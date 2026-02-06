//! Mixed-precision linear layers: F16, EightBit, Ternary, Binary (BitLinear).
//!
//! **Sandwich Rule:** Use [`QuantMode`] to keep embedding and LM head in F16/EightBit,
//! and hidden layers in Ternary or Binary. Each quantized layer stores full-precision
//! (f32) "latent" weights; forward uses quantized values; backward uses STE.
//!
//! # Thread safety
//!
//! The inference cache uses [`parking_lot::Mutex`] instead of `RefCell`,
//! making these layers `Send + Sync`.

use parking_lot::Mutex;

use candle_core::{Result, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder};

use ternary_common::{OneBitLlmConfig, QuantMode};

use crate::quantize::{
    current_anneal_frac, debug_binary_distribution, debug_ternary_distribution, matmul_reshape,
    ste_sign_scaled, ste_tanh_scaled, ternary_absmean_ste, ternary_quantize_forward,
};

/// Small init for 1-bit layers.
///
/// σ = 0.02 ensures most initial weights land inside the ternary "zero band"
/// (|w| < δ ≈ 0.014), giving the optimiser room to push them toward ±1
/// during training rather than having them all saturate at init.
const BIT_LAYER_INIT: Init = Init::Randn {
    mean: 0.,
    stdev: 0.02,
};

// ── BinaryLinear ────────────────────────────────────────────────────────────

/// Binary linear layer: weights are quantised to ±1 via `sign()` with STE.
///
/// Forward path:
/// 1. Clamp latent weights to [-C, +C].
/// 2. Quantise weights and activations (soft or hard depending on annealing).
/// 3. Matmul.
/// 4. Scale by γ / √d_in.
pub struct BinaryLinear {
    weight: Linear,
    ste_scale_factor: f64,
    latent_clamp_max: f64,
    /// Inference cache: `Some((quantised_weight, scale))` when set via
    /// [`cache_quantized`]. Forward uses this to skip re-quantisation.
    cache: Mutex<Option<(Tensor, f64)>>,
}

impl BinaryLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        ste_scale_factor: f64,
        latent_clamp_max: f64,
    ) -> Result<Self> {
        let ws = vb.get_with_hints((out_dim, in_dim), "weight", BIT_LAYER_INIT)?;
        let weight = Linear::new(ws, None);
        Ok(Self {
            weight,
            ste_scale_factor,
            latent_clamp_max,
            cache: Mutex::new(None),
        })
    }

    /// Debug: count ±1 in the quantised weight.
    pub fn debug_weight_distribution(&self) -> Result<(u64, u64)> {
        debug_binary_distribution(self.weight.weight())
    }

    /// Pre-compute quantised weight + scale for inference (no STE).
    pub fn cache_quantized(&self) -> Result<()> {
        let w = self.weight.weight();
        let w_use = w.clamp(-self.latent_clamp_max, self.latent_clamp_max)?;
        let in_dim = w_use.dim(1)?;
        let w_bin = w_use.sign()?;
        let gamma = w_use.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
        let scale = gamma / (in_dim as f64).sqrt();
        self.cache.lock().replace((w_bin, scale));
        Ok(())
    }

    /// Clear inference cache (call before training resumes).
    pub fn clear_cache(&self) {
        self.cache.lock().take();
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Fast path: cached inference weights
        {
            let guard = self.cache.lock();
            if let Some((ref w_bin, scale)) = *guard {
                let x_bin = ste_sign_scaled(x, self.ste_scale_factor)?;
                let out = matmul_reshape(&x_bin, &w_bin.t()?)?;
                return out.affine(scale, 0.0);
            }
        }

        // Training path
        let w = self.weight.weight();
        let w_use = w.clamp(-self.latent_clamp_max, self.latent_clamp_max)?;
        let in_dim = w_use.dim(1)?;
        let anneal = current_anneal_frac();

        let (w_bin, x_bin) = if anneal < 1.0 {
            // Soft regime: α grows from 1→8 over the annealing schedule.
            let alpha = 1.0 + 7.0 * anneal;
            (
                ste_tanh_scaled(&w_use, alpha, self.ste_scale_factor)?,
                ste_tanh_scaled(x, alpha, self.ste_scale_factor)?,
            )
        } else {
            // Hard regime: true sign with STE.
            (
                ste_sign_scaled(&w_use, self.ste_scale_factor)?,
                ste_sign_scaled(x, self.ste_scale_factor)?,
            )
        };

        let out = matmul_reshape(&x_bin, &w_bin.t()?)?;
        let gamma = w_use.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
        let scale = gamma / (in_dim as f64).sqrt();
        out.affine(scale, 0.0)
    }
}

impl Module for BinaryLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

// ── TernaryLinear ───────────────────────────────────────────────────────────

/// Ternary linear layer: weights are quantised to {-1, 0, +1} (BitNet b1.58).
///
/// Two quantisation strategies (selected by `use_dynamic_threshold`):
/// * **Dynamic Threshold:** δ = 0.7 × mean(|W|). Weights inside the band → 0.
/// * **AbsMean:** Scale by 1/mean(|W|), round, clamp.
pub struct TernaryLinear {
    weight: Linear,
    use_dynamic_threshold: bool,
    ste_scale_factor: f64,
    latent_clamp_max: f64,
    cache: Mutex<Option<(Tensor, f64)>>,
}

impl TernaryLinear {
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilder,
        use_dynamic_threshold: bool,
        ste_scale_factor: f64,
        latent_clamp_max: f64,
    ) -> Result<Self> {
        let ws = vb.get_with_hints((out_dim, in_dim), "weight", BIT_LAYER_INIT)?;
        let weight = Linear::new(ws, None);
        Ok(Self {
            weight,
            use_dynamic_threshold,
            ste_scale_factor,
            latent_clamp_max,
            cache: Mutex::new(None),
        })
    }

    /// Debug: count {-1, 0, +1} in the quantised weight.
    pub fn debug_weight_distribution(&self) -> Result<(u64, u64, u64)> {
        debug_ternary_distribution(self.weight.weight(), self.use_dynamic_threshold)
    }

    /// Pre-compute quantised weight + scale for inference.
    pub fn cache_quantized(&self) -> Result<()> {
        let w = self.weight.weight();
        let w_use = w.clamp(-self.latent_clamp_max, self.latent_clamp_max)?;
        let in_dim = w_use.dim(1)?;
        let w_ternary = ternary_quantize_forward(&w_use, self.use_dynamic_threshold)?;
        let gamma = w_use.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
        let scale = gamma / (in_dim as f64).sqrt();
        self.cache.lock().replace((w_ternary, scale));
        Ok(())
    }

    /// Clear inference cache.
    pub fn clear_cache(&self) {
        self.cache.lock().take();
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Fast path: cached inference weights
        {
            let guard = self.cache.lock();
            if let Some((ref w_ternary, scale)) = *guard {
                let out = matmul_reshape(x, &w_ternary.t()?)?;
                return out.affine(scale, 0.0);
            }
        }

        // Training path
        let w = self.weight.weight();
        let w_use = w.clamp(-self.latent_clamp_max, self.latent_clamp_max)?;
        let in_dim = w_use.dim(1)?;
        let anneal = current_anneal_frac();

        // Soft activation smoothing during annealing (stabilises early grads).
        let x_in = if anneal < 1.0 {
            let alpha = 1.0 + 7.0 * anneal;
            ste_tanh_scaled(x, alpha, self.ste_scale_factor)?
        } else {
            x.clone()
        };

        let w_ternary =
            ternary_absmean_ste(&w_use, self.use_dynamic_threshold, self.ste_scale_factor)?;
        let out = matmul_reshape(&x_in, &w_ternary.t()?)?;
        let gamma = w_use.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
        let scale = gamma / (in_dim as f64).sqrt();
        out.affine(scale, 0.0)
    }
}

impl Module for TernaryLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

// ── F16Linear (Full precision) ───────────────────────────────────────────────

/// Full-precision linear layer (F16/f32). Used for embedding and LM head in the Sandwich Rule.
pub struct F16Linear {
    inner: Linear,
}

impl F16Linear {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::linear(in_dim, out_dim, vb)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

impl Module for F16Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

// ── EightBitLinear (Stub: full precision for now) ───────────────────────────

/// 8-bit quantized linear layer. Currently implemented as full precision;
/// export/inference can apply 8-bit quantization. Used in Sandwich Rule for
/// optional embedding/lm_head middle ground.
pub struct EightBitLinear {
    inner: Linear,
}

impl EightBitLinear {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = candle_nn::linear(in_dim, out_dim, vb)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

impl Module for EightBitLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

// ── BitLinearLayer (Mixed precision dispatch) ────────────────────────────────

/// Dispatch enum: F16, EightBit, Ternary, or Binary per layer (Sandwich Rule).
///
/// This is the type used by attention and FFN blocks. Construct with
/// [`BitLinearLayer::new_with_mode`] when using a layer bit-map.
pub enum BitLinearLayer {
    F16(F16Linear),
    EightBit(EightBitLinear),
    Binary(BinaryLinear),
    Ternary(TernaryLinear),
}

impl BitLinearLayer {
    /// Construct from config: uses `config.use_ternary` for Binary vs Ternary only.
    /// For Sandwich Rule use [`BitLinearLayer::new_with_mode`].
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        config: &OneBitLlmConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        if config.use_ternary {
            Ok(Self::Ternary(TernaryLinear::new(
                in_dim,
                out_dim,
                vb,
                config.use_dynamic_threshold,
                config.ste_scale_factor,
                config.latent_clamp_max,
            )?))
        } else {
            Ok(Self::Binary(BinaryLinear::new(
                in_dim,
                out_dim,
                vb,
                config.ste_scale_factor,
                config.latent_clamp_max,
            )?))
        }
    }

    /// Construct with explicit [`QuantMode`] for mixed-precision (Sandwich Rule).
    ///
    /// * F16 / EightBit: no STE, no annealing; standard forward.
    /// * Ternary / Binary: latent weights, STE, annealing; `config` supplies
    ///   ste_scale_factor, latent_clamp_max, use_dynamic_threshold.
    pub fn new_with_mode(
        in_dim: usize,
        out_dim: usize,
        mode: QuantMode,
        config: &OneBitLlmConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        match mode {
            QuantMode::F16 => Ok(Self::F16(F16Linear::new(in_dim, out_dim, vb)?)),
            QuantMode::EightBit => Ok(Self::EightBit(EightBitLinear::new(in_dim, out_dim, vb)?)),
            QuantMode::Ternary => Ok(Self::Ternary(TernaryLinear::new(
                in_dim,
                out_dim,
                vb,
                config.use_dynamic_threshold,
                config.ste_scale_factor,
                config.latent_clamp_max,
            )?)),
            QuantMode::Binary => Ok(Self::Binary(BinaryLinear::new(
                in_dim,
                out_dim,
                vb,
                config.ste_scale_factor,
                config.latent_clamp_max,
            )?)),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::F16(l) => l.forward(x),
            Self::EightBit(l) => l.forward(x),
            Self::Binary(l) => l.forward(x),
            Self::Ternary(l) => l.forward(x),
        }
    }

    /// Human-readable weight distribution string.
    pub fn debug_weight_distribution(&self) -> Result<String> {
        match self {
            Self::F16(_) => Ok("f16 (full precision)".to_string()),
            Self::EightBit(_) => Ok("8bit (stub)".to_string()),
            Self::Binary(l) => {
                let (neg, pos) = l.debug_weight_distribution()?;
                Ok(format!("binary -1:{neg} +1:{pos}"))
            }
            Self::Ternary(l) => {
                let (neg, zero, pos) = l.debug_weight_distribution()?;
                Ok(format!("ternary -1:{neg} 0:{zero} +1:{pos}"))
            }
        }
    }

    pub fn cache_quantized(&self) -> Result<()> {
        match self {
            Self::F16(_) | Self::EightBit(_) => Ok(()),
            Self::Binary(l) => l.cache_quantized(),
            Self::Ternary(l) => l.cache_quantized(),
        }
    }

    pub fn clear_cache(&self) {
        match self {
            Self::F16(_) | Self::EightBit(_) => {}
            Self::Binary(l) => l.clear_cache(),
            Self::Ternary(l) => l.clear_cache(),
        }
    }
}
