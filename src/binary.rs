//! Binary and ternary 1-bit layers with straight-through estimator (STE).
//!
//! Forward: quantize weights (and optionally activations); backward: gradients
//! flow to full-precision parameters via STE. Supports binary ±1 and ternary {-1,0,+1} (AbsMean).
//! Optional inference cache: pre-compute quantized weights once for faster repeated forwards.

use std::cell::RefCell;

use candle::{DType, Result, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder};

/// Straight-through estimator for sign with optional scale: forward ≈ sign(x), backward = scale * identity so latent weights get stronger updates.
#[inline]
fn ste_sign_scaled(x: &Tensor, scale: f64) -> Result<Tensor> {
    let sign_x = x.sign()?;
    let detach_x = x.detach();
    let residual = (x - &detach_x)?;
    Ok((&sign_x + &residual.affine(scale, 0.0)?)?)
}

/// Ternary in *original* space with Δ = 0.7 × mean(|W|): |w| <= Δ -> 0, else sign(w). No 1/beta scaling.
fn ternary_delta_quantize(w: &Tensor, delta: f64, ste_scale: f64, apply_ste: bool) -> Result<Tensor> {
    let abs_w = w.abs()?;
    let mask = abs_w.gt(delta)?.to_dtype(DType::F32)?;
    let out = (w.sign()? * mask)?;
    if apply_ste {
        let residual = (w - &w.detach())?;
        Ok((out.detach() + residual.affine(ste_scale, 0.0)?)?)
    } else {
        Ok(out)
    }
}

/// Ternary quantization (forward only, no STE). For debug stats.
fn ternary_quantize_forward(w: &Tensor, use_dynamic_threshold: bool) -> Result<Tensor> {
    let abs_w = w.abs()?;
    let beta = abs_w.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
    if use_dynamic_threshold {
        let delta = 0.7 * beta;
        ternary_delta_quantize(w, delta, 1.0, false)
    } else {
        let inv_beta = 1.0f64 / beta;
        let w_scaled = w.affine(inv_beta, 0.0)?;
        let rounded = w_scaled.round()?;
        Ok(rounded.clamp(-1f64, 1f64)?)
    }
}

/// Ternary quantization: either Δ = 0.7×mean(|W|) (dynamic) or AbsMean (scale by 1/beta, round). STE with scale.
fn ternary_absmean_ste(w: &Tensor, use_dynamic_threshold: bool, ste_scale: f64) -> Result<Tensor> {
    let abs_w = w.abs()?;
    let beta = abs_w.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
    if use_dynamic_threshold {
        let delta = 0.7 * beta;
        ternary_delta_quantize(w, delta, ste_scale, true)
    } else {
        let inv_beta = 1.0f64 / beta;
        let w_scaled = w.affine(inv_beta, 0.0)?;
        let rounded = w_scaled.round()?;
        let clamped = rounded.clamp(-1f64, 1f64)?;
        let residual = (w - &w.detach())?;
        Ok((clamped.detach() + residual.affine(ste_scale, 0.0)?)?)
    }
}

/// Binary linear layer: forward uses sign(W) and sign(x) with STE; optional latent clamp and STE scale.
/// Optional cache: (quantized_weight_t, scale) for inference-only fast path (no re-quantize per forward).
pub struct BinaryLinear {
    weight: Linear,
    ste_scale_factor: f64,
    latent_clamp_max: f64,
    /// Cached (quantized_weight, scale) for inference when set via cache_quantized().
    cache: RefCell<Option<(Tensor, f64)>>,
}

/// Small init for 1-bit layers so more weights land in zero band (ternary) or don't all saturate (binary).
const BIT_LAYER_INIT: Init = Init::Randn {
    mean: 0.,
    stdev: 0.02,
};

impl BinaryLinear {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder, ste_scale_factor: f64, latent_clamp_max: f64) -> Result<Self> {
        let ws = vb.get_with_hints((out_dim, in_dim), "weight", BIT_LAYER_INIT)?;
        let weight = Linear::new(ws, None);
        Ok(Self { weight, ste_scale_factor, latent_clamp_max, cache: RefCell::new(None) })
    }

    /// Debug: counts of -1 and +1 in quantized (sign) weight.
    pub fn debug_weight_distribution(&self) -> Result<(u64, u64)> {
        let w = self.weight.weight();
        debug_binary_distribution(w)
    }

    /// Pre-compute quantized weight and scale; forward will use cache when set (inference-only path).
    pub fn cache_quantized(&self) -> Result<()> {
        let w = self.weight.weight();
        let w_use = w.clamp(-self.latent_clamp_max, self.latent_clamp_max)?;
        let in_dim = w_use.dim(1)?;
        let w_bin = w_use.sign()?; // no STE for inference cache
        let gamma = w_use.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
        let scale = (1.0 / (in_dim as f64).sqrt()) * gamma;
        self.cache.borrow_mut().replace((w_bin, scale));
        Ok(())
    }

    /// Clear inference cache (e.g. before training step).
    pub fn clear_cache(&self) {
        self.cache.borrow_mut().take();
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some((ref w_bin, scale)) = self.cache.borrow().as_ref() {
            let x_bin = ste_sign_scaled(x, self.ste_scale_factor)?;
            let out = matmul_reshape(&x_bin, &w_bin.t()?)?;
            return out.affine(*scale, 0.0);
        }
        let w = self.weight.weight();
        let w_use = w.clamp(-self.latent_clamp_max, self.latent_clamp_max)?;
        let in_dim = w_use.dim(1)?;
        let w_bin = ste_sign_scaled(&w_use, self.ste_scale_factor)?;
        let x_bin = ste_sign_scaled(x, self.ste_scale_factor)?;
        let out = matmul_reshape(&x_bin, &w_bin.t()?)?;
        let gamma = w_use.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
        let scale = (1.0 / (in_dim as f64).sqrt()) * gamma;
        out.affine(scale, 0.0)
    }
}

impl Module for BinaryLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

/// Ternary linear layer (BitNet-style): AbsMean or Δ=0.7×mean(|W|) ternary, STE with scale; latent clamp.
/// Optional cache: (quantized_weight_t, scale) for inference-only fast path.
pub struct TernaryLinear {
    weight: Linear,
    use_dynamic_threshold: bool,
    ste_scale_factor: f64,
    latent_clamp_max: f64,
    cache: RefCell<Option<(Tensor, f64)>>,
}

impl TernaryLinear {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder, use_dynamic_threshold: bool, ste_scale_factor: f64, latent_clamp_max: f64) -> Result<Self> {
        let ws = vb.get_with_hints((out_dim, in_dim), "weight", BIT_LAYER_INIT)?;
        let weight = Linear::new(ws, None);
        Ok(Self { weight, use_dynamic_threshold, ste_scale_factor, latent_clamp_max, cache: RefCell::new(None) })
    }

    /// Debug: counts of -1, 0, +1 in quantized ternary weight.
    pub fn debug_weight_distribution(&self) -> Result<(u64, u64, u64)> {
        let w = self.weight.weight();
        debug_ternary_distribution(w, self.use_dynamic_threshold)
    }

    /// Pre-compute quantized weight and scale; forward will use cache when set (inference-only path).
    pub fn cache_quantized(&self) -> Result<()> {
        let w = self.weight.weight();
        let w_use = w.clamp(-self.latent_clamp_max, self.latent_clamp_max)?;
        let in_dim = w_use.dim(1)?;
        let w_ternary = ternary_quantize_forward(&w_use, self.use_dynamic_threshold)?;
        let gamma = w_use.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
        let scale = (1.0 / (in_dim as f64).sqrt()) * gamma;
        self.cache.borrow_mut().replace((w_ternary, scale));
        Ok(())
    }

    /// Clear inference cache.
    pub fn clear_cache(&self) {
        self.cache.borrow_mut().take();
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some((ref w_ternary, scale)) = self.cache.borrow().as_ref() {
            let out = matmul_reshape(x, &w_ternary.t()?)?;
            return out.affine(*scale, 0.0);
        }
        let w = self.weight.weight();
        let w_use = w.clamp(-self.latent_clamp_max, self.latent_clamp_max)?;
        let in_dim = w_use.dim(1)?;
        let w_ternary = ternary_absmean_ste(&w_use, self.use_dynamic_threshold, self.ste_scale_factor)?;
        let out = matmul_reshape(x, &w_ternary.t()?)?;
        let gamma = w_use.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
        let scale = (1.0 / (in_dim as f64).sqrt()) * gamma;
        out.affine(scale, 0.0)
    }
}

impl Module for TernaryLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

// --- Debug: weight value distribution (-1, 0, +1 for ternary; -1, +1 for binary) ---

/// Counts of -1, 0, +1 in quantized ternary weight (no STE, forward-only quantize).
pub fn debug_ternary_distribution(w: &Tensor, use_dynamic_threshold: bool) -> Result<(u64, u64, u64)> {
    let q = ternary_quantize_forward(w, use_dynamic_threshold)?;
    let flat = q.flatten_all()?.to_vec1::<f32>()?;
    let mut n_neg = 0u64;
    let mut n_zero = 0u64;
    let mut n_pos = 0u64;
    for &v in flat.iter() {
        if v < -0.5 {
            n_neg += 1;
        } else if v > 0.5 {
            n_pos += 1;
        } else {
            n_zero += 1;
        }
    }
    Ok((n_neg, n_zero, n_pos))
}

/// Counts of -1, +1 in quantized binary weight (sign, no STE).
pub fn debug_binary_distribution(w: &Tensor) -> Result<(u64, u64)> {
    let s = w.sign()?;
    let flat = s.flatten_all()?.to_vec1::<f32>()?;
    let mut n_neg = 0u64;
    let mut n_pos = 0u64;
    for &v in flat.iter() {
        if v < 0.0 {
            n_neg += 1;
        } else {
            n_pos += 1;
        }
    }
    Ok((n_neg, n_pos))
}

/// Reshape x to 2D, matmul with w_t, reshape back to original leading dims.
fn matmul_reshape(x: &Tensor, w_t: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let out_dim = w_t.dim(1)?;
    match dims.as_ref() {
        [b, m, k] => {
            let x_2d = x.reshape((*b * *m, *k))?;
            let y = x_2d.matmul(w_t)?;
            y.reshape((*b, *m, out_dim))
        }
        [b1, b2, m, k] => {
            let x_2d = x.reshape((*b1 * *b2 * *m, *k))?;
            let y = x_2d.matmul(w_t)?;
            y.reshape((*b1, *b2, *m, out_dim))
        }
        _ => {
            let last = dims[dims.len() - 1];
            let prod: usize = dims[..dims.len() - 1].iter().product();
            let x_2d = x.reshape((prod, last))?;
            let y = x_2d.matmul(w_t)?;
            let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
            out_shape.push(out_dim);
            y.reshape(out_shape.as_slice())
        }
    }
}
