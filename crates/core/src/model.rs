//! Decoder-only transformer (GPT-style) with binary/ternary layers.
//!
//! Supports: RMSNorm (subln), RoPE, QK-norm, residual scaling, Arenas
//! residual, SwiGLU / ReLU² / SiLU activations. Weight tying between
//! token embedding and output projection (lm_head shares `wte`).

use candle_core::{Result, Tensor};
use candle_nn::{embedding, Embedding, Module, VarBuilder};

use ternary_common::OneBitLlmConfig;

use crate::activation::FfnLayer;
use crate::attention::CausalSelfAttention;
use crate::norm::NormLayer;

// ── Decoder Block ───────────────────────────────────────────────────────────

/// Single decoder block: pre-norm → attention → residual → pre-norm → FFN → residual.
///
/// Optional Arenas: adds a full-precision shortcut `arenas_coef × block_input`
/// to the residual (stabilises early training, anneals to 0).
struct DecoderBlock {
    attn: CausalSelfAttention,
    ln1: NormLayer,
    ffn: FfnLayer,
    ln2: NormLayer,
    residual_scale: f64,
}

impl DecoderBlock {
    fn new(config: &OneBitLlmConfig, vb: VarBuilder) -> Result<Self> {
        let attn = CausalSelfAttention::new(config, vb.pp("attn"))?;
        let ln1 = NormLayer::new(config, vb.pp("ln1"))?;
        let ffn = FfnLayer::new(config, vb.pp("mlp"))?;
        let ln2 = NormLayer::new(config, vb.pp("ln2"))?;

        let residual_scale = if config.use_residual_scaling {
            1.0 / 2.0_f64.sqrt()
        } else {
            1.0
        };

        Ok(Self {
            attn,
            ln1,
            ffn,
            ln2,
            residual_scale,
        })
    }

    fn forward(&self, x: &Tensor, arenas_coef: Option<f32>) -> Result<Tensor> {
        let block_input = x;

        // Attention sub-layer
        let residual = x;
        let normed = self.ln1.forward(x)?;
        let attn_out = self.attn.forward(&normed)?;
        let scaled = attn_out.affine(self.residual_scale, 0.0)?;
        let mut x = (residual + scaled)?;
        if let Some(c) = arenas_coef {
            let c_t = Tensor::new(&[c], block_input.device())?;
            x = (x + block_input.broadcast_mul(&c_t)?)?;
        }

        // FFN sub-layer
        let residual = &x;
        let normed = self.ln2.forward(&x)?;
        let ff_out = self.ffn.forward(&normed)?;
        let scaled = ff_out.affine(self.residual_scale, 0.0)?;
        let mut x = (residual + scaled)?;
        if let Some(c) = arenas_coef {
            let c_t = Tensor::new(&[c], block_input.device())?;
            x = (x + block_input.broadcast_mul(&c_t)?)?;
        }

        Ok(x)
    }

    fn debug_weight_distributions(&self, block_idx: usize) -> Vec<(String, Result<String>)> {
        let prefix = format!("h.{block_idx}");
        let mut out = self
            .attn
            .debug_weight_distributions(&format!("{prefix}.attn"));
        out.extend(
            self.ffn
                .debug_weight_distributions(&format!("{prefix}.mlp")),
        );
        out
    }

    fn cache_quantized(&self) -> Result<()> {
        self.attn.cache_quantized()?;
        self.ffn.cache_quantized()
    }

    fn clear_cache(&self) {
        self.attn.clear_cache();
        self.ffn.clear_cache();
    }
}

// ── OneBitLlm ───────────────────────────────────────────────────────────────

/// Decoder-only transformer with binary/ternary linear layers.
///
/// Weight tying: the token embedding `wte` and the output projection share
/// the same weight matrix. No separate `lm_head` is stored.
pub struct OneBitLlm {
    wte: Embedding,
    blocks: Vec<DecoderBlock>,
    ln_f: NormLayer,
    config: OneBitLlmConfig,
}

impl OneBitLlm {
    pub fn new(vb: VarBuilder, config: &OneBitLlmConfig) -> Result<Self> {
        let wte = embedding(config.vocab_size, config.hidden_size, vb.pp("wte"))?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let block = DecoderBlock::new(config, vb.pp(format!("h.{i}")))?;
            blocks.push(block);
        }

        let ln_f = NormLayer::new(config, vb.pp("ln_f"))?;

        Ok(Self {
            wte,
            blocks,
            ln_f,
            config: config.clone(),
        })
    }

    /// Forward pass (inference — no Arenas residual).
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_with_arenas(input_ids, None)
    }

    /// Forward pass with optional Arenas residual coefficient.
    pub fn forward_with_arenas(
        &self,
        input_ids: &Tensor,
        arenas_coef: Option<f32>,
    ) -> Result<Tensor> {
        let mut x = self.wte.forward(input_ids)?;
        for block in &self.blocks {
            x = block.forward(&x, arenas_coef)?;
        }
        x = self.ln_f.forward(&x)?;

        // Weight-tied output projection: logits = x @ wte^T
        let wte_weight = self.wte.embeddings();
        let (b, t, h) = x.dims3()?;
        let x_2d = x.reshape((b * t, h))?;
        let logits = x_2d.matmul(&wte_weight.t()?)?;
        logits.reshape((b, t, self.config.vocab_size))
    }

    pub fn config(&self) -> &OneBitLlmConfig {
        &self.config
    }

    /// Collect weight distributions for all bit-linear layers (debug).
    pub fn debug_weight_distributions(&self) -> Result<Vec<(String, String)>> {
        let mut out = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            for (name, res) in block.debug_weight_distributions(i) {
                let s = res?;
                out.push((name, s));
            }
        }
        Ok(out)
    }

    /// Pre-compute quantised weights for all layers (inference-only fast path).
    pub fn cache_quantized_weights(&self) -> Result<()> {
        for block in &self.blocks {
            block.cache_quantized()?;
        }
        Ok(())
    }

    /// Clear cached quantised weights (call before training resumes).
    pub fn clear_quantized_cache(&self) {
        for block in &self.blocks {
            block.clear_cache();
        }
    }
}

// ── Compression Stats ───────────────────────────────────────────────────────

/// Approximate parameter counts and compression statistics.
///
/// Computed from config alone (no model instance needed).
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub total_params: usize,
    pub quantized_params: usize,
    /// Effective bits per weight: (quant×2 + full×32) / total.
    pub effective_bits_per_param: f64,
    /// Compression ratio vs FP32.
    pub compression_ratio_vs_f32: f64,
}

/// Compute approximate parameter counts and compression ratio from config.
///
/// No model instance needed — pure arithmetic from hyper-parameters.
pub fn compression_stats(config: &OneBitLlmConfig) -> CompressionStats {
    let vocab = config.vocab_size;
    let h = config.hidden_size;
    let n = config.num_layers;
    let inter = config.intermediate_size;

    let wte = vocab * h;
    let per_block_attn = h * (3 * h) + h * h; // c_attn + c_proj

    let per_block_ff = if config.use_swiglu {
        // SwiGLU: gate + up + down = 3 × h × inter
        3 * h * inter
    } else {
        // Standard: c_fc + c_proj = 2 × h × inter
        2 * h * inter
    };
    let quantized_per_block = per_block_attn + per_block_ff;

    let norm_params = (n * 2 + 1) * h;
    let qk_norm = if config.use_qk_norm { n * 2 * h } else { 0 };
    let total = wte + n * (per_block_attn + per_block_ff + norm_params) + qk_norm;
    let quantized = n * quantized_per_block;

    let bits_quantized = if config.use_ternary { 2.0 } else { 1.0 };
    let bits_full = 32.0;
    let effective_bits = (quantized as f64 * bits_quantized
        + (total.saturating_sub(quantized)) as f64 * bits_full)
        / total.max(1) as f64;
    let compression_ratio_vs_f32 = bits_full / effective_bits;

    CompressionStats {
        total_params: total,
        quantized_params: quantized,
        effective_bits_per_param: effective_bits,
        compression_ratio_vs_f32,
    }
}
