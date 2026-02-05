//! Causal self-attention with binary/ternary Q, K, V projections.
//!
//! Features:
//! * **RoPE** — Rotary position embeddings (optional).
//! * **QK-Norm** — RMSNorm on Q and K before attention (stabilises training).
//! * Fused Q/K/V projection via a single `BitLinearLayer` (3 × hidden).

use candle_core::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{rms_norm, Module, RmsNorm, VarBuilder};

use ternary_common::OneBitLlmConfig;

use crate::linear::BitLinearLayer;

/// Build RoPE cosine and sine tables: shape (seq_len, head_dim / 2).
///
/// Standard RoPE: θ_i = 10000^{-2i/d}.
fn rope_cos_sin(
    device: &candle_core::Device,
    seq_len: usize,
    head_dim: usize,
) -> Result<(Tensor, Tensor)> {
    let d2 = head_dim / 2;
    let inv_freq: Vec<f32> = (0..d2)
        .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, d2), device)?;
    let positions = Tensor::arange(0u32, seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((seq_len, 1))?;
    let freqs = positions.broadcast_mul(&inv_freq)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    Ok((cos, sin))
}

/// Multi-head causal self-attention.
pub struct CausalSelfAttention {
    c_attn: BitLinearLayer,
    c_proj: BitLinearLayer,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    use_rope: bool,
}

impl CausalSelfAttention {
    pub fn new(config: &OneBitLlmConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = config.hidden_size;
        let num_heads = config.num_heads;
        let head_dim = config.head_dim();

        let c_attn = BitLinearLayer::new(hidden, 3 * hidden, config, vb.pp("c_attn"))?;
        let c_proj = BitLinearLayer::new(hidden, hidden, config, vb.pp("c_proj"))?;

        let (q_norm, k_norm) = if config.use_qk_norm {
            let q = rms_norm(head_dim, config.layer_norm_eps, vb.pp("q_norm"))?;
            let k = rms_norm(head_dim, config.layer_norm_eps, vb.pp("k_norm"))?;
            (Some(q), Some(k))
        } else {
            (None, None)
        };

        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            c_attn,
            c_proj,
            q_norm,
            k_norm,
            num_heads,
            head_dim,
            scale,
            use_rope: config.use_rope,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;

        // Fused QKV projection
        let qkv = self.c_attn.forward(x)?;
        let qkv = qkv.reshape((b, t, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((0, 3, 1, 4, 2))?; // (b, heads, t, head_dim, 3)

        let mut q = qkv.i((.., .., .., .., 0))?.contiguous()?;
        let mut k = qkv.i((.., .., .., .., 1))?.contiguous()?;
        let v = qkv.i((.., .., .., .., 2))?.contiguous()?;

        // Optional QK-norm
        if let Some(ref q_norm) = self.q_norm {
            q = q_norm.forward(&q)?;
        }
        if let Some(ref k_norm) = self.k_norm {
            k = k_norm.forward(&k)?;
        }

        // Optional RoPE
        if self.use_rope {
            let device = x.device();
            let (cos, sin) = rope_cos_sin(device, t, self.head_dim)?;
            q = candle_nn::rotary_emb::rope_i(&q, &cos, &sin)?;
            k = candle_nn::rotary_emb::rope_i(&k, &cos, &sin)?;
        }

        // Scaled dot-product attention with causal mask
        let scores = (q.matmul(&k.t()?)? * self.scale)?;
        let device = x.device();
        let mask = Tensor::tril2(t, DType::F32, device)?;
        let mask = mask.reshape((1, 1, t, t))?;
        let ones = Tensor::ones((1, 1, t, t), DType::F32, device)?;
        let one_minus_mask = (&ones - &mask)?;
        let neg_inf = (-1e9f64 * &one_minus_mask)?;
        let scores = scores.broadcast_add(&neg_inf)?;

        let att = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let y = att.contiguous()?.matmul(&v)?;
        let y = y.transpose(1, 2)?;
        let y = y.reshape((b, t, c))?;

        self.c_proj.forward(&y)
    }

    /// Debug weight distributions for attention projections.
    pub fn debug_weight_distributions(&self, prefix: &str) -> Vec<(String, Result<String>)> {
        vec![
            (
                format!("{prefix}.c_attn"),
                self.c_attn.debug_weight_distribution(),
            ),
            (
                format!("{prefix}.c_proj"),
                self.c_proj.debug_weight_distribution(),
            ),
        ]
    }

    pub fn cache_quantized(&self) -> Result<()> {
        self.c_attn.cache_quantized()?;
        self.c_proj.cache_quantized()?;
        Ok(())
    }

    pub fn clear_cache(&self) {
        self.c_attn.clear_cache();
        self.c_proj.clear_cache();
    }
}
