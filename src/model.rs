//! Decoder-only transformer (GPT-style) with binary/ternary layers.
//! Supports BitNet-style: ReLU², subln (RMSNorm), RoPE, Arenas. All comments in English.

use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm_no_bias, linear_no_bias, rms_norm, Embedding, LayerNorm, Module, RmsNorm,
    VarBuilder,
};

use crate::binary::{BinaryLinear, TernaryLinear};
use crate::config::OneBitLlmConfig;

/// Shared linear layer: either binary (±1) or ternary {-1,0,+1} AbsMean, depending on config.
enum BitLinearLayer {
    Binary(BinaryLinear),
    Ternary(TernaryLinear),
}

impl BitLinearLayer {
    fn new(in_dim: usize, out_dim: usize, config: &OneBitLlmConfig, vb: VarBuilder) -> Result<Self> {
        if config.use_ternary {
            Ok(BitLinearLayer::Ternary(TernaryLinear::new(
                in_dim,
                out_dim,
                vb,
                config.use_dynamic_threshold,
                config.ste_scale_factor,
                config.latent_clamp_max,
            )?))
        } else {
            Ok(BitLinearLayer::Binary(BinaryLinear::new(
                in_dim,
                out_dim,
                vb,
                config.ste_scale_factor,
                config.latent_clamp_max,
            )?))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            BitLinearLayer::Binary(l) => l.forward(x),
            BitLinearLayer::Ternary(l) => l.forward(x),
        }
    }

    /// Debug: -1, 0, +1 (ternary) or -1, +1 (binary) distribution as string.
    fn debug_weight_distribution(&self) -> Result<String> {
        match self {
            BitLinearLayer::Binary(l) => {
                let (n_neg, n_pos) = l.debug_weight_distribution()?;
                Ok(format!("binary -1:{} +1:{}", n_neg, n_pos))
            }
            BitLinearLayer::Ternary(l) => {
                let (n_neg, n_zero, n_pos) = l.debug_weight_distribution()?;
                Ok(format!("ternary -1:{} 0:{} +1:{}", n_neg, n_zero, n_pos))
            }
        }
    }

    fn cache_quantized(&self) -> Result<()> {
        match self {
            BitLinearLayer::Binary(l) => l.cache_quantized(),
            BitLinearLayer::Ternary(l) => l.cache_quantized(),
        }
    }

    fn clear_cache(&self) {
        match self {
            BitLinearLayer::Binary(l) => l.clear_cache(),
            BitLinearLayer::Ternary(l) => l.clear_cache(),
        }
    }
}

/// Build RoPE cos and sin for shape (seq_len, head_dim/2). Standard RoPE: theta_i = 10000^(-2i/d).
fn rope_cos_sin(device: &candle::Device, seq_len: usize, head_dim: usize) -> Result<(Tensor, Tensor)> {
    let d2 = head_dim / 2;
    let inv_freq: Vec<f32> = (0..d2)
        .map(|i| 1.0 / 10000f32.powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (1, d2), device)?;
    let positions = Tensor::arange(0u32, seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((seq_len, 1))?;
    let freqs = positions.broadcast_mul(&inv_freq)?; // (seq_len, d2)
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    Ok((cos, sin))
}

/// Causal self-attention with binary/ternary Q,K,V; optional RoPE on q,k; optional QK-norm (RMSNorm on Q,K).
struct CausalSelfAttention {
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
    fn new(config: &OneBitLlmConfig, vb: VarBuilder) -> Result<Self> {
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

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let qkv = self.c_attn.forward(x)?;
        let qkv = qkv.reshape((b, t, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((0, 3, 1, 4, 2))?; // (b, heads, t, head_dim, 3)
        let mut q = qkv.i((.., .., .., .., 0))?.contiguous()?;
        let mut k = qkv.i((.., .., .., .., 1))?.contiguous()?;
        let v = qkv.i((.., .., .., .., 2))?.contiguous()?;
        if let Some(ref q_norm) = self.q_norm {
            q = q_norm.forward(&q)?;
        }
        if let Some(ref k_norm) = self.k_norm {
            k = k_norm.forward(&k)?;
        }
        if self.use_rope {
            let device = x.device();
            let (cos, sin) = rope_cos_sin(device, t, self.head_dim)?; // (t, head_dim/2)
            q = candle_nn::rotary_emb::rope_i(&q, &cos, &sin)?;
            k = candle_nn::rotary_emb::rope_i(&k, &cos, &sin)?;
        }
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

    fn debug_weight_distributions(&self, prefix: &str) -> Vec<(String, Result<String>)> {
        vec![
            (
                format!("{}.c_attn", prefix),
                self.c_attn.debug_weight_distribution(),
            ),
            (
                format!("{}.c_proj", prefix),
                self.c_proj.debug_weight_distribution(),
            ),
        ]
    }

    fn cache_quantized(&self) -> Result<()> {
        self.c_attn.cache_quantized()?;
        self.c_proj.cache_quantized()?;
        Ok(())
    }

    fn clear_cache(&self) {
        self.c_attn.clear_cache();
        self.c_proj.clear_cache();
    }
}

/// ReLU²: squared ReLU (BitNet-style activation).
fn relu2(x: &Tensor) -> Result<Tensor> {
    let r = x.relu()?;
    r.sqr()
}

/// FFN: two binary/ternary linears with SiLU or ReLU² in between.
struct FeedForward {
    c_fc: BitLinearLayer,
    c_proj: BitLinearLayer,
    use_relu2: bool,
}

impl FeedForward {
    fn new(config: &OneBitLlmConfig, vb: VarBuilder) -> Result<Self> {
        let c_fc = BitLinearLayer::new(config.hidden_size, config.intermediate_size, config, vb.pp("c_fc"))?;
        let c_proj = BitLinearLayer::new(config.intermediate_size, config.hidden_size, config, vb.pp("c_proj"))?;
        Ok(Self {
            c_fc,
            c_proj,
            use_relu2: config.use_relu2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = if self.use_relu2 {
            relu2(&x)?
        } else {
            candle_nn::ops::silu(&x)?
        };
        self.c_proj.forward(&x)
    }

    fn debug_weight_distributions(&self, prefix: &str) -> Vec<(String, Result<String>)> {
        vec![
            (
                format!("{}.c_fc", prefix),
                self.c_fc.debug_weight_distribution(),
            ),
            (
                format!("{}.c_proj", prefix),
                self.c_proj.debug_weight_distribution(),
            ),
        ]
    }

    fn cache_quantized(&self) -> Result<()> {
        self.c_fc.cache_quantized()?;
        self.c_proj.cache_quantized()?;
        Ok(())
    }

    fn clear_cache(&self) {
        self.c_fc.clear_cache();
        self.c_proj.clear_cache();
    }
}

/// Normalization: LayerNorm or RMSNorm (subln).
enum NormLayer {
    LayerNorm(LayerNorm),
    SubLn(RmsNorm),
}

impl NormLayer {
    fn new(config: &OneBitLlmConfig, vb: VarBuilder) -> Result<Self> {
        if config.use_subln {
            Ok(NormLayer::SubLn(rms_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb,
            )?))
        } else {
            Ok(NormLayer::LayerNorm(layer_norm_no_bias(
                config.hidden_size,
                config.layer_norm_eps,
                vb,
            )?))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            NormLayer::LayerNorm(l) => l.forward(x),
            NormLayer::SubLn(r) => r.forward(x),
        }
    }
}

/// Single decoder block: pre-norm attention + residual, pre-norm FFN + residual; optional Arenas; optional residual scaling.
struct DecoderBlock {
    attn: CausalSelfAttention,
    ln1: NormLayer,
    ff: FeedForward,
    ln2: NormLayer,
    residual_scale: f64,
}

impl DecoderBlock {
    fn new(config: &OneBitLlmConfig, vb: VarBuilder) -> Result<Self> {
        let attn = CausalSelfAttention::new(config, vb.pp("attn"))?;
        let ln1 = NormLayer::new(config, vb.pp("ln1"))?;
        let ff = FeedForward::new(config, vb.pp("mlp"))?;
        let ln2 = NormLayer::new(config, vb.pp("ln2"))?;
        let residual_scale = if config.use_residual_scaling {
            1.0 / 2.0_f64.sqrt()
        } else {
            1.0
        };
        Ok(Self {
            attn,
            ln1,
            ff,
            ln2,
            residual_scale,
        })
    }

    fn forward(&self, x: &Tensor, arenas_coef: Option<f32>) -> Result<Tensor> {
        let block_input = x;
        let residual = x;
        let x = self.ln1.forward(x)?;
        let attn_out = self.attn.forward(&x)?;
        let scaled = attn_out.affine(self.residual_scale, 0.0)?;
        let mut x = (residual + scaled)?;
        if let Some(c) = arenas_coef {
            let device = block_input.device();
            let c_t = Tensor::new(&[c], device)?;
            x = (x + block_input.broadcast_mul(&c_t)?)?;
        }
        let residual = &x;
        let x = self.ln2.forward(&x)?;
        let ff_out = self.ff.forward(&x)?;
        let scaled = ff_out.affine(self.residual_scale, 0.0)?;
        let mut x = (residual + scaled)?;
        if let Some(c) = arenas_coef {
            let device = block_input.device();
            let c_t = Tensor::new(&[c], device)?;
            x = (x + block_input.broadcast_mul(&c_t)?)?;
        }
        Ok(x)
    }

    fn debug_weight_distributions(&self, block_idx: usize) -> Vec<(String, Result<String>)> {
        let prefix = format!("h.{}", block_idx);
        let mut out = self.attn.debug_weight_distributions(&format!("{}.attn", prefix));
        out.extend(self.ff.debug_weight_distributions(&format!("{}.mlp", prefix)));
        out
    }

    fn cache_quantized(&self) -> Result<()> {
        self.attn.cache_quantized()?;
        self.ff.cache_quantized()?;
        Ok(())
    }

    fn clear_cache(&self) {
        self.attn.clear_cache();
        self.ff.clear_cache();
    }
}

/// OneBit-LLM: decoder-only transformer with binary/ternary attention and FFN; optional RoPE, ReLU², subln, Arenas.
pub struct OneBitLlm {
    wte: Embedding,
    blocks: Vec<DecoderBlock>,
    ln_f: NormLayer,
    lm_head: candle_nn::Linear,
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
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
            config: config.clone(),
        })
    }

    /// Forward pass. arenas_coef: when training with Arenas, pass current coefficient (e.g. annealed from arenas_initial to 0); None = disabled.
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.forward_with_arenas(input_ids, None)
    }

    pub fn forward_with_arenas(
        &self,
        input_ids: &Tensor,
        arenas_coef: Option<f32>,
    ) -> Result<Tensor> {
        let x = self.wte.forward(input_ids)?;
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, arenas_coef)?;
        }
        x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn config(&self) -> &OneBitLlmConfig {
        &self.config
    }

    /// Debug: collect -1, 0, +1 (or -1, +1) weight distribution for all bit-linear layers.
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

    /// Pre-compute quantized weights for all bit-linear layers (inference-only). Forward will skip re-quantize.
    pub fn cache_quantized_weights(&self) -> Result<()> {
        for block in &self.blocks {
            block.cache_quantized()?;
        }
        Ok(())
    }

    /// Clear cached quantized weights (e.g. before training).
    pub fn clear_quantized_cache(&self) {
        for block in &self.blocks {
            block.clear_cache();
        }
    }
}

/// Compression stats: total and quantized parameter counts from config (for logging).
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub total_params: usize,
    pub quantized_params: usize,
    /// Effective bits per weight: (quantized * 2 + (total - quantized) * 32) / total (ternary ≈ 2 bit).
    pub effective_bits_per_param: f64,
    /// Compression ratio vs full F32 (32 bit).
    pub compression_ratio_vs_f32: f64,
}

impl OneBitLlmConfig {
    /// Approximate parameter counts and compression stats from config (no model needed).
    pub fn compression_stats(&self) -> CompressionStats {
        let vocab = self.vocab_size;
        let h = self.hidden_size;
        let n = self.num_layers;
        let i = self.intermediate_size;

        let wte = vocab * h;
        let lm_head = h * vocab;
        let per_block_attn = h * (3 * h) + h * h;
        let per_block_ff = h * i + i * h;
        let quantized_per_block = per_block_attn + per_block_ff;

        let norm_params = (n * 2 + 1) * h; // ln1, ln2 per block + ln_f; approx
        let qk_norm = if self.use_qk_norm { n * 2 * h } else { 0 };
        let total = wte + lm_head + n * (per_block_attn + per_block_ff + norm_params) + qk_norm;
        let quantized = n * quantized_per_block;

        let bits_quantized = 2.0; // ternary
        let bits_full = 32.0;
        let effective_bits = (quantized as f64 * bits_quantized + (total.saturating_sub(quantized)) as f64 * bits_full) / total.max(1) as f64;
        let compression_ratio_vs_f32 = bits_full / effective_bits;

        CompressionStats {
            total_params: total,
            quantized_params: quantized,
            effective_bits_per_param: effective_bits,
            compression_ratio_vs_f32,
        }
    }
}
