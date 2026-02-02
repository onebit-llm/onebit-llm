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
    fn new(in_dim: usize, out_dim: usize, use_ternary: bool, vb: VarBuilder) -> Result<Self> {
        if use_ternary {
            Ok(BitLinearLayer::Ternary(TernaryLinear::new(in_dim, out_dim, vb)?))
        } else {
            Ok(BitLinearLayer::Binary(BinaryLinear::new(in_dim, out_dim, vb)?))
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            BitLinearLayer::Binary(l) => l.forward(x),
            BitLinearLayer::Ternary(l) => l.forward(x),
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

/// Causal self-attention with binary/ternary Q,K,V; optional RoPE on q,k.
struct CausalSelfAttention {
    c_attn: BitLinearLayer,
    c_proj: BitLinearLayer,
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
        let c_attn = BitLinearLayer::new(hidden, 3 * hidden, config.use_ternary, vb.pp("c_attn"))?;
        let c_proj = BitLinearLayer::new(hidden, hidden, config.use_ternary, vb.pp("c_proj"))?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        Ok(Self {
            c_attn,
            c_proj,
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
        let c_fc = BitLinearLayer::new(
            config.hidden_size,
            config.intermediate_size,
            config.use_ternary,
            vb.pp("c_fc"),
        )?;
        let c_proj = BitLinearLayer::new(
            config.intermediate_size,
            config.hidden_size,
            config.use_ternary,
            vb.pp("c_proj"),
        )?;
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

/// Single decoder block: pre-norm attention + residual, pre-norm FFN + residual; optional Arenas residual.
struct DecoderBlock {
    attn: CausalSelfAttention,
    ln1: NormLayer,
    ff: FeedForward,
    ln2: NormLayer,
}

impl DecoderBlock {
    fn new(config: &OneBitLlmConfig, vb: VarBuilder) -> Result<Self> {
        let attn = CausalSelfAttention::new(config, vb.pp("attn"))?;
        let ln1 = NormLayer::new(config, vb.pp("ln1"))?;
        let ff = FeedForward::new(config, vb.pp("mlp"))?;
        let ln2 = NormLayer::new(config, vb.pp("ln2"))?;
        Ok(Self { attn, ln1, ff, ln2 })
    }

    fn forward(&self, x: &Tensor, arenas_coef: Option<f32>) -> Result<Tensor> {
        let block_input = x;
        let residual = x;
        let x = self.ln1.forward(x)?;
        let x = self.attn.forward(&x)?;
        let mut x = (x + residual)?;
        if let Some(c) = arenas_coef {
            let device = block_input.device();
            let c_t = Tensor::new(&[c], device)?;
            x = (x + block_input.broadcast_mul(&c_t)?)?;
        }
        let residual = &x;
        let x = self.ln2.forward(&x)?;
        let x = self.ff.forward(&x)?;
        let mut x = (x + residual)?;
        if let Some(c) = arenas_coef {
            let device = block_input.device();
            let c_t = Tensor::new(&[c], device)?;
            x = (x + block_input.broadcast_mul(&c_t)?)?;
        }
        Ok(x)
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
}
