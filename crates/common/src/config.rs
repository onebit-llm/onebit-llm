//! Model configuration for Ternary-Core.
//!
//! Serialised as JSON for export/import. Every field has a sensible default so
//! a minimal `{}` JSON will produce a working (if small) model.
//!
//! **Sandwich Rule:** Use [`QuantMode`] and [`LayerBitMap`] to keep embedding
//! and LM head in high precision (F16 / EightBit) while middle layers use
//! Ternary or Binary.

use serde::{Deserialize, Serialize};

// ── QuantMode (Mixed Precision) ─────────────────────────────────────────────

/// Per-layer quantization mode for the "Sandwich Rule".
///
/// * **F16** — Full float16 (or f32 in candle); use for embedding and LM head.
/// * **EightBit** — 8-bit quantization; optional middle ground.
/// * **Ternary** — 1.58-bit {-1, 0, +1}; default for hidden layers.
/// * **Binary** — 1-bit ±1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantMode {
    F16,
    EightBit,
    Ternary,
    Binary,
}

impl Default for QuantMode {
    fn default() -> Self {
        QuantMode::Ternary
    }
}

impl std::fmt::Display for QuantMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantMode::F16 => write!(f, "f16"),
            QuantMode::EightBit => write!(f, "8bit"),
            QuantMode::Ternary => write!(f, "ternary"),
            QuantMode::Binary => write!(f, "binary"),
        }
    }
}

/// Bit-map: which bit-width to use for embedding, each decoder layer, and lm_head.
///
/// Produced by the search crate (min-perplexity constraint) and consumed by
/// training and inference. **Pinned:** embedding and lm_head are typically
/// F16 or EightBit to avoid information collapse.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerBitMap {
    /// Quantization mode for the token embedding (and tied lm_head).
    #[serde(default)]
    pub embedding: QuantMode,
    /// Per-decoder-layer mode (length = num_layers). Applies to both attn and FFN in that block.
    #[serde(default)]
    pub layer_modes: Vec<QuantMode>,
    /// Explicit lm_head mode; if None, uses same as embedding (weight-tied).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lm_head: Option<QuantMode>,
}

impl LayerBitMap {
    /// Sandwich default: embedding and lm_head F16, all hidden layers Ternary.
    pub fn sandwich_default(num_layers: usize) -> Self {
        Self {
            embedding: QuantMode::F16,
            lm_head: None,
            layer_modes: std::iter::repeat(QuantMode::Ternary).take(num_layers).collect(),
        }
    }

    /// Mode for the LM head (tied to embedding if None).
    pub fn lm_head_mode(&self) -> QuantMode {
        self.lm_head.unwrap_or(self.embedding)
    }

    /// Mode for decoder layer `i` (0..num_layers).
    pub fn layer_mode(&self, i: usize) -> QuantMode {
        self.layer_modes.get(i).copied().unwrap_or(QuantMode::Ternary)
    }
}

/// Configuration for the 1-bit decoder-only transformer.
///
/// Stored alongside weights for reproducible import. Backwards-compatible:
/// missing fields fall back to their `#[serde(default)]` values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneBitLlmConfig {
    // ── Core dimensions ─────────────────────────────────────────────────────
    /// Vocabulary size (must match tokeniser).
    pub vocab_size: usize,
    /// Hidden size (model dimension d_model).
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of decoder layers.
    pub num_layers: usize,
    /// FFN intermediate dimension.
    ///
    /// * When `use_swiglu = false`: 2 projections, total = 2 * hidden * intermediate.
    /// * When `use_swiglu = true`:  3 projections (gate, up, down), total = 3 * hidden * intermediate.
    ///   Set this to `(4 * hidden_size * 2) / 3` rounded to the nearest multiple of 256
    ///   for equivalent capacity to a 4×-expanded ReLU² FFN.
    pub intermediate_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Layer norm / RMSNorm epsilon.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,

    // ── Architecture switches ───────────────────────────────────────────────
    /// Use ternary weights {-1,0,+1} (AbsMean) instead of binary ±1.
    #[serde(default)]
    pub use_ternary: bool,
    /// Use ReLU² activation in FFN (BitNet-style). Ignored when `use_swiglu = true`.
    #[serde(default)]
    pub use_relu2: bool,
    /// **NEW (v0.2):** Use SwiGLU activation in FFN (LLaMA/Mistral-style).
    /// Takes precedence over `use_relu2`. When true the FFN has 3 bit-linear
    /// projections (gate, up, down) instead of 2.
    #[serde(default)]
    pub use_swiglu: bool,
    /// Use RMSNorm (sub-layer norm) instead of LayerNorm.
    #[serde(default)]
    pub use_subln: bool,
    /// Use Rotary Position Embeddings in attention.
    #[serde(default)]
    pub use_rope: bool,
    /// QK-norm: RMSNorm on Q and K before attention (Olmo2/LLaMA-style).
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    /// Scale sublayer output by 1/√2 before adding to residual.
    #[serde(default = "default_true")]
    pub use_residual_scaling: bool,
    /// Dynamic threshold for ternary: δ = 0.7 × mean(|W|).
    #[serde(default = "default_true")]
    pub use_dynamic_threshold: bool,

    // ── Quantisation tuning ─────────────────────────────────────────────────
    /// STE gradient multiplier (>1 strengthens latent weight updates).
    #[serde(default = "default_ste_scale")]
    pub ste_scale_factor: f64,
    /// Latent weight clamp bound (forward clamps to [-C, +C]).
    #[serde(default = "default_latent_clamp_max")]
    pub latent_clamp_max: f64,
    /// Latent weight clamp after each optimizer step (training only). Default 1.2.
    #[serde(default = "default_latent_clip_training")]
    pub latent_clip_max_training: f64,

    // ── Annealing ───────────────────────────────────────────────────────────
    /// **NEW (v0.2):** Fraction of total training steps spent in the soft
    /// annealing regime (tanh → sign). Default 0.3 = first 30%.
    #[serde(default = "default_anneal_fraction")]
    pub anneal_fraction: f32,

    // ── Arenas residual ─────────────────────────────────────────────────────
    /// Arenas initial coefficient (None = disabled). Anneals to 0.
    #[serde(default)]
    pub arenas_initial: Option<f64>,
    /// Steps over which Arenas coefficient anneals to 0.
    #[serde(default = "default_arenas_anneal_steps")]
    pub arenas_anneal_steps: usize,

    // ── Mixed precision (Sandwich Rule) ────────────────────────────────────
    /// Optional per-layer bit map. If None, uses global `use_ternary` and
    /// embedding/lm_head are treated as F16 when sandwich rule is desired.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_bit_map: Option<LayerBitMap>,
}

// ── Default value functions ─────────────────────────────────────────────────

fn default_layer_norm_eps() -> f64 {
    1e-5
}
fn default_true() -> bool {
    true
}
fn default_ste_scale() -> f64 {
    2.0
}
fn default_latent_clamp_max() -> f64 {
    1.5
}
fn default_latent_clip_training() -> f64 {
    1.2
}
fn default_anneal_fraction() -> f32 {
    0.3
}
fn default_arenas_anneal_steps() -> usize {
    10_000
}

// ── Impl ────────────────────────────────────────────────────────────────────

impl Default for OneBitLlmConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257, // GPT-2 tokeniser
            hidden_size: 256,
            num_heads: 8,
            num_layers: 6,
            intermediate_size: 1024,
            max_seq_len: 512,
            layer_norm_eps: 1e-5,
            use_ternary: false,
            use_relu2: false,
            use_swiglu: false,
            use_subln: false,
            use_rope: false,
            use_qk_norm: true,
            use_residual_scaling: true,
            use_dynamic_threshold: true,
            ste_scale_factor: 2.0,
            latent_clamp_max: 1.5,
            latent_clip_max_training: 1.2,
            anneal_fraction: 0.3,
            arenas_initial: None,
            arenas_anneal_steps: 10_000,
            layer_bit_map: None,
        }
    }
}

impl OneBitLlmConfig {
    /// Resolve embedding/lm_head QuantMode (Sandwich: F16 when using layer_bit_map).
    pub fn embedding_quant_mode(&self) -> QuantMode {
        self.layer_bit_map
            .as_ref()
            .map(|m| m.embedding)
            .unwrap_or(QuantMode::F16)
    }

    /// Resolve QuantMode for decoder layer `i` (0..num_layers).
    pub fn decoder_layer_quant_mode(&self, i: usize) -> QuantMode {
        self.layer_bit_map
            .as_ref()
            .map(|m| m.layer_mode(i))
            .unwrap_or(if self.use_ternary {
                QuantMode::Ternary
            } else {
                QuantMode::Binary
            })
    }

    /// Head dimension (`hidden_size / num_heads`). Panics if not divisible.
    pub fn head_dim(&self) -> usize {
        assert!(
            self.hidden_size % self.num_heads == 0,
            "hidden_size ({}) must be divisible by num_heads ({})",
            self.hidden_size,
            self.num_heads,
        );
        self.hidden_size / self.num_heads
    }

    /// Save config to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load config from a JSON file.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let config = serde_json::from_str(&json)?;
        Ok(config)
    }

    /// Resolve which FFN activation to use (priority: swiglu > relu2 > silu).
    pub fn ffn_activation(&self) -> FfnActivation {
        if self.use_swiglu {
            FfnActivation::SwiGLU
        } else if self.use_relu2 {
            FfnActivation::ReLU2
        } else {
            FfnActivation::SiLU
        }
    }
}

/// Resolved FFN activation choice (avoids checking multiple booleans everywhere).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FfnActivation {
    SiLU,
    ReLU2,
    SwiGLU,
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_json_round_trip() {
        let config = OneBitLlmConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let loaded: OneBitLlmConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config.vocab_size, loaded.vocab_size);
        assert_eq!(config.hidden_size, loaded.hidden_size);
        assert_eq!(config.num_heads, loaded.num_heads);
        assert_eq!(config.num_layers, loaded.num_layers);
        assert_eq!(config.max_seq_len, loaded.max_seq_len);
        assert!(!loaded.use_swiglu);
        assert_eq!(loaded.anneal_fraction, 0.3);
    }

    #[test]
    fn config_head_dim() {
        let config = OneBitLlmConfig {
            hidden_size: 256,
            num_heads: 8,
            ..Default::default()
        };
        assert_eq!(config.head_dim(), 32);
    }

    #[test]
    fn backward_compat_missing_fields() {
        // A JSON from the old format (no use_swiglu, no anneal_fraction)
        let old_json = r#"{
            "vocab_size": 50257,
            "hidden_size": 512,
            "num_heads": 8,
            "num_layers": 6,
            "intermediate_size": 2048,
            "max_seq_len": 512,
            "use_ternary": true,
            "use_relu2": true,
            "use_subln": true,
            "use_rope": true
        }"#;
        let loaded: OneBitLlmConfig = serde_json::from_str(old_json).unwrap();
        // New fields should default correctly
        assert!(!loaded.use_swiglu);
        assert_eq!(loaded.anneal_fraction, 0.3);
        assert_eq!(loaded.latent_clamp_max, 1.5);
        assert_eq!(loaded.ste_scale_factor, 2.0);
    }

    #[test]
    fn ffn_activation_priority() {
        let mut c = OneBitLlmConfig::default();
        assert_eq!(c.ffn_activation(), FfnActivation::SiLU);

        c.use_relu2 = true;
        assert_eq!(c.ffn_activation(), FfnActivation::ReLU2);

        c.use_swiglu = true; // SwiGLU takes precedence
        assert_eq!(c.ffn_activation(), FfnActivation::SwiGLU);
    }
}
