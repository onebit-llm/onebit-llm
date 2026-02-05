//! Model configuration for OneBit-LLM.
//!
//! Serialised as JSON for export/import. Every field has a sensible default so
//! a minimal `{}` JSON will produce a working (if small) model.

use serde::{Deserialize, Serialize};

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
            anneal_fraction: 0.3,
            arenas_initial: None,
            arenas_anneal_steps: 10_000,
        }
    }
}

impl OneBitLlmConfig {
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
