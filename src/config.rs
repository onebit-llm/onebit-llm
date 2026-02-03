//! Model configuration for OneBit-LLM.
//! Serialized as JSON for export/import.

use serde::{Deserialize, Serialize};

/// Configuration for the 1-bit decoder-only transformer.
/// Stored alongside weights for reproducible import.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneBitLlmConfig {
    /// Vocabulary size (must match tokenizer).
    pub vocab_size: usize,
    /// Hidden size (model dimension).
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of decoder layers.
    pub num_layers: usize,
    /// FFN intermediate dimension (typically 4 * hidden_size).
    pub intermediate_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Layer norm epsilon.
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,

    // --- BitNet / paper framework options (all default false for backward compat) ---
    /// Use ternary weights {-1,0,+1} with AbsMean instead of binary ±1.
    #[serde(default)]
    pub use_ternary: bool,
    /// Use ReLU² activation instead of SiLU in FFN (BitNet-style).
    #[serde(default)]
    pub use_relu2: bool,
    /// Use RMSNorm (subln) instead of LayerNorm before attention/FFN.
    #[serde(default)]
    pub use_subln: bool,
    /// Use RoPE (rotary position embeddings) in attention.
    #[serde(default)]
    pub use_rope: bool,
    /// QK-norm: RMSNorm on Q and K before attention (Olmo2/LLaMA-style). Stabilizes training, lowers loss.
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    /// Residual scaling: scale sublayer output by 1/sqrt(2) before adding to residual. Improves gradient flow in deep 1-bit nets.
    #[serde(default = "default_true")]
    pub use_residual_scaling: bool,
    /// Dynamic threshold for ternary: Δ = 0.7 × mean(|W|) in original space; |w| <= Δ -> 0, else sign(w). Makes weights flip more easily.
    #[serde(default = "default_true")]
    pub use_dynamic_threshold: bool,
    /// STE scale factor: gradient flowing to latent weights is multiplied by this (>1 strengthens updates so ternary counts can change). Default 2.0.
    #[serde(default = "default_ste_scale")]
    pub ste_scale_factor: f64,
    /// Latent weight clamp: in forward, clamp latent weights to [-latent_clamp_max, +latent_clamp_max] so they stay near threshold. Default 1.5.
    #[serde(default = "default_latent_clamp_max")]
    pub latent_clamp_max: f64,
    /// Arenas: initial coefficient for full-precision residual path (None = disabled). Anneals to 0 over arenas_anneal_steps.
    #[serde(default)]
    pub arenas_initial: Option<f64>,
    /// Number of steps over which Arenas coefficient anneals from arenas_initial to 0 (used only if arenas_initial is set).
    #[serde(default = "default_arenas_anneal_steps")]
    pub arenas_anneal_steps: usize,
}

fn default_arenas_anneal_steps() -> usize {
    10_000
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

fn default_layer_norm_eps() -> f64 {
    1e-5
}

impl Default for OneBitLlmConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257, // GPT-2
            hidden_size: 256,
            num_heads: 8,
            num_layers: 6,
            intermediate_size: 1024,
            max_seq_len: 512,
            layer_norm_eps: 1e-5,
            use_ternary: false,
            use_relu2: false,
            use_subln: false,
            use_rope: false,
            use_qk_norm: true,
            use_residual_scaling: true,
            use_dynamic_threshold: true,
            ste_scale_factor: 2.0,
            latent_clamp_max: 1.5,
            arenas_initial: None,
            arenas_anneal_steps: 10_000,
        }
    }
}

impl OneBitLlmConfig {
    /// Head dimension (hidden_size / num_heads). Panics if not divisible.
    pub fn head_dim(&self) -> usize {
        assert!(
            self.hidden_size.is_multiple_of(self.num_heads),
            "hidden_size must be divisible by num_heads"
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
}

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
}
