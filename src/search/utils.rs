//! Helper functions for quantization search

use super::types::QuantConfig;

/// Generate a unique string key for a quantization config (for caching / dedup).
pub fn config_key(config: &QuantConfig) -> String {
    (0..config.num_layers)
        .map(|i| format!("{:?}", config.get_layer(i)))
        .collect::<Vec<_>>()
        .join("-")
}
