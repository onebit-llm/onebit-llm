//! Core types for quantisation search.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Per-layer quantisation level. Search space: Binary or Ternary only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantLevel {
    Binary,  // 1-bit
    Ternary, // 1.58-bit
}

impl QuantLevel {
    pub fn bits(&self) -> f32 {
        match self {
            Self::Binary => 1.0,
            Self::Ternary => 1.58,
        }
    }
    pub fn all_levels() -> Vec<Self> {
        vec![Self::Binary, Self::Ternary]
    }
}

/// Complete layer-wise quantisation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    pub layer_quant: HashMap<usize, QuantLevel>,
    pub num_layers: usize,
}

impl QuantConfig {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layer_quant: HashMap::new(),
            num_layers,
        }
    }
    pub fn set_layer(&mut self, layer: usize, level: QuantLevel) {
        self.layer_quant.insert(layer, level);
    }
    pub fn get_layer(&self, layer: usize) -> QuantLevel {
        self.layer_quant
            .get(&layer)
            .copied()
            .unwrap_or(QuantLevel::Ternary)
    }
    pub fn total_size_mb(&self, params_per_layer: &[usize]) -> f64 {
        let mut total_bits = 0.0;
        for (layer, &params) in params_per_layer.iter().enumerate().take(self.num_layers) {
            total_bits += params as f64 * self.get_layer(layer).bits() as f64;
        }
        total_bits / (8.0 * 1024.0 * 1024.0)
    }
    pub fn compression_ratio(&self, params_per_layer: &[usize]) -> f64 {
        let fp32_size = params_per_layer.iter().sum::<usize>() as f64 * 32.0;
        let quant_size: f64 = (0..self.num_layers)
            .map(|i| params_per_layer[i] as f64 * self.get_layer(i).bits() as f64)
            .sum();
        fp32_size / quant_size
    }
}

/// Search result with all metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub config: QuantConfig,
    pub accuracy: f64,
    pub perplexity: f64,
    pub size_mb: f64,
    pub compression_ratio: f64,
    pub search_time_secs: f64,
    pub evaluations: usize,
    pub total_evaluations: usize,
    pub valid_partitions: usize,
}

/// Search configuration (CLI-level knobs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub model_config: String,
    pub checkpoint: String,
    pub val_data: String,
    pub tokenizer: String,
    pub max_size_mb: Option<f64>,
    pub min_accuracy: Option<f64>,
    pub max_evaluations: usize,
    pub partition_size: usize,
    pub overlap_ratio: f64,
    pub num_threads: usize,
    pub output_path: String,
}

/// Unique key for a `QuantConfig` (for caching / dedup).
pub fn config_key(config: &QuantConfig) -> String {
    (0..config.num_layers)
        .map(|i| format!("{:?}", config.get_layer(i)))
        .collect::<Vec<_>>()
        .join("-")
}
