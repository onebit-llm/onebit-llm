//! Core types for quantization search

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantization configuration for a single layer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantLevel {
    Binary,   // 1-bit
    Ternary,  // 1.58-bit
    FourBit,  // 4-bit
    EightBit, // 8-bit
    Float16,  // 16-bit
}

impl QuantLevel {
    pub fn bits(&self) -> f32 {
        match self {
            Self::Binary => 1.0,
            Self::Ternary => 1.58,
            Self::FourBit => 4.0,
            Self::EightBit => 8.0,
            Self::Float16 => 16.0,
        }
    }

    pub fn all_levels() -> Vec<Self> {
        vec![
            Self::Binary,
            Self::Ternary,
            Self::FourBit,
            Self::EightBit,
            Self::Float16,
        ]
    }
}

/// Complete quantization configuration for entire model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    /// Layer index → quantization level
    pub layer_quant: HashMap<usize, QuantLevel>,
    /// Model configuration reference
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
            .unwrap_or(QuantLevel::Float16) // Default: no quantization
    }

    /// Calculate total model size in MB
    pub fn total_size_mb(&self, params_per_layer: &[usize]) -> f64 {
        let mut total_bits = 0.0;
        for (layer, &params) in params_per_layer.iter().enumerate().take(self.num_layers) {
            let level = self.get_layer(layer);
            total_bits += params as f64 * level.bits() as f64;
        }
        total_bits / (8.0 * 1024.0 * 1024.0) // bits → MB
    }

    /// Calculate compression ratio vs FP32
    pub fn compression_ratio(&self, params_per_layer: &[usize]) -> f64 {
        let fp32_size = params_per_layer.iter().sum::<usize>() as f64 * 32.0;
        let quant_size: f64 = (0..self.num_layers)
            .map(|i| params_per_layer[i] as f64 * self.get_layer(i).bits() as f64)
            .sum();
        fp32_size / quant_size
    }
}

/// Search result with metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub config: QuantConfig,
    pub accuracy: f64, // Validation accuracy or negative loss
    pub perplexity: f64,
    pub size_mb: f64,
    pub compression_ratio: f64,
    pub search_time_secs: f64,
    pub evaluations: usize, // Number of configs evaluated
}

/// Search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Model config path
    pub model_config: String,
    /// Checkpoint path
    pub checkpoint: String,
    /// Validation data path
    pub val_data: String,
    /// Tokenizer path (e.g. tokenizer.json) for validation dataset
    pub tokenizer: String,

    /// Search constraints
    pub max_size_mb: Option<f64>,
    pub min_accuracy: Option<f64>,
    pub max_evaluations: usize,

    /// Expander parameters
    pub partition_size: usize, // Target partition size (√V typically)
    pub overlap_ratio: f64, // Partition overlap (0.1-0.2)

    /// Parallelism
    pub num_threads: usize,

    /// Output
    pub output_path: String,
}
