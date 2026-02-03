//! OneBit-LLM: 1-bit decoder-only LLM with binary/XNOR-style layers.
//!
//! Provides config, binary layers, transformer model, and data pipeline
//! for training, export, and import.

pub mod binary;
pub mod config;
pub mod data;
pub mod model;

pub use binary::{ternary_quantize_forward, BinaryLinear, TernaryLinear};
pub use config::OneBitLlmConfig;
pub use data::{batch_to_tensors, StreamingBatchIter, TextDataset};
pub use model::{CompressionStats, OneBitLlm};
