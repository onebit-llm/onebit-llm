//! OneBit-LLM: 1-bit decoder-only LLM with binary/XNOR-style layers.
//!
//! Provides config, binary layers, transformer model, and data pipeline
//! for training, export, and import.

pub mod config;
pub mod binary;
pub mod model;
pub mod data;

pub use config::OneBitLlmConfig;
pub use binary::{BinaryLinear, TernaryLinear};
pub use model::OneBitLlm;
pub use data::{batch_to_tensors, StreamingBatchIter, TextDataset};
