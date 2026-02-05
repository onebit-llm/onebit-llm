//! # ternary-common — Shared Primitives
//!
//! Types and utilities shared across every crate in the workspace:
//!
//! * **[`OneBitLlmConfig`]** — model hyper-parameters (serialised as JSON).
//! * **[`FfnActivation`]** — resolved FFN activation choice.
//! * **[`TextDataset`]** / **[`StreamingBatchIter`]** — data loading & batching.
//! * **[`batch_to_tensors`]** — raw batch → Candle tensors.

pub mod config;
pub mod data;

pub use config::{FfnActivation, OneBitLlmConfig};
pub use data::{batch_to_tensors, StreamingBatchIter, TextDataset};
