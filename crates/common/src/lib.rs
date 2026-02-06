//! # ternary-common — Shared Primitives
//!
//! Types and utilities shared across every crate in the workspace:
//!
//! * **[`OneBitLlmConfig`]** — model hyper-parameters (serialised as JSON).
//! * **[`FfnActivation`]** — resolved FFN activation choice.
//! * **[`TextDataset`]** / **[`StreamingBatchIter`]** / **[`MmapDataset`]** — data loading & batching.
//! * **[`BatchDataset`]** — common trait for in-memory and mmap datasets.
//! * **[`batch_to_tensors`]** / **[`write_tokenized_file`]** — raw batch → tensors; save tokenized.

pub mod config;
pub mod data;

pub use config::{FfnActivation, OneBitLlmConfig};
pub use data::{
    batch_to_tensors, write_tokenized_file, AnyBatchDataset, BatchDataset, MmapDataset,
    StreamingBatchIter, TextDataset,
};
