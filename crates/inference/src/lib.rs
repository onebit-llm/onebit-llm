//! # ternary-infer — Lightweight Inference Runtime
//!
//! * **[`Sampler`]** — top-k, top-p, temperature, repetition penalty.
//! * **[`InferenceRuntime`]** — load model + generate tokens.
//! * **[`export`]** — `.1bit` binary format with C-compatible header.

pub mod export;
pub mod runtime;
pub mod sampler;

pub use export::{export_1bit, generate_c_header, PackMode};
pub use runtime::InferenceRuntime;
pub use sampler::{Sampler, SamplerConfig};
