//! # ternary-core — The Mathematical Engine
//!
//! Every compute primitive needed to build, train, and run a 1-bit / 1.58-bit
//! LLM lives in this crate:
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`quantize`] | STE primitives, soft→hard annealing, ternary/binary quantise |
//! | [`linear`] | `BinaryLinear`, `TernaryLinear`, `BitLinearLayer` |
//! | [`norm`] | `NormLayer` (RMSNorm / LayerNorm) |
//! | [`activation`] | `relu_squared`, `SwiGLUFeedForward`, `FfnLayer` |
//! | [`attention`] | `CausalSelfAttention` with RoPE and QK-Norm |
//! | [`model`] | `OneBitLlm` (full transformer), `CompressionStats` |
//!
//! ## Design principles
//!
//! 1. **Pure Rust hot path.** Everything goes through `candle-core`/`candle-nn`.
//!    Compiles to CPU, CUDA, Metal, and (eventually) WASM.
//! 2. **`Send + Sync`-safe.** Inference caches use `parking_lot::Mutex`, not `RefCell`.
//! 3. **Deterministic.** Same inputs + same annealing fraction = same output.

pub mod activation;
pub mod attention;
pub mod linear;
pub mod model;
pub mod norm;
pub mod quantize;

// ── Public re-exports ───────────────────────────────────────────────────────

pub use linear::{BinaryLinear, BitLinearLayer, TernaryLinear};
pub use model::{compression_stats, CompressionStats, OneBitLlm};
pub use quantize::{
    current_anneal_frac, set_quant_anneal_frac, ste_sign_scaled, ste_tanh_scaled,
    ternary_quantize_forward,
};
