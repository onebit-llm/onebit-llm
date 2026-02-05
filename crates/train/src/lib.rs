//! # ternary-train — The Stability Engine
//!
//! Training loop, optimiser, and scheduling for 1-bit LLMs:
//!
//! * **[`Trainer`]** — owns model + optimiser + schedules. One call to
//!   [`Trainer::step`] runs forward, backward, gradient clipping, AdamW,
//!   latent clamping, and schedule advancement.
//! * **[`LrScheduler`]** — warmup → cosine / linear / constant.
//! * **[`AnnealSchedule`]** — soft→hard quantisation annealing.

pub mod scheduler;
pub mod trainer;

pub use scheduler::{AnnealSchedule, LrDecay, LrScheduler};
pub use trainer::{StepMetrics, Trainer, TrainerConfig};
