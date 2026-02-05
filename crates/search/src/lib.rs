//! # ternary-search — The Optimisation Engine
//!
//! Find the optimal layer-wise bit-width map (Binary vs Ternary) for a
//! trained model without retraining.

pub mod coordinator;
pub mod evaluator;
pub mod expander;
pub mod graph;
pub mod types;

pub use coordinator::SearchCoordinator;
pub use graph::{GraphBuilder, QuantGraph};
pub use types::{config_key, QuantConfig, QuantLevel, SearchConfig, SearchResult};
