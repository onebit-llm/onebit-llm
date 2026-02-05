//! Normalisation layers for 1-bit transformers.
//!
//! RMSNorm is strongly preferred for ternary models because it does **not**
//! subtract the mean. Mean subtraction can shift activations across the
//! quantisation threshold, causing "flip-flop" instability.

use candle_core::{Result, Tensor};
use candle_nn::{layer_norm_no_bias, rms_norm, LayerNorm, Module, RmsNorm, VarBuilder};

use ternary_common::OneBitLlmConfig;

/// Normalisation layer: RMSNorm (when `use_subln = true`) or LayerNorm.
pub enum NormLayer {
    LayerNorm(LayerNorm),
    RmsNorm(RmsNorm),
}

impl NormLayer {
    /// Construct from config. `vb` should be scoped to the layer prefix
    /// (e.g. `vb.pp("ln1")`).
    pub fn new(config: &OneBitLlmConfig, vb: VarBuilder) -> Result<Self> {
        if config.use_subln {
            Ok(Self::RmsNorm(rms_norm(
                config.hidden_size,
                config.layer_norm_eps,
                vb,
            )?))
        } else {
            Ok(Self::LayerNorm(layer_norm_no_bias(
                config.hidden_size,
                config.layer_norm_eps,
                vb,
            )?))
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::LayerNorm(l) => l.forward(x),
            Self::RmsNorm(r) => r.forward(x),
        }
    }
}
