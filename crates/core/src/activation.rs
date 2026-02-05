//! Activation functions for 1-bit FFN layers.
//!
//! Three options, selectable via [`OneBitLlmConfig`]:
//!
//! | Activation | Formula | Notes |
//! |------------|---------|-------|
//! | SiLU       | x · σ(x) | Default, works well with STE. |
//! | ReLU²      | max(0, x)² | BitNet-style, amplifies strong signals. |
//! | SwiGLU     | SiLU(gate) ⊙ up | LLaMA/Mistral-style, 3 projections. |
//!
//! SwiGLU is implemented at the FFN level (it requires two separate
//! linear projections to produce `gate` and `up`). This module provides
//! the pointwise helper and the ReLU² function.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use ternary_common::{FfnActivation, OneBitLlmConfig};

use crate::linear::BitLinearLayer;

/// ReLU²: `max(0, x)²`. Amplifies large positive activations.
#[inline]
pub fn relu_squared(x: &Tensor) -> Result<Tensor> {
    x.relu()?.sqr()
}

// ── Feed-Forward Networks ───────────────────────────────────────────────────

/// Standard 2-projection FFN (SiLU or ReLU²):
///
/// ```text
/// out = W_down( activation( W_up(x) ) )
/// ```
pub struct FeedForward {
    w_up: BitLinearLayer,
    w_down: BitLinearLayer,
    activation: FfnActivation,
}

/// SwiGLU 3-projection FFN:
///
/// ```text
/// gate = SiLU( W_gate(x) )
/// up   = W_up(x)
/// out  = W_down( gate ⊙ up )
/// ```
///
/// The gating mechanism allows the network to learn which features to
/// amplify and which to suppress, crucial for maintaining signal integrity
/// through ternary layers.
pub struct SwiGLUFeedForward {
    w_gate: BitLinearLayer,
    w_up: BitLinearLayer,
    w_down: BitLinearLayer,
}

/// Unified FFN: dispatches to either [`FeedForward`] or [`SwiGLUFeedForward`].
pub enum FfnLayer {
    Standard(FeedForward),
    SwiGLU(SwiGLUFeedForward),
}

impl FfnLayer {
    /// Construct from config.
    ///
    /// * Standard FFN: 2 projections (`c_fc`, `c_proj`).
    /// * SwiGLU FFN: 3 projections (`w_gate`, `w_up`, `w_down`).
    pub fn new(config: &OneBitLlmConfig, vb: VarBuilder) -> Result<Self> {
        match config.ffn_activation() {
            FfnActivation::SwiGLU => {
                let w_gate = BitLinearLayer::new(
                    config.hidden_size,
                    config.intermediate_size,
                    config,
                    vb.pp("w_gate"),
                )?;
                let w_up = BitLinearLayer::new(
                    config.hidden_size,
                    config.intermediate_size,
                    config,
                    vb.pp("w_up"),
                )?;
                let w_down = BitLinearLayer::new(
                    config.intermediate_size,
                    config.hidden_size,
                    config,
                    vb.pp("w_down"),
                )?;
                Ok(Self::SwiGLU(SwiGLUFeedForward {
                    w_gate,
                    w_up,
                    w_down,
                }))
            }
            activation => {
                // Standard 2-projection FFN (SiLU or ReLU²)
                let w_up = BitLinearLayer::new(
                    config.hidden_size,
                    config.intermediate_size,
                    config,
                    vb.pp("c_fc"),
                )?;
                let w_down = BitLinearLayer::new(
                    config.intermediate_size,
                    config.hidden_size,
                    config,
                    vb.pp("c_proj"),
                )?;
                Ok(Self::Standard(FeedForward {
                    w_up,
                    w_down,
                    activation,
                }))
            }
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Standard(ff) => ff.forward(x),
            Self::SwiGLU(ff) => ff.forward(x),
        }
    }

    /// Debug weight distributions for all linear layers in this FFN.
    pub fn debug_weight_distributions(&self, prefix: &str) -> Vec<(String, Result<String>)> {
        match self {
            Self::Standard(ff) => vec![
                (format!("{prefix}.c_fc"), ff.w_up.debug_weight_distribution()),
                (
                    format!("{prefix}.c_proj"),
                    ff.w_down.debug_weight_distribution(),
                ),
            ],
            Self::SwiGLU(ff) => vec![
                (
                    format!("{prefix}.w_gate"),
                    ff.w_gate.debug_weight_distribution(),
                ),
                (format!("{prefix}.w_up"), ff.w_up.debug_weight_distribution()),
                (
                    format!("{prefix}.w_down"),
                    ff.w_down.debug_weight_distribution(),
                ),
            ],
        }
    }

    pub fn cache_quantized(&self) -> Result<()> {
        match self {
            Self::Standard(ff) => {
                ff.w_up.cache_quantized()?;
                ff.w_down.cache_quantized()
            }
            Self::SwiGLU(ff) => {
                ff.w_gate.cache_quantized()?;
                ff.w_up.cache_quantized()?;
                ff.w_down.cache_quantized()
            }
        }
    }

    pub fn clear_cache(&self) {
        match self {
            Self::Standard(ff) => {
                ff.w_up.clear_cache();
                ff.w_down.clear_cache();
            }
            Self::SwiGLU(ff) => {
                ff.w_gate.clear_cache();
                ff.w_up.clear_cache();
                ff.w_down.clear_cache();
            }
        }
    }
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.w_up.forward(x)?;
        let h = match self.activation {
            FfnActivation::ReLU2 => relu_squared(&h)?,
            _ => candle_nn::ops::silu(&h)?,
        };
        self.w_down.forward(&h)
    }
}

impl SwiGLUFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w_gate.forward(x)?)?;
        let up = self.w_up.forward(x)?;
        let activated = (gate * up)?;
        self.w_down.forward(&activated)
    }
}
