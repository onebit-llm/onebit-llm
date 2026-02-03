//! Binary and ternary 1-bit layers with straight-through estimator (STE).
//!
//! Forward: quantize weights (and optionally activations); backward: gradients
//! flow to full-precision parameters via STE. Supports binary ±1 and ternary {-1,0,+1} (AbsMean).

use candle::{Result, Tensor};
use candle_nn::{Init, Linear, Module, VarBuilder};

/// Straight-through estimator for sign: forward = sign(x), backward = identity.
#[inline]
fn ste_sign(x: &Tensor) -> Result<Tensor> {
    let sign_x = x.sign()?;
    let detach_x = x.detach();
    let residual = (x - &detach_x)?;
    Ok((&sign_x + &residual)?)
}

/// Ternary quantization with AbsMean: beta = mean(|W|), quantize to {-1,0,+1} with STE.
/// Forward: scale by 1/beta, round, clamp to [-1,0,+1]; backward: gradient flows to w via STE.
fn ternary_absmean_ste(w: &Tensor) -> Result<Tensor> {
    let abs_w = w.abs()?;
    let beta = abs_w.mean_all()?;
    let beta_s = beta.to_scalar::<f32>()?.max(1e-8);
    let inv_beta = 1.0f64 / beta_s as f64;
    let w_scaled = w.affine(inv_beta, 0.0)?;
    let rounded = w_scaled.round()?;
    let clamped = rounded.clamp(-1f64, 1f64)?;
    // STE: forward value = clamped, backward = identity so dL/dw = dL/d_out
    Ok((clamped.detach() + (w - &w.detach())?)?)
}

/// Binary linear layer: forward uses sign(W) and sign(x) with STE; no bias.
pub struct BinaryLinear {
    weight: Linear,
}

/// Small init for 1-bit layers so more weights land in zero band (ternary) or don't all saturate (binary).
const BIT_LAYER_INIT: Init = Init::Randn {
    mean: 0.,
    stdev: 0.02,
};

impl BinaryLinear {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let ws = vb.get_with_hints((out_dim, in_dim), "weight", BIT_LAYER_INIT)?;
        let weight = Linear::new(ws, None);
        Ok(Self { weight })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.weight.weight();
        let in_dim = w.dim(1)?;
        let w_bin = ste_sign(w)?;
        let x_bin = ste_sign(x)?;
        let out = matmul_reshape(&x_bin, &w_bin.t()?)?;
        // Weight scaling (γ): preserve output energy; γ = mean(|W|) balances quantized layer.
        let gamma = w.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
        let scale = (1.0 / (in_dim as f64).sqrt()) * gamma;
        out.affine(scale, 0.0)
    }
}

impl Module for BinaryLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

/// Ternary linear layer (BitNet-style): AbsMean ternary weights, STE; activations kept full-precision in forward (W1.58A8 inference path can add 8-bit later).
pub struct TernaryLinear {
    weight: Linear,
}

impl TernaryLinear {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let ws = vb.get_with_hints((out_dim, in_dim), "weight", BIT_LAYER_INIT)?;
        let weight = Linear::new(ws, None);
        Ok(Self { weight })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.weight.weight();
        let in_dim = w.dim(1)?;
        let w_ternary = ternary_absmean_ste(w)?;
        // Activations: full-precision in training; optional 8-bit AbsMax for inference later.
        let out = matmul_reshape(x, &w_ternary.t()?)?;
        // Weight scaling (γ): mean(|W|) so quantized layer output energy is balanced.
        let gamma = w.abs()?.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
        let scale = (1.0 / (in_dim as f64).sqrt()) * gamma;
        out.affine(scale, 0.0)
    }
}

impl Module for TernaryLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }
}

/// Reshape x to 2D, matmul with w_t, reshape back to original leading dims.
fn matmul_reshape(x: &Tensor, w_t: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let out_dim = w_t.dim(1)?;
    match dims.as_ref() {
        [b, m, k] => {
            let x_2d = x.reshape((*b * *m, *k))?;
            let y = x_2d.matmul(w_t)?;
            y.reshape((*b, *m, out_dim))
        }
        [b1, b2, m, k] => {
            let x_2d = x.reshape((*b1 * *b2 * *m, *k))?;
            let y = x_2d.matmul(w_t)?;
            y.reshape((*b1, *b2, *m, out_dim))
        }
        _ => {
            let last = dims[dims.len() - 1];
            let prod: usize = dims[..dims.len() - 1].iter().product();
            let x_2d = x.reshape((prod, last))?;
            let y = x_2d.matmul(w_t)?;
            let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
            out_shape.push(out_dim);
            y.reshape(out_shape.as_slice())
        }
    }
}
