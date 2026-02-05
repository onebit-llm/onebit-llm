//! Quantisation primitives: STE, annealing, ternary/binary quantise.
//!
//! This module is the mathematical foundation of the framework. Every
//! `BitLinear` layer delegates to these functions for the forward-pass
//! quantisation and the backward-pass gradient routing.
//!
//! # Global annealing schedule
//!
//! All `BitLinear` layers in the process share a single annealing fraction
//! stored in an [`AtomicU32`]. This is intentional: layers must anneal in
//! lockstep, otherwise gradient scale mismatches between soft and hard layers
//! destabilise training.

use std::sync::atomic::{AtomicU32, Ordering};

use candle_core::{DType, Result, Tensor};

// ── Global annealing state ──────────────────────────────────────────────────

/// Annealing fraction stored as millis: 0 → fully soft, 1000 → fully hard.
static ANNEAL_FRAC_MILLIS: AtomicU32 = AtomicU32::new(1000);

/// Set the global annealing fraction in \[0, 1\].
///
/// * `0.0` = fully soft (`tanh(α·x)` approximation).
/// * `1.0` = fully hard (`sign(x)` with STE).
///
/// Called by the trainer at every step. Thread-safe (relaxed atomic).
pub fn set_quant_anneal_frac(frac: f32) {
    let clamped = frac.clamp(0.0, 1.0);
    let clamped = if clamped.is_nan() { 1.0 } else { clamped };
    let v = (clamped * 1000.0) as u32;
    ANNEAL_FRAC_MILLIS.store(v.min(1000), Ordering::Relaxed);
}

/// Read the current annealing fraction as f64 in \[0.0, 1.0\].
#[inline]
pub fn current_anneal_frac() -> f64 {
    ANNEAL_FRAC_MILLIS.load(Ordering::Relaxed) as f64 / 1000.0
}

// ── STE primitives ──────────────────────────────────────────────────────────

/// Hard STE: `forward ≈ sign(x)`, `backward = scale · identity`.
///
/// The trick: `sign(x) + (x - x.detach()) * scale`.
/// In the forward pass the residual is zero so the output is `sign(x)`.
/// In the backward pass `∂(residual)/∂x = scale`, so gradients flow.
#[inline]
pub fn ste_sign_scaled(x: &Tensor, scale: f64) -> Result<Tensor> {
    let sign_x = x.sign()?;
    let detach_x = x.detach();
    let residual = (x - &detach_x)?;
    &sign_x + &residual.affine(scale, 0.0)?
}

/// Soft STE: `forward ≈ tanh(α·x)`, `backward = scale · identity`.
///
/// Used during the annealing phase (α grows from 1→8). As α→∞, `tanh(α·x)`
/// converges to `sign(x)`, providing a smooth curriculum.
#[inline]
pub fn ste_tanh_scaled(x: &Tensor, alpha: f64, scale: f64) -> Result<Tensor> {
    let y = x.affine(alpha, 0.0)?;
    let tanh_y = y.tanh()?;
    let detach_x = x.detach();
    let residual = (x - &detach_x)?;
    &tanh_y + &residual.affine(scale, 0.0)?
}

// ── Ternary quantisation ────────────────────────────────────────────────────

/// Ternary quantisation using dynamic threshold Δ = 0.7 × mean(|W|).
///
/// ```text
/// W_q[i] = sign(W[i])  if |W[i]| > Δ
///          0            otherwise
/// ```
///
/// When `apply_ste = true`, the STE residual is added so gradients flow
/// through to the latent weight.
fn ternary_delta_quantize(
    w: &Tensor,
    delta: f64,
    ste_scale: f64,
    apply_ste: bool,
) -> Result<Tensor> {
    let abs_w = w.abs()?;
    let mask = abs_w.gt(delta)?.to_dtype(DType::F32)?;
    let out = (w.sign()? * mask)?;
    if apply_ste {
        let residual = (w - &w.detach())?;
        Ok((out.detach() + residual.affine(ste_scale, 0.0)?)?)
    } else {
        Ok(out)
    }
}

/// Ternary quantisation (forward only, **no STE**).
///
/// Returns an F32 tensor with values in {-1, 0, 1}. Used for:
/// * Inference cache (pre-compute once).
/// * Export to quantised checkpoint.
pub fn ternary_quantize_forward(w: &Tensor, use_dynamic_threshold: bool) -> Result<Tensor> {
    let abs_w = w.abs()?;
    let beta = abs_w.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
    if use_dynamic_threshold {
        let delta = 0.7 * beta;
        ternary_delta_quantize(w, delta, 1.0, false)
    } else {
        // AbsMean: scale by 1/β, round, clamp to [-1, 1].
        let inv_beta = 1.0f64 / beta;
        let w_scaled = w.affine(inv_beta, 0.0)?;
        let rounded = w_scaled.round()?;
        Ok(rounded.clamp(-1f64, 1f64)?)
    }
}

/// Ternary quantisation **with STE** (training path).
///
/// Quantises the weight to {-1, 0, +1} and attaches the STE residual
/// so that `∂loss/∂w_latent` is `ste_scale × identity`.
pub fn ternary_absmean_ste(
    w: &Tensor,
    use_dynamic_threshold: bool,
    ste_scale: f64,
) -> Result<Tensor> {
    let abs_w = w.abs()?;
    let beta = abs_w.mean_all()?.to_scalar::<f32>()?.max(1e-8) as f64;
    if use_dynamic_threshold {
        let delta = 0.7 * beta;
        ternary_delta_quantize(w, delta, ste_scale, true)
    } else {
        let inv_beta = 1.0f64 / beta;
        let w_scaled = w.affine(inv_beta, 0.0)?;
        let rounded = w_scaled.round()?;
        let clamped = rounded.clamp(-1f64, 1f64)?;
        let residual = (w - &w.detach())?;
        Ok((clamped.detach() + residual.affine(ste_scale, 0.0)?)?)
    }
}

// ── Debug helpers ───────────────────────────────────────────────────────────

/// Count {-1, 0, +1} occurrences in a ternary-quantised weight.
pub fn debug_ternary_distribution(
    w: &Tensor,
    use_dynamic_threshold: bool,
) -> Result<(u64, u64, u64)> {
    let q = ternary_quantize_forward(w, use_dynamic_threshold)?;
    let flat = q.flatten_all()?.to_vec1::<f32>()?;
    let (mut n_neg, mut n_zero, mut n_pos) = (0u64, 0u64, 0u64);
    for &v in &flat {
        if v < -0.5 {
            n_neg += 1;
        } else if v > 0.5 {
            n_pos += 1;
        } else {
            n_zero += 1;
        }
    }
    Ok((n_neg, n_zero, n_pos))
}

/// Count {-1, +1} occurrences in a binary-quantised weight.
pub fn debug_binary_distribution(w: &Tensor) -> Result<(u64, u64)> {
    let s = w.sign()?;
    let flat = s.flatten_all()?.to_vec1::<f32>()?;
    let (mut n_neg, mut n_pos) = (0u64, 0u64);
    for &v in &flat {
        if v < 0.0 {
            n_neg += 1;
        } else {
            n_pos += 1;
        }
    }
    Ok((n_neg, n_pos))
}

// ── Tensor helpers ──────────────────────────────────────────────────────────

/// Reshape `x` to 2-D, multiply by `w_t`, reshape back.
///
/// Handles (B, T, K), (B1, B2, T, K), and arbitrary leading dims.
pub fn matmul_reshape(x: &Tensor, w_t: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let out_dim = w_t.dim(1)?;
    match dims {
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

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn annealing_round_trip() {
        set_quant_anneal_frac(0.0);
        assert!((current_anneal_frac() - 0.0).abs() < 0.002);
        set_quant_anneal_frac(0.5);
        assert!((current_anneal_frac() - 0.5).abs() < 0.002);
        set_quant_anneal_frac(1.0);
        assert!((current_anneal_frac() - 1.0).abs() < 0.002);
        // Reset for other tests
        set_quant_anneal_frac(1.0);
    }

    #[test]
    fn annealing_clamps_out_of_range() {
        set_quant_anneal_frac(-5.0);
        assert!((current_anneal_frac() - 0.0).abs() < 0.002);
        set_quant_anneal_frac(99.0);
        assert!((current_anneal_frac() - 1.0).abs() < 0.002);
        set_quant_anneal_frac(1.0);
    }

    #[test]
    fn ternary_quantize_dynamic() {
        let dev = Device::Cpu;
        let w = Tensor::new(&[-0.9f32, -0.1, 0.0, 0.05, 0.8], &dev).unwrap();
        let q = ternary_quantize_forward(&w, true).unwrap();
        let vals: Vec<f32> = q.to_vec1().unwrap();
        // With dynamic threshold δ ≈ 0.7 * mean(|W|):
        // mean(|W|) = (0.9+0.1+0.0+0.05+0.8)/5 = 0.37 → δ ≈ 0.259
        // -0.9: |0.9| > 0.259 → -1
        // -0.1: |0.1| < 0.259 → 0
        //  0.0: 0 < 0.259 → 0
        //  0.05: 0.05 < 0.259 → 0
        //  0.8: |0.8| > 0.259 → +1
        assert_eq!(vals, vec![-1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn ternary_quantize_absmean() {
        let dev = Device::Cpu;
        let w = Tensor::new(&[-0.9f32, -0.1, 0.0, 0.05, 0.8], &dev).unwrap();
        let q = ternary_quantize_forward(&w, false).unwrap();
        let vals: Vec<f32> = q.to_vec1().unwrap();
        // AbsMean: β = 0.37, 1/β ≈ 2.70
        // -0.9 * 2.70 = -2.43 → round -2 → clamp -1
        // -0.1 * 2.70 = -0.27 → round 0
        //  0.0 * 2.70 = 0 → 0
        //  0.05 * 2.70 = 0.135 → round 0
        //  0.8 * 2.70 = 2.16 → round 2 → clamp 1
        assert_eq!(vals, vec![-1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn binary_distribution_counts() {
        let dev = Device::Cpu;
        let w = Tensor::new(&[-0.5f32, 0.3, -0.1, 0.9, 0.0], &dev).unwrap();
        let (neg, pos) = debug_binary_distribution(&w).unwrap();
        // sign: [-1, 1, -1, 1, 0] → candle sign(0)=0 which is ≥0 → pos
        assert_eq!(neg, 2);
        assert_eq!(pos, 3); // includes the 0
    }
}
