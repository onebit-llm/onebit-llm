//! Learning rate and quantisation annealing schedules.

use ternary_core::set_quant_anneal_frac;

// ── LR Scheduler ────────────────────────────────────────────────────────────

/// Learning rate schedule: warmup → cosine / linear / constant decay.
#[derive(Clone)]
pub struct LrScheduler {
    step: usize,
    lr: f64,
    lr_min: f64,
    warmup_steps: usize,
    max_steps: usize,
    decay: LrDecay,
}

/// Decay mode after warmup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LrDecay {
    Cosine,
    Linear,
    None,
}

impl LrDecay {
    pub fn from_str(s: &str) -> Self {
        match s {
            "cosine" => Self::Cosine,
            "linear" => Self::Linear,
            _ => Self::None,
        }
    }
}

impl LrScheduler {
    pub fn new(
        lr: f64,
        lr_min: f64,
        warmup_steps: usize,
        max_steps: usize,
        decay: LrDecay,
    ) -> Self {
        Self {
            step: 0,
            lr,
            lr_min,
            warmup_steps,
            max_steps,
            decay,
        }
    }

    /// Current learning rate at the current step.
    pub fn current_lr(&self) -> f64 {
        let step = self.step;

        // Warmup phase: linear ramp from 0 to lr.
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.lr * (step as f64 + 1.0) / self.warmup_steps as f64;
        }

        // No decay or no max_steps → constant lr.
        if self.max_steps == 0 || self.decay == LrDecay::None {
            return self.lr;
        }

        let step = step.min(self.max_steps);
        if step <= self.warmup_steps {
            return self.lr;
        }

        let decay_steps = (self.max_steps - self.warmup_steps).max(1);
        let progress = (step - self.warmup_steps) as f64 / decay_steps as f64;

        match self.decay {
            LrDecay::Cosine => {
                let cos = (std::f64::consts::PI * progress).cos();
                self.lr_min + 0.5 * (self.lr - self.lr_min) * (1.0 + cos)
            }
            LrDecay::Linear => self.lr - (self.lr - self.lr_min) * progress,
            LrDecay::None => self.lr,
        }
    }

    pub fn advance(&mut self) {
        self.step += 1;
    }

    pub fn step(&self) -> usize {
        self.step
    }
}

// ── Anneal Schedule ─────────────────────────────────────────────────────────

/// Quantisation annealing: controls the global soft→hard schedule.
///
/// All `BitLinear` layers read the same atomic fraction. This struct
/// computes the fraction from the current step and writes it.
pub struct AnnealSchedule {
    total_steps: usize,
    anneal_fraction: f32,
}

impl AnnealSchedule {
    /// * `total_steps` — total training steps (0 = always hard).
    /// * `anneal_fraction` — what fraction of training is spent annealing (default 0.3).
    pub fn new(total_steps: usize, anneal_fraction: f32) -> Self {
        Self {
            total_steps,
            anneal_fraction,
        }
    }

    /// Advance the annealing schedule based on the current global step.
    pub fn step(&self, global_step: usize) {
        if self.total_steps == 0 {
            set_quant_anneal_frac(1.0);
            return;
        }
        let anneal_steps = (self.total_steps as f32 * self.anneal_fraction).max(1.0) as usize;
        let frac = (global_step as f32 / anneal_steps as f32).min(1.0);
        set_quant_anneal_frac(frac);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lr_warmup() {
        let sched = LrScheduler::new(1e-3, 1e-6, 100, 1000, LrDecay::Cosine);
        // Step 0: (0+1)/100 * 1e-3 = 1e-5
        assert!((sched.current_lr() - 1e-5).abs() < 1e-9);
    }

    #[test]
    fn lr_cosine_midpoint() {
        let mut sched = LrScheduler::new(1e-3, 0.0, 0, 1000, LrDecay::Cosine);
        for _ in 0..500 {
            sched.advance();
        }
        // Midpoint of cosine: cos(π·0.5) = 0 → lr = 0.5 * 1e-3
        assert!((sched.current_lr() - 5e-4).abs() < 1e-6);
    }

    #[test]
    fn lr_no_decay() {
        let mut sched = LrScheduler::new(1e-3, 1e-6, 0, 1000, LrDecay::None);
        for _ in 0..500 {
            sched.advance();
        }
        assert!((sched.current_lr() - 1e-3).abs() < 1e-9);
    }
}
