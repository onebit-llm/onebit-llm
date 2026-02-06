//! Trainer: encapsulates the full training loop.
//!
//! Decouples the compute graph (forward + loss) from the optimisation step
//! (backward, gradient clipping, AdamW, latent clamping, schedule advance).

use std::path::PathBuf;

use candle_core::{backprop::GradStore, DType, Device, Tensor, Var};
use candle_nn::{loss, ops, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use ternary_common::{batch_to_tensors, BatchDataset, OneBitLlmConfig};
use ternary_core::{compression_stats, OneBitLlm};

use crate::scheduler::{AnnealSchedule, LrDecay, LrScheduler};

// ── Config ──────────────────────────────────────────────────────────────────

/// All training hyper-parameters (CLI-level knobs).
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub batch_size: usize,
    pub accumulation_steps: usize,
    pub max_steps: usize,
    pub max_epochs: usize,
    pub lr: f64,
    pub lr_min: f64,
    pub lr_warmup_steps: usize,
    pub lr_decay: LrDecay,
    pub weight_decay: f64,
    pub grad_clip_max_norm: f64,
    pub label_smoothing: f64,
    pub save_every: usize,
    pub log_every: usize,
    pub debug_every: usize,
    pub eval_every: usize,
    pub eval_batches: usize,
    pub output_dir: PathBuf,
}

/// Metrics returned after each training step.
#[derive(Debug, Clone)]
pub struct StepMetrics {
    pub step: usize,
    pub loss: f32,
    pub lr: f64,
    pub grad_norm: Option<f64>,
}

// ── Trainer ─────────────────────────────────────────────────────────────────

/// The training engine. Owns the model, optimiser, and all schedules.
pub struct Trainer {
    pub model: OneBitLlm,
    pub varmap: VarMap,
    vars: Vec<Var>,
    optimizer: AdamW,
    lr_scheduler: LrScheduler,
    anneal_schedule: AnnealSchedule,
    pub config: TrainerConfig,
    model_config: OneBitLlmConfig,
    pub global_step: usize,
    device: Device,
}

impl Trainer {
    /// Construct a new Trainer. Builds the model from config.
    pub fn new(
        model_config: OneBitLlmConfig,
        trainer_config: TrainerConfig,
        device: Device,
    ) -> anyhow::Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = OneBitLlm::new(vb, &model_config)?;
        let vars = varmap.all_vars();

        let stats = compression_stats(&model_config);
        tracing::info!(
            total_params = stats.total_params,
            quantized_params = stats.quantized_params,
            effective_bits = format!("{:.2}", stats.effective_bits_per_param),
            compression = format!("{:.2}x", stats.compression_ratio_vs_f32),
            "Model compression stats"
        );

        let lr_scheduler = LrScheduler::new(
            trainer_config.lr,
            trainer_config.lr_min,
            trainer_config.lr_warmup_steps,
            trainer_config.max_steps,
            trainer_config.lr_decay,
        );
        let anneal_schedule =
            AnnealSchedule::new(trainer_config.max_steps, model_config.anneal_fraction);

        let optimizer = AdamW::new(
            vars.clone(),
            ParamsAdamW {
                lr: trainer_config.lr,
                weight_decay: trainer_config.weight_decay,
                ..Default::default()
            },
        )?;

        Ok(Self {
            model,
            varmap,
            vars,
            optimizer,
            lr_scheduler,
            anneal_schedule,
            config: trainer_config,
            model_config,
            global_step: 0,
            device,
        })
    }

    /// Execute one optimiser step over N accumulated micro-batches.
    pub fn step(&mut self, batches: &[(Vec<u32>, Vec<u32>)]) -> anyhow::Result<StepMetrics> {
        let n = batches.len();
        let seq_len = self.model_config.max_seq_len;
        let batch_size = self.config.batch_size;

        // Arenas coefficient
        let arenas_coef = self.model_config.arenas_initial.map(|init| {
            let progress = self.global_step as f64 / self.model_config.arenas_anneal_steps as f64;
            (init * (1.0 - progress.min(1.0))) as f32
        });

        // Forward + loss accumulation
        let mut total_loss: Option<Tensor> = None;
        let mut loss_sum = 0.0f32;

        for (input_ids, labels) in batches {
            let (input_ids, labels) =
                batch_to_tensors(input_ids, labels, batch_size, seq_len, &self.device)?;
            let logits = self.model.forward_with_arenas(&input_ids, arenas_coef)?;
            let (b, t, v) = logits.dims3()?;
            let logits_flat = logits.reshape((b * t, v))?;
            let labels_flat = labels.reshape((b * t,))?.to_dtype(DType::U32)?;
            let step_loss = cross_entropy_with_label_smoothing(
                &logits_flat,
                &labels_flat,
                self.config.label_smoothing,
                self.model_config.vocab_size,
            )?;
            loss_sum += step_loss.to_scalar::<f32>()?;
            let scaled = step_loss.affine(1.0 / n as f64, 0.0)?;
            total_loss = Some(match total_loss {
                None => scaled,
                Some(prev) => (prev + scaled)?,
            });
        }
        let total_loss = total_loss.unwrap();
        let loss_val = loss_sum / n as f32;

        // Backward
        self.optimizer
            .set_learning_rate(self.lr_scheduler.current_lr());
        let mut grads = total_loss.backward()?;

        // Debug gradient norm
        let debug_grad_norm =
            if self.config.debug_every > 0 && self.global_step % self.config.debug_every == 0 {
                Some(grad_norm(&grads, &self.vars)?)
            } else {
                None
            };

        // Gradient clipping
        if self.config.grad_clip_max_norm > 0.0 {
            clip_grad_norm(&mut grads, &self.vars, self.config.grad_clip_max_norm)?;
        }

        // Optimiser step
        self.optimizer.step(&grads)?;

        // Latent weight clamping (uses config value, not hard-coded 1.2)
        let clamp = self.model_config.latent_clamp_max;
        for var in &self.vars {
            let t = var.as_tensor();
            if t.dtype() == DType::F32 {
                let clamped = t.clamp(-clamp as f32, clamp as f32)?;
                var.set(&clamped)?;
            }
        }

        // Advance schedules
        let lr = self.lr_scheduler.current_lr();
        self.lr_scheduler.advance();
        self.anneal_schedule.step(self.global_step);
        self.global_step += 1;

        Ok(StepMetrics {
            step: self.global_step - 1,
            loss: loss_val,
            lr,
            grad_norm: debug_grad_norm,
        })
    }

    /// Save checkpoint.
    pub fn save_checkpoint(&self) -> anyhow::Result<PathBuf> {
        std::fs::create_dir_all(&self.config.output_dir)?;
        let path = self
            .config
            .output_dir
            .join(format!("checkpoint-{}.safetensors", self.global_step));
        self.varmap.save(&path)?;
        self.model_config
            .save(&self.config.output_dir.join("config.json"))?;
        Ok(path)
    }

    /// Save final model.
    pub fn save_final(&self) -> anyhow::Result<PathBuf> {
        std::fs::create_dir_all(&self.config.output_dir)?;
        let path = self.config.output_dir.join("model.safetensors");
        self.varmap.save(&path)?;
        self.model_config
            .save(&self.config.output_dir.join("config.json"))?;
        Ok(path)
    }

    /// Evaluate validation perplexity.
    pub fn evaluate(&self, val_ds: &impl BatchDataset) -> anyhow::Result<(f64, f64)> {
        let batch_size = self.config.batch_size;
        let seq_len = self.model_config.max_seq_len;
        let mut loss_sum = 0.0f64;
        let mut count = 0usize;

        for (input_ids, labels) in val_ds.batches(batch_size).take(self.config.eval_batches) {
            let (input_ids, labels) =
                batch_to_tensors(&input_ids, &labels, batch_size, seq_len, &self.device)?;
            let logits = self.model.forward(&input_ids)?;
            let (b, t, v) = logits.dims3()?;
            let logits_flat = logits.reshape((b * t, v))?;
            let labels_flat = labels.reshape((b * t,))?.to_dtype(DType::U32)?;
            let l = cross_entropy_with_label_smoothing(
                &logits_flat,
                &labels_flat,
                self.config.label_smoothing,
                self.model_config.vocab_size,
            )?;
            loss_sum += l.to_scalar::<f32>()? as f64;
            count += 1;
        }

        if count == 0 {
            return Ok((f64::MAX, f64::MAX));
        }
        let avg_loss = loss_sum / count as f64;
        let ppl = avg_loss.exp();
        Ok((avg_loss, ppl))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

// ── Loss ────────────────────────────────────────────────────────────────────

/// Cross-entropy with label smoothing.
fn cross_entropy_with_label_smoothing(
    logits: &Tensor,
    labels: &Tensor,
    smoothing: f64,
    vocab_size: usize,
) -> candle_core::Result<Tensor> {
    if smoothing <= 0.0 {
        return loss::cross_entropy(logits, labels);
    }
    let log_probs = ops::log_softmax(logits, 1)?;
    let nll = loss::nll(&log_probs, labels)?;
    let sum_log = log_probs.sum(1)?;
    let neg_sum_mean = (sum_log.neg()?.mean_all()?.to_scalar::<f32>()?) as f64;
    let s = smoothing;
    let v = vocab_size as f64;
    nll.affine(1.0 - s, s / v * neg_sum_mean)
}

// ── Gradient utilities ──────────────────────────────────────────────────────

/// Total L2 norm of gradients.
fn grad_norm(grads: &GradStore, vars: &[Var]) -> anyhow::Result<f64> {
    let mut total = 0.0f64;
    for var in vars {
        if let Some(g) = grads.get(var.as_tensor()) {
            total += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
    }
    Ok(total.sqrt().max(1e-12))
}

/// Clip gradients so their global L2 norm ≤ `max_norm`.
fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> anyhow::Result<()> {
    let mut total = 0.0f64;
    for var in vars {
        if let Some(g) = grads.get(var.as_tensor()) {
            total += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
    }
    let norm = total.sqrt().max(1e-12);
    let scale = if norm > max_norm {
        max_norm / norm
    } else {
        1.0
    };
    for var in vars {
        if let Some(g) = grads.remove(var.as_tensor()) {
            let clipped = g.affine(scale, 0.0)?;
            grads.insert(var.as_tensor(), clipped);
        }
    }
    Ok(())
}
