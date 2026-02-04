//! CLI for training OneBit-LLM from scratch.
//!
//! Usage: train --config config.json --data-dir /path/to/texts --output-dir ./checkpoints [options]
//! For 1-bit/ternary stability: use gradient clipping (default 1.0), lower LR (e.g. 1e-4), and warmup.

use std::io::Write;
use std::path::PathBuf;

use candle::{backprop::GradStore, DType, Device, Var};
use candle_nn::{loss, ops, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;

use onebit_llm::{batch_to_tensors, OneBitLlm, OneBitLlmConfig, StreamingBatchIter, TextDataset};

/// Cross-entropy with label smoothing: (1-s)*NLL + s/V * (-sum_c log p(c)).
fn cross_entropy_with_label_smoothing(
    logits: &candle::Tensor,
    labels: &candle::Tensor,
    smoothing: f64,
    vocab_size: usize,
) -> candle::Result<candle::Tensor> {
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

/// Learning rate schedule: warmup then constant or decay (cosine/linear) when max_steps > 0.
#[derive(Clone)]
struct LrScheduler {
    step: usize,
    lr: f64,
    lr_min: f64,
    warmup_steps: usize,
    max_steps: usize,
    lr_decay: String,
}

impl LrScheduler {
    fn new(lr: f64, lr_min: f64, warmup_steps: usize, max_steps: usize, lr_decay: String) -> Self {
        Self {
            step: 0,
            lr,
            lr_min,
            warmup_steps,
            max_steps,
            lr_decay,
        }
    }

    fn current_lr(&self) -> f64 {
        let step = self.step;
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return self.lr * (step as f64 + 1.0) / self.warmup_steps as f64;
        }
        if self.max_steps == 0 || self.lr_decay == "none" {
            return self.lr;
        }
        let step = step.min(self.max_steps);
        if step <= self.warmup_steps {
            return self.lr;
        }
        let decay_steps = (self.max_steps - self.warmup_steps).max(1);
        let progress = (step - self.warmup_steps) as f64 / decay_steps as f64;
        match self.lr_decay.as_str() {
            "cosine" => {
                let cos = (std::f64::consts::PI * progress).cos();
                self.lr_min + 0.5 * (self.lr - self.lr_min) * (1.0 + cos)
            }
            "linear" => self.lr - (self.lr - self.lr_min) * progress,
            _ => self.lr,
        }
    }

    fn advance(&mut self) {
        self.step += 1;
    }
}

/// Total L2 norm of gradients (for debug logging).
fn grad_norm(grads: &GradStore, vars: &[Var]) -> anyhow::Result<f64> {
    let mut total_norm_sq = 0.0f64;
    for var in vars.iter() {
        if let Some(g) = grads.get(var.as_tensor()) {
            let s = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            total_norm_sq += s;
        }
    }
    Ok(total_norm_sq.sqrt().max(1e-12))
}

/// Clip gradient norm to max_norm; scale all grads so global L2 norm <= max_norm.
fn clip_grad_norm(grads: &mut GradStore, vars: &[Var], max_norm: f64) -> anyhow::Result<()> {
    let mut total_norm_sq = 0.0f64;
    for var in vars.iter() {
        if let Some(g) = grads.get(var.as_tensor()) {
            let s = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            total_norm_sq += s;
        }
    }
    let total_norm = total_norm_sq.sqrt().max(1e-12);
    let scale = if total_norm > max_norm {
        max_norm / total_norm
    } else {
        1.0
    };
    for var in vars.iter() {
        if let Some(g) = grads.remove(var.as_tensor()) {
            let clipped = g.affine(scale, 0.0)?;
            grads.insert(var.as_tensor(), clipped);
        }
    }
    Ok(())
}

#[derive(Parser, Debug)]
#[command(name = "train", about = "Train OneBit-LLM from scratch")]
struct Args {
    /// Path to config JSON (created with defaults if missing).
    #[arg(long, default_value = "config.json")]
    config: PathBuf,

    /// Path to data: file or directory of .txt / .jsonl files.
    #[arg(long)]
    data_dir: PathBuf,

    /// Path to tokenizer.json (e.g. GPT-2 BPE).
    #[arg(long)]
    tokenizer: PathBuf,

    /// Output directory for checkpoints (safetensors + config.json).
    #[arg(long, default_value = "checkpoints")]
    output_dir: PathBuf,

    /// Use streaming: read files line-by-line (no full load into RAM). Use for large datasets (e.g. 25GB+).
    #[arg(long)]
    streaming: bool,

    /// Batch size.
    #[arg(long, default_value = "8")]
    batch_size: usize,

    /// Max training steps (0 = no step limit; in non-streaming, use max_epochs to stop).
    #[arg(long, default_value = "10000")]
    max_steps: usize,

    /// Max epochs (non-streaming only). 0 = no limit. E.g. 10 = stop after 10 full passes over the data.
    #[arg(long, default_value = "0")]
    max_epochs: usize,

    /// Save checkpoint every N steps.
    #[arg(long, default_value = "1000")]
    save_every: usize,

    /// Learning rate. 1-bit/ternary often needs higher LR (e.g. 5e-3 or 1e-2) so latent weights can cross the quantization threshold; default 5e-3.
    #[arg(long, default_value = "5e-3")]
    lr: f64,

    /// Weight decay (L2). Use 0 to avoid shrinking latent weights so they can flip -1/0/+1; non-zero may cause "frozen" ternary counts.
    #[arg(long, default_value = "0.0")]
    weight_decay: f64,

    /// Gradient clipping: max L2 norm of gradients (0 = disabled). Use 1.0 for 1-bit/ternary.
    #[arg(long, default_value = "1.0")]
    grad_clip_max_norm: f64,

    /// LR warmup steps: linear warmup from 0 to lr (use 100–200 for small datasets like Wikitext2).
    #[arg(long, default_value = "200")]
    lr_warmup_steps: usize,

    /// Minimum LR for decay (used only when lr_decay is cosine or linear and max_steps > 0). Use 1e-6 for cosine annealing.
    #[arg(long, default_value = "1e-6")]
    lr_min: f64,

    /// LR decay after warmup: cosine, linear, or none. Only applies when max_steps > 0.
    #[arg(long, default_value = "cosine", value_parser = ["cosine", "linear", "none"])]
    lr_decay: String,

    /// Label smoothing for cross-entropy (e.g. 0.1). Softens targets so model learns probabilities.
    #[arg(long, default_value = "0.1")]
    label_smoothing: f64,

    /// Log loss every N steps (higher = smaller log file; 0 = only progress lines).
    #[arg(long, default_value = "100")]
    log_every: usize,

    /// Debug: every N steps print weight distribution (-1, 0, +1 counts) for all bit-linear layers and total gradient norm. 0 = disabled.
    #[arg(long, default_value = "0")]
    debug_every: usize,

    /// Gradient accumulation steps: effective batch = batch_size * accumulation_steps. Use for large models when OOM.
    #[arg(long, default_value = "1")]
    accumulation_steps: usize,

    /// Validation data path (file or dir). If set, validation perplexity is computed every eval_every steps.
    #[arg(long)]
    val_data_dir: Option<PathBuf>,

    /// Run validation and log perplexity every N steps (requires --val-data-dir). 0 = disabled.
    #[arg(long, default_value = "500")]
    eval_every: usize,

    /// Max validation batches per eval (0 = use all available val batches up to memory).
    #[arg(long, default_value = "50")]
    eval_batches: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let config = if args.config.exists() {
        OneBitLlmConfig::load(&args.config)?
    } else {
        let default = OneBitLlmConfig::default();
        default.save(&args.config)?;
        eprintln!("Created default config at {}", args.config.display());
        default
    };

    std::fs::create_dir_all(&args.output_dir)?;

    let device = Device::cuda_if_available(0)?;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = OneBitLlm::new(vb, &config)?;

    let vars = varmap.all_vars();
    let compression = config.compression_stats();
    eprintln!(
        "Compression: total_params={} quantized_params={} effective_bits={:.2} compression_ratio={:.2}x",
        compression.total_params,
        compression.quantized_params,
        compression.effective_bits_per_param,
        compression.compression_ratio_vs_f32
    );
    eprintln!(
        "Training: lr={}, lr_min={}, lr_decay={}, grad_clip_max_norm={}, lr_warmup_steps={}, accumulation_steps={}",
        args.lr, args.lr_min, args.lr_decay, args.grad_clip_max_norm, args.lr_warmup_steps, args.accumulation_steps
    );
    let mut lr_scheduler = LrScheduler::new(
        args.lr,
        args.lr_min,
        args.lr_warmup_steps,
        args.max_steps,
        args.lr_decay.clone(),
    );
    let mut optimizer = AdamW::new(
        vars.clone(),
        ParamsAdamW {
            lr: args.lr,
            weight_decay: args.weight_decay,
            ..Default::default()
        },
    )?;

    let mut global_step = 0usize;
    let batch_size = args.batch_size;
    let seq_len = config.max_seq_len;

    let val_dataset: Option<TextDataset> = if let Some(ref p) = args.val_data_dir {
        let mut ds = TextDataset::new(p, &args.tokenizer, seq_len)?;
        ds.load()?;
        eprintln!(
            "Validation: loaded {} sequences (eval_every={}, eval_batches={})",
            ds.num_sequences(),
            args.eval_every,
            args.eval_batches
        );
        Some(ds)
    } else {
        None
    };

    let mut metrics_file: Option<std::fs::File> = if val_dataset.is_some() {
        let p = args.output_dir.join("metrics.csv");
        let mut f = std::fs::File::create(&p)?;
        writeln!(f, "step,val_loss,perplexity")?;
        eprintln!("Metrics log: {}", p.display());
        Some(f)
    } else {
        None
    };

    if args.streaming {
        eprintln!("Streaming mode: reading files line-by-line (stage1 + stage2)");
        let mut stream =
            StreamingBatchIter::new(&args.data_dir, &args.tokenizer, seq_len, batch_size)?;
        loop {
            if args.max_steps > 0 && global_step >= args.max_steps {
                break;
            }
            let mut batches = Vec::with_capacity(args.accumulation_steps);
            for _ in 0..args.accumulation_steps {
                match stream.next_batch()? {
                    Some(b) => batches.push(b),
                    None => break,
                }
            }
            if batches.is_empty() {
                break;
            }
            let n = batches.len();
            let arenas_coef = config.arenas_initial.map(|init| {
                let progress = global_step as f64 / config.arenas_anneal_steps as f64;
                (init * (1.0 - progress.min(1.0))) as f32
            });
            let mut total_loss = None;
            let mut loss_sum = 0.0f32;
            for (input_ids, labels) in batches {
                let (input_ids, labels) =
                    batch_to_tensors(&input_ids, &labels, batch_size, seq_len, &device)?;
                let logits = model.forward_with_arenas(&input_ids, arenas_coef)?;
                let (b, t, v) = logits.dims3()?;
                let logits_flat = logits.reshape((b * t, v))?;
                let labels_flat = labels.reshape((b * t,))?.to_dtype(DType::U32)?;
                let loss = cross_entropy_with_label_smoothing(
                    &logits_flat,
                    &labels_flat,
                    args.label_smoothing,
                    config.vocab_size,
                )?;
                loss_sum += loss.to_scalar::<f32>()?;
                let scaled = loss.affine(1.0 / n as f64, 0.0)?;
                total_loss = Some(match total_loss {
                    None => scaled,
                    Some(prev) => (prev + scaled)?,
                });
            }
            let total_loss = total_loss.unwrap();
            let loss_val = loss_sum / n as f32;

            optimizer.set_learning_rate(lr_scheduler.current_lr());
            let mut grads = total_loss.backward()?;
            let debug_grad_norm =
                if args.debug_every > 0 && global_step.is_multiple_of(args.debug_every) {
                    Some(grad_norm(&grads, &vars)?)
                } else {
                    None
                };
            if args.grad_clip_max_norm > 0.0 {
                clip_grad_norm(&mut grads, &vars, args.grad_clip_max_norm)?;
            }
            optimizer.step(&grads)?;
            lr_scheduler.advance();
            if args.log_every > 0 && global_step.is_multiple_of(args.log_every) {
                eprintln!("step {} loss {:.4}", global_step, loss_val);
            }
            if let Some(gn) = debug_grad_norm {
                eprintln!("  [debug] step {} grad_norm {:.4}", global_step, gn);
                if let Ok(dists) = model.debug_weight_distributions() {
                    for (name, s) in dists {
                        eprintln!("  [debug]   {} {}", name, s);
                    }
                }
            }
            if let Some(ref val_ds) = val_dataset {
                if args.eval_every > 0
                    && global_step > 0
                    && global_step.is_multiple_of(args.eval_every)
                {
                    let mut val_loss_sum = 0.0f64;
                    let mut val_count = 0usize;
                    for (input_ids, labels) in val_ds.batches(batch_size).take(args.eval_batches) {
                        let (input_ids, labels) =
                            batch_to_tensors(&input_ids, &labels, batch_size, seq_len, &device)?;
                        let logits = model.forward(&input_ids)?;
                        let (b, t, v) = logits.dims3()?;
                        let logits_flat = logits.reshape((b * t, v))?;
                        let labels_flat = labels.reshape((b * t,))?.to_dtype(DType::U32)?;
                        let loss = cross_entropy_with_label_smoothing(
                            &logits_flat,
                            &labels_flat,
                            args.label_smoothing,
                            config.vocab_size,
                        )?;
                        val_loss_sum += loss.to_scalar::<f32>()? as f64;
                        val_count += 1;
                    }
                    if val_count > 0 {
                        let val_loss = val_loss_sum / val_count as f64;
                        let ppl = val_loss.exp();
                        eprintln!(
                            "  [eval] step {} val_loss={:.4} perplexity={:.2}",
                            global_step, val_loss, ppl
                        );
                        if let Some(ref mut f) = metrics_file {
                            writeln!(f, "{},{},{}", global_step, val_loss, ppl)?;
                        }
                    }
                }
            }
            if global_step > 0 && global_step.is_multiple_of(500) && args.save_every > 0 {
                let next_ckpt = ((global_step / args.save_every) + 1) * args.save_every;
                eprintln!(
                    "  -> progress: {} steps (next checkpoint at step {})",
                    global_step, next_ckpt
                );
            }
            if global_step > 0 && global_step.is_multiple_of(500) {
                let _ = std::fs::write(
                    args.output_dir.join("progress.txt"),
                    format!("step {} loss {:.4}\n", global_step, loss_val),
                );
            }

            global_step += 1;

            if args.save_every > 0 && global_step.is_multiple_of(args.save_every) {
                let ckpt_path = args
                    .output_dir
                    .join(format!("checkpoint-{}.safetensors", global_step));
                varmap.save(&ckpt_path)?;
                config.save(&args.output_dir.join("config.json"))?;
                eprintln!("Saved checkpoint to {}", ckpt_path.display());
            }
        }
    } else {
        let mut dataset = TextDataset::new(&args.data_dir, &args.tokenizer, config.max_seq_len)?;
        dataset.load()?;

        let num_tokens = dataset.num_tokens();
        let num_seq = dataset.num_sequences();
        eprintln!(
            "Loaded {} tokens, {} sequences (seq_len={})",
            num_tokens, num_seq, config.max_seq_len
        );

        if num_seq == 0 {
            anyhow::bail!("No sequences; need at least seq_len+1 tokens");
        }

        let mut epoch = 0u32;
        loop {
            if args.max_steps > 0 && global_step >= args.max_steps {
                break;
            }
            if args.max_epochs > 0 && (epoch as usize) >= args.max_epochs {
                eprintln!("Completed {} epochs, stopping.", args.max_epochs);
                break;
            }
            eprintln!("epoch {} (step {})", epoch, global_step);

            let mut batch_iter = dataset.batches(batch_size).enumerate().peekable();
            while batch_iter.peek().is_some() {
                if args.max_steps > 0 && global_step >= args.max_steps {
                    break;
                }
                let batches: Vec<_> = batch_iter.by_ref().take(args.accumulation_steps).collect();
                if batches.is_empty() {
                    break;
                }
                let n = batches.len();
                let arenas_coef = config.arenas_initial.map(|init| {
                    let progress = global_step as f64 / config.arenas_anneal_steps as f64;
                    (init * (1.0 - progress.min(1.0))) as f32
                });
                let mut total_loss = None;
                let mut loss_sum = 0.0f32;
                for (_batch_idx, (input_ids, labels)) in batches {
                    let (input_ids, labels) =
                        batch_to_tensors(&input_ids, &labels, batch_size, seq_len, &device)?;
                    let logits = model.forward_with_arenas(&input_ids, arenas_coef)?;
                    let (b, t, v) = logits.dims3()?;
                    let logits_flat = logits.reshape((b * t, v))?;
                    let labels_flat = labels.reshape((b * t,))?.to_dtype(DType::U32)?;
                    let loss = cross_entropy_with_label_smoothing(
                        &logits_flat,
                        &labels_flat,
                        args.label_smoothing,
                        config.vocab_size,
                    )?;
                    loss_sum += loss.to_scalar::<f32>()?;
                    let scaled = loss.affine(1.0 / n as f64, 0.0)?;
                    total_loss = Some(match total_loss {
                        None => scaled,
                        Some(prev) => (prev + scaled)?,
                    });
                }
                let total_loss = total_loss.unwrap();
                let loss_val = loss_sum / n as f32;

                optimizer.set_learning_rate(lr_scheduler.current_lr());
                let mut grads = total_loss.backward()?;
                let debug_grad_norm =
                    if args.debug_every > 0 && global_step.is_multiple_of(args.debug_every) {
                        Some(grad_norm(&grads, &vars)?)
                    } else {
                        None
                    };
                if args.grad_clip_max_norm > 0.0 {
                    clip_grad_norm(&mut grads, &vars, args.grad_clip_max_norm)?;
                }
                optimizer.step(&grads)?;
                lr_scheduler.advance();

                if args.log_every > 0 && global_step.is_multiple_of(args.log_every) {
                    eprintln!(
                        "step {} epoch {} loss {:.4} (effective_batch={})",
                        global_step,
                        epoch,
                        loss_val,
                        batch_size * n
                    );
                }
                if let Some(gn) = debug_grad_norm {
                    eprintln!("  [debug] step {} grad_norm {:.4}", global_step, gn);
                    if let Ok(dists) = model.debug_weight_distributions() {
                        for (name, s) in dists {
                            eprintln!("  [debug]   {} {}", name, s);
                        }
                    }
                }
                if let Some(ref val_ds) = val_dataset {
                    if args.eval_every > 0
                        && global_step > 0
                        && global_step.is_multiple_of(args.eval_every)
                    {
                        let mut val_loss_sum = 0.0f64;
                        let mut val_count = 0usize;
                        for (input_ids, labels) in
                            val_ds.batches(batch_size).take(args.eval_batches)
                        {
                            let (input_ids, labels) = batch_to_tensors(
                                &input_ids, &labels, batch_size, seq_len, &device,
                            )?;
                            let logits = model.forward(&input_ids)?;
                            let (b, t, v) = logits.dims3()?;
                            let logits_flat = logits.reshape((b * t, v))?;
                            let labels_flat = labels.reshape((b * t,))?.to_dtype(DType::U32)?;
                            let loss = cross_entropy_with_label_smoothing(
                                &logits_flat,
                                &labels_flat,
                                args.label_smoothing,
                                config.vocab_size,
                            )?;
                            val_loss_sum += loss.to_scalar::<f32>()? as f64;
                            val_count += 1;
                        }
                        if val_count > 0 {
                            let val_loss = val_loss_sum / val_count as f64;
                            let ppl = val_loss.exp();
                            eprintln!(
                                "  [eval] step {} val_loss={:.4} perplexity={:.2}",
                                global_step, val_loss, ppl
                            );
                            if let Some(ref mut f) = metrics_file {
                                writeln!(f, "{},{},{}", global_step, val_loss, ppl)?;
                            }
                        }
                    }
                }
                if global_step > 0 && global_step.is_multiple_of(500) && args.save_every > 0 {
                    let next_ckpt = ((global_step / args.save_every) + 1) * args.save_every;
                    eprintln!(
                        "  -> progress: {} steps (next checkpoint at step {})",
                        global_step, next_ckpt
                    );
                }
                if global_step > 0 && global_step.is_multiple_of(500) {
                    let _ = std::fs::write(
                        args.output_dir.join("progress.txt"),
                        format!("step {} loss {:.4}\n", global_step, loss_val),
                    );
                }

                global_step += 1;

                if args.save_every > 0 && global_step.is_multiple_of(args.save_every) {
                    let ckpt_path = args
                        .output_dir
                        .join(format!("checkpoint-{}.safetensors", global_step));
                    varmap.save(&ckpt_path)?;
                    config.save(&args.output_dir.join("config.json"))?;
                    eprintln!("Saved checkpoint to {}", ckpt_path.display());
                }
            }
            epoch += 1;
        }
    }

    let final_path = args.output_dir.join("model.safetensors");
    varmap.save(&final_path)?;
    config.save(&args.output_dir.join("config.json"))?;
    eprintln!(
        "Training done. Saved final weights to {}",
        final_path.display()
    );

    Ok(())
}
