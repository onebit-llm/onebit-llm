//! CLI for training OneBit-LLM from scratch.
//!
//! Usage: train --config config.json --data-dir /path/to/texts --output-dir ./checkpoints [options]
//! For 1-bit/ternary stability: use gradient clipping (default 1.0), lower LR (e.g. 1e-4), and warmup.

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

/// Effective learning rate: warmup then constant or decay (cosine/linear) when max_steps > 0.
fn effective_lr(
    step: usize,
    lr: f64,
    lr_min: f64,
    warmup_steps: usize,
    max_steps: usize,
    lr_decay: &str,
) -> f64 {
    if warmup_steps > 0 && step < warmup_steps {
        return lr * (step as f64 + 1.0) / warmup_steps as f64;
    }
    if max_steps == 0 || lr_decay == "none" {
        return lr;
    }
    let step = step.min(max_steps);
    if step <= warmup_steps {
        return lr;
    }
    let decay_steps = (max_steps - warmup_steps).max(1);
    let progress = (step - warmup_steps) as f64 / decay_steps as f64;
    match lr_decay {
        "cosine" => {
            let cos = (std::f64::consts::PI * progress).cos();
            lr_min + 0.5 * (lr - lr_min) * (1.0 + cos)
        }
        "linear" => lr - (lr - lr_min) * progress,
        _ => lr,
    }
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
    let scale = if total_norm > max_norm { max_norm / total_norm } else { 1.0 };
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

    /// Learning rate. 1-bit/ternary training often benefits from higher LR than FP16 (e.g. 1e-3 or 3e-3); try raising if loss plateaus (BitNet literature).
    #[arg(long, default_value = "1e-3")]
    lr: f64,

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
    eprintln!(
        "Training: lr={}, lr_min={}, lr_decay={}, grad_clip_max_norm={}, lr_warmup_steps={}",
        args.lr, args.lr_min, args.lr_decay, args.grad_clip_max_norm, args.lr_warmup_steps
    );
    let mut optimizer = AdamW::new(
        vars.clone(),
        ParamsAdamW {
            lr: args.lr,
            ..Default::default()
        },
    )?;

    let mut global_step = 0usize;
    let batch_size = args.batch_size;
    let seq_len = config.max_seq_len;

    if args.streaming {
        eprintln!("Streaming mode: reading files line-by-line (stage1 + stage2)");
        let mut stream = StreamingBatchIter::new(
            &args.data_dir,
            &args.tokenizer,
            seq_len,
            batch_size,
        )?;
        while let Some((input_ids, labels)) = stream.next_batch()? {
            if args.max_steps > 0 && global_step >= args.max_steps {
                break;
            }
            let (input_ids, labels) = batch_to_tensors(
                &input_ids,
                &labels,
                batch_size,
                seq_len,
                &device,
            )?;

            let arenas_coef = config.arenas_initial.map(|init| {
                let progress = global_step as f64 / config.arenas_anneal_steps as f64;
                (init * (1.0 - progress.min(1.0))) as f32
            });
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

            let effective_lr = effective_lr(
                global_step,
                args.lr,
                args.lr_min,
                args.lr_warmup_steps,
                args.max_steps,
                &args.lr_decay,
            );
            optimizer.set_learning_rate(effective_lr);

            let mut grads = loss.backward()?;
            if args.grad_clip_max_norm > 0.0 {
                clip_grad_norm(&mut grads, &vars, args.grad_clip_max_norm)?;
            }
            optimizer.step(&grads)?;

            let loss_val = loss.to_scalar::<f32>()?;
            if args.log_every > 0 && global_step % args.log_every == 0 {
                eprintln!("step {} loss {:.4}", global_step, loss_val);
            }
            if global_step > 0 && global_step % 500 == 0 && args.save_every > 0 {
                let next_ckpt = ((global_step / args.save_every) + 1) * args.save_every;
                eprintln!("  -> progress: {} steps (next checkpoint at step {})", global_step, next_ckpt);
            }
            if global_step > 0 && global_step % 500 == 0 {
                let _ = std::fs::write(
                    args.output_dir.join("progress.txt"),
                    format!("step {} loss {:.4}\n", global_step, loss_val),
                );
            }

            global_step += 1;

            if args.save_every > 0 && global_step % args.save_every == 0 {
                let ckpt_path = args.output_dir.join(format!("checkpoint-{}.safetensors", global_step));
                varmap.save(&ckpt_path)?;
                config.save(&args.output_dir.join("config.json"))?;
                eprintln!("Saved checkpoint to {}", ckpt_path.display());
            }
        }
    } else {
        let mut dataset = TextDataset::new(
            &args.data_dir,
            &args.tokenizer,
            config.max_seq_len,
        )?;
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

            for (batch_idx, (input_ids, labels)) in dataset.batches(batch_size).enumerate() {
                if args.max_steps > 0 && global_step >= args.max_steps {
                    break;
                }

                let (input_ids, labels) = batch_to_tensors(
            &input_ids,
            &labels,
            batch_size,
            seq_len,
            &device,
        )?;

        let arenas_coef = config.arenas_initial.map(|init| {
            let progress = global_step as f64 / config.arenas_anneal_steps as f64;
            (init * (1.0 - progress.min(1.0))) as f32
        });
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

        let effective_lr = effective_lr(
            global_step,
            args.lr,
            args.lr_min,
            args.lr_warmup_steps,
            args.max_steps,
            &args.lr_decay,
        );
        optimizer.set_learning_rate(effective_lr);

        let mut grads = loss.backward()?;
        if args.grad_clip_max_norm > 0.0 {
            clip_grad_norm(&mut grads, &vars, args.grad_clip_max_norm)?;
        }
        optimizer.step(&grads)?;

        let loss_val = loss.to_scalar::<f32>()?;
        if args.log_every > 0 && global_step % args.log_every == 0 {
            eprintln!("step {} epoch {} batch {} loss {:.4}", global_step, epoch, batch_idx, loss_val);
        }
        if global_step > 0 && global_step % 500 == 0 && args.save_every > 0 {
            let next_ckpt = ((global_step / args.save_every) + 1) * args.save_every;
            eprintln!("  -> progress: {} steps (next checkpoint at step {})", global_step, next_ckpt);
        }
        if global_step > 0 && global_step % 500 == 0 {
            let _ = std::fs::write(
                args.output_dir.join("progress.txt"),
                format!("step {} loss {:.4}\n", global_step, loss_val),
            );
        }

        global_step += 1;

            if args.save_every > 0 && global_step % args.save_every == 0 {
                let ckpt_path = args.output_dir.join(format!("checkpoint-{}.safetensors", global_step));
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
    eprintln!("Training done. Saved final weights to {}", final_path.display());

    Ok(())
}
