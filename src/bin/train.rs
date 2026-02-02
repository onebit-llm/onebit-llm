//! CLI for training OneBit-LLM from scratch.
//!
//! Usage: train --config config.json --data-dir /path/to/texts --output-dir ./checkpoints [options]
//! For 1-bit/ternary stability: use gradient clipping (default 1.0), lower LR (e.g. 1e-4), and warmup.

use std::path::PathBuf;

use candle::{backprop::GradStore, DType, Device, Var};
use candle_nn::{loss, AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use clap::Parser;

use onebit_llm::{batch_to_tensors, OneBitLlm, OneBitLlmConfig, StreamingBatchIter, TextDataset};

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

    /// Max training steps (0 = run until data exhausted).
    #[arg(long, default_value = "10000")]
    max_steps: usize,

    /// Save checkpoint every N steps.
    #[arg(long, default_value = "1000")]
    save_every: usize,

    /// Learning rate. Use lower values (e.g. 1e-4 or 3e-5) for 1-bit/ternary stability.
    #[arg(long, default_value = "1e-4")]
    lr: f64,

    /// Gradient clipping: max L2 norm of gradients (0 = disabled). Use 1.0 for 1-bit/ternary.
    #[arg(long, default_value = "1.0")]
    grad_clip_max_norm: f64,

    /// LR warmup steps: linear warmup from 0 to lr over this many steps (0 = no warmup).
    #[arg(long, default_value = "1000")]
    lr_warmup_steps: usize,

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
        "Training: lr={}, grad_clip_max_norm={}, lr_warmup_steps={}",
        args.lr, args.grad_clip_max_norm, args.lr_warmup_steps
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
            let loss = loss::cross_entropy(&logits_flat, &labels_flat)?;

            let effective_lr = if args.lr_warmup_steps > 0 && global_step < args.lr_warmup_steps {
                args.lr * (global_step as f64 + 1.0) / args.lr_warmup_steps as f64
            } else {
                args.lr
            };
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
        let loss = loss::cross_entropy(&logits_flat, &labels_flat)?;

        let effective_lr = if args.lr_warmup_steps > 0 && global_step < args.lr_warmup_steps {
            args.lr * (global_step as f64 + 1.0) / args.lr_warmup_steps as f64
        } else {
            args.lr
        };
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
