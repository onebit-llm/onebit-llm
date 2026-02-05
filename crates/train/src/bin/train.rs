//! CLI for training OneBit-LLM from scratch.

use std::io::Write;
use std::path::PathBuf;

use candle_core::Device;
use clap::Parser;

use ternary_common::{OneBitLlmConfig, StreamingBatchIter, TextDataset};
use ternary_train::{LrDecay, Trainer, TrainerConfig};

#[derive(Parser, Debug)]
#[command(name = "onebit-train", about = "Train a 1-bit LLM from scratch")]
struct Args {
    #[arg(long, default_value = "config.json")]
    config: PathBuf,
    #[arg(long)]
    data_dir: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long, default_value = "checkpoints")]
    output_dir: PathBuf,
    #[arg(long)]
    streaming: bool,
    #[arg(long, default_value = "8")]
    batch_size: usize,
    #[arg(long, default_value = "10000")]
    max_steps: usize,
    #[arg(long, default_value = "0")]
    max_epochs: usize,
    #[arg(long, default_value = "1000")]
    save_every: usize,
    #[arg(long, default_value = "5e-3")]
    lr: f64,
    #[arg(long, default_value = "0.0")]
    weight_decay: f64,
    #[arg(long, default_value = "1.0")]
    grad_clip_max_norm: f64,
    #[arg(long, default_value = "200")]
    lr_warmup_steps: usize,
    #[arg(long, default_value = "1e-6")]
    lr_min: f64,
    #[arg(long, default_value = "cosine", value_parser = ["cosine", "linear", "none"])]
    lr_decay: String,
    #[arg(long, default_value = "0.1")]
    label_smoothing: f64,
    #[arg(long, default_value = "100")]
    log_every: usize,
    #[arg(long, default_value = "0")]
    debug_every: usize,
    #[arg(long, default_value = "1")]
    accumulation_steps: usize,
    #[arg(long)]
    val_data_dir: Option<PathBuf>,
    #[arg(long, default_value = "500")]
    eval_every: usize,
    #[arg(long, default_value = "50")]
    eval_batches: usize,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    // Load or create config
    let model_config = if args.config.exists() {
        OneBitLlmConfig::load(&args.config)?
    } else {
        let default = OneBitLlmConfig::default();
        default.save(&args.config)?;
        eprintln!("Created default config at {}", args.config.display());
        default
    };

    let trainer_config = TrainerConfig {
        batch_size: args.batch_size,
        accumulation_steps: args.accumulation_steps,
        max_steps: args.max_steps,
        max_epochs: args.max_epochs,
        lr: args.lr,
        lr_min: args.lr_min,
        lr_warmup_steps: args.lr_warmup_steps,
        lr_decay: LrDecay::from_str(&args.lr_decay),
        weight_decay: args.weight_decay,
        grad_clip_max_norm: args.grad_clip_max_norm,
        label_smoothing: args.label_smoothing,
        save_every: args.save_every,
        log_every: args.log_every,
        debug_every: args.debug_every,
        eval_every: args.eval_every,
        eval_batches: args.eval_batches,
        output_dir: args.output_dir.clone(),
    };

    let device = Device::cuda_if_available(0)?;
    let mut trainer = Trainer::new(model_config.clone(), trainer_config, device)?;

    // Validation dataset (optional)
    let val_dataset: Option<TextDataset> = if let Some(ref p) = args.val_data_dir {
        let mut ds = TextDataset::new(p, &args.tokenizer, model_config.max_seq_len)?;
        ds.load()?;
        eprintln!("Validation: {} sequences", ds.num_sequences());
        Some(ds)
    } else {
        None
    };

    let mut metrics_file: Option<std::fs::File> = if val_dataset.is_some() {
        let p = args.output_dir.join("metrics.csv");
        let mut f = std::fs::File::create(&p)?;
        writeln!(f, "step,val_loss,perplexity")?;
        Some(f)
    } else {
        None
    };

    if args.streaming {
        eprintln!("Streaming mode");
        let mut stream = StreamingBatchIter::new(
            &args.data_dir,
            &args.tokenizer,
            model_config.max_seq_len,
            args.batch_size,
        )?;

        loop {
            if args.max_steps > 0 && trainer.global_step >= args.max_steps {
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

            let m = trainer.step(&batches)?;

            if args.log_every > 0 && m.step % args.log_every == 0 {
                eprintln!("step {} loss {:.4} lr {:.2e}", m.step, m.loss, m.lr);
            }
            if let Some(gn) = m.grad_norm {
                eprintln!("  [debug] grad_norm {gn:.4}");
            }

            run_eval_and_checkpoint(&trainer, &val_dataset, &mut metrics_file, &args)?;
        }
    } else {
        let mut dataset =
            TextDataset::new(&args.data_dir, &args.tokenizer, model_config.max_seq_len)?;
        dataset.load()?;
        eprintln!(
            "Loaded {} tokens, {} sequences",
            dataset.num_tokens(),
            dataset.num_sequences()
        );
        if dataset.num_sequences() == 0 {
            anyhow::bail!("No sequences; need at least seq_len+1 tokens");
        }

        let mut epoch = 0u32;
        loop {
            if args.max_steps > 0 && trainer.global_step >= args.max_steps {
                break;
            }
            if args.max_epochs > 0 && (epoch as usize) >= args.max_epochs {
                eprintln!("Completed {} epochs.", args.max_epochs);
                break;
            }
            eprintln!("epoch {epoch} (step {})", trainer.global_step);

            let mut batch_iter = dataset
                .batches(args.batch_size)
                .collect::<Vec<_>>()
                .into_iter()
                .peekable();

            while batch_iter.peek().is_some() {
                if args.max_steps > 0 && trainer.global_step >= args.max_steps {
                    break;
                }
                let batches: Vec<_> = batch_iter.by_ref().take(args.accumulation_steps).collect();
                if batches.is_empty() {
                    break;
                }

                let m = trainer.step(&batches)?;

                if args.log_every > 0 && m.step % args.log_every == 0 {
                    eprintln!(
                        "step {} epoch {epoch} loss {:.4} lr {:.2e}",
                        m.step, m.loss, m.lr
                    );
                }
                if let Some(gn) = m.grad_norm {
                    eprintln!("  [debug] grad_norm {gn:.4}");
                }

                run_eval_and_checkpoint(&trainer, &val_dataset, &mut metrics_file, &args)?;
            }
            epoch += 1;
        }
    }

    let path = trainer.save_final()?;
    eprintln!("Training done. Saved to {}", path.display());
    Ok(())
}

fn run_eval_and_checkpoint(
    trainer: &Trainer,
    val_dataset: &Option<TextDataset>,
    metrics_file: &mut Option<std::fs::File>,
    args: &Args,
) -> anyhow::Result<()> {
    let step = trainer.global_step;

    // Validation
    if let Some(ref val_ds) = val_dataset {
        if args.eval_every > 0 && step > 0 && step % args.eval_every == 0 {
            let (val_loss, ppl) = trainer.evaluate(val_ds)?;
            eprintln!("  [eval] step {step} val_loss={val_loss:.4} perplexity={ppl:.2}");
            if let Some(ref mut f) = metrics_file {
                writeln!(f, "{step},{val_loss},{ppl}")?;
            }
        }
    }

    // Checkpoint
    if args.save_every > 0 && step > 0 && step % args.save_every == 0 {
        let path = trainer.save_checkpoint()?;
        eprintln!("  Saved checkpoint to {}", path.display());
    }
    Ok(())
}
