//! Example: Train a mixed-precision (Sandwich Rule) model on WikiText.
//!
//! Prerequisites:
//!   1. Download data: `python scripts/download_wikitext.py`
//!   2. Tokenize: `cargo run -p ternary-train --bin onebit-tokenize -- --data_dir data/wikitext-103-raw --tokenizer tokenizer.json --out_dir data/tokenized`
//!
//! Run:
//!   cargo run -p ternary-train --example train_wikitext -- --config config_wikitext.json --data_dir data/tokenized --tokenizer tokenizer.json --output_dir checkpoints

use std::path::PathBuf;

use candle_core::Device;
use clap::Parser;

use ternary_common::{OneBitLlmConfig, TextDataset};
use ternary_train::{LrDecay, Trainer, TrainerConfig};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value = "config.json")]
    config: PathBuf,
    #[arg(long)]
    data_dir: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long, default_value = "checkpoints")]
    output_dir: PathBuf,
    #[arg(long, default_value = "8")]
    batch_size: usize,
    #[arg(long, default_value = "1000")]
    max_steps: usize,
    #[arg(long, default_value = "5e-3")]
    lr: f64,
    #[arg(long, default_value = "cosine", value_parser = ["cosine", "linear", "none"])]
    lr_decay: String,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let model_config = if args.config.exists() {
        OneBitLlmConfig::load(&args.config)?
    } else {
        let default = OneBitLlmConfig::default();
        default.save(&args.config)?;
        eprintln!("Created default config at {}", args.config.display());
        default
    };

    let lr_decay = match args.lr_decay.as_str() {
        "cosine" => LrDecay::Cosine,
        "linear" => LrDecay::Linear,
        _ => LrDecay::None,
    };

    let trainer_config = TrainerConfig {
        batch_size: args.batch_size,
        accumulation_steps: 1,
        max_steps: args.max_steps,
        max_epochs: 0,
        lr: args.lr,
        lr_min: 1e-6,
        lr_warmup_steps: 200,
        lr_decay,
        weight_decay: 0.0,
        grad_clip_max_norm: 1.0,
        label_smoothing: 0.1,
        save_every: 500,
        log_every: 50,
        debug_every: 0,
        eval_every: 0,
        eval_batches: 0,
        output_dir: args.output_dir,
    };

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let mut trainer = Trainer::new(model_config.clone(), trainer_config, device)?;

    let seq_len = model_config.max_seq_len + 1;
    let mut dataset = TextDataset::new(&args.data_dir, &args.tokenizer, seq_len)?;
    dataset.load()?;
    if dataset.num_sequences() == 0 {
        anyhow::bail!("No sequences; need at least seq_len+1 tokens");
    }

    let all_batches: Vec<_> = dataset.batches(trainer.config.batch_size).collect();
    let mut batch_iter = all_batches.into_iter().peekable();

    while batch_iter.peek().is_some() && trainer.global_step < trainer.config.max_steps {
        let batches: Vec<_> = batch_iter.by_ref().take(trainer.config.accumulation_steps).collect();
        if batches.is_empty() {
            break;
        }
        let metrics = trainer.step(&batches)?;
        if (metrics.step + 1) % trainer.config.log_every == 0 {
            tracing::info!(step = metrics.step, loss = %metrics.loss, lr = %metrics.lr, "step");
        }
        if trainer.config.save_every > 0 && (metrics.step + 1) % trainer.config.save_every == 0 {
            let path = trainer.save_checkpoint()?;
            tracing::info!(path = %path.display(), "checkpoint saved");
        }
    }

    let path = trainer.save_final()?;
    tracing::info!(path = %path.display(), "final model saved");
    Ok(())
}
