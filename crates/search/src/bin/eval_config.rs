//! Evaluate a single quantisation config on the validation set.

use std::path::PathBuf;

use candle_core::{DType, Device};
use clap::Parser;
use serde::Serialize;

use ternary_common::{batch_to_tensors, OneBitLlmConfig, TextDataset};
use ternary_core::OneBitLlm;
use ternary_search::{config_key, QuantConfig, QuantLevel};

#[derive(Parser, Debug)]
#[command(name = "onebit-eval-config", about = "Evaluate a single quant config")]
struct Args {
    #[arg(long)]
    model_config: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    val_data: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    quant_config: PathBuf,
    #[arg(long, default_value_t = 10)]
    max_eval_batches: usize,
    #[arg(long, default_value_t = 8)]
    batch_size: usize,
}

#[derive(Serialize)]
struct EvalResult {
    config_key: String,
    loss: f64,
    perplexity: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;

    let mut model_config = OneBitLlmConfig::load(&args.model_config)?;
    let quant_config: QuantConfig = {
        let data = std::fs::read_to_string(&args.quant_config)?;
        serde_json::from_str(&data)?
    };

    let binary_count = (0..quant_config.num_layers)
        .filter(|&i| quant_config.get_layer(i) == QuantLevel::Binary)
        .count();
    let ternary_count = (0..quant_config.num_layers)
        .filter(|&i| quant_config.get_layer(i) == QuantLevel::Ternary)
        .count();
    model_config.use_ternary = ternary_count > binary_count;

    let mut val_dataset =
        TextDataset::new(&args.val_data, &args.tokenizer, model_config.max_seq_len)?;
    val_dataset.load()?;

    let mut varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = OneBitLlm::new(vb, &model_config)?;
    varmap.load(&args.checkpoint)?;

    let seq_len = model_config.max_seq_len;
    let mut total_loss = 0.0;
    let mut num_batches = 0usize;

    for (input_ids, labels) in val_dataset
        .batches(args.batch_size)
        .take(args.max_eval_batches)
    {
        let (input_ids, labels) =
            batch_to_tensors(&input_ids, &labels, args.batch_size, seq_len, &device)?;
        let logits = model.forward(&input_ids)?;
        let (b, t, v) = logits.dims3()?;
        let logits_flat = logits.reshape((b * t, v))?;
        let labels_flat = labels.reshape((b * t,))?.to_dtype(DType::U32)?;
        let loss = candle_nn::loss::cross_entropy(&logits_flat, &labels_flat)?;
        total_loss += loss.to_scalar::<f32>()? as f64;
        num_batches += 1;
    }

    let avg_loss = if num_batches > 0 {
        total_loss / num_batches as f64
    } else {
        f64::MAX
    };

    let result = EvalResult {
        config_key: config_key(&quant_config),
        loss: avg_loss,
        perplexity: avg_loss.exp(),
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
