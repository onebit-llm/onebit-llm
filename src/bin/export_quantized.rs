//! CLI for exporting a true ternary quantized checkpoint (inference-ready).
//!
//! Loads F32 checkpoint, quantizes c_attn/c_proj/c_fc layers to {-1,0,+1} (stored as F32),
//! keeps embeddings and norms in F32. Quantized layers use 4 bytes per param but only 3 values;
//! for further size reduction use external tools to pack to 2 bits. Saves to a new safetensors file.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::Context;
use candle::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use clap::Parser;

use onebit_llm::{ternary_quantize_forward, OneBitLlm, OneBitLlmConfig};

fn quantize_ternary(tensor: &Tensor, use_dynamic_threshold: bool) -> Result<Tensor> {
    ternary_quantize_forward(tensor, use_dynamic_threshold)
}

#[derive(Parser, Debug)]
#[command(
    name = "export_quantized",
    about = "Export ternary quantized checkpoint (F32 -> i8 for bit layers)"
)]
struct Args {
    /// Path to config JSON.
    #[arg(long, default_value = "config.json")]
    config: PathBuf,

    /// Path to F32 checkpoint (model.safetensors or checkpoint-*.safetensors).
    #[arg(long)]
    checkpoint: PathBuf,

    /// Output path for quantized checkpoint (e.g. model_quantized.safetensors).
    #[arg(long)]
    output: PathBuf,

    /// Use dynamic threshold (0.7 * mean(|W|)) for ternary; same as config.
    #[arg(long, default_value = "true")]
    dynamic_threshold: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let config = OneBitLlmConfig::load(&args.config).context("load config")?;
    let device = Device::cuda_if_available(0)?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = OneBitLlm::new(vb, &config)?;
    varmap
        .load(&args.checkpoint)
        .context("load F32 checkpoint")?;

    let mut quantized_tensors: HashMap<String, Tensor> = HashMap::new();
    let data = varmap
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("varmap lock"))?;
    for (name, var) in data.iter() {
        let tensor = var.as_tensor().clone();
        let out = if name.contains("c_attn") || name.contains("c_proj") || name.contains("c_fc") {
            quantize_ternary(&tensor, args.dynamic_threshold)?
        } else {
            tensor
        };
        quantized_tensors.insert(name.clone(), out);
    }
    drop(data);

    candle::safetensors::save(&quantized_tensors, &args.output)
        .context("save quantized checkpoint")?;

    println!(
        "Exported quantized checkpoint to: {}",
        args.output.display()
    );
    Ok(())
}
