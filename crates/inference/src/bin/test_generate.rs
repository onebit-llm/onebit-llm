//! Quick inference test: generate text from a prompt without interactive mode.

use std::path::PathBuf;

use candle_core::Device;
use clap::Parser;

use ternary_infer::{InferenceRuntime, SamplerConfig};

#[derive(Parser)]
#[command(name = "onebit-test-generate", about = "Generate text from a trained model")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value = "The cat sat")]
    prompt: String,
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,
    #[arg(long, default_value_t = 50)]
    top_k: usize,
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,
    #[arg(long, default_value_t = 1.2)]
    repetition_penalty: f64,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::cuda_if_available(0)?;

    let sampler = SamplerConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        ..Default::default()
    };

    eprintln!("Loading model from {} ...", args.model_dir.display());
    let mut runtime = InferenceRuntime::load(&args.model_dir, sampler, device)?;
    eprintln!("Model loaded. Generating...\n");

    // Test multiple prompts
    let prompts = [
        &args.prompt as &str,
        "The rain fell",
        "A small boat",
        "The scientist",
        "Stars twinkled",
    ];

    for prompt in prompts {
        eprintln!("--- Prompt: \"{}\" ---", prompt);
        let output = runtime.generate(prompt, args.max_tokens)?;
        println!("{}", output);
        println!();
    }

    Ok(())
}
