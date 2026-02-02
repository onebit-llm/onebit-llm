//! CLI for exporting model (safetensors + config + optional tokenizer).
//!
//! Usage: export --checkpoint-dir ./checkpoints --output-dir ./exported [--tokenizer tokenizer.json]

use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;

use onebit_llm::OneBitLlmConfig;

#[derive(Parser, Debug)]
#[command(name = "export", about = "Export model to a directory (safetensors + config)")]
struct Args {
    /// Checkpoint directory (contains model.safetensors or checkpoint-*.safetensors and config.json).
    #[arg(long)]
    checkpoint_dir: PathBuf,

    /// Output directory (will contain model.safetensors, config.json).
    #[arg(long)]
    output_dir: PathBuf,

    /// Optional: path to tokenizer.json to copy into output_dir.
    #[arg(long)]
    tokenizer: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    std::fs::create_dir_all(&args.output_dir)?;

    let config_path = args.checkpoint_dir.join("config.json");
    let config = OneBitLlmConfig::load(&config_path)
        .context("load config (run train first or provide config.json in checkpoint dir)")?;
    config.save(&args.output_dir.join("config.json"))?;

    let weights_source = if args.checkpoint_dir.join("model.safetensors").exists() {
        args.checkpoint_dir.join("model.safetensors")
    } else {
        let mut latest: Option<(u64, PathBuf)> = None;
        for entry in std::fs::read_dir(&args.checkpoint_dir)? {
            let entry = entry?;
            let p = entry.path();
            if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("checkpoint-") && name.ends_with(".safetensors") {
                    let n = name
                        .trim_start_matches("checkpoint-")
                        .trim_end_matches(".safetensors");
                    if let Ok(step) = n.parse::<u64>() {
                        if latest.as_ref().map(|(s, _)| step > *s).unwrap_or(true) {
                            latest = Some((step, p));
                        }
                        continue;
                    }
                }
            }
        }
        latest
            .map(|(_, p)| p)
            .ok_or_else(|| anyhow::anyhow!("no model.safetensors or checkpoint-*.safetensors in {}", args.checkpoint_dir.display()))?
    };

    let weights_dest = args.output_dir.join("model.safetensors");
    std::fs::copy(&weights_source, &weights_dest)
        .context("copy weights to output dir")?;
    eprintln!("Copied weights to {}", weights_dest.display());

    if let Some(tok_path) = &args.tokenizer {
        if tok_path.exists() {
            let dest = args.output_dir.join("tokenizer.json");
            std::fs::copy(tok_path, &dest).context("copy tokenizer")?;
            eprintln!("Copied tokenizer to {}", dest.display());
        }
    }

    eprintln!("Export done. Output: {}", args.output_dir.display());
    Ok(())
}
