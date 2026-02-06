//! Quantisation search CLI (tokio-based async coordinator).

use clap::Parser;
use ternary_search::{SearchConfig, SearchCoordinator};

#[derive(Parser)]
#[command(name = "onebit-search", about = "Expander-based quantisation search")]
struct Args {
    #[arg(long, default_value = "config.json")]
    model_config: String,
    #[arg(long, default_value = "checkpoints/model.safetensors")]
    checkpoint: String,
    #[arg(long, default_value = "data/val.txt")]
    val_data: String,
    #[arg(long, default_value = "tokenizer.json")]
    tokenizer: String,
    #[arg(long)]
    max_size_mb: Option<f64>,
    #[arg(long)]
    min_accuracy: Option<f64>,
    /// Reject configs with perplexity above this (min-perplexity constraint).
    #[arg(long)]
    min_perplexity_max: Option<f64>,
    #[arg(long, default_value_t = 1000)]
    max_evaluations: usize,
    #[arg(long, default_value_t = 100)]
    partition_size: usize,
    #[arg(long, default_value_t = 0.15)]
    overlap_ratio: f64,
    #[arg(long, default_value_t = 8)]
    num_threads: usize,
    #[arg(long, default_value = "search_result.json")]
    output: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()
        .unwrap();

    let search_config = SearchConfig {
        model_config: args.model_config,
        checkpoint: args.checkpoint,
        val_data: args.val_data,
        tokenizer: args.tokenizer,
        max_size_mb: args.max_size_mb,
        min_accuracy: args.min_accuracy,
        min_perplexity_max: args.min_perplexity_max,
        max_evaluations: args.max_evaluations,
        partition_size: args.partition_size,
        overlap_ratio: args.overlap_ratio,
        num_threads: args.num_threads,
        output_path: args.output.clone(),
    };

    let coordinator = SearchCoordinator::new(search_config)?;
    let result = coordinator.search_async().await?;

    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&args.output, json)?;

    println!("\n=== Search Complete ===");
    println!("Perplexity: {:.2}", result.perplexity);
    println!("Model size: {:.1} MB", result.size_mb);
    println!("Compression: {:.1}x", result.compression_ratio);
    println!("Result saved to: {}", args.output);
    Ok(())
}
