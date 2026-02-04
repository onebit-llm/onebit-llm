//! Quantization search CLI

use clap::Parser;
use onebit_llm::{SearchConfig, SearchCoordinator};

#[derive(Parser)]
#[command(name = "onebit-search")]
#[command(about = "Expander-based quantization search for onebit-llm")]
struct Args {
    /// Model config path
    #[arg(long, default_value = "config.json")]
    model_config: String,

    /// Checkpoint path
    #[arg(long, default_value = "checkpoints/model.safetensors")]
    checkpoint: String,

    /// Validation data path
    #[arg(long, default_value = "data/val.txt")]
    val_data: String,

    /// Tokenizer path (e.g. tokenizer.json)
    #[arg(long, default_value = "tokenizer.json")]
    tokenizer: String,

    /// Maximum model size in MB
    #[arg(long)]
    max_size_mb: Option<f64>,

    /// Minimum accuracy (negative loss)
    #[arg(long)]
    min_accuracy: Option<f64>,

    /// Max evaluations
    #[arg(long, default_value_t = 1000)]
    max_evaluations: usize,

    /// Partition size (√V typically)
    #[arg(long, default_value_t = 100)]
    partition_size: usize,

    /// Partition overlap ratio
    #[arg(long, default_value_t = 0.15)]
    overlap_ratio: f64,

    /// Number of threads
    #[arg(long, default_value_t = 8)]
    num_threads: usize,

    /// Output path
    #[arg(long, default_value = "search_result.json")]
    output: String,
}

fn main() -> anyhow::Result<()> {
    // Setup logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    // Setup rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.num_threads)
        .build_global()
        .unwrap();

    // Build search config
    let search_config = SearchConfig {
        model_config: args.model_config,
        checkpoint: args.checkpoint,
        val_data: args.val_data,
        tokenizer: args.tokenizer,
        max_size_mb: args.max_size_mb,
        min_accuracy: args.min_accuracy,
        max_evaluations: args.max_evaluations,
        partition_size: args.partition_size,
        overlap_ratio: args.overlap_ratio,
        num_threads: args.num_threads,
        output_path: args.output.clone(),
    };

    // Run search
    let coordinator = SearchCoordinator::new(search_config)?;
    let result = coordinator.search()?;

    // Save result
    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&args.output, json)?;

    println!("\n=== Search Complete ===");
    println!("Perplexity: {:.2}", result.perplexity);
    println!("Model size: {:.1} MB", result.size_mb);
    println!("Compression: {:.1}x", result.compression_ratio);
    println!("Search time: {:.1}s", result.search_time_secs);
    println!("Result saved to: {}", args.output);

    Ok(())
}
