//! CLI for importing and running inference (or optionally resume training).
//!
//! Usage: run --model-dir ./exported [--prompt "Hello world"] [--max-tokens 64]
//!        run --model-dir ./checkpoints --chat  (interactive chat)

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::Context;
use candle::{DType, Device, IndexOp};
use candle_nn::{sampling, VarBuilder, VarMap};
use clap::Parser;

/// Sample next token from logits: greedy (temperature 0) or temperature sampling via Gumbel-Softmax.
fn sample_next_token(logits: &candle::Tensor, temperature: f64) -> anyhow::Result<u32> {
    let sampled = sampling::gumbel_softmax(logits, temperature, 0)?;
    let token: u32 = if let Ok(t) = sampled.to_scalar::<u32>() {
        t
    } else {
        sampled.to_scalar::<i64>()? as u32
    };
    Ok(token)
}
use tokenizers::Tokenizer;

use onebit_llm::{batch_to_tensors, OneBitLlm, OneBitLlmConfig, TextDataset};

#[derive(Parser, Debug)]
#[command(name = "run", about = "Load model and run inference")]
struct Args {
    /// Model directory (contains model.safetensors and config.json).
    #[arg(long)]
    model_dir: PathBuf,

    /// Optional: tokenizer.json path (default: model_dir/tokenizer.json).
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Optional: prompt for one-shot generation (if omitted and not --chat, runs dummy forward).
    #[arg(long)]
    prompt: Option<String>,

    /// Interactive chat: read lines from stdin, generate reply, repeat. Requires tokenizer.
    #[arg(long)]
    chat: bool,

    /// Max tokens to generate per turn (when using --prompt or --chat).
    #[arg(long, default_value = "64")]
    max_tokens: usize,

    /// Sampling temperature (0 = greedy, 0.7–1.0 = more diverse; reduces repetition).
    #[arg(long, default_value = "0.8")]
    temperature: f64,

    /// Pre-compute and cache quantized weights once for faster repeated forwards (inference-only).
    #[arg(long)]
    use_cached_quantized: bool,

    /// Run N forward passes and report throughput (tokens/sec or time per forward). 0 = disabled.
    #[arg(long, default_value = "0")]
    benchmark: usize,

    /// Evaluate perplexity on a text file (one sentence per line). Requires --tokenizer. Disables chat/prompt.
    #[arg(long)]
    eval_perplexity: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let config_path = args.model_dir.join("config.json");
    let config = OneBitLlmConfig::load(&config_path)
        .context("load config (run export first)")?;

    let weights_path = {
        let primary = args.model_dir.join("model.safetensors");
        if primary.exists() {
            primary
        } else {
            // Fallback: load latest checkpoint-N.safetensors
            let mut best: Option<(u64, PathBuf)> = None;
            if let Ok(entries) = std::fs::read_dir(&args.model_dir) {
                for e in entries.flatten() {
                    let p = e.path();
                    if p.extension().map_or(false, |e| e == "safetensors") {
                        if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                            if let Some(n) = stem.strip_prefix("checkpoint-").and_then(|s| s.parse::<u64>().ok()) {
                                if best.as_ref().map_or(true, |(b, _)| n > *b) {
                                    best = Some((n, p));
                                }
                            }
                        }
                    }
                }
            }
            best.map(|(_, p)| p).unwrap_or_else(|| primary)
        }
    };
    if !weights_path.exists() {
        anyhow::bail!("weights not found at {} (no model.safetensors or checkpoint-*.safetensors)", weights_path.display());
    }

    let device = Device::cuda_if_available(0)?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = OneBitLlm::new(vb, &config)?;
    varmap.load(&weights_path)?;

    eprintln!("Loaded model from {}", args.model_dir.display());

    if args.use_cached_quantized {
        model.cache_quantized_weights().context("cache quantized weights")?;
        eprintln!("Cached quantized weights for inference.");
    }

    let tokenizer_path = args
        .tokenizer
        .clone()
        .unwrap_or_else(|| args.model_dir.join("tokenizer.json"));

    if let Some(eval_path) = &args.eval_perplexity {
        if !tokenizer_path.exists() {
            anyhow::bail!(
                "eval-perplexity requires tokenizer at {} (use --tokenizer or copy tokenizer.json)",
                tokenizer_path.display()
            );
        }
        let _tokenizer = Tokenizer::from_file(tokenizer_path.as_os_str().to_string_lossy().into_owned())
            .map_err(|e| anyhow::anyhow!("load tokenizer: {}", e))?;
        let mut dataset = TextDataset::new(eval_path, &tokenizer_path, config.max_seq_len)
            .context("create eval dataset")?;
        dataset.load()?;
        let batch_size = 8usize;
        let seq_len = config.max_seq_len;
        let mut loss_sum = 0.0f64;
        let mut count = 0usize;
        for (input_ids, labels) in dataset.batches(batch_size).take(500) {
            let (input_ids, labels) = batch_to_tensors(&input_ids, &labels, batch_size, seq_len, &device)?;
            let logits = model.forward(&input_ids)?;
            let (b, t, v) = logits.dims3()?;
            let logits_flat = logits.reshape((b * t, v))?;
            let labels_flat = labels.reshape((b * t,))?.to_dtype(DType::U32)?;
            let loss = candle_nn::loss::cross_entropy(&logits_flat, &labels_flat)?;
            loss_sum += loss.to_scalar::<f32>()? as f64;
            count += 1;
        }
        if count > 0 {
            let avg_loss = loss_sum / count as f64;
            let ppl = avg_loss.exp();
            eprintln!("Eval perplexity: loss={:.4} perplexity={:.2} ({} batches)", avg_loss, ppl, count);
        } else {
            eprintln!("No batches for eval.");
        }
        return Ok(());
    }

    if args.benchmark > 0 {
        let batch_size = 1usize;
        let seq_len = config.max_seq_len.min(128);
        let dummy_ids = vec![0u32; batch_size * seq_len];
        let input = candle::Tensor::from_vec(dummy_ids.clone(), (batch_size, seq_len), &device)?;
        let warmup = 3;
        for _ in 0..warmup {
            let _ = model.forward(&input)?;
        }
        let start = std::time::Instant::now();
        for _ in 0..args.benchmark {
            let input = candle::Tensor::from_vec(dummy_ids.clone(), (batch_size, seq_len), &device)?;
            let _ = model.forward(&input)?;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let throughput = args.benchmark as f64 / elapsed;
        eprintln!(
            "Benchmark: {} forwards in {:.2}s, {:.1} forwards/sec, {:.2} ms/forward",
            args.benchmark,
            elapsed,
            throughput,
            1000.0 / throughput
        );
        return Ok(());
    }

    if args.chat {
        if !tokenizer_path.exists() {
            anyhow::bail!(
                "tokenizer not found at {} (use --tokenizer or copy tokenizer.json to model dir)",
                tokenizer_path.display()
            );
        }
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_os_str().to_string_lossy().into_owned())
            .map_err(|e| anyhow::anyhow!("load tokenizer: {}", e))?;
        eprintln!("Chat mode. Type a message and press Enter. Empty line or 'quit' to exit.");
        eprintln!("---");
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        loop {
            print!("You: ");
            stdout.flush()?;
            let mut line = String::new();
            stdin.lock().read_line(&mut line)?;
            let line = line.trim();
            if line.is_empty() || line.eq_ignore_ascii_case("quit") || line.eq_ignore_ascii_case("exit") {
                break;
            }
            let enc = tokenizer
                .encode(line, true)
                .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
            let mut input_ids: Vec<u32> = enc.get_ids().to_vec();
            for _ in 0..args.max_tokens.saturating_sub(1) {
                let seq_len = input_ids.len().min(config.max_seq_len);
                let start = input_ids.len().saturating_sub(seq_len);
                let context: Vec<u32> = input_ids[start..].to_vec();
                let input = candle::Tensor::from_vec(context.clone(), (1, seq_len), &device)?;
                let logits = model.forward(&input)?;
                let logits = logits.i((0, seq_len - 1))?;
                let next_token = sample_next_token(&logits, args.temperature)?;
                input_ids.push(next_token);
                if next_token == tokenizer.token_to_id("[EOS]").unwrap_or(0)
                    || next_token == tokenizer.get_vocab(true).len() as u32
                {
                    break;
                }
            }
            let decoded = tokenizer
                .decode(&input_ids, true)
                .map_err(|e| anyhow::anyhow!("decode: {}", e))?;
            println!("Model: {}", decoded);
            println!();
        }
    } else if let Some(prompt) = &args.prompt {
        if !tokenizer_path.exists() {
            anyhow::bail!(
                "tokenizer not found at {} (use --tokenizer or copy tokenizer.json to model dir)",
                tokenizer_path.display()
            );
        }
        let tokenizer = Tokenizer::from_file(tokenizer_path.as_os_str().to_string_lossy().into_owned())
            .map_err(|e| anyhow::anyhow!("load tokenizer: {}", e))?;

        let enc = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| anyhow::anyhow!("encode: {}", e))?;
        let mut input_ids: Vec<u32> = enc.get_ids().to_vec();

        for _ in 0..args.max_tokens.saturating_sub(1) {
            let seq_len = input_ids.len().min(config.max_seq_len);
            let start = input_ids.len().saturating_sub(seq_len);
            let context: Vec<u32> = input_ids[start..].to_vec();
            let input = candle::Tensor::from_vec(
                context.clone(),
                (1, seq_len),
                &device,
            )?;
            let logits = model.forward(&input)?;
            let logits = logits.i((0, seq_len - 1))?;
            let next_token = sample_next_token(&logits, args.temperature)?;
            input_ids.push(next_token);
            if next_token == tokenizer.token_to_id("[EOS]").unwrap_or(0)
                || next_token == tokenizer.get_vocab(true).len() as u32
            {
                break;
            }
        }

        let decoded = tokenizer
            .decode(&input_ids, true)
            .map_err(|e| anyhow::anyhow!("decode: {}", e))?;
        println!("{}", decoded);
    } else {
        let batch_size = 1usize;
        let seq_len = config.max_seq_len.min(32);
        let dummy_ids = vec![0u32; batch_size * seq_len];
        let input = candle::Tensor::from_vec(dummy_ids, (batch_size, seq_len), &device)?;
        let logits = model.forward(&input)?;
        eprintln!(
            "Dummy forward pass: logits shape {:?}",
            logits.shape()
        );
    }

    Ok(())
}
