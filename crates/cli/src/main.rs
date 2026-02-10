use std::path::PathBuf;

use anyhow::Result;
use candle_core::Device;
use clap::{Parser, Subcommand};

use ternary_common::{
    batch_to_tensors, AnyBatchDataset, BatchDataset, OneBitLlmConfig, StreamingBatchIter,
    TextDataset,
};
use ternary_core::OneBitLlm;
use ternary_infer::{export_1bit, generate_c_header, InferenceRuntime, PackMode, SamplerConfig};
use ternary_search::{config_key, QuantConfig, SearchConfig, SearchCoordinator};
use ternary_train::{LrDecay, Trainer, TrainerConfig};

/// Number of batches to prefetch so the training thread is not starved.
const PREFETCH_BUFFER: usize = 8;

#[derive(Parser, Debug)]
#[command(name = "onebit", about = "Unified CLI for OneBit-LLM")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Train a 1-bit / ternary LLM from scratch or continue training.
    Train(TrainArgs),
    /// Pre-tokenise text data into a .tokens mmap file.
    Tokenize(TokenizeArgs),
    /// Interactive chat with a trained model directory.
    Chat(ChatArgs),
    /// One-off text generation for quick smoke tests.
    Generate(GenerateArgs),
    /// Export a trained model to packed .1bit format (optional C header).
    Export(ExportArgs),
    /// Run quantisation search over layer-wise bit-width maps.
    Search(SearchArgs),
    /// Evaluate a single quantisation config on a validation set.
    EvalConfig(EvalConfigArgs),
}

// ── Train ──────────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
struct TrainArgs {
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
    /// Batch size per micro-step; larger values improve GPU utilisation (e.g. 32–64).
    #[arg(long, default_value = "32")]
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
    /// Gradient accumulation steps; effective batch = batch_size * accumulation_steps.
    #[arg(long, default_value = "4")]
    accumulation_steps: usize,
    #[arg(long)]
    val_data_dir: Option<PathBuf>,
    #[arg(long, default_value = "500")]
    eval_every: usize,
    #[arg(long, default_value = "50")]
    eval_batches: usize,
    /// QAT warmup steps before enabling any quantisation in BitLinear.
    #[arg(long, default_value = "2000")]
    quant_warmup_steps: usize,
    /// QAT annealing steps (after warmup) to go from soft→hard quantisation.
    #[arg(long, default_value = "8000")]
    quant_anneal_steps: usize,
}

// ── Tokenize ───────────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
struct TokenizeArgs {
    #[arg(long)]
    data_dir: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    seq_len: usize,
    #[arg(long)]
    output: PathBuf,
}

// ── Chat / Generate / Export ───────────────────────────────────────────────────

#[derive(Parser, Debug)]
struct ChatArgs {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value_t = 256)]
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

#[derive(Parser, Debug)]
struct GenerateArgs {
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

#[derive(Parser, Debug)]
struct ExportArgs {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long, default_value = "model.1bit")]
    output: PathBuf,
    #[arg(long, default_value = "ternary", value_parser = ["binary", "ternary"])]
    pack_mode: String,
    #[arg(long)]
    emit_c_header: bool,
}

// ── Search / Eval-config ───────────────────────────────────────────────────────

#[derive(Parser, Debug)]
struct SearchArgs {
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

#[derive(Parser, Debug)]
struct EvalConfigArgs {
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

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Train(args) => cmd_train(args),
        Command::Tokenize(args) => cmd_tokenize(args),
        Command::Chat(args) => cmd_chat(args),
        Command::Generate(args) => cmd_generate(args),
        Command::Export(args) => cmd_export(args),
        Command::Search(args) => cmd_search(args),
        Command::EvalConfig(args) => cmd_eval_config(args),
    }
}

// ── Command implementations ────────────────────────────────────────────────────

fn cmd_train(args: TrainArgs) -> Result<()> {
    use std::io::Write;
    use std::sync::mpsc;
    use std::thread;

    enum PrefetchMessage {
        Batch((Vec<u32>, Vec<u32>)),
        EpochEnd,
        Done,
    }

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
        quant_warmup_steps: args.quant_warmup_steps,
        quant_anneal_steps: args.quant_anneal_steps,
    };

    let device = Device::cuda_if_available(0)?;
    let mut trainer = Trainer::new(model_config.clone(), trainer_config, device)?;

    // Optional validation dataset.
    let val_dataset: Option<AnyBatchDataset> = if let Some(ref p) = args.val_data_dir {
        let seq_len = model_config.max_seq_len;
        if p.is_file() && p.extension().map(|e| e == "tokens").unwrap_or(false) {
            let ds = ternary_common::MmapDataset::open(p, seq_len)?;
            eprintln!("Validation (mmap): {} sequences", ds.num_sequences());
            Some(AnyBatchDataset::Mmap(ds))
        } else {
            let mut ds = TextDataset::new(p, &args.tokenizer, seq_len)?;
            ds.load()?;
            eprintln!("Validation: {} sequences", ds.num_sequences());
            Some(AnyBatchDataset::Text(ds))
        }
    } else {
        None
    };

    std::fs::create_dir_all(&args.output_dir)?;

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

        let mut last_loss: Option<f32> = None;
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
            last_loss = Some(m.loss);

            if args.log_every > 0 && m.step % args.log_every == 0 {
                eprintln!("step {} loss {:.4} lr {:.2e}", m.step, m.loss, m.lr);
            }
            if m.grad_norm.is_some() || m.weight_mean_0.is_some() || m.weight_mean_1.is_some() {
                let gn = m
                    .grad_norm
                    .map(|g| format!("{g:.6}"))
                    .unwrap_or_else(|| "n/a".to_string());
                let w0 = match (m.weight_mean_0, m.weight_std_0) {
                    (Some(mu), Some(s)) => format!("w0 mean={mu:.6} std={s:.6}"),
                    _ => String::new(),
                };
                let w1 = match (m.weight_mean_1, m.weight_std_1) {
                    (Some(mu), Some(s)) => format!("w1 mean={mu:.6} std={s:.6}"),
                    _ => String::new(),
                };
                eprintln!("  [debug] grad_norm={gn}  {w0}  {w1}");
            }

            run_eval_and_checkpoint(&trainer, &val_dataset, &mut metrics_file, &args)?;
        }
        if let Some(loss) = last_loss {
            eprintln!("step {} epoch 0 loss {:.4} (final)", trainer.global_step, loss);
        }
    } else {
        let seq_len = model_config.max_seq_len;
        let dataset: AnyBatchDataset = if args.data_dir.is_file()
            && args
                .data_dir
                .extension()
                .map(|e| e == "tokens")
                .unwrap_or(false)
        {
            eprintln!("Using mmap dataset: {}", args.data_dir.display());
            let ds = ternary_common::MmapDataset::open(&args.data_dir, seq_len)?;
            eprintln!(
                "Loaded {} tokens, {} sequences (zero-copy)",
                ds.num_tokens(),
                ds.num_sequences()
            );
            AnyBatchDataset::Mmap(ds)
        } else {
            let mut ds = TextDataset::new(&args.data_dir, &args.tokenizer, seq_len)?;
            ds.load()?;
            eprintln!(
                "Loaded {} tokens, {} sequences",
                ds.num_tokens(),
                ds.num_sequences()
            );
            AnyBatchDataset::Text(ds)
        };

        if dataset.num_sequences() == 0 {
            anyhow::bail!("No sequences; need at least seq_len+1 tokens");
        }

        eprintln!(
            "Prefetch buffer: {} batches (batch_size={}, accumulation_steps={}, effective_batch={})",
            PREFETCH_BUFFER,
            args.batch_size,
            args.accumulation_steps,
            args.batch_size * args.accumulation_steps
        );

        // Producer thread owns the dataset and prefetches batches so the GPU is not starved.
        let max_epochs_producer = if args.max_epochs == 0 {
            10000
        } else {
            args.max_epochs
        };
        let (tx, rx) = mpsc::sync_channel::<PrefetchMessage>(PREFETCH_BUFFER);
        let batch_size = args.batch_size;
        let producer = thread::spawn(move || {
            for _ in 0..max_epochs_producer {
                for batch in dataset.batches(batch_size) {
                    if tx.send(PrefetchMessage::Batch(batch)).is_err() {
                        return;
                    }
                }
                if tx.send(PrefetchMessage::EpochEnd).is_err() {
                    return;
                }
            }
            let _ = tx.send(PrefetchMessage::Done);
        });

        let mut epoch = 0u32;
        let mut acc = Vec::with_capacity(args.accumulation_steps);
        let mut last_loss: Option<f32> = None;
        while let Ok(msg) = rx.recv() {
            if args.max_steps > 0 && trainer.global_step >= args.max_steps {
                break;
            }
            if args.max_epochs > 0 && (epoch as usize) >= args.max_epochs {
                break;
            }
            match msg {
                PrefetchMessage::Batch(batch) => {
                    acc.push(batch);
                    if acc.len() >= args.accumulation_steps {
                        let m = trainer.step(&acc)?;
                        last_loss = Some(m.loss);
                        acc.clear();
                        if args.log_every > 0 && m.step % args.log_every == 0 {
                            eprintln!(
                                "step {} epoch {epoch} loss {:.4} lr {:.2e}",
                                m.step, m.loss, m.lr
                            );
                        }
                        if m.grad_norm.is_some()
                            || m.weight_mean_0.is_some()
                            || m.weight_mean_1.is_some()
                        {
                            let gn = m
                                .grad_norm
                                .map(|g| format!("{g:.6}"))
                                .unwrap_or_else(|| "n/a".to_string());
                            let w0 = match (m.weight_mean_0, m.weight_std_0) {
                                (Some(mu), Some(s)) => format!("w0 mean={mu:.6} std={s:.6}"),
                                _ => String::new(),
                            };
                            let w1 = match (m.weight_mean_1, m.weight_std_1) {
                                (Some(mu), Some(s)) => format!("w1 mean={mu:.6} std={s:.6}"),
                                _ => String::new(),
                            };
                            eprintln!("  [debug] grad_norm={gn}  {w0}  {w1}");
                        }
                        run_eval_and_checkpoint(&trainer, &val_dataset, &mut metrics_file, &args)?;
                    }
                }
                PrefetchMessage::EpochEnd => {
                    epoch += 1;
                    eprintln!("epoch {epoch} (step {})", trainer.global_step);
                }
                PrefetchMessage::Done => break,
            }
        }
        if !acc.is_empty() {
            let m = trainer.step(&acc)?;
            last_loss = Some(m.loss);
            if args.log_every > 0 {
                eprintln!(
                    "step {} epoch {epoch} loss {:.4} lr {:.2e}",
                    m.step, m.loss, m.lr
                );
            }
            run_eval_and_checkpoint(&trainer, &val_dataset, &mut metrics_file, &args)?;
        }
        drop(rx);
        let _ = producer.join();
        if let Some(loss) = last_loss {
            eprintln!(
                "step {} epoch {epoch} loss {:.4} (final)",
                trainer.global_step, loss
            );
        }
    }

    let path = trainer.save_final()?;
    eprintln!("Training done. Saved to {}", path.display());
    Ok(())
}

fn run_eval_and_checkpoint(
    trainer: &Trainer,
    val_dataset: &Option<AnyBatchDataset>,
    metrics_file: &mut Option<std::fs::File>,
    args: &TrainArgs,
) -> Result<()> {
    use std::io::Write;

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

fn cmd_tokenize(args: TokenizeArgs) -> Result<()> {
    let mut dataset = TextDataset::new(&args.data_dir, &args.tokenizer, args.seq_len)?;
    dataset.load()?;
    let n = dataset.num_tokens();
    let seqs = dataset.num_sequences();
    dataset.write_tokenized(&args.output)?;
    eprintln!(
        "Wrote {} tokens ({} sequences) to {}",
        n,
        seqs,
        args.output.display()
    );
    eprintln!(
        "Train with mmap: --data-dir {} (zero-copy)",
        args.output.display()
    );
    Ok(())
}

fn build_sampler(
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: f64,
) -> SamplerConfig {
    SamplerConfig {
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        ..Default::default()
    }
}

fn cmd_chat(args: ChatArgs) -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let sampler = build_sampler(
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
    );

    eprintln!("Loading model from {} ...", args.model_dir.display());
    let mut runtime = InferenceRuntime::load(&args.model_dir, sampler, device)?;
    eprintln!("Ready. Type 'quit' to exit.\n");
    runtime.chat_loop(args.max_tokens)?;
    Ok(())
}

fn cmd_generate(args: GenerateArgs) -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    let sampler = build_sampler(
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
    );

    eprintln!("Loading model from {} ...", args.model_dir.display());
    let mut runtime = InferenceRuntime::load(&args.model_dir, sampler, device)?;
    eprintln!("Model loaded. Generating...\n");

    let prompts = [
        args.prompt.as_str(),
        "The rain fell",
        "A small boat",
        "The scientist",
        "Stars twinkled",
    ];

    for prompt in prompts {
        eprintln!("--- Prompt: \"{}\" ---", prompt);
        let output = runtime.generate(prompt, args.max_tokens)?;
        println!("{output}\n");
    }

    Ok(())
}

fn cmd_export(args: ExportArgs) -> Result<()> {
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};

    let device = Device::Cpu;
    let config = OneBitLlmConfig::load(&args.model_dir.join("config.json"))?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let _model = OneBitLlm::new(vb, &config)?;
    varmap.load(&args.model_dir.join("model.safetensors"))?;

    let pack_mode = match args.pack_mode.as_str() {
        "binary" => PackMode::Binary,
        _ => PackMode::Ternary,
    };

    eprintln!("Exporting to {} ...", args.output.display());
    let total = export_1bit(&varmap, &config, &args.output, pack_mode)?;
    eprintln!(
        "Done. Packed data: {} bytes ({:.2} MB)",
        total,
        total as f64 / 1024.0 / 1024.0
    );

    if args.emit_c_header {
        let header = generate_c_header();
        let header_path = args.output.with_extension("h");
        std::fs::write(&header_path, header)?;
        eprintln!("C header: {}", header_path.display());
    }

    Ok(())
}

fn cmd_search(args: SearchArgs) -> Result<()> {
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
    let rt = tokio::runtime::Runtime::new()?;
    let result = rt.block_on(coordinator.search_async())?;

    let json = serde_json::to_string_pretty(&result)?;
    std::fs::write(&args.output, json)?;

    println!("\n=== Search Complete ===");
    println!("Perplexity: {:.2}", result.perplexity);
    println!("Model size: {:.1} MB", result.size_mb);
    println!("Compression: {:.1}x", result.compression_ratio);
    println!("Result saved to: {}", args.output);
    Ok(())
}

fn cmd_eval_config(args: EvalConfigArgs) -> Result<()> {
    use candle_core::DType;
    use candle_nn::VarBuilder;
    use candle_nn::VarMap;
    use serde::Serialize;

    #[derive(Serialize)]
    struct EvalResult {
        config_key: String,
        loss: f64,
        perplexity: f64,
    }

    let device = Device::cuda_if_available(0)?;

    let mut model_config = OneBitLlmConfig::load(&args.model_config)?;
    let quant_config: QuantConfig = {
        let data = std::fs::read_to_string(&args.quant_config)?;
        serde_json::from_str(&data)?
    };

    let binary_count = (0..quant_config.num_layers)
        .filter(|&i| quant_config.get_layer(i) == ternary_search::QuantLevel::Binary)
        .count();
    let ternary_count = (0..quant_config.num_layers)
        .filter(|&i| quant_config.get_layer(i) == ternary_search::QuantLevel::Ternary)
        .count();
    model_config.use_ternary = ternary_count > binary_count;

    let mut val_dataset =
        TextDataset::new(&args.val_data, &args.tokenizer, model_config.max_seq_len)?;
    val_dataset.load()?;

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
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

