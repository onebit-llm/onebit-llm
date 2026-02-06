//! Create a pre-tokenized .tokens file for use with MmapDataset (zero-copy training).
//!
//! Usage: onebit-tokenize --data-dir data/wikitext-2 --tokenizer data/tokenizer.json \
//!        --seq-len 256 --output data/wikitext-2/train.tokens

use std::path::PathBuf;

use clap::Parser;
use ternary_common::TextDataset;

#[derive(Parser)]
#[command(name = "onebit-tokenize", about = "Create .tokens file for MmapDataset")]
struct Args {
    #[arg(long)]
    data_dir: PathBuf,
    #[arg(long)]
    tokenizer: PathBuf,
    #[arg(long)]
    seq_len: usize,
    #[arg(long)]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut dataset =
        TextDataset::new(&args.data_dir, &args.tokenizer, args.seq_len)?;
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
    eprintln!("Train with: --data-dir {} (mmap zero-copy)", args.output.display());
    Ok(())
}
