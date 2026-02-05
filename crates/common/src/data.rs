//! Data pipeline: text loading, tokenisation, batching.
//!
//! Supports text files or JSONL (one text field per line). Tokeniser loaded
//! from `tokenizer.json` (e.g. GPT-2 BPE). Batches are `(batch_size, seq_len)`
//! token IDs; labels for next-token prediction are `input_ids` shifted by one.
//!
//! Use [`StreamingBatchIter`] for large datasets without loading all into RAM.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result as AnyhowResult};
use candle_core::{Device, Result, Tensor};
use tokenizers::Tokenizer;

// ── TextDataset (in-memory) ─────────────────────────────────────────────────

/// Dataset over text: yields tokenised sequences from files or a directory.
pub struct TextDataset {
    path: PathBuf,
    tokenizer: Tokenizer,
    seq_len: usize,
    token_ids: Vec<u32>,
}

impl TextDataset {
    /// Create dataset. Call [`load`](Self::load) after construction.
    pub fn new(path: &Path, tokenizer_path: &Path, seq_len: usize) -> AnyhowResult<Self> {
        let tokenizer =
            Tokenizer::from_file(tokenizer_path.as_os_str().to_string_lossy().to_string())
                .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        Ok(Self {
            path: path.to_path_buf(),
            tokenizer,
            seq_len,
            token_ids: Vec::new(),
        })
    }

    /// Load and tokenise all text from path (file or directory).
    pub fn load(&mut self) -> AnyhowResult<()> {
        self.token_ids.clear();
        let path = self.path.clone();
        if path.is_file() {
            self.load_file(&path)?;
        } else if path.is_dir() {
            for entry in std::fs::read_dir(&path)? {
                let p = entry?.path();
                if p.is_file() {
                    if let Some(ext) = p.extension() {
                        if ext == "txt" || ext == "raw" || ext == "jsonl" || ext == "json" {
                            self.load_file(&p)?;
                        }
                    }
                }
            }
        } else {
            anyhow::bail!("path is neither file nor directory: {}", path.display());
        }
        Ok(())
    }

    fn load_file(&mut self, path: &Path) -> AnyhowResult<()> {
        let reader = BufReader::new(File::open(path).context("open file")?);
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let text = extract_text(line);
            let enc = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
            self.token_ids.extend(enc.get_ids());
        }
        Ok(())
    }

    pub fn num_tokens(&self) -> usize {
        self.token_ids.len()
    }

    pub fn num_sequences(&self) -> usize {
        if self.seq_len == 0 {
            0
        } else {
            self.token_ids.len().saturating_sub(1) / self.seq_len
        }
    }

    /// Yield `(input_ids, labels)` batches. Labels are shifted by 1.
    pub fn batches(&self, batch_size: usize) -> impl Iterator<Item = (Vec<u32>, Vec<u32>)> + '_ {
        let seq_len = self.seq_len;
        let tokens = &self.token_ids;
        let total = tokens.len();
        let step = seq_len;
        let mut start = 0usize;
        std::iter::from_fn(move || {
            if start + batch_size * seq_len + 1 > total {
                return None;
            }
            let mut input_batch = Vec::with_capacity(batch_size * seq_len);
            let mut label_batch = Vec::with_capacity(batch_size * seq_len);
            for b in 0..batch_size {
                let base = start + b * step;
                for i in 0..seq_len {
                    input_batch.push(tokens[base + i]);
                    label_batch.push(tokens[base + i + 1]);
                }
            }
            start += batch_size * step;
            Some((input_batch, label_batch))
        })
    }
}

// ── StreamingBatchIter ──────────────────────────────────────────────────────

/// Streaming batch iterator: reads files line-by-line without loading the
/// full dataset into memory. Use for large datasets (25 GB+).
pub struct StreamingBatchIter {
    files: Vec<PathBuf>,
    current_file_index: usize,
    current_reader: Option<BufReader<File>>,
    buffer: Vec<u32>,
    tokenizer: Tokenizer,
    seq_len: usize,
    batch_size: usize,
}

impl StreamingBatchIter {
    pub fn new(
        path: &Path,
        tokenizer_path: &Path,
        seq_len: usize,
        batch_size: usize,
    ) -> AnyhowResult<Self> {
        let tokenizer =
            Tokenizer::from_file(tokenizer_path.as_os_str().to_string_lossy().to_string())
                .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        let files = collect_files(path)?;
        let current_reader = if files.is_empty() {
            None
        } else {
            let f = File::open(&files[0]).context("open first file")?;
            Some(BufReader::new(f))
        };
        Ok(Self {
            files,
            current_file_index: 0,
            current_reader,
            buffer: Vec::with_capacity(batch_size * (seq_len + 1) * 2),
            tokenizer,
            seq_len,
            batch_size,
        })
    }

    fn fill_buffer(&mut self) -> AnyhowResult<bool> {
        let need = self.batch_size * (self.seq_len + 1);
        let mut line_buf = String::new();
        while self.buffer.len() < need {
            let reader = match &mut self.current_reader {
                Some(r) => r,
                None => return Ok(false),
            };
            line_buf.clear();
            let bytes_read = reader.read_line(&mut line_buf)?;
            if bytes_read == 0 {
                // EOF on current file — advance to next
                self.current_reader = None;
                self.current_file_index += 1;
                if self.current_file_index >= self.files.len() {
                    return Ok(false);
                }
                let f = File::open(&self.files[self.current_file_index])
                    .context("open next file")?;
                self.current_reader = Some(BufReader::new(f));
                continue;
            }
            let line = line_buf.trim();
            if line.is_empty() {
                continue;
            }
            let text = extract_text(line);
            let enc = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
            self.buffer.extend(enc.get_ids());
        }
        Ok(true)
    }

    /// Yield the next batch, or `None` if data is exhausted.
    pub fn next_batch(&mut self) -> AnyhowResult<Option<(Vec<u32>, Vec<u32>)>> {
        let need = self.batch_size * (self.seq_len + 1);
        if !self.fill_buffer()? && self.buffer.len() < need {
            return Ok(None);
        }
        if self.buffer.len() < need {
            return Ok(None);
        }
        let seq_len = self.seq_len;
        let batch_size = self.batch_size;
        let step = seq_len;
        let mut input_batch = Vec::with_capacity(batch_size * seq_len);
        let mut label_batch = Vec::with_capacity(batch_size * seq_len);
        for b in 0..batch_size {
            let base = b * step;
            for i in 0..seq_len {
                input_batch.push(self.buffer[base + i]);
                label_batch.push(self.buffer[base + i + 1]);
            }
        }
        self.buffer.drain(0..(batch_size * (seq_len + 1)));
        Ok(Some((input_batch, label_batch)))
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Convert a raw batch of `(input_ids, labels)` to Candle tensors.
pub fn batch_to_tensors(
    input_ids: &[u32],
    labels: &[u32],
    batch_size: usize,
    seq_len: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let input = Tensor::from_vec(input_ids.to_vec(), (batch_size, seq_len), device)?;
    let labels = Tensor::from_vec(labels.to_vec(), (batch_size, seq_len), device)?;
    Ok((input, labels))
}

/// Collect text/JSONL files from a path (file or directory), sorted.
fn collect_files(path: &Path) -> AnyhowResult<Vec<PathBuf>> {
    let mut out = Vec::new();
    if path.is_file() {
        out.push(path.to_path_buf());
    } else if path.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension()
                        .map(|e| e == "jsonl" || e == "json" || e == "txt" || e == "raw")
                        .unwrap_or(false)
            })
            .collect();
        entries.sort();
        out = entries;
    }
    Ok(out)
}

/// Extract text from a line: supports plain text, JSONL with `"text"`,
/// or JSONL with `"input"` + `"output"`.
fn extract_text(line: &str) -> String {
    if line.starts_with('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
                return t.to_string();
            }
            if let (Some(inp), Some(out)) = (
                v.get("input").and_then(|x| x.as_str()),
                v.get("output").and_then(|x| x.as_str()),
            ) {
                return format!("{inp}\n{out}");
            }
        }
    }
    line.to_string()
}
