//! Data pipeline: text loading, tokenisation, batching.
//!
//! Supports text files or JSONL (one text field per line). Tokeniser loaded
//! from `tokenizer.json` (e.g. GPT-2 BPE). Batches are `(batch_size, seq_len)`
//! token IDs; labels for next-token prediction are `input_ids` shifted by one.
//!
//! * **[`TextDataset`]** — load and tokenise text into memory; call [`TextDataset::batches`].
//! * **[`StreamingBatchIter`]** — stream over files without loading all into RAM.
//! * **[`MmapDataset`]** — zero-copy access to a pre-tokenised binary file via `memmap2`.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result as AnyhowResult};
use candle_core::{Device, Result, Tensor};
use memmap2::Mmap;
use tokenizers::Tokenizer;

// ── Tokenized binary format ──────────────────────────────────────────────────

/// Magic bytes for the tokenized binary format (version 2).
const TOKENIZED_MAGIC: &[u8; 4] = b"TKN2";
/// Header size: magic (4) + num_tokens (8).
const TOKENIZED_HEADER_LEN: usize = 4 + 8;

/// Write a pre-tokenised sequence to a binary file for use with [`MmapDataset`].
///
/// Format: magic "TKN2" (4 bytes), `num_tokens` as u64 LE (8 bytes), then
/// `num_tokens` × u32 LE. No other metadata; use the same `seq_len` when opening.
pub fn write_tokenized_file(path: &Path, token_ids: &[u32]) -> AnyhowResult<()> {
    let mut f = File::create(path).context("create tokenized file")?;
    f.write_all(TOKENIZED_MAGIC)?;
    f.write_all(&(token_ids.len() as u64).to_le_bytes())?;
    for &id in token_ids {
        f.write_all(&id.to_le_bytes())?;
    }
    f.sync_all().context("sync tokenized file")?;
    Ok(())
}

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

    /// Write tokenised data to a binary file for use with [`MmapDataset`].
    pub fn write_tokenized(&self, path: &Path) -> AnyhowResult<()> {
        write_tokenized_file(path, &self.token_ids)
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
                let f =
                    File::open(&self.files[self.current_file_index]).context("open next file")?;
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

// ── MmapDataset (zero-copy) ──────────────────────────────────────────────────

/// Zero-copy dataset over a pre-tokenised binary file.
///
/// The file is memory-mapped; only the pages touched for each batch are paged in.
/// Use [`write_tokenized_file`] or [`TextDataset::write_tokenized`] to create the file.
pub struct MmapDataset {
    mmap: Mmap,
    num_tokens: usize,
    seq_len: usize,
}

impl MmapDataset {
    /// Open a tokenized binary file. `seq_len` must match the sequence length used in training.
    pub fn open(path: &Path, seq_len: usize) -> AnyhowResult<Self> {
        let file = File::open(path).context("open tokenized file for mmap")?;
        let mmap = unsafe { Mmap::map(&file).context("mmap tokenized file")? };
        if mmap.len() < TOKENIZED_HEADER_LEN {
            anyhow::bail!("tokenized file too short");
        }
        if &mmap[0..4] != TOKENIZED_MAGIC {
            anyhow::bail!("invalid tokenized file: bad magic");
        }
        let num_tokens = u64::from_le_bytes(mmap[4..12].try_into().unwrap()) as usize;
        let expected_len = TOKENIZED_HEADER_LEN + num_tokens * 4;
        if mmap.len() < expected_len {
            anyhow::bail!(
                "tokenized file truncated: expected {} bytes, got {}",
                expected_len,
                mmap.len()
            );
        }
        Ok(Self {
            mmap,
            num_tokens,
            seq_len,
        })
    }

    pub fn num_tokens(&self) -> usize {
        self.num_tokens
    }

    pub fn num_sequences(&self) -> usize {
        if self.seq_len == 0 {
            0
        } else {
            self.num_tokens.saturating_sub(1) / self.seq_len
        }
    }

    /// Read one u32 at byte offset (after header).
    #[inline]
    fn read_u32_at(&self, byte_offset: usize) -> u32 {
        let i = TOKENIZED_HEADER_LEN + byte_offset;
        u32::from_le_bytes(self.mmap[i..i + 4].try_into().unwrap())
    }

    /// Yield `(input_ids, labels)` batches by reading from the mmap. Labels are shifted by 1.
    pub fn batches(
        &self,
        batch_size: usize,
    ) -> impl Iterator<Item = (Vec<u32>, Vec<u32>)> + '_ {
        let seq_len = self.seq_len;
        let step = seq_len;
        let total = self.num_tokens;
        let mut start = 0usize;
        let dataset = self;
        std::iter::from_fn(move || {
            if start + batch_size * seq_len + 1 > total {
                return None;
            }
            let mut input_batch = Vec::with_capacity(batch_size * seq_len);
            let mut label_batch = Vec::with_capacity(batch_size * seq_len);
            for b in 0..batch_size {
                let base = start + b * step;
                for i in 0..seq_len {
                    let input_byte = (base + i) * 4;
                    let label_byte = (base + i + 1) * 4;
                    input_batch.push(dataset.read_u32_at(input_byte));
                    label_batch.push(dataset.read_u32_at(label_byte));
                }
            }
            start += batch_size * step;
            Some((input_batch, label_batch))
        })
    }
}

// ── BatchDataset trait ──────────────────────────────────────────────────────

/// Common interface for datasets that yield (input_ids, labels) batches.
pub trait BatchDataset {
    fn num_tokens(&self) -> usize;
    fn num_sequences(&self) -> usize;
    fn batches(
        &self,
        batch_size: usize,
    ) -> Box<dyn Iterator<Item = (Vec<u32>, Vec<u32>)> + '_>;
}

impl BatchDataset for TextDataset {
    fn num_tokens(&self) -> usize {
        self.num_tokens()
    }
    fn num_sequences(&self) -> usize {
        self.num_sequences()
    }
    fn batches(
        &self,
        batch_size: usize,
    ) -> Box<dyn Iterator<Item = (Vec<u32>, Vec<u32>)> + '_> {
        Box::new(self.batches(batch_size))
    }
}

impl BatchDataset for MmapDataset {
    fn num_tokens(&self) -> usize {
        self.num_tokens()
    }
    fn num_sequences(&self) -> usize {
        self.num_sequences()
    }
    fn batches(
        &self,
        batch_size: usize,
    ) -> Box<dyn Iterator<Item = (Vec<u32>, Vec<u32>)> + '_> {
        Box::new(self.batches(batch_size))
    }
}

/// Either a [`TextDataset`] or an [`MmapDataset`], both implementing [`BatchDataset`].
pub enum AnyBatchDataset {
    Text(TextDataset),
    Mmap(MmapDataset),
}

impl BatchDataset for AnyBatchDataset {
    fn num_tokens(&self) -> usize {
        match self {
            AnyBatchDataset::Text(d) => d.num_tokens(),
            AnyBatchDataset::Mmap(d) => d.num_tokens(),
        }
    }
    fn num_sequences(&self) -> usize {
        match self {
            AnyBatchDataset::Text(d) => d.num_sequences(),
            AnyBatchDataset::Mmap(d) => d.num_sequences(),
        }
    }
    fn batches(
        &self,
        batch_size: usize,
    ) -> Box<dyn Iterator<Item = (Vec<u32>, Vec<u32>)> + '_> {
        match self {
            AnyBatchDataset::Text(d) => Box::new(d.batches(batch_size)),
            AnyBatchDataset::Mmap(d) => Box::new(d.batches(batch_size)),
        }
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
