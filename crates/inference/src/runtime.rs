//! Inference runtime: load model, generate tokens.
//!
//! Uses a KV cache for O(1) per-token decoding after an initial prefill pass.

use std::path::Path;

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, VarMap};

use ternary_common::OneBitLlmConfig;
use ternary_core::{LayerKVCache, OneBitLlm};

use crate::sampler::{Sampler, SamplerConfig};

/// High-level inference runtime.
pub struct InferenceRuntime {
    model: OneBitLlm,
    #[allow(dead_code)]
    varmap: VarMap,
    config: OneBitLlmConfig,
    tokenizer: tokenizers::Tokenizer,
    sampler: Sampler,
    device: Device,
    /// KV cache for incremental decoding; one entry per layer.
    kvcache: Vec<LayerKVCache>,
}

impl InferenceRuntime {
    pub fn load(
        model_dir: &Path,
        sampler_config: SamplerConfig,
        device: Device,
    ) -> anyhow::Result<Self> {
        let config = OneBitLlmConfig::load(&model_dir.join("config.json"))?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_path.to_string_lossy().to_string())
                .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = OneBitLlm::new(vb, &config)?;

        // Load weights
        let safetensors = model_dir.join("model.safetensors");
        varmap.load(&safetensors)?;

        // Pre-compute quantised weights for fast inference
        model.cache_quantized_weights()?;

        let sampler = Sampler::new(sampler_config);

        Ok(Self {
            model,
            varmap,
            config,
            tokenizer,
            sampler,
            device,
            kvcache: Vec::new(),
        })
    }

    /// Generate text from a prompt using KV cache for O(1) per-token decoding.
    ///
    /// Prefill: run the prompt (up to `max_seq_len` tokens) once and fill the KV cache.
    /// Decode: each new token is run with seq_len 1, reusing the cache.
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> anyhow::Result<String> {
        self.sampler.reset();
        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
        let mut tokens: Vec<u32> = enc.get_ids().to_vec();

        // One KV cache per layer; cleared and resized at start of each generate
        self.kvcache.clear();
        self.kvcache
            .resize_with(self.config.num_layers, LayerKVCache::default);

        // Prefill: process full prompt (or up to max_seq_len)
        let prefill_len = tokens.len().min(self.config.max_seq_len);
        let input_ids = &tokens[..prefill_len];
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?; // (1, prefill_len)
        let logits = self.model.forward_with_cache(
            &input,
            None,
            Some(self.kvcache.as_mut_slice()),
        )?;
        let (_, t, _) = logits.dims3()?;
        let mut next_token = self.sampler.sample(&logits.i((0, t - 1))?)?;

        if let Some(eos) = self.tokenizer.token_to_id("<|endoftext|>") {
            if next_token == eos {
                let output = self
                    .tokenizer
                    .decode(&tokens, true)
                    .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
                return Ok(output);
            }
        }
        tokens.push(next_token);

        // Decode: one token at a time with KV cache
        for _ in 1..max_tokens {
            // Single token: shape (1, 1) for (batch=1, seq_len=1)
            let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward_with_cache(
                &next_input,
                None,
                Some(self.kvcache.as_mut_slice()),
            )?;
            let (_, logit_t, _) = logits.dims3()?;
            next_token = self.sampler.sample(&logits.i((0, logit_t - 1))?)?;

            if let Some(eos) = self.tokenizer.token_to_id("<|endoftext|>") {
                if next_token == eos {
                    break;
                }
            }
            tokens.push(next_token);
        }

        let output = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        Ok(output)
    }

    /// Interactive chat loop (returns on empty input).
    pub fn chat_loop(&mut self, max_tokens: usize) -> anyhow::Result<()> {
        use std::io::{self, Write};
        loop {
            print!("You: ");
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();
            if input.is_empty() || input == "quit" || input == "exit" {
                break;
            }
            let response = self.generate(input, max_tokens)?;
            println!("AI: {response}");
        }
        Ok(())
    }
}
