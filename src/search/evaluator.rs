//! Config evaluation using Candle model

use super::types::{QuantConfig, QuantLevel};
use super::utils::config_key;
use crate::{batch_to_tensors, OneBitLlm, OneBitLlmConfig, TextDataset};
use anyhow::Result as AnyhowResult;
use candle::{DType, Device, Result as CandleResult};
use candle_nn::{VarBuilder, VarMap};
use std::path::Path;
use std::sync::Mutex;

const MAX_EVAL_BATCHES: usize = 50;
const BATCH_SIZE: usize = 8;

/// Config evaluator (uses Candle)
pub struct ConfigEvaluator {
    base_model_config: OneBitLlmConfig,
    checkpoint_path: String,
    val_dataset: TextDataset,
    device: Device,
    cache: Mutex<lru::LruCache<String, (f64, f64)>>,
}

impl ConfigEvaluator {
    pub fn new(
        config_path: &str,
        checkpoint_path: &str,
        val_data_path: &str,
        tokenizer_path: &str,
        device: Device,
    ) -> AnyhowResult<Self> {
        let base_config = OneBitLlmConfig::load(Path::new(config_path))?;

        let mut val_dataset = TextDataset::new(
            Path::new(val_data_path),
            Path::new(tokenizer_path),
            base_config.max_seq_len,
        )?;
        val_dataset.load()?;

        let cache = Mutex::new(lru::LruCache::new(
            std::num::NonZeroUsize::new(1000).unwrap(),
        ));

        Ok(Self {
            base_model_config: base_config,
            checkpoint_path: checkpoint_path.to_string(),
            val_dataset,
            device,
            cache,
        })
    }

    /// Evaluate a quantization config
    /// Returns: (validation_loss, perplexity)
    pub fn evaluate(&self, config: &QuantConfig) -> AnyhowResult<(f64, f64)> {
        let config_key = config_key(config);

        // Check cache
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(result) = cache.get(&config_key) {
                return Ok(*result);
            }
        }

        // Build model with this quantization config
        let model = self.build_model_with_config(config)?;

        // Evaluate on validation set
        let (loss, ppl) = self.compute_validation_metrics(&model)?;

        // Cache result
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(config_key, (loss, ppl));
        }

        Ok((loss, ppl))
    }

    fn build_model_with_config(&self, config: &QuantConfig) -> AnyhowResult<OneBitLlm> {
        let mut model_config = self.base_model_config.clone();

        let ternary_count = (0..config.num_layers)
            .filter(|&i| config.get_layer(i) == QuantLevel::Ternary)
            .count();

        model_config.use_ternary = ternary_count > config.num_layers / 2;

        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &self.device);
        let model = OneBitLlm::new(vb, &model_config)?;
        varmap.load(Path::new(&self.checkpoint_path))?;

        Ok(model)
    }

    fn compute_validation_metrics(&self, model: &OneBitLlm) -> CandleResult<(f64, f64)> {
        let seq_len = self.base_model_config.max_seq_len;
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        for (input_ids, labels) in self.val_dataset.batches(BATCH_SIZE).take(MAX_EVAL_BATCHES) {
            let (input_ids, labels) =
                batch_to_tensors(&input_ids, &labels, BATCH_SIZE, seq_len, &self.device)?;

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
        let perplexity = avg_loss.exp();

        Ok((avg_loss, perplexity))
    }
}
