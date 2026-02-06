//! Config evaluation using Candle model.

use super::types::{config_key, QuantConfig};
use candle_core::{DType, Device, Result as CandleResult};
use candle_nn::{VarBuilder, VarMap};
use std::path::Path;
use std::sync::Mutex;

use ternary_common::{batch_to_tensors, OneBitLlmConfig, TextDataset};
use ternary_core::OneBitLlm;

const MAX_EVAL_BATCHES: usize = 10;
const BATCH_SIZE: usize = 8;

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
    ) -> anyhow::Result<Self> {
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

    pub fn evaluate(&self, config: &QuantConfig) -> anyhow::Result<(f64, f64)> {
        let key = config_key(config);
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(result) = cache.get(&key) {
                return Ok(*result);
            }
        }
        let model = self.build_model_with_config(config)?;
        let (loss, ppl) = self.compute_validation_metrics(&model)?;
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(key, (loss, ppl));
        }
        Ok((loss, ppl))
    }

    fn build_model_with_config(&self, config: &QuantConfig) -> anyhow::Result<OneBitLlm> {
        let mut model_config = self.base_model_config.clone();
        // Sandwich Rule: use layer_bit_map with embedding/lm_head pinned F16.
        model_config.layer_bit_map = Some(config.to_layer_bit_map());

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
        Ok((avg_loss, avg_loss.exp()))
    }
}
