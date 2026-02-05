//! Decoding strategies: greedy, top-k, top-p, temperature, repetition penalty.

use candle_core::{Result, Tensor};

/// Sampling configuration.
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
    pub repetition_window: usize,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.2,
            repetition_window: 64,
        }
    }
}

pub struct Sampler {
    config: SamplerConfig,
    recent_tokens: Vec<u32>,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            config,
            recent_tokens: Vec::new(),
        }
    }

    /// Sample one token from logits (1-D tensor of vocab_size).
    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let mut logits_vec: Vec<f32> = logits.to_vec1()?;

        // Apply repetition penalty
        if self.config.repetition_penalty != 1.0 {
            let window_start =
                self.recent_tokens.len().saturating_sub(self.config.repetition_window);
            for &tok in &self.recent_tokens[window_start..] {
                let idx = tok as usize;
                if idx < logits_vec.len() {
                    if logits_vec[idx] > 0.0 {
                        logits_vec[idx] /= self.config.repetition_penalty as f32;
                    } else {
                        logits_vec[idx] *= self.config.repetition_penalty as f32;
                    }
                }
            }
        }

        // Greedy if temperature ~0
        if self.config.temperature < 1e-6 {
            let token = argmax(&logits_vec);
            self.recent_tokens.push(token);
            return Ok(token);
        }

        // Temperature scaling
        let temp = self.config.temperature as f32;
        for v in &mut logits_vec {
            *v /= temp;
        }

        // Top-k filter
        if self.config.top_k > 0 && self.config.top_k < logits_vec.len() {
            let mut indexed: Vec<(usize, f32)> =
                logits_vec.iter().cloned().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let threshold = indexed[self.config.top_k].1;
            for v in &mut logits_vec {
                if *v < threshold {
                    *v = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax
        let max_val = logits_vec
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits_vec.iter().map(|&v| (v - max_val).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }

        // Top-p (nucleus) filter
        if self.config.top_p < 1.0 {
            let mut sorted: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut cumsum = 0.0;
            let mut keep = std::collections::HashSet::new();
            for (idx, p) in &sorted {
                keep.insert(*idx);
                cumsum += p;
                if cumsum >= self.config.top_p as f32 {
                    break;
                }
            }
            for (i, p) in probs.iter_mut().enumerate() {
                if !keep.contains(&i) {
                    *p = 0.0;
                }
            }
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 {
                for p in &mut probs {
                    *p /= sum;
                }
            }
        }

        // Weighted random sample
        let token = weighted_sample(&probs);
        self.recent_tokens.push(token);
        Ok(token)
    }

    pub fn reset(&mut self) {
        self.recent_tokens.clear();
    }
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn weighted_sample(probs: &[f32]) -> u32 {
    let r: f32 = rand::random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn greedy_sampling() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            ..Default::default()
        });
        let logits = Tensor::new(&[0.1f32, 0.9, 0.3, 0.5], &Device::Cpu).unwrap();
        let token = sampler.sample(&logits).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn repetition_penalty_reduces_repeat() {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            repetition_penalty: 100.0,
            ..Default::default()
        });
        // Force token 1 into history
        sampler.recent_tokens.push(1);
        let logits = Tensor::new(&[0.1f32, 0.9, 0.8, 0.5], &Device::Cpu).unwrap();
        let token = sampler.sample(&logits).unwrap();
        // Token 1 should be heavily penalised, so token 2 wins
        assert_eq!(token, 2);
    }
}
