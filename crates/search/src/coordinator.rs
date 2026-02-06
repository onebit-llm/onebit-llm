//! Main search coordinator (combines all components).
//!
//! Supports both synchronous [`search`](Self::search) and tokio-based async
//! [`search_async`](Self::search_async). The async path runs the partition loop
//! in [`tokio::task::spawn_blocking`] so the runtime is not blocked by GPU work.

use super::evaluator::ConfigEvaluator;
use super::expander::{ExpanderDecomposer, ExpanderParams, ExpanderPartition};
use super::graph::GraphBuilder;
use super::types::{QuantConfig, SearchConfig, SearchResult};
use ternary_common::OneBitLlmConfig;

use anyhow::Result;
use candle_core::Device;
use indicatif::{ProgressBar, ProgressStyle};
use petgraph::algo::dijkstra;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const MAX_EVALS_PER_PARTITION: usize = 5;

pub struct SearchCoordinator {
    config: SearchConfig,
    evaluator: Arc<ConfigEvaluator>,
}

impl SearchCoordinator {
    pub fn new(config: SearchConfig) -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        let evaluator = ConfigEvaluator::new(
            &config.model_config,
            &config.checkpoint,
            &config.val_data,
            &config.tokenizer,
            device,
        )?;
        Ok(Self {
            config,
            evaluator: Arc::new(evaluator),
        })
    }

    pub fn search(&self) -> Result<SearchResult> {
        let start_time = Instant::now();
        tracing::info!("Starting expander-based quantisation search");

        let model_config = OneBitLlmConfig::load(Path::new(&self.config.model_config))?;
        let params_per_layer = Self::estimate_params_per_layer(&model_config);
        let builder = GraphBuilder::new(model_config.num_layers, params_per_layer.clone());
        let (graph, start_node, goal_nodes) = builder.build();

        tracing::info!(goals = goal_nodes.len(), "Graph built");

        let expander_params = ExpanderParams {
            target_partition_size: self.config.partition_size,
            overlap_ratio: self.config.overlap_ratio,
            ..Default::default()
        };
        let decomposer = ExpanderDecomposer::new(expander_params);
        let partitions = decomposer.decompose(&graph);

        tracing::info!(partitions = partitions.len(), "Searching partitions");

        let pb = ProgressBar::new(partitions.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40} {pos}/{len} partitions")
                .unwrap()
                .progress_chars("=>-"),
        );

        let min_perplexity_max = self.config.min_perplexity_max;
        let mut partition_results = Vec::new();
        for partition in &partitions {
            let result = Self::search_partition(
                self.evaluator.as_ref(),
                &graph,
                partition,
                start_node,
                min_perplexity_max,
            );
            partition_results.push(result);
            pb.inc(1);
        }
        pb.finish_with_message("done");

        let valid_results: Vec<_> = partition_results
            .iter()
            .filter_map(|r| r.as_ref())
            .cloned()
            .collect();
        let valid_partitions = valid_results.len();

        let best_config = if let Some((cfg, _)) = valid_results
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            cfg.clone()
        } else {
            tracing::warn!("No valid partition results, using start config");
            graph[start_node].clone()
        };

        let (loss, perplexity) = self.evaluator.evaluate(&best_config)?;
        let size_mb = best_config.total_size_mb(&params_per_layer);
        let compression_ratio = best_config.compression_ratio(&params_per_layer);

        let result = SearchResult {
            config: best_config.clone(),
            layer_bit_map: best_config.to_layer_bit_map(),
            accuracy: -loss,
            perplexity,
            size_mb,
            compression_ratio,
            search_time_secs: start_time.elapsed().as_secs_f64(),
            evaluations: valid_partitions.max(1),
            total_evaluations: partitions.len() * MAX_EVALS_PER_PARTITION,
            valid_partitions,
        };

        tracing::info!(
            ppl = format!("{:.2}", result.perplexity),
            size = format!("{:.1} MB", result.size_mb),
            compression = format!("{:.1}x", result.compression_ratio),
            "Search complete"
        );

        Ok(result)
    }

    /// Run the same search on the tokio runtime without blocking it.
    ///
    /// The partition loop (and thus GPU evaluation) runs inside
    /// `tokio::task::spawn_blocking`, so the async runtime remains responsive.
    /// Use this from async code (e.g. `#[tokio::main]`) to allow cancellation
    /// and composition with other async tasks.
    pub async fn search_async(&self) -> Result<SearchResult> {
        let config = self.config.clone();
        let evaluator = Arc::clone(&self.evaluator);
        tokio::task::spawn_blocking(move || {
            let coordinator = SearchCoordinator { config, evaluator };
            coordinator.search()
        })
        .await
        .map_err(|e| anyhow::anyhow!("search task join: {}", e))?
    }

    fn search_partition(
        evaluator: &ConfigEvaluator,
        graph: &super::graph::QuantGraph,
        partition: &ExpanderPartition,
        start_node: NodeIndex,
        min_perplexity_max: Option<f64>,
    ) -> Option<(QuantConfig, f64)> {
        let partition_set: HashSet<NodeIndex> = partition.nodes.iter().copied().collect();
        let paths = dijkstra(graph, start_node, None, |e| {
            if partition_set.contains(&e.target()) {
                *e.weight()
            } else {
                f64::INFINITY
            }
        });

        let mut best: Option<(QuantConfig, f64)> = None;
        let mut eval_count = 0;

        for &node in &partition.nodes {
            if eval_count >= MAX_EVALS_PER_PARTITION {
                break;
            }
            if let Some(&path_cost) = paths.get(&node) {
                let config = &graph[node];
                if let Ok((loss, ppl)) = evaluator.evaluate(config) {
                    eval_count += 1;
                    // Min-perplexity constraint: reject if above threshold.
                    if let Some(max_ppl) = min_perplexity_max {
                        if ppl > max_ppl {
                            continue;
                        }
                    }
                    let score = -loss - 0.1 * path_cost;
                    if score.is_finite() {
                        if best.as_ref().map_or(true, |(_, s)| score > *s) {
                            best = Some((config.clone(), score));
                        }
                    }
                }
            }
        }
        best
    }

    fn estimate_params_per_layer(config: &OneBitLlmConfig) -> Vec<usize> {
        let d = config.hidden_size;
        let ff = config.intermediate_size;
        let attn = 3 * d * d + d * d;
        let ffn = if config.use_swiglu {
            3 * d * ff
        } else {
            2 * d * ff
        };
        vec![attn + ffn; config.num_layers]
    }
}
