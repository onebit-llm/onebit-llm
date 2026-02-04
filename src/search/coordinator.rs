//! Main search coordinator (combines all components)

use super::evaluator::ConfigEvaluator;
use super::expander::{ExpanderDecomposer, ExpanderParams, ExpanderPartition};
use super::graph::{GraphBuilder, QuantGraph};
use super::types::{QuantConfig, SearchConfig, SearchResult};
use crate::OneBitLlmConfig;
use anyhow::Result;
use candle::Device;
use indicatif::{ProgressBar, ProgressStyle};
use petgraph::algo::dijkstra;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

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

    /// Run complete search
    pub fn search(&self) -> Result<SearchResult> {
        let start_time = Instant::now();

        tracing::info!("Starting expander-based quantization search");

        // 1. Build search graph
        tracing::info!("Building search graph...");
        let model_config = OneBitLlmConfig::load(Path::new(&self.config.model_config))?;
        let params_per_layer = Self::estimate_params_per_layer(&model_config);

        let builder = GraphBuilder::new(model_config.num_layers, params_per_layer.clone());
        let (graph, start_node, goal_nodes) = builder.build();

        tracing::info!(
            "Graph built: {} nodes, {} potential goals",
            graph.node_count(),
            goal_nodes.len()
        );

        // 2. Expander decomposition
        tracing::info!("Performing expander decomposition...");
        let expander_params = ExpanderParams {
            target_partition_size: self.config.partition_size,
            overlap_ratio: self.config.overlap_ratio,
            ..Default::default()
        };
        let decomposer = ExpanderDecomposer::new(expander_params);
        let partitions = decomposer.decompose(&graph);

        tracing::info!("Decomposed into {} partitions", partitions.len());

        // 3. Parallel search in partitions
        tracing::info!("Searching in partitions (parallel)...");
        let pb = ProgressBar::new(partitions.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40} {pos}/{len} {msg}")
                .unwrap(),
        );

        let evaluator = Arc::clone(&self.evaluator);
        let partition_results: Vec<_> = partitions
            .par_iter()
            .map(|partition| {
                let result = Self::search_partition(&evaluator, &graph, partition, start_node);
                pb.inc(1);
                result
            })
            .collect();

        pb.finish_with_message("Partition search complete");

        // 4. Find global best
        tracing::info!("Finding global optimum...");
        let evaluations = partition_results
            .iter()
            .filter(|r| r.is_some())
            .count()
            .max(1);
        let best_config = Self::find_global_best(&partition_results)?;

        // 5. Final evaluation
        tracing::info!("Final evaluation of best config...");
        let (loss, perplexity) = self.evaluator.evaluate(&best_config)?;
        let accuracy = -loss; // Higher is better
        let size_mb = best_config.total_size_mb(&params_per_layer);
        let compression_ratio = best_config.compression_ratio(&params_per_layer);

        let search_time = start_time.elapsed().as_secs_f64();

        let result = SearchResult {
            config: best_config,
            accuracy,
            perplexity,
            size_mb,
            compression_ratio,
            search_time_secs: search_time,
            evaluations,
        };

        tracing::info!(
            "Search complete! Perplexity: {:.2}, Size: {:.1} MB, Compression: {:.1}x, Time: {:.1}s",
            result.perplexity,
            result.size_mb,
            result.compression_ratio,
            result.search_time_secs
        );

        Ok(result)
    }

    fn search_partition(
        evaluator: &ConfigEvaluator,
        graph: &QuantGraph,
        partition: &ExpanderPartition,
        start_node: NodeIndex,
    ) -> Option<(QuantConfig, f64)> {
        let partition_set: HashSet<NodeIndex> = partition.nodes.iter().copied().collect();

        let paths = dijkstra(graph, start_node, None, |e| {
            if partition_set.contains(&e.target()) {
                *e.weight()
            } else {
                f64::INFINITY
            }
        });

        let mut best_config = None;
        let mut best_score = f64::NEG_INFINITY;

        for &node in &partition.nodes {
            if let Some(&path_cost) = paths.get(&node) {
                let config = &graph[node];

                if let Ok((loss, _ppl)) = evaluator.evaluate(config) {
                    let score = -loss - 0.1 * path_cost; // Balance accuracy and search cost

                    if score > best_score {
                        best_score = score;
                        best_config = Some(config.clone());
                    }
                }
            }
        }

        best_config.map(|c| (c, best_score))
    }

    fn find_global_best(partition_results: &[Option<(QuantConfig, f64)>]) -> Result<QuantConfig> {
        let valid_results: Vec<_> = partition_results
            .iter()
            .filter_map(|r| r.as_ref())
            .cloned()
            .collect();

        if valid_results.is_empty() {
            anyhow::bail!("No valid configurations found");
        }

        let (best_config, best_score) = valid_results
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        tracing::info!("Global best score: {:.4}", best_score);

        Ok(best_config)
    }

    fn estimate_params_per_layer(config: &OneBitLlmConfig) -> Vec<usize> {
        let d = config.hidden_size;
        let ff = config.intermediate_size;

        let attn_params = 3 * d * d + d * d; // QKV + proj
        let ffn_params = d * ff + ff * d; // Two linear layers
        let layer_params = attn_params + ffn_params;

        vec![layer_params; config.num_layers]
    }
}
