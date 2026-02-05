//! Expander graph decomposition (simplified STOC 2024).

use super::graph::QuantGraph;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use rand::seq::SliceRandom;
use std::collections::{HashSet, VecDeque};

#[derive(Debug, Clone)]
pub struct ExpanderPartition {
    pub nodes: Vec<NodeIndex>,
    pub internal_edges: usize,
    pub boundary_edges: usize,
    pub expansion: f64,
}

#[derive(Debug, Clone)]
pub struct ExpanderParams {
    pub target_partition_size: usize,
    pub min_expansion: f64,
    pub overlap_ratio: f64,
}

impl Default for ExpanderParams {
    fn default() -> Self {
        Self {
            target_partition_size: 100,
            min_expansion: 2.0,
            overlap_ratio: 0.15,
        }
    }
}

pub struct ExpanderDecomposer {
    params: ExpanderParams,
}

impl ExpanderDecomposer {
    pub fn new(params: ExpanderParams) -> Self {
        Self { params }
    }

    pub fn decompose(&self, graph: &QuantGraph) -> Vec<ExpanderPartition> {
        let num_nodes = graph.node_count();
        let target_size = self.params.target_partition_size.min(num_nodes / 2).max(10);

        let mut partitions = Vec::new();
        let mut covered = HashSet::new();
        let mut seeds: Vec<_> = graph.node_indices().collect();
        seeds.shuffle(&mut rand::thread_rng());

        for seed in seeds {
            if covered.contains(&seed) {
                continue;
            }
            if let Some(partition) = self.grow_partition(graph, seed, target_size, &covered) {
                if partition.expansion >= self.params.min_expansion {
                    for &node in &partition.nodes {
                        if rand::random::<f64>() > self.params.overlap_ratio {
                            covered.insert(node);
                        }
                    }
                    partitions.push(partition);
                    if covered.len() >= num_nodes * 95 / 100 {
                        break;
                    }
                }
            }
        }

        tracing::info!(
            partitions = partitions.len(),
            coverage = format!("{:.1}%", covered.len() as f64 / num_nodes as f64 * 100.0),
            "Expander decomposition complete"
        );
        partitions
    }

    fn grow_partition(
        &self,
        graph: &QuantGraph,
        seed: NodeIndex,
        target_size: usize,
        avoid: &HashSet<NodeIndex>,
    ) -> Option<ExpanderPartition> {
        let mut nodes_set = HashSet::new();
        let mut queue = VecDeque::new();
        nodes_set.insert(seed);
        queue.push_back(seed);

        while let Some(node) = queue.pop_front() {
            if nodes_set.len() >= target_size {
                break;
            }
            for edge in graph.edges(node) {
                let nb = edge.target();
                if nodes_set.contains(&nb) || avoid.contains(&nb) {
                    continue;
                }
                nodes_set.insert(nb);
                queue.push_back(nb);
                if nodes_set.len() >= target_size {
                    break;
                }
            }
        }

        if nodes_set.len() < 5 {
            return None;
        }

        let nodes: Vec<_> = nodes_set.iter().copied().collect();
        let (internal, boundary) = self.count_edges(graph, &nodes_set);
        let expansion = internal as f64 / nodes.len() as f64;

        Some(ExpanderPartition {
            nodes,
            internal_edges: internal,
            boundary_edges: boundary,
            expansion,
        })
    }

    fn count_edges(&self, graph: &QuantGraph, partition: &HashSet<NodeIndex>) -> (usize, usize) {
        let (mut internal, mut boundary) = (0, 0);
        for &node in partition {
            for edge in graph.edges(node) {
                if partition.contains(&edge.target()) {
                    internal += 1;
                } else {
                    boundary += 1;
                }
            }
        }
        (internal, boundary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphBuilder;

    #[test]
    fn test_expander_decomposition() {
        let builder = GraphBuilder::new(6, vec![1000; 6]);
        let (graph, _, _) = builder.build();
        let decomposer = ExpanderDecomposer::new(ExpanderParams::default());
        let partitions = decomposer.decompose(&graph);
        assert!(!partitions.is_empty());
    }
}
