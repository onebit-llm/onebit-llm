//! Expander graph decomposition algorithm
//!
//! Based on: "Deterministic Distributed Expander Decomposition" (STOC 2024)
//! Simplified version for quantization search

use super::graph::QuantGraph;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use rand::seq::SliceRandom;
use std::collections::{HashSet, VecDeque};

/// Expander partition
#[derive(Debug, Clone)]
pub struct ExpanderPartition {
    pub nodes: Vec<NodeIndex>,
    pub internal_edges: usize,
    pub boundary_edges: usize,
    pub expansion: f64, // Internal edges / nodes (quality metric)
}

/// Expander decomposition parameters
#[derive(Debug, Clone)]
pub struct ExpanderParams {
    pub target_partition_size: usize, // √V typically
    pub min_expansion: f64,           // Minimum expansion ratio (e.g., 2.0)
    pub overlap_ratio: f64,           // Overlap between partitions (0.1-0.2)
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

/// Expander decomposition algorithm
pub struct ExpanderDecomposer {
    params: ExpanderParams,
}

impl ExpanderDecomposer {
    pub fn new(params: ExpanderParams) -> Self {
        Self { params }
    }

    /// Decompose graph into expander partitions
    ///
    /// Algorithm (simplified):
    /// 1. Start with random seed nodes
    /// 2. Grow partitions via BFS until target size
    /// 3. Verify expander property (high internal connectivity)
    /// 4. Add overlap for smooth transitions
    pub fn decompose(&self, graph: &QuantGraph) -> Vec<ExpanderPartition> {
        let num_nodes = graph.node_count();

        // Adjust target partition size based on graph size
        let target_size = self.params.target_partition_size.min(num_nodes / 2).max(10);

        tracing::info!(
            "Starting expander decomposition: {} nodes, target partition size: {}",
            num_nodes,
            target_size
        );

        let mut partitions = Vec::new();
        let mut covered_nodes = HashSet::new();
        let mut seed_candidates: Vec<_> = graph.node_indices().collect();

        // Shuffle seeds for better coverage
        seed_candidates.shuffle(&mut rand::thread_rng());

        for seed in seed_candidates {
            // Skip if already well-covered
            if covered_nodes.contains(&seed) {
                continue;
            }

            // Grow partition from seed
            if let Some(partition) = self.grow_partition(graph, seed, target_size, &covered_nodes) {
                // Verify expander property
                if partition.expansion >= self.params.min_expansion {
                    // Mark nodes as covered (with some overlap allowed)
                    for &node in &partition.nodes {
                        if rand::random::<f64>() > self.params.overlap_ratio {
                            covered_nodes.insert(node);
                        }
                    }

                    partitions.push(partition);

                    // Stop if we've covered enough
                    if covered_nodes.len() >= num_nodes * 95 / 100 {
                        break;
                    }
                }
            }
        }

        tracing::info!(
            "Decomposition complete: {} partitions, coverage: {:.1}%",
            partitions.len(),
            covered_nodes.len() as f64 / num_nodes as f64 * 100.0
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
        let mut partition_nodes = HashSet::new();
        let mut queue = VecDeque::new();

        partition_nodes.insert(seed);
        queue.push_back(seed);

        // BFS growth
        while let Some(node) = queue.pop_front() {
            if partition_nodes.len() >= target_size {
                break;
            }

            // Explore neighbors
            for edge in graph.edges(node) {
                let neighbor = edge.target();

                // Skip if already in partition or should avoid
                if partition_nodes.contains(&neighbor) || avoid.contains(&neighbor) {
                    continue;
                }

                // Add to partition
                partition_nodes.insert(neighbor);
                queue.push_back(neighbor);

                if partition_nodes.len() >= target_size {
                    break;
                }
            }
        }

        // Check minimum size
        if partition_nodes.len() < 5 {
            return None;
        }

        // Calculate metrics
        let nodes: Vec<_> = partition_nodes.iter().copied().collect();
        let (internal_edges, boundary_edges) = self.count_edges(graph, &partition_nodes);
        let expansion = internal_edges as f64 / nodes.len() as f64;

        Some(ExpanderPartition {
            nodes,
            internal_edges,
            boundary_edges,
            expansion,
        })
    }

    fn count_edges(&self, graph: &QuantGraph, partition: &HashSet<NodeIndex>) -> (usize, usize) {
        let mut internal = 0;
        let mut boundary = 0;

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
    use crate::search::graph::GraphBuilder;

    #[test]
    fn test_expander_decomposition() {
        let builder = GraphBuilder::new(6, vec![1000; 6]);
        let (graph, _, _) = builder.build();

        let decomposer = ExpanderDecomposer::new(ExpanderParams::default());
        let partitions = decomposer.decompose(&graph);

        assert!(!partitions.is_empty());
        for (i, partition) in partitions.iter().enumerate() {
            println!(
                "Partition {}: {} nodes, expansion: {:.2}",
                i,
                partition.nodes.len(),
                partition.expansion
            );
        }
    }
}
