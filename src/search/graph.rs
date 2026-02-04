//! Quantization search space as a graph

use super::types::{QuantConfig, QuantLevel};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Directed;
use std::collections::{HashMap, VecDeque};

pub type QuantGraph = Graph<QuantConfig, f64, Directed>;

/// Build quantization search graph
pub struct GraphBuilder {
    num_layers: usize,
    #[allow(dead_code)] // reserved for size-based goals / edge costs
    params_per_layer: Vec<usize>,
    /// Optional cap to avoid graph explosion for large num_layers
    pub max_nodes: Option<usize>,
}

impl GraphBuilder {
    pub fn new(num_layers: usize, params_per_layer: Vec<usize>) -> Self {
        Self {
            num_layers,
            params_per_layer,
            max_nodes: None,
        }
    }

    /// Set optional max nodes cap (for large num_layers).
    pub fn with_max_nodes(mut self, max: usize) -> Self {
        self.max_nodes = Some(max);
        self
    }

    /// Build complete search graph
    ///
    /// Graph structure:
    /// - Node: QuantConfig (specific layer-wise quantization)
    /// - Edge: Transition cost (accuracy loss estimate)
    /// - Start: All FP16 (baseline)
    /// - Goal: Size constraint satisfied
    pub fn build(&self) -> (QuantGraph, NodeIndex, Vec<NodeIndex>) {
        let mut graph = Graph::new();
        let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

        // Start node: Baseline (all FP16)
        let start_config = self.baseline_config();
        let start_node = graph.add_node(start_config.clone());
        node_map.insert(self.config_key(&start_config), start_node);

        // Generate all reachable configs (layer-wise transitions)
        let mut goals = Vec::new();
        self.generate_configs_bfs(&mut graph, &mut node_map, &mut goals, start_node);

        (graph, start_node, goals)
    }

    fn baseline_config(&self) -> QuantConfig {
        let mut config = QuantConfig::new(self.num_layers);
        for layer in 0..self.num_layers {
            config.set_layer(layer, QuantLevel::Float16);
        }
        config
    }

    fn generate_configs_bfs(
        &self,
        graph: &mut QuantGraph,
        node_map: &mut HashMap<String, NodeIndex>,
        goals: &mut Vec<NodeIndex>,
        start: NodeIndex,
    ) {
        let mut queue = VecDeque::new();
        queue.push_back(start);

        while let Some(current_node) = queue.pop_front() {
            if let Some(max) = self.max_nodes {
                if graph.node_count() >= max {
                    break;
                }
            }

            let current_config = graph[current_node].clone();

            // Try quantizing each layer one step lower
            for layer in 0..self.num_layers {
                let current_level = current_config.get_layer(layer);

                // Get next quantization level (lower bits)
                if let Some(next_level) = self.next_quant_level(current_level) {
                    let mut next_config = current_config.clone();
                    next_config.set_layer(layer, next_level);

                    let config_key = self.config_key(&next_config);

                    // Add node if not exists
                    let next_node = *node_map.entry(config_key.clone()).or_insert_with(|| {
                        let node = graph.add_node(next_config.clone());
                        queue.push_back(node);

                        // Check if this is a goal (highly quantized)
                        if self.is_goal_config(&next_config) {
                            goals.push(node);
                        }

                        node
                    });

                    // Add edge: cost = estimated accuracy loss
                    let cost = self.estimate_transition_cost(current_level, next_level, layer);
                    graph.add_edge(current_node, next_node, cost);
                }
            }
        }

        tracing::info!(
            "Built graph: {} nodes, {} edges, {} goals",
            graph.node_count(),
            graph.edge_count(),
            goals.len()
        );
    }

    fn next_quant_level(&self, current: QuantLevel) -> Option<QuantLevel> {
        match current {
            QuantLevel::Float16 => Some(QuantLevel::EightBit),
            QuantLevel::EightBit => Some(QuantLevel::FourBit),
            QuantLevel::FourBit => Some(QuantLevel::Ternary),
            QuantLevel::Ternary => Some(QuantLevel::Binary),
            QuantLevel::Binary => None, // Can't go lower
        }
    }

    fn estimate_transition_cost(&self, from: QuantLevel, to: QuantLevel, layer: usize) -> f64 {
        // Heuristic: accuracy loss ~ bit reduction × layer sensitivity
        let bit_reduction = from.bits() - to.bits();
        let layer_sensitivity = self.layer_sensitivity(layer);
        bit_reduction as f64 * layer_sensitivity
    }

    fn layer_sensitivity(&self, layer: usize) -> f64 {
        // Heuristic: earlier layers more sensitive
        // Layer 0: 2.0, Layer N: 1.0
        let normalized_depth = layer as f64 / self.num_layers as f64;
        2.0 - normalized_depth
    }

    fn is_goal_config(&self, config: &QuantConfig) -> bool {
        // Goal: majority of layers quantized to ternary or below
        let highly_quant_count = (0..self.num_layers)
            .filter(|&i| {
                matches!(
                    config.get_layer(i),
                    QuantLevel::Binary | QuantLevel::Ternary
                )
            })
            .count();

        highly_quant_count >= self.num_layers * 2 / 3
    }

    fn config_key(&self, config: &QuantConfig) -> String {
        // Unique key: concatenate layer quantization levels
        (0..self.num_layers)
            .map(|i| format!("{:?}", config.get_layer(i)))
            .collect::<Vec<_>>()
            .join("-")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_build() {
        let builder = GraphBuilder::new(4, vec![1000, 2000, 3000, 4000]);
        let (graph, _start, goals) = builder.build();

        assert!(graph.node_count() > 0);
        assert!(!goals.is_empty());
        println!("Graph: {} nodes, {} goals", graph.node_count(), goals.len());
    }
}
