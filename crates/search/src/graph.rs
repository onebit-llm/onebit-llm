//! Quantisation search space as a graph.

use super::types::{QuantConfig, QuantLevel};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Directed;
use std::collections::{HashMap, VecDeque};

pub type QuantGraph = Graph<QuantConfig, f64, Directed>;

/// Build quantisation search graph via BFS.
pub struct GraphBuilder {
    num_layers: usize,
    #[allow(dead_code)]
    params_per_layer: Vec<usize>,
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

    pub fn with_max_nodes(mut self, max: usize) -> Self {
        self.max_nodes = Some(max);
        self
    }

    pub fn build(&self) -> (QuantGraph, NodeIndex, Vec<NodeIndex>) {
        let mut graph = Graph::new();
        let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

        let start_config = self.baseline_config();
        let start_node = graph.add_node(start_config.clone());
        node_map.insert(self.config_key(&start_config), start_node);

        let mut goals = Vec::new();
        self.generate_configs_bfs(&mut graph, &mut node_map, &mut goals, start_node);

        (graph, start_node, goals)
    }

    fn baseline_config(&self) -> QuantConfig {
        let mut config = QuantConfig::new(self.num_layers);
        for layer in 0..self.num_layers {
            config.set_layer(layer, QuantLevel::Ternary);
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

            for layer in 0..self.num_layers {
                let current_level = current_config.get_layer(layer);
                if let Some(next_level) = self.next_quant_level(current_level) {
                    let mut next_config = current_config.clone();
                    next_config.set_layer(layer, next_level);
                    let key = self.config_key(&next_config);

                    let next_node = *node_map.entry(key).or_insert_with(|| {
                        let node = graph.add_node(next_config.clone());
                        queue.push_back(node);
                        if self.is_goal_config(&next_config) {
                            goals.push(node);
                        }
                        node
                    });

                    let cost = self.estimate_transition_cost(current_level, next_level, layer);
                    graph.add_edge(current_node, next_node, cost);
                }
            }
        }

        tracing::info!(
            nodes = graph.node_count(),
            edges = graph.edge_count(),
            goals = goals.len(),
            "Search graph built"
        );
    }

    fn next_quant_level(&self, current: QuantLevel) -> Option<QuantLevel> {
        match current {
            QuantLevel::Ternary => Some(QuantLevel::Binary),
            QuantLevel::Binary => None,
        }
    }

    fn estimate_transition_cost(&self, from: QuantLevel, to: QuantLevel, layer: usize) -> f64 {
        let bit_reduction = from.bits() - to.bits();
        let sensitivity = 2.0 - layer as f64 / self.num_layers as f64;
        bit_reduction as f64 * sensitivity
    }

    fn is_goal_config(&self, config: &QuantConfig) -> bool {
        let highly_quant = (0..self.num_layers)
            .filter(|&i| matches!(config.get_layer(i), QuantLevel::Binary | QuantLevel::Ternary))
            .count();
        highly_quant >= self.num_layers * 2 / 3
    }

    fn config_key(&self, config: &QuantConfig) -> String {
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
    }
}
