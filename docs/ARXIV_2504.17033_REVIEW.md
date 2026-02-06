# Review: arXiv:2504.17033 — Breaking the Sorting Barrier for Directed SSSP

**Paper:** [Breaking the Sorting Barrier for Directed Single-Source Shortest Paths](https://arxiv.org/abs/2504.17033) (Duan, Mao, Mao, Shu, Yin), v2 (2025).

## Summary

The paper gives a **deterministic** \(O(m \log^{2/3} n)\)-time algorithm for **single-source shortest paths (SSSP)** on directed graphs with real non-negative edge weights (comparison-addition model). This is the first result to break the \(O(m + n \log n)\) bound of Dijkstra’s algorithm on sparse graphs.

**Main idea:** Merge Dijkstra and Bellman–Ford via a **recursive partitioning** technique (related to bottleneck path algorithms). The bottleneck in Dijkstra is maintaining a large “frontier” and repeatedly extracting the minimum (sorting). They reduce the effective frontier size by:

- Classifying vertices as “complete” vs “incomplete” and tracking dependence on a pivot set \(S\).
- A **FindPivots** procedure that returns a small pivot set \(P \subseteq S\) and a set \(W\) of vertices.
- Running **Bellman–Ford-style** steps from pivots to complete vertices in a distance band without full sorting.
- A recursive **BMSSP** (bounded multiple-source shortest path) subroutine and a custom data structure \(\mathcal{D}\) for batched extraction and updates.
- Parameters \(k\) and recursion level \(\ell\); complexity comes from the analysis of the recursion and the data structure.

So the optimization is **algorithmic**: a strictly faster SSSP than Dijkstra when only distances (not the full distance ordering) are required.

## Relevance to OneBit-LLM

In **ternary-search** we use SSSP as follows:

- **QuantGraph:** nodes = quantisation configs, edges = transitions with real costs.
- **Per partition:** we run **Dijkstra** from the start node restricted to the partition (via infinite weight on edges leaving the partition) to get shortest-path distances to each node in the partition.
- We then **evaluate** a limited number of configs (e.g. top by path cost) with the GPU evaluator.

So we do **one Dijkstra (or equivalent SSSP) per partition** and only need **distances to partition nodes**, not the full vertex ordering.

### Can we use this optimization?

**Theoretically yes:**

- The paper’s algorithm is for directed graphs with non-negative weights, which matches our graph.
- We only need distances to a subset of nodes (partition nodes), not the full ordering; the paper explicitly targets the setting where the full order is not required.
- Replacing Dijkstra with an \(O(m \log^{2/3} n)\) SSSP in `search_partition` would reduce the cost of the shortest-path step when the graph is large.

**Practically:**

1. **Current bottleneck:** In our setup, cost is dominated by **ConfigEvaluator::evaluate()** (model load + GPU forward), not by Dijkstra. So replacing Dijkstra alone may yield only a small end-to-end speedup unless the graph is very large and we run many partitions.
2. **Implementation cost:** The algorithm is complex: FindPivots, BMSSP recursion, and the data structure \(\mathcal{D}\) are non-trivial. The paper does not provide code; we would be implementing from the description.
3. **When it becomes attractive:** If we scale the quantisation search (e.g. much larger config graphs or many more partitions), the SSSP step could start to matter. Then implementing or reusing an implementation of this algorithm (or a follow-up) would be a natural optimization.

## Recommendation

- **Short term:** Keep using Dijkstra in the coordinator. No code change required.
- **Documentation:** Reference this paper where we describe the search (e.g. coordinator or MASTER_PLAN) as a possible future optimization if the graph grows.
- **Future work:** If we ever optimize the “graph” part of the search (e.g. larger \(n\)/\(m\) or more partitions), consider:
  - Implementing the algorithm of arXiv:2504.17033 (or a simpler variant), or
  - Using a library that provides faster directed SSSP when available.

## References

- arXiv:2504.17033 [cs.DS]: *Breaking the Sorting Barrier for Directed Single-Source Shortest Paths* (HTML: [2504.17033v2](https://arxiv.org/html/2504.17033v2)).
