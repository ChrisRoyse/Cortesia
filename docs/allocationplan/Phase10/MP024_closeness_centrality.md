# MP024: Closeness Centrality Algorithm

## Task Description
Implement closeness centrality to identify neurons that can quickly reach all other neurons in the network.

## Prerequisites
- MP011-MP014 completed (shortest path algorithms)
- MP023 completed (eigenvector centrality)
- Understanding of distance-based centrality

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/centrality/closeness.rs`

2. Implement basic closeness centrality:
   ```rust
   pub fn closeness_centrality<G: Graph>(
       graph: &G,
   ) -> HashMap<G::Node::Id, f32> {
       let mut centrality = HashMap::new();
       
       for node in graph.nodes() {
           let node_id = node.id();
           let distances = dijkstra_all_distances(graph, node_id.clone());
           
           let total_distance: f32 = distances.values()
               .filter(|&&d| d != f32::INFINITY)
               .sum();
           
           let reachable_nodes = distances.values()
               .filter(|&&d| d != f32::INFINITY && d > 0.0)
               .count() as f32;
           
           let closeness = if total_distance > 0.0 {
               reachable_nodes / total_distance
           } else {
               0.0
           };
           
           centrality.insert(node_id, closeness);
       }
       
       centrality
   }
   ```

3. Implement harmonic centrality (for disconnected graphs):
   ```rust
   pub fn harmonic_centrality<G: Graph>(
       graph: &G,
   ) -> HashMap<G::Node::Id, f32> {
       let mut centrality = HashMap::new();
       let n = graph.node_count() as f32;
       
       for node in graph.nodes() {
           let node_id = node.id();
           let distances = dijkstra_all_distances(graph, node_id.clone());
           
           let harmonic_sum: f32 = distances.values()
               .filter(|&&d| d != f32::INFINITY && d > 0.0)
               .map(|&d| 1.0 / d)
               .sum();
           
           let normalized_centrality = harmonic_sum / (n - 1.0);
           centrality.insert(node_id, normalized_centrality);
       }
       
       centrality
   }
   ```

4. Add weighted closeness centrality:
   ```rust
   pub fn weighted_closeness_centrality<G: Graph>(
       graph: &G,
       use_edge_weights: bool,
   ) -> HashMap<G::Node::Id, f32> {
       let mut centrality = HashMap::new();
       
       for node in graph.nodes() {
           let node_id = node.id();
           
           let distances = if use_edge_weights {
               dijkstra_all_distances(graph, node_id.clone())
           } else {
               bfs_all_distances(graph, node_id.clone())
           };
           
           let sum_inverse_distances: f32 = distances.iter()
               .filter(|(_, &d)| d != f32::INFINITY && d > 0.0)
               .map(|(_, &d)| 1.0 / d)
               .sum();
           
           centrality.insert(node_id, sum_inverse_distances);
       }
       
       centrality
   }
   ```

5. Implement neuromorphic-specific variants:
   ```rust
   pub fn neural_closeness_centrality<G: NeuromorphicGraph>(
       graph: &G,
   ) -> HashMap<G::Node::Id, f32> {
       // Consider spike propagation delays and refractory periods
       let mut centrality = HashMap::new();
       
       for node in graph.nodes() {
           let neural_distances = compute_neural_distances(graph, node.id());
           let effective_closeness = calculate_spike_efficiency(&neural_distances);
           centrality.insert(node.id(), effective_closeness);
       }
       
       centrality
   }
   ```

## Expected Output
```rust
pub struct ClosenessResult<Id> {
    pub centrality: HashMap<Id, f32>,
    pub harmonic_centrality: HashMap<Id, f32>,
    pub average_distance: HashMap<Id, f32>,
}

pub trait ClosenessCentrality: Graph {
    fn closeness_centrality(&self) -> HashMap<Self::Node::Id, f32>;
    fn harmonic_centrality(&self) -> HashMap<Self::Node::Id, f32>;
    fn weighted_closeness(&self, use_weights: bool) -> HashMap<Self::Node::Id, f32>;
}
```

## Verification Steps
1. Test on star graphs (center should have highest centrality)
2. Verify normalization for different graph sizes
3. Test harmonic centrality on disconnected components
4. Compare with manual calculations on small graphs

## Time Estimate
25 minutes

## Dependencies
- MP011-MP014: Shortest path algorithms for distance computation
- MP016: BFS for unweighted distances
- MP001-MP010: Graph infrastructure