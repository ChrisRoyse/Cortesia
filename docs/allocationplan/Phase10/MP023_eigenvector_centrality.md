# MP023: Eigenvector Centrality Algorithm

## Task Description
Implement eigenvector centrality using power iteration method to identify influential neurons based on connection quality.

## Prerequisites
- MP021-MP022 completed (PageRank, Betweenness)
- Understanding of eigenvectors
- Basic linear algebra

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/centrality/eigenvector.rs`

2. Implement power iteration method:
   ```rust
   pub fn eigenvector_centrality<G: Graph>(
       graph: &G,
       max_iterations: usize,
       tolerance: f32,
   ) -> HashMap<G::Node::Id, f32> {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let n = nodes.len();
       
       // Initialize with random values
       let mut x: Vec<f32> = vec![1.0 / n as f32; n];
       let mut x_new = vec![0.0; n];
       
       for _ in 0..max_iterations {
           // Matrix-vector multiplication: x_new = A * x
           for (i, node_i) in nodes.iter().enumerate() {
               x_new[i] = 0.0;
               
               if let Some(node) = graph.get_node(node_i) {
                   for neighbor in node.neighbors() {
                       if let Some(j) = nodes.iter().position(|id| id == &neighbor) {
                           let weight = graph.edge_weight(node_i, &neighbor).unwrap_or(1.0);
                           x_new[i] += weight * x[j];
                       }
                   }
               }
           }
           
           // Normalize
           let norm: f32 = x_new.iter().map(|&v| v * v).sum::<f32>().sqrt();
           if norm > 0.0 {
               for v in &mut x_new {
                   *v /= norm;
               }
           }
           
           // Check convergence
           let diff: f32 = x.iter().zip(&x_new)
               .map(|(a, b)| (a - b).abs())
               .sum();
           
           if diff < tolerance {
               break;
           }
           
           x.copy_from_slice(&x_new);
       }
       
       nodes.into_iter()
           .zip(x_new)
           .map(|(id, score)| (id, score.abs()))
           .collect()
   }
   ```

3. Add weighted variant:
   ```rust
   pub fn weighted_eigenvector_centrality<G: Graph>(
       graph: &G,
       config: EigenvectorConfig,
   ) -> EigenvectorResult<G::Node::Id> {
       // Similar to above but use edge weights in adjacency matrix
   }
   ```

4. Implement neuromorphic-specific variant:
   ```rust
   pub fn neural_eigenvector_centrality<G: NeuromorphicGraph>(
       graph: &G,
   ) -> HashMap<G::Node::Id, f32> {
       // Weight by synaptic strength and activation levels
   }
   ```

## Expected Output
```rust
pub struct EigenvectorResult<Id> {
    pub centrality: HashMap<Id, f32>,
    pub eigenvalue: f32,
    pub iterations: usize,
    pub converged: bool,
}

pub trait EigenvectorCentrality: Graph {
    fn eigenvector_centrality(&self, config: EigenvectorConfig) -> EigenvectorResult<Self::Node::Id>;
}
```

## Verification Steps
1. Test on graphs with known eigenvector centrality
2. Verify convergence properties
3. Compare with analytical solutions for simple graphs
4. Test stability with different initializations

## Time Estimate
25 minutes

## Dependencies
- MP021-MP022: Other centrality measures for comparison
- MP001-MP010: Graph infrastructure