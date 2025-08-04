# MP014: Floyd-Warshall Algorithm Implementation

## Task Description
Implement Floyd-Warshall algorithm for all-pairs shortest paths, useful for precomputing neural pathway distances.

## Prerequisites
- MP011-MP013 completed
- Understanding of dynamic programming
- Matrix operations knowledge

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/floyd_warshall.rs`

2. Implement basic Floyd-Warshall:
   ```rust
   pub fn floyd_warshall<G: Graph>(
       graph: &G,
   ) -> FloydWarshallResult<G::Node::Id> {
       let nodes: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let n = nodes.len();
       
       // Initialize distance matrix
       let mut dist = vec![vec![f32::INFINITY; n]; n];
       let mut next = vec![vec![None; n]; n];
       
       // Set diagonal to 0
       for i in 0..n {
           dist[i][i] = 0.0;
       }
       
       // Initialize with direct edges
       for edge in graph.edges() {
           let u_idx = nodes.iter().position(|id| id == &edge.source()).unwrap();
           let v_idx = nodes.iter().position(|id| id == &edge.target()).unwrap();
           dist[u_idx][v_idx] = edge.weight();
           next[u_idx][v_idx] = Some(v_idx);
       }
       
       // Main algorithm
       for k in 0..n {
           for i in 0..n {
               for j in 0..n {
                   if dist[i][k] + dist[k][j] < dist[i][j] {
                       dist[i][j] = dist[i][k] + dist[k][j];
                       next[i][j] = next[i][k];
                   }
               }
           }
       }
       
       FloydWarshallResult {
           distances: dist,
           next_node: next,
           node_mapping: nodes,
       }
   }
   ```

3. Add path reconstruction:
   ```rust
   impl<Id: Clone> FloydWarshallResult<Id> {
       pub fn get_path(&self, from: &Id, to: &Id) -> Option<Vec<Id>> {
           let i = self.node_mapping.iter().position(|id| id == from)?;
           let j = self.node_mapping.iter().position(|id| id == to)?;
           
           if self.distances[i][j] == f32::INFINITY {
               return None;
           }
           
           let mut path = vec![from.clone()];
           let mut current = i;
           
           while current != j {
               current = self.next_node[current][j]?;
               path.push(self.node_mapping[current].clone());
           }
           
           Some(path)
       }
       
       pub fn distance(&self, from: &Id, to: &Id) -> Option<f32> {
           let i = self.node_mapping.iter().position(|id| id == from)?;
           let j = self.node_mapping.iter().position(|id| id == to)?;
           
           let dist = self.distances[i][j];
           if dist == f32::INFINITY {
               None
           } else {
               Some(dist)
           }
       }
   }
   ```

4. Implement blocked Floyd-Warshall:
   - Cache-friendly blocked matrix multiplication
   - Parallel block processing
   - Memory-efficient for large graphs

5. Add incremental updates:
   - Handle edge insertions/deletions
   - Avoid full recomputation
   - Maintain consistency

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/floyd_warshall.rs
pub struct FloydWarshallResult<Id> {
    distances: Vec<Vec<f32>>,
    next_node: Vec<Vec<Option<usize>>>,
    node_mapping: Vec<Id>,
}

pub trait AllPairsShortestPath: Graph {
    fn floyd_warshall(&self) -> FloydWarshallResult<Self::Node::Id>;
    
    fn floyd_warshall_parallel(&self, threads: usize) -> FloydWarshallResult<Self::Node::Id>;
}

impl<Id> FloydWarshallResult<Id> {
    pub fn diameter(&self) -> f32 {
        // Maximum shortest path distance
    }
    
    pub fn is_connected(&self) -> bool {
        // Check if graph is fully connected
    }
}
```

## Verification Steps
1. Test on small graphs with known all-pairs distances
2. Verify path reconstruction accuracy
3. Benchmark against repeated Dijkstra
4. Test memory usage for large graphs

## Time Estimate
30 minutes

## Dependencies
- MP011-MP013: Previous shortest path algorithms
- MP001-MP010: Graph infrastructure