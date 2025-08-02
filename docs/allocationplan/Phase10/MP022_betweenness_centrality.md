# MP022: Betweenness Centrality Algorithm

## Task Description
Implement Brandes' algorithm for betweenness centrality computation, identifying critical pathway nodes in neural networks.

## Prerequisites
- MP015-MP016 completed (BFS/DFS)
- Understanding of betweenness centrality
- Stack and queue operations

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/centrality/betweenness.rs`

2. Implement Brandes' algorithm:
   ```rust
   pub fn betweenness_centrality<G: Graph>(
       graph: &G,
   ) -> HashMap<G::Node::Id, f32> {
       let mut cb: HashMap<G::Node::Id, f32> = HashMap::new();
       
       // Initialize centrality scores
       for node in graph.nodes() {
           cb.insert(node.id(), 0.0);
       }
       
       for s in graph.nodes() {
           let s_id = s.id();
           let mut stack = Vec::new();
           let mut paths: HashMap<G::Node::Id, Vec<G::Node::Id>> = HashMap::new();
           let mut sigma: HashMap<G::Node::Id, f32> = HashMap::new();
           let mut d: HashMap<G::Node::Id, i32> = HashMap::new();
           let mut delta: HashMap<G::Node::Id, f32> = HashMap::new();
           
           // Single-source shortest-paths problem
           for v in graph.nodes() {
               paths.insert(v.id(), Vec::new());
               sigma.insert(v.id(), 0.0);
               d.insert(v.id(), -1);
               delta.insert(v.id(), 0.0);
           }
           
           sigma.insert(s_id.clone(), 1.0);
           d.insert(s_id.clone(), 0);
           
           let mut queue = VecDeque::new();
           queue.push_back(s_id.clone());
           
           while let Some(v) = queue.pop_front() {
               stack.push(v.clone());
               
               if let Some(node) = graph.get_node(&v) {
                   for w in node.neighbors() {
                       // First time we reach w?
                       if d[&w] < 0 {
                           queue.push_back(w.clone());
                           d.insert(w.clone(), d[&v] + 1);
                       }
                       
                       // Shortest path to w via v?
                       if d[&w] == d[&v] + 1 {
                           sigma.insert(w.clone(), sigma[&w] + sigma[&v]);
                           paths.get_mut(&w).unwrap().push(v.clone());
                       }
                   }
               }
           }
           
           // Accumulation
           while let Some(w) = stack.pop() {
               for v in &paths[&w] {
                   let delta_v = delta[v] + (sigma[v] / sigma[&w]) * (1.0 + delta[&w]);
                   delta.insert(v.clone(), delta_v);
               }
               
               if w != s_id {
                   cb.insert(w.clone(), cb[&w] + delta[&w]);
               }
           }
       }
       
       // Normalize for undirected graphs
       let n = graph.node_count() as f32;
       let normalization = 1.0 / ((n - 1.0) * (n - 2.0));
       
       for score in cb.values_mut() {
           *score *= normalization;
       }
       
       cb
   }
   ```

3. Implement approximate betweenness:
   ```rust
   pub fn approximate_betweenness<G: Graph>(
       graph: &G,
       k: usize, // Number of sample vertices
   ) -> HashMap<G::Node::Id, f32> {
       let mut rng = thread_rng();
       let vertices: Vec<_> = graph.nodes().map(|n| n.id()).collect();
       let sample_vertices: Vec<_> = vertices.choose_multiple(&mut rng, k).cloned().collect();
       
       let mut cb = HashMap::new();
       for node in graph.nodes() {
           cb.insert(node.id(), 0.0);
       }
       
       for s in sample_vertices {
           // Run single-source betweenness
           let single_source_cb = single_source_betweenness(graph, s);
           
           for (node, score) in single_source_cb {
               *cb.get_mut(&node).unwrap() += score;
           }
       }
       
       // Scale by sampling factor
       let scale = graph.node_count() as f32 / k as f32;
       for score in cb.values_mut() {
           *score *= scale;
       }
       
       cb
   }
   ```

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/centrality/betweenness.rs
pub trait BetweennessCentrality: Graph {
    fn betweenness_centrality(&self) -> HashMap<Self::Node::Id, f32>;
    fn approximate_betweenness(&self, samples: usize) -> HashMap<Self::Node::Id, f32>;
    fn edge_betweenness(&self) -> HashMap<(Self::Node::Id, Self::Node::Id), f32>;
}

pub struct BetweennessResult<Id> {
    pub node_centrality: HashMap<Id, f32>,
    pub edge_centrality: HashMap<(Id, Id), f32>,
    pub computation_time: Duration,
}
```

## Verification Steps
1. Test on graphs with known betweenness values
2. Compare exact vs approximate results
3. Verify normalization correctness
4. Benchmark performance scaling

## Time Estimate
30 minutes

## Dependencies
- MP015-MP016: BFS for shortest paths
- MP001-MP010: Graph infrastructure