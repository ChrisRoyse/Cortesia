# MP019: Maximum Flow Algorithms

## Task Description
Implement Ford-Fulkerson and Edmonds-Karp algorithms for maximum flow computation, useful for neural capacity analysis.

## Prerequisites
- MP015-MP016 completed (BFS for Edmonds-Karp)
- Understanding of flow networks
- Residual graph concepts

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/max_flow.rs`

2. Define flow network traits:
   ```rust
   pub trait FlowNetwork: Graph {
       fn capacity(&self, from: &Self::Node::Id, to: &Self::Node::Id) -> f32;
       fn set_flow(&mut self, from: &Self::Node::Id, to: &Self::Node::Id, flow: f32);
       fn get_flow(&self, from: &Self::Node::Id, to: &Self::Node::Id) -> f32;
   }
   
   pub struct FlowEdge<Id> {
       pub from: Id,
       pub to: Id,
       pub capacity: f32,
       pub flow: f32,
   }
   ```

3. Implement Ford-Fulkerson with DFS:
   ```rust
   pub fn ford_fulkerson<G: FlowNetwork>(
       graph: &mut G,
       source: G::Node::Id,
       sink: G::Node::Id,
   ) -> f32 {
       // Initialize all flows to 0
       for edge in graph.edges() {
           graph.set_flow(&edge.source(), &edge.target(), 0.0);
       }
       
       let mut max_flow = 0.0;
       
       while let Some(path) = find_augmenting_path_dfs(graph, &source, &sink) {
           // Find minimum residual capacity along path
           let mut path_flow = f32::INFINITY;
           
           for i in 0..path.len() - 1 {
               let residual = graph.capacity(&path[i], &path[i + 1]) 
                   - graph.get_flow(&path[i], &path[i + 1]);
               path_flow = path_flow.min(residual);
           }
           
           // Update flows along path
           for i in 0..path.len() - 1 {
               let current_flow = graph.get_flow(&path[i], &path[i + 1]);
               graph.set_flow(&path[i], &path[i + 1], current_flow + path_flow);
               
               // Update reverse flow
               let reverse_flow = graph.get_flow(&path[i + 1], &path[i]);
               graph.set_flow(&path[i + 1], &path[i], reverse_flow - path_flow);
           }
           
           max_flow += path_flow;
       }
       
       max_flow
   }
   ```

4. Implement Edmonds-Karp with BFS:
   ```rust
   pub fn edmonds_karp<G: FlowNetwork>(
       graph: &mut G,
       source: G::Node::Id,
       sink: G::Node::Id,
   ) -> MaxFlowResult<G::Node::Id> {
       let mut max_flow = 0.0;
       let mut parent = HashMap::new();
       let mut augmenting_paths = Vec::new();
       
       loop {
           parent.clear();
           
           // BFS to find shortest augmenting path
           let mut queue = VecDeque::new();
           queue.push_back(source.clone());
           parent.insert(source.clone(), None);
           
           let mut found_path = false;
           
           while let Some(u) = queue.pop_front() {
               if u == sink {
                   found_path = true;
                   break;
               }
               
               if let Some(node) = graph.get_node(&u) {
                   for v in node.neighbors() {
                       let residual = graph.capacity(&u, &v) - graph.get_flow(&u, &v);
                       
                       if !parent.contains_key(&v) && residual > 0.0 {
                           parent.insert(v.clone(), Some(u.clone()));
                           queue.push_back(v);
                       }
                   }
               }
           }
           
           if !found_path {
               break;
           }
           
           // Find minimum residual capacity
           let path = reconstruct_path(&parent, &sink);
           let path_flow = compute_path_flow(graph, &path);
           
           // Update flows
           update_flows(graph, &path, path_flow);
           
           augmenting_paths.push((path, path_flow));
           max_flow += path_flow;
       }
       
       MaxFlowResult {
           max_flow,
           augmenting_paths,
           min_cut: find_min_cut(graph, &source),
       }
   }
   ```

5. Add push-relabel algorithm for better performance:
   ```rust
   pub fn push_relabel<G: FlowNetwork>(
       graph: &G,
       source: G::Node::Id,
       sink: G::Node::Id,
   ) -> f32 {
       // Implementation of push-relabel algorithm
       // More efficient for dense graphs
   }
   ```

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/max_flow.rs
pub struct MaxFlowResult<Id> {
    pub max_flow: f32,
    pub augmenting_paths: Vec<(Vec<Id>, f32)>,
    pub min_cut: (Vec<Id>, Vec<Id>),
}

pub trait MaxFlow: FlowNetwork {
    fn ford_fulkerson(&mut self, source: Self::Node::Id, sink: Self::Node::Id) -> f32;
    fn edmonds_karp(&mut self, source: Self::Node::Id, sink: Self::Node::Id) -> MaxFlowResult<Self::Node::Id>;
    fn push_relabel(&self, source: Self::Node::Id, sink: Self::Node::Id) -> f32;
}
```

## Verification Steps
1. Test on networks with known max flow
2. Verify min-cut theorem (max flow = min cut)
3. Compare algorithm performance
4. Test with multiple sources/sinks

## Time Estimate
30 minutes

## Dependencies
- MP015-MP016: DFS/BFS for path finding
- MP001-MP010: Graph infrastructure