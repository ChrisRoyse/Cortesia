# MP011: Dijkstra's Algorithm Implementation

## Task Description
Implement Dijkstra's shortest path algorithm optimized for neuromorphic graphs with synaptic weights.

## Prerequisites
- MP001-MP010 completed
- Understanding of Dijkstra's algorithm
- Priority queue knowledge

## Detailed Steps

1. Create `src/neuromorphic/graph/algorithms/shortest_path.rs`

2. Implement basic Dijkstra:
   ```rust
   use std::collections::{BinaryHeap, HashMap};
   use std::cmp::Reverse;
   
   pub fn dijkstra<G: Graph>(
       graph: &G,
       start: G::Node::Id,
       end: G::Node::Id,
   ) -> Option<(Vec<G::Node::Id>, f32)> {
       let mut distances = HashMap::new();
       let mut previous = HashMap::new();
       let mut heap = BinaryHeap::new();
       
       distances.insert(start.clone(), 0.0);
       heap.push(Reverse((OrderedFloat(0.0), start.clone())));
       
       while let Some(Reverse((dist, node_id))) = heap.pop() {
           if node_id == end {
               return Some(reconstruct_path(&previous, end));
           }
           
           if dist.0 > distances.get(&node_id).copied().unwrap_or(f32::INFINITY) {
               continue;
           }
           
           // Process neighbors
           if let Some(node) = graph.get_node(&node_id) {
               for neighbor_id in node.neighbors() {
                   let edge_weight = graph.edge_weight(&node_id, &neighbor_id)
                       .unwrap_or(1.0);
                   let new_dist = dist.0 + edge_weight;
                   
                   if new_dist < distances.get(&neighbor_id).copied().unwrap_or(f32::INFINITY) {
                       distances.insert(neighbor_id.clone(), new_dist);
                       previous.insert(neighbor_id.clone(), node_id.clone());
                       heap.push(Reverse((OrderedFloat(new_dist), neighbor_id)));
                   }
               }
           }
       }
       
       None
   }
   ```

3. Add neuromorphic-specific variant:
   - Consider spike propagation delays
   - Account for refractory periods
   - Handle dynamic edge weights

4. Implement bidirectional Dijkstra:
   - Search from both start and end
   - Meet in the middle optimization
   - Reduce search space for large graphs

5. Add path reconstruction:
   ```rust
   fn reconstruct_path<Id: Clone>(
       previous: &HashMap<Id, Id>,
       end: Id,
   ) -> (Vec<Id>, f32) {
       let mut path = vec![end.clone()];
       let mut current = end;
       
       while let Some(prev) = previous.get(&current) {
           path.push(prev.clone());
           current = prev.clone();
       }
       
       path.reverse();
       (path, total_distance)
   }
   ```

6. Optimize with fibonacci heap (optional):
   - Better theoretical complexity
   - Consider practical performance

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/shortest_path.rs
pub struct DijkstraResult<Id> {
    pub path: Vec<Id>,
    pub distance: f32,
    pub visited_nodes: usize,
}

impl<G: Graph> ShortestPath for G {
    fn dijkstra(
        &self,
        start: Self::Node::Id,
        end: Self::Node::Id,
    ) -> Option<DijkstraResult<Self::Node::Id>> {
        // Implementation
    }
    
    fn dijkstra_all(
        &self,
        start: Self::Node::Id,
    ) -> HashMap<Self::Node::Id, (f32, Vec<Self::Node::Id>)> {
        // All shortest paths from start
    }
}
```

## Verification Steps
1. Test on small graphs with known shortest paths
2. Verify correctness with negative edge weights (should fail)
3. Benchmark against reference implementation
4. Test on disconnected graphs

## Time Estimate
30 minutes

## Dependencies
- MP001-MP010: Graph infrastructure