# MP012: A* Algorithm Implementation

## Task Description
Implement A* pathfinding algorithm with neuromorphic-specific heuristics for optimal path planning in spiking neural networks.

## Prerequisites
- MP011 completed (Dijkstra implementation)
- Understanding of A* algorithm
- Heuristic function design

## Detailed Steps

1. Create heuristic trait in `src/neuromorphic/graph/algorithms/heuristics.rs`:
   ```rust
   pub trait Heuristic<G: Graph> {
       fn estimate(&self, graph: &G, from: &G::Node::Id, to: &G::Node::Id) -> f32;
   }
   ```

2. Implement standard heuristics:
   ```rust
   pub struct EuclideanHeuristic;
   pub struct ManhattanHeuristic;
   pub struct NeuralActivityHeuristic;
   
   impl<G: SpatialGraph> Heuristic<G> for EuclideanHeuristic {
       fn estimate(&self, graph: &G, from: &G::Node::Id, to: &G::Node::Id) -> f32 {
           let from_pos = graph.position(from);
           let to_pos = graph.position(to);
           euclidean_distance(from_pos, to_pos)
       }
   }
   ```

3. Implement A* algorithm:
   ```rust
   pub fn astar<G: Graph, H: Heuristic<G>>(
       graph: &G,
       start: G::Node::Id,
       goal: G::Node::Id,
       heuristic: &H,
   ) -> Option<AStarResult<G::Node::Id>> {
       let mut open_set = BinaryHeap::new();
       let mut g_score = HashMap::new();
       let mut f_score = HashMap::new();
       let mut came_from = HashMap::new();
       
       g_score.insert(start.clone(), 0.0);
       let h = heuristic.estimate(graph, &start, &goal);
       f_score.insert(start.clone(), h);
       open_set.push(Reverse((OrderedFloat(h), start.clone())));
       
       while let Some(Reverse((_, current))) = open_set.pop() {
           if current == goal {
               return Some(reconstruct_astar_path(&came_from, current, g_score));
           }
           
           // Process neighbors with heuristic
       }
       None
   }
   ```

4. Add neuromorphic-specific heuristics:
   - Spike propagation likelihood
   - Energy efficiency paths
   - Refractory state avoidance

5. Implement adaptive A*:
   - Dynamic heuristic weight adjustment
   - Learning from previous searches
   - Cache frequently used paths

## Expected Output
```rust
// src/neuromorphic/graph/algorithms/astar.rs
pub struct AStarResult<Id> {
    pub path: Vec<Id>,
    pub cost: f32,
    pub nodes_explored: usize,
    pub heuristic_accuracy: f32,
}

pub trait AStarGraph: Graph {
    fn astar<H: Heuristic<Self>>(
        &self,
        start: Self::Node::Id,
        goal: Self::Node::Id,
        heuristic: &H,
    ) -> Option<AStarResult<Self::Node::Id>>;
}
```

## Verification Steps
1. Compare A* results with Dijkstra for optimality
2. Test heuristic admissibility
3. Benchmark speedup over Dijkstra
4. Verify path quality with different heuristics

## Time Estimate
25 minutes

## Dependencies
- MP011: Dijkstra implementation for comparison
- MP001-MP010: Graph infrastructure