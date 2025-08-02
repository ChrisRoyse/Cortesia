# MP061: Unit Test Framework

## Task Description
Establish comprehensive unit testing framework for all graph algorithms with neuromorphic-specific assertions.

## Prerequisites
- MP001-MP060 completed
- Understanding of testing principles
- Knowledge of Rust test frameworks

## Detailed Steps

1. Create `tests/unit/graph_algorithms/mod.rs`

2. Implement test utilities:
   ```rust
   pub struct GraphTestUtils;
   
   impl GraphTestUtils {
       pub fn create_test_graph(nodes: usize, edges: usize) -> TestGraph {
           let mut graph = TestGraph::new();
           
           // Create deterministic nodes
           for i in 0..nodes {
               graph.add_node(NodeId(i), format!("node_{}", i));
           }
           
           // Create deterministic edges with predictable weights
           let mut rng = StdRng::seed_from_u64(42);
           for _ in 0..edges {
               let from = NodeId(rng.gen_range(0..nodes));
               let to = NodeId(rng.gen_range(0..nodes));
               let weight = rng.gen_range(1.0..10.0);
               graph.add_edge(from, to, weight);
           }
           
           graph
       }
       
       pub fn assert_graph_invariants<G: Graph>(graph: &G) {
           // Verify no orphaned edges
           for edge in graph.edges() {
               assert!(graph.contains_node(edge.source()));
               assert!(graph.contains_node(edge.target()));
           }
           
           // Check node ID uniqueness
           let mut node_ids = std::collections::HashSet::new();
           for node in graph.nodes() {
               assert!(node_ids.insert(node.id()));
           }
           
           // Validate edge consistency
           for edge in graph.edges() {
               assert!(edge.weight() >= 0.0);
               assert!(!edge.weight().is_nan());
               assert!(edge.weight().is_finite());
           }
       }
       
       pub fn assert_neuromorphic_properties<G: NeuromorphicGraph>(graph: &G) {
           // Verify spike timing constraints
           for spike in graph.spike_events() {
               assert!(spike.timestamp() >= 0.0);
               assert!(spike.amplitude() >= 0.0 && spike.amplitude() <= 1.0);
           }
           
           // Check activation patterns
           for node in graph.active_nodes() {
               assert!(graph.activation_level(node) > 0.0);
               assert!(graph.activation_level(node) <= 1.0);
           }
       }
   }
   ```

3. Create algorithm-specific test modules:
   ```rust
   mod dijkstra_tests {
       use super::*;
       
       #[test]
       fn test_dijkstra_shortest_path() {
           let graph = GraphTestUtils::create_test_graph(5, 8);
           let result = dijkstra(&graph, NodeId(0), NodeId(4));
           
           assert!(result.is_some());
           let (distance, path) = result.unwrap();
           assert!(distance >= 0.0);
           assert!(path.starts_with(&[NodeId(0)]));
           assert!(path.ends_with(&[NodeId(4)]));
           GraphTestUtils::assert_path_validity(&graph, &path);
       }
       
       #[test]
       fn test_dijkstra_unreachable_node() {
           let mut graph = TestGraph::new();
           graph.add_node(NodeId(0), "isolated_0");
           graph.add_node(NodeId(1), "isolated_1");
           
           let result = dijkstra(&graph, NodeId(0), NodeId(1));
           assert!(result.is_none());
       }
   }
   ```

4. Implement neuromorphic-specific test assertions:
   ```rust
   pub struct NeuromorphicTestAssertions;
   
   impl NeuromorphicTestAssertions {
       pub fn assert_spike_timing_valid(spikes: &[SpikeEvent]) {
           for window in spikes.windows(2) {
               assert!(window[0].timestamp() <= window[1].timestamp());
           }
       }
       
       pub fn assert_activation_conservation<G: NeuromorphicGraph>(graph: &G) {
           let total_activation: f64 = graph.nodes()
               .map(|node| graph.activation_level(node.id()))
               .sum();
           
           // Total activation should not exceed theoretical maximum
           assert!(total_activation <= graph.node_count() as f64);
       }
       
       pub fn assert_lateral_inhibition_effect<G: NeuromorphicGraph>(
           graph: &G, 
           activated_node: NodeId
       ) {
           let neighbors = graph.neighbors(activated_node);
           let activation = graph.activation_level(activated_node);
           
           for neighbor in neighbors {
               let neighbor_activation = graph.activation_level(neighbor);
               // Neighbors should have reduced activation due to inhibition
               assert!(neighbor_activation < activation * 0.8);
           }
       }
   }
   ```

5. Create performance benchmarking utilities:
   ```rust
   pub struct PerformanceTestUtils;
   
   impl PerformanceTestUtils {
       pub fn time_algorithm<F, R>(algorithm: F) -> (R, Duration)
       where
           F: FnOnce() -> R,
       {
           let start = Instant::now();
           let result = algorithm();
           let duration = start.elapsed();
           (result, duration)
       }
       
       pub fn assert_performance_within_bounds<F, R>(
           algorithm: F,
           max_duration: Duration,
           description: &str
       ) -> R
       where
           F: FnOnce() -> R,
       {
           let (result, duration) = Self::time_algorithm(algorithm);
           assert!(
               duration <= max_duration,
               "{} took {:?}, expected <= {:?}",
               description, duration, max_duration
           );
           result
       }
   }
   ```

## Expected Output
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_algorithms::*;
    
    #[test]
    fn test_dijkstra_correctness() {
        let graph = GraphTestUtils::create_test_graph(10, 20);
        GraphTestUtils::assert_graph_invariants(&graph);
        
        let result = dijkstra(&graph, NodeId(0), NodeId(9));
        assert!(result.is_some());
        
        let (distance, path) = result.unwrap();
        assert!(distance >= 0.0);
        assert_eq!(path.first(), Some(&NodeId(0)));
        assert_eq!(path.last(), Some(&NodeId(9)));
    }
    
    #[test]
    fn test_neuromorphic_graph_properties() {
        let mut graph = NeuromorphicTestGraph::new();
        graph.simulate_spike_train(100);
        
        NeuromorphicTestAssertions::assert_spike_timing_valid(&graph.spike_events());
        NeuromorphicTestAssertions::assert_activation_conservation(&graph);
    }
    
    #[test]
    fn test_performance_requirements() {
        let large_graph = GraphTestUtils::create_test_graph(1000, 5000);
        
        PerformanceTestUtils::assert_performance_within_bounds(
            || dijkstra(&large_graph, NodeId(0), NodeId(999)),
            Duration::from_millis(100),
            "Dijkstra on 1000-node graph"
        );
    }
}
```

## Verification Steps
1. Run test suite and verify 100% pass rate
2. Check code coverage metrics exceed 90%
3. Validate test isolation (tests don't affect each other)
4. Verify test performance meets requirements
5. Ensure neuromorphic-specific assertions work correctly

## Time Estimate
25 minutes

## Dependencies
- MP001-MP060: All implementations to test
- Rust testing framework
- Test data generation utilities