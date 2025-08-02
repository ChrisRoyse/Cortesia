# MP007: Test Harness Creation

## Task Description
Create a comprehensive test harness for graph algorithms with utilities for generating test graphs and validating algorithm correctness.

## Prerequisites
- MP001-MP006 completed
- Understanding of property-based testing
- Familiarity with test fixtures

## Detailed Steps

1. Create `src/neuromorphic/graph/testing/mod.rs`

2. Implement graph generators:
   ```rust
   pub struct GraphGenerator {
       rng: StdRng,
   }
   
   impl GraphGenerator {
       pub fn random_graph(nodes: usize, edge_prob: f64) -> NeuromorphicGraph;
       pub fn scale_free_graph(nodes: usize, m: usize) -> NeuromorphicGraph;
       pub fn small_world_graph(nodes: usize, k: usize, p: f64) -> NeuromorphicGraph;
       pub fn grid_graph(width: usize, height: usize) -> NeuromorphicGraph;
       pub fn complete_graph(nodes: usize) -> NeuromorphicGraph;
   }
   ```

3. Create test fixtures:
   - Predefined small graphs for unit tests
   - Benchmark graphs with known properties
   - Pathological cases (disconnected, cycles, etc.)

4. Implement property-based tests:
   ```rust
   #[cfg(test)]
   mod property_tests {
       use proptest::prelude::*;
       
       proptest! {
           #[test]
           fn graph_invariants(nodes in 1..100usize, edges in 0..1000usize) {
               let graph = generate_random_graph(nodes, edges);
               assert!(graph.node_count() == nodes);
               assert!(graph.edge_count() <= edges);
           }
       }
   }
   ```

5. Add performance benchmarks:
   - Micro-benchmarks for basic operations
   - Scaling tests for large graphs
   - Memory usage profiling

6. Create validation utilities:
   - Graph integrity checker
   - Algorithm output validator
   - Performance regression detector

## Expected Output
```rust
// src/neuromorphic/graph/testing/mod.rs
pub mod generators;
pub mod fixtures;
pub mod validators;

#[cfg(test)]
pub fn assert_graph_valid<G: Graph>(graph: &G) {
    // Check no orphaned edges
    for edge in graph.edges() {
        assert!(graph.has_node(edge.source()));
        assert!(graph.has_node(edge.target()));
    }
    
    // Check node ID uniqueness
    let mut seen = HashSet::new();
    for node in graph.nodes() {
        assert!(seen.insert(node.id()));
    }
}

#[cfg(test)]
pub fn benchmark_operation<F, R>(name: &str, op: F) -> R 
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = op();
    let duration = start.elapsed();
    println!("Operation '{}' took: {:?}", name, duration);
    result
}
```

## Verification Steps
1. Generate graphs of various types and validate properties
2. Run property-based tests with 1000+ iterations
3. Execute benchmarks and check performance
4. Test edge cases with malformed graphs

## Time Estimate
30 minutes

## Dependencies
- MP001-MP006: Complete graph infrastructure