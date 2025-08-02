# Micro Task 06: Activation Tests

**Priority**: CRITICAL  
**Estimated Time**: 40 minutes  
**Dependencies**: 05_convergence_detection.md  
**Skills Required**: Rust testing, property-based testing, performance analysis

## Objective

Create a comprehensive test suite covering all Day 1 activation components with unit tests, integration tests, and performance benchmarks.

## Context

Testing is crucial for ensuring the activation system works correctly under all conditions. This includes unit tests for individual components, integration tests for component interactions, property-based tests for mathematical correctness, and performance benchmarks for optimization validation.

## Specifications

### Test Coverage Requirements

1. **Unit Tests** - Individual component functionality
2. **Integration Tests** - Component interaction verification  
3. **Property Tests** - Mathematical invariant verification
4. **Performance Tests** - Benchmark and profiling
5. **Edge Case Tests** - Boundary condition handling
6. **Stress Tests** - Large-scale operation validation

### Test Organization
- Separate test files for each component
- Common test utilities and fixtures
- Property-based testing with QuickCheck
- Performance benchmarks with criterion
- Mock objects for isolated testing

## Implementation Guide

### Step 1: Create Test Infrastructure
```rust
// File: tests/activation/mod.rs

pub mod test_utils;
pub mod fixtures;

pub use test_utils::*;
pub use fixtures::*;

use crate::core::{Graph, NodeId, ActivationState};
use std::collections::HashMap;

// Test utilities for activation testing
pub struct ActivationTestUtils;

impl ActivationTestUtils {
    pub fn create_linear_graph(nodes: usize) -> Graph {
        let mut graph = Graph::new();
        
        for i in 0..nodes {
            graph.add_node(NodeId(i)).unwrap();
        }
        
        for i in 0..nodes-1 {
            graph.add_edge(NodeId(i), NodeId(i+1), 1.0).unwrap();
        }
        
        graph
    }
    
    pub fn create_star_graph(spokes: usize) -> Graph {
        let mut graph = Graph::new();
        
        // Center node
        graph.add_node(NodeId(0)).unwrap();
        
        // Spoke nodes
        for i in 1..=spokes {
            graph.add_node(NodeId(i)).unwrap();
            graph.add_edge(NodeId(0), NodeId(i), 1.0).unwrap();
            graph.add_edge(NodeId(i), NodeId(0), 1.0).unwrap();
        }
        
        graph
    }
    
    pub fn create_ring_graph(nodes: usize) -> Graph {
        let mut graph = Graph::new();
        
        for i in 0..nodes {
            graph.add_node(NodeId(i)).unwrap();
        }
        
        for i in 0..nodes {
            let next = (i + 1) % nodes;
            graph.add_edge(NodeId(i), NodeId(next), 1.0).unwrap();
        }
        
        graph
    }
    
    pub fn create_random_graph(nodes: usize, edge_probability: f32) -> Graph {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut graph = Graph::new();
        
        // Add nodes
        for i in 0..nodes {
            graph.add_node(NodeId(i)).unwrap();
        }
        
        // Add random edges
        for i in 0..nodes {
            for j in i+1..nodes {
                if rng.gen::<f32>() < edge_probability {
                    let weight = rng.gen_range(0.1..1.0);
                    graph.add_edge(NodeId(i), NodeId(j), weight).unwrap();
                    graph.add_edge(NodeId(j), NodeId(i), weight).unwrap();
                }
            }
        }
        
        graph
    }
    
    pub fn assert_activation_normalized(state: &ActivationState) {
        for node in state.activated_nodes() {
            let activation = state.get_activation(node);
            assert!(activation >= 0.0 && activation <= 1.0, 
                   "Activation {} for node {:?} out of range [0,1]", activation, node);
        }
    }
    
    pub fn assert_energy_conservation(
        initial: &ActivationState, 
        final_state: &ActivationState,
        tolerance: f32
    ) {
        let initial_energy: f32 = initial.activated_nodes().iter()
            .map(|&node| initial.get_activation(node).powi(2))
            .sum();
        
        let final_energy: f32 = final_state.activated_nodes().iter()
            .map(|&node| final_state.get_activation(node).powi(2))
            .sum();
        
        let energy_diff = (final_energy - initial_energy).abs();
        assert!(energy_diff <= tolerance, 
               "Energy not conserved: initial={}, final={}, diff={}", 
               initial_energy, final_energy, energy_diff);
    }
}
```

### Step 2: Activation State Tests
```rust
// File: tests/activation/state_tests.rs

use super::*;
use crate::core::activation::ActivationState;

#[cfg(test)]
mod activation_state_tests {
    use super::*;
    
    #[test]
    fn test_basic_state_operations() {
        let mut state = ActivationState::new();
        
        // Test setting activation
        state.set_activation(NodeId(1), 0.5);
        assert_eq!(state.get_activation(NodeId(1)), 0.5);
        
        // Test getting non-existent activation
        assert_eq!(state.get_activation(NodeId(999)), 0.0);
        
        // Test adding activation
        state.add_activation(NodeId(1), 0.3);
        assert_eq!(state.get_activation(NodeId(1)), 0.8);
        
        // Test clamping
        state.set_activation(NodeId(2), 1.5);
        assert_eq!(state.get_activation(NodeId(2)), 1.0);
        
        state.set_activation(NodeId(3), -0.5);
        assert_eq!(state.get_activation(NodeId(3)), 0.0);
    }
    
    #[test]
    fn test_threshold_cleanup() {
        let mut state = ActivationState::new();
        
        // Below threshold should be removed
        state.set_activation(NodeId(1), 0.0005);
        assert_eq!(state.activated_nodes().len(), 0);
        
        // Above threshold should remain
        state.set_activation(NodeId(2), 0.002);
        assert_eq!(state.activated_nodes().len(), 1);
        assert!(state.activated_nodes().contains(&NodeId(2)));
    }
    
    #[test]
    fn test_energy_calculation() {
        let mut state = ActivationState::new();
        
        state.set_activation(NodeId(1), 0.6);
        state.set_activation(NodeId(2), 0.8);
        
        let expected_energy = 0.6_f32.powi(2) + 0.8_f32.powi(2);
        assert!((state.total_energy() - expected_energy).abs() < 0.001);
    }
    
    #[test]
    fn test_state_statistics() {
        let mut state = ActivationState::new();
        
        state.set_activation(NodeId(1), 0.3);
        state.set_activation(NodeId(2), 0.7);
        state.set_activation(NodeId(3), 0.5);
        
        assert_eq!(state.total_activation(), 1.5);
        assert_eq!(state.max_activation(), 0.7);
        assert_eq!(state.activated_nodes().len(), 3);
    }
    
    #[test]
    fn test_state_clear() {
        let mut state = ActivationState::new();
        
        state.set_activation(NodeId(1), 0.5);
        state.set_activation(NodeId(2), 0.8);
        
        state.clear();
        
        assert_eq!(state.activated_nodes().len(), 0);
        assert_eq!(state.total_activation(), 0.0);
        assert_eq!(state.total_energy(), 0.0);
    }
    
    #[test]
    fn test_concurrent_access() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let state = Arc::new(Mutex::new(ActivationState::new()));
        let mut handles = vec![];
        
        // Spawn multiple threads to modify state
        for i in 0..10 {
            let state_clone = Arc::clone(&state);
            let handle = thread::spawn(move || {
                let mut state = state_clone.lock().unwrap();
                state.set_activation(NodeId(i), 0.5);
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        let final_state = state.lock().unwrap();
        assert_eq!(final_state.activated_nodes().len(), 10);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use quickcheck::{quickcheck, TestResult};
    
    quickcheck! {
        fn prop_activation_always_in_range(activation: f32) -> TestResult {
            let mut state = ActivationState::new();
            state.set_activation(NodeId(1), activation);
            
            let stored = state.get_activation(NodeId(1));
            TestResult::from_bool(stored >= 0.0 && stored <= 1.0)
        }
        
        fn prop_energy_never_negative(activations: Vec<(u32, f32)>) -> TestResult {
            let mut state = ActivationState::new();
            
            for (node_id, activation) in activations {
                state.set_activation(NodeId(node_id), activation);
            }
            
            TestResult::from_bool(state.total_energy() >= 0.0)
        }
        
        fn prop_total_activation_equals_sum(activations: Vec<(u32, f32)>) -> TestResult {
            let mut state = ActivationState::new();
            
            for (node_id, activation) in activations {
                state.set_activation(NodeId(node_id), activation);
            }
            
            let calculated_total: f32 = state.activated_nodes().iter()
                .map(|&node| state.get_activation(node))
                .sum();
            
            let diff = (state.total_activation() - calculated_total).abs();
            TestResult::from_bool(diff < 0.001)
        }
    }
}
```

### Step 3: Spreader Algorithm Tests
```rust
// File: tests/activation/spreader_tests.rs

use super::*;
use crate::core::activation::{ActivationSpreader, ActivationState};

#[cfg(test)]
mod spreader_tests {
    use super::*;
    
    #[test]
    fn test_basic_spreading() {
        let spreader = ActivationSpreader::new();
        let graph = ActivationTestUtils::create_linear_graph(3); // A -> B -> C
        
        let mut initial_state = ActivationState::new();
        initial_state.set_activation(NodeId(0), 1.0);
        
        let result = spreader.spread_activation(&initial_state, &graph).unwrap();
        
        // Verify spreading occurred
        assert!(result.final_state.get_activation(NodeId(1)) > 0.0);
        assert!(result.final_state.get_activation(NodeId(2)) > 0.0);
        
        // Verify convergence
        assert!(result.converged || result.iterations < 50);
    }
    
    #[test]
    fn test_spreading_conservation() {
        let spreader = ActivationSpreader::with_config(0.9, 0.001, 100);
        let graph = ActivationTestUtils::create_ring_graph(5);
        
        let mut initial_state = ActivationState::new();
        initial_state.set_activation(NodeId(0), 1.0);
        
        let result = spreader.spread_activation(&initial_state, &graph).unwrap();
        
        // Energy should decay but not disappear completely
        let initial_energy = initial_state.total_energy();
        let final_energy = result.final_state.total_energy();
        
        assert!(final_energy > 0.0);
        assert!(final_energy <= initial_energy);
    }
    
    #[test]
    fn test_spreading_convergence() {
        let spreader = ActivationSpreader::new();
        let graph = ActivationTestUtils::create_star_graph(4);
        
        let mut initial_state = ActivationState::new();
        initial_state.set_activation(NodeId(0), 1.0); // Center
        
        let result = spreader.spread_activation(&initial_state, &graph).unwrap();
        
        assert!(result.converged);
        assert!(result.iterations > 0);
        
        // All spokes should have some activation
        for i in 1..=4 {
            assert!(result.final_state.get_activation(NodeId(i)) > 0.0);
        }
    }
    
    #[test]
    fn test_multiple_source_spreading() {
        let spreader = ActivationSpreader::new();
        let graph = ActivationTestUtils::create_linear_graph(5);
        
        let mut initial_state = ActivationState::new();
        initial_state.set_activation(NodeId(0), 0.5);
        initial_state.set_activation(NodeId(4), 0.5);
        
        let result = spreader.spread_activation(&initial_state, &graph).unwrap();
        
        // Middle node should receive activation from both sources
        assert!(result.final_state.get_activation(NodeId(2)) > 0.0);
    }
    
    #[test]
    fn test_disconnected_graph_spreading() {
        let mut graph = Graph::new();
        
        // Create two disconnected components
        for i in 0..6 {
            graph.add_node(NodeId(i)).unwrap();
        }
        
        // Component 1: 0-1-2
        graph.add_edge(NodeId(0), NodeId(1), 1.0).unwrap();
        graph.add_edge(NodeId(1), NodeId(2), 1.0).unwrap();
        
        // Component 2: 3-4-5
        graph.add_edge(NodeId(3), NodeId(4), 1.0).unwrap();
        graph.add_edge(NodeId(4), NodeId(5), 1.0).unwrap();
        
        let spreader = ActivationSpreader::new();
        let mut initial_state = ActivationState::new();
        initial_state.set_activation(NodeId(0), 1.0);
        
        let result = spreader.spread_activation(&initial_state, &graph).unwrap();
        
        // Activation should only spread within component 1
        assert!(result.final_state.get_activation(NodeId(1)) > 0.0);
        assert!(result.final_state.get_activation(NodeId(2)) > 0.0);
        
        // Component 2 should remain unactivated
        assert_eq!(result.final_state.get_activation(NodeId(3)), 0.0);
        assert_eq!(result.final_state.get_activation(NodeId(4)), 0.0);
        assert_eq!(result.final_state.get_activation(NodeId(5)), 0.0);
    }
}
```

### Step 4: Integration Tests
```rust
// File: tests/activation/integration_tests.rs

use super::*;
use crate::core::activation::*;

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_spreading_with_decay() {
        let decay_fn = DecayFunction::exponential(2.0);
        let spreader = ActivationSpreader::new().with_decay(decay_fn);
        
        let graph = ActivationTestUtils::create_linear_graph(4);
        let mut initial_state = ActivationState::new();
        initial_state.set_activation(NodeId(0), 1.0);
        
        let result = spreader.spread_activation(&initial_state, &graph).unwrap();
        
        // Decay should reduce total activation over time
        let initial_total = initial_state.total_activation();
        let final_total = result.final_state.total_activation();
        
        assert!(final_total < initial_total);
        assert!(final_total > 0.0); // But not completely gone
    }
    
    #[test]
    fn test_spreading_with_inhibition() {
        let inhibition = LateralInhibition::global_winner_take_all(0.2);
        let spreader = ActivationSpreader::new().with_inhibition(inhibition);
        
        let graph = ActivationTestUtils::create_star_graph(3);
        let mut initial_state = ActivationState::new();
        
        // Multiple competing sources
        initial_state.set_activation(NodeId(1), 0.8);
        initial_state.set_activation(NodeId(2), 0.6);
        initial_state.set_activation(NodeId(3), 0.7);
        
        let result = spreader.spread_with_inhibition(&initial_state, &graph).unwrap();
        
        // Should converge to winner-take-all
        let activated_nodes = result.final_state.activated_nodes();
        let non_zero_count = activated_nodes.iter()
            .filter(|&&node| result.final_state.get_activation(node) > 0.01)
            .count();
        
        assert!(non_zero_count <= 2); // Winner + possibly center node
    }
    
    #[test]
    fn test_spreading_with_convergence_detection() {
        let detector = ConvergenceDetector::energy_based(0.01);
        let spreader = ActivationSpreader::new()
            .with_convergence_detector(detector);
        
        let graph = ActivationTestUtils::create_ring_graph(6);
        let mut initial_state = ActivationState::new();
        initial_state.set_activation(NodeId(0), 1.0);
        
        let result = spreader.spread_with_convergence_detection(&initial_state, &graph).unwrap();
        
        assert!(result.converged);
        assert!(result.convergence_info.is_some());
        
        let convergence_info = result.convergence_info.unwrap();
        assert_eq!(convergence_info.status, ConvergenceStatus::Converged);
    }
    
    #[test]
    fn test_full_pipeline_integration() {
        // Test complete pipeline: spreading + decay + inhibition + convergence
        let decay_fn = DecayFunction::linear(0.1);
        let inhibition = LateralInhibition::k_winners_take_all(2, 0.3);
        let detector = ConvergenceDetector::combined(0.01, 0.01);
        
        let spreader = ActivationSpreader::with_config(0.8, 0.01, 50)
            .with_decay(decay_fn)
            .with_inhibition(inhibition)
            .with_convergence_detector(detector);
        
        let graph = ActivationTestUtils::create_random_graph(10, 0.3);
        let mut initial_state = ActivationState::new();
        
        // Multiple competing sources
        for i in 0..5 {
            initial_state.set_activation(NodeId(i), 0.2 + 0.1 * i as f32);
        }
        
        let result = spreader.spread_activation(&initial_state, &graph).unwrap();
        
        // Should produce valid result
        assert!(result.iterations > 0);
        ActivationTestUtils::assert_activation_normalized(&result.final_state);
        
        // Should have limited number of winners due to inhibition
        let active_count = result.final_state.activated_nodes().len();
        assert!(active_count <= 5); // At most original sources
    }
}
```

### Step 5: Performance Benchmarks
```rust
// File: tests/activation/performance_tests.rs

use super::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_activation_state_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activation_state");
    
    group.bench_function("set_activation_1k", |b| {
        let mut state = ActivationState::new();
        b.iter(|| {
            for i in 0..1000 {
                state.set_activation(NodeId(i), black_box(0.5));
            }
        });
    });
    
    group.bench_function("get_activation_1k", |b| {
        let mut state = ActivationState::new();
        for i in 0..1000 {
            state.set_activation(NodeId(i), 0.5);
        }
        
        b.iter(|| {
            for i in 0..1000 {
                black_box(state.get_activation(NodeId(i)));
            }
        });
    });
    
    group.finish();
}

fn bench_spreading_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("spreading");
    
    let spreader = ActivationSpreader::new();
    let graph = ActivationTestUtils::create_random_graph(1000, 0.01);
    let mut initial_state = ActivationState::new();
    initial_state.set_activation(NodeId(0), 1.0);
    
    group.bench_function("spread_1k_nodes", |b| {
        b.iter(|| {
            black_box(spreader.spread_activation(&initial_state, &graph))
        });
    });
    
    let large_graph = ActivationTestUtils::create_random_graph(10000, 0.001);
    group.bench_function("spread_10k_nodes", |b| {
        b.iter(|| {
            black_box(spreader.spread_activation(&initial_state, &large_graph))
        });
    });
    
    group.finish();
}

fn bench_convergence_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("convergence");
    
    group.bench_function("energy_detection", |b| {
        let mut detector = ConvergenceDetector::energy_based(0.001);
        let mut state = ActivationState::new();
        
        for i in 0..1000 {
            state.set_activation(NodeId(i), rand::random::<f32>());
        }
        
        b.iter(|| {
            detector.add_state(&state);
            black_box(detector.has_converged())
        });
    });
    
    group.finish();
}

fn bench_lateral_inhibition(c: &mut Criterion) {
    let mut group = c.benchmark_group("inhibition");
    
    let inhibition = LateralInhibition::global_winner_take_all(0.2);
    let graph = ActivationTestUtils::create_random_graph(1000, 0.01);
    
    group.bench_function("global_wta_1k", |b| {
        let mut state = ActivationState::new();
        for i in 0..1000 {
            state.set_activation(NodeId(i), rand::random::<f32>());
        }
        
        b.iter(|| {
            let mut test_state = state.clone();
            black_box(inhibition.apply_inhibition(&mut test_state, &graph))
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_activation_state_operations,
    bench_spreading_algorithms,
    bench_convergence_detection,
    bench_lateral_inhibition
);
criterion_main!(benches);
```

### Step 6: Stress Tests
```rust
// File: tests/activation/stress_tests.rs

use super::*;

#[cfg(test)]
mod stress_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    #[ignore] // Run with: cargo test stress_ -- --ignored
    fn stress_test_large_graph_spreading() {
        let spreader = ActivationSpreader::new();
        let graph = ActivationTestUtils::create_random_graph(100_000, 0.0001);
        
        let mut initial_state = ActivationState::new();
        for i in 0..100 {
            initial_state.set_activation(NodeId(i), 0.01);
        }
        
        let start = Instant::now();
        let result = spreader.spread_activation(&initial_state, &graph).unwrap();
        let elapsed = start.elapsed();
        
        println!("100k node spreading took: {:?}", elapsed);
        println!("Converged: {}, Iterations: {}", result.converged, result.iterations);
        
        assert!(elapsed.as_millis() < 1000); // Should complete in < 1 second
        ActivationTestUtils::assert_activation_normalized(&result.final_state);
    }
    
    #[test]
    #[ignore]
    fn stress_test_memory_usage() {
        let spreader = ActivationSpreader::new();
        
        // Test with incrementally larger graphs
        for size in &[1_000, 10_000, 100_000] {
            let graph = ActivationTestUtils::create_random_graph(*size, 0.001);
            
            let mut initial_state = ActivationState::new();
            initial_state.set_activation(NodeId(0), 1.0);
            
            let result = spreader.spread_activation(&initial_state, &graph).unwrap();
            
            // Verify memory usage is reasonable
            let active_nodes = result.final_state.activated_nodes().len();
            assert!(active_nodes < *size / 10); // Should be sparse
            
            println!("Graph size: {}, Active nodes: {}", size, active_nodes);
        }
    }
    
    #[test]
    #[ignore]
    fn stress_test_concurrent_spreading() {
        use std::sync::Arc;
        use std::thread;
        
        let spreader = Arc::new(ActivationSpreader::new());
        let graph = Arc::new(ActivationTestUtils::create_random_graph(1000, 0.01));
        
        let mut handles = vec![];
        
        // Spawn multiple threads doing spreading
        for i in 0..10 {
            let spreader_clone = Arc::clone(&spreader);
            let graph_clone = Arc::clone(&graph);
            
            let handle = thread::spawn(move || {
                let mut initial_state = ActivationState::new();
                initial_state.set_activation(NodeId(i), 1.0);
                
                let result = spreader_clone.spread_activation(&initial_state, &graph_clone).unwrap();
                (result.converged, result.iterations)
            });
            
            handles.push(handle);
        }
        
        // Collect results
        let results: Vec<_> = handles.into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        
        // All should complete successfully
        for (converged, iterations) in results {
            assert!(iterations > 0);
            println!("Converged: {}, Iterations: {}", converged, iterations);
        }
    }
    
    #[test]
    #[ignore]
    fn stress_test_numerical_stability() {
        let spreader = ActivationSpreader::new();
        let graph = ActivationTestUtils::create_ring_graph(100);
        
        // Test with various extreme initial conditions
        let test_cases = vec![
            vec![1e-10], // Very small activation
            vec![0.999999], // Very close to maximum
            vec![0.5; 50], // Many equal activations
            (0..100).map(|i| i as f32 / 100.0).collect(), // Gradient
        ];
        
        for initial_activations in test_cases {
            let mut initial_state = ActivationState::new();
            for (i, &activation) in initial_activations.iter().enumerate() {
                initial_state.set_activation(NodeId(i), activation);
            }
            
            let result = spreader.spread_activation(&initial_state, &graph).unwrap();
            
            // Verify numerical stability
            ActivationTestUtils::assert_activation_normalized(&result.final_state);
            
            // Check for NaN or infinity
            for node in result.final_state.activated_nodes() {
                let activation = result.final_state.get_activation(node);
                assert!(activation.is_finite(), "Non-finite activation detected: {}", activation);
            }
        }
    }
}
```

## File Locations

- `tests/activation/mod.rs` - Test infrastructure and utilities
- `tests/activation/state_tests.rs` - ActivationState unit tests
- `tests/activation/spreader_tests.rs` - Spreading algorithm tests
- `tests/activation/integration_tests.rs` - Component integration tests
- `tests/activation/performance_tests.rs` - Performance benchmarks
- `tests/activation/stress_tests.rs` - Stress and load tests

## Success Criteria

- [ ] All unit tests pass (100% success rate)
- [ ] Integration tests verify component interactions
- [ ] Property tests validate mathematical invariants
- [ ] Performance benchmarks meet targets:
  - State operations: < 100ns per operation
  - Spreading on 1k nodes: < 10ms
  - Convergence detection: < 1ms overhead
- [ ] Stress tests handle large graphs (100k+ nodes)
- [ ] Memory usage remains bounded under load
- [ ] All tests are deterministic and reproducible

## Test Requirements

```rust
// Example test runner command
#[test]
fn run_all_activation_tests() {
    // This test ensures all components work together
    let test_suite = ActivationTestSuite::new();
    
    // Unit tests
    test_suite.run_state_tests();
    test_suite.run_spreader_tests();
    test_suite.run_decay_tests();
    test_suite.run_inhibition_tests();
    test_suite.run_convergence_tests();
    
    // Integration tests
    test_suite.run_integration_tests();
    
    // Performance validation
    test_suite.validate_performance_targets();
    
    println!("All activation system tests passed!");
}

// Property-based test example
#[quickcheck]
fn prop_spreading_preserves_bounds(
    graph_size: usize,
    initial_activations: Vec<(usize, f32)>
) -> TestResult {
    if graph_size > 1000 || initial_activations.len() > 100 {
        return TestResult::discard();
    }
    
    let graph = ActivationTestUtils::create_random_graph(graph_size, 0.1);
    let spreader = ActivationSpreader::new();
    
    let mut initial_state = ActivationState::new();
    for (node_idx, activation) in initial_activations {
        if node_idx < graph_size {
            initial_state.set_activation(NodeId(node_idx), activation);
        }
    }
    
    if let Ok(result) = spreader.spread_activation(&initial_state, &graph) {
        // All activations should remain in [0,1]
        for node in result.final_state.activated_nodes() {
            let activation = result.final_state.get_activation(node);
            if activation < 0.0 || activation > 1.0 {
                return TestResult::failed();
            }
        }
        TestResult::passed()
    } else {
        TestResult::failed()
    }
}
```

## Performance Targets

```rust
// Performance benchmark thresholds
const PERFORMANCE_TARGETS: &[(&str, Duration)] = &[
    ("activation_state_set", Duration::from_nanos(100)),
    ("activation_state_get", Duration::from_nanos(50)),
    ("spreading_1k_nodes", Duration::from_millis(10)),
    ("spreading_10k_nodes", Duration::from_millis(100)),
    ("convergence_detection", Duration::from_millis(1)),
    ("global_wta_1k", Duration::from_millis(5)),
];

#[test]
fn validate_performance_targets() {
    for (test_name, target_duration) in PERFORMANCE_TARGETS {
        let actual_duration = run_performance_test(test_name);
        assert!(
            actual_duration <= *target_duration,
            "Performance target failed for {}: {:?} > {:?}",
            test_name, actual_duration, target_duration
        );
    }
}
```

## Quality Gates

- [ ] Code coverage > 95% for all activation components
- [ ] No memory leaks detected in stress tests
- [ ] All property tests pass with 1000+ generated cases
- [ ] Performance regression tests pass
- [ ] Deterministic behavior across multiple test runs
- [ ] Error handling coverage for all failure modes
- [ ] Thread safety verified for concurrent operations

## Next Task

Upon completion, proceed to **07_query_intent_types.md**