# Micro Task 02: Basic Spreader Algorithm

**Priority**: CRITICAL  
**Estimated Time**: 60 minutes  
**Dependencies**: 01_activation_state_structure.md  
**Skills Required**: Graph algorithms, neural network concepts

## Objective

Implement the core spreading activation algorithm that propagates activation through the knowledge graph.

## Context

The spreader algorithm is the heart of the activation system. It takes an initial activation state and iteratively spreads activation through connected nodes until convergence, mimicking neural propagation.

## Specifications

### Core Algorithm Requirements

1. **ActivationSpreader struct**
   - Configurable spread factor (0.0-1.0)
   - Configurable threshold for activation cutoff
   - Maximum iteration limit
   - Support for different spreading strategies

2. **Spreading Mechanics**
   - Proportional spreading based on edge weights
   - Activation conservation with controlled decay
   - Parallel processing capability
   - Convergence detection

3. **Performance Targets**
   - < 10ms for 1M node graphs
   - Linear scaling with relevant subgraph size
   - Memory efficient (no unnecessary copies)

## Implementation Guide

### Step 1: Create Spreader Structure
```rust
// File: src/core/activation/spreader.rs

use std::collections::HashMap;
use crate::core::{Graph, NodeId, ActivationState, ActivationResult};

pub struct ActivationSpreader {
    spread_factor: f32,
    threshold: f32,
    max_iterations: usize,
    convergence_epsilon: f32,
}

impl ActivationSpreader {
    pub fn new() -> Self {
        Self {
            spread_factor: 0.8,
            threshold: 0.01,
            max_iterations: 100,
            convergence_epsilon: 0.001,
        }
    }
    
    pub fn with_config(
        spread_factor: f32,
        threshold: f32,
        max_iterations: usize,
    ) -> Self {
        Self {
            spread_factor,
            threshold,
            max_iterations,
            convergence_epsilon: 0.001,
        }
    }
}
```

### Step 2: Implement Core Spreading
```rust
impl ActivationSpreader {
    pub fn spread_activation(
        &self,
        initial_state: &ActivationState,
        graph: &Graph,
    ) -> Result<ActivationResult> {
        let start_time = Instant::now();
        let mut state = initial_state.clone();
        let mut history = vec![state.clone()];
        
        for iteration in 0..self.max_iterations {
            // Perform one spreading step
            let new_state = self.spread_step(&state, graph)?;
            
            // Check for convergence
            let change = self.calculate_state_change(&state, &new_state);
            
            state = new_state;
            history.push(state.clone());
            
            if change < self.convergence_epsilon {
                return Ok(ActivationResult {
                    final_state: state,
                    history,
                    iterations: iteration + 1,
                    converged: true,
                    processing_time: start_time.elapsed(),
                });
            }
        }
        
        // Max iterations reached without convergence
        Ok(ActivationResult {
            final_state: state,
            history,
            iterations: self.max_iterations,
            converged: false,
            processing_time: start_time.elapsed(),
        })
    }
}
```

### Step 3: Implement Single Spread Step
```rust
impl ActivationSpreader {
    pub fn spread_step(
        &self,
        current_state: &ActivationState,
        graph: &Graph,
    ) -> Result<ActivationState> {
        let mut new_state = ActivationState::new();
        
        // Collect all current activations
        let current_activations = current_state.all_activations();
        
        // Spread from each activated node
        for (&source_node, &source_activation) in current_activations {
            if source_activation > self.threshold {
                self.spread_from_node(
                    source_node,
                    source_activation,
                    graph,
                    &mut new_state,
                )?;
            }
        }
        
        Ok(new_state)
    }
    
    fn spread_from_node(
        &self,
        source: NodeId,
        activation: f32,
        graph: &Graph,
        target_state: &mut ActivationState,
    ) -> Result<()> {
        // Get neighbors and their edge weights
        let neighbors = graph.neighbors(source)?;
        let total_outgoing_weight: f32 = neighbors.iter()
            .map(|(_, weight)| *weight)
            .sum();
        
        if total_outgoing_weight > 0.0 {
            // Spread activation proportionally
            for (neighbor, edge_weight) in neighbors {
                let spread_proportion = edge_weight / total_outgoing_weight;
                let spread_amount = activation * spread_proportion * self.spread_factor;
                
                target_state.add_activation(neighbor, spread_amount);
            }
        }
        
        // Source retains some activation (with decay)
        let retained_activation = activation * (1.0 - self.spread_factor);
        target_state.set_activation(source, retained_activation);
        
        Ok(())
    }
}
```

### Step 4: Add Convergence Detection
```rust
impl ActivationSpreader {
    fn calculate_state_change(
        &self,
        old_state: &ActivationState,
        new_state: &ActivationState,
    ) -> f32 {
        let old_nodes = old_state.activated_nodes();
        let new_nodes = new_state.activated_nodes();
        
        // Combine all nodes that have activation in either state
        let all_nodes: HashSet<_> = old_nodes.iter()
            .chain(new_nodes.iter())
            .copied()
            .collect();
        
        let mut total_change = 0.0;
        let mut total_activation = 0.0;
        
        for node in all_nodes {
            let old_activation = old_state.get_activation(node);
            let new_activation = new_state.get_activation(node);
            let change = (new_activation - old_activation).abs();
            
            total_change += change;
            total_activation += new_activation.max(old_activation);
        }
        
        if total_activation > 0.0 {
            total_change / total_activation // Relative change
        } else {
            0.0
        }
    }
    
    pub fn has_converged(
        &self,
        current_state: &ActivationState,
        previous_state: &ActivationState,
    ) -> bool {
        self.calculate_state_change(previous_state, current_state) 
            < self.convergence_epsilon
    }
}
```

### Step 5: Add Parallel Support
```rust
use rayon::prelude::*;

impl ActivationSpreader {
    pub fn spread_step_parallel(
        &self,
        current_state: &ActivationState,
        graph: &Graph,
    ) -> Result<ActivationState> {
        let current_activations = current_state.all_activations();
        let active_nodes: Vec<_> = current_activations
            .iter()
            .filter(|(_, &activation)| activation > self.threshold)
            .collect();
        
        // Process nodes in parallel
        let partial_states: Vec<_> = active_nodes
            .par_iter()
            .map(|(&source_node, &source_activation)| {
                let mut partial_state = ActivationState::new();
                self.spread_from_node(
                    source_node,
                    source_activation,
                    graph,
                    &mut partial_state,
                ).ok();
                partial_state
            })
            .collect();
        
        // Merge partial states
        let mut final_state = ActivationState::new();
        for partial in partial_states {
            for node in partial.activated_nodes() {
                let activation = partial.get_activation(node);
                final_state.add_activation(node, activation);
            }
        }
        
        Ok(final_state)
    }
}
```

## File Locations

- `src/core/activation/spreader.rs` - Main implementation
- `src/core/activation/mod.rs` - Module exports
- `tests/activation/spreader_tests.rs` - Test implementation

## Success Criteria

- [ ] Basic spreading algorithm works correctly
- [ ] Activation is conserved (with controlled decay)
- [ ] Convergence detection functional
- [ ] Performance target met (< 10ms for 1M nodes)
- [ ] Parallel processing works correctly
- [ ] All tests pass

## Test Requirements

```rust
#[test]
fn test_basic_activation_spreading() {
    let spreader = ActivationSpreader::new();
    let graph = create_simple_graph(); // A -> B -> C
    
    let mut initial_state = ActivationState::new();
    initial_state.set_activation(NodeId(0), 1.0); // Node A
    
    let result = spreader.spread_activation(&initial_state, &graph).unwrap();
    
    // Verify spreading occurred
    assert!(result.final_state.get_activation(NodeId(1)) > 0.0); // Node B
    assert!(result.final_state.get_activation(NodeId(2)) > 0.0); // Node C
    
    // Verify convergence
    assert!(result.converged);
    assert!(result.iterations < 50);
}

#[test]
fn test_activation_conservation() {
    let spreader = ActivationSpreader::with_config(0.9, 0.001, 100);
    let graph = create_ring_graph(10);
    
    let mut initial_state = ActivationState::new();
    initial_state.set_activation(NodeId(0), 1.0);
    
    let result = spreader.spread_activation(&initial_state, &graph).unwrap();
    
    let initial_total = initial_state.total_activation();
    let final_total = result.final_state.total_activation();
    
    // Should have decay but not complete loss
    assert!(final_total < initial_total);
    assert!(final_total > initial_total * 0.5);
}

#[test]
fn test_convergence_detection() {
    let spreader = ActivationSpreader::new();
    let graph = create_star_graph(5); // Central node with 5 spokes
    
    let mut initial_state = ActivationState::new();
    initial_state.set_activation(NodeId(0), 1.0); // Center
    
    let result = spreader.spread_activation(&initial_state, &graph).unwrap();
    
    assert!(result.converged);
    assert!(result.iterations < spreader.max_iterations);
}

#[test]
fn test_parallel_spreading() {
    let spreader = ActivationSpreader::new();
    let graph = create_large_random_graph(1000);
    
    let mut initial_state = ActivationState::new();
    for i in 0..10 {
        initial_state.set_activation(NodeId(i), 0.1);
    }
    
    let start = Instant::now();
    let result = spreader.spread_activation(&initial_state, &graph).unwrap();
    let elapsed = start.elapsed();
    
    assert!(elapsed < Duration::from_millis(10));
    assert!(result.final_state.activated_nodes().len() > 10);
}
```

## Performance Benchmarks

```rust
#[bench]
fn bench_spreading_1m_nodes(b: &mut Bencher) {
    let spreader = ActivationSpreader::new();
    let graph = create_scale_free_graph(1_000_000);
    
    let mut initial_state = ActivationState::new();
    initial_state.set_activation(NodeId(0), 1.0);
    
    b.iter(|| {
        black_box(spreader.spread_activation(&initial_state, &graph))
    });
}
```

## Quality Gates

- [ ] No memory leaks during long-running spreads
- [ ] Numerical stability (no NaN or infinity values)
- [ ] Deterministic results (same inputs = same outputs)
- [ ] Graceful handling of disconnected graphs
- [ ] Proper error handling for invalid inputs

## Next Task

Upon completion, proceed to **03_decay_mechanisms.md**