# Micro Task 01: Activation State Structure

**Priority**: CRITICAL  
**Estimated Time**: 45 minutes  
**Dependencies**: None  
**Skills Required**: Rust data structures, memory management

## Objective

Create the core data structures for managing activation states in the spreading activation engine.

## Context

The activation state is the fundamental data structure that tracks which nodes are activated and their activation levels throughout the spreading process. This forms the foundation for all subsequent activation operations.

## Specifications

### Required Data Structures

1. **ActivationState struct**
   - HashMap<NodeId, f32> for node activations (0.0 to 1.0)
   - Timestamp for temporal tracking
   - Total energy calculation
   - Efficient insertion/removal with threshold

2. **ActivationResult struct**
   - Final activation state
   - History of activation frames
   - Convergence information
   - Performance metrics

3. **ActivationFrame struct**
   - Snapshot of activation at one time step
   - Step number and timestamp
   - Change magnitude from previous frame

### Performance Requirements

- Get/set operations: O(1) average case
- Memory usage: < 8 bytes per activated node
- Support for 1M+ nodes with sparse activation
- Thread-safe for concurrent queries

## Implementation Guide

### Step 1: Create Core Types
```rust
// File: src/core/activation/state.rs

use std::collections::HashMap;
use std::time::Instant;
use crate::core::types::{NodeId, ActivationLevel};

#[derive(Debug, Clone)]
pub struct ActivationState {
    activations: HashMap<NodeId, f32>,
    timestamp: Instant,
    energy: f32,
    generation: usize,
}

impl ActivationState {
    pub fn new() -> Self {
        Self {
            activations: HashMap::new(),
            timestamp: Instant::now(),
            energy: 0.0,
            generation: 0,
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            activations: HashMap::with_capacity(capacity),
            timestamp: Instant::now(),
            energy: 0.0,
            generation: 0,
        }
    }
}
```

### Step 2: Implement Core Operations
```rust
impl ActivationState {
    pub fn set_activation(&mut self, node: NodeId, activation: f32) {
        let clamped = activation.clamp(0.0, 1.0);
        
        if clamped > 0.001 {
            self.activations.insert(node, clamped);
        } else {
            self.activations.remove(&node);
        }
        
        self.update_energy();
    }
    
    pub fn get_activation(&self, node: NodeId) -> f32 {
        self.activations.get(&node).copied().unwrap_or(0.0)
    }
    
    pub fn add_activation(&mut self, node: NodeId, delta: f32) {
        let current = self.get_activation(node);
        self.set_activation(node, current + delta);
    }
    
    fn update_energy(&mut self) {
        self.energy = self.activations.values()
            .map(|&a| a * a)
            .sum();
    }
}
```

### Step 3: Add Utility Methods
```rust
impl ActivationState {
    pub fn activated_nodes(&self) -> Vec<NodeId> {
        self.activations.keys().copied().collect()
    }
    
    pub fn total_activation(&self) -> f32 {
        self.activations.values().sum()
    }
    
    pub fn max_activation(&self) -> f32 {
        self.activations.values()
            .fold(0.0, |max, &a| max.max(a))
    }
    
    pub fn clear(&mut self) {
        self.activations.clear();
        self.energy = 0.0;
        self.generation += 1;
    }
}
```

### Step 4: Create Result Structures
```rust
#[derive(Debug, Clone)]
pub struct ActivationResult {
    pub final_state: ActivationState,
    pub history: Vec<ActivationFrame>,
    pub iterations: usize,
    pub converged: bool,
    pub processing_time: Duration,
}

#[derive(Debug, Clone)]
pub struct ActivationFrame {
    pub step: usize,
    pub state: ActivationState,
    pub change_magnitude: f32,
    pub timestamp: Instant,
}
```

## File Locations

- `src/core/activation/state.rs` - Main implementation
- `src/core/activation/mod.rs` - Module exports
- `tests/activation/state_tests.rs` - Test implementation

## Success Criteria

- [ ] ActivationState struct compiles and runs
- [ ] All core operations (get/set/add) work correctly
- [ ] Energy calculation accurate
- [ ] Memory usage efficient (verified with 100k nodes)
- [ ] Thread safety verified
- [ ] All tests pass:
  - Basic activation operations
  - Threshold-based cleanup
  - Energy conservation
  - Performance benchmarks

## Test Requirements

```rust
#[test]
fn test_basic_activation_operations() {
    let mut state = ActivationState::new();
    
    // Test set/get
    state.set_activation(NodeId(1), 0.5);
    assert_eq!(state.get_activation(NodeId(1)), 0.5);
    
    // Test add
    state.add_activation(NodeId(1), 0.3);
    assert_eq!(state.get_activation(NodeId(1)), 0.8);
    
    // Test clamping
    state.set_activation(NodeId(2), 1.5);
    assert_eq!(state.get_activation(NodeId(2)), 1.0);
}

#[test]
fn test_threshold_cleanup() {
    let mut state = ActivationState::new();
    
    state.set_activation(NodeId(1), 0.0005); // Below threshold
    assert_eq!(state.activated_nodes().len(), 0);
    
    state.set_activation(NodeId(2), 0.002); // Above threshold
    assert_eq!(state.activated_nodes().len(), 1);
}

#[test]
fn test_energy_calculation() {
    let mut state = ActivationState::new();
    
    state.set_activation(NodeId(1), 0.6);
    state.set_activation(NodeId(2), 0.8);
    
    let expected_energy = 0.6 * 0.6 + 0.8 * 0.8;
    assert!((state.energy - expected_energy).abs() < 0.001);
}
```

## Quality Gates

- [ ] Memory usage < 50MB for 1M nodes
- [ ] Get/set operations < 100ns average
- [ ] Thread safety verified with concurrent access
- [ ] Zero memory leaks under stress testing
- [ ] All edge cases handled (NaN, infinity, negative values)

## Next Task

Upon completion, proceed to **02_basic_spreader_algorithm.md**