# Micro Task 03: Decay Mechanisms

**Priority**: HIGH  
**Estimated Time**: 30 minutes  
**Dependencies**: 02_basic_spreader_algorithm.md  
**Skills Required**: Mathematical functions, biological modeling

## Objective

Implement various activation decay functions that simulate neural fatigue and prevent unlimited activation accumulation.

## Context

Decay mechanisms are crucial for system stability and biological realism. They prevent runaway activation, implement forgetting, and allow the system to reach equilibrium states.

## Specifications

### Decay Function Types

1. **Exponential Decay** - Biological standard (e^(-t/τ))
2. **Linear Decay** - Simple computational model
3. **Sigmoid Decay** - Smooth cutoff behavior
4. **Custom Functions** - User-defined decay patterns

### Requirements
- Time-based and iteration-based decay options
- Configurable decay rates per node type
- Efficient batch processing for large activation sets
- Integration with spreading algorithm

## Implementation Guide

### Step 1: Define Decay Types
```rust
// File: src/core/activation/decay.rs

use std::time::Duration;

#[derive(Debug, Clone, Copy)]
pub enum DecayType {
    Exponential { time_constant: f32 },
    Linear { rate: f32 },
    Sigmoid { steepness: f32, midpoint: f32 },
    Custom(fn(f32, f32) -> f32),
}

pub struct DecayFunction {
    pub decay_type: DecayType,
    pub min_threshold: f32,
}

impl DecayFunction {
    pub fn exponential(time_constant: f32) -> Self {
        Self {
            decay_type: DecayType::Exponential { time_constant },
            min_threshold: 0.001,
        }
    }
    
    pub fn linear(rate: f32) -> Self {
        Self {
            decay_type: DecayType::Linear { rate },
            min_threshold: 0.001,
        }
    }
}
```

### Step 2: Implement Decay Application
```rust
impl DecayFunction {
    pub fn apply(&self, activation: f32, time_elapsed: f32) -> f32 {
        let decayed = match self.decay_type {
            DecayType::Exponential { time_constant } => {
                activation * (-time_elapsed / time_constant).exp()
            }
            DecayType::Linear { rate } => {
                (activation - rate * time_elapsed).max(0.0)
            }
            DecayType::Sigmoid { steepness, midpoint } => {
                let x = time_elapsed - midpoint;
                activation / (1.0 + (steepness * x).exp())
            }
            DecayType::Custom(func) => {
                func(activation, time_elapsed)
            }
        };
        
        if decayed < self.min_threshold {
            0.0
        } else {
            decayed
        }
    }
    
    pub fn apply_to_state(
        &self,
        state: &mut ActivationState,
        time_elapsed: f32,
    ) {
        let nodes: Vec<_> = state.activated_nodes();
        
        for node in nodes {
            let current = state.get_activation(node);
            let decayed = self.apply(current, time_elapsed);
            state.set_activation(node, decayed);
        }
    }
}
```

### Step 3: Add Adaptive Decay
```rust
pub struct AdaptiveDecayManager {
    node_decay_rates: HashMap<NodeId, f32>,
    global_decay: DecayFunction,
    usage_tracking: HashMap<NodeId, Duration>,
}

impl AdaptiveDecayManager {
    pub fn new(global_decay: DecayFunction) -> Self {
        Self {
            node_decay_rates: HashMap::new(),
            global_decay,
            usage_tracking: HashMap::new(),
        }
    }
    
    pub fn set_node_decay_rate(&mut self, node: NodeId, rate: f32) {
        self.node_decay_rates.insert(node, rate);
    }
    
    pub fn apply_decay(&mut self, state: &mut ActivationState) {
        let current_time = Instant::now();
        let nodes: Vec<_> = state.activated_nodes();
        
        for node in nodes {
            let activation = state.get_activation(node);
            
            // Get last usage time
            let last_used = self.usage_tracking
                .get(&node)
                .copied()
                .unwrap_or(current_time);
            
            let time_elapsed = current_time
                .duration_since(last_used)
                .as_secs_f32();
            
            // Apply node-specific or global decay
            let decay_rate = self.node_decay_rates
                .get(&node)
                .copied()
                .unwrap_or(1.0);
            
            let effective_time = time_elapsed * decay_rate;
            let decayed = self.global_decay.apply(activation, effective_time);
            
            state.set_activation(node, decayed);
            
            // Update usage tracking
            if decayed > 0.0 {
                self.usage_tracking.insert(node, current_time);
            }
        }
    }
}
```

### Step 4: Integration with Spreader
```rust
// Modify ActivationSpreader to include decay

impl ActivationSpreader {
    pub fn with_decay(mut self, decay_function: DecayFunction) -> Self {
        self.decay_function = Some(decay_function);
        self
    }
    
    pub fn spread_step_with_decay(
        &self,
        current_state: &ActivationState,
        graph: &Graph,
        time_step: f32,
    ) -> Result<ActivationState> {
        // Perform spreading
        let mut new_state = self.spread_step(current_state, graph)?;
        
        // Apply decay if configured
        if let Some(ref decay_fn) = self.decay_function {
            decay_fn.apply_to_state(&mut new_state, time_step);
        }
        
        Ok(new_state)
    }
}
```

## File Locations

- `src/core/activation/decay.rs` - Main implementation
- `src/core/activation/spreader.rs` - Integration points
- `tests/activation/decay_tests.rs` - Test implementation

## Success Criteria

- [ ] All decay types implemented correctly
- [ ] Mathematical accuracy verified
- [ ] Integration with spreader functional
- [ ] Adaptive decay working
- [ ] Performance acceptable (< 1ms for 10k nodes)
- [ ] All tests pass

## Test Requirements

```rust
#[test]
fn test_exponential_decay() {
    let decay = DecayFunction::exponential(1.0);
    
    let initial = 1.0;
    let after_1_tc = decay.apply(initial, 1.0);
    let after_2_tc = decay.apply(initial, 2.0);
    
    assert!((after_1_tc - (1.0 / std::f32::consts::E)).abs() < 0.01);
    assert!((after_2_tc - (1.0 / std::f32::consts::E.powi(2))).abs() < 0.01);
}

#[test]
fn test_linear_decay() {
    let decay = DecayFunction::linear(0.1);
    
    assert_eq!(decay.apply(1.0, 5.0), 0.5);
    assert_eq!(decay.apply(1.0, 10.0), 0.0);
    assert_eq!(decay.apply(0.5, 6.0), 0.0); // Clips to 0
}

#[test]
fn test_adaptive_decay() {
    let mut manager = AdaptiveDecayManager::new(
        DecayFunction::exponential(1.0)
    );
    
    // Set different decay rates
    manager.set_node_decay_rate(NodeId(1), 2.0); // Faster decay
    manager.set_node_decay_rate(NodeId(2), 0.5); // Slower decay
    
    let mut state = ActivationState::new();
    state.set_activation(NodeId(1), 1.0);
    state.set_activation(NodeId(2), 1.0);
    
    manager.apply_decay(&mut state);
    
    // Fast decay node should have lower activation
    assert!(state.get_activation(NodeId(1)) < state.get_activation(NodeId(2)));
}
```

## Quality Gates

- [ ] Numerical stability (no overflow/underflow)
- [ ] Monotonic decay (never increases activation)
- [ ] Efficient batch processing
- [ ] Memory usage remains constant

## Next Task

Upon completion, proceed to **04_lateral_inhibition.md**