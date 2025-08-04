# Micro Task 04: Lateral Inhibition

**Priority**: CRITICAL  
**Estimated Time**: 35 minutes  
**Dependencies**: 03_decay_mechanisms.md  
**Skills Required**: Neural network algorithms, competitive dynamics

## Objective

Implement lateral inhibition mechanisms for winner-take-all competition between activated nodes.

## Context

Lateral inhibition is a fundamental neural computation that enhances contrast and implements competitive selection. It suppresses weakly activated nodes while strengthening highly activated ones, leading to sparse, focused activation patterns.

## Specifications

### Core Inhibition Mechanisms

1. **LateralInhibition struct**
   - Winner-take-all competition within neighborhoods
   - Configurable inhibition strength and radius
   - Support for different competition strategies
   - Integration with activation spreading

2. **Competition Strategies**
   - Global winner-take-all (single winner)
   - K-winners-take-all (top K nodes)
   - Neighborhood-based competition
   - Adaptive threshold selection

3. **Performance Requirements**
   - < 5ms for 100k node competitions
   - Stable convergence behavior
   - Configurable aggression levels

## Implementation Guide

### Step 1: Define Inhibition Types
```rust
// File: src/core/activation/inhibition.rs

use std::collections::{HashMap, HashSet};
use crate::core::{NodeId, ActivationState, Graph};

#[derive(Debug, Clone, Copy)]
pub enum InhibitionStrategy {
    GlobalWTA,                          // Single global winner
    KWinnersWTA { k: usize },          // Top K winners
    LocalWTA { radius: usize },        // Winners per neighborhood
    ThresholdBased { threshold: f32 }, // Above-threshold competition
}

pub struct LateralInhibition {
    strategy: InhibitionStrategy,
    inhibition_strength: f32,
    convergence_iterations: usize,
    min_activation_diff: f32,
}

impl LateralInhibition {
    pub fn global_winner_take_all(strength: f32) -> Self {
        Self {
            strategy: InhibitionStrategy::GlobalWTA,
            inhibition_strength: strength,
            convergence_iterations: 10,
            min_activation_diff: 0.01,
        }
    }
    
    pub fn k_winners_take_all(k: usize, strength: f32) -> Self {
        Self {
            strategy: InhibitionStrategy::KWinnersWTA { k },
            inhibition_strength: strength,
            convergence_iterations: 10,
            min_activation_diff: 0.01,
        }
    }
    
    pub fn local_competition(radius: usize, strength: f32) -> Self {
        Self {
            strategy: InhibitionStrategy::LocalWTA { radius },
            inhibition_strength: strength,
            convergence_iterations: 10,
            min_activation_diff: 0.01,
        }
    }
}
```

### Step 2: Implement Core Competition Logic
```rust
impl LateralInhibition {
    pub fn apply_inhibition(
        &self,
        state: &mut ActivationState,
        graph: &Graph,
    ) -> Result<bool> {
        match self.strategy {
            InhibitionStrategy::GlobalWTA => {
                self.apply_global_wta(state)
            }
            InhibitionStrategy::KWinnersWTA { k } => {
                self.apply_k_winners_wta(state, k)
            }
            InhibitionStrategy::LocalWTA { radius } => {
                self.apply_local_wta(state, graph, radius)
            }
            InhibitionStrategy::ThresholdBased { threshold } => {
                self.apply_threshold_competition(state, threshold)
            }
        }
    }
    
    fn apply_global_wta(&self, state: &mut ActivationState) -> Result<bool> {
        let activated_nodes = state.activated_nodes();
        if activated_nodes.is_empty() {
            return Ok(false);
        }
        
        // Find global winner
        let winner = activated_nodes.iter()
            .max_by(|&&a, &&b| {
                state.get_activation(a)
                    .partial_cmp(&state.get_activation(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied();
        
        if let Some(winner_node) = winner {
            let winner_activation = state.get_activation(winner_node);
            
            // Clear all other activations
            for &node in &activated_nodes {
                if node != winner_node {
                    state.set_activation(node, 0.0);
                }
            }
            
            // Strengthen winner
            let enhanced_activation = (winner_activation * (1.0 + self.inhibition_strength))
                .min(1.0);
            state.set_activation(winner_node, enhanced_activation);
            
            Ok(true)
        } else {
            Ok(false)
        }
    }
}
```

### Step 3: Implement K-Winners Competition
```rust
impl LateralInhibition {
    fn apply_k_winners_wta(
        &self,
        state: &mut ActivationState,
        k: usize,
    ) -> Result<bool> {
        let mut node_activations: Vec<_> = state.activated_nodes()
            .into_iter()
            .map(|node| (node, state.get_activation(node)))
            .collect();
        
        if node_activations.is_empty() {
            return Ok(false);
        }
        
        // Sort by activation (descending)
        node_activations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Determine winners and losers
        let num_winners = k.min(node_activations.len());
        let winners: HashSet<_> = node_activations[..num_winners]
            .iter()
            .map(|(node, _)| *node)
            .collect();
        
        // Apply inhibition
        for (node, current_activation) in node_activations {
            if winners.contains(&node) {
                // Enhance winners
                let enhanced = (current_activation * (1.0 + self.inhibition_strength))
                    .min(1.0);
                state.set_activation(node, enhanced);
            } else {
                // Suppress losers
                let suppressed = current_activation * (1.0 - self.inhibition_strength);
                state.set_activation(node, suppressed);
            }
        }
        
        Ok(true)
    }
    
    fn apply_local_wta(
        &self,
        state: &mut ActivationState,
        graph: &Graph,
        radius: usize,
    ) -> Result<bool> {
        let activated_nodes = state.activated_nodes();
        let mut processed: HashSet<NodeId> = HashSet::new();
        let mut any_changes = false;
        
        for &center_node in &activated_nodes {
            if processed.contains(&center_node) {
                continue;
            }
            
            // Get neighborhood
            let neighborhood = self.get_neighborhood(center_node, graph, radius)?;
            let local_nodes: Vec<_> = neighborhood.intersection(
                &activated_nodes.iter().copied().collect()
            ).copied().collect();
            
            if local_nodes.len() <= 1 {
                processed.extend(&neighborhood);
                continue;
            }
            
            // Find local winner
            let local_winner = local_nodes.iter()
                .max_by(|&&a, &&b| {
                    state.get_activation(a)
                        .partial_cmp(&state.get_activation(b))
                        .unwrap()
                })
                .copied();
            
            if let Some(winner) = local_winner {
                let winner_activation = state.get_activation(winner);
                
                // Suppress other nodes in neighborhood
                for &node in &local_nodes {
                    if node != winner {
                        let current = state.get_activation(node);
                        let suppressed = current * (1.0 - self.inhibition_strength);
                        state.set_activation(node, suppressed);
                        any_changes = true;
                    }
                }
                
                // Enhance winner
                let enhanced = (winner_activation * (1.0 + self.inhibition_strength))
                    .min(1.0);
                state.set_activation(winner, enhanced);
                
                processed.extend(&neighborhood);
            }
        }
        
        Ok(any_changes)
    }
    
    fn get_neighborhood(
        &self,
        center: NodeId,
        graph: &Graph,
        radius: usize,
    ) -> Result<HashSet<NodeId>> {
        let mut neighborhood = HashSet::new();
        let mut current_layer = vec![center];
        neighborhood.insert(center);
        
        for _ in 0..radius {
            let mut next_layer = Vec::new();
            
            for &node in &current_layer {
                let neighbors = graph.neighbors(node)?;
                for (neighbor, _) in neighbors {
                    if neighborhood.insert(neighbor) {
                        next_layer.push(neighbor);
                    }
                }
            }
            
            if next_layer.is_empty() {
                break;
            }
            current_layer = next_layer;
        }
        
        Ok(neighborhood)
    }
}
```

### Step 4: Add Iterative Competition
```rust
impl LateralInhibition {
    pub fn apply_iterative_inhibition(
        &self,
        state: &mut ActivationState,
        graph: &Graph,
    ) -> Result<InhibitionResult> {
        let mut iteration = 0;
        let mut converged = false;
        let initial_state = state.clone();
        
        while iteration < self.convergence_iterations && !converged {
            let prev_state = state.clone();
            
            // Apply one round of inhibition
            self.apply_inhibition(state, graph)?;
            
            // Check convergence
            converged = self.has_converged(&prev_state, state);
            iteration += 1;
        }
        
        Ok(InhibitionResult {
            initial_state,
            final_state: state.clone(),
            iterations: iteration,
            converged,
        })
    }
    
    fn has_converged(
        &self,
        prev_state: &ActivationState,
        current_state: &ActivationState,
    ) -> bool {
        let prev_nodes = prev_state.activated_nodes();
        let current_nodes = current_state.activated_nodes();
        
        // Check if same nodes are activated
        let prev_set: HashSet<_> = prev_nodes.iter().copied().collect();
        let current_set: HashSet<_> = current_nodes.iter().copied().collect();
        
        if prev_set != current_set {
            return false;
        }
        
        // Check activation differences
        for &node in &current_nodes {
            let prev_activation = prev_state.get_activation(node);
            let current_activation = current_state.get_activation(node);
            let diff = (current_activation - prev_activation).abs();
            
            if diff > self.min_activation_diff {
                return false;
            }
        }
        
        true
    }
}

#[derive(Debug, Clone)]
pub struct InhibitionResult {
    pub initial_state: ActivationState,
    pub final_state: ActivationState,
    pub iterations: usize,
    pub converged: bool,
}
```

### Step 5: Integration with Spreader
```rust
// Modify ActivationSpreader to include inhibition

impl ActivationSpreader {
    pub fn with_inhibition(mut self, inhibition: LateralInhibition) -> Self {
        self.inhibition = Some(inhibition);
        self
    }
    
    pub fn spread_with_inhibition(
        &self,
        initial_state: &ActivationState,
        graph: &Graph,
    ) -> Result<ActivationResult> {
        let start_time = Instant::now();
        let mut state = initial_state.clone();
        let mut history = vec![state.clone()];
        
        for iteration in 0..self.max_iterations {
            // Spreading step
            let new_state = self.spread_step(&state, graph)?;
            
            // Apply inhibition if configured
            let mut final_state = new_state;
            if let Some(ref inhibition) = self.inhibition {
                inhibition.apply_inhibition(&mut final_state, graph)?;
            }
            
            // Check convergence
            let change = self.calculate_state_change(&state, &final_state);
            
            state = final_state;
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

## File Locations

- `src/core/activation/inhibition.rs` - Main implementation
- `src/core/activation/spreader.rs` - Integration points
- `tests/activation/inhibition_tests.rs` - Test implementation

## Success Criteria

- [ ] All inhibition strategies implemented correctly
- [ ] Winner-take-all behavior verified
- [ ] Local competition working
- [ ] Integration with spreader functional
- [ ] Performance targets met (< 5ms for 100k nodes)
- [ ] All tests pass

## Test Requirements

```rust
#[test]
fn test_global_winner_take_all() {
    let inhibition = LateralInhibition::global_winner_take_all(0.2);
    
    let mut state = ActivationState::new();
    state.set_activation(NodeId(1), 0.8);
    state.set_activation(NodeId(2), 0.6);
    state.set_activation(NodeId(3), 0.4);
    
    let graph = create_empty_graph();
    inhibition.apply_inhibition(&mut state, &graph).unwrap();
    
    // Only highest activation should remain (enhanced)
    assert!(state.get_activation(NodeId(1)) > 0.8);
    assert_eq!(state.get_activation(NodeId(2)), 0.0);
    assert_eq!(state.get_activation(NodeId(3)), 0.0);
}

#[test]
fn test_k_winners_take_all() {
    let inhibition = LateralInhibition::k_winners_take_all(2, 0.3);
    
    let mut state = ActivationState::new();
    state.set_activation(NodeId(1), 0.9);
    state.set_activation(NodeId(2), 0.7);
    state.set_activation(NodeId(3), 0.5);
    state.set_activation(NodeId(4), 0.3);
    
    let graph = create_empty_graph();
    inhibition.apply_inhibition(&mut state, &graph).unwrap();
    
    // Top 2 should be enhanced, others suppressed
    assert!(state.get_activation(NodeId(1)) > 0.9);
    assert!(state.get_activation(NodeId(2)) > 0.7);
    assert!(state.get_activation(NodeId(3)) < 0.5);
    assert!(state.get_activation(NodeId(4)) < 0.3);
}

#[test]
fn test_local_competition() {
    let inhibition = LateralInhibition::local_competition(1, 0.4);
    
    // Create graph: 1-2-3  4-5 (two separate components)
    let graph = create_linear_graph();
    
    let mut state = ActivationState::new();
    state.set_activation(NodeId(1), 0.8);  // Winner in first component
    state.set_activation(NodeId(2), 0.6);
    state.set_activation(NodeId(4), 0.7);  // Winner in second component
    state.set_activation(NodeId(5), 0.5);
    
    inhibition.apply_inhibition(&mut state, &graph).unwrap();
    
    // Should have one winner per component
    assert!(state.get_activation(NodeId(1)) > 0.8);
    assert!(state.get_activation(NodeId(2)) < 0.6);
    assert!(state.get_activation(NodeId(4)) > 0.7);
    assert!(state.get_activation(NodeId(5)) < 0.5);
}

#[test]
fn test_inhibition_convergence() {
    let inhibition = LateralInhibition::global_winner_take_all(0.1);
    
    let mut state = ActivationState::new();
    state.set_activation(NodeId(1), 0.6);
    state.set_activation(NodeId(2), 0.55);  // Very close competition
    
    let graph = create_empty_graph();
    let result = inhibition.apply_iterative_inhibition(&mut state, &graph).unwrap();
    
    assert!(result.converged);
    assert!(result.iterations > 1);  // Should take multiple iterations
    
    // Should still have clear winner
    let final_activations: Vec<_> = result.final_state.activated_nodes()
        .into_iter()
        .map(|node| result.final_state.get_activation(node))
        .collect();
    assert_eq!(final_activations.len(), 1);
}
```

## Performance Benchmarks

```rust
#[bench]
fn bench_global_wta_100k_nodes(b: &mut Bencher) {
    let inhibition = LateralInhibition::global_winner_take_all(0.2);
    let graph = create_empty_graph();
    
    let mut state = ActivationState::new();
    for i in 0..100_000 {
        state.set_activation(NodeId(i), rand::random::<f32>());
    }
    
    b.iter(|| {
        let mut test_state = state.clone();
        black_box(inhibition.apply_inhibition(&mut test_state, &graph))
    });
}
```

## Quality Gates

- [ ] Deterministic winner selection with identical inputs
- [ ] Numerical stability under repeated applications
- [ ] Memory usage remains constant during operation
- [ ] Graceful handling of empty activation states
- [ ] No infinite loops in iterative competition

## Next Task

Upon completion, proceed to **05_convergence_detection.md**