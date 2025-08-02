# Micro Task 05: Convergence Detection

**Priority**: HIGH  
**Estimated Time**: 30 minutes  
**Dependencies**: 04_lateral_inhibition.md  
**Skills Required**: Mathematical analysis, signal processing, optimization theory

## Objective

Implement robust convergence detection mechanisms to determine when activation spreading has reached stable equilibrium.

## Context

Convergence detection is critical for system efficiency and correctness. It prevents unnecessary computation when the system has stabilized and provides guarantees about solution quality. Multiple detection strategies are needed for different convergence behaviors.

## Specifications

### Convergence Detection Types

1. **Energy-based Convergence** - Total system energy stabilization
2. **State-based Convergence** - Activation pattern stability  
3. **Gradient-based Convergence** - Rate of change analysis
4. **Oscillation Detection** - Cyclic behavior identification
5. **Adaptive Thresholds** - Context-sensitive convergence criteria

### Requirements
- Multiple convergence criteria with logical operators (AND/OR)
- Early termination when convergence detected
- Oscillation and cycling detection
- Performance monitoring and diagnostics
- Configurable sensitivity levels

## Implementation Guide

### Step 1: Define Convergence Types
```rust
// File: src/core/activation/convergence.rs

use std::collections::VecDeque;
use crate::core::{ActivationState, ActivationFrame};

#[derive(Debug, Clone, Copy)]
pub enum ConvergenceStrategy {
    Energy { threshold: f32 },
    StateChange { threshold: f32 },
    Gradient { window_size: usize, threshold: f32 },
    Combined { energy_threshold: f32, state_threshold: f32 },
}

pub struct ConvergenceDetector {
    strategy: ConvergenceStrategy,
    history_size: usize,
    oscillation_detection: bool,
    min_stable_iterations: usize,
    state_history: VecDeque<ActivationState>,
    energy_history: VecDeque<f32>,
    change_history: VecDeque<f32>,
}

impl ConvergenceDetector {
    pub fn energy_based(threshold: f32) -> Self {
        Self {
            strategy: ConvergenceStrategy::Energy { threshold },
            history_size: 10,
            oscillation_detection: true,
            min_stable_iterations: 3,
            state_history: VecDeque::new(),
            energy_history: VecDeque::new(),
            change_history: VecDeque::new(),
        }
    }
    
    pub fn state_change_based(threshold: f32) -> Self {
        Self {
            strategy: ConvergenceStrategy::StateChange { threshold },
            history_size: 10,
            oscillation_detection: true,
            min_stable_iterations: 3,
            state_history: VecDeque::new(),
            energy_history: VecDeque::new(),
            change_history: VecDeque::new(),
        }
    }
    
    pub fn gradient_based(window_size: usize, threshold: f32) -> Self {
        Self {
            strategy: ConvergenceStrategy::Gradient { window_size, threshold },
            history_size: window_size.max(10),
            oscillation_detection: true,
            min_stable_iterations: 3,
            state_history: VecDeque::new(),
            energy_history: VecDeque::new(),
            change_history: VecDeque::new(),
        }
    }
}
```

### Step 2: Implement Core Detection Logic
```rust
impl ConvergenceDetector {
    pub fn add_state(&mut self, state: &ActivationState) {
        // Add to history and maintain size limit
        self.state_history.push_back(state.clone());
        if self.state_history.len() > self.history_size {
            self.state_history.pop_front();
        }
        
        // Calculate and store energy
        let energy = self.calculate_energy(state);
        self.energy_history.push_back(energy);
        if self.energy_history.len() > self.history_size {
            self.energy_history.pop_front();
        }
        
        // Calculate state change if we have previous state
        if self.state_history.len() >= 2 {
            let prev_state = &self.state_history[self.state_history.len() - 2];
            let change = self.calculate_state_change(prev_state, state);
            self.change_history.push_back(change);
            if self.change_history.len() > self.history_size {
                self.change_history.pop_front();
            }
        }
    }
    
    pub fn has_converged(&self) -> ConvergenceResult {
        if self.state_history.len() < self.min_stable_iterations {
            return ConvergenceResult::continuing();
        }
        
        match self.strategy {
            ConvergenceStrategy::Energy { threshold } => {
                self.check_energy_convergence(threshold)
            }
            ConvergenceStrategy::StateChange { threshold } => {
                self.check_state_change_convergence(threshold)
            }
            ConvergenceStrategy::Gradient { window_size, threshold } => {
                self.check_gradient_convergence(window_size, threshold)
            }
            ConvergenceStrategy::Combined { energy_threshold, state_threshold } => {
                self.check_combined_convergence(energy_threshold, state_threshold)
            }
        }
    }
    
    fn calculate_energy(&self, state: &ActivationState) -> f32 {
        state.activated_nodes()
            .iter()
            .map(|&node| {
                let activation = state.get_activation(node);
                activation * activation
            })
            .sum()
    }
    
    fn calculate_state_change(&self, prev: &ActivationState, current: &ActivationState) -> f32 {
        let prev_nodes = prev.activated_nodes();
        let current_nodes = current.activated_nodes();
        
        // Get union of all nodes
        let mut all_nodes = std::collections::HashSet::new();
        all_nodes.extend(&prev_nodes);
        all_nodes.extend(&current_nodes);
        
        let mut total_change = 0.0;
        let mut total_magnitude = 0.0;
        
        for &node in &all_nodes {
            let prev_activation = prev.get_activation(node);
            let current_activation = current.get_activation(node);
            
            let change = (current_activation - prev_activation).abs();
            let magnitude = current_activation.max(prev_activation);
            
            total_change += change;
            total_magnitude += magnitude;
        }
        
        if total_magnitude > 0.0 {
            total_change / total_magnitude
        } else {
            0.0
        }
    }
}
```

### Step 3: Implement Specific Convergence Checks
```rust
impl ConvergenceDetector {
    fn check_energy_convergence(&self, threshold: f32) -> ConvergenceResult {
        if self.energy_history.len() < self.min_stable_iterations {
            return ConvergenceResult::continuing();
        }
        
        // Check if energy has stabilized
        let recent_energies = &self.energy_history.iter()
            .rev()
            .take(self.min_stable_iterations)
            .collect::<Vec<_>>();
        
        let min_energy = recent_energies.iter().fold(f32::INFINITY, |min, &&e| min.min(e));
        let max_energy = recent_energies.iter().fold(0.0, |max, &&e| max.max(e));
        let energy_range = max_energy - min_energy;
        
        if energy_range <= threshold {
            // Check for oscillations
            if self.oscillation_detection && self.detect_oscillation() {
                ConvergenceResult::oscillating(self.energy_history.len())
            } else {
                ConvergenceResult::converged(self.energy_history.len())
            }
        } else {
            ConvergenceResult::continuing()
        }
    }
    
    fn check_state_change_convergence(&self, threshold: f32) -> ConvergenceResult {
        if self.change_history.len() < self.min_stable_iterations {
            return ConvergenceResult::continuing();
        }
        
        // Check if recent changes are below threshold
        let recent_changes = self.change_history.iter()
            .rev()
            .take(self.min_stable_iterations);
        
        let all_below_threshold = recent_changes.all(|&change| change <= threshold);
        
        if all_below_threshold {
            if self.oscillation_detection && self.detect_oscillation() {
                ConvergenceResult::oscillating(self.change_history.len())
            } else {
                ConvergenceResult::converged(self.change_history.len())
            }
        } else {
            ConvergenceResult::continuing()
        }
    }
    
    fn check_gradient_convergence(&self, window_size: usize, threshold: f32) -> ConvergenceResult {
        if self.change_history.len() < window_size {
            return ConvergenceResult::continuing();
        }
        
        // Calculate gradient (trend) of recent changes
        let recent_changes: Vec<f32> = self.change_history.iter()
            .rev()
            .take(window_size)
            .copied()
            .collect();
        
        let gradient = self.calculate_linear_gradient(&recent_changes);
        
        if gradient.abs() <= threshold {
            if self.oscillation_detection && self.detect_oscillation() {
                ConvergenceResult::oscillating(self.change_history.len())
            } else {
                ConvergenceResult::converged(self.change_history.len())
            }
        } else {
            ConvergenceResult::continuing()
        }
    }
    
    fn check_combined_convergence(
        &self,
        energy_threshold: f32,
        state_threshold: f32,
    ) -> ConvergenceResult {
        let energy_result = self.check_energy_convergence(energy_threshold);
        let state_result = self.check_state_change_convergence(state_threshold);
        
        match (energy_result.status, state_result.status) {
            (ConvergenceStatus::Converged, ConvergenceStatus::Converged) => {
                ConvergenceResult::converged(energy_result.iterations.max(state_result.iterations))
            }
            (ConvergenceStatus::Oscillating, _) | (_, ConvergenceStatus::Oscillating) => {
                ConvergenceResult::oscillating(energy_result.iterations.max(state_result.iterations))
            }
            _ => ConvergenceResult::continuing()
        }
    }
}
```

### Step 4: Add Oscillation Detection
```rust
impl ConvergenceDetector {
    fn detect_oscillation(&self) -> bool {
        if self.energy_history.len() < 6 {
            return false;
        }
        
        // Look for repeating patterns in energy history
        let recent_energies: Vec<_> = self.energy_history.iter().rev().take(6).collect();
        
        // Check for 2-cycle oscillation (A-B-A-B pattern)
        if self.detect_2cycle(&recent_energies) {
            return true;
        }
        
        // Check for 3-cycle oscillation (A-B-C-A-B-C pattern)
        if recent_energies.len() >= 6 && self.detect_3cycle(&recent_energies) {
            return true;
        }
        
        false
    }
    
    fn detect_2cycle(&self, energies: &[&f32]) -> bool {
        if energies.len() < 4 {
            return false;
        }
        
        let tolerance = 0.01;
        
        // Check A-B-A-B pattern
        let e0_e2_diff = (energies[0] - energies[2]).abs();
        let e1_e3_diff = (energies[1] - energies[3]).abs();
        let e0_e1_diff = (energies[0] - energies[1]).abs();
        
        e0_e2_diff < tolerance && 
        e1_e3_diff < tolerance && 
        e0_e1_diff > tolerance  // Must actually be different values
    }
    
    fn detect_3cycle(&self, energies: &[&f32]) -> bool {
        if energies.len() < 6 {
            return false;
        }
        
        let tolerance = 0.01;
        
        // Check A-B-C-A-B-C pattern
        let cycle1_match = (energies[0] - energies[3]).abs() < tolerance &&
                          (energies[1] - energies[4]).abs() < tolerance &&
                          (energies[2] - energies[5]).abs() < tolerance;
        
        let values_different = (energies[0] - energies[1]).abs() > tolerance ||
                              (energies[1] - energies[2]).abs() > tolerance ||
                              (energies[0] - energies[2]).abs() > tolerance;
        
        cycle1_match && values_different
    }
    
    fn calculate_linear_gradient(&self, values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let n = values.len() as f32;
        let x_sum: f32 = (0..values.len()).map(|i| i as f32).sum();
        let y_sum: f32 = values.iter().sum();
        let xy_sum: f32 = values.iter().enumerate()
            .map(|(i, &y)| i as f32 * y)
            .sum();
        let x2_sum: f32 = (0..values.len()).map(|i| (i as f32).powi(2)).sum();
        
        let denominator = n * x2_sum - x_sum * x_sum;
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            (n * xy_sum - x_sum * y_sum) / denominator
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceStatus {
    Continuing,
    Converged,
    Oscillating,
}

#[derive(Debug, Clone)]
pub struct ConvergenceResult {
    pub status: ConvergenceStatus,
    pub iterations: usize,
    pub confidence: f32,
    pub diagnostics: ConvergenceDiagnostics,
}

#[derive(Debug, Clone)]
pub struct ConvergenceDiagnostics {
    pub final_energy: f32,
    pub final_change_rate: f32,
    pub oscillation_detected: bool,
    pub stability_score: f32,
}

impl ConvergenceResult {
    pub fn continuing() -> Self {
        Self {
            status: ConvergenceStatus::Continuing,
            iterations: 0,
            confidence: 0.0,
            diagnostics: ConvergenceDiagnostics::default(),
        }
    }
    
    pub fn converged(iterations: usize) -> Self {
        Self {
            status: ConvergenceStatus::Converged,
            iterations,
            confidence: 1.0,
            diagnostics: ConvergenceDiagnostics::default(),
        }
    }
    
    pub fn oscillating(iterations: usize) -> Self {
        Self {
            status: ConvergenceStatus::Oscillating,
            iterations,
            confidence: 0.8,
            diagnostics: ConvergenceDiagnostics {
                oscillation_detected: true,
                ..Default::default()
            },
        }
    }
}
```

### Step 5: Integration with Spreader
```rust
// Modify ActivationSpreader to use convergence detection

impl ActivationSpreader {
    pub fn with_convergence_detector(mut self, detector: ConvergenceDetector) -> Self {
        self.convergence_detector = Some(detector);
        self
    }
    
    pub fn spread_with_convergence_detection(
        &self,
        initial_state: &ActivationState,
        graph: &Graph,
    ) -> Result<ActivationResult> {
        let start_time = Instant::now();
        let mut state = initial_state.clone();
        let mut history = vec![state.clone()];
        let mut detector = self.convergence_detector.clone()
            .unwrap_or_else(|| ConvergenceDetector::energy_based(0.001));
        
        detector.add_state(&state);
        
        for iteration in 0..self.max_iterations {
            // Perform spreading step
            state = self.spread_step(&state, graph)?;
            history.push(state.clone());
            
            // Add state to convergence detector
            detector.add_state(&state);
            
            // Check for convergence
            let convergence_result = detector.has_converged();
            
            match convergence_result.status {
                ConvergenceStatus::Converged => {
                    return Ok(ActivationResult {
                        final_state: state,
                        history,
                        iterations: iteration + 1,
                        converged: true,
                        processing_time: start_time.elapsed(),
                        convergence_info: Some(convergence_result),
                    });
                }
                ConvergenceStatus::Oscillating => {
                    // Handle oscillation - could return best state or continue with damping
                    return Ok(ActivationResult {
                        final_state: state,
                        history,
                        iterations: iteration + 1,
                        converged: false,
                        processing_time: start_time.elapsed(),
                        convergence_info: Some(convergence_result),
                    });
                }
                ConvergenceStatus::Continuing => {
                    // Continue iteration
                    continue;
                }
            }
        }
        
        // Max iterations reached
        Ok(ActivationResult {
            final_state: state,
            history,
            iterations: self.max_iterations,
            converged: false,
            processing_time: start_time.elapsed(),
            convergence_info: Some(detector.has_converged()),
        })
    }
}
```

## File Locations

- `src/core/activation/convergence.rs` - Main implementation
- `src/core/activation/spreader.rs` - Integration points
- `tests/activation/convergence_tests.rs` - Test implementation

## Success Criteria

- [ ] All convergence strategies implemented correctly
- [ ] Oscillation detection working reliably
- [ ] Early termination when convergence detected
- [ ] Integration with spreader functional
- [ ] Performance acceptable (< 1ms detection overhead)
- [ ] All tests pass

## Test Requirements

```rust
#[test]
fn test_energy_convergence_detection() {
    let mut detector = ConvergenceDetector::energy_based(0.01);
    
    // Simulate converging energy sequence
    let energies = vec![1.0, 0.9, 0.82, 0.81, 0.805, 0.803, 0.802];
    
    for energy in energies {
        let mut state = ActivationState::new();
        state.set_activation(NodeId(1), energy.sqrt());
        detector.add_state(&state);
    }
    
    let result = detector.has_converged();
    assert_eq!(result.status, ConvergenceStatus::Converged);
}

#[test]
fn test_oscillation_detection() {
    let mut detector = ConvergenceDetector::energy_based(0.01);
    
    // Simulate 2-cycle oscillation: high-low-high-low
    let pattern = vec![0.8, 0.4, 0.8, 0.4, 0.8, 0.4];
    
    for energy in pattern {
        let mut state = ActivationState::new();
        state.set_activation(NodeId(1), energy.sqrt());
        detector.add_state(&state);
    }
    
    let result = detector.has_converged();
    assert_eq!(result.status, ConvergenceStatus::Oscillating);
    assert!(result.diagnostics.oscillation_detected);
}

#[test]
fn test_gradient_convergence() {
    let mut detector = ConvergenceDetector::gradient_based(5, 0.1);
    
    // Simulate gradually decreasing changes
    let changes = vec![0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01];
    
    for (i, &change) in changes.iter().enumerate() {
        let mut state = ActivationState::new();
        state.set_activation(NodeId(1), 1.0 - change);
        detector.add_state(&state);
        
        if i >= 4 {  // After window fills
            let result = detector.has_converged();
            if i == changes.len() - 1 {
                assert_eq!(result.status, ConvergenceStatus::Converged);
            }
        }
    }
}

#[test]
fn test_combined_convergence() {
    let mut detector = ConvergenceDetector::combined(0.01, 0.01);
    
    // Create states that satisfy both energy and state change criteria
    let mut prev_state = ActivationState::new();
    prev_state.set_activation(NodeId(1), 0.8);
    detector.add_state(&prev_state);
    
    // Add states with minimal changes
    for i in 0..5 {
        let mut state = ActivationState::new();
        state.set_activation(NodeId(1), 0.8 + 0.001 * (i as f32));
        detector.add_state(&state);
    }
    
    let result = detector.has_converged();
    assert_eq!(result.status, ConvergenceStatus::Converged);
}

#[test]
fn test_early_termination_integration() {
    let spreader = ActivationSpreader::new()
        .with_convergence_detector(ConvergenceDetector::energy_based(0.001));
    
    let graph = create_simple_chain_graph(5);
    let mut initial_state = ActivationState::new();
    initial_state.set_activation(NodeId(0), 1.0);
    
    let result = spreader.spread_with_convergence_detection(&initial_state, &graph).unwrap();
    
    assert!(result.converged);
    assert!(result.iterations < spreader.max_iterations);
    assert!(result.convergence_info.is_some());
}
```

## Performance Benchmarks

```rust
#[bench]
fn bench_convergence_detection_overhead(b: &mut Bencher) {
    let mut detector = ConvergenceDetector::energy_based(0.001);
    
    let mut state = ActivationState::new();
    for i in 0..1000 {
        state.set_activation(NodeId(i), rand::random::<f32>());
    }
    
    b.iter(|| {
        detector.add_state(&state);
        black_box(detector.has_converged())
    });
}
```

## Quality Gates

- [ ] Convergence detection is deterministic and repeatable
- [ ] No false positives in convergence detection
- [ ] Oscillation detection catches common patterns
- [ ] Memory usage bounded by history size
- [ ] Numerical stability with extreme activation values

## Next Task

Upon completion, proceed to **06_activation_tests.md**