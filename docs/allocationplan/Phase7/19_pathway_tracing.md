# Micro Task 19: Pathway Tracing

**Priority**: CRITICAL  
**Estimated Time**: 50 minutes  
**Dependencies**: 18_attention_tests.md completed  
**Skills Required**: Graph traversal, path analysis, brain-inspired learning systems

## Objective

Implement activation pathway tracing to identify and record neural-like pathways through the knowledge graph during spreading activation, enabling pathway-based learning and reinforcement.

## Context

In biological neural networks, learning occurs through the strengthening of specific pathways that fire together. This task implements pathway tracing to capture activation routes during spreading activation, forming the foundation for Hebbian-like learning mechanisms.

## Specifications

### Core Pathway Components

1. **PathwayTracer struct**
   - Real-time activation path recording
   - Multi-hop pathway tracking
   - Temporal sequence preservation
   - Branching and convergence detection

2. **ActivationPathway struct**
   - Ordered sequence of activated nodes
   - Activation strengths at each step
   - Timing information
   - Path significance metrics

3. **PathwaySegment struct**
   - Individual edge in activation path
   - Source and target nodes
   - Activation transfer strength
   - Temporal relationship

4. **PathwayMetrics struct**
   - Path efficiency measures
   - Activation propagation speed
   - Signal strength preservation
   - Convergence quality

### Performance Requirements

- Track pathways in real-time during activation
- Support multiple concurrent pathway traces
- Minimal overhead on activation performance (< 5%)
- Memory efficient for long pathways
- Thread-safe pathway recording

## Implementation Guide

### Step 1: Core Pathway Types

```rust
// File: src/cognitive/learning/pathway_tracing.rs

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use crate::core::types::{NodeId, EntityId, ActivationLevel};
use crate::core::activation::state::ActivationState;

#[derive(Debug, Clone)]
pub struct PathwaySegment {
    pub source_node: NodeId,
    pub target_node: NodeId,
    pub activation_transfer: f32,
    pub timestamp: Instant,
    pub propagation_delay: Duration,
    pub edge_weight: f32,
}

#[derive(Debug, Clone)]
pub struct ActivationPathway {
    pub pathway_id: PathwayId,
    pub segments: Vec<PathwaySegment>,
    pub source_query: String,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub total_activation: f32,
    pub path_efficiency: f32,
    pub significance_score: f32,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct PathwayId(pub u64);

#[derive(Debug, Clone)]
pub struct PathwayMetrics {
    pub total_length: usize,
    pub activation_decay: f32,
    pub propagation_speed: f32,
    pub branching_factor: f32,
    pub convergence_points: usize,
    pub signal_to_noise_ratio: f32,
}

#[derive(Debug)]
pub struct PathwayTracer {
    active_pathways: HashMap<PathwayId, ActivationPathway>,
    pathway_history: VecDeque<ActivationPathway>,
    tracing_enabled: bool,
    next_pathway_id: u64,
    max_pathway_length: usize,
    min_activation_threshold: f32,
    history_capacity: usize,
}

#[derive(Debug, Clone)]
pub struct TracingConfig {
    pub max_pathway_length: usize,
    pub min_activation_threshold: f32,
    pub history_capacity: usize,
    pub enable_branching_detection: bool,
    pub enable_convergence_tracking: bool,
}
```

### Step 2: Pathway Tracer Implementation

```rust
impl PathwayTracer {
    pub fn new() -> Self {
        Self {
            active_pathways: HashMap::new(),
            pathway_history: VecDeque::new(),
            tracing_enabled: true,
            next_pathway_id: 1,
            max_pathway_length: 50,
            min_activation_threshold: 0.01,
            history_capacity: 1000,
        }
    }
    
    pub fn with_config(config: TracingConfig) -> Self {
        Self {
            active_pathways: HashMap::new(),
            pathway_history: VecDeque::new(),
            tracing_enabled: true,
            next_pathway_id: 1,
            max_pathway_length: config.max_pathway_length,
            min_activation_threshold: config.min_activation_threshold,
            history_capacity: config.history_capacity,
        }
    }
    
    pub fn start_pathway_trace(&mut self, source_query: String) -> PathwayId {
        if !self.tracing_enabled {
            return PathwayId(0); // Null pathway ID
        }
        
        let pathway_id = PathwayId(self.next_pathway_id);
        self.next_pathway_id += 1;
        
        let pathway = ActivationPathway {
            pathway_id,
            segments: Vec::new(),
            source_query,
            start_time: Instant::now(),
            end_time: None,
            total_activation: 0.0,
            path_efficiency: 0.0,
            significance_score: 0.0,
        };
        
        self.active_pathways.insert(pathway_id, pathway);
        pathway_id
    }
    
    pub fn record_activation_step(
        &mut self,
        pathway_id: PathwayId,
        source_node: NodeId,
        target_node: NodeId,
        activation_transfer: f32,
        edge_weight: f32,
        propagation_delay: Duration,
    ) -> Result<(), TracingError> {
        if !self.tracing_enabled || pathway_id.0 == 0 {
            return Ok(());
        }
        
        let pathway = self.active_pathways.get_mut(&pathway_id)
            .ok_or(TracingError::PathwayNotFound)?;
        
        // Check activation threshold
        if activation_transfer < self.min_activation_threshold {
            return Ok(());
        }
        
        // Check pathway length limit
        if pathway.segments.len() >= self.max_pathway_length {
            self.finalize_pathway(pathway_id)?;
            return Err(TracingError::PathwayTooLong);
        }
        
        let segment = PathwaySegment {
            source_node,
            target_node,
            activation_transfer,
            timestamp: Instant::now(),
            propagation_delay,
            edge_weight,
        };
        
        pathway.segments.push(segment);
        pathway.total_activation += activation_transfer;
        
        Ok(())
    }
    
    pub fn record_branching_point(
        &mut self,
        pathway_id: PathwayId,
        branching_node: NodeId,
        target_nodes: &[NodeId],
        activation_splits: &[f32],
    ) -> Result<Vec<PathwayId>, TracingError> {
        if !self.tracing_enabled || pathway_id.0 == 0 {
            return Ok(vec![]);
        }
        
        let mut new_pathway_ids = Vec::new();
        
        // Create new pathways for each branch
        for (i, (&target_node, &activation)) in target_nodes.iter()
            .zip(activation_splits.iter())
            .enumerate() {
            
            if i == 0 {
                // Continue with original pathway for first branch
                self.record_activation_step(
                    pathway_id,
                    branching_node,
                    target_node,
                    activation,
                    1.0, // Default edge weight
                    Duration::from_micros(10),
                )?;
                new_pathway_ids.push(pathway_id);
            } else {
                // Create new pathway for additional branches
                let source_pathway = self.active_pathways.get(&pathway_id)
                    .ok_or(TracingError::PathwayNotFound)?;
                
                let new_pathway_id = self.start_pathway_trace(
                    source_pathway.source_query.clone()
                );
                
                // Copy existing segments
                if let Some(new_pathway) = self.active_pathways.get_mut(&new_pathway_id) {
                    new_pathway.segments = source_pathway.segments.clone();
                    new_pathway.total_activation = source_pathway.total_activation;
                    
                    // Add branching segment
                    self.record_activation_step(
                        new_pathway_id,
                        branching_node,
                        target_node,
                        activation,
                        1.0,
                        Duration::from_micros(10),
                    )?;
                }
                
                new_pathway_ids.push(new_pathway_id);
            }
        }
        
        Ok(new_pathway_ids)
    }
    
    pub fn finalize_pathway(&mut self, pathway_id: PathwayId) -> Result<ActivationPathway, TracingError> {
        let mut pathway = self.active_pathways.remove(&pathway_id)
            .ok_or(TracingError::PathwayNotFound)?;
        
        pathway.end_time = Some(Instant::now());
        pathway.path_efficiency = self.calculate_path_efficiency(&pathway);
        pathway.significance_score = self.calculate_significance_score(&pathway);
        
        // Add to history
        self.pathway_history.push_back(pathway.clone());
        
        // Manage history capacity
        if self.pathway_history.len() > self.history_capacity {
            self.pathway_history.pop_front();
        }
        
        Ok(pathway)
    }
    
    fn calculate_path_efficiency(&self, pathway: &ActivationPathway) -> f32 {
        if pathway.segments.is_empty() {
            return 0.0;
        }
        
        // Calculate efficiency as ratio of final activation to initial activation
        let initial_activation = pathway.segments.first()
            .map(|s| s.activation_transfer)
            .unwrap_or(0.0);
        
        let final_activation = pathway.segments.last()
            .map(|s| s.activation_transfer)
            .unwrap_or(0.0);
        
        if initial_activation > 0.0 {
            final_activation / initial_activation
        } else {
            0.0
        }
    }
    
    fn calculate_significance_score(&self, pathway: &ActivationPathway) -> f32 {
        let length_factor = (pathway.segments.len() as f32 / self.max_pathway_length as f32).min(1.0);
        let activation_factor = pathway.total_activation.min(1.0);
        let efficiency_factor = pathway.path_efficiency;
        
        // Weighted combination of factors
        (length_factor * 0.3 + activation_factor * 0.4 + efficiency_factor * 0.3).clamp(0.0, 1.0)
    }
}
```

### Step 3: Pathway Analysis and Metrics

```rust
impl PathwayTracer {
    pub fn analyze_pathway(&self, pathway_id: PathwayId) -> Result<PathwayMetrics, TracingError> {
        let pathway = self.active_pathways.get(&pathway_id)
            .or_else(|| {
                self.pathway_history.iter()
                    .find(|p| p.pathway_id == pathway_id)
            })
            .ok_or(TracingError::PathwayNotFound)?;
        
        let metrics = PathwayMetrics {
            total_length: pathway.segments.len(),
            activation_decay: self.calculate_activation_decay(pathway),
            propagation_speed: self.calculate_propagation_speed(pathway),
            branching_factor: self.calculate_branching_factor(pathway),
            convergence_points: self.count_convergence_points(pathway),
            signal_to_noise_ratio: self.calculate_signal_to_noise_ratio(pathway),
        };
        
        Ok(metrics)
    }
    
    fn calculate_activation_decay(&self, pathway: &ActivationPathway) -> f32 {
        if pathway.segments.len() < 2 {
            return 0.0;
        }
        
        let initial = pathway.segments.first().unwrap().activation_transfer;
        let final_activation = pathway.segments.last().unwrap().activation_transfer;
        
        if initial > 0.0 {
            (initial - final_activation) / initial
        } else {
            0.0
        }
    }
    
    fn calculate_propagation_speed(&self, pathway: &ActivationPathway) -> f32 {
        if pathway.segments.is_empty() {
            return 0.0;
        }
        
        let total_time = pathway.segments.iter()
            .map(|s| s.propagation_delay.as_secs_f32())
            .sum::<f32>();
        
        if total_time > 0.0 {
            pathway.segments.len() as f32 / total_time
        } else {
            0.0
        }
    }
    
    fn calculate_branching_factor(&self, pathway: &ActivationPathway) -> f32 {
        let mut branching_nodes = HashMap::new();
        
        for segment in &pathway.segments {
            *branching_nodes.entry(segment.source_node).or_insert(0) += 1;
        }
        
        let total_branches: usize = branching_nodes.values().sum();
        let unique_nodes = branching_nodes.len();
        
        if unique_nodes > 0 {
            total_branches as f32 / unique_nodes as f32
        } else {
            0.0
        }
    }
    
    fn count_convergence_points(&self, pathway: &ActivationPathway) -> usize {
        let mut target_counts = HashMap::new();
        
        for segment in &pathway.segments {
            *target_counts.entry(segment.target_node).or_insert(0) += 1;
        }
        
        target_counts.values().filter(|&&count| count > 1).count()
    }
    
    fn calculate_signal_to_noise_ratio(&self, pathway: &ActivationPathway) -> f32 {
        if pathway.segments.is_empty() {
            return 0.0;
        }
        
        let activations: Vec<f32> = pathway.segments.iter()
            .map(|s| s.activation_transfer)
            .collect();
        
        let mean = activations.iter().sum::<f32>() / activations.len() as f32;
        let variance = activations.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / activations.len() as f32;
        
        let std_dev = variance.sqrt();
        
        if std_dev > 0.0 {
            mean / std_dev
        } else {
            0.0
        }
    }
    
    pub fn get_pathway_statistics(&self) -> PathwayStatistics {
        let active_count = self.active_pathways.len();
        let completed_count = self.pathway_history.len();
        
        let avg_length = if completed_count > 0 {
            self.pathway_history.iter()
                .map(|p| p.segments.len())
                .sum::<usize>() as f32 / completed_count as f32
        } else {
            0.0
        };
        
        let avg_efficiency = if completed_count > 0 {
            self.pathway_history.iter()
                .map(|p| p.path_efficiency)
                .sum::<f32>() / completed_count as f32
        } else {
            0.0
        };
        
        PathwayStatistics {
            active_pathways: active_count,
            completed_pathways: completed_count,
            average_pathway_length: avg_length,
            average_efficiency: avg_efficiency,
            total_traced_activations: self.pathway_history.iter()
                .map(|p| p.total_activation)
                .sum(),
        }
    }
    
    pub fn get_significant_pathways(&self, min_significance: f32) -> Vec<&ActivationPathway> {
        self.pathway_history.iter()
            .filter(|p| p.significance_score >= min_significance)
            .collect()
    }
    
    pub fn clear_pathway_history(&mut self) {
        self.pathway_history.clear();
    }
    
    pub fn set_tracing_enabled(&mut self, enabled: bool) {
        self.tracing_enabled = enabled;
        
        if !enabled {
            // Finalize all active pathways
            let pathway_ids: Vec<PathwayId> = self.active_pathways.keys().copied().collect();
            for pathway_id in pathway_ids {
                let _ = self.finalize_pathway(pathway_id);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PathwayStatistics {
    pub active_pathways: usize,
    pub completed_pathways: usize,
    pub average_pathway_length: f32,
    pub average_efficiency: f32,
    pub total_traced_activations: f32,
}

#[derive(Debug, Clone)]
pub enum TracingError {
    PathwayNotFound,
    PathwayTooLong,
    InvalidActivation,
    TracingDisabled,
}

impl std::fmt::Display for TracingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TracingError::PathwayNotFound => write!(f, "Pathway not found"),
            TracingError::PathwayTooLong => write!(f, "Pathway exceeds maximum length"),
            TracingError::InvalidActivation => write!(f, "Invalid activation value"),
            TracingError::TracingDisabled => write!(f, "Pathway tracing is disabled"),
        }
    }
}

impl std::error::Error for TracingError {}
```

### Step 4: Integration with Spreading Activation

```rust
// File: src/core/activation/spreading_with_tracing.rs

use crate::cognitive::learning::pathway_tracing::{PathwayTracer, PathwayId};
use crate::core::activation::state::ActivationState;

pub struct TracingSpreadingActivation {
    pub pathway_tracer: PathwayTracer,
    // ... other spreading activation components
}

impl TracingSpreadingActivation {
    pub fn new() -> Self {
        Self {
            pathway_tracer: PathwayTracer::new(),
        }
    }
    
    pub fn spread_with_tracing(
        &mut self,
        initial_state: &ActivationState,
        query: String,
    ) -> (ActivationState, Vec<PathwayId>) {
        let mut traced_pathways = Vec::new();
        
        // Start pathway tracing for each initially activated node
        for &node_id in &initial_state.activated_nodes() {
            let pathway_id = self.pathway_tracer.start_pathway_trace(query.clone());
            traced_pathways.push(pathway_id);
        }
        
        // Perform spreading activation with tracing
        let mut current_state = initial_state.clone();
        
        for iteration in 0..10 { // Max iterations
            let mut next_state = current_state.clone();
            
            for (i, &source_node) in current_state.activated_nodes().iter().enumerate() {
                let source_activation = current_state.get_activation(source_node);
                
                // Simulate spreading to neighbors
                let neighbors = self.get_neighbors(source_node);
                
                for neighbor in neighbors {
                    let transfer_amount = source_activation * 0.1; // 10% transfer
                    
                    if transfer_amount > 0.01 {
                        next_state.add_activation(neighbor, transfer_amount);
                        
                        // Record pathway step
                        if i < traced_pathways.len() {
                            let _ = self.pathway_tracer.record_activation_step(
                                traced_pathways[i],
                                source_node,
                                neighbor,
                                transfer_amount,
                                1.0, // Edge weight
                                std::time::Duration::from_micros(100),
                            );
                        }
                    }
                }
            }
            
            current_state = next_state;
        }
        
        // Finalize pathways
        for pathway_id in &traced_pathways {
            let _ = self.pathway_tracer.finalize_pathway(*pathway_id);
        }
        
        (current_state, traced_pathways)
    }
    
    fn get_neighbors(&self, _node: NodeId) -> Vec<NodeId> {
        // Placeholder - would integrate with actual graph structure
        vec![NodeId(1), NodeId(2), NodeId(3)]
    }
}
```

## File Locations

- `src/cognitive/learning/pathway_tracing.rs` - Main implementation
- `src/cognitive/learning/mod.rs` - Module exports
- `src/core/activation/spreading_with_tracing.rs` - Integration module
- `tests/cognitive/learning/pathway_tracing_tests.rs` - Test implementation

## Success Criteria

- [ ] PathwayTracer struct compiles and runs
- [ ] Real-time pathway recording works correctly
- [ ] Branching and convergence detection functional
- [ ] Pathway metrics calculation accurate
- [ ] Memory usage reasonable for long pathways
- [ ] Thread-safe concurrent access
- [ ] All tests pass:
  - Basic pathway tracing
  - Branching point handling
  - Metrics calculation
  - Integration with spreading activation

## Test Requirements

```rust
#[test]
fn test_basic_pathway_tracing() {
    let mut tracer = PathwayTracer::new();
    
    let pathway_id = tracer.start_pathway_trace("test query".to_string());
    
    // Record activation steps
    tracer.record_activation_step(
        pathway_id,
        NodeId(1),
        NodeId(2),
        0.5,
        1.0,
        Duration::from_micros(100),
    ).unwrap();
    
    tracer.record_activation_step(
        pathway_id,
        NodeId(2),
        NodeId(3),
        0.3,
        1.0,
        Duration::from_micros(150),
    ).unwrap();
    
    let pathway = tracer.finalize_pathway(pathway_id).unwrap();
    
    assert_eq!(pathway.segments.len(), 2);
    assert_eq!(pathway.total_activation, 0.8);
}

#[test]
fn test_branching_detection() {
    let mut tracer = PathwayTracer::new();
    
    let pathway_id = tracer.start_pathway_trace("branching test".to_string());
    
    let target_nodes = vec![NodeId(2), NodeId(3), NodeId(4)];
    let activations = vec![0.3, 0.2, 0.1];
    
    let new_pathways = tracer.record_branching_point(
        pathway_id,
        NodeId(1),
        &target_nodes,
        &activations,
    ).unwrap();
    
    assert_eq!(new_pathways.len(), 3);
    
    // All pathways should exist
    for &new_pathway_id in &new_pathways {
        let pathway = tracer.finalize_pathway(new_pathway_id).unwrap();
        assert!(!pathway.segments.is_empty());
    }
}

#[test]
fn test_pathway_metrics() {
    let mut tracer = PathwayTracer::new();
    
    let pathway_id = tracer.start_pathway_trace("metrics test".to_string());
    
    // Create a pathway with decreasing activation
    tracer.record_activation_step(
        pathway_id, NodeId(1), NodeId(2), 1.0, 1.0, Duration::from_micros(100)
    ).unwrap();
    
    tracer.record_activation_step(
        pathway_id, NodeId(2), NodeId(3), 0.8, 1.0, Duration::from_micros(120)
    ).unwrap();
    
    tracer.record_activation_step(
        pathway_id, NodeId(3), NodeId(4), 0.6, 1.0, Duration::from_micros(110)
    ).unwrap();
    
    let metrics = tracer.analyze_pathway(pathway_id).unwrap();
    
    assert_eq!(metrics.total_length, 3);
    assert!(metrics.activation_decay > 0.0);
    assert!(metrics.propagation_speed > 0.0);
}

#[test]
fn test_pathway_significance_scoring() {
    let mut tracer = PathwayTracer::new();
    
    // Test high significance pathway
    let high_sig_id = tracer.start_pathway_trace("high significance".to_string());
    for i in 1..=10 {
        tracer.record_activation_step(
            high_sig_id,
            NodeId(i),
            NodeId(i + 1),
            0.8,
            1.0,
            Duration::from_micros(100),
        ).unwrap();
    }
    
    let high_pathway = tracer.finalize_pathway(high_sig_id).unwrap();
    
    // Test low significance pathway
    let low_sig_id = tracer.start_pathway_trace("low significance".to_string());
    tracer.record_activation_step(
        low_sig_id,
        NodeId(100),
        NodeId(101),
        0.05,
        1.0,
        Duration::from_micros(100),
    ).unwrap();
    
    let low_pathway = tracer.finalize_pathway(low_sig_id).unwrap();
    
    assert!(high_pathway.significance_score > low_pathway.significance_score);
}

#[test]
fn test_pathway_history_management() {
    let config = TracingConfig {
        max_pathway_length: 10,
        min_activation_threshold: 0.01,
        history_capacity: 3,
        enable_branching_detection: true,
        enable_convergence_tracking: true,
    };
    
    let mut tracer = PathwayTracer::with_config(config);
    
    // Create more pathways than history capacity
    for i in 0..5 {
        let pathway_id = tracer.start_pathway_trace(format!("test {}", i));
        tracer.record_activation_step(
            pathway_id,
            NodeId(i),
            NodeId(i + 1),
            0.5,
            1.0,
            Duration::from_micros(100),
        ).unwrap();
        tracer.finalize_pathway(pathway_id).unwrap();
    }
    
    let stats = tracer.get_pathway_statistics();
    assert_eq!(stats.completed_pathways, 3); // Should be limited by capacity
}
```

## Quality Gates

- [ ] Memory usage < 100MB for 10k pathways
- [ ] Tracing overhead < 5% of activation time
- [ ] Thread-safe concurrent pathway recording
- [ ] No memory leaks during continuous operation
- [ ] Accurate pathway metrics calculation
- [ ] Proper handling of long and complex pathways

## Next Task

Upon completion, proceed to **20_pathway_reinforcement.md**