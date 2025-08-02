# Micro Task 20: Pathway Reinforcement

**Priority**: CRITICAL  
**Estimated Time**: 45 minutes  
**Dependencies**: 19_pathway_tracing.md completed  
**Skills Required**: Hebbian learning, synaptic plasticity, reinforcement algorithms

## Objective

Implement Hebbian-like learning mechanisms to strengthen frequently used activation pathways and implement long-term potentiation (LTP) and long-term depression (LTD) for pathway adaptation.

## Context

Hebbian learning follows the principle "cells that fire together, wire together." This task implements pathway reinforcement that strengthens connections based on correlated activation patterns, enabling the system to learn and optimize successful query resolution paths.

## Specifications

### Core Reinforcement Components

1. **PathwayReinforcer struct**
   - Hebbian weight updates
   - Long-term potentiation (LTP) mechanisms
   - Long-term depression (LTD) for unused pathways
   - Activity-dependent plasticity

2. **SynapticWeight struct**
   - Current connection strength
   - Baseline weight value
   - Potentiation and depression factors
   - Usage history tracking

3. **LearningRule enum**
   - StandardHebbian (basic correlation)
   - BCM (Bienenstock-Cooper-Munro)
   - STDP (Spike-Timing Dependent Plasticity)
   - Oja (normalized Hebbian)

4. **PlasticityParams struct**
   - Learning rate parameters
   - Potentiation/depression thresholds
   - Decay time constants
   - Saturation limits

### Performance Requirements

- Real-time weight updates during activation
- Support for 1M+ synaptic connections
- Configurable learning rules and parameters
- Thread-safe concurrent weight modifications
- Persistent weight storage

## Implementation Guide

### Step 1: Core Reinforcement Types

```rust
// File: src/cognitive/learning/pathway_reinforcement.rs

use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::core::types::{NodeId, EntityId};
use crate::cognitive::learning::pathway_tracing::{ActivationPathway, PathwayId, PathwaySegment};

#[derive(Debug, Clone)]
pub struct SynapticWeight {
    pub current_strength: f32,
    pub baseline_strength: f32,
    pub potentiation_factor: f32,
    pub depression_factor: f32,
    pub last_active: Instant,
    pub activation_count: u64,
    pub total_activation: f32,
    pub average_activation: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum LearningRule {
    StandardHebbian { learning_rate: f32 },
    BCM { 
        learning_rate: f32, 
        threshold_adaptation_rate: f32 
    },
    STDP { 
        ltp_window: Duration, 
        ltd_window: Duration,
        max_weight_change: f32 
    },
    Oja { 
        learning_rate: f32, 
        normalization_factor: f32 
    },
}

#[derive(Debug, Clone)]
pub struct PlasticityParams {
    pub ltp_threshold: f32,          // Threshold for long-term potentiation
    pub ltd_threshold: f32,          // Threshold for long-term depression
    pub max_weight: f32,             // Maximum synaptic weight
    pub min_weight: f32,             // Minimum synaptic weight
    pub decay_rate: f32,             // Weight decay rate
    pub consolidation_threshold: f32, // Threshold for pathway consolidation
    pub homeostatic_scaling: f32,    // Homeostatic scaling factor
}

#[derive(Debug)]
pub struct PathwayReinforcer {
    synaptic_weights: HashMap<(NodeId, NodeId), SynapticWeight>,
    learning_rule: LearningRule,
    plasticity_params: PlasticityParams,
    pathway_strengths: HashMap<PathwayId, f32>,
    reinforcement_history: Vec<ReinforcementEvent>,
    global_activity_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct ReinforcementEvent {
    pub timestamp: Instant,
    pub event_type: ReinforcementType,
    pub source_node: NodeId,
    pub target_node: NodeId,
    pub weight_change: f32,
    pub pathway_id: Option<PathwayId>,
}

#[derive(Debug, Clone)]
pub enum ReinforcementType {
    Potentiation,
    Depression,
    Homeostatic,
    Consolidation,
}
```

### Step 2: Pathway Reinforcer Implementation

```rust
impl PathwayReinforcer {
    pub fn new(learning_rule: LearningRule) -> Self {
        let plasticity_params = PlasticityParams {
            ltp_threshold: 0.3,
            ltd_threshold: 0.1,
            max_weight: 5.0,
            min_weight: 0.01,
            decay_rate: 0.001,
            consolidation_threshold: 0.8,
            homeostatic_scaling: 0.1,
        };
        
        Self {
            synaptic_weights: HashMap::new(),
            learning_rule,
            plasticity_params,
            pathway_strengths: HashMap::new(),
            reinforcement_history: Vec::new(),
            global_activity_threshold: 0.2,
        }
    }
    
    pub fn with_plasticity_params(
        learning_rule: LearningRule, 
        params: PlasticityParams
    ) -> Self {
        Self {
            synaptic_weights: HashMap::new(),
            learning_rule,
            plasticity_params: params,
            pathway_strengths: HashMap::new(),
            reinforcement_history: Vec::new(),
            global_activity_threshold: 0.2,
        }
    }
    
    pub fn reinforce_pathway(
        &mut self, 
        pathway: &ActivationPathway,
    ) -> Result<f32, ReinforcementError> {
        if pathway.segments.is_empty() {
            return Err(ReinforcementError::EmptyPathway);
        }
        
        let pathway_strength = self.calculate_pathway_strength(pathway);
        
        // Apply reinforcement to each segment
        for segment in &pathway.segments {
            self.reinforce_connection(
                segment.source_node,
                segment.target_node,
                segment.activation_transfer,
                pathway.pathway_id,
            )?;
        }
        
        // Update pathway strength record
        self.pathway_strengths.insert(pathway.pathway_id, pathway_strength);
        
        // Apply homeostatic scaling if needed
        if pathway_strength > self.plasticity_params.consolidation_threshold {
            self.apply_homeostatic_scaling()?;
        }
        
        Ok(pathway_strength)
    }
    
    fn calculate_pathway_strength(&self, pathway: &ActivationPathway) -> f32 {
        if pathway.segments.is_empty() {
            return 0.0;
        }
        
        // Calculate weighted average of segment strengths
        let total_weight: f32 = pathway.segments.iter()
            .map(|s| self.get_synaptic_weight(s.source_node, s.target_node))
            .sum();
        
        total_weight / pathway.segments.len() as f32
    }
    
    fn reinforce_connection(
        &mut self,
        source: NodeId,
        target: NodeId,
        activation_strength: f32,
        pathway_id: PathwayId,
    ) -> Result<(), ReinforcementError> {
        let connection_key = (source, target);
        
        // Initialize connection if it doesn't exist
        if !self.synaptic_weights.contains_key(&connection_key) {
            self.initialize_connection(source, target);
        }
        
        let synaptic_weight = self.synaptic_weights.get_mut(&connection_key)
            .ok_or(ReinforcementError::ConnectionNotFound)?;
        
        // Apply learning rule
        let weight_change = self.calculate_weight_change(
            synaptic_weight,
            activation_strength,
        )?;
        
        // Update synaptic weight
        self.update_synaptic_weight(synaptic_weight, weight_change, activation_strength);
        
        // Log reinforcement event
        self.log_reinforcement_event(
            if weight_change > 0.0 { 
                ReinforcementType::Potentiation 
            } else { 
                ReinforcementType::Depression 
            },
            source,
            target,
            weight_change,
            Some(pathway_id),
        );
        
        Ok(())
    }
    
    fn initialize_connection(&mut self, source: NodeId, target: NodeId) {
        let initial_weight = SynapticWeight {
            current_strength: 1.0,
            baseline_strength: 1.0,
            potentiation_factor: 1.0,
            depression_factor: 1.0,
            last_active: Instant::now(),
            activation_count: 0,
            total_activation: 0.0,
            average_activation: 0.0,
        };
        
        self.synaptic_weights.insert((source, target), initial_weight);
    }
    
    fn calculate_weight_change(
        &self,
        synaptic_weight: &SynapticWeight,
        activation_strength: f32,
    ) -> Result<f32, ReinforcementError> {
        match self.learning_rule {
            LearningRule::StandardHebbian { learning_rate } => {
                Ok(learning_rate * activation_strength * synaptic_weight.current_strength)
            },
            
            LearningRule::BCM { learning_rate, .. } => {
                // BCM rule: weight change depends on post-synaptic activity relative to threshold
                let threshold = synaptic_weight.average_activation;
                let post_activity = activation_strength;
                let weight_change = learning_rate * post_activity * 
                    (post_activity - threshold) * synaptic_weight.current_strength;
                Ok(weight_change)
            },
            
            LearningRule::STDP { max_weight_change, .. } => {
                // Simplified STDP: assume causally related if activation is strong
                if activation_strength > self.plasticity_params.ltp_threshold {
                    Ok(max_weight_change * 0.5) // LTP
                } else if activation_strength < self.plasticity_params.ltd_threshold {
                    Ok(-max_weight_change * 0.3) // LTD
                } else {
                    Ok(0.0)
                }
            },
            
            LearningRule::Oja { learning_rate, normalization_factor } => {
                // Oja's rule: Hebbian learning with weight normalization
                let hebbian_term = learning_rate * activation_strength * synaptic_weight.current_strength;
                let normalization_term = normalization_factor * synaptic_weight.current_strength.powi(2);
                Ok(hebbian_term - normalization_term)
            },
        }
    }
    
    fn update_synaptic_weight(
        &mut self,
        synaptic_weight: &mut SynapticWeight,
        weight_change: f32,
        activation_strength: f32,
    ) {
        // Update weight with bounds checking
        synaptic_weight.current_strength = (synaptic_weight.current_strength + weight_change)
            .clamp(self.plasticity_params.min_weight, self.plasticity_params.max_weight);
        
        // Update statistics
        synaptic_weight.last_active = Instant::now();
        synaptic_weight.activation_count += 1;
        synaptic_weight.total_activation += activation_strength;
        synaptic_weight.average_activation = 
            synaptic_weight.total_activation / synaptic_weight.activation_count as f32;
        
        // Update potentiation/depression factors based on recent activity
        if activation_strength > self.plasticity_params.ltp_threshold {
            synaptic_weight.potentiation_factor = 
                (synaptic_weight.potentiation_factor * 1.1).min(2.0);
        } else if activation_strength < self.plasticity_params.ltd_threshold {
            synaptic_weight.depression_factor = 
                (synaptic_weight.depression_factor * 1.1).min(2.0);
        }
    }
    
    fn apply_homeostatic_scaling(&mut self) -> Result<(), ReinforcementError> {
        // Calculate global activity level
        let total_weights: f32 = self.synaptic_weights.values()
            .map(|w| w.current_strength)
            .sum();
        
        let avg_weight = total_weights / self.synaptic_weights.len() as f32;
        
        // Apply scaling if average weight is too high
        if avg_weight > self.global_activity_threshold {
            let scaling_factor = self.global_activity_threshold / avg_weight;
            
            for ((source, target), weight) in self.synaptic_weights.iter_mut() {
                let old_strength = weight.current_strength;
                weight.current_strength *= scaling_factor;
                
                let weight_change = weight.current_strength - old_strength;
                
                self.log_reinforcement_event(
                    ReinforcementType::Homeostatic,
                    *source,
                    *target,
                    weight_change,
                    None,
                );
            }
        }
        
        Ok(())
    }
}
```

### Step 3: Weight Management and Decay

```rust
impl PathwayReinforcer {
    pub fn apply_weight_decay(&mut self, time_step: Duration) {
        let decay_factor = (-self.plasticity_params.decay_rate * time_step.as_secs_f32()).exp();
        
        let mut to_remove = Vec::new();
        
        for ((source, target), weight) in self.synaptic_weights.iter_mut() {
            // Apply exponential decay
            weight.current_strength *= decay_factor;
            
            // Remove weights that have decayed below minimum
            if weight.current_strength < self.plasticity_params.min_weight {
                to_remove.push((*source, *target));
            }
            
            // Decay potentiation and depression factors
            weight.potentiation_factor = (weight.potentiation_factor * 0.99).max(1.0);
            weight.depression_factor = (weight.depression_factor * 0.99).max(1.0);
        }
        
        // Remove weak connections
        for connection in to_remove {
            self.synaptic_weights.remove(&connection);
            
            self.log_reinforcement_event(
                ReinforcementType::Depression,
                connection.0,
                connection.1,
                -self.plasticity_params.min_weight,
                None,
            );
        }
    }
    
    pub fn consolidate_strong_pathways(&mut self, min_strength: f32) -> Vec<PathwayId> {
        let mut consolidated_pathways = Vec::new();
        
        for (&pathway_id, &strength) in &self.pathway_strengths {
            if strength >= min_strength {
                // Strengthen all connections in this pathway
                consolidated_pathways.push(pathway_id);
                
                // Apply consolidation boost to pathway connections
                // (This would require storing pathway->connection mappings)
                // For now, we mark it as consolidated
            }
        }
        
        consolidated_pathways
    }
    
    pub fn get_synaptic_weight(&self, source: NodeId, target: NodeId) -> f32 {
        self.synaptic_weights.get(&(source, target))
            .map(|w| w.current_strength)
            .unwrap_or(1.0) // Default weight
    }
    
    pub fn get_connection_statistics(&self) -> ConnectionStatistics {
        let weights: Vec<f32> = self.synaptic_weights.values()
            .map(|w| w.current_strength)
            .collect();
        
        let total_connections = weights.len();
        let average_weight = if total_connections > 0 {
            weights.iter().sum::<f32>() / total_connections as f32
        } else {
            0.0
        };
        
        let strong_connections = weights.iter()
            .filter(|&&w| w > self.plasticity_params.ltp_threshold)
            .count();
        
        let weak_connections = weights.iter()
            .filter(|&&w| w < self.plasticity_params.ltd_threshold)
            .count();
        
        ConnectionStatistics {
            total_connections,
            average_weight,
            strong_connections,
            weak_connections,
            max_weight: weights.iter().fold(0.0, |a, &b| a.max(b)),
            min_weight: weights.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        }
    }
    
    pub fn get_pathway_strength(&self, pathway_id: PathwayId) -> Option<f32> {
        self.pathway_strengths.get(&pathway_id).copied()
    }
    
    pub fn get_reinforcement_history(&self) -> &[ReinforcementEvent] {
        &self.reinforcement_history
    }
    
    pub fn reset_all_weights(&mut self) {
        for weight in self.synaptic_weights.values_mut() {
            weight.current_strength = weight.baseline_strength;
            weight.potentiation_factor = 1.0;
            weight.depression_factor = 1.0;
            weight.activation_count = 0;
            weight.total_activation = 0.0;
            weight.average_activation = 0.0;
        }
        
        self.pathway_strengths.clear();
        self.reinforcement_history.clear();
    }
    
    fn log_reinforcement_event(
        &mut self,
        event_type: ReinforcementType,
        source: NodeId,
        target: NodeId,
        weight_change: f32,
        pathway_id: Option<PathwayId>,
    ) {
        let event = ReinforcementEvent {
            timestamp: Instant::now(),
            event_type,
            source_node: source,
            target_node: target,
            weight_change,
            pathway_id,
        };
        
        self.reinforcement_history.push(event);
        
        // Keep history manageable
        if self.reinforcement_history.len() > 10000 {
            self.reinforcement_history.drain(..5000);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConnectionStatistics {
    pub total_connections: usize,
    pub average_weight: f32,
    pub strong_connections: usize,
    pub weak_connections: usize,
    pub max_weight: f32,
    pub min_weight: f32,
}

#[derive(Debug, Clone)]
pub enum ReinforcementError {
    EmptyPathway,
    ConnectionNotFound,
    InvalidActivation,
    LearningDisabled,
}

impl std::fmt::Display for ReinforcementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReinforcementError::EmptyPathway => write!(f, "Cannot reinforce empty pathway"),
            ReinforcementError::ConnectionNotFound => write!(f, "Synaptic connection not found"),
            ReinforcementError::InvalidActivation => write!(f, "Invalid activation strength"),
            ReinforcementError::LearningDisabled => write!(f, "Learning is disabled"),
        }
    }
}

impl std::error::Error for ReinforcementError {}
```

### Step 4: Integration with Pathway Tracing

```rust
// File: src/cognitive/learning/learning_system.rs

use crate::cognitive::learning::pathway_tracing::PathwayTracer;
use crate::cognitive::learning::pathway_reinforcement::{PathwayReinforcer, LearningRule};

pub struct LearningSystem {
    pub pathway_tracer: PathwayTracer,
    pub pathway_reinforcer: PathwayReinforcer,
    learning_enabled: bool,
}

impl LearningSystem {
    pub fn new(learning_rule: LearningRule) -> Self {
        Self {
            pathway_tracer: PathwayTracer::new(),
            pathway_reinforcer: PathwayReinforcer::new(learning_rule),
            learning_enabled: true,
        }
    }
    
    pub fn process_activation_with_learning(
        &mut self,
        query: String,
        initial_activations: Vec<(NodeId, f32)>,
    ) -> Result<LearningResult, LearningError> {
        if !self.learning_enabled {
            return Err(LearningError::LearningDisabled);
        }
        
        // Start pathway tracing
        let pathway_id = self.pathway_tracer.start_pathway_trace(query);
        
        // Simulate activation spreading with tracing
        for (i, (source, activation)) in initial_activations.iter().enumerate() {
            let target = NodeId(source.0 + 1); // Simple progression
            
            self.pathway_tracer.record_activation_step(
                pathway_id,
                *source,
                target,
                *activation,
                1.0,
                Duration::from_micros(100),
            )?;
        }
        
        // Finalize pathway
        let pathway = self.pathway_tracer.finalize_pathway(pathway_id)?;
        
        // Apply reinforcement learning
        let pathway_strength = self.pathway_reinforcer.reinforce_pathway(&pathway)?;
        
        Ok(LearningResult {
            pathway_id,
            final_pathway_strength: pathway_strength,
            connections_modified: pathway.segments.len(),
            learning_events: self.pathway_reinforcer.get_reinforcement_history().len(),
        })
    }
    
    pub fn apply_maintenance_decay(&mut self, time_step: Duration) {
        self.pathway_reinforcer.apply_weight_decay(time_step);
    }
    
    pub fn get_system_statistics(&self) -> LearningSystemStatistics {
        let pathway_stats = self.pathway_tracer.get_pathway_statistics();
        let connection_stats = self.pathway_reinforcer.get_connection_statistics();
        
        LearningSystemStatistics {
            total_pathways_traced: pathway_stats.completed_pathways,
            average_pathway_length: pathway_stats.average_pathway_length,
            total_connections: connection_stats.total_connections,
            average_connection_strength: connection_stats.average_weight,
            strong_connections: connection_stats.strong_connections,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LearningResult {
    pub pathway_id: PathwayId,
    pub final_pathway_strength: f32,
    pub connections_modified: usize,
    pub learning_events: usize,
}

#[derive(Debug, Clone)]
pub struct LearningSystemStatistics {
    pub total_pathways_traced: usize,
    pub average_pathway_length: f32,
    pub total_connections: usize,
    pub average_connection_strength: f32,
    pub strong_connections: usize,
}

#[derive(Debug, Clone)]
pub enum LearningError {
    LearningDisabled,
    TracingError(TracingError),
    ReinforcementError(ReinforcementError),
}

impl From<TracingError> for LearningError {
    fn from(err: TracingError) -> Self {
        LearningError::TracingError(err)
    }
}

impl From<ReinforcementError> for LearningError {
    fn from(err: ReinforcementError) -> Self {
        LearningError::ReinforcementError(err)
    }
}
```

## File Locations

- `src/cognitive/learning/pathway_reinforcement.rs` - Main implementation
- `src/cognitive/learning/learning_system.rs` - Integration module
- `src/cognitive/learning/mod.rs` - Module exports
- `tests/cognitive/learning/pathway_reinforcement_tests.rs` - Test implementation

## Success Criteria

- [ ] PathwayReinforcer struct compiles and runs
- [ ] Hebbian learning rules implemented correctly
- [ ] LTP and LTD mechanisms functional
- [ ] Homeostatic scaling prevents runaway strengthening
- [ ] Weight decay removes unused connections
- [ ] Multiple learning rules supported
- [ ] All tests pass:
  - Basic Hebbian reinforcement
  - Weight bounds enforcement
  - Homeostatic scaling
  - Integration with pathway tracing

## Test Requirements

```rust
#[test]
fn test_basic_hebbian_reinforcement() {
    let learning_rule = LearningRule::StandardHebbian { learning_rate: 0.1 };
    let mut reinforcer = PathwayReinforcer::new(learning_rule);
    
    // Create a simple pathway
    let mut pathway = ActivationPathway {
        pathway_id: PathwayId(1),
        segments: vec![
            PathwaySegment {
                source_node: NodeId(1),
                target_node: NodeId(2),
                activation_transfer: 0.8,
                timestamp: Instant::now(),
                propagation_delay: Duration::from_micros(100),
                edge_weight: 1.0,
            }
        ],
        source_query: "test".to_string(),
        start_time: Instant::now(),
        end_time: Some(Instant::now()),
        total_activation: 0.8,
        path_efficiency: 1.0,
        significance_score: 0.9,
    };
    
    let initial_weight = reinforcer.get_synaptic_weight(NodeId(1), NodeId(2));
    
    reinforcer.reinforce_pathway(&pathway).unwrap();
    
    let final_weight = reinforcer.get_synaptic_weight(NodeId(1), NodeId(2));
    
    // Weight should have increased due to reinforcement
    assert!(final_weight > initial_weight);
}

#[test]
fn test_weight_bounds_enforcement() {
    let learning_rule = LearningRule::StandardHebbian { learning_rate: 10.0 }; // High learning rate
    let mut reinforcer = PathwayReinforcer::new(learning_rule);
    
    // Apply many reinforcements to test upper bound
    for _ in 0..100 {
        let pathway = create_test_pathway(NodeId(1), NodeId(2), 1.0);
        reinforcer.reinforce_pathway(&pathway).unwrap();
    }
    
    let weight = reinforcer.get_synaptic_weight(NodeId(1), NodeId(2));
    assert!(weight <= reinforcer.plasticity_params.max_weight);
}

#[test]
fn test_homeostatic_scaling() {
    let learning_rule = LearningRule::StandardHebbian { learning_rate: 1.0 };
    let mut reinforcer = PathwayReinforcer::new(learning_rule);
    
    // Create many strong connections
    for i in 0..10 {
        let pathway = create_test_pathway(NodeId(i), NodeId(i + 1), 0.9);
        reinforcer.reinforce_pathway(&pathway).unwrap();
    }
    
    let stats_before = reinforcer.get_connection_statistics();
    
    // Trigger homeostatic scaling by creating a very strong pathway
    let strong_pathway = create_test_pathway(NodeId(100), NodeId(101), 1.0);
    reinforcer.reinforce_pathway(&strong_pathway).unwrap();
    
    let stats_after = reinforcer.get_connection_statistics();
    
    // Average weight should be regulated
    assert!(stats_after.average_weight <= stats_before.average_weight * 1.1);
}

#[test]
fn test_weight_decay() {
    let learning_rule = LearningRule::StandardHebbian { learning_rate: 0.1 };
    let mut reinforcer = PathwayReinforcer::new(learning_rule);
    
    // Reinforce a connection
    let pathway = create_test_pathway(NodeId(1), NodeId(2), 0.8);
    reinforcer.reinforce_pathway(&pathway).unwrap();
    
    let weight_before = reinforcer.get_synaptic_weight(NodeId(1), NodeId(2));
    
    // Apply decay
    reinforcer.apply_weight_decay(Duration::from_secs(10));
    
    let weight_after = reinforcer.get_synaptic_weight(NodeId(1), NodeId(2));
    
    // Weight should have decayed
    assert!(weight_after < weight_before);
}

#[test]
fn test_bcm_learning_rule() {
    let learning_rule = LearningRule::BCM { 
        learning_rate: 0.1, 
        threshold_adaptation_rate: 0.01 
    };
    let mut reinforcer = PathwayReinforcer::new(learning_rule);
    
    // Apply weak activation (below threshold)
    let weak_pathway = create_test_pathway(NodeId(1), NodeId(2), 0.1);
    reinforcer.reinforce_pathway(&weak_pathway).unwrap();
    
    let weight_after_weak = reinforcer.get_synaptic_weight(NodeId(1), NodeId(2));
    
    // Apply strong activation (above threshold)
    let strong_pathway = create_test_pathway(NodeId(1), NodeId(2), 0.9);
    reinforcer.reinforce_pathway(&strong_pathway).unwrap();
    
    let weight_after_strong = reinforcer.get_synaptic_weight(NodeId(1), NodeId(2));
    
    // Strong activation should increase weight more than weak activation
    assert!(weight_after_strong > weight_after_weak);
}

#[test]
fn test_learning_system_integration() {
    let learning_rule = LearningRule::StandardHebbian { learning_rate: 0.1 };
    let mut learning_system = LearningSystem::new(learning_rule);
    
    let initial_activations = vec![
        (NodeId(1), 0.8),
        (NodeId(2), 0.6),
        (NodeId(3), 0.4),
    ];
    
    let result = learning_system.process_activation_with_learning(
        "test query".to_string(),
        initial_activations,
    ).unwrap();
    
    assert!(result.final_pathway_strength > 0.0);
    assert!(result.connections_modified > 0);
    
    let stats = learning_system.get_system_statistics();
    assert_eq!(stats.total_pathways_traced, 1);
    assert!(stats.total_connections > 0);
}

// Helper function for tests
fn create_test_pathway(source: NodeId, target: NodeId, activation: f32) -> ActivationPathway {
    ActivationPathway {
        pathway_id: PathwayId(1),
        segments: vec![
            PathwaySegment {
                source_node: source,
                target_node: target,
                activation_transfer: activation,
                timestamp: Instant::now(),
                propagation_delay: Duration::from_micros(100),
                edge_weight: 1.0,
            }
        ],
        source_query: "test".to_string(),
        start_time: Instant::now(),
        end_time: Some(Instant::now()),
        total_activation: activation,
        path_efficiency: 1.0,
        significance_score: 0.8,
    }
}
```

## Quality Gates

- [ ] Weight updates are numerically stable
- [ ] Homeostatic mechanisms prevent runaway growth
- [ ] Multiple learning rules work correctly
- [ ] Thread-safe concurrent weight updates
- [ ] Memory usage remains bounded during continuous learning
- [ ] Learning performance acceptable (< 1ms per pathway)

## Next Task

Upon completion, proceed to **21_pathway_memory.md**