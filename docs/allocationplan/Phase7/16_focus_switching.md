# Micro Task 16: Focus Switching

**Priority**: HIGH  
**Estimated Time**: 30 minutes  
**Dependencies**: 15_attention_weighting.md completed  
**Skills Required**: State machines, cognitive switching, transition modeling

## Objective

Implement dynamic attention focus switching mechanisms that model cognitive switching costs, inhibition of return, and context-dependent switching strategies.

## Context

Focus switching is a critical cognitive process that determines when and how attention moves between different targets. This involves switching costs (time/energy penalties), inhibition of return (reduced likelihood of returning to recently abandoned targets), and strategic switching based on task demands.

## Specifications

### Core Switching Components

1. **FocusSwitcher struct**
   - State machine for focus transitions
   - Switching cost calculation
   - Inhibition of return tracking
   - Context-adaptive switching policies

2. **SwitchingCost struct**
   - Time-based switching penalties
   - Cognitive load adjustments
   - Context similarity factors
   - Frequency-based cost reductions

3. **SwitchingPolicy enum**
   - Immediate switching
   - Threshold-based switching
   - Hysteresis switching
   - Goal-driven switching

### Performance Requirements

- Switching decision latency < 50ms
- Support rapid switching sequences
- Maintain switching history for analysis
- Adaptive switching cost learning
- Thread-safe concurrent access

## Implementation Guide

### Step 1: Core Switching Types

```rust
// File: src/cognitive/attention/focus_switching.rs

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use crate::core::types::EntityId;
use crate::cognitive::attention::{AttentionTarget, ContextState};

#[derive(Debug, Clone)]
pub struct SwitchingCost {
    pub time_penalty: Duration,
    pub cognitive_load_increase: f32,
    pub accuracy_reduction: f32,
    pub energy_expenditure: f32,
}

#[derive(Debug, Clone)]
pub enum SwitchingPolicy {
    Immediate,
    Threshold { switch_threshold: f32 },
    Hysteresis { switch_up: f32, switch_down: f32 },
    GoalDriven { goal_priority_threshold: f32 },
    Adaptive { learning_rate: f32 },
}

#[derive(Debug, Clone)]
pub struct SwitchingEvent {
    pub timestamp: Instant,
    pub from_entity: Option<EntityId>,
    pub to_entity: EntityId,
    pub switch_reason: SwitchReason,
    pub switching_cost: SwitchingCost,
    pub context: ContextState,
}

#[derive(Debug, Clone)]
pub enum SwitchReason {
    WeightThresholdExceeded,
    GoalChange,
    BottomUpCapture,
    TaskCompletion,
    TimeoutExpired,
    UserDirected,
    ContextChange,
}

#[derive(Debug)]
pub struct FocusSwitcher {
    current_focus: Option<EntityId>,
    switching_policy: SwitchingPolicy,
    inhibition_map: HashMap<EntityId, InhibitionRecord>,
    switching_history: VecDeque<SwitchingEvent>,
    base_switching_costs: HashMap<String, SwitchingCost>,
    context_similarity_cache: HashMap<(EntityId, EntityId), f32>,
    switching_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct InhibitionRecord {
    pub entity_id: EntityId,
    pub inhibition_strength: f32,
    pub inhibition_start: Instant,
    pub decay_rate: f32,
    pub minimum_return_delay: Duration,
}

#[derive(Debug, Clone)]
pub enum SwitchingDecision {
    Switch { to_entity: EntityId, reason: SwitchReason },
    Maintain { current_entity: EntityId },
    Delay { until: Instant, reason: String },
}
```

### Step 2: Focus Switching Implementation

```rust
impl FocusSwitcher {
    pub fn new(policy: SwitchingPolicy) -> Self {
        Self {
            current_focus: None,
            switching_policy: policy,
            inhibition_map: HashMap::new(),
            switching_history: VecDeque::new(),
            base_switching_costs: Self::initialize_base_costs(),
            context_similarity_cache: HashMap::new(),
            switching_threshold: 0.3,
        }
    }
    
    fn initialize_base_costs() -> HashMap<String, SwitchingCost> {
        let mut costs = HashMap::new();
        
        costs.insert("same_domain".to_string(), SwitchingCost {
            time_penalty: Duration::from_millis(100),
            cognitive_load_increase: 0.1,
            accuracy_reduction: 0.05,
            energy_expenditure: 0.1,
        });
        
        costs.insert("different_domain".to_string(), SwitchingCost {
            time_penalty: Duration::from_millis(300),
            cognitive_load_increase: 0.3,
            accuracy_reduction: 0.15,
            energy_expenditure: 0.25,
        });
        
        costs.insert("task_switch".to_string(), SwitchingCost {
            time_penalty: Duration::from_millis(500),
            cognitive_load_increase: 0.4,
            accuracy_reduction: 0.2,
            energy_expenditure: 0.3,
        });
        
        costs
    }
    
    pub fn evaluate_switching_decision(
        &mut self,
        candidate_targets: &[(EntityId, f32)],
        context: &ContextState
    ) -> SwitchingDecision {
        
        // Find highest weighted candidate
        let best_candidate = candidate_targets.iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(entity_id, weight)| (*entity_id, *weight));
        
        let current_focus = match self.current_focus {
            Some(focus) => focus,
            None => {
                // No current focus, switch to best candidate if available
                if let Some((entity_id, _)) = best_candidate {
                    return SwitchingDecision::Switch {
                        to_entity: entity_id,
                        reason: SwitchReason::TaskCompletion,
                    };
                } else {
                    return SwitchingDecision::Delay {
                        until: Instant::now() + Duration::from_millis(100),
                        reason: "No suitable candidates".to_string(),
                    };
                }
            }
        };
        
        // Get current focus weight
        let current_weight = candidate_targets.iter()
            .find(|(entity_id, _)| *entity_id == current_focus)
            .map(|(_, weight)| *weight)
            .unwrap_or(0.0);
        
        if let Some((best_entity, best_weight)) = best_candidate {
            if best_entity == current_focus {
                // Best candidate is current focus
                return SwitchingDecision::Maintain { current_entity: current_focus };
            }
            
            // Check if switching is warranted
            let should_switch = self.should_switch(
                current_focus,
                current_weight,
                best_entity,
                best_weight,
                context
            );
            
            match should_switch {
                Ok(reason) => SwitchingDecision::Switch {
                    to_entity: best_entity,
                    reason,
                },
                Err(delay_reason) => {
                    if delay_reason.contains("inhibition") {
                        SwitchingDecision::Delay {
                            until: Instant::now() + Duration::from_millis(200),
                            reason: delay_reason,
                        }
                    } else {
                        SwitchingDecision::Maintain { current_entity: current_focus }
                    }
                }
            }
        } else {
            SwitchingDecision::Maintain { current_entity: current_focus }
        }
    }
    
    fn should_switch(
        &self,
        current_entity: EntityId,
        current_weight: f32,
        candidate_entity: EntityId,
        candidate_weight: f32,
        context: &ContextState
    ) -> Result<SwitchReason, String> {
        
        // Check inhibition of return
        if self.is_inhibited(candidate_entity) {
            return Err("Target is inhibited from recent switching".to_string());
        }
        
        // Apply switching policy
        match &self.switching_policy {
            SwitchingPolicy::Immediate => {
                if candidate_weight > current_weight {
                    Ok(SwitchReason::WeightThresholdExceeded)
                } else {
                    Err("Candidate weight not higher".to_string())
                }
            },
            
            SwitchingPolicy::Threshold { switch_threshold } => {
                let weight_difference = candidate_weight - current_weight;
                if weight_difference > *switch_threshold {
                    Ok(SwitchReason::WeightThresholdExceeded)
                } else {
                    Err("Weight difference below threshold".to_string())
                }
            },
            
            SwitchingPolicy::Hysteresis { switch_up, switch_down } => {
                let weight_difference = candidate_weight - current_weight;
                
                // Use different thresholds based on whether we're switching up or down
                let threshold = if candidate_weight > current_weight {
                    *switch_up
                } else {
                    *switch_down
                };
                
                if weight_difference.abs() > threshold {
                    Ok(SwitchReason::WeightThresholdExceeded)
                } else {
                    Err("Hysteresis threshold not met".to_string())
                }
            },
            
            SwitchingPolicy::GoalDriven { goal_priority_threshold } => {
                // Check if candidate aligns better with current goals
                if context.current_goal.is_some() && candidate_weight > *goal_priority_threshold {
                    Ok(SwitchReason::GoalChange)
                } else {
                    Err("Goal-driven threshold not met".to_string())
                }
            },
            
            SwitchingPolicy::Adaptive { learning_rate: _ } => {
                // Use learned switching patterns (simplified implementation)
                let adaptive_threshold = self.calculate_adaptive_threshold(current_entity, candidate_entity);
                let weight_difference = candidate_weight - current_weight;
                
                if weight_difference > adaptive_threshold {
                    Ok(SwitchReason::WeightThresholdExceeded)
                } else {
                    Err("Adaptive threshold not met".to_string())
                }
            },
        }
    }
    
    fn is_inhibited(&self, entity_id: EntityId) -> bool {
        if let Some(inhibition) = self.inhibition_map.get(&entity_id) {
            let elapsed = inhibition.inhibition_start.elapsed();
            
            // Check minimum return delay
            if elapsed < inhibition.minimum_return_delay {
                return true;
            }
            
            // Check if inhibition has decayed below threshold
            let current_inhibition = inhibition.inhibition_strength * 
                (-inhibition.decay_rate * elapsed.as_secs_f32()).exp();
            
            current_inhibition > 0.1 // Inhibition threshold
        } else {
            false
        }
    }
    
    fn calculate_adaptive_threshold(&self, current_entity: EntityId, candidate_entity: EntityId) -> f32 {
        // Calculate adaptive threshold based on switching history
        let base_threshold = 0.2;
        
        // Check recent switching patterns
        let recent_switches = self.switching_history.iter()
            .take(10)
            .filter(|event| {
                event.from_entity == Some(current_entity) && 
                event.to_entity == candidate_entity
            })
            .count();
        
        // Increase threshold if we've been switching frequently between these entities
        let frequency_adjustment = recent_switches as f32 * 0.05;
        
        // Check context similarity
        let context_adjustment = self.get_context_similarity(current_entity, candidate_entity) * 0.1;
        
        base_threshold + frequency_adjustment - context_adjustment
    }
    
    fn get_context_similarity(&self, entity1: EntityId, entity2: EntityId) -> f32 {
        // Simplified context similarity calculation
        // In practice, this would analyze semantic/contextual relationships
        self.context_similarity_cache.get(&(entity1, entity2))
            .or_else(|| self.context_similarity_cache.get(&(entity2, entity1)))
            .copied()
            .unwrap_or(0.5) // Default moderate similarity
    }
}
```

### Step 3: Switching Execution and Inhibition

```rust
impl FocusSwitcher {
    pub fn execute_switch(
        &mut self,
        to_entity: EntityId,
        reason: SwitchReason,
        context: &ContextState
    ) -> Result<SwitchingCost, SwitchingError> {
        
        let from_entity = self.current_focus;
        
        // Calculate switching cost
        let switching_cost = self.calculate_switching_cost(from_entity, to_entity, &reason, context);
        
        // Apply inhibition to previous focus
        if let Some(prev_entity) = from_entity {
            self.apply_inhibition_of_return(prev_entity);
        }
        
        // Update current focus
        self.current_focus = Some(to_entity);
        
        // Record switching event
        let switching_event = SwitchingEvent {
            timestamp: Instant::now(),
            from_entity,
            to_entity,
            switch_reason: reason,
            switching_cost: switching_cost.clone(),
            context: context.clone(),
        };
        
        self.switching_history.push_back(switching_event);
        
        // Keep history manageable
        if self.switching_history.len() > 100 {
            self.switching_history.pop_front();
        }
        
        Ok(switching_cost)
    }
    
    fn calculate_switching_cost(
        &self,
        from_entity: Option<EntityId>,
        to_entity: EntityId,
        reason: &SwitchReason,
        context: &ContextState
    ) -> SwitchingCost {
        
        let base_cost = match reason {
            SwitchReason::GoalChange => {
                self.base_switching_costs.get("task_switch").unwrap().clone()
            },
            SwitchReason::BottomUpCapture => {
                self.base_switching_costs.get("different_domain").unwrap().clone()
            },
            _ => {
                self.base_switching_costs.get("same_domain").unwrap().clone()
            }
        };
        
        // Adjust cost based on context
        let context_multiplier = 1.0 + context.cognitive_load * 0.5 + context.time_pressure * 0.3;
        
        // Adjust cost based on similarity
        let similarity_adjustment = if let Some(from) = from_entity {
            let similarity = self.get_context_similarity(from, to_entity);
            1.0 - (similarity * 0.3) // Higher similarity = lower cost
        } else {
            1.0 // No previous entity, full cost
        };
        
        SwitchingCost {
            time_penalty: Duration::from_millis(
                (base_cost.time_penalty.as_millis() as f32 * context_multiplier * similarity_adjustment) as u64
            ),
            cognitive_load_increase: base_cost.cognitive_load_increase * context_multiplier,
            accuracy_reduction: base_cost.accuracy_reduction * context_multiplier,
            energy_expenditure: base_cost.energy_expenditure * context_multiplier,
        }
    }
    
    fn apply_inhibition_of_return(&mut self, entity_id: EntityId) {
        let inhibition_strength = self.calculate_inhibition_strength(entity_id);
        
        let inhibition = InhibitionRecord {
            entity_id,
            inhibition_strength,
            inhibition_start: Instant::now(),
            decay_rate: 0.5, // Half-life of ~1.4 seconds
            minimum_return_delay: Duration::from_millis(500),
        };
        
        self.inhibition_map.insert(entity_id, inhibition);
    }
    
    fn calculate_inhibition_strength(&self, entity_id: EntityId) -> f32 {
        // Base inhibition strength
        let base_strength = 0.7;
        
        // Increase inhibition if entity was focused on recently
        let recent_focus_bonus = self.switching_history.iter()
            .take(5)
            .filter(|event| event.to_entity == entity_id)
            .count() as f32 * 0.1;
        
        (base_strength + recent_focus_bonus).clamp(0.0, 1.0)
    }
    
    pub fn update_inhibition(&mut self, delta_time: Duration) {
        let dt = delta_time.as_secs_f32();
        let mut to_remove = Vec::new();
        
        for (entity_id, inhibition) in self.inhibition_map.iter_mut() {
            // Apply decay
            let elapsed = inhibition.inhibition_start.elapsed().as_secs_f32();
            inhibition.inhibition_strength *= (-inhibition.decay_rate * dt).exp();
            
            // Remove if below threshold and past minimum delay
            if inhibition.inhibition_strength < 0.1 && 
               inhibition.inhibition_start.elapsed() > inhibition.minimum_return_delay {
                to_remove.push(*entity_id);
            }
        }
        
        for entity_id in to_remove {
            self.inhibition_map.remove(&entity_id);
        }
    }
    
    pub fn get_current_focus(&self) -> Option<EntityId> {
        self.current_focus
    }
    
    pub fn get_switching_history(&self) -> &VecDeque<SwitchingEvent> {
        &self.switching_history
    }
    
    pub fn get_inhibition_strength(&self, entity_id: EntityId) -> f32 {
        self.inhibition_map.get(&entity_id)
            .map(|inhibition| {
                let elapsed = inhibition.inhibition_start.elapsed().as_secs_f32();
                inhibition.inhibition_strength * (-inhibition.decay_rate * elapsed).exp()
            })
            .unwrap_or(0.0)
    }
    
    pub fn force_switch(&mut self, to_entity: EntityId, context: &ContextState) -> Result<SwitchingCost, SwitchingError> {
        // Forced switch bypasses normal switching logic
        self.execute_switch(to_entity, SwitchReason::UserDirected, context)
    }
}

#[derive(Debug, Clone)]
pub enum SwitchingError {
    NoCurrentFocus,
    InvalidTarget,
    SwitchingBlocked(String),
}

impl std::fmt::Display for SwitchingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SwitchingError::NoCurrentFocus => write!(f, "No current focus to switch from"),
            SwitchingError::InvalidTarget => write!(f, "Invalid switching target"),
            SwitchingError::SwitchingBlocked(reason) => write!(f, "Switching blocked: {}", reason),
        }
    }
}

impl std::error::Error for SwitchingError {}
```

## File Locations

- `src/cognitive/attention/focus_switching.rs` - Main implementation
- `src/cognitive/attention/mod.rs` - Module exports
- `tests/cognitive/attention/focus_switching_tests.rs` - Test implementation

## Success Criteria

- [ ] FocusSwitcher compiles and runs correctly
- [ ] All switching policies implemented and functional
- [ ] Inhibition of return working correctly
- [ ] Switching costs calculated appropriately
- [ ] Context-sensitive switching behavior
- [ ] Thread-safe concurrent access
- [ ] All tests pass:
  - Switching decision logic
  - Cost calculation accuracy
  - Inhibition mechanics
  - Policy application

## Test Requirements

```rust
#[test]
fn test_threshold_switching_policy() {
    let mut switcher = FocusSwitcher::new(
        SwitchingPolicy::Threshold { switch_threshold: 0.3 }
    );
    
    let context = ContextState::default();
    
    // Initial switch to establish focus
    switcher.execute_switch(EntityId(1), SwitchReason::TaskCompletion, &context).unwrap();
    
    let candidates = vec![
        (EntityId(1), 0.5), // Current focus
        (EntityId(2), 0.6), // Slight improvement
        (EntityId(3), 0.9), // Large improvement
    ];
    
    let decision = switcher.evaluate_switching_decision(&candidates, &context);
    
    // Should switch to EntityId(3) because 0.9 - 0.5 > 0.3
    match decision {
        SwitchingDecision::Switch { to_entity, .. } => {
            assert_eq!(to_entity, EntityId(3));
        },
        _ => panic!("Expected switch decision"),
    }
}

#[test]
fn test_inhibition_of_return() {
    let mut switcher = FocusSwitcher::new(
        SwitchingPolicy::Immediate
    );
    
    let context = ContextState::default();
    
    // Switch from 1 to 2
    switcher.execute_switch(EntityId(1), SwitchReason::TaskCompletion, &context).unwrap();
    switcher.execute_switch(EntityId(2), SwitchReason::WeightThresholdExceeded, &context).unwrap();
    
    // Entity 1 should now be inhibited
    assert!(switcher.get_inhibition_strength(EntityId(1)) > 0.0);
    
    let candidates = vec![
        (EntityId(1), 0.9), // High weight but recently abandoned
        (EntityId(2), 0.5), // Current focus
        (EntityId(3), 0.7), // Alternative
    ];
    
    let decision = switcher.evaluate_switching_decision(&candidates, &context);
    
    // Should not switch to EntityId(1) due to inhibition
    match decision {
        SwitchingDecision::Switch { to_entity, .. } => {
            assert_ne!(to_entity, EntityId(1));
        },
        SwitchingDecision::Delay { .. } => {
            // Acceptable if switching is delayed due to inhibition
        },
        _ => {}, // Other decisions are acceptable
    }
}

#[test]
fn test_switching_cost_calculation() {
    let switcher = FocusSwitcher::new(SwitchingPolicy::Immediate);
    
    let context = ContextState {
        cognitive_load: 0.8,
        time_pressure: 0.6,
        ..Default::default()
    };
    
    let cost = switcher.calculate_switching_cost(
        Some(EntityId(1)),
        EntityId(2),
        &SwitchReason::GoalChange,
        &context
    );
    
    // High cognitive load and time pressure should increase costs
    assert!(cost.time_penalty > Duration::from_millis(300));
    assert!(cost.cognitive_load_increase > 0.3);
}

#[test]
fn test_hysteresis_switching() {
    let mut switcher = FocusSwitcher::new(
        SwitchingPolicy::Hysteresis { 
            switch_up: 0.4,   // Higher threshold for upward switches
            switch_down: 0.2  // Lower threshold for downward switches
        }
    );
    
    let context = ContextState::default();
    
    // Establish initial focus
    switcher.execute_switch(EntityId(1), SwitchReason::TaskCompletion, &context).unwrap();
    
    // Test upward switch (requires higher threshold)
    let candidates_up = vec![
        (EntityId(1), 0.5),
        (EntityId(2), 0.8), // 0.3 difference < 0.4 threshold
    ];
    
    let decision_up = switcher.evaluate_switching_decision(&candidates_up, &context);
    
    // Should maintain focus due to hysteresis
    match decision_up {
        SwitchingDecision::Maintain { .. } => {}, // Expected
        _ => panic!("Expected maintain decision for upward hysteresis"),
    }
    
    // Test with higher difference
    let candidates_high = vec![
        (EntityId(1), 0.5),
        (EntityId(2), 1.0), // 0.5 difference > 0.4 threshold
    ];
    
    let decision_high = switcher.evaluate_switching_decision(&candidates_high, &context);
    
    // Should switch due to threshold being exceeded
    match decision_high {
        SwitchingDecision::Switch { to_entity, .. } => {
            assert_eq!(to_entity, EntityId(2));
        },
        _ => panic!("Expected switch decision when threshold exceeded"),
    }
}

#[test]
fn test_inhibition_decay() {
    let mut switcher = FocusSwitcher::new(SwitchingPolicy::Immediate);
    
    let context = ContextState::default();
    
    // Create inhibition
    switcher.execute_switch(EntityId(1), SwitchReason::TaskCompletion, &context).unwrap();
    switcher.execute_switch(EntityId(2), SwitchReason::WeightThresholdExceeded, &context).unwrap();
    
    let initial_inhibition = switcher.get_inhibition_strength(EntityId(1));
    assert!(initial_inhibition > 0.0);
    
    // Apply decay over time
    switcher.update_inhibition(Duration::from_secs(2));
    
    let decayed_inhibition = switcher.get_inhibition_strength(EntityId(1));
    
    // Inhibition should have decayed
    assert!(decayed_inhibition < initial_inhibition);
}
```

## Quality Gates

- [ ] Switching decisions consistent with policies
- [ ] Inhibition of return prevents immediate returns
- [ ] Switching costs reflect cognitive realism  
- [ ] Performance < 50ms for switching decisions
- [ ] Thread-safe concurrent access verified
- [ ] No memory leaks in switching history

## Next Task

Upon completion, proceed to **17_salience_calculation.md**