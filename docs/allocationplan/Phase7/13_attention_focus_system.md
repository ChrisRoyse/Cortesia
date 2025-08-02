# Micro Task 13: Attention Focus System

**Priority**: CRITICAL  
**Estimated Time**: 50 minutes  
**Dependencies**: 12_intent_tests.md completed  
**Skills Required**: Cognitive modeling, attention mechanisms, priority queues

## Objective

Implement a brain-inspired attention focus system that manages selective attention, working memory capacity limits, and winner-take-all attention dynamics.

## Context

Human attention operates through both top-down (goal-driven) and bottom-up (stimulus-driven) mechanisms. This task implements the core attention focus system that manages multiple competing attention targets and enforces cognitive capacity limits.

## Specifications

### Core Attention Components

1. **AttentionFocus struct**
   - Active attention targets with salience scores
   - Working memory capacity management (7±2 rule)
   - Top-down goal influences
   - Bottom-up stimulus captures

2. **AttentionTarget struct**
   - Entity or concept being attended to
   - Current attention strength (0.0-1.0)
   - Persistence duration
   - Source (top-down vs bottom-up)

3. **AttentionState enum**
   - Focused (single dominant target)
   - Divided (multiple balanced targets)
   - Scanning (searching for targets)
   - Overloaded (exceeding capacity)

### Performance Requirements

- Support 5-9 simultaneous attention targets (Miller's 7±2)
- Focus switching latency < 100ms
- Attention decay follows exponential curves
- Thread-safe concurrent access
- Memory efficient for continuous operation

## Implementation Guide

### Step 1: Core Attention Types

```rust
// File: src/cognitive/attention/focus_system.rs

use std::collections::{HashMap, BinaryHeap};
use std::time::{Duration, Instant};
use crate::core::types::{NodeId, EntityId};

#[derive(Debug, Clone, PartialEq)]
pub enum AttentionSource {
    TopDown { goal: String, priority: f32 },
    BottomUp { stimulus_strength: f32, novelty: f32 },
    Maintenance { importance: f32 },
}

#[derive(Debug, Clone)]
pub struct AttentionTarget {
    pub entity_id: EntityId,
    pub attention_strength: f32,  // 0.0 to 1.0
    pub source: AttentionSource,
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub persistence: Duration,
    pub decay_rate: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AttentionState {
    Focused { dominant_target: EntityId },
    Divided { target_count: usize },
    Scanning { search_query: String },
    Overloaded { exceeding_by: usize },
}

#[derive(Debug)]
pub struct AttentionFocus {
    targets: HashMap<EntityId, AttentionTarget>,
    focus_history: Vec<AttentionEvent>,
    current_state: AttentionState,
    working_memory_capacity: usize,
    total_attention_budget: f32,
    inhibition_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct AttentionEvent {
    pub timestamp: Instant,
    pub event_type: AttentionEventType,
    pub entity_id: EntityId,
    pub attention_change: f32,
}

#[derive(Debug, Clone)]
pub enum AttentionEventType {
    Focus,
    Unfocus,
    Strengthen,
    Weaken,
    Maintain,
}
```

### Step 2: Attention Focus Implementation

```rust
impl AttentionFocus {
    pub fn new() -> Self {
        Self {
            targets: HashMap::new(),
            focus_history: Vec::new(),
            current_state: AttentionState::Scanning { 
                search_query: String::new() 
            },
            working_memory_capacity: 7, // Miller's magic number
            total_attention_budget: 1.0,
            inhibition_threshold: 0.1,
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            working_memory_capacity: capacity.clamp(3, 12), // Realistic range
            ..Self::new()
        }
    }
    
    pub fn focus_on(&mut self, entity_id: EntityId, source: AttentionSource) -> Result<(), AttentionError> {
        // Check capacity constraints
        if self.targets.len() >= self.working_memory_capacity {
            if !self.try_make_space(entity_id, &source)? {
                return Err(AttentionError::CapacityExceeded);
            }
        }
        
        let attention_strength = self.calculate_initial_attention(&source);
        
        let target = AttentionTarget {
            entity_id,
            attention_strength,
            source,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            persistence: Duration::from_secs(30), // Default persistence
            decay_rate: 0.1, // Attention decay per second
        };
        
        self.targets.insert(entity_id, target);
        self.log_attention_event(AttentionEventType::Focus, entity_id, attention_strength);
        self.update_attention_state();
        
        Ok(())
    }
    
    fn calculate_initial_attention(&self, source: &AttentionSource) -> f32 {
        match source {
            AttentionSource::TopDown { priority, .. } => {
                (*priority * 0.8).clamp(0.0, 1.0)
            },
            AttentionSource::BottomUp { stimulus_strength, novelty } => {
                (stimulus_strength * 0.7 + novelty * 0.3).clamp(0.0, 1.0)
            },
            AttentionSource::Maintenance { importance } => {
                (*importance * 0.6).clamp(0.0, 1.0)
            },
        }
    }
    
    fn try_make_space(&mut self, new_entity: EntityId, new_source: &AttentionSource) -> Result<bool, AttentionError> {
        let new_importance = self.calculate_source_importance(new_source);
        
        // Find weakest target
        if let Some((weakest_id, weakest_strength)) = self.find_weakest_target() {
            if new_importance > weakest_strength {
                self.unfocus(weakest_id);
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    fn find_weakest_target(&self) -> Option<(EntityId, f32)> {
        self.targets.iter()
            .min_by(|a, b| a.1.attention_strength.partial_cmp(&b.1.attention_strength).unwrap())
            .map(|(id, target)| (*id, target.attention_strength))
    }
    
    fn calculate_source_importance(&self, source: &AttentionSource) -> f32 {
        match source {
            AttentionSource::TopDown { priority, .. } => *priority,
            AttentionSource::BottomUp { stimulus_strength, novelty } => {
                stimulus_strength + novelty * 0.5
            },
            AttentionSource::Maintenance { importance } => *importance * 0.8,
        }
    }
}
```

### Step 3: Attention Dynamics and Updates

```rust
impl AttentionFocus {
    pub fn update_attention(&mut self, delta_time: Duration) {
        let dt = delta_time.as_secs_f32();
        let mut to_remove = Vec::new();
        
        // Apply attention decay
        for (entity_id, target) in self.targets.iter_mut() {
            // Exponential decay
            target.attention_strength *= (-target.decay_rate * dt).exp();
            
            // Check if attention fell below threshold
            if target.attention_strength < self.inhibition_threshold {
                to_remove.push(*entity_id);
            }
            
            // Check persistence timeout
            if target.created_at.elapsed() > target.persistence {
                to_remove.push(*entity_id);
            }
        }
        
        // Remove weak targets
        for entity_id in to_remove {
            self.unfocus(entity_id);
        }
        
        // Apply winner-take-all competition
        self.apply_competition();
        
        // Update attention state
        self.update_attention_state();
    }
    
    fn apply_competition(&mut self) {
        if self.targets.len() <= 1 {
            return;
        }
        
        // Normalize attention strengths to enforce budget constraint
        let total_attention: f32 = self.targets.values()
            .map(|t| t.attention_strength)
            .sum();
        
        if total_attention > self.total_attention_budget {
            let normalization_factor = self.total_attention_budget / total_attention;
            
            for target in self.targets.values_mut() {
                target.attention_strength *= normalization_factor;
            }
        }
        
        // Winner-take-all enhancement for dominant target
        if let Some((dominant_id, max_strength)) = self.find_dominant_target() {
            if max_strength > 0.6 { // Strong dominance threshold
                self.enhance_dominant_target(dominant_id, 0.1);
                self.suppress_competing_targets(dominant_id, 0.05);
            }
        }
    }
    
    fn find_dominant_target(&self) -> Option<(EntityId, f32)> {
        self.targets.iter()
            .max_by(|a, b| a.1.attention_strength.partial_cmp(&b.1.attention_strength).unwrap())
            .map(|(id, target)| (*id, target.attention_strength))
    }
    
    fn enhance_dominant_target(&mut self, dominant_id: EntityId, enhancement: f32) {
        if let Some(target) = self.targets.get_mut(&dominant_id) {
            target.attention_strength = (target.attention_strength + enhancement).clamp(0.0, 1.0);
            target.last_accessed = Instant::now();
        }
    }
    
    fn suppress_competing_targets(&mut self, dominant_id: EntityId, suppression: f32) {
        for (entity_id, target) in self.targets.iter_mut() {
            if *entity_id != dominant_id {
                target.attention_strength = (target.attention_strength - suppression).max(0.0);
            }
        }
    }
    
    pub fn unfocus(&mut self, entity_id: EntityId) {
        if let Some(target) = self.targets.remove(&entity_id) {
            self.log_attention_event(
                AttentionEventType::Unfocus, 
                entity_id, 
                -target.attention_strength
            );
            self.update_attention_state();
        }
    }
    
    fn update_attention_state(&mut self) {
        let target_count = self.targets.len();
        
        self.current_state = if target_count == 0 {
            AttentionState::Scanning { search_query: String::new() }
        } else if target_count > self.working_memory_capacity {
            AttentionState::Overloaded { 
                exceeding_by: target_count - self.working_memory_capacity 
            }
        } else if let Some((dominant_id, strength)) = self.find_dominant_target() {
            if strength > 0.7 && target_count == 1 {
                AttentionState::Focused { dominant_target: dominant_id }
            } else {
                AttentionState::Divided { target_count }
            }
        } else {
            AttentionState::Divided { target_count }
        };
    }
    
    fn log_attention_event(&mut self, event_type: AttentionEventType, entity_id: EntityId, change: f32) {
        let event = AttentionEvent {
            timestamp: Instant::now(),
            event_type,
            entity_id,
            attention_change: change,
        };
        
        self.focus_history.push(event);
        
        // Keep history manageable
        if self.focus_history.len() > 1000 {
            self.focus_history.drain(..500);
        }
    }
}
```

### Step 4: Query Interface and Analysis

```rust
impl AttentionFocus {
    pub fn get_current_targets(&self) -> Vec<(EntityId, f32)> {
        self.targets.iter()
            .map(|(id, target)| (*id, target.attention_strength))
            .collect()
    }
    
    pub fn get_dominant_target(&self) -> Option<EntityId> {
        self.find_dominant_target().map(|(id, _)| id)
    }
    
    pub fn get_attention_strength(&self, entity_id: EntityId) -> f32 {
        self.targets.get(&entity_id)
            .map(|t| t.attention_strength)
            .unwrap_or(0.0)
    }
    
    pub fn is_attending_to(&self, entity_id: EntityId) -> bool {
        self.targets.contains_key(&entity_id)
    }
    
    pub fn get_current_state(&self) -> &AttentionState {
        &self.current_state
    }
    
    pub fn get_capacity_utilization(&self) -> f32 {
        self.targets.len() as f32 / self.working_memory_capacity as f32
    }
    
    pub fn get_attention_distribution(&self) -> HashMap<EntityId, f32> {
        self.targets.iter()
            .map(|(id, target)| (*id, target.attention_strength))
            .collect()
    }
    
    pub fn boost_attention(&mut self, entity_id: EntityId, boost: f32) {
        if let Some(target) = self.targets.get_mut(&entity_id) {
            target.attention_strength = (target.attention_strength + boost).clamp(0.0, 1.0);
            target.last_accessed = Instant::now();
            self.log_attention_event(AttentionEventType::Strengthen, entity_id, boost);
        }
    }
    
    pub fn clear_all_attention(&mut self) {
        let entity_ids: Vec<EntityId> = self.targets.keys().copied().collect();
        for entity_id in entity_ids {
            self.unfocus(entity_id);
        }
    }
}

#[derive(Debug, Clone)]
pub enum AttentionError {
    CapacityExceeded,
    InvalidEntity,
    InvalidStrength,
}

impl std::fmt::Display for AttentionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttentionError::CapacityExceeded => write!(f, "Working memory capacity exceeded"),
            AttentionError::InvalidEntity => write!(f, "Invalid entity for attention"),
            AttentionError::InvalidStrength => write!(f, "Invalid attention strength"),
        }
    }
}

impl std::error::Error for AttentionError {}
```

## File Locations

- `src/cognitive/attention/focus_system.rs` - Main implementation
- `src/cognitive/attention/mod.rs` - Module exports
- `tests/cognitive/attention/focus_system_tests.rs` - Test implementation

## Success Criteria

- [ ] AttentionFocus struct compiles and runs
- [ ] Working memory capacity properly enforced (7±2 rule)
- [ ] Winner-take-all dynamics working correctly
- [ ] Attention decay follows exponential curves
- [ ] Top-down and bottom-up attention sources supported
- [ ] Thread-safe concurrent access
- [ ] All tests pass:
  - Capacity limit enforcement
  - Attention competition dynamics
  - Focus switching behavior
  - Decay and persistence mechanisms

## Test Requirements

```rust
#[test]
fn test_working_memory_capacity() {
    let mut focus = AttentionFocus::with_capacity(5);
    
    // Fill to capacity
    for i in 0..5 {
        let result = focus.focus_on(
            EntityId(i), 
            AttentionSource::TopDown { goal: "test".into(), priority: 0.5 }
        );
        assert!(result.is_ok());
    }
    
    // Should be at capacity
    assert_eq!(focus.get_current_targets().len(), 5);
    
    // Adding one more should succeed by removing weakest
    let result = focus.focus_on(
        EntityId(5), 
        AttentionSource::TopDown { goal: "strong".into(), priority: 0.9 }
    );
    assert!(result.is_ok());
    assert_eq!(focus.get_current_targets().len(), 5);
}

#[test]
fn test_winner_take_all_competition() {
    let mut focus = AttentionFocus::new();
    
    // Add competing targets
    focus.focus_on(
        EntityId(1), 
        AttentionSource::TopDown { goal: "weak".into(), priority: 0.3 }
    ).unwrap();
    
    focus.focus_on(
        EntityId(2), 
        AttentionSource::TopDown { goal: "strong".into(), priority: 0.8 }
    ).unwrap();
    
    // Apply competition
    focus.update_attention(Duration::from_millis(100));
    
    let strength1 = focus.get_attention_strength(EntityId(1));
    let strength2 = focus.get_attention_strength(EntityId(2));
    
    // Stronger target should dominate
    assert!(strength2 > strength1);
    
    // Check if focused state achieved
    if let AttentionState::Focused { dominant_target } = focus.get_current_state() {
        assert_eq!(*dominant_target, EntityId(2));
    }
}

#[test]
fn test_attention_decay() {
    let mut focus = AttentionFocus::new();
    
    focus.focus_on(
        EntityId(1), 
        AttentionSource::TopDown { goal: "test".into(), priority: 0.8 }
    ).unwrap();
    
    let initial_strength = focus.get_attention_strength(EntityId(1));
    
    // Apply decay over time
    focus.update_attention(Duration::from_secs(1));
    
    let decayed_strength = focus.get_attention_strength(EntityId(1));
    
    // Attention should have decayed
    assert!(decayed_strength < initial_strength);
    assert!(decayed_strength > 0.0); // But not disappeared
}

#[test]
fn test_bottom_up_attention_capture() {
    let mut focus = AttentionFocus::new();
    
    // Strong bottom-up stimulus
    focus.focus_on(
        EntityId(1), 
        AttentionSource::BottomUp { 
            stimulus_strength: 0.9, 
            novelty: 0.8 
        }
    ).unwrap();
    
    let attention_strength = focus.get_attention_strength(EntityId(1));
    assert!(attention_strength > 0.6); // Should be strong
    
    // Should achieve focused state quickly
    focus.update_attention(Duration::from_millis(50));
    
    if let AttentionState::Focused { dominant_target } = focus.get_current_state() {
        assert_eq!(*dominant_target, EntityId(1));
    }
}

#[test]
fn test_attention_state_transitions() {
    let mut focus = AttentionFocus::new();
    
    // Start in scanning state
    assert!(matches!(focus.get_current_state(), AttentionState::Scanning { .. }));
    
    // Add single target -> should become focused
    focus.focus_on(
        EntityId(1), 
        AttentionSource::TopDown { goal: "test".into(), priority: 0.8 }
    ).unwrap();
    
    focus.update_attention(Duration::from_millis(10));
    
    // Should be focused on single target
    assert!(matches!(focus.get_current_state(), AttentionState::Focused { .. }));
    
    // Add second target -> should become divided
    focus.focus_on(
        EntityId(2), 
        AttentionSource::TopDown { goal: "test2".into(), priority: 0.7 }
    ).unwrap();
    
    focus.update_attention(Duration::from_millis(10));
    
    // Should be in divided attention state
    assert!(matches!(focus.get_current_state(), AttentionState::Divided { .. }));
}
```

## Quality Gates

- [ ] Working memory capacity strictly enforced
- [ ] Attention budget never exceeded
- [ ] No memory leaks during continuous operation
- [ ] Thread-safe concurrent access verified
- [ ] Performance acceptable (< 1ms update cycles)
- [ ] Realistic attention dynamics (matches cognitive psychology)

## Next Task

Upon completion, proceed to **14_working_memory.md**