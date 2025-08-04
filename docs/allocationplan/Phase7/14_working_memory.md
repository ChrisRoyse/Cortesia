# Micro Task 14: Working Memory

**Priority**: CRITICAL  
**Estimated Time**: 40 minutes  
**Dependencies**: 13_attention_focus_system.md completed  
**Skills Required**: Cognitive modeling, memory simulation, capacity management

## Objective

Implement a brain-inspired working memory system that manages temporary information storage, capacity limits, and interference patterns following cognitive psychology principles.

## Context

Working memory is the cognitive system responsible for temporarily holding and manipulating information. It has strict capacity limits (Miller's 7±2 rule), temporal decay, and interference effects. This implementation models working memory as the active storage component of the attention system.

## Specifications

### Core Working Memory Components

1. **WorkingMemory struct**
   - Memory slots with capacity limits (3-9 items)
   - Temporal decay of stored information
   - Interference between similar items
   - Rehearsal and refreshing mechanisms

2. **MemoryItem struct**
   - Stored entity or concept
   - Activation strength (0.0-1.0)
   - Storage duration and last access
   - Similarity relationships

3. **MemorySlot struct**
   - Individual storage location
   - Capacity utilization tracking
   - Interference calculation
   - Access frequency monitoring

### Performance Requirements

- Support 7±2 concurrent memory items
- Item retrieval latency < 10ms
- Decay follows cognitive timing models
- Interference effects based on similarity
- Thread-safe concurrent access

## Implementation Guide

### Step 1: Core Working Memory Types

```rust
// File: src/cognitive/memory/working_memory.rs

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use crate::core::types::{EntityId, NodeId};
use crate::cognitive::attention::AttentionSource;

#[derive(Debug, Clone)]
pub struct MemoryItem {
    pub entity_id: EntityId,
    pub content: MemoryContent,
    pub activation_strength: f32,  // 0.0 to 1.0
    pub created_at: Instant,
    pub last_accessed: Instant,
    pub access_count: usize,
    pub rehearsal_count: usize,
    pub decay_rate: f32,
    pub interference_susceptibility: f32,
}

#[derive(Debug, Clone)]
pub enum MemoryContent {
    Entity { properties: HashMap<String, String> },
    Relationship { from: EntityId, to: EntityId, relation_type: String },
    Concept { name: String, attributes: Vec<String> },
    Episode { sequence: Vec<EntityId>, context: String },
    Goal { description: String, priority: f32 },
}

#[derive(Debug, Clone)]
pub struct MemorySlot {
    pub slot_id: usize,
    pub item: Option<MemoryItem>,
    pub utilization: f32,
    pub interference_level: f32,
    pub last_update: Instant,
}

#[derive(Debug)]
pub struct WorkingMemory {
    slots: Vec<MemorySlot>,
    capacity: usize,
    total_activation: f32,
    interference_matrix: HashMap<(usize, usize), f32>,
    rehearsal_buffer: VecDeque<EntityId>,
    decay_scheduler: DecayScheduler,
    access_history: Vec<MemoryAccess>,
}

#[derive(Debug, Clone)]
pub struct MemoryAccess {
    pub timestamp: Instant,
    pub slot_id: usize,
    pub access_type: AccessType,
    pub success: bool,
}

#[derive(Debug, Clone)]
pub enum AccessType {
    Store,
    Retrieve,
    Update,
    Rehearse,
    Decay,
}

#[derive(Debug)]
pub struct DecayScheduler {
    decay_events: VecDeque<DecayEvent>,
    next_decay_time: Option<Instant>,
}

#[derive(Debug, Clone)]
pub struct DecayEvent {
    pub target_slot: usize,
    pub scheduled_time: Instant,
    pub decay_amount: f32,
}
```

### Step 2: Working Memory Implementation

```rust
impl WorkingMemory {
    pub fn new() -> Self {
        Self::with_capacity(7) // Default to Miller's magic number
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        let clamped_capacity = capacity.clamp(3, 9); // Realistic cognitive limits
        let mut slots = Vec::with_capacity(clamped_capacity);
        
        for i in 0..clamped_capacity {
            slots.push(MemorySlot {
                slot_id: i,
                item: None,
                utilization: 0.0,
                interference_level: 0.0,
                last_update: Instant::now(),
            });
        }
        
        Self {
            slots,
            capacity: clamped_capacity,
            total_activation: 0.0,
            interference_matrix: HashMap::new(),
            rehearsal_buffer: VecDeque::new(),
            decay_scheduler: DecayScheduler::new(),
            access_history: Vec::new(),
        }
    }
    
    pub fn store(&mut self, entity_id: EntityId, content: MemoryContent, source: AttentionSource) -> Result<usize, MemoryError> {
        // Find available slot or least important item to replace
        let slot_id = self.find_available_slot()
            .or_else(|| self.find_replaceable_slot(&content))
            .ok_or(MemoryError::CapacityExceeded)?;
        
        // Calculate initial activation based on source
        let initial_activation = self.calculate_initial_activation(&source);
        
        // Create memory item
        let item = MemoryItem {
            entity_id,
            content,
            activation_strength: initial_activation,
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 1,
            rehearsal_count: 0,
            decay_rate: self.calculate_decay_rate(&source),
            interference_susceptibility: 0.5, // Default susceptibility
        };
        
        // Store in slot
        self.slots[slot_id].item = Some(item);
        self.slots[slot_id].utilization = initial_activation;
        self.slots[slot_id].last_update = Instant::now();
        
        // Update interference matrix
        self.update_interference_matrix(slot_id);
        
        // Schedule decay
        self.schedule_decay(slot_id);
        
        // Log access
        self.log_access(slot_id, AccessType::Store, true);
        
        // Update total activation
        self.update_total_activation();
        
        Ok(slot_id)
    }
    
    fn find_available_slot(&self) -> Option<usize> {
        self.slots.iter()
            .position(|slot| slot.item.is_none())
    }
    
    fn find_replaceable_slot(&self, new_content: &MemoryContent) -> Option<usize> {
        let new_importance = self.calculate_content_importance(new_content);
        
        self.slots.iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                slot.item.as_ref().map(|item| {
                    let current_importance = self.calculate_item_importance(item);
                    (i, current_importance)
                })
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .and_then(|(slot_id, importance)| {
                if new_importance > importance {
                    Some(slot_id)
                } else {
                    None
                }
            })
    }
    
    fn calculate_initial_activation(&self, source: &AttentionSource) -> f32 {
        match source {
            AttentionSource::TopDown { priority, .. } => {
                (*priority * 0.9).clamp(0.0, 1.0)
            },
            AttentionSource::BottomUp { stimulus_strength, novelty } => {
                (stimulus_strength * 0.7 + novelty * 0.2).clamp(0.0, 1.0)
            },
            AttentionSource::Maintenance { importance } => {
                (*importance * 0.8).clamp(0.0, 1.0)
            },
        }
    }
    
    fn calculate_decay_rate(&self, source: &AttentionSource) -> f32 {
        match source {
            AttentionSource::TopDown { .. } => 0.05, // Slower decay for goal-directed
            AttentionSource::BottomUp { .. } => 0.15, // Faster decay for stimulus-driven
            AttentionSource::Maintenance { .. } => 0.03, // Slowest decay for maintained items
        }
    }
    
    fn calculate_content_importance(&self, content: &MemoryContent) -> f32 {
        match content {
            MemoryContent::Goal { priority, .. } => *priority,
            MemoryContent::Entity { .. } => 0.5,
            MemoryContent::Relationship { .. } => 0.6,
            MemoryContent::Concept { .. } => 0.4,
            MemoryContent::Episode { .. } => 0.7,
        }
    }
    
    fn calculate_item_importance(&self, item: &MemoryItem) -> f32 {
        let base_importance = self.calculate_content_importance(&item.content);
        let recency_bonus = (-item.last_accessed.elapsed().as_secs_f32() / 60.0).exp() * 0.2;
        let frequency_bonus = (item.access_count as f32).ln() * 0.1;
        
        base_importance + recency_bonus + frequency_bonus
    }
}
```

### Step 3: Memory Retrieval and Interference

```rust
impl WorkingMemory {
    pub fn retrieve(&mut self, entity_id: EntityId) -> Option<&MemoryItem> {
        let slot_id = self.find_item_slot(entity_id)?;
        
        if let Some(ref mut item) = self.slots[slot_id].item {
            // Update access information
            item.last_accessed = Instant::now();
            item.access_count += 1;
            
            // Apply retrieval boost
            item.activation_strength = (item.activation_strength + 0.1).clamp(0.0, 1.0);
            
            // Update slot information
            self.slots[slot_id].utilization = item.activation_strength;
            self.slots[slot_id].last_update = Instant::now();
            
            // Log successful retrieval
            self.log_access(slot_id, AccessType::Retrieve, true);
            
            Some(item)
        } else {
            // Log failed retrieval
            self.log_access(slot_id, AccessType::Retrieve, false);
            None
        }
    }
    
    fn find_item_slot(&self, entity_id: EntityId) -> Option<usize> {
        self.slots.iter()
            .position(|slot| {
                slot.item.as_ref()
                    .map(|item| item.entity_id == entity_id)
                    .unwrap_or(false)
            })
    }
    
    pub fn rehearse(&mut self, entity_id: EntityId) -> Result<(), MemoryError> {
        let slot_id = self.find_item_slot(entity_id)
            .ok_or(MemoryError::ItemNotFound)?;
        
        if let Some(ref mut item) = self.slots[slot_id].item {
            // Rehearsal strengthens memory and reduces decay
            item.rehearsal_count += 1;
            item.activation_strength = (item.activation_strength + 0.05).clamp(0.0, 1.0);
            item.decay_rate *= 0.9; // Reduce decay rate
            item.last_accessed = Instant::now();
            
            // Add to rehearsal buffer for maintenance
            self.rehearsal_buffer.push_back(entity_id);
            if self.rehearsal_buffer.len() > 3 {
                self.rehearsal_buffer.pop_front();
            }
            
            // Reschedule decay
            self.schedule_decay(slot_id);
            
            self.log_access(slot_id, AccessType::Rehearse, true);
            
            Ok(())
        } else {
            Err(MemoryError::ItemNotFound)
        }
    }
    
    fn update_interference_matrix(&mut self, target_slot: usize) {
        for i in 0..self.capacity {
            if i != target_slot {
                let interference = self.calculate_interference(target_slot, i);
                self.interference_matrix.insert((target_slot, i), interference);
                self.interference_matrix.insert((i, target_slot), interference);
            }
        }
        
        // Apply interference effects
        self.apply_interference_effects(target_slot);
    }
    
    fn calculate_interference(&self, slot1: usize, slot2: usize) -> f32 {
        let item1 = self.slots[slot1].item.as_ref();
        let item2 = self.slots[slot2].item.as_ref();
        
        match (item1, item2) {
            (Some(i1), Some(i2)) => {
                self.calculate_similarity(&i1.content, &i2.content)
            },
            _ => 0.0,
        }
    }
    
    fn calculate_similarity(&self, content1: &MemoryContent, content2: &MemoryContent) -> f32 {
        match (content1, content2) {
            (MemoryContent::Entity { .. }, MemoryContent::Entity { .. }) => 0.3,
            (MemoryContent::Relationship { .. }, MemoryContent::Relationship { .. }) => 0.4,
            (MemoryContent::Concept { .. }, MemoryContent::Concept { .. }) => 0.5,
            (MemoryContent::Goal { .. }, MemoryContent::Goal { .. }) => 0.6,
            (MemoryContent::Episode { .. }, MemoryContent::Episode { .. }) => 0.2,
            _ => 0.1, // Different types have low similarity
        }
    }
    
    fn apply_interference_effects(&mut self, target_slot: usize) {
        for i in 0..self.capacity {
            if i != target_slot {
                if let Some(interference) = self.interference_matrix.get(&(target_slot, i)) {
                    if let Some(ref mut item) = self.slots[i].item {
                        // Reduce activation due to interference
                        let interference_reduction = interference * 0.1;
                        item.activation_strength = (item.activation_strength - interference_reduction).max(0.0);
                        
                        // Update slot interference level
                        self.slots[i].interference_level = *interference;
                    }
                }
            }
        }
    }
}
```

### Step 4: Decay and Maintenance

```rust
impl WorkingMemory {
    pub fn update(&mut self, delta_time: Duration) {
        let dt = delta_time.as_secs_f32();
        
        // Process decay events
        self.process_decay_events();
        
        // Apply natural decay to all items
        for slot in &mut self.slots {
            if let Some(ref mut item) = slot.item {
                // Exponential decay based on time
                let decay_factor = (-item.decay_rate * dt).exp();
                item.activation_strength *= decay_factor;
                
                // Apply interference-based decay
                let interference_decay = slot.interference_level * 0.05 * dt;
                item.activation_strength = (item.activation_strength - interference_decay).max(0.0);
                
                // Update slot utilization
                slot.utilization = item.activation_strength;
                
                // Remove items below threshold
                if item.activation_strength < 0.05 {
                    slot.item = None;
                    slot.utilization = 0.0;
                    slot.interference_level = 0.0;
                }
            }
        }
        
        // Automatic rehearsal from buffer
        self.process_rehearsal_buffer();
        
        // Update total activation
        self.update_total_activation();
    }
    
    fn process_decay_events(&mut self) {
        let now = Instant::now();
        
        while let Some(event) = self.decay_scheduler.decay_events.front() {
            if event.scheduled_time <= now {
                let event = self.decay_scheduler.decay_events.pop_front().unwrap();
                self.apply_decay_event(event);
            } else {
                break;
            }
        }
    }
    
    fn apply_decay_event(&mut self, event: DecayEvent) {
        if let Some(ref mut item) = self.slots[event.target_slot].item {
            item.activation_strength = (item.activation_strength - event.decay_amount).max(0.0);
            self.slots[event.target_slot].utilization = item.activation_strength;
            
            self.log_access(event.target_slot, AccessType::Decay, true);
        }
    }
    
    fn schedule_decay(&mut self, slot_id: usize) {
        if let Some(ref item) = self.slots[slot_id].item {
            let decay_delay = Duration::from_secs_f32(1.0 / item.decay_rate);
            let decay_time = Instant::now() + decay_delay;
            
            let decay_event = DecayEvent {
                target_slot: slot_id,
                scheduled_time: decay_time,
                decay_amount: 0.1, // Standard decay amount
            };
            
            self.decay_scheduler.decay_events.push_back(decay_event);
        }
    }
    
    fn process_rehearsal_buffer(&mut self) {
        // Automatic maintenance of rehearsed items
        for entity_id in self.rehearsal_buffer.clone() {
            if let Some(slot_id) = self.find_item_slot(entity_id) {
                if let Some(ref mut item) = self.slots[slot_id].item {
                    // Small boost for rehearsed items
                    item.activation_strength = (item.activation_strength + 0.02).clamp(0.0, 1.0);
                }
            }
        }
    }
    
    fn update_total_activation(&mut self) {
        self.total_activation = self.slots.iter()
            .filter_map(|slot| slot.item.as_ref())
            .map(|item| item.activation_strength)
            .sum();
    }
    
    fn log_access(&mut self, slot_id: usize, access_type: AccessType, success: bool) {
        let access = MemoryAccess {
            timestamp: Instant::now(),
            slot_id,
            access_type,
            success,
        };
        
        self.access_history.push(access);
        
        // Keep history manageable
        if self.access_history.len() > 1000 {
            self.access_history.drain(..500);
        }
    }
}

impl DecayScheduler {
    fn new() -> Self {
        Self {
            decay_events: VecDeque::new(),
            next_decay_time: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum MemoryError {
    CapacityExceeded,
    ItemNotFound,
    InvalidActivation,
    InterferenceOverload,
}

impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryError::CapacityExceeded => write!(f, "Working memory capacity exceeded"),
            MemoryError::ItemNotFound => write!(f, "Memory item not found"),
            MemoryError::InvalidActivation => write!(f, "Invalid activation level"),
            MemoryError::InterferenceOverload => write!(f, "Interference level too high"),
        }
    }
}

impl std::error::Error for MemoryError {}
```

## File Locations

- `src/cognitive/memory/working_memory.rs` - Main implementation
- `src/cognitive/memory/mod.rs` - Module exports  
- `tests/cognitive/memory/working_memory_tests.rs` - Test implementation

## Success Criteria

- [ ] WorkingMemory struct compiles and runs
- [ ] Capacity limits properly enforced (7±2 rule)
- [ ] Interference effects implemented correctly
- [ ] Temporal decay follows cognitive models
- [ ] Rehearsal mechanisms functional
- [ ] Thread-safe concurrent access
- [ ] All tests pass:
  - Capacity enforcement
  - Interference calculations
  - Decay and maintenance
  - Retrieval accuracy

## Test Requirements

```rust
#[test]
fn test_capacity_limits() {
    let mut memory = WorkingMemory::with_capacity(5);
    
    // Fill to capacity
    for i in 0..5 {
        let result = memory.store(
            EntityId(i),
            MemoryContent::Entity { properties: HashMap::new() },
            AttentionSource::TopDown { goal: "test".into(), priority: 0.5 }
        );
        assert!(result.is_ok());
    }
    
    // Should have 5 items
    assert_eq!(memory.get_active_item_count(), 5);
    
    // Adding 6th should replace weakest
    let result = memory.store(
        EntityId(5),
        MemoryContent::Goal { description: "important".into(), priority: 0.9 },
        AttentionSource::TopDown { goal: "priority".into(), priority: 0.9 }
    );
    assert!(result.is_ok());
    assert_eq!(memory.get_active_item_count(), 5);
}

#[test]
fn test_interference_effects() {
    let mut memory = WorkingMemory::new();
    
    // Store similar items
    memory.store(
        EntityId(1),
        MemoryContent::Concept { name: "dog".into(), attributes: vec!["animal".into()] },
        AttentionSource::TopDown { goal: "test".into(), priority: 0.7 }
    ).unwrap();
    
    let initial_strength = memory.retrieve(EntityId(1)).unwrap().activation_strength;
    
    memory.store(
        EntityId(2),
        MemoryContent::Concept { name: "cat".into(), attributes: vec!["animal".into()] },
        AttentionSource::TopDown { goal: "test".into(), priority: 0.7 }
    ).unwrap();
    
    // First item should have reduced activation due to interference
    let reduced_strength = memory.retrieve(EntityId(1)).unwrap().activation_strength;
    assert!(reduced_strength <= initial_strength);
}

#[test]
fn test_decay_and_rehearsal() {
    let mut memory = WorkingMemory::new();
    
    memory.store(
        EntityId(1),
        MemoryContent::Entity { properties: HashMap::new() },
        AttentionSource::TopDown { goal: "test".into(), priority: 0.8 }
    ).unwrap();
    
    let initial_strength = memory.retrieve(EntityId(1)).unwrap().activation_strength;
    
    // Apply decay
    memory.update(Duration::from_secs(1));
    
    let decayed_strength = memory.retrieve(EntityId(1)).unwrap().activation_strength;
    assert!(decayed_strength < initial_strength);
    
    // Rehearse to strengthen
    memory.rehearse(EntityId(1)).unwrap();
    
    let rehearsed_strength = memory.retrieve(EntityId(1)).unwrap().activation_strength;
    assert!(rehearsed_strength > decayed_strength);
}

#[test]
fn test_retrieval_boost() {
    let mut memory = WorkingMemory::new();
    
    memory.store(
        EntityId(1),
        MemoryContent::Entity { properties: HashMap::new() },
        AttentionSource::TopDown { goal: "test".into(), priority: 0.5 }
    ).unwrap();
    
    let initial_strength = memory.retrieve(EntityId(1)).unwrap().activation_strength;
    
    // Multiple retrievals should boost activation
    memory.retrieve(EntityId(1));
    memory.retrieve(EntityId(1));
    
    let boosted_strength = memory.retrieve(EntityId(1)).unwrap().activation_strength;
    assert!(boosted_strength > initial_strength);
}
```

## Quality Gates

- [ ] Capacity never exceeded beyond configured limits
- [ ] Interference calculations mathematically sound  
- [ ] Decay follows exponential curves
- [ ] Memory efficient for continuous operation
- [ ] Thread-safe concurrent access verified
- [ ] Retrieval performance < 10ms average

## Next Task

Upon completion, proceed to **15_attention_weighting.md**