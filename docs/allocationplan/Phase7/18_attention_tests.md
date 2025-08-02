# Micro Task 18: Attention System Tests

**Priority**: CRITICAL  
**Estimated Time**: 45 minutes  
**Dependencies**: 17_salience_calculation.md completed  
**Skills Required**: Testing frameworks, cognitive validation, integration testing

## Objective

Implement comprehensive test suite for the attention system that validates cognitive realism, performance characteristics, and integration between attention components.

## Context

The attention system tests must verify not only functional correctness but also cognitive plausibility. Tests should validate that attention behaviors match psychological research findings, including Miller's 7±2 rule, switching costs, inhibition of return, and salience-driven capture.

## Specifications

### Test Categories

1. **Unit Tests**
   - Individual component functionality
   - Boundary condition handling
   - Error case validation
   - Performance benchmarks

2. **Integration Tests**
   - Component interaction validation
   - End-to-end attention workflows
   - Cross-component data consistency
   - Resource management verification

3. **Cognitive Validation Tests**
   - Psychological realism checks
   - Attention capacity limits
   - Switching cost validation
   - Salience behavior verification

### Test Requirements

- Test coverage > 90% for attention modules
- Performance benchmarks for real-time operation
- Cognitive behavior validation against literature
- Stress testing for continuous operation
- Memory leak detection
- Thread safety verification

## Implementation Guide

### Step 1: Test Infrastructure Setup

```rust
// File: tests/cognitive/attention/mod.rs

pub mod attention_integration_tests;
pub mod focus_system_tests;
pub mod working_memory_tests;
pub mod weighting_tests;
pub mod switching_tests;
pub mod salience_tests;
pub mod cognitive_validation_tests;
pub mod performance_benchmarks;

use std::time::{Duration, Instant};
use std::collections::HashMap;
use crate::cognitive::attention::*;
use crate::core::types::EntityId;

// Test utilities and helpers
pub struct AttentionTestHarness {
    pub focus_system: AttentionFocus,
    pub working_memory: WorkingMemory,
    pub weight_calculator: AttentionWeightCalculator,
    pub focus_switcher: FocusSwitcher,
    pub salience_calculator: SalienceCalculator,
}

impl AttentionTestHarness {
    pub fn new() -> Self {
        Self {
            focus_system: AttentionFocus::new(),
            working_memory: WorkingMemory::new(),
            weight_calculator: AttentionWeightCalculator::new(
                WeightingStrategy::Linear { weights: vec![1.0] }
            ),
            focus_switcher: FocusSwitcher::new(
                SwitchingPolicy::Threshold { switch_threshold: 0.3 }
            ),
            salience_calculator: SalienceCalculator::new(),
        }
    }
    
    pub fn create_test_targets(&self, count: usize) -> Vec<AttentionTarget> {
        (0..count).map(|i| {
            AttentionTarget {
                entity_id: EntityId(i),
                attention_strength: 0.5 + (i as f32 * 0.1),
                source: AttentionSource::TopDown { 
                    goal: format!("goal_{}", i), 
                    priority: 0.5 + (i as f32 * 0.1) 
                },
                created_at: Instant::now(),
                last_accessed: Instant::now(),
                persistence: Duration::from_secs(30),
                decay_rate: 0.1,
            }
        }).collect()
    }
    
    pub fn create_test_context(&self) -> ContextState {
        ContextState {
            current_goal: Some("test_goal".to_string()),
            task_difficulty: 0.5,
            cognitive_load: 0.3,
            environmental_noise: 0.2,
            time_pressure: 0.1,
            domain_context: "test".to_string(),
        }
    }
}

// Test assertion helpers
pub fn assert_attention_sum_normalized(weights: &HashMap<EntityId, f32>, tolerance: f32) {
    let sum: f32 = weights.values().sum();
    assert!((sum - 1.0).abs() < tolerance, "Attention weights sum to {}, expected ~1.0", sum);
}

pub fn assert_capacity_limit_respected(active_count: usize, capacity: usize) {
    assert!(active_count <= capacity, "Active count {} exceeds capacity {}", active_count, capacity);
}

pub fn assert_cognitive_realistic_timing(duration: Duration, expected_min: Duration, expected_max: Duration) {
    assert!(duration >= expected_min && duration <= expected_max, 
           "Duration {:?} outside realistic range {:?} - {:?}", duration, expected_min, expected_max);
}
```

### Step 2: Integration Tests

```rust
// File: tests/cognitive/attention/attention_integration_tests.rs

use super::*;
use std::thread;
use std::sync::{Arc, Mutex};

#[test]
fn test_full_attention_pipeline() {
    let mut harness = AttentionTestHarness::new();
    
    // Step 1: Create stimuli with different salience levels
    let high_salience_stimulus = StimulusProperties {
        intensity: 0.9,
        size_ratio: 0.8,
        color_distinctiveness: 0.7,
        color_saturation: 0.9,
        semantic_unexpectedness: 0.6,
        position: (0.5, 0.5),
        timestamp: Instant::now(),
    };
    
    let low_salience_stimulus = StimulusProperties {
        intensity: 0.3,
        size_ratio: 0.2,
        color_distinctiveness: 0.1,
        color_saturation: 0.2,
        semantic_unexpectedness: 0.1,
        position: (0.3, 0.3),
        timestamp: Instant::now(),
    };
    
    let context = harness.create_test_context();
    
    // Step 2: Calculate salience
    let high_salience_record = harness.salience_calculator.calculate_salience(
        EntityId(1), &high_salience_stimulus, &SalienceContext::default()
    );
    
    let low_salience_record = harness.salience_calculator.calculate_salience(
        EntityId(2), &low_salience_stimulus, &SalienceContext::default()
    );
    
    // Step 3: Create attention targets from salience
    let targets = vec![
        AttentionTarget {
            entity_id: EntityId(1),
            attention_strength: high_salience_record.total_salience,
            source: AttentionSource::BottomUp { 
                stimulus_strength: high_salience_record.total_salience,
                novelty: high_salience_record.feature_contributions.novelty,
            },
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            persistence: Duration::from_secs(30),
            decay_rate: 0.1,
        },
        AttentionTarget {
            entity_id: EntityId(2),
            attention_strength: low_salience_record.total_salience,
            source: AttentionSource::BottomUp { 
                stimulus_strength: low_salience_record.total_salience,
                novelty: low_salience_record.feature_contributions.novelty,
            },
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            persistence: Duration::from_secs(30),
            decay_rate: 0.1,
        },
    ];
    
    // Step 4: Calculate attention weights
    let weights = harness.weight_calculator.calculate_attention_weights(&targets, &context);
    
    // Step 5: Apply focus switching logic
    let candidate_targets: Vec<(EntityId, f32)> = weights.into_iter().collect();
    let switching_decision = harness.focus_switcher.evaluate_switching_decision(&candidate_targets, &context);
    
    // Step 6: Update working memory if switching occurs
    match switching_decision {
        SwitchingDecision::Switch { to_entity, reason } => {
            let switching_cost = harness.focus_switcher.execute_switch(to_entity, reason, &context).unwrap();
            
            // Store in working memory
            let memory_content = MemoryContent::Entity { properties: HashMap::new() };
            let result = harness.working_memory.store(
                to_entity, 
                memory_content, 
                AttentionSource::TopDown { goal: "test".to_string(), priority: 0.8 }
            );
            
            assert!(result.is_ok());
            
            // Verify attention focused on high salience stimulus
            assert_eq!(to_entity, EntityId(1), "Should focus on high salience stimulus");
            
            // Verify switching cost is reasonable
            assert!(switching_cost.time_penalty < Duration::from_millis(1000));
        },
        _ => {
            // If no switching occurs, verify it's for a good reason
            assert!(candidate_targets.is_empty() || candidate_targets[0].1 < 0.1);
        }
    }
}

#[test]
fn test_attention_overload_handling() {
    let mut harness = AttentionTestHarness::new();
    let context = harness.create_test_context();
    
    // Create more targets than working memory capacity
    let excess_targets = harness.create_test_targets(12); // Exceeds typical 7±2 capacity
    
    let mut successful_focuses = 0;
    
    for target in excess_targets {
        let result = harness.focus_system.focus_on(target.entity_id, target.source);
        if result.is_ok() {
            successful_focuses += 1;
        }
    }
    
    // Should not exceed capacity
    assert!(successful_focuses <= 9); // Upper bound of 7±2
    
    // System should be in overloaded state if we tried to add too many
    if successful_focuses >= 9 {
        let current_state = harness.focus_system.get_current_state();
        assert!(matches!(current_state, AttentionState::Overloaded { .. }));
    }
}

#[test]
fn test_cross_component_consistency() {
    let mut harness = AttentionTestHarness::new();
    let context = harness.create_test_context();
    
    // Add target to focus system
    let entity_id = EntityId(1);
    let source = AttentionSource::TopDown { goal: "test".to_string(), priority: 0.8 };
    
    harness.focus_system.focus_on(entity_id, source.clone()).unwrap();
    
    // Add same entity to working memory
    let memory_content = MemoryContent::Entity { properties: HashMap::new() };
    harness.working_memory.store(entity_id, memory_content, source).unwrap();
    
    // Verify consistency
    assert!(harness.focus_system.is_attending_to(entity_id));
    assert!(harness.working_memory.retrieve(entity_id).is_some());
    
    // Update both systems and verify they remain consistent
    harness.focus_system.update_attention(Duration::from_millis(100));
    harness.working_memory.update(Duration::from_millis(100));
    
    let focus_strength = harness.focus_system.get_attention_strength(entity_id);
    let memory_item = harness.working_memory.retrieve(entity_id);
    
    if focus_strength > 0.0 {
        assert!(memory_item.is_some(), "Entity in focus should also be in working memory");
    }
}

#[test]
fn test_concurrent_attention_operations() {
    let harness = Arc::new(Mutex::new(AttentionTestHarness::new()));
    let mut handles = vec![];
    
    // Spawn multiple threads performing attention operations
    for i in 0..4 {
        let harness_clone = Arc::clone(&harness);
        let handle = thread::spawn(move || {
            let entity_id = EntityId(i);
            let source = AttentionSource::TopDown { 
                goal: format!("goal_{}", i), 
                priority: 0.5 + (i as f32 * 0.1) 
            };
            
            // Acquire lock and perform operations
            {
                let mut h = harness_clone.lock().unwrap();
                let _ = h.focus_system.focus_on(entity_id, source.clone());
                
                let memory_content = MemoryContent::Entity { properties: HashMap::new() };
                let _ = h.working_memory.store(entity_id, memory_content, source);
            }
            
            // Simulate some processing time
            thread::sleep(Duration::from_millis(10));
            
            // Check results
            {
                let h = harness_clone.lock().unwrap();
                h.focus_system.is_attending_to(entity_id)
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    let results: Vec<bool> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    
    // Verify no crashes and some successes
    assert!(results.iter().any(|&success| success), "At least one thread should succeed");
    
    // Verify system state is consistent
    let harness = harness.lock().unwrap();
    let active_targets = harness.focus_system.get_current_targets();
    assert!(active_targets.len() <= 9); // Capacity limit respected
}
```

### Step 3: Cognitive Validation Tests

```rust
// File: tests/cognitive/attention/cognitive_validation_tests.rs

use super::*;

#[test]
fn test_millers_magic_number() {
    let mut working_memory = WorkingMemory::with_capacity(7);
    let context = ContextState::default();
    
    // Fill working memory to capacity
    for i in 0..7 {
        let result = working_memory.store(
            EntityId(i),
            MemoryContent::Entity { properties: HashMap::new() },
            AttentionSource::TopDown { goal: "test".to_string(), priority: 0.5 }
        );
        assert!(result.is_ok(), "Should be able to store {} items", i + 1);
    }
    
    // Verify capacity is at limit
    assert_eq!(working_memory.get_active_item_count(), 7);
    
    // Adding 8th item should either fail or replace existing
    let result = working_memory.store(
        EntityId(7),
        MemoryContent::Goal { description: "priority".to_string(), priority: 0.9 },
        AttentionSource::TopDown { goal: "priority".to_string(), priority: 0.9 }
    );
    
    if result.is_ok() {
        // If successful, verify capacity still not exceeded
        assert!(working_memory.get_active_item_count() <= 7);
    }
}

#[test]
fn test_switching_cost_validation() {
    let mut switcher = FocusSwitcher::new(
        SwitchingPolicy::Threshold { switch_threshold: 0.3 }
    );
    
    let context = ContextState {
        cognitive_load: 0.8, // High cognitive load
        time_pressure: 0.7,  // High time pressure
        ..Default::default()
    };
    
    // Perform a task switch (should be expensive)
    let cost = switcher.execute_switch(
        EntityId(1), 
        SwitchReason::GoalChange, 
        &context
    ).unwrap();
    
    // Validate switching cost matches cognitive research
    assert!(cost.time_penalty >= Duration::from_millis(100), 
           "Task switching should have significant time cost");
    assert!(cost.cognitive_load_increase > 0.2, 
           "Task switching should increase cognitive load");
    assert!(cost.accuracy_reduction > 0.0, 
           "Task switching should reduce accuracy");
    
    // High cognitive load should increase costs
    let low_load_context = ContextState {
        cognitive_load: 0.2,
        time_pressure: 0.1,
        ..Default::default()
    };
    
    let low_load_cost = switcher.calculate_switching_cost(
        Some(EntityId(1)),
        EntityId(2),
        &SwitchReason::GoalChange,
        &low_load_context
    );
    
    assert!(cost.time_penalty > low_load_cost.time_penalty,
           "High cognitive load should increase switching costs");
}

#[test]
fn test_inhibition_of_return_timing() {
    let mut switcher = FocusSwitcher::new(SwitchingPolicy::Immediate);
    let context = ContextState::default();
    
    // Switch from A to B
    switcher.execute_switch(EntityId(1), SwitchReason::TaskCompletion, &context).unwrap();
    switcher.execute_switch(EntityId(2), SwitchReason::WeightThresholdExceeded, &context).unwrap();
    
    // Check immediate inhibition
    let immediate_inhibition = switcher.get_inhibition_strength(EntityId(1));
    assert!(immediate_inhibition > 0.5, "Should have strong immediate inhibition");
    
    // Simulate time passing
    switcher.update_inhibition(Duration::from_millis(300));
    let early_inhibition = switcher.get_inhibition_strength(EntityId(1));
    
    switcher.update_inhibition(Duration::from_secs(2));
    let later_inhibition = switcher.get_inhibition_strength(EntityId(1));
    
    // Verify inhibition decays over time
    assert!(immediate_inhibition > early_inhibition);
    assert!(early_inhibition > later_inhibition);
    
    // Verify minimum return delay is respected
    let candidates = vec![(EntityId(1), 0.9), (EntityId(2), 0.5)];
    let decision = switcher.evaluate_switching_decision(&candidates, &context);
    
    // Should not immediately return to inhibited target
    match decision {
        SwitchingDecision::Switch { to_entity, .. } => {
            assert_ne!(to_entity, EntityId(1), "Should not immediately return to inhibited target");
        },
        SwitchingDecision::Delay { .. } => {
            // Acceptable - switching delayed due to inhibition
        },
        _ => {}, // Other decisions acceptable
    }
}

#[test]
fn test_salience_capture_behavior() {
    let mut calculator = SalienceCalculator::new();
    
    // High salience stimulus should capture attention
    let high_salience_stimulus = StimulusProperties {
        intensity: 0.95,
        size_ratio: 0.8,
        color_distinctiveness: 0.9,
        color_saturation: 0.9,
        semantic_unexpectedness: 0.8,
        position: (0.5, 0.5),
        timestamp: Instant::now(),
    };
    
    let context = SalienceContext {
        background_activity: 0.2, // Low background
        adaptation_level: 0.1,    // Low adaptation
        attention_allocation: HashMap::new(), // No competing attention
        temporal_expectation: 0.3,
        noise_level: 0.1,
    };
    
    let record = calculator.calculate_salience(EntityId(1), &high_salience_stimulus, &context);
    
    // Should have high salience and capture probability
    assert!(record.total_salience > 0.7, "High contrast stimulus should have high salience");
    assert!(record.capture_probability > 0.6, "High salience should lead to high capture probability");
    
    // Test with competing attention
    let mut competing_attention = HashMap::new();
    competing_attention.insert(EntityId(2), 0.8); // Strong competing attention
    
    let competing_context = SalienceContext {
        attention_allocation: competing_attention,
        ..context
    };
    
    let competing_record = calculator.calculate_salience(EntityId(1), &high_salience_stimulus, &competing_context);
    
    // Capture probability should be reduced with competition
    assert!(competing_record.capture_probability < record.capture_probability,
           "Competing attention should reduce capture probability");
}

#[test]
fn test_attention_capacity_curve() {
    let mut focus_system = AttentionFocus::with_capacity(7);
    
    // Test attention allocation as targets are added
    let mut attention_efficiency = Vec::new();
    
    for i in 1..=10 {
        let source = AttentionSource::TopDown { 
            goal: format!("goal_{}", i), 
            priority: 0.6 
        };
        
        let result = focus_system.focus_on(EntityId(i), source);
        
        if result.is_ok() {
            let utilization = focus_system.get_capacity_utilization();
            let targets = focus_system.get_current_targets();
            let efficiency = targets.len() as f32 / (utilization + 0.01); // Avoid division by zero
            
            attention_efficiency.push(efficiency);
        }
    }
    
    // Efficiency should decrease as capacity is approached (cognitive load increases)
    if attention_efficiency.len() >= 5 {
        let early_efficiency = attention_efficiency[2]; // 3rd item
        let late_efficiency = attention_efficiency[attention_efficiency.len() - 1]; // Last item
        
        assert!(early_efficiency >= late_efficiency,
               "Attention efficiency should decrease as capacity is approached");
    }
}

#[test]
fn test_attention_decay_realism() {
    let mut focus_system = AttentionFocus::new();
    
    // Add target with specific decay rate
    let source = AttentionSource::BottomUp { 
        stimulus_strength: 0.8, 
        novelty: 0.6 
    };
    
    focus_system.focus_on(EntityId(1), source).unwrap();
    
    let initial_strength = focus_system.get_attention_strength(EntityId(1));
    
    // Apply decay over realistic time intervals
    let time_intervals = vec![
        Duration::from_millis(500),  // 0.5 seconds
        Duration::from_secs(1),      // 1 second  
        Duration::from_secs(5),      // 5 seconds
        Duration::from_secs(15),     // 15 seconds
    ];
    
    let mut previous_strength = initial_strength;
    
    for interval in time_intervals {
        focus_system.update_attention(interval);
        let current_strength = focus_system.get_attention_strength(EntityId(1));
        
        // Verify exponential decay pattern
        assert!(current_strength <= previous_strength, 
               "Attention should decay over time");
        
        // Verify realistic decay rate (not too fast, not too slow)
        let decay_ratio = current_strength / previous_strength;
        assert!(decay_ratio > 0.1 && decay_ratio <= 1.0,
               "Decay rate should be realistic: ratio = {}", decay_ratio);
        
        previous_strength = current_strength;
    }
}
```

### Step 4: Performance Benchmarks

```rust
// File: tests/cognitive/attention/performance_benchmarks.rs

use super::*;
use std::time::Instant;

#[test]
fn benchmark_attention_weight_calculation() {
    let mut calculator = AttentionWeightCalculator::new(
        WeightingStrategy::WeightedSum { normalization: true }
    );
    
    let targets = (0..20).map(|i| {
        AttentionTarget {
            entity_id: EntityId(i),
            attention_strength: 0.5 + (i as f32 * 0.02),
            source: AttentionSource::TopDown { 
                goal: format!("goal_{}", i), 
                priority: 0.5 + (i as f32 * 0.02) 
            },
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            persistence: Duration::from_secs(30),
            decay_rate: 0.1,
        }
    }).collect::<Vec<_>>();
    
    let context = ContextState::default();
    
    // Benchmark weight calculation
    let start = Instant::now();
    let iterations = 1000;
    
    for _ in 0..iterations {
        let _weights = calculator.calculate_attention_weights(&targets, &context);
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed / iterations;
    
    // Should complete within 5ms on average
    assert!(avg_time < Duration::from_millis(5),
           "Weight calculation too slow: {:?} average", avg_time);
    
    println!("Weight calculation average time: {:?}", avg_time);
}

#[test]
fn benchmark_focus_switching() {
    let mut switcher = FocusSwitcher::new(
        SwitchingPolicy::Threshold { switch_threshold: 0.3 }
    );
    
    let context = ContextState::default();
    let candidates = vec![
        (EntityId(1), 0.7),
        (EntityId(2), 0.8),
        (EntityId(3), 0.6),
        (EntityId(4), 0.9),
        (EntityId(5), 0.5),
    ];
    
    // Benchmark switching decision
    let start = Instant::now();
    let iterations = 10000;
    
    for _ in 0..iterations {
        let _decision = switcher.evaluate_switching_decision(&candidates, &context);
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed / iterations;
    
    // Should complete within 50μs on average  
    assert!(avg_time < Duration::from_micros(50),
           "Focus switching too slow: {:?} average", avg_time);
    
    println!("Focus switching average time: {:?}", avg_time);
}

#[test]
fn benchmark_salience_calculation() {
    let mut calculator = SalienceCalculator::new();
    
    let stimulus = StimulusProperties {
        intensity: 0.7,
        size_ratio: 0.5,
        color_distinctiveness: 0.6,
        color_saturation: 0.8,
        semantic_unexpectedness: 0.4,
        position: (0.5, 0.5),
        timestamp: Instant::now(),
    };
    
    let context = SalienceContext::default();
    
    // Benchmark salience calculation
    let start = Instant::now();
    let iterations = 1000;
    
    for i in 0..iterations {
        let _record = calculator.calculate_salience(EntityId(i), &stimulus, &context);
    }
    
    let elapsed = start.elapsed();
    let avg_time = elapsed / iterations;
    
    // Should complete within 10ms on average
    assert!(avg_time < Duration::from_millis(10),
           "Salience calculation too slow: {:?} average", avg_time);
    
    println!("Salience calculation average time: {:?}", avg_time);
}

#[test]
fn benchmark_working_memory_operations() {
    let mut memory = WorkingMemory::new();
    
    // Benchmark store operations
    let start = Instant::now();
    let iterations = 1000;
    
    for i in 0..iterations {
        let entity_id = EntityId(i % 7); // Cycle through capacity
        let content = MemoryContent::Entity { properties: HashMap::new() };
        let source = AttentionSource::TopDown { goal: "test".to_string(), priority: 0.5 };
        
        let _ = memory.store(entity_id, content, source);
    }
    
    let store_elapsed = start.elapsed();
    
    // Benchmark retrieve operations
    let start = Instant::now();
    
    for i in 0..iterations {
        let entity_id = EntityId(i % 7);
        let _ = memory.retrieve(entity_id);
    }
    
    let retrieve_elapsed = start.elapsed();
    
    let avg_store_time = store_elapsed / iterations;
    let avg_retrieve_time = retrieve_elapsed / iterations;
    
    // Store should complete within 100μs, retrieve within 10μs
    assert!(avg_store_time < Duration::from_micros(100),
           "Working memory store too slow: {:?} average", avg_store_time);
    assert!(avg_retrieve_time < Duration::from_micros(10),
           "Working memory retrieve too slow: {:?} average", avg_retrieve_time);
    
    println!("Working memory store average time: {:?}", avg_store_time);
    println!("Working memory retrieve average time: {:?}", avg_retrieve_time);
}

#[test]
fn stress_test_continuous_operation() {
    let mut harness = AttentionTestHarness::new();
    let context = harness.create_test_context();
    
    let start = Instant::now();
    let test_duration = Duration::from_secs(5);
    let mut operations = 0;
    
    while start.elapsed() < test_duration {
        // Simulate continuous attention operations
        let entity_id = EntityId(operations % 10);
        let source = AttentionSource::TopDown { 
            goal: "continuous".to_string(), 
            priority: 0.5 
        };
        
        // Attempt focus
        let _ = harness.focus_system.focus_on(entity_id, source.clone());
        
        // Store in memory
        let content = MemoryContent::Entity { properties: HashMap::new() };
        let _ = harness.working_memory.store(entity_id, content, source);
        
        // Update systems
        harness.focus_system.update_attention(Duration::from_millis(10));
        harness.working_memory.update(Duration::from_millis(10));
        harness.focus_switcher.update_inhibition(Duration::from_millis(10));
        
        operations += 1;
        
        // Brief pause to simulate realistic timing
        thread::sleep(Duration::from_micros(100));
    }
    
    let operations_per_second = operations as f32 / test_duration.as_secs_f32();
    
    // Should handle at least 1000 operations per second
    assert!(operations_per_second > 1000.0,
           "System too slow under continuous load: {} ops/sec", operations_per_second);
    
    println!("Continuous operation rate: {:.0} ops/sec", operations_per_second);
    
    // Verify system state is still consistent
    let active_targets = harness.focus_system.get_current_targets();
    assert!(active_targets.len() <= 9); // Capacity respected
}
```

## File Locations

- `tests/cognitive/attention/mod.rs` - Test module organization
- `tests/cognitive/attention/attention_integration_tests.rs` - Integration tests
- `tests/cognitive/attention/cognitive_validation_tests.rs` - Cognitive realism tests
- `tests/cognitive/attention/performance_benchmarks.rs` - Performance benchmarks
- `tests/cognitive/attention/focus_system_tests.rs` - Focus system unit tests
- `tests/cognitive/attention/working_memory_tests.rs` - Working memory unit tests
- `tests/cognitive/attention/weighting_tests.rs` - Weight calculation tests
- `tests/cognitive/attention/switching_tests.rs` - Focus switching tests
- `tests/cognitive/attention/salience_tests.rs` - Salience calculation tests

## Success Criteria

- [ ] All test categories implemented and passing
- [ ] Test coverage > 90% for attention modules
- [ ] Performance benchmarks within acceptable limits
- [ ] Cognitive validation tests confirm realistic behavior
- [ ] Integration tests verify component interaction
- [ ] Stress tests demonstrate system stability
- [ ] Thread safety verified under concurrent access
- [ ] Memory leak detection shows clean operation

## Test Execution Requirements

```bash
# Run all attention tests
cargo test cognitive::attention

# Run specific test categories
cargo test cognitive::attention::integration
cargo test cognitive::attention::cognitive_validation
cargo test cognitive::attention::performance_benchmarks

# Run with performance output
cargo test cognitive::attention::performance_benchmarks -- --nocapture

# Run stress tests
cargo test stress_test_continuous_operation -- --nocapture
```

## Quality Gates

- [ ] All tests pass consistently
- [ ] No flaky tests (intermittent failures)
- [ ] Performance benchmarks within realistic cognitive timing
- [ ] Memory usage stable during stress testing
- [ ] Thread safety verified with concurrent tests
- [ ] Cognitive behaviors match psychological research
- [ ] Integration between components seamless

## Documentation Requirements

Each test should include:
- Clear description of what cognitive behavior is being tested
- Expected outcomes based on psychological research
- Performance expectations with justification
- Error conditions and edge cases covered
- Instructions for interpreting test results

## Next Task

Upon completion, this concludes Day 3 (Attention Mechanisms) tasks for Phase 7. The attention system should now be fully implemented with:

1. ✅ **13_attention_focus_system.md** - Core attention focus management
2. ✅ **14_working_memory.md** - Working memory with capacity limits
3. ✅ **15_attention_weighting.md** - Multi-factor attention weighting
4. ✅ **16_focus_switching.md** - Dynamic focus switching with costs
5. ✅ **17_salience_calculation.md** - Bottom-up salience detection
6. ✅ **18_attention_tests.md** - Comprehensive test suite

The attention system is now ready for integration with the spreading activation (Day 1) and intent recognition (Day 2) systems to create a complete brain-inspired query processing engine.