/// Advanced temporal decay scenarios for relationship dynamics
/// 
/// This module extends temporal relationship testing with:
/// - Multi-modal decay patterns and phase transitions
/// - Competitive decay dynamics between relationships
/// - Temporal interference and consolidation effects
/// - Circadian and ultradian rhythm simulation
/// - Memory reconsolidation and updating patterns
/// - Stress-induced temporal modulation

use llmkg::core::brain_types::{BrainInspiredRelationship, RelationType};
use llmkg::core::types::EntityKey;
use std::time::{SystemTime, Duration};
use std::thread;
use super::test_constants;
use super::test_helpers::*;

// ==================== Multi-Modal Decay Pattern Tests ====================

#[test]
fn test_multi_phase_decay_patterns() {
    let mut relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::STRONG_EXCITATORY,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    // Phase 1: Rapid initial decay (short-term memory loss)
    let rapid_decay_phase = test_constants::FAST_DECAY_RATE;
    relationship.temporal_decay = rapid_decay_phase;
    
    // Simulate first phase
    relationship.last_strengthened = SystemTime::now() - Duration::from_millis(50);
    let phase1_weight = relationship.apply_decay();
    
    // Phase 2: Slower decay (consolidation)
    let slow_decay_phase = test_constants::SLOW_DECAY_RATE;
    relationship.temporal_decay = slow_decay_phase;
    
    // Simulate second phase
    thread::sleep(Duration::from_millis(30));
    let phase2_weight = relationship.apply_decay();
    
    // Phase 3: Stable long-term memory
    let minimal_decay_phase = 0.001;
    relationship.temporal_decay = minimal_decay_phase;
    
    thread::sleep(Duration::from_millis(20));
    let phase3_weight = relationship.apply_decay();
    
    // Verify multi-phase pattern
    assert!(phase1_weight < test_constants::STRONG_EXCITATORY, "Phase 1 should show rapid decay");
    assert!(phase2_weight < phase1_weight, "Phase 2 should continue decay");
    assert!(phase3_weight > phase2_weight * 0.9, "Phase 3 should stabilize");
    
    // Overall should follow exponential decay curve with changing rates
    let total_decay = test_constants::STRONG_EXCITATORY - phase3_weight;
    assert!(total_decay > 0.1, "Should show significant total decay");
    assert!(phase3_weight > 0.3, "Should retain substantial long-term memory");
}

#[test]
fn test_decay_rate_adaptation() {
    let mut relationship = create_test_relationship(
        RelationType::Temporal,
        test_constants::MEDIUM_EXCITATORY,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    let initial_decay_rate = relationship.temporal_decay;
    let usage_threshold = 5;
    
    // Simulate usage-dependent decay rate adaptation
    for activation_count in 1..=10 {
        relationship.strengthen(test_constants::STANDARD_LEARNING_RATE);
        
        // Adapt decay rate based on usage frequency
        if relationship.activation_count >= usage_threshold {
            // Frequently used connections decay slower
            relationship.temporal_decay = initial_decay_rate * 0.5;
        } else {
            // Infrequently used connections decay faster
            relationship.temporal_decay = initial_decay_rate * 1.2;
        }
        
        thread::sleep(Duration::from_millis(2));
        relationship.apply_decay();
    }
    
    // High-usage relationship should have reduced decay rate
    assert!(
        relationship.temporal_decay < initial_decay_rate,
        "Frequently used relationships should decay slower: {} vs {}",
        relationship.temporal_decay, initial_decay_rate
    );
    
    // Should maintain higher weight due to adaptation
    assert!(
        relationship.weight > test_constants::WEAK_EXCITATORY,
        "Adapted relationship should maintain strength: {}",
        relationship.weight
    );
}

#[test]
fn test_decay_saturation_effects() {
    let mut relationship = create_test_relationship(
        RelationType::RelatedTo,
        test_constants::SATURATION_LEVEL,
        false,
        test_constants::FAST_DECAY_RATE
    );
    
    let decay_steps = 20;
    let mut weight_history = vec![relationship.weight];
    
    // Apply decay steps and track saturation behavior
    for _ in 0..decay_steps {
        thread::sleep(Duration::from_millis(5));
        relationship.apply_decay();
        weight_history.push(relationship.weight);
    }
    
    // Calculate decay rates between steps
    let mut decay_rates = Vec::new();
    for i in 1..weight_history.len() {
        let rate = (weight_history[i-1] - weight_history[i]) / weight_history[i-1];
        decay_rates.push(rate);
    }
    
    // Early decay should be faster than late decay (saturation effect)
    let early_avg = decay_rates[0..3].iter().sum::<f32>() / 3.0;
    let late_avg = decay_rates[decay_rates.len()-3..].iter().sum::<f32>() / 3.0;
    
    assert!(
        early_avg > late_avg * 1.1, // Allow some tolerance
        "Early decay should be faster than late decay: early={}, late={}",
        early_avg, late_avg
    );
    
    // Should never reach exactly zero
    assert!(weight_history.last().unwrap() > &0.0, "Weight should never reach exactly zero");
}

// ==================== Competitive Decay Dynamics Tests ====================

#[test]
fn test_competitive_relationship_decay() {
    let source = EntityKey::from(slotmap::KeyData::from_ffi(1));
    let target1 = EntityKey::from(slotmap::KeyData::from_ffi(2));
    let target2 = EntityKey::from(slotmap::KeyData::from_ffi(3));
    
    let mut strong_relationship = BrainInspiredRelationship::new(source, target1, RelationType::Learned);
    let mut weak_relationship = BrainInspiredRelationship::new(source, target2, RelationType::Learned);
    
    // Establish different initial strengths
    for _ in 0..8 {
        strong_relationship.strengthen(test_constants::STANDARD_LEARNING_RATE);
    }
    for _ in 0..3 {
        weak_relationship.strengthen(test_constants::STANDARD_LEARNING_RATE);
    }
    
    let initial_strong = strong_relationship.weight;
    let initial_weak = weak_relationship.weight;
    
    // Simulate competitive decay where strong connections interfere with weak ones
    let competition_factor = 1.5;
    
    for _ in 0..10 {
        // Apply normal decay
        strong_relationship.apply_decay();
        weak_relationship.apply_decay();
        
        // Apply competitive interference (stronger connections suppress weaker ones)
        if strong_relationship.weight > weak_relationship.weight {
            weak_relationship.temporal_decay *= competition_factor;
        }
        
        thread::sleep(Duration::from_millis(1));
    }
    
    // Strong relationship should decay less than weak relationship
    let strong_retention = strong_relationship.weight / initial_strong;
    let weak_retention = weak_relationship.weight / initial_weak;
    
    assert!(
        strong_retention > weak_retention,
        "Strong relationships should compete better: strong_retention={}, weak_retention={}",
        strong_retention, weak_retention
    );
}

#[test]
fn test_resource_limited_decay() {
    // Simulate limited metabolic resources affecting decay rates
    let num_relationships = 5;
    let mut relationships = Vec::new();
    
    for i in 0..num_relationships {
        let source = EntityKey::from(slotmap::KeyData::from_ffi(i));
        let target = EntityKey::from(slotmap::KeyData::from_ffi(i + 10));
        relationships.push(BrainInspiredRelationship::new(source, target, RelationType::RelatedTo));
    }
    
    // Strengthen relationships to different levels
    for (i, rel) in relationships.iter_mut().enumerate() {
        for _ in 0..(i + 1) * 2 {
            rel.strengthen(test_constants::STANDARD_LEARNING_RATE);
        }
    }
    
    let total_initial_strength: f32 = relationships.iter().map(|r| r.weight).sum();
    let resource_limit = total_initial_strength * 0.7; // 70% of total strength can be maintained
    
    // Apply resource-limited decay
    for _ in 0..10 {
        let current_total: f32 = relationships.iter().map(|r| r.weight).sum();
        
        if current_total > resource_limit {
            // Increase decay rate when over resource limit
            let overuse_factor = current_total / resource_limit;
            for rel in &mut relationships {
                rel.temporal_decay *= overuse_factor;
                rel.apply_decay();
            }
        } else {
            // Normal decay when under resource limit
            for rel in &mut relationships {
                rel.apply_decay();
            }
        }
        
        thread::sleep(Duration::from_millis(2));
    }
    
    let final_total: f32 = relationships.iter().map(|r| r.weight).sum();
    
    // Should converge toward resource limit
    assert!(
        final_total <= resource_limit * 1.1,
        "Should respect resource limits: final={}, limit={}",
        final_total, resource_limit
    );
    
    // Stronger relationships should retain more of their original strength
    let strongest_initial = relationships.iter().map(|r| r.weight).fold(0.0, f32::max);
    let strongest_final = relationships.iter().map(|r| r.weight).fold(0.0, f32::max);
    let weakest_final = relationships.iter().map(|r| r.weight).fold(f32::INFINITY, f32::min);
    
    assert!(strongest_final > weakest_final, "Strength hierarchy should be preserved");
}

// ==================== Temporal Interference Tests ====================

#[test]
fn test_retroactive_interference() {
    let source = EntityKey::from(slotmap::KeyData::from_ffi(1));
    let old_target = EntityKey::from(slotmap::KeyData::from_ffi(2));
    let new_target = EntityKey::from(slotmap::KeyData::from_ffi(3));
    
    let mut old_memory = BrainInspiredRelationship::new(source, old_target, RelationType::Learned);
    let mut new_memory = BrainInspiredRelationship::new(source, new_target, RelationType::Learned);
    
    // Learn old association first
    for _ in 0..6 {
        old_memory.strengthen(test_constants::STANDARD_LEARNING_RATE);
    }
    let old_after_learning = old_memory.weight;
    
    // Wait for some consolidation
    thread::sleep(Duration::from_millis(20));
    
    // Learn competing new association (retroactive interference)
    for _ in 0..6 {
        new_memory.strengthen(test_constants::STANDARD_LEARNING_RATE);
        
        // New learning interferes with old memory
        old_memory.temporal_decay *= 1.2; // Increased decay due to interference
        old_memory.apply_decay();
    }
    
    let old_after_interference = old_memory.weight;
    let new_final = new_memory.weight;
    
    // Old memory should be weakened by interference
    assert!(
        old_after_interference < old_after_learning,
        "Old memory should be weakened by interference: {} -> {}",
        old_after_learning, old_after_interference
    );
    
    // New memory should be stronger than weakened old memory
    assert!(
        new_final > old_after_interference,
        "New memory should dominate: new={}, old={}",
        new_final, old_after_interference
    );
}

#[test]
fn test_proactive_interference() {
    let source = EntityKey::from(slotmap::KeyData::from_ffi(1));
    let first_target = EntityKey::from(slotmap::KeyData::from_ffi(2));
    let second_target = EntityKey::from(slotmap::KeyData::from_ffi(3));
    
    let mut first_memory = BrainInspiredRelationship::new(source, first_target, RelationType::Learned);
    let mut second_memory = BrainInspiredRelationship::new(source, second_target, RelationType::Learned);
    
    // Strongly establish first memory
    for _ in 0..8 {
        first_memory.strengthen(test_constants::FAST_LEARNING_RATE);
    }
    
    // Let it consolidate
    thread::sleep(Duration::from_millis(15));
    
    // Try to learn second memory (proactive interference from first)
    let interference_factor = first_memory.weight; // Stronger first memory = more interference
    
    for _ in 0..5 {
        // Reduced learning rate due to proactive interference
        let reduced_rate = test_constants::STANDARD_LEARNING_RATE * (1.0 - interference_factor * 0.3);
        second_memory.strengthen(reduced_rate.max(0.01));
    }
    
    // Second memory should be weaker due to proactive interference
    assert!(
        second_memory.weight < first_memory.weight * 0.8,
        "Second memory should be impaired by proactive interference: second={}, first={}",
        second_memory.weight, first_memory.weight
    );
}

// ==================== Circadian Rhythm Simulation Tests ====================

#[test]
fn test_circadian_decay_modulation() {
    let mut relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::MEDIUM_EXCITATORY,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    let base_decay_rate = relationship.temporal_decay;
    let circadian_period_ms = 100; // Compressed circadian cycle for testing
    let num_cycles = 3;
    
    let mut weight_over_time = Vec::new();
    
    for time_step in 0..(circadian_period_ms * num_cycles) {
        // Simulate circadian modulation of decay rate
        let cycle_position = (time_step as f32) / (circadian_period_ms as f32);
        let circadian_phase = 2.0 * std::f32::consts::PI * cycle_position;
        
        // Decay rate varies sinusoidally (higher at "night", lower at "day")
        let circadian_modifier = 1.0 + 0.5 * circadian_phase.sin();
        relationship.temporal_decay = base_decay_rate * circadian_modifier;
        
        // Apply learning during "day" phase
        if circadian_phase.cos() > 0.0 {
            relationship.strengthen(test_constants::SLOW_LEARNING_RATE);
        }
        
        relationship.apply_decay();
        weight_over_time.push((time_step, relationship.weight));
        
        thread::sleep(Duration::from_millis(1));
    }
    
    // Should show periodic oscillations
    let mid_point = weight_over_time.len() / 2;
    let early_weights: Vec<f32> = weight_over_time[0..mid_point].iter().map(|(_, w)| *w).collect();
    let late_weights: Vec<f32> = weight_over_time[mid_point..].iter().map(|(_, w)| *w).collect();
    
    // Should show some cyclical pattern (not strictly monotonic decay)
    let early_trend = early_weights.last().unwrap() - early_weights.first().unwrap();
    let late_trend = late_weights.last().unwrap() - late_weights.first().unwrap();
    
    // At least one period should show growth (due to learning during "day")
    assert!(
        early_trend > -0.1 || late_trend > -0.1,
        "Should show periods of growth due to circadian learning: early_trend={}, late_trend={}",
        early_trend, late_trend
    );
}

#[test]
fn test_sleep_consolidation_effect() {
    let mut relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::WEAK_EXCITATORY,
        false,
        test_constants::FAST_DECAY_RATE
    );
    
    // Learning phase (awake)
    let wake_learning_steps = 5;
    for _ in 0..wake_learning_steps {
        relationship.strengthen(test_constants::STANDARD_LEARNING_RATE);
        relationship.apply_decay(); // Normal decay during wake
        thread::sleep(Duration::from_millis(2));
    }
    
    let pre_sleep_weight = relationship.weight;
    
    // Sleep phase - reduced decay and consolidation
    let sleep_decay_rate = relationship.temporal_decay * 0.3; // Much slower decay
    relationship.temporal_decay = sleep_decay_rate;
    
    // Simulate memory replay/consolidation during sleep
    for _ in 0..3 {
        relationship.strengthen(test_constants::SLOW_LEARNING_RATE); // Weak replay strengthening
        relationship.apply_decay();
        thread::sleep(Duration::from_millis(5));
    }
    
    let post_sleep_weight = relationship.weight;
    
    // Should show consolidation effect (maintained or slight increase)
    assert!(
        post_sleep_weight >= pre_sleep_weight * 0.95,
        "Sleep should consolidate memory: pre={}, post={}",
        pre_sleep_weight, post_sleep_weight
    );
}

// ==================== Memory Reconsolidation Tests ====================

#[test]
fn test_memory_reconsolidation_window() {
    let mut relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::STRONG_EXCITATORY,
        false,
        test_constants::SLOW_DECAY_RATE
    );
    
    // Initial consolidation
    for _ in 0..5 {
        relationship.strengthen(test_constants::STANDARD_LEARNING_RATE);
    }
    
    // Let memory consolidate
    thread::sleep(Duration::from_millis(30));
    let consolidated_weight = relationship.weight;
    
    // Memory reactivation opens reconsolidation window
    relationship.strengthen(test_constants::WEAK_EXCITATORY); // Reactivation
    
    // During reconsolidation window, memory is labile again
    let reconsolidation_window_ms = 20;
    let window_start = SystemTime::now();
    
    while window_start.elapsed().unwrap().as_millis() < reconsolidation_window_ms as u128 {
        // Memory can be modified during reconsolidation
        relationship.strengthen(test_constants::FAST_LEARNING_RATE);
        thread::sleep(Duration::from_millis(2));
    }
    
    let post_reconsolidation_weight = relationship.weight;
    
    // Should show enhanced strengthening during reconsolidation
    assert!(
        post_reconsolidation_weight > consolidated_weight * 1.1,
        "Reconsolidation should allow memory enhancement: {} -> {}",
        consolidated_weight, post_reconsolidation_weight
    );
}

#[test]
fn test_memory_updating_during_reconsolidation() {
    let source = EntityKey::from(slotmap::KeyData::from_ffi(1));
    let original_target = EntityKey::from(slotmap::KeyData::from_ffi(2));
    let updated_target = EntityKey::from(slotmap::KeyData::from_ffi(3));
    
    let mut original_memory = BrainInspiredRelationship::new(source, original_target, RelationType::Learned);
    let mut updated_memory = BrainInspiredRelationship::new(source, updated_target, RelationType::Learned);
    
    // Establish original memory
    for _ in 0..6 {
        original_memory.strengthen(test_constants::STANDARD_LEARNING_RATE);
    }
    
    // Consolidation period
    thread::sleep(Duration::from_millis(25));
    let original_consolidated = original_memory.weight;
    
    // Reactivate original memory (trigger reconsolidation)
    original_memory.strengthen(test_constants::WEAK_EXCITATORY);
    
    // During reconsolidation, introduce competing/updating information
    for _ in 0..4 {
        updated_memory.strengthen(test_constants::STANDARD_LEARNING_RATE);
        
        // Original memory becomes labile and can be weakened
        original_memory.temporal_decay *= 1.3;
        original_memory.apply_decay();
        
        thread::sleep(Duration::from_millis(3));
    }
    
    let original_after_update = original_memory.weight;
    let updated_final = updated_memory.weight;
    
    // Original memory should be weakened during reconsolidation update
    assert!(
        original_after_update < original_consolidated,
        "Original memory should be weakened during updating: {} -> {}",
        original_consolidated, original_after_update
    );
    
    // Updated memory should dominate
    assert!(
        updated_final > original_after_update,
        "Updated memory should be stronger: updated={}, original={}",
        updated_final, original_after_update
    );
}

// ==================== Stress-Induced Temporal Modulation Tests ====================

#[test]
fn test_acute_stress_effects() {
    let mut normal_relationship = create_test_relationship(
        RelationType::RelatedTo,
        test_constants::MEDIUM_EXCITATORY,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    let mut stressed_relationship = create_test_relationship(
        RelationType::RelatedTo,
        test_constants::MEDIUM_EXCITATORY,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    // Normal learning condition
    for _ in 0..5 {
        normal_relationship.strengthen(test_constants::STANDARD_LEARNING_RATE);
        normal_relationship.apply_decay();
        thread::sleep(Duration::from_millis(2));
    }
    
    // Acute stress condition - enhanced encoding but increased decay
    let stress_learning_boost = 1.5;
    let stress_decay_increase = 2.0;
    
    for _ in 0..5 {
        stressed_relationship.strengthen(test_constants::STANDARD_LEARNING_RATE * stress_learning_boost);
        stressed_relationship.temporal_decay *= stress_decay_increase;
        stressed_relationship.apply_decay();
        thread::sleep(Duration::from_millis(2));
    }
    
    // Acute stress should show initial enhancement but worse retention
    assert!(
        stressed_relationship.weight < normal_relationship.weight,
        "Acute stress should impair long-term retention: stressed={}, normal={}",
        stressed_relationship.weight, normal_relationship.weight
    );
}

#[test]
fn test_chronic_stress_effects() {
    let mut relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::WEAK_EXCITATORY,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    let baseline_learning_rate = test_constants::STANDARD_LEARNING_RATE;
    let baseline_decay_rate = relationship.temporal_decay;
    
    // Chronic stress - progressive impairment
    let stress_duration_steps = 15;
    
    for step in 0..stress_duration_steps {
        // Progressive impairment with chronic stress
        let stress_factor = 1.0 + (step as f32) / (stress_duration_steps as f32) * 0.5;
        
        let impaired_learning_rate = baseline_learning_rate / stress_factor;
        let increased_decay_rate = baseline_decay_rate * stress_factor;
        
        relationship.strengthen(impaired_learning_rate);
        relationship.temporal_decay = increased_decay_rate;
        relationship.apply_decay();
        
        thread::sleep(Duration::from_millis(2));
    }
    
    // Chronic stress should severely impair final weight
    assert!(
        relationship.weight < test_constants::WEAK_EXCITATORY * 0.8,
        "Chronic stress should severely impair memory: {}",
        relationship.weight
    );
    
    // Decay rate should be significantly elevated
    assert!(
        relationship.temporal_decay > baseline_decay_rate * 1.3,
        "Chronic stress should increase decay rate: {} vs {}",
        relationship.temporal_decay, baseline_decay_rate
    );
}

// ==================== Complex Temporal Pattern Integration Tests ====================

#[test]
fn test_integrated_temporal_dynamics() {
    // Combine multiple temporal effects in realistic scenario
    let mut relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::RESTING_POTENTIAL,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    let simulation_steps = 50;
    let base_decay = relationship.temporal_decay;
    
    for step in 0..simulation_steps {
        // Circadian modulation
        let circadian_phase = 2.0 * std::f32::consts::PI * (step as f32) / 24.0; // 24-step "day"
        let circadian_modifier = 1.0 + 0.3 * circadian_phase.sin();
        
        // Usage-dependent decay adaptation
        let usage_factor = if relationship.activation_count > 3 { 0.8 } else { 1.2 };
        
        // Stress modulation (peaks at middle of simulation)
        let stress_peak = (simulation_steps / 2) as f32;
        let stress_distance = ((step as f32) - stress_peak).abs() / stress_peak;
        let stress_modifier = 1.0 + 0.5 * (-stress_distance * 2.0).exp();
        
        // Combined decay rate
        relationship.temporal_decay = base_decay * circadian_modifier * usage_factor * stress_modifier;
        
        // Learning events during favorable conditions
        if circadian_phase.cos() > 0.5 && stress_modifier < 1.2 {
            relationship.strengthen(test_constants::STANDARD_LEARNING_RATE);
        }
        
        relationship.apply_decay();
        thread::sleep(Duration::from_millis(1));
    }
    
    // Should show complex but realistic temporal dynamics
    assert!(
        relationship.weight > test_constants::RESTING_POTENTIAL,
        "Should show net learning despite complex dynamics: {}",
        relationship.weight
    );
    
    assert!(
        relationship.activation_count > 0,
        "Should have learning events: {}",
        relationship.activation_count
    );
    
    // Decay rate should be modulated from baseline
    assert!(
        (relationship.temporal_decay - base_decay).abs() > 0.01,
        "Decay rate should be modulated: {} vs {}",
        relationship.temporal_decay, base_decay
    );
}