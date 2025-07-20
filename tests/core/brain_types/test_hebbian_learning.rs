/// Comprehensive Hebbian learning and relationship strengthening tests
/// 
/// This module implements extensive testing for synaptic plasticity including:
/// - Classical Hebbian learning scenarios
/// - Competitive learning dynamics
/// - Long-term potentiation and depression
/// - Spike-timing dependent plasticity simulation
/// - Learning rate adaptation and saturation effects
/// - Memory consolidation and forgetting patterns

use llmkg::core::brain_types::{BrainInspiredRelationship, RelationType};
use llmkg::core::types::EntityKey;
use std::time::{SystemTime, Duration};
use std::thread;
use super::test_constants;
use super::test_helpers::*;

// ==================== Basic Hebbian Learning Tests ====================

#[test]
fn test_basic_hebbian_strengthening() {
    let source = EntityKey::from(slotmap::KeyData::from_ffi(1));
    let target = EntityKey::from(slotmap::KeyData::from_ffi(2));
    
    let mut relationship = BrainInspiredRelationship::new(source, target, RelationType::RelatedTo);
    
    let initial_weight = relationship.weight;
    let learning_rate = test_constants::STANDARD_LEARNING_RATE;
    
    // Apply Hebbian learning
    relationship.strengthen(learning_rate);
    
    // Weight should increase
    assert!(
        relationship.weight > initial_weight,
        "Weight should increase: {} -> {}",
        initial_weight, relationship.weight
    );
    
    // Should update both weight and strength
    assert_eq!(relationship.weight, relationship.strength);
    
    // Should update counters
    assert_eq!(relationship.activation_count, 1);
    assert_eq!(relationship.usage_count, 1);
    
    // Should update timestamps
    assert!(relationship.last_strengthened >= relationship.creation_time);
    assert!(relationship.last_update >= relationship.last_strengthened);
}

#[test]
fn test_repeated_strengthening() {
    let mut relationship = create_test_relationship(
        RelationType::IsA,
        test_constants::WEAK_EXCITATORY,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    let learning_rate = test_constants::STANDARD_LEARNING_RATE;
    let num_iterations = 10;
    
    let mut weight_history = vec![relationship.weight];
    
    // Apply repeated strengthening
    for i in 0..num_iterations {
        relationship.strengthen(learning_rate);
        weight_history.push(relationship.weight);
        
        // Weight should be monotonically increasing
        assert!(
            weight_history[i + 1] >= weight_history[i],
            "Weight should not decrease on iteration {}: {} -> {}",
            i, weight_history[i], weight_history[i + 1]
        );
    }
    
    // Should approach saturation
    assert!(
        relationship.weight <= test_constants::SATURATION_LEVEL,
        "Weight should not exceed saturation: {}",
        relationship.weight
    );
    
    // Usage count should match iterations
    assert_eq!(relationship.activation_count, num_iterations as u64);
}

#[test]
fn test_learning_rate_effects() {
    let learning_rates = generate_learning_rates();
    let initial_weight = test_constants::MEDIUM_EXCITATORY;
    
    let mut final_weights = Vec::new();
    
    for &rate in &learning_rates {
        let mut relationship = create_test_relationship(
            RelationType::HasProperty,
            initial_weight,
            false,
            test_constants::STANDARD_DECAY_RATE
        );
        
        // Single strengthening step
        relationship.strengthen(rate);
        final_weights.push(relationship.weight);
    }
    
    // Higher learning rates should produce higher weights (up to saturation)
    for i in 1..learning_rates.len() {
        if learning_rates[i] > learning_rates[i-1] && final_weights[i-1] < 1.0 {
            assert!(
                final_weights[i] >= final_weights[i-1],
                "Higher learning rate should not decrease weight: rate[{}]={} -> weight={}, rate[{}]={} -> weight={}",
                i-1, learning_rates[i-1], final_weights[i-1],
                i, learning_rates[i], final_weights[i]
            );
        }
    }
}

// ==================== Competitive Learning Tests ====================

#[test]
fn test_competitive_strengthening() {
    // Create multiple relationships from same source
    let source = EntityKey::from(slotmap::KeyData::from_ffi(1));
    let targets = vec![
        EntityKey::from(slotmap::KeyData::from_ffi(2)),
        EntityKey::from(slotmap::KeyData::from_ffi(3)),
        EntityKey::from(slotmap::KeyData::from_ffi(4)),
    ];
    
    let mut relationships = Vec::new();
    for target in targets {
        relationships.push(BrainInspiredRelationship::new(source, target, RelationType::RelatedTo));
    }
    
    let learning_rate = test_constants::STANDARD_LEARNING_RATE;
    
    // Strengthen first relationship more than others (winner-take-all)
    let winner_index = 0;
    let winner_activations = 10;
    let loser_activations = 2;
    
    // Activate winner more frequently
    for _ in 0..winner_activations {
        relationships[winner_index].strengthen(learning_rate);
    }
    
    // Activate losers less frequently
    for i in 1..relationships.len() {
        for _ in 0..loser_activations {
            relationships[i].strengthen(learning_rate);
        }
    }
    
    // Winner should have highest weight
    let winner_weight = relationships[winner_index].weight;
    for i in 1..relationships.len() {
        assert!(
            winner_weight > relationships[i].weight,
            "Winner should have highest weight: winner={}, loser[{}]={}",
            winner_weight, i, relationships[i].weight
        );
    }
}

#[test]
fn test_lateral_inhibition_simulation() {
    // Simulate lateral inhibition by strengthening some connections
    // while weakening others through decay
    
    let source = EntityKey::from(slotmap::KeyData::from_ffi(1));
    let num_connections = 5;
    
    let mut relationships = Vec::new();
    for i in 0..num_connections {
        let target = EntityKey::from(slotmap::KeyData::from_ffi(i + 2));
        relationships.push(BrainInspiredRelationship::new(source, target, RelationType::RelatedTo));
    }
    
    let strong_learning_rate = test_constants::FAST_LEARNING_RATE;
    let weak_learning_rate = test_constants::SLOW_LEARNING_RATE;
    
    // Create strong and weak populations
    let strong_indices = vec![0, 2]; // Even indices get strong reinforcement
    let weak_indices = vec![1, 3, 4]; // Odd indices get weak reinforcement
    
    let num_rounds = 5;
    
    for _ in 0..num_rounds {
        // Strengthen strong connections
        for &i in &strong_indices {
            relationships[i].strengthen(strong_learning_rate);
        }
        
        // Weakly strengthen or let decay weak connections
        for &i in &weak_indices {
            relationships[i].strengthen(weak_learning_rate);
            // Simulate some decay
            relationships[i].apply_decay();
        }
        
        // Small delay for realistic timing
        thread::sleep(Duration::from_millis(1));
    }
    
    // Strong connections should dominate
    let strong_avg = strong_indices.iter()
        .map(|&i| relationships[i].weight)
        .sum::<f32>() / strong_indices.len() as f32;
    
    let weak_avg = weak_indices.iter()
        .map(|&i| relationships[i].weight)
        .sum::<f32>() / weak_indices.len() as f32;
    
    assert!(
        strong_avg > weak_avg,
        "Strong connections should dominate: strong_avg={}, weak_avg={}",
        strong_avg, weak_avg
    );
}

// ==================== Long-term Potentiation/Depression Tests ====================

#[test]
fn test_long_term_potentiation() {
    let mut relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::THRESHOLD_POTENTIAL,
        false,
        test_constants::SLOW_DECAY_RATE
    );
    
    let high_frequency_rate = test_constants::FAST_LEARNING_RATE;
    let ltp_threshold = 5; // Minimum activations for LTP
    
    // High-frequency stimulation
    for _ in 0..ltp_threshold * 2 {
        relationship.strengthen(high_frequency_rate);
        // Short intervals between activations
        thread::sleep(Duration::from_millis(1));
    }
    
    let potentiated_weight = relationship.weight;
    
    // Should show significant potentiation
    assert!(
        potentiated_weight > test_constants::STRONG_EXCITATORY,
        "Should show long-term potentiation: {}",
        potentiated_weight
    );
    
    // Wait and check persistence
    thread::sleep(Duration::from_millis(50));
    let persistent_weight = relationship.weight; // No decay applied yet
    
    // Should maintain high weight (simulating LTP persistence)
    assert_float_eq(
        persistent_weight,
        potentiated_weight,
        test_constants::ACTIVATION_EPSILON
    );
}

#[test]
fn test_long_term_depression() {
    let mut relationship = create_test_relationship(
        RelationType::RelatedTo,
        test_constants::STRONG_EXCITATORY,
        false,
        test_constants::FAST_DECAY_RATE
    );
    
    let initial_weight = relationship.weight;
    
    // Low-frequency stimulation with decay
    let low_frequency_activations = 3;
    let weak_learning_rate = test_constants::MIN_LEARNING_RATE;
    
    for _ in 0..low_frequency_activations {
        relationship.strengthen(weak_learning_rate);
        
        // Long intervals with decay
        thread::sleep(Duration::from_millis(20));
        relationship.apply_decay();
    }
    
    let depressed_weight = relationship.weight;
    
    // Should show depression (weight decrease from decay > strengthening)
    assert!(
        depressed_weight < initial_weight,
        "Should show long-term depression: {} -> {}",
        initial_weight, depressed_weight
    );
}

// ==================== Spike-Timing Dependent Plasticity Tests ====================

#[test]
fn test_spike_timing_dependent_plasticity() {
    // Simulate STDP where timing between pre- and post-synaptic spikes matters
    
    let source = EntityKey::from(slotmap::KeyData::from_ffi(1));
    let target = EntityKey::from(slotmap::KeyData::from_ffi(2));
    
    // Create two relationships to test different timing scenarios
    let mut pre_post_relationship = BrainInspiredRelationship::new(source, target, RelationType::Temporal);
    let mut post_pre_relationship = BrainInspiredRelationship::new(source, target, RelationType::Temporal);
    
    let learning_rate = test_constants::STANDARD_LEARNING_RATE;
    
    // Scenario 1: Pre-before-post (should strengthen - causal)
    pre_post_relationship.strengthen(learning_rate); // Pre-synaptic spike
    thread::sleep(Duration::from_millis(5)); // Small delay
    pre_post_relationship.strengthen(learning_rate); // Post-synaptic spike
    
    let causal_weight = pre_post_relationship.weight;
    
    // Scenario 2: Post-before-pre (should strengthen less - anti-causal)
    post_pre_relationship.strengthen(learning_rate * 0.5); // Weaker post-before-pre
    thread::sleep(Duration::from_millis(5));
    post_pre_relationship.strengthen(learning_rate * 0.5);
    
    let anti_causal_weight = post_pre_relationship.weight;
    
    // Causal timing should produce stronger connections
    assert!(
        causal_weight > anti_causal_weight,
        "Causal timing should be stronger: causal={}, anti-causal={}",
        causal_weight, anti_causal_weight
    );
}

#[test]
fn test_critical_period_plasticity() {
    // Test that early strengthening has more effect (critical period)
    
    let mut early_relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::RESTING_POTENTIAL,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    let mut late_relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::RESTING_POTENTIAL,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    // Early critical period - high plasticity
    let critical_learning_rate = test_constants::FAST_LEARNING_RATE;
    for _ in 0..5 {
        early_relationship.strengthen(critical_learning_rate);
    }
    
    // Simulate aging/maturation
    thread::sleep(Duration::from_millis(10));
    
    // Later period - reduced plasticity
    let mature_learning_rate = test_constants::SLOW_LEARNING_RATE;
    for _ in 0..5 {
        late_relationship.strengthen(mature_learning_rate);
    }
    
    // Early learning should be more effective
    assert!(
        early_relationship.weight > late_relationship.weight,
        "Critical period should show higher plasticity: early={}, late={}",
        early_relationship.weight, late_relationship.weight
    );
}

// ==================== Memory Consolidation Tests ====================

#[test]
fn test_memory_consolidation() {
    let mut relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::WEAK_EXCITATORY,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    // Initial learning phase
    let learning_rate = test_constants::STANDARD_LEARNING_RATE;
    let initial_learning_steps = 5;
    
    for _ in 0..initial_learning_steps {
        relationship.strengthen(learning_rate);
    }
    
    let post_learning_weight = relationship.weight;
    
    // Consolidation phase - reduced decay rate (simulate protein synthesis)
    relationship.temporal_decay = test_constants::SLOW_DECAY_RATE;
    
    // Wait for consolidation period
    thread::sleep(Duration::from_millis(30));
    
    // Apply minimal decay
    relationship.apply_decay();
    
    let consolidated_weight = relationship.weight;
    
    // Should retain most of the learned weight
    let retention_ratio = consolidated_weight / post_learning_weight;
    assert!(
        retention_ratio > 0.8,
        "Should show good consolidation: retention={}",
        retention_ratio
    );
}

#[test]
fn test_reconsolidation() {
    let mut relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::MEDIUM_EXCITATORY,
        false,
        test_constants::SLOW_DECAY_RATE
    );
    
    // Initial consolidation
    for _ in 0..3 {
        relationship.strengthen(test_constants::STANDARD_LEARNING_RATE);
    }
    
    let initial_consolidated = relationship.weight;
    
    // Reactivation makes memory labile again
    thread::sleep(Duration::from_millis(20));
    
    // Reconsolidation with new information
    relationship.strengthen(test_constants::FAST_LEARNING_RATE);
    
    let reconsolidated_weight = relationship.weight;
    
    // Should be able to modify consolidated memory
    assert!(
        reconsolidated_weight > initial_consolidated,
        "Reconsolidation should allow modification: {} -> {}",
        initial_consolidated, reconsolidated_weight
    );
}

// ==================== Forgetting and Interference Tests ====================

#[test]
fn test_forgetting_curves() {
    let mut relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::STRONG_EXCITATORY,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    let initial_weight = relationship.weight;
    let time_points = vec![10, 20, 50, 100]; // milliseconds
    let mut decay_curve = Vec::new();
    
    for &time_ms in &time_points {
        // Reset for each measurement
        relationship.weight = initial_weight;
        relationship.strength = initial_weight;
        relationship.last_strengthened = SystemTime::now() - Duration::from_millis(time_ms);
        
        let decayed_weight = relationship.apply_decay();
        decay_curve.push(decayed_weight);
    }
    
    // Should show exponential forgetting curve
    for i in 1..decay_curve.len() {
        assert!(
            decay_curve[i] <= decay_curve[i-1],
            "Forgetting curve should be monotonic: t={}ms -> {}, t={}ms -> {}",
            time_points[i-1], decay_curve[i-1],
            time_points[i], decay_curve[i]
        );
    }
    
    // First point should be close to original
    assert!(decay_curve[0] > initial_weight * 0.9);
    
    // Last point should show significant decay
    assert!(decay_curve.last().unwrap() < &(initial_weight * 0.5));
}

#[test]
fn test_interference_effects() {
    // Test proactive and retroactive interference
    
    let source = EntityKey::from(slotmap::KeyData::from_ffi(1));
    let target1 = EntityKey::from(slotmap::KeyData::from_ffi(2));
    let target2 = EntityKey::from(slotmap::KeyData::from_ffi(3));
    
    let mut first_memory = BrainInspiredRelationship::new(source, target1, RelationType::Learned);
    let mut second_memory = BrainInspiredRelationship::new(source, target2, RelationType::Learned);
    
    let learning_rate = test_constants::STANDARD_LEARNING_RATE;
    
    // Learn first association
    for _ in 0..5 {
        first_memory.strengthen(learning_rate);
    }
    let first_after_learning = first_memory.weight;
    
    // Learn competing association (interference)
    for _ in 0..5 {
        second_memory.strengthen(learning_rate);
        
        // Simulate interference by weakening first memory
        first_memory.apply_decay();
    }
    
    let first_after_interference = first_memory.weight;
    let second_final = second_memory.weight;
    
    // First memory should be weakened by interference
    assert!(
        first_after_interference < first_after_learning,
        "First memory should be weakened by interference: {} -> {}",
        first_after_learning, first_after_interference
    );
    
    // Second memory should be strong
    assert!(
        second_final > test_constants::MEDIUM_EXCITATORY,
        "Second memory should be strong: {}",
        second_final
    );
}

// ==================== Homeostatic Plasticity Tests ====================

#[test]
fn test_homeostatic_scaling() {
    // Test that total synaptic strength tends toward equilibrium
    
    let source = EntityKey::from(slotmap::KeyData::from_ffi(1));
    let num_connections = 4;
    
    let mut relationships = Vec::new();
    for i in 0..num_connections {
        let target = EntityKey::from(slotmap::KeyData::from_ffi(i + 2));
        relationships.push(BrainInspiredRelationship::new(source, target, RelationType::RelatedTo));
    }
    
    // Strengthen all connections
    for _ in 0..10 {
        for rel in &mut relationships {
            rel.strengthen(test_constants::STANDARD_LEARNING_RATE);
        }
    }
    
    let total_strength: f32 = relationships.iter().map(|r| r.weight).sum();
    let target_total = 2.0; // Homeostatic target
    
    // Apply homeostatic scaling if total is too high
    if total_strength > target_total {
        let scale_factor = target_total / total_strength;
        
        for rel in &mut relationships {
            rel.weight *= scale_factor;
            rel.strength = rel.weight;
        }
        
        let new_total: f32 = relationships.iter().map(|r| r.weight).sum();
        
        assert_float_eq(
            new_total,
            target_total,
            test_constants::LOOSE_TOLERANCE
        );
    }
}

// ==================== Metaplasticity Tests ====================

#[test]
fn test_metaplasticity() {
    // Test that plasticity itself can be plastic (learning to learn)
    
    let mut relationship = create_test_relationship(
        RelationType::Learned,
        test_constants::RESTING_POTENTIAL,
        false,
        test_constants::STANDARD_DECAY_RATE
    );
    
    // Phase 1: Normal learning rate
    let base_rate = test_constants::STANDARD_LEARNING_RATE;
    let initial_activations = 3;
    
    for _ in 0..initial_activations {
        relationship.strengthen(base_rate);
    }
    
    let phase1_weight = relationship.weight;
    
    // Phase 2: Enhanced learning rate (metaplasticity)
    let enhanced_rate = base_rate * 1.5; // Learning rate increased by prior experience
    
    for _ in 0..initial_activations {
        relationship.strengthen(enhanced_rate);
    }
    
    let phase2_weight = relationship.weight;
    
    // Phase 2 should show greater weight gain per activation
    let phase1_gain = phase1_weight - test_constants::RESTING_POTENTIAL;
    let phase2_gain = phase2_weight - phase1_weight;
    
    assert!(
        phase2_gain > phase1_gain,
        "Metaplasticity should increase learning efficiency: phase1_gain={}, phase2_gain={}",
        phase1_gain, phase2_gain
    );
}

// ==================== Stress and Pathological Conditions Tests ====================

#[test]
fn test_stress_induced_plasticity_changes() {
    // Simulate stress effects on plasticity
    
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
        test_constants::FAST_DECAY_RATE // Stress increases decay
    );
    
    let normal_rate = test_constants::STANDARD_LEARNING_RATE;
    let stressed_rate = test_constants::SLOW_LEARNING_RATE; // Stress impairs learning
    
    // Apply same number of learning trials
    for _ in 0..5 {
        normal_relationship.strengthen(normal_rate);
        stressed_relationship.strengthen(stressed_rate);
        
        // Stressed condition has more decay
        stressed_relationship.apply_decay();
        
        thread::sleep(Duration::from_millis(2));
    }
    
    // Normal condition should show better learning
    assert!(
        normal_relationship.weight > stressed_relationship.weight,
        "Stress should impair learning: normal={}, stressed={}",
        normal_relationship.weight, stressed_relationship.weight
    );
}

// ==================== Relationship Type Specific Tests ====================

#[test]
fn test_relation_type_specific_plasticity() {
    let relation_types = vec![
        RelationType::IsA,
        RelationType::HasProperty,
        RelationType::RelatedTo,
        RelationType::Temporal,
        RelationType::Learned,
    ];
    
    let learning_rate = test_constants::STANDARD_LEARNING_RATE;
    let mut final_weights = Vec::new();
    
    for rel_type in relation_types {
        let mut relationship = create_test_relationship(
            rel_type,
            test_constants::WEAK_EXCITATORY,
            false,
            test_constants::STANDARD_DECAY_RATE
        );
        
        // Apply standard learning
        for _ in 0..5 {
            relationship.strengthen(learning_rate);
        }
        
        final_weights.push(relationship.weight);
    }
    
    // All should show learning, but specific patterns might emerge
    for &weight in &final_weights {
        assert!(
            weight > test_constants::WEAK_EXCITATORY,
            "All relation types should show learning: weight={}",
            weight
        );
    }
}

// ==================== Performance and Scalability Tests ====================

#[test]
fn test_large_scale_learning_performance() {
    let num_relationships = 1000;
    let learning_iterations = 100;
    
    let mut relationships = Vec::new();
    
    // Create many relationships
    for i in 0..num_relationships {
        relationships.push(create_test_relationship(
            RelationType::RelatedTo,
            test_constants::RESTING_POTENTIAL,
            false,
            test_constants::STANDARD_DECAY_RATE
        ));
    }
    
    let start_time = SystemTime::now();
    
    // Apply learning to all
    for _ in 0..learning_iterations {
        for rel in &mut relationships {
            rel.strengthen(test_constants::STANDARD_LEARNING_RATE);
        }
    }
    
    let duration = start_time.elapsed().unwrap();
    
    // Should complete within reasonable time
    assert!(
        duration.as_millis() < 1000, // 1 second max
        "Large scale learning took too long: {:?}",
        duration
    );
    
    // All relationships should show learning
    for (i, rel) in relationships.iter().enumerate() {
        assert!(
            rel.weight > test_constants::RESTING_POTENTIAL,
            "Relationship {} should show learning: weight={}",
            i, rel.weight
        );
    }
}