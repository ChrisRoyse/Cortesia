/// Comprehensive tests for BrainInspiredRelationship strengthen and decay methods
/// 
/// This module tests Hebbian learning, temporal decay, and relationship dynamics
/// with various scenarios including edge cases and stress conditions.

use llmkg::core::brain_types::{BrainInspiredRelationship, RelationType};
use llmkg::core::types::EntityKey;
use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use std::thread;
use super::test_helpers::*;
use super::test_helpers::{
    generate_learning_rates, generate_decay_rates, generate_edge_case_activations, assert_float_eq
};

#[test]
fn test_relationship_creation() {
    let source = EntityKey::default();
    let target = EntityKey::default();
    let rel = BrainInspiredRelationship::new(source, target, RelationType::IsA);
    
    assert_eq!(rel.source, source);
    assert_eq!(rel.target, target);
    assert_eq!(rel.source_key, source); // Duplicate field
    assert_eq!(rel.target_key, target); // Duplicate field
    assert_eq!(rel.relation_type, RelationType::IsA);
    assert_float_eq(rel.weight, 1.0, 0.001);
    assert_float_eq(rel.strength, 1.0, 0.001);
    assert_eq!(rel.is_inhibitory, false);
    assert_float_eq(rel.temporal_decay, 0.1, 0.001);
    assert_eq!(rel.activation_count, 0);
    assert_eq!(rel.usage_count, 0);
    assert!(rel.metadata.is_empty());
    
    // Timestamps should be approximately now
    let now = SystemTime::now();
    assert!(rel.last_strengthened <= now);
    assert!(rel.last_update <= now);
    assert!(rel.creation_time <= now);
    assert!(rel.ingestion_time <= now);
}

#[test]
fn test_all_relation_types() {
    let relation_types = vec![
        RelationType::IsA,
        RelationType::HasInstance,
        RelationType::HasProperty,
        RelationType::RelatedTo,
        RelationType::PartOf,
        RelationType::Similar,
        RelationType::Opposite,
        RelationType::Temporal,
        RelationType::Learned,
    ];
    
    for rel_type in relation_types {
        let rel = BrainInspiredRelationship::new(
            EntityKey::default(),
            EntityKey::default(),
            rel_type
        );
        assert_eq!(rel.relation_type, rel_type);
    }
}

#[test]
fn test_basic_strengthening() {
    let mut rel = create_test_relationship(RelationType::RelatedTo, 0.5, false, 0.1);
    
    let initial_weight = rel.weight;
    let initial_count = rel.activation_count;
    let initial_strengthened = rel.last_strengthened;
    
    // Small delay to ensure timestamp difference
    thread::sleep(Duration::from_millis(10));
    
    rel.strengthen(0.2);
    
    // Weight should increase
    assert_float_eq(rel.weight, initial_weight + 0.2, 0.001);
    assert_float_eq(rel.strength, rel.weight, 0.001); // Should match weight
    
    // Counters should increment
    assert_eq!(rel.activation_count, initial_count + 1);
    assert_eq!(rel.usage_count, initial_count + 1);
    
    // Timestamps should be updated
    assert!(rel.last_strengthened > initial_strengthened);
    assert!(rel.last_update >= rel.last_strengthened);
}

#[test]
fn test_strengthening_clamping() {
    let mut rel = create_test_relationship(RelationType::Similar, 0.9, false, 0.1);
    
    // Strengthen beyond 1.0
    rel.strengthen(0.5);
    
    // Should be clamped to 1.0
    assert_float_eq(rel.weight, 1.0, 0.001);
    assert_float_eq(rel.strength, 1.0, 0.001);
}

#[test]
fn test_multiple_strengthenings() {
    let mut rel = create_test_relationship(RelationType::Learned, 0.1, false, 0.1);
    
    let learning_steps = vec![0.1, 0.15, 0.2, 0.05, 0.3];
    let mut expected_weight = 0.1;
    
    for (i, learning_rate) in learning_steps.iter().enumerate() {
        rel.strengthen(*learning_rate);
        expected_weight = (expected_weight + learning_rate).min(1.0);
        
        assert_float_eq(rel.weight, expected_weight, 0.001);
        assert_eq!(rel.activation_count, i as u64 + 1);
        assert_eq!(rel.usage_count, i as u64 + 1);
    }
}

#[test]
fn test_negative_learning_rate() {
    let mut rel = create_test_relationship(RelationType::RelatedTo, 0.7, false, 0.1);
    
    // Negative learning rate should still work (though may be unusual)
    rel.strengthen(-0.2);
    
    assert_float_eq(rel.weight, 0.5, 0.001); // 0.7 + (-0.2) = 0.5
    assert_eq!(rel.activation_count, 1);
}

#[test]
fn test_large_learning_rates() {
    let mut rel = create_test_relationship(RelationType::HasProperty, 0.1, false, 0.1);
    
    // Very large learning rate
    rel.strengthen(5.0);
    
    // Should be clamped to 1.0
    assert_float_eq(rel.weight, 1.0, 0.001);
}

#[test]
fn test_basic_decay() {
    let mut rel = create_test_relationship(RelationType::Temporal, 0.8, false, 0.2);
    
    // Set an old timestamp
    rel.last_strengthened = SystemTime::now() - Duration::from_secs(5);
    
    let result = rel.apply_decay();
    
    // Expected: 0.8 * exp(-0.2 * 5) = 0.8 * exp(-1) ≈ 0.8 * 0.368 ≈ 0.294
    let expected = 0.8 * (-0.2 * 5.0).exp();
    assert_float_eq(result, expected, 0.01);
    assert_float_eq(rel.weight, expected, 0.01);
    assert_float_eq(rel.strength, expected, 0.01);
}

#[test]
fn test_different_decay_rates() {
    let decay_scenarios = vec![
        (0.0, 1.0),    // No decay
        (0.05, 0.95),  // Very slow decay (1 second)
        (0.1, 0.90),   // Slow decay
        (0.3, 0.74),   // Moderate decay
        (0.7, 0.50),   // Fast decay
        (1.0, 0.37),   // Very fast decay
    ];
    
    for (decay_rate, expected_factor) in decay_scenarios {
        let mut rel = create_test_relationship(RelationType::Similar, 1.0, false, decay_rate);
        rel.last_strengthened = SystemTime::now() - Duration::from_secs(1);
        
        let result = rel.apply_decay();
        assert_float_eq(result, expected_factor, 0.02);
    }
}

#[test]
fn test_immediate_decay() {
    let mut rel = create_test_relationship(RelationType::RelatedTo, 0.9, false, 0.5);
    
    // Apply decay immediately (no time elapsed)
    let result = rel.apply_decay();
    
    // Should be virtually unchanged
    assert_float_eq(result, 0.9, 0.001);
}

#[test]
fn test_extreme_decay_scenarios() {
    let mut rel = create_test_relationship(RelationType::Opposite, 1.0, false, 1.0);
    
    // Very old relationship
    rel.last_strengthened = SystemTime::now() - Duration::from_secs(100);
    
    let result = rel.apply_decay();
    
    // Should be nearly zero
    assert!(result < 0.001);
    assert!(rel.weight < 0.001);
}

#[test]
fn test_strengthen_after_decay() {
    let mut rel = create_test_relationship(RelationType::Learned, 0.8, false, 0.3);
    
    // Apply some decay first
    rel.last_strengthened = SystemTime::now() - Duration::from_secs(2);
    rel.apply_decay();
    
    let decayed_weight = rel.weight;
    assert!(decayed_weight < 0.8);
    
    // Now strengthen
    rel.strengthen(0.4);
    
    // Should add to decayed weight and clamp
    let expected = (decayed_weight + 0.4).min(1.0);
    assert_float_eq(rel.weight, expected, 0.01);
}

#[test]
fn test_decay_after_strengthen() {
    let mut rel = create_test_relationship(RelationType::PartOf, 0.3, false, 0.2);
    
    // Strengthen first
    rel.strengthen(0.5);
    assert_float_eq(rel.weight, 0.8, 0.001);
    
    // Manually set old timestamp for decay
    rel.last_strengthened = SystemTime::now() - Duration::from_secs(3);
    
    let result = rel.apply_decay();
    let expected = 0.8 * (-0.2 * 3.0).exp(); // ≈ 0.8 * 0.549 ≈ 0.439
    assert_float_eq(result, expected, 0.02);
}

#[test]
fn test_inhibitory_relationships() {
    let mut rel = create_test_relationship(RelationType::Opposite, 0.6, true, 0.1);
    
    assert!(rel.is_inhibitory);
    
    // Inhibitory property doesn't affect strengthen/decay logic
    rel.strengthen(0.2);
    assert_float_eq(rel.weight, 0.8, 0.001);
    
    rel.last_strengthened = SystemTime::now() - Duration::from_secs(1);
    let result = rel.apply_decay();
    let expected = 0.8 * (-0.1 * 1.0).exp();
    assert_float_eq(result, expected, 0.01);
}

#[test]
fn test_zero_weight_scenarios() {
    let mut rel = create_test_relationship(RelationType::RelatedTo, 0.0, false, 0.5);
    
    // Strengthen from zero
    rel.strengthen(0.3);
    assert_float_eq(rel.weight, 0.3, 0.001);
    
    // Reset to zero and apply decay
    rel.weight = 0.0;
    rel.strength = 0.0;
    rel.last_strengthened = SystemTime::now() - Duration::from_secs(10);
    
    let result = rel.apply_decay();
    assert_float_eq(result, 0.0, 0.001); // Zero remains zero
}

#[test]
fn test_activation_counting() {
    let mut rel = create_test_relationship(RelationType::HasInstance, 0.5, false, 0.1);
    
    // Multiple strengthenings should increment counters
    for i in 1..=5 {
        rel.strengthen(0.05);
        assert_eq!(rel.activation_count, i);
        assert_eq!(rel.usage_count, i); // Should be identical
    }
    
    // Decay doesn't affect counters
    rel.apply_decay();
    assert_eq!(rel.activation_count, 5);
    assert_eq!(rel.usage_count, 5);
}

#[test]
fn test_timestamp_updates() {
    let mut rel = create_test_relationship(RelationType::Temporal, 0.4, false, 0.2);
    
    let initial_strengthened = rel.last_strengthened;
    let initial_update = rel.last_update;
    
    thread::sleep(Duration::from_millis(10));
    
    // Strengthen should update both timestamps
    rel.strengthen(0.1);
    assert!(rel.last_strengthened > initial_strengthened);
    assert!(rel.last_update >= rel.last_strengthened);
    
    let after_strengthen_update = rel.last_update;
    thread::sleep(Duration::from_millis(10));
    
    // Decay should update last_update but not last_strengthened
    rel.apply_decay();
    assert!(rel.last_update > after_strengthen_update);
    // last_strengthened should remain unchanged from strengthen call
}

#[test]
fn test_metadata_management() {
    let mut rel = create_test_relationship(RelationType::RelatedTo, 0.5, false, 0.1);
    
    // Add metadata
    rel.metadata.insert("source".to_string(), "test_data".to_string());
    rel.metadata.insert("confidence".to_string(), "0.85".to_string());
    rel.metadata.insert("created_by".to_string(), "neural_network".to_string());
    
    assert_eq!(rel.metadata.len(), 3);
    assert_eq!(rel.metadata.get("source"), Some(&"test_data".to_string()));
    assert_eq!(rel.metadata.get("confidence"), Some(&"0.85".to_string()));
    
    // Strengthen and decay shouldn't affect metadata
    rel.strengthen(0.2);
    rel.apply_decay();
    assert_eq!(rel.metadata.len(), 3);
}

#[test]
fn test_concurrent_operations_simulation() {
    let mut rel = create_test_relationship(RelationType::Learned, 0.2, false, 0.05);
    
    // Simulate rapid alternating strengthen/decay operations
    for i in 0..10 {
        if i % 2 == 0 {
            rel.strengthen(0.1);
        } else {
            // Introduce small delay and apply decay
            rel.last_strengthened = SystemTime::now() - Duration::from_millis(100);
            rel.apply_decay();
        }
        
        // Weight should remain in valid range
        assert!(rel.weight >= 0.0);
        assert!(rel.weight <= 1.0);
    }
}

#[test]
fn test_rapid_strengthening_sequence() {
    let mut rel = create_test_relationship(RelationType::Similar, 0.1, false, 0.1);
    
    // Rapid successive strengthenings
    let learning_rates = vec![0.05, 0.1, 0.15, 0.2, 0.05, 0.1];
    
    for rate in learning_rates {
        rel.strengthen(rate);
        assert!(rel.weight <= 1.0); // Should never exceed 1.0
        assert!(rel.weight >= 0.1); // Should generally increase
    }
    
    // Final weight should be 1.0 (clamped)
    assert_float_eq(rel.weight, 1.0, 0.001);
}

#[test]
fn test_long_term_decay_behavior() {
    let mut rel = create_test_relationship(RelationType::Temporal, 1.0, false, 0.1);
    
    // Simulate decay over multiple time periods
    let time_periods = vec![1, 2, 5, 10, 20]; // seconds
    let mut previous_weight = 1.0;
    
    for period in time_periods {
        rel.last_strengthened = SystemTime::now() - Duration::from_secs(period);
        let current_weight = rel.apply_decay();
        
        // Weight should monotonically decrease
        assert!(current_weight < previous_weight);
        assert!(current_weight >= 0.0);
        
        previous_weight = current_weight;
    }
}

#[test]
fn test_weight_strength_consistency() {
    let mut rel = create_test_relationship(RelationType::RelatedTo, 0.3, false, 0.2);
    
    // Throughout all operations, weight and strength should remain equal
    assert_float_eq(rel.weight, rel.strength, 0.001);
    
    rel.strengthen(0.4);
    assert_float_eq(rel.weight, rel.strength, 0.001);
    
    rel.last_strengthened = SystemTime::now() - Duration::from_secs(2);
    rel.apply_decay();
    assert_float_eq(rel.weight, rel.strength, 0.001);
    
    rel.strengthen(0.1);
    assert_float_eq(rel.weight, rel.strength, 0.001);
}

#[test]
fn test_boundary_learning_rates() {
    let test_cases = vec![
        (0.0, 0.5, 0.5),      // Zero learning rate
        (0.001, 0.5, 0.501),  // Minimal learning rate
        (0.5, 0.5, 1.0),      // Exactly to maximum
        (1.0, 0.0, 1.0),      // Maximum learning rate from zero
        (2.0, 0.5, 1.0),      // Excessive learning rate (clamped)
    ];
    
    for (learning_rate, initial_weight, expected_final) in test_cases {
        let mut rel = create_test_relationship(RelationType::RelatedTo, initial_weight, false, 0.1);
        rel.strengthen(learning_rate);
        assert_float_eq(rel.weight, expected_final, 0.001);
    }
}

#[test]
fn test_boundary_decay_rates() {
    let test_cases = vec![
        (0.0, 1.0, 1.0),     // Zero decay rate
        (0.001, 1.0, 0.999), // Minimal decay rate (1 second)
        (10.0, 1.0, 0.0),    // Extreme decay rate
    ];
    
    for (decay_rate, initial_weight, expected_factor) in test_cases {
        let mut rel = create_test_relationship(RelationType::Temporal, initial_weight, false, decay_rate);
        rel.last_strengthened = SystemTime::now() - Duration::from_secs(1);
        
        let result = rel.apply_decay();
        assert_float_eq(result, expected_factor, 0.02);
    }
}

// ==================== Enhanced Learning Rate Variation Tests ====================

#[test]
fn test_comprehensive_learning_rate_variations() {
    let learning_rates = generate_learning_rates();
    let initial_weights = vec![0.0, 0.1, 0.5, 0.9, 1.0];
    
    for &initial_weight in &initial_weights {
        for &learning_rate in &learning_rates {
            let mut rel = create_test_relationship(RelationType::Learned, initial_weight, false, 0.1);
            let initial_count = rel.activation_count;
            
            rel.strengthen(learning_rate);
            
            // Check weight calculation
            let expected_weight = (initial_weight + learning_rate).min(1.0);
            assert_float_eq(rel.weight, expected_weight, 0.001);
            assert_float_eq(rel.strength, expected_weight, 0.001);
            
            // Check counters incremented
            assert_eq!(rel.activation_count, initial_count + 1);
            assert_eq!(rel.usage_count, initial_count + 1);
            
            // Check timestamps updated
            assert!(rel.last_strengthened <= SystemTime::now());
            assert!(rel.last_update >= rel.last_strengthened);
        }
    }
}

#[test]
fn test_extreme_learning_rate_edge_cases() {
    let extreme_rates = vec![
        f32::NEG_INFINITY,
        -1000.0,
        -1.0,
        f32::INFINITY,
        f32::NAN,
        1000.0,
    ];
    
    for &rate in &extreme_rates {
        let mut rel = create_test_relationship(RelationType::Learned, 0.5, false, 0.1);
        let initial_weight = rel.weight;
        
        rel.strengthen(rate);
        
        if rate.is_nan() {
            // NaN learning rate might produce NaN weight
            assert!(rel.weight.is_nan() || (rel.weight >= 0.0 && rel.weight <= 1.0));
        } else if rate.is_infinite() || rate >= 1000.0 {
            // Very large positive rates should saturate to 1.0
            assert_eq!(rel.weight, 1.0, "Large positive learning rate should saturate");
        } else if rate <= -1000.0 || rate == f32::NEG_INFINITY {
            // Very large negative rates might underflow but shouldn't break
            assert!(rel.weight >= 0.0 || rel.weight.is_infinite(), "Should handle negative rates gracefully");
        }
        
        // Counters should still increment regardless of rate value
        assert_eq!(rel.activation_count, 1);
    }
}

#[test]
fn test_sequential_learning_rate_applications() {
    let mut rel = create_test_relationship(RelationType::Learned, 0.1, false, 0.1);
    
    // Apply a sequence of different learning rates
    let learning_sequence = vec![0.1, 0.05, 0.2, 0.15, 0.3, 0.1];
    let mut expected_weight = 0.1;
    
    for (i, &rate) in learning_sequence.iter().enumerate() {
        rel.strengthen(rate);
        expected_weight = (expected_weight + rate).min(1.0);
        
        assert_float_eq(rel.weight, expected_weight, 0.001);
        assert_eq!(rel.activation_count, i as u64 + 1);
        
        // Small delay to ensure timestamps are different
        thread::sleep(Duration::from_millis(1));
    }
    
    // Final weight should be 1.0 (saturated)
    assert_eq!(rel.weight, 1.0);
}

#[test]
fn test_learning_rate_with_different_relation_types() {
    let relation_types = vec![
        RelationType::IsA,
        RelationType::HasInstance,
        RelationType::HasProperty,
        RelationType::RelatedTo,
        RelationType::PartOf,
        RelationType::Similar,
        RelationType::Opposite,
        RelationType::Temporal,
        RelationType::Learned,
    ];
    
    let learning_rate = 0.3;
    let initial_weight = 0.2;
    
    for rel_type in relation_types {
        let mut rel = create_test_relationship(rel_type, initial_weight, false, 0.1);
        
        rel.strengthen(learning_rate);
        
        // Learning should work the same regardless of relation type
        let expected = initial_weight + learning_rate;
        assert_float_eq(rel.weight, expected, 0.001);
        assert_eq!(rel.relation_type, rel_type);
    }
}

#[test]
fn test_learning_rate_inhibitory_vs_excitatory() {
    let learning_rate = 0.4;
    let initial_weight = 0.3;
    
    // Test excitatory relationship
    let mut excitatory = create_test_relationship(RelationType::RelatedTo, initial_weight, false, 0.1);
    excitatory.strengthen(learning_rate);
    
    // Test inhibitory relationship
    let mut inhibitory = create_test_relationship(RelationType::RelatedTo, initial_weight, true, 0.1);
    inhibitory.strengthen(learning_rate);
    
    // Learning should work the same way regardless of inhibitory flag
    assert_float_eq(excitatory.weight, inhibitory.weight, 0.001);
    assert_eq!(excitatory.weight, initial_weight + learning_rate);
    
    // But inhibitory flag should be preserved
    assert!(!excitatory.is_inhibitory);
    assert!(inhibitory.is_inhibitory);
}

// ==================== Enhanced Decay Testing ====================

#[test]
fn test_comprehensive_decay_rate_variations() {
    let decay_rates = generate_decay_rates();
    let initial_weights = vec![0.1, 0.5, 0.9, 1.0];
    let time_intervals = vec![
        Duration::from_millis(100),
        Duration::from_secs(1),
        Duration::from_secs(5),
        Duration::from_secs(10),
    ];
    
    for &initial_weight in &initial_weights {
        for &decay_rate in &decay_rates {
            for &time_interval in &time_intervals {
                let mut rel = create_test_relationship(RelationType::Temporal, initial_weight, false, decay_rate);
                rel.last_strengthened = SystemTime::now() - time_interval;
                
                let result = rel.apply_decay();
                
                // Calculate expected decay
                let time_secs = time_interval.as_secs_f32();
                let expected = initial_weight * (-decay_rate * time_secs).exp();
                
                if decay_rate == 0.0 {
                    // No decay
                    assert_float_eq(result, initial_weight, 0.001);
                } else if decay_rate.is_finite() && decay_rate > 0.0 {
                    // Normal decay
                    assert_float_eq(result, expected, 0.02);
                    assert!(result <= initial_weight, "Decay should not increase weight");
                    assert!(result >= 0.0, "Weight should not go negative");
                } else {
                    // Handle extreme decay rates gracefully
                    assert!(result >= 0.0, "Weight should remain non-negative");
                }
            }
        }
    }
}

#[test]
fn test_decay_with_extreme_timestamps() {
    let mut rel = create_test_relationship(RelationType::Temporal, 0.8, false, 0.1);
    
    // Test with future timestamp (clock skew)
    rel.last_strengthened = SystemTime::now() + Duration::from_secs(1);
    let result = rel.apply_decay();
    
    // Should handle gracefully (elapsed() returns Ok(Duration::ZERO) or Default)
    assert!(result >= 0.0 && result <= 1.0);
    
    // Test with very old timestamp
    rel.weight = 0.9;
    rel.strength = 0.9;
    rel.last_strengthened = SystemTime::UNIX_EPOCH;
    let result = rel.apply_decay();
    
    // Should decay to essentially zero
    assert!(result < 0.001);
}

#[test]
fn test_decay_and_strengthen_interaction_patterns() {
    let mut rel = create_test_relationship(RelationType::Learned, 0.5, false, 0.2);
    
    // Pattern 1: Strengthen, wait, decay, strengthen again
    rel.strengthen(0.3); // Weight becomes 0.8
    assert_float_eq(rel.weight, 0.8, 0.001);
    
    thread::sleep(Duration::from_millis(100));
    rel.apply_decay(); // Some decay occurs
    let after_decay = rel.weight;
    assert!(after_decay < 0.8);
    
    rel.strengthen(0.1); // Strengthen again
    assert!(rel.weight > after_decay);
    
    // Pattern 2: Rapid strengthen-decay cycles
    for _ in 0..5 {
        let before = rel.weight;
        rel.strengthen(0.05);
        let after_strengthen = rel.weight;
        
        // Small delay for decay
        rel.last_strengthened = SystemTime::now() - Duration::from_millis(50);
        rel.apply_decay();
        let after_decay = rel.weight;
        
        assert!(after_strengthen > before, "Strengthen should increase weight");
        assert!(after_decay <= after_strengthen, "Decay should not increase weight");
    }
}

// ==================== Property-Based Testing ====================

#[test]
fn test_relationship_weight_bounds_property() {
    let learning_rates = generate_learning_rates();
    let initial_weights = vec![0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
    
    for &initial in &initial_weights {
        for &rate in &learning_rates {
            let mut rel = create_test_relationship(RelationType::RelatedTo, initial, false, 0.1);
            
            rel.strengthen(rate);
            
            // Property: Weight should always be in [0, 1] range after strengthening
            if !rel.weight.is_nan() {
                assert!(rel.weight >= 0.0, "Weight should be non-negative");
                assert!(rel.weight <= 1.0, "Weight should not exceed 1.0");
            }
            
            // Property: Weight and strength should always be equal
            assert_float_eq(rel.weight, rel.strength, 0.001);
        }
    }
}

#[test]
fn test_relationship_monotonic_strengthening_property() {
    let mut rel = create_test_relationship(RelationType::RelatedTo, 0.1, false, 0.0); // No decay
    
    // Property: Successive positive strengthening should not decrease weight
    let positive_rates = vec![0.01, 0.05, 0.1, 0.2, 0.05, 0.1];
    let mut previous_weight = rel.weight;
    
    for &rate in &positive_rates {
        rel.strengthen(rate);
        
        if rate >= 0.0 {
            assert!(rel.weight >= previous_weight, 
                   "Positive strengthening should not decrease weight: {} -> {} (rate: {})",
                   previous_weight, rel.weight, rate);
        }
        
        previous_weight = rel.weight;
        if previous_weight >= 1.0 {
            break; // Stop at saturation
        }
    }
}

#[test]
fn test_relationship_decay_monotonicity_property() {
    let mut rel = create_test_relationship(RelationType::Temporal, 1.0, false, 0.3);
    
    // Property: Successive decay applications should not increase weight
    for i in 1..=10 {
        let previous_weight = rel.weight;
        rel.last_strengthened = SystemTime::now() - Duration::from_millis(i * 50);
        
        let result = rel.apply_decay();
        
        assert!(result <= previous_weight,
               "Decay should not increase weight: {} -> {} (iteration {})",
               previous_weight, result, i);
    }
}

#[test]
fn test_relationship_counter_monotonicity_property() {
    let mut rel = create_test_relationship(RelationType::RelatedTo, 0.5, false, 0.1);
    
    // Property: Strengthening should always increment counters
    for i in 1..=10 {
        let prev_activation = rel.activation_count;
        let prev_usage = rel.usage_count;
        
        rel.strengthen(0.01);
        
        assert_eq!(rel.activation_count, prev_activation + 1,
                  "Activation count should increment");
        assert_eq!(rel.usage_count, prev_usage + 1,
                  "Usage count should increment");
        assert_eq!(rel.activation_count, rel.usage_count,
                  "Counters should remain equal");
    }
    
    // Property: Decay should not affect counters
    let count_before_decay = rel.activation_count;
    rel.apply_decay();
    assert_eq!(rel.activation_count, count_before_decay,
              "Decay should not affect counters");
}

#[test]
fn test_relationship_temporal_consistency_property() {
    let mut rel = create_test_relationship(RelationType::Temporal, 0.5, false, 0.1);
    
    let start_time = SystemTime::now();
    
    // Property: Timestamps should be consistent and non-decreasing
    rel.strengthen(0.1);
    let first_strengthen_time = rel.last_strengthened;
    let first_update_time = rel.last_update;
    
    assert!(first_strengthen_time >= start_time, "Strengthen time should be recent");
    assert!(first_update_time >= first_strengthen_time, "Update time should be >= strengthen time");
    
    thread::sleep(Duration::from_millis(10));
    
    rel.apply_decay();
    let after_decay_update = rel.last_update;
    
    assert!(after_decay_update > first_update_time, "Decay should update timestamp");
    assert_eq!(rel.last_strengthened, first_strengthen_time, "Decay should not change strengthen time");
    
    thread::sleep(Duration::from_millis(10));
    
    rel.strengthen(0.05);
    let second_strengthen_time = rel.last_strengthened;
    
    assert!(second_strengthen_time > first_strengthen_time, "Second strengthen should have later timestamp");
}

#[test]
fn test_relationship_saturation_stability_property() {
    let mut rel = create_test_relationship(RelationType::RelatedTo, 0.9, false, 0.0); // No decay
    
    // Reach saturation
    rel.strengthen(0.2); // Should reach 1.0
    assert_eq!(rel.weight, 1.0);
    
    // Property: Once saturated, further strengthening should not change weight
    for _ in 0..5 {
        let prev_weight = rel.weight;
        rel.strengthen(0.1);
        assert_eq!(rel.weight, prev_weight, "Saturated weight should remain stable");
    }
    
    // But counters should still increment
    assert_eq!(rel.activation_count, 6); // Initial strengthen + 5 more
}

#[test]
fn test_relationship_metadata_persistence_property() {
    let mut rel = create_test_relationship(RelationType::RelatedTo, 0.5, false, 0.1);
    
    // Add metadata
    rel.metadata.insert("source".to_string(), "test".to_string());
    rel.metadata.insert("confidence".to_string(), "0.95".to_string());
    let original_metadata = rel.metadata.clone();
    
    // Property: Metadata should persist through strengthen/decay operations
    for _ in 0..3 {
        rel.strengthen(0.1);
        assert_eq!(rel.metadata, original_metadata, "Metadata should persist through strengthening");
        
        rel.apply_decay();
        assert_eq!(rel.metadata, original_metadata, "Metadata should persist through decay");
    }
}

// ==================== Stress Testing ====================

#[test]
fn test_relationship_high_frequency_operations() {
    let mut rel = create_test_relationship(RelationType::Learned, 0.1, false, 0.05);
    
    // Rapid alternating strengthen/decay operations
    for i in 0..100 {
        if i % 2 == 0 {
            rel.strengthen(0.01);
        } else {
            // Introduce small delay and decay
            rel.last_strengthened = SystemTime::now() - Duration::from_millis(1);
            rel.apply_decay();
        }
        
        // Properties should hold throughout
        assert!(rel.weight >= 0.0 && rel.weight <= 1.0, "Weight should remain in bounds");
        assert_float_eq(rel.weight, rel.strength, 0.001, "Weight and strength should match");
    }
    
    // Should have reasonable final values
    assert!(rel.activation_count >= 50, "Should have many activations");
    assert!(rel.weight > 0.0, "Should have some weight remaining");
}