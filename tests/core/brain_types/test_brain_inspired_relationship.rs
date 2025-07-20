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