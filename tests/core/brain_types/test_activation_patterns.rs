/// Comprehensive tests for ActivationPattern and ActivationStep functionality
/// 
/// This module tests neural propagation patterns, reasoning traces, and
/// activation management with various scenarios and edge cases.

use llmkg::core::brain_types::{
    ActivationPattern, ActivationStep, ActivationOperation,
    BrainInspiredEntity, EntityDirection
};
use llmkg::core::types::EntityKey;
use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use super::test_helpers::*;

#[test]
fn test_activation_pattern_creation() {
    let pattern = ActivationPattern::new("test_query".to_string());
    
    assert_eq!(pattern.query, "test_query");
    assert!(pattern.activations.is_empty());
    assert!(pattern.timestamp <= SystemTime::now());
}

#[test]
fn test_activation_pattern_basic_operations() {
    let mut pattern = ActivationPattern::new("basic_test".to_string());
    
    // Add some activations
    let key1 = EntityKey::default();
    let key2 = EntityKey::default();
    let key3 = EntityKey::default();
    
    pattern.activations.insert(key1, 0.8);
    pattern.activations.insert(key2, 0.3);
    pattern.activations.insert(key3, 0.9);
    
    assert_eq!(pattern.activations.len(), 3);
    assert_float_eq(*pattern.activations.get(&key1).unwrap(), 0.8, 0.001);
    assert_float_eq(*pattern.activations.get(&key2).unwrap(), 0.3, 0.001);
    assert_float_eq(*pattern.activations.get(&key3).unwrap(), 0.9, 0.001);
}

#[test]
fn test_get_top_activations_basic() {
    let mut pattern = ActivationPattern::new("top_test".to_string());
    
    // Create test keys (using placeholder approach since we can't easily create unique keys)
    let keys: Vec<EntityKey> = (0..5).map(|_| EntityKey::default()).collect();
    let activations = vec![0.2, 0.8, 0.1, 0.9, 0.5];
    
    for (i, &activation) in activations.iter().enumerate() {
        pattern.activations.insert(keys[i], activation);
    }
    
    // Get top 3 activations
    let top_3 = pattern.get_top_activations(3);
    
    assert_eq!(top_3.len(), 3);
    // Should be sorted in descending order: 0.9, 0.8, 0.5
    assert_float_eq(top_3[0].1, 0.9, 0.001);
    assert_float_eq(top_3[1].1, 0.8, 0.001);
    assert_float_eq(top_3[2].1, 0.5, 0.001);
}

#[test]
fn test_get_top_activations_edge_cases() {
    let mut pattern = ActivationPattern::new("edge_test".to_string());
    
    // Test with empty pattern
    let empty_top = pattern.get_top_activations(5);
    assert!(empty_top.is_empty());
    
    // Add single activation
    let key = EntityKey::default();
    pattern.activations.insert(key, 0.7);
    
    // Request more than available
    let single_top = pattern.get_top_activations(10);
    assert_eq!(single_top.len(), 1);
    assert_float_eq(single_top[0].1, 0.7, 0.001);
    
    // Request zero
    let zero_top = pattern.get_top_activations(0);
    assert!(zero_top.is_empty());
}

#[test]
fn test_get_top_activations_with_ties() {
    let mut pattern = ActivationPattern::new("ties_test".to_string());
    
    let keys: Vec<EntityKey> = (0..4).map(|_| EntityKey::default()).collect();
    let activations = vec![0.8, 0.8, 0.5, 0.8]; // Multiple ties
    
    for (i, &activation) in activations.iter().enumerate() {
        pattern.activations.insert(keys[i], activation);
    }
    
    let top_2 = pattern.get_top_activations(2);
    assert_eq!(top_2.len(), 2);
    
    // All top results should have value 0.8 (due to ties)
    for (_, value) in &top_2 {
        assert_float_eq(*value, 0.8, 0.001);
    }
}

#[test]
fn test_get_top_activations_boundary_values() {
    let mut pattern = ActivationPattern::new("boundary_test".to_string());
    
    let keys: Vec<EntityKey> = (0..5).map(|_| EntityKey::default()).collect();
    let boundary_values = vec![0.0, 0.001, 0.5, 0.999, 1.0];
    
    for (i, &activation) in boundary_values.iter().enumerate() {
        pattern.activations.insert(keys[i], activation);
    }
    
    let top_all = pattern.get_top_activations(5);
    
    // Should be sorted: 1.0, 0.999, 0.5, 0.001, 0.0
    assert_float_eq(top_all[0].1, 1.0, 0.001);
    assert_float_eq(top_all[1].1, 0.999, 0.001);
    assert_float_eq(top_all[2].1, 0.5, 0.001);
    assert_float_eq(top_all[3].1, 0.001, 0.001);
    assert_float_eq(top_all[4].1, 0.0, 0.001);
}

#[test]
fn test_activation_pattern_large_scale() {
    let mut pattern = ActivationPattern::new("large_scale".to_string());
    
    // Create many activations
    let num_activations = 100;
    let keys: Vec<EntityKey> = (0..num_activations).map(|_| EntityKey::default()).collect();
    
    for (i, key) in keys.iter().enumerate() {
        let activation = (i as f32) / (num_activations as f32); // 0.0 to ~1.0
        pattern.activations.insert(*key, activation);
    }
    
    assert_eq!(pattern.activations.len(), num_activations);
    
    // Test top activations with large dataset
    let top_10 = pattern.get_top_activations(10);
    assert_eq!(top_10.len(), 10);
    
    // Should be in descending order
    for i in 0..9 {
        assert!(top_10[i].1 >= top_10[i + 1].1);
    }
    
    // Highest should be close to 1.0
    assert!(top_10[0].1 > 0.9);
}

#[test]
fn test_activation_step_creation() {
    let step = ActivationStep {
        step_id: 1,
        entity_key: EntityKey::default(),
        concept_id: "test_concept".to_string(),
        activation_level: 0.75,
        operation_type: ActivationOperation::Initialize,
        timestamp: SystemTime::now(),
    };
    
    assert_eq!(step.step_id, 1);
    assert_eq!(step.concept_id, "test_concept");
    assert_float_eq(step.activation_level, 0.75, 0.001);
    assert_eq!(step.operation_type, ActivationOperation::Initialize);
    assert!(step.timestamp <= SystemTime::now());
}

#[test]
fn test_all_activation_operations() {
    let operations = vec![
        ActivationOperation::Initialize,
        ActivationOperation::Propagate,
        ActivationOperation::Inhibit,
        ActivationOperation::Reinforce,
        ActivationOperation::Decay,
    ];
    
    for (i, operation) in operations.iter().enumerate() {
        let step = ActivationStep {
            step_id: i,
            entity_key: EntityKey::default(),
            concept_id: format!("concept_{}", i),
            activation_level: 0.5,
            operation_type: *operation,
            timestamp: SystemTime::now(),
        };
        
        assert_eq!(step.operation_type, *operation);
    }
}

#[test]
fn test_activation_trace_sequence() {
    let mut trace = Vec::new();
    let base_time = SystemTime::now();
    
    // Create a sequence of activation steps
    let steps_data = vec![
        (ActivationOperation::Initialize, 0.0, "Initialize network"),
        (ActivationOperation::Propagate, 0.3, "Forward propagation"),
        (ActivationOperation::Reinforce, 0.5, "Reinforce active paths"),
        (ActivationOperation::Inhibit, 0.4, "Apply inhibition"),
        (ActivationOperation::Decay, 0.2, "Temporal decay"),
    ];
    
    for (i, (operation, activation, concept)) in steps_data.iter().enumerate() {
        let step = ActivationStep {
            step_id: i,
            entity_key: EntityKey::default(),
            concept_id: concept.to_string(),
            activation_level: *activation,
            operation_type: *operation,
            timestamp: base_time + Duration::from_millis(i as u64 * 100),
        };
        trace.push(step);
    }
    
    assert_eq!(trace.len(), 5);
    
    // Verify sequence properties
    for (i, step) in trace.iter().enumerate() {
        assert_eq!(step.step_id, i);
        if i > 0 {
            assert!(step.timestamp >= trace[i-1].timestamp);
        }
    }
    
    // Verify specific operations
    assert_eq!(trace[0].operation_type, ActivationOperation::Initialize);
    assert_eq!(trace[4].operation_type, ActivationOperation::Decay);
    assert_float_eq(trace[2].activation_level, 0.5, 0.001);
}

#[test]
fn test_activation_pattern_query_variations() {
    let test_queries = vec![
        "",
        "simple query",
        "complex query with symbols !@#$%",
        "🧠 unicode query 🔥",
        &"very long query ".repeat(100),
        "query\nwith\nnewlines",
        "query\twith\ttabs",
    ];
    
    for query in test_queries {
        let pattern = ActivationPattern::new(query.clone());
        assert_eq!(pattern.query, query);
        assert!(pattern.activations.is_empty());
    }
}

#[test]
fn test_activation_pattern_timestamp_consistency() {
    let start_time = SystemTime::now();
    let pattern = ActivationPattern::new("timestamp_test".to_string());
    let end_time = SystemTime::now();
    
    // Pattern timestamp should be between start and end
    assert!(pattern.timestamp >= start_time);
    assert!(pattern.timestamp <= end_time);
}

#[test]
fn test_activation_pattern_modification() {
    let mut pattern = ActivationPattern::new("modification_test".to_string());
    
    let key = EntityKey::default();
    
    // Add activation
    pattern.activations.insert(key, 0.5);
    assert_float_eq(*pattern.activations.get(&key).unwrap(), 0.5, 0.001);
    
    // Modify activation
    pattern.activations.insert(key, 0.8);
    assert_float_eq(*pattern.activations.get(&key).unwrap(), 0.8, 0.001);
    
    // Remove activation
    pattern.activations.remove(&key);
    assert!(pattern.activations.get(&key).is_none());
}

#[test]
fn test_activation_step_boundary_values() {
    let boundary_activations = vec![0.0, 0.001, 0.5, 0.999, 1.0];
    
    for (i, &activation) in boundary_activations.iter().enumerate() {
        let step = ActivationStep {
            step_id: i,
            entity_key: EntityKey::default(),
            concept_id: format!("boundary_{}", i),
            activation_level: activation,
            operation_type: ActivationOperation::Propagate,
            timestamp: SystemTime::now(),
        };
        
        assert_float_eq(step.activation_level, activation, 0.001);
        assert_valid_activation(step.activation_level);
    }
}

#[test]
fn test_activation_pattern_performance_characteristics() {
    let mut pattern = ActivationPattern::new("performance_test".to_string());
    
    // Add many activations quickly
    let num_activations = 1000;
    let start_time = SystemTime::now();
    
    for i in 0..num_activations {
        let key = EntityKey::default(); // Note: all same key in this test
        pattern.activations.insert(key, i as f32 / num_activations as f32);
    }
    
    let insertion_time = start_time.elapsed().unwrap();
    
    // Test top activations performance
    let top_start = SystemTime::now();
    let _top_100 = pattern.get_top_activations(100);
    let top_time = top_start.elapsed().unwrap();
    
    // These should be reasonably fast (exact timing depends on system)
    assert!(insertion_time < Duration::from_millis(100));
    assert!(top_time < Duration::from_millis(50));
}

#[test]
fn test_activation_step_concept_id_variations() {
    let concept_ids = vec![
        "simple",
        "concept_with_underscores",
        "concept-with-dashes",
        "concept.with.dots",
        "concept with spaces",
        "123numeric_concept",
        "🧠neural_concept",
        "", // Empty concept ID
    ];
    
    for (i, concept_id) in concept_ids.iter().enumerate() {
        let step = ActivationStep {
            step_id: i,
            entity_key: EntityKey::default(),
            concept_id: concept_id.to_string(),
            activation_level: 0.5,
            operation_type: ActivationOperation::Initialize,
            timestamp: SystemTime::now(),
        };
        
        assert_eq!(step.concept_id, *concept_id);
    }
}

#[test]
fn test_activation_pattern_sorting_stability() {
    let mut pattern = ActivationPattern::new("stability_test".to_string());
    
    // Add activations with some identical values
    let keys: Vec<EntityKey> = (0..6).map(|_| EntityKey::default()).collect();
    let activations = vec![0.5, 0.8, 0.5, 0.9, 0.5, 0.8]; // Multiple duplicates
    
    for (i, &activation) in activations.iter().enumerate() {
        pattern.activations.insert(keys[i], activation);
    }
    
    // Get top activations multiple times - should be consistent
    let top1 = pattern.get_top_activations(6);
    let top2 = pattern.get_top_activations(6);
    
    assert_eq!(top1.len(), top2.len());
    for i in 0..top1.len() {
        assert_float_eq(top1[i].1, top2[i].1, 0.001);
    }
    
    // Verify descending order
    for i in 0..top1.len()-1 {
        assert!(top1[i].1 >= top1[i+1].1);
    }
}

#[test]
fn test_activation_trace_temporal_ordering() {
    let base_time = SystemTime::now();
    let mut steps = Vec::new();
    
    // Create steps with specific time intervals
    let time_offsets = vec![0, 50, 100, 150, 200]; // milliseconds
    
    for (i, &offset) in time_offsets.iter().enumerate() {
        let step = ActivationStep {
            step_id: i,
            entity_key: EntityKey::default(),
            concept_id: format!("step_{}", i),
            activation_level: 0.5,
            operation_type: ActivationOperation::Propagate,
            timestamp: base_time + Duration::from_millis(offset),
        };
        steps.push(step);
    }
    
    // Verify temporal ordering
    for i in 1..steps.len() {
        assert!(steps[i].timestamp > steps[i-1].timestamp);
        let duration = steps[i].timestamp.duration_since(steps[i-1].timestamp).unwrap();
        assert!(duration >= Duration::from_millis(40)); // Should be ~50ms apart
    }
}

#[test]
fn test_activation_pattern_integration_with_entities() {
    // Test pattern usage with actual entities
    let mut pattern = ActivationPattern::new("integration_test".to_string());
    
    let entity1 = BrainInspiredEntity::new("concept1".to_string(), EntityDirection::Input);
    let entity2 = BrainInspiredEntity::new("concept2".to_string(), EntityDirection::Output);
    let entity3 = BrainInspiredEntity::new("concept3".to_string(), EntityDirection::Hidden);
    
    // Use entity IDs in pattern
    pattern.activations.insert(entity1.id, 0.7);
    pattern.activations.insert(entity2.id, 0.4);
    pattern.activations.insert(entity3.id, 0.9);
    
    let top_activations = pattern.get_top_activations(2);
    assert_eq!(top_activations.len(), 2);
    
    // Highest should be entity3 (0.9), then entity1 (0.7)
    assert_float_eq(top_activations[0].1, 0.9, 0.001);
    assert_float_eq(top_activations[1].1, 0.7, 0.001);
}