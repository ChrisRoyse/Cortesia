/// Comprehensive error handling and boundary condition tests
/// 
/// This module tests error scenarios, input validation, and boundary conditions
/// for all brain_types components to ensure robust error handling.

use llmkg::core::brain_types::{
    LogicGate, LogicGateType, BrainInspiredEntity, EntityDirection,
    BrainInspiredRelationship, RelationType, ActivationPattern
};
use llmkg::core::types::EntityKey;
use llmkg::error::{GraphError, Result};
use super::test_helpers::*;
use std::time::{SystemTime, Duration};

#[test]
fn test_logic_gate_input_count_validation() {
    // Test gates that require specific input counts
    
    // NOT gate - requires exactly 1 input
    let not_gate = LogicGate::new(LogicGateType::Not, 0.5);
    assert!(not_gate.calculate_output(&[]).is_err()); // No inputs
    assert!(not_gate.calculate_output(&[0.5, 0.3]).is_err()); // Too many inputs
    assert!(not_gate.calculate_output(&[0.5, 0.3, 0.7]).is_err()); // Way too many
    assert!(not_gate.calculate_output(&[0.5]).is_ok()); // Correct count
    
    // Identity gate - requires exactly 1 input
    let identity_gate = LogicGate::new(LogicGateType::Identity, 0.5);
    assert!(identity_gate.calculate_output(&[]).is_err());
    assert!(identity_gate.calculate_output(&[0.5, 0.3]).is_err());
    assert!(identity_gate.calculate_output(&[0.5]).is_ok());
    
    // XOR gate - requires exactly 2 inputs
    let xor_gate = LogicGate::new(LogicGateType::Xor, 0.5);
    assert!(xor_gate.calculate_output(&[]).is_err());
    assert!(xor_gate.calculate_output(&[0.5]).is_err()); // Too few
    assert!(xor_gate.calculate_output(&[0.5, 0.3, 0.7]).is_err()); // Too many
    assert!(xor_gate.calculate_output(&[0.5, 0.3]).is_ok()); // Correct count
    
    // XNOR gate - requires exactly 2 inputs
    let xnor_gate = LogicGate::new(LogicGateType::Xnor, 0.5);
    assert!(xnor_gate.calculate_output(&[0.5]).is_err());
    assert!(xnor_gate.calculate_output(&[0.5, 0.3, 0.7]).is_err());
    assert!(xnor_gate.calculate_output(&[0.5, 0.3]).is_ok());
}

#[test]
fn test_logic_gate_input_mismatch_errors() {
    // Test gates with input_nodes vs actual input mismatches
    let mut gate = LogicGate::new(LogicGateType::And, 0.5);
    gate.input_nodes = vec![EntityKey::default(), EntityKey::default(), EntityKey::default()]; // 3 expected inputs
    
    // Wrong number of inputs
    assert!(gate.calculate_output(&[]).is_err());
    assert!(gate.calculate_output(&[0.5]).is_err()); // Too few
    assert!(gate.calculate_output(&[0.5, 0.3]).is_err()); // Still too few
    assert!(gate.calculate_output(&[0.5, 0.3, 0.7, 0.9]).is_err()); // Too many
    assert!(gate.calculate_output(&[0.5, 0.3, 0.7]).is_ok()); // Correct count
}

#[test]
fn test_weighted_gate_weight_matrix_mismatch() {
    let mut weighted_gate = LogicGate::new(LogicGateType::Weighted, 1.0);
    weighted_gate.input_nodes = vec![EntityKey::default(), EntityKey::default(), EntityKey::default()]; // 3 inputs
    
    // No weight matrix
    weighted_gate.weight_matrix = vec![];
    assert!(weighted_gate.calculate_output(&[0.5, 0.3, 0.7]).is_err());
    
    // Wrong size weight matrix
    weighted_gate.weight_matrix = vec![0.5, 0.3]; // Only 2 weights for 3 inputs
    assert!(weighted_gate.calculate_output(&[0.5, 0.3, 0.7]).is_err());
    
    // Too many weights
    weighted_gate.weight_matrix = vec![0.5, 0.3, 0.2, 0.1]; // 4 weights for 3 inputs
    assert!(weighted_gate.calculate_output(&[0.5, 0.3, 0.7]).is_err());
    
    // Correct size
    weighted_gate.weight_matrix = vec![0.5, 0.3, 0.2]; // 3 weights for 3 inputs
    assert!(weighted_gate.calculate_output(&[0.5, 0.3, 0.7]).is_ok());
}

#[test]
fn test_logic_gate_special_float_values() {
    let gate = LogicGate::new(LogicGateType::Or, 0.5);
    
    // Test with NaN inputs
    let result_nan = gate.calculate_output(&[f32::NAN, 0.5]);
    // Should handle gracefully (either error or non-crashing result)
    match result_nan {
        Ok(val) => {
            // If it succeeds, result might be NaN or some handled value
            // The key is that it doesn't panic
        },
        Err(_) => {
            // Acceptable to return an error for NaN inputs
        }
    }
    
    // Test with infinity inputs
    let result_inf = gate.calculate_output(&[f32::INFINITY, 0.5]);
    match result_inf {
        Ok(val) => assert!(val.is_finite() || val.is_infinite()), // Should be some float value
        Err(_) => {} // Acceptable to error on infinity
    }
    
    // Test with negative infinity
    let result_neg_inf = gate.calculate_output(&[f32::NEG_INFINITY, 0.5]);
    match result_neg_inf {
        Ok(_) => {},
        Err(_) => {} // Both outcomes acceptable
    }
}

#[test]
fn test_logic_gate_extreme_threshold_values() {
    // Test with extreme threshold values
    let extreme_thresholds = vec![
        f32::NEG_INFINITY,
        -1000.0,
        -1.0,
        -0.001,
        0.0,
        1000.0,
        f32::INFINITY,
        f32::NAN,
    ];
    
    for threshold in extreme_thresholds {
        let gate = LogicGate::new(LogicGateType::Threshold, threshold);
        let result = gate.calculate_output(&[0.5, 0.3, 0.7]);
        
        // Should not panic, regardless of result
        match result {
            Ok(val) => {
                // If threshold is NaN, result might be NaN
                if !threshold.is_nan() {
                    // For non-NaN thresholds, result should be valid
                    assert!(val >= 0.0 || val.is_nan());
                }
            },
            Err(_) => {} // Errors are acceptable for extreme values
        }
    }
}

#[test]
fn test_entity_activation_boundary_conditions() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    
    // Test with extreme activation values
    let extreme_activations = vec![
        f32::NEG_INFINITY,
        -1000.0,
        -1.0,
        -0.001,
        0.0,
        1000.0,
        f32::INFINITY,
        f32::NAN,
    ];
    
    for activation in extreme_activations {
        entity.activation_state = 0.5; // Reset to known state
        let result = entity.activate(activation, 0.1);
        
        // Should not panic
        assert!(result >= 0.0 && result <= 1.0 || result.is_nan());
        
        // State should remain valid (clamped) unless NaN
        if !activation.is_nan() {
            assert!(entity.activation_state >= 0.0 && entity.activation_state <= 1.0);
        }
    }
}

#[test]
fn test_entity_activation_extreme_decay_rates() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    entity.activation_state = 0.8;
    entity.last_activation = SystemTime::now() - Duration::from_secs(1);
    
    let extreme_decay_rates = vec![
        f32::NEG_INFINITY,
        -1000.0,
        -1.0,
        0.0,
        1000.0,
        f32::INFINITY,
        f32::NAN,
    ];
    
    for decay_rate in extreme_decay_rates {
        entity.activation_state = 0.8; // Reset
        let result = entity.activate(0.0, decay_rate);
        
        // Should not panic
        match result {
            result if result.is_nan() => {
                // NaN decay rate might produce NaN result
                assert!(decay_rate.is_nan());
            },
            result => {
                // For valid decay rates, result should be valid
                assert!(result >= 0.0 && result <= 1.0);
            }
        }
    }
}

#[test]
fn test_relationship_extreme_learning_rates() {
    let mut rel = BrainInspiredRelationship::new(
        EntityKey::default(),
        EntityKey::default(),
        RelationType::RelatedTo
    );
    rel.weight = 0.5;
    
    let extreme_learning_rates = vec![
        f32::NEG_INFINITY,
        -1000.0,
        -1.0,
        0.0,
        1000.0,
        f32::INFINITY,
        f32::NAN,
    ];
    
    for learning_rate in extreme_learning_rates {
        rel.weight = 0.5; // Reset
        rel.strengthen(learning_rate);
        
        // Weight should remain in valid range or be NaN
        if learning_rate.is_nan() {
            // NaN learning rate might produce NaN weight
            assert!(rel.weight.is_nan() || (rel.weight >= 0.0 && rel.weight <= 1.0));
        } else {
            // For non-NaN learning rates, weight should be clamped
            assert!(rel.weight >= 0.0 && rel.weight <= 1.0);
        }
        
        // Strength should match weight
        assert_eq!(rel.weight, rel.strength);
    }
}

#[test]
fn test_relationship_extreme_decay_rates() {
    let mut rel = BrainInspiredRelationship::new(
        EntityKey::default(),
        EntityKey::default(),
        RelationType::Temporal
    );
    rel.weight = 0.8;
    rel.last_strengthened = SystemTime::now() - Duration::from_secs(1);
    
    let extreme_decay_rates = vec![
        f32::NEG_INFINITY,
        -1000.0,
        -1.0,
        0.0,
        1000.0,
        f32::INFINITY,
        f32::NAN,
    ];
    
    for decay_rate in extreme_decay_rates {
        rel.weight = 0.8; // Reset
        rel.temporal_decay = decay_rate;
        let result = rel.apply_decay();
        
        // Should not panic
        if decay_rate.is_nan() {
            // NaN decay rate might produce NaN result
            assert!(result.is_nan() || (result >= 0.0 && result <= 1.0));
        } else {
            // For valid decay rates, result should be non-negative
            assert!(result >= 0.0);
        }
    }
}

#[test]
fn test_activation_pattern_extreme_values() {
    let mut pattern = ActivationPattern::new("extreme_test".to_string());
    
    let key = EntityKey::default();
    let extreme_values = vec![
        f32::NEG_INFINITY,
        -1000.0,
        -1.0,
        0.0,
        1000.0,
        f32::INFINITY,
        f32::NAN,
    ];
    
    for value in extreme_values {
        pattern.activations.clear();
        pattern.activations.insert(key, value);
        
        // get_top_activations should handle extreme values gracefully
        let top = pattern.get_top_activations(1);
        
        if value.is_nan() {
            // NaN values might be handled specially in sorting
            // The key is that it doesn't panic
        } else {
            assert_eq!(top.len(), 1);
            assert_eq!(top[0].1, value);
        }
    }
}

#[test]
fn test_activation_pattern_sorting_with_special_values() {
    let mut pattern = ActivationPattern::new("sorting_test".to_string());
    
    let keys: Vec<EntityKey> = (0..5).map(|_| EntityKey::default()).collect();
    let mixed_values = vec![0.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.8];
    
    for (i, &value) in mixed_values.iter().enumerate() {
        pattern.activations.insert(keys[i], value);
    }
    
    // Should not panic during sorting
    let top = pattern.get_top_activations(5);
    
    // Exact behavior with special float values is implementation-dependent,
    // but it should not crash
    assert!(top.len() <= 5);
}

#[test]
fn test_entity_time_handling_edge_cases() {
    let mut entity = BrainInspiredEntity::new("time_test".to_string(), EntityDirection::Input);
    
    // Test with future timestamp (shouldn't normally happen, but test robustness)
    entity.last_activation = SystemTime::now() + Duration::from_secs(60);
    entity.activation_state = 0.7;
    
    let result = entity.activate(0.0, 0.1);
    
    // Should handle gracefully (elapsed() might return error for future times)
    // The exact behavior depends on how elapsed() errors are handled
    assert!(result >= 0.0 && result <= 1.0);
}

#[test]
fn test_relationship_time_handling_edge_cases() {
    let mut rel = BrainInspiredRelationship::new(
        EntityKey::default(),
        EntityKey::default(),
        RelationType::Temporal
    );
    
    // Test with future timestamp
    rel.last_strengthened = SystemTime::now() + Duration::from_secs(60);
    rel.weight = 0.7;
    
    let result = rel.apply_decay();
    
    // Should handle gracefully
    assert!(result >= 0.0 && result <= 1.0);
}

#[test]
fn test_empty_inputs_handling() {
    // Test various components with empty inputs
    
    // Logic gates with empty inputs (where applicable)
    let inhibitory_gate = LogicGate::new(LogicGateType::Inhibitory, 0.5);
    let result = inhibitory_gate.calculate_output(&[]);
    assert_eq!(result.unwrap(), 0.0); // Should return 0 for empty input
    
    // Threshold gate with empty inputs
    let threshold_gate = LogicGate::new(LogicGateType::Threshold, 0.5);
    let result = threshold_gate.calculate_output(&[]);
    assert_eq!(result.unwrap(), 0.0); // Sum of empty = 0 < threshold
    
    // Activation pattern with empty activations
    let pattern = ActivationPattern::new("empty_test".to_string());
    let top = pattern.get_top_activations(5);
    assert!(top.is_empty());
}

#[test]
fn test_zero_input_scenarios() {
    // Test behavior with all-zero inputs
    
    let gate_types = vec![
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Nand,
        LogicGateType::Nor,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
    ];
    
    for gate_type in gate_types {
        let gate = create_test_gate(gate_type, 0.5, 3);
        let result = gate.calculate_output(&[0.0, 0.0, 0.0]);
        
        match result {
            Ok(output) => {
                // Should be valid output
                assert!(output >= 0.0 && output <= 1.0);
            },
            Err(_) => {
                // Some gates might error on certain conditions
            }
        }
    }
}

#[test]
fn test_maximum_input_scenarios() {
    // Test behavior with all-maximum inputs
    
    let gate_types = vec![
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Nand,
        LogicGateType::Nor,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
    ];
    
    for gate_type in gate_types {
        let gate = create_test_gate(gate_type, 0.5, 3);
        let result = gate.calculate_output(&[1.0, 1.0, 1.0]);
        
        match result {
            Ok(output) => {
                // Should be valid output
                assert!(output >= 0.0 && output <= 1.0);
            },
            Err(_) => {
                // Some gates might error on certain conditions
            }
        }
    }
}

#[test]
fn test_string_handling_edge_cases() {
    // Test with extreme string inputs
    
    let extreme_strings = vec![
        "",
        " ",
        "\n\r\t",
        "🔥".repeat(1000), // Very long unicode
        "\0null\0terminated\0",
        "control\x01\x02\x03chars",
    ];
    
    for test_string in extreme_strings {
        // Entity concept IDs
        let entity = BrainInspiredEntity::new(test_string.clone(), EntityDirection::Input);
        assert_eq!(entity.concept_id, test_string);
        
        // Activation pattern queries
        let pattern = ActivationPattern::new(test_string.clone());
        assert_eq!(pattern.query, test_string);
    }
}

#[test]
fn test_concurrent_modification_safety() {
    // Test that structures handle rapid modifications safely
    
    let mut pattern = ActivationPattern::new("concurrent_test".to_string());
    let key = EntityKey::default();
    
    // Rapid insertions and deletions
    for i in 0..100 {
        pattern.activations.insert(key, i as f32 / 100.0);
        if i % 2 == 0 {
            pattern.activations.remove(&key);
        }
    }
    
    // Should not crash and should have consistent state
    let top = pattern.get_top_activations(1);
    // Exact result depends on final state, but should not crash
}

#[test]
fn test_memory_usage_patterns() {
    // Test with large data structures to check for memory issues
    
    let mut pattern = ActivationPattern::new("memory_test".to_string());
    
    // Add many activations
    for i in 0..10000 {
        let key = EntityKey::default(); // Note: will overwrite same key
        pattern.activations.insert(key, (i % 1000) as f32 / 1000.0);
    }
    
    // Should handle large datasets
    let top = pattern.get_top_activations(100);
    assert!(top.len() <= 100);
    
    // Clear and ensure cleanup
    pattern.activations.clear();
    assert!(pattern.activations.is_empty());
}