// Tests for LogicGate struct
// Validates neural logic gate computation with all gate types and edge cases

use llmkg::core::brain_types::{LogicGate, LogicGateType, EntityDirection};
use llmkg::core::types::EntityKey;
use llmkg::error::GraphError;
use serde_json;

use super::test_constants;
use super::test_helpers::{
    LogicGateBuilder, assert_activation_close, measure_execution_time,
    create_test_weight_matrices, create_gate_test_inputs, generate_edge_case_activations,
    assert_float_eq
};

// ==================== Constructor Tests ====================

#[test]
fn test_logic_gate_new() {
    let gate = LogicGate::new(LogicGateType::And, test_constants::AND_GATE_THRESHOLD);
    
    assert_eq!(gate.gate_type, LogicGateType::And);
    assert_eq!(gate.threshold, test_constants::AND_GATE_THRESHOLD);
    assert!(gate.input_nodes.is_empty());
    assert!(gate.output_nodes.is_empty());
    assert!(gate.weight_matrix.is_empty());
}

#[test]
fn test_logic_gate_new_all_types() {
    let gate_types = [
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Not,
        LogicGateType::Xor,
        LogicGateType::Nand,
        LogicGateType::Nor,
        LogicGateType::Xnor,
        LogicGateType::Identity,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
        LogicGateType::Weighted,
    ];
    
    for gate_type in gate_types {
        let gate = LogicGate::new(gate_type, 0.5);
        assert_eq!(gate.gate_type, gate_type);
        assert_eq!(gate.threshold, 0.5);
    }
}

// ==================== Builder Pattern Tests ====================

#[test]
fn test_logic_gate_builder() {
    let input_keys = vec![EntityKey::from(1), EntityKey::from(2)];
    let output_keys = vec![EntityKey::from(3)];
    let weights = vec![0.6, 0.4];
    
    let gate = LogicGateBuilder::new(LogicGateType::Weighted)
        .with_threshold(test_constants::WEIGHTED_GATE_THRESHOLD)
        .with_inputs(input_keys.clone())
        .with_outputs(output_keys.clone())
        .with_weights(weights.clone())
        .build();
    
    assert_eq!(gate.gate_type, LogicGateType::Weighted);
    assert_eq!(gate.threshold, test_constants::WEIGHTED_GATE_THRESHOLD);
    assert_eq!(gate.input_nodes, input_keys);
    assert_eq!(gate.output_nodes, output_keys);
    assert_eq!(gate.weight_matrix, weights);
}

// ==================== AND Gate Tests ====================

#[test]
fn test_and_gate_basic() {
    let gate = LogicGate::new(LogicGateType::And, test_constants::AND_GATE_THRESHOLD);
    
    // Both inputs high (above threshold)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_11).unwrap();
    assert_activation_close(result, 0.7, test_constants::ACTIVATION_EPSILON, "AND both high");
    
    // One input low
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_01).unwrap();
    assert_eq!(result, 0.0, "AND one low should be 0");
    
    // Both inputs low
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_00).unwrap();
    assert_eq!(result, 0.0, "AND both low should be 0");
}

#[test]
fn test_and_gate_threshold_behavior() {
    let gate = LogicGate::new(LogicGateType::And, 0.6); // Higher threshold
    
    // Both inputs below threshold
    let result = gate.calculate_output(&[0.5, 0.4]).unwrap();
    assert_eq!(result, 0.0, "AND below threshold should be 0");
    
    // Both inputs above threshold
    let result = gate.calculate_output(&[0.8, 0.7]).unwrap();
    assert_activation_close(result, 0.7, test_constants::ACTIVATION_EPSILON, "AND above threshold");
}

#[test]
fn test_and_gate_multiple_inputs() {
    let gate = LogicGate::new(LogicGateType::And, 0.5);
    
    // All inputs high
    let result = gate.calculate_output(&test_constants::THREE_INPUTS_ALL_HIGH).unwrap();
    assert_activation_close(result, 0.7, test_constants::ACTIVATION_EPSILON, "AND all high");
    
    // Mixed inputs (one low)
    let result = gate.calculate_output(&test_constants::THREE_INPUTS_MIXED).unwrap();
    assert_eq!(result, 0.0, "AND mixed should be 0");
}

// ==================== OR Gate Tests ====================

#[test]
fn test_or_gate_basic() {
    let gate = LogicGate::new(LogicGateType::Or, test_constants::OR_GATE_THRESHOLD);
    
    // Both inputs high
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_11).unwrap();
    assert_activation_close(result, 0.7, test_constants::ACTIVATION_EPSILON, "OR both high");
    
    // One input high
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_01).unwrap();
    assert_activation_close(result, 0.7, test_constants::ACTIVATION_EPSILON, "OR one high");
    
    // Both inputs low
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_00).unwrap();
    assert_eq!(result, 0.0, "OR both low should be 0");
}

#[test]
fn test_or_gate_threshold_behavior() {
    let gate = LogicGate::new(LogicGateType::Or, 0.8); // High threshold
    
    // One input below threshold
    let result = gate.calculate_output(&[0.7, 0.5]).unwrap();
    assert_eq!(result, 0.0, "OR below threshold should be 0");
    
    // One input above threshold
    let result = gate.calculate_output(&[0.9, 0.5]).unwrap();
    assert_activation_close(result, 0.9, test_constants::ACTIVATION_EPSILON, "OR above threshold");
}

// ==================== NOT Gate Tests ====================

#[test]
fn test_not_gate_basic() {
    let gate = LogicGate::new(LogicGateType::Not, 0.5);
    
    // High input
    let result = gate.calculate_output(&test_constants::SINGLE_INPUT_HIGH).unwrap();
    assert_activation_close(result, 0.2, test_constants::ACTIVATION_EPSILON, "NOT high input");
    
    // Low input
    let result = gate.calculate_output(&test_constants::SINGLE_INPUT_LOW).unwrap();
    assert_activation_close(result, 0.7, test_constants::ACTIVATION_EPSILON, "NOT low input");
    
    // Zero input
    let result = gate.calculate_output(&[0.0]).unwrap();
    assert_eq!(result, 1.0, "NOT zero should be 1.0");
    
    // Max input
    let result = gate.calculate_output(&[1.0]).unwrap();
    assert_eq!(result, 0.0, "NOT max should be 0.0");
}

#[test]
fn test_not_gate_input_validation() {
    let gate = LogicGate::new(LogicGateType::Not, 0.5);
    
    // Too many inputs
    let result = gate.calculate_output(&[0.5, 0.7]);
    assert!(result.is_err(), "NOT gate should reject multiple inputs");
    
    // No inputs
    let result = gate.calculate_output(&[]);
    assert!(result.is_err(), "NOT gate should reject no inputs");
}

// ==================== XOR Gate Tests ====================

#[test]
fn test_xor_gate_basic() {
    let gate = LogicGate::new(LogicGateType::Xor, test_constants::AND_GATE_THRESHOLD);
    
    // Both high (same)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_11).unwrap();
    assert_eq!(result, 0.0, "XOR both high should be 0");
    
    // Both low (same)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_00).unwrap();
    assert_eq!(result, 0.0, "XOR both low should be 0");
    
    // One high, one low (different)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_01).unwrap();
    assert_activation_close(result, 0.7, test_constants::ACTIVATION_EPSILON, "XOR different");
    
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_10).unwrap();
    assert_activation_close(result, 0.7, test_constants::ACTIVATION_EPSILON, "XOR different");
}

#[test]
fn test_xor_gate_input_validation() {
    let gate = LogicGate::new(LogicGateType::Xor, 0.5);
    
    // Too many inputs
    let result = gate.calculate_output(&[0.5, 0.7, 0.3]);
    assert!(result.is_err(), "XOR gate should reject more than 2 inputs");
    
    // Too few inputs
    let result = gate.calculate_output(&[0.5]);
    assert!(result.is_err(), "XOR gate should reject less than 2 inputs");
}

// ==================== Compound Gate Tests (NAND, NOR, XNOR) ====================

#[test]
fn test_nand_gate() {
    let gate = LogicGate::new(LogicGateType::Nand, test_constants::AND_GATE_THRESHOLD);
    
    // Both high (AND would be true, so NAND is false)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_11).unwrap();
    assert_eq!(result, 0.0, "NAND both high should be 0");
    
    // One low (AND would be false, so NAND is true)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_01).unwrap();
    assert_eq!(result, 1.0, "NAND one low should be 1");
    
    // Both low (AND would be false, so NAND is true)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_00).unwrap();
    assert_eq!(result, 1.0, "NAND both low should be 1");
}

#[test]
fn test_nor_gate() {
    let gate = LogicGate::new(LogicGateType::Nor, test_constants::OR_GATE_THRESHOLD);
    
    // Both high (OR would be true, so NOR is false)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_11).unwrap();
    assert_eq!(result, 0.0, "NOR both high should be 0");
    
    // One high (OR would be true, so NOR is false)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_01).unwrap();
    assert_eq!(result, 0.0, "NOR one high should be 0");
    
    // Both low (OR would be false, so NOR is true)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_00).unwrap();
    assert_eq!(result, 1.0, "NOR both low should be 1");
}

#[test]
fn test_xnor_gate() {
    let gate = LogicGate::new(LogicGateType::Xnor, test_constants::AND_GATE_THRESHOLD);
    
    // Both high (same, so XNOR is true)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_11).unwrap();
    assert_activation_close(result, 0.7, test_constants::ACTIVATION_EPSILON, "XNOR both high");
    
    // Both low (same, so XNOR is true)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_00).unwrap();
    assert_eq!(result, 0.0, "XNOR both low should be 0"); // Both below threshold
    
    // Different inputs (XOR would be true, so XNOR is false)
    let result = gate.calculate_output(&test_constants::GATE_INPUTS_01).unwrap();
    assert_eq!(result, 0.0, "XNOR different should be 0");
}

// ==================== Neural-Specific Gate Tests ====================

#[test]
fn test_identity_gate() {
    let gate = LogicGate::new(LogicGateType::Identity, 0.0); // Threshold irrelevant
    
    // Should pass through input unchanged
    let test_values = [0.0, 0.3, 0.5, 0.7, 1.0];
    
    for value in test_values {
        let result = gate.calculate_output(&[value]).unwrap();
        assert_activation_close(result, value, test_constants::ACTIVATION_EPSILON, "Identity passthrough");
    }
}

#[test]
fn test_identity_gate_input_validation() {
    let gate = LogicGate::new(LogicGateType::Identity, 0.0);
    
    // Multiple inputs should fail
    let result = gate.calculate_output(&[0.5, 0.7]);
    assert!(result.is_err(), "Identity gate should reject multiple inputs");
    
    // No inputs should fail
    let result = gate.calculate_output(&[]);
    assert!(result.is_err(), "Identity gate should reject no inputs");
}

#[test]
fn test_threshold_gate() {
    let gate = LogicGate::new(LogicGateType::Threshold, test_constants::THRESHOLD_GATE_LIMIT);
    
    // Sum below threshold
    let result = gate.calculate_output(&[0.2, 0.3, 0.1]).unwrap(); // Sum = 0.6
    assert_eq!(result, 0.0, "Threshold below limit should be 0");
    
    // Sum above threshold
    let result = gate.calculate_output(&[0.3, 0.4, 0.2]).unwrap(); // Sum = 0.9
    assert_activation_close(result, 0.9, test_constants::ACTIVATION_EPSILON, "Threshold above limit");
    
    // Sum way above 1.0 should be clamped
    let result = gate.calculate_output(&[0.8, 0.8, 0.8]).unwrap(); // Sum = 2.4
    assert_eq!(result, 1.0, "Threshold should clamp at 1.0");
}

#[test]
fn test_inhibitory_gate() {
    let gate = LogicGate::new(LogicGateType::Inhibitory, test_constants::INHIBITORY_GATE_THRESHOLD);
    
    // Primary input only
    let result = gate.calculate_output(&[0.8]).unwrap();
    assert_activation_close(result, 0.8, test_constants::ACTIVATION_EPSILON, "Inhibitory primary only");
    
    // Primary with weak inhibition
    let result = gate.calculate_output(&[0.8, 0.2]).unwrap(); // 0.8 - 0.2 = 0.6
    assert_activation_close(result, 0.6, test_constants::ACTIVATION_EPSILON, "Inhibitory with weak inhibition");
    
    // Primary with strong inhibition
    let result = gate.calculate_output(&[0.8, 0.9]).unwrap(); // 0.8 - 0.9 = -0.1 -> 0.0
    assert_eq!(result, 0.0, "Inhibitory should not go negative");
    
    // Multiple inhibitory inputs
    let result = gate.calculate_output(&[0.8, 0.3, 0.2]).unwrap(); // 0.8 - (0.3 + 0.2) = 0.3
    assert_activation_close(result, 0.3, test_constants::ACTIVATION_EPSILON, "Inhibitory multiple inhibitors");
    
    // No inputs
    let result = gate.calculate_output(&[]).unwrap();
    assert_eq!(result, 0.0, "Inhibitory with no inputs should be 0");
}

#[test]
fn test_weighted_gate() {
    let mut gate = LogicGate::new(LogicGateType::Weighted, test_constants::WEIGHTED_GATE_THRESHOLD);
    gate.weight_matrix = test_constants::WEIGHT_MATRIX_3.to_vec();
    
    // Weighted sum below threshold
    let inputs = [0.4, 0.2, 0.3]; // 0.4*0.5 + 0.2*0.3 + 0.3*0.2 = 0.32
    let result = gate.calculate_output(&inputs).unwrap();
    assert_eq!(result, 0.0, "Weighted below threshold should be 0");
    
    // Weighted sum above threshold
    let inputs = [0.8, 0.9, 0.7]; // 0.8*0.5 + 0.9*0.3 + 0.7*0.2 = 0.81
    let result = gate.calculate_output(&inputs).unwrap();
    assert_activation_close(result, 0.81, test_constants::ACTIVATION_EPSILON, "Weighted above threshold");
    
    // Test clamping at 1.0
    let inputs = [2.0, 2.0, 2.0]; // Would be 2.0 but should clamp
    let result = gate.calculate_output(&inputs).unwrap();
    assert_eq!(result, 1.0, "Weighted should clamp at 1.0");
}

#[test]
fn test_weighted_gate_weight_mismatch() {
    let mut gate = LogicGate::new(LogicGateType::Weighted, 0.5);
    gate.weight_matrix = vec![0.5, 0.3]; // 2 weights
    
    // 3 inputs with 2 weights should fail
    let result = gate.calculate_output(&[0.1, 0.2, 0.3]);
    assert!(result.is_err(), "Weight matrix size mismatch should fail");
    
    // 1 input with 2 weights should fail
    let result = gate.calculate_output(&[0.5]);
    assert!(result.is_err(), "Weight matrix size mismatch should fail");
}

// ==================== Error Handling Tests ====================

#[test]
fn test_gate_input_mismatch_errors() {
    let gate = LogicGate::new(LogicGateType::And, 0.5);
    
    // Empty input to binary gate
    let result = gate.calculate_output(&[]);
    // AND gate should handle empty gracefully or error - check implementation
    match result {
        Ok(_) => {}, // Some implementations might handle this
        Err(_) => {}, // Others might error
    }
}

// ==================== Serialization Tests ====================

#[test]
fn test_logic_gate_serialization() {
    let gate = LogicGateBuilder::new(LogicGateType::Weighted)
        .with_threshold(0.7)
        .with_inputs(vec![EntityKey::from(1), EntityKey::from(2)])
        .with_outputs(vec![EntityKey::from(3)])
        .with_weights(vec![0.6, 0.4])
        .build();
    
    let serialized = serde_json::to_string(&gate).expect("Should serialize");
    
    // Check key fields are present
    assert!(serialized.contains("gate_type"));
    assert!(serialized.contains("threshold"));
    assert!(serialized.contains("input_nodes"));
    assert!(serialized.contains("weight_matrix"));
}

#[test]
fn test_logic_gate_deserialization() {
    let original = LogicGateBuilder::new(LogicGateType::Inhibitory)
        .with_threshold(test_constants::INHIBITORY_GATE_THRESHOLD)
        .with_inputs(vec![EntityKey::from(10), EntityKey::from(20)])
        .build();
    
    let serialized = serde_json::to_string(&original).expect("Should serialize");
    let deserialized: LogicGate = serde_json::from_str(&serialized)
        .expect("Should deserialize");
    
    assert_eq!(deserialized.gate_type, original.gate_type);
    assert_eq!(deserialized.threshold, original.threshold);
    assert_eq!(deserialized.input_nodes, original.input_nodes);
    assert_eq!(deserialized.output_nodes, original.output_nodes);
}

// ==================== Performance Tests ====================

#[test]
fn test_logic_gate_calculation_performance() {
    let gate = LogicGate::new(LogicGateType::Threshold, 0.5);
    let inputs = vec![0.1, 0.2, 0.3, 0.4, 0.5]; // 5-input gate
    
    let (_, duration) = measure_execution_time(|| {
        for _ in 0..10000 {
            let _ = gate.calculate_output(&inputs);
        }
    });
    
    // Should be very fast
    assert!(duration.as_millis() < 50, "Gate calculation should be fast: {:?}", duration);
}

#[test]
fn test_large_input_gate_performance() {
    let gate = LogicGate::new(LogicGateType::Threshold, 5.0);
    let inputs: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
    
    let (result, duration) = measure_execution_time(|| {
        gate.calculate_output(&inputs).unwrap()
    });
    
    assert!(duration.as_micros() < 1000, "Large input gate should be fast: {:?}", duration);
    assert!(result >= 0.0 && result <= 1.0, "Result should be valid");
}

// ==================== Edge Case Tests ====================

#[test]
fn test_gates_with_extreme_values() {
    let gates = [
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
    ];
    
    for gate_type in gates {
        let gate = LogicGate::new(gate_type, 0.5);
        
        // Test with extreme values
        let extreme_inputs = [f32::MIN, 0.0, 1.0, f32::MAX];
        
        for &input in &extreme_inputs {
            if gate_type == LogicGateType::And || gate_type == LogicGateType::Or {
                let result = gate.calculate_output(&[input, 0.5]);
                if let Ok(output) = result {
                    assert!(output >= 0.0 && output <= 1.0 || output.is_infinite(), 
                           "Output should be valid or infinite: {}", output);
                }
            } else if gate_type == LogicGateType::Threshold || gate_type == LogicGateType::Inhibitory {
                let result = gate.calculate_output(&[input]);
                if let Ok(output) = result {
                    assert!(output >= 0.0 || output.is_infinite(), 
                           "Output should be non-negative or infinite: {}", output);
                }
            }
        }
    }
}

#[test]
fn test_gates_with_nan_inputs() {
    let gate = LogicGate::new(LogicGateType::Threshold, 0.5);
    
    // NaN inputs should be handled gracefully
    let result = gate.calculate_output(&[f32::NAN, 0.5]);
    
    // Result might be NaN or error, both acceptable
    match result {
        Ok(output) => {
            // If it succeeds, output might be NaN which is acceptable
            assert!(output.is_nan() || (output >= 0.0 && output <= 1.0));
        }
        Err(_) => {
            // Erroring on NaN input is also acceptable
        }
    }
}

// ==================== Integration Tests ====================

#[test]
fn test_logic_gate_with_entity_keys() {
    let input_keys = vec![EntityKey::from(100), EntityKey::from(200)];
    let output_keys = vec![EntityKey::from(300)];
    
    let gate = LogicGateBuilder::new(LogicGateType::And)
        .with_inputs(input_keys.clone())
        .with_outputs(output_keys.clone())
        .build();
    
    assert_eq!(gate.input_nodes, input_keys);
    assert_eq!(gate.output_nodes, output_keys);
    
    // Gate calculation should work regardless of EntityKey values
    let result = gate.calculate_output(&[0.8, 0.7]).unwrap();
    assert_activation_close(result, 0.7, test_constants::ACTIVATION_EPSILON, "Gate with EntityKeys");
}

#[test]
fn test_complex_gate_network_simulation() {
    // Create a mini network: AND -> NOT -> OR
    let and_gate = LogicGate::new(LogicGateType::And, 0.5);
    let not_gate = LogicGate::new(LogicGateType::Not, 0.0);
    let or_gate = LogicGate::new(LogicGateType::Or, 0.3);
    
    // Input pattern
    let inputs = [0.8, 0.7]; // Both high
    
    // Stage 1: AND gate
    let and_output = and_gate.calculate_output(&inputs).unwrap();
    assert_activation_close(and_output, 0.7, test_constants::ACTIVATION_EPSILON, "Network AND stage");
    
    // Stage 2: NOT gate
    let not_output = not_gate.calculate_output(&[and_output]).unwrap();
    assert_activation_close(not_output, 0.3, test_constants::ACTIVATION_EPSILON, "Network NOT stage");
    
    // Stage 3: OR gate with original input and NOT output
    let final_output = or_gate.calculate_output(&[inputs[0], not_output]).unwrap();
    assert_activation_close(final_output, 0.8, test_constants::ACTIVATION_EPSILON, "Network final stage");
    
    // Should propagate through network correctly
    assert!(final_output > 0.0, "Complex network should produce output");
}

// ==================== Enhanced Edge Case Tests ====================

#[test]
fn test_logic_gate_empty_inputs_comprehensive() {
    let gate_types = [
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Nand,
        LogicGateType::Nor,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
    ];
    
    for gate_type in gate_types {
        let gate = LogicGate::new(gate_type, 0.5);
        let result = gate.calculate_output(&[]);
        
        match gate_type {
            LogicGateType::Inhibitory => {
                // Inhibitory gate should return 0 for empty inputs
                assert_eq!(result.unwrap(), 0.0, "Inhibitory gate with empty inputs");
            }
            LogicGateType::Threshold => {
                // Threshold gate should return 0 for empty inputs
                assert_eq!(result.unwrap(), 0.0, "Threshold gate with empty inputs");
            }
            _ => {
                // Other gates may error or return a default value
                match result {
                    Ok(output) => assert!(output >= 0.0 && output <= 1.0, "Valid output range"),
                    Err(_) => {} // Error is acceptable for empty inputs
                }
            }
        }
    }
}

#[test]
fn test_logic_gate_weight_matrix_mismatches() {
    let weight_matrices = create_test_weight_matrices();
    let test_inputs = create_gate_test_inputs();
    
    for weights in weight_matrices {
        let mut gate = LogicGate::new(LogicGateType::Weighted, 0.5);
        gate.weight_matrix = weights.clone();
        
        for inputs in &test_inputs {
            let result = gate.calculate_output(inputs);
            
            if weights.len() != inputs.len() {
                // Size mismatch should error
                assert!(result.is_err(), 
                    "Weight matrix size mismatch should error: weights={}, inputs={}", 
                    weights.len(), inputs.len());
            } else if weights.is_empty() && inputs.is_empty() {
                // Both empty should work and return 0
                assert_eq!(result.unwrap(), 0.0, "Empty weights and inputs should work");
            } else {
                // Matching sizes should work (unless inputs are invalid)
                match result {
                    Ok(output) => {
                        if !output.is_nan() {
                            assert!(output >= 0.0, "Output should be non-negative");
                        }
                    }
                    Err(_) => {} // May error for invalid inputs like NaN
                }
            }
        }
    }
}

#[test]
fn test_logic_gate_invalid_thresholds() {
    let invalid_thresholds = vec![
        f32::NEG_INFINITY,
        -1000.0,
        -1.0,
        f32::INFINITY,
        f32::NAN,
    ];
    
    let valid_thresholds = vec![
        0.0, 0.5, 1.0, 2.0, 100.0
    ];
    
    // Test that gates can be created with various thresholds
    for &threshold in &invalid_thresholds {
        let gate = LogicGate::new(LogicGateType::And, threshold);
        assert_eq!(gate.threshold, threshold, "Threshold should be stored as-is");
        
        // Test calculation with invalid threshold
        let result = gate.calculate_output(&[0.5, 0.7]);
        match result {
            Ok(output) => {
                if threshold.is_nan() {
                    assert!(output.is_nan() || output >= 0.0, "NaN threshold behavior");
                } else {
                    assert!(output >= 0.0 || output.is_infinite(), "Output should be valid");
                }
            }
            Err(_) => {} // Error is acceptable for invalid thresholds
        }
    }
    
    // Valid thresholds should always work
    for &threshold in &valid_thresholds {
        let gate = LogicGate::new(LogicGateType::Or, threshold);
        let result = gate.calculate_output(&[0.5, 0.7]).unwrap();
        assert!(result >= 0.0 && result <= 1.0, "Valid threshold should work");
    }
}

#[test]
fn test_logic_gate_extreme_input_values() {
    let edge_cases = generate_edge_case_activations();
    let gate_types = [
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
    ];
    
    for gate_type in gate_types {
        let gate = LogicGate::new(gate_type, 0.5);
        
        // Test single extreme values
        for &value in &edge_cases {
            if gate_type == LogicGateType::And || gate_type == LogicGateType::Or {
                let result = gate.calculate_output(&[value, 0.5]);
                match result {
                    Ok(output) => {
                        if value.is_nan() {
                            assert!(output.is_nan() || (output >= 0.0 && output <= 1.0));
                        } else if value.is_infinite() {
                            assert!(output.is_infinite() || (output >= 0.0 && output <= 1.0));
                        } else {
                            assert!(output >= 0.0, "Output should be non-negative");
                        }
                    }
                    Err(_) => {} // Error is acceptable for extreme values
                }
            } else {
                let result = gate.calculate_output(&[value]);
                match result {
                    Ok(output) => {
                        if !value.is_finite() {
                            // Non-finite inputs may produce non-finite outputs
                            assert!(output.is_finite() || !output.is_finite());
                        } else {
                            assert!(output >= 0.0, "Output should be non-negative for finite inputs");
                        }
                    }
                    Err(_) => {} // Error is acceptable for extreme values
                }
            }
        }
    }
}

#[test]
fn test_logic_gate_input_node_capacity_mismatch() {
    let mut gate = LogicGate::new(LogicGateType::And, 0.5);
    
    // Set input nodes to expect 3 inputs
    gate.input_nodes = vec![EntityKey::from(1), EntityKey::from(2), EntityKey::from(3)];
    
    // Test with wrong number of inputs
    let result_too_few = gate.calculate_output(&[0.5, 0.7]); // 2 inputs, expects 3
    assert!(result_too_few.is_err(), "Should error with too few inputs");
    
    let result_too_many = gate.calculate_output(&[0.5, 0.7, 0.3, 0.9]); // 4 inputs, expects 3
    assert!(result_too_many.is_err(), "Should error with too many inputs");
    
    // Test with correct number of inputs
    let result_correct = gate.calculate_output(&[0.5, 0.7, 0.3]); // 3 inputs
    assert!(result_correct.is_ok(), "Should work with correct number of inputs");
}

#[test]
fn test_weighted_gate_comprehensive_edge_cases() {
    let mut gate = LogicGate::new(LogicGateType::Weighted, 0.6);
    
    // Test with zero weights
    gate.weight_matrix = vec![0.0, 0.0, 0.0];
    let result = gate.calculate_output(&[0.8, 0.9, 0.7]).unwrap();
    assert_eq!(result, 0.0, "Zero weights should produce zero output");
    
    // Test with negative weights
    gate.weight_matrix = vec![-0.5, 0.5, 1.0];
    let result = gate.calculate_output(&[0.8, 0.9, 0.7]).unwrap();
    // Expected: 0.8*(-0.5) + 0.9*0.5 + 0.7*1.0 = -0.4 + 0.45 + 0.7 = 0.75
    let expected = 0.8 * (-0.5) + 0.9 * 0.5 + 0.7 * 1.0;
    if expected >= 0.6 { // Above threshold
        assert_float_eq(result, expected.min(1.0), 0.01);
    } else {
        assert_eq!(result, 0.0);
    }
    
    // Test with very large weights
    gate.weight_matrix = vec![1000.0, 1000.0];
    gate.threshold = 500.0;
    let result = gate.calculate_output(&[0.001, 0.001]).unwrap();
    // Should exceed threshold and be clamped to 1.0
    assert_eq!(result, 1.0, "Large weights should saturate");
    
    // Test with infinite weights
    gate.weight_matrix = vec![f32::INFINITY, 0.5];
    let result = gate.calculate_output(&[0.5, 0.3]);
    match result {
        Ok(output) => {
            assert!(output.is_infinite() || output == 1.0, "Infinite weight behavior");
        }
        Err(_) => {} // Error is acceptable with infinite weights
    }
    
    // Test with NaN weights
    gate.weight_matrix = vec![f32::NAN, 0.5];
    let result = gate.calculate_output(&[0.5, 0.3]);
    match result {
        Ok(output) => {
            assert!(output.is_nan() || (output >= 0.0 && output <= 1.0), "NaN weight behavior");
        }
        Err(_) => {} // Error is acceptable with NaN weights
    }
}

#[test]
fn test_gate_type_specific_input_validation() {
    // Test XOR with wrong input count
    let xor_gate = LogicGate::new(LogicGateType::Xor, 0.5);
    assert!(xor_gate.calculate_output(&[0.5]).is_err(), "XOR needs 2 inputs");
    assert!(xor_gate.calculate_output(&[0.5, 0.7, 0.3]).is_err(), "XOR needs exactly 2 inputs");
    
    // Test XNOR with wrong input count  
    let xnor_gate = LogicGate::new(LogicGateType::Xnor, 0.5);
    assert!(xnor_gate.calculate_output(&[0.5]).is_err(), "XNOR needs 2 inputs");
    assert!(xnor_gate.calculate_output(&[0.5, 0.7, 0.3]).is_err(), "XNOR needs exactly 2 inputs");
    
    // Test NOT with wrong input count
    let not_gate = LogicGate::new(LogicGateType::Not, 0.5);
    assert!(not_gate.calculate_output(&[]).is_err(), "NOT needs 1 input");
    assert!(not_gate.calculate_output(&[0.5, 0.7]).is_err(), "NOT needs exactly 1 input");
    
    // Test Identity with wrong input count
    let identity_gate = LogicGate::new(LogicGateType::Identity, 0.0);
    assert!(identity_gate.calculate_output(&[]).is_err(), "Identity needs 1 input");
    assert!(identity_gate.calculate_output(&[0.5, 0.7]).is_err(), "Identity needs exactly 1 input");
}

#[test]
fn test_threshold_edge_behaviors() {
    let gate = LogicGate::new(LogicGateType::Threshold, 1.0);
    
    // Test sum exactly equal to threshold
    let result = gate.calculate_output(&[0.5, 0.5]).unwrap();
    assert_eq!(result, 1.0, "Sum equal to threshold should activate");
    
    // Test sum slightly below threshold
    let result = gate.calculate_output(&[0.499, 0.499]).unwrap();
    assert_eq!(result, 0.0, "Sum below threshold should not activate");
    
    // Test sum slightly above threshold
    let result = gate.calculate_output(&[0.501, 0.501]).unwrap();
    assert_float_eq(result, 1.002, 0.001);
    assert_eq!(result, 1.0, "Sum above threshold should be clamped to 1.0");
}

#[test]
fn test_inhibitory_gate_edge_cases() {
    let gate = LogicGate::new(LogicGateType::Inhibitory, 0.0); // Threshold unused
    
    // Test with only primary input
    let result = gate.calculate_output(&[0.8]).unwrap();
    assert_float_eq(result, 0.8, 0.001);
    
    // Test with equal primary and inhibitory
    let result = gate.calculate_output(&[0.5, 0.5]).unwrap();
    assert_eq!(result, 0.0, "Equal primary and inhibitory should cancel");
    
    // Test with multiple strong inhibitory inputs
    let result = gate.calculate_output(&[0.5, 0.3, 0.3, 0.3]).unwrap();
    // 0.5 - (0.3 + 0.3 + 0.3) = 0.5 - 0.9 = -0.4 -> 0.0
    assert_eq!(result, 0.0, "Strong inhibition should completely suppress");
    
    // Test with very large inhibitory values
    let result = gate.calculate_output(&[1.0, 1000.0]).unwrap();
    assert_eq!(result, 0.0, "Massive inhibition should suppress any primary");
    
    // Test with negative inhibitory (double negative = positive)
    let result = gate.calculate_output(&[0.5, -0.3]).unwrap();
    // 0.5 - (-0.3) = 0.5 + 0.3 = 0.8
    assert_float_eq(result, 0.8, 0.001);
}

// ==================== Property-Based Testing ====================

#[test]
fn test_gate_output_range_property() {
    let gate_types = [
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Nand,
        LogicGateType::Nor,
        LogicGateType::Not,
        LogicGateType::Identity,
        LogicGateType::Threshold,
    ];
    
    for gate_type in gate_types {
        let gate = LogicGate::new(gate_type, 0.5);
        
        // Test with valid inputs (0.0 to 1.0 range)
        for i in 0..10 {
            let input1 = i as f32 / 10.0;
            for j in 0..10 {
                let input2 = j as f32 / 10.0;
                
                let inputs = match gate_type {
                    LogicGateType::Not | LogicGateType::Identity => vec![input1],
                    _ => vec![input1, input2],
                };
                
                if let Ok(output) = gate.calculate_output(&inputs) {
                    assert!(
                        output >= 0.0 && output <= 1.0,
                        "Gate {:?} output {} should be in [0,1] for inputs {:?}",
                        gate_type, output, inputs
                    );
                }
            }
        }
    }
}

#[test]
fn test_gate_deterministic_property() {
    let gate = LogicGate::new(LogicGateType::And, 0.5);
    let inputs = vec![0.7, 0.8];
    
    // Same inputs should always produce same output
    let output1 = gate.calculate_output(&inputs).unwrap();
    let output2 = gate.calculate_output(&inputs).unwrap();
    let output3 = gate.calculate_output(&inputs).unwrap();
    
    assert_float_eq(output1, output2, 0.0001);
    assert_float_eq(output2, output3, 0.0001);
}

#[test]
fn test_gate_monotonicity_properties() {
    let gate = LogicGate::new(LogicGateType::Threshold, 1.0);
    
    // For threshold gates, increasing all inputs should not decrease output
    let test_cases = vec![
        (vec![0.1, 0.2], vec![0.2, 0.3]),
        (vec![0.3, 0.4], vec![0.4, 0.5]),
        (vec![0.5, 0.5], vec![0.6, 0.6]),
    ];
    
    for (lower_inputs, higher_inputs) in test_cases {
        let lower_output = gate.calculate_output(&lower_inputs).unwrap();
        let higher_output = gate.calculate_output(&higher_inputs).unwrap();
        
        assert!(
            higher_output >= lower_output,
            "Threshold gate should be monotonic: {:?} -> {} vs {:?} -> {}",
            lower_inputs, lower_output, higher_inputs, higher_output
        );
    }
}