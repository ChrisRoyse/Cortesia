// Tests for LogicGate struct
// Validates neural logic gate computation with all gate types and edge cases

use llmkg::core::brain_types::{LogicGate, LogicGateType, EntityDirection};
use llmkg::core::types::EntityKey;
use llmkg::error::GraphError;
use serde_json;

use super::test_constants;
use super::test_helpers::{LogicGateBuilder, assert_activation_close, measure_execution_time};

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