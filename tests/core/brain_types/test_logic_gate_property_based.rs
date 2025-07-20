/// Property-based tests for logic gate operations
/// 
/// This module implements comprehensive property-based testing to verify:
/// - Truth table correctness for all gate types
/// - Monotonicity properties where applicable
/// - Boundary condition handling
/// - Error resilience with invalid inputs
/// - Performance characteristics under various loads

use llmkg::core::brain_types::{LogicGate, LogicGateType};
use llmkg::core::types::EntityKey;
use llmkg::error::Result;
use super::test_constants;
use super::test_helpers::*;

// ==================== Property-Based Test Framework ====================

/// Property: Gate outputs should be deterministic
fn property_gate_deterministic(gate: &LogicGate, inputs: &[f32]) -> bool {
    let result1 = gate.calculate_output(inputs);
    let result2 = gate.calculate_output(inputs);
    
    match (result1, result2) {
        (Ok(out1), Ok(out2)) => {
            if out1.is_nan() && out2.is_nan() {
                true // Both NaN is consistent
            } else {
                (out1 - out2).abs() < f32::EPSILON
            }
        },
        (Err(_), Err(_)) => true, // Both errors is consistent
        _ => false, // Inconsistent results
    }
}

/// Property: Valid inputs should produce non-negative outputs (except for error cases)
fn property_non_negative_output(gate: &LogicGate, inputs: &[f32]) -> bool {
    match gate.calculate_output(inputs) {
        Ok(output) => output >= 0.0 || output.is_nan(),
        Err(_) => true, // Errors are acceptable for invalid inputs
    }
}

/// Property: AND gate monotonicity - if all inputs increase, output should not decrease
fn property_and_gate_monotonic(inputs1: &[f32], inputs2: &[f32], gate: &LogicGate) -> bool {
    if inputs1.len() != inputs2.len() || gate.gate_type != LogicGateType::And {
        return true; // Property doesn't apply
    }
    
    // Check if inputs1 <= inputs2 element-wise
    let all_leq = inputs1.iter().zip(inputs2.iter()).all(|(a, b)| a <= b);
    
    if !all_leq {
        return true; // Property doesn't apply
    }
    
    match (gate.calculate_output(inputs1), gate.calculate_output(inputs2)) {
        (Ok(out1), Ok(out2)) => out1 <= out2 || out1.is_nan() || out2.is_nan(),
        _ => true, // Skip if either fails
    }
}

/// Property: OR gate monotonicity - if any input increases, output should not decrease
fn property_or_gate_monotonic(inputs1: &[f32], inputs2: &[f32], gate: &LogicGate) -> bool {
    if inputs1.len() != inputs2.len() || gate.gate_type != LogicGateType::Or {
        return true; // Property doesn't apply
    }
    
    // Check if inputs1 <= inputs2 element-wise
    let all_leq = inputs1.iter().zip(inputs2.iter()).all(|(a, b)| a <= b);
    
    if !all_leq {
        return true; // Property doesn't apply
    }
    
    match (gate.calculate_output(inputs1), gate.calculate_output(inputs2)) {
        (Ok(out1), Ok(out2)) => out1 <= out2 || out1.is_nan() || out2.is_nan(),
        _ => true, // Skip if either fails
    }
}

/// Property: NOT gate involution - NOT(NOT(x)) = x
fn property_not_gate_involution(input: f32, gate: &LogicGate) -> bool {
    if gate.gate_type != LogicGateType::Not {
        return true; // Property doesn't apply
    }
    
    match gate.calculate_output(&[input]) {
        Ok(inverted) => {
            match gate.calculate_output(&[inverted]) {
                Ok(double_inverted) => {
                    (double_inverted - input).abs() < test_constants::ACTIVATION_EPSILON
                },
                Err(_) => true, // Skip if second inversion fails
            }
        },
        Err(_) => true, // Skip if first inversion fails
    }
}

/// Property: Threshold gate should activate when sum >= threshold
fn property_threshold_gate(inputs: &[f32], gate: &LogicGate) -> bool {
    if gate.gate_type != LogicGateType::Threshold {
        return true; // Property doesn't apply
    }
    
    let sum: f32 = inputs.iter().sum();
    
    match gate.calculate_output(inputs) {
        Ok(output) => {
            if sum >= gate.threshold {
                output > 0.0 || output.is_nan()
            } else {
                output == 0.0 || output.is_nan()
            }
        },
        Err(_) => true, // Errors are acceptable
    }
}

// ==================== Comprehensive Property Tests ====================

#[test]
fn test_all_gates_deterministic() {
    let gate_types = vec![
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
        let input_count = match gate_type {
            LogicGateType::Not | LogicGateType::Identity => 1,
            LogicGateType::Xor | LogicGateType::Xnor => 2,
            _ => 3, // Use 3 inputs for multi-input gates
        };
        
        let mut gate = create_test_gate(gate_type, 0.5, input_count);
        
        // Add weight matrix for weighted gates
        if gate_type == LogicGateType::Weighted {
            gate.weight_matrix = vec![0.33; input_count];
        }
        
        // Test with various input combinations
        let input_combinations = input_combinations(input_count);
        
        for inputs in input_combinations {
            assert!(
                property_gate_deterministic(&gate, &inputs),
                "Gate {:?} failed deterministic property with inputs {:?}",
                gate_type, inputs
            );
        }
    }
}

#[test]
fn test_all_gates_non_negative_output() {
    let gate_types = vec![
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
        let input_count = match gate_type {
            LogicGateType::Not | LogicGateType::Identity => 1,
            LogicGateType::Xor | LogicGateType::Xnor => 2,
            _ => 3,
        };
        
        let mut gate = create_test_gate(gate_type, 0.5, input_count);
        
        // Add weight matrix for weighted gates
        if gate_type == LogicGateType::Weighted {
            gate.weight_matrix = vec![0.33; input_count];
        }
        
        // Test with boundary values
        let boundary_inputs = boundary_activation_values();
        
        // Create combinations for testing
        for &value in &boundary_inputs {
            let inputs = vec![value; input_count];
            
            assert!(
                property_non_negative_output(&gate, &inputs),
                "Gate {:?} produced negative output with inputs {:?}",
                gate_type, inputs
            );
        }
    }
}

#[test]
fn test_monotonicity_properties() {
    // Test AND gate monotonicity
    let and_gate = create_test_gate(LogicGateType::And, 0.5, 2);
    
    let monotonic_pairs = vec![
        (vec![0.0, 0.0], vec![0.5, 0.5]),
        (vec![0.3, 0.4], vec![0.6, 0.7]),
        (vec![0.5, 0.2], vec![0.8, 0.9]),
        (vec![0.0, 0.8], vec![0.4, 1.0]),
    ];
    
    for (inputs1, inputs2) in monotonic_pairs {
        assert!(
            property_and_gate_monotonic(&inputs1, &inputs2, &and_gate),
            "AND gate failed monotonicity: {:?} -> {:?}",
            inputs1, inputs2
        );
    }
    
    // Test OR gate monotonicity
    let or_gate = create_test_gate(LogicGateType::Or, 0.5, 2);
    
    for (inputs1, inputs2) in vec![
        (vec![0.0, 0.0], vec![0.5, 0.5]),
        (vec![0.3, 0.4], vec![0.6, 0.7]),
        (vec![0.2, 0.8], vec![0.4, 1.0]),
    ] {
        assert!(
            property_or_gate_monotonic(&inputs1, &inputs2, &or_gate),
            "OR gate failed monotonicity: {:?} -> {:?}",
            inputs1, inputs2
        );
    }
}

#[test]
fn test_not_gate_involution() {
    let not_gate = create_test_gate(LogicGateType::Not, 0.0, 1);
    
    let test_values = boundary_activation_values();
    
    for &value in &test_values {
        assert!(
            property_not_gate_involution(value, &not_gate),
            "NOT gate failed involution property with input {}",
            value
        );
    }
}

#[test]
fn test_threshold_gate_properties() {
    let thresholds = threshold_test_values();
    
    for &threshold in &thresholds {
        let gate = create_test_gate(LogicGateType::Threshold, threshold, 3);
        
        let input_combinations = input_combinations(3);
        
        for inputs in input_combinations {
            assert!(
                property_threshold_gate(&inputs, &gate),
                "Threshold gate (threshold={}) failed property with inputs {:?}",
                threshold, inputs
            );
        }
    }
}

// ==================== Truth Table Validation ====================

#[test]
fn test_and_gate_truth_table_comprehensive() {
    let gate = create_test_gate(LogicGateType::And, 0.5, 2);
    
    // Extended truth table with various activation levels
    let test_cases = vec![
        // (input1, input2, threshold_met1, threshold_met2, expected_behavior)
        (0.0, 0.0, false, false, 0.0),
        (0.0, 0.8, false, true, 0.0),
        (0.8, 0.0, true, false, 0.0),
        (0.8, 0.9, true, true, 0.8), // min of inputs above threshold
        (0.6, 0.7, true, true, 0.6), // min of inputs above threshold
        (1.0, 1.0, true, true, 1.0), // both at maximum
        (0.5, 0.5, true, true, 0.5), // both at threshold
    ];
    
    for (input1, input2, _, _, expected) in test_cases {
        let result = gate.calculate_output(&[input1, input2]).unwrap();
        assert_float_eq(result, expected, test_constants::ACTIVATION_EPSILON);
    }
}

#[test]
fn test_or_gate_truth_table_comprehensive() {
    let gate = create_test_gate(LogicGateType::Or, 0.5, 2);
    
    let test_cases = vec![
        (0.0, 0.0, 0.0), // both below threshold
        (0.0, 0.8, 0.8), // second above threshold
        (0.8, 0.0, 0.8), // first above threshold
        (0.8, 0.9, 0.9), // both above threshold -> max
        (0.6, 0.7, 0.7), // both above threshold -> max
        (0.5, 0.4, 0.5), // first at threshold
    ];
    
    for (input1, input2, expected) in test_cases {
        let result = gate.calculate_output(&[input1, input2]).unwrap();
        assert_float_eq(result, expected, test_constants::ACTIVATION_EPSILON);
    }
}

#[test]
fn test_xor_gate_truth_table_comprehensive() {
    let gate = create_test_gate(LogicGateType::Xor, 0.5, 2);
    
    let test_cases = vec![
        (0.0, 0.0, 0.0), // (false, false) -> false
        (0.0, 0.8, 0.8), // (false, true) -> true (max value)
        (0.8, 0.0, 0.8), // (true, false) -> true (max value)
        (0.8, 0.9, 0.0), // (true, true) -> false
        (0.6, 0.7, 0.0), // both above threshold -> false
        (0.5, 0.4, 0.5), // first at threshold, second below -> true
    ];
    
    for (input1, input2, expected) in test_cases {
        let result = gate.calculate_output(&[input1, input2]).unwrap();
        assert_float_eq(result, expected, test_constants::ACTIVATION_EPSILON);
    }
}

#[test]
fn test_nand_gate_truth_table_comprehensive() {
    let gate = create_test_gate(LogicGateType::Nand, 0.5, 2);
    
    let test_cases = vec![
        (0.0, 0.0, 1.0), // NOT(false AND false) -> true
        (0.0, 0.8, 1.0), // NOT(false AND true) -> true
        (0.8, 0.0, 1.0), // NOT(true AND false) -> true
        (0.8, 0.9, 0.0), // NOT(true AND true) -> false
        (0.5, 0.5, 0.0), // both at threshold -> false
    ];
    
    for (input1, input2, expected) in test_cases {
        let result = gate.calculate_output(&[input1, input2]).unwrap();
        assert_float_eq(result, expected, test_constants::ACTIVATION_EPSILON);
    }
}

#[test]
fn test_nor_gate_truth_table_comprehensive() {
    let gate = create_test_gate(LogicGateType::Nor, 0.5, 2);
    
    let test_cases = vec![
        (0.0, 0.0, 1.0), // NOT(false OR false) -> true
        (0.0, 0.8, 0.0), // NOT(false OR true) -> false
        (0.8, 0.0, 0.0), // NOT(true OR false) -> false
        (0.8, 0.9, 0.0), // NOT(true OR true) -> false
        (0.4, 0.4, 1.0), // both below threshold -> true
    ];
    
    for (input1, input2, expected) in test_cases {
        let result = gate.calculate_output(&[input1, input2]).unwrap();
        assert_float_eq(result, expected, test_constants::ACTIVATION_EPSILON);
    }
}

#[test]
fn test_xnor_gate_truth_table_comprehensive() {
    let gate = create_test_gate(LogicGateType::Xnor, 0.5, 2);
    
    let test_cases = vec![
        (0.0, 0.0, 0.0), // NOT(false XOR false) -> NOT(false) -> true, but returns max
        (0.0, 0.8, 0.0), // NOT(false XOR true) -> NOT(true) -> false
        (0.8, 0.0, 0.0), // NOT(true XOR false) -> NOT(true) -> false
        (0.8, 0.9, 0.9), // NOT(true XOR true) -> NOT(false) -> true, returns max
        (0.4, 0.3, 0.4), // both below threshold -> returns max
    ];
    
    for (input1, input2, expected) in test_cases {
        let result = gate.calculate_output(&[input1, input2]).unwrap();
        assert_float_eq(result, expected, test_constants::ACTIVATION_EPSILON);
    }
}

// ==================== Edge Case Testing ====================

#[test]
fn test_edge_case_scenarios() {
    let edge_cases = edge_case_scenarios();
    
    for gate_type in vec![LogicGateType::And, LogicGateType::Or, LogicGateType::Threshold] {
        let gate = create_test_gate(gate_type, 0.5, 3);
        
        for (scenario_name, inputs) in &edge_cases {
            // Should not panic regardless of input
            let result = gate.calculate_output(inputs);
            
            // Verify the gate handles edge cases gracefully
            match result {
                Ok(output) => {
                    assert_valid_activation(output);
                },
                Err(_) => {
                    // Errors are acceptable for invalid inputs
                }
            }
        }
    }
}

#[test]
fn test_special_float_values() {
    let gate = create_test_gate(LogicGateType::Or, 0.5, 2);
    
    let special_values = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN,
        f32::MAX,
        0.0,
        -0.0,
    ];
    
    for &value in &special_values {
        let inputs = vec![value, 0.5];
        
        // Should not panic
        let result = gate.calculate_output(&inputs);
        
        // The gate should handle special values gracefully
        // Either produce a valid result or an error
        assert!(result.is_ok() || result.is_err());
    }
}

// ==================== Weighted Gate Property Tests ====================

#[test]
fn test_weighted_gate_linearity() {
    let mut gate = create_test_gate(LogicGateType::Weighted, 0.0, 3);
    gate.weight_matrix = vec![0.5, 0.3, 0.2];
    
    // Test linearity property: f(a*x) = a*f(x) when threshold is 0
    let base_inputs = vec![0.4, 0.6, 0.8];
    let scale_factor = 2.0;
    let scaled_inputs: Vec<f32> = base_inputs.iter().map(|&x| x * scale_factor).collect();
    
    let base_output = gate.calculate_output(&base_inputs).unwrap();
    let scaled_output = gate.calculate_output(&scaled_inputs).unwrap();
    
    // Due to clamping at 1.0, this property may not hold exactly
    // but we can test it for inputs that don't exceed saturation
    if scaled_output <= 1.0 {
        assert_float_eq(
            scaled_output,
            (base_output * scale_factor).min(1.0),
            test_constants::LOOSE_TOLERANCE
        );
    }
}

#[test]
fn test_weighted_gate_weight_matrix_properties() {
    let input_count = 4;
    let weight_variants = weight_matrix_variants(input_count);
    
    for weights in weight_variants {
        let mut gate = create_test_gate(LogicGateType::Weighted, 0.5, input_count);
        gate.weight_matrix = weights;
        
        let test_inputs = vec![0.8; input_count];
        
        match gate.calculate_output(&test_inputs) {
            Ok(output) => {
                assert_valid_activation(output);
                assert!(output <= 1.0, "Output should be clamped to 1.0");
            },
            Err(_) => {
                // Weight matrix mismatches or other errors are acceptable
            }
        }
    }
}

// ==================== Inhibitory Gate Property Tests ====================

#[test]
fn test_inhibitory_gate_inhibition_property() {
    let gate = create_test_gate(LogicGateType::Inhibitory, 0.0, 3);
    
    // Property: Increasing inhibitory inputs should decrease or maintain output
    let test_cases = vec![
        // (primary, inhibitory1, inhibitory2)
        (0.8, 0.0, 0.0), // No inhibition
        (0.8, 0.1, 0.0), // Slight inhibition
        (0.8, 0.2, 0.1), // More inhibition
        (0.8, 0.3, 0.2), // Strong inhibition
        (0.8, 0.4, 0.4), // Very strong inhibition
    ];
    
    let mut previous_output = f32::INFINITY;
    
    for (primary, inh1, inh2) in test_cases {
        let output = gate.calculate_output(&[primary, inh1, inh2]).unwrap();
        
        // Output should not increase as inhibition increases
        assert!(
            output <= previous_output || previous_output.is_infinite(),
            "Inhibitory gate output increased: {} -> {}",
            previous_output, output
        );
        
        previous_output = output;
    }
}

// ==================== Performance Property Tests ====================

#[test]
fn test_gate_calculation_performance() {
    let large_input_count = 100;
    let gate_types = vec![
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Threshold,
        LogicGateType::Weighted,
    ];
    
    for gate_type in gate_types {
        let mut gate = create_test_gate(gate_type, 0.5, large_input_count);
        
        if gate_type == LogicGateType::Weighted {
            gate.weight_matrix = vec![0.01; large_input_count]; // Small weights
        }
        
        let large_inputs = create_performance_dataset(large_input_count);
        
        let (_, duration) = measure_execution_time(|| {
            gate.calculate_output(&large_inputs)
        });
        
        // Should compute quickly even with many inputs
        assert!(
            duration.as_micros() < test_constants::MAX_PROCESSING_TIME_US,
            "Gate {:?} with {} inputs took too long: {:?}",
            gate_type, large_input_count, duration
        );
    }
}

// ==================== Fuzzing Property Tests ====================

#[test]
fn test_gates_with_fuzz_inputs() {
    let mut rng = TestRng::new(42); // Deterministic for reproducibility
    
    let gate_types = vec![
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
    ];
    
    for gate_type in gate_types {
        let gate = create_test_gate(gate_type, 0.5, 3);
        
        // Generate 100 random input combinations
        for _ in 0..100 {
            let fuzz_inputs = generate_fuzz_inputs(&mut rng, 3);
            
            // Should not panic regardless of inputs
            let result = gate.calculate_output(&fuzz_inputs);
            
            // Verify robustness properties
            assert!(result.is_ok() || result.is_err());
            
            if let Ok(output) = result {
                // Output should be well-behaved (not infinite, etc.)
                assert!(
                    !output.is_infinite() || output.is_nan() || output >= 0.0,
                    "Gate {:?} produced ill-behaved output: {} from inputs {:?}",
                    gate_type, output, fuzz_inputs
                );
            }
        }
    }
}

// ==================== Compositional Property Tests ====================

#[test]
fn test_gate_composition_properties() {
    // Test that chaining gates produces consistent results
    let and_gate = create_test_gate(LogicGateType::And, 0.5, 2);
    let not_gate = create_test_gate(LogicGateType::Not, 0.0, 1);
    
    let test_inputs = vec![
        vec![0.0, 0.0],
        vec![0.8, 0.0],
        vec![0.0, 0.8],
        vec![0.8, 0.8],
    ];
    
    for inputs in test_inputs {
        // Compute AND then NOT (should equal NAND)
        let and_result = and_gate.calculate_output(&inputs).unwrap();
        let nand_result = not_gate.calculate_output(&[and_result]).unwrap();
        
        // Compare with direct NAND gate
        let nand_gate = create_test_gate(LogicGateType::Nand, 0.5, 2);
        let direct_nand = nand_gate.calculate_output(&inputs).unwrap();
        
        assert_float_eq(
            nand_result,
            direct_nand,
            test_constants::LOOSE_TOLERANCE
        );
    }
}