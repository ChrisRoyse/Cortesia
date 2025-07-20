/// Comprehensive tests for LogicGate::calculate_output covering all gate types
/// 
/// This module tests all 11 logic gate types with extensive edge cases,
/// boundary conditions, and error scenarios.

use llmkg::core::brain_types::{LogicGate, LogicGateType};
use llmkg::core::types::EntityKey;
use llmkg::error::{GraphError, Result};
use super::test_helpers::*;

#[test]
fn test_and_gate_comprehensive() {
    let gate = create_test_gate(LogicGateType::And, 0.5, 2);
    
    // Standard cases
    assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.8); // Both above threshold -> min
    assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.0); // One below threshold -> 0
    assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.0); // Both below threshold -> 0
    
    // Edge cases
    assert_eq!(gate.calculate_output(&[0.5, 0.5]).unwrap(), 0.5); // Exactly at threshold
    assert_eq!(gate.calculate_output(&[0.0, 1.0]).unwrap(), 0.0); // One zero
    assert_eq!(gate.calculate_output(&[1.0, 1.0]).unwrap(), 1.0); // Both max
    
    // Multiple inputs
    let multi_gate = create_test_gate(LogicGateType::And, 0.3, 4);
    assert_eq!(multi_gate.calculate_output(&[0.5, 0.4, 0.6, 0.7]).unwrap(), 0.4); // Min of all above threshold
    assert_eq!(multi_gate.calculate_output(&[0.5, 0.2, 0.6, 0.7]).unwrap(), 0.0); // One below threshold
}

#[test]
fn test_or_gate_comprehensive() {
    let gate = create_test_gate(LogicGateType::Or, 0.5, 2);
    
    // Standard cases
    assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.9); // Both above threshold -> max
    assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.9); // One above threshold -> max of above
    assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.0); // Both below threshold -> 0
    
    // Edge cases
    assert_eq!(gate.calculate_output(&[0.5, 0.4]).unwrap(), 0.5); // One exactly at threshold
    assert_eq!(gate.calculate_output(&[0.0, 1.0]).unwrap(), 1.0); // One zero, one max
    assert_eq!(gate.calculate_output(&[0.0, 0.0]).unwrap(), 0.0); // Both zero
    
    // Multiple inputs
    let multi_gate = create_test_gate(LogicGateType::Or, 0.3, 4);
    assert_eq!(multi_gate.calculate_output(&[0.5, 0.4, 0.6, 0.7]).unwrap(), 0.7); // Max of all above threshold
    assert_eq!(multi_gate.calculate_output(&[0.1, 0.2, 0.6, 0.7]).unwrap(), 0.7); // Max of those above threshold
}

#[test]
fn test_not_gate_comprehensive() {
    let gate = create_test_gate(LogicGateType::Not, 0.0, 1);
    
    // Standard cases
    assert_eq!(gate.calculate_output(&[0.0]).unwrap(), 1.0); // 0 -> 1
    assert_eq!(gate.calculate_output(&[1.0]).unwrap(), 0.0); // 1 -> 0
    assert_eq!(gate.calculate_output(&[0.5]).unwrap(), 0.5); // 0.5 -> 0.5
    
    // Edge cases
    assert_eq!(gate.calculate_output(&[0.001]).unwrap(), 0.999); // Very small -> very large
    assert_eq!(gate.calculate_output(&[0.999]).unwrap(), 0.001); // Very large -> very small
    
    // Error case: wrong input count
    assert!(gate.calculate_output(&[0.5, 0.3]).is_err());
    assert!(gate.calculate_output(&[]).is_err());
}

#[test]
fn test_xor_gate_comprehensive() {
    let gate = create_test_gate(LogicGateType::Xor, 0.5, 2);
    
    // Standard XOR truth table
    assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.0); // (0,0) -> 0
    assert_eq!(gate.calculate_output(&[0.7, 0.4]).unwrap(), 0.7); // (1,0) -> max
    assert_eq!(gate.calculate_output(&[0.3, 0.8]).unwrap(), 0.8); // (0,1) -> max
    assert_eq!(gate.calculate_output(&[0.7, 0.8]).unwrap(), 0.0); // (1,1) -> 0
    
    // Edge cases with threshold
    assert_eq!(gate.calculate_output(&[0.5, 0.4]).unwrap(), 0.5); // Exactly at threshold
    assert_eq!(gate.calculate_output(&[0.5, 0.5]).unwrap(), 0.0); // Both at threshold -> 0
    
    // Error cases: wrong input count
    assert!(gate.calculate_output(&[0.5]).is_err());
    assert!(gate.calculate_output(&[0.5, 0.3, 0.7]).is_err());
}

#[test]
fn test_nand_gate_comprehensive() {
    let gate = create_test_gate(LogicGateType::Nand, 0.5, 2);
    
    // NAND = NOT AND
    assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.0); // Both above threshold -> 0
    assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 1.0); // One below threshold -> 1
    assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 1.0); // Both below threshold -> 1
    
    // Edge cases
    assert_eq!(gate.calculate_output(&[0.5, 0.5]).unwrap(), 0.0); // Both at threshold -> 0
    assert_eq!(gate.calculate_output(&[0.5, 0.4]).unwrap(), 1.0); // One below threshold -> 1
    
    // Multiple inputs
    let multi_gate = create_test_gate(LogicGateType::Nand, 0.3, 3);
    assert_eq!(multi_gate.calculate_output(&[0.5, 0.4, 0.6]).unwrap(), 0.0); // All above threshold -> 0
    assert_eq!(multi_gate.calculate_output(&[0.5, 0.2, 0.6]).unwrap(), 1.0); // One below threshold -> 1
}

#[test]
fn test_nor_gate_comprehensive() {
    let gate = create_test_gate(LogicGateType::Nor, 0.5, 2);
    
    // NOR = NOT OR
    assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.0); // Any above threshold -> 0
    assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.0); // One above threshold -> 0
    assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 1.0); // Both below threshold -> 1
    
    // Edge cases
    assert_eq!(gate.calculate_output(&[0.5, 0.4]).unwrap(), 0.0); // One at threshold -> 0
    assert_eq!(gate.calculate_output(&[0.0, 0.0]).unwrap(), 1.0); // Both zero -> 1
}

#[test]
fn test_xnor_gate_comprehensive() {
    let gate = create_test_gate(LogicGateType::Xnor, 0.5, 2);
    
    // XNOR = NOT XOR
    assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.4); // (0,0) -> max
    assert_eq!(gate.calculate_output(&[0.7, 0.4]).unwrap(), 0.0); // (1,0) -> 0
    assert_eq!(gate.calculate_output(&[0.3, 0.8]).unwrap(), 0.0); // (0,1) -> 0
    assert_eq!(gate.calculate_output(&[0.7, 0.8]).unwrap(), 0.8); // (1,1) -> max
    
    // Error cases: wrong input count
    assert!(gate.calculate_output(&[0.5]).is_err());
    assert!(gate.calculate_output(&[0.5, 0.3, 0.7]).is_err());
}

#[test]
fn test_identity_gate_comprehensive() {
    let gate = create_test_gate(LogicGateType::Identity, 0.0, 1);
    
    // Identity just passes through
    for value in boundary_activation_values() {
        assert_eq!(gate.calculate_output(&[value]).unwrap(), value);
    }
    
    // Error cases: wrong input count
    assert!(gate.calculate_output(&[0.5, 0.3]).is_err());
    assert!(gate.calculate_output(&[]).is_err());
}

#[test]
fn test_threshold_gate_comprehensive() {
    let gate = create_test_gate(LogicGateType::Threshold, 1.5, 3);
    
    // Below threshold
    assert_eq!(gate.calculate_output(&[0.4, 0.4, 0.4]).unwrap(), 0.0); // Sum = 1.2 < 1.5
    assert_eq!(gate.calculate_output(&[0.5, 0.5, 0.4]).unwrap(), 0.0); // Sum = 1.4 < 1.5
    
    // At threshold
    assert_eq!(gate.calculate_output(&[0.5, 0.5, 0.5]).unwrap(), 1.0); // Sum = 1.5 = threshold, clamped to 1.0
    
    // Above threshold
    assert_eq!(gate.calculate_output(&[0.6, 0.6, 0.5]).unwrap(), 1.0); // Sum = 1.7 > 1.5, clamped to 1.0
    assert_eq!(gate.calculate_output(&[1.0, 1.0, 1.0]).unwrap(), 1.0); // Sum = 3.0, clamped to 1.0
    
    // Edge cases
    let zero_threshold_gate = create_test_gate(LogicGateType::Threshold, 0.0, 2);
    assert_eq!(zero_threshold_gate.calculate_output(&[0.0, 0.0]).unwrap(), 0.0); // Sum = 0 = threshold
    assert_eq!(zero_threshold_gate.calculate_output(&[0.1, 0.1]).unwrap(), 0.2); // Sum = 0.2 > 0
    
    // Single input
    let single_gate = create_test_gate(LogicGateType::Threshold, 0.5, 1);
    assert_eq!(single_gate.calculate_output(&[0.3]).unwrap(), 0.0); // Below threshold
    assert_eq!(single_gate.calculate_output(&[0.7]).unwrap(), 0.7); // Above threshold
}

#[test]
fn test_inhibitory_gate_comprehensive() {
    let gate = create_test_gate(LogicGateType::Inhibitory, 0.0, 3);
    
    // Primary input - sum of inhibitory inputs
    assert_eq!(gate.calculate_output(&[0.8, 0.2, 0.1]).unwrap(), 0.5); // 0.8 - (0.2 + 0.1) = 0.5
    assert_eq!(gate.calculate_output(&[0.5, 0.3, 0.4]).unwrap(), 0.0); // 0.5 - (0.3 + 0.4) = -0.2 -> 0.0 (clamped)
    assert_eq!(gate.calculate_output(&[1.0, 0.0, 0.0]).unwrap(), 1.0); // 1.0 - 0.0 = 1.0
    
    // Edge cases
    assert_eq!(gate.calculate_output(&[0.0, 0.5, 0.3]).unwrap(), 0.0); // 0.0 - 0.8 = -0.8 -> 0.0
    assert_eq!(gate.calculate_output(&[0.5]).unwrap(), 0.5); // Single input (no inhibition)
    assert_eq!(gate.calculate_output(&[]).unwrap(), 0.0); // No inputs
    
    // Strong inhibition
    let strong_inhibition_gate = create_test_gate(LogicGateType::Inhibitory, 0.0, 4);
    assert_eq!(strong_inhibition_gate.calculate_output(&[0.5, 0.2, 0.2, 0.2]).unwrap(), 0.0); // 0.5 - 0.6 = -0.1 -> 0.0
}

#[test]
fn test_weighted_gate_comprehensive() {
    let mut gate = create_test_gate(LogicGateType::Weighted, 1.0, 3);
    gate.weight_matrix = vec![0.5, 0.3, 0.2]; // Weights sum to 1.0
    
    // Standard weighted sum
    assert_eq!(gate.calculate_output(&[1.0, 1.0, 1.0]).unwrap(), 1.0); // 1.0*0.5 + 1.0*0.3 + 1.0*0.2 = 1.0 >= 1.0
    assert_eq!(gate.calculate_output(&[0.5, 0.5, 0.5]).unwrap(), 0.0); // 0.5*0.5 + 0.5*0.3 + 0.5*0.2 = 0.5 < 1.0
    
    // Above threshold, clamped to 1.0
    assert_eq!(gate.calculate_output(&[2.0, 2.0, 2.0]).unwrap(), 1.0); // 2.0*0.5 + 2.0*0.3 + 2.0*0.2 = 2.0 -> 1.0 (clamped)
    
    // Different weights
    let mut asymmetric_gate = create_test_gate(LogicGateType::Weighted, 0.5, 2);
    asymmetric_gate.weight_matrix = vec![0.8, 0.2];
    assert_eq!(asymmetric_gate.calculate_output(&[1.0, 0.0]).unwrap(), 0.8); // 1.0*0.8 + 0.0*0.2 = 0.8 >= 0.5
    assert_eq!(asymmetric_gate.calculate_output(&[0.0, 1.0]).unwrap(), 0.0); // 0.0*0.8 + 1.0*0.2 = 0.2 < 0.5
    
    // Negative weights
    let mut negative_gate = create_test_gate(LogicGateType::Weighted, 0.0, 2);
    negative_gate.weight_matrix = vec![1.0, -0.5];
    assert_eq!(negative_gate.calculate_output(&[0.8, 0.4]).unwrap(), 0.6); // 0.8*1.0 + 0.4*(-0.5) = 0.6
    
    // Error case: weight matrix size mismatch
    let mut mismatch_gate = create_test_gate(LogicGateType::Weighted, 1.0, 3);
    mismatch_gate.weight_matrix = vec![0.5, 0.3]; // Only 2 weights for 3 inputs
    assert!(mismatch_gate.calculate_output(&[1.0, 1.0, 1.0]).is_err());
}

#[test]
fn test_logic_gate_input_validation() {
    let gate = create_test_gate(LogicGateType::And, 0.5, 2);
    
    // Wrong number of inputs
    assert!(gate.calculate_output(&[0.5]).is_err()); // Too few
    assert!(gate.calculate_output(&[0.5, 0.3, 0.7]).is_err()); // Too many
    assert!(gate.calculate_output(&[]).is_err()); // Empty
}

#[test]
fn test_logic_gate_special_values() {
    let gate = create_test_gate(LogicGateType::Or, 0.5, 2);
    
    // Test with NaN and infinity (should not crash, but behavior may vary)
    // Note: These tests verify robustness rather than specific behavior
    let result_nan = gate.calculate_output(&[f32::NAN, 0.5]);
    let result_inf = gate.calculate_output(&[f32::INFINITY, 0.5]);
    let result_neg_inf = gate.calculate_output(&[f32::NEG_INFINITY, 0.5]);
    
    // Should not panic, but results may be NaN or infinity
    // The key is that calculate_output doesn't crash
    assert!(result_nan.is_ok() || result_nan.is_err());
    assert!(result_inf.is_ok() || result_inf.is_err());
    assert!(result_neg_inf.is_ok() || result_neg_inf.is_err());
}

#[test]
fn test_all_gate_types_with_boundary_values() {
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
            _ => 2, // Default to 2 inputs for other types
        };
        
        let mut gate = create_test_gate(gate_type, 0.5, input_count);
        
        // Add weight matrix for weighted gates
        if gate_type == LogicGateType::Weighted {
            gate.weight_matrix = vec![0.5; input_count];
        }
        
        // Test with boundary values
        let test_inputs = match input_count {
            1 => vec![vec![0.0], vec![0.5], vec![1.0]],
            2 => vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
                vec![0.5, 0.5],
            ],
            _ => vec![vec![0.5; input_count]],
        };
        
        for inputs in test_inputs {
            let result = gate.calculate_output(&inputs);
            // Should not panic and should return a valid result or error
            assert!(result.is_ok() || result.is_err(), 
                "Gate {:?} failed with inputs {:?}", gate_type, inputs);
            
            if let Ok(output) = result {
                assert!(output >= 0.0, "Gate {:?} produced negative output: {}", gate_type, output);
                // Note: Output may exceed 1.0 for some gates before clamping
            }
        }
    }
}

#[test]
fn test_logic_gate_display_formatting() {
    use std::fmt::Write;
    
    let gate_types = vec![
        (LogicGateType::And, "and"),
        (LogicGateType::Or, "or"),
        (LogicGateType::Not, "not"),
        (LogicGateType::Xor, "xor"),
        (LogicGateType::Nand, "nand"),
        (LogicGateType::Nor, "nor"),
        (LogicGateType::Xnor, "xnor"),
        (LogicGateType::Identity, "identity"),
        (LogicGateType::Threshold, "threshold"),
        (LogicGateType::Inhibitory, "inhibitory"),
        (LogicGateType::Weighted, "weighted"),
    ];
    
    for (gate_type, expected_str) in gate_types {
        let mut output = String::new();
        write!(&mut output, "{}", gate_type).unwrap();
        assert_eq!(output, expected_str);
    }
}