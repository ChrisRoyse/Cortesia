use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    LogicGate, LogicGateType
};
use llmkg::core::types::EntityKey;

#[tokio::test]
async fn test_xor_gate_operation() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create inputs and output for XOR gate
    let input_a = BrainInspiredEntity::new("InputA".to_string(), EntityDirection::Input);
    let input_b = BrainInspiredEntity::new("InputB".to_string(), EntityDirection::Input);
    let output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);

    let key_a = engine.add_entity(input_a).await.unwrap();
    let key_b = engine.add_entity(input_b).await.unwrap();
    let key_output = engine.add_entity(output).await.unwrap();

    // Create XOR gate
    let mut xor_gate = LogicGate::new(LogicGateType::Xor, 0.5);
    xor_gate.input_nodes.push(key_a);
    xor_gate.input_nodes.push(key_b);
    xor_gate.output_nodes.push(key_output);

    engine.add_logic_gate(xor_gate).await.unwrap();

    // Test 1: Different inputs (1, 0) -> 1
    let mut pattern1 = ActivationPattern::new("xor_10".to_string());
    pattern1.activations.insert(key_a, 0.8);
    pattern1.activations.insert(key_b, 0.2);

    let result1 = engine.propagate_activation(&pattern1).await.unwrap();
    let output1 = result1.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output1 > 0.5, "XOR(1,0) should produce high output");

    // Test 2: Different inputs (0, 1) -> 1
    let mut pattern2 = ActivationPattern::new("xor_01".to_string());
    pattern2.activations.insert(key_a, 0.2);
    pattern2.activations.insert(key_b, 0.8);

    let result2 = engine.propagate_activation(&pattern2).await.unwrap();
    let output2 = result2.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output2 > 0.5, "XOR(0,1) should produce high output");

    // Test 3: Same inputs (1, 1) -> 0
    let mut pattern3 = ActivationPattern::new("xor_11".to_string());
    pattern3.activations.insert(key_a, 0.8);
    pattern3.activations.insert(key_b, 0.8);

    let result3 = engine.propagate_activation(&pattern3).await.unwrap();
    let output3 = result3.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert_eq!(output3, 0.0, "XOR(1,1) should produce zero output");

    // Test 4: Same inputs (0, 0) -> 0
    let mut pattern4 = ActivationPattern::new("xor_00".to_string());
    pattern4.activations.insert(key_a, 0.2);
    pattern4.activations.insert(key_b, 0.2);

    let result4 = engine.propagate_activation(&pattern4).await.unwrap();
    let output4 = result4.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert_eq!(output4, 0.0, "XOR(0,0) should produce zero output");
}

#[tokio::test]
async fn test_threshold_gate_operation() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create multiple inputs for threshold gate
    let input_a = BrainInspiredEntity::new("InputA".to_string(), EntityDirection::Input);
    let input_b = BrainInspiredEntity::new("InputB".to_string(), EntityDirection::Input);
    let input_c = BrainInspiredEntity::new("InputC".to_string(), EntityDirection::Input);
    let output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);

    let key_a = engine.add_entity(input_a).await.unwrap();
    let key_b = engine.add_entity(input_b).await.unwrap();
    let key_c = engine.add_entity(input_c).await.unwrap();
    let key_output = engine.add_entity(output).await.unwrap();

    // Create threshold gate (fires when sum >= 1.5)
    let mut threshold_gate = LogicGate::new(LogicGateType::Threshold, 1.5);
    threshold_gate.input_nodes.push(key_a);
    threshold_gate.input_nodes.push(key_b);
    threshold_gate.input_nodes.push(key_c);
    threshold_gate.output_nodes.push(key_output);

    engine.add_logic_gate(threshold_gate).await.unwrap();

    // Test 1: Sum below threshold
    let mut pattern1 = ActivationPattern::new("threshold_low".to_string());
    pattern1.activations.insert(key_a, 0.4);
    pattern1.activations.insert(key_b, 0.4);
    pattern1.activations.insert(key_c, 0.4); // Sum = 1.2

    let result1 = engine.propagate_activation(&pattern1).await.unwrap();
    let output1 = result1.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert_eq!(output1, 0.0, "Below threshold should produce zero output");

    // Test 2: Sum exactly at threshold
    let mut pattern2 = ActivationPattern::new("threshold_exact".to_string());
    pattern2.activations.insert(key_a, 0.5);
    pattern2.activations.insert(key_b, 0.5);
    pattern2.activations.insert(key_c, 0.5); // Sum = 1.5

    let result2 = engine.propagate_activation(&pattern2).await.unwrap();
    let output2 = result2.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output2 > 0.0, "At threshold should produce output");

    // Test 3: Sum above threshold (clamped to 1.0)
    let mut pattern3 = ActivationPattern::new("threshold_high".to_string());
    pattern3.activations.insert(key_a, 0.8);
    pattern3.activations.insert(key_b, 0.8);
    pattern3.activations.insert(key_c, 0.8); // Sum = 2.4

    let result3 = engine.propagate_activation(&pattern3).await.unwrap();
    let output3 = result3.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert_eq!(output3, 1.0, "High sum should be clamped to 1.0");
}

#[tokio::test]
async fn test_weighted_gate_operation() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create inputs with different importance
    let input_important = BrainInspiredEntity::new("Important".to_string(), EntityDirection::Input);
    let input_medium = BrainInspiredEntity::new("Medium".to_string(), EntityDirection::Input);
    let input_low = BrainInspiredEntity::new("Low".to_string(), EntityDirection::Input);
    let output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);

    let key_important = engine.add_entity(input_important).await.unwrap();
    let key_medium = engine.add_entity(input_medium).await.unwrap();
    let key_low = engine.add_entity(input_low).await.unwrap();
    let key_output = engine.add_entity(output).await.unwrap();

    // Create weighted gate with different weights
    let mut weighted_gate = LogicGate::new(LogicGateType::Weighted, 0.5);
    weighted_gate.input_nodes.push(key_important);
    weighted_gate.input_nodes.push(key_medium);
    weighted_gate.input_nodes.push(key_low);
    weighted_gate.weight_matrix = vec![0.6, 0.3, 0.1]; // Weights sum to 1.0
    weighted_gate.output_nodes.push(key_output);

    engine.add_logic_gate(weighted_gate).await.unwrap();

    // Test 1: Only important input active
    let mut pattern1 = ActivationPattern::new("weighted_important".to_string());
    pattern1.activations.insert(key_important, 1.0);
    pattern1.activations.insert(key_medium, 0.0);
    pattern1.activations.insert(key_low, 0.0);

    let result1 = engine.propagate_activation(&pattern1).await.unwrap();
    let output1 = result1.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output1 > 0.5, "Important input alone should exceed threshold");
    assert!((output1 - 0.48).abs() < 0.1, "Output should be ~0.6 * 0.8 propagation");

    // Test 2: All low-weight inputs active
    let mut pattern2 = ActivationPattern::new("weighted_low_priority".to_string());
    pattern2.activations.insert(key_important, 0.0);
    pattern2.activations.insert(key_medium, 1.0);
    pattern2.activations.insert(key_low, 1.0);

    let result2 = engine.propagate_activation(&pattern2).await.unwrap();
    let output2 = result2.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output2 < 0.5, "Low-weight inputs shouldn't exceed threshold");
}

#[tokio::test]
async fn test_inhibitory_gate_operation() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create primary input and inhibitory inputs
    let input_primary = BrainInspiredEntity::new("Primary".to_string(), EntityDirection::Input);
    let input_inhib1 = BrainInspiredEntity::new("Inhibitor1".to_string(), EntityDirection::Input);
    let input_inhib2 = BrainInspiredEntity::new("Inhibitor2".to_string(), EntityDirection::Input);
    let output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);

    let key_primary = engine.add_entity(input_primary).await.unwrap();
    let key_inhib1 = engine.add_entity(input_inhib1).await.unwrap();
    let key_inhib2 = engine.add_entity(input_inhib2).await.unwrap();
    let key_output = engine.add_entity(output).await.unwrap();

    // Create inhibitory gate (first input minus others)
    let mut inhib_gate = LogicGate::new(LogicGateType::Inhibitory, 0.0);
    inhib_gate.input_nodes.push(key_primary);
    inhib_gate.input_nodes.push(key_inhib1);
    inhib_gate.input_nodes.push(key_inhib2);
    inhib_gate.output_nodes.push(key_output);

    engine.add_logic_gate(inhib_gate).await.unwrap();

    // Test 1: No inhibition
    let mut pattern1 = ActivationPattern::new("no_inhibition".to_string());
    pattern1.activations.insert(key_primary, 0.8);
    pattern1.activations.insert(key_inhib1, 0.0);
    pattern1.activations.insert(key_inhib2, 0.0);

    let result1 = engine.propagate_activation(&pattern1).await.unwrap();
    let output1 = result1.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output1 > 0.5, "Without inhibition, primary should pass through");

    // Test 2: Partial inhibition
    let mut pattern2 = ActivationPattern::new("partial_inhibition".to_string());
    pattern2.activations.insert(key_primary, 0.8);
    pattern2.activations.insert(key_inhib1, 0.3);
    pattern2.activations.insert(key_inhib2, 0.2);

    let result2 = engine.propagate_activation(&pattern2).await.unwrap();
    let output2 = result2.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output2 > 0.0 && output2 < output1, "Partial inhibition should reduce output");

    // Test 3: Complete inhibition
    let mut pattern3 = ActivationPattern::new("complete_inhibition".to_string());
    pattern3.activations.insert(key_primary, 0.8);
    pattern3.activations.insert(key_inhib1, 0.5);
    pattern3.activations.insert(key_inhib2, 0.5);

    let result3 = engine.propagate_activation(&pattern3).await.unwrap();
    let output3 = result3.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert_eq!(output3, 0.0, "Strong inhibition should eliminate output");
}

#[tokio::test]
async fn test_nand_gate_operation() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create inputs and output for NAND gate
    let input_a = BrainInspiredEntity::new("InputA".to_string(), EntityDirection::Input);
    let input_b = BrainInspiredEntity::new("InputB".to_string(), EntityDirection::Input);
    let output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);

    let key_a = engine.add_entity(input_a).await.unwrap();
    let key_b = engine.add_entity(input_b).await.unwrap();
    let key_output = engine.add_entity(output).await.unwrap();

    // Create NAND gate
    let mut nand_gate = LogicGate::new(LogicGateType::Nand, 0.5);
    nand_gate.input_nodes.push(key_a);
    nand_gate.input_nodes.push(key_b);
    nand_gate.output_nodes.push(key_output);

    engine.add_logic_gate(nand_gate).await.unwrap();

    // Test: NAND truth table
    let test_cases = vec![
        ((0.8, 0.8), false, "NAND(1,1) should be 0"),
        ((0.8, 0.2), true,  "NAND(1,0) should be 1"),
        ((0.2, 0.8), true,  "NAND(0,1) should be 1"),
        ((0.2, 0.2), true,  "NAND(0,0) should be 1"),
    ];

    for (i, ((a_val, b_val), expected_high, msg)) in test_cases.iter().enumerate() {
        let mut pattern = ActivationPattern::new(format!("nand_test_{}", i));
        pattern.activations.insert(key_a, *a_val);
        pattern.activations.insert(key_b, *b_val);

        let result = engine.propagate_activation(&pattern).await.unwrap();
        let output = result.final_activations.get(&key_output).copied().unwrap_or(0.0);
        
        if *expected_high {
            assert!(output > 0.5, "{}", msg);
        } else {
            assert!(output < 0.1, "{}", msg);
        }
    }
}