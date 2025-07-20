use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    LogicGate, LogicGateType
};
use llmkg::core::types::EntityKey;

#[tokio::test]
async fn test_and_gate_operation() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create inputs and output for AND gate
    let input_a = BrainInspiredEntity::new("InputA".to_string(), EntityDirection::Input);
    let input_b = BrainInspiredEntity::new("InputB".to_string(), EntityDirection::Input);
    let output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);

    let key_a = input_a.id;
    let key_b = input_b.id;
    let key_output = output.id;

    engine.add_entity(input_a).await.unwrap();
    engine.add_entity(input_b).await.unwrap();
    engine.add_entity(output).await.unwrap();

    // Create AND gate
    let mut and_gate = LogicGate::new(LogicGateType::And, 0.5);
    and_gate.input_nodes.push(key_a);
    and_gate.input_nodes.push(key_b);
    and_gate.output_nodes.push(key_output);

    let gate_key = and_gate.gate_id;
    engine.add_logic_gate(and_gate).await.unwrap();

    // Test 1: Both inputs high (1, 1) -> 1
    let mut pattern1 = ActivationPattern::new("and_11".to_string());
    pattern1.activations.insert(key_a, 0.8);
    pattern1.activations.insert(key_b, 0.7);

    let result1 = engine.propagate_activation(&pattern1).await.unwrap();
    let output1 = result1.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output1 > 0.5, "AND(1,1) should produce high output");

    // Test 2: One input low (1, 0) -> 0
    let mut pattern2 = ActivationPattern::new("and_10".to_string());
    pattern2.activations.insert(key_a, 0.8);
    pattern2.activations.insert(key_b, 0.2);

    let result2 = engine.propagate_activation(&pattern2).await.unwrap();
    let output2 = result2.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output2 < 0.1, "AND(1,0) should produce low output");

    // Test 3: Both inputs low (0, 0) -> 0
    let mut pattern3 = ActivationPattern::new("and_00".to_string());
    pattern3.activations.insert(key_a, 0.1);
    pattern3.activations.insert(key_b, 0.2);

    let result3 = engine.propagate_activation(&pattern3).await.unwrap();
    let output3 = result3.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert_eq!(output3, 0.0, "AND(0,0) should produce zero output");
}

#[tokio::test]
async fn test_or_gate_operation() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create inputs and output for OR gate
    let input_a = BrainInspiredEntity::new("InputA".to_string(), EntityDirection::Input);
    let input_b = BrainInspiredEntity::new("InputB".to_string(), EntityDirection::Input);
    let output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);

    let key_a = input_a.id;
    let key_b = input_b.id;
    let key_output = output.id;

    engine.add_entity(input_a).await.unwrap();
    engine.add_entity(input_b).await.unwrap();
    engine.add_entity(output).await.unwrap();

    // Create OR gate
    let mut or_gate = LogicGate::new(LogicGateType::Or, 0.5);
    or_gate.input_nodes.push(key_a);
    or_gate.input_nodes.push(key_b);
    or_gate.output_nodes.push(key_output);

    engine.add_logic_gate(or_gate).await.unwrap();

    // Test 1: Both inputs high (1, 1) -> 1
    let mut pattern1 = ActivationPattern::new("or_11".to_string());
    pattern1.activations.insert(key_a, 0.8);
    pattern1.activations.insert(key_b, 0.7);

    let result1 = engine.propagate_activation(&pattern1).await.unwrap();
    let output1 = result1.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output1 > 0.6, "OR(1,1) should produce high output");

    // Test 2: One input high (1, 0) -> 1
    let mut pattern2 = ActivationPattern::new("or_10".to_string());
    pattern2.activations.insert(key_a, 0.8);
    pattern2.activations.insert(key_b, 0.2);

    let result2 = engine.propagate_activation(&pattern2).await.unwrap();
    let output2 = result2.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output2 > 0.5, "OR(1,0) should produce high output");

    // Test 3: Both inputs low (0, 0) -> 0
    let mut pattern3 = ActivationPattern::new("or_00".to_string());
    pattern3.activations.insert(key_a, 0.1);
    pattern3.activations.insert(key_b, 0.2);

    let result3 = engine.propagate_activation(&pattern3).await.unwrap();
    let output3 = result3.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert_eq!(output3, 0.0, "OR(0,0) should produce zero output");
}

#[tokio::test]
async fn test_not_gate_operation() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create input and output for NOT gate
    let input = BrainInspiredEntity::new("Input".to_string(), EntityDirection::Input);
    let output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);

    let key_input = input.id;
    let key_output = output.id;

    engine.add_entity(input).await.unwrap();
    engine.add_entity(output).await.unwrap();

    // Create NOT gate
    let mut not_gate = LogicGate::new(LogicGateType::Not, 0.0);
    not_gate.input_nodes.push(key_input);
    not_gate.output_nodes.push(key_output);

    engine.add_logic_gate(not_gate).await.unwrap();

    // Test 1: High input -> Low output
    let mut pattern1 = ActivationPattern::new("not_1".to_string());
    pattern1.activations.insert(key_input, 0.9);

    let result1 = engine.propagate_activation(&pattern1).await.unwrap();
    let output1 = result1.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output1 < 0.2, "NOT(1) should produce low output");

    // Test 2: Low input -> High output
    let mut pattern2 = ActivationPattern::new("not_0".to_string());
    pattern2.activations.insert(key_input, 0.1);

    let result2 = engine.propagate_activation(&pattern2).await.unwrap();
    let output2 = result2.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output2 > 0.8, "NOT(0) should produce high output");
}

#[tokio::test]
async fn test_xor_gate_operation() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create inputs and output for XOR gate
    let input_a = BrainInspiredEntity::new("InputA".to_string(), EntityDirection::Input);
    let input_b = BrainInspiredEntity::new("InputB".to_string(), EntityDirection::Input);
    let output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);

    let key_a = input_a.id;
    let key_b = input_b.id;
    let key_output = output.id;

    engine.add_entity(input_a).await.unwrap();
    engine.add_entity(input_b).await.unwrap();
    engine.add_entity(output).await.unwrap();

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

    // Test 2: Same inputs (1, 1) -> 0
    let mut pattern2 = ActivationPattern::new("xor_11".to_string());
    pattern2.activations.insert(key_a, 0.8);
    pattern2.activations.insert(key_b, 0.7);

    let result2 = engine.propagate_activation(&pattern2).await.unwrap();
    let output2 = result2.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert_eq!(output2, 0.0, "XOR(1,1) should produce zero output");
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

    let key_a = input_a.id;
    let key_b = input_b.id;
    let key_c = input_c.id;
    let key_output = output.id;

    engine.add_entity(input_a).await.unwrap();
    engine.add_entity(input_b).await.unwrap();
    engine.add_entity(input_c).await.unwrap();
    engine.add_entity(output).await.unwrap();

    // Create threshold gate (fires if sum >= 1.5)
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
    pattern1.activations.insert(key_c, 0.4); // Sum = 1.2 < 1.5

    let result1 = engine.propagate_activation(&pattern1).await.unwrap();
    let output1 = result1.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert_eq!(output1, 0.0, "Below threshold should produce zero output");

    // Test 2: Sum above threshold
    let mut pattern2 = ActivationPattern::new("threshold_high".to_string());
    pattern2.activations.insert(key_a, 0.6);
    pattern2.activations.insert(key_b, 0.6);
    pattern2.activations.insert(key_c, 0.5); // Sum = 1.7 > 1.5

    let result2 = engine.propagate_activation(&pattern2).await.unwrap();
    let output2 = result2.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert!(output2 > 0.5, "Above threshold should produce high output");
}

#[tokio::test]
async fn test_cascaded_logic_gates() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create complex circuit: (A AND B) OR C
    let input_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let input_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Input);
    let input_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Input);
    let intermediate = BrainInspiredEntity::new("AndOutput".to_string(), EntityDirection::Hidden);
    let output = BrainInspiredEntity::new("FinalOutput".to_string(), EntityDirection::Output);

    let key_a = input_a.id;
    let key_b = input_b.id;
    let key_c = input_c.id;
    let key_intermediate = intermediate.id;
    let key_output = output.id;

    engine.add_entity(input_a).await.unwrap();
    engine.add_entity(input_b).await.unwrap();
    engine.add_entity(input_c).await.unwrap();
    engine.add_entity(intermediate).await.unwrap();
    engine.add_entity(output).await.unwrap();

    // AND gate: A AND B -> intermediate
    let mut and_gate = LogicGate::new(LogicGateType::And, 0.5);
    and_gate.input_nodes.push(key_a);
    and_gate.input_nodes.push(key_b);
    and_gate.output_nodes.push(key_intermediate);

    // OR gate: intermediate OR C -> output
    let mut or_gate = LogicGate::new(LogicGateType::Or, 0.5);
    or_gate.input_nodes.push(key_intermediate);
    or_gate.input_nodes.push(key_c);
    or_gate.output_nodes.push(key_output);

    engine.add_logic_gate(and_gate).await.unwrap();
    engine.add_logic_gate(or_gate).await.unwrap();

    // Test: A=1, B=0, C=1 -> (0) OR 1 -> 1
    let mut pattern = ActivationPattern::new("cascaded_test".to_string());
    pattern.activations.insert(key_a, 0.8);
    pattern.activations.insert(key_b, 0.2);
    pattern.activations.insert(key_c, 0.9);

    let result = engine.propagate_activation(&pattern).await.unwrap();
    let final_output = result.final_activations.get(&key_output).copied().unwrap_or(0.0);
    
    assert!(final_output > 0.5, "Circuit should output high when C is high");
}