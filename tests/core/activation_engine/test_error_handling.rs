use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    LogicGate, LogicGateType
};
use llmkg::core::types::EntityKey;

#[tokio::test]
async fn test_not_gate_multiple_inputs_error() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create multiple inputs for NOT gate (invalid)
    let input_a = BrainInspiredEntity::new("InputA".to_string(), EntityDirection::Input);
    let input_b = BrainInspiredEntity::new("InputB".to_string(), EntityDirection::Input);
    let output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);

    let key_a = input_a.id;
    let key_b = input_b.id;
    let key_output = output.id;

    engine.add_entity(input_a).await.unwrap();
    engine.add_entity(input_b).await.unwrap();
    engine.add_entity(output).await.unwrap();

    // Create NOT gate with multiple inputs (should fail during processing)
    let mut not_gate = LogicGate::new(LogicGateType::Not, 0.0);
    not_gate.input_nodes.push(key_a);
    not_gate.input_nodes.push(key_b); // Invalid: NOT should have only 1 input
    not_gate.output_nodes.push(key_output);

    engine.add_logic_gate(not_gate).await.unwrap();

    // Try to propagate - should handle error gracefully
    let mut pattern = ActivationPattern::new("not_multiple_inputs".to_string());
    pattern.activations.insert(key_a, 0.5);
    pattern.activations.insert(key_b, 0.5);

    // The propagation should still complete, but the gate output should be 0
    let result = engine.propagate_activation(&pattern).await.unwrap();
    let output = result.final_activations.get(&key_output).copied().unwrap_or(0.0);
    
    // Gate calculation will fail, so output should remain at default (0.0)
    assert_eq!(output, 0.0, "Invalid NOT gate should produce no output");
}

#[tokio::test]
async fn test_xor_gate_wrong_input_count() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create three inputs for XOR gate (invalid - needs exactly 2)
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

    // Create XOR gate with 3 inputs (should only have 2)
    let mut xor_gate = LogicGate::new(LogicGateType::Xor, 0.5);
    xor_gate.input_nodes.push(key_a);
    xor_gate.input_nodes.push(key_b);
    xor_gate.input_nodes.push(key_c); // Invalid third input
    xor_gate.output_nodes.push(key_output);

    engine.add_logic_gate(xor_gate).await.unwrap();

    // Try to propagate
    let mut pattern = ActivationPattern::new("xor_three_inputs".to_string());
    pattern.activations.insert(key_a, 0.8);
    pattern.activations.insert(key_b, 0.2);
    pattern.activations.insert(key_c, 0.5);

    let result = engine.propagate_activation(&pattern).await.unwrap();
    let output = result.final_activations.get(&key_output).copied().unwrap_or(0.0);
    
    // XOR with wrong input count should produce no output
    assert_eq!(output, 0.0, "Invalid XOR gate should produce no output");
}

#[tokio::test]
async fn test_weighted_gate_matrix_mismatch() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create inputs
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

    // Create weighted gate with mismatched weight matrix
    let mut weighted_gate = LogicGate::new(LogicGateType::Weighted, 0.5);
    weighted_gate.input_nodes.push(key_a);
    weighted_gate.input_nodes.push(key_b);
    weighted_gate.input_nodes.push(key_c);
    weighted_gate.weight_matrix = vec![0.5, 0.5]; // Only 2 weights for 3 inputs!
    weighted_gate.output_nodes.push(key_output);

    engine.add_logic_gate(weighted_gate).await.unwrap();

    // Try to propagate
    let mut pattern = ActivationPattern::new("weighted_mismatch".to_string());
    pattern.activations.insert(key_a, 0.8);
    pattern.activations.insert(key_b, 0.8);
    pattern.activations.insert(key_c, 0.8);

    let result = engine.propagate_activation(&pattern).await.unwrap();
    let output = result.final_activations.get(&key_output).copied().unwrap_or(0.0);
    
    // Weighted gate with matrix mismatch should produce no output
    assert_eq!(output, 0.0, "Weighted gate with matrix mismatch should produce no output");
}

#[tokio::test]
async fn test_extreme_activation_values() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create simple network
    let input = BrainInspiredEntity::new("Input".to_string(), EntityDirection::Input);
    let hidden = BrainInspiredEntity::new("Hidden".to_string(), EntityDirection::Hidden);
    let output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);

    let key_input = input.id;
    let key_hidden = hidden.id;
    let key_output = output.id;

    engine.add_entity(input).await.unwrap();
    engine.add_entity(hidden).await.unwrap();
    engine.add_entity(output).await.unwrap();

    // Create relationships with extreme weights
    let mut rel1 = llmkg::core::brain_types::BrainInspiredRelationship::new(
        key_input, key_hidden, llmkg::core::brain_types::RelationType::RelatedTo
    );
    rel1.weight = 5.0; // Very high weight

    let mut rel2 = llmkg::core::brain_types::BrainInspiredRelationship::new(
        key_hidden, key_output, llmkg::core::brain_types::RelationType::RelatedTo
    );
    rel2.weight = 3.0; // Very high weight

    engine.add_relationship(rel1).await.unwrap();
    engine.add_relationship(rel2).await.unwrap();

    // Test with extreme input activation
    let mut pattern = ActivationPattern::new("extreme_values".to_string());
    pattern.activations.insert(key_input, 10.0); // Way above normal range

    let result = engine.propagate_activation(&pattern).await.unwrap();
    
    // Check all activations are clamped to valid range
    for (entity_key, activation) in result.final_activations.iter() {
        assert!(
            *activation >= 0.0 && *activation <= 1.0,
            "Activation for {:?} = {} should be clamped to [0,1]", 
            entity_key, activation
        );
    }

    // Output should be clamped to 1.0 despite extreme inputs
    let output_activation = result.final_activations.get(&key_output).copied().unwrap_or(0.0);
    assert_eq!(output_activation, 1.0, "Extreme activation should be clamped to 1.0");
}

#[tokio::test]
async fn test_empty_gate_inputs() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create output only
    let output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);
    let key_output = output.id;
    engine.add_entity(output).await.unwrap();

    // Create gate with no inputs
    let mut empty_gate = LogicGate::new(LogicGateType::And, 0.5);
    empty_gate.output_nodes.push(key_output);
    // No input nodes added!

    engine.add_logic_gate(empty_gate).await.unwrap();

    // Try to propagate with empty pattern
    let pattern = ActivationPattern::new("empty_gate".to_string());
    
    let result = engine.propagate_activation(&pattern).await.unwrap();
    
    // Should handle gracefully without crashing
    assert!(result.converged || result.iterations_completed > 0);
}

#[tokio::test]
async fn test_self_referential_inhibition() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create entity that inhibits itself
    let entity = BrainInspiredEntity::new("SelfInhibitor".to_string(), EntityDirection::Hidden);
    let key = entity.id;
    engine.add_entity(entity).await.unwrap();

    // Create self-inhibitory connection
    let mut self_inhibition = llmkg::core::brain_types::BrainInspiredRelationship::new(
        key, key, llmkg::core::brain_types::RelationType::RelatedTo
    );
    self_inhibition.is_inhibitory = true;
    self_inhibition.weight = 0.5;

    engine.add_relationship(self_inhibition).await.unwrap();

    // Start with high activation
    let mut pattern = ActivationPattern::new("self_inhibition".to_string());
    pattern.activations.insert(key, 0.8);

    let result = engine.propagate_activation(&pattern).await.unwrap();
    let final_activation = result.final_activations.get(&key).copied().unwrap_or(0.0);
    
    // Self-inhibition should reduce but not eliminate activation
    assert!(
        final_activation < 0.8 && final_activation > 0.0,
        "Self-inhibition should reduce activation from {} to {}", 0.8, final_activation
    );
}

#[tokio::test]
async fn test_gate_with_missing_entities() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create only input, not output
    let input = BrainInspiredEntity::new("Input".to_string(), EntityDirection::Input);
    let key_input = input.id;
    engine.add_entity(input).await.unwrap();

    // Create gate referencing non-existent output
    let fake_output_entity = BrainInspiredEntity::new("FakeOutput".to_string(), EntityDirection::Output);
    let fake_output_key = fake_output_entity.id;
    // Note: We intentionally don't add the fake output entity to the engine
    let mut gate = LogicGate::new(LogicGateType::Identity, 0.0);
    gate.input_nodes.push(key_input);
    gate.output_nodes.push(fake_output_key); // Non-existent entity!

    engine.add_logic_gate(gate).await.unwrap();

    // Try to propagate
    let mut pattern = ActivationPattern::new("missing_entity".to_string());
    pattern.activations.insert(key_input, 0.8);

    // Should handle gracefully
    let result = engine.propagate_activation(&pattern).await.unwrap();
    
    // The propagation should complete despite the missing entity
    assert!(result.iterations_completed > 0);
}