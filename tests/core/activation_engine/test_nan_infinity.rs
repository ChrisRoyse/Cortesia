use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    BrainInspiredRelationship, RelationType, LogicGate, LogicGateType
};
use llmkg::core::types::EntityKey;
use std::collections::HashMap;

// ==================== NAN AND INFINITY EDGE CASE TESTS ====================

#[tokio::test]
async fn test_nan_propagation() {
    // Test that NaN inputs don't crash the system and are handled properly
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create simple chain: A -> B -> C
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();

    // Add relationships
    let rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    let rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);

    engine.add_relationship(rel_ab).await.unwrap();
    engine.add_relationship(rel_bc).await.unwrap();

    // Create activation pattern with NaN
    let mut pattern = ActivationPattern::new("test_nan".to_string());
    pattern.activations.insert(key_a, f32::NAN);

    // Test propagation - should not panic
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify NaN is handled (likely treated as 0.0 or filtered out)
    for (key, activation) in &result.final_activations {
        assert!(!activation.is_nan(), 
            "Entity {:?} should not have NaN activation, got {}", key, activation);
        assert!(activation.is_finite(), 
            "Entity {:?} should have finite activation, got {}", key, activation);
    }
}

#[tokio::test]
async fn test_infinity_propagation() {
    // Test that Infinity values are properly clamped
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create simple network
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();

    // Add relationship
    let rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    engine.add_relationship(rel).await.unwrap();

    // Create activation pattern with positive infinity
    let mut pattern = ActivationPattern::new("test_infinity".to_string());
    pattern.activations.insert(key_a, f32::INFINITY);

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify infinity is clamped to maximum allowed value (1.0)
    for (key, activation) in &result.final_activations {
        assert!(activation.is_finite(), 
            "Entity {:?} should have finite activation", key);
        assert!(*activation <= 1.0, 
            "Entity {:?} activation should be clamped to max 1.0, got {}", key, activation);
        assert!(*activation >= 0.0, 
            "Entity {:?} activation should be non-negative, got {}", key, activation);
    }
}

#[tokio::test]
async fn test_negative_infinity_propagation() {
    // Test that negative infinity values are handled properly
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create network
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();

    let rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    engine.add_relationship(rel).await.unwrap();

    // Create activation pattern with negative infinity
    let mut pattern = ActivationPattern::new("test_neg_infinity".to_string());
    pattern.activations.insert(key_a, f32::NEG_INFINITY);

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify negative infinity is handled (likely clamped to 0.0)
    for (key, activation) in &result.final_activations {
        assert!(activation.is_finite(), 
            "Entity {:?} should have finite activation", key);
        assert!(*activation >= 0.0, 
            "Entity {:?} activation should be non-negative, got {}", key, activation);
    }
}

#[tokio::test]
async fn test_mixed_nan_infinity_normal() {
    // Test network with mixed NaN, Infinity, and normal values
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create network with multiple inputs
    let entity_nan = BrainInspiredEntity::new("NaN_input".to_string(), EntityDirection::Input);
    let entity_inf = BrainInspiredEntity::new("Inf_input".to_string(), EntityDirection::Input);
    let entity_normal = BrainInspiredEntity::new("Normal_input".to_string(), EntityDirection::Input);
    let entity_output = BrainInspiredEntity::new("Output".to_string(), EntityDirection::Output);

    let key_nan = entity_nan.id;
    let key_inf = entity_inf.id;
    let key_normal = entity_normal.id;
    let key_output = entity_output.id;

    engine.add_entity(entity_nan).await.unwrap();
    engine.add_entity(entity_inf).await.unwrap();
    engine.add_entity(entity_normal).await.unwrap();
    engine.add_entity(entity_output).await.unwrap();

    // Connect all inputs to output
    engine.add_relationship(BrainInspiredRelationship::new(key_nan, key_output, RelationType::RelatedTo)).await.unwrap();
    engine.add_relationship(BrainInspiredRelationship::new(key_inf, key_output, RelationType::RelatedTo)).await.unwrap();
    engine.add_relationship(BrainInspiredRelationship::new(key_normal, key_output, RelationType::RelatedTo)).await.unwrap();

    // Create mixed activation pattern
    let mut pattern = ActivationPattern::new("test_mixed".to_string());
    pattern.activations.insert(key_nan, f32::NAN);
    pattern.activations.insert(key_inf, f32::INFINITY);
    pattern.activations.insert(key_normal, 0.5);

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify all values are finite and within bounds
    for (key, activation) in &result.final_activations {
        assert!(!activation.is_nan(), 
            "Entity {:?} should not have NaN activation", key);
        assert!(activation.is_finite(), 
            "Entity {:?} should have finite activation", key);
        assert!(*activation >= 0.0 && *activation <= 1.0, 
            "Entity {:?} activation should be in range [0, 1], got {}", key, activation);
    }

    // Verify output received some activation from normal input
    assert!(result.final_activations.get(&key_output).copied().unwrap_or(0.0) > 0.0,
        "Output should receive activation from normal input");
}

#[tokio::test]
async fn test_arithmetic_overflow_prevention() {
    // Test that large multiplications don't cause overflow
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create network with extreme weights
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();

    // Add relationships with extreme weights
    let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel_ab.weight = f32::MAX / 2.0; // Very large weight
    engine.add_relationship(rel_ab).await.unwrap();

    let mut rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
    rel_bc.weight = f32::MAX / 2.0; // Very large weight
    engine.add_relationship(rel_bc).await.unwrap();

    // High activation
    let mut pattern = ActivationPattern::new("test_overflow".to_string());
    pattern.activations.insert(key_a, 0.99);

    // Test propagation - should not overflow
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify all values are finite and clamped
    for (key, activation) in &result.final_activations {
        assert!(activation.is_finite(), 
            "Entity {:?} should have finite activation despite large weights", key);
        assert!(*activation <= 1.0, 
            "Entity {:?} activation should be clamped to max 1.0", key);
    }
}

#[tokio::test]
async fn test_division_by_zero_inhibition() {
    // Test that inhibitory connections handle division by zero scenarios
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create network with inhibitory connection
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();

    // Add inhibitory relationship
    let mut rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel.is_inhibitory = true;
    rel.weight = 0.0; // Zero weight that could cause issues
    engine.add_relationship(rel).await.unwrap();

    // Activate both entities
    let mut pattern = ActivationPattern::new("test_div_zero".to_string());
    pattern.activations.insert(key_a, 0.8);
    pattern.activations.insert(key_b, 0.5);

    // Test propagation - should not crash on division
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify all values are valid
    for (key, activation) in &result.final_activations {
        assert!(!activation.is_nan(), 
            "Entity {:?} should not have NaN from division", key);
        assert!(activation.is_finite(), 
            "Entity {:?} should have finite activation", key);
    }
}

#[tokio::test]
async fn test_nan_in_logic_gates() {
    // Test logic gates with NaN inputs
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create entities for AND gate
    let input1 = BrainInspiredEntity::new("input1".to_string(), EntityDirection::Input);
    let input2 = BrainInspiredEntity::new("input2".to_string(), EntityDirection::Input);
    let gate_entity = BrainInspiredEntity::new("and_gate".to_string(), EntityDirection::Gate);
    let output = BrainInspiredEntity::new("output".to_string(), EntityDirection::Output);

    let key_input1 = input1.id;
    let key_input2 = input2.id;
    let key_gate = gate_entity.id;
    let key_output = output.id;

    engine.add_entity(input1).await.unwrap();
    engine.add_entity(input2).await.unwrap();
    engine.add_entity(gate_entity).await.unwrap();
    engine.add_entity(output).await.unwrap();

    // Create AND gate
    let mut and_gate = LogicGate::new(LogicGateType::And, 0.5);
    and_gate.gate_id = key_gate;
    and_gate.input_nodes.push(key_input1);
    and_gate.input_nodes.push(key_input2);
    and_gate.output_nodes.push(key_output);
    engine.add_logic_gate(and_gate).await.unwrap();

    // Test with NaN input
    let mut pattern = ActivationPattern::new("test_gate_nan".to_string());
    pattern.activations.insert(key_input1, f32::NAN);
    pattern.activations.insert(key_input2, 0.7);

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify gate and output handle NaN properly
    assert!(!result.final_activations[&key_gate].is_nan(), 
        "Gate should not output NaN");
    assert!(!result.final_activations[&key_output].is_nan(), 
        "Output should not receive NaN");
}

#[tokio::test]
async fn test_extreme_inhibition_values() {
    // Test inhibitory connections with extreme values
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create network
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();

    // Add extremely strong inhibitory connection
    let mut rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel.is_inhibitory = true;
    rel.weight = f32::INFINITY; // Infinite inhibition
    engine.add_relationship(rel).await.unwrap();

    // Activate both entities
    let mut pattern = ActivationPattern::new("test_extreme_inhibition".to_string());
    pattern.activations.insert(key_a, 1.0);
    pattern.activations.insert(key_b, 1.0);

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify B is inhibited but not to NaN or negative
    let b_activation = result.final_activations[&key_b];
    assert!(!b_activation.is_nan(), "Inhibited value should not be NaN");
    assert!(b_activation.is_finite(), "Inhibited value should be finite");
    assert!(b_activation >= 0.0, "Inhibited value should not be negative");
    assert!(b_activation < 1.0, "B should be inhibited from its initial value");
}

#[tokio::test]
async fn test_negative_activation_values() {
    // Test that negative activation values are handled properly
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create simple network
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();

    let rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    engine.add_relationship(rel).await.unwrap();

    // Try various negative values
    let negative_values = vec![-1.0, -0.5, -100.0, -f32::INFINITY];
    
    for neg_val in negative_values {
        let mut pattern = ActivationPattern::new(format!("test_neg_{}", neg_val));
        pattern.activations.insert(key_a, neg_val);

        let result = engine.propagate_activation(&pattern).await.unwrap();

        // Verify all activations are non-negative
        for (key, activation) in &result.final_activations {
            assert!(*activation >= 0.0, 
                "Entity {:?} should have non-negative activation when input was {}, got {}", 
                key, neg_val, activation);
            assert!(activation.is_finite(), 
                "Entity {:?} should have finite activation", key);
        }
    }
}

#[tokio::test]
async fn test_nan_in_inhibitory_chain() {
    // Test NaN propagation through inhibitory connections
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create chain with inhibitory connections
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();

    // A activates B, B inhibits C
    let rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    engine.add_relationship(rel_ab).await.unwrap();

    let mut rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
    rel_bc.is_inhibitory = true;
    engine.add_relationship(rel_bc).await.unwrap();

    // Start with NaN in A
    let mut pattern = ActivationPattern::new("test_nan_inhibitory".to_string());
    pattern.activations.insert(key_a, f32::NAN);
    pattern.activations.insert(key_c, 0.8); // C starts activated

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify NaN doesn't propagate through inhibition
    for (key, activation) in &result.final_activations {
        assert!(!activation.is_nan(), 
            "Entity {:?} should not have NaN activation", key);
        assert!(activation.is_finite(), 
            "Entity {:?} should have finite activation", key);
    }
}

#[tokio::test]
async fn test_zero_activation_stability() {
    // Test that zero activations remain stable and don't become NaN
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create network
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();

    // Add relationships with zero weights
    let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel_ab.weight = 0.0;
    engine.add_relationship(rel_ab).await.unwrap();

    let mut rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
    rel_bc.weight = 0.0;
    engine.add_relationship(rel_bc).await.unwrap();

    // All zero activations
    let mut pattern = ActivationPattern::new("test_zero_stability".to_string());
    pattern.activations.insert(key_a, 0.0);
    pattern.activations.insert(key_b, 0.0);
    pattern.activations.insert(key_c, 0.0);

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify zeros remain zeros and don't become NaN
    for (key, activation) in &result.final_activations {
        assert_eq!(*activation, 0.0, 
            "Entity {:?} should remain at zero activation", key);
        assert!(!activation.is_nan(), 
            "Entity {:?} should not have NaN", key);
    }
}

#[tokio::test]
async fn test_subnormal_values() {
    // Test handling of subnormal (denormalized) floating point values
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create simple network
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();

    let rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    engine.add_relationship(rel).await.unwrap();

    // Use very small subnormal value
    let subnormal = f32::MIN_POSITIVE / 2.0;
    let mut pattern = ActivationPattern::new("test_subnormal".to_string());
    pattern.activations.insert(key_a, subnormal);

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify subnormal values are handled
    for (key, activation) in &result.final_activations {
        assert!(activation.is_finite(), 
            "Entity {:?} should have finite activation with subnormal input", key);
        assert!(!activation.is_nan(), 
            "Entity {:?} should not have NaN with subnormal input", key);
    }
}