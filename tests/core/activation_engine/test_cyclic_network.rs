use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    BrainInspiredRelationship, RelationType
};
use llmkg::core::types::EntityKey;

#[tokio::test]
async fn test_simple_cycle_convergence() {
    let mut config = ActivationConfig::default();
    config.convergence_threshold = 0.01;
    config.max_iterations = 100;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create a simple cycle: A -> B -> C -> A
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Hidden);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();

    // Create cycle with decreasing weights to ensure convergence
    let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel_ab.weight = 0.8;
    
    let mut rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
    rel_bc.weight = 0.7;
    
    let mut rel_ca = BrainInspiredRelationship::new(key_c, key_a, RelationType::RelatedTo);
    rel_ca.weight = 0.6;

    engine.add_relationship(rel_ab).await.unwrap();
    engine.add_relationship(rel_bc).await.unwrap();
    engine.add_relationship(rel_ca).await.unwrap();

    // Initial activation
    let mut pattern = ActivationPattern::new("cycle_test".to_string());
    pattern.activations.insert(key_a, 1.0);

    // Propagate activation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify convergence despite cycle
    assert!(result.converged, "Cyclic network should eventually converge");
    assert!(result.iterations_completed > 1, "Should take multiple iterations due to cycle");
    
    // All entities should have non-zero activation due to cycle
    let activation_a = result.final_activations.get(&key_a).copied().unwrap_or(0.0);
    let activation_b = result.final_activations.get(&key_b).copied().unwrap_or(0.0);
    let activation_c = result.final_activations.get(&key_c).copied().unwrap_or(0.0);
    
    assert!(activation_a > 0.0, "Entity A should maintain activation");
    assert!(activation_b > 0.0, "Entity B should have activation from cycle");
    assert!(activation_c > 0.0, "Entity C should have activation from cycle");
}

#[tokio::test]
async fn test_multiple_cycles_network() {
    let mut config = ActivationConfig::default();
    config.convergence_threshold = 0.01;
    config.max_iterations = 150;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create network with multiple cycles: 
    // A -> B -> C -> A (cycle 1)
    // B -> D -> C (cycle 2 via B and C)
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Hidden);
    let entity_d = BrainInspiredEntity::new("D".to_string(), EntityDirection::Hidden);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;
    let key_d = entity_d.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();
    engine.add_entity(entity_d).await.unwrap();

    // First cycle
    let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel_ab.weight = 0.7;
    
    let mut rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
    rel_bc.weight = 0.6;
    
    let mut rel_ca = BrainInspiredRelationship::new(key_c, key_a, RelationType::RelatedTo);
    rel_ca.weight = 0.5;

    // Second path through D
    let mut rel_bd = BrainInspiredRelationship::new(key_b, key_d, RelationType::RelatedTo);
    rel_bd.weight = 0.8;
    
    let mut rel_dc = BrainInspiredRelationship::new(key_d, key_c, RelationType::RelatedTo);
    rel_dc.weight = 0.7;

    engine.add_relationship(rel_ab).await.unwrap();
    engine.add_relationship(rel_bc).await.unwrap();
    engine.add_relationship(rel_ca).await.unwrap();
    engine.add_relationship(rel_bd).await.unwrap();
    engine.add_relationship(rel_dc).await.unwrap();

    // Initial activation
    let mut pattern = ActivationPattern::new("multi_cycle_test".to_string());
    pattern.activations.insert(key_a, 0.9);

    // Propagate activation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify convergence
    assert!(result.converged, "Multiple cycle network should converge");
    
    // All entities should be activated
    for key in [key_a, key_b, key_c, key_d] {
        let activation = result.final_activations.get(&key).copied().unwrap_or(0.0);
        assert!(activation > 0.0, "All entities in cycles should have activation");
    }
}

#[tokio::test]
async fn test_damped_oscillation_in_cycle() {
    let mut config = ActivationConfig::default();
    config.convergence_threshold = 0.005;
    config.max_iterations = 200;
    config.decay_rate = 0.1; // Add decay to dampen oscillations
    
    let engine = ActivationPropagationEngine::new(config);

    // Create a two-node oscillator: A <-> B
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);

    let key_a = entity_a.id;
    let key_b = entity_b.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();

    // Bidirectional connections with high weights
    let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel_ab.weight = 0.9;
    
    let mut rel_ba = BrainInspiredRelationship::new(key_b, key_a, RelationType::RelatedTo);
    rel_ba.weight = 0.9;

    engine.add_relationship(rel_ab).await.unwrap();
    engine.add_relationship(rel_ba).await.unwrap();

    // Start with asymmetric activation
    let mut pattern = ActivationPattern::new("oscillator_test".to_string());
    pattern.activations.insert(key_a, 1.0);
    pattern.activations.insert(key_b, 0.0);

    // Propagate activation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Should converge due to decay
    assert!(result.converged, "Oscillator should converge with decay");
    
    // Both should have similar final activations due to oscillation
    let activation_a = result.final_activations.get(&key_a).copied().unwrap_or(0.0);
    let activation_b = result.final_activations.get(&key_b).copied().unwrap_or(0.0);
    
    assert!((activation_a - activation_b).abs() < 0.1, 
        "Oscillator should reach near-equilibrium");
}

#[tokio::test] 
async fn test_cycle_with_external_input() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create cycle with external input: X -> A -> B -> C -> A
    let entity_x = BrainInspiredEntity::new("X".to_string(), EntityDirection::Input);
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Output);

    let key_x = entity_x.id;
    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;

    engine.add_entity(entity_x).await.unwrap();
    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();

    // External input
    let mut rel_xa = BrainInspiredRelationship::new(key_x, key_a, RelationType::RelatedTo);
    rel_xa.weight = 1.0;

    // Cycle
    let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel_ab.weight = 0.6;
    
    let mut rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
    rel_bc.weight = 0.5;
    
    let mut rel_ca = BrainInspiredRelationship::new(key_c, key_a, RelationType::RelatedTo);
    rel_ca.weight = 0.4;

    engine.add_relationship(rel_xa).await.unwrap();
    engine.add_relationship(rel_ab).await.unwrap();
    engine.add_relationship(rel_bc).await.unwrap();
    engine.add_relationship(rel_ca).await.unwrap();

    // Activate external input
    let mut pattern = ActivationPattern::new("external_cycle_test".to_string());
    pattern.activations.insert(key_x, 0.8);

    // Propagate
    let result = engine.propagate_activation(&pattern).await.unwrap();

    assert!(result.converged, "Cycle with external input should converge");
    
    // Verify external input maintains highest activation
    let activation_x = result.final_activations.get(&key_x).copied().unwrap_or(0.0);
    let activation_a = result.final_activations.get(&key_a).copied().unwrap_or(0.0);
    
    assert_eq!(activation_x, 0.8, "External input should maintain activation");
    assert!(activation_a > 0.0, "Cycle entry point should be activated");
}