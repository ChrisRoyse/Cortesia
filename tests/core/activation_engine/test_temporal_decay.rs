use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern,
    BrainInspiredRelationship, RelationType
};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::test]
async fn test_basic_temporal_decay() {
    let mut config = ActivationConfig::default();
    config.decay_rate = 0.5; // Moderate decay
    config.max_iterations = 10;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create simple entity
    let mut entity = BrainInspiredEntity::new("Decaying".to_string(), EntityDirection::Hidden);
    // Set last activation to past time to simulate decay
    entity.last_activation = std::time::SystemTime::now() - Duration::from_secs(2);
    
    let key = entity.id;
    engine.add_entity(entity).await.unwrap();

    // Initial high activation
    let mut pattern = ActivationPattern::new("decay_test".to_string());
    pattern.activations.insert(key, 1.0);

    // Propagate and let decay work
    let result = engine.propagate_activation(&pattern).await.unwrap();

    let final_activation = result.final_activations.get(&key).copied().unwrap_or(0.0);
    
    // Activation should decay over time
    assert!(
        final_activation < 1.0,
        "Activation should decay from initial value"
    );
    assert!(
        final_activation > 0.0,
        "Activation should not decay to zero immediately"
    );
}

#[tokio::test]
async fn test_decay_in_network_propagation() {
    let mut config = ActivationConfig::default();
    config.decay_rate = 0.3;
    config.max_iterations = 20;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create chain: A -> B -> C
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();

    // Weak connections to emphasize decay effect
    let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel_ab.weight = 0.6;
    
    let mut rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
    rel_bc.weight = 0.6;

    engine.add_relationship(rel_ab).await.unwrap();
    engine.add_relationship(rel_bc).await.unwrap();

    // Initial activation
    let mut pattern = ActivationPattern::new("chain_decay_test".to_string());
    pattern.activations.insert(key_a, 0.8);

    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Each step should show decay effect
    let activation_a = result.final_activations.get(&key_a).copied().unwrap_or(0.0);
    let activation_b = result.final_activations.get(&key_b).copied().unwrap_or(0.0);
    let activation_c = result.final_activations.get(&key_c).copied().unwrap_or(0.0);

    assert!(activation_a > activation_b, "Decay should reduce activation along chain");
    assert!(activation_b > activation_c, "Further propagation should show more decay");
}

#[tokio::test]
async fn test_zero_decay_rate() {
    let mut config = ActivationConfig::default();
    config.decay_rate = 0.0; // No decay
    config.max_iterations = 5;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create entity with old activation time
    let mut entity = BrainInspiredEntity::new("NoDecay".to_string(), EntityDirection::Hidden);
    entity.last_activation = std::time::SystemTime::now() - Duration::from_secs(10);
    
    let key = entity.id;
    engine.add_entity(entity).await.unwrap();

    // Set activation
    let mut pattern = ActivationPattern::new("no_decay_test".to_string());
    pattern.activations.insert(key, 0.7);

    let result = engine.propagate_activation(&pattern).await.unwrap();

    // With zero decay rate, activation should remain unchanged
    assert_eq!(
        result.final_activations.get(&key).copied().unwrap_or(0.0),
        0.7,
        "Zero decay rate should preserve activation"
    );
}

#[tokio::test]
async fn test_high_decay_rate() {
    let mut config = ActivationConfig::default();
    config.decay_rate = 2.0; // Very high decay
    config.max_iterations = 10;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create network with delayed propagation
    let entity_a = BrainInspiredEntity::new("Source".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("Target".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();

    // Weak connection
    let mut rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel.weight = 0.5;

    engine.add_relationship(rel).await.unwrap();

    // Initial activation
    let mut pattern = ActivationPattern::new("high_decay_test".to_string());
    pattern.activations.insert(key_a, 1.0);

    let result = engine.propagate_activation(&pattern).await.unwrap();

    // With high decay, activations should be significantly reduced
    let activation_b = result.final_activations.get(&key_b).copied().unwrap_or(0.0);
    
    assert!(
        activation_b < 0.3,
        "High decay rate should significantly reduce propagated activation"
    );
}

#[tokio::test]
async fn test_decay_with_reinforcement() {
    let mut config = ActivationConfig::default();
    config.decay_rate = 0.4;
    config.max_iterations = 30;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create a loop that reinforces itself: A -> B -> A
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);

    let key_a = entity_a.id;
    let key_b = entity_b.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();

    // Create reinforcing loop with moderate weights
    let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel_ab.weight = 0.7;
    
    let mut rel_ba = BrainInspiredRelationship::new(key_b, key_a, RelationType::RelatedTo);
    rel_ba.weight = 0.7;

    engine.add_relationship(rel_ab).await.unwrap();
    engine.add_relationship(rel_ba).await.unwrap();

    // Initial activation
    let mut pattern = ActivationPattern::new("reinforcement_decay_test".to_string());
    pattern.activations.insert(key_a, 1.0);

    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Despite reinforcement, decay should eventually stabilize activations
    assert!(result.converged, "Decay should allow convergence even with reinforcement");
    
    let activation_a = result.final_activations.get(&key_a).copied().unwrap_or(0.0);
    let activation_b = result.final_activations.get(&key_b).copied().unwrap_or(0.0);
    
    // Both should have non-zero but reduced activations
    assert!(activation_a > 0.0 && activation_a < 0.8, "A should maintain reduced activation");
    assert!(activation_b > 0.0 && activation_b < 0.8, "B should maintain reduced activation");
}

#[tokio::test]
async fn test_differential_decay_rates() {
    let mut config = ActivationConfig::default();
    config.decay_rate = 0.5;
    config.max_iterations = 15;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create entities with different activation histories
    let mut entity_fresh = BrainInspiredEntity::new("Fresh".to_string(), EntityDirection::Hidden);
    entity_fresh.last_activation = std::time::SystemTime::now();
    
    let mut entity_old = BrainInspiredEntity::new("Old".to_string(), EntityDirection::Hidden);
    entity_old.last_activation = std::time::SystemTime::now() - Duration::from_secs(5);

    let key_fresh = entity_fresh.id;
    let key_old = entity_old.id;

    engine.add_entity(entity_fresh).await.unwrap();
    engine.add_entity(entity_old).await.unwrap();

    // Same initial activation
    let mut pattern = ActivationPattern::new("differential_decay_test".to_string());
    pattern.activations.insert(key_fresh, 0.8);
    pattern.activations.insert(key_old, 0.8);

    let result = engine.propagate_activation(&pattern).await.unwrap();

    let activation_fresh = result.final_activations.get(&key_fresh).copied().unwrap_or(0.0);
    let activation_old = result.final_activations.get(&key_old).copied().unwrap_or(0.0);

    // Older activation should decay more
    assert!(
        activation_fresh > activation_old,
        "Entity with older last_activation should decay more"
    );
}