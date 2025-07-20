use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    BrainInspiredRelationship, RelationType
};
use llmkg::core::types::EntityKey;

#[tokio::test]
async fn test_fully_disconnected_network() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create multiple isolated entities with no connections
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Output);
    let entity_d = BrainInspiredEntity::new("D".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;
    let key_d = entity_d.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();
    engine.add_entity(entity_d).await.unwrap();

    // No relationships added - completely disconnected

    // Activate multiple entities
    let mut pattern = ActivationPattern::new("disconnected_test".to_string());
    pattern.activations.insert(key_a, 1.0);
    pattern.activations.insert(key_b, 0.5);

    // Propagate activation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify no spread of activation
    assert!(result.converged, "Disconnected network should converge immediately");
    
    // Only initially activated entities should have activation
    assert_eq!(
        result.final_activations.get(&key_a).copied().unwrap_or(0.0),
        1.0,
        "Entity A should maintain its initial activation"
    );
    assert_eq!(
        result.final_activations.get(&key_b).copied().unwrap_or(0.0),
        0.5,
        "Entity B should maintain its initial activation"
    );
    assert_eq!(
        result.final_activations.get(&key_c).copied().unwrap_or(0.0),
        0.0,
        "Entity C should have no activation"
    );
    assert_eq!(
        result.final_activations.get(&key_d).copied().unwrap_or(0.0),
        0.0,
        "Entity D should have no activation"
    );
}

#[tokio::test]
async fn test_partially_connected_network_islands() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create two separate islands: (A -> B) and (C -> D)
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Output);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Input);
    let entity_d = BrainInspiredEntity::new("D".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;
    let key_d = entity_d.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();
    engine.add_entity(entity_d).await.unwrap();

    // Connect A to B
    let rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    engine.add_relationship(rel_ab).await.unwrap();

    // Connect C to D
    let rel_cd = BrainInspiredRelationship::new(key_c, key_d, RelationType::RelatedTo);
    engine.add_relationship(rel_cd).await.unwrap();

    // Activate only A
    let mut pattern = ActivationPattern::new("island_test".to_string());
    pattern.activations.insert(key_a, 0.8);

    // Propagate activation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify activation only spreads within the island
    let activation_a = result.final_activations.get(&key_a).copied().unwrap_or(0.0);
    let activation_b = result.final_activations.get(&key_b).copied().unwrap_or(0.0);
    let activation_c = result.final_activations.get(&key_c).copied().unwrap_or(0.0);
    let activation_d = result.final_activations.get(&key_d).copied().unwrap_or(0.0);

    assert!(activation_a > 0.0, "Entity A should maintain activation");
    assert!(activation_b > 0.0, "Entity B should receive activation from A");
    assert_eq!(activation_c, 0.0, "Entity C should have no activation");
    assert_eq!(activation_d, 0.0, "Entity D should have no activation");
}

#[tokio::test]
async fn test_single_entity_network() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create a single entity with no connections
    let entity = BrainInspiredEntity::new("Solo".to_string(), EntityDirection::Hidden);
    let key = entity.id;

    engine.add_entity(entity).await.unwrap();

    // Activate the entity
    let mut pattern = ActivationPattern::new("single_entity_test".to_string());
    pattern.activations.insert(key, 0.7);

    // Propagate activation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify the entity maintains its activation
    assert!(result.converged, "Single entity should converge immediately");
    assert_eq!(
        result.final_activations.get(&key).copied().unwrap_or(0.0),
        0.7,
        "Single entity should maintain its initial activation"
    );
    assert_eq!(result.final_activations.len(), 1, "Only one activation should exist");
}

#[tokio::test]
async fn test_self_referential_entity() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create an entity that references itself
    let entity = BrainInspiredEntity::new("Self".to_string(), EntityDirection::Hidden);
    let key = entity.id;

    engine.add_entity(entity).await.unwrap();

    // Create self-referential relationship
    let mut self_rel = BrainInspiredRelationship::new(key, key, RelationType::RelatedTo);
    self_rel.weight = 0.5;

    engine.add_relationship(self_rel).await.unwrap();

    // Activate the entity
    let mut pattern = ActivationPattern::new("self_reference_test".to_string());
    pattern.activations.insert(key, 0.6);

    // Propagate activation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify the behavior with self-reference
    assert!(result.converged, "Self-referential network should converge");
    let final_activation = result.final_activations.get(&key).copied().unwrap_or(0.0);
    assert!(
        final_activation >= 0.6,
        "Self-referential entity should maintain or increase activation"
    );
}