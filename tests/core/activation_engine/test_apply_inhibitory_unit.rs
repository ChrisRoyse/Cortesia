use llmkg::core::activation_config::ActivationConfig;
use llmkg::core::activation_processors::ActivationProcessors;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, BrainInspiredRelationship, 
    RelationType
};
use llmkg::core::types::EntityKey;
use std::collections::HashMap;
use ahash::AHashMap;

// Helper function to create test entities and get their keys
fn create_test_entities() -> (EntityKey, EntityKey, EntityKey, EntityKey) {
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Hidden);
    let entity_d = BrainInspiredEntity::new("D".to_string(), EntityDirection::Hidden);
    
    (entity_a.id, entity_b.id, entity_c.id, entity_d.id)
}

// ==================== UNIT TESTS FOR apply_inhibitory_connections ====================

#[tokio::test]
async fn test_apply_inhibitory_connections_happy_path() {
    // Test normal inhibitory connection application
    let config = ActivationConfig::default();
    let processors = ActivationProcessors::new(config);

    // Create test data
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    
    let key_a = entity_a.id;
    let key_b = entity_b.id;

    // Create inhibitory relationship
    let mut rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel.is_inhibitory = true;
    rel.weight = 0.5;

    let mut relationships = AHashMap::new();
    relationships.insert((key_a, key_b), rel);

    // Set up activations
    let mut activations = HashMap::new();
    activations.insert(key_a, 0.8); // Inhibitor active
    activations.insert(key_b, 0.6); // Target has initial activation

    let mut trace = Vec::new();

    // Apply inhibitory connections
    processors.apply_inhibitory_connections(&mut activations, &relationships, &mut trace, 0).await.unwrap();

    // Verify inhibition was applied
    assert!(activations[&key_b] < 0.6, "Target activation should be reduced");
    assert!(activations[&key_b] > 0.0, "Target should not be completely suppressed");
    assert_eq!(activations[&key_a], 0.8, "Inhibitor activation should remain unchanged");
    assert!(!trace.is_empty(), "Should have trace entries for inhibition");
}

#[tokio::test]
async fn test_apply_inhibitory_connections_zero_weight() {
    // Test with zero weight inhibitory connection
    let config = ActivationConfig::default();
    let processors = ActivationProcessors::new(config);

    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let key_a = entity_a.id;
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let key_b = entity_b.id;

    // Create zero-weight inhibitory relationship
    let mut rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel.is_inhibitory = true;
    rel.weight = 0.0; // Zero weight

    let mut relationships = AHashMap::new();
    relationships.insert((key_a, key_b), rel);

    // Set up activations
    let mut activations = HashMap::new();
    activations.insert(key_a, 1.0);
    activations.insert(key_b, 0.5);

    let initial_b = activations[&key_b];
    let mut trace = Vec::new();

    // Apply inhibitory connections
    processors.apply_inhibitory_connections(&mut activations, &relationships, &mut trace, 0).await.unwrap();

    // Zero weight should have no effect
    assert_eq!(activations[&key_b], initial_b, "Zero weight should not change activation");
}

#[tokio::test]
async fn test_apply_inhibitory_connections_negative_weight() {
    // Test with negative weight (should be treated as positive for inhibition)
    let config = ActivationConfig::default();
    let processors = ActivationProcessors::new(config);

    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let key_a = entity_a.id;
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let key_b = entity_b.id;

    // Create negative-weight inhibitory relationship
    let mut rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel.is_inhibitory = true;
    rel.weight = -0.5; // Negative weight

    let mut relationships = AHashMap::new();
    relationships.insert((key_a, key_b), rel);

    // Set up activations
    let mut activations = HashMap::new();
    activations.insert(key_a, 0.8);
    activations.insert(key_b, 0.6);

    let mut trace = Vec::new();

    // Apply inhibitory connections - should handle gracefully
    let result = processors.apply_inhibitory_connections(&mut activations, &relationships, &mut trace, 0).await;
    
    // Should not panic with negative weight
    assert!(result.is_ok(), "Should handle negative weight gracefully");
}

#[tokio::test]
async fn test_apply_inhibitory_connections_zero_source_activation() {
    // Test when inhibitor has zero activation
    let config = ActivationConfig::default();
    let processors = ActivationProcessors::new(config);

    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let key_a = entity_a.id;
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let key_b = entity_b.id;

    let mut rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel.is_inhibitory = true;
    rel.weight = 0.8;

    let mut relationships = AHashMap::new();
    relationships.insert((key_a, key_b), rel);

    // Set up activations
    let mut activations = HashMap::new();
    activations.insert(key_a, 0.0); // Inhibitor is inactive
    activations.insert(key_b, 0.7);

    let initial_b = activations[&key_b];
    let mut trace = Vec::new();

    // Apply inhibitory connections
    processors.apply_inhibitory_connections(&mut activations, &relationships, &mut trace, 0).await.unwrap();

    // No inhibition should occur
    assert_eq!(activations[&key_b], initial_b, "Inactive inhibitor should not affect target");
}

#[tokio::test]
async fn test_apply_inhibitory_connections_zero_target_activation() {
    // Test when target has zero activation
    let config = ActivationConfig::default();
    let processors = ActivationProcessors::new(config);

    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let key_a = entity_a.id;
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let key_b = entity_b.id;

    let mut rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel.is_inhibitory = true;
    rel.weight = 0.8;

    let mut relationships = AHashMap::new();
    relationships.insert((key_a, key_b), rel);

    // Set up activations
    let mut activations = HashMap::new();
    activations.insert(key_a, 0.9);
    activations.insert(key_b, 0.0); // Target already at zero

    let mut trace = Vec::new();

    // Apply inhibitory connections
    processors.apply_inhibitory_connections(&mut activations, &relationships, &mut trace, 0).await.unwrap();

    // Target should remain at zero
    assert_eq!(activations[&key_b], 0.0, "Zero activation should remain zero");
}

#[tokio::test]
async fn test_apply_inhibitory_connections_multiple_inhibitors() {
    // Test cumulative inhibition from multiple sources
    let mut config = ActivationConfig::default();
    config.inhibition_strength = 1.0;
    let processors = ActivationProcessors::new(config);

    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let key_a = entity_a.id;
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let key_b = entity_b.id;
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Hidden);
    let key_c = entity_c.id;
    let entity_target = BrainInspiredEntity::new("Target".to_string(), EntityDirection::Hidden);
    let key_target = entity_target.id;

    // Multiple inhibitors targeting same node
    let mut rel1 = BrainInspiredRelationship::new(key_a, key_target, RelationType::RelatedTo);
    rel1.is_inhibitory = true;
    rel1.weight = 0.5;

    let mut rel2 = BrainInspiredRelationship::new(key_b, key_target, RelationType::RelatedTo);
    rel2.is_inhibitory = true;
    rel2.weight = 0.5;

    let mut rel3 = BrainInspiredRelationship::new(key_c, key_target, RelationType::RelatedTo);
    rel3.is_inhibitory = true;
    rel3.weight = 0.5;

    let mut relationships = AHashMap::new();
    relationships.insert((key_a, key_target), rel1);
    relationships.insert((key_b, key_target), rel2);
    relationships.insert((key_c, key_target), rel3);

    // Set up activations
    let mut activations = HashMap::new();
    activations.insert(key_a, 0.6);
    activations.insert(key_b, 0.6);
    activations.insert(key_c, 0.6);
    activations.insert(key_target, 0.9);

    let mut trace = Vec::new();

    // Apply inhibitory connections
    processors.apply_inhibitory_connections(&mut activations, &relationships, &mut trace, 0).await.unwrap();

    // Cumulative inhibition should significantly reduce target
    assert!(activations[&key_target] < 0.5, "Multiple inhibitors should have cumulative effect");
    assert!(activations[&key_target] > 0.0, "Should not completely suppress (divisive inhibition)");
}

#[tokio::test]
async fn test_apply_inhibitory_connections_chain_effect() {
    // Test inhibitory chain: A inhibits B, B inhibits C
    let config = ActivationConfig::default();
    let processors = ActivationProcessors::new(config);

    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let key_a = entity_a.id;
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let key_b = entity_b.id;
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Hidden);
    let key_c = entity_c.id;

    // A inhibits B
    let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel_ab.is_inhibitory = true;
    rel_ab.weight = 0.8;

    // B inhibits C
    let mut rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
    rel_bc.is_inhibitory = true;
    rel_bc.weight = 0.8;

    let mut relationships = AHashMap::new();
    relationships.insert((key_a, key_b), rel_ab);
    relationships.insert((key_b, key_c), rel_bc);

    // Set up activations
    let mut activations = HashMap::new();
    activations.insert(key_a, 0.9);
    activations.insert(key_b, 0.8);
    activations.insert(key_c, 0.7);

    let mut trace = Vec::new();

    // Apply inhibitory connections (includes 2 passes for chain effects)
    processors.apply_inhibitory_connections(&mut activations, &relationships, &mut trace, 0).await.unwrap();

    // B should be inhibited by A
    assert!(activations[&key_b] < 0.8, "B should be inhibited by A");
    
    // C's inhibition should be affected by B's reduced activation
    let final_c = activations[&key_c];
    assert!(final_c < 0.7, "C should be inhibited");
    assert!(final_c > activations[&key_b] * 0.5, "C inhibition should be less due to B's reduced state");
}

#[tokio::test]
async fn test_apply_inhibitory_connections_extreme_inhibition() {
    // Test with very high inhibition strength
    let mut config = ActivationConfig::default();
    config.inhibition_strength = 10.0; // Very strong inhibition
    let processors = ActivationProcessors::new(config);

    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let key_a = entity_a.id;
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let key_b = entity_b.id;

    let mut rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel.is_inhibitory = true;
    rel.weight = 1.0;

    let mut relationships = AHashMap::new();
    relationships.insert((key_a, key_b), rel);

    // Set up activations
    let mut activations = HashMap::new();
    activations.insert(key_a, 1.0);
    activations.insert(key_b, 1.0);

    let mut trace = Vec::new();

    // Apply inhibitory connections
    processors.apply_inhibitory_connections(&mut activations, &relationships, &mut trace, 0).await.unwrap();

    // Even with extreme inhibition, divisive model prevents complete suppression
    assert!(activations[&key_b] > 0.0, "Divisive inhibition should prevent complete suppression");
    assert!(activations[&key_b] < 0.1, "Strong inhibition should greatly reduce activation");
}

#[tokio::test]
async fn test_apply_inhibitory_connections_no_relationships() {
    // Test with empty relationships map
    let config = ActivationConfig::default();
    let processors = ActivationProcessors::new(config);

    let relationships = AHashMap::new(); // Empty

    // Set up activations
    let mut activations = HashMap::new();
    let entity_1 = BrainInspiredEntity::new("Entity1".to_string(), EntityDirection::Hidden);
    let key_1 = entity_1.id;
    let entity_2 = BrainInspiredEntity::new("Entity2".to_string(), EntityDirection::Hidden);
    let key_2 = entity_2.id;
    activations.insert(key_1, 0.5);
    activations.insert(key_2, 0.7);

    let initial_activations = activations.clone();
    let mut trace = Vec::new();

    // Apply inhibitory connections
    processors.apply_inhibitory_connections(&mut activations, &relationships, &mut trace, 0).await.unwrap();

    // No changes should occur
    assert_eq!(activations, initial_activations, "No relationships should mean no changes");
    assert!(trace.is_empty(), "No trace entries without inhibitory connections");
}

#[tokio::test]
async fn test_apply_inhibitory_connections_self_inhibition() {
    // Test self-inhibition (node inhibiting itself)
    let config = ActivationConfig::default();
    let processors = ActivationProcessors::new(config);

    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let key_a = entity_a.id;

    // Self-inhibitory connection
    let mut rel = BrainInspiredRelationship::new(key_a, key_a, RelationType::RelatedTo);
    rel.is_inhibitory = true;
    rel.weight = 0.5;

    let mut relationships = AHashMap::new();
    relationships.insert((key_a, key_a), rel);

    // Set up activations
    let mut activations = HashMap::new();
    activations.insert(key_a, 0.8);

    let mut trace = Vec::new();

    // Apply inhibitory connections
    processors.apply_inhibitory_connections(&mut activations, &relationships, &mut trace, 0).await.unwrap();

    // Self-inhibition should reduce own activation
    assert!(activations[&key_a] < 0.8, "Self-inhibition should reduce activation");
    assert!(activations[&key_a] > 0.0, "Should not completely self-suppress");
}

#[tokio::test]
async fn test_apply_inhibitory_connections_missing_source() {
    // Test when inhibitory source is not in activations map
    let config = ActivationConfig::default();
    let processors = ActivationProcessors::new(config);

    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let key_a = entity_a.id;
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let key_b = entity_b.id;

    let mut rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel.is_inhibitory = true;
    rel.weight = 0.8;

    let mut relationships = AHashMap::new();
    relationships.insert((key_a, key_b), rel);

    // Set up activations - source not included
    let mut activations = HashMap::new();
    activations.insert(key_b, 0.7); // Only target

    let initial_b = activations[&key_b];
    let mut trace = Vec::new();

    // Apply inhibitory connections
    processors.apply_inhibitory_connections(&mut activations, &relationships, &mut trace, 0).await.unwrap();

    // No inhibition should occur without source
    assert_eq!(activations[&key_b], initial_b, "Missing source should not affect target");
}

#[tokio::test]
async fn test_apply_inhibitory_connections_missing_target() {
    // Test when inhibitory target is not in activations map
    let config = ActivationConfig::default();
    let processors = ActivationProcessors::new(config);

    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let key_a = entity_a.id;
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let key_b = entity_b.id;

    let mut rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel.is_inhibitory = true;
    rel.weight = 0.8;

    let mut relationships = AHashMap::new();
    relationships.insert((key_a, key_b), rel);

    // Set up activations - target not included
    let mut activations = HashMap::new();
    activations.insert(key_a, 0.9); // Only source

    let mut trace = Vec::new();

    // Apply inhibitory connections - should handle gracefully
    let result = processors.apply_inhibitory_connections(&mut activations, &relationships, &mut trace, 0).await;
    
    assert!(result.is_ok(), "Should handle missing target gracefully");
    assert!(!activations.contains_key(&key_b), "Should not create missing target");
}