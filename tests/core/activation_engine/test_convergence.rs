use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    BrainInspiredRelationship, RelationType
};
use llmkg::core::types::EntityKey;

#[tokio::test]
async fn test_simple_linear_network_convergence() {
    let mut config = ActivationConfig::default();
    config.convergence_threshold = 0.01;
    config.max_iterations = 50;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create a simple linear network: A -> B -> C
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();

    // Create relationships
    let rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    let rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);

    engine.add_relationship(rel_ab).await.unwrap();
    engine.add_relationship(rel_bc).await.unwrap();

    // Create initial activation pattern
    let mut pattern = ActivationPattern::new("linear_test".to_string());
    pattern.activations.insert(key_a, 1.0);

    // Propagate activation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify convergence
    assert!(result.converged, "Simple linear network should converge");
    assert!(result.iterations_completed < 50, "Should converge before max iterations");
    
    // Verify activation propagated through the network
    let activation_b = result.final_activations.get(&key_b).copied().unwrap_or(0.0);
    let activation_c = result.final_activations.get(&key_c).copied().unwrap_or(0.0);
    
    assert!(activation_b > 0.0, "Entity B should have received activation");
    assert!(activation_c > 0.0, "Entity C should have received activation");
    assert!(activation_b >= activation_c, "Activation should decrease along the chain");
}

#[tokio::test]
async fn test_divergent_network_convergence() {
    let mut config = ActivationConfig::default();
    config.convergence_threshold = 0.01;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create a divergent network: A -> {B, C, D}
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Output);
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

    // Create relationships with different weights
    let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel_ab.weight = 0.8;
    
    let mut rel_ac = BrainInspiredRelationship::new(key_a, key_c, RelationType::RelatedTo);
    rel_ac.weight = 0.6;
    
    let mut rel_ad = BrainInspiredRelationship::new(key_a, key_d, RelationType::RelatedTo);
    rel_ad.weight = 0.4;

    engine.add_relationship(rel_ab).await.unwrap();
    engine.add_relationship(rel_ac).await.unwrap();
    engine.add_relationship(rel_ad).await.unwrap();

    // Create initial activation
    let mut pattern = ActivationPattern::new("divergent_test".to_string());
    pattern.activations.insert(key_a, 1.0);

    // Debug: Check entities and relationships before propagation
    let stats = engine.get_activation_statistics().await.unwrap();
    println!("Total entities: {}", stats.total_entities);
    println!("Total relationships: {}", stats.total_relationships);
    
    // Propagate activation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify convergence
    assert!(result.converged, "Divergent network should converge");
    
    // Verify activations reflect weights
    let activation_b = result.final_activations.get(&key_b).copied().unwrap_or(0.0);
    let activation_c = result.final_activations.get(&key_c).copied().unwrap_or(0.0);
    let activation_d = result.final_activations.get(&key_d).copied().unwrap_or(0.0);
    
    println!("Activation A: {}", result.final_activations.get(&key_a).copied().unwrap_or(0.0));
    println!("Activation B: {}", activation_b);
    println!("Activation C: {}", activation_c);
    println!("Activation D: {}", activation_d);
    println!("All activations: {:?}", result.final_activations);
    
    assert!(activation_b > activation_c, "B should have higher activation due to higher weight");
    assert!(activation_c > activation_d, "C should have higher activation than D");
    assert!(activation_d > 0.0, "All outputs should receive some activation");
}

#[tokio::test]
async fn test_convergence_with_very_small_changes() {
    let mut config = ActivationConfig::default();
    config.convergence_threshold = 0.1; // Large threshold
    config.max_iterations = 10;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create simple network
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();

    // Weak relationship
    let mut rel = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel.weight = 0.1; // Very weak connection

    engine.add_relationship(rel).await.unwrap();

    // Small initial activation
    let mut pattern = ActivationPattern::new("small_change_test".to_string());
    pattern.activations.insert(key_a, 0.2);

    // Propagate
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Should converge quickly with large threshold
    assert!(result.converged, "Should converge with large threshold");
    assert!(result.iterations_completed <= 3, "Should converge very quickly");
}