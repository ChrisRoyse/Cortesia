use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{BrainInspiredEntity, EntityDirection, ActivationPattern};

#[tokio::test]
async fn test_empty_initial_pattern_propagation() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create a network with entities
    let entity1 = BrainInspiredEntity::new("entity1".to_string(), EntityDirection::Input);
    let entity2 = BrainInspiredEntity::new("entity2".to_string(), EntityDirection::Hidden);
    let entity3 = BrainInspiredEntity::new("entity3".to_string(), EntityDirection::Output);

    engine.add_entity(entity1.clone()).await.unwrap();
    engine.add_entity(entity2.clone()).await.unwrap();
    engine.add_entity(entity3.clone()).await.unwrap();

    // Create empty activation pattern
    let empty_pattern = ActivationPattern::new("empty_test".to_string());
    
    // Propagate with empty pattern
    let result = engine.propagate_activation(&empty_pattern).await.unwrap();

    // Verify results
    assert_eq!(result.final_activations.len(), 0, "Empty pattern should result in no activations");
    assert!(result.converged, "Empty pattern should converge immediately");
    assert_eq!(result.iterations_completed, 0, "Should complete in 0 iterations for empty pattern");
    assert_eq!(result.total_energy, 0.0, "Total energy should be 0 for empty pattern");
}

#[tokio::test]
async fn test_empty_network_with_initial_pattern() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // No entities added to the network
    
    // Create activation pattern with non-existent entities
    let mut pattern = ActivationPattern::new("test_pattern".to_string());
    let non_existent_entity = BrainInspiredEntity::new("NonExistent".to_string(), EntityDirection::Hidden);
    let non_existent_key = non_existent_entity.id;
    // Note: We intentionally don't add this entity to the engine
    pattern.activations.insert(non_existent_key, 0.8);
    
    // Propagate in empty network
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify that activation persists but doesn't spread
    assert_eq!(result.final_activations.len(), 1, "Should maintain initial activation");
    assert!(result.converged, "Should converge with no network to propagate through");
}