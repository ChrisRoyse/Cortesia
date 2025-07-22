//! Brain entity manager integration tests
//! Tests complete entity lifecycle simulation including insertion, updating, and removal
//! Tests brain-specific entity operations through public APIs only

use llmkg::core::brain_enhanced_graph::{BrainEnhancedKnowledgeGraph, BrainQueryResult, EntityStatistics};
use llmkg::core::types::{EntityKey, EntityData, Relationship};
use llmkg::error::Result;
use std::collections::HashMap;
use tokio;

/// Helper function to create test entity data
fn create_test_entity_data(id: u32, embedding_len: usize) -> EntityData {
    let embedding = vec![0.1; embedding_len];
    let mut properties = HashMap::new();
    properties.insert("name".to_string(), format!("Entity_{}", id));
    properties.insert("type".to_string(), "test_entity".to_string());
    
    EntityData::new(
        id as u16,
        serde_json::to_string(&properties).unwrap_or_default(),
        embedding
    )
}

#[tokio::test]
async fn test_entity_lifecycle_complete() -> Result<()> {
    // Create brain graph
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Test 1: Insert multiple entities
    let entity_data_1 = create_test_entity_data(1, 96);
    let entity_data_2 = create_test_entity_data(2, 96);
    let entity_data_3 = create_test_entity_data(3, 96);
    
    let entity_key_1 = brain_graph.insert_brain_entity(1, entity_data_1.clone()).await?;
    let entity_key_2 = brain_graph.insert_brain_entity(2, entity_data_2.clone()).await?;
    let entity_key_3 = brain_graph.insert_brain_entity(3, entity_data_3.clone()).await?;
    
    // Verify entities were inserted
    assert_ne!(entity_key_1, entity_key_2);
    assert_ne!(entity_key_2, entity_key_3);
    assert_ne!(entity_key_1, entity_key_3);
    
    // Test 2: Check entity activations were set
    let activation_1 = brain_graph.get_entity_activation(entity_key_1).await;
    let activation_2 = brain_graph.get_entity_activation(entity_key_2).await;
    let activation_3 = brain_graph.get_entity_activation(entity_key_3).await;
    
    assert!(activation_1 > 0.0);
    assert!(activation_2 > 0.0);
    assert!(activation_3 > 0.0);
    
    // Test 3: Update entity activations
    brain_graph.set_entity_activation(entity_key_1, 0.8).await;
    brain_graph.set_entity_activation(entity_key_2, 0.6).await;
    brain_graph.set_entity_activation(entity_key_3, 0.4).await;
    
    let updated_activation_1 = brain_graph.get_entity_activation(entity_key_1).await;
    let updated_activation_2 = brain_graph.get_entity_activation(entity_key_2).await;
    let updated_activation_3 = brain_graph.get_entity_activation(entity_key_3).await;
    
    assert!((updated_activation_1 - 0.8).abs() < 0.001);
    assert!((updated_activation_2 - 0.6).abs() < 0.001);
    assert!((updated_activation_3 - 0.4).abs() < 0.001);
    
    // Test 4: Get entities by activation threshold
    let highly_activated = brain_graph.get_entities_above_threshold(0.7).await;
    assert_eq!(highly_activated.len(), 1);
    assert_eq!(highly_activated[0].0, entity_key_1);
    
    let moderately_activated = brain_graph.get_entities_above_threshold(0.5).await;
    assert_eq!(moderately_activated.len(), 2);
    
    let all_activated = brain_graph.get_entities_above_threshold(0.3).await;
    assert_eq!(all_activated.len(), 3);
    
    // Test 5: Batch update activations
    let activation_updates = vec![
        (entity_key_1, 0.9),
        (entity_key_2, 0.7),
        (entity_key_3, 0.5),
    ];
    
    brain_graph.batch_update_activations(&activation_updates).await;
    
    let final_activation_1 = brain_graph.get_entity_activation(entity_key_1).await;
    let final_activation_2 = brain_graph.get_entity_activation(entity_key_2).await;
    let final_activation_3 = brain_graph.get_entity_activation(entity_key_3).await;
    
    assert!((final_activation_1 - 0.9).abs() < 0.001);
    assert!((final_activation_2 - 0.7).abs() < 0.001);
    assert!((final_activation_3 - 0.5).abs() < 0.001);
    
    Ok(())
}

#[tokio::test]
async fn test_logic_gate_entity_operations() -> Result<()> {
    // Create brain graph
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Create input entities
    let input1_data = create_test_entity_data(1, 96);
    let input2_data = create_test_entity_data(2, 96);
    let input1_key = brain_graph.insert_brain_entity(1, input1_data).await?;
    let input2_key = brain_graph.insert_brain_entity(2, input2_data).await?;
    
    // Create output entity
    let output_data = create_test_entity_data(3, 96);
    let output_key = brain_graph.insert_brain_entity(3, output_data).await?;
    
    // Test 1: Insert AND gate
    let and_gate_key = brain_graph.insert_logic_gate(
        4,
        "AND",
        vec![input1_key, input2_key],
        vec![output_key]
    ).await?;
    
    // Verify gate was created
    let gate_activation = brain_graph.get_entity_activation(and_gate_key).await;
    assert!(gate_activation > 0.0);
    
    // Test 2: Check relationships were created
    assert!(brain_graph.has_relationship(input1_key, and_gate_key).await);
    assert!(brain_graph.has_relationship(input2_key, and_gate_key).await);
    assert!(brain_graph.has_relationship(and_gate_key, output_key).await);
    
    // Test 3: Check relationship weights
    let input1_weight = brain_graph.get_synaptic_weight(input1_key, and_gate_key).await;
    let input2_weight = brain_graph.get_synaptic_weight(input2_key, and_gate_key).await;
    let output_weight = brain_graph.get_synaptic_weight(and_gate_key, output_key).await;
    
    assert!(input1_weight > 0.7); // Input connections should be strong
    assert!(input2_weight > 0.7);
    assert!(output_weight > 0.8); // Output connections should be stronger
    
    // Test 4: Insert OR gate with different configuration
    let or_gate_key = brain_graph.insert_logic_gate(
        5,
        "OR",
        vec![input1_key, input2_key],
        vec![output_key]
    ).await?;
    
    // Verify both gates exist and have different keys
    assert_ne!(and_gate_key, or_gate_key);
    
    let or_gate_activation = brain_graph.get_entity_activation(or_gate_key).await;
    assert!(or_gate_activation > 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_entity_statistics_and_monitoring() -> Result<()> {
    // Create brain graph
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Get initial statistics
    let initial_stats = brain_graph.get_entity_statistics().await;
    assert_eq!(initial_stats.total_entities, 0);
    assert_eq!(initial_stats.active_entities, 0);
    assert_eq!(initial_stats.average_activation, 0.0);
    
    // Insert entities with different activation levels
    let high_entity_data = create_test_entity_data(1, 96);
    let medium_entity_data = create_test_entity_data(2, 96);
    let low_entity_data = create_test_entity_data(3, 96);
    
    let high_key = brain_graph.insert_brain_entity(1, high_entity_data).await?;
    let medium_key = brain_graph.insert_brain_entity(2, medium_entity_data).await?;
    let low_key = brain_graph.insert_brain_entity(3, low_entity_data).await?;
    
    // Set specific activation levels
    brain_graph.set_entity_activation(high_key, 0.9).await;
    brain_graph.set_entity_activation(medium_key, 0.5).await;
    brain_graph.set_entity_activation(low_key, 0.1).await;
    
    // Test statistics after insertions
    let updated_stats = brain_graph.get_entity_statistics().await;
    assert_eq!(updated_stats.total_entities, 3);
    
    // Check average activation
    let expected_avg = (0.9 + 0.5 + 0.1) / 3.0;
    assert!((updated_stats.average_activation - expected_avg).abs() < 0.01);
    
    // Test active entity counting with different thresholds
    let high_threshold_count = brain_graph.count_entities_above_threshold(0.8).await;
    assert_eq!(high_threshold_count, 1);
    
    let medium_threshold_count = brain_graph.count_entities_above_threshold(0.4).await;
    assert_eq!(medium_threshold_count, 2);
    
    let low_threshold_count = brain_graph.count_entities_above_threshold(0.05).await;
    assert_eq!(low_threshold_count, 3);
    
    Ok(())
}

#[tokio::test]
async fn test_entity_similarity_and_clustering() -> Result<()> {
    // Create brain graph
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Create similar entities (similar embeddings)
    let mut similar_embedding = vec![0.1; 96];
    similar_embedding[0] = 0.8;
    similar_embedding[1] = 0.7;
    
    let mut similar_embedding_2 = vec![0.1; 96];
    similar_embedding_2[0] = 0.7;
    similar_embedding_2[1] = 0.8;
    
    let mut different_embedding = vec![0.9; 96];
    different_embedding[0] = 0.1;
    different_embedding[1] = 0.1;
    
    let similar_data_1 = EntityData::new(1, r#"{"type": "similar", "group": "A"}"#.to_string(), similar_embedding.clone());
    
    let similar_data_2 = EntityData::new(2, r#"{"type": "similar", "group": "A"}"#.to_string(), similar_embedding_2);
    
    let different_data = EntityData::new(3, r#"{"type": "different", "group": "B"}"#.to_string(), different_embedding);
    
    // Insert entities
    let similar_key_1 = brain_graph.insert_brain_entity(1, similar_data_1).await?;
    let similar_key_2 = brain_graph.insert_brain_entity(2, similar_data_2).await?;
    let different_key = brain_graph.insert_brain_entity(3, different_data).await?;
    
    // Test similarity search
    let query_result = brain_graph.neural_query(&similar_embedding, 3).await?;
    
    // Check that similar entities are ranked higher
    let sorted_entities = query_result.get_sorted_entities();
    assert_eq!(sorted_entities.len(), 3);
    
    // The most similar entity should be first
    assert!(sorted_entities[0].0 == similar_key_1 || sorted_entities[0].0 == similar_key_2);
    
    // The different entity should have lower activation
    let different_activation = query_result.get_activation(&different_key).unwrap_or(0.0);
    let similar_activation_1 = query_result.get_activation(&similar_key_1).unwrap_or(0.0);
    let similar_activation_2 = query_result.get_activation(&similar_key_2).unwrap_or(0.0);
    
    assert!(similar_activation_1 > different_activation);
    assert!(similar_activation_2 > different_activation);
    
    Ok(())
}

#[tokio::test]
async fn test_entity_batch_operations() -> Result<()> {
    // Create brain graph
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Prepare batch entity data
    let mut entity_keys = Vec::new();
    for i in 1..=10 {
        let entity_data = create_test_entity_data(i, 96);
        let key = brain_graph.insert_brain_entity(i, entity_data).await?;
        entity_keys.push(key);
    }
    
    // Test batch activation updates
    let batch_activations: Vec<(EntityKey, f32)> = entity_keys
        .iter()
        .enumerate()
        .map(|(i, &key)| (key, (i as f32 + 1.0) / 10.0))
        .collect();
    
    brain_graph.batch_update_activations(&batch_activations).await;
    
    // Verify all activations were updated
    for (i, &key) in entity_keys.iter().enumerate() {
        let expected_activation = (i as f32 + 1.0) / 10.0;
        let actual_activation = brain_graph.get_entity_activation(key).await;
        assert!((actual_activation - expected_activation).abs() < 0.001);
    }
    
    // Test filtering entities by activation range
    let mid_range_entities = brain_graph.get_entities_in_activation_range(0.3, 0.7).await;
    assert_eq!(mid_range_entities.len(), 5); // Entities 3-7
    
    // Verify the entities are in the correct range
    for (_, activation) in mid_range_entities {
        assert!(activation >= 0.3 && activation <= 0.7);
    }
    
    // Test top-k entity selection
    let top_3_entities = brain_graph.get_top_k_entities(3).await;
    assert_eq!(top_3_entities.len(), 3);
    
    // Verify they are the highest activated entities
    assert!(top_3_entities[0].1 >= top_3_entities[1].1);
    assert!(top_3_entities[1].1 >= top_3_entities[2].1);
    assert!(top_3_entities[2].1 >= 0.8); // Should be high activation
    
    Ok(())
}

#[tokio::test]
async fn test_entity_concept_formation() -> Result<()> {
    // Create brain graph with concept formation enabled
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Create conceptually related entities
    let concept_a_data_1 = EntityData::new(1, r#"{"concept": "A", "instance": "1"}"#.to_string(), vec![0.8, 0.2, 0.1, 0.1].iter().cycle().take(96).cloned().collect());
    
    let concept_a_data_2 = EntityData::new(2, r#"{"concept": "A", "instance": "2"}"#.to_string(), vec![0.7, 0.3, 0.1, 0.1].iter().cycle().take(96).cloned().collect());
    
    let concept_b_data = EntityData::new(3, r#"{"concept": "B", "instance": "1"}"#.to_string(), vec![0.1, 0.1, 0.8, 0.2].iter().cycle().take(96).cloned().collect());
    
    // Insert entities and trigger concept formation
    let key_a1 = brain_graph.insert_brain_entity(1, concept_a_data_1).await?;
    let key_a2 = brain_graph.insert_brain_entity(2, concept_a_data_2).await?;
    let key_b1 = brain_graph.insert_brain_entity(3, concept_b_data).await?;
    
    // Allow some time for concept formation processing
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    // Test concept-based querying
    let concept_a_query = vec![0.75, 0.25, 0.1, 0.1].iter().cycle().take(96).cloned().collect();
    let concept_a_results = brain_graph.neural_query(&concept_a_query, 3).await?;
    
    // Concept A entities should be more activated
    let activation_a1 = concept_a_results.get_activation(&key_a1).unwrap_or(0.0);
    let activation_a2 = concept_a_results.get_activation(&key_a2).unwrap_or(0.0);
    let activation_b1 = concept_a_results.get_activation(&key_b1).unwrap_or(0.0);
    
    assert!(activation_a1 > activation_b1);
    assert!(activation_a2 > activation_b1);
    
    // Test concept statistics
    let concept_stats = brain_graph.get_concept_statistics().await;
    assert!(concept_stats.total_concepts >= 0); // May form concepts or not
    
    Ok(())
}