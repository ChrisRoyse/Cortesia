//! Brain graph types integration tests
//! Tests data structure compatibility across brain graph components through public APIs
//! Tests serialization/deserialization workflows using only public interfaces

use llmkg::core::brain_enhanced_graph::{
    BrainEnhancedKnowledgeGraph, BrainQueryResult, ConceptStructure, 
    BrainStatistics, BrainEnhancedConfig, BrainMemoryUsage
};
use llmkg::core::types::{EntityKey, EntityData, Relationship};
use llmkg::error::Result;
use std::collections::HashMap;
use tokio;
use serde_json;

/// Helper function to create test entity data
fn create_test_entity_data(id: u32, embedding_len: usize) -> EntityData {
    let embedding = (0..embedding_len).map(|i| (i as f32 + id as f32) * 0.01).collect();
    let mut properties = HashMap::new();
    properties.insert("id".to_string(), id.to_string());
    properties.insert("type".to_string(), "test".to_string());
    
    EntityData::new(
        id as u16,
        serde_json::to_string(&properties).unwrap_or_default(),
        embedding
    )
}

#[tokio::test]
async fn test_brain_query_result_operations() -> Result<()> {
    // Test 1: Create and manipulate BrainQueryResult
    let mut query_result = BrainQueryResult::new();
    
    // Verify initial state
    assert!(query_result.is_empty());
    assert_eq!(query_result.entity_count(), 0);
    assert_eq!(query_result.get_average_activation(), 0.0);
    assert_eq!(query_result.total_activation, 0.0);
    
    // Test 2: Create mock entity keys (using actual brain graph for realistic keys)
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let entity1_data = create_test_entity_data(1, 96);
    let entity2_data = create_test_entity_data(2, 96);
    let entity3_data = create_test_entity_data(3, 96);
    
    let entity1_key = brain_graph.insert_brain_entity(1, entity1_data).await?;
    let entity2_key = brain_graph.insert_brain_entity(2, entity2_data).await?;
    let entity3_key = brain_graph.insert_brain_entity(3, entity3_data).await?;
    
    // Test 3: Add entities with different activations
    query_result.add_entity(entity1_key, 0.9);
    query_result.add_entity(entity2_key, 0.7);
    query_result.add_entity(entity3_key, 0.5);
    
    // Verify state after additions
    assert!(!query_result.is_empty());
    assert_eq!(query_result.entity_count(), 3);
    assert_eq!(query_result.total_activation, 2.1);
    assert!((query_result.get_average_activation() - 0.7).abs() < 0.001);
    
    // Test 4: Check individual entity activations
    assert_eq!(query_result.get_activation(&entity1_key), Some(0.9));
    assert_eq!(query_result.get_activation(&entity2_key), Some(0.7));
    assert_eq!(query_result.get_activation(&entity3_key), Some(0.5));
    
    // Test 5: Test sorted entities
    let sorted_entities = query_result.get_sorted_entities();
    assert_eq!(sorted_entities.len(), 3);
    assert_eq!(sorted_entities[0].0, entity1_key); // Highest activation first
    assert_eq!(sorted_entities[0].1, 0.9);
    assert_eq!(sorted_entities[2].0, entity3_key); // Lowest activation last
    assert_eq!(sorted_entities[2].1, 0.5);
    
    // Test 6: Test top-k selection
    let top_2 = query_result.get_top_k(2);
    assert_eq!(top_2.len(), 2);
    assert_eq!(top_2[0].0, entity1_key);
    assert_eq!(top_2[1].0, entity2_key);
    
    let top_5 = query_result.get_top_k(5); // More than available
    assert_eq!(top_5.len(), 3); // Should return all available
    
    // Test 7: Test threshold filtering
    let above_0_8 = query_result.get_entities_above_threshold(0.8);
    assert_eq!(above_0_8.len(), 1);
    assert_eq!(above_0_8[0].0, entity1_key);
    
    let above_0_6 = query_result.get_entities_above_threshold(0.6);
    assert_eq!(above_0_6.len(), 2);
    
    let above_0_4 = query_result.get_entities_above_threshold(0.4);
    assert_eq!(above_0_4.len(), 3);
    
    let above_1_0 = query_result.get_entities_above_threshold(1.0);
    assert_eq!(above_1_0.len(), 0);
    
    Ok(())
}

#[tokio::test]
async fn test_brain_query_result_serialization() -> Result<()> {
    // Test 1: Create query result with realistic data
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let entity1_data = create_test_entity_data(1, 96);
    let entity2_data = create_test_entity_data(2, 96);
    
    let entity1_key = brain_graph.insert_brain_entity(1, entity1_data).await?;
    let entity2_key = brain_graph.insert_brain_entity(2, entity2_data).await?;
    
    let mut original_result = BrainQueryResult::new();
    original_result.add_entity(entity1_key, 0.8);
    original_result.add_entity(entity2_key, 0.6);
    
    // Test 2: Serialize to JSON
    let serialized = serde_json::to_string(&original_result)?;
    assert!(!serialized.is_empty());
    assert!(serialized.contains("entities"));
    assert!(serialized.contains("activations"));
    assert!(serialized.contains("total_activation"));
    
    // Test 3: Deserialize from JSON
    let deserialized_result: BrainQueryResult = serde_json::from_str(&serialized)?;
    
    // Test 4: Verify deserialized data integrity
    assert_eq!(deserialized_result.entity_count(), original_result.entity_count());
    assert_eq!(deserialized_result.total_activation, original_result.total_activation);
    assert!((deserialized_result.get_average_activation() - original_result.get_average_activation()).abs() < 0.001);
    
    // Check individual entities
    assert_eq!(deserialized_result.get_activation(&entity1_key), Some(0.8));
    assert_eq!(deserialized_result.get_activation(&entity2_key), Some(0.6));
    
    // Test 5: Round-trip serialization
    let re_serialized = serde_json::to_string(&deserialized_result)?;
    let re_deserialized: BrainQueryResult = serde_json::from_str(&re_serialized)?;
    
    assert_eq!(re_deserialized.entity_count(), original_result.entity_count());
    assert_eq!(re_deserialized.total_activation, original_result.total_activation);
    
    Ok(())
}

#[tokio::test]
async fn test_concept_structure_compatibility() -> Result<()> {
    // Test 1: Create brain graph and build concept structure
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Create input entities
    let input1_data = create_test_entity_data(1, 96);
    let input2_data = create_test_entity_data(2, 96);
    let input1_key = brain_graph.insert_brain_entity(1, input1_data).await?;
    let input2_key = brain_graph.insert_brain_entity(2, input2_data).await?;
    
    // Create gate entities
    let gate1_key = brain_graph.insert_logic_gate(3, "AND", vec![input1_key, input2_key], vec![]).await?;
    let gate2_key = brain_graph.insert_logic_gate(4, "OR", vec![input1_key, input2_key], vec![]).await?;
    
    // Create output entity
    let output_data = create_test_entity_data(5, 96);
    let output_key = brain_graph.insert_brain_entity(5, output_data).await?;
    
    // Connect gates to output
    let gate1_to_output = Relationship {
        from: gate1_key,
        to: output_key,
        rel_type: 1,
        weight: 0.8,
    };
    let gate2_to_output = Relationship {
        from: gate2_key,
        to: output_key,
        rel_type: 1,
        weight: 0.8,
    };
    
    brain_graph.insert_brain_relationship(gate1_to_output).await?;
    brain_graph.insert_brain_relationship(gate2_to_output).await?;
    
    // Test 2: Create concept structure manually
    let concept_structure = ConceptStructure {
        input_entities: vec![input1_key, input2_key],
        output_entities: vec![output_key],
        gate_entities: vec![gate1_key, gate2_key],
        activation_threshold: 0.5,
        concept_weight: 0.7,
        stability_score: 0.8,
        formation_time: std::time::SystemTime::now(),
    };
    
    // Test 3: Serialize concept structure
    let serialized_concept = serde_json::to_string(&concept_structure)?;
    assert!(!serialized_concept.is_empty());
    
    // Test 4: Deserialize concept structure
    let deserialized_concept: ConceptStructure = serde_json::from_str(&serialized_concept)?;
    
    // Verify structure integrity
    assert_eq!(deserialized_concept.input_entities.len(), 2);
    assert_eq!(deserialized_concept.output_entities.len(), 1);
    assert_eq!(deserialized_concept.gate_entities.len(), 2);
    assert_eq!(deserialized_concept.activation_threshold, 0.5);
    assert_eq!(deserialized_concept.concept_weight, 0.7);
    assert_eq!(deserialized_concept.stability_score, 0.8);
    
    // Test 5: Store and retrieve concept structure through brain graph
    brain_graph.store_concept_structure("test_concept", concept_structure.clone()).await?;
    let retrieved_concept = brain_graph.get_concept_structure("test_concept").await;
    
    assert!(retrieved_concept.is_some());
    let retrieved = retrieved_concept.unwrap();
    assert_eq!(retrieved.input_entities, concept_structure.input_entities);
    assert_eq!(retrieved.output_entities, concept_structure.output_entities);
    assert_eq!(retrieved.gate_entities, concept_structure.gate_entities);
    
    Ok(())
}

#[tokio::test]
async fn test_brain_statistics_tracking() -> Result<()> {
    // Test 1: Create brain graph and verify initial statistics
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let initial_stats = brain_graph.get_brain_statistics().await;
    assert_eq!(initial_stats.entity_count, 0);
    assert_eq!(initial_stats.relationship_count, 0);
    assert_eq!(initial_stats.avg_activation, 0.0);
    assert_eq!(initial_stats.max_activation, 0.0);
    assert_eq!(initial_stats.min_activation, 0.0);
    
    // Test 2: Add entities and verify statistics updates
    let entity1_data = create_test_entity_data(1, 96);
    let entity2_data = create_test_entity_data(2, 96);
    let entity3_data = create_test_entity_data(3, 96);
    
    let entity1_key = brain_graph.insert_brain_entity(1, entity1_data).await?;
    let entity2_key = brain_graph.insert_brain_entity(2, entity2_data).await?;
    let entity3_key = brain_graph.insert_brain_entity(3, entity3_data).await?;
    
    let after_entities_stats = brain_graph.get_brain_statistics().await;
    assert_eq!(after_entities_stats.entity_count, 3);
    assert!(after_entities_stats.avg_activation > 0.0);
    assert!(after_entities_stats.max_activation > 0.0);
    assert!(after_entities_stats.min_activation > 0.0);
    
    // Test 3: Add relationships and verify updates
    let relationship1 = Relationship {
        from: entity1_key,
        to: entity2_key,
        rel_type: 1,
        weight: 0.8,
    };
    let relationship2 = Relationship {
        from: entity2_key,
        to: entity3_key,
        rel_type: 1,
        weight: 0.7,
    };
    
    brain_graph.insert_brain_relationship(relationship1).await?;
    brain_graph.insert_brain_relationship(relationship2).await?;
    
    let after_relationships_stats = brain_graph.get_brain_statistics().await;
    assert_eq!(after_relationships_stats.relationship_count, 2);
    
    // Test 4: Modify activations and verify statistics tracking
    brain_graph.set_entity_activation(entity1_key, 1.0).await;
    brain_graph.set_entity_activation(entity2_key, 0.5).await;
    brain_graph.set_entity_activation(entity3_key, 0.1).await;
    
    // Allow statistics to update
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    
    let updated_stats = brain_graph.get_brain_statistics().await;
    assert_eq!(updated_stats.max_activation, 1.0);
    assert_eq!(updated_stats.min_activation, 0.1);
    
    let expected_avg = (1.0 + 0.5 + 0.1) / 3.0;
    assert!((updated_stats.avg_activation - expected_avg).abs() < 0.1);
    
    // Test 5: Statistics serialization
    let serialized_stats = serde_json::to_string(&updated_stats)?;
    let deserialized_stats: BrainStatistics = serde_json::from_str(&serialized_stats)?;
    
    assert_eq!(deserialized_stats.entity_count, updated_stats.entity_count);
    assert_eq!(deserialized_stats.relationship_count, updated_stats.relationship_count);
    assert!((deserialized_stats.avg_activation - updated_stats.avg_activation).abs() < 0.001);
    assert_eq!(deserialized_stats.max_activation, updated_stats.max_activation);
    assert_eq!(deserialized_stats.min_activation, updated_stats.min_activation);
    
    Ok(())
}

#[tokio::test]
async fn test_configuration_compatibility() -> Result<()> {
    // Test 1: Default configuration
    let default_config = BrainEnhancedConfig::default();
    
    assert!(default_config.activation_threshold > 0.0);
    assert!(default_config.activation_decay_rate > 0.0 && default_config.activation_decay_rate < 1.0);
    assert!(default_config.propagation_steps > 0);
    assert!(default_config.concept_formation_threshold > 0.0);
    
    // Test 2: Testing configuration
    let test_config = BrainEnhancedConfig::for_testing();
    
    // Testing config should be more permissive
    assert!(test_config.activation_threshold <= default_config.activation_threshold);
    assert!(test_config.concept_formation_threshold <= default_config.concept_formation_threshold);
    
    // Test 3: Configuration serialization
    let serialized_config = serde_json::to_string(&default_config)?;
    let deserialized_config: BrainEnhancedConfig = serde_json::from_str(&serialized_config)?;
    
    assert_eq!(deserialized_config.activation_threshold, default_config.activation_threshold);
    assert_eq!(deserialized_config.activation_decay_rate, default_config.activation_decay_rate);
    assert_eq!(deserialized_config.propagation_steps, default_config.propagation_steps);
    assert_eq!(deserialized_config.enable_concept_formation, default_config.enable_concept_formation);
    
    // Test 4: Create brain graph with custom configuration
    let mut custom_config = BrainEnhancedConfig::for_testing();
    custom_config.activation_threshold = 0.3;
    custom_config.propagation_steps = 5;
    custom_config.enable_concept_formation = true;
    
    let brain_graph = BrainEnhancedKnowledgeGraph::new_with_config(96, custom_config.clone())?;
    
    // Verify configuration is applied
    let retrieved_config = brain_graph.get_configuration().await;
    assert_eq!(retrieved_config.activation_threshold, custom_config.activation_threshold);
    assert_eq!(retrieved_config.propagation_steps, custom_config.propagation_steps);
    assert_eq!(retrieved_config.enable_concept_formation, custom_config.enable_concept_formation);
    
    Ok(())
}

#[tokio::test]
async fn test_cross_component_data_flow() -> Result<()> {
    // Test 1: Create brain graph and components
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Test 2: Insert entities and verify they work across all components
    let entity1_data = create_test_entity_data(1, 96);
    let entity2_data = create_test_entity_data(2, 96);
    
    let entity1_key = brain_graph.insert_brain_entity(1, entity1_data.clone()).await?;
    let entity2_key = brain_graph.insert_brain_entity(2, entity2_data.clone()).await?;
    
    // Test entity manager integration
    let entity1_activation = brain_graph.get_entity_activation(entity1_key).await;
    assert!(entity1_activation > 0.0);
    
    // Test 3: Create relationships and verify relationship manager integration
    let relationship = Relationship {
        from: entity1_key,
        to: entity2_key,
        rel_type: 1,
        weight: 0.8,
    };
    brain_graph.insert_brain_relationship(relationship).await?;
    
    let neighbors = brain_graph.get_neighbors_with_weights(entity1_key).await;
    assert_eq!(neighbors.len(), 1);
    assert_eq!(neighbors[0].0, entity2_key);
    
    // Test 4: Query engine integration with entity and relationship data
    let query_result = brain_graph.neural_query(&entity1_data.embedding, 2).await?;
    assert!(!query_result.is_empty());
    
    // Query should find both entities
    assert_eq!(query_result.entity_count(), 2);
    
    // Test 5: Verify data consistency across components
    let graph_stats = brain_graph.get_entity_statistics().await;
    let relationship_stats = brain_graph.get_relationship_statistics().await;
    let query_stats = brain_graph.get_query_statistics().await;
    
    assert_eq!(graph_stats.total_entities, 2);
    assert_eq!(relationship_stats.total_relationships, 1);
    assert!(query_stats.total_queries > 0);
    
    // Test 6: Cross-component activation propagation
    brain_graph.set_entity_activation(entity1_key, 1.0).await;
    
    // Propagation should affect relationship weights and query results
    let propagation_result = brain_graph.propagate_activation_from_entity(entity1_key, 0.1).await?;
    assert!(propagation_result.affected_entities > 0);
    
    let entity2_activation = brain_graph.get_entity_activation(entity2_key).await;
    assert!(entity2_activation > 0.0); // Should be activated through relationship
    
    // Test 7: Verify query results reflect activation changes
    let updated_query = brain_graph.neural_query(&entity1_data.embedding, 2).await?;
    let entity1_result_activation = updated_query.get_activation(&entity1_key).unwrap_or(0.0);
    let entity2_result_activation = updated_query.get_activation(&entity2_key).unwrap_or(0.0);
    
    assert!(entity1_result_activation > entity2_result_activation); // entity1 should have higher activation
    
    Ok(())
}

#[tokio::test]
async fn test_data_structure_edge_cases() -> Result<()> {
    // Test 1: Empty query result behavior
    let empty_result = BrainQueryResult::new();
    assert!(empty_result.is_empty());
    assert_eq!(empty_result.get_sorted_entities().len(), 0);
    assert_eq!(empty_result.get_top_k(5).len(), 0);
    assert_eq!(empty_result.get_entities_above_threshold(0.5).len(), 0);
    assert_eq!(empty_result.get_average_activation(), 0.0);
    
    // Test 2: Query result with zero activations
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    let entity_data = create_test_entity_data(1, 96);
    let entity_key = brain_graph.insert_brain_entity(1, entity_data).await?;
    
    let mut zero_activation_result = BrainQueryResult::new();
    zero_activation_result.add_entity(entity_key, 0.0);
    
    assert!(!zero_activation_result.is_empty());
    assert_eq!(zero_activation_result.get_average_activation(), 0.0);
    assert_eq!(zero_activation_result.get_entities_above_threshold(0.1).len(), 0);
    assert_eq!(zero_activation_result.get_top_k(1).len(), 1); // Should still return the entity
    
    // Test 3: Very large activation values
    let mut large_activation_result = BrainQueryResult::new();
    large_activation_result.add_entity(entity_key, 1000.0);
    
    assert_eq!(large_activation_result.get_average_activation(), 1000.0);
    assert_eq!(large_activation_result.get_entities_above_threshold(500.0).len(), 1);
    
    // Test 4: Negative activation values (edge case)
    let mut negative_result = BrainQueryResult::new();
    negative_result.add_entity(entity_key, -0.5);
    
    assert_eq!(negative_result.total_activation, -0.5);
    assert_eq!(negative_result.get_average_activation(), -0.5);
    assert_eq!(negative_result.get_entities_above_threshold(0.0).len(), 0);
    
    // Test 5: Serialization of edge cases
    let serialized_negative = serde_json::to_string(&negative_result)?;
    let deserialized_negative: BrainQueryResult = serde_json::from_str(&serialized_negative)?;
    assert_eq!(deserialized_negative.total_activation, -0.5);
    
    Ok(())
}

#[tokio::test]
async fn test_memory_usage_data_consistency() -> Result<()> {
    // Test 1: Create brain graph and track memory usage
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let initial_memory = brain_graph.get_memory_usage().await;
    assert_eq!(initial_memory.total_entities, 0);
    assert_eq!(initial_memory.total_relationships, 0);
    assert_eq!(initial_memory.activation_memory, 0);
    
    // Test 2: Add entities and verify memory tracking
    let mut entity_keys = Vec::new();
    for i in 1..6 {
        let entity_data = create_test_entity_data(i, 96);
        let key = brain_graph.insert_brain_entity(i, entity_data).await?;
        entity_keys.push(key);
    }
    
    let after_entities_memory = brain_graph.get_memory_usage().await;
    assert_eq!(after_entities_memory.total_entities, 5);
    assert!(after_entities_memory.entity_memory > 0);
    assert!(after_entities_memory.activation_memory > 0);
    
    // Test 3: Add relationships and verify memory updates
    for i in 0..4 {
        let relationship = Relationship {
            from: entity_keys[i],
            to: entity_keys[i + 1],
            rel_type: 1,
            weight: 0.7,
        };
        brain_graph.insert_brain_relationship(relationship).await?;
    }
    
    let after_relationships_memory = brain_graph.get_memory_usage().await;
    assert_eq!(after_relationships_memory.total_relationships, 4);
    assert!(after_relationships_memory.relationship_memory > 0);
    assert!(after_relationships_memory.synaptic_weight_memory > 0);
    
    // Test 4: Perform queries and verify cache memory
    for i in 0..3 {
        let query_embedding = (0..96).map(|j| ((i * 10 + j) as f32) * 0.01).collect();
        brain_graph.neural_query(&query_embedding, 3).await?;
    }
    
    let after_queries_memory = brain_graph.get_memory_usage().await;
    assert!(after_queries_memory.query_cache_memory > 0);
    
    // Test 5: Memory usage serialization and consistency
    let serialized_memory = serde_json::to_string(&after_queries_memory)?;
    let deserialized_memory: BrainMemoryUsage = serde_json::from_str(&serialized_memory)?;
    
    assert_eq!(deserialized_memory.total_entities, after_queries_memory.total_entities);
    assert_eq!(deserialized_memory.total_relationships, after_queries_memory.total_relationships);
    assert_eq!(deserialized_memory.entity_memory, after_queries_memory.entity_memory);
    assert_eq!(deserialized_memory.relationship_memory, after_queries_memory.relationship_memory);
    
    // Test 6: Verify total memory calculation
    let expected_total = deserialized_memory.entity_memory 
        + deserialized_memory.relationship_memory 
        + deserialized_memory.activation_memory
        + deserialized_memory.synaptic_weight_memory
        + deserialized_memory.query_cache_memory
        + deserialized_memory.concept_structure_memory;
    
    assert_eq!(deserialized_memory.total_memory, expected_total);
    
    Ok(())
}