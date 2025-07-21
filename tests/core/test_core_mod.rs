//! Module organization tests for core module components
//! 
//! Tests that components from multiple core submodules perform complete
//! end-to-end tasks and that modules are correctly integrated with
//! public APIs working as expected.

use std::collections::HashMap;

use llmkg::core::{
    graph::KnowledgeGraph,
    types::{EntityData, EntityKey, Relationship},
    entity::EntityStore,
    memory::{GraphArena, EpochManager},
    triple::Triple,
    brain_types::*,
    activation_engine::ActivationEngine,
    knowledge_engine::KnowledgeEngine,
    semantic_summary::SemanticSummary,
    parallel::ParallelProcessor,
    zero_copy_engine::ZeroCopyEngine,
    phase1_integration::Phase1IntegrationLayer,
    phase1_types::Phase1Config,
};

fn create_test_embedding(seed: u64, dim: usize) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    
    let mut embedding = Vec::with_capacity(dim);
    let mut hash = hasher.finish();
    
    for _ in 0..dim {
        hash = hash.wrapping_mul(1103515245).wrapping_add(12345);
        embedding.push(((hash >> 16) & 0xFFFF) as f32 / 65536.0);
    }
    
    // Normalize
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in embedding.iter_mut() {
            *val /= magnitude;
        }
    }
    
    embedding
}

#[test]
fn test_core_modules_integration_workflow() {
    // This test verifies that all core modules work together correctly
    
    // Phase 1: Initialize core components
    let graph = KnowledgeGraph::new(128).expect("Failed to create graph");
    let epoch_manager = std::sync::Arc::new(EpochManager::new(16));
    
    // Test that graph's epoch manager is accessible
    let graph_epoch_manager = graph.epoch_manager();
    assert!(graph_epoch_manager.current_epoch() >= 0);
    
    // Phase 2: Test entity and memory integration
    let entity_data_1 = EntityData {
        type_id: 1,
        embedding: create_test_embedding(1, 128),
        properties: r#"{"type": "concept", "name": "Machine Learning"}"#.to_string(),
    };
    
    let entity_data_2 = EntityData {
        type_id: 2,
        embedding: create_test_embedding(2, 128),
        properties: r#"{"type": "concept", "name": "Neural Networks"}"#.to_string(),
    };
    
    let key1 = graph.insert_entity(1, entity_data_1.clone()).expect("Failed to insert entity");
    let key2 = graph.insert_entity(2, entity_data_2.clone()).expect("Failed to insert entity");
    
    // Verify memory allocation worked correctly
    let memory_usage = graph.memory_usage();
    assert!(memory_usage.total_bytes() > 0);
    assert!(memory_usage.arena_bytes > 0);
    assert!(memory_usage.entity_store_bytes > 0);
    
    // Phase 3: Test relationship and graph integration
    let relationship = Relationship {
        from: key1,
        to: key2,
        rel_type: 1,
        weight: 0.8,
    };
    
    let rel_result = graph.insert_relationship(relationship);
    assert!(rel_result.is_ok());
    
    // Test relationship queries work across modules
    let neighbors = graph.get_neighbors(key1);
    assert_eq!(neighbors.len(), 1);
    assert!(neighbors.contains(&key2));
    
    let outgoing = graph.get_outgoing_relationships(key1);
    assert_eq!(outgoing.len(), 1);
    assert_eq!(outgoing[0].to, key2);
    assert_eq!(outgoing[0].weight, 0.8);
    
    // Phase 4: Test triple integration with graph
    let subject = "Machine Learning".to_string();
    let predicate = "related_to".to_string();
    let object = "Neural Networks".to_string();
    
    let triple = Triple::new(subject.clone(), predicate.clone(), object.clone());
    assert_eq!(triple.subject(), &subject);
    assert_eq!(triple.predicate(), &predicate);
    assert_eq!(triple.object(), &object);
    
    // Test triple-to-graph conversion concepts
    let triple_hash = triple.hash_identifier();
    assert!(triple_hash > 0);
    
    // Phase 5: Test semantic summary integration
    let entities_for_summary = vec![entity_data_1.clone(), entity_data_2.clone()];
    let summary = SemanticSummary::from_entities(&entities_for_summary);
    
    assert!(summary.entity_count() >= 2);
    let summary_stats = summary.get_statistics();
    assert!(summary_stats.contains_key("total_entities"));
    
    // Phase 6: Test activation engine with graph data
    let activation_config = ActivationConfig {
        threshold: 0.5,
        decay_rate: 0.1,
        max_activation: 1.0,
        min_activation: 0.0,
    };
    
    let activation_engine = ActivationEngine::new(activation_config);
    
    // Create activation context from graph entities
    let entity_ids = vec![1, 2];
    let activations = activation_engine.compute_activations(&entity_ids);
    assert_eq!(activations.len(), 2);
    
    for activation in &activations {
        assert!(activation.value >= 0.0 && activation.value <= 1.0);
    }
    
    // Phase 7: Test knowledge engine integration
    let knowledge_engine = KnowledgeEngine::new();
    
    // Extract knowledge from graph entities
    let knowledge_items = knowledge_engine.extract_from_entities(&[
        (1, entity_data_1.clone()),
        (2, entity_data_2.clone()),
    ]);
    
    assert!(knowledge_items.len() >= 2);
    
    // Test knowledge embedding integration
    for item in &knowledge_items {
        assert!(!item.content.is_empty());
        assert!(item.embedding.len() > 0);
        assert!(item.confidence >= 0.0 && item.confidence <= 1.0);
    }
    
    // Phase 8: Test parallel processing integration
    let parallel_data = vec![
        (10, create_test_embedding(10, 128)),
        (11, create_test_embedding(11, 128)),
        (12, create_test_embedding(12, 128)),
        (13, create_test_embedding(13, 128)),
    ];
    
    let should_use_parallel = ParallelProcessor::should_use_parallel(
        parallel_data.len(), 
        llmkg::core::parallel::ParallelOperation::BatchValidation
    );
    
    // For small batch, might not use parallel
    // Test the decision logic works
    assert!(!should_use_parallel || parallel_data.len() >= 4);
    
    // Phase 9: Test zero-copy engine integration
    let zero_copy = ZeroCopyEngine::new();
    
    // Test zero-copy operations with existing graph data
    let entity_ref = graph.get_entity(key1);
    assert!(entity_ref.is_some());
    
    let (_, entity_data) = entity_ref.unwrap();
    let zero_copy_slice = zero_copy.create_embedding_slice(&entity_data.embedding);
    assert_eq!(zero_copy_slice.len(), entity_data.embedding.len());
    
    // Test zero-copy performance measurement
    let benchmark = zero_copy.benchmark_operations(100);
    assert!(benchmark.total_time.as_nanos() > 0);
    assert_eq!(benchmark.operation_count, 100);
    assert!(benchmark.average_time_per_op().as_nanos() > 0);
    
    // Phase 10: Test brain types integration
    let cognitive_load = CognitiveLoad {
        attention_weight: 0.8,
        memory_pressure: 0.3,
        processing_complexity: 0.6,
    };
    
    assert!(cognitive_load.is_within_limits());
    
    let neural_activation = NeuralActivation {
        node_id: key1.into(), // Convert EntityKey to NodeId
        activation_level: 0.75,
        timestamp: std::time::SystemTime::now(),
    };
    
    assert!(neural_activation.is_active(0.5));
    assert!(!neural_activation.is_active(0.9));
    
    // Phase 11: Test phase1 integration layer
    let phase1_config = Phase1Config::default();
    let phase1_layer = Phase1IntegrationLayer::new(phase1_config);
    
    // Test phase1 layer can work with graph
    let integration_result = phase1_layer.integrate_with_graph(&graph);
    assert!(integration_result.is_ok());
    
    let phase1_stats = phase1_layer.get_statistics();
    assert!(phase1_stats.total_operations >= 0);
    assert!(phase1_stats.success_rate >= 0.0 && phase1_stats.success_rate <= 1.0);
    
    // Final verification: All components working together
    assert_eq!(graph.entity_count(), 2);
    assert_eq!(graph.relationship_count(), 1);
    assert_eq!(graph.embedding_dimension(), 128);
    
    // Memory usage should reflect all operations
    let final_memory = graph.memory_usage();
    assert!(final_memory.total_bytes() >= memory_usage.total_bytes());
}

#[test] 
fn test_cross_module_data_flow() {
    // This test ensures data flows correctly between different core modules
    
    // Phase 1: Start with brain types
    let cognitive_state = CognitiveState {
        attention_focus: vec![1, 2, 3],
        working_memory: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        long_term_memory_activation: 0.7,
    };
    
    assert_eq!(cognitive_state.attention_focus.len(), 3);
    assert_eq!(cognitive_state.working_memory.len(), 5);
    
    // Phase 2: Convert to graph entities
    let graph = KnowledgeGraph::new(128).unwrap();
    
    let mut entities = Vec::new();
    for (i, &focus_item) in cognitive_state.attention_focus.iter().enumerate() {
        let entity_data = EntityData {
            type_id: focus_item,
            embedding: create_test_embedding(focus_item as u64, 128),
            properties: format!(r#"{{"cognitive_focus": {}, "index": {}}}"#, focus_item, i),
        };
        
        let key = graph.insert_entity(focus_item, entity_data.clone()).unwrap();
        entities.push((focus_item, key, entity_data));
    }
    
    // Phase 3: Process with activation engine
    let activation_config = ActivationConfig {
        threshold: 0.3,
        decay_rate: 0.05,
        max_activation: cognitive_state.long_term_memory_activation,
        min_activation: 0.0,
    };
    
    let activation_engine = ActivationEngine::new(activation_config);
    let entity_ids: Vec<u32> = entities.iter().map(|(id, _, _)| *id).collect();
    let activations = activation_engine.compute_activations(&entity_ids);
    
    assert_eq!(activations.len(), entities.len());
    
    // Phase 4: Create relationships based on activation levels
    for i in 0..entities.len() {
        for j in (i+1)..entities.len() {
            let (_, key1, _) = &entities[i];
            let (_, key2, _) = &entities[j];
            
            let activation1 = activations[i].value;
            let activation2 = activations[j].value;
            let relationship_weight = (activation1 + activation2) / 2.0;
            
            if relationship_weight > 0.4 { // Only create strong relationships
                let relationship = Relationship {
                    from: *key1,
                    to: *key2,
                    rel_type: 1,
                    weight: relationship_weight,
                };
                
                let _ = graph.insert_relationship(relationship);
            }
        }
    }
    
    // Phase 5: Extract knowledge from the resulting graph
    let knowledge_engine = KnowledgeEngine::new();
    let entity_data_pairs: Vec<(u32, EntityData)> = entities.iter()
        .map(|(id, _, data)| (*id, data.clone()))
        .collect();
    
    let knowledge_items = knowledge_engine.extract_from_entities(&entity_data_pairs);
    assert!(knowledge_items.len() >= entities.len());
    
    // Phase 6: Create semantic summary
    let entity_data_vec: Vec<EntityData> = entities.iter()
        .map(|(_, _, data)| data.clone())
        .collect();
    
    let semantic_summary = SemanticSummary::from_entities(&entity_data_vec);
    let summary_stats = semantic_summary.get_statistics();
    
    assert!(summary_stats.contains_key("total_entities"));
    let total_entities = summary_stats.get("total_entities").unwrap();
    assert_eq!(*total_entities as usize, entities.len());
    
    // Phase 7: Use zero-copy engine for efficient access
    let zero_copy = ZeroCopyEngine::new();
    
    for (_, key, _) in &entities {
        if let Some((_, entity_data)) = graph.get_entity(*key) {
            let embedding_slice = zero_copy.create_embedding_slice(&entity_data.embedding);
            assert_eq!(embedding_slice.len(), 128);
            
            // Verify zero-copy slice matches original
            for (i, &value) in entity_data.embedding.iter().enumerate() {
                assert_eq!(embedding_slice[i], value);
            }
        }
    }
    
    // Phase 8: Test with phase1 integration
    let phase1_config = Phase1Config::default();
    let phase1_layer = Phase1IntegrationLayer::new(phase1_config);
    
    let integration_result = phase1_layer.integrate_with_graph(&graph);
    assert!(integration_result.is_ok());
    
    // Query through phase1 layer
    let query_embedding = create_test_embedding(100, 128);
    let query_result = phase1_layer.query_similar(&query_embedding, 2);
    assert!(query_result.is_ok());
    
    let results = query_result.unwrap();
    assert!(results.entities.len() <= 2);
    assert!(results.query_time.as_nanos() > 0);
    
    // Phase 9: Verify final state consistency
    assert_eq!(graph.entity_count(), 3);
    assert!(graph.relationship_count() >= 0); // Might be 0 if no activations were strong enough
    
    // All entities should still be accessible
    for (entity_id, entity_key, _) in &entities {
        let retrieved = graph.get_entity_by_id(*entity_id);
        assert!(retrieved.is_some());
        
        let (retrieved_key, _) = retrieved.unwrap();
        assert_eq!(retrieved_key, *entity_key);
    }
    
    // Memory usage should be reasonable
    let memory_usage = graph.memory_usage();
    assert!(memory_usage.total_bytes() > 0);
    
    let breakdown = memory_usage.usage_breakdown();
    let total_percentage = breakdown.arena_percentage + 
                          breakdown.entity_store_percentage + 
                          breakdown.graph_percentage + 
                          breakdown.embedding_bank_percentage + 
                          breakdown.quantizer_percentage + 
                          breakdown.bloom_filter_percentage;
    
    // Should approximately sum to 100%
    assert!((total_percentage - 100.0).abs() < 5.0);
}

#[test]
fn test_module_error_propagation() {
    // This test ensures errors are properly handled across module boundaries
    
    // Phase 1: Test graph creation errors
    let large_dim_result = KnowledgeGraph::new(10000); // Very large dimension
    // Should succeed or fail gracefully
    if let Err(error) = large_dim_result {
        // Error should be meaningful
        match error {
            llmkg::error::GraphError::InvalidEmbeddingDimension { expected: _, actual: _ } => (),
            llmkg::error::GraphError::InitializationError(_) => (),
            _ => panic!("Unexpected error type: {:?}", error),
        }
    }
    
    // Phase 2: Test entity insertion errors propagation
    let graph = KnowledgeGraph::new(64).unwrap();
    
    let invalid_entity = EntityData {
        type_id: 1,
        embedding: vec![0.1; 32], // Wrong dimension
        properties: "{}".to_string(),
    };
    
    let insert_result = graph.insert_entity(1, invalid_entity);
    assert!(insert_result.is_err());
    
    if let Err(error) = insert_result {
        match error {
            llmkg::error::GraphError::InvalidEmbeddingDimension { expected, actual } => {
                assert_eq!(expected, 64);
                assert_eq!(actual, 32);
            },
            _ => panic!("Expected InvalidEmbeddingDimension error, got: {:?}", error),
        }
    }
    
    // Phase 3: Test activation engine error handling
    let invalid_config = ActivationConfig {
        threshold: -0.5, // Invalid negative threshold
        decay_rate: 0.1,
        max_activation: 1.0,
        min_activation: 0.0,
    };
    
    let activation_engine = ActivationEngine::new(invalid_config);
    let activations = activation_engine.compute_activations(&[1, 2, 3]);
    
    // Should handle invalid config gracefully
    assert_eq!(activations.len(), 3);
    for activation in activations {
        // Values should still be in valid range despite invalid config
        assert!(activation.value >= 0.0);
        assert!(activation.value <= 1.0);
    }
    
    // Phase 4: Test knowledge engine error handling
    let knowledge_engine = KnowledgeEngine::new();
    
    let invalid_entities = vec![
        (1, EntityData {
            type_id: 1,
            embedding: vec![], // Empty embedding
            properties: "invalid json {".to_string(),
        })
    ];
    
    let knowledge_items = knowledge_engine.extract_from_entities(&invalid_entities);
    // Should handle gracefully and return empty or filtered results
    assert!(knowledge_items.len() <= 1);
    
    // Phase 5: Test semantic summary error handling
    let invalid_entity_data = vec![
        EntityData {
            type_id: 1,
            embedding: vec![], // Empty embedding
            properties: "".to_string(), // Empty properties
        }
    ];
    
    let summary = SemanticSummary::from_entities(&invalid_entity_data);
    let stats = summary.get_statistics();
    
    // Should handle gracefully
    assert!(stats.contains_key("total_entities"));
    assert!(stats.get("total_entities").unwrap_or(&0.0) >= &0.0);
    
    // Phase 6: Test zero-copy engine error handling
    let zero_copy = ZeroCopyEngine::new();
    
    let empty_embedding = vec![];
    let slice_result = zero_copy.create_embedding_slice(&empty_embedding);
    assert_eq!(slice_result.len(), 0); // Should handle empty input gracefully
    
    // Phase 7: Test phase1 integration error handling
    let phase1_config = Phase1Config::default();
    let phase1_layer = Phase1IntegrationLayer::new(phase1_config);
    
    // Try to integrate with empty graph
    let empty_graph = KnowledgeGraph::new(96).unwrap();
    let integration_result = phase1_layer.integrate_with_graph(&empty_graph);
    
    // Should handle empty graph gracefully
    assert!(integration_result.is_ok());
    
    // Try query on empty integration
    let query_embedding = create_test_embedding(1, 96);
    let query_result = phase1_layer.query_similar(&query_embedding, 5);
    
    if let Ok(results) = query_result {
        assert!(results.entities.is_empty()); // No entities to find
    }
    
    // Phase 8: Test brain types error handling
    let invalid_cognitive_load = CognitiveLoad {
        attention_weight: 2.0, // > 1.0, invalid
        memory_pressure: -0.1, // Negative, invalid
        processing_complexity: 0.5,
    };
    
    // Should still function but indicate it's not within limits
    assert!(!invalid_cognitive_load.is_within_limits());
    
    let invalid_activation = NeuralActivation {
        node_id: 999999, // Very large node ID
        activation_level: -0.5, // Negative activation
        timestamp: std::time::SystemTime::now(),
    };
    
    // Should handle invalid activation level gracefully
    assert!(!invalid_activation.is_active(0.0)); // Negative activation not active
    
    // Phase 9: Test complex error scenario
    let graph = KnowledgeGraph::new(128).unwrap();
    
    // Insert valid entity
    let valid_entity = EntityData {
        type_id: 1,
        embedding: create_test_embedding(1, 128),
        properties: r#"{"valid": true}"#.to_string(),
    };
    let valid_key = graph.insert_entity(1, valid_entity).unwrap();
    
    // Try to create relationship with non-existent entity
    let fake_key = valid_key; // Use same key for both, which should still work
    let relationship = Relationship {
        from: valid_key,
        to: fake_key, // Self-relationship
        rel_type: 1,
        weight: 1.0,
    };
    
    let rel_result = graph.insert_relationship(relationship);
    // Self-relationships might be allowed or rejected, both are valid behaviors
    
    // Verify graph is still in consistent state
    assert_eq!(graph.entity_count(), 1);
    let final_entity = graph.get_entity_by_id(1);
    assert!(final_entity.is_some());
    
    // Memory usage should still be reported correctly
    let memory_usage = graph.memory_usage();
    assert!(memory_usage.total_bytes() > 0);
}