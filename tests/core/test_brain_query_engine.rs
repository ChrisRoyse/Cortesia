//! Brain query engine integration tests
//! Tests complete query workflow including creating graph, adding entities/concepts, performing various queries
//! Tests query engine returns correct results and performs efficiently through public APIs

use llmkg::core::brain_enhanced_graph::{BrainEnhancedKnowledgeGraph, BrainQueryResult, QueryStatistics};
use llmkg::core::types::{EntityKey, EntityData, Relationship, ContextEntity, QueryResult};
use llmkg::error::Result;
use std::collections::HashMap;
use std::time::Duration;
use tokio;

/// Helper function to create test entity data with specific embedding patterns
fn create_test_entity_data_with_embedding(id: u32, embedding: Vec<f32>, properties: HashMap<String, String>) -> EntityData {
    EntityData {
        type_id: id,
        embedding,
        properties: serde_json::to_string(&properties).unwrap_or_default(),
    }
}

/// Helper function to create clustered embeddings for similarity testing
fn create_clustered_embedding(cluster_id: u32, instance_id: u32, embedding_dim: usize) -> Vec<f32> {
    let mut embedding = vec![0.1; embedding_dim];
    
    match cluster_id {
        1 => {
            // Cluster 1: High values in first third
            for i in 0..(embedding_dim / 3) {
                embedding[i] = 0.8 + (instance_id as f32 * 0.02);
            }
        },
        2 => {
            // Cluster 2: High values in middle third
            let start = embedding_dim / 3;
            let end = 2 * embedding_dim / 3;
            for i in start..end {
                embedding[i] = 0.7 + (instance_id as f32 * 0.02);
            }
        },
        3 => {
            // Cluster 3: High values in last third
            let start = 2 * embedding_dim / 3;
            for i in start..embedding_dim {
                embedding[i] = 0.9 + (instance_id as f32 * 0.02);
            }
        },
        _ => {
            // Random pattern
            for i in 0..embedding_dim {
                embedding[i] = ((cluster_id * instance_id + i as u32) as f32 * 0.01) % 1.0;
            }
        }
    }
    
    embedding
}

#[tokio::test]
async fn test_neural_query_complete_workflow() -> Result<()> {
    // Test 1: Create brain graph and populate with test data
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Create entities with different similarity clusters
    let mut entity_keys = Vec::new();
    
    // Cluster 1 entities (similar to each other)
    for i in 1..4 {
        let embedding = create_clustered_embedding(1, i, 96);
        let mut properties = HashMap::new();
        properties.insert("cluster".to_string(), "1".to_string());
        properties.insert("instance".to_string(), i.to_string());
        properties.insert("type".to_string(), "cluster1".to_string());
        
        let data = create_test_entity_data_with_embedding(i, embedding, properties);
        let key = brain_graph.insert_brain_entity(i, data).await?;
        entity_keys.push((key, 1)); // (key, cluster_id)
    }
    
    // Cluster 2 entities
    for i in 4..7 {
        let embedding = create_clustered_embedding(2, i - 3, 96);
        let mut properties = HashMap::new();
        properties.insert("cluster".to_string(), "2".to_string());
        properties.insert("instance".to_string(), (i - 3).to_string());
        properties.insert("type".to_string(), "cluster2".to_string());
        
        let data = create_test_entity_data_with_embedding(i, embedding, properties);
        let key = brain_graph.insert_brain_entity(i, data).await?;
        entity_keys.push((key, 2));
    }
    
    // Cluster 3 entities
    for i in 7..10 {
        let embedding = create_clustered_embedding(3, i - 6, 96);
        let mut properties = HashMap::new();
        properties.insert("cluster".to_string(), "3".to_string());
        properties.insert("instance".to_string(), (i - 6).to_string());
        properties.insert("type".to_string(), "cluster3".to_string());
        
        let data = create_test_entity_data_with_embedding(i, embedding, properties);
        let key = brain_graph.insert_brain_entity(i, data).await?;
        entity_keys.push((key, 3));
    }
    
    // Test 2: Perform neural query with cluster 1 pattern
    let cluster1_query = create_clustered_embedding(1, 2, 96); // Similar to cluster 1
    let query_result = brain_graph.neural_query(&cluster1_query, 5).await?;
    
    // Verify query found entities
    assert!(!query_result.is_empty());
    assert!(query_result.entity_count() > 0);
    assert!(query_result.query_time > Duration::from_nanos(1));
    
    // Test 3: Verify similarity-based ranking
    let sorted_results = query_result.get_sorted_entities();
    
    // Top results should be from cluster 1
    let top_3_keys: Vec<EntityKey> = sorted_results.iter().take(3).map(|(key, _)| *key).collect();
    let cluster1_keys: Vec<EntityKey> = entity_keys.iter()
        .filter(|(_, cluster)| *cluster == 1)
        .map(|(key, _)| *key)
        .collect();
    
    // Check if top results include cluster 1 entities
    let cluster1_in_top = top_3_keys.iter().any(|key| cluster1_keys.contains(key));
    assert!(cluster1_in_top);
    
    // Test 4: Query with different patterns and verify different results
    let cluster2_query = create_clustered_embedding(2, 2, 96);
    let cluster2_result = brain_graph.neural_query(&cluster2_query, 5).await?;
    
    let cluster3_query = create_clustered_embedding(3, 2, 96);
    let cluster3_result = brain_graph.neural_query(&cluster3_query, 5).await?;
    
    // Results should be different for different query patterns
    assert_ne!(query_result.get_sorted_entities(), cluster2_result.get_sorted_entities());
    assert_ne!(cluster2_result.get_sorted_entities(), cluster3_result.get_sorted_entities());
    
    Ok(())
}

#[tokio::test]
async fn test_activation_propagation_in_queries() -> Result<()> {
    // Test 1: Create brain graph with connected entities
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Create a chain of connected entities
    let mut chain_keys = Vec::new();
    for i in 1..6 {
        let embedding = vec![(i as f32) * 0.1; 96];
        let mut properties = HashMap::new();
        properties.insert("chain_position".to_string(), i.to_string());
        properties.insert("type".to_string(), "chain_entity".to_string());
        
        let data = create_test_entity_data_with_embedding(i, embedding, properties);
        let key = brain_graph.insert_brain_entity(i, data).await?;
        chain_keys.push(key);
        
        // Connect to previous entity in chain
        if i > 1 {
            let relationship = Relationship {
                from: chain_keys[i - 2],
                to: key,
                rel_type: 1,
                weight: 0.8,
            };
            brain_graph.insert_brain_relationship(relationship).await?;
        }
    }
    
    // Test 2: Set high activation on first entity
    brain_graph.set_entity_activation(chain_keys[0], 1.0).await;
    
    // Test 3: Perform query and check activation propagation
    let query_embedding = vec![0.1; 96]; // Similar to first entity
    let query_result = brain_graph.neural_query(&query_embedding, 5).await?;
    
    // Check that activation propagated through the chain
    for i in 0..chain_keys.len() {
        let activation = query_result.get_activation(&chain_keys[i]).unwrap_or(0.0);
        if i == 0 {
            // First entity should have highest activation
            assert!(activation > 0.5);
        } else {
            // Connected entities should have some activation
            assert!(activation > 0.0);
        }
    }
    
    // Test 4: Verify activation decreases with distance
    let first_activation = query_result.get_activation(&chain_keys[0]).unwrap_or(0.0);
    let last_activation = query_result.get_activation(&chain_keys[4]).unwrap_or(0.0);
    assert!(first_activation > last_activation);
    
    // Test 5: Test neural dampening
    brain_graph.set_entity_activation(chain_keys[0], 2.0).await; // Very high activation
    let dampened_result = brain_graph.neural_query(&query_embedding, 5).await?;
    
    // Even with very high initial activation, results should be reasonable
    let max_result_activation = dampened_result.get_sorted_entities()
        .first()
        .map(|(_, activation)| *activation)
        .unwrap_or(0.0);
    assert!(max_result_activation <= 1.2); // Should be dampened
    
    Ok(())
}

#[tokio::test]
async fn test_query_caching_and_performance() -> Result<()> {
    // Test 1: Create brain graph with entities
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Add test entities
    let mut entity_keys = Vec::new();
    for i in 1..11 {
        let embedding = (0..96).map(|j| (i * j) as f32 * 0.001).collect();
        let mut properties = HashMap::new();
        properties.insert("id".to_string(), i.to_string());
        
        let data = create_test_entity_data_with_embedding(i, embedding, properties);
        let key = brain_graph.insert_brain_entity(i, data).await?;
        entity_keys.push(key);
    }
    
    // Test 2: Perform initial query and measure time
    let query_embedding = (0..96).map(|i| i as f32 * 0.01).collect::<Vec<f32>>();
    
    let start_time = std::time::Instant::now();
    let first_result = brain_graph.neural_query(&query_embedding, 5).await?;
    let first_query_time = start_time.elapsed();
    
    assert!(!first_result.is_empty());
    assert!(first_query_time > Duration::from_nanos(1));
    
    // Test 3: Perform same query again (should be cached)
    let start_time = std::time::Instant::now();
    let cached_result = brain_graph.neural_query(&query_embedding, 5).await?;
    let cached_query_time = start_time.elapsed();
    
    // Results should be identical
    assert_eq!(first_result.entity_count(), cached_result.entity_count());
    assert_eq!(first_result.total_activation, cached_result.total_activation);
    
    // Cached query might be faster (implementation dependent)
    // We just verify it completes successfully
    assert!(cached_query_time > Duration::from_nanos(1));
    
    // Test 4: Test different query parameters to verify cache specificity
    let different_k_result = brain_graph.neural_query(&query_embedding, 3).await?;
    let different_embedding = (0..96).map(|i| (i + 1) as f32 * 0.01).collect::<Vec<f32>>();
    let different_embedding_result = brain_graph.neural_query(&different_embedding, 5).await?;
    
    // Should get different results for different parameters
    assert_ne!(first_result.entity_count(), different_k_result.entity_count());
    assert_ne!(first_result.get_sorted_entities(), different_embedding_result.get_sorted_entities());
    
    // Test 5: Check query statistics
    let query_stats = brain_graph.get_query_statistics().await;
    assert!(query_stats.total_queries >= 4); // We performed 4 queries
    assert!(query_stats.average_query_time > 0.0);
    assert!(query_stats.cache_hit_rate >= 0.0 && query_stats.cache_hit_rate <= 1.0);
    
    Ok(())
}

#[tokio::test]
async fn test_context_aware_querying() -> Result<()> {
    // Test 1: Create brain graph with contextual entities
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    // Create entities representing different contexts
    let mut context_entities = HashMap::new();
    
    // Science context
    let science_keywords = vec!["physics", "chemistry", "biology"];
    for (i, keyword) in science_keywords.iter().enumerate() {
        let embedding = create_clustered_embedding(1, i as u32 + 1, 96);
        let mut properties = HashMap::new();
        properties.insert("context".to_string(), "science".to_string());
        properties.insert("keyword".to_string(), keyword.to_string());
        properties.insert("domain".to_string(), "academic".to_string());
        
        let data = create_test_entity_data_with_embedding(i as u32 + 1, embedding, properties);
        let key = brain_graph.insert_brain_entity(i as u32 + 1, data).await?;
        context_entities.insert(format!("science_{}", keyword), key);
    }
    
    // Technology context
    let tech_keywords = vec!["programming", "algorithms", "databases"];
    for (i, keyword) in tech_keywords.iter().enumerate() {
        let embedding = create_clustered_embedding(2, i as u32 + 1, 96);
        let mut properties = HashMap::new();
        properties.insert("context".to_string(), "technology".to_string());
        properties.insert("keyword".to_string(), keyword.to_string());
        properties.insert("domain".to_string(), "technical".to_string());
        
        let data = create_test_entity_data_with_embedding(i as u32 + 4, embedding, properties);
        let key = brain_graph.insert_brain_entity(i as u32 + 4, data).await?;
        context_entities.insert(format!("tech_{}", keyword), key);
    }
    
    // Create cross-context relationships
    let physics_key = context_entities["science_physics"];
    let algorithms_key = context_entities["tech_algorithms"];
    
    let cross_context_relationship = Relationship {
        from: physics_key,
        to: algorithms_key,
        rel_type: 2, // Cross-domain relationship
        weight: 0.6,
    };
    brain_graph.insert_brain_relationship(cross_context_relationship).await?;
    
    // Test 2: Query with science context
    let science_query = create_clustered_embedding(1, 2, 96);
    let science_result = brain_graph.neural_query(&science_query, 5).await?;
    
    // Should find science-related entities first
    let sorted_science = science_result.get_sorted_entities();
    assert!(!sorted_science.is_empty());
    
    // Test 3: Query with technology context
    let tech_query = create_clustered_embedding(2, 2, 96);
    let tech_result = brain_graph.neural_query(&tech_query, 5).await?;
    
    let sorted_tech = tech_result.get_sorted_entities();
    assert!(!sorted_tech.is_empty());
    
    // Results should be different for different contexts
    assert_ne!(sorted_science[0].0, sorted_tech[0].0);
    
    // Test 4: Query for cross-context concepts
    let cross_context_query = vec![0.5; 96]; // Neutral query
    let cross_result = brain_graph.neural_query(&cross_context_query, 6).await?;
    
    // Should find entities from both contexts
    assert!(cross_result.entity_count() > 3);
    
    // Test 5: Verify relationship influence on context queries
    brain_graph.set_entity_activation(physics_key, 1.0).await;
    
    let influenced_result = brain_graph.neural_query(&science_query, 5).await?;
    let algorithms_activation = influenced_result.get_activation(&algorithms_key).unwrap_or(0.0);
    
    // Algorithm entity should have some activation due to relationship
    assert!(algorithms_activation > 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_query_result_filtering_and_ranking() -> Result<()> {
    // Test 1: Create brain graph with diverse entities
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let mut all_entities = Vec::new();
    
    // Create entities with different quality scores
    let quality_scores = vec![0.9, 0.7, 0.5, 0.3, 0.8, 0.6, 0.4, 0.2, 0.95, 0.1];
    
    for (i, &quality) in quality_scores.iter().enumerate() {
        let embedding = vec![quality; 96];
        let mut properties = HashMap::new();
        properties.insert("quality_score".to_string(), quality.to_string());
        properties.insert("rank".to_string(), i.to_string());
        properties.insert("type".to_string(), "ranked_entity".to_string());
        
        let data = create_test_entity_data_with_embedding(i as u32 + 1, embedding, properties);
        let key = brain_graph.insert_brain_entity(i as u32 + 1, data).await?;
        all_entities.push((key, quality));
    }
    
    // Test 2: Query all entities and verify ranking
    let query_embedding = vec![0.6; 96]; // Mid-range query
    let all_results = brain_graph.neural_query(&query_embedding, 10).await?;
    
    assert_eq!(all_results.entity_count(), 10);
    
    // Test 3: Filter by activation threshold
    let high_quality = all_results.get_entities_above_threshold(0.7);
    let medium_quality = all_results.get_entities_above_threshold(0.4);
    let all_quality = all_results.get_entities_above_threshold(0.1);
    
    // Should have fewer high-quality than medium-quality entities
    assert!(high_quality.len() <= medium_quality.len());
    assert!(medium_quality.len() <= all_quality.len());
    assert_eq!(all_quality.len(), 10); // All entities above 0.1
    
    // Test 4: Test top-k selection with different k values
    let top_1 = all_results.get_top_k(1);
    let top_3 = all_results.get_top_k(3);
    let top_5 = all_results.get_top_k(5);
    
    assert_eq!(top_1.len(), 1);
    assert_eq!(top_3.len(), 3);
    assert_eq!(top_5.len(), 5);
    
    // Top entities should have higher activation
    assert!(top_1[0].1 >= top_3[2].1);
    assert!(top_3[2].1 >= top_5[4].1);
    
    // Test 5: Verify ranking consistency
    let sorted_entities = all_results.get_sorted_entities();
    
    // Check that sorting is consistent (descending order)
    for i in 1..sorted_entities.len() {
        assert!(sorted_entities[i - 1].1 >= sorted_entities[i].1);
    }
    
    // Test 6: Test empty result filtering
    let impossible_threshold = all_results.get_entities_above_threshold(2.0);
    assert_eq!(impossible_threshold.len(), 0);
    
    let excessive_k = all_results.get_top_k(20); // More than available
    assert_eq!(excessive_k.len(), 10); // Should return all available
    
    Ok(())
}

#[tokio::test]
async fn test_query_performance_optimization() -> Result<()> {
    // Test 1: Create large brain graph for performance testing
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let mut entity_keys = Vec::new();
    
    // Create 50 entities with varied embeddings
    for i in 1..51 {
        let embedding = (0..96).map(|j| ((i * j) as f32 * 0.0001) % 1.0).collect();
        let mut properties = HashMap::new();
        properties.insert("batch_id".to_string(), (i / 10).to_string());
        properties.insert("index".to_string(), i.to_string());
        
        let data = create_test_entity_data_with_embedding(i, embedding, properties);
        let key = brain_graph.insert_brain_entity(i, data).await?;
        entity_keys.push(key);
    }
    
    // Add some relationships for realistic graph structure
    for i in 0..40 {
        if i + 10 < entity_keys.len() {
            let relationship = Relationship {
                from: entity_keys[i],
                to: entity_keys[i + 10],
                rel_type: 1,
                weight: 0.5,
            };
            brain_graph.insert_brain_relationship(relationship).await?;
        }
    }
    
    // Test 2: Benchmark different query sizes
    let query_embedding = (0..96).map(|i| i as f32 * 0.01).collect::<Vec<f32>>();
    
    let mut query_times = Vec::new();
    let k_values = vec![1, 5, 10, 20, 30];
    
    for &k in &k_values {
        let start_time = std::time::Instant::now();
        let result = brain_graph.neural_query(&query_embedding, k).await?;
        let query_time = start_time.elapsed();
        
        query_times.push((k, query_time, result.entity_count()));
        
        // Verify we get the requested number of entities (or all available)
        assert_eq!(result.entity_count(), std::cmp::min(k, 50));
    }
    
    // Test 3: Verify query times are reasonable and scale appropriately
    for (k, time, count) in &query_times {
        assert!(time < Duration::from_secs(1)); // Should complete within 1 second
        assert_eq!(*count, std::cmp::min(*k, 50));
    }
    
    // Test 4: Test query with activation propagation
    brain_graph.set_entity_activation(entity_keys[0], 1.0).await;
    
    let start_time = std::time::Instant::now();
    let propagated_result = brain_graph.neural_query(&query_embedding, 10).await?;
    let propagation_time = start_time.elapsed();
    
    assert!(propagation_time < Duration::from_secs(2)); // Should handle propagation efficiently
    assert!(!propagated_result.is_empty());
    
    // Test 5: Test concurrent queries
    let mut query_handles = Vec::new();
    
    for i in 0..5 {
        let graph_clone = std::sync::Arc::new(&brain_graph);
        let query_embedding_clone = query_embedding.clone();
        
        let handle = tokio::spawn(async move {
            let start = std::time::Instant::now();
            let result = graph_clone.neural_query(&query_embedding_clone, 5).await;
            let duration = start.elapsed();
            (result, duration)
        });
        
        query_handles.push(handle);
    }
    
    // Wait for all concurrent queries
    let mut concurrent_results = Vec::new();
    for handle in query_handles {
        let (result, duration) = handle.await.unwrap();
        assert!(result.is_ok());
        assert!(duration < Duration::from_secs(1));
        concurrent_results.push(result.unwrap());
    }
    
    // All concurrent queries should return consistent results
    let first_result = &concurrent_results[0];
    for result in &concurrent_results[1..] {
        assert_eq!(result.entity_count(), first_result.entity_count());
        assert_eq!(result.total_activation, first_result.total_activation);
    }
    
    // Test 6: Check final performance statistics
    let final_stats = brain_graph.get_query_statistics().await;
    assert!(final_stats.total_queries > 10); // We performed many queries
    assert!(final_stats.average_query_time > 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_standard_query_compatibility() -> Result<()> {
    // Test 1: Create brain graph with entities
    let brain_graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    
    let mut entity_keys = Vec::new();
    
    for i in 1..6 {
        let embedding = (0..96).map(|j| (i * j) as f32 * 0.001).collect();
        let mut properties = HashMap::new();
        properties.insert("id".to_string(), i.to_string());
        properties.insert("type".to_string(), "standard_test".to_string());
        
        let data = create_test_entity_data_with_embedding(i, embedding, properties);
        let key = brain_graph.insert_brain_entity(i, data).await?;
        entity_keys.push(key);
    }
    
    // Add relationships
    for i in 0..4 {
        let relationship = Relationship {
            from: entity_keys[i],
            to: entity_keys[i + 1],
            rel_type: 1,
            weight: 0.7,
        };
        brain_graph.insert_brain_relationship(relationship).await?;
    }
    
    // Test 2: Use standard query method (compatibility wrapper)
    let query_embedding = (0..96).map(|i| i as f32 * 0.01).collect::<Vec<f32>>();
    let context_entities = vec![]; // Empty context for simplicity
    
    let standard_result = brain_graph.query(&query_embedding, &context_entities, 3).await?;
    
    // Test 3: Verify standard query result structure
    assert!(!standard_result.entities.is_empty());
    assert!(standard_result.entities.len() <= 3);
    assert!(standard_result.confidence > 0.0);
    assert!(standard_result.query_time_ms > 0);
    
    // Test 4: Compare with neural query
    let neural_result = brain_graph.neural_query(&query_embedding, 3).await?;
    
    // Results should be related (both should find entities)
    assert_eq!(standard_result.entities.len(), neural_result.entity_count());
    
    // Test 5: Verify standard query result format
    for context_entity in &standard_result.entities {
        // Check that entity has required fields
        assert!(context_entity.similarity > 0.0);
        assert!(!context_entity.properties.is_empty());
        
        // Properties should be valid JSON
        let parsed_properties: HashMap<String, String> = 
            serde_json::from_str(&context_entity.properties).unwrap();
        assert!(parsed_properties.contains_key("id"));
        assert!(parsed_properties.contains_key("type"));
    }
    
    // Test 6: Test with context entities
    let context_entity = ContextEntity {
        id: entity_keys[0],
        similarity: 0.9,
        neighbors: vec![entity_keys[1]],
        properties: r#"{"context": "test"}"#.to_string(),
    };
    
    let context_entities = vec![context_entity];
    let contextual_result = brain_graph.query(&query_embedding, &context_entities, 3).await?;
    
    // Should still return valid results
    assert!(!contextual_result.entities.is_empty());
    assert!(contextual_result.confidence > 0.0);
    
    // Test 7: Verify relationship information in standard query
    assert!(!standard_result.relationships.is_empty());
    
    for relationship in &standard_result.relationships {
        assert!(relationship.weight > 0.0);
        assert!(relationship.rel_type > 0);
    }
    
    Ok(())
}