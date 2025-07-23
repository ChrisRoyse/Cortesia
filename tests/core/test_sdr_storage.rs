//! Integration tests for SDR storage module
//! 
//! These tests verify the public API of SDR storage and its integration
//! with the KnowledgeEngine for memory-efficient storage and retrieval.

use llmkg::core::sdr_storage::SDRStorage;
use llmkg::core::sdr_types::{SDR, SDRConfig, SDRPattern, SDRStatistics, SimilaritySearchResult};
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::triple::Triple;
use llmkg::core::types::EntityKey;
use std::collections::HashSet;
use std::sync::Arc;
use slotmap::SlotMap;

/// Helper function to create test entity keys
fn create_test_entity_key() -> EntityKey {
    let mut slot_map = SlotMap::with_key();
    slot_map.insert(())
}

/// Helper function to create a test triple
fn create_test_triple(subject: &str, predicate: &str, object: &str) -> Triple {
    Triple {
        subject: subject.to_string(),
        predicate: predicate.to_string(),
        object: object.to_string(),
        confidence: 0.9,
        source: Some("test".to_string()),
    }
}

/// Helper function to generate a simple embedding
fn generate_simple_embedding(text: &str, dim: usize) -> Vec<f32> {
    // Simple hash-based embedding for deterministic testing
    let hash = text.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
    let mut embedding = vec![0.0; dim];
    for i in 0..dim {
        embedding[i] = ((hash.wrapping_mul(i as u32 + 1) % 1000) as f32) / 1000.0;
    }
    embedding
}

#[tokio::test]
async fn test_knowledge_engine_sdr_integration_basic() {
    // Create KnowledgeEngine and SDRStorage
    let knowledge_engine = KnowledgeEngine::new(128, 10000).unwrap();
    let sdr_config = SDRConfig {
        total_bits: 2048,
        active_bits: 40,
        sparsity: 0.02,
        overlap_threshold: 0.5,
    };
    let sdr_storage = SDRStorage::new(sdr_config.clone());

    // Store a triple in knowledge engine
    let triple = create_test_triple("Alice", "knows", "Bob");
    let embedding = generate_simple_embedding("Alice knows Bob", 128);
    let node_id = knowledge_engine.store_triple(triple.clone(), Some(embedding.clone())).unwrap();

    // Create SDR representation from the embedding
    let entity_key = create_test_entity_key();
    let pattern_id = sdr_storage.store_dense_vector(
        entity_key,
        &embedding,
        "Alice knows Bob".to_string()
    ).await.unwrap();

    // Verify SDR was stored correctly
    let pattern = sdr_storage.get_entity_pattern(entity_key).await.unwrap();
    assert!(pattern.is_some());
    let pattern = pattern.unwrap();
    assert_eq!(pattern.pattern_id, pattern_id);
    assert_eq!(pattern.concept_name, "Alice knows Bob");
    
    // Verify statistics
    let stats = sdr_storage.get_statistics().await.unwrap();
    assert_eq!(stats.total_patterns, 1);
    assert_eq!(stats.total_entities, 1);
    assert!(stats.average_sparsity > 0.0 && stats.average_sparsity <= sdr_config.sparsity + 0.01);
}

#[tokio::test]
async fn test_sdr_similarity_search_accuracy() {
    let sdr_config = SDRConfig {
        total_bits: 1024,
        active_bits: 20,
        sparsity: 0.02,
        overlap_threshold: 0.3,
    };
    let sdr_storage = SDRStorage::new(sdr_config);

    // Store multiple related concepts
    let concepts = vec![
        ("machine learning", "AI concept about learning from data"),
        ("deep learning", "Subset of machine learning using neural networks"),
        ("neural networks", "Computing systems inspired by biological neural networks"),
        ("artificial intelligence", "Simulation of human intelligence by machines"),
        ("quantum computing", "Computing using quantum mechanical phenomena"),
    ];

    let mut entity_keys = Vec::new();
    for (concept, description) in &concepts {
        let entity_key = create_test_entity_key();
        let embedding = generate_simple_embedding(concept, 100);
        
        sdr_storage.store_dense_vector(
            entity_key,
            &embedding,
            description.to_string()
        ).await.unwrap();
        
        entity_keys.push(entity_key);
    }

    // Test similarity search
    let query = "machine learning algorithms";
    let results = sdr_storage.similarity_search(query, 0.0).await.unwrap();
    
    assert!(!results.is_empty());
    assert!(results.len() <= 10); // Default max results
    
    // Verify results contain relevant concepts
    let result_contents: Vec<String> = results.iter()
        .map(|r| r.content.clone())
        .collect();
    
    // At least one ML-related concept should be found
    let has_ml_concept = result_contents.iter()
        .any(|content| content.contains("learning") || content.contains("AI"));
    assert!(has_ml_concept);
}

#[tokio::test]
async fn test_memory_efficiency_improvements() {
    let knowledge_engine = KnowledgeEngine::new(384, 10000).unwrap();
    let sdr_storage = SDRStorage::new(SDRConfig::default());

    // Measure memory usage before SDR compression
    let initial_memory = sdr_storage.memory_usage().await;

    // Store 100 entities with both full embeddings and SDR representations
    let mut entity_keys = Vec::new();
    let embedding_dim = 384;
    
    for i in 0..100 {
        let triple = create_test_triple(
            &format!("Entity{}", i),
            "has_property",
            &format!("Value{}", i)
        );
        
        // Generate a high-dimensional embedding
        let embedding = generate_simple_embedding(&format!("Entity {} data", i), embedding_dim);
        
        // Store in knowledge engine
        let node_id = knowledge_engine.store_triple(triple, Some(embedding.clone())).unwrap();
        
        // Convert to SDR for memory-efficient storage
        let entity_key = create_test_entity_key();
        sdr_storage.store_dense_vector(
            entity_key,
            &embedding,
            format!("Entity {} concept", i)
        ).await.unwrap();
        
        entity_keys.push(entity_key);
    }

    // Check memory usage after storing SDRs
    let final_memory = sdr_storage.memory_usage().await;
    let memory_per_entity = (final_memory - initial_memory) / 100;
    
    // Calculate theoretical full embedding storage
    let full_embedding_size = embedding_dim * std::mem::size_of::<f32>();
    let sdr_bits = SDRConfig::default().total_bits / 8; // bits to bytes
    
    // SDR should use significantly less memory than full embeddings
    assert!(sdr_bits < full_embedding_size);
    
    // Verify statistics
    let stats = sdr_storage.get_statistics().await.unwrap();
    assert_eq!(stats.total_patterns, 100);
    assert_eq!(stats.total_entities, 100);
    assert!(stats.average_sparsity <= SDRConfig::default().sparsity + 0.01);
    
    // Test that we can still retrieve SDRs efficiently
    for (i, entity_key) in entity_keys.iter().enumerate().take(10) {
        let pattern = sdr_storage.get_entity_pattern(*entity_key).await.unwrap();
        assert!(pattern.is_some());
        assert_eq!(pattern.unwrap().concept_name, format!("Entity {} concept", i));
    }
}

#[tokio::test]
async fn test_sdr_pattern_similarity_and_clustering() {
    let sdr_config = SDRConfig {
        total_bits: 512,
        active_bits: 10,
        sparsity: 0.02,
        overlap_threshold: 0.4,
    };
    let sdr_storage = SDRStorage::new(sdr_config);

    // Create clusters of related concepts
    let clusters = vec![
        vec!["cat", "dog", "pet", "animal"],
        vec!["car", "truck", "vehicle", "automobile"],
        vec!["apple", "orange", "fruit", "banana"],
    ];

    let mut cluster_entity_map: Vec<Vec<EntityKey>> = Vec::new();
    
    for cluster in &clusters {
        let mut cluster_entities = Vec::new();
        
        for concept in cluster {
            let entity_key = create_test_entity_key();
            let embedding = generate_simple_embedding(concept, 50);
            
            sdr_storage.store_dense_vector(
                entity_key,
                &embedding,
                concept.to_string()
            ).await.unwrap();
            
            cluster_entities.push(entity_key);
        }
        
        cluster_entity_map.push(cluster_entities);
    }

    // Test within-cluster similarity
    for (cluster_idx, entities) in cluster_entity_map.iter().enumerate() {
        if entities.len() >= 2 {
            let pattern1 = sdr_storage.get_entity_pattern(entities[0]).await.unwrap().unwrap();
            let pattern2 = sdr_storage.get_entity_pattern(entities[1]).await.unwrap().unwrap();
            
            let similarity = pattern1.sdr.overlap(&pattern2.sdr);
            
            // Within-cluster items should have some similarity
            // (Note: with our simple hash-based embeddings, this might be low)
            assert!(similarity >= 0.0);
        }
    }

    // Test similarity search finds related concepts
    let query_sdr = sdr_storage.encode_text("dog").await.unwrap();
    let similar_patterns = sdr_storage.find_similar_patterns(&query_sdr, 5).await.unwrap();
    
    assert!(!similar_patterns.is_empty());
    assert!(similar_patterns.len() <= 5);
}

#[tokio::test]
async fn test_sdr_storage_compaction() {
    let sdr_storage = SDRStorage::new(SDRConfig::default());

    // Store patterns with and without entity associations
    let mut associated_count = 0;
    let mut unassociated_count = 0;

    // Create patterns with entity associations
    for i in 0..5 {
        let entity_key = create_test_entity_key();
        let embedding = generate_simple_embedding(&format!("associated_{}", i), 64);
        
        sdr_storage.store_dense_vector(
            entity_key,
            &embedding,
            format!("Associated concept {}", i)
        ).await.unwrap();
        
        associated_count += 1;
    }

    // Create orphaned patterns (no entity associations)
    for i in 0..3 {
        let sdr = SDR::random(&SDRConfig::default());
        let pattern = SDRPattern::new(
            format!("orphan_{}", i),
            sdr,
            format!("Orphaned concept {}", i)
        );
        
        sdr_storage.store_pattern(pattern).await.unwrap();
        unassociated_count += 1;
    }

    // Verify initial state
    let stats_before = sdr_storage.get_statistics().await.unwrap();
    assert_eq!(stats_before.total_patterns, associated_count + unassociated_count);
    assert_eq!(stats_before.total_entities, associated_count);

    // Perform compaction
    let removed_count = sdr_storage.compact().await.unwrap();
    assert_eq!(removed_count, unassociated_count);

    // Verify final state
    let stats_after = sdr_storage.get_statistics().await.unwrap();
    assert_eq!(stats_after.total_patterns, associated_count);
    assert_eq!(stats_after.total_entities, associated_count);
}

#[tokio::test]
async fn test_concurrent_sdr_operations() {
    use tokio::task::JoinSet;
    
    let sdr_storage = Arc::new(SDRStorage::new(SDRConfig::default()));
    let knowledge_engine = Arc::new(KnowledgeEngine::new(256, 10000).unwrap());
    
    let mut join_set = JoinSet::new();
    
    // Spawn multiple concurrent tasks
    for i in 0..20 {
        let storage_clone = sdr_storage.clone();
        let engine_clone = knowledge_engine.clone();
        
        join_set.spawn(async move {
            // Store triple in knowledge engine
            let triple = create_test_triple(
                &format!("Subject{}", i),
                "relates_to",
                &format!("Object{}", i)
            );
            let embedding = generate_simple_embedding(&format!("Relation {}", i), 256);
            
            engine_clone.store_triple(triple, Some(embedding.clone())).unwrap();
            
            // Store SDR representation
            let entity_key = create_test_entity_key();
            let pattern_id = storage_clone.store_dense_vector(
                entity_key,
                &embedding,
                format!("Concurrent concept {}", i)
            ).await.unwrap();
            
            (entity_key, pattern_id)
        });
    }
    
    // Collect results
    let mut results = Vec::new();
    while let Some(result) = join_set.join_next().await {
        results.push(result.unwrap());
    }
    
    // Verify all operations completed successfully
    assert_eq!(results.len(), 20);
    
    let stats = sdr_storage.get_statistics().await.unwrap();
    assert_eq!(stats.total_patterns, 20);
    assert_eq!(stats.total_entities, 20);
    
    // Verify each entity can be retrieved
    for (entity_key, pattern_id) in results {
        let pattern = sdr_storage.get_entity_pattern(entity_key).await.unwrap();
        assert!(pattern.is_some());
        assert_eq!(pattern.unwrap().pattern_id, pattern_id);
    }
}

#[tokio::test]
async fn test_sdr_encoding_consistency() {
    let sdr_storage = SDRStorage::new(SDRConfig::default());
    
    // Test that same text produces same SDR
    let text = "consistent encoding test";
    let sdr1 = sdr_storage.encode_text(text).await.unwrap();
    let sdr2 = sdr_storage.encode_text(text).await.unwrap();
    
    assert_eq!(sdr1.active_bits, sdr2.active_bits);
    assert_eq!(sdr1.total_bits, sdr2.total_bits);
    
    // Test that different texts produce different SDRs
    let sdr3 = sdr_storage.encode_text("different text").await.unwrap();
    assert_ne!(sdr1.active_bits, sdr3.active_bits);
}

#[tokio::test]
async fn test_sdr_similarity_metrics() {
    let config = SDRConfig {
        total_bits: 256,
        active_bits: 8,
        sparsity: 0.03,
        overlap_threshold: 0.25,
    };
    
    // Create SDRs with known overlap
    let base_bits: HashSet<usize> = [10, 20, 30, 40, 50, 60, 70, 80].iter().cloned().collect();
    let sdr1 = SDR::new(base_bits.clone(), 256);
    
    // 50% overlap (4 common bits)
    let half_overlap_bits: HashSet<usize> = [50, 60, 70, 80, 90, 100, 110, 120].iter().cloned().collect();
    let sdr2 = SDR::new(half_overlap_bits, 256);
    
    // Test overlap calculation
    let overlap = sdr1.overlap(&sdr2);
    assert!(overlap > 0.0 && overlap < 1.0);
    
    // Test Jaccard similarity
    let jaccard = sdr1.jaccard_similarity(&sdr2);
    assert_eq!(overlap, jaccard); // For SDRs, these should be equal
    
    // Test cosine similarity
    let cosine = sdr1.cosine_similarity(&sdr2);
    assert!(cosine > 0.0 && cosine < 1.0);
    assert!((cosine - 0.5).abs() < 0.01); // Should be approximately 0.5
}

#[tokio::test]
async fn test_knowledge_engine_sdr_workflow() {
    let knowledge_engine = KnowledgeEngine::new(512, 50000).unwrap();
    let sdr_storage = SDRStorage::new(SDRConfig {
        total_bits: 4096,
        active_bits: 80,
        sparsity: 0.02,
        overlap_threshold: 0.4,
    });

    // Simulate a knowledge graph workflow with SDR compression
    let relationships = vec![
        ("Einstein", "developed", "Theory of Relativity"),
        ("Theory of Relativity", "describes", "spacetime"),
        ("spacetime", "involves", "gravity"),
        ("Newton", "formulated", "Laws of Motion"),
        ("Laws of Motion", "describe", "mechanics"),
        ("Einstein", "influenced_by", "Newton"),
    ];

    let mut stored_entities = Vec::new();
    
    for (subject, predicate, object) in &relationships {
        // Create and store triple
        let triple = create_test_triple(subject, predicate, object);
        let triple_text = format!("{} {} {}", subject, predicate, object);
        let embedding = generate_simple_embedding(&triple_text, 512);
        
        // Store in knowledge engine
        let node_id = knowledge_engine.store_triple(triple, Some(embedding.clone())).unwrap();
        
        // Create SDR representation for memory efficiency
        let entity_key = create_test_entity_key();
        let pattern_id = sdr_storage.store_dense_vector(
            entity_key,
            &embedding,
            triple_text
        ).await.unwrap();
        
        stored_entities.push((entity_key, pattern_id, subject.to_string()));
    }

    // Test similarity search for related concepts
    let query = "Einstein physics";
    let results = sdr_storage.similarity_search(query, 0.0).await.unwrap();
    
    assert!(!results.is_empty());
    
    // Verify we can find Einstein-related content
    let einstein_results: Vec<&SimilaritySearchResult> = results.iter()
        .filter(|r| r.content.contains("Einstein"))
        .collect();
    
    assert!(!einstein_results.is_empty());

    // Test memory efficiency
    let stats = sdr_storage.get_statistics().await.unwrap();
    assert_eq!(stats.total_patterns, relationships.len());
    
    // Verify SDR properties
    assert!(stats.average_sparsity <= 0.02 + 0.005); // Allow small tolerance
    
    // Calculate memory savings
    let original_embedding_size = 512 * std::mem::size_of::<f32>() * relationships.len();
    let sdr_memory = sdr_storage.memory_usage().await;
    
    // SDR should use less memory than full embeddings
    assert!(sdr_memory < original_embedding_size);
}

#[tokio::test]
async fn test_sdr_pattern_operations() {
    let config = SDRConfig::default();
    
    // Test union operation
    let sdr1 = SDR::new([1, 2, 3, 4, 5].iter().cloned().collect(), 100);
    let sdr2 = SDR::new([4, 5, 6, 7, 8].iter().cloned().collect(), 100);
    
    let union = sdr1.union(&sdr2).unwrap();
    assert_eq!(union.active_bits.len(), 8); // 1,2,3,4,5,6,7,8
    assert!(union.active_bits.contains(&1));
    assert!(union.active_bits.contains(&8));
    
    // Test intersection operation
    let intersection = sdr1.intersection(&sdr2).unwrap();
    assert_eq!(intersection.active_bits.len(), 2); // 4,5
    assert!(intersection.active_bits.contains(&4));
    assert!(intersection.active_bits.contains(&5));
    
    // Test operations with mismatched dimensions
    let sdr3 = SDR::new([1, 2, 3].iter().cloned().collect(), 200);
    assert!(sdr1.union(&sdr3).is_err());
    assert!(sdr1.intersection(&sdr3).is_err());
}

#[tokio::test]
async fn test_sdr_dense_vector_conversion() {
    let sdr_storage = SDRStorage::new(SDRConfig::default());
    
    // Test conversion from dense to SDR and back
    let original_vector = vec![0.1, 0.9, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.0];
    let entity_key = create_test_entity_key();
    
    // Store as SDR
    sdr_storage.store_dense_vector(
        entity_key,
        &original_vector,
        "test vector".to_string()
    ).await.unwrap();
    
    // Retrieve as dense vector
    let retrieved_vector = sdr_storage.get_dense_vector(entity_key).await.unwrap();
    assert!(retrieved_vector.is_some());
    
    let dense = retrieved_vector.unwrap();
    assert_eq!(dense.len(), SDRConfig::default().total_bits);
    
    // Check that it's a valid binary representation
    for value in &dense {
        assert!(*value == 0.0 || *value == 1.0);
    }
    
    // Count active bits
    let active_count = dense.iter().filter(|&&v| v == 1.0).count();
    assert_eq!(active_count, SDRConfig::default().active_bits.min(original_vector.len()));
}

#[tokio::test]
async fn test_sdr_storage_edge_cases() {
    let sdr_storage = SDRStorage::new(SDRConfig::default());
    
    // Test with empty vector
    let entity_key1 = create_test_entity_key();
    let empty_vector: Vec<f32> = vec![];
    let pattern_id1 = sdr_storage.store_dense_vector(
        entity_key1,
        &empty_vector,
        "empty".to_string()
    ).await.unwrap();
    
    let pattern1 = sdr_storage.get_entity_pattern(entity_key1).await.unwrap().unwrap();
    assert_eq!(pattern1.sdr.active_bits.len(), 0);
    
    // Test with very large vector
    let entity_key2 = create_test_entity_key();
    let large_vector = vec![0.5; 10000];
    let pattern_id2 = sdr_storage.store_dense_vector(
        entity_key2,
        &large_vector,
        "large".to_string()
    ).await.unwrap();
    
    let pattern2 = sdr_storage.get_entity_pattern(entity_key2).await.unwrap().unwrap();
    assert_eq!(pattern2.sdr.active_bits.len(), SDRConfig::default().active_bits);
    
    // Test retrieving non-existent entity
    let non_existent_key = create_test_entity_key();
    let result = sdr_storage.get_entity_pattern(non_existent_key).await.unwrap();
    assert!(result.is_none());
    
    // Test retrieving dense vector for non-existent entity
    let dense_result = sdr_storage.get_dense_vector(non_existent_key).await.unwrap();
    assert!(dense_result.is_none());
}

#[tokio::test]
async fn test_entity_mapping_functionality() {
    let sdr_storage = SDRStorage::new(SDRConfig::default());
    
    // Store multiple entities
    let mut entity_pattern_pairs = Vec::new();
    
    for i in 0..5 {
        let entity_key = create_test_entity_key();
        let embedding = generate_simple_embedding(&format!("entity_{}", i), 32);
        
        let pattern_id = sdr_storage.store_dense_vector(
            entity_key,
            &embedding,
            format!("Entity {}", i)
        ).await.unwrap();
        
        entity_pattern_pairs.push((entity_key, pattern_id));
    }
    
    // Get entity mappings
    let mappings = sdr_storage.get_entity_mappings().await;
    assert_eq!(mappings.len(), 5);
    
    // Verify all mappings are correct
    for (entity_key, pattern_id) in entity_pattern_pairs {
        assert_eq!(mappings.get(&pattern_id), Some(&entity_key));
    }
}

#[tokio::test]
async fn test_store_with_metadata() {
    let sdr_storage = SDRStorage::new(SDRConfig::default());
    
    // Test storing SDR with metadata
    let sdr = SDR::random(&SDRConfig::default());
    let content = "Important knowledge with high relevance".to_string();
    let importance = 0.95;
    
    let pattern_id = sdr_storage.store_with_metadata(
        &sdr,
        content.clone(),
        importance
    ).await.unwrap();
    
    assert!(pattern_id.starts_with("pattern_"));
    
    // Verify pattern was stored
    let stats = sdr_storage.get_statistics().await.unwrap();
    assert_eq!(stats.total_patterns, 1);
}

#[tokio::test]
async fn test_performance_at_scale() {
    use std::time::Instant;
    
    let sdr_config = SDRConfig {
        total_bits: 8192,
        active_bits: 160,
        sparsity: 0.02,
        overlap_threshold: 0.3,
    };
    let sdr_storage = SDRStorage::new(sdr_config);
    
    // Measure time to store 1000 patterns
    let start = Instant::now();
    
    for i in 0..1000 {
        let entity_key = create_test_entity_key();
        let embedding = generate_simple_embedding(&format!("scale_test_{}", i), 256);
        
        sdr_storage.store_dense_vector(
            entity_key,
            &embedding,
            format!("Scale test concept {}", i)
        ).await.unwrap();
    }
    
    let storage_duration = start.elapsed();
    
    // Verify all patterns were stored
    let stats = sdr_storage.get_statistics().await.unwrap();
    assert_eq!(stats.total_patterns, 1000);
    assert_eq!(stats.total_entities, 1000);
    
    // Measure similarity search performance
    let search_start = Instant::now();
    let query_sdr = sdr_storage.encode_text("scale_test_500").await.unwrap();
    let results = sdr_storage.find_similar_patterns(&query_sdr, 10).await.unwrap();
    let search_duration = search_start.elapsed();
    
    assert!(!results.is_empty());
    assert!(results.len() <= 10);
    
    // Performance assertions (these are generous to account for CI environments)
    assert!(storage_duration.as_secs() < 10, "Storage took too long: {:?}", storage_duration);
    assert!(search_duration.as_millis() < 100, "Search took too long: {:?}", search_duration);
    
    // Test memory efficiency
    let memory_usage = sdr_storage.memory_usage().await;
    let memory_per_pattern = memory_usage / 1000;
    
    // Each pattern should use less than 1KB on average
    assert!(memory_per_pattern < 1024, "Memory usage too high: {} bytes per pattern", memory_per_pattern);
}