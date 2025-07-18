// Phase 4.4: Zero-Copy Serialization Tests
// Comprehensive test suite for zero-copy serialization functionality

use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::zero_copy_engine::ZeroCopyKnowledgeEngine;
use llmkg::core::types::{EntityData, Relationship};
use llmkg::storage::zero_copy::{ZeroCopySerializer, ZeroCopyDeserializer, ZeroCopyGraphStorage};
use llmkg::storage::string_interner::StringInterner;
use std::sync::Arc;
use std::time::Instant;

#[tokio::test]
async fn test_zero_copy_serialization_roundtrip() {
    let mut serializer = ZeroCopySerializer::new();
    
    // Create test entities
    let entities = vec![
        EntityData {
            type_id: 1,
            properties: "Entity 1 with properties".to_string(),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
        },
        EntityData {
            type_id: 2,
            properties: "Entity 2 with different properties".to_string(),
            embedding: vec![0.5, 0.6, 0.7, 0.8],
        },
    ];
    
    // Add entities to serializer
    for entity in &entities {
        serializer.add_entity(entity, 4).unwrap();
    }
    
    // Add test relationships
    let relationships = vec![
        Relationship { from: 0, to: 1, rel_type: 1, weight: 0.5 },
        Relationship { from: 1, to: 0, rel_type: 2, weight: 0.8 },
    ];
    
    for relationship in &relationships {
        serializer.add_relationship(relationship).unwrap();
    }
    
    // Finalize serialization
    let data = serializer.finalize().unwrap();
    assert!(!data.is_empty());
    
    // Deserialize with zero-copy
    let deserializer = unsafe { ZeroCopyDeserializer::new(&data).unwrap() };
    
    // Verify entity count
    assert_eq!(deserializer.entity_count(), 2);
    assert_eq!(deserializer.relationship_count(), 2);
    
    // Verify entity data
    let entity1 = deserializer.get_entity(0).unwrap();
    assert_eq!(entity1.type_id, 1);
    
    let properties1 = deserializer.get_entity_properties(entity1);
    assert_eq!(properties1, "Entity 1 with properties");
    
    let entity2 = deserializer.get_entity(1).unwrap();
    assert_eq!(entity2.type_id, 2);
    
    let properties2 = deserializer.get_entity_properties(entity2);
    assert_eq!(properties2, "Entity 2 with different properties");
    
    // Verify relationship data
    let rel1 = deserializer.get_relationship(0).unwrap();
    assert_eq!(rel1.from, 0);
    assert_eq!(rel1.to, 1);
    assert_eq!(rel1.rel_type, 1);
    assert_eq!(rel1.weight, 0.5);
    
    let rel2 = deserializer.get_relationship(1).unwrap();
    assert_eq!(rel2.from, 1);
    assert_eq!(rel2.to, 0);
    assert_eq!(rel2.rel_type, 2);
    assert_eq!(rel2.weight, 0.8);
}

#[tokio::test]
async fn test_zero_copy_engine_integration() {
    let base_engine = Arc::new(KnowledgeEngine::new(96).unwrap());
    let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine.clone(), 96);
    
    // Add test entities to base engine
    let entities = vec![
        EntityData {
            type_id: 1,
            properties: "Test entity 1".to_string(),
            embedding: vec![0.1; 96],
        },
        EntityData {
            type_id: 2,
            properties: "Test entity 2".to_string(),
            embedding: vec![0.2; 96],
        },
        EntityData {
            type_id: 3,
            properties: "Test entity 3".to_string(),
            embedding: vec![0.3; 96],
        },
    ];
    
    for (i, entity) in entities.iter().enumerate() {
        base_engine.insert_entity(i as u32, entity.clone()).await.unwrap();
    }
    
    // Serialize to zero-copy format
    let data = zero_copy_engine.serialize_to_zero_copy().await.unwrap();
    assert!(!data.is_empty());
    
    // Load zero-copy data
    zero_copy_engine.load_zero_copy_data(data).unwrap();
    
    // Test zero-copy access
    for i in 0..3 {
        let handle = zero_copy_engine.get_entity_zero_copy(i).unwrap();
        assert_eq!(handle.id(), i);
        assert_eq!(handle.type_id(), (i + 1) as u16);
        assert_eq!(handle.properties(), format!("Test entity {}", i + 1));
    }
    
    // Test metrics
    let metrics = zero_copy_engine.get_metrics();
    assert_eq!(metrics.entities_processed, 3);
    assert!(metrics.serialization_time_ns > 0);
}

#[tokio::test]
async fn test_zero_copy_similarity_search() {
    let base_engine = Arc::new(KnowledgeEngine::new(4).unwrap());
    let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine.clone(), 4);
    
    // Add entities with known embeddings for similarity testing
    let test_embeddings = vec![
        vec![1.0, 0.0, 0.0, 0.0], // Should be most similar to query [1,0,0,0]
        vec![0.9, 0.1, 0.0, 0.0], // Second most similar
        vec![0.5, 0.5, 0.0, 0.0], // Third most similar
        vec![0.0, 1.0, 0.0, 0.0], // Less similar
        vec![0.0, 0.0, 1.0, 0.0], // Even less similar
    ];
    
    for (i, embedding) in test_embeddings.iter().enumerate() {
        let entity = EntityData {
            type_id: i as u16,
            properties: format!("Entity {}", i),
            embedding: embedding.clone(),
        };
        base_engine.insert_entity(i as u32, entity).await.unwrap();
    }
    
    // Serialize and load
    let data = zero_copy_engine.serialize_to_zero_copy().await.unwrap();
    zero_copy_engine.load_zero_copy_data(data).unwrap();
    
    // Test similarity search
    let query = vec![1.0, 0.0, 0.0, 0.0];
    let results = zero_copy_engine.similarity_search_zero_copy(&query, 3).unwrap();
    
    assert_eq!(results.len(), 3);
    
    // Results should be sorted by similarity (descending)
    for i in 1..results.len() {
        assert!(results[i-1].similarity >= results[i].similarity,
                "Results not sorted: {} >= {} failed", 
                results[i-1].similarity, results[i].similarity);
    }
    
    // Most similar should be entity 0
    assert_eq!(results[0].entity_id, 0);
}

#[tokio::test]
async fn test_zero_copy_performance_vs_standard() {
    let base_engine = Arc::new(KnowledgeEngine::new(96).unwrap());
    let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine.clone(), 96);
    
    // Add a reasonable number of entities for performance testing
    let entity_count = 1000;
    for i in 0..entity_count {
        let entity = EntityData {
            type_id: (i % 10) as u16,
            properties: format!("Performance test entity {}", i),
            embedding: (0..96).map(|j| (i + j) as f32 / 1000.0).collect(),
        };
        base_engine.insert_entity(i, entity).await.unwrap();
    }
    
    // Serialize and load
    let data = zero_copy_engine.serialize_to_zero_copy().await.unwrap();
    zero_copy_engine.load_zero_copy_data(data).unwrap();
    
    // Run performance benchmark
    let query = vec![0.5; 96];
    let benchmark = zero_copy_engine.benchmark_against_standard(&query, 50).await.unwrap();
    
    // Zero-copy should be faster
    assert!(benchmark.speedup > 1.0, "Zero-copy should be faster than standard access");
    assert!(benchmark.zero_copy_ops_per_sec() > 0.0);
    assert!(benchmark.standard_ops_per_sec() > 0.0);
    
    println!("Performance benchmark results:");
    println!("  Zero-copy: {:.2} ops/sec", benchmark.zero_copy_ops_per_sec());
    println!("  Standard: {:.2} ops/sec", benchmark.standard_ops_per_sec());
    println!("  Speedup: {:.2}x", benchmark.speedup);
}

#[tokio::test]
async fn test_zero_copy_memory_efficiency() {
    let base_engine = Arc::new(KnowledgeEngine::new(96).unwrap());
    let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine.clone(), 96);
    
    // Add entities with varying property sizes
    let mut total_property_bytes = 0;
    for i in 0..500 {
        let property_size = 50 + (i % 200); // 50-250 chars
        let properties = "x".repeat(property_size);
        total_property_bytes += properties.len();
        
        let entity = EntityData {
            type_id: (i % 5) as u16,
            properties,
            embedding: vec![i as f32 / 500.0; 96],
        };
        base_engine.insert_entity(i, entity).await.unwrap();
    }
    
    // Serialize to zero-copy format
    let data = zero_copy_engine.serialize_to_zero_copy().await.unwrap();
    zero_copy_engine.load_zero_copy_data(data.clone()).unwrap();
    
    // Calculate memory efficiency
    let raw_size = total_property_bytes + (500 * 96 * 4) + (500 * 16); // properties + embeddings + metadata
    let zero_copy_size = data.len();
    let compression_ratio = raw_size as f32 / zero_copy_size as f32;
    
    println!("Memory efficiency test:");
    println!("  Raw size: {} bytes", raw_size);
    println!("  Zero-copy size: {} bytes", zero_copy_size);
    println!("  Compression ratio: {:.2}:1", compression_ratio);
    
    // Should achieve some level of compression
    assert!(compression_ratio > 1.0, "Zero-copy should be more memory efficient than raw data");
    
    // Test metrics
    let metrics = zero_copy_engine.get_metrics();
    assert_eq!(metrics.entities_processed, 500);
    assert!(metrics.compression_ratio > 1.0);
    assert!(metrics.memory_efficiency_bytes_per_entity() > 0.0);
}

#[tokio::test]
async fn test_zero_copy_iterator_performance() {
    let mut serializer = ZeroCopySerializer::new();
    
    let entity_count = 10000;
    for i in 0..entity_count {
        let entity = EntityData {
            type_id: (i % 100) as u16,
            properties: format!("Iterator test entity {}", i),
            embedding: vec![i as f32; 32],
        };
        serializer.add_entity(&entity, 32).unwrap();
    }
    
    let data = serializer.finalize().unwrap();
    let deserializer = unsafe { ZeroCopyDeserializer::new(&data).unwrap() };
    
    // Test iterator performance
    let start = Instant::now();
    let count = deserializer.iter_entities().count();
    let iteration_time = start.elapsed();
    
    assert_eq!(count, entity_count);
    
    // Iteration should be very fast due to zero-copy access
    let entities_per_ms = count as f64 / iteration_time.as_millis() as f64;
    println!("Iterator performance: {:.0} entities/ms", entities_per_ms);
    
    // Should be able to iterate thousands of entities per millisecond
    assert!(entities_per_ms > 1000.0, "Iterator should be very fast with zero-copy access");
    
    // Test iterator exactness
    let mut iter = deserializer.iter_entities();
    assert_eq!(iter.size_hint(), (entity_count, Some(entity_count)));
    
    // Verify we can iterate through all entities
    let mut iterated_count = 0;
    for entity in deserializer.iter_entities() {
        assert!(entity.id < entity_count as u32);
        iterated_count += 1;
    }
    assert_eq!(iterated_count, entity_count);
}

#[tokio::test]
async fn test_zero_copy_batch_access() {
    let base_engine = Arc::new(KnowledgeEngine::new(64).unwrap());
    let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine.clone(), 64);
    
    // Add test entities
    for i in 0..100 {
        let entity = EntityData {
            type_id: i as u16,
            properties: format!("Batch test entity {}", i),
            embedding: vec![i as f32 / 100.0; 64],
        };
        base_engine.insert_entity(i, entity).await.unwrap();
    }
    
    // Serialize and load
    let data = zero_copy_engine.serialize_to_zero_copy().await.unwrap();
    zero_copy_engine.load_zero_copy_data(data).unwrap();
    
    // Test batch access
    let entity_ids: Vec<u32> = (0..50).collect();
    let handles = zero_copy_engine.get_entities_batch_zero_copy(&entity_ids);
    
    assert_eq!(handles.len(), 50);
    
    // All handles should be valid
    for (i, handle) in handles.iter().enumerate() {
        assert!(handle.is_some(), "Handle {} should be valid", i);
        let handle = handle.as_ref().unwrap();
        assert_eq!(handle.id(), i as u32);
        assert_eq!(handle.properties(), format!("Batch test entity {}", i));
    }
    
    // Test with some invalid IDs
    let mixed_ids = vec![0, 1, 200, 3, 300]; // 200 and 300 don't exist
    let mixed_handles = zero_copy_engine.get_entities_batch_zero_copy(&mixed_ids);
    
    assert_eq!(mixed_handles.len(), 5);
    assert!(mixed_handles[0].is_some());
    assert!(mixed_handles[1].is_some());
    assert!(mixed_handles[2].is_none()); // 200 doesn't exist
    assert!(mixed_handles[3].is_some());
    assert!(mixed_handles[4].is_none()); // 300 doesn't exist
}

#[test]
fn test_zero_copy_storage_creation() {
    let string_interner = Arc::new(StringInterner::new());
    
    // Create simple test data
    let mut serializer = ZeroCopySerializer::new();
    let entity = EntityData {
        type_id: 1,
        properties: "test".to_string(),
        embedding: vec![1.0, 2.0, 3.0, 4.0],
    };
    serializer.add_entity(&entity, 4).unwrap();
    let data = serializer.finalize().unwrap();
    
    // Create storage
    let storage = ZeroCopyGraphStorage::from_data(data, string_interner).unwrap();
    
    assert_eq!(storage.entity_count(), 1);
    assert!(storage.memory_usage() > 0);
    
    let entity = storage.get_entity(0).unwrap();
    assert_eq!(entity.type_id, 1);
    
    let properties = storage.get_entity_properties(entity);
    assert_eq!(properties, "test");
}