//! Integration tests for entity_compat.rs module
//! Tests only PUBLIC APIs - no access to private methods or fields
//! Focuses on complete compatibility workflows and performance validation

use llmkg::core::entity_compat::{Entity, Relationship, SimilarityResult};
use llmkg::core::types::EntityKey;
use llmkg::core::brain_enhanced_graph::brain_graph_core::BrainEnhancedKnowledgeGraph;
use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::triple::Triple;
use std::collections::HashMap;
use std::time::Instant;

/// Helper function to create test entities with various configurations
fn create_test_entities() -> Vec<Entity> {
    vec![
        Entity::new_with_type("entity_1".to_string(), "Person".to_string()),
        Entity::new_with_type("entity_2".to_string(), "Organization".to_string()),
        Entity::with_attributes(
            "entity_3".to_string(),
            "Product".to_string(),
            {
                let mut attrs = HashMap::new();
                attrs.insert("price".to_string(), "99.99".to_string());
                attrs.insert("category".to_string(), "Electronics".to_string());
                attrs
            }
        ),
        Entity::with_attributes(
            "entity_4".to_string(),
            "Location".to_string(),
            {
                let mut attrs = HashMap::new();
                attrs.insert("latitude".to_string(), "37.7749".to_string());
                attrs.insert("longitude".to_string(), "-122.4194".to_string());
                attrs.insert("city".to_string(), "San Francisco".to_string());
                attrs
            }
        ),
    ]
}

/// Helper function to create test relationships
fn create_test_relationships() -> Vec<Relationship> {
    vec![
        Relationship::new(
            "works_for".to_string(),
            {
                let mut attrs = HashMap::new();
                attrs.insert("since".to_string(), "2020".to_string());
                attrs.insert("role".to_string(), "Engineer".to_string());
                attrs
            }
        ),
        Relationship::new(
            "produces".to_string(),
            {
                let mut attrs = HashMap::new();
                attrs.insert("quantity".to_string(), "1000".to_string());
                attrs
            }
        ),
        Relationship::new(
            "located_in".to_string(),
            HashMap::new()
        ),
    ]
}

#[test]
fn test_entity_creation_workflow() {
    // Test complete entity creation workflow
    let entities = create_test_entities();
    
    // Verify entity creation
    assert_eq!(entities.len(), 4);
    
    // Test entity properties
    assert_eq!(entities[0].entity_type(), "Person");
    assert_eq!(entities[1].entity_type(), "Organization");
    assert_eq!(entities[2].entity_type(), "Product");
    assert_eq!(entities[3].entity_type(), "Location");
    
    // Test attribute access
    assert_eq!(entities[2].get_attribute("price"), Some("99.99"));
    assert_eq!(entities[3].get_attribute("city"), Some("San Francisco"));
    
    // Test entity IDs
    assert_eq!(entities[0].id(), "entity_1");
    assert_eq!(entities[1].id(), "entity_2");
}

#[test]
fn test_entity_key_operations() {
    // Test EntityKey creation and operations
    let key1 = EntityKey::new("test_key_1".to_string());
    let key2 = EntityKey::new("test_key_2".to_string());
    let key3 = EntityKey::new("test_key_1".to_string()); // Same as key1
    
    // Test deterministic key generation
    assert_eq!(key1, key3);
    assert_ne!(key1, key2);
    
    // Test key conversion operations
    let key_string = key1.to_string();
    assert!(!key_string.is_empty());
    
    let key_id = key1.id();
    assert!(!key_id.is_empty());
    
    // Test from_id consistency
    let key_from_id = EntityKey::from_id("test_key_1".to_string());
    assert_eq!(key_from_id, key1);
    
    // Test raw conversion round-trip
    let raw_value = key1.as_raw();
    let key_from_raw = EntityKey::from_raw(raw_value);
    assert_eq!(key_from_raw, key1);
}

#[test]
fn test_entity_serialization_workflow() {
    // Test entity serialization/deserialization
    let mut entity = Entity::with_attributes(
        "serialize_test".to_string(),
        "TestEntity".to_string(),
        {
            let mut attrs = HashMap::new();
            attrs.insert("attr1".to_string(), "value1".to_string());
            attrs.insert("attr2".to_string(), "value2".to_string());
            attrs.insert("attr3".to_string(), "value3".to_string());
            attrs
        }
    );
    
    // Set a specific key
    let key = EntityKey::new("serialize_key".to_string());
    entity.set_key(key);
    
    // Serialize
    let serialized = entity.serialize();
    assert!(!serialized.is_empty());
    
    // Deserialize
    let deserialized = Entity::deserialize(&serialized).expect("Deserialization should succeed");
    
    // Verify all properties are preserved
    assert_eq!(deserialized.id(), entity.id());
    assert_eq!(deserialized.name(), entity.name());
    assert_eq!(deserialized.entity_type(), entity.entity_type());
    assert_eq!(deserialized.attributes(), entity.attributes());
    assert_eq!(deserialized.key(), entity.key());
}

#[test]
fn test_relationship_workflow() {
    // Test complete relationship workflow
    let mut relationships = create_test_relationships();
    
    // Test relationship properties
    assert_eq!(relationships[0].relationship_type, "works_for");
    assert_eq!(relationships[0].attributes.get("role"), Some(&"Engineer".to_string()));
    
    // Test setting targets
    let target_key1 = EntityKey::new("target_1".to_string());
    let target_key2 = EntityKey::new("target_2".to_string());
    
    relationships[0].set_target(target_key1);
    relationships[1].set_target(target_key2);
    
    assert_eq!(relationships[0].target(), target_key1);
    assert_eq!(relationships[1].target(), target_key2);
}

#[test]
fn test_similarity_result_workflow() {
    // Test similarity result creation and usage
    let entity_key = EntityKey::new("similar_entity".to_string());
    
    let results = vec![
        SimilarityResult::new(entity_key, 0.95),
        SimilarityResult::new(EntityKey::new("entity2".to_string()), 0.87),
        SimilarityResult::new(EntityKey::new("entity3".to_string()), 0.73),
    ];
    
    // Verify results
    assert_eq!(results[0].entity, entity_key);
    assert_eq!(results[0].similarity, 0.95);
    
    // Test sorting by similarity
    let mut sorted_results = results.clone();
    sorted_results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    
    assert_eq!(sorted_results[0].similarity, 0.95);
    assert_eq!(sorted_results[1].similarity, 0.87);
    assert_eq!(sorted_results[2].similarity, 0.73);
}

#[test]
fn test_memory_usage_calculation() {
    // Test memory usage calculation for entities
    let small_entity = Entity::new_with_type("small".to_string(), "Type".to_string());
    let small_usage = small_entity.memory_usage();
    
    let mut large_entity = Entity::with_attributes(
        "large_entity_with_long_id".to_string(),
        "VeryLongEntityTypeName".to_string(),
        {
            let mut attrs = HashMap::new();
            for i in 0..100 {
                attrs.insert(
                    format!("attribute_key_{}", i),
                    format!("attribute_value_with_some_content_{}", i)
                );
            }
            attrs
        }
    );
    
    let large_usage = large_entity.memory_usage();
    
    // Large entity should use more memory
    assert!(large_usage > small_usage);
    
    // Adding attributes should increase memory usage
    let before_add = large_entity.memory_usage();
    large_entity.add_attribute("new_key", "new_value");
    let after_add = large_entity.memory_usage();
    assert!(after_add > before_add);
}

#[tokio::test]
async fn test_integration_with_knowledge_graph() {
    // Test integration with KnowledgeGraph through public APIs
    let graph = KnowledgeGraph::new(96).expect("Failed to create knowledge graph");
    
    // Create entities using compatibility layer
    let entity1 = Entity::new_with_type("person_1".to_string(), "Person".to_string());
    let entity2 = Entity::new_with_type("company_1".to_string(), "Company".to_string());
    
    // Create entity keys
    let key1 = EntityKey::new(entity1.id().clone());
    let key2 = EntityKey::new(entity2.id().clone());
    
    // Add entities to graph (simulating compatibility workflow)
    let entity_data1 = llmkg::core::types::EntityData {
        type_id: 0,
        properties: entity1.attributes().iter()
            .map(|(k, v)| (k.clone(), llmkg::core::types::AttributeValue::String(v.clone())))
            .collect(),
        embedding: None,
    };
    
    let entity_data2 = llmkg::core::types::EntityData {
        type_id: 0,
        properties: entity2.attributes().iter()
            .map(|(k, v)| (k.clone(), llmkg::core::types::AttributeValue::String(v.clone())))
            .collect(),
        embedding: None,
    };
    
    let actual_key1 = graph.add_entity(entity1.name().to_string(), vec![0.1; 96], Some(entity_data1))
        .expect("Failed to add entity1");
    let actual_key2 = graph.add_entity(entity2.name().to_string(), vec![0.2; 96], Some(entity_data2))
        .expect("Failed to add entity2");
    
    // Verify entities were added
    assert_eq!(graph.entity_count(), 2);
    
    // Add relationship
    graph.add_relationship(actual_key1, actual_key2, "employs".to_string(), 1.0)
        .expect("Failed to add relationship");
    
    assert_eq!(graph.relationship_count(), 1);
}

#[tokio::test]
async fn test_integration_with_brain_enhanced_graph() {
    // Test integration with BrainEnhancedKnowledgeGraph
    let brain_graph = BrainEnhancedKnowledgeGraph::new(96)
        .expect("Failed to create brain enhanced graph");
    
    // Create test entities
    let entities = create_test_entities();
    
    // Add entities to brain graph through core graph
    for entity in &entities {
        let entity_data = llmkg::core::types::EntityData {
            type_id: 0,
            properties: entity.attributes().iter()
                .map(|(k, v)| (k.clone(), llmkg::core::types::AttributeValue::String(v.clone())))
                .collect(),
            embedding: None,
        };
        
        brain_graph.core_graph.add_entity(
            entity.name().to_string(),
            vec![0.1; 96],
            Some(entity_data)
        ).expect("Failed to add entity to brain graph");
    }
    
    // Verify entities were added
    assert_eq!(brain_graph.entity_count(), entities.len());
    
    // Test activation levels (brain-specific feature)
    let activations = brain_graph.entity_activations.read().await;
    
    // Initially, activations should be empty or default
    assert!(activations.is_empty() || activations.values().all(|&v| v == 0.0));
}

#[test]
fn test_performance_entity_creation() {
    // Performance test for entity creation
    let start = Instant::now();
    let num_entities = 10000;
    
    let mut entities = Vec::with_capacity(num_entities);
    for i in 0..num_entities {
        let entity = Entity::new_with_type(
            format!("entity_{}", i),
            format!("Type_{}", i % 10)
        );
        entities.push(entity);
    }
    
    let duration = start.elapsed();
    println!("Created {} entities in {:?}", num_entities, duration);
    
    // Verify reasonable performance (should be < 1 second for 10k entities)
    assert!(duration.as_secs() < 1);
    
    // Test bulk operations
    let start = Instant::now();
    for entity in &mut entities[0..1000] {
        entity.add_attribute("bulk_attr", "bulk_value");
    }
    let duration = start.elapsed();
    println!("Added attributes to 1000 entities in {:?}", duration);
    
    // Should be very fast
    assert!(duration.as_millis() < 100);
}

#[test]
fn test_performance_entity_key_operations() {
    // Performance test for EntityKey operations
    let num_keys = 100000;
    
    // Test key creation performance
    let start = Instant::now();
    let keys: Vec<EntityKey> = (0..num_keys)
        .map(|i| EntityKey::new(format!("key_{}", i)))
        .collect();
    let duration = start.elapsed();
    println!("Created {} EntityKeys in {:?}", num_keys, duration);
    
    // Test key lookup performance
    let start = Instant::now();
    let lookups = 10000;
    for i in 0..lookups {
        let _ = keys[i % keys.len()].to_string();
    }
    let duration = start.elapsed();
    println!("Performed {} key lookups in {:?}", lookups, duration);
    
    // Test raw conversion performance
    let start = Instant::now();
    for key in &keys[0..1000] {
        let raw = key.as_raw();
        let _ = EntityKey::from_raw(raw);
    }
    let duration = start.elapsed();
    println!("Performed 1000 raw conversions in {:?}", duration);
    
    // All operations should be very fast
    assert!(duration.as_millis() < 50);
}

#[test]
fn test_performance_serialization() {
    // Performance test for serialization/deserialization
    let mut entities = Vec::new();
    
    // Create entities with varying complexity
    for i in 0..1000 {
        let mut attrs = HashMap::new();
        for j in 0..10 {
            attrs.insert(format!("attr_{}", j), format!("value_{}_{}", i, j));
        }
        
        let entity = Entity::with_attributes(
            format!("entity_{}", i),
            format!("Type_{}", i % 5),
            attrs
        );
        entities.push(entity);
    }
    
    // Test serialization performance
    let start = Instant::now();
    let serialized: Vec<Vec<u8>> = entities.iter()
        .map(|e| e.serialize())
        .collect();
    let ser_duration = start.elapsed();
    println!("Serialized 1000 entities in {:?}", ser_duration);
    
    // Test deserialization performance
    let start = Instant::now();
    let deserialized: Vec<Entity> = serialized.iter()
        .filter_map(|data| Entity::deserialize(data).ok())
        .collect();
    let deser_duration = start.elapsed();
    println!("Deserialized 1000 entities in {:?}", deser_duration);
    
    // Verify all entities were deserialized
    assert_eq!(deserialized.len(), entities.len());
    
    // Both operations should complete in reasonable time
    assert!(ser_duration.as_millis() < 500);
    assert!(deser_duration.as_millis() < 500);
}

#[test]
fn test_edge_cases_and_error_handling() {
    // Test edge cases for entity creation
    
    // Empty strings
    let empty_entity = Entity::new_with_type(String::new(), String::new());
    assert_eq!(empty_entity.id(), "");
    assert_eq!(empty_entity.name(), "");
    assert_eq!(empty_entity.entity_type(), "");
    
    // Very long strings
    let long_string = "a".repeat(10000);
    let long_entity = Entity::new_with_type(long_string.clone(), long_string.clone());
    assert_eq!(long_entity.id(), &long_string);
    
    // Unicode and special characters
    let unicode_id = "🦀 Rust 语言 #1".to_string();
    let unicode_entity = Entity::new_with_type(unicode_id.clone(), "UnicodeType".to_string());
    assert_eq!(unicode_entity.id(), &unicode_id);
    
    // Test entity key edge cases
    let special_key = EntityKey::new("!@#$%^&*()".to_string());
    let _ = special_key.to_string(); // Should not panic
    
    // Test deserialization with invalid data
    let invalid_data = vec![0xFF, 0xFE, 0xFD];
    let result = Entity::deserialize(&invalid_data);
    assert!(result.is_err());
    
    // Test empty serialization
    let empty_data = Vec::new();
    let result = Entity::deserialize(&empty_data);
    assert!(result.is_err());
}

#[test]
fn test_concurrent_access_simulation() {
    // Simulate concurrent access patterns
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let shared_entities = Arc::new(Mutex::new(Vec::new()));
    let num_threads = 10;
    let entities_per_thread = 100;
    
    let mut handles = vec![];
    
    for thread_id in 0..num_threads {
        let entities_clone = Arc::clone(&shared_entities);
        
        let handle = thread::spawn(move || {
            for i in 0..entities_per_thread {
                let entity = Entity::new_with_type(
                    format!("thread_{}_entity_{}", thread_id, i),
                    format!("ThreadType_{}", thread_id)
                );
                
                let mut entities = entities_clone.lock().unwrap();
                entities.push(entity);
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify all entities were created
    let entities = shared_entities.lock().unwrap();
    assert_eq!(entities.len(), num_threads * entities_per_thread);
}

#[test]
fn test_compatibility_with_legacy_patterns() {
    // Test compatibility with specific legacy patterns
    let test_patterns = vec![
        "visual_pattern",
        "temporal_pattern",
        "concept_hierarchy",
        "auditory_sequence",
        "semantic_cluster",
        "spatial_relation",
        "color_gradient",
        "rhythm_complex",
        "abstract_relation",
    ];
    
    for pattern in test_patterns {
        let key = EntityKey::from_hash(pattern);
        let original_id = key.get_original_id();
        
        // Some patterns might be recognized
        if let Some(id) = original_id {
            println!("Pattern '{}' mapped to '{}'", pattern, id);
            assert!(id.contains(pattern) || pattern.contains(&id[..5]));
        }
    }
}

#[tokio::test]
async fn test_end_to_end_knowledge_engine_workflow() {
    // Test complete workflow with KnowledgeEngine
    let engine = KnowledgeEngine::new(96, 1000)
        .expect("Failed to create knowledge engine");
    
    // Create entities
    let entity1 = Entity::new_with_type("subject_1".to_string(), "Person".to_string());
    let entity2 = Entity::new_with_type("object_1".to_string(), "Organization".to_string());
    
    // Create a triple
    let triple = Triple {
        subject: entity1.name().to_string(),
        predicate: "works_for".to_string(),
        object: entity2.name().to_string(),
        confidence: 0.8,
        source: Some("test".to_string()),
    };
    
    // Store triple in knowledge engine
    let node_id = engine.store_triple(triple, None)
        .expect("Failed to store triple");
    
    assert!(!node_id.is_empty());
    
    // Query the stored triple
    let query = llmkg::core::knowledge_types::TripleQuery {
        subject: Some(entity1.name().to_string()),
        predicate: None,
        object: None,
    };
    
    let results = engine.query_triples(&query)
        .expect("Failed to query triples");
    
    assert!(!results.is_empty());
}

#[test]
fn test_attribute_operations_comprehensive() {
    // Comprehensive test of attribute operations
    let mut entity = Entity::new_with_type("attr_test".to_string(), "TestType".to_string());
    
    // Test adding various attribute types
    entity.add_attribute("string_attr", "test_value");
    entity.add_attribute("numeric_attr", "42.5");
    entity.add_attribute("boolean_attr", "true");
    entity.add_attribute("empty_attr", "");
    entity.add_attribute("", "empty_key_value");
    
    // Test retrieval
    assert_eq!(entity.get_attribute("string_attr"), Some("test_value"));
    assert_eq!(entity.get_attribute("numeric_attr"), Some("42.5"));
    assert_eq!(entity.get_attribute("boolean_attr"), Some("true"));
    assert_eq!(entity.get_attribute("empty_attr"), Some(""));
    assert_eq!(entity.get_attribute(""), Some("empty_key_value"));
    assert_eq!(entity.get_attribute("non_existent"), None);
    
    // Test overwriting
    entity.add_attribute("string_attr", "new_value");
    assert_eq!(entity.get_attribute("string_attr"), Some("new_value"));
    
    // Test attribute count
    assert_eq!(entity.attributes().len(), 5);
    
    // Test memory impact
    let initial_memory = entity.memory_usage();
    
    // Add many attributes
    for i in 0..100 {
        entity.add_attribute(&format!("attr_{}", i), &format!("value_{}", i));
    }
    
    let final_memory = entity.memory_usage();
    assert!(final_memory > initial_memory);
    
    // Verify all attributes are accessible
    for i in 0..100 {
        assert_eq!(
            entity.get_attribute(&format!("attr_{}", i)),
            Some(&format!("value_{}", i)[..])
        );
    }
}

#[test]
fn test_relationship_attributes_comprehensive() {
    // Comprehensive test of relationship attributes
    let mut attrs = HashMap::new();
    attrs.insert("weight".to_string(), "0.8".to_string());
    attrs.insert("confidence".to_string(), "0.95".to_string());
    attrs.insert("timestamp".to_string(), "2024-01-01".to_string());
    attrs.insert("metadata".to_string(), r#"{"type": "professional", "verified": true}"#.to_string());
    
    let mut relationship = Relationship::new("complex_relation".to_string(), attrs);
    
    // Test attribute access
    assert_eq!(relationship.attributes.get("weight"), Some(&"0.8".to_string()));
    assert_eq!(relationship.attributes.get("confidence"), Some(&"0.95".to_string()));
    assert_eq!(relationship.attributes.get("timestamp"), Some(&"2024-01-01".to_string()));
    
    // Test JSON-like metadata
    let metadata = relationship.attributes.get("metadata").unwrap();
    assert!(metadata.contains("professional"));
    assert!(metadata.contains("verified"));
    
    // Test modifying attributes
    relationship.attributes.insert("updated".to_string(), "true".to_string());
    assert_eq!(relationship.attributes.len(), 5);
    
    // Test with target
    let target = EntityKey::new("target_entity".to_string());
    relationship.set_target(target);
    assert_eq!(relationship.target(), target);
}

#[test]
fn test_similarity_result_operations() {
    // Test various similarity result operations
    let entities = vec![
        EntityKey::new("entity_a".to_string()),
        EntityKey::new("entity_b".to_string()),
        EntityKey::new("entity_c".to_string()),
        EntityKey::new("entity_d".to_string()),
        EntityKey::new("entity_e".to_string()),
    ];
    
    let results: Vec<SimilarityResult> = entities.into_iter()
        .enumerate()
        .map(|(i, entity)| {
            let similarity = 1.0 - (i as f32 * 0.2);
            SimilarityResult::new(entity, similarity)
        })
        .collect();
    
    // Test similarity values
    assert_eq!(results[0].similarity, 1.0);
    assert_eq!(results[1].similarity, 0.8);
    assert_eq!(results[2].similarity, 0.6);
    assert_eq!(results[3].similarity, 0.4);
    assert_eq!(results[4].similarity, 0.2);
    
    // Test filtering by threshold
    let threshold = 0.5;
    let filtered: Vec<&SimilarityResult> = results.iter()
        .filter(|r| r.similarity >= threshold)
        .collect();
    
    assert_eq!(filtered.len(), 3);
    assert!(filtered.iter().all(|r| r.similarity >= threshold));
    
    // Test finding top-k results
    let top_k = 3;
    let mut sorted = results.clone();
    sorted.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    let top_results: Vec<&SimilarityResult> = sorted.iter().take(top_k).collect();
    
    assert_eq!(top_results.len(), top_k);
    assert_eq!(top_results[0].similarity, 1.0);
    assert_eq!(top_results[1].similarity, 0.8);
    assert_eq!(top_results[2].similarity, 0.6);
}

#[test]
fn test_entity_lifecycle_complete() {
    // Test complete entity lifecycle
    
    // 1. Creation
    let mut entity = Entity::new_with_type("lifecycle_test".to_string(), "TestEntity".to_string());
    assert_eq!(entity.key(), EntityKey::default());
    
    // 2. Configuration
    entity.add_attribute("status", "active");
    entity.add_attribute("created_at", "2024-01-01");
    
    // 3. Key assignment
    let assigned_key = EntityKey::new("lifecycle_key".to_string());
    entity.set_key(assigned_key);
    assert_eq!(entity.key(), assigned_key);
    
    // 4. Modification
    entity.add_attribute("status", "modified");
    entity.add_attribute("modified_at", "2024-01-02");
    
    // 5. Serialization
    let serialized = entity.serialize();
    
    // 6. Transmission (simulated)
    let transmitted_data = serialized.clone();
    
    // 7. Deserialization
    let restored = Entity::deserialize(&transmitted_data)
        .expect("Failed to deserialize");
    
    // 8. Verification
    assert_eq!(restored.id(), entity.id());
    assert_eq!(restored.name(), entity.name());
    assert_eq!(restored.entity_type(), entity.entity_type());
    assert_eq!(restored.get_attribute("status"), Some("modified"));
    assert_eq!(restored.get_attribute("created_at"), Some("2024-01-01"));
    assert_eq!(restored.get_attribute("modified_at"), Some("2024-01-02"));
    assert_eq!(restored.key(), entity.key());
    
    // 9. Memory calculation
    let memory = restored.memory_usage();
    assert!(memory > 0);
    println!("Entity lifecycle complete. Final memory usage: {} bytes", memory);
}