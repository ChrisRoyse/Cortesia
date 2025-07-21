//! Basic integration tests for entity_compat.rs module
//! Tests only PUBLIC APIs - no access to private methods or fields

use llmkg::core::entity_compat::{Entity, Relationship, SimilarityResult};
use llmkg::core::types::EntityKey;
use std::collections::HashMap;

#[test]
fn test_entity_creation_and_attributes() {
    // Test basic entity creation
    let entity1 = Entity::new_with_type("entity_1".to_string(), "Person".to_string());
    assert_eq!(entity1.id(), "entity_1");
    assert_eq!(entity1.entity_type(), "Person");
    assert_eq!(entity1.name(), "Person");
    assert!(entity1.attributes().is_empty());
    
    // Test entity with attributes
    let mut attrs = HashMap::new();
    attrs.insert("age".to_string(), "30".to_string());
    attrs.insert("city".to_string(), "San Francisco".to_string());
    
    let entity2 = Entity::with_attributes(
        "entity_2".to_string(),
        "Person".to_string(),
        attrs
    );
    
    assert_eq!(entity2.get_attribute("age"), Some("30"));
    assert_eq!(entity2.get_attribute("city"), Some("San Francisco"));
    assert_eq!(entity2.get_attribute("missing"), None);
}

#[test]
fn test_entity_key_operations() {
    // Test EntityKey creation
    let key1 = EntityKey::new("test_key_1".to_string());
    let key2 = EntityKey::new("test_key_2".to_string());
    let key3 = EntityKey::new("test_key_1".to_string());
    
    // Keys should be deterministic
    assert_eq!(key1, key3);
    assert_ne!(key1, key2);
    
    // Test conversions
    let key_string = key1.to_string();
    assert!(!key_string.is_empty());
    
    let raw = key1.as_raw();
    let key_from_raw = EntityKey::from_raw(raw);
    assert_eq!(key1, key_from_raw);
}

#[test]
fn test_entity_serialization() {
    let mut entity = Entity::with_attributes(
        "test_entity".to_string(),
        "TestType".to_string(),
        {
            let mut attrs = HashMap::new();
            attrs.insert("key1".to_string(), "value1".to_string());
            attrs.insert("key2".to_string(), "value2".to_string());
            attrs
        }
    );
    
    // Set a key
    let key = EntityKey::new("test_key".to_string());
    entity.set_key(key);
    
    // Serialize
    let serialized = entity.serialize();
    assert!(!serialized.is_empty());
    
    // Deserialize
    let deserialized = Entity::deserialize(&serialized).expect("Deserialization failed");
    assert_eq!(deserialized.id(), entity.id());
    assert_eq!(deserialized.name(), entity.name());
    assert_eq!(deserialized.entity_type(), entity.entity_type());
    assert_eq!(deserialized.attributes(), entity.attributes());
    assert_eq!(deserialized.key(), entity.key());
}

#[test]
fn test_relationship_operations() {
    let mut attrs = HashMap::new();
    attrs.insert("weight".to_string(), "0.8".to_string());
    attrs.insert("type".to_string(), "professional".to_string());
    
    let mut relationship = Relationship::new("works_with".to_string(), attrs);
    
    // Test attributes
    assert_eq!(relationship.relationship_type, "works_with");
    assert_eq!(relationship.attributes.get("weight"), Some(&"0.8".to_string()));
    
    // Test target setting
    let target = EntityKey::new("target_entity".to_string());
    relationship.set_target(target);
    assert_eq!(relationship.target(), target);
}

#[test]
fn test_similarity_results() {
    let key1 = EntityKey::new("entity1".to_string());
    let key2 = EntityKey::new("entity2".to_string());
    let key3 = EntityKey::new("entity3".to_string());
    
    let mut results = vec![
        SimilarityResult::new(key1, 0.95),
        SimilarityResult::new(key2, 0.75),
        SimilarityResult::new(key3, 0.85),
    ];
    
    // Sort by similarity
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
    
    assert_eq!(results[0].similarity, 0.95);
    assert_eq!(results[1].similarity, 0.85);
    assert_eq!(results[2].similarity, 0.75);
}

#[test]
fn test_entity_memory_usage() {
    let small_entity = Entity::new_with_type("small".to_string(), "Type".to_string());
    let small_usage = small_entity.memory_usage();
    
    let mut large_entity = Entity::with_attributes(
        "large_entity".to_string(),
        "ComplexType".to_string(),
        {
            let mut attrs = HashMap::new();
            for i in 0..50 {
                attrs.insert(format!("key_{}", i), format!("value_{}", i));
            }
            attrs
        }
    );
    
    let large_usage = large_entity.memory_usage();
    assert!(large_usage > small_usage);
    
    // Adding more attributes increases memory
    let before = large_entity.memory_usage();
    large_entity.add_attribute("new_key", "new_value");
    let after = large_entity.memory_usage();
    assert!(after > before);
}

#[test]
fn test_entity_modification() {
    let mut entity = Entity::new_with_type("test".to_string(), "Type".to_string());
    
    // Add attributes
    entity.add_attribute("attr1", "value1");
    entity.add_attribute("attr2", "value2");
    
    assert_eq!(entity.get_attribute("attr1"), Some("value1"));
    assert_eq!(entity.get_attribute("attr2"), Some("value2"));
    
    // Overwrite attribute
    entity.add_attribute("attr1", "new_value");
    assert_eq!(entity.get_attribute("attr1"), Some("new_value"));
    
    // Set key
    let new_key = EntityKey::new("new_key".to_string());
    entity.set_key(new_key);
    assert_eq!(entity.key(), new_key);
}

#[test]
fn test_edge_cases() {
    // Empty strings
    let empty_entity = Entity::new_with_type(String::new(), String::new());
    assert_eq!(empty_entity.id(), "");
    assert_eq!(empty_entity.name(), "");
    
    // Unicode
    let unicode_entity = Entity::new_with_type(
        "🦀_rust_实体".to_string(),
        "UnicodeType".to_string()
    );
    assert_eq!(unicode_entity.id(), "🦀_rust_实体");
    
    // Invalid deserialization
    let invalid_data = vec![0xFF, 0xFE];
    assert!(Entity::deserialize(&invalid_data).is_err());
}

#[test]
fn test_performance_batch_operations() {
    use std::time::Instant;
    
    // Create many entities
    let start = Instant::now();
    let entities: Vec<Entity> = (0..1000)
        .map(|i| Entity::new_with_type(
            format!("entity_{}", i),
            format!("Type_{}", i % 10)
        ))
        .collect();
    let creation_time = start.elapsed();
    
    println!("Created 1000 entities in {:?}", creation_time);
    assert!(creation_time.as_millis() < 100); // Should be fast
    
    // Batch serialization
    let start = Instant::now();
    let _serialized: Vec<Vec<u8>> = entities.iter()
        .map(|e| e.serialize())
        .collect();
    let ser_time = start.elapsed();
    
    println!("Serialized 1000 entities in {:?}", ser_time);
    assert!(ser_time.as_millis() < 500); // Should be reasonably fast
}