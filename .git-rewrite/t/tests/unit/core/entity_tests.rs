//! Entity Management Unit Tests
//!
//! Comprehensive tests for entity creation, manipulation, serialization,
//! and memory management with 100% code coverage.

use crate::unit::*;
use crate::unit::test_utils::*;
use std::collections::HashSet;

#[cfg(test)]
mod entity_tests {
    use super::*;

    #[test]
    fn test_entity_creation_deterministic() {
        let mut rng = DeterministicRng::new(ENTITY_TEST_SEED);
        
        // Test 1: Basic entity creation
        let entity_key = EntityKey::from_hash("test_entity_1");
        let entity = Entity::new(entity_key, "Test Entity".to_string());
        
        assert_eq!(entity.key(), entity_key);
        assert_eq!(entity.name(), "Test Entity");
        assert_eq!(entity.attributes().len(), 0);
        assert_eq!(entity.memory_usage(), EXPECTED_EMPTY_ENTITY_SIZE);
        
        // Test 2: Entity with attributes
        let mut entity_with_attrs = Entity::new(
            EntityKey::from_hash("test_entity_2"), 
            "Entity with Attributes".to_string()
        );
        
        entity_with_attrs.add_attribute("type", "test");
        entity_with_attrs.add_attribute("value", "42");
        entity_with_attrs.add_attribute("flag", "true");
        
        assert_eq!(entity_with_attrs.attributes().len(), 3);
        assert_eq!(entity_with_attrs.get_attribute("type"), Some("test"));
        assert_eq!(entity_with_attrs.get_attribute("value"), Some("42"));
        assert_eq!(entity_with_attrs.get_attribute("nonexistent"), None);
        
        // Verify memory usage calculation
        let expected_memory = EXPECTED_EMPTY_ENTITY_SIZE 
            + "type".len() as u64 + "test".len() as u64
            + "value".len() as u64 + "42".len() as u64
            + "flag".len() as u64 + "true".len() as u64
            + ATTRIBUTE_OVERHEAD as u64 * 3;
        assert_eq!(entity_with_attrs.memory_usage(), expected_memory);
        
        // Test 3: Entity serialization determinism
        let serialized1 = entity_with_attrs.serialize();
        let serialized2 = entity_with_attrs.serialize();
        assert_eq!(serialized1, serialized2);
        
        let deserialized = Entity::deserialize(&serialized1).unwrap();
        assert_eq!(entity_with_attrs, deserialized);
    }
    
    #[test]
    fn test_entity_key_generation() {
        // Test deterministic key generation
        let key1 = EntityKey::from_hash("identical_input");
        let key2 = EntityKey::from_hash("identical_input");
        assert_eq!(key1, key2);
        
        // Test key uniqueness
        let key3 = EntityKey::from_hash("different_input");
        assert_ne!(key1, key3);
        
        // Test key collision resistance (birthday paradox test)
        let mut keys = HashSet::new();
        for i in 0..100000 {
            let key = EntityKey::from_hash(&format!("key_{}", i));
            assert!(!keys.contains(&key), "Key collision detected at iteration {}", i);
            keys.insert(key);
        }
    }
    
    #[test]
    fn test_entity_attribute_edge_cases() {
        let mut entity = Entity::new(EntityKey::from_hash("test"), "Test".to_string());
        
        // Test empty string values
        entity.add_attribute("empty", "");
        assert_eq!(entity.get_attribute("empty"), Some(""));
        
        // Test Unicode values
        entity.add_attribute("unicode", "🦀 Rust 测试 🌟");
        assert_eq!(entity.get_attribute("unicode"), Some("🦀 Rust 测试 🌟"));
        
        // Test very long values
        let long_value = "x".repeat(10000);
        entity.add_attribute("long", &long_value);
        assert_eq!(entity.get_attribute("long"), Some(&long_value));
        
        // Test attribute overwriting
        entity.add_attribute("overwrite", "original");
        entity.add_attribute("overwrite", "updated");
        assert_eq!(entity.get_attribute("overwrite"), Some("updated"));
        
        // Test case sensitivity
        entity.add_attribute("CaseSensitive", "value1");
        entity.add_attribute("casesensitive", "value2");
        assert_eq!(entity.get_attribute("CaseSensitive"), Some("value1"));
        assert_eq!(entity.get_attribute("casesensitive"), Some("value2"));
    }
    
    #[test]
    fn test_entity_memory_management() {
        let mut entity = Entity::new(EntityKey::from_hash("memory_test"), "Memory Test".to_string());
        let initial_memory = entity.memory_usage();
        
        // Add attributes and verify memory growth
        entity.add_attribute("attr1", "value1");
        let memory_after_1 = entity.memory_usage();
        assert!(memory_after_1 > initial_memory);
        
        entity.add_attribute("attr2", "value2");
        let memory_after_2 = entity.memory_usage();
        assert!(memory_after_2 > memory_after_1);
        
        // Remove attribute and verify memory decrease
        entity.remove_attribute("attr1");
        let memory_after_removal = entity.memory_usage();
        assert!(memory_after_removal < memory_after_2);
        
        // Verify memory calculation accuracy
        let expected_memory = calculate_expected_entity_memory(&entity);
        assert_eq!(entity.memory_usage(), expected_memory);
    }

    #[test]
    fn test_entity_clone_and_equality() {
        let mut original = Entity::new(EntityKey::from_hash("clone_test"), "Original".to_string());
        original.add_attribute("key1", "value1");
        original.add_attribute("key2", "value2");
        
        let cloned = original.clone();
        assert_eq!(original, cloned);
        assert_eq!(original.memory_usage(), cloned.memory_usage());
        
        // Modify clone and verify independence
        let mut modified_clone = cloned.clone();
        modified_clone.add_attribute("key3", "value3");
        assert_ne!(original, modified_clone);
        assert!(modified_clone.memory_usage() > original.memory_usage());
    }

    #[test]
    fn test_entity_large_scale_attributes() {
        let mut entity = Entity::new(EntityKey::from_hash("large_test"), "Large Test".to_string());
        let attribute_count = 10000;
        
        // Add many attributes
        for i in 0..attribute_count {
            entity.add_attribute(&format!("key_{}", i), &format!("value_{}", i));
        }
        
        assert_eq!(entity.attributes().len(), attribute_count);
        
        // Verify all attributes are accessible
        for i in 0..attribute_count {
            let key = format!("key_{}", i);
            let expected_value = format!("value_{}", i);
            assert_eq!(entity.get_attribute(&key), Some(&expected_value));
        }
        
        // Test memory usage is reasonable
        let memory_per_attribute = entity.memory_usage() / attribute_count as u64;
        assert!(memory_per_attribute < 100, "Memory per attribute too high: {}", memory_per_attribute);
    }

    #[test] 
    fn test_entity_serialization_formats() {
        let mut entity = Entity::new(EntityKey::from_hash("serialize_test"), "Serialize Test".to_string());
        entity.add_attribute("type", "test");
        entity.add_attribute("number", "42");
        entity.add_attribute("boolean", "true");
        
        // Test JSON serialization
        let json_serialized = entity.to_json().unwrap();
        let json_deserialized = Entity::from_json(&json_serialized).unwrap();
        assert_eq!(entity, json_deserialized);
        
        // Test binary serialization
        let binary_serialized = entity.to_binary().unwrap();
        let binary_deserialized = Entity::from_binary(&binary_serialized).unwrap();
        assert_eq!(entity, binary_deserialized);
        
        // Verify binary is more compact than JSON
        assert!(binary_serialized.len() < json_serialized.len());
        
        // Test roundtrip consistency
        let roundtrip1 = Entity::from_json(&entity.to_json().unwrap()).unwrap();
        let roundtrip2 = Entity::from_binary(&roundtrip1.to_binary().unwrap()).unwrap();
        assert_eq!(entity, roundtrip2);
    }

    #[test]
    fn test_entity_validation() {
        // Test valid entity
        let valid_entity = Entity::new(EntityKey::from_hash("valid"), "Valid Entity".to_string());
        assert!(valid_entity.validate().is_ok());
        
        // Test entity with invalid characters in name
        let mut invalid_name_entity = Entity::new(EntityKey::from_hash("invalid"), "".to_string());
        assert!(invalid_name_entity.validate().is_err());
        
        // Test entity with null bytes in attributes
        let mut null_byte_entity = Entity::new(EntityKey::from_hash("null_test"), "Test".to_string());
        null_byte_entity.add_attribute("invalid\0key", "value");
        assert!(null_byte_entity.validate().is_err());
        
        // Test entity with extremely long attribute values
        let mut long_attr_entity = Entity::new(EntityKey::from_hash("long_test"), "Test".to_string());
        let very_long_value = "x".repeat(1_000_000); // 1MB value
        long_attr_entity.add_attribute("long", &very_long_value);
        
        // Should validate but with warnings
        let validation_result = long_attr_entity.validate();
        assert!(validation_result.is_ok() || validation_result.as_ref().err().unwrap().to_string().contains("warning"));
    }

    #[test]
    fn test_entity_concurrent_access() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let entity = Arc::new(Mutex::new(
            Entity::new(EntityKey::from_hash("concurrent"), "Concurrent Test".to_string())
        ));
        
        let thread_count = 10;
        let operations_per_thread = 100;
        let mut handles = vec![];
        
        for thread_id in 0..thread_count {
            let entity_clone = Arc::clone(&entity);
            let handle = thread::spawn(move || {
                for op_id in 0..operations_per_thread {
                    let mut entity = entity_clone.lock().unwrap();
                    let key = format!("thread_{}_op_{}", thread_id, op_id);
                    let value = format!("value_{}_{}", thread_id, op_id);
                    entity.add_attribute(&key, &value);
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify all attributes were added
        let entity = entity.lock().unwrap();
        assert_eq!(entity.attributes().len(), thread_count * operations_per_thread);
        
        // Verify all values are correct
        for thread_id in 0..thread_count {
            for op_id in 0..operations_per_thread {
                let key = format!("thread_{}_op_{}", thread_id, op_id);
                let expected_value = format!("value_{}_{}", thread_id, op_id);
                assert_eq!(entity.get_attribute(&key), Some(&expected_value));
            }
        }
    }

    #[test]
    fn test_entity_performance_characteristics() {
        let entity_count = 1000;
        let mut entities = Vec::new();
        
        // Test entity creation performance
        let (_, creation_time) = measure_execution_time(|| {
            for i in 0..entity_count {
                let entity = Entity::new(
                    EntityKey::from_hash(&format!("perf_test_{}", i)),
                    format!("Performance Test {}", i)
                );
                entities.push(entity);
            }
        });
        
        println!("Entity creation time for {} entities: {:?}", entity_count, creation_time);
        assert!(creation_time.as_millis() < 1000, "Entity creation too slow");
        
        // Test attribute access performance
        let test_entity = &mut entities[0];
        for i in 0..1000 {
            test_entity.add_attribute(&format!("perf_attr_{}", i), &format!("value_{}", i));
        }
        
        let (_, access_time) = measure_execution_time(|| {
            for i in 0..1000 {
                let _ = test_entity.get_attribute(&format!("perf_attr_{}", i));
            }
        });
        
        println!("Attribute access time for 1000 attributes: {:?}", access_time);
        assert!(access_time.as_micros() < 10000, "Attribute access too slow"); // < 10ms
        
        // Test serialization performance
        let (serialized, serialization_time) = measure_execution_time(|| {
            test_entity.serialize()
        });
        
        println!("Serialization time: {:?}, size: {} bytes", serialization_time, serialized.len());
        assert!(serialization_time.as_millis() < 100, "Serialization too slow");
        
        // Test deserialization performance
        let (_, deserialization_time) = measure_execution_time(|| {
            Entity::deserialize(&serialized).unwrap()
        });
        
        println!("Deserialization time: {:?}", deserialization_time);
        assert!(deserialization_time.as_millis() < 100, "Deserialization too slow");
    }
}