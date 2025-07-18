//! Core Types Unit Tests
//!
//! Tests for fundamental data types, type conversions,
//! serialization, and type safety validation.

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::core::types::*;

#[cfg(test)]
mod types_tests {
    use super::*;

    #[test]
    fn test_entity_key_operations() {
        // Test key creation from different inputs
        let key1 = EntityKey::from_hash("test_string");
        let key2 = EntityKey::from_hash("test_string");
        let key3 = EntityKey::from_hash("different_string");
        
        // Test deterministic generation
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
        
        // Test key serialization
        let serialized = key1.to_bytes();
        let deserialized = EntityKey::from_bytes(&serialized).unwrap();
        assert_eq!(key1, deserialized);
        
        // Test string representation
        let key_str = key1.to_string();
        let from_str = EntityKey::from_string(&key_str).unwrap();
        assert_eq!(key1, from_str);
        
        // Test hash distribution
        let mut keys = std::collections::HashSet::new();
        for i in 0..10000 {
            let key = EntityKey::from_hash(&format!("key_{}", i));
            assert!(!keys.contains(&key), "Duplicate key found for input {}", i);
            keys.insert(key);
        }
    }

    #[test]
    fn test_relationship_types() {
        // Test directed relationship
        let directed = RelationshipType::Directed;
        assert_eq!(directed.is_bidirectional(), false);
        assert_eq!(directed.to_string(), "directed");
        
        // Test undirected relationship
        let undirected = RelationshipType::Undirected;
        assert_eq!(undirected.is_bidirectional(), true);
        assert_eq!(undirected.to_string(), "undirected");
        
        // Test weighted relationship
        let weighted = RelationshipType::Weighted;
        assert_eq!(weighted.supports_weights(), true);
        assert_eq!(weighted.to_string(), "weighted");
        
        // Test type comparison
        assert_ne!(directed, undirected);
        assert_ne!(weighted, directed);
        
        // Test serialization
        let serialized_directed = serde_json::to_string(&directed).unwrap();
        let deserialized_directed: RelationshipType = serde_json::from_str(&serialized_directed).unwrap();
        assert_eq!(directed, deserialized_directed);
    }

    #[test]
    fn test_weight_validation() {
        // Test valid weights
        assert!(Weight::new(0.0).is_ok());
        assert!(Weight::new(0.5).is_ok());
        assert!(Weight::new(1.0).is_ok());
        
        // Test invalid weights
        assert!(Weight::new(-0.1).is_err());
        assert!(Weight::new(1.1).is_err());
        assert!(Weight::new(f32::NAN).is_err());
        assert!(Weight::new(f32::INFINITY).is_err());
        assert!(Weight::new(f32::NEG_INFINITY).is_err());
        
        // Test weight operations
        let w1 = Weight::new(0.3).unwrap();
        let w2 = Weight::new(0.7).unwrap();
        
        let sum = w1 + w2;
        assert_eq!(sum.value(), 1.0);
        
        let product = w1 * w2;
        assert_eq!(product.value(), 0.21);
        
        // Test weight normalization
        let weights = vec![
            Weight::new(0.2).unwrap(),
            Weight::new(0.3).unwrap(),
            Weight::new(0.5).unwrap(),
        ];
        
        let normalized = Weight::normalize(&weights);
        let sum: f32 = normalized.iter().map(|w| w.value()).sum();
        assert!((sum - 1.0).abs() < 1e-6, "Normalized weights should sum to 1.0");
    }

    #[test]
    fn test_attribute_value_types() {
        // Test string attributes
        let string_attr = AttributeValue::String("test_value".to_string());
        assert_eq!(string_attr.as_string(), Some("test_value"));
        assert_eq!(string_attr.as_number(), None);
        
        // Test numeric attributes
        let number_attr = AttributeValue::Number(42.5);
        assert_eq!(number_attr.as_number(), Some(42.5));
        assert_eq!(number_attr.as_string(), None);
        
        // Test boolean attributes
        let bool_attr = AttributeValue::Boolean(true);
        assert_eq!(bool_attr.as_boolean(), Some(true));
        assert_eq!(bool_attr.as_string(), None);
        
        // Test array attributes
        let array_attr = AttributeValue::Array(vec![
            AttributeValue::String("item1".to_string()),
            AttributeValue::Number(123.0),
            AttributeValue::Boolean(false),
        ]);
        
        if let Some(array) = array_attr.as_array() {
            assert_eq!(array.len(), 3);
            assert_eq!(array[0].as_string(), Some("item1"));
            assert_eq!(array[1].as_number(), Some(123.0));
            assert_eq!(array[2].as_boolean(), Some(false));
        } else {
            panic!("Array attribute should return array");
        }
        
        // Test object attributes
        let mut object = std::collections::HashMap::new();
        object.insert("key1".to_string(), AttributeValue::String("value1".to_string()));
        object.insert("key2".to_string(), AttributeValue::Number(456.0));
        
        let object_attr = AttributeValue::Object(object);
        if let Some(obj) = object_attr.as_object() {
            assert_eq!(obj.get("key1").unwrap().as_string(), Some("value1"));
            assert_eq!(obj.get("key2").unwrap().as_number(), Some(456.0));
        } else {
            panic!("Object attribute should return object");
        }
    }

    #[test]
    fn test_attribute_value_conversion() {
        // Test automatic type conversion
        let string_number = AttributeValue::from_string("123.45");
        assert_eq!(string_number.try_as_number(), Ok(123.45));
        
        let string_bool = AttributeValue::from_string("true");
        assert_eq!(string_bool.try_as_boolean(), Ok(true));
        
        let number_string = AttributeValue::Number(789.0);
        assert_eq!(number_string.to_string(), "789");
        
        // Test conversion errors
        let invalid_number = AttributeValue::from_string("not_a_number");
        assert!(invalid_number.try_as_number().is_err());
        
        let invalid_bool = AttributeValue::from_string("maybe");
        assert!(invalid_bool.try_as_boolean().is_err());
    }

    #[test]
    fn test_attribute_value_serialization() {
        let test_values = vec![
            AttributeValue::String("test".to_string()),
            AttributeValue::Number(42.5),
            AttributeValue::Boolean(true),
            AttributeValue::Array(vec![
                AttributeValue::String("item1".to_string()),
                AttributeValue::Number(123.0),
            ]),
        ];
        
        for value in test_values {
            // Test JSON serialization
            let json = value.to_json().unwrap();
            let from_json = AttributeValue::from_json(&json).unwrap();
            assert_eq!(value, from_json);
            
            // Test binary serialization
            let binary = value.to_binary().unwrap();
            let from_binary = AttributeValue::from_binary(&binary).unwrap();
            assert_eq!(value, from_binary);
            
            // Verify binary is more compact for complex types
            if value.is_complex() {
                assert!(binary.len() <= json.len());
            }
        }
    }

    #[test]
    fn test_timestamp_types() {
        // Test timestamp creation
        let now = Timestamp::now();
        let epoch = Timestamp::epoch();
        let custom = Timestamp::from_millis(1234567890123);
        
        assert!(now.as_millis() > epoch.as_millis());
        assert_eq!(custom.as_millis(), 1234567890123);
        
        // Test timestamp operations
        let earlier = Timestamp::from_millis(1000);
        let later = Timestamp::from_millis(2000);
        
        assert!(later > earlier);
        assert!(earlier < later);
        assert_eq!(later - earlier, Duration::from_millis(1000));
        
        // Test timestamp formatting
        let formatted = now.format_iso8601();
        let parsed = Timestamp::parse_iso8601(&formatted).unwrap();
        assert_eq!(now.as_millis(), parsed.as_millis());
        
        // Test timestamp arithmetic
        let base = Timestamp::from_millis(5000);
        let duration = Duration::from_millis(1000);
        
        let future = base + duration;
        assert_eq!(future.as_millis(), 6000);
        
        let past = base - duration;
        assert_eq!(past.as_millis(), 4000);
    }

    #[test]
    fn test_coordinate_types() {
        // Test 2D coordinates
        let coord_2d = Coordinate2D::new(10.5, 20.7);
        assert_eq!(coord_2d.x(), 10.5);
        assert_eq!(coord_2d.y(), 20.7);
        
        let distance_2d = coord_2d.distance_to(&Coordinate2D::new(13.5, 24.7));
        assert!((distance_2d - 5.0).abs() < 0.1); // Approximately 5.0
        
        // Test 3D coordinates
        let coord_3d = Coordinate3D::new(1.0, 2.0, 3.0);
        assert_eq!(coord_3d.x(), 1.0);
        assert_eq!(coord_3d.y(), 2.0);
        assert_eq!(coord_3d.z(), 3.0);
        
        let distance_3d = coord_3d.distance_to(&Coordinate3D::new(4.0, 6.0, 7.0));
        assert!((distance_3d - 7.07).abs() < 0.1); // Approximately sqrt(50)
        
        // Test coordinate serialization
        let serialized_2d = coord_2d.to_string();
        let parsed_2d = Coordinate2D::from_string(&serialized_2d).unwrap();
        assert_eq!(coord_2d, parsed_2d);
        
        let serialized_3d = coord_3d.to_string();
        let parsed_3d = Coordinate3D::from_string(&serialized_3d).unwrap();
        assert_eq!(coord_3d, parsed_3d);
    }

    #[test]
    fn test_version_types() {
        // Test version creation
        let v1 = Version::new(1, 0, 0);
        let v2 = Version::new(1, 2, 3);
        let v3 = Version::new(2, 0, 0);
        
        // Test version comparison
        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1 < v3);
        
        // Test version parsing
        let parsed = Version::parse("1.2.3").unwrap();
        assert_eq!(parsed, v2);
        
        let invalid_parse = Version::parse("invalid.version");
        assert!(invalid_parse.is_err());
        
        // Test version formatting
        assert_eq!(v2.to_string(), "1.2.3");
        
        // Test version compatibility
        assert!(v2.is_compatible_with(&Version::new(1, 2, 4))); // Patch difference OK
        assert!(v2.is_compatible_with(&Version::new(1, 3, 0))); // Minor increase OK
        assert!(!v2.is_compatible_with(&Version::new(2, 0, 0))); // Major difference not OK
        assert!(!v2.is_compatible_with(&Version::new(1, 1, 9))); // Minor decrease not OK
    }

    #[test]
    fn test_uuid_types() {
        // Test UUID generation
        let uuid1 = Uuid::generate();
        let uuid2 = Uuid::generate();
        
        assert_ne!(uuid1, uuid2);
        assert_eq!(uuid1.to_string().len(), 36); // Standard UUID string length
        
        // Test UUID parsing
        let uuid_str = uuid1.to_string();
        let parsed_uuid = Uuid::parse(&uuid_str).unwrap();
        assert_eq!(uuid1, parsed_uuid);
        
        // Test nil UUID
        let nil_uuid = Uuid::nil();
        assert_eq!(nil_uuid.to_string(), "00000000-0000-0000-0000-000000000000");
        
        // Test UUID validation
        assert!(Uuid::is_valid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!Uuid::is_valid("invalid-uuid"));
        assert!(!Uuid::is_valid("550e8400-e29b-41d4-a716")); // Too short
        
        // Test UUID bytes
        let bytes = uuid1.as_bytes();
        assert_eq!(bytes.len(), 16);
        
        let from_bytes = Uuid::from_bytes(bytes).unwrap();
        assert_eq!(uuid1, from_bytes);
    }

    #[test]
    fn test_type_safety() {
        // Test that type conversions fail appropriately
        let string_val = AttributeValue::String("not_a_number".to_string());
        assert!(string_val.try_as_number().is_err());
        
        let number_val = AttributeValue::Number(42.0);
        assert!(number_val.try_as_boolean().is_err());
        
        // Test weight bounds enforcement
        assert!(Weight::new(-1.0).is_err());
        assert!(Weight::new(2.0).is_err());
        
        // Test coordinate validation
        assert!(Coordinate2D::new(f32::NAN, 0.0).is_err());
        assert!(Coordinate3D::new(0.0, f32::INFINITY, 0.0).is_err());
        
        // Test timestamp bounds
        assert!(Timestamp::from_millis(u64::MAX).is_err()); // Too large
        
        // Test version component limits
        assert!(Version::new(u32::MAX, 0, 0).is_err()); // Major too large
    }

    #[test]
    fn test_type_performance() {
        let iterations = 100000;
        
        // Test EntityKey creation performance
        let (keys, key_time) = measure_execution_time(|| {
            let mut keys = Vec::new();
            for i in 0..iterations {
                let key = EntityKey::from_hash(&format!("key_{}", i));
                keys.push(key);
            }
            keys
        });
        
        println!("EntityKey creation time for {} keys: {:?}", iterations, key_time);
        assert!(key_time.as_millis() < 1000, "EntityKey creation too slow");
        
        // Test Weight operations performance
        let weights: Vec<Weight> = (0..iterations)
            .map(|i| Weight::new((i as f32) / (iterations as f32)).unwrap())
            .collect();
        
        let (_, weight_ops_time) = measure_execution_time(|| {
            let mut sum = Weight::new(0.0).unwrap();
            for weight in &weights {
                sum = sum + *weight;
            }
        });
        
        println!("Weight operations time for {} operations: {:?}", iterations, weight_ops_time);
        assert!(weight_ops_time.as_millis() < 100, "Weight operations too slow");
        
        // Test AttributeValue serialization performance
        let test_value = AttributeValue::Array((0..1000)
            .map(|i| AttributeValue::String(format!("item_{}", i)))
            .collect());
        
        let (_, serialization_time) = measure_execution_time(|| {
            for _ in 0..100 {
                let _ = test_value.to_json().unwrap();
            }
        });
        
        println!("AttributeValue serialization time for 100 operations: {:?}", serialization_time);
        assert!(serialization_time.as_millis() < 1000, "AttributeValue serialization too slow");
    }

    #[test]
    fn test_type_memory_usage() {
        // Test memory usage of core types
        let entity_key = EntityKey::from_hash("test");
        assert_eq!(std::mem::size_of_val(&entity_key), 32); // 256-bit key
        
        let weight = Weight::new(0.5).unwrap();
        assert_eq!(std::mem::size_of_val(&weight), 4); // f32
        
        let timestamp = Timestamp::now();
        assert_eq!(std::mem::size_of_val(&timestamp), 8); // u64 milliseconds
        
        let coord_2d = Coordinate2D::new(1.0, 2.0);
        assert_eq!(std::mem::size_of_val(&coord_2d), 8); // 2 * f32
        
        let coord_3d = Coordinate3D::new(1.0, 2.0, 3.0);
        assert_eq!(std::mem::size_of_val(&coord_3d), 12); // 3 * f32
        
        let uuid = Uuid::generate();
        assert_eq!(std::mem::size_of_val(&uuid), 16); // 128-bit UUID
        
        let version = Version::new(1, 2, 3);
        assert_eq!(std::mem::size_of_val(&version), 12); // 3 * u32
        
        // Test AttributeValue memory for different types
        let string_attr = AttributeValue::String("test".to_string());
        let number_attr = AttributeValue::Number(42.0);
        let bool_attr = AttributeValue::Boolean(true);
        
        // String should be larger due to heap allocation
        let string_size = string_attr.memory_usage();
        let number_size = number_attr.memory_usage();
        let bool_size = bool_attr.memory_usage();
        
        assert!(string_size > number_size);
        assert!(number_size > bool_size);
        
        println!("AttributeValue memory usage - String: {}, Number: {}, Boolean: {}", 
                string_size, number_size, bool_size);
    }
}