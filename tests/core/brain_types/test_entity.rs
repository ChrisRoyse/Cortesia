//! Comprehensive tests for BrainInspiredEntity functionality
//! 
//! This module provides exhaustive testing for brain-inspired entities including:
//! - Entity creation and initialization
//! - Activation state management with temporal decay
//! - Property and embedding management
//! - Temporal tracking and updates
//! - Edge cases and boundary conditions
//! - Performance characteristics

use crate::core::brain_types::{BrainInspiredEntity, EntityDirection};
use crate::core::types::{EntityKey, AttributeValue};
use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use std::thread;

/// Test fixture for creating various brain-inspired entities
pub struct EntityTestFixture {
    pub input_entity: BrainInspiredEntity,
    pub output_entity: BrainInspiredEntity,
    pub gate_entity: BrainInspiredEntity,
    pub hidden_entity: BrainInspiredEntity,
}

impl EntityTestFixture {
    pub fn new() -> Self {
        Self {
            input_entity: BrainInspiredEntity::new("input_concept".to_string(), EntityDirection::Input),
            output_entity: BrainInspiredEntity::new("output_concept".to_string(), EntityDirection::Output),
            gate_entity: BrainInspiredEntity::new("gate_concept".to_string(), EntityDirection::Gate),
            hidden_entity: BrainInspiredEntity::new("hidden_concept".to_string(), EntityDirection::Hidden),
        }
    }
    
    pub fn create_entity_with_properties() -> BrainInspiredEntity {
        let mut entity = BrainInspiredEntity::new("test_concept".to_string(), EntityDirection::Input);
        entity.properties.insert("name".to_string(), AttributeValue::String("Test Entity".to_string()));
        entity.properties.insert("score".to_string(), AttributeValue::Number(0.85));
        entity.properties.insert("active".to_string(), AttributeValue::Boolean(true));
        entity.properties.insert("tags".to_string(), AttributeValue::Array(vec![
            AttributeValue::String("neural".to_string()),
            AttributeValue::String("test".to_string()),
        ]));
        entity.embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        entity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Entity Creation Tests
    #[test]
    fn test_entity_creation() {
        let entity = BrainInspiredEntity::new("test_concept".to_string(), EntityDirection::Input);
        
        assert_eq!(entity.concept_id, "test_concept");
        assert_eq!(entity.direction, EntityDirection::Input);
        assert!(entity.properties.is_empty());
        assert!(entity.embedding.is_empty());
        assert_eq!(entity.activation_state, 0.0);
        
        // Temporal fields should be initialized to now
        let now = SystemTime::now();
        let time_diff = now.duration_since(entity.last_activation).unwrap_or_default();
        assert!(time_diff < Duration::from_millis(100), "last_activation should be recent");
        
        let time_diff = now.duration_since(entity.last_update).unwrap_or_default();
        assert!(time_diff < Duration::from_millis(100), "last_update should be recent");
    }

    #[test]
    fn test_entity_creation_different_directions() {
        let input_entity = BrainInspiredEntity::new("input".to_string(), EntityDirection::Input);
        let output_entity = BrainInspiredEntity::new("output".to_string(), EntityDirection::Output);
        let gate_entity = BrainInspiredEntity::new("gate".to_string(), EntityDirection::Gate);
        let hidden_entity = BrainInspiredEntity::new("hidden".to_string(), EntityDirection::Hidden);
        
        assert_eq!(input_entity.direction, EntityDirection::Input);
        assert_eq!(output_entity.direction, EntityDirection::Output);
        assert_eq!(gate_entity.direction, EntityDirection::Gate);
        assert_eq!(hidden_entity.direction, EntityDirection::Hidden);
    }

    // Activation Tests
    #[test]
    fn test_basic_activation() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        let result = entity.activate(0.7, 0.1);
        assert_eq!(result, 0.7, "First activation should return the activation level");
        assert_eq!(entity.activation_state, 0.7, "Entity state should be updated");
        
        let activation_time = entity.last_activation;
        let update_time = entity.last_update;
        assert!(activation_time <= SystemTime::now(), "Activation time should be updated");
        assert!(update_time <= activation_time, "Update time should be before or equal to activation time");
    }

    #[test]
    fn test_activation_accumulation() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        // First activation
        entity.activate(0.5, 0.1);
        assert_eq!(entity.activation_state, 0.5);
        
        // Second activation immediately (no decay)
        let result = entity.activate(0.4, 0.1);
        assert_eq!(result, 0.9, "Activations should accumulate");
        assert_eq!(entity.activation_state, 0.9);
    }

    #[test]
    fn test_activation_clamping() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        // Activate beyond maximum
        let result = entity.activate(0.8, 0.1);
        assert_eq!(result, 0.8);
        
        let result = entity.activate(0.5, 0.1);
        assert_eq!(result, 1.0, "Activation should be clamped to 1.0");
        assert_eq!(entity.activation_state, 1.0);
    }

    #[test]
    fn test_activation_with_zero() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        let result = entity.activate(0.0, 0.1);
        assert_eq!(result, 0.0, "Zero activation should work");
        assert_eq!(entity.activation_state, 0.0);
    }

    #[test]
    fn test_negative_activation() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        entity.activation_state = 0.5; // Set initial state
        
        // Negative activation should still work (though unusual)
        let result = entity.activate(-0.2, 0.1);
        assert_eq!(result, 0.3, "Negative activation should reduce total activation");
        assert_eq!(entity.activation_state, 0.3);
    }

    // Temporal Decay Tests
    #[test]
    fn test_temporal_decay_immediate() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        entity.activation_state = 1.0;
        
        // Immediate activation (no time passed, no decay)
        let result = entity.activate(0.0, 0.1);
        assert_eq!(result, 1.0, "No decay should occur with immediate activation");
    }

    #[test]
    fn test_temporal_decay_calculation() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        entity.activation_state = 1.0;
        
        // Manually set last_activation to simulate time passing
        entity.last_activation = SystemTime::now() - Duration::from_secs(1);
        
        let result = entity.activate(0.0, 0.1); // decay_rate = 0.1, time = 1 sec
        // Expected: 1.0 * exp(-0.1 * 1) ≈ 1.0 * 0.9048 ≈ 0.9048
        assert!((result - 0.9048).abs() < 0.01, "Decay should follow exponential formula");
    }

    #[test]
    fn test_high_decay_rate() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        entity.activation_state = 1.0;
        entity.last_activation = SystemTime::now() - Duration::from_secs(2);
        
        let result = entity.activate(0.0, 1.0); // High decay rate
        // Expected: 1.0 * exp(-1.0 * 2) ≈ 1.0 * 0.1353 ≈ 0.1353
        assert!((result - 0.1353).abs() < 0.01, "High decay rate should significantly reduce activation");
    }

    #[test]
    fn test_decay_with_new_activation() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        entity.activation_state = 0.8;
        entity.last_activation = SystemTime::now() - Duration::from_secs(1);
        
        let result = entity.activate(0.3, 0.2);
        // Decayed: 0.8 * exp(-0.2 * 1) ≈ 0.8 * 0.8187 ≈ 0.6550
        // Plus new: 0.6550 + 0.3 = 0.9550
        assert!((result - 0.9550).abs() < 0.01, "Should combine decay and new activation");
    }

    #[test]
    fn test_very_long_decay() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        entity.activation_state = 1.0;
        entity.last_activation = SystemTime::now() - Duration::from_secs(3600); // 1 hour
        
        let result = entity.activate(0.0, 0.1);
        // After 1 hour with decay rate 0.1, activation should be very small
        assert!(result < 0.01, "Very long decay should reduce activation to near zero");
    }

    // Property Management Tests
    #[test]
    fn test_property_management() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        // Add various property types
        entity.properties.insert("name".to_string(), AttributeValue::String("Test Entity".to_string()));
        entity.properties.insert("score".to_string(), AttributeValue::Number(0.95));
        entity.properties.insert("enabled".to_string(), AttributeValue::Boolean(true));
        entity.properties.insert("metadata".to_string(), AttributeValue::Null);
        
        assert_eq!(entity.properties.len(), 4);
        
        // Test property retrieval
        if let Some(AttributeValue::String(name)) = entity.properties.get("name") {
            assert_eq!(name, "Test Entity");
        } else {
            panic!("Expected string property");
        }
        
        if let Some(AttributeValue::Number(score)) = entity.properties.get("score") {
            assert_eq!(*score, 0.95);
        } else {
            panic!("Expected number property");
        }
    }

    #[test]
    fn test_complex_properties() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        // Complex array property
        entity.properties.insert("tags".to_string(), AttributeValue::Array(vec![
            AttributeValue::String("neural".to_string()),
            AttributeValue::String("ai".to_string()),
            AttributeValue::Number(1.0),
        ]));
        
        // Complex object property
        let mut obj = HashMap::new();
        obj.insert("x".to_string(), AttributeValue::Number(10.0));
        obj.insert("y".to_string(), AttributeValue::Number(20.0));
        entity.properties.insert("position".to_string(), AttributeValue::Object(obj));
        
        // Vector property
        entity.properties.insert("weights".to_string(), AttributeValue::Vector(vec![0.1, 0.2, 0.3]));
        
        assert_eq!(entity.properties.len(), 3);
        
        // Verify complex properties
        if let Some(AttributeValue::Array(tags)) = entity.properties.get("tags") {
            assert_eq!(tags.len(), 3);
        } else {
            panic!("Expected array property");
        }
        
        if let Some(AttributeValue::Object(pos)) = entity.properties.get("position") {
            assert_eq!(pos.len(), 2);
            assert!(pos.contains_key("x"));
            assert!(pos.contains_key("y"));
        } else {
            panic!("Expected object property");
        }
    }

    // Embedding Tests
    #[test]
    fn test_embedding_management() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        assert!(entity.embedding.is_empty());
        
        // Set embedding
        entity.embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        assert_eq!(entity.embedding.len(), 5);
        assert_eq!(entity.embedding[0], 0.1);
        assert_eq!(entity.embedding[4], 0.5);
    }

    #[test]
    fn test_large_embedding() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        // Large embedding vector
        let large_embedding: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        entity.embedding = large_embedding;
        
        assert_eq!(entity.embedding.len(), 1000);
        assert_eq!(entity.embedding[0], 0.0);
        assert_eq!(entity.embedding[999], 0.999);
    }

    #[test]
    fn test_embedding_with_special_values() {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        
        entity.embedding = vec![
            0.0,
            -1.0,
            1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        
        assert_eq!(entity.embedding.len(), 6);
        assert_eq!(entity.embedding[0], 0.0);
        assert_eq!(entity.embedding[1], -1.0);
        assert_eq!(entity.embedding[2], 1.0);
        assert_eq!(entity.embedding[3], f32::INFINITY);
        assert_eq!(entity.embedding[4], f32::NEG_INFINITY);
        assert!(entity.embedding[5].is_nan());
    }

    // Serialization Tests
    #[test]
    fn test_entity_serialization() {
        let entity = EntityTestFixture::create_entity_with_properties();
        
        // Test serialization to JSON
        let serialized = serde_json::to_string(&entity);
        assert!(serialized.is_ok(), "Entity should be serializable");
        
        let json_str = serialized.unwrap();
        assert!(json_str.contains("test_concept"));
        assert!(json_str.contains("Input"));
        assert!(json_str.contains("Test Entity"));
    }

    #[test]
    fn test_entity_deserialization() {
        let entity = EntityTestFixture::create_entity_with_properties();
        
        // Serialize and deserialize
        let serialized = serde_json::to_string(&entity).unwrap();
        let deserialized: Result<BrainInspiredEntity, _> = serde_json::from_str(&serialized);
        
        assert!(deserialized.is_ok(), "Entity should be deserializable");
        
        let restored_entity = deserialized.unwrap();
        assert_eq!(restored_entity.concept_id, entity.concept_id);
        assert_eq!(restored_entity.direction, entity.direction);
        assert_eq!(restored_entity.properties.len(), entity.properties.len());
        assert_eq!(restored_entity.embedding.len(), entity.embedding.len());
        assert_eq!(restored_entity.activation_state, entity.activation_state);
    }

    // Edge Cases and Error Handling
    #[test]
    fn test_empty_concept_id() {
        let entity = BrainInspiredEntity::new("".to_string(), EntityDirection::Input);
        assert_eq!(entity.concept_id, "");
        // Should still work with empty concept ID
    }

    #[test]
    fn test_very_long_concept_id() {
        let long_id = "a".repeat(10000);
        let entity = BrainInspiredEntity::new(long_id.clone(), EntityDirection::Input);
        assert_eq!(entity.concept_id, long_id);
        assert_eq!(entity.concept_id.len(), 10000);
    }

    #[test]
    fn test_unicode_concept_id() {
        let unicode_id = "概念_🧠_test_αβγ".to_string();
        let entity = BrainInspiredEntity::new(unicode_id.clone(), EntityDirection::Input);
        assert_eq!(entity.concept_id, unicode_id);
    }

    #[test]
    fn test_entity_cloning() {
        let entity = EntityTestFixture::create_entity_with_properties();
        let cloned_entity = entity.clone();
        
        assert_eq!(entity.concept_id, cloned_entity.concept_id);
        assert_eq!(entity.direction, cloned_entity.direction);
        assert_eq!(entity.properties.len(), cloned_entity.properties.len());
        assert_eq!(entity.embedding.len(), cloned_entity.embedding.len());
        assert_eq!(entity.activation_state, cloned_entity.activation_state);
        
        // Ensure it's a deep clone (not sharing references)
        // This would be more relevant for mutable references, but worth checking
    }

    // Concurrency and Thread Safety Tests
    #[test]
    fn test_entity_across_threads() {
        use std::sync::{Arc, Mutex};
        use std::thread;
        
        let entity = Arc::new(Mutex::new(BrainInspiredEntity::new(
            "concurrent_test".to_string(),
            EntityDirection::Input
        )));
        
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let entity_clone = Arc::clone(&entity);
                thread::spawn(move || {
                    let mut entity_guard = entity_clone.lock().unwrap();
                    entity_guard.activate(0.1, 0.01);
                    entity_guard.properties.insert(
                        format!("thread_{}", i),
                        AttributeValue::Number(i as f64)
                    );
                })
            })
            .collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let final_entity = entity.lock().unwrap();
        assert!(final_entity.activation_state > 0.0, "Entity should have been activated");
        assert_eq!(final_entity.properties.len(), 10, "Should have properties from all threads");
    }

    // Performance Tests
    #[test]
    fn test_activation_performance() {
        let mut entity = BrainInspiredEntity::new("perf_test".to_string(), EntityDirection::Input);
        
        let start = std::time::Instant::now();
        for i in 0..10000 {
            entity.activate(0.001, 0.01);
            if i % 1000 == 0 {
                // Reset occasionally to prevent saturation
                entity.activation_state = 0.0;
            }
        }
        let duration = start.elapsed();
        
        assert!(duration.as_millis() < 1000, "10000 activations should complete quickly");
    }

    #[test]
    fn test_property_access_performance() {
        let entity = EntityTestFixture::create_entity_with_properties();
        
        let start = std::time::Instant::now();
        for _ in 0..100000 {
            let _ = entity.properties.get("name");
            let _ = entity.properties.get("score");
            let _ = entity.properties.get("active");
            let _ = entity.properties.get("nonexistent");
        }
        let duration = start.elapsed();
        
        assert!(duration.as_millis() < 1000, "Property access should be fast");
    }

    #[test]
    fn test_large_properties_performance() {
        let mut entity = BrainInspiredEntity::new("large_props".to_string(), EntityDirection::Input);
        
        // Add many properties
        for i in 0..1000 {
            entity.properties.insert(
                format!("prop_{}", i),
                AttributeValue::Number(i as f64)
            );
        }
        
        assert_eq!(entity.properties.len(), 1000);
        
        let start = std::time::Instant::now();
        for i in 0..1000 {
            let _ = entity.properties.get(&format!("prop_{}", i));
        }
        let duration = start.elapsed();
        
        assert!(duration.as_millis() < 100, "Large property map access should be efficient");
    }

    // Boundary Value Tests
    #[test]
    fn test_extreme_activation_values() {
        let mut entity = BrainInspiredEntity::new("extreme".to_string(), EntityDirection::Input);
        
        // Test with very small values
        entity.activate(f32::MIN_POSITIVE, 0.1);
        assert!(entity.activation_state >= 0.0);
        
        // Test with very large values
        entity.activation_state = 0.0; // Reset
        entity.activate(f32::MAX, 0.1);
        assert_eq!(entity.activation_state, 1.0, "Should clamp to 1.0");
        
        // Test with infinity
        entity.activation_state = 0.0; // Reset
        entity.activate(f32::INFINITY, 0.1);
        assert_eq!(entity.activation_state, 1.0, "Should handle infinity");
    }

    #[test]
    fn test_extreme_decay_rates() {
        let mut entity = BrainInspiredEntity::new("decay_test".to_string(), EntityDirection::Input);
        entity.activation_state = 1.0;
        entity.last_activation = SystemTime::now() - Duration::from_secs(1);
        
        // Very small decay rate
        let result = entity.activate(0.0, f32::MIN_POSITIVE);
        assert!(result > 0.99, "Very small decay should barely affect activation");
        
        // Very large decay rate
        entity.activation_state = 1.0;
        entity.last_activation = SystemTime::now() - Duration::from_secs(1);
        let result = entity.activate(0.0, f32::MAX);
        assert!(result < 0.01, "Very large decay should nearly eliminate activation");
    }

    // State Consistency Tests
    #[test]
    fn test_activation_state_consistency() {
        let mut entity = BrainInspiredEntity::new("consistency".to_string(), EntityDirection::Input);
        
        // Multiple activations should maintain consistency
        for i in 1..=10 {
            let before_state = entity.activation_state;
            let activation_amount = 0.05;
            let result = entity.activate(activation_amount, 0.01);
            
            // Result should match entity state
            assert_eq!(result, entity.activation_state, "Returned value should match entity state");
            
            // State should never exceed 1.0
            assert!(entity.activation_state <= 1.0, "State should be clamped to 1.0");
            
            // State should be non-negative
            assert!(entity.activation_state >= 0.0, "State should be non-negative");
        }
    }

    #[test]
    fn test_temporal_consistency() {
        let mut entity = BrainInspiredEntity::new("temporal".to_string(), EntityDirection::Input);
        let initial_time = entity.last_activation;
        
        // Small delay
        thread::sleep(Duration::from_millis(10));
        entity.activate(0.5, 0.1);
        
        // Activation time should be updated
        assert!(entity.last_activation > initial_time, "Activation time should be updated");
        assert!(entity.last_update >= entity.last_activation, "Update time should be recent");
    }

    // Integration with EntityKey Tests
    #[test]
    fn test_entity_key_handling() {
        let entity = BrainInspiredEntity::new("key_test".to_string(), EntityDirection::Input);
        
        // EntityKey should be initialized (default)
        // We can't easily test specific values since EntityKey is opaque
        // But we can verify it exists and can be used
        let _key = entity.id;
        
        // Test that entities can have different keys
        let entity2 = BrainInspiredEntity::new("key_test2".to_string(), EntityDirection::Output);
        // Keys might be the same if using default(), but that's implementation-dependent
    }

    // Memory and Resource Tests
    #[test]
    fn test_memory_usage() {
        // Create many entities to test memory usage patterns
        let mut entities = Vec::new();
        
        for i in 0..1000 {
            let mut entity = BrainInspiredEntity::new(
                format!("entity_{}", i),
                match i % 4 {
                    0 => EntityDirection::Input,
                    1 => EntityDirection::Output,
                    2 => EntityDirection::Gate,
                    _ => EntityDirection::Hidden,
                }
            );
            
            // Add some properties and embedding
            entity.properties.insert("id".to_string(), AttributeValue::Number(i as f64));
            entity.embedding = vec![0.1; 100]; // 100-dimensional embedding
            
            entities.push(entity);
        }
        
        assert_eq!(entities.len(), 1000);
        
        // Test accessing random entities
        for _ in 0..100 {
            let idx = (SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos() % 1000) as usize;
            let entity = &entities[idx];
            assert!(entity.concept_id.starts_with("entity_"));
        }
    }
}