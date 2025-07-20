// Tests for BrainInspiredEntity struct
// Validates neural entity behavior including activation, decay, and temporal properties

use llmkg::core::brain_types::{BrainInspiredEntity, EntityDirection};
use llmkg::core::types::{EntityKey, AttributeValue};
use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use std::thread;
use serde_json;

use super::test_constants;
use super::test_helpers::{EntityBuilder, assert_activation_close, assert_entity_direction, measure_execution_time};

// ==================== Constructor Tests ====================

#[test]
fn test_brain_inspired_entity_new() {
    let concept_id = test_constants::TEST_CONCEPT_INPUT.to_string();
    let direction = EntityDirection::Input;
    
    let entity = BrainInspiredEntity::new(concept_id.clone(), direction);
    
    assert_eq!(entity.concept_id, concept_id);
    assert_eq!(entity.direction, direction);
    assert_eq!(entity.activation_state, test_constants::RESTING_POTENTIAL);
    assert!(entity.properties.is_empty());
    assert!(entity.embedding.is_empty());
    
    // Check that timestamps are recent
    let now = SystemTime::now();
    let time_diff = now.duration_since(entity.last_activation).unwrap();
    assert!(time_diff < Duration::from_secs(1), "Timestamp should be recent");
}

#[test]
fn test_brain_inspired_entity_new_all_directions() {
    let directions = [
        EntityDirection::Input,
        EntityDirection::Output,
        EntityDirection::Gate,
        EntityDirection::Hidden,
    ];
    
    for direction in directions {
        let entity = BrainInspiredEntity::new("test_concept".to_string(), direction);
        assert_entity_direction(&entity, direction, "Constructor should set correct direction");
        assert_eq!(entity.activation_state, test_constants::RESTING_POTENTIAL);
    }
}

// ==================== Builder Pattern Tests ====================

#[test]
fn test_entity_builder_basic() {
    let entity = EntityBuilder::new(test_constants::TEST_CONCEPT_INPUT, EntityDirection::Input)
        .with_activation(test_constants::ACTION_POTENTIAL)
        .build();
    
    assert_eq!(entity.concept_id, test_constants::TEST_CONCEPT_INPUT);
    assert_eq!(entity.direction, EntityDirection::Input);
    assert_activation_close(entity.activation_state, test_constants::ACTION_POTENTIAL, 
                           test_constants::ACTIVATION_EPSILON, "Builder activation");
}

#[test]
fn test_entity_builder_with_properties() {
    let mut properties = HashMap::new();
    properties.insert("type".to_string(), AttributeValue::String("neuron".to_string()));
    
    let entity = EntityBuilder::new(test_constants::TEST_CONCEPT_HIDDEN, EntityDirection::Hidden)
        .with_property("type", AttributeValue::String("neuron".to_string()))
        .with_property("layer", AttributeValue::Integer(2))
        .build();
    
    assert_eq!(entity.properties.len(), 2);
    assert_eq!(entity.properties.get("type"), Some(&AttributeValue::String("neuron".to_string())));
    assert_eq!(entity.properties.get("layer"), Some(&AttributeValue::Integer(2)));
}

#[test]
fn test_entity_builder_with_embedding() {
    let embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    
    let entity = EntityBuilder::new(test_constants::TEST_CONCEPT_OUTPUT, EntityDirection::Output)
        .with_embedding(embedding.clone())
        .build();
    
    assert_eq!(entity.embedding, embedding);
    assert_eq!(entity.embedding.len(), 5);
}

// ==================== Activation Method Tests ====================

#[test]
fn test_entity_activate_basic() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    
    // Initial activation
    let result = entity.activate(test_constants::ACTION_POTENTIAL, test_constants::STANDARD_DECAY_RATE);
    
    assert_activation_close(result, test_constants::ACTION_POTENTIAL, 
                           test_constants::ACTIVATION_EPSILON, "First activation");
    assert_activation_close(entity.activation_state, test_constants::ACTION_POTENTIAL,
                           test_constants::ACTIVATION_EPSILON, "Entity state after activation");
}

#[test]
fn test_entity_activate_accumulation() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    
    // First activation
    entity.activate(test_constants::THRESHOLD_POTENTIAL, test_constants::STANDARD_DECAY_RATE);
    
    // Second activation (should accumulate but cap at 1.0)
    let result = entity.activate(test_constants::THRESHOLD_POTENTIAL, test_constants::STANDARD_DECAY_RATE);
    
    assert_activation_close(result, 1.0, test_constants::ACTIVATION_EPSILON, "Accumulated activation");
    assert!(entity.activation_state <= test_constants::SATURATION_LEVEL, 
            "Activation should not exceed saturation");
}

#[test]
fn test_entity_activate_saturation_clamping() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    
    // Activate with above-saturation value
    let result = entity.activate(test_constants::ABOVE_SATURATION, test_constants::STANDARD_DECAY_RATE);
    
    assert_activation_close(result, test_constants::SATURATION_LEVEL, 
                           test_constants::ACTIVATION_EPSILON, "Clamped activation");
    assert_eq!(entity.activation_state, test_constants::SATURATION_LEVEL);
}

#[test]
fn test_entity_activate_temporal_decay() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    
    // First activation
    entity.activate(test_constants::ACTION_POTENTIAL, test_constants::FAST_DECAY_RATE);
    let first_activation = entity.activation_state;
    
    // Wait a short time (simulate time passage)
    thread::sleep(Duration::from_millis(test_constants::DECAY_WAIT_MS));
    
    // Second activation with decay
    let result = entity.activate(test_constants::WEAK_EXCITATORY, test_constants::FAST_DECAY_RATE);
    
    // Should be less than simple addition due to decay
    assert!(result < first_activation + test_constants::WEAK_EXCITATORY,
            "Decay should reduce accumulated activation");
}

#[test]
fn test_entity_activate_no_decay() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    
    // Activate with no decay
    entity.activate(test_constants::THRESHOLD_POTENTIAL, test_constants::NO_DECAY_RATE);
    let first_state = entity.activation_state;
    
    // Wait and activate again
    thread::sleep(Duration::from_millis(10));
    entity.activate(test_constants::THRESHOLD_POTENTIAL, test_constants::NO_DECAY_RATE);
    
    // Should be exactly double (no decay, perfect accumulation)
    assert_activation_close(entity.activation_state, 
                           (first_state + test_constants::THRESHOLD_POTENTIAL).min(1.0),
                           test_constants::ACTIVATION_EPSILON, "No decay accumulation");
}

#[test]
fn test_entity_activate_different_decay_rates() {
    let decay_rates = [
        test_constants::NO_DECAY_RATE,
        test_constants::SLOW_DECAY_RATE,
        test_constants::STANDARD_DECAY_RATE,
        test_constants::FAST_DECAY_RATE,
    ];
    
    for &decay_rate in &decay_rates {
        let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        entity.activate(test_constants::ACTION_POTENTIAL, test_constants::STANDARD_DECAY_RATE);
        
        // Wait for decay
        thread::sleep(Duration::from_millis(50));
        
        let result = entity.activate(test_constants::WEAK_EXCITATORY, decay_rate);
        
        // Higher decay rates should result in more activation loss
        assert!(result >= 0.0 && result <= 1.0, "Activation should be in valid range");
    }
}

// ==================== Temporal Behavior Tests ====================

#[test]
fn test_entity_timestamp_updates() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    let initial_activation_time = entity.last_activation;
    let initial_update_time = entity.last_update;
    
    // Wait a bit
    thread::sleep(Duration::from_millis(10));
    
    // Activate the entity
    entity.activate(test_constants::ACTION_POTENTIAL, test_constants::STANDARD_DECAY_RATE);
    
    // Check that last_activation was updated
    assert!(entity.last_activation > initial_activation_time, 
            "last_activation should be updated");
    
    // last_update should still be the original (not explicitly updated by activate)
    assert_eq!(entity.last_update, initial_update_time);
}

#[test]
fn test_entity_activation_decay_over_time() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    
    // Activate to high level
    entity.activate(test_constants::ACTION_POTENTIAL, test_constants::STANDARD_DECAY_RATE);
    let high_activation = entity.activation_state;
    
    // Wait for significant time
    thread::sleep(Duration::from_millis(100));
    
    // Activate with zero input to just apply decay
    entity.activate(0.0, test_constants::STANDARD_DECAY_RATE);
    let decayed_activation = entity.activation_state;
    
    assert!(decayed_activation < high_activation, 
            "Activation should decay over time: {} -> {}", high_activation, decayed_activation);
}

// ==================== Serialization Tests ====================

#[test]
fn test_entity_serialization() {
    let entity = EntityBuilder::new(test_constants::TEST_CONCEPT_INPUT, EntityDirection::Input)
        .with_activation(test_constants::ACTION_POTENTIAL)
        .with_embedding(vec![0.1, 0.2, 0.3])
        .build();
    
    let serialized = serde_json::to_string(&entity).expect("Should serialize");
    
    // Check that key fields are present in JSON
    assert!(serialized.contains("concept_id"));
    assert!(serialized.contains("direction"));
    assert!(serialized.contains("activation_state"));
    assert!(serialized.contains("embedding"));
}

#[test]
fn test_entity_deserialization() {
    let entity = EntityBuilder::new(test_constants::TEST_CONCEPT_OUTPUT, EntityDirection::Output)
        .with_activation(test_constants::THRESHOLD_POTENTIAL)
        .build();
    
    let serialized = serde_json::to_string(&entity).expect("Should serialize");
    let deserialized: BrainInspiredEntity = serde_json::from_str(&serialized)
        .expect("Should deserialize");
    
    assert_eq!(deserialized.concept_id, entity.concept_id);
    assert_eq!(deserialized.direction, entity.direction);
    assert_activation_close(deserialized.activation_state, entity.activation_state,
                           test_constants::ACTIVATION_EPSILON, "Deserialized activation");
}

#[test]
fn test_entity_round_trip_serialization() {
    let original = EntityBuilder::new(test_constants::TEST_CONCEPT_GATE, EntityDirection::Gate)
        .with_activation(test_constants::ACTION_POTENTIAL)
        .with_embedding(vec![0.5, 0.7, 0.2, 0.9])
        .with_property("type", AttributeValue::String("logic_gate".to_string()))
        .build();
    
    let serialized = serde_json::to_string(&original).expect("Should serialize");
    let deserialized: BrainInspiredEntity = serde_json::from_str(&serialized)
        .expect("Should deserialize");
    
    assert_eq!(deserialized.concept_id, original.concept_id);
    assert_eq!(deserialized.direction, original.direction);
    assert_eq!(deserialized.embedding, original.embedding);
    assert_eq!(deserialized.properties.len(), original.properties.len());
}

// ==================== Edge Case Tests ====================

#[test]
fn test_entity_zero_activation() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    
    let result = entity.activate(0.0, test_constants::STANDARD_DECAY_RATE);
    
    assert_eq!(result, 0.0);
    assert_eq!(entity.activation_state, 0.0);
}

#[test]
fn test_entity_negative_activation_handling() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    
    // Start with some activation
    entity.activate(test_constants::WEAK_EXCITATORY, test_constants::NO_DECAY_RATE);
    
    // Try to apply negative activation (should not go below 0)
    let result = entity.activate(-0.5, test_constants::NO_DECAY_RATE);
    
    assert!(result >= 0.0, "Activation should not go negative");
    assert!(entity.activation_state >= 0.0, "Entity state should not go negative");
}

#[test]
fn test_entity_extreme_decay_rate() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    
    // Activate with extreme decay rate
    entity.activate(test_constants::ACTION_POTENTIAL, 100.0); // Very high decay
    
    thread::sleep(Duration::from_millis(10));
    
    // Should decay to essentially zero very quickly
    entity.activate(0.0, 100.0);
    
    assert!(entity.activation_state < 0.1, 
            "High decay rate should reduce activation significantly");
}

#[test]
fn test_entity_empty_concept_id() {
    let entity = BrainInspiredEntity::new("".to_string(), EntityDirection::Input);
    
    assert_eq!(entity.concept_id, "");
    assert_eq!(entity.direction, EntityDirection::Input);
}

#[test]
fn test_entity_large_embedding() {
    let large_embedding: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
    
    let entity = EntityBuilder::new("test", EntityDirection::Hidden)
        .with_embedding(large_embedding.clone())
        .build();
    
    assert_eq!(entity.embedding.len(), 1000);
    assert_eq!(entity.embedding, large_embedding);
}

// ==================== Performance Tests ====================

#[test]
fn test_entity_activation_performance() {
    let mut entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    
    let (_, duration) = measure_execution_time(|| {
        for _ in 0..1000 {
            entity.activate(test_constants::ACTION_POTENTIAL, test_constants::STANDARD_DECAY_RATE);
        }
    });
    
    // Should complete 1000 activations quickly
    assert!(duration.as_millis() < 10, "Activation should be fast: {:?}", duration);
}

#[test]
fn test_entity_memory_usage() {
    use std::mem;
    
    let entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    let size = mem::size_of_val(&entity);
    
    // Entity should be reasonably sized (not exact due to heap allocations)
    assert!(size < 1000, "Entity size {} bytes seems excessive", size);
}

#[test]
fn test_entity_clone_performance() {
    let entity = EntityBuilder::new("test", EntityDirection::Input)
        .with_embedding(vec![0.1; 100])
        .with_activation(test_constants::ACTION_POTENTIAL)
        .build();
    
    let (cloned, duration) = measure_execution_time(|| entity.clone());
    
    assert_eq!(cloned.concept_id, entity.concept_id);
    assert_eq!(cloned.embedding, entity.embedding);
    assert!(duration.as_micros() < 1000, "Clone should be fast: {:?}", duration);
}

// ==================== Multi-threading Safety Tests ====================

#[test]
fn test_entity_send_sync() {
    // Test that BrainInspiredEntity can be sent between threads
    let entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
    
    let handle = std::thread::spawn(move || {
        let mut local_entity = entity;
        local_entity.activate(test_constants::ACTION_POTENTIAL, test_constants::STANDARD_DECAY_RATE);
        local_entity.activation_state
    });
    
    let result = handle.join().unwrap();
    assert_activation_close(result, test_constants::ACTION_POTENTIAL,
                           test_constants::ACTIVATION_EPSILON, "Thread activation");
}

// ==================== Integration Tests ====================

#[test]
fn test_entity_with_different_directions() {
    let directions = [EntityDirection::Input, EntityDirection::Output, EntityDirection::Gate, EntityDirection::Hidden];
    
    for direction in directions {
        let mut entity = BrainInspiredEntity::new("test".to_string(), direction);
        
        // All directions should support activation
        entity.activate(test_constants::ACTION_POTENTIAL, test_constants::STANDARD_DECAY_RATE);
        assert_activation_close(entity.activation_state, test_constants::ACTION_POTENTIAL,
                               test_constants::ACTIVATION_EPSILON, "Direction activation");
        
        // All directions should support properties
        entity.properties.insert("test_prop".to_string(), AttributeValue::Float(0.5));
        assert_eq!(entity.properties.len(), 1);
    }
}

#[test]
fn test_entity_realistic_neural_behavior() {
    let mut entity = BrainInspiredEntity::new("cortical_neuron".to_string(), EntityDirection::Hidden);
    
    // Simulate realistic neural activation pattern
    let activations = [0.1, 0.3, 0.7, 0.9, 0.6, 0.3, 0.1]; // Spike pattern
    let mut results = Vec::new();
    
    for activation in activations {
        let result = entity.activate(activation, test_constants::STANDARD_DECAY_RATE);
        results.push(result);
        thread::sleep(Duration::from_millis(5)); // Simulate time between spikes
    }
    
    // Should show accumulation then decay pattern
    assert!(results[3] > results[0], "Should show accumulation");
    assert!(results[6] < results[3], "Should show decay");
    
    // All values should be in valid range
    for &result in &results {
        assert!(result >= 0.0 && result <= 1.0, "All activations should be valid");
    }
}