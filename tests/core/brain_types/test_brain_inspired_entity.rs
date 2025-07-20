// Tests for BrainInspiredEntity struct
// Validates neural entity behavior including activation, decay, and temporal properties

use llmkg::core::brain_types::{BrainInspiredEntity, EntityDirection};
use llmkg::core::types::{EntityKey, AttributeValue};
use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use std::thread;
use serde_json;

use super::test_constants;
use super::test_helpers::{
    EntityBuilder, assert_activation_close, assert_entity_direction, measure_execution_time,
    generate_edge_case_activations, generate_decay_rates, assert_float_eq
};

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

// ==================== Enhanced Temporal Decay Tests ====================

#[test]
fn test_entity_temporal_decay_mathematical_precision() {
    let mut entity = BrainInspiredEntity::new("precision_test".to_string(), EntityDirection::Input);
    
    // Test precise exponential decay formula: new_state = old_state * exp(-decay_rate * time)
    let initial_activation = 0.8;
    let decay_rate = 0.1;
    
    entity.activate(initial_activation, decay_rate);
    let activated_time = entity.last_activation;
    
    // Manually set time to known value for testing
    entity.last_activation = activated_time - Duration::from_secs(2);
    
    // Calculate expected decay
    let time_elapsed = 2.0; // seconds
    let expected_decayed = initial_activation * (-decay_rate * time_elapsed).exp();
    
    // Apply decay with zero new activation
    let result = entity.activate(0.0, decay_rate);
    
    assert_float_eq(result, expected_decayed, 0.01);
    assert_float_eq(entity.activation_state, expected_decayed, 0.01);
}

#[test]
fn test_entity_decay_rate_variations_systematic() {
    let decay_rates = generate_decay_rates();
    let initial_activation = test_constants::ACTION_POTENTIAL;
    let wait_time = Duration::from_millis(500);
    
    for &decay_rate in &decay_rates {
        let mut entity = BrainInspiredEntity::new("decay_test".to_string(), EntityDirection::Input);
        
        // Initial activation
        entity.activate(initial_activation, decay_rate);
        let initial_state = entity.activation_state;
        
        // Wait and apply decay
        entity.last_activation = SystemTime::now() - wait_time;
        let result = entity.activate(0.0, decay_rate);
        
        if decay_rate == 0.0 {
            // No decay should preserve activation
            assert_float_eq(result, initial_activation, test_constants::ACTIVATION_EPSILON);
        } else {
            // Higher decay rates should result in lower activation
            assert!(result <= initial_state, "Decay should reduce activation");
            assert!(result >= 0.0, "Activation should not go negative");
            
            // Very high decay rates should result in near-zero activation
            if decay_rate >= 5.0 {
                assert!(result < 0.1, "High decay rate should significantly reduce activation");
            }
        }
    }
}

#[test]
fn test_entity_temporal_decay_boundary_conditions() {
    let mut entity = BrainInspiredEntity::new("boundary_test".to_string(), EntityDirection::Input);
    
    // Test decay with zero initial activation
    entity.activation_state = 0.0;
    entity.last_activation = SystemTime::now() - Duration::from_secs(5);
    let result = entity.activate(0.0, test_constants::STANDARD_DECAY_RATE);
    assert_eq!(result, 0.0, "Zero activation should remain zero after decay");
    
    // Test decay with maximum activation
    entity.activation_state = test_constants::SATURATION_LEVEL;
    entity.last_activation = SystemTime::now() - Duration::from_secs(1);
    let result = entity.activate(0.0, test_constants::SLOW_DECAY_RATE);
    assert!(result < test_constants::SATURATION_LEVEL, "Maximum activation should decay");
    assert!(result > 0.5, "Slow decay should not reduce activation too much");
    
    // Test decay with very old timestamp
    entity.activation_state = test_constants::ACTION_POTENTIAL;
    entity.last_activation = SystemTime::UNIX_EPOCH; // Very old timestamp
    let result = entity.activate(0.0, test_constants::STANDARD_DECAY_RATE);
    assert!(result < 0.001, "Very old activation should decay to near zero");
}

#[test]
fn test_entity_activation_accumulation_with_decay() {
    let mut entity = BrainInspiredEntity::new("accumulation_test".to_string(), EntityDirection::Input);
    
    // Test accumulation pattern with intermediate decay
    let activations = vec![0.3, 0.2, 0.4, 0.1, 0.5];
    let wait_times = vec![10, 50, 20, 100, 30]; // milliseconds
    let decay_rate = test_constants::STANDARD_DECAY_RATE;
    
    let mut previous_state = 0.0;
    
    for (activation, wait_ms) in activations.iter().zip(wait_times.iter()) {
        // Wait before next activation
        thread::sleep(Duration::from_millis(*wait_ms));
        
        let result = entity.activate(*activation, decay_rate);
        
        // Result should be influenced by both decay and new activation
        assert!(result >= *activation, "Should include new activation");
        assert!(result <= previous_state + activation, "Should account for decay");
        assert!(result <= test_constants::SATURATION_LEVEL, "Should not exceed saturation");
        
        previous_state = result;
    }
}

#[test]
fn test_entity_rapid_activation_sequence() {
    let mut entity = BrainInspiredEntity::new("rapid_test".to_string(), EntityDirection::Input);
    
    // Rapid successive activations with minimal time between them
    let rapid_activations = vec![0.1, 0.1, 0.1, 0.1, 0.1]; // 5 x 0.1 = 0.5 total
    let decay_rate = test_constants::SLOW_DECAY_RATE; // Minimal decay
    
    for &activation in &rapid_activations {
        let result = entity.activate(activation, decay_rate);
        assert!(result >= activation, "Each activation should contribute");
        assert!(result <= test_constants::SATURATION_LEVEL, "Should not exceed saturation");
    }
    
    // Final state should be close to sum of activations (minimal decay due to rapid succession)
    let expected_total = rapid_activations.iter().sum::<f32>().min(test_constants::SATURATION_LEVEL);
    assert_float_eq(entity.activation_state, expected_total, 0.1);
}

#[test]
fn test_entity_decay_time_calculation_precision() {
    let mut entity = BrainInspiredEntity::new("timing_test".to_string(), EntityDirection::Input);
    
    // Test that time calculation is precise for different durations
    let test_durations = vec![
        Duration::from_millis(1),
        Duration::from_millis(10),
        Duration::from_millis(100),
        Duration::from_secs(1),
        Duration::from_secs(10),
    ];
    
    for duration in test_durations {
        entity.activation_state = test_constants::ACTION_POTENTIAL;
        entity.last_activation = SystemTime::now() - duration;
        
        let time_secs = duration.as_secs_f32();
        let decay_rate = 0.1;
        let expected = test_constants::ACTION_POTENTIAL * (-decay_rate * time_secs).exp();
        
        let result = entity.activate(0.0, decay_rate);
        
        // Allow some tolerance for timing precision
        assert_float_eq(result, expected, 0.01);
    }
}

// ==================== Enhanced Boundary Condition Tests ====================

#[test]
fn test_entity_activation_extreme_values() {
    let mut entity = BrainInspiredEntity::new("extreme_test".to_string(), EntityDirection::Input);
    let edge_cases = generate_edge_case_activations();
    
    for &activation in &edge_cases {
        let result = entity.activate(activation, test_constants::STANDARD_DECAY_RATE);
        
        if activation.is_nan() {
            // NaN input may produce NaN or be handled gracefully
            assert!(result.is_nan() || (result >= 0.0 && result <= 1.0));
        } else if activation.is_infinite() {
            // Infinite activation should be clamped to 1.0
            assert!(result == test_constants::SATURATION_LEVEL || result.is_infinite());
        } else if activation < 0.0 {
            // Negative activations should not decrease the current state below 0
            assert!(result >= 0.0, "Activation should not go negative");
        } else if activation > test_constants::SATURATION_LEVEL {
            // Large activations should be clamped
            assert_eq!(result, test_constants::SATURATION_LEVEL, "Should clamp at saturation");
        } else {
            // Normal values should work correctly
            assert!(result >= 0.0 && result <= test_constants::SATURATION_LEVEL);
        }
        
        // Reset for next test
        entity.activation_state = test_constants::RESTING_POTENTIAL;
    }
}

#[test]
fn test_entity_decay_rate_extreme_values() {
    let mut entity = BrainInspiredEntity::new("decay_extreme_test".to_string(), EntityDirection::Input);
    
    let extreme_decay_rates = vec![
        f32::NEG_INFINITY,
        -1000.0,
        -1.0,
        f32::INFINITY,
        f32::NAN,
        1000.0,
    ];
    
    for &decay_rate in &extreme_decay_rates {
        entity.activation_state = test_constants::ACTION_POTENTIAL;
        entity.last_activation = SystemTime::now() - Duration::from_secs(1);
        
        let result = entity.activate(0.1, decay_rate);
        
        if decay_rate.is_nan() {
            // NaN decay rate may produce NaN or be handled gracefully
            assert!(result.is_nan() || (result >= 0.0 && result <= 1.0));
        } else if decay_rate < 0.0 {
            // Negative decay rate might cause growth instead of decay
            // Implementation choice - may clamp or handle differently
            assert!(result >= 0.0, "Should not go negative");
        } else if decay_rate.is_infinite() || decay_rate > 100.0 {
            // Very high decay should approach zero very quickly
            assert!(result <= 0.2, "High decay should dramatically reduce activation");
        }
        
        // Reset for next test
        entity.activation_state = test_constants::RESTING_POTENTIAL;
    }
}

#[test]
fn test_entity_timestamp_edge_cases() {
    let mut entity = BrainInspiredEntity::new("timestamp_test".to_string(), EntityDirection::Input);
    
    // Test with future timestamp (clock skew)
    entity.last_activation = SystemTime::now() + Duration::from_secs(1);
    let result = entity.activate(test_constants::ACTION_POTENTIAL, test_constants::STANDARD_DECAY_RATE);
    
    // Should handle gracefully (elapsed() might return default)
    assert!(result >= 0.0 && result <= test_constants::SATURATION_LEVEL);
    
    // Test with UNIX_EPOCH (very old)
    entity.last_activation = SystemTime::UNIX_EPOCH;
    let result = entity.activate(test_constants::WEAK_EXCITATORY, test_constants::STANDARD_DECAY_RATE);
    
    // Should decay to essentially zero and then add new activation
    assert_float_eq(result, test_constants::WEAK_EXCITATORY, 0.01);
}

#[test]
fn test_entity_activation_clamping_precision() {
    let mut entity = BrainInspiredEntity::new("clamping_test".to_string(), EntityDirection::Input);
    
    // Test values very close to 1.0
    let near_saturation_tests = vec![
        0.999, 0.9999, 0.99999, 1.0, 1.00001, 1.0001, 1.001,
    ];
    
    for &activation in &near_saturation_tests {
        entity.activation_state = 0.0;
        let result = entity.activate(activation, test_constants::NO_DECAY_RATE);
        
        if activation <= test_constants::SATURATION_LEVEL {
            assert_float_eq(result, activation, test_constants::ACTIVATION_EPSILON);
        } else {
            assert_eq!(result, test_constants::SATURATION_LEVEL);
        }
    }
}

#[test]
fn test_entity_concurrent_activation_safety() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let entity = Arc::new(Mutex::new(
        BrainInspiredEntity::new("concurrent_test".to_string(), EntityDirection::Input)
    ));
    
    let mut handles = vec![];
    
    // Spawn multiple threads that activate the entity
    for i in 0..5 {
        let entity_clone = Arc::clone(&entity);
        let handle = thread::spawn(move || {
            let activation = 0.1 + (i as f32) * 0.1;
            let mut entity_lock = entity_clone.lock().unwrap();
            entity_lock.activate(activation, test_constants::STANDARD_DECAY_RATE)
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    let mut results = vec![];
    for handle in handles {
        results.push(handle.join().unwrap());
    }
    
    // All results should be valid
    for result in results {
        assert!(result >= 0.0 && result <= test_constants::SATURATION_LEVEL);
    }
    
    // Final state should be valid
    let final_entity = entity.lock().unwrap();
    assert!(final_entity.activation_state >= 0.0 && final_entity.activation_state <= test_constants::SATURATION_LEVEL);
}

// ==================== Property-Based Testing for Entities ====================

#[test]
fn test_entity_activation_monotonicity_property() {
    let mut entity = BrainInspiredEntity::new("monotonicity_test".to_string(), EntityDirection::Input);
    
    // Property: Activation with positive input should not decrease current state (without decay)
    let test_activations = vec![0.1, 0.2, 0.05, 0.3, 0.15];
    
    for &activation in &test_activations {
        let initial_state = entity.activation_state;
        let result = entity.activate(activation, test_constants::NO_DECAY_RATE);
        
        assert!(result >= initial_state, 
               "Activation should not decrease without decay: {} -> {}", 
               initial_state, result);
    }
}

#[test]
fn test_entity_decay_monotonicity_property() {
    let mut entity = BrainInspiredEntity::new("decay_monotonicity_test".to_string(), EntityDirection::Input);
    
    // Property: Pure decay (zero new activation) should never increase activation
    entity.activate(test_constants::ACTION_POTENTIAL, test_constants::STANDARD_DECAY_RATE);
    let initial_state = entity.activation_state;
    
    // Wait and apply pure decay multiple times
    for _ in 0..5 {
        thread::sleep(Duration::from_millis(50));
        let previous_state = entity.activation_state;
        let result = entity.activate(0.0, test_constants::STANDARD_DECAY_RATE);
        
        assert!(result <= previous_state, 
               "Pure decay should not increase activation: {} -> {}", 
               previous_state, result);
    }
    
    assert!(entity.activation_state < initial_state, 
           "Overall decay should reduce activation");
}

#[test]
fn test_entity_idempotence_property() {
    let mut entity = BrainInspiredEntity::new("idempotence_test".to_string(), EntityDirection::Input);
    
    // Property: Activating with 0.0 immediately (no time elapsed) should not change state
    entity.activate(test_constants::ACTION_POTENTIAL, test_constants::STANDARD_DECAY_RATE);
    let state_after_activation = entity.activation_state;
    
    // Immediately activate with 0.0 (no time should have elapsed)
    let result = entity.activate(0.0, test_constants::STANDARD_DECAY_RATE);
    
    // Should be essentially the same (allowing for tiny timing differences)
    assert_float_eq(result, state_after_activation, 0.001);
}

#[test]
fn test_entity_saturation_property() {
    let mut entity = BrainInspiredEntity::new("saturation_test".to_string(), EntityDirection::Input);
    
    // Property: No matter how much activation is applied, result should never exceed 1.0
    let large_activations = vec![2.0, 10.0, 100.0, 1000.0, f32::INFINITY];
    
    for &activation in &large_activations {
        entity.activation_state = 0.0; // Reset
        let result = entity.activate(activation, test_constants::NO_DECAY_RATE);
        
        if activation.is_finite() {
            assert_eq!(result, test_constants::SATURATION_LEVEL,
                      "Large finite activation should clamp to saturation");
        } else {
            // Infinite activation might be handled specially
            assert!(result == test_constants::SATURATION_LEVEL || result.is_infinite(),
                   "Infinite activation should be handled appropriately");
        }
    }
}