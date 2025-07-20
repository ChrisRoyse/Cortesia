// Tests for ActivationStep struct
// Validates activation step tracking for neural reasoning traces

use llmkg::core::brain_types::{ActivationStep, ActivationOperation};
use llmkg::core::types::EntityKey;
use std::time::SystemTime;
use serde_json;

use super::test_constants;
use super::test_helpers::{create_activation_step, measure_execution_time};

// ==================== Constructor Tests ====================

#[test]
fn test_activation_step_creation() {
    let entity_key = EntityKey::from(slotmap::KeyData::from_ffi(42));
    let concept_id = test_constants::TEST_CONCEPT_INPUT;
    let activation_level = test_constants::ACTION_POTENTIAL;
    let operation = ActivationOperation::Initialize;
    
    let step = create_activation_step(
        1,
        entity_key,
        concept_id,
        activation_level,
        operation
    );
    
    assert_eq!(step.step_id, 1);
    assert_eq!(step.entity_key, entity_key);
    assert_eq!(step.concept_id, concept_id);
    assert_eq!(step.activation_level, activation_level);
    assert!(matches!(step.operation_type, ActivationOperation::Initialize));
    
    // Timestamp should be recent
    let now = SystemTime::now();
    let duration = now.duration_since(step.timestamp).unwrap();
    assert!(duration.as_secs() < 1, "Timestamp should be recent");
}

#[test]
fn test_activation_step_all_operations() {
    let operations = [
        ActivationOperation::Initialize,
        ActivationOperation::Propagate,
        ActivationOperation::Inhibit,
        ActivationOperation::Reinforce,
        ActivationOperation::Decay,
    ];
    
    for (index, operation) in operations.iter().enumerate() {
        let step = create_activation_step(
            index,
            EntityKey::from(slotmap::KeyData::from_ffi(index as u64)),
            test_constants::TEST_CONCEPT_GATE,
            test_constants::THRESHOLD_POTENTIAL,
            *operation
        );
        
        assert_eq!(step.step_id, index);
        assert!(matches!(step.operation_type, operation));
    }
}

// ==================== Serialization Tests ====================

#[test]
fn test_activation_step_serialization() {
    let step = create_activation_step(
        5,
        EntityKey::from(slotmap::KeyData::from_ffi(100),
        test_constants::TEST_CONCEPT_OUTPUT,
        test_constants::STRONG_EXCITATORY,
        ActivationOperation::Propagate
    );
    
    let serialized = serde_json::to_string(&step).expect("Should serialize");
    
    // Verify key fields are present
    assert!(serialized.contains("step_id"));
    assert!(serialized.contains("entity_key"));
    assert!(serialized.contains("concept_id"));
    assert!(serialized.contains("activation_level"));
    assert!(serialized.contains("operation_type"));
    assert!(serialized.contains("timestamp"));
}

#[test]
fn test_activation_step_deserialization() {
    let original = create_activation_step(
        10,
        EntityKey::from(slotmap::KeyData::from_ffi(200),
        test_constants::TEST_CONCEPT_HIDDEN,
        test_constants::MEDIUM_EXCITATORY,
        ActivationOperation::Reinforce
    );
    
    let serialized = serde_json::to_string(&original).expect("Should serialize");
    let deserialized: ActivationStep = serde_json::from_str(&serialized)
        .expect("Should deserialize");
    
    assert_eq!(deserialized.step_id, original.step_id);
    assert_eq!(deserialized.entity_key, original.entity_key);
    assert_eq!(deserialized.concept_id, original.concept_id);
    assert_eq!(deserialized.activation_level, original.activation_level);
    
    // Operation types should match
    match (&original.operation_type, &deserialized.operation_type) {
        (ActivationOperation::Reinforce, ActivationOperation::Reinforce) => {},
        _ => panic!("Operation type mismatch"),
    }
}

// ==================== Activation Level Tests ====================

#[test]
fn test_activation_step_valid_levels() {
    let valid_levels = [
        test_constants::RESTING_POTENTIAL,
        test_constants::THRESHOLD_POTENTIAL,
        test_constants::ACTION_POTENTIAL,
        test_constants::SATURATION_LEVEL,
    ];
    
    for (index, &level) in valid_levels.iter().enumerate() {
        let step = create_activation_step(
            index,
            EntityKey::from(slotmap::KeyData::from_ffi(index as u64)),
            test_constants::TEST_CONCEPT_INPUT,
            level,
            ActivationOperation::Initialize
        );
        
        assert_eq!(step.activation_level, level);
        assert!(step.activation_level >= 0.0);
        assert!(step.activation_level <= 1.0);
    }
}

#[test]
fn test_activation_step_extreme_levels() {
    // Test with levels outside normal range (should be allowed for flexibility)
    let extreme_levels = [
        -0.5, // Negative
        test_constants::ABOVE_SATURATION, // Above 1.0
        f32::MIN,
        f32::MAX,
    ];
    
    for (index, &level) in extreme_levels.iter().enumerate() {
        let step = create_activation_step(
            index,
            EntityKey::from(slotmap::KeyData::from_ffi(index as u64)),
            test_constants::TEST_CONCEPT_OUTPUT,
            level,
            ActivationOperation::Decay
        );
        
        assert_eq!(step.activation_level, level);
    }
}

// ==================== Operation Type Tests ====================

#[test]
fn test_operation_type_initialize() {
    let step = create_activation_step(
        1,
        EntityKey::from(slotmap::KeyData::from_ffi(1),
        test_constants::TEST_CONCEPT_INPUT,
        test_constants::RESTING_POTENTIAL,
        ActivationOperation::Initialize
    );
    
    match step.operation_type {
        ActivationOperation::Initialize => {
            // Should typically have low activation when initializing
            assert!(step.activation_level <= test_constants::THRESHOLD_POTENTIAL);
        },
        _ => panic!("Expected Initialize operation"),
    }
}

#[test]
fn test_operation_type_propagate() {
    let step = create_activation_step(
        2,
        EntityKey::from(slotmap::KeyData::from_ffi(2),
        test_constants::TEST_CONCEPT_GATE,
        test_constants::ACTION_POTENTIAL,
        ActivationOperation::Propagate
    );
    
    match step.operation_type {
        ActivationOperation::Propagate => {
            // Propagation usually involves significant activation
            assert!(step.activation_level >= test_constants::THRESHOLD_POTENTIAL);
        },
        _ => panic!("Expected Propagate operation"),
    }
}

#[test]
fn test_operation_type_inhibit() {
    let step = create_activation_step(
        3,
        EntityKey::from(slotmap::KeyData::from_ffi(3),
        test_constants::TEST_CONCEPT_OUTPUT,
        test_constants::WEAK_EXCITATORY,
        ActivationOperation::Inhibit
    );
    
    match step.operation_type {
        ActivationOperation::Inhibit => {
            // Inhibition typically reduces activation
            assert!(step.activation_level <= test_constants::MEDIUM_EXCITATORY);
        },
        _ => panic!("Expected Inhibit operation"),
    }
}

#[test]
fn test_operation_type_reinforce() {
    let step = create_activation_step(
        4,
        EntityKey::from(slotmap::KeyData::from_ffi(4),
        test_constants::TEST_CONCEPT_HIDDEN,
        test_constants::STRONG_EXCITATORY,
        ActivationOperation::Reinforce
    );
    
    match step.operation_type {
        ActivationOperation::Reinforce => {
            // Reinforcement should increase activation
            assert!(step.activation_level >= test_constants::MEDIUM_EXCITATORY);
        },
        _ => panic!("Expected Reinforce operation"),
    }
}

#[test]
fn test_operation_type_decay() {
    let step = create_activation_step(
        5,
        EntityKey::from(slotmap::KeyData::from_ffi(5),
        test_constants::TEST_CONCEPT_INPUT,
        test_constants::WEAK_EXCITATORY,
        ActivationOperation::Decay
    );
    
    match step.operation_type {
        ActivationOperation::Decay => {
            // Decay should result in lower activation
            assert!(step.activation_level <= test_constants::THRESHOLD_POTENTIAL);
        },
        _ => panic!("Expected Decay operation"),
    }
}

// ==================== Concept ID Tests ====================

#[test]
fn test_activation_step_concept_ids() {
    let concept_ids = [
        test_constants::TEST_CONCEPT_INPUT,
        test_constants::TEST_CONCEPT_OUTPUT,
        test_constants::TEST_CONCEPT_GATE,
        test_constants::TEST_CONCEPT_HIDDEN,
        "custom_concept_123",
        "neural_pathway_abc",
    ];
    
    for (index, &concept_id) in concept_ids.iter().enumerate() {
        let step = create_activation_step(
            index,
            EntityKey::from(slotmap::KeyData::from_ffi(index as u64)),
            concept_id,
            test_constants::MEDIUM_EXCITATORY,
            ActivationOperation::Propagate
        );
        
        assert_eq!(step.concept_id, concept_id);
        assert!(!step.concept_id.is_empty());
    }
}

#[test]
fn test_activation_step_empty_concept_id() {
    // Test with empty concept ID (should be allowed but not recommended)
    let step = create_activation_step(
        1,
        EntityKey::from(slotmap::KeyData::from_ffi(1),
        test_constants::EMPTY_CONCEPT_ID,
        test_constants::RESTING_POTENTIAL,
        ActivationOperation::Initialize
    );
    
    assert_eq!(step.concept_id, test_constants::EMPTY_CONCEPT_ID);
    assert!(step.concept_id.is_empty());
}

// ==================== Entity Key Tests ====================

#[test]
fn test_activation_step_entity_keys() {
    let entity_keys = [
        EntityKey::from(slotmap::KeyData::from_ffi(0)),
        EntityKey::from(slotmap::KeyData::from_ffi(1)),
        EntityKey::from(slotmap::KeyData::from_ffi(42)),
        EntityKey::from(slotmap::KeyData::from_ffi(12345)),
        EntityKey::from(slotmap::KeyData::from_ffi(u64::MAX)),
    ];
    
    for (index, &key) in entity_keys.iter().enumerate() {
        let step = create_activation_step(
            index,
            key,
            test_constants::TEST_CONCEPT_GATE,
            test_constants::THRESHOLD_POTENTIAL,
            ActivationOperation::Initialize
        );
        
        assert_eq!(step.entity_key, key);
    }
}

// ==================== Step ID Tests ====================

#[test]
fn test_activation_step_sequence() {
    let step_count = 10;
    let mut steps = Vec::new();
    
    // Create a sequence of activation steps
    for i in 0..step_count {
        let step = create_activation_step(
            i,
            EntityKey::from(slotmap::KeyData::from_ffi(i as u64),
            test_constants::TEST_CONCEPT_INPUT,
            (i as f32) / (step_count as f32),
            ActivationOperation::Propagate
        );
        steps.push(step);
    }
    
    // Verify sequence integrity
    for (expected_id, step) in steps.iter().enumerate() {
        assert_eq!(step.step_id, expected_id);
        
        // Later steps should have higher activation (artificial progression)
        if expected_id > 0 {
            assert!(step.activation_level >= steps[expected_id - 1].activation_level);
        }
    }
}

#[test]
fn test_activation_step_large_sequence() {
    let large_step_count = 1000;
    
    // Test with large step IDs
    let step = create_activation_step(
        large_step_count,
        EntityKey::from(slotmap::KeyData::from_ffi(large_step_count as u64)),
        test_constants::TEST_CONCEPT_OUTPUT,
        test_constants::ACTION_POTENTIAL,
        ActivationOperation::Reinforce
    );
    
    assert_eq!(step.step_id, large_step_count);
}

// ==================== Timestamp Tests ====================

#[test]
fn test_activation_step_timestamp_ordering() {
    let mut steps = Vec::new();
    
    // Create steps with small delays to ensure different timestamps
    for i in 0..5 {
        std::thread::sleep(std::time::Duration::from_millis(1));
        
        let step = create_activation_step(
            i,
            EntityKey::from(slotmap::KeyData::from_ffi(i as u64),
            test_constants::TEST_CONCEPT_GATE,
            test_constants::MEDIUM_EXCITATORY,
            ActivationOperation::Propagate
        );
        steps.push(step);
    }
    
    // Verify timestamps are in order (later steps have later timestamps)
    for i in 1..steps.len() {
        assert!(
            steps[i].timestamp >= steps[i-1].timestamp,
            "Timestamps should be in order"
        );
    }
}

// ==================== Performance Tests ====================

#[test]
fn test_activation_step_creation_performance() {
    let step_count = 10000;
    
    let (_, duration) = measure_execution_time(|| {
        for i in 0..step_count {
            let _ = create_activation_step(
                i,
                EntityKey::from(slotmap::KeyData::from_ffi(i as u64),
                test_constants::TEST_CONCEPT_INPUT,
                test_constants::THRESHOLD_POTENTIAL,
                ActivationOperation::Initialize
            );
        }
    });
    
    // Should be able to create many steps quickly
    assert!(
        duration.as_millis() < 100,
        "Creating {} steps took too long: {:?}",
        step_count, duration
    );
}

#[test]
fn test_activation_step_serialization_performance() {
    let step = create_activation_step(
        42,
        EntityKey::from(slotmap::KeyData::from_ffi(123),
        test_constants::TEST_CONCEPT_HIDDEN,
        test_constants::STRONG_EXCITATORY,
        ActivationOperation::Propagate
    );
    
    let (_, duration) = measure_execution_time(|| {
        for _ in 0..1000 {
            let _ = serde_json::to_string(&step);
        }
    });
    
    // Serialization should be fast
    assert!(
        duration.as_millis() < 50,
        "Serialization performance too slow: {:?}",
        duration
    );
}

// ==================== Debug and Display Tests ====================

#[test]
fn test_activation_step_debug_format() {
    let step = create_activation_step(
        7,
        EntityKey::from(slotmap::KeyData::from_ffi(99),
        test_constants::TEST_CONCEPT_OUTPUT,
        test_constants::ACTION_POTENTIAL,
        ActivationOperation::Inhibit
    );
    
    let debug_str = format!("{:?}", step);
    
    // Should contain key information
    assert!(debug_str.contains("ActivationStep"));
    assert!(debug_str.contains("step_id"));
    assert!(debug_str.contains("entity_key"));
    assert!(debug_str.contains("activation_level"));
    assert!(debug_str.contains("operation_type"));
}

// ==================== Clone Tests ====================

#[test]
fn test_activation_step_clone() {
    let original = create_activation_step(
        15,
        EntityKey::from(slotmap::KeyData::from_ffi(500),
        test_constants::TEST_CONCEPT_GATE,
        test_constants::MEDIUM_EXCITATORY,
        ActivationOperation::Decay
    );
    
    let cloned = original.clone();
    
    assert_eq!(cloned.step_id, original.step_id);
    assert_eq!(cloned.entity_key, original.entity_key);
    assert_eq!(cloned.concept_id, original.concept_id);
    assert_eq!(cloned.activation_level, original.activation_level);
    assert_eq!(cloned.timestamp, original.timestamp);
    
    // Operation types should match
    match (&original.operation_type, &cloned.operation_type) {
        (ActivationOperation::Decay, ActivationOperation::Decay) => {},
        _ => panic!("Operation type mismatch in clone"),
    }
}

// ==================== Memory Tests ====================

#[test]
fn test_activation_step_memory_usage() {
    use std::mem;
    
    let step = create_activation_step(
        1,
        EntityKey::from(slotmap::KeyData::from_ffi(1),
        test_constants::TEST_CONCEPT_INPUT,
        test_constants::RESTING_POTENTIAL,
        ActivationOperation::Initialize
    );
    
    let size = mem::size_of_val(&step);
    
    // Should be reasonably sized (exact size may vary with platform)
    assert!(size < 200, "ActivationStep should be compact, got {} bytes", size);
}