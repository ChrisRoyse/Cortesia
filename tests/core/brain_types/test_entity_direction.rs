// Tests for EntityDirection enum
// Validates brain-inspired entity direction classification system

use llmkg::core::brain_types::EntityDirection;
use serde_json;

use super::test_constants;

// ==================== Basic Enum Tests ====================

#[test]
fn test_entity_direction_variants() {
    // Test all enum variants exist and are distinct
    let input = EntityDirection::Input;
    let output = EntityDirection::Output;
    let gate = EntityDirection::Gate;
    let hidden = EntityDirection::Hidden;

    // Verify distinctness
    assert_ne!(input, output);
    assert_ne!(input, gate);
    assert_ne!(input, hidden);
    assert_ne!(output, gate);
    assert_ne!(output, hidden);
    assert_ne!(gate, hidden);
}

#[test]
fn test_entity_direction_copy_clone() {
    let original = EntityDirection::Input;
    let copied = original; // Test Copy trait
    let cloned = original.clone(); // Test Clone trait

    assert_eq!(original, copied);
    assert_eq!(original, cloned);
    assert_eq!(copied, cloned);
}

#[test]
fn test_entity_direction_debug() {
    // Test Debug trait implementation
    let input = EntityDirection::Input;
    let debug_str = format!("{:?}", input);
    assert!(debug_str.contains("Input"));
}

#[test]
fn test_entity_direction_partial_eq() {
    // Test PartialEq implementation
    assert_eq!(EntityDirection::Input, EntityDirection::Input);
    assert_eq!(EntityDirection::Output, EntityDirection::Output);
    assert_eq!(EntityDirection::Gate, EntityDirection::Gate);
    assert_eq!(EntityDirection::Hidden, EntityDirection::Hidden);

    // Test inequality
    assert_ne!(EntityDirection::Input, EntityDirection::Output);
    assert_ne!(EntityDirection::Gate, EntityDirection::Hidden);
}

// ==================== Serialization Tests ====================

#[test]
fn test_entity_direction_serialization() {
    // Test serde serialization
    let input = EntityDirection::Input;
    let serialized = serde_json::to_string(&input).expect("Should serialize");
    assert!(serialized.contains("Input"));

    let output = EntityDirection::Output;
    let serialized = serde_json::to_string(&output).expect("Should serialize");
    assert!(serialized.contains("Output"));

    let gate = EntityDirection::Gate;
    let serialized = serde_json::to_string(&gate).expect("Should serialize");
    assert!(serialized.contains("Gate"));

    let hidden = EntityDirection::Hidden;
    let serialized = serde_json::to_string(&hidden).expect("Should serialize");
    assert!(serialized.contains("Hidden"));
}

#[test]
fn test_entity_direction_deserialization() {
    // Test serde deserialization
    let input_json = "\"Input\"";
    let deserialized: EntityDirection = serde_json::from_str(input_json)
        .expect("Should deserialize Input");
    assert_eq!(deserialized, EntityDirection::Input);

    let output_json = "\"Output\"";
    let deserialized: EntityDirection = serde_json::from_str(output_json)
        .expect("Should deserialize Output");
    assert_eq!(deserialized, EntityDirection::Output);

    let gate_json = "\"Gate\"";
    let deserialized: EntityDirection = serde_json::from_str(gate_json)
        .expect("Should deserialize Gate");
    assert_eq!(deserialized, EntityDirection::Gate);

    let hidden_json = "\"Hidden\"";
    let deserialized: EntityDirection = serde_json::from_str(hidden_json)
        .expect("Should deserialize Hidden");
    assert_eq!(deserialized, EntityDirection::Hidden);
}

#[test]
fn test_entity_direction_round_trip() {
    // Test serialization round trip
    let directions = [
        EntityDirection::Input,
        EntityDirection::Output,
        EntityDirection::Gate,
        EntityDirection::Hidden,
    ];

    for direction in directions {
        let serialized = serde_json::to_string(&direction)
            .expect("Should serialize");
        let deserialized: EntityDirection = serde_json::from_str(&serialized)
            .expect("Should deserialize");
        assert_eq!(direction, deserialized);
    }
}

// ==================== Semantic Meaning Tests ====================

#[test]
fn test_entity_direction_semantic_meaning() {
    // Test that directions represent their intended neural functions
    
    // Input neurons receive external stimuli
    let input = EntityDirection::Input;
    assert_eq!(input, EntityDirection::Input);
    
    // Output neurons produce system responses
    let output = EntityDirection::Output;
    assert_eq!(output, EntityDirection::Output);
    
    // Gate neurons perform logic operations
    let gate = EntityDirection::Gate;
    assert_eq!(gate, EntityDirection::Gate);
    
    // Hidden neurons process internal computations
    let hidden = EntityDirection::Hidden;
    assert_eq!(hidden, EntityDirection::Hidden);
}

#[test]
fn test_entity_direction_neural_hierarchy() {
    // Test that directions represent valid neural network layers
    let directions = [
        EntityDirection::Input,   // Input layer
        EntityDirection::Hidden,  // Hidden layer(s)
        EntityDirection::Gate,    // Logic processing layer
        EntityDirection::Output,  // Output layer
    ];

    // All directions should be valid
    for direction in directions {
        match direction {
            EntityDirection::Input => {
                // Input neurons are entry points
                assert_eq!(direction, EntityDirection::Input);
            }
            EntityDirection::Hidden => {
                // Hidden neurons perform intermediate processing
                assert_eq!(direction, EntityDirection::Hidden);
            }
            EntityDirection::Gate => {
                // Gate neurons perform logical operations
                assert_eq!(direction, EntityDirection::Gate);
            }
            EntityDirection::Output => {
                // Output neurons produce final results
                assert_eq!(direction, EntityDirection::Output);
            }
        }
    }
}

// ==================== Collection and Iteration Tests ====================

#[test]
fn test_entity_direction_in_collections() {
    use std::collections::{HashMap, HashSet};

    // Test in HashMap (requires Eq + Hash)
    let mut direction_map = HashMap::new();
    direction_map.insert(EntityDirection::Input, "sensory");
    direction_map.insert(EntityDirection::Output, "motor");
    direction_map.insert(EntityDirection::Gate, "logic");
    direction_map.insert(EntityDirection::Hidden, "processing");

    assert_eq!(direction_map.get(&EntityDirection::Input), Some(&"sensory"));
    assert_eq!(direction_map.get(&EntityDirection::Output), Some(&"motor"));

    // Test in HashSet (requires Eq + Hash)
    let mut direction_set = HashSet::new();
    direction_set.insert(EntityDirection::Input);
    direction_set.insert(EntityDirection::Output);
    direction_set.insert(EntityDirection::Gate);
    direction_set.insert(EntityDirection::Hidden);

    assert_eq!(direction_set.len(), 4);
    assert!(direction_set.contains(&EntityDirection::Input));
    assert!(direction_set.contains(&EntityDirection::Hidden));
}

#[test]
fn test_entity_direction_iteration() {
    // Test iteration over directions
    let directions = vec![
        EntityDirection::Input,
        EntityDirection::Hidden,
        EntityDirection::Gate,
        EntityDirection::Output,
    ];

    let count = directions.iter().count();
    assert_eq!(count, 4);

    let input_count = directions.iter()
        .filter(|&&d| d == EntityDirection::Input)
        .count();
    assert_eq!(input_count, 1);
}

// ==================== Error Handling Tests ====================

#[test]
fn test_entity_direction_invalid_deserialization() {
    // Test invalid JSON deserialization
    let invalid_json = "\"InvalidDirection\"";
    let result: Result<EntityDirection, _> = serde_json::from_str(invalid_json);
    assert!(result.is_err());

    let invalid_number = "42";
    let result: Result<EntityDirection, _> = serde_json::from_str(invalid_number);
    assert!(result.is_err());

    let invalid_object = "{}";
    let result: Result<EntityDirection, _> = serde_json::from_str(invalid_object);
    assert!(result.is_err());
}

// ==================== Edge Case Tests ====================

#[test]
fn test_entity_direction_memory_efficiency() {
    // Test that enum variants don't take excessive memory
    use std::mem;
    
    let size = mem::size_of::<EntityDirection>();
    // Enum should be small (typically 1 byte for simple enums)
    assert!(size <= 8, "EntityDirection size {} bytes is too large", size);
}

#[test]
fn test_entity_direction_pattern_matching() {
    // Test comprehensive pattern matching
    let directions = [
        EntityDirection::Input,
        EntityDirection::Output,
        EntityDirection::Gate,
        EntityDirection::Hidden,
    ];

    for direction in directions {
        match direction {
            EntityDirection::Input => {
                assert_eq!(direction, EntityDirection::Input);
            }
            EntityDirection::Output => {
                assert_eq!(direction, EntityDirection::Output);
            }
            EntityDirection::Gate => {
                assert_eq!(direction, EntityDirection::Gate);
            }
            EntityDirection::Hidden => {
                assert_eq!(direction, EntityDirection::Hidden);
            }
        }
    }
}