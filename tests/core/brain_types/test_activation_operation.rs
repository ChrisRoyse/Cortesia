/// Comprehensive tests for ActivationOperation enum
/// 
/// This module tests neural activation operation types including:
/// - Operation type semantics and usage patterns
/// - Serialization and deserialization of operations
/// - Operation sequencing and state transitions
/// - Debug and display formatting
/// - Performance characteristics

use llmkg::core::brain_types::ActivationOperation;
use serde_json;
use super::test_constants;

// ==================== Basic Operation Type Tests ====================

#[test]
fn test_activation_operation_variants() {
    let operations = vec![
        ActivationOperation::Initialize,
        ActivationOperation::Propagate,
        ActivationOperation::Inhibit,
        ActivationOperation::Reinforce,
        ActivationOperation::Decay,
    ];
    
    // Each variant should be distinct
    for (i, op1) in operations.iter().enumerate() {
        for (j, op2) in operations.iter().enumerate() {
            if i == j {
                assert_eq!(op1, op2, "Same operations should be equal");
            } else {
                assert_ne!(op1, op2, "Different operations should not be equal");
            }
        }
    }
}

#[test]
fn test_activation_operation_copy_clone() {
    let original = ActivationOperation::Propagate;
    
    // Test Copy trait
    let copied = original;
    assert_eq!(original, copied);
    
    // Test Clone trait
    let cloned = original.clone();
    assert_eq!(original, cloned);
    
    // All should be equal
    assert_eq!(copied, cloned);
}

#[test]
fn test_activation_operation_debug() {
    let operations = vec![
        (ActivationOperation::Initialize, "Initialize"),
        (ActivationOperation::Propagate, "Propagate"),
        (ActivationOperation::Inhibit, "Inhibit"),
        (ActivationOperation::Reinforce, "Reinforce"),
        (ActivationOperation::Decay, "Decay"),
    ];
    
    for (operation, expected_name) in operations {
        let debug_str = format!("{:?}", operation);
        assert!(
            debug_str.contains(expected_name),
            "Debug string '{}' should contain '{}'",
            debug_str, expected_name
        );
    }
}

// ==================== Serialization Tests ====================

#[test]
fn test_activation_operation_serialization() {
    let operations = vec![
        ActivationOperation::Initialize,
        ActivationOperation::Propagate,
        ActivationOperation::Inhibit,
        ActivationOperation::Reinforce,
        ActivationOperation::Decay,
    ];
    
    for operation in operations {
        let serialized = serde_json::to_string(&operation)
            .expect("Should serialize successfully");
        
        // Should be valid JSON
        assert!(serialized.len() > 2, "Serialized form should not be empty");
        
        // Should contain the operation name
        let operation_name = format!("{:?}", operation);
        assert!(
            serialized.contains(&operation_name),
            "Serialized form '{}' should contain operation name '{}'",
            serialized, operation_name
        );
    }
}

#[test]
fn test_activation_operation_deserialization() {
    let test_cases = vec![
        (ActivationOperation::Initialize, "Initialize"),
        (ActivationOperation::Propagate, "Propagate"),
        (ActivationOperation::Inhibit, "Inhibit"),
        (ActivationOperation::Reinforce, "Reinforce"),
        (ActivationOperation::Decay, "Decay"),
    ];
    
    for (original_op, expected_name) in test_cases {
        // Test round-trip serialization
        let serialized = serde_json::to_string(&original_op).unwrap();
        let deserialized: ActivationOperation = serde_json::from_str(&serialized)
            .expect("Should deserialize successfully");
        
        assert_eq!(original_op, deserialized);
        
        // Test direct JSON deserialization
        let json_str = format!("\"{}\"", expected_name);
        let from_json: ActivationOperation = serde_json::from_str(&json_str)
            .expect("Should deserialize from JSON string");
        
        assert_eq!(original_op, from_json);
    }
}

#[test]
fn test_activation_operation_json_format() {
    let operation = ActivationOperation::Propagate;
    let serialized = serde_json::to_string(&operation).unwrap();
    
    // Should be a simple JSON string
    assert!(serialized.starts_with('"'));
    assert!(serialized.ends_with('"'));
    assert!(serialized.contains("Propagate"));
    
    // Should be minimal format
    assert_eq!(serialized, "\"Propagate\"");
}

// ==================== Operation Semantics Tests ====================

#[test]
fn test_operation_semantic_meaning() {
    // Test that operations have appropriate semantic associations
    
    // Initialize should be used for setting up initial state
    let init_op = ActivationOperation::Initialize;
    assert_eq!(format!("{:?}", init_op), "Initialize");
    
    // Propagate should be used for spreading activation
    let prop_op = ActivationOperation::Propagate;
    assert_eq!(format!("{:?}", prop_op), "Propagate");
    
    // Inhibit should be used for suppressing activation
    let inhib_op = ActivationOperation::Inhibit;
    assert_eq!(format!("{:?}", inhib_op), "Inhibit");
    
    // Reinforce should be used for strengthening activation
    let reinf_op = ActivationOperation::Reinforce;
    assert_eq!(format!("{:?}", reinf_op), "Reinforce");
    
    // Decay should be used for reducing activation over time
    let decay_op = ActivationOperation::Decay;
    assert_eq!(format!("{:?}", decay_op), "Decay");
}

#[test]
fn test_operation_usage_patterns() {
    // Test typical usage patterns in activation sequences
    
    // Common initialization sequence
    let init_sequence = vec![
        ActivationOperation::Initialize,
        ActivationOperation::Propagate,
        ActivationOperation::Decay,
    ];
    
    assert_eq!(init_sequence.len(), 3);
    assert_eq!(init_sequence[0], ActivationOperation::Initialize);
    assert_eq!(init_sequence[1], ActivationOperation::Propagate);
    assert_eq!(init_sequence[2], ActivationOperation::Decay);
    
    // Learning sequence
    let learning_sequence = vec![
        ActivationOperation::Propagate,
        ActivationOperation::Reinforce,
        ActivationOperation::Propagate,
    ];
    
    assert_eq!(learning_sequence.len(), 3);
    assert!(learning_sequence.contains(&ActivationOperation::Reinforce));
    
    // Inhibition sequence
    let inhibition_sequence = vec![
        ActivationOperation::Propagate,
        ActivationOperation::Inhibit,
        ActivationOperation::Decay,
    ];
    
    assert_eq!(inhibition_sequence.len(), 3);
    assert!(inhibition_sequence.contains(&ActivationOperation::Inhibit));
}

// ==================== Operation State Transition Tests ====================

#[test]
fn test_typical_operation_transitions() {
    // Test typical state transitions in neural processing
    
    let transitions = vec![
        (ActivationOperation::Initialize, ActivationOperation::Propagate),
        (ActivationOperation::Propagate, ActivationOperation::Reinforce),
        (ActivationOperation::Propagate, ActivationOperation::Inhibit),
        (ActivationOperation::Reinforce, ActivationOperation::Propagate),
        (ActivationOperation::Inhibit, ActivationOperation::Decay),
        (ActivationOperation::Decay, ActivationOperation::Initialize),
    ];
    
    // Verify transitions are valid (all operations can follow any other)
    for (from_op, to_op) in transitions {
        // In a real system, you might have rules about valid transitions
        // For now, we just verify the operations exist and are different
        assert_ne!(from_op, to_op, "Transition should be between different operations");
        
        // Both operations should be valid
        let from_str = format!("{:?}", from_op);
        let to_str = format!("{:?}", to_op);
        assert!(!from_str.is_empty());
        assert!(!to_str.is_empty());
    }
}

#[test]
fn test_operation_cycle_detection() {
    // Test for potential cycles in operation sequences
    
    let cycle_sequence = vec![
        ActivationOperation::Initialize,
        ActivationOperation::Propagate,
        ActivationOperation::Reinforce,
        ActivationOperation::Propagate,
        ActivationOperation::Decay,
        ActivationOperation::Initialize, // Back to start
    ];
    
    // Check for cycle (Initialize appears twice)
    let first_init = cycle_sequence.iter().position(|&op| op == ActivationOperation::Initialize);
    let last_init = cycle_sequence.iter().rposition(|&op| op == ActivationOperation::Initialize);
    
    assert_eq!(first_init, Some(0));
    assert_eq!(last_init, Some(5));
    assert_ne!(first_init, last_init, "Should detect cycle");
}

// ==================== Batch Operation Tests ====================

#[test]
fn test_operation_collections() {
    let all_operations = vec![
        ActivationOperation::Initialize,
        ActivationOperation::Propagate,
        ActivationOperation::Inhibit,
        ActivationOperation::Reinforce,
        ActivationOperation::Decay,
    ];
    
    // Test that we can work with collections of operations
    assert_eq!(all_operations.len(), 5);
    
    // Test filtering
    let propagate_ops: Vec<_> = all_operations.iter()
        .filter(|&&op| op == ActivationOperation::Propagate)
        .collect();
    assert_eq!(propagate_ops.len(), 1);
    
    // Test mapping
    let debug_strings: Vec<_> = all_operations.iter()
        .map(|op| format!("{:?}", op))
        .collect();
    assert_eq!(debug_strings.len(), 5);
    assert!(debug_strings.contains(&"Initialize".to_string()));
    assert!(debug_strings.contains(&"Propagate".to_string()));
    assert!(debug_strings.contains(&"Inhibit".to_string()));
    assert!(debug_strings.contains(&"Reinforce".to_string()));
    assert!(debug_strings.contains(&"Decay".to_string()));
}

#[test]
fn test_operation_frequency_analysis() {
    // Simulate a sequence of operations and analyze frequency
    let operation_sequence = vec![
        ActivationOperation::Initialize,
        ActivationOperation::Propagate,
        ActivationOperation::Propagate,
        ActivationOperation::Reinforce,
        ActivationOperation::Propagate,
        ActivationOperation::Decay,
        ActivationOperation::Propagate,
        ActivationOperation::Inhibit,
        ActivationOperation::Decay,
    ];
    
    // Count frequencies
    let propagate_count = operation_sequence.iter()
        .filter(|&&op| op == ActivationOperation::Propagate)
        .count();
    let decay_count = operation_sequence.iter()
        .filter(|&&op| op == ActivationOperation::Decay)
        .count();
    let init_count = operation_sequence.iter()
        .filter(|&&op| op == ActivationOperation::Initialize)
        .count();
    
    assert_eq!(propagate_count, 4); // Most common
    assert_eq!(decay_count, 2);
    assert_eq!(init_count, 1);
    
    // Propagate should be most frequent (typical in neural processing)
    assert!(propagate_count >= decay_count);
    assert!(propagate_count >= init_count);
}

// ==================== Serialization Edge Cases ====================

#[test]
fn test_operation_serialization_edge_cases() {
    let operations = vec![
        ActivationOperation::Initialize,
        ActivationOperation::Propagate,
        ActivationOperation::Inhibit,
        ActivationOperation::Reinforce,
        ActivationOperation::Decay,
    ];
    
    // Test array serialization
    let serialized_array = serde_json::to_string(&operations).unwrap();
    let deserialized_array: Vec<ActivationOperation> = serde_json::from_str(&serialized_array).unwrap();
    
    assert_eq!(operations.len(), deserialized_array.len());
    for (orig, deser) in operations.iter().zip(deserialized_array.iter()) {
        assert_eq!(orig, deser);
    }
    
    // Test nested structure serialization
    let nested_structure = vec![
        vec![ActivationOperation::Initialize, ActivationOperation::Propagate],
        vec![ActivationOperation::Reinforce],
        vec![ActivationOperation::Inhibit, ActivationOperation::Decay],
    ];
    
    let nested_serialized = serde_json::to_string(&nested_structure).unwrap();
    let nested_deserialized: Vec<Vec<ActivationOperation>> = serde_json::from_str(&nested_serialized).unwrap();
    
    assert_eq!(nested_structure.len(), nested_deserialized.len());
    for (orig_vec, deser_vec) in nested_structure.iter().zip(nested_deserialized.iter()) {
        assert_eq!(orig_vec.len(), deser_vec.len());
        for (orig_op, deser_op) in orig_vec.iter().zip(deser_vec.iter()) {
            assert_eq!(orig_op, deser_op);
        }
    }
}

// ==================== Performance Tests ====================

#[test]
fn test_operation_performance() {
    let num_operations = 100000;
    let mut operations = Vec::new();
    
    let start_time = std::time::Instant::now();
    
    // Create many operation instances
    for i in 0..num_operations {
        let op = match i % 5 {
            0 => ActivationOperation::Initialize,
            1 => ActivationOperation::Propagate,
            2 => ActivationOperation::Inhibit,
            3 => ActivationOperation::Reinforce,
            _ => ActivationOperation::Decay,
        };
        operations.push(op);
    }
    
    let creation_time = start_time.elapsed();
    
    // Should create operations very quickly (they're just enums)
    assert!(
        creation_time.as_millis() < 10,
        "Creating {} operations took too long: {:?}",
        num_operations, creation_time
    );
    
    // Test iteration performance
    let iterate_start = std::time::Instant::now();
    let propagate_count = operations.iter()
        .filter(|&&op| op == ActivationOperation::Propagate)
        .count();
    let iterate_time = iterate_start.elapsed();
    
    assert!(
        iterate_time.as_millis() < 10,
        "Iterating through {} operations took too long: {:?}",
        num_operations, iterate_time
    );
    
    // Verify count is reasonable
    assert!(propagate_count > 0);
    assert!(propagate_count <= num_operations);
}

#[test]
fn test_operation_memory_usage() {
    use std::mem;
    
    let operation = ActivationOperation::Propagate;
    let size = mem::size_of_val(&operation);
    
    // Should be very small (just an enum)
    assert!(
        size <= 8,
        "ActivationOperation should be very compact, got {} bytes",
        size
    );
    
    // Test array memory usage
    let operations = vec![ActivationOperation::Propagate; 1000];
    let array_size = mem::size_of_val(&operations);
    let expected_overhead = mem::size_of::<Vec<ActivationOperation>>();
    
    // Array should have reasonable memory overhead
    assert!(
        array_size >= expected_overhead,
        "Array size should include Vec overhead"
    );
}

// ==================== Hash and Comparison Tests ====================

#[test]
fn test_operation_in_hashmap() {
    use std::collections::HashMap;
    
    let mut operation_counts = HashMap::new();
    
    // Use operations as keys
    operation_counts.insert(ActivationOperation::Initialize, 0);
    operation_counts.insert(ActivationOperation::Propagate, 0);
    operation_counts.insert(ActivationOperation::Inhibit, 0);
    operation_counts.insert(ActivationOperation::Reinforce, 0);
    operation_counts.insert(ActivationOperation::Decay, 0);
    
    // Increment counters
    *operation_counts.get_mut(&ActivationOperation::Propagate).unwrap() += 1;
    *operation_counts.get_mut(&ActivationOperation::Propagate).unwrap() += 1;
    *operation_counts.get_mut(&ActivationOperation::Decay).unwrap() += 1;
    
    assert_eq!(operation_counts[&ActivationOperation::Initialize], 0);
    assert_eq!(operation_counts[&ActivationOperation::Propagate], 2);
    assert_eq!(operation_counts[&ActivationOperation::Decay], 1);
}

#[test]
fn test_operation_in_hashset() {
    use std::collections::HashSet;
    
    let mut operation_set = HashSet::new();
    
    // Add operations to set
    operation_set.insert(ActivationOperation::Initialize);
    operation_set.insert(ActivationOperation::Propagate);
    operation_set.insert(ActivationOperation::Propagate); // Duplicate
    
    // Set should contain unique operations only
    assert_eq!(operation_set.len(), 2);
    assert!(operation_set.contains(&ActivationOperation::Initialize));
    assert!(operation_set.contains(&ActivationOperation::Propagate));
    assert!(!operation_set.contains(&ActivationOperation::Inhibit));
}

// ==================== Pattern Matching Tests ====================

#[test]
fn test_operation_pattern_matching() {
    let operations = vec![
        ActivationOperation::Initialize,
        ActivationOperation::Propagate,
        ActivationOperation::Inhibit,
        ActivationOperation::Reinforce,
        ActivationOperation::Decay,
    ];
    
    for operation in operations {
        let category = match operation {
            ActivationOperation::Initialize => "setup",
            ActivationOperation::Propagate => "forward",
            ActivationOperation::Inhibit => "suppress",
            ActivationOperation::Reinforce => "strengthen",
            ActivationOperation::Decay => "reduce",
        };
        
        // Verify categorization
        match operation {
            ActivationOperation::Initialize => assert_eq!(category, "setup"),
            ActivationOperation::Propagate => assert_eq!(category, "forward"),
            ActivationOperation::Inhibit => assert_eq!(category, "suppress"),
            ActivationOperation::Reinforce => assert_eq!(category, "strengthen"),
            ActivationOperation::Decay => assert_eq!(category, "reduce"),
        }
    }
}

#[test]
fn test_operation_conditional_logic() {
    let operation = ActivationOperation::Propagate;
    
    // Test various conditional patterns
    let is_modifying = matches!(operation, 
        ActivationOperation::Propagate | 
        ActivationOperation::Reinforce | 
        ActivationOperation::Inhibit |
        ActivationOperation::Decay
    );
    assert!(is_modifying);
    
    let is_initializing = matches!(operation, ActivationOperation::Initialize);
    assert!(!is_initializing);
    
    let is_positive = matches!(operation,
        ActivationOperation::Initialize |
        ActivationOperation::Propagate |
        ActivationOperation::Reinforce
    );
    assert!(is_positive);
    
    let is_negative = matches!(operation,
        ActivationOperation::Inhibit |
        ActivationOperation::Decay
    );
    assert!(!is_negative);
}