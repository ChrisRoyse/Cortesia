// Tests for LogicGateType enum and Display trait
// Validates neural logic gate type classification and string representation

use llmkg::core::brain_types::LogicGateType;
use serde_json;
use std::fmt::Display;

use super::test_constants;

// ==================== Basic Enum Tests ====================

#[test]
fn test_logic_gate_type_variants() {
    // Test all enum variants exist and are distinct
    let gates = [
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Not,
        LogicGateType::Xor,
        LogicGateType::Nand,
        LogicGateType::Nor,
        LogicGateType::Xnor,
        LogicGateType::Identity,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
        LogicGateType::Weighted,
    ];

    // Verify all variants are distinct
    for (i, gate1) in gates.iter().enumerate() {
        for (j, gate2) in gates.iter().enumerate() {
            if i != j {
                assert_ne!(gate1, gate2, "Gates at indices {} and {} should be different", i, j);
            } else {
                assert_eq!(gate1, gate2, "Same gate should equal itself");
            }
        }
    }
}

#[test]
fn test_logic_gate_type_copy_clone() {
    let original = LogicGateType::And;
    let copied = original; // Test Copy trait
    let cloned = original.clone(); // Test Clone trait

    assert_eq!(original, copied);
    assert_eq!(original, cloned);
    assert_eq!(copied, cloned);
}

#[test]
fn test_logic_gate_type_debug() {
    // Test Debug trait implementation
    let and_gate = LogicGateType::And;
    let debug_str = format!("{:?}", and_gate);
    assert!(debug_str.contains("And"));

    let inhibitory_gate = LogicGateType::Inhibitory;
    let debug_str = format!("{:?}", inhibitory_gate);
    assert!(debug_str.contains("Inhibitory"));
}

// ==================== Display Trait Tests ====================

#[test]
fn test_logic_gate_type_display_basic() {
    // Test Display trait for all variants
    assert_eq!(LogicGateType::And.to_string(), "and");
    assert_eq!(LogicGateType::Or.to_string(), "or");
    assert_eq!(LogicGateType::Not.to_string(), "not");
    assert_eq!(LogicGateType::Xor.to_string(), "xor");
}

#[test]
fn test_logic_gate_type_display_extended() {
    // Test Display trait for extended variants
    assert_eq!(LogicGateType::Nand.to_string(), "nand");
    assert_eq!(LogicGateType::Nor.to_string(), "nor");
    assert_eq!(LogicGateType::Xnor.to_string(), "xnor");
    assert_eq!(LogicGateType::Identity.to_string(), "identity");
}

#[test]
fn test_logic_gate_type_display_advanced() {
    // Test Display trait for advanced gate types
    assert_eq!(LogicGateType::Threshold.to_string(), "threshold");
    assert_eq!(LogicGateType::Inhibitory.to_string(), "inhibitory");
    assert_eq!(LogicGateType::Weighted.to_string(), "weighted");
}

#[test]
fn test_logic_gate_type_display_format() {
    // Test that display format is consistent (lowercase)
    let gates = [
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Not,
        LogicGateType::Xor,
        LogicGateType::Nand,
        LogicGateType::Nor,
        LogicGateType::Xnor,
        LogicGateType::Identity,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
        LogicGateType::Weighted,
    ];

    for gate in gates {
        let display_str = gate.to_string();
        assert_eq!(display_str, display_str.to_lowercase(), "Display should be lowercase");
        assert!(!display_str.is_empty(), "Display should not be empty");
        assert!(!display_str.contains(' '), "Display should not contain spaces");
    }
}

#[test]
fn test_logic_gate_type_display_uniqueness() {
    // Test that all display strings are unique
    let gates = [
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Not,
        LogicGateType::Xor,
        LogicGateType::Nand,
        LogicGateType::Nor,
        LogicGateType::Xnor,
        LogicGateType::Identity,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
        LogicGateType::Weighted,
    ];

    let mut display_strings = Vec::new();
    for gate in gates {
        let display_str = gate.to_string();
        assert!(!display_strings.contains(&display_str), 
                "Display string '{}' is not unique", display_str);
        display_strings.push(display_str);
    }
}

// ==================== Serialization Tests ====================

#[test]
fn test_logic_gate_type_serialization() {
    // Test serde serialization
    let and_gate = LogicGateType::And;
    let serialized = serde_json::to_string(&and_gate).expect("Should serialize");
    assert!(serialized.contains("And"));

    let inhibitory_gate = LogicGateType::Inhibitory;
    let serialized = serde_json::to_string(&inhibitory_gate).expect("Should serialize");
    assert!(serialized.contains("Inhibitory"));

    let weighted_gate = LogicGateType::Weighted;
    let serialized = serde_json::to_string(&weighted_gate).expect("Should serialize");
    assert!(serialized.contains("Weighted"));
}

#[test]
fn test_logic_gate_type_deserialization() {
    // Test serde deserialization
    let and_json = "\"And\"";
    let deserialized: LogicGateType = serde_json::from_str(and_json)
        .expect("Should deserialize And");
    assert_eq!(deserialized, LogicGateType::And);

    let inhibitory_json = "\"Inhibitory\"";
    let deserialized: LogicGateType = serde_json::from_str(inhibitory_json)
        .expect("Should deserialize Inhibitory");
    assert_eq!(deserialized, LogicGateType::Inhibitory);

    let threshold_json = "\"Threshold\"";
    let deserialized: LogicGateType = serde_json::from_str(threshold_json)
        .expect("Should deserialize Threshold");
    assert_eq!(deserialized, LogicGateType::Threshold);
}

#[test]
fn test_logic_gate_type_round_trip() {
    // Test serialization round trip for all variants
    let gates = [
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Not,
        LogicGateType::Xor,
        LogicGateType::Nand,
        LogicGateType::Nor,
        LogicGateType::Xnor,
        LogicGateType::Identity,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
        LogicGateType::Weighted,
    ];

    for gate in gates {
        let serialized = serde_json::to_string(&gate)
            .expect("Should serialize");
        let deserialized: LogicGateType = serde_json::from_str(&serialized)
            .expect("Should deserialize");
        assert_eq!(gate, deserialized);
    }
}

// ==================== Semantic Meaning Tests ====================

#[test]
fn test_logic_gate_type_basic_gates() {
    // Test basic Boolean logic gates
    assert_eq!(LogicGateType::And, LogicGateType::And);
    assert_eq!(LogicGateType::Or, LogicGateType::Or);
    assert_eq!(LogicGateType::Not, LogicGateType::Not);
    assert_eq!(LogicGateType::Xor, LogicGateType::Xor);
}

#[test]
fn test_logic_gate_type_compound_gates() {
    // Test compound logic gates (NAND, NOR, XNOR)
    assert_eq!(LogicGateType::Nand, LogicGateType::Nand); // NOT AND
    assert_eq!(LogicGateType::Nor, LogicGateType::Nor);   // NOT OR
    assert_eq!(LogicGateType::Xnor, LogicGateType::Xnor); // NOT XOR
}

#[test]
fn test_logic_gate_type_neural_gates() {
    // Test neural-specific gate types
    assert_eq!(LogicGateType::Identity, LogicGateType::Identity);     // Pass-through
    assert_eq!(LogicGateType::Threshold, LogicGateType::Threshold);   // Threshold activation
    assert_eq!(LogicGateType::Inhibitory, LogicGateType::Inhibitory); // Inhibitory processing
    assert_eq!(LogicGateType::Weighted, LogicGateType::Weighted);     // Weighted sum
}

// ==================== Collection and Usage Tests ====================

#[test]
fn test_logic_gate_type_in_collections() {
    use std::collections::{HashMap, HashSet};

    // Test in HashMap
    let mut gate_descriptions = HashMap::new();
    gate_descriptions.insert(LogicGateType::And, "Logical AND operation");
    gate_descriptions.insert(LogicGateType::Or, "Logical OR operation");
    gate_descriptions.insert(LogicGateType::Not, "Logical NOT operation");
    gate_descriptions.insert(LogicGateType::Inhibitory, "Neural inhibition");

    assert_eq!(gate_descriptions.get(&LogicGateType::And), Some(&"Logical AND operation"));
    assert_eq!(gate_descriptions.get(&LogicGateType::Inhibitory), Some(&"Neural inhibition"));

    // Test in HashSet
    let mut supported_gates = HashSet::new();
    supported_gates.insert(LogicGateType::And);
    supported_gates.insert(LogicGateType::Or);
    supported_gates.insert(LogicGateType::Threshold);
    supported_gates.insert(LogicGateType::Weighted);

    assert!(supported_gates.contains(&LogicGateType::And));
    assert!(supported_gates.contains(&LogicGateType::Threshold));
    assert!(!supported_gates.contains(&LogicGateType::Not));
}

#[test]
fn test_logic_gate_type_categorization() {
    // Test logical categorization of gate types
    let basic_gates = [LogicGateType::And, LogicGateType::Or, LogicGateType::Not, LogicGateType::Xor];
    let compound_gates = [LogicGateType::Nand, LogicGateType::Nor, LogicGateType::Xnor];
    let neural_gates = [LogicGateType::Identity, LogicGateType::Threshold, LogicGateType::Inhibitory, LogicGateType::Weighted];

    // Verify categorization completeness
    let mut all_gates = Vec::new();
    all_gates.extend_from_slice(&basic_gates);
    all_gates.extend_from_slice(&compound_gates);
    all_gates.extend_from_slice(&neural_gates);

    assert_eq!(all_gates.len(), 11); // Total number of gate types

    // Verify no overlaps between categories
    for basic in basic_gates {
        assert!(!compound_gates.contains(&basic));
        assert!(!neural_gates.contains(&basic));
    }
}

// ==================== Error Handling Tests ====================

#[test]
fn test_logic_gate_type_invalid_deserialization() {
    // Test invalid JSON deserialization
    let invalid_json = "\"InvalidGate\"";
    let result: Result<LogicGateType, _> = serde_json::from_str(invalid_json);
    assert!(result.is_err());

    let invalid_number = "42";
    let result: Result<LogicGateType, _> = serde_json::from_str(invalid_number);
    assert!(result.is_err());

    let invalid_object = "{}";
    let result: Result<LogicGateType, _> = serde_json::from_str(invalid_object);
    assert!(result.is_err());
}

// ==================== Performance Tests ====================

#[test]
fn test_logic_gate_type_memory_efficiency() {
    use std::mem;
    
    let size = mem::size_of::<LogicGateType>();
    // Enum should be small (typically 1 byte for simple enums)
    assert!(size <= 8, "LogicGateType size {} bytes is too large", size);
}

#[test]
fn test_logic_gate_type_pattern_matching_performance() {
    // Test that pattern matching is efficient
    let gates = [
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Not,
        LogicGateType::Inhibitory,
        LogicGateType::Weighted,
    ];

    let start = std::time::Instant::now();
    
    for _ in 0..10000 {
        for gate in gates {
            match gate {
                LogicGateType::And => { /* AND logic */ }
                LogicGateType::Or => { /* OR logic */ }
                LogicGateType::Not => { /* NOT logic */ }
                LogicGateType::Xor => { /* XOR logic */ }
                LogicGateType::Nand => { /* NAND logic */ }
                LogicGateType::Nor => { /* NOR logic */ }
                LogicGateType::Xnor => { /* XNOR logic */ }
                LogicGateType::Identity => { /* Identity logic */ }
                LogicGateType::Threshold => { /* Threshold logic */ }
                LogicGateType::Inhibitory => { /* Inhibitory logic */ }
                LogicGateType::Weighted => { /* Weighted logic */ }
            }
        }
    }
    
    let duration = start.elapsed();
    // Pattern matching should be very fast
    assert!(duration.as_millis() < 100, "Pattern matching took too long: {:?}", duration);
}

// ==================== Display Format Validation Tests ====================

#[test]
fn test_logic_gate_type_display_consistency() {
    // Test that Display trait matches expected naming conventions
    let expected_displays = [
        (LogicGateType::And, "and"),
        (LogicGateType::Or, "or"),
        (LogicGateType::Not, "not"),
        (LogicGateType::Xor, "xor"),
        (LogicGateType::Nand, "nand"),
        (LogicGateType::Nor, "nor"),
        (LogicGateType::Xnor, "xnor"),
        (LogicGateType::Identity, "identity"),
        (LogicGateType::Threshold, "threshold"),
        (LogicGateType::Inhibitory, "inhibitory"),
        (LogicGateType::Weighted, "weighted"),
    ];

    for (gate, expected) in expected_displays {
        assert_eq!(gate.to_string(), expected);
        assert_eq!(format!("{}", gate), expected);
    }
}

#[test]
fn test_logic_gate_type_display_immutability() {
    // Test that display strings don't change between calls
    let gate = LogicGateType::Inhibitory;
    let first_display = gate.to_string();
    let second_display = gate.to_string();
    let third_display = format!("{}", gate);
    
    assert_eq!(first_display, second_display);
    assert_eq!(second_display, third_display);
}

#[test]
fn test_logic_gate_type_display_validation() {
    // Test display strings are valid identifiers (useful for code generation)
    let gates = [
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Not,
        LogicGateType::Xor,
        LogicGateType::Nand,
        LogicGateType::Nor,
        LogicGateType::Xnor,
        LogicGateType::Identity,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
        LogicGateType::Weighted,
    ];

    for gate in gates {
        let display_str = gate.to_string();
        
        // Valid identifier rules: lowercase, no spaces, no special chars
        assert!(display_str.chars().all(|c| c.is_ascii_lowercase() || c == '_'), 
                "Display string '{}' contains invalid characters", display_str);
        assert!(!display_str.is_empty(), "Display string should not be empty");
        assert!(display_str.len() <= 20, "Display string '{}' is too long", display_str);
    }
}