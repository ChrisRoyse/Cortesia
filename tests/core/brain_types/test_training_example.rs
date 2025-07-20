/// Comprehensive tests for TrainingExample struct
/// 
/// This module tests training data structures for neural structure prediction including:
/// - Training example creation and validation
/// - Metadata management and access patterns
/// - Expected operations sequence handling
/// - Serialization and deserialization of training data
/// - Complex training scenarios and edge cases

use llmkg::core::brain_types::{TrainingExample, GraphOperation, EntityDirection, LogicGateType, RelationType};
use std::collections::HashMap;
use serde_json;
use super::test_constants;
use super::test_helpers::*;

// ==================== Basic Training Example Tests ====================

#[test]
fn test_training_example_creation() {
    let text = "Create a neural connection between input and output nodes";
    let operations = vec![
        create_node_operation("input_node", EntityDirection::Input),
        create_node_operation("output_node", EntityDirection::Output),
        create_relationship_operation("input_node", "output_node", RelationType::RelatedTo, 0.8),
    ];
    let metadata = vec![
        ("domain", "neural_networks"),
        ("difficulty", "basic"),
    ];
    
    let example = create_training_example(text, operations.clone(), metadata);
    
    assert_eq!(example.text, text);
    assert_eq!(example.expected_operations.len(), operations.len());
    assert_eq!(example.metadata.len(), 2);
    assert_eq!(example.metadata.get("domain"), Some(&"neural_networks".to_string()));
    assert_eq!(example.metadata.get("difficulty"), Some(&"basic".to_string()));
}

#[test]
fn test_training_example_empty_components() {
    // Test with empty operations
    let empty_ops_example = create_training_example(
        "No operations needed",
        vec![],
        vec![("type", "null_operation")]
    );
    
    assert!(empty_ops_example.expected_operations.is_empty());
    assert_eq!(empty_ops_example.metadata.len(), 1);
    
    // Test with empty metadata
    let empty_meta_example = create_training_example(
        "Simple operation",
        vec![create_node_operation("test", EntityDirection::Input)],
        vec![]
    );
    
    assert!(empty_meta_example.metadata.is_empty());
    assert_eq!(empty_meta_example.expected_operations.len(), 1);
    
    // Test with empty text
    let empty_text_example = create_training_example(
        "",
        vec![create_node_operation("empty_text_test", EntityDirection::Hidden)],
        vec![("note", "empty_text_case")]
    );
    
    assert!(empty_text_example.text.is_empty());
    assert_eq!(empty_text_example.expected_operations.len(), 1);
}

// ==================== Complex Training Example Tests ====================

#[test]
fn test_complex_neural_network_training_example() {
    let text = "Build a two-layer neural network with AND gates and weighted connections";
    
    let operations = vec![
        // Input layer
        create_node_operation("input_1", EntityDirection::Input),
        create_node_operation("input_2", EntityDirection::Input),
        
        // Hidden layer with logic gates
        create_gate_operation(
            vec!["input_1", "input_2"],
            vec!["hidden_gate_1"],
            LogicGateType::And
        ),
        create_gate_operation(
            vec!["input_1", "input_2"],
            vec!["hidden_gate_2"],
            LogicGateType::Or
        ),
        
        // Output layer
        create_node_operation("output", EntityDirection::Output),
        
        // Connections
        create_relationship_operation("hidden_gate_1", "output", RelationType::Learned, 0.7),
        create_relationship_operation("hidden_gate_2", "output", RelationType::Learned, 0.3),
    ];
    
    let metadata = vec![
        ("architecture", "two_layer"),
        ("gate_types", "and_or_mixed"),
        ("learning_type", "supervised"),
        ("complexity", "intermediate"),
    ];
    
    let example = create_training_example(text, operations, metadata);
    
    // Verify structure
    assert!(example.text.contains("neural network"));
    assert_eq!(example.expected_operations.len(), 7);
    assert_eq!(example.metadata.len(), 4);
    
    // Count operation types
    let node_ops = example.expected_operations.iter()
        .filter(|op| matches!(op, GraphOperation::CreateNode { .. }))
        .count();
    let gate_ops = example.expected_operations.iter()
        .filter(|op| matches!(op, GraphOperation::CreateLogicGate { .. }))
        .count();
    let rel_ops = example.expected_operations.iter()
        .filter(|op| matches!(op, GraphOperation::CreateRelationship { .. }))
        .count();
    
    assert_eq!(node_ops, 3); // 2 inputs + 1 output
    assert_eq!(gate_ops, 2); // AND and OR gates
    assert_eq!(rel_ops, 2); // 2 connections to output
}

#[test]
fn test_ontological_training_example() {
    let text = "Model the relationship: 'A cat is a mammal and has the property of being furry'";
    
    let operations = vec![
        create_node_operation("cat", EntityDirection::Input),
        create_node_operation("mammal", EntityDirection::Output),
        create_node_operation("furry", EntityDirection::Output),
        
        create_relationship_operation("cat", "mammal", RelationType::IsA, 1.0),
        create_relationship_operation("cat", "furry", RelationType::HasProperty, 0.9),
    ];
    
    let metadata = vec![
        ("domain", "ontology"),
        ("relation_types", "is_a,has_property"),
        ("entities", "3"),
        ("source", "knowledge_base"),
    ];
    
    let example = create_training_example(text, operations, metadata);
    
    assert!(example.text.contains("cat"));
    assert!(example.text.contains("mammal"));
    assert!(example.text.contains("furry"));
    
    // Verify relationship types
    let has_isa = example.expected_operations.iter()
        .any(|op| match op {
            GraphOperation::CreateRelationship { relation_type: RelationType::IsA, .. } => true,
            _ => false,
        });
    let has_property = example.expected_operations.iter()
        .any(|op| match op {
            GraphOperation::CreateRelationship { relation_type: RelationType::HasProperty, .. } => true,
            _ => false,
        });
    
    assert!(has_isa, "Should contain IsA relationship");
    assert!(has_property, "Should contain HasProperty relationship");
}

#[test]
fn test_temporal_sequence_training_example() {
    let text = "Model temporal sequence: first A happens, then B, then C, with causal relationships";
    
    let operations = vec![
        create_node_operation("event_A", EntityDirection::Input),
        create_node_operation("event_B", EntityDirection::Hidden),
        create_node_operation("event_C", EntityDirection::Output),
        
        create_relationship_operation("event_A", "event_B", RelationType::Temporal, 0.8),
        create_relationship_operation("event_B", "event_C", RelationType::Temporal, 0.8),
        create_relationship_operation("event_A", "event_C", RelationType::Temporal, 0.4), // Indirect
    ];
    
    let metadata = vec![
        ("sequence_type", "linear"),
        ("causality", "strong"),
        ("temporal_distance", "short_medium_long"),
    ];
    
    let example = create_training_example(text, operations, metadata);
    
    // All relationships should be temporal
    let temporal_count = example.expected_operations.iter()
        .filter(|op| match op {
            GraphOperation::CreateRelationship { relation_type: RelationType::Temporal, .. } => true,
            _ => false,
        })
        .count();
    
    assert_eq!(temporal_count, 3);
}

// ==================== Metadata Management Tests ====================

#[test]
fn test_metadata_various_types() {
    let metadata_variants = vec![
        vec![
            ("string_value", "text"),
            ("number_as_string", "42"),
            ("boolean_as_string", "true"),
            ("float_as_string", "3.14159"),
        ],
        vec![
            ("empty_value", ""),
            ("special_chars", "!@#$%^&*()"),
            ("unicode", "日本語"),
            ("whitespace", "   spaces   "),
        ],
        vec![
            ("very_long_key_that_exceeds_normal_expectations", "value"),
            ("key", "very_long_value_that_contains_lots_of_information_and_details"),
            ("nested_structure_simulation", "key1:value1,key2:value2"),
        ],
    ];
    
    for (i, metadata) in metadata_variants.into_iter().enumerate() {
        let example = create_training_example(
            &format!("Test case {}", i),
            vec![create_node_operation("test", EntityDirection::Input)],
            metadata.clone()
        );
        
        assert_eq!(example.metadata.len(), metadata.len());
        
        for (key, expected_value) in metadata {
            assert_eq!(
                example.metadata.get(key),
                Some(&expected_value.to_string()),
                "Metadata key '{}' should have value '{}'",
                key, expected_value
            );
        }
    }
}

#[test]
fn test_metadata_duplicate_keys() {
    // Note: HashMap will keep the last value for duplicate keys
    let metadata = vec![
        ("duplicate_key", "first_value"),
        ("unique_key", "unique_value"),
        ("duplicate_key", "second_value"), // This should override the first
    ];
    
    let example = create_training_example(
        "Duplicate key test",
        vec![],
        metadata
    );
    
    // Should have 2 unique keys
    assert_eq!(example.metadata.len(), 2);
    assert_eq!(example.metadata.get("duplicate_key"), Some(&"second_value".to_string()));
    assert_eq!(example.metadata.get("unique_key"), Some(&"unique_value".to_string()));
}

#[test]
fn test_metadata_access_patterns() {
    let example = create_training_example(
        "Metadata access test",
        vec![],
        vec![
            ("difficulty", "hard"),
            ("domain", "neural_networks"),
            ("version", "1.0"),
        ]
    );
    
    // Test various access patterns
    assert!(example.metadata.contains_key("difficulty"));
    assert!(!example.metadata.contains_key("nonexistent"));
    
    // Test iteration
    let mut keys: Vec<_> = example.metadata.keys().collect();
    keys.sort();
    assert_eq!(keys, vec!["difficulty", "domain", "version"]);
    
    // Test values
    let mut values: Vec<_> = example.metadata.values().cloned().collect();
    values.sort();
    assert_eq!(values, vec!["1.0", "hard", "neural_networks"]);
}

// ==================== Serialization Tests ====================

#[test]
fn test_training_example_serialization() {
    let example = create_training_example(
        "Serialization test example",
        vec![
            create_node_operation("input", EntityDirection::Input),
            create_gate_operation(vec!["input"], vec!["output"], LogicGateType::Identity),
        ],
        vec![
            ("test_type", "serialization"),
            ("complexity", "simple"),
        ]
    );
    
    let serialized = serde_json::to_string(&example)
        .expect("Should serialize successfully");
    
    // Verify JSON structure
    assert!(serialized.contains("text"));
    assert!(serialized.contains("expected_operations"));
    assert!(serialized.contains("metadata"));
    assert!(serialized.contains("Serialization test example"));
    assert!(serialized.contains("CreateNode"));
    assert!(serialized.contains("CreateLogicGate"));
    assert!(serialized.contains("test_type"));
}

#[test]
fn test_training_example_deserialization() {
    let original = create_training_example(
        "Round-trip test",
        vec![
            create_relationship_operation("source", "target", RelationType::Similar, 0.5),
        ],
        vec![
            ("round_trip", "true"),
            ("format", "json"),
        ]
    );
    
    let serialized = serde_json::to_string(&original).unwrap();
    let deserialized: TrainingExample = serde_json::from_str(&serialized)
        .expect("Should deserialize successfully");
    
    // Verify round-trip correctness
    assert_eq!(deserialized.text, original.text);
    assert_eq!(deserialized.expected_operations.len(), original.expected_operations.len());
    assert_eq!(deserialized.metadata.len(), original.metadata.len());
    
    // Verify metadata
    for (key, value) in &original.metadata {
        assert_eq!(deserialized.metadata.get(key), Some(value));
    }
    
    // Verify operations
    for (orig_op, deser_op) in original.expected_operations.iter().zip(deserialized.expected_operations.iter()) {
        match (orig_op, deser_op) {
            (
                GraphOperation::CreateRelationship { source: s1, target: t1, relation_type: r1, weight: w1 },
                GraphOperation::CreateRelationship { source: s2, target: t2, relation_type: r2, weight: w2 }
            ) => {
                assert_eq!(s1, s2);
                assert_eq!(t1, t2);
                assert_eq!(r1, r2);
                assert_float_eq(*w1, *w2, test_constants::ACTIVATION_EPSILON);
            },
            _ => panic!("Operation types should match after deserialization"),
        }
    }
}

#[test]
fn test_training_example_json_structure() {
    let example = create_training_example(
        "JSON structure validation",
        vec![create_node_operation("test_node", EntityDirection::Gate)],
        vec![("key1", "value1"), ("key2", "value2")]
    );
    
    let json_value: serde_json::Value = serde_json::to_value(&example).unwrap();
    
    // Verify top-level structure
    assert!(json_value.is_object());
    let obj = json_value.as_object().unwrap();
    
    assert!(obj.contains_key("text"));
    assert!(obj.contains_key("expected_operations"));
    assert!(obj.contains_key("metadata"));
    
    // Verify text field
    assert!(obj["text"].is_string());
    assert_eq!(obj["text"].as_str().unwrap(), "JSON structure validation");
    
    // Verify operations array
    assert!(obj["expected_operations"].is_array());
    let ops_array = obj["expected_operations"].as_array().unwrap();
    assert_eq!(ops_array.len(), 1);
    
    // Verify metadata object
    assert!(obj["metadata"].is_object());
    let metadata_obj = obj["metadata"].as_object().unwrap();
    assert_eq!(metadata_obj.len(), 2);
    assert!(metadata_obj.contains_key("key1"));
    assert!(metadata_obj.contains_key("key2"));
}

// ==================== Large Training Example Tests ====================

#[test]
fn test_large_training_example() {
    let num_operations = 1000;
    let mut operations = Vec::new();
    
    // Create many operations
    for i in 0..num_operations {
        let operation = match i % 3 {
            0 => create_node_operation(&format!("node_{}", i), EntityDirection::Hidden),
            1 => create_gate_operation(
                vec![&format!("input_{}", i)],
                vec![&format!("output_{}", i)],
                LogicGateType::Threshold
            ),
            _ => create_relationship_operation(
                &format!("source_{}", i),
                &format!("target_{}", i),
                RelationType::Learned,
                (i as f32) / (num_operations as f32)
            ),
        };
        operations.push(operation);
    }
    
    let metadata = vec![
        ("size", "large"),
        ("operations_count", &num_operations.to_string()),
        ("generated", "true"),
    ];
    
    let example = create_training_example(
        "Large-scale neural network construction",
        operations,
        metadata
    );
    
    assert_eq!(example.expected_operations.len(), num_operations);
    assert_eq!(example.metadata.len(), 3);
    
    // Test serialization performance
    let start_time = std::time::Instant::now();
    let _serialized = serde_json::to_string(&example).unwrap();
    let duration = start_time.elapsed();
    
    assert!(
        duration.as_millis() < 1000,
        "Large example serialization took too long: {:?}",
        duration
    );
}

// ==================== Training Example Collections Tests ====================

#[test]
fn test_training_example_collection() {
    let examples = vec![
        create_training_example(
            "Basic node creation",
            vec![create_node_operation("simple", EntityDirection::Input)],
            vec![("type", "basic")]
        ),
        create_training_example(
            "Logic gate creation",
            vec![create_gate_operation(vec!["in"], vec!["out"], LogicGateType::And)],
            vec![("type", "logic")]
        ),
        create_training_example(
            "Relationship creation",
            vec![create_relationship_operation("a", "b", RelationType::RelatedTo, 0.5)],
            vec![("type", "relationship")]
        ),
    ];
    
    // Test collection serialization
    let serialized = serde_json::to_string(&examples).unwrap();
    let deserialized: Vec<TrainingExample> = serde_json::from_str(&serialized).unwrap();
    
    assert_eq!(examples.len(), deserialized.len());
    
    // Verify each example
    for (orig, deser) in examples.iter().zip(deserialized.iter()) {
        assert_eq!(orig.text, deser.text);
        assert_eq!(orig.expected_operations.len(), deser.expected_operations.len());
        assert_eq!(orig.metadata.len(), deser.metadata.len());
    }
}

#[test]
fn test_heterogeneous_training_examples() {
    // Create examples with different characteristics
    let examples = vec![
        // Simple example
        create_training_example("Simple", vec![], vec![]),
        
        // Complex example
        create_training_example(
            "Complex neural network with multiple layers and connections",
            vec![
                create_node_operation("input1", EntityDirection::Input),
                create_node_operation("input2", EntityDirection::Input),
                create_gate_operation(vec!["input1", "input2"], vec!["hidden"], LogicGateType::And),
                create_node_operation("output", EntityDirection::Output),
                create_relationship_operation("hidden", "output", RelationType::Learned, 0.8),
            ],
            vec![
                ("complexity", "high"),
                ("layers", "3"),
                ("connections", "1"),
            ]
        ),
        
        // Metadata-heavy example
        create_training_example(
            "Metadata test",
            vec![create_node_operation("meta", EntityDirection::Hidden)],
            vec![
                ("author", "test_suite"),
                ("created", "2024"),
                ("version", "1.0"),
                ("tags", "metadata,test,comprehensive"),
                ("difficulty", "easy"),
                ("domain", "testing"),
                ("language", "english"),
                ("format", "structured"),
            ]
        ),
    ];
    
    // Should handle heterogeneous examples gracefully
    for (i, example) in examples.iter().enumerate() {
        assert!(!example.text.is_empty() || example.text == "Simple");
        
        if i == 1 { // Complex example
            assert!(example.expected_operations.len() >= 5);
        }
        
        if i == 2 { // Metadata-heavy example
            assert!(example.metadata.len() >= 7);
        }
    }
}

// ==================== Error Handling and Edge Cases ====================

#[test]
fn test_training_example_edge_cases() {
    // Very long text
    let long_text = "A".repeat(10000);
    let long_text_example = create_training_example(
        &long_text,
        vec![create_node_operation("long_text_test", EntityDirection::Input)],
        vec![("text_length", "10000")]
    );
    assert_eq!(long_text_example.text.len(), 10000);
    
    // Special characters in text
    let special_text = "Text with special chars: !@#$%^&*()[]{}|;':\",./<>?`~";
    let special_example = create_training_example(
        special_text,
        vec![],
        vec![("contains", "special_chars")]
    );
    assert_eq!(special_example.text, special_text);
    
    // Unicode text
    let unicode_text = "Multiple languages: English, 中文, 日本語, العربية, Русский";
    let unicode_example = create_training_example(
        unicode_text,
        vec![],
        vec![("languages", "multiple")]
    );
    assert_eq!(unicode_example.text, unicode_text);
}

// ==================== Memory and Performance Tests ====================

#[test]
fn test_training_example_memory_usage() {
    use std::mem;
    
    let example = create_training_example(
        "Memory test",
        vec![create_node_operation("memory", EntityDirection::Input)],
        vec![("test", "memory")]
    );
    
    let size = mem::size_of_val(&example);
    
    // Should be reasonably sized (this will vary by platform)
    assert!(
        size < 1000,
        "TrainingExample should be reasonably compact, got {} bytes",
        size
    );
}

#[test]
fn test_training_example_performance() {
    let num_examples = 1000;
    let start_time = std::time::Instant::now();
    
    let mut examples = Vec::new();
    for i in 0..num_examples {
        let example = create_training_example(
            &format!("Performance test example {}", i),
            vec![
                create_node_operation(&format!("node_{}", i), EntityDirection::Input),
                create_relationship_operation(
                    &format!("source_{}", i),
                    &format!("target_{}", i),
                    RelationType::RelatedTo,
                    (i as f32) / (num_examples as f32)
                ),
            ],
            vec![
                ("index", &i.to_string()),
                ("batch", "performance_test"),
            ]
        );
        examples.push(example);
    }
    
    let creation_time = start_time.elapsed();
    
    assert!(
        creation_time.as_millis() < 100,
        "Creating {} training examples took too long: {:?}",
        num_examples, creation_time
    );
    
    // Test batch serialization
    let serialize_start = std::time::Instant::now();
    let _serialized = serde_json::to_string(&examples).unwrap();
    let serialize_time = serialize_start.elapsed();
    
    assert!(
        serialize_time.as_millis() < 2000,
        "Serializing {} examples took too long: {:?}",
        num_examples, serialize_time
    );
}