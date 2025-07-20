/// Comprehensive tests for GraphOperation enum
/// 
/// This module tests neural structure prediction operations including:
/// - Node creation operations and validation
/// - Logic gate creation with input/output specification
/// - Relationship creation with proper typing and weights
/// - Serialization and deserialization of operations
/// - Operation validation and error handling

use llmkg::core::brain_types::{GraphOperation, EntityDirection, LogicGateType, RelationType};
use serde_json;
use super::test_constants;
use super::test_helpers::*;

// ==================== Node Creation Operation Tests ====================

#[test]
fn test_create_node_operation() {
    let concept = test_constants::TEST_CONCEPT_INPUT;
    let node_type = EntityDirection::Input;
    
    let operation = create_node_operation(concept, node_type);
    
    match operation {
        GraphOperation::CreateNode { concept: op_concept, node_type: op_type } => {
            assert_eq!(op_concept, concept);
            assert_eq!(op_type, node_type);
        },
        _ => panic!("Expected CreateNode operation"),
    }
}

#[test]
fn test_create_node_all_directions() {
    let directions = vec![
        EntityDirection::Input,
        EntityDirection::Output,
        EntityDirection::Gate,
        EntityDirection::Hidden,
    ];
    
    for (i, direction) in directions.iter().enumerate() {
        let concept = format!("test_concept_{}", i);
        let operation = create_node_operation(&concept, *direction);
        
        match operation {
            GraphOperation::CreateNode { concept: op_concept, node_type } => {
                assert_eq!(op_concept, concept);
                assert_eq!(node_type, *direction);
            },
            _ => panic!("Expected CreateNode operation for direction {:?}", direction),
        }
    }
}

#[test]
fn test_create_node_various_concepts() {
    let concepts = vec![
        test_constants::TEST_CONCEPT_INPUT,
        test_constants::TEST_CONCEPT_OUTPUT,
        test_constants::TEST_CONCEPT_GATE,
        test_constants::TEST_CONCEPT_HIDDEN,
        "custom_concept_123",
        "very_long_concept_name_with_special_chars_!@#",
        "", // Empty concept
        "单个字符", // Unicode characters
    ];
    
    for concept in concepts {
        let operation = create_node_operation(concept, EntityDirection::Input);
        
        match operation {
            GraphOperation::CreateNode { concept: op_concept, .. } => {
                assert_eq!(op_concept, concept);
            },
            _ => panic!("Expected CreateNode operation"),
        }
    }
}

// ==================== Logic Gate Creation Operation Tests ====================

#[test]
fn test_create_logic_gate_operation() {
    let inputs = vec!["input1", "input2"];
    let outputs = vec!["output1"];
    let gate_type = LogicGateType::And;
    
    let operation = create_gate_operation(inputs.clone(), outputs.clone(), gate_type);
    
    match operation {
        GraphOperation::CreateLogicGate { inputs: op_inputs, outputs: op_outputs, gate_type: op_type } => {
            let expected_inputs: Vec<String> = inputs.into_iter().map(|s| s.to_string()).collect();
            let expected_outputs: Vec<String> = outputs.into_iter().map(|s| s.to_string()).collect();
            
            assert_eq!(op_inputs, expected_inputs);
            assert_eq!(op_outputs, expected_outputs);
            assert_eq!(op_type, gate_type);
        },
        _ => panic!("Expected CreateLogicGate operation"),
    }
}

#[test]
fn test_create_logic_gate_all_types() {
    let gate_types = vec![
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
    
    for gate_type in gate_types {
        let inputs = vec!["input1", "input2"];
        let outputs = vec!["output1"];
        
        let operation = create_gate_operation(inputs, outputs, gate_type);
        
        match operation {
            GraphOperation::CreateLogicGate { gate_type: op_type, .. } => {
                assert_eq!(op_type, gate_type);
            },
            _ => panic!("Expected CreateLogicGate operation for {:?}", gate_type),
        }
    }
}

#[test]
fn test_create_logic_gate_various_configurations() {
    let test_configs = vec![
        // (inputs, outputs, expected_input_count, expected_output_count)
        (vec!["single_input"], vec!["single_output"], 1, 1),
        (vec!["in1", "in2"], vec!["out1"], 2, 1),
        (vec!["in1", "in2", "in3"], vec!["out1", "out2"], 3, 2),
        (vec![], vec!["out1"], 0, 1), // No inputs
        (vec!["in1"], vec![], 1, 0), // No outputs
        (vec![], vec![], 0, 0), // No inputs or outputs
    ];
    
    for (inputs, outputs, expected_in_count, expected_out_count) in test_configs {
        let operation = create_gate_operation(inputs, outputs, LogicGateType::And);
        
        match operation {
            GraphOperation::CreateLogicGate { inputs: op_inputs, outputs: op_outputs, .. } => {
                assert_eq!(op_inputs.len(), expected_in_count);
                assert_eq!(op_outputs.len(), expected_out_count);
            },
            _ => panic!("Expected CreateLogicGate operation"),
        }
    }
}

#[test]
fn test_create_logic_gate_long_names() {
    let long_inputs = vec![
        "very_long_input_name_that_exceeds_normal_length_expectations",
        "another_extremely_long_input_name_with_many_underscores_and_descriptive_text",
    ];
    let long_outputs = vec![
        "similarly_long_output_name_for_comprehensive_testing_purposes",
    ];
    
    let operation = create_gate_operation(long_inputs.clone(), long_outputs.clone(), LogicGateType::Or);
    
    match operation {
        GraphOperation::CreateLogicGate { inputs, outputs, .. } => {
            let expected_inputs: Vec<String> = long_inputs.into_iter().map(|s| s.to_string()).collect();
            let expected_outputs: Vec<String> = long_outputs.into_iter().map(|s| s.to_string()).collect();
            
            assert_eq!(inputs, expected_inputs);
            assert_eq!(outputs, expected_outputs);
        },
        _ => panic!("Expected CreateLogicGate operation"),
    }
}

// ==================== Relationship Creation Operation Tests ====================

#[test]
fn test_create_relationship_operation() {
    let source = test_constants::TEST_CONCEPT_INPUT;
    let target = test_constants::TEST_CONCEPT_OUTPUT;
    let relation_type = RelationType::RelatedTo;
    let weight = test_constants::STRONG_EXCITATORY;
    
    let operation = create_relationship_operation(source, target, relation_type, weight);
    
    match operation {
        GraphOperation::CreateRelationship { source: op_source, target: op_target, relation_type: op_type, weight: op_weight } => {
            assert_eq!(op_source, source);
            assert_eq!(op_target, target);
            assert_eq!(op_type, relation_type);
            assert_eq!(op_weight, weight);
        },
        _ => panic!("Expected CreateRelationship operation"),
    }
}

#[test]
fn test_create_relationship_all_types() {
    let relation_types = vec![
        RelationType::IsA,
        RelationType::HasInstance,
        RelationType::HasProperty,
        RelationType::RelatedTo,
        RelationType::PartOf,
        RelationType::Similar,
        RelationType::Opposite,
        RelationType::Temporal,
        RelationType::Learned,
    ];
    
    for relation_type in relation_types {
        let operation = create_relationship_operation(
            "source",
            "target", 
            relation_type,
            test_constants::MEDIUM_EXCITATORY
        );
        
        match operation {
            GraphOperation::CreateRelationship { relation_type: op_type, .. } => {
                assert_eq!(op_type, relation_type);
            },
            _ => panic!("Expected CreateRelationship operation for {:?}", relation_type),
        }
    }
}

#[test]
fn test_create_relationship_various_weights() {
    let weights = vec![
        0.0,
        test_constants::WEAK_EXCITATORY,
        test_constants::MEDIUM_EXCITATORY,
        test_constants::STRONG_EXCITATORY,
        1.0,
        test_constants::ABOVE_SATURATION,
        -0.5, // Negative weight
        f32::MAX,
        f32::MIN,
    ];
    
    for weight in weights {
        let operation = create_relationship_operation(
            "source",
            "target",
            RelationType::RelatedTo,
            weight
        );
        
        match operation {
            GraphOperation::CreateRelationship { weight: op_weight, .. } => {
                assert_eq!(op_weight, weight);
            },
            _ => panic!("Expected CreateRelationship operation"),
        }
    }
}

#[test]
fn test_create_relationship_edge_case_names() {
    let edge_case_names = vec![
        ("", "target"), // Empty source
        ("source", ""), // Empty target
        ("", ""), // Both empty
        ("source with spaces", "target with spaces"),
        ("source_with_underscores", "target_with_underscores"),
        ("source-with-dashes", "target-with-dashes"),
        ("source123", "target456"), // With numbers
        ("UPPERCASE_SOURCE", "UPPERCASE_TARGET"),
        ("mixedCaseSource", "mixedCaseTarget"),
    ];
    
    for (source, target) in edge_case_names {
        let operation = create_relationship_operation(
            source,
            target,
            RelationType::RelatedTo,
            test_constants::MEDIUM_EXCITATORY
        );
        
        match operation {
            GraphOperation::CreateRelationship { source: op_source, target: op_target, .. } => {
                assert_eq!(op_source, source);
                assert_eq!(op_target, target);
            },
            _ => panic!("Expected CreateRelationship operation"),
        }
    }
}

// ==================== Serialization Tests ====================

#[test]
fn test_graph_operation_serialization() {
    let operations = vec![
        create_node_operation(test_constants::TEST_CONCEPT_INPUT, EntityDirection::Input),
        create_gate_operation(
            vec!["input1", "input2"],
            vec!["output1"],
            LogicGateType::And
        ),
        create_relationship_operation(
            "source",
            "target",
            RelationType::RelatedTo,
            test_constants::STRONG_EXCITATORY
        ),
    ];
    
    for operation in operations {
        let serialized = serde_json::to_string(&operation)
            .expect("Should serialize");
        
        // Should contain operation type discriminant
        assert!(serialized.len() > 10, "Serialized form should not be empty");
        
        // Should be valid JSON
        let _: serde_json::Value = serde_json::from_str(&serialized)
            .expect("Should be valid JSON");
    }
}

#[test]
fn test_graph_operation_deserialization() {
    let operations = vec![
        create_node_operation(test_constants::TEST_CONCEPT_OUTPUT, EntityDirection::Output),
        create_gate_operation(
            vec!["in1", "in2", "in3"],
            vec!["out1", "out2"],
            LogicGateType::Threshold
        ),
        create_relationship_operation(
            "entity1",
            "entity2",
            RelationType::IsA,
            test_constants::WEAK_EXCITATORY
        ),
    ];
    
    for original in operations {
        let serialized = serde_json::to_string(&original)
            .expect("Should serialize");
        
        let deserialized: GraphOperation = serde_json::from_str(&serialized)
            .expect("Should deserialize");
        
        // Verify round-trip correctness
        match (&original, &deserialized) {
            (
                GraphOperation::CreateNode { concept: c1, node_type: t1 },
                GraphOperation::CreateNode { concept: c2, node_type: t2 }
            ) => {
                assert_eq!(c1, c2);
                assert_eq!(t1, t2);
            },
            (
                GraphOperation::CreateLogicGate { inputs: i1, outputs: o1, gate_type: g1 },
                GraphOperation::CreateLogicGate { inputs: i2, outputs: o2, gate_type: g2 }
            ) => {
                assert_eq!(i1, i2);
                assert_eq!(o1, o2);
                assert_eq!(g1, g2);
            },
            (
                GraphOperation::CreateRelationship { source: s1, target: t1, relation_type: r1, weight: w1 },
                GraphOperation::CreateRelationship { source: s2, target: t2, relation_type: r2, weight: w2 }
            ) => {
                assert_eq!(s1, s2);
                assert_eq!(t1, t2);
                assert_eq!(r1, r2);
                assert_float_eq(*w1, *w2, test_constants::ACTIVATION_EPSILON);
            },
            _ => panic!("Operation types don't match after round-trip"),
        }
    }
}

#[test]
fn test_graph_operation_json_structure() {
    let node_op = create_node_operation("test_node", EntityDirection::Gate);
    let serialized = serde_json::to_string(&node_op).unwrap();
    
    // Should contain expected fields
    assert!(serialized.contains("CreateNode"));
    assert!(serialized.contains("concept"));
    assert!(serialized.contains("node_type"));
    assert!(serialized.contains("test_node"));
    assert!(serialized.contains("Gate"));
    
    let gate_op = create_gate_operation(
        vec!["input"],
        vec!["output"],
        LogicGateType::Not
    );
    let gate_serialized = serde_json::to_string(&gate_op).unwrap();
    
    assert!(gate_serialized.contains("CreateLogicGate"));
    assert!(gate_serialized.contains("inputs"));
    assert!(gate_serialized.contains("outputs"));
    assert!(gate_serialized.contains("gate_type"));
    assert!(gate_serialized.contains("Not"));
    
    let rel_op = create_relationship_operation(
        "src",
        "tgt",
        RelationType::Temporal,
        0.75
    );
    let rel_serialized = serde_json::to_string(&rel_op).unwrap();
    
    assert!(rel_serialized.contains("CreateRelationship"));
    assert!(rel_serialized.contains("source"));
    assert!(rel_serialized.contains("target"));
    assert!(rel_serialized.contains("relation_type"));
    assert!(rel_serialized.contains("weight"));
    assert!(rel_serialized.contains("Temporal"));
    assert!(rel_serialized.contains("0.75"));
}

// ==================== Complex Operation Tests ====================

#[test]
fn test_complex_graph_operations() {
    // Create a complex operation sequence
    let operations = vec![
        create_node_operation("input_layer_1", EntityDirection::Input),
        create_node_operation("input_layer_2", EntityDirection::Input),
        create_node_operation("hidden_layer_1", EntityDirection::Hidden),
        create_gate_operation(
            vec!["input_layer_1", "input_layer_2"],
            vec!["gate_output"],
            LogicGateType::And
        ),
        create_relationship_operation(
            "gate_output",
            "hidden_layer_1",
            RelationType::Learned,
            test_constants::STRONG_EXCITATORY
        ),
        create_node_operation("output_layer", EntityDirection::Output),
        create_relationship_operation(
            "hidden_layer_1",
            "output_layer",
            RelationType::RelatedTo,
            test_constants::MEDIUM_EXCITATORY
        ),
    ];
    
    // Should be able to serialize/deserialize entire sequence
    let serialized = serde_json::to_string(&operations).unwrap();
    let deserialized: Vec<GraphOperation> = serde_json::from_str(&serialized).unwrap();
    
    assert_eq!(operations.len(), deserialized.len());
    
    // Verify each operation type is preserved
    let node_count = operations.iter()
        .filter(|op| matches!(op, GraphOperation::CreateNode { .. }))
        .count();
    let gate_count = operations.iter()
        .filter(|op| matches!(op, GraphOperation::CreateLogicGate { .. }))
        .count();
    let rel_count = operations.iter()
        .filter(|op| matches!(op, GraphOperation::CreateRelationship { .. }))
        .count();
    
    let deserialized_node_count = deserialized.iter()
        .filter(|op| matches!(op, GraphOperation::CreateNode { .. }))
        .count();
    let deserialized_gate_count = deserialized.iter()
        .filter(|op| matches!(op, GraphOperation::CreateLogicGate { .. }))
        .count();
    let deserialized_rel_count = deserialized.iter()
        .filter(|op| matches!(op, GraphOperation::CreateRelationship { .. }))
        .count();
    
    assert_eq!(node_count, deserialized_node_count);
    assert_eq!(gate_count, deserialized_gate_count);
    assert_eq!(rel_count, deserialized_rel_count);
}

// ==================== Debug and Display Tests ====================

#[test]
fn test_graph_operation_debug() {
    let operations = vec![
        create_node_operation("debug_node", EntityDirection::Hidden),
        create_gate_operation(vec!["in"], vec!["out"], LogicGateType::Identity),
        create_relationship_operation("a", "b", RelationType::PartOf, 0.5),
    ];
    
    for operation in operations {
        let debug_str = format!("{:?}", operation);
        
        // Should contain operation variant name
        assert!(
            debug_str.contains("CreateNode") ||
            debug_str.contains("CreateLogicGate") ||
            debug_str.contains("CreateRelationship"),
            "Debug output should contain operation type: {}",
            debug_str
        );
        
        // Should not be empty
        assert!(debug_str.len() > 10, "Debug output should be substantial");
    }
}

#[test]
fn test_graph_operation_clone() {
    let original = create_relationship_operation(
        "original_source",
        "original_target",
        RelationType::Similar,
        test_constants::MEDIUM_EXCITATORY
    );
    
    let cloned = original.clone();
    
    // Should be equal after cloning
    match (&original, &cloned) {
        (
            GraphOperation::CreateRelationship { source: s1, target: t1, relation_type: r1, weight: w1 },
            GraphOperation::CreateRelationship { source: s2, target: t2, relation_type: r2, weight: w2 }
        ) => {
            assert_eq!(s1, s2);
            assert_eq!(t1, t2);
            assert_eq!(r1, r2);
            assert_eq!(w1, w2);
        },
        _ => panic!("Clone should preserve operation type and content"),
    }
}

// ==================== Memory and Performance Tests ====================

#[test]
fn test_graph_operation_memory_usage() {
    use std::mem;
    
    let operations = vec![
        create_node_operation("memory_test", EntityDirection::Input),
        create_gate_operation(vec!["input"], vec!["output"], LogicGateType::And),
        create_relationship_operation("src", "tgt", RelationType::RelatedTo, 1.0),
    ];
    
    for operation in operations {
        let size = mem::size_of_val(&operation);
        
        // Should be reasonably sized
        assert!(
            size < 200,
            "GraphOperation should be compact, got {} bytes",
            size
        );
    }
}

#[test]
fn test_large_operation_collection_performance() {
    let num_operations = 10000;
    let mut operations = Vec::new();
    
    let start_time = std::time::Instant::now();
    
    // Create many operations
    for i in 0..num_operations {
        let operation = match i % 3 {
            0 => create_node_operation(&format!("node_{}", i), EntityDirection::Input),
            1 => create_gate_operation(
                vec![&format!("input_{}", i)],
                vec![&format!("output_{}", i)],
                LogicGateType::Or
            ),
            _ => create_relationship_operation(
                &format!("source_{}", i),
                &format!("target_{}", i),
                RelationType::RelatedTo,
                (i as f32) / (num_operations as f32)
            ),
        };
        operations.push(operation);
    }
    
    let creation_time = start_time.elapsed();
    
    // Should create operations quickly
    assert!(
        creation_time.as_millis() < 100,
        "Creating {} operations took too long: {:?}",
        num_operations, creation_time
    );
    
    // Test serialization performance
    let serialize_start = std::time::Instant::now();
    let _serialized = serde_json::to_string(&operations).unwrap();
    let serialize_time = serialize_start.elapsed();
    
    assert!(
        serialize_time.as_millis() < 1000,
        "Serializing {} operations took too long: {:?}",
        num_operations, serialize_time
    );
}