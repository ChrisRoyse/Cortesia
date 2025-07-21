//! Integration tests for brain_types module
//! Tests public API, serialization/deserialization, and cross-component compatibility

use llmkg::core::brain_types::*;
use llmkg::core::types::{EntityKey, AttributeValue};
use serde_json;
use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use std::thread;

/// Helper function to create test EntityKeys
fn create_test_key(id: u64) -> EntityKey {
    EntityKey::from(slotmap::KeyData::from_ffi(id))
}

/// Helper function to create a sample BrainInspiredEntity
fn create_sample_entity(id: &str, direction: EntityDirection) -> BrainInspiredEntity {
    let mut entity = BrainInspiredEntity::new(id.to_string(), direction);
    entity.id = create_test_key(1);
    entity.properties.insert("type".to_string(), AttributeValue::String("concept".to_string()));
    entity.properties.insert("confidence".to_string(), AttributeValue::Number(0.85));
    entity.embedding = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    entity
}

#[test]
fn test_entity_direction_serialization_deserialization() {
    // Test all EntityDirection variants
    let directions = vec![
        EntityDirection::Input,
        EntityDirection::Output,
        EntityDirection::Gate,
        EntityDirection::Hidden,
    ];

    for direction in directions {
        // Serialize to JSON
        let json = serde_json::to_string(&direction).expect("Failed to serialize EntityDirection");
        
        // Deserialize back
        let deserialized: EntityDirection = serde_json::from_str(&json)
            .expect("Failed to deserialize EntityDirection");
        
        assert_eq!(direction, deserialized);
    }
}

#[test]
fn test_logic_gate_type_serialization_deserialization() {
    // Test all LogicGateType variants
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
        // Serialize to JSON
        let json = serde_json::to_string(&gate_type).expect("Failed to serialize LogicGateType");
        
        // Deserialize back
        let deserialized: LogicGateType = serde_json::from_str(&json)
            .expect("Failed to deserialize LogicGateType");
        
        assert_eq!(gate_type, deserialized);
        
        // Test Display trait
        let display_str = format!("{}", gate_type);
        assert!(!display_str.is_empty());
    }
}

#[test]
fn test_relation_type_serialization_deserialization() {
    // Test all RelationType variants
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
        // Serialize to JSON
        let json = serde_json::to_string(&relation_type).expect("Failed to serialize RelationType");
        
        // Deserialize back
        let deserialized: RelationType = serde_json::from_str(&json)
            .expect("Failed to deserialize RelationType");
        
        assert_eq!(relation_type, deserialized);
    }
}

#[test]
fn test_brain_inspired_entity_full_serialization_workflow() {
    // Create entity with all fields populated
    let mut entity = create_sample_entity("test_concept", EntityDirection::Input);
    entity.activation_state = 0.75;
    
    // Add complex properties
    entity.properties.insert("nested".to_string(), AttributeValue::Object({
        let mut map = HashMap::new();
        map.insert("key1".to_string(), AttributeValue::String("value1".to_string()));
        map.insert("key2".to_string(), AttributeValue::Number(42.0));
        map
    }));
    
    entity.properties.insert("array".to_string(), AttributeValue::Array(vec![
        AttributeValue::Number(1.0),
        AttributeValue::Number(2.0),
        AttributeValue::Number(3.0),
    ]));

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&entity)
        .expect("Failed to serialize BrainInspiredEntity");
    
    // Deserialize back
    let deserialized: BrainInspiredEntity = serde_json::from_str(&json)
        .expect("Failed to deserialize BrainInspiredEntity");
    
    // Verify all fields
    assert_eq!(entity.id, deserialized.id);
    assert_eq!(entity.concept_id, deserialized.concept_id);
    assert_eq!(entity.direction, deserialized.direction);
    assert_eq!(entity.properties, deserialized.properties);
    assert_eq!(entity.embedding, deserialized.embedding);
    assert_eq!(entity.activation_state, deserialized.activation_state);
}

#[test]
fn test_logic_gate_full_serialization_workflow() {
    // Create a complete logic gate
    let mut gate = LogicGate::new(LogicGateType::Weighted, 0.7);
    gate.gate_id = create_test_key(100);
    gate.input_nodes = vec![create_test_key(1), create_test_key(2), create_test_key(3)];
    gate.output_nodes = vec![create_test_key(4), create_test_key(5)];
    gate.weight_matrix = vec![0.5, 0.8, 0.3];

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&gate)
        .expect("Failed to serialize LogicGate");
    
    // Deserialize back
    let deserialized: LogicGate = serde_json::from_str(&json)
        .expect("Failed to deserialize LogicGate");
    
    // Verify all fields
    assert_eq!(gate.gate_id, deserialized.gate_id);
    assert_eq!(gate.gate_type, deserialized.gate_type);
    assert_eq!(gate.input_nodes, deserialized.input_nodes);
    assert_eq!(gate.output_nodes, deserialized.output_nodes);
    assert_eq!(gate.threshold, deserialized.threshold);
    assert_eq!(gate.weight_matrix, deserialized.weight_matrix);
}

#[test]
fn test_brain_inspired_relationship_full_serialization_workflow() {
    // Create a complete relationship
    let mut rel = BrainInspiredRelationship::new(
        create_test_key(1),
        create_test_key(2),
        RelationType::HasProperty
    );
    
    rel.weight = 0.8;
    rel.strength = 0.8;
    rel.is_inhibitory = true;
    rel.temporal_decay = 0.05;
    rel.activation_count = 42;
    rel.usage_count = 42;
    rel.metadata.insert("source_type".to_string(), "concept".to_string());
    rel.metadata.insert("target_type".to_string(), "property".to_string());

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&rel)
        .expect("Failed to serialize BrainInspiredRelationship");
    
    // Deserialize back
    let deserialized: BrainInspiredRelationship = serde_json::from_str(&json)
        .expect("Failed to deserialize BrainInspiredRelationship");
    
    // Verify all fields
    assert_eq!(rel.source, deserialized.source);
    assert_eq!(rel.target, deserialized.target);
    assert_eq!(rel.source_key, deserialized.source_key);
    assert_eq!(rel.target_key, deserialized.target_key);
    assert_eq!(rel.relation_type, deserialized.relation_type);
    assert_eq!(rel.weight, deserialized.weight);
    assert_eq!(rel.strength, deserialized.strength);
    assert_eq!(rel.is_inhibitory, deserialized.is_inhibitory);
    assert_eq!(rel.temporal_decay, deserialized.temporal_decay);
    assert_eq!(rel.activation_count, deserialized.activation_count);
    assert_eq!(rel.usage_count, deserialized.usage_count);
    assert_eq!(rel.metadata, deserialized.metadata);
}

#[test]
fn test_graph_operation_serialization_workflow() {
    let operations = vec![
        GraphOperation::CreateNode {
            concept: "test_concept".to_string(),
            node_type: EntityDirection::Input,
        },
        GraphOperation::CreateLogicGate {
            inputs: vec!["input1".to_string(), "input2".to_string()],
            outputs: vec!["output1".to_string()],
            gate_type: LogicGateType::And,
        },
        GraphOperation::CreateRelationship {
            source: "entity1".to_string(),
            target: "entity2".to_string(),
            relation_type: RelationType::RelatedTo,
            weight: 0.75,
        },
    ];

    for op in operations {
        // Serialize to JSON
        let json = serde_json::to_string(&op).expect("Failed to serialize GraphOperation");
        
        // Deserialize back
        let deserialized: GraphOperation = serde_json::from_str(&json)
            .expect("Failed to deserialize GraphOperation");
        
        // Pattern match to verify
        match (op, deserialized) {
            (
                GraphOperation::CreateNode { concept: c1, node_type: n1 },
                GraphOperation::CreateNode { concept: c2, node_type: n2 }
            ) => {
                assert_eq!(c1, c2);
                assert_eq!(n1, n2);
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
                assert!((w1 - w2).abs() < f32::EPSILON);
            },
            _ => panic!("Operation type mismatch after deserialization"),
        }
    }
}

#[test]
fn test_training_example_serialization_workflow() {
    let mut example = TrainingExample {
        text: "The cat is a mammal".to_string(),
        expected_operations: vec![
            GraphOperation::CreateNode {
                concept: "cat".to_string(),
                node_type: EntityDirection::Input,
            },
            GraphOperation::CreateNode {
                concept: "mammal".to_string(),
                node_type: EntityDirection::Output,
            },
            GraphOperation::CreateRelationship {
                source: "cat".to_string(),
                target: "mammal".to_string(),
                relation_type: RelationType::IsA,
                weight: 1.0,
            },
        ],
        metadata: HashMap::new(),
    };
    
    example.metadata.insert("source".to_string(), "biology".to_string());
    example.metadata.insert("confidence".to_string(), "high".to_string());

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&example)
        .expect("Failed to serialize TrainingExample");
    
    // Deserialize back
    let deserialized: TrainingExample = serde_json::from_str(&json)
        .expect("Failed to deserialize TrainingExample");
    
    assert_eq!(example.text, deserialized.text);
    assert_eq!(example.expected_operations.len(), deserialized.expected_operations.len());
    assert_eq!(example.metadata, deserialized.metadata);
}

#[test]
fn test_activation_pattern_serialization_workflow() {
    let mut pattern = ActivationPattern::new("find related concepts".to_string());
    pattern.activations.insert(create_test_key(1), 0.9);
    pattern.activations.insert(create_test_key(2), 0.7);
    pattern.activations.insert(create_test_key(3), 0.5);

    // Serialize to JSON
    let json = serde_json::to_string(&pattern)
        .expect("Failed to serialize ActivationPattern");
    
    // Deserialize back
    let deserialized: ActivationPattern = serde_json::from_str(&json)
        .expect("Failed to deserialize ActivationPattern");
    
    assert_eq!(pattern.query, deserialized.query);
    assert_eq!(pattern.activations.len(), deserialized.activations.len());
    
    // Verify top activations work after deserialization
    let top = deserialized.get_top_activations(2);
    assert_eq!(top.len(), 2);
    assert!(top[0].1 >= top[1].1); // Should be sorted
}

#[test]
fn test_activation_step_serialization_workflow() {
    let step = ActivationStep {
        step_id: 42,
        entity_key: create_test_key(100),
        concept_id: "neural_network".to_string(),
        activation_level: 0.85,
        operation_type: ActivationOperation::Propagate,
        timestamp: SystemTime::now(),
    };

    // Test all ActivationOperation variants
    let operations = vec![
        ActivationOperation::Initialize,
        ActivationOperation::Propagate,
        ActivationOperation::Inhibit,
        ActivationOperation::Reinforce,
        ActivationOperation::Decay,
    ];

    for op in operations {
        let mut test_step = step.clone();
        test_step.operation_type = op;
        
        // Serialize to JSON
        let json = serde_json::to_string(&test_step)
            .expect("Failed to serialize ActivationStep");
        
        // Deserialize back
        let deserialized: ActivationStep = serde_json::from_str(&json)
            .expect("Failed to deserialize ActivationStep");
        
        assert_eq!(test_step.step_id, deserialized.step_id);
        assert_eq!(test_step.entity_key, deserialized.entity_key);
        assert_eq!(test_step.concept_id, deserialized.concept_id);
        assert_eq!(test_step.activation_level, deserialized.activation_level);
        assert_eq!(test_step.operation_type, deserialized.operation_type);
    }
}

#[test]
fn test_entity_activation_workflow_end_to_end() {
    let mut entity = BrainInspiredEntity::new("neuron".to_string(), EntityDirection::Hidden);
    
    // Initial state
    assert_eq!(entity.activation_state, 0.0);
    
    // First activation
    let result1 = entity.activate(0.5, 0.1);
    assert!((result1 - 0.5).abs() < 0.01);
    
    // Immediate second activation (no decay)
    let result2 = entity.activate(0.3, 0.1);
    assert!(result2 > 0.7); // Should accumulate
    assert!(result2 <= 1.0); // Should be capped
    
    // Wait a bit to test decay
    thread::sleep(Duration::from_millis(10));
    
    // Third activation with decay
    let result3 = entity.activate(0.2, 5.0); // High decay rate
    assert!(result3 < result2); // Should show some decay effect
}

#[test]
fn test_logic_gate_computation_workflow() {
    // Test AND gate workflow
    let mut and_gate = LogicGate::new(LogicGateType::And, 0.5);
    and_gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
    
    let output = and_gate.calculate_output(&[0.8, 0.7]).expect("AND calculation failed");
    assert!((output - 0.7).abs() < 0.01);
    
    // Test OR gate workflow
    let mut or_gate = LogicGate::new(LogicGateType::Or, 0.5);
    or_gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
    
    let output = or_gate.calculate_output(&[0.3, 0.7]).expect("OR calculation failed");
    assert!((output - 0.7).abs() < 0.01);
    
    // Test weighted gate workflow
    let mut weighted_gate = LogicGate::new(LogicGateType::Weighted, 0.8);
    weighted_gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
    weighted_gate.weight_matrix = vec![0.6, 0.4];
    
    let output = weighted_gate.calculate_output(&[1.0, 1.0]).expect("Weighted calculation failed");
    assert_eq!(output, 1.0); // 0.6 + 0.4 = 1.0, meets threshold
}

#[test]
fn test_relationship_strengthening_workflow() {
    let mut rel = BrainInspiredRelationship::new(
        create_test_key(1),
        create_test_key(2),
        RelationType::Learned
    );
    
    // Initial state
    assert_eq!(rel.weight, 1.0);
    assert_eq!(rel.activation_count, 0);
    
    // Strengthen multiple times
    for i in 0..5 {
        rel.strengthen(0.1);
        assert_eq!(rel.activation_count, i + 1);
        assert_eq!(rel.usage_count, i + 1);
    }
    
    // Weight should remain at 1.0 (capped)
    assert_eq!(rel.weight, 1.0);
    assert_eq!(rel.strength, 1.0);
    
    // Test decay
    thread::sleep(Duration::from_millis(10));
    let decayed_weight = rel.apply_decay();
    assert!(decayed_weight < 1.0);
    assert_eq!(rel.weight, decayed_weight);
}

#[test]
fn test_cross_component_compatibility() {
    // Create a mini knowledge graph with entities, relationships, and gates
    let entity1 = create_sample_entity("input_concept", EntityDirection::Input);
    let entity2 = create_sample_entity("output_concept", EntityDirection::Output);
    
    let mut gate = LogicGate::new(LogicGateType::And, 0.5);
    gate.input_nodes = vec![entity1.id];
    gate.output_nodes = vec![entity2.id];
    
    let relationship = BrainInspiredRelationship::new(
        entity1.id,
        entity2.id,
        RelationType::RelatedTo
    );
    
    // Create activation pattern
    let mut pattern = ActivationPattern::new("test query".to_string());
    pattern.activations.insert(entity1.id, 0.8);
    pattern.activations.insert(entity2.id, 0.0);
    
    // Simulate propagation through gate
    let gate_output = gate.calculate_output(&[0.8]).expect("Gate calculation failed");
    pattern.activations.insert(entity2.id, gate_output);
    
    // Create activation steps for tracing
    let steps = vec![
        ActivationStep {
            step_id: 1,
            entity_key: entity1.id,
            concept_id: entity1.concept_id.clone(),
            activation_level: 0.8,
            operation_type: ActivationOperation::Initialize,
            timestamp: SystemTime::now(),
        },
        ActivationStep {
            step_id: 2,
            entity_key: entity2.id,
            concept_id: entity2.concept_id.clone(),
            activation_level: gate_output,
            operation_type: ActivationOperation::Propagate,
            timestamp: SystemTime::now(),
        },
    ];
    
    // Serialize entire workflow
    let workflow_data = serde_json::json!({
        "entities": vec![entity1, entity2],
        "gates": vec![gate],
        "relationships": vec![relationship],
        "activation_pattern": pattern,
        "activation_steps": steps,
    });
    
    let json = serde_json::to_string_pretty(&workflow_data)
        .expect("Failed to serialize workflow");
    
    // Verify JSON is valid
    let _parsed: serde_json::Value = serde_json::from_str(&json)
        .expect("Failed to parse workflow JSON");
}

#[test]
fn test_data_consistency_across_types() {
    // Test that duplicate fields in BrainInspiredRelationship remain consistent
    let mut rel = BrainInspiredRelationship::new(
        create_test_key(10),
        create_test_key(20),
        RelationType::HasProperty
    );
    
    // Verify initial consistency
    assert_eq!(rel.source, rel.source_key);
    assert_eq!(rel.target, rel.target_key);
    assert_eq!(rel.weight, rel.strength);
    assert_eq!(rel.activation_count, rel.usage_count);
    
    // Test after strengthening
    rel.strengthen(0.2);
    assert_eq!(rel.weight, rel.strength);
    assert_eq!(rel.activation_count, rel.usage_count);
    
    // Test after decay
    rel.apply_decay();
    assert_eq!(rel.weight, rel.strength);
}

#[test]
fn test_temporal_tracking_consistency() {
    let entity = BrainInspiredEntity::new("temporal_test".to_string(), EntityDirection::Hidden);
    let relationship = BrainInspiredRelationship::new(
        create_test_key(1),
        create_test_key(2),
        RelationType::Temporal
    );
    
    // Verify all temporal fields are initialized
    assert!(entity.last_activation <= SystemTime::now());
    assert!(entity.last_update <= SystemTime::now());
    assert!(relationship.last_strengthened <= SystemTime::now());
    assert!(relationship.last_update <= SystemTime::now());
    assert!(relationship.creation_time <= SystemTime::now());
    assert!(relationship.ingestion_time <= SystemTime::now());
}

#[test]
fn test_complex_property_handling() {
    let mut entity = BrainInspiredEntity::new("complex_props".to_string(), EntityDirection::Input);
    
    // Add various property types
    entity.properties.insert("string_prop".to_string(), 
        AttributeValue::String("test_value".to_string()));
    entity.properties.insert("number_prop".to_string(), 
        AttributeValue::Number(42.5));
    entity.properties.insert("bool_prop".to_string(), 
        AttributeValue::Boolean(true));
    entity.properties.insert("null_prop".to_string(), 
        AttributeValue::Null);
    entity.properties.insert("vector_prop".to_string(), 
        AttributeValue::Vector(vec![0.1, 0.2, 0.3]));
    
    // Complex nested structure
    let mut nested_obj = HashMap::new();
    nested_obj.insert("nested_key".to_string(), 
        AttributeValue::Array(vec![
            AttributeValue::Number(1.0),
            AttributeValue::String("nested".to_string()),
            AttributeValue::Boolean(false),
        ]));
    entity.properties.insert("object_prop".to_string(), 
        AttributeValue::Object(nested_obj));
    
    // Serialize and deserialize
    let json = serde_json::to_string(&entity).expect("Failed to serialize");
    let deserialized: BrainInspiredEntity = serde_json::from_str(&json)
        .expect("Failed to deserialize");
    
    // Verify all properties preserved
    assert_eq!(entity.properties.len(), deserialized.properties.len());
    for (key, value) in &entity.properties {
        assert_eq!(value, deserialized.properties.get(key).unwrap());
    }
}

#[test]
fn test_error_handling_in_gate_calculations() {
    // Test various error conditions through public API
    let mut gate = LogicGate::new(LogicGateType::Not, 0.5);
    gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
    
    // NOT gate with multiple inputs should error
    assert!(gate.calculate_output(&[0.5, 0.5]).is_err());
    
    // Mismatched input count
    let mut and_gate = LogicGate::new(LogicGateType::And, 0.5);
    and_gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
    assert!(and_gate.calculate_output(&[0.5]).is_err());
    
    // Weighted gate without matching weights
    let mut weighted_gate = LogicGate::new(LogicGateType::Weighted, 0.5);
    weighted_gate.input_nodes = vec![create_test_key(1), create_test_key(2)];
    weighted_gate.weight_matrix = vec![0.5]; // Too few weights
    assert!(weighted_gate.calculate_output(&[0.5, 0.5]).is_err());
}

#[test]
fn test_activation_pattern_ordering_stability() {
    let mut pattern = ActivationPattern::new("test".to_string());
    
    // Add activations in specific order
    let keys = vec![
        (create_test_key(1), 0.3),
        (create_test_key(2), 0.9),
        (create_test_key(3), 0.6),
        (create_test_key(4), 0.8),
        (create_test_key(5), 0.1),
    ];
    
    for (key, value) in keys {
        pattern.activations.insert(key, value);
    }
    
    // Get top activations multiple times
    let top1 = pattern.get_top_activations(3);
    let top2 = pattern.get_top_activations(3);
    let top3 = pattern.get_top_activations(3);
    
    // Verify consistent ordering
    assert_eq!(top1, top2);
    assert_eq!(top2, top3);
    
    // Verify correct order (descending by activation)
    assert_eq!(top1[0].1, 0.9);
    assert_eq!(top1[1].1, 0.8);
    assert_eq!(top1[2].1, 0.6);
}

#[test]
fn test_binary_serialization_compatibility() {
    // Test that types can be serialized to binary format (e.g., for storage)
    let entity = create_sample_entity("binary_test", EntityDirection::Gate);
    let gate = LogicGate::new(LogicGateType::Xor, 0.5);
    let rel = BrainInspiredRelationship::new(
        create_test_key(1),
        create_test_key(2),
        RelationType::Similar
    );
    
    // Serialize to binary using bincode
    let entity_bytes = bincode::serialize(&entity).expect("Failed to serialize entity to binary");
    let gate_bytes = bincode::serialize(&gate).expect("Failed to serialize gate to binary");
    let rel_bytes = bincode::serialize(&rel).expect("Failed to serialize relationship to binary");
    
    // Deserialize back
    let entity_restored: BrainInspiredEntity = bincode::deserialize(&entity_bytes)
        .expect("Failed to deserialize entity from binary");
    let gate_restored: LogicGate = bincode::deserialize(&gate_bytes)
        .expect("Failed to deserialize gate from binary");
    let rel_restored: BrainInspiredRelationship = bincode::deserialize(&rel_bytes)
        .expect("Failed to deserialize relationship from binary");
    
    // Verify core fields
    assert_eq!(entity.concept_id, entity_restored.concept_id);
    assert_eq!(gate.gate_type, gate_restored.gate_type);
    assert_eq!(rel.relation_type, rel_restored.relation_type);
}

#[test]
fn test_complete_neural_network_workflow() {
    // Create a small neural network: Input -> Gate -> Output
    let mut input1 = BrainInspiredEntity::new("sensory_input_1".to_string(), EntityDirection::Input);
    let mut input2 = BrainInspiredEntity::new("sensory_input_2".to_string(), EntityDirection::Input);
    let mut output = BrainInspiredEntity::new("motor_output".to_string(), EntityDirection::Output);
    
    input1.id = create_test_key(1);
    input2.id = create_test_key(2);
    output.id = create_test_key(3);
    
    // Create AND gate connecting inputs to output
    let mut gate = LogicGate::new(LogicGateType::And, 0.6);
    gate.gate_id = create_test_key(100);
    gate.input_nodes = vec![input1.id, input2.id];
    gate.output_nodes = vec![output.id];
    
    // Create relationships
    let rel1 = BrainInspiredRelationship::new(input1.id, gate.gate_id, RelationType::PartOf);
    let rel2 = BrainInspiredRelationship::new(input2.id, gate.gate_id, RelationType::PartOf);
    let rel3 = BrainInspiredRelationship::new(gate.gate_id, output.id, RelationType::PartOf);
    
    // Simulate activation propagation
    input1.activate(0.8, 0.1);
    input2.activate(0.7, 0.1);
    
    // Calculate gate output
    let gate_activation = gate.calculate_output(&[
        input1.activation_state,
        input2.activation_state
    ]).expect("Gate calculation failed");
    
    // Propagate to output
    output.activate(gate_activation, 0.1);
    
    // Create activation pattern to track the flow
    let mut pattern = ActivationPattern::new("neural_propagation".to_string());
    pattern.activations.insert(input1.id, input1.activation_state);
    pattern.activations.insert(input2.id, input2.activation_state);
    pattern.activations.insert(gate.gate_id, gate_activation);
    pattern.activations.insert(output.id, output.activation_state);
    
    // Verify the propagation worked correctly
    assert!(input1.activation_state > 0.0);
    assert!(input2.activation_state > 0.0);
    assert!(gate_activation > 0.0); // AND gate with both inputs above threshold
    assert_eq!(output.activation_state, gate_activation);
    
    // Test serialization of complete network
    let network = serde_json::json!({
        "entities": vec![input1, input2, output],
        "gates": vec![gate],
        "relationships": vec![rel1, rel2, rel3],
        "activation_pattern": pattern,
    });
    
    let json = serde_json::to_string_pretty(&network)
        .expect("Failed to serialize network");
    
    // Verify we can parse it back
    let _parsed: serde_json::Value = serde_json::from_str(&json)
        .expect("Failed to parse network JSON");
}

#[test]
fn test_inhibitory_network_workflow() {
    // Create network with inhibitory connections
    let mut excitatory = BrainInspiredEntity::new("excitatory_neuron".to_string(), EntityDirection::Hidden);
    let mut inhibitory = BrainInspiredEntity::new("inhibitory_neuron".to_string(), EntityDirection::Hidden);
    let mut target = BrainInspiredEntity::new("target_neuron".to_string(), EntityDirection::Output);
    
    excitatory.id = create_test_key(1);
    inhibitory.id = create_test_key(2);
    target.id = create_test_key(3);
    
    // Create inhibitory gate
    let mut gate = LogicGate::new(LogicGateType::Inhibitory, 0.0);
    gate.input_nodes = vec![excitatory.id, inhibitory.id];
    gate.output_nodes = vec![target.id];
    
    // Create relationships with inhibitory flag
    let mut inhibitory_rel = BrainInspiredRelationship::new(
        inhibitory.id,
        target.id,
        RelationType::RelatedTo
    );
    inhibitory_rel.is_inhibitory = true;
    
    // Test inhibition: high excitation, higher inhibition
    excitatory.activate(0.8, 0.0);
    inhibitory.activate(0.9, 0.0);
    
    let gate_output = gate.calculate_output(&[
        excitatory.activation_state,
        inhibitory.activation_state
    ]).expect("Inhibitory gate failed");
    
    // Output should be suppressed (0.8 - 0.9 = -0.1, clamped to 0.0)
    assert_eq!(gate_output, 0.0);
    
    // Test with lower inhibition
    inhibitory.activation_state = 0.3;
    let gate_output2 = gate.calculate_output(&[
        excitatory.activation_state,
        inhibitory.activation_state
    ]).expect("Inhibitory gate failed");
    
    // Output should be positive (0.8 - 0.3 = 0.5)
    assert!((gate_output2 - 0.5).abs() < 0.01);
}

#[test]
fn test_learning_and_decay_over_time() {
    let mut rel = BrainInspiredRelationship::new(
        create_test_key(1),
        create_test_key(2),
        RelationType::Learned
    );
    
    // Initial strengthening
    rel.weight = 0.5;
    rel.strength = 0.5;
    
    // Simulate learning over multiple cycles
    let mut weights = vec![rel.weight];
    
    for _ in 0..3 {
        rel.strengthen(0.15);
        weights.push(rel.weight);
        thread::sleep(Duration::from_millis(5));
    }
    
    // Verify learning increased weight
    assert!(weights[3] > weights[0]);
    
    // Now test decay
    thread::sleep(Duration::from_millis(20));
    let decayed = rel.apply_decay();
    
    // Weight should have decreased
    assert!(decayed < weights[3]);
}

#[test] 
fn test_type_validation_through_public_api() {
    // Test that public constructors enforce valid states
    
    // Entity with empty concept_id is allowed but might not be meaningful
    let entity = BrainInspiredEntity::new("".to_string(), EntityDirection::Input);
    assert_eq!(entity.concept_id, "");
    
    // Gate with negative threshold is allowed
    let gate = LogicGate::new(LogicGateType::Threshold, -1.0);
    assert_eq!(gate.threshold, -1.0);
    
    // Relationship with same source and target is allowed
    let same_key = create_test_key(1);
    let rel = BrainInspiredRelationship::new(same_key, same_key, RelationType::Similar);
    assert_eq!(rel.source, rel.target);
    
    // Very large decay rate
    let mut rel2 = BrainInspiredRelationship::new(
        create_test_key(1),
        create_test_key(2),
        RelationType::Temporal
    );
    rel2.temporal_decay = f32::MAX;
    let result = rel2.apply_decay();
    assert_eq!(result, 0.0); // Should decay to zero immediately
}