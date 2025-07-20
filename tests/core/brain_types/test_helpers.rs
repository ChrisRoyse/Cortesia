// Test helpers for brain_types tests
// Utility functions and builders for creating test scenarios

use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, BrainInspiredRelationship, RelationType,
    LogicGate, LogicGateType, ActivationPattern, ActivationStep, ActivationOperation,
    GraphOperation, TrainingExample
};
use llmkg::core::types::EntityKey;
use std::collections::HashMap;
use std::time::SystemTime;

use super::test_constants;

// ==================== Entity Builders ====================

/// Builder for creating test entities with fluent API
pub struct EntityBuilder {
    concept_id: String,
    direction: EntityDirection,
    properties: HashMap<String, llmkg::core::types::AttributeValue>,
    embedding: Vec<f32>,
    activation_state: f32,
}

impl EntityBuilder {
    pub fn new(concept_id: &str, direction: EntityDirection) -> Self {
        Self {
            concept_id: concept_id.to_string(),
            direction,
            properties: HashMap::new(),
            embedding: Vec::new(),
            activation_state: 0.0,
        }
    }

    pub fn with_property(mut self, key: &str, value: llmkg::core::types::AttributeValue) -> Self {
        self.properties.insert(key.to_string(), value);
        self
    }

    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }

    pub fn with_activation(mut self, activation: f32) -> Self {
        self.activation_state = activation;
        self
    }

    pub fn build(self) -> BrainInspiredEntity {
        let mut entity = BrainInspiredEntity::new(self.concept_id, self.direction);
        entity.properties = self.properties;
        entity.embedding = self.embedding;
        entity.activation_state = self.activation_state;
        entity
    }
}

// ==================== Relationship Builders ====================

/// Builder for creating test relationships
pub struct RelationshipBuilder {
    source: EntityKey,
    target: EntityKey,
    relation_type: RelationType,
    weight: f32,
    is_inhibitory: bool,
    temporal_decay: f32,
    metadata: HashMap<String, String>,
}

impl RelationshipBuilder {
    pub fn new(source: EntityKey, target: EntityKey, relation_type: RelationType) -> Self {
        Self {
            source,
            target,
            relation_type,
            weight: 1.0,
            is_inhibitory: false,
            temporal_decay: test_constants::STANDARD_DECAY_RATE,
            metadata: HashMap::new(),
        }
    }

    pub fn with_weight(mut self, weight: f32) -> Self {
        self.weight = weight;
        self
    }

    pub fn inhibitory(mut self) -> Self {
        self.is_inhibitory = true;
        self
    }

    pub fn with_decay(mut self, decay_rate: f32) -> Self {
        self.temporal_decay = decay_rate;
        self
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    pub fn build(self) -> BrainInspiredRelationship {
        let mut relationship = BrainInspiredRelationship::new(self.source, self.target, self.relation_type);
        relationship.weight = self.weight;
        relationship.strength = self.weight;
        relationship.is_inhibitory = self.is_inhibitory;
        relationship.temporal_decay = self.temporal_decay;
        relationship.metadata = self.metadata;
        relationship
    }
}

// ==================== Logic Gate Builders ====================

/// Builder for creating test logic gates
pub struct LogicGateBuilder {
    gate_type: LogicGateType,
    threshold: f32,
    input_nodes: Vec<EntityKey>,
    output_nodes: Vec<EntityKey>,
    weight_matrix: Vec<f32>,
}

impl LogicGateBuilder {
    pub fn new(gate_type: LogicGateType) -> Self {
        let threshold = match gate_type {
            LogicGateType::And => test_constants::AND_GATE_THRESHOLD,
            LogicGateType::Or => test_constants::OR_GATE_THRESHOLD,
            LogicGateType::Threshold => test_constants::THRESHOLD_GATE_LIMIT,
            LogicGateType::Weighted => test_constants::WEIGHTED_GATE_THRESHOLD,
            LogicGateType::Inhibitory => test_constants::INHIBITORY_GATE_THRESHOLD,
            _ => 0.5,
        };

        Self {
            gate_type,
            threshold,
            input_nodes: Vec::new(),
            output_nodes: Vec::new(),
            weight_matrix: Vec::new(),
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_inputs(mut self, inputs: Vec<EntityKey>) -> Self {
        self.input_nodes = inputs;
        self
    }

    pub fn with_outputs(mut self, outputs: Vec<EntityKey>) -> Self {
        self.output_nodes = outputs;
        self
    }

    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        self.weight_matrix = weights;
        self
    }

    pub fn build(self) -> LogicGate {
        let mut gate = LogicGate::new(self.gate_type, self.threshold);
        gate.input_nodes = self.input_nodes;
        gate.output_nodes = self.output_nodes;
        gate.weight_matrix = self.weight_matrix;
        gate
    }
}

// ==================== Activation Pattern Helpers ====================

/// Create a simple activation pattern for testing
pub fn create_test_pattern(query: &str, size: usize) -> ActivationPattern {
    let mut pattern = ActivationPattern::new(query.to_string());
    
    for i in 0..size {
        let key = EntityKey::from(i as u64);
        let activation = (i as f32 + 1.0) / (size as f32 + 1.0); // Normalized activation
        pattern.activations.insert(key, activation);
    }
    
    pattern
}

/// Create pattern with specific activations
pub fn create_pattern_with_activations(query: &str, activations: Vec<(u64, f32)>) -> ActivationPattern {
    let mut pattern = ActivationPattern::new(query.to_string());
    
    for (key_val, activation) in activations {
        pattern.activations.insert(EntityKey::from(key_val), activation);
    }
    
    pattern
}

// ==================== Activation Step Helpers ====================

/// Create a test activation step
pub fn create_activation_step(
    step_id: usize,
    entity_key: EntityKey,
    concept_id: &str,
    activation_level: f32,
    operation_type: ActivationOperation
) -> ActivationStep {
    ActivationStep {
        step_id,
        entity_key,
        concept_id: concept_id.to_string(),
        activation_level,
        operation_type,
        timestamp: SystemTime::now(),
    }
}

// ==================== Graph Operation Helpers ====================

/// Create a test node creation operation
pub fn create_node_operation(concept: &str, node_type: EntityDirection) -> GraphOperation {
    GraphOperation::CreateNode {
        concept: concept.to_string(),
        node_type,
    }
}

/// Create a test logic gate operation
pub fn create_gate_operation(
    inputs: Vec<&str>,
    outputs: Vec<&str>,
    gate_type: LogicGateType
) -> GraphOperation {
    GraphOperation::CreateLogicGate {
        inputs: inputs.into_iter().map(|s| s.to_string()).collect(),
        outputs: outputs.into_iter().map(|s| s.to_string()).collect(),
        gate_type,
    }
}

/// Create a test relationship operation
pub fn create_relationship_operation(
    source: &str,
    target: &str,
    relation_type: RelationType,
    weight: f32
) -> GraphOperation {
    GraphOperation::CreateRelationship {
        source: source.to_string(),
        target: target.to_string(),
        relation_type,
        weight,
    }
}

// ==================== Training Example Helpers ====================

/// Create a test training example
pub fn create_training_example(
    text: &str,
    operations: Vec<GraphOperation>,
    metadata: Vec<(&str, &str)>
) -> TrainingExample {
    let metadata_map: HashMap<String, String> = metadata
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

    TrainingExample {
        text: text.to_string(),
        expected_operations: operations,
        metadata: metadata_map,
    }
}

// ==================== Test Entity Collections ====================

/// Create a set of test entities for multi-entity tests
pub fn create_test_entities() -> Vec<BrainInspiredEntity> {
    vec![
        EntityBuilder::new(test_constants::TEST_CONCEPT_INPUT, EntityDirection::Input)
            .with_activation(test_constants::ACTION_POTENTIAL)
            .build(),
        EntityBuilder::new(test_constants::TEST_CONCEPT_OUTPUT, EntityDirection::Output)
            .with_activation(test_constants::RESTING_POTENTIAL)
            .build(),
        EntityBuilder::new(test_constants::TEST_CONCEPT_GATE, EntityDirection::Gate)
            .with_activation(test_constants::THRESHOLD_POTENTIAL)
            .build(),
        EntityBuilder::new(test_constants::TEST_CONCEPT_HIDDEN, EntityDirection::Hidden)
            .with_activation(test_constants::RESTING_POTENTIAL)
            .build(),
    ]
}

/// Create a set of test relationships connecting entities
pub fn create_test_relationships(entities: &[BrainInspiredEntity]) -> Vec<BrainInspiredRelationship> {
    if entities.len() < 4 {
        return Vec::new();
    }

    vec![
        RelationshipBuilder::new(entities[0].id, entities[1].id, RelationType::RelatedTo)
            .with_weight(test_constants::STRONG_EXCITATORY)
            .build(),
        RelationshipBuilder::new(entities[1].id, entities[2].id, RelationType::IsA)
            .with_weight(test_constants::MEDIUM_EXCITATORY)
            .build(),
        RelationshipBuilder::new(entities[2].id, entities[3].id, RelationType::HasProperty)
            .inhibitory()
            .with_weight(test_constants::MEDIUM_INHIBITORY)
            .build(),
    ]
}

// ==================== Assertion Helpers ====================

/// Assert activation is within expected range with tolerance
pub fn assert_activation_close(actual: f32, expected: f32, tolerance: f32, context: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{}: Expected activation {}, got {}, tolerance {}",
        context, expected, actual, tolerance
    );
}

/// Assert entity has expected direction
pub fn assert_entity_direction(entity: &BrainInspiredEntity, expected: EntityDirection, context: &str) {
    assert_eq!(
        entity.direction, expected,
        "{}: Expected direction {:?}, got {:?}",
        context, expected, entity.direction
    );
}

/// Assert relationship has expected properties
pub fn assert_relationship_properties(
    rel: &BrainInspiredRelationship,
    expected_weight: f32,
    expected_inhibitory: bool,
    context: &str
) {
    assert_activation_close(rel.weight, expected_weight, test_constants::ACTIVATION_EPSILON, context);
    assert_eq!(
        rel.is_inhibitory, expected_inhibitory,
        "{}: Expected inhibitory {}, got {}",
        context, expected_inhibitory, rel.is_inhibitory
    );
}

// ==================== Timing Helpers ====================

/// Measure execution time of an operation
pub fn measure_execution_time<F, T>(operation: F) -> (T, std::time::Duration)
where
    F: FnOnce() -> T,
{
    let start = std::time::Instant::now();
    let result = operation();
    let duration = start.elapsed();
    (result, duration)
}

/// Assert operation completes within time limit
pub fn assert_within_time_limit<F, T>(
    operation: F,
    max_duration: std::time::Duration,
    operation_name: &str
) -> T
where
    F: FnOnce() -> T,
{
    let (result, duration) = measure_execution_time(operation);
    assert!(
        duration <= max_duration,
        "{} took {:?}, exceeding limit of {:?}",
        operation_name, duration, max_duration
    );
    result
}