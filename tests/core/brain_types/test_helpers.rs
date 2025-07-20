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

/// Create a test logic gate with specified parameters
pub fn create_test_gate(gate_type: LogicGateType, threshold: f32, num_inputs: usize) -> LogicGate {
    let inputs: Vec<EntityKey> = (0..num_inputs).map(|i| EntityKey::from(slotmap::KeyData::from_ffi(i as u64))).collect();
    let outputs = vec![EntityKey::from(slotmap::KeyData::from_ffi(num_inputs as u64))];
    let weights = vec![1.0; num_inputs];
    
    LogicGateBuilder::new(gate_type)
        .with_threshold(threshold)
        .with_inputs(inputs)
        .with_outputs(outputs)
        .with_weights(weights)
        .build()
}

// ==================== Activation Pattern Helpers ====================

/// Create a simple activation pattern for testing
pub fn create_test_pattern(query: &str, size: usize) -> ActivationPattern {
    let mut pattern = ActivationPattern::new(query.to_string());
    
    for i in 0..size {
        let key = EntityKey::from(slotmap::KeyData::from_ffi(i as u64));
        let activation = (i as f32 + 1.0) / (size as f32 + 1.0); // Normalized activation
        pattern.activations.insert(key, activation);
    }
    
    pattern
}

/// Create pattern with specific activations
pub fn create_pattern_with_activations(query: &str, activations: Vec<(u64, f32)>) -> ActivationPattern {
    let mut pattern = ActivationPattern::new(query.to_string());
    
    for (key_val, activation) in activations {
        pattern.activations.insert(EntityKey::from(slotmap::KeyData::from_ffi(key_val)), activation);
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

// ==================== Additional Helper Functions ====================

/// Create a test relationship with specific parameters
pub fn create_test_relationship(
    relation_type: RelationType,
    weight: f32,
    is_inhibitory: bool,
    temporal_decay: f32
) -> BrainInspiredRelationship {
    let source = EntityKey::default();
    let target = EntityKey::default();
    let mut rel = BrainInspiredRelationship::new(source, target, relation_type);
    rel.weight = weight;
    rel.strength = weight;
    rel.is_inhibitory = is_inhibitory;
    rel.temporal_decay = temporal_decay;
    rel
}

/// Assert floating point equality with tolerance
pub fn assert_float_eq(actual: f32, expected: f32, tolerance: f32) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "Expected {}, got {}, tolerance {}",
        expected, actual, tolerance
    );
}

/// Assert activation value is valid (typically between 0.0 and 1.0 but allows flexibility)
pub fn assert_valid_activation(activation: f32) {
    assert!(
        !activation.is_nan() && !activation.is_infinite(),
        "Activation should be a valid number, got {}",
        activation
    );
}

/// Generate test data for property-based testing
pub fn generate_valid_activations(count: usize) -> Vec<f32> {
    (0..count).map(|i| (i as f32) / (count as f32)).collect()
}

/// Generate test data for learning rates
pub fn generate_learning_rates() -> Vec<f32> {
    vec![
        0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0
    ]
}

/// Generate test data for decay rates
pub fn generate_decay_rates() -> Vec<f32> {
    vec![
        0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0
    ]
}

/// Generate edge case activation values for testing
pub fn generate_edge_case_activations() -> Vec<f32> {
    vec![
        f32::NEG_INFINITY,
        -1000.0,
        -1.0,
        -0.1,
        0.0,
        0.1,
        0.5,
        0.9,
        1.0,
        1.1,
        2.0,
        1000.0,
        f32::INFINITY,
        f32::NAN,
    ]
}

/// Create weight matrix with different characteristics
pub fn create_test_weight_matrices() -> Vec<Vec<f32>> {
    vec![
        vec![], // Empty
        vec![1.0], // Single weight
        vec![0.5, 0.5], // Balanced
        vec![0.8, 0.2], // Unbalanced
        vec![0.33, 0.33, 0.34], // Three weights
        vec![0.1, 0.2, 0.3, 0.4], // Four weights
        vec![0.0, 0.0, 0.0], // All zeros
        vec![1.0, 1.0, 1.0], // All ones
        vec![-0.5, 0.5, 1.0], // Including negative
        vec![f32::NAN, 0.5, 0.3], // Including NaN
    ]
}

/// Create test input combinations for logic gates
pub fn create_gate_test_inputs() -> Vec<Vec<f32>> {
    vec![
        vec![], // Empty inputs
        vec![0.0], // Single zero
        vec![1.0], // Single one
        vec![0.5], // Single middle
        vec![0.0, 0.0], // Both zero
        vec![0.0, 1.0], // Mixed
        vec![1.0, 0.0], // Mixed reverse
        vec![1.0, 1.0], // Both one
        vec![0.5, 0.5], // Both middle
        vec![0.3, 0.7, 0.5], // Three inputs
        vec![f32::NAN, 0.5], // With NaN
        vec![f32::INFINITY, 0.5], // With infinity
        vec![-0.5, 0.5], // With negative
    ]
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

// ==================== Property-Based Testing Helpers ====================

/// Generate boundary activation values for comprehensive testing
pub fn boundary_activation_values() -> Vec<f32> {
    vec![
        0.0,              // Minimum activation
        0.001,            // Very small positive
        0.1,              // Low activation
        0.25,             // Quarter activation
        0.5,              // Half activation
        0.75,             // Three-quarter activation
        0.9,              // High activation
        0.999,            // Very high activation
        1.0,              // Maximum activation
    ]
}

/// Generate comprehensive test thresholds
pub fn threshold_test_values() -> Vec<f32> {
    vec![
        0.0,              // Zero threshold
        0.001,            // Very low threshold
        0.1,              // Low threshold
        0.3,              // Low-medium threshold
        0.5,              // Medium threshold
        0.7,              // Medium-high threshold
        0.9,              // High threshold
        1.0,              // Maximum threshold
        1.5,              // Above maximum threshold
        2.0,              // Well above maximum
    ]
}

/// Generate test weight matrices with various characteristics
pub fn weight_matrix_variants(input_count: usize) -> Vec<Vec<f32>> {
    let mut variants = Vec::new();
    
    // Uniform weights
    variants.push(vec![1.0 / input_count as f32; input_count]);
    
    // All ones
    variants.push(vec![1.0; input_count]);
    
    // All zeros
    variants.push(vec![0.0; input_count]);
    
    // Decreasing weights
    variants.push((0..input_count).map(|i| 1.0 - (i as f32 / input_count as f32)).collect());
    
    // Increasing weights
    variants.push((0..input_count).map(|i| (i + 1) as f32 / input_count as f32).collect());
    
    // Random-like weights (deterministic for testing)
    variants.push((0..input_count).map(|i| ((i * 37 + 17) % 100) as f32 / 100.0).collect());
    
    // Single dominant weight
    let mut single_dominant = vec![0.1; input_count];
    if input_count > 0 {
        single_dominant[0] = 0.9;
    }
    variants.push(single_dominant);
    
    // Alternating weights
    variants.push((0..input_count).map(|i| if i % 2 == 0 { 0.8 } else { 0.2 }).collect());
    
    variants
}

/// Generate comprehensive input combinations for property-based testing
pub fn input_combinations(input_count: usize) -> Vec<Vec<f32>> {
    let mut combinations = Vec::new();
    let boundary_values = boundary_activation_values();
    
    // All same values
    for &value in &boundary_values {
        combinations.push(vec![value; input_count]);
    }
    
    // All zeros
    combinations.push(vec![0.0; input_count]);
    
    // All ones
    combinations.push(vec![1.0; input_count]);
    
    // Mixed values
    if input_count >= 2 {
        combinations.push(vec![0.0, 1.0].into_iter().cycle().take(input_count).collect());
        combinations.push(vec![1.0, 0.0].into_iter().cycle().take(input_count).collect());
        combinations.push(vec![0.5, 0.3, 0.8].into_iter().cycle().take(input_count).collect());
    }
    
    // Gradual increase
    combinations.push((0..input_count).map(|i| i as f32 / input_count.max(1) as f32).collect());
    
    // Gradual decrease
    combinations.push((0..input_count).map(|i| 1.0 - (i as f32 / input_count.max(1) as f32)).collect());
    
    combinations
}

/// Generate edge case scenarios for testing
pub fn edge_case_scenarios() -> Vec<(&'static str, Vec<f32>)> {
    vec![
        ("all_zero", vec![0.0, 0.0, 0.0]),
        ("all_max", vec![1.0, 1.0, 1.0]),
        ("mixed_extreme", vec![0.0, 1.0, 0.0]),
        ("very_small", vec![0.001, 0.001, 0.001]),
        ("near_threshold", vec![0.499, 0.501, 0.5]),
        ("above_saturation", vec![1.1, 1.2, 1.5]),
        ("negative_values", vec![-0.1, -0.5, -1.0]),
        ("large_values", vec![10.0, 100.0, 1000.0]),
    ]
}

/// Property: All gate outputs should be non-negative (except for special float values)
pub fn property_non_negative_output<F>(gate_fn: F, inputs: &[f32]) -> bool
where
    F: Fn(&[f32]) -> llmkg::error::Result<f32>,
{
    match gate_fn(inputs) {
        Ok(output) => output >= 0.0 || output.is_nan(),
        Err(_) => true, // Errors are acceptable for invalid inputs
    }
}

/// Property: Gate outputs should be deterministic (same inputs = same outputs)
pub fn property_deterministic_output<F>(gate_fn: F, inputs: &[f32]) -> bool
where
    F: Fn(&[f32]) -> llmkg::error::Result<f32>,
{
    let result1 = gate_fn(inputs);
    let result2 = gate_fn(inputs);
    
    match (result1, result2) {
        (Ok(out1), Ok(out2)) => {
            if out1.is_nan() && out2.is_nan() {
                true // Both NaN is consistent
            } else {
                (out1 - out2).abs() < f32::EPSILON
            }
        },
        (Err(_), Err(_)) => true, // Both errors is consistent
        _ => false, // Inconsistent results
    }
}

/// Property: Monotonicity for certain gate types (AND, OR)
pub fn property_monotonic_and(inputs1: &[f32], inputs2: &[f32], gate_fn: impl Fn(&[f32]) -> llmkg::error::Result<f32>) -> bool {
    if inputs1.len() != inputs2.len() {
        return true; // Skip if different lengths
    }
    
    // Check if inputs1 <= inputs2 element-wise
    let all_leq = inputs1.iter().zip(inputs2.iter()).all(|(a, b)| a <= b);
    
    if !all_leq {
        return true; // Property doesn't apply
    }
    
    match (gate_fn(inputs1), gate_fn(inputs2)) {
        (Ok(out1), Ok(out2)) => out1 <= out2 || out1.is_nan() || out2.is_nan(),
        _ => true, // Skip if either fails
    }
}

// ==================== Performance Testing Helpers ====================

/// Benchmark a function with multiple iterations
pub fn benchmark_function<F, T>(
    operation: F,
    iterations: usize,
    operation_name: &str
) -> (T, std::time::Duration)
where
    F: Fn() -> T,
    T: Clone,
{
    let start = std::time::Instant::now();
    let mut result = None;
    
    for _ in 0..iterations {
        result = Some(operation());
    }
    
    let total_duration = start.elapsed();
    let avg_duration = total_duration / iterations as u32;
    
    println!("{}: {} iterations, avg {:?}/op", operation_name, iterations, avg_duration);
    
    (result.unwrap(), total_duration)
}

/// Memory usage estimation helper
pub fn estimate_memory_usage<T>() -> usize {
    std::mem::size_of::<T>()
}

/// Create performance test dataset
pub fn create_performance_dataset(size: usize) -> Vec<f32> {
    (0..size).map(|i| {
        let val = (i as f32 * 1.618034) % 1.0; // Golden ratio for pseudo-random distribution
        val
    }).collect()
}

// ==================== Fuzzing Helpers ====================

/// Simple deterministic "random" number generator for testing
pub struct TestRng {
    state: u64,
}

impl TestRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    
    pub fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state / 65536) % 32768) as f32 / 32768.0
    }
    
    pub fn next_bool(&mut self) -> bool {
        self.next_f32() > 0.5
    }
    
    pub fn next_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}

/// Generate fuzz test inputs
pub fn generate_fuzz_inputs(rng: &mut TestRng, count: usize) -> Vec<f32> {
    (0..count).map(|_| {
        match rng.next_f32() {
            x if x < 0.1 => 0.0,                    // 10% zeros
            x if x < 0.2 => 1.0,                    // 10% ones
            x if x < 0.3 => rng.next_f32(),         // 10% normal range
            x if x < 0.4 => -rng.next_f32(),        // 10% negative
            x if x < 0.5 => rng.next_range(1.0, 10.0), // 10% above normal
            x if x < 0.6 => f32::NAN,               // 10% NaN
            x if x < 0.7 => f32::INFINITY,          // 10% infinity
            x if x < 0.8 => f32::NEG_INFINITY,      // 10% negative infinity
            _ => rng.next_range(0.0, 1.0),          // 20% normal range
        }
    }).collect()
}

// ==================== Test Data Validation ====================

/// Validate that test data produces expected results
pub fn validate_test_data() -> bool {
    // Validate boundary values
    let boundaries = boundary_activation_values();
    assert_eq!(boundaries.len(), 9);
    assert_eq!(boundaries[0], 0.0);
    assert_eq!(boundaries[8], 1.0);
    
    // Validate threshold values
    let thresholds = threshold_test_values();
    assert_eq!(thresholds.len(), 10);
    assert_eq!(thresholds[0], 0.0);
    
    // Validate weight matrices
    let weights = weight_matrix_variants(3);
    assert!(weights.len() >= 6);
    assert_eq!(weights[0].len(), 3); // Uniform weights
    
    true
}

// ==================== Assertion Macros ====================

/// Assert that a value is within a specific range
#[macro_export]
macro_rules! assert_in_range {
    ($value:expr, $min:expr, $max:expr) => {
        assert!(
            $value >= $min && $value <= $max,
            "Value {} not in range [{}, {}]",
            $value, $min, $max
        );
    };
    ($value:expr, $min:expr, $max:expr, $msg:expr) => {
        assert!(
            $value >= $min && $value <= $max,
            "{}: Value {} not in range [{}, {}]",
            $msg, $value, $min, $max
        );
    };
}

/// Assert that a value is approximately equal with custom tolerance
#[macro_export]
macro_rules! assert_approx_eq {
    ($left:expr, $right:expr, $tolerance:expr) => {
        assert!(
            ($left - $right).abs() <= $tolerance,
            "Values not approximately equal: {} != {} (tolerance: {})",
            $left, $right, $tolerance
        );
    };
    ($left:expr, $right:expr, $tolerance:expr, $msg:expr) => {
        assert!(
            ($left - $right).abs() <= $tolerance,
            "{}: Values not approximately equal: {} != {} (tolerance: {})",
            $msg, $left, $right, $tolerance
        );
    };
}