// Test helpers and builders for activation engine tests

use llmkg::core::activation_config::ActivationConfig;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, BrainInspiredRelationship, 
    RelationType, ActivationPattern, LogicGate, LogicGateType
};
use llmkg::core::types::EntityKey;
use std::collections::HashMap;

// Import test constants from the same module
use super::test_constants;

// ==================== Test Network Builder ====================

/// Builder for creating test neural networks with fluent API
pub struct TestNetworkBuilder {
    config: ActivationConfig,
    entities: Vec<(String, BrainInspiredEntity)>,
    relationships: Vec<BrainInspiredRelationship>,
    logic_gates: Vec<LogicGate>,
}

impl TestNetworkBuilder {
    /// Create a new test network builder
    pub fn new() -> Self {
        Self {
            config: ActivationConfig::default(),
            entities: Vec::new(),
            relationships: Vec::new(),
            logic_gates: Vec::new(),
        }
    }

    /// Set custom configuration
    pub fn with_config(mut self, config: ActivationConfig) -> Self {
        self.config = config;
        self
    }

    /// Add an input neuron
    pub fn add_input_neuron(mut self, name: &str) -> Self {
        let entity = BrainInspiredEntity::new(name.to_string(), EntityDirection::Input);
        self.entities.push((name.to_string(), entity));
        self
    }

    /// Add a hidden neuron
    pub fn add_hidden_neuron(mut self, name: &str) -> Self {
        let entity = BrainInspiredEntity::new(name.to_string(), EntityDirection::Hidden);
        self.entities.push((name.to_string(), entity));
        self
    }

    /// Add an output neuron
    pub fn add_output_neuron(mut self, name: &str) -> Self {
        let entity = BrainInspiredEntity::new(name.to_string(), EntityDirection::Output);
        self.entities.push((name.to_string(), entity));
        self
    }

    /// Add a gate neuron
    pub fn add_gate_neuron(mut self, name: &str) -> Self {
        let entity = BrainInspiredEntity::new(name.to_string(), EntityDirection::Gate);
        self.entities.push((name.to_string(), entity));
        self
    }

    /// Connect two neurons with excitatory connection
    pub fn connect(mut self, from: &str, to: &str, weight: f32) -> Self {
        let from_key = self.get_entity_key(from);
        let to_key = self.get_entity_key(to);
        
        let mut rel = BrainInspiredRelationship::new(from_key, to_key, RelationType::RelatedTo);
        rel.weight = weight;
        self.relationships.push(rel);
        self
    }

    /// Connect two neurons with inhibitory connection
    pub fn connect_inhibitory(mut self, from: &str, to: &str, weight: f32) -> Self {
        let from_key = self.get_entity_key(from);
        let to_key = self.get_entity_key(to);
        
        let mut rel = BrainInspiredRelationship::new(from_key, to_key, RelationType::RelatedTo);
        rel.weight = weight;
        rel.is_inhibitory = true;
        self.relationships.push(rel);
        self
    }

    /// Add an AND gate
    pub fn add_and_gate(mut self, inputs: Vec<&str>, outputs: Vec<&str>, threshold: f32) -> Self {
        let mut gate = LogicGate::new(LogicGateType::And, threshold);
        
        for input in inputs {
            gate.input_nodes.push(self.get_entity_key(input));
        }
        
        for output in outputs {
            gate.output_nodes.push(self.get_entity_key(output));
        }
        
        self.logic_gates.push(gate);
        self
    }

    /// Add an OR gate
    pub fn add_or_gate(mut self, inputs: Vec<&str>, outputs: Vec<&str>, threshold: f32) -> Self {
        let mut gate = LogicGate::new(LogicGateType::Or, threshold);
        
        for input in inputs {
            gate.input_nodes.push(self.get_entity_key(input));
        }
        
        for output in outputs {
            gate.output_nodes.push(self.get_entity_key(output));
        }
        
        self.logic_gates.push(gate);
        self
    }

    /// Build the network and return engine and entity map
    pub async fn build(self) -> (ActivationPropagationEngine, HashMap<String, EntityKey>) {
        let engine = ActivationPropagationEngine::new(self.config);
        let mut entity_map = HashMap::new();

        // Add all entities
        for (name, entity) in &self.entities {
            entity_map.insert(name.clone(), entity.id);
            engine.add_entity(entity.clone()).await.unwrap();
        }

        // Add all relationships
        for rel in self.relationships {
            engine.add_relationship(rel).await.unwrap();
        }

        // Add all logic gates
        for gate in self.logic_gates {
            engine.add_logic_gate(gate).await.unwrap();
        }

        (engine, entity_map)
    }

    /// Helper to get entity key by name
    fn get_entity_key(&self, name: &str) -> EntityKey {
        self.entities.iter()
            .find(|(n, _)| n == name)
            .map(|(_, e)| e.id)
            .expect(&format!("Entity '{}' not found", name))
    }
}

// ==================== Common Test Networks ====================

/// Create a simple linear chain: A -> B -> C
pub async fn create_linear_chain() -> (ActivationPropagationEngine, HashMap<String, EntityKey>) {
    TestNetworkBuilder::new()
        .add_input_neuron("sensory_input")
        .add_hidden_neuron("interneuron")
        .add_output_neuron("motor_output")
        .connect("sensory_input", "interneuron", super::test_constants::MEDIUM_WEIGHT)
        .connect("interneuron", "motor_output", super::test_constants::MEDIUM_WEIGHT)
        .build()
        .await
}

/// Create a network with lateral inhibition
pub async fn create_lateral_inhibition_network() -> (ActivationPropagationEngine, HashMap<String, EntityKey>) {
    TestNetworkBuilder::new()
        .add_input_neuron("input_a")
        .add_input_neuron("input_b")
        .add_output_neuron("output_a")
        .add_output_neuron("output_b")
        .connect("input_a", "output_a", super::test_constants::STRONG_WEIGHT)
        .connect("input_b", "output_b", super::test_constants::STRONG_WEIGHT)
        .connect_inhibitory("output_a", "output_b", super::test_constants::MEDIUM_WEIGHT)
        .connect_inhibitory("output_b", "output_a", super::test_constants::MEDIUM_WEIGHT)
        .build()
        .await
}

/// Create a convergent network: multiple inputs to single output
pub async fn create_convergent_network() -> (ActivationPropagationEngine, HashMap<String, EntityKey>) {
    TestNetworkBuilder::new()
        .add_input_neuron("visual_input")
        .add_input_neuron("auditory_input")
        .add_input_neuron("tactile_input")
        .add_hidden_neuron("integration_neuron")
        .add_output_neuron("decision_neuron")
        .connect("visual_input", "integration_neuron", super::test_constants::MEDIUM_WEIGHT)
        .connect("auditory_input", "integration_neuron", super::test_constants::MEDIUM_WEIGHT)
        .connect("tactile_input", "integration_neuron", super::test_constants::WEAK_WEIGHT)
        .connect("integration_neuron", "decision_neuron", super::test_constants::STRONG_WEIGHT)
        .build()
        .await
}

// ==================== Activation Pattern Builders ====================

/// Create an activation pattern with specified neurons activated
pub fn create_activation_pattern(name: &str, activations: Vec<(&str, f32)>, entity_map: &HashMap<String, EntityKey>) -> ActivationPattern {
    let mut pattern = ActivationPattern::new(name.to_string());
    
    for (neuron_name, activation) in activations {
        if let Some(&key) = entity_map.get(neuron_name) {
            pattern.activations.insert(key, activation);
        }
    }
    
    pattern
}

// ==================== Assertion Helpers ====================

/// Assert activation is within expected range with descriptive message
pub fn assert_activation_in_range(
    actual: f32, 
    min: f32, 
    max: f32, 
    neuron_name: &str,
    context: &str
) {
    assert!(
        actual >= min && actual <= max,
        "{}: Neuron '{}' activation {} is outside expected range [{}, {}]",
        context, neuron_name, actual, min, max
    );
}

/// Assert activation equals expected value within epsilon
pub fn assert_activation_equals(
    actual: f32,
    expected: f32,
    neuron_name: &str,
    context: &str
) {
    assert!(
        (actual - expected).abs() < super::test_constants::EPSILON,
        "{}: Neuron '{}' activation {} differs from expected {} by more than epsilon",
        context, neuron_name, actual, expected
    );
}

/// Assert that a neuron was activated (non-zero)
pub fn assert_neuron_activated(
    activations: &HashMap<EntityKey, f32>,
    key: &EntityKey,
    neuron_name: &str,
    context: &str
) {
    let activation = activations.get(key).copied().unwrap_or(0.0);
    assert!(
        activation > 0.0,
        "{}: Neuron '{}' should be activated but has activation {}",
        context, neuron_name, activation
    );
}

/// Assert that a neuron was not activated (zero or missing)
pub fn assert_neuron_silent(
    activations: &HashMap<EntityKey, f32>,
    key: &EntityKey,
    neuron_name: &str,
    context: &str
) {
    let activation = activations.get(key).copied().unwrap_or(0.0);
    assert_eq!(
        activation, 0.0,
        "{}: Neuron '{}' should be silent but has activation {}",
        context, neuron_name, activation
    );
}

// ==================== Timing Helpers ====================

/// Measure execution time and assert it's within limits
pub async fn assert_timed_execution<F, Fut, T>(
    operation: F,
    max_duration_ms: u128,
    operation_name: &str
) -> T
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = T>,
{
    let start = std::time::Instant::now();
    let result = operation().await;
    let duration = start.elapsed().as_millis();
    
    assert!(
        duration <= max_duration_ms,
        "{} took {}ms, exceeding limit of {}ms",
        operation_name, duration, max_duration_ms
    );
    
    result
}