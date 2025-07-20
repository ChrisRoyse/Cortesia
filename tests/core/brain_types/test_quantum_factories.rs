/// Quantum Test Data Factories - Advanced Fixture Generation
/// 
/// Provides comprehensive data generation for property-based testing,
/// stress testing, and edge case exploration across all brain types.

use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, BrainInspiredRelationship, RelationType,
    LogicGate, LogicGateType, ActivationPattern, ActivationStep, ActivationOperation,
    GraphOperation, TrainingExample
};
use llmkg::core::types::{EntityKey, AttributeValue};
use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use super::test_constants;

// ==================== Quantum Entity Factory ====================

/// Factory for generating diverse entity test data
pub struct QuantumEntityFactory {
    entity_id_counter: u64,
    concept_id_counter: u64,
    randomizer: QuantumRandomizer,
}

impl QuantumEntityFactory {
    pub fn new(seed: u64) -> Self {
        Self {
            entity_id_counter: 1,
            concept_id_counter: 1,
            randomizer: QuantumRandomizer::new(seed),
        }
    }

    /// Generate entities with specific characteristics
    pub fn generate_entity_collection(&mut self, spec: EntityCollectionSpec) -> Vec<BrainInspiredEntity> {
        let mut entities = Vec::new();

        // Input layer entities
        for _ in 0..spec.input_count {
            entities.push(self.create_input_entity(spec.embedding_dimension));
        }

        // Hidden layer entities
        for _ in 0..spec.hidden_count {
            entities.push(self.create_hidden_entity(spec.embedding_dimension));
        }

        // Output layer entities
        for _ in 0..spec.output_count {
            entities.push(self.create_output_entity(spec.embedding_dimension));
        }

        // Gate entities (special processing nodes)
        for _ in 0..spec.gate_count {
            entities.push(self.create_gate_entity(spec.embedding_dimension));
        }

        entities
    }

    fn create_input_entity(&mut self, embedding_dim: usize) -> BrainInspiredEntity {
        let mut entity = BrainInspiredEntity::new(
            format!("input_concept_{}", self.concept_id_counter),
            EntityDirection::Input
        );
        
        entity.id = EntityKey::from(slotmap::KeyData::from_ffi(self.entity_id_counter));
        entity.embedding = self.randomizer.generate_embedding(embedding_dim, EmbeddingDistribution::Gaussian);
        entity.activation_state = self.randomizer.next_activation();
        
        // Add realistic properties for input entities
        entity.properties.insert("sensitivity".to_string(), AttributeValue::Float(self.randomizer.next_range(0.1, 1.0)));
        entity.properties.insert("input_type".to_string(), AttributeValue::String("sensory".to_string()));
        entity.properties.insert("learning_rate".to_string(), AttributeValue::Float(test_constants::STANDARD_LEARNING_RATE));

        self.entity_id_counter += 1;
        self.concept_id_counter += 1;
        entity
    }

    fn create_hidden_entity(&mut self, embedding_dim: usize) -> BrainInspiredEntity {
        let mut entity = BrainInspiredEntity::new(
            format!("hidden_concept_{}", self.concept_id_counter),
            EntityDirection::Hidden
        );
        
        entity.id = EntityKey::from(slotmap::KeyData::from_ffi(self.entity_id_counter));
        entity.embedding = self.randomizer.generate_embedding(embedding_dim, EmbeddingDistribution::Uniform);
        entity.activation_state = self.randomizer.next_activation();
        
        // Hidden entities often have processing characteristics
        entity.properties.insert("processing_type".to_string(), AttributeValue::String("associative".to_string()));
        entity.properties.insert("memory_capacity".to_string(), AttributeValue::Float(self.randomizer.next_range(0.5, 2.0)));
        entity.properties.insert("lateral_inhibition".to_string(), AttributeValue::Bool(self.randomizer.next_bool()));

        self.entity_id_counter += 1;
        self.concept_id_counter += 1;
        entity
    }

    fn create_output_entity(&mut self, embedding_dim: usize) -> BrainInspiredEntity {
        let mut entity = BrainInspiredEntity::new(
            format!("output_concept_{}", self.concept_id_counter),
            EntityDirection::Output
        );
        
        entity.id = EntityKey::from(slotmap::KeyData::from_ffi(self.entity_id_counter));
        entity.embedding = self.randomizer.generate_embedding(embedding_dim, EmbeddingDistribution::Sparse);
        entity.activation_state = self.randomizer.next_activation();
        
        // Output entities have decision characteristics
        entity.properties.insert("decision_threshold".to_string(), AttributeValue::Float(self.randomizer.next_range(0.3, 0.8)));
        entity.properties.insert("output_type".to_string(), AttributeValue::String("motor".to_string()));
        entity.properties.insert("confidence_level".to_string(), AttributeValue::Float(self.randomizer.next_range(0.6, 1.0)));

        self.entity_id_counter += 1;
        self.concept_id_counter += 1;
        entity
    }

    fn create_gate_entity(&mut self, embedding_dim: usize) -> BrainInspiredEntity {
        let mut entity = BrainInspiredEntity::new(
            format!("gate_concept_{}", self.concept_id_counter),
            EntityDirection::Gate
        );
        
        entity.id = EntityKey::from(slotmap::KeyData::from_ffi(self.entity_id_counter));
        entity.embedding = self.randomizer.generate_embedding(embedding_dim, EmbeddingDistribution::Binary);
        entity.activation_state = self.randomizer.next_activation();
        
        // Gate entities have logical processing characteristics
        entity.properties.insert("gate_function".to_string(), AttributeValue::String("logical".to_string()));
        entity.properties.insert("temporal_integration".to_string(), AttributeValue::Float(self.randomizer.next_range(0.1, 0.5)));
        entity.properties.insert("noise_tolerance".to_string(), AttributeValue::Float(self.randomizer.next_range(0.05, 0.2)));

        self.entity_id_counter += 1;
        self.concept_id_counter += 1;
        entity
    }

    /// Generate entities with specific activation patterns
    pub fn generate_entities_with_activation_pattern(&mut self, pattern: ActivationPatternType, count: usize) -> Vec<BrainInspiredEntity> {
        let mut entities = Vec::new();
        
        for i in 0..count {
            let mut entity = BrainInspiredEntity::new(
                format!("pattern_entity_{}", i),
                EntityDirection::Hidden
            );
            
            entity.id = EntityKey::from(slotmap::KeyData::from_ffi(self.entity_id_counter));
            entity.embedding = self.randomizer.generate_embedding(16, EmbeddingDistribution::Gaussian);
            
            entity.activation_state = match pattern {
                ActivationPatternType::Linear => i as f32 / count as f32,
                ActivationPatternType::Exponential => (2.0_f32.powi(i as i32) / 2.0_f32.powi(count as i32)).min(1.0),
                ActivationPatternType::Sinusoidal => (0.5 + 0.5 * (2.0 * std::f32::consts::PI * i as f32 / count as f32).sin()).abs(),
                ActivationPatternType::Random => self.randomizer.next_activation(),
                ActivationPatternType::Threshold => if i < count / 2 { 0.1 } else { 0.9 },
            };
            
            entities.push(entity);
            self.entity_id_counter += 1;
        }
        
        entities
    }

    /// Generate entities for stress testing
    pub fn generate_stress_test_entities(&mut self, stress_level: StressLevel) -> Vec<BrainInspiredEntity> {
        let (count, embedding_dim) = match stress_level {
            StressLevel::Light => (100, 8),
            StressLevel::Moderate => (1000, 16),
            StressLevel::Heavy => (10000, 32),
            StressLevel::Extreme => (100000, 64),
        };

        let spec = EntityCollectionSpec {
            input_count: count / 4,
            hidden_count: count / 2,
            output_count: count / 8,
            gate_count: count / 8,
            embedding_dimension: embedding_dim,
        };

        self.generate_entity_collection(spec)
    }
}

#[derive(Debug, Clone)]
pub struct EntityCollectionSpec {
    pub input_count: usize,
    pub hidden_count: usize,
    pub output_count: usize,
    pub gate_count: usize,
    pub embedding_dimension: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationPatternType {
    Linear,
    Exponential,
    Sinusoidal,
    Random,
    Threshold,
}

#[derive(Debug, Clone, Copy)]
pub enum StressLevel {
    Light,
    Moderate,
    Heavy,
    Extreme,
}

// ==================== Quantum Logic Gate Factory ====================

/// Factory for generating comprehensive logic gate test scenarios
pub struct QuantumLogicGateFactory {
    gate_id_counter: u64,
    randomizer: QuantumRandomizer,
}

impl QuantumLogicGateFactory {
    pub fn new(seed: u64) -> Self {
        Self {
            gate_id_counter: 1,
            randomizer: QuantumRandomizer::new(seed),
        }
    }

    /// Generate comprehensive gate test suite
    pub fn generate_gate_test_suite(&mut self) -> GateTestSuite {
        let mut basic_gates = Vec::new();
        let mut complex_gates = Vec::new();
        let mut edge_case_gates = Vec::new();

        // Basic gates with standard configurations
        for &gate_type in &[LogicGateType::And, LogicGateType::Or, LogicGateType::Not, LogicGateType::Xor] {
            basic_gates.push(self.create_standard_gate(gate_type));
        }

        // Complex gates with varied configurations
        for &gate_type in &[LogicGateType::Threshold, LogicGateType::Inhibitory, LogicGateType::Weighted] {
            complex_gates.push(self.create_complex_gate(gate_type));
        }

        // Edge case gates
        edge_case_gates.push(self.create_edge_case_gate(LogicGateType::And, 0.0)); // Zero threshold
        edge_case_gates.push(self.create_edge_case_gate(LogicGateType::Or, 1.0)); // Maximum threshold
        edge_case_gates.push(self.create_edge_case_gate(LogicGateType::Threshold, 0.5)); // Standard threshold

        GateTestSuite {
            basic_gates,
            complex_gates,
            edge_case_gates,
        }
    }

    fn create_standard_gate(&mut self, gate_type: LogicGateType) -> LogicGate {
        let input_count = match gate_type {
            LogicGateType::Not | LogicGateType::Identity => 1,
            LogicGateType::Xor | LogicGateType::Xnor => 2,
            _ => 2 + self.randomizer.next_int() % 4, // 2-5 inputs
        };

        let threshold = match gate_type {
            LogicGateType::And => test_constants::AND_GATE_THRESHOLD,
            LogicGateType::Or => test_constants::OR_GATE_THRESHOLD,
            LogicGateType::Threshold => test_constants::THRESHOLD_GATE_LIMIT,
            _ => 0.5,
        };

        self.create_gate_with_config(gate_type, threshold, input_count, 1)
    }

    fn create_complex_gate(&mut self, gate_type: LogicGateType) -> LogicGate {
        let input_count = 3 + self.randomizer.next_int() % 8; // 3-10 inputs
        let output_count = 1 + self.randomizer.next_int() % 3; // 1-3 outputs
        let threshold = self.randomizer.next_range(0.2, 0.8);

        let mut gate = self.create_gate_with_config(gate_type, threshold, input_count, output_count);

        // Add complex weight matrix for weighted gates
        if gate_type == LogicGateType::Weighted {
            gate.weight_matrix = (0..input_count)
                .map(|_| self.randomizer.next_range(0.1, 1.0))
                .collect();
        }

        gate
    }

    fn create_edge_case_gate(&mut self, gate_type: LogicGateType, threshold: f32) -> LogicGate {
        let input_count = match gate_type {
            LogicGateType::Not => 1,
            _ => 2,
        };

        self.create_gate_with_config(gate_type, threshold, input_count, 1)
    }

    fn create_gate_with_config(&mut self, gate_type: LogicGateType, threshold: f32, input_count: usize, output_count: usize) -> LogicGate {
        let mut gate = LogicGate::new(gate_type, threshold);
        gate.gate_id = EntityKey::from(slotmap::KeyData::from_ffi(self.gate_id_counter));
        
        gate.input_nodes = (0..input_count)
            .map(|i| EntityKey::from(slotmap::KeyData::from_ffi(self.gate_id_counter + 1 + i as u64)))
            .collect();
            
        gate.output_nodes = (0..output_count)
            .map(|i| EntityKey::from(slotmap::KeyData::from_ffi(self.gate_id_counter + 1 + input_count as u64 + i as u64)))
            .collect();
            
        gate.weight_matrix = vec![1.0; input_count];
        
        self.gate_id_counter += 1 + input_count as u64 + output_count as u64;
        gate
    }

    /// Generate input test vectors for comprehensive gate testing
    pub fn generate_comprehensive_test_vectors(&mut self, input_count: usize) -> Vec<Vec<f32>> {
        let mut vectors = Vec::new();

        // Boundary cases
        vectors.push(vec![0.0; input_count]); // All zeros
        vectors.push(vec![1.0; input_count]); // All ones
        vectors.push(vec![0.5; input_count]); // All half

        // Binary combinations (for small input counts)
        if input_count <= 4 {
            for i in 0..(1 << input_count) {
                let mut vector = Vec::new();
                for j in 0..input_count {
                    vector.push(if (i >> j) & 1 == 1 { 0.8 } else { 0.2 });
                }
                vectors.push(vector);
            }
        }

        // Random vectors
        for _ in 0..20 {
            let vector = (0..input_count)
                .map(|_| self.randomizer.next_activation())
                .collect();
            vectors.push(vector);
        }

        // Gradient vectors
        for _ in 0..5 {
            let base = self.randomizer.next_activation();
            let vector = (0..input_count)
                .map(|i| (base + i as f32 * 0.1) % 1.0)
                .collect();
            vectors.push(vector);
        }

        // Spike patterns
        for spike_pos in 0..input_count.min(5) {
            let mut vector = vec![0.1; input_count];
            vector[spike_pos] = 0.9;
            vectors.push(vector);
        }

        vectors
    }
}

#[derive(Debug)]
pub struct GateTestSuite {
    pub basic_gates: Vec<LogicGate>,
    pub complex_gates: Vec<LogicGate>,
    pub edge_case_gates: Vec<LogicGate>,
}

// ==================== Quantum Relationship Factory ====================

/// Factory for generating diverse relationship test data
pub struct QuantumRelationshipFactory {
    randomizer: QuantumRandomizer,
}

impl QuantumRelationshipFactory {
    pub fn new(seed: u64) -> Self {
        Self {
            randomizer: QuantumRandomizer::new(seed),
        }
    }

    /// Generate relationships for different network topologies
    pub fn generate_network_relationships(&mut self, entities: &[BrainInspiredEntity], topology: NetworkTopology) -> Vec<BrainInspiredRelationship> {
        match topology {
            NetworkTopology::FullyConnected => self.create_fully_connected(entities),
            NetworkTopology::SmallWorld => self.create_small_world(entities),
            NetworkTopology::ScaleFree => self.create_scale_free(entities),
            NetworkTopology::Random => self.create_random_network(entities),
            NetworkTopology::Layered => self.create_layered_network(entities),
        }
    }

    fn create_fully_connected(&mut self, entities: &[BrainInspiredEntity]) -> Vec<BrainInspiredRelationship> {
        let mut relationships = Vec::new();
        
        for (i, source) in entities.iter().enumerate() {
            for (j, target) in entities.iter().enumerate() {
                if i != j {
                    let relation_type = self.select_random_relation_type();
                    let weight = self.randomizer.next_range(0.1, 0.9);
                    let inhibitory = self.randomizer.next_float() < 0.2; // 20% inhibitory
                    
                    relationships.push(self.create_relationship(
                        source.id, target.id, relation_type, weight, inhibitory
                    ));
                }
            }
        }
        
        relationships
    }

    fn create_small_world(&mut self, entities: &[BrainInspiredEntity]) -> Vec<BrainInspiredRelationship> {
        let mut relationships = Vec::new();
        let n = entities.len();
        let k = 4; // Each node connected to k nearest neighbors
        let p = 0.3; // Rewiring probability

        // Create ring lattice
        for i in 0..n {
            for j in 1..=k/2 {
                let target_idx = (i + j) % n;
                
                // Rewire with probability p
                let final_target_idx = if self.randomizer.next_float() < p {
                    self.randomizer.next_int() % n
                } else {
                    target_idx
                };

                if i != final_target_idx {
                    let relation_type = self.select_random_relation_type();
                    let weight = self.randomizer.next_range(0.3, 0.8);
                    
                    relationships.push(self.create_relationship(
                        entities[i].id, entities[final_target_idx].id, relation_type, weight, false
                    ));
                }
            }
        }

        relationships
    }

    fn create_scale_free(&mut self, entities: &[BrainInspiredEntity]) -> Vec<BrainInspiredRelationship> {
        let mut relationships = Vec::new();
        let mut degrees = vec![0; entities.len()];

        // Preferential attachment
        for i in 1..entities.len() {
            let m = 2.min(i); // Number of connections to add
            
            for _ in 0..m {
                // Select target based on degree (preferential attachment)
                let target_idx = self.select_preferential_target(&degrees, i);
                
                if target_idx < i {
                    let relation_type = self.select_random_relation_type();
                    let weight = self.randomizer.next_range(0.4, 0.9);
                    
                    relationships.push(self.create_relationship(
                        entities[i].id, entities[target_idx].id, relation_type, weight, false
                    ));
                    
                    degrees[i] += 1;
                    degrees[target_idx] += 1;
                }
            }
        }

        relationships
    }

    fn create_random_network(&mut self, entities: &[BrainInspiredEntity]) -> Vec<BrainInspiredRelationship> {
        let mut relationships = Vec::new();
        let connection_probability = 0.3;

        for (i, source) in entities.iter().enumerate() {
            for (j, target) in entities.iter().enumerate() {
                if i != j && self.randomizer.next_float() < connection_probability {
                    let relation_type = self.select_random_relation_type();
                    let weight = self.randomizer.next_range(0.2, 0.8);
                    let inhibitory = self.randomizer.next_float() < 0.15; // 15% inhibitory
                    
                    relationships.push(self.create_relationship(
                        source.id, target.id, relation_type, weight, inhibitory
                    ));
                }
            }
        }

        relationships
    }

    fn create_layered_network(&mut self, entities: &[BrainInspiredEntity]) -> Vec<BrainInspiredRelationship> {
        let mut relationships = Vec::new();
        let layer_size = entities.len() / 3; // Assume 3 layers

        // Inter-layer connections
        for layer in 0..2 {
            let layer_start = layer * layer_size;
            let next_layer_start = (layer + 1) * layer_size;
            let layer_end = (layer_start + layer_size).min(entities.len());
            let next_layer_end = (next_layer_start + layer_size).min(entities.len());

            for i in layer_start..layer_end {
                for j in next_layer_start..next_layer_end {
                    if self.randomizer.next_float() < 0.6 { // 60% connectivity between layers
                        let relation_type = RelationType::RelatedTo;
                        let weight = self.randomizer.next_range(0.5, 0.9);
                        
                        relationships.push(self.create_relationship(
                            entities[i].id, entities[j].id, relation_type, weight, false
                        ));
                    }
                }
            }
        }

        // Intra-layer connections (sparse)
        for layer in 0..3 {
            let layer_start = layer * layer_size;
            let layer_end = (layer_start + layer_size).min(entities.len());

            for i in layer_start..layer_end {
                for j in layer_start..layer_end {
                    if i != j && self.randomizer.next_float() < 0.1 { // 10% intra-layer connectivity
                        let relation_type = RelationType::Similar;
                        let weight = self.randomizer.next_range(0.2, 0.5);
                        let inhibitory = self.randomizer.next_float() < 0.3; // 30% inhibitory within layer
                        
                        relationships.push(self.create_relationship(
                            entities[i].id, entities[j].id, relation_type, weight, inhibitory
                        ));
                    }
                }
            }
        }

        relationships
    }

    fn create_relationship(&self, source: EntityKey, target: EntityKey, relation_type: RelationType, weight: f32, inhibitory: bool) -> BrainInspiredRelationship {
        let mut rel = BrainInspiredRelationship::new(source, target, relation_type);
        rel.weight = weight;
        rel.strength = weight;
        rel.is_inhibitory = inhibitory;
        rel.temporal_decay = test_constants::STANDARD_DECAY_RATE;
        rel
    }

    fn select_random_relation_type(&mut self) -> RelationType {
        let relation_types = [
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
        
        relation_types[self.randomizer.next_int() % relation_types.len()]
    }

    fn select_preferential_target(&mut self, degrees: &[usize], max_idx: usize) -> usize {
        let total_degree: usize = degrees.iter().take(max_idx).sum();
        if total_degree == 0 {
            return self.randomizer.next_int() % max_idx;
        }

        let mut cumulative = 0;
        let target_sum = self.randomizer.next_int() % total_degree;

        for (i, &degree) in degrees.iter().take(max_idx).enumerate() {
            cumulative += degree;
            if cumulative > target_sum {
                return i;
            }
        }

        max_idx - 1
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NetworkTopology {
    FullyConnected,
    SmallWorld,
    ScaleFree,
    Random,
    Layered,
}

// ==================== Quantum Randomizer ====================

/// Advanced randomizer for generating diverse test data
pub struct QuantumRandomizer {
    state: u64,
}

impl QuantumRandomizer {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_int(&mut self) -> usize {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state / 65536) % 32768) as usize
    }

    pub fn next_float(&mut self) -> f32 {
        self.next_int() as f32 / 32768.0
    }

    pub fn next_activation(&mut self) -> f32 {
        self.next_range(0.0, 1.0)
    }

    pub fn next_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_float() * (max - min)
    }

    pub fn next_bool(&mut self) -> bool {
        self.next_float() > 0.5
    }

    pub fn generate_embedding(&mut self, dimension: usize, distribution: EmbeddingDistribution) -> Vec<f32> {
        match distribution {
            EmbeddingDistribution::Gaussian => self.generate_gaussian_embedding(dimension),
            EmbeddingDistribution::Uniform => self.generate_uniform_embedding(dimension),
            EmbeddingDistribution::Sparse => self.generate_sparse_embedding(dimension),
            EmbeddingDistribution::Binary => self.generate_binary_embedding(dimension),
        }
    }

    fn generate_gaussian_embedding(&mut self, dimension: usize) -> Vec<f32> {
        (0..dimension).map(|_| {
            // Box-Muller transform for Gaussian distribution
            let u1 = self.next_float();
            let u2 = self.next_float();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            z0 * 0.3 + 0.5 // Scale and center
        }).collect()
    }

    fn generate_uniform_embedding(&mut self, dimension: usize) -> Vec<f32> {
        (0..dimension).map(|_| self.next_float()).collect()
    }

    fn generate_sparse_embedding(&mut self, dimension: usize) -> Vec<f32> {
        let mut embedding = vec![0.0; dimension];
        let active_count = (dimension / 10).max(1); // 10% active

        for _ in 0..active_count {
            let idx = self.next_int() % dimension;
            embedding[idx] = self.next_range(0.7, 1.0);
        }

        embedding
    }

    fn generate_binary_embedding(&mut self, dimension: usize) -> Vec<f32> {
        (0..dimension).map(|_| if self.next_bool() { 1.0 } else { 0.0 }).collect()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EmbeddingDistribution {
    Gaussian,
    Uniform,
    Sparse,
    Binary,
}

// ==================== Quantum Pattern Factory ====================

/// Factory for generating complex activation and training patterns
pub struct QuantumPatternFactory {
    randomizer: QuantumRandomizer,
    pattern_counter: usize,
}

impl QuantumPatternFactory {
    pub fn new(seed: u64) -> Self {
        Self {
            randomizer: QuantumRandomizer::new(seed),
            pattern_counter: 0,
        }
    }

    /// Generate training examples for different learning scenarios
    pub fn generate_training_examples(&mut self, scenario: LearningScenario, count: usize) -> Vec<TrainingExample> {
        match scenario {
            LearningScenario::SimpleClassification => self.generate_classification_examples(count),
            LearningScenario::SequentialPrediction => self.generate_sequential_examples(count),
            LearningScenario::PatternCompletion => self.generate_completion_examples(count),
            LearningScenario::AssociativeMemory => self.generate_associative_examples(count),
            LearningScenario::TemporalDynamics => self.generate_temporal_examples(count),
        }
    }

    fn generate_classification_examples(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        let classes = ["positive", "negative", "neutral"];

        for i in 0..count {
            let class = classes[i % classes.len()];
            let text = format!("Classification example {} for class {}", i, class);
            
            let operations = vec![
                GraphOperation::CreateNode {
                    concept: format!("input_{}", i),
                    node_type: EntityDirection::Input,
                },
                GraphOperation::CreateNode {
                    concept: format!("output_{}", class),
                    node_type: EntityDirection::Output,
                },
                GraphOperation::CreateRelationship {
                    source: format!("input_{}", i),
                    target: format!("output_{}", class),
                    relation_type: RelationType::RelatedTo,
                    weight: 0.8,
                },
            ];

            let mut metadata = HashMap::new();
            metadata.insert("class".to_string(), class.to_string());
            metadata.insert("difficulty".to_string(), format!("{}", self.randomizer.next_range(0.1, 1.0)));

            examples.push(TrainingExample {
                text,
                expected_operations: operations,
                metadata,
            });
        }

        examples
    }

    fn generate_sequential_examples(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for i in 0..count {
            let sequence_length = 3 + self.randomizer.next_int() % 5; // 3-7 steps
            let text = format!("Sequential pattern {} with {} steps", i, sequence_length);
            
            let mut operations = Vec::new();
            
            // Create sequence nodes
            for step in 0..sequence_length {
                operations.push(GraphOperation::CreateNode {
                    concept: format!("seq_{}_{}", i, step),
                    node_type: EntityDirection::Hidden,
                });
                
                if step > 0 {
                    operations.push(GraphOperation::CreateRelationship {
                        source: format!("seq_{}_{}", i, step - 1),
                        target: format!("seq_{}_{}", i, step),
                        relation_type: RelationType::Temporal,
                        weight: 0.7,
                    });
                }
            }

            let mut metadata = HashMap::new();
            metadata.insert("sequence_length".to_string(), sequence_length.to_string());
            metadata.insert("pattern_type".to_string(), "temporal".to_string());

            examples.push(TrainingExample {
                text,
                expected_operations: operations,
                metadata,
            });
        }

        examples
    }

    fn generate_completion_examples(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        let patterns = ["ABC", "123", "XYZ", "abc"];

        for i in 0..count {
            let pattern = patterns[i % patterns.len()];
            let text = format!("Complete the pattern starting with {}", pattern);
            
            let mut operations = Vec::new();
            
            // Create pattern nodes
            for (j, ch) in pattern.chars().enumerate() {
                operations.push(GraphOperation::CreateNode {
                    concept: format!("pattern_{}_{}", pattern, ch),
                    node_type: EntityDirection::Input,
                });
                
                if j > 0 {
                    operations.push(GraphOperation::CreateRelationship {
                        source: format!("pattern_{}_{}", pattern, pattern.chars().nth(j-1).unwrap()),
                        target: format!("pattern_{}_{}", pattern, ch),
                        relation_type: RelationType::RelatedTo,
                        weight: 0.9,
                    });
                }
            }

            let mut metadata = HashMap::new();
            metadata.insert("pattern".to_string(), pattern.to_string());
            metadata.insert("completion_task".to_string(), "true".to_string());

            examples.push(TrainingExample {
                text,
                expected_operations: operations,
                metadata,
            });
        }

        examples
    }

    fn generate_associative_examples(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();
        let associations = [
            ("sun", "light"), ("water", "wet"), ("fire", "hot"),
            ("ice", "cold"), ("music", "sound"), ("book", "read"),
        ];

        for i in 0..count {
            let (source, target) = associations[i % associations.len()];
            let text = format!("Associate {} with {}", source, target);
            
            let operations = vec![
                GraphOperation::CreateNode {
                    concept: source.to_string(),
                    node_type: EntityDirection::Input,
                },
                GraphOperation::CreateNode {
                    concept: target.to_string(),
                    node_type: EntityDirection::Output,
                },
                GraphOperation::CreateRelationship {
                    source: source.to_string(),
                    target: target.to_string(),
                    relation_type: RelationType::RelatedTo,
                    weight: 0.85,
                },
            ];

            let mut metadata = HashMap::new();
            metadata.insert("association_strength".to_string(), format!("{}", self.randomizer.next_range(0.6, 1.0)));
            metadata.insert("semantic_type".to_string(), "conceptual".to_string());

            examples.push(TrainingExample {
                text,
                expected_operations: operations,
                metadata,
            });
        }

        examples
    }

    fn generate_temporal_examples(&mut self, count: usize) -> Vec<TrainingExample> {
        let mut examples = Vec::new();

        for i in 0..count {
            let time_steps = 4 + self.randomizer.next_int() % 4; // 4-7 time steps
            let text = format!("Temporal dynamics example {} with {} time steps", i, time_steps);
            
            let mut operations = Vec::new();
            
            for t in 0..time_steps {
                operations.push(GraphOperation::CreateNode {
                    concept: format!("state_{}_{}", i, t),
                    node_type: EntityDirection::Hidden,
                });
                
                if t > 0 {
                    let decay_weight = 0.9 * (0.8_f32.powi(t as i32));
                    operations.push(GraphOperation::CreateRelationship {
                        source: format!("state_{}_{}", i, t - 1),
                        target: format!("state_{}_{}", i, t),
                        relation_type: RelationType::Temporal,
                        weight: decay_weight,
                    });
                }
            }

            let mut metadata = HashMap::new();
            metadata.insert("time_steps".to_string(), time_steps.to_string());
            metadata.insert("decay_rate".to_string(), "0.8".to_string());
            metadata.insert("temporal_type".to_string(), "sequential".to_string());

            examples.push(TrainingExample {
                text,
                expected_operations: operations,
                metadata,
            });
        }

        examples
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LearningScenario {
    SimpleClassification,
    SequentialPrediction,
    PatternCompletion,
    AssociativeMemory,
    TemporalDynamics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_factory() {
        let mut factory = QuantumEntityFactory::new(42);
        let spec = EntityCollectionSpec {
            input_count: 3,
            hidden_count: 2,
            output_count: 1,
            gate_count: 1,
            embedding_dimension: 16,
        };
        
        let entities = factory.generate_entity_collection(spec);
        assert_eq!(entities.len(), 7);
        
        let input_count = entities.iter().filter(|e| e.direction == EntityDirection::Input).count();
        let hidden_count = entities.iter().filter(|e| e.direction == EntityDirection::Hidden).count();
        let output_count = entities.iter().filter(|e| e.direction == EntityDirection::Output).count();
        let gate_count = entities.iter().filter(|e| e.direction == EntityDirection::Gate).count();
        
        assert_eq!(input_count, 3);
        assert_eq!(hidden_count, 2);
        assert_eq!(output_count, 1);
        assert_eq!(gate_count, 1);
    }

    #[test]
    fn test_gate_factory() {
        let mut factory = QuantumLogicGateFactory::new(42);
        let suite = factory.generate_gate_test_suite();
        
        assert_eq!(suite.basic_gates.len(), 4);
        assert_eq!(suite.complex_gates.len(), 3);
        assert_eq!(suite.edge_case_gates.len(), 3);
    }

    #[test]
    fn test_relationship_factory() {
        let mut entity_factory = QuantumEntityFactory::new(42);
        let mut rel_factory = QuantumRelationshipFactory::new(42);
        
        let spec = EntityCollectionSpec {
            input_count: 3,
            hidden_count: 3,
            output_count: 2,
            gate_count: 0,
            embedding_dimension: 8,
        };
        
        let entities = entity_factory.generate_entity_collection(spec);
        let relationships = rel_factory.generate_network_relationships(&entities, NetworkTopology::Random);
        
        assert!(!relationships.is_empty());
        
        // Verify relationship properties
        for rel in &relationships {
            assert!(rel.weight >= 0.0 && rel.weight <= 1.0);
            assert!(!rel.source.data().is_null());
            assert!(!rel.target.data().is_null());
        }
    }

    #[test]
    fn test_pattern_factory() {
        let mut factory = QuantumPatternFactory::new(42);
        let examples = factory.generate_training_examples(LearningScenario::SimpleClassification, 5);
        
        assert_eq!(examples.len(), 5);
        
        for example in &examples {
            assert!(!example.text.is_empty());
            assert!(!example.expected_operations.is_empty());
            assert!(example.metadata.contains_key("class"));
        }
    }
}