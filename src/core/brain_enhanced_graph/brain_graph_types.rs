//! Type definitions for brain-enhanced knowledge graph

use crate::core::types::EntityKey;
use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Result of activation propagation from a specific entity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationPropagationResult {
    pub affected_entities: usize,
    pub total_activation_spread: f32,
    pub propagation_time: Duration,
}

/// Query result structure for brain-enhanced graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainQueryResult {
    pub entities: Vec<EntityKey>,
    pub activations: HashMap<EntityKey, f32>,
    pub query_time: Duration,
    pub total_activation: f32,
}

impl BrainQueryResult {
    /// Create new empty query result
    pub fn new() -> Self {
        Self {
            entities: Vec::new(),
            activations: HashMap::new(),
            query_time: Duration::from_millis(0),
            total_activation: 0.0,
        }
    }

    /// Add entity with activation
    pub fn add_entity(&mut self, entity: EntityKey, activation: f32) {
        self.entities.push(entity);
        self.activations.insert(entity, activation);
        self.total_activation += activation;
    }

    /// Get activation for entity
    pub fn get_activation(&self, entity: &EntityKey) -> Option<f32> {
        self.activations.get(entity).copied()
    }

    /// Get entities sorted by activation
    pub fn get_sorted_entities(&self) -> Vec<(EntityKey, f32)> {
        let mut entities_with_activation: Vec<(EntityKey, f32)> = self
            .entities
            .iter()
            .filter_map(|entity| {
                self.activations.get(entity).map(|activation| (*entity, *activation))
            })
            .collect();

        entities_with_activation.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entities_with_activation
    }

    /// Get top k entities by activation
    pub fn get_top_k(&self, k: usize) -> Vec<(EntityKey, f32)> {
        let mut sorted = self.get_sorted_entities();
        sorted.truncate(k);
        sorted
    }

    /// Get entities above threshold
    pub fn get_entities_above_threshold(&self, threshold: f32) -> Vec<(EntityKey, f32)> {
        self.get_sorted_entities()
            .into_iter()
            .filter(|(_, activation)| *activation >= threshold)
            .collect()
    }

    /// Get average activation
    pub fn get_average_activation(&self) -> f32 {
        if self.entities.is_empty() {
            0.0
        } else {
            self.total_activation / self.entities.len() as f32
        }
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get entity count
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }
}

impl Default for BrainQueryResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Structured concept representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptStructure {
    pub input_entities: Vec<EntityKey>,
    pub output_entities: Vec<EntityKey>,
    pub gate_entities: Vec<EntityKey>,
    pub concept_activation: f32,
    pub coherence_score: f32,
}

impl ConceptStructure {
    /// Create new concept structure
    pub fn new() -> Self {
        Self {
            input_entities: Vec::new(),
            output_entities: Vec::new(),
            gate_entities: Vec::new(),
            concept_activation: 0.0,
            coherence_score: 0.0,
        }
    }

    /// Add input entity
    pub fn add_input(&mut self, entity: EntityKey) {
        self.input_entities.push(entity);
    }

    /// Add output entity
    pub fn add_output(&mut self, entity: EntityKey) {
        self.output_entities.push(entity);
    }

    /// Add gate entity
    pub fn add_gate(&mut self, entity: EntityKey) {
        self.gate_entities.push(entity);
    }

    /// Get total entity count
    pub fn total_entities(&self) -> usize {
        self.input_entities.len() + self.output_entities.len() + self.gate_entities.len()
    }

    /// Check if concept is well-formed
    pub fn is_well_formed(&self) -> bool {
        !self.input_entities.is_empty() && 
        !self.output_entities.is_empty() && 
        self.coherence_score > 0.5
    }

    /// Get all entities
    pub fn get_all_entities(&self) -> Vec<EntityKey> {
        let mut all_entities = Vec::new();
        all_entities.extend(&self.input_entities);
        all_entities.extend(&self.output_entities);
        all_entities.extend(&self.gate_entities);
        all_entities
    }

    /// Calculate activation density
    pub fn activation_density(&self) -> f32 {
        if self.total_entities() == 0 {
            0.0
        } else {
            self.concept_activation / self.total_entities() as f32
        }
    }
}

impl Default for ConceptStructure {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive brain statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainStatistics {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub avg_activation: f32,
    pub max_activation: f32,
    pub min_activation: f32,
    pub graph_density: f32,
    pub clustering_coefficient: f32,
    pub average_path_length: f32,
    pub betweenness_centrality: HashMap<EntityKey, f32>,
    pub activation_distribution: HashMap<String, usize>,
    pub concept_coherence: f32,
    pub learning_efficiency: f32,
    pub max_degree: usize,
    pub average_degree: f32,
}

impl BrainStatistics {
    /// Create new statistics
    pub fn new() -> Self {
        Self {
            entity_count: 0,
            relationship_count: 0,
            avg_activation: 0.0,
            max_activation: 0.0,
            min_activation: 0.0,
            graph_density: 0.0,
            clustering_coefficient: 0.0,
            average_path_length: 0.0,
            betweenness_centrality: HashMap::new(),
            activation_distribution: HashMap::new(),
            concept_coherence: 0.0,
            learning_efficiency: 0.0,
            max_degree: 0,
            average_degree: 0.0,
        }
    }

    /// Calculate graph health score
    pub fn graph_health_score(&self) -> f32 {
        let density_score = self.graph_density * 0.3;
        let clustering_score = self.clustering_coefficient * 0.2;
        let coherence_score = self.concept_coherence * 0.3;
        let efficiency_score = self.learning_efficiency * 0.2;
        
        density_score + clustering_score + coherence_score + efficiency_score
    }

    /// Get activation statistics
    pub fn get_activation_stats(&self) -> ActivationStats {
        ActivationStats {
            average: self.avg_activation,
            maximum: self.max_activation,
            minimum: self.min_activation,
            range: self.max_activation - self.min_activation,
            distribution: self.activation_distribution.clone(),
        }
    }

    /// Get connectivity statistics
    pub fn get_connectivity_stats(&self) -> ConnectivityStats {
        ConnectivityStats {
            entity_count: self.entity_count,
            relationship_count: self.relationship_count,
            density: self.graph_density,
            clustering_coefficient: self.clustering_coefficient,
            average_path_length: self.average_path_length,
        }
    }

    /// Check if graph is well-connected
    pub fn is_well_connected(&self) -> bool {
        self.graph_density > 0.1 && 
        self.clustering_coefficient > 0.3 && 
        self.average_path_length < 6.0
    }

    /// Get most central entities
    pub fn get_most_central_entities(&self, k: usize) -> Vec<(EntityKey, f32)> {
        let mut centrality: Vec<(EntityKey, f32)> = self.betweenness_centrality.iter()
            .map(|(entity, score)| (*entity, *score))
            .collect();
        
        centrality.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        centrality.truncate(k);
        centrality
    }
}

impl Default for BrainStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Activation statistics
#[derive(Debug, Clone)]
pub struct ActivationStats {
    pub average: f32,
    pub maximum: f32,
    pub minimum: f32,
    pub range: f32,
    pub distribution: HashMap<String, usize>,
}

/// Connectivity statistics
#[derive(Debug, Clone)]
pub struct ConnectivityStats {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub density: f32,
    pub clustering_coefficient: f32,
    pub average_path_length: f32,
}

/// Activation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationConfig {
    pub default_threshold: f32,
    pub max_iterations: usize,
    pub decay_factor: f32,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            default_threshold: 0.5,
            max_iterations: 100,
            decay_factor: 0.95,
        }
    }
}

/// Brain graph configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainEnhancedConfig {
    pub learning_rate: f32,
    pub activation_threshold: f32,
    pub max_activation_spread: usize,
    pub neural_dampening: f32,
    pub concept_coherence_threshold: f32,
    pub enable_hebbian_learning: bool,
    pub enable_concept_formation: bool,
    pub enable_neural_plasticity: bool,
    pub memory_consolidation_threshold: f32,
    pub synaptic_strength_decay: f32,
    pub embedding_dim: usize,
    pub activation_config: ActivationConfig,
    pub enable_temporal_tracking: bool,
    pub enable_sdr_storage: bool,
}

impl Default for BrainEnhancedConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            activation_threshold: 0.5,
            max_activation_spread: 5,
            neural_dampening: 0.95,
            concept_coherence_threshold: 0.7,
            enable_hebbian_learning: true,
            enable_concept_formation: true,
            enable_neural_plasticity: true,
            memory_consolidation_threshold: 0.8,
            synaptic_strength_decay: 0.99,
            embedding_dim: 384,
            activation_config: ActivationConfig::default(),
            enable_temporal_tracking: true,
            enable_sdr_storage: true,
        }
    }
}

impl BrainEnhancedConfig {
    /// Create configuration for testing
    pub fn for_testing() -> Self {
        Self {
            learning_rate: 0.2,
            activation_threshold: 0.3,
            max_activation_spread: 3,
            neural_dampening: 0.9,
            concept_coherence_threshold: 0.5,
            enable_hebbian_learning: true,
            enable_concept_formation: true,
            enable_neural_plasticity: true,
            memory_consolidation_threshold: 0.6,
            synaptic_strength_decay: 0.95,
            embedding_dim: 128,
            activation_config: ActivationConfig::default(),
            enable_temporal_tracking: true,
            enable_sdr_storage: false,
        }
    }

    /// Create high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            learning_rate: 0.05,
            activation_threshold: 0.7,
            max_activation_spread: 3,
            neural_dampening: 0.98,
            concept_coherence_threshold: 0.8,
            enable_hebbian_learning: true,
            enable_concept_formation: false, // Disable for performance
            enable_neural_plasticity: true,
            memory_consolidation_threshold: 0.9,
            synaptic_strength_decay: 0.99,
            embedding_dim: 256,
            activation_config: ActivationConfig::default(),
            enable_temporal_tracking: true,
            enable_sdr_storage: true,
        }
    }

    /// Create exploratory configuration
    pub fn exploratory() -> Self {
        Self {
            learning_rate: 0.15,
            activation_threshold: 0.4,
            max_activation_spread: 7,
            neural_dampening: 0.9,
            concept_coherence_threshold: 0.6,
            enable_hebbian_learning: true,
            enable_concept_formation: true,
            enable_neural_plasticity: true,
            memory_consolidation_threshold: 0.7,
            synaptic_strength_decay: 0.98,
            embedding_dim: 192,
            activation_config: ActivationConfig::default(),
            enable_temporal_tracking: true,
            enable_sdr_storage: true,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err("Learning rate must be between 0 and 1".to_string());
        }
        
        if self.activation_threshold < 0.0 || self.activation_threshold > 1.0 {
            return Err("Activation threshold must be between 0 and 1".to_string());
        }
        
        if self.max_activation_spread == 0 {
            return Err("Max activation spread must be greater than 0".to_string());
        }
        
        if self.neural_dampening <= 0.0 || self.neural_dampening > 1.0 {
            return Err("Neural dampening must be between 0 and 1".to_string());
        }
        
        Ok(())
    }
}

/// Neural activation patterns
#[derive(Debug, Clone)]
pub enum ActivationPattern {
    Focused,    // High activation in few entities
    Distributed, // Moderate activation across many entities
    Sparse,     // Low activation in many entities
    Clustered,  // High activation in connected groups
}

/// Learning mode for brain operations
#[derive(Debug, Clone)]
pub enum LearningMode {
    Supervised,     // Guided learning with targets
    Unsupervised,   // Self-organizing learning
    Reinforcement,  // Reward-based learning
    Hebbian,        // Connection-based learning
}

/// Query mode for brain operations
#[derive(Debug, Clone)]
pub enum QueryMode {
    Exact,          // Exact match queries
    Fuzzy,          // Approximate matching
    Associative,    // Association-based queries
    Conceptual,     // Concept-based queries
}

/// Graph health metrics
#[derive(Debug, Clone)]
pub struct GraphHealthMetrics {
    pub connectivity_score: f32,
    pub activation_balance: f32,
    pub learning_stability: f32,
    pub concept_coherence: f32,
    pub overall_health: f32,
}

impl GraphHealthMetrics {
    /// Check if graph is healthy
    pub fn is_healthy(&self) -> bool {
        self.overall_health > 0.7 && 
        self.connectivity_score > 0.6 &&
        self.activation_balance > 0.5 &&
        self.learning_stability > 0.6
    }
    
    /// Get health report
    pub fn get_health_report(&self) -> String {
        format!(
            "Graph Health Report:\n\
            - Overall Health: {:.2}\n\
            - Connectivity: {:.2}\n\
            - Activation Balance: {:.2}\n\
            - Learning Stability: {:.2}\n\
            - Concept Coherence: {:.2}\n\
            - Status: {}",
            self.overall_health,
            self.connectivity_score,
            self.activation_balance,
            self.learning_stability,
            self.concept_coherence,
            if self.is_healthy() { "Healthy" } else { "Needs Attention" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Helper function to create a test EntityKey
    fn create_test_entity(id: u64) -> EntityKey {
        EntityKey::new(id.to_string())
    }

    #[cfg(test)]
    mod brain_query_result_tests {
        use super::*;

        #[test]
        fn test_new_query_result() {
            let result = BrainQueryResult::new();
            assert!(result.entities.is_empty());
            assert!(result.activations.is_empty());
            assert_eq!(result.query_time, Duration::from_millis(0));
            assert_eq!(result.total_activation, 0.0);
        }

        #[test]
        fn test_default_query_result() {
            let result = BrainQueryResult::default();
            assert!(result.entities.is_empty());
            assert!(result.activations.is_empty());
            assert_eq!(result.query_time, Duration::from_millis(0));
            assert_eq!(result.total_activation, 0.0);
        }

        #[test]
        fn test_add_entity() {
            let mut result = BrainQueryResult::new();
            let entity1 = create_test_entity(1);
            let entity2 = create_test_entity(2);

            result.add_entity(entity1, 0.5);
            result.add_entity(entity2, 0.8);

            assert_eq!(result.entities.len(), 2);
            assert_eq!(result.activations.len(), 2);
            assert_eq!(result.total_activation, 1.3);
            assert_eq!(result.get_activation(&entity1), Some(0.5));
            assert_eq!(result.get_activation(&entity2), Some(0.8));
        }

        #[test]
        fn test_get_activation_nonexistent() {
            let result = BrainQueryResult::new();
            let entity = create_test_entity(999);
            assert_eq!(result.get_activation(&entity), None);
        }

        #[test]
        fn test_get_sorted_entities() {
            let mut result = BrainQueryResult::new();
            let entity1 = create_test_entity(1);
            let entity2 = create_test_entity(2);
            let entity3 = create_test_entity(3);

            result.add_entity(entity1, 0.3);
            result.add_entity(entity2, 0.8);
            result.add_entity(entity3, 0.5);

            let sorted = result.get_sorted_entities();
            assert_eq!(sorted.len(), 3);
            assert_eq!(sorted[0], (entity2, 0.8));
            assert_eq!(sorted[1], (entity3, 0.5));
            assert_eq!(sorted[2], (entity1, 0.3));
        }

        #[test]
        fn test_get_sorted_entities_empty() {
            let result = BrainQueryResult::new();
            let sorted = result.get_sorted_entities();
            assert!(sorted.is_empty());
        }

        #[test]
        fn test_get_top_k() {
            let mut result = BrainQueryResult::new();
            let entity1 = create_test_entity(1);
            let entity2 = create_test_entity(2);
            let entity3 = create_test_entity(3);

            result.add_entity(entity1, 0.3);
            result.add_entity(entity2, 0.8);
            result.add_entity(entity3, 0.5);

            let top_2 = result.get_top_k(2);
            assert_eq!(top_2.len(), 2);
            assert_eq!(top_2[0], (entity2, 0.8));
            assert_eq!(top_2[1], (entity3, 0.5));

            let top_10 = result.get_top_k(10);
            assert_eq!(top_10.len(), 3); // Should only return available entities
        }

        #[test]
        fn test_get_top_k_empty() {
            let result = BrainQueryResult::new();
            let top_k = result.get_top_k(5);
            assert!(top_k.is_empty());
        }

        #[test]
        fn test_get_entities_above_threshold() {
            let mut result = BrainQueryResult::new();
            let entity1 = create_test_entity(1);
            let entity2 = create_test_entity(2);
            let entity3 = create_test_entity(3);

            result.add_entity(entity1, 0.3);
            result.add_entity(entity2, 0.8);
            result.add_entity(entity3, 0.5);

            let above_threshold = result.get_entities_above_threshold(0.4);
            assert_eq!(above_threshold.len(), 2);
            assert!(above_threshold.contains(&(entity2, 0.8)));
            assert!(above_threshold.contains(&(entity3, 0.5)));
        }

        #[test]
        fn test_get_entities_above_threshold_none() {
            let mut result = BrainQueryResult::new();
            let entity1 = create_test_entity(1);
            result.add_entity(entity1, 0.3);

            let above_threshold = result.get_entities_above_threshold(0.5);
            assert!(above_threshold.is_empty());
        }

        #[test]
        fn test_get_average_activation() {
            let mut result = BrainQueryResult::new();
            let entity1 = create_test_entity(1);
            let entity2 = create_test_entity(2);

            result.add_entity(entity1, 0.4);
            result.add_entity(entity2, 0.6);

            assert_eq!(result.get_average_activation(), 0.5);
        }

        #[test]
        fn test_get_average_activation_empty() {
            let result = BrainQueryResult::new();
            assert_eq!(result.get_average_activation(), 0.0);
        }

        #[test]
        fn test_is_empty() {
            let mut result = BrainQueryResult::new();
            assert!(result.is_empty());

            let entity = create_test_entity(1);
            result.add_entity(entity, 0.5);
            assert!(!result.is_empty());
        }

        #[test]
        fn test_entity_count() {
            let mut result = BrainQueryResult::new();
            assert_eq!(result.entity_count(), 0);

            let entity1 = create_test_entity(1);
            let entity2 = create_test_entity(2);
            result.add_entity(entity1, 0.5);
            result.add_entity(entity2, 0.7);

            assert_eq!(result.entity_count(), 2);
        }

        #[test]
        fn test_add_entity_with_zero_activation() {
            let mut result = BrainQueryResult::new();
            let entity = create_test_entity(1);
            result.add_entity(entity, 0.0);

            assert_eq!(result.entity_count(), 1);
            assert_eq!(result.get_activation(&entity), Some(0.0));
            assert_eq!(result.total_activation, 0.0);
        }

        #[test]
        fn test_add_entity_with_negative_activation() {
            let mut result = BrainQueryResult::new();
            let entity = create_test_entity(1);
            result.add_entity(entity, -0.5);

            assert_eq!(result.entity_count(), 1);
            assert_eq!(result.get_activation(&entity), Some(-0.5));
            assert_eq!(result.total_activation, -0.5);
        }
    }

    #[cfg(test)]
    mod concept_structure_tests {
        use super::*;

        #[test]
        fn test_new_concept_structure() {
            let concept = ConceptStructure::new();
            assert!(concept.input_entities.is_empty());
            assert!(concept.output_entities.is_empty());
            assert!(concept.gate_entities.is_empty());
            assert_eq!(concept.concept_activation, 0.0);
            assert_eq!(concept.coherence_score, 0.0);
        }

        #[test]
        fn test_default_concept_structure() {
            let concept = ConceptStructure::default();
            assert!(concept.input_entities.is_empty());
            assert!(concept.output_entities.is_empty());
            assert!(concept.gate_entities.is_empty());
            assert_eq!(concept.concept_activation, 0.0);
            assert_eq!(concept.coherence_score, 0.0);
        }

        #[test]
        fn test_add_input() {
            let mut concept = ConceptStructure::new();
            let entity1 = create_test_entity(1);
            let entity2 = create_test_entity(2);

            concept.add_input(entity1);
            concept.add_input(entity2);

            assert_eq!(concept.input_entities.len(), 2);
            assert!(concept.input_entities.contains(&entity1));
            assert!(concept.input_entities.contains(&entity2));
        }

        #[test]
        fn test_add_output() {
            let mut concept = ConceptStructure::new();
            let entity1 = create_test_entity(1);
            let entity2 = create_test_entity(2);

            concept.add_output(entity1);
            concept.add_output(entity2);

            assert_eq!(concept.output_entities.len(), 2);
            assert!(concept.output_entities.contains(&entity1));
            assert!(concept.output_entities.contains(&entity2));
        }

        #[test]
        fn test_add_gate() {
            let mut concept = ConceptStructure::new();
            let entity1 = create_test_entity(1);
            let entity2 = create_test_entity(2);

            concept.add_gate(entity1);
            concept.add_gate(entity2);

            assert_eq!(concept.gate_entities.len(), 2);
            assert!(concept.gate_entities.contains(&entity1));
            assert!(concept.gate_entities.contains(&entity2));
        }

        #[test]
        fn test_total_entities() {
            let mut concept = ConceptStructure::new();
            assert_eq!(concept.total_entities(), 0);

            concept.add_input(create_test_entity(1));
            concept.add_input(create_test_entity(2));
            concept.add_output(create_test_entity(3));
            concept.add_gate(create_test_entity(4));

            assert_eq!(concept.total_entities(), 4);
        }

        #[test]
        fn test_is_well_formed_valid() {
            let mut concept = ConceptStructure::new();
            concept.add_input(create_test_entity(1));
            concept.add_output(create_test_entity(2));
            concept.coherence_score = 0.8;

            assert!(concept.is_well_formed());
        }

        #[test]
        fn test_is_well_formed_no_inputs() {
            let mut concept = ConceptStructure::new();
            concept.add_output(create_test_entity(2));
            concept.coherence_score = 0.8;

            assert!(!concept.is_well_formed());
        }

        #[test]
        fn test_is_well_formed_no_outputs() {
            let mut concept = ConceptStructure::new();
            concept.add_input(create_test_entity(1));
            concept.coherence_score = 0.8;

            assert!(!concept.is_well_formed());
        }

        #[test]
        fn test_is_well_formed_low_coherence() {
            let mut concept = ConceptStructure::new();
            concept.add_input(create_test_entity(1));
            concept.add_output(create_test_entity(2));
            concept.coherence_score = 0.3;

            assert!(!concept.is_well_formed());
        }

        #[test]
        fn test_get_all_entities() {
            let mut concept = ConceptStructure::new();
            let input_entity = create_test_entity(1);
            let output_entity = create_test_entity(2);
            let gate_entity = create_test_entity(3);

            concept.add_input(input_entity);
            concept.add_output(output_entity);
            concept.add_gate(gate_entity);

            let all_entities = concept.get_all_entities();
            assert_eq!(all_entities.len(), 3);
            assert!(all_entities.contains(&input_entity));
            assert!(all_entities.contains(&output_entity));
            assert!(all_entities.contains(&gate_entity));
        }

        #[test]
        fn test_get_all_entities_empty() {
            let concept = ConceptStructure::new();
            let all_entities = concept.get_all_entities();
            assert!(all_entities.is_empty());
        }

        #[test]
        fn test_activation_density() {
            let mut concept = ConceptStructure::new();
            concept.add_input(create_test_entity(1));
            concept.add_output(create_test_entity(2));
            concept.concept_activation = 1.0;

            assert_eq!(concept.activation_density(), 0.5);
        }

        #[test]
        fn test_activation_density_empty() {
            let concept = ConceptStructure::new();
            assert_eq!(concept.activation_density(), 0.0);
        }

        #[test]
        fn test_activation_density_zero_activation() {
            let mut concept = ConceptStructure::new();
            concept.add_input(create_test_entity(1));
            concept.add_output(create_test_entity(2));
            concept.concept_activation = 0.0;

            assert_eq!(concept.activation_density(), 0.0);
        }
    }

    #[cfg(test)]
    mod brain_statistics_tests {
        use super::*;

        #[test]
        fn test_new_brain_statistics() {
            let stats = BrainStatistics::new();
            assert_eq!(stats.entity_count, 0);
            assert_eq!(stats.relationship_count, 0);
            assert_eq!(stats.avg_activation, 0.0);
            assert_eq!(stats.max_activation, 0.0);
            assert_eq!(stats.min_activation, 0.0);
            assert_eq!(stats.graph_density, 0.0);
            assert_eq!(stats.clustering_coefficient, 0.0);
            assert_eq!(stats.average_path_length, 0.0);
            assert!(stats.betweenness_centrality.is_empty());
            assert!(stats.activation_distribution.is_empty());
            assert_eq!(stats.concept_coherence, 0.0);
            assert_eq!(stats.learning_efficiency, 0.0);
        }

        #[test]
        fn test_default_brain_statistics() {
            let stats = BrainStatistics::default();
            assert_eq!(stats.entity_count, 0);
            assert_eq!(stats.relationship_count, 0);
            assert_eq!(stats.avg_activation, 0.0);
        }

        #[test]
        fn test_graph_health_score() {
            let mut stats = BrainStatistics::new();
            stats.graph_density = 0.5;
            stats.clustering_coefficient = 0.6;
            stats.concept_coherence = 0.7;
            stats.learning_efficiency = 0.8;

            let expected = 0.5 * 0.3 + 0.6 * 0.2 + 0.7 * 0.3 + 0.8 * 0.2;
            assert_eq!(stats.graph_health_score(), expected);
        }

        #[test]
        fn test_graph_health_score_zero() {
            let stats = BrainStatistics::new();
            assert_eq!(stats.graph_health_score(), 0.0);
        }

        #[test]
        fn test_get_activation_stats() {
            let mut stats = BrainStatistics::new();
            stats.avg_activation = 0.5;
            stats.max_activation = 1.0;
            stats.min_activation = 0.1;
            stats.activation_distribution.insert("high".to_string(), 10);

            let activation_stats = stats.get_activation_stats();
            assert_eq!(activation_stats.average, 0.5);
            assert_eq!(activation_stats.maximum, 1.0);
            assert_eq!(activation_stats.minimum, 0.1);
            assert_eq!(activation_stats.range, 0.9);
            assert_eq!(activation_stats.distribution.get("high"), Some(&10));
        }

        #[test]
        fn test_get_connectivity_stats() {
            let mut stats = BrainStatistics::new();
            stats.entity_count = 100;
            stats.relationship_count = 200;
            stats.graph_density = 0.4;
            stats.clustering_coefficient = 0.6;
            stats.average_path_length = 3.5;

            let connectivity_stats = stats.get_connectivity_stats();
            assert_eq!(connectivity_stats.entity_count, 100);
            assert_eq!(connectivity_stats.relationship_count, 200);
            assert_eq!(connectivity_stats.density, 0.4);
            assert_eq!(connectivity_stats.clustering_coefficient, 0.6);
            assert_eq!(connectivity_stats.average_path_length, 3.5);
        }

        #[test]
        fn test_is_well_connected_valid() {
            let mut stats = BrainStatistics::new();
            stats.graph_density = 0.2;
            stats.clustering_coefficient = 0.4;
            stats.average_path_length = 4.0;

            assert!(stats.is_well_connected());
        }

        #[test]
        fn test_is_well_connected_low_density() {
            let mut stats = BrainStatistics::new();
            stats.graph_density = 0.05;
            stats.clustering_coefficient = 0.4;
            stats.average_path_length = 4.0;

            assert!(!stats.is_well_connected());
        }

        #[test]
        fn test_is_well_connected_low_clustering() {
            let mut stats = BrainStatistics::new();
            stats.graph_density = 0.2;
            stats.clustering_coefficient = 0.2;
            stats.average_path_length = 4.0;

            assert!(!stats.is_well_connected());
        }

        #[test]
        fn test_is_well_connected_long_path() {
            let mut stats = BrainStatistics::new();
            stats.graph_density = 0.2;
            stats.clustering_coefficient = 0.4;
            stats.average_path_length = 8.0;

            assert!(!stats.is_well_connected());
        }

        #[test]
        fn test_get_most_central_entities() {
            let mut stats = BrainStatistics::new();
            let entity1 = create_test_entity(1);
            let entity2 = create_test_entity(2);
            let entity3 = create_test_entity(3);

            stats.betweenness_centrality.insert(entity1, 0.3);
            stats.betweenness_centrality.insert(entity2, 0.8);
            stats.betweenness_centrality.insert(entity3, 0.5);

            let most_central = stats.get_most_central_entities(2);
            assert_eq!(most_central.len(), 2);
            assert_eq!(most_central[0], (entity2, 0.8));
            assert_eq!(most_central[1], (entity3, 0.5));
        }

        #[test]
        fn test_get_most_central_entities_empty() {
            let stats = BrainStatistics::new();
            let most_central = stats.get_most_central_entities(5);
            assert!(most_central.is_empty());
        }

        #[test]
        fn test_get_most_central_entities_more_than_available() {
            let mut stats = BrainStatistics::new();
            let entity1 = create_test_entity(1);
            stats.betweenness_centrality.insert(entity1, 0.5);

            let most_central = stats.get_most_central_entities(5);
            assert_eq!(most_central.len(), 1);
            assert_eq!(most_central[0], (entity1, 0.5));
        }
    }

    #[cfg(test)]
    mod brain_enhanced_config_tests {
        use super::*;

        #[test]
        fn test_default_config() {
            let config = BrainEnhancedConfig::default();
            assert_eq!(config.learning_rate, 0.1);
            assert_eq!(config.activation_threshold, 0.5);
            assert_eq!(config.max_activation_spread, 5);
            assert_eq!(config.neural_dampening, 0.95);
            assert_eq!(config.concept_coherence_threshold, 0.7);
            assert!(config.enable_hebbian_learning);
            assert!(config.enable_concept_formation);
            assert!(config.enable_neural_plasticity);
            assert_eq!(config.memory_consolidation_threshold, 0.8);
            assert_eq!(config.synaptic_strength_decay, 0.99);
            assert_eq!(config.embedding_dim, 384);
            assert!(config.enable_temporal_tracking);
            assert!(config.enable_sdr_storage);
        }

        #[test]
        fn test_for_testing_config() {
            let config = BrainEnhancedConfig::for_testing();
            assert_eq!(config.learning_rate, 0.2);
            assert_eq!(config.activation_threshold, 0.3);
            assert_eq!(config.max_activation_spread, 3);
            assert_eq!(config.neural_dampening, 0.9);
            assert_eq!(config.concept_coherence_threshold, 0.5);
            assert_eq!(config.memory_consolidation_threshold, 0.6);
            assert_eq!(config.synaptic_strength_decay, 0.95);
            assert_eq!(config.embedding_dim, 128);
            assert!(!config.enable_sdr_storage);
        }

        #[test]
        fn test_high_performance_config() {
            let config = BrainEnhancedConfig::high_performance();
            assert_eq!(config.learning_rate, 0.05);
            assert_eq!(config.activation_threshold, 0.7);
            assert_eq!(config.max_activation_spread, 3);
            assert_eq!(config.neural_dampening, 0.98);
            assert_eq!(config.concept_coherence_threshold, 0.8);
            assert!(!config.enable_concept_formation); // Disabled for performance
            assert_eq!(config.memory_consolidation_threshold, 0.9);
            assert_eq!(config.embedding_dim, 256);
        }

        #[test]
        fn test_exploratory_config() {
            let config = BrainEnhancedConfig::exploratory();
            assert_eq!(config.learning_rate, 0.15);
            assert_eq!(config.activation_threshold, 0.4);
            assert_eq!(config.max_activation_spread, 7);
            assert_eq!(config.neural_dampening, 0.9);
            assert_eq!(config.concept_coherence_threshold, 0.6);
            assert_eq!(config.memory_consolidation_threshold, 0.7);
            assert_eq!(config.synaptic_strength_decay, 0.98);
            assert_eq!(config.embedding_dim, 192);
        }

        #[test]
        fn test_validate_valid_config() {
            let config = BrainEnhancedConfig::default();
            assert!(config.validate().is_ok());
        }

        #[test]
        fn test_validate_invalid_learning_rate_zero() {
            let mut config = BrainEnhancedConfig::default();
            config.learning_rate = 0.0;
            assert!(config.validate().is_err());
            assert!(config.validate().unwrap_err().contains("Learning rate"));
        }

        #[test]
        fn test_validate_invalid_learning_rate_too_high() {
            let mut config = BrainEnhancedConfig::default();
            config.learning_rate = 1.5;
            assert!(config.validate().is_err());
            assert!(config.validate().unwrap_err().contains("Learning rate"));
        }

        #[test]
        fn test_validate_invalid_activation_threshold_negative() {
            let mut config = BrainEnhancedConfig::default();
            config.activation_threshold = -0.1;
            assert!(config.validate().is_err());
            assert!(config.validate().unwrap_err().contains("Activation threshold"));
        }

        #[test]
        fn test_validate_invalid_activation_threshold_too_high() {
            let mut config = BrainEnhancedConfig::default();
            config.activation_threshold = 1.1;
            assert!(config.validate().is_err());
            assert!(config.validate().unwrap_err().contains("Activation threshold"));
        }

        #[test]
        fn test_validate_invalid_max_activation_spread_zero() {
            let mut config = BrainEnhancedConfig::default();
            config.max_activation_spread = 0;
            assert!(config.validate().is_err());
            assert!(config.validate().unwrap_err().contains("Max activation spread"));
        }

        #[test]
        fn test_validate_invalid_neural_dampening_zero() {
            let mut config = BrainEnhancedConfig::default();
            config.neural_dampening = 0.0;
            assert!(config.validate().is_err());
            assert!(config.validate().unwrap_err().contains("Neural dampening"));
        }

        #[test]
        fn test_validate_invalid_neural_dampening_too_high() {
            let mut config = BrainEnhancedConfig::default();
            config.neural_dampening = 1.1;
            assert!(config.validate().is_err());
            assert!(config.validate().unwrap_err().contains("Neural dampening"));
        }

        #[test]
        fn test_validate_edge_case_values() {
            let mut config = BrainEnhancedConfig::default();
            config.learning_rate = 1.0; // Edge case: exactly 1.0
            config.activation_threshold = 0.0; // Edge case: exactly 0.0
            config.neural_dampening = 1.0; // Edge case: exactly 1.0
            assert!(config.validate().is_ok());
        }
    }

    #[cfg(test)]
    mod activation_config_tests {
        use super::*;

        #[test]
        fn test_default_activation_config() {
            let config = ActivationConfig::default();
            assert_eq!(config.default_threshold, 0.5);
            assert_eq!(config.max_iterations, 100);
            assert_eq!(config.decay_factor, 0.95);
        }
    }

    #[cfg(test)]
    mod graph_health_metrics_tests {
        use super::*;

        #[test]
        fn test_is_healthy_valid() {
            let metrics = GraphHealthMetrics {
                connectivity_score: 0.7,
                activation_balance: 0.6,
                learning_stability: 0.7,
                concept_coherence: 0.8,
                overall_health: 0.8,
            };
            assert!(metrics.is_healthy());
        }

        #[test]
        fn test_is_healthy_low_overall_health() {
            let metrics = GraphHealthMetrics {
                connectivity_score: 0.7,
                activation_balance: 0.6,
                learning_stability: 0.7,
                concept_coherence: 0.8,
                overall_health: 0.6, // Too low
            };
            assert!(!metrics.is_healthy());
        }

        #[test]
        fn test_is_healthy_low_connectivity() {
            let metrics = GraphHealthMetrics {
                connectivity_score: 0.5, // Too low
                activation_balance: 0.6,
                learning_stability: 0.7,
                concept_coherence: 0.8,
                overall_health: 0.8,
            };
            assert!(!metrics.is_healthy());
        }

        #[test]
        fn test_is_healthy_low_activation_balance() {
            let metrics = GraphHealthMetrics {
                connectivity_score: 0.7,
                activation_balance: 0.4, // Too low
                learning_stability: 0.7,
                concept_coherence: 0.8,
                overall_health: 0.8,
            };
            assert!(!metrics.is_healthy());
        }

        #[test]
        fn test_is_healthy_low_learning_stability() {
            let metrics = GraphHealthMetrics {
                connectivity_score: 0.7,
                activation_balance: 0.6,
                learning_stability: 0.5, // Too low
                concept_coherence: 0.8,
                overall_health: 0.8,
            };
            assert!(!metrics.is_healthy());
        }

        #[test]
        fn test_get_health_report_healthy() {
            let metrics = GraphHealthMetrics {
                connectivity_score: 0.7,
                activation_balance: 0.6,
                learning_stability: 0.7,
                concept_coherence: 0.8,
                overall_health: 0.8,
            };
            let report = metrics.get_health_report();
            assert!(report.contains("Overall Health: 0.80"));
            assert!(report.contains("Connectivity: 0.70"));
            assert!(report.contains("Activation Balance: 0.60"));
            assert!(report.contains("Learning Stability: 0.70"));
            assert!(report.contains("Concept Coherence: 0.80"));
            assert!(report.contains("Status: Healthy"));
        }

        #[test]
        fn test_get_health_report_unhealthy() {
            let metrics = GraphHealthMetrics {
                connectivity_score: 0.3,
                activation_balance: 0.2,
                learning_stability: 0.4,
                concept_coherence: 0.3,
                overall_health: 0.4,
            };
            let report = metrics.get_health_report();
            assert!(report.contains("Status: Needs Attention"));
        }
    }

    #[cfg(test)]
    mod activation_stats_tests {
        use super::*;

        #[test]
        fn test_activation_stats_creation() {
            let mut distribution = HashMap::new();
            distribution.insert("low".to_string(), 10);
            distribution.insert("high".to_string(), 5);

            let stats = ActivationStats {
                average: 0.5,
                maximum: 1.0,
                minimum: 0.1,
                range: 0.9,
                distribution,
            };

            assert_eq!(stats.average, 0.5);
            assert_eq!(stats.maximum, 1.0);
            assert_eq!(stats.minimum, 0.1);
            assert_eq!(stats.range, 0.9);
            assert_eq!(stats.distribution.get("low"), Some(&10));
            assert_eq!(stats.distribution.get("high"), Some(&5));
        }
    }

    #[cfg(test)]
    mod connectivity_stats_tests {
        use super::*;

        #[test]
        fn test_connectivity_stats_creation() {
            let stats = ConnectivityStats {
                entity_count: 100,
                relationship_count: 200,
                density: 0.4,
                clustering_coefficient: 0.6,
                average_path_length: 3.5,
            };

            assert_eq!(stats.entity_count, 100);
            assert_eq!(stats.relationship_count, 200);
            assert_eq!(stats.density, 0.4);
            assert_eq!(stats.clustering_coefficient, 0.6);
            assert_eq!(stats.average_path_length, 3.5);
        }
    }

    #[cfg(test)]
    mod enum_tests {
        use super::*;

        #[test]
        fn test_activation_pattern_variants() {
            let focused = ActivationPattern::Focused;
            let distributed = ActivationPattern::Distributed;
            let sparse = ActivationPattern::Sparse;
            let clustered = ActivationPattern::Clustered;

            // Test that enums can be created and matched
            match focused {
                ActivationPattern::Focused => assert!(true),
                _ => assert!(false),
            }

            match distributed {
                ActivationPattern::Distributed => assert!(true),
                _ => assert!(false),
            }

            match sparse {
                ActivationPattern::Sparse => assert!(true),
                _ => assert!(false),
            }

            match clustered {
                ActivationPattern::Clustered => assert!(true),
                _ => assert!(false),
            }
        }

        #[test]
        fn test_learning_mode_variants() {
            let supervised = LearningMode::Supervised;
            let unsupervised = LearningMode::Unsupervised;
            let reinforcement = LearningMode::Reinforcement;
            let hebbian = LearningMode::Hebbian;

            match supervised {
                LearningMode::Supervised => assert!(true),
                _ => assert!(false),
            }

            match unsupervised {
                LearningMode::Unsupervised => assert!(true),
                _ => assert!(false),
            }

            match reinforcement {
                LearningMode::Reinforcement => assert!(true),
                _ => assert!(false),
            }

            match hebbian {
                LearningMode::Hebbian => assert!(true),
                _ => assert!(false),
            }
        }

        #[test]
        fn test_query_mode_variants() {
            let exact = QueryMode::Exact;
            let fuzzy = QueryMode::Fuzzy;
            let associative = QueryMode::Associative;
            let conceptual = QueryMode::Conceptual;

            match exact {
                QueryMode::Exact => assert!(true),
                _ => assert!(false),
            }

            match fuzzy {
                QueryMode::Fuzzy => assert!(true),
                _ => assert!(false),
            }

            match associative {
                QueryMode::Associative => assert!(true),
                _ => assert!(false),
            }

            match conceptual {
                QueryMode::Conceptual => assert!(true),
                _ => assert!(false),
            }
        }
    }

    #[cfg(test)]
    mod integration_tests {
        use super::*;

        #[test]
        fn test_brain_query_result_with_concept_structure() {
            let mut query_result = BrainQueryResult::new();
            let mut concept = ConceptStructure::new();

            let entity1 = create_test_entity(1);
            let entity2 = create_test_entity(2);
            let entity3 = create_test_entity(3);

            // Add entities to concept
            concept.add_input(entity1);
            concept.add_output(entity2);
            concept.add_gate(entity3);

            // Add same entities to query result
            query_result.add_entity(entity1, 0.7);
            query_result.add_entity(entity2, 0.8);
            query_result.add_entity(entity3, 0.5);

            // Verify integration
            assert_eq!(query_result.entity_count(), concept.total_entities());
            
            let all_concept_entities = concept.get_all_entities();
            for entity in all_concept_entities {
                assert!(query_result.get_activation(&entity).is_some());
            }
        }

        #[test]
        fn test_config_validation_with_activation_config() {
            let mut config = BrainEnhancedConfig::default();
            
            // Test that activation config is properly embedded
            assert_eq!(config.activation_config.default_threshold, 0.5);
            assert_eq!(config.activation_config.max_iterations, 100);
            assert_eq!(config.activation_config.decay_factor, 0.95);

            // Modify activation config
            config.activation_config.default_threshold = 0.3;
            config.activation_config.max_iterations = 50;
            config.activation_config.decay_factor = 0.9;

            // Config should still be valid
            assert!(config.validate().is_ok());
        }

        #[test]
        fn test_statistics_health_integration() {
            let mut stats = BrainStatistics::new();
            
            // Set up stats for healthy graph
            stats.graph_density = 0.3;
            stats.clustering_coefficient = 0.5;
            stats.concept_coherence = 0.8;
            stats.learning_efficiency = 0.7;
            stats.average_path_length = 4.0;

            // Test health calculations
            assert!(stats.is_well_connected());
            assert!(stats.graph_health_score() > 0.5);

            // Create health metrics from stats
            let health_metrics = GraphHealthMetrics {
                connectivity_score: stats.graph_density,
                activation_balance: stats.clustering_coefficient,
                learning_stability: stats.learning_efficiency,
                concept_coherence: stats.concept_coherence,
                overall_health: stats.graph_health_score(),
            };

            assert!(health_metrics.is_healthy());
        }
    }
}