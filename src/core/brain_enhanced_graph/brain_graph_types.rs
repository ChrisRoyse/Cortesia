//! Type definitions for brain-enhanced knowledge graph

use crate::core::types::EntityKey;
use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};

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