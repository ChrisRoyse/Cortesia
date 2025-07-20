//! Core brain-enhanced knowledge graph structure

use super::brain_graph_types::*;
use crate::core::graph::KnowledgeGraph;
use crate::core::sdr_storage::SDRStorage;
use crate::core::sdr_types::SDRConfig;
use crate::core::types::{EntityKey, EntityData};
use crate::error::Result;
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;

/// Brain-enhanced knowledge graph that combines traditional graph with brain-like processing
pub struct BrainEnhancedKnowledgeGraph {
    /// Core knowledge graph for entity storage and relationships
    pub core_graph: Arc<KnowledgeGraph>,
    /// SDR storage for concept representation
    pub sdr_storage: Arc<SDRStorage>,
    /// Brain-specific configuration
    pub config: BrainEnhancedConfig,
    /// Entity activation levels
    pub entity_activations: RwLock<HashMap<EntityKey, f32>>,
    /// Relationship weights (separate from core graph for brain-specific weights)
    pub synaptic_weights: RwLock<HashMap<(EntityKey, EntityKey), f32>>,
    /// Concept structures
    pub concept_structures: RwLock<HashMap<String, ConceptStructure>>,
    /// Learning statistics
    pub learning_stats: RwLock<BrainStatistics>,
    /// Query cache for performance
    pub query_cache: RwLock<HashMap<String, BrainQueryResult>>,
}

impl std::fmt::Debug for BrainEnhancedKnowledgeGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrainEnhancedKnowledgeGraph")
            .field("core_graph", &"KnowledgeGraph")
            .field("sdr_storage", &"SDRStorage")
            .field("config", &self.config)
            .field("entity_activations", &"RwLock<HashMap>")
            .field("synaptic_weights", &"RwLock<HashMap>")
            .field("concept_structures", &"RwLock<HashMap>")
            .field("learning_stats", &"RwLock<BrainStatistics>")
            .field("query_cache", &"RwLock<HashMap>")
            .finish()
    }
}

impl BrainEnhancedKnowledgeGraph {
    /// Create new brain-enhanced knowledge graph
    pub fn new(embedding_dim: usize) -> Result<Self> {
        let core_graph = Arc::new(KnowledgeGraph::new(embedding_dim)?);
        let sdr_config = SDRConfig {
            total_bits: embedding_dim * 4,
            active_bits: (embedding_dim * 4) / 50, // 2% sparsity
            sparsity: 0.02,
            overlap_threshold: 0.5,
        };
        let sdr_storage = Arc::new(SDRStorage::new(sdr_config));
        
        Ok(Self {
            core_graph,
            sdr_storage,
            config: BrainEnhancedConfig::default(),
            entity_activations: RwLock::new(HashMap::new()),
            synaptic_weights: RwLock::new(HashMap::new()),
            concept_structures: RwLock::new(HashMap::new()),
            learning_stats: RwLock::new(BrainStatistics::new()),
            query_cache: RwLock::new(HashMap::new()),
        })
    }

    /// Create new brain-enhanced graph for testing
    pub fn new_for_test() -> Result<Self> {
        let core_graph = Arc::new(KnowledgeGraph::new(96)?);
        let sdr_storage = Arc::new(SDRStorage::new(SDRConfig::default()));
        
        Ok(Self {
            core_graph,
            sdr_storage,
            config: BrainEnhancedConfig::for_testing(),
            entity_activations: RwLock::new(HashMap::new()),
            synaptic_weights: RwLock::new(HashMap::new()),
            concept_structures: RwLock::new(HashMap::new()),
            learning_stats: RwLock::new(BrainStatistics::new()),
            query_cache: RwLock::new(HashMap::new()),
        })
    }

    /// Create new brain-enhanced graph asynchronously
    pub async fn new_async(embedding_dim: usize) -> Result<Self> {
        let core_graph = Arc::new(KnowledgeGraph::new(embedding_dim)?);
        let sdr_config = SDRConfig {
            total_bits: embedding_dim * 4,
            active_bits: (embedding_dim * 4) / 50, // 2% sparsity
            sparsity: 0.02,
            overlap_threshold: 0.5,
        };
        let sdr_storage = Arc::new(SDRStorage::new(sdr_config));
        
        Ok(Self {
            core_graph,
            sdr_storage,
            config: BrainEnhancedConfig::default(),
            entity_activations: RwLock::new(HashMap::new()),
            synaptic_weights: RwLock::new(HashMap::new()),
            concept_structures: RwLock::new(HashMap::new()),
            learning_stats: RwLock::new(BrainStatistics::new()),
            query_cache: RwLock::new(HashMap::new()),
        })
    }

    /// Create new brain-enhanced graph with custom configuration
    pub fn new_with_config(embedding_dim: usize, config: BrainEnhancedConfig) -> Result<Self> {
        // Validate configuration
        config.validate().map_err(|e| crate::error::GraphError::InvalidConfiguration(e))?;
        
        let core_graph = Arc::new(KnowledgeGraph::new(embedding_dim)?);
        let sdr_config = SDRConfig {
            total_bits: embedding_dim * 4,
            active_bits: (embedding_dim * 4) / 50, // 2% sparsity
            sparsity: 0.02,
            overlap_threshold: 0.5,
        };
        let sdr_storage = Arc::new(SDRStorage::new(sdr_config));
        
        Ok(Self {
            core_graph,
            sdr_storage,
            config,
            entity_activations: RwLock::new(HashMap::new()),
            synaptic_weights: RwLock::new(HashMap::new()),
            concept_structures: RwLock::new(HashMap::new()),
            learning_stats: RwLock::new(BrainStatistics::new()),
            query_cache: RwLock::new(HashMap::new()),
        })
    }

    /// Get entity count
    pub fn entity_count(&self) -> usize {
        self.core_graph.entity_count()
    }

    /// Get relationship count
    pub fn relationship_count(&self) -> usize {
        self.core_graph.relationship_count() as usize
    }

    /// Get embedding dimension
    pub fn embedding_dimension(&self) -> usize {
        self.core_graph.embedding_dimension()
    }

    /// Get current configuration
    pub fn get_config(&self) -> &BrainEnhancedConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: BrainEnhancedConfig) -> Result<()> {
        config.validate().map_err(|e| crate::error::GraphError::InvalidConfiguration(e))?;
        self.config = config;
        Ok(())
    }

    /// Get entity activation level
    pub async fn get_entity_activation(&self, entity: EntityKey) -> f32 {
        let activations = self.entity_activations.read().await;
        activations.get(&entity).copied().unwrap_or(0.0)
    }

    /// Set entity activation level
    pub async fn set_entity_activation(&self, entity: EntityKey, activation: f32) {
        let mut activations = self.entity_activations.write().await;
        activations.insert(entity, activation.clamp(0.0, 1.0));
    }

    /// Get synaptic weight between entities
    pub async fn get_synaptic_weight(&self, source: EntityKey, target: EntityKey) -> f32 {
        let weights = self.synaptic_weights.read().await;
        weights.get(&(source, target)).copied().unwrap_or(0.0)
    }

    /// Set synaptic weight between entities
    pub async fn set_synaptic_weight(&self, source: EntityKey, target: EntityKey, weight: f32) {
        let mut weights = self.synaptic_weights.write().await;
        weights.insert((source, target), weight.clamp(0.0, 1.0));
    }

    /// Get all entity activations
    pub async fn get_all_activations(&self) -> HashMap<EntityKey, f32> {
        let activations = self.entity_activations.read().await;
        activations.clone()
    }

    /// Clear all activations
    pub async fn clear_activations(&self) {
        let mut activations = self.entity_activations.write().await;
        activations.clear();
    }

    /// Reset graph state
    pub async fn reset(&self) {
        self.clear_activations().await;
        
        let mut weights = self.synaptic_weights.write().await;
        weights.clear();
        
        let mut concepts = self.concept_structures.write().await;
        concepts.clear();
        
        let mut cache = self.query_cache.write().await;
        cache.clear();
        
        let mut stats = self.learning_stats.write().await;
        *stats = BrainStatistics::new();
    }

    /// Get concept structure by name
    pub async fn get_concept_structure(&self, concept_name: &str) -> Option<ConceptStructure> {
        let concepts = self.concept_structures.read().await;
        concepts.get(concept_name).cloned()
    }

    /// Store concept structure
    pub async fn store_concept_structure(&self, concept_name: String, structure: ConceptStructure) {
        let mut concepts = self.concept_structures.write().await;
        concepts.insert(concept_name, structure);
    }

    /// Get all concept names
    pub async fn get_concept_names(&self) -> Vec<String> {
        let concepts = self.concept_structures.read().await;
        concepts.keys().cloned().collect()
    }

    /// Remove concept structure
    pub async fn remove_concept_structure(&self, concept_name: &str) -> bool {
        let mut concepts = self.concept_structures.write().await;
        concepts.remove(concept_name).is_some()
    }

    /// Get learning statistics
    pub async fn get_learning_stats(&self) -> BrainStatistics {
        let stats = self.learning_stats.read().await;
        stats.clone()
    }

    /// Update learning statistics
    pub async fn update_learning_stats<F>(&self, updater: F) 
    where
        F: FnOnce(&mut BrainStatistics),
    {
        let mut stats = self.learning_stats.write().await;
        updater(&mut stats);
    }

    /// Check if entity exists
    pub fn contains_entity(&self, entity: EntityKey) -> bool {
        self.core_graph.contains_entity_key(entity)
    }

    /// Get entity data
    pub fn get_entity_data(&self, entity: EntityKey) -> Option<EntityData> {
        self.core_graph.get_entity_data(entity)
    }

    /// Get all entity keys
    pub fn get_all_entity_keys(&self) -> Vec<EntityKey> {
        self.core_graph.get_all_entity_keys()
    }

    /// Get neighbors of entity
    pub fn get_neighbors(&self, entity: EntityKey) -> Vec<EntityKey> {
        self.core_graph.get_neighbors(entity)
    }

    /// Get memory usage
    pub async fn get_memory_usage(&self) -> BrainMemoryUsage {
        let core_usage = self.core_graph.memory_usage();
        let sdr_usage = self.sdr_storage.memory_usage().await;
        
        BrainMemoryUsage {
            core_graph_bytes: core_usage.total_bytes(),
            sdr_storage_bytes: sdr_usage,
            activation_bytes: std::mem::size_of::<HashMap<EntityKey, f32>>() * 1000, // Estimate
            synaptic_weights_bytes: std::mem::size_of::<HashMap<(EntityKey, EntityKey), f32>>() * 1000, // Estimate
            concept_structures_bytes: std::mem::size_of::<HashMap<String, ConceptStructure>>() * 100, // Estimate
            total_bytes: 0, // Will be calculated
        }
    }

    /// Validate graph consistency
    pub async fn validate_consistency(&self) -> Vec<String> {
        let mut issues = Vec::new();
        
        // Check activation consistency
        let activations = self.entity_activations.read().await;
        for (entity, activation) in activations.iter() {
            if !self.contains_entity(*entity) {
                issues.push(format!("Activation found for non-existent entity: {:?}", entity));
            }
            
            if *activation < 0.0 || *activation > 1.0 {
                issues.push(format!("Invalid activation value {} for entity {:?}", activation, entity));
            }
        }
        
        // Check synaptic weight consistency
        let weights = self.synaptic_weights.read().await;
        for ((source, target), weight) in weights.iter() {
            if !self.contains_entity(*source) {
                issues.push(format!("Synaptic weight found for non-existent source entity: {:?}", source));
            }
            
            if !self.contains_entity(*target) {
                issues.push(format!("Synaptic weight found for non-existent target entity: {:?}", target));
            }
            
            if *weight < 0.0 || *weight > 1.0 {
                issues.push(format!("Invalid synaptic weight {} between {:?} and {:?}", weight, source, target));
            }
        }
        
        // Check concept structure consistency
        let concepts = self.concept_structures.read().await;
        for (concept_name, structure) in concepts.iter() {
            for entity in structure.get_all_entities() {
                if !self.contains_entity(entity) {
                    issues.push(format!("Concept '{}' references non-existent entity: {:?}", concept_name, entity));
                }
            }
        }
        
        issues
    }

    /// Get graph health metrics
    pub async fn get_health_metrics(&self) -> GraphHealthMetrics {
        let stats = self.learning_stats.read().await;
        let entity_count = self.entity_count();
        let relationship_count = self.relationship_count();
        
        // Calculate connectivity score
        let connectivity_score = if entity_count > 0 {
            (relationship_count as f32 / entity_count as f32).min(1.0)
        } else {
            0.0
        };
        
        // Calculate activation balance
        let activations = self.entity_activations.read().await;
        let activation_balance = if !activations.is_empty() {
            let total_activation: f32 = activations.values().sum();
            let avg_activation = total_activation / activations.len() as f32;
            1.0 - (avg_activation - 0.5).abs() * 2.0 // Closer to 0.5 is better balance
        } else {
            0.0
        };
        
        // Use stats values for other metrics
        let learning_stability = stats.learning_efficiency;
        let concept_coherence = stats.concept_coherence;
        
        let overall_health = (connectivity_score + activation_balance + learning_stability + concept_coherence) / 4.0;
        
        GraphHealthMetrics {
            connectivity_score,
            activation_balance,
            learning_stability,
            concept_coherence,
            overall_health,
        }
    }

    /// Clear query cache
    pub async fn clear_query_cache(&self) {
        let mut cache = self.query_cache.write().await;
        cache.clear();
    }

    /// Get query cache statistics
    pub async fn get_cache_stats(&self) -> (usize, usize) {
        let cache = self.query_cache.read().await;
        (cache.len(), cache.capacity())
    }
}

/// Memory usage information for brain-enhanced graph
#[derive(Debug, Clone)]
pub struct BrainMemoryUsage {
    pub core_graph_bytes: usize,
    pub sdr_storage_bytes: usize,
    pub activation_bytes: usize,
    pub synaptic_weights_bytes: usize,
    pub concept_structures_bytes: usize,
    pub total_bytes: usize,
}

impl BrainMemoryUsage {
    /// Calculate total memory usage
    pub fn calculate_total(&mut self) {
        self.total_bytes = self.core_graph_bytes + 
                          self.sdr_storage_bytes + 
                          self.activation_bytes + 
                          self.synaptic_weights_bytes + 
                          self.concept_structures_bytes;
    }
    
    /// Get memory usage breakdown
    pub fn get_breakdown(&self) -> HashMap<String, f64> {
        let mut breakdown = HashMap::new();
        let total = self.total_bytes as f64;
        
        if total > 0.0 {
            breakdown.insert("core_graph".to_string(), (self.core_graph_bytes as f64 / total) * 100.0);
            breakdown.insert("sdr_storage".to_string(), (self.sdr_storage_bytes as f64 / total) * 100.0);
            breakdown.insert("activations".to_string(), (self.activation_bytes as f64 / total) * 100.0);
            breakdown.insert("synaptic_weights".to_string(), (self.synaptic_weights_bytes as f64 / total) * 100.0);
            breakdown.insert("concept_structures".to_string(), (self.concept_structures_bytes as f64 / total) * 100.0);
        }
        
        breakdown
    }
}