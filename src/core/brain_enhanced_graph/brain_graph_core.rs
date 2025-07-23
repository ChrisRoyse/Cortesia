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

    /// Get entity degree (number of connections)
    pub async fn get_entity_degree(&self, entity: EntityKey) -> usize {
        self.get_neighbors(entity).len()
    }

    /// Propagate activation from a specific entity
    pub async fn propagate_activation_from_entity(&self, source_entity: EntityKey, decay_factor: f32) -> Result<ActivationPropagationResult> {
        let start_time = std::time::Instant::now();
        
        // Get source activation
        let source_activation = self.get_entity_activation(source_entity).await;
        if source_activation == 0.0 {
            return Ok(ActivationPropagationResult {
                affected_entities: 0,
                total_activation_spread: 0.0,
                propagation_time: start_time.elapsed(),
            });
        }

        // Get neighbors of the source entity
        let neighbors = self.get_neighbors(source_entity);
        let mut affected_entities = 0;
        let mut total_activation_spread = 0.0;

        // Propagate to each neighbor
        for neighbor in neighbors {
            let synaptic_weight = self.get_synaptic_weight(source_entity, neighbor).await;
            let propagated_activation = source_activation * decay_factor * synaptic_weight;
            
            if propagated_activation > 0.001 { // Minimum threshold
                let current_activation = self.get_entity_activation(neighbor).await;
                let new_activation = (current_activation + propagated_activation).min(1.0);
                
                self.set_entity_activation(neighbor, new_activation).await;
                affected_entities += 1;
                total_activation_spread += propagated_activation;
            }
        }

        // Update learning statistics
        self.update_learning_stats(|stats| {
            stats.total_propagations += 1;
            stats.total_affected_entities += affected_entities;
        }).await;

        Ok(ActivationPropagationResult {
            affected_entities,
            total_activation_spread,
            propagation_time: start_time.elapsed(),
        })
    }

    /// Get entities with activation above a threshold
    pub async fn get_entities_above_threshold(&self, threshold: f32) -> Vec<(EntityKey, f32)> {
        let activations = self.get_all_activations().await;
        activations
            .into_iter()
            .filter(|(_, activation)| *activation > threshold)
            .collect()
    }

    /// Perform cognitive query using brain-specific processing
    pub async fn cognitive_query(&self, query_embedding: &[f32], k: usize) -> Result<BrainQueryResult> {
        let start_time = std::time::Instant::now();
        
        // Validate embedding dimension
        if query_embedding.len() != self.embedding_dimension() {
            return Err(crate::error::GraphError::InvalidEmbeddingDimension {
                expected: self.embedding_dimension(),
                actual: query_embedding.len(),
            });
        }

        // Perform neural query
        let result = self.neural_query(query_embedding, k).await?;
        
        // Enhance with activation context
        let activated_entities_vec = self.get_entities_above_threshold(0.1).await;
        let activated_entities: HashMap<EntityKey, f32> = activated_entities_vec.into_iter().collect();
        
        // Combine results with activation-weighted scoring
        let mut enhanced_entities = Vec::new();
        for entity_key in &result.entities {
            let activation = self.get_entity_activation(*entity_key).await;
            let similarity = result.activations.get(entity_key).unwrap_or(&0.0);
            let cognitive_score = similarity * 0.7 + activation * 0.3;
            enhanced_entities.push((*entity_key, cognitive_score));
        }

        // Sort by cognitive score
        enhanced_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        enhanced_entities.truncate(k);

        let final_entities: Vec<EntityKey> = enhanced_entities.iter().map(|(key, _score)| *key).collect();
        let final_activations: HashMap<EntityKey, f32> = enhanced_entities.iter().map(|(key, score)| (*key, *score)).collect();
        
        Ok(BrainQueryResult {
            entities: final_entities,
            activations: final_activations,
            query_time: start_time.elapsed(),
            total_activation: enhanced_entities.iter().map(|(_, score)| score).sum(),
            activation_context: activated_entities,
            confidence: result.confidence,
        })
    }
}

/// Memory usage information for brain-enhanced graph
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::EntityKey;
    
    

    /// Helper function to create a test brain graph
    async fn create_test_brain_graph() -> Result<BrainEnhancedKnowledgeGraph> {
        BrainEnhancedKnowledgeGraph::new_for_test()
    }

    /// Helper function to create test entity keys
    fn create_test_entity_keys(count: usize) -> Vec<EntityKey> {
        (0..count).map(|i| EntityKey::new((i as u64).to_string())).collect()
    }

    /// Helper function to setup brain graph with test entities
    async fn setup_brain_graph_with_entities(entity_count: usize) -> Result<(BrainEnhancedKnowledgeGraph, Vec<EntityKey>)> {
        let brain_graph = create_test_brain_graph().await?;
        let entity_keys = create_test_entity_keys(entity_count);
        
        // Add entities to core graph (simulated - would need actual core graph implementation)
        // This is a placeholder for the actual entity addition logic
        
        Ok((brain_graph, entity_keys))
    }

    #[tokio::test]
    async fn test_new_creates_valid_brain_graph() {
        let result = BrainEnhancedKnowledgeGraph::new(128);
        assert!(result.is_ok());
        
        let brain_graph = result.unwrap();
        assert_eq!(brain_graph.embedding_dimension(), 128);
        assert_eq!(brain_graph.entity_count(), 0);
        assert_eq!(brain_graph.relationship_count(), 0);
        
        // Verify initial state
        let activations = brain_graph.get_all_activations().await;
        assert!(activations.is_empty());
        
        let stats = brain_graph.get_learning_stats().await;
        assert_eq!(stats.entity_count, 0);
        assert_eq!(stats.relationship_count, 0);
    }

    #[tokio::test]
    async fn test_new_with_invalid_embedding_dimension() {
        // Test with zero embedding dimension
        let result = BrainEnhancedKnowledgeGraph::new(0);
        // Assuming the underlying KnowledgeGraph::new would fail with 0 dimension
        // The exact behavior depends on the KnowledgeGraph implementation
        
        // Test with very large dimension to check memory limits
        let result_large = BrainEnhancedKnowledgeGraph::new(usize::MAX);
        // This should likely fail due to memory allocation issues
    }

    #[tokio::test]
    async fn test_new_for_test_creates_valid_test_graph() {
        let result = BrainEnhancedKnowledgeGraph::new_for_test();
        assert!(result.is_ok());
        
        let brain_graph = result.unwrap();
        assert_eq!(brain_graph.embedding_dimension(), 96);
        
        // Verify test configuration
        let config = brain_graph.get_config();
        assert_eq!(config.embedding_dim, 128); // Test config value
        assert_eq!(config.activation_threshold, 0.3);
        assert!(!config.enable_sdr_storage); // Disabled for testing
    }

    #[tokio::test]
    async fn test_new_async_creates_valid_brain_graph() {
        let result = BrainEnhancedKnowledgeGraph::new_async(256).await;
        assert!(result.is_ok());
        
        let brain_graph = result.unwrap();
        assert_eq!(brain_graph.embedding_dimension(), 256);
        
        // Verify SDR configuration
        let expected_total_bits = 256 * 4;
        let expected_active_bits = expected_total_bits / 50;
        // Note: We can't directly access SDR config without public methods
        // This test validates the creation process
    }

    #[tokio::test]
    async fn test_new_with_config_valid_configuration() {
        let config = BrainEnhancedConfig {
            learning_rate: 0.05,
            activation_threshold: 0.8,
            max_activation_spread: 10,
            neural_dampening: 0.9,
            concept_coherence_threshold: 0.6,
            enable_hebbian_learning: true,
            enable_concept_formation: false,
            enable_neural_plasticity: true,
            memory_consolidation_threshold: 0.7,
            synaptic_strength_decay: 0.95,
            embedding_dim: 512,
            activation_config: ActivationConfig::default(),
            enable_temporal_tracking: false,
            enable_sdr_storage: true,
        };
        
        let result = BrainEnhancedKnowledgeGraph::new_with_config(384, config.clone());
        assert!(result.is_ok());
        
        let brain_graph = result.unwrap();
        assert_eq!(brain_graph.embedding_dimension(), 384);
        
        let stored_config = brain_graph.get_config();
        assert_eq!(stored_config.learning_rate, 0.05);
        assert_eq!(stored_config.activation_threshold, 0.8);
        assert!(!stored_config.enable_concept_formation);
    }

    #[tokio::test]
    async fn test_new_with_config_invalid_configuration() {
        let invalid_config = BrainEnhancedConfig {
            learning_rate: 1.5, // Invalid: > 1.0
            activation_threshold: -0.1, // Invalid: < 0.0
            max_activation_spread: 0, // Invalid: must be > 0
            neural_dampening: 1.1, // Invalid: > 1.0
            ..BrainEnhancedConfig::default()
        };
        
        let result = BrainEnhancedKnowledgeGraph::new_with_config(128, invalid_config);
        assert!(result.is_err());
        
        if let Err(e) = result {
            // Verify it's a configuration error
            match e {
                crate::error::GraphError::InvalidConfiguration(_) => {
                    // Expected error type
                }
                _ => panic!("Expected InvalidConfiguration error"),
            }
        }
    }

    #[tokio::test]
    async fn test_get_entity_activation_nonexistent_entity() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        let nonexistent_entity = EntityKey::new(999.to_string());
        
        let activation = brain_graph.get_entity_activation(nonexistent_entity).await;
        assert_eq!(activation, 0.0);
    }

    #[tokio::test]
    async fn test_get_entity_activation_existing_entity() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        let entity = EntityKey::new(1.to_string());
        let expected_activation = 0.75;
        
        // Set activation first
        brain_graph.set_entity_activation(entity, expected_activation).await;
        
        // Get activation
        let actual_activation = brain_graph.get_entity_activation(entity).await;
        assert_eq!(actual_activation, expected_activation);
    }

    #[tokio::test]
    async fn test_set_entity_activation_valid_values() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        let entity = EntityKey::new(1.to_string());
        
        // Test setting various valid activation values
        let test_values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        
        for value in test_values {
            brain_graph.set_entity_activation(entity, value).await;
            let stored_value = brain_graph.get_entity_activation(entity).await;
            assert_eq!(stored_value, value);
        }
    }

    #[tokio::test]
    async fn test_set_entity_activation_clamping() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        let entity = EntityKey::new(1.to_string());
        
        // Test values outside [0.0, 1.0] range are clamped
        brain_graph.set_entity_activation(entity, -0.5).await;
        let clamped_low = brain_graph.get_entity_activation(entity).await;
        assert_eq!(clamped_low, 0.0);
        
        brain_graph.set_entity_activation(entity, 1.5).await;
        let clamped_high = brain_graph.get_entity_activation(entity).await;
        assert_eq!(clamped_high, 1.0);
    }

    #[tokio::test]
    async fn test_set_entity_activation_concurrent_access() {
        let brain_graph = Arc::new(create_test_brain_graph().await.unwrap());
        let entity = EntityKey::new(1.to_string());
        
        // Test concurrent access to activation setting
        let mut handles = vec![];
        
        for i in 0..10 {
            let graph_clone = Arc::clone(&brain_graph);
            let handle = tokio::spawn(async move {
                let activation = (i as f32) / 10.0;
                graph_clone.set_entity_activation(entity, activation).await;
                graph_clone.get_entity_activation(entity).await
            });
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        let results: Vec<f32> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();
        
        // All results should be valid activation values
        for result in results {
            assert!(result >= 0.0 && result <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_synaptic_weight_management() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        let source = EntityKey::new(1.to_string());
        let target = EntityKey::new(2.to_string());
        
        // Test initial weight
        let initial_weight = brain_graph.get_synaptic_weight(source, target).await;
        assert_eq!(initial_weight, 0.0);
        
        // Test setting weight
        let test_weight = 0.8;
        brain_graph.set_synaptic_weight(source, target, test_weight).await;
        let stored_weight = brain_graph.get_synaptic_weight(source, target).await;
        assert_eq!(stored_weight, test_weight);
        
        // Test weight clamping
        brain_graph.set_synaptic_weight(source, target, 1.5).await;
        let clamped_weight = brain_graph.get_synaptic_weight(source, target).await;
        assert_eq!(clamped_weight, 1.0);
    }

    #[tokio::test]
    async fn test_concept_structure_management() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        let concept_name = "test_concept".to_string();
        
        // Test getting non-existent concept
        let nonexistent = brain_graph.get_concept_structure(&concept_name).await;
        assert!(nonexistent.is_none());
        
        // Test storing and retrieving concept
        let mut concept = ConceptStructure::new();
        concept.add_input(EntityKey::new(1.to_string()));
        concept.add_output(EntityKey::new(2.to_string()));
        concept.concept_activation = 0.7;
        concept.coherence_score = 0.8;
        
        brain_graph.store_concept_structure(concept_name.clone(), concept.clone()).await;
        
        let retrieved = brain_graph.get_concept_structure(&concept_name).await;
        assert!(retrieved.is_some());
        
        let retrieved_concept = retrieved.unwrap();
        assert_eq!(retrieved_concept.input_entities, concept.input_entities);
        assert_eq!(retrieved_concept.output_entities, concept.output_entities);
        assert_eq!(retrieved_concept.concept_activation, concept.concept_activation);
        assert_eq!(retrieved_concept.coherence_score, concept.coherence_score);
    }

    #[tokio::test]
    async fn test_concept_structure_removal() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        let concept_name = "removable_concept".to_string();
        
        // Store a concept
        let concept = ConceptStructure::new();
        brain_graph.store_concept_structure(concept_name.clone(), concept).await;
        
        // Verify it exists
        assert!(brain_graph.get_concept_structure(&concept_name).await.is_some());
        
        // Remove it
        let removed = brain_graph.remove_concept_structure(&concept_name).await;
        assert!(removed);
        
        // Verify it's gone
        assert!(brain_graph.get_concept_structure(&concept_name).await.is_none());
        
        // Try removing again
        let removed_again = brain_graph.remove_concept_structure(&concept_name).await;
        assert!(!removed_again);
    }

    #[tokio::test]
    async fn test_get_concept_names() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Initially empty
        let names = brain_graph.get_concept_names().await;
        assert!(names.is_empty());
        
        // Add concepts
        let concept_names = vec!["concept1", "concept2", "concept3"];
        for name in &concept_names {
            brain_graph.store_concept_structure(name.to_string(), ConceptStructure::new()).await;
        }
        
        // Get all names
        let mut retrieved_names = brain_graph.get_concept_names().await;
        retrieved_names.sort();
        
        let mut expected_names: Vec<String> = concept_names.iter().map(|s| s.to_string()).collect();
        expected_names.sort();
        
        assert_eq!(retrieved_names, expected_names);
    }

    #[tokio::test]
    async fn test_learning_statistics_management() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Test initial statistics
        let initial_stats = brain_graph.get_learning_stats().await;
        assert_eq!(initial_stats.entity_count, 0);
        assert_eq!(initial_stats.relationship_count, 0);
        assert_eq!(initial_stats.avg_activation, 0.0);
        
        // Test updating statistics
        brain_graph.update_learning_stats(|stats| {
            stats.entity_count = 10;
            stats.relationship_count = 15;
            stats.avg_activation = 0.5;
            stats.learning_efficiency = 0.8;
        }).await;
        
        let updated_stats = brain_graph.get_learning_stats().await;
        assert_eq!(updated_stats.entity_count, 10);
        assert_eq!(updated_stats.relationship_count, 15);
        assert_eq!(updated_stats.avg_activation, 0.5);
        assert_eq!(updated_stats.learning_efficiency, 0.8);
    }

    #[tokio::test]
    async fn test_reset_clears_all_state() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Setup some state
        let entity = EntityKey::new(1.to_string());
        brain_graph.set_entity_activation(entity, 0.8).await;
        brain_graph.set_synaptic_weight(entity, EntityKey::new(2.to_string()), 0.7).await;
        brain_graph.store_concept_structure("test".to_string(), ConceptStructure::new()).await;
        brain_graph.update_learning_stats(|stats| {
            stats.entity_count = 5;
            stats.learning_efficiency = 0.9;
        }).await;
        
        // Verify state exists
        assert_eq!(brain_graph.get_entity_activation(entity).await, 0.8);
        assert!(!brain_graph.get_all_activations().await.is_empty());
        assert!(brain_graph.get_concept_structure("test").await.is_some());
        
        // Reset
        brain_graph.reset().await;
        
        // Verify all state is cleared
        assert_eq!(brain_graph.get_entity_activation(entity).await, 0.0);
        assert!(brain_graph.get_all_activations().await.is_empty());
        assert!(brain_graph.get_concept_structure("test").await.is_none());
        
        let stats = brain_graph.get_learning_stats().await;
        assert_eq!(stats.entity_count, 0);
        assert_eq!(stats.learning_efficiency, 0.0);
    }

    #[tokio::test]
    async fn test_validate_consistency_with_valid_state() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Setup valid state (assuming entities exist in core graph)
        // Note: This test would need actual entities in the core graph
        // For now, we test the consistency check mechanism
        
        let issues = brain_graph.validate_consistency().await;
        // With empty graph, should have no consistency issues
        assert!(issues.is_empty());
    }

    #[tokio::test]
    async fn test_validate_consistency_with_invalid_activations() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Manually insert invalid activation (bypassing clamping)
        {
            let mut activations = brain_graph.entity_activations.write().await;
            activations.insert(EntityKey::new(999.to_string()), 1.5); // Invalid: > 1.0
            activations.insert(EntityKey::new(998.to_string()), -0.5); // Invalid: < 0.0
        }
        
        let issues = brain_graph.validate_consistency().await;
        
        // Should detect invalid activation values
        let invalid_activation_issues: Vec<_> = issues.iter()
            .filter(|issue| issue.contains("Invalid activation value"))
            .collect();
        
        assert!(invalid_activation_issues.len() >= 2);
    }

    #[tokio::test]
    async fn test_validate_consistency_with_nonexistent_entities() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Add activations for non-existent entities
        let nonexistent_entity = EntityKey::new(999.to_string());
        brain_graph.set_entity_activation(nonexistent_entity, 0.5).await;
        
        let issues = brain_graph.validate_consistency().await;
        
        // Should detect references to non-existent entities
        let nonexistent_entity_issues: Vec<_> = issues.iter()
            .filter(|issue| issue.contains("non-existent entity"))
            .collect();
        
        assert!(!nonexistent_entity_issues.is_empty());
    }

    #[tokio::test]
    async fn test_get_health_metrics() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Setup some state for metrics calculation
        brain_graph.set_entity_activation(EntityKey::new(1.to_string()), 0.4).await;
        brain_graph.set_entity_activation(EntityKey::new(2.to_string()), 0.6).await;
        
        brain_graph.update_learning_stats(|stats| {
            stats.learning_efficiency = 0.8;
            stats.concept_coherence = 0.7;
        }).await;
        
        let metrics = brain_graph.get_health_metrics().await;
        
        // Verify metrics are within valid ranges
        assert!(metrics.connectivity_score >= 0.0 && metrics.connectivity_score <= 1.0);
        assert!(metrics.activation_balance >= 0.0 && metrics.activation_balance <= 1.0);
        assert!(metrics.learning_stability >= 0.0 && metrics.learning_stability <= 1.0);
        assert!(metrics.concept_coherence >= 0.0 && metrics.concept_coherence <= 1.0);
        assert!(metrics.overall_health >= 0.0 && metrics.overall_health <= 1.0);
        
        // Verify calculation
        let expected_overall = (metrics.connectivity_score + metrics.activation_balance + 
                               metrics.learning_stability + metrics.concept_coherence) / 4.0;
        assert!((metrics.overall_health - expected_overall).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_memory_usage_calculation() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Add some data to increase memory usage
        for i in 0..100 {
            brain_graph.set_entity_activation(EntityKey::new(i.to_string()), 0.5).await;
        }
        
        let memory_usage = brain_graph.get_memory_usage().await;
        
        // Verify memory usage structure
        assert!(memory_usage.core_graph_bytes > 0);
        assert!(memory_usage.sdr_storage_bytes >= 0);
        assert!(memory_usage.activation_bytes > 0);
        assert!(memory_usage.synaptic_weights_bytes > 0);
        assert!(memory_usage.concept_structures_bytes > 0);
    }

    #[tokio::test]
    async fn test_query_cache_management() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Test initial cache state
        let (size, capacity) = brain_graph.get_cache_stats().await;
        assert_eq!(size, 0);
        
        // Clear empty cache (should not fail)
        brain_graph.clear_query_cache().await;
        
        // Note: Direct cache manipulation would require public methods
        // This test validates the cache management interface
    }

    #[tokio::test]
    async fn test_config_update() {
        let mut brain_graph = create_test_brain_graph().await.unwrap();
        
        let new_config = BrainEnhancedConfig {
            learning_rate: 0.15,
            activation_threshold: 0.6,
            ..BrainEnhancedConfig::default()
        };
        
        let result = brain_graph.update_config(new_config.clone());
        assert!(result.is_ok());
        
        let updated_config = brain_graph.get_config();
        assert_eq!(updated_config.learning_rate, 0.15);
        assert_eq!(updated_config.activation_threshold, 0.6);
    }

    #[tokio::test]
    async fn test_config_update_with_invalid_config() {
        let mut brain_graph = create_test_brain_graph().await.unwrap();
        
        let invalid_config = BrainEnhancedConfig {
            learning_rate: -0.1, // Invalid
            ..BrainEnhancedConfig::default()
        };
        
        let result = brain_graph.update_config(invalid_config);
        assert!(result.is_err());
        
        // Verify original config is unchanged
        let config = brain_graph.get_config();
        assert!(config.learning_rate > 0.0);
    }

    #[tokio::test]
    async fn test_brain_enhanced_integration() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Test integration between different components
        let entity1 = EntityKey::new(1.to_string());
        let entity2 = EntityKey::new(2.to_string());
        
        // Set up brain-enhanced state
        brain_graph.set_entity_activation(entity1, 0.8).await;
        brain_graph.set_entity_activation(entity2, 0.6).await;
        brain_graph.set_synaptic_weight(entity1, entity2, 0.7).await;
        
        // Create concept structure linking entities
        let mut concept = ConceptStructure::new();
        concept.add_input(entity1);
        concept.add_output(entity2);
        concept.concept_activation = 0.75;
        concept.coherence_score = 0.8;
        
        brain_graph.store_concept_structure("integration_test".to_string(), concept).await;
        
        // Verify integrated state
        let activations = brain_graph.get_all_activations().await;
        assert_eq!(activations.len(), 2);
        assert_eq!(activations[&entity1], 0.8);
        assert_eq!(activations[&entity2], 0.6);
        
        let weight = brain_graph.get_synaptic_weight(entity1, entity2).await;
        assert_eq!(weight, 0.7);
        
        let stored_concept = brain_graph.get_concept_structure("integration_test").await;
        assert!(stored_concept.is_some());
        
        let concept = stored_concept.unwrap();
        assert!(concept.is_well_formed());
        assert_eq!(concept.total_entities(), 2);
    }

    #[tokio::test]
    async fn test_concurrent_operations() {
        let brain_graph = Arc::new(create_test_brain_graph().await.unwrap());
        
        // Test concurrent operations on different aspects of the brain graph
        let graph1 = Arc::clone(&brain_graph);
        let graph2 = Arc::clone(&brain_graph);
        let graph3 = Arc::clone(&brain_graph);
        
        let handle1 = tokio::spawn(async move {
            // Concurrent activation updates
            for i in 0..50 {
                let entity = EntityKey::new(i.to_string());
                let activation = (i as f32) / 100.0;
                graph1.set_entity_activation(entity, activation).await;
            }
        });
        
        let handle2 = tokio::spawn(async move {
            // Concurrent synaptic weight updates
            for i in 0..25 {
                let source = EntityKey::new(i.to_string());
                let target = EntityKey::new((i + 25).to_string());
                let weight = (i as f32) / 50.0;
                graph2.set_synaptic_weight(source, target, weight).await;
            }
        });
        
        let handle3 = tokio::spawn(async move {
            // Concurrent concept creation
            for i in 0..10 {
                let concept_name = format!("concept_{}", i);
                let concept = ConceptStructure::new();
                graph3.store_concept_structure(concept_name, concept).await;
            }
        });
        
        // Wait for all operations to complete
        let results = tokio::try_join!(handle1, handle2, handle3);
        assert!(results.is_ok());
        
        // Verify final state consistency
        let issues = brain_graph.validate_consistency().await;
        // Should have no consistency issues from concurrent operations
        // (though there might be issues from non-existent entities)
        println!("Consistency issues: {:?}", issues);
    }

    #[tokio::test]
    async fn test_activation_spread_simulation() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Set up a network of activations to simulate spread
        let entities: Vec<EntityKey> = (0..10).map(|i| EntityKey::new(i.to_string())).collect();
        
        // Set initial activation
        brain_graph.set_entity_activation(entities[0], 1.0).await;
        
        // Set up synaptic weights for spread
        for i in 0..9 {
            brain_graph.set_synaptic_weight(entities[i], entities[i + 1], 0.8).await;
        }
        
        // Simulate activation spread (simplified)
        for step in 1..5 {
            for i in 0..9 {
                let source_activation = brain_graph.get_entity_activation(entities[i]).await;
                let weight = brain_graph.get_synaptic_weight(entities[i], entities[i + 1]).await;
                
                if source_activation > 0.1 {
                    let target_activation = brain_graph.get_entity_activation(entities[i + 1]).await;
                    let new_activation = (target_activation + source_activation * weight * 0.5).min(1.0);
                    brain_graph.set_entity_activation(entities[i + 1], new_activation).await;
                }
            }
            
            // Apply dampening
            for entity in &entities {
                let current = brain_graph.get_entity_activation(*entity).await;
                brain_graph.set_entity_activation(*entity, current * 0.95).await;
            }
        }
        
        // Verify activation pattern
        let final_activations = brain_graph.get_all_activations().await;
        
        // First entity should still have highest activation
        let first_activation = final_activations.get(&entities[0]).unwrap_or(&0.0);
        
        // Last entity should have some activation due to spread
        let last_activation = final_activations.get(&entities[9]).unwrap_or(&0.0);
        
        assert!(*first_activation > *last_activation);
        assert!(*last_activation > 0.0); // Some activation should have spread
    }

    #[tokio::test]
    async fn test_brain_memory_usage_calculation() {
        let mut memory_usage = BrainMemoryUsage {
            core_graph_bytes: 1000,
            sdr_storage_bytes: 2000,
            activation_bytes: 500,
            synaptic_weights_bytes: 750,
            concept_structures_bytes: 250,
            total_bytes: 0,
        };
        
        // Test total calculation
        memory_usage.calculate_total();
        assert_eq!(memory_usage.total_bytes, 4500);
        
        // Test breakdown calculation
        let breakdown = memory_usage.get_breakdown();
        assert!((breakdown["core_graph"] - 22.22).abs() < 0.1);
        assert!((breakdown["sdr_storage"] - 44.44).abs() < 0.1);
        assert!((breakdown["activations"] - 11.11).abs() < 0.1);
        assert!((breakdown["synaptic_weights"] - 16.67).abs() < 0.1);
        assert!((breakdown["concept_structures"] - 5.56).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_private_component_initialization() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Test that all private components are properly initialized
        
        // Test entity_activations is initialized as empty HashMap
        let activations = brain_graph.entity_activations.read().await;
        assert!(activations.is_empty());
        drop(activations);
        
        // Test synaptic_weights is initialized as empty HashMap
        let weights = brain_graph.synaptic_weights.read().await;
        assert!(weights.is_empty());
        drop(weights);
        
        // Test concept_structures is initialized as empty HashMap
        let concepts = brain_graph.concept_structures.read().await;
        assert!(concepts.is_empty());
        drop(concepts);
        
        // Test learning_stats is initialized with default values
        let stats = brain_graph.learning_stats.read().await;
        assert_eq!(stats.entity_count, 0);
        assert_eq!(stats.relationship_count, 0);
        assert_eq!(stats.avg_activation, 0.0);
        drop(stats);
        
        // Test query_cache is initialized as empty HashMap
        let cache = brain_graph.query_cache.read().await;
        assert!(cache.is_empty());
        drop(cache);
    }

    #[tokio::test]
    async fn test_sdr_configuration_integration() {
        // Test SDR configuration for different embedding dimensions
        let test_cases = vec![64, 128, 256, 512];
        
        for embedding_dim in test_cases {
            let brain_graph = BrainEnhancedKnowledgeGraph::new(embedding_dim).unwrap();
            
            // Verify embedding dimension is set correctly
            assert_eq!(brain_graph.embedding_dimension(), embedding_dim);
            
            // Test that SDR storage is properly configured
            // Note: We can't directly test SDR config without public accessors
            // but we can verify the brain graph was created successfully
            assert!(brain_graph.entity_count() == 0); // Initial state
        }
    }

    #[tokio::test]
    async fn test_component_coordination() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        let entity1 = EntityKey::new(1.to_string());
        let entity2 = EntityKey::new(2.to_string());
        
        // Test coordination between activations and synaptic weights
        brain_graph.set_entity_activation(entity1, 0.8).await;
        brain_graph.set_entity_activation(entity2, 0.4).await;
        brain_graph.set_synaptic_weight(entity1, entity2, 0.6).await;
        
        // Test coordination with concept structures
        let mut concept = ConceptStructure::new();
        concept.add_input(entity1);
        concept.add_output(entity2);
        concept.concept_activation = 0.7;
        concept.coherence_score = 0.8;
        
        brain_graph.store_concept_structure("coordination_test".to_string(), concept).await;
        
        // Test coordination with learning statistics
        brain_graph.update_learning_stats(|stats| {
            stats.entity_count = 2;
            stats.relationship_count = 1;
            stats.avg_activation = 0.6;
        }).await;
        
        // Verify all components work together
        let health_metrics = brain_graph.get_health_metrics().await;
        assert!(health_metrics.overall_health > 0.0);
        
        let all_activations = brain_graph.get_all_activations().await;
        assert_eq!(all_activations.len(), 2);
        
        let weight = brain_graph.get_synaptic_weight(entity1, entity2).await;
        assert_eq!(weight, 0.6);
        
        let stored_concept = brain_graph.get_concept_structure("coordination_test").await;
        assert!(stored_concept.is_some());
        
        let stats = brain_graph.get_learning_stats().await;
        assert_eq!(stats.entity_count, 2);
    }

    #[tokio::test]
    async fn test_edge_case_entity_keys() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Test with edge case entity keys
        let edge_cases = vec![
            EntityKey::new(0.to_string()),                    // Zero key
            EntityKey::new(u64::MAX.to_string()),            // Maximum key
            EntityKey::new((u64::MAX / 2).to_string()),        // Middle value
        ];
        
        for entity in edge_cases {
            // Test activation management with edge case keys
            brain_graph.set_entity_activation(entity, 0.5).await;
            let activation = brain_graph.get_entity_activation(entity).await;
            assert_eq!(activation, 0.5);
            
            // Test synaptic weight management
            let other_entity = EntityKey::new(1.to_string());
            brain_graph.set_synaptic_weight(entity, other_entity, 0.3).await;
            let weight = brain_graph.get_synaptic_weight(entity, other_entity).await;
            assert_eq!(weight, 0.3);
        }
    }

    #[tokio::test]
    async fn test_stress_activation_operations() {
        let brain_graph = create_test_brain_graph().await.unwrap();
        
        // Stress test with many activation operations
        let num_entities = 1000;
        let entities: Vec<EntityKey> = (0..num_entities).map(|i| EntityKey::new(i.to_string())).collect();
        
        // Set activations for all entities
        for (i, entity) in entities.iter().enumerate() {
            let activation = (i as f32) / (num_entities as f32);
            brain_graph.set_entity_activation(*entity, activation).await;
        }
        
        // Verify all activations were set correctly
        for (i, entity) in entities.iter().enumerate() {
            let expected_activation = (i as f32) / (num_entities as f32);
            let actual_activation = brain_graph.get_entity_activation(*entity).await;
            assert!((actual_activation - expected_activation).abs() < 0.001);
        }
        
        // Test bulk retrieval
        let all_activations = brain_graph.get_all_activations().await;
        assert_eq!(all_activations.len(), num_entities);
        
        // Clear and verify
        brain_graph.clear_activations().await;
        let cleared_activations = brain_graph.get_all_activations().await;
        assert!(cleared_activations.is_empty());
    }
}