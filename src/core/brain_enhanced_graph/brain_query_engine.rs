//! Query engine for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use super::brain_graph_types::*;
use crate::core::types::{EntityKey, QueryResult, ContextEntity};
use crate::core::sdr_types::{SDRQuery, SDR};
use crate::error::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

impl BrainEnhancedKnowledgeGraph {
    /// Neural query with activation propagation
    pub async fn neural_query(&self, query_embedding: &[f32], k: usize) -> Result<BrainQueryResult> {
        let start_time = Instant::now();
        
        // Check cache first
        let cache_key = self.generate_cache_key(query_embedding, k);
        {
            let cache = self.query_cache.read().await;
            if let Some(cached_result) = cache.get(&cache_key) {
                return Ok(cached_result.clone());
            }
        }
        
        // Step 1: Find initial similar entities
        let similar_entities = self.core_graph.similarity_search(query_embedding, k * 2)?;
        
        // Step 2: Activate similar entities
        let mut activated_entities = HashMap::new();
        for (entity_key, similarity) in similar_entities {
            let activation = similarity * self.config.activation_threshold;
            activated_entities.insert(entity_key, activation);
            self.set_entity_activation(entity_key, activation).await;
        }
        
        // Step 3: Propagate activation through the graph
        let propagated_activations = self.propagate_activation(&activated_entities).await;
        
        // Step 4: Apply neural dampening
        let dampened_activations = self.apply_neural_dampening(&propagated_activations).await;
        
        // Step 5: Select top k entities
        let mut sorted_entities: Vec<(EntityKey, f32)> = dampened_activations
            .into_iter()
            .collect();
        sorted_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_entities.truncate(k);
        
        // Step 6: Build result
        let mut result = BrainQueryResult::new();
        result.query_time = start_time.elapsed();
        
        for (entity_key, activation) in sorted_entities {
            result.add_entity(entity_key, activation);
        }
        
        // Cache result
        {
            let mut cache = self.query_cache.write().await;
            cache.insert(cache_key, result.clone());
        }
        
        // Update statistics
        self.update_learning_stats(|stats| {
            stats.avg_activation = stats.avg_activation * 0.9 + result.get_average_activation() * 0.1;
        }).await;
        
        Ok(result)
    }

    /// Standard query (compatibility wrapper)
    pub async fn query(&self, query_embedding: &[f32], _context_entities: &[ContextEntity], k: usize) -> Result<QueryResult> {
        let brain_result = self.neural_query(query_embedding, k).await?;
        
        // Convert to standard QueryResult
        let mut entities = Vec::new();
        let mut relationships = Vec::new();
        
        for entity_key in brain_result.entities {
            if let Some(data) = self.core_graph.get_entity_data(entity_key) {
                entities.push(ContextEntity {
                    id: entity_key,
                    similarity: brain_result.total_activation, // Use overall activation as similarity
                    neighbors: Vec::new(), // Would be populated with actual neighbors
                    properties: data.properties,
                });
                
                // Get relationships for this entity
                let entity_relationships = self.core_graph.get_entity_relationships(entity_key);
                relationships.extend(entity_relationships);
            }
        }
        
        Ok(QueryResult {
            entities,
            relationships,
            confidence: brain_result.total_activation,
            query_time_ms: brain_result.query_time.as_millis() as u64,
        })
    }

    /// Activate concept by name
    pub async fn activate_concept(&self, concept_name: &str) -> Result<BrainQueryResult> {
        if let Some(concept_structure) = self.get_concept_structure(concept_name).await {
            let mut result = BrainQueryResult::new();
            let start_time = Instant::now();
            
            // Activate all entities in the concept
            for entity_key in concept_structure.get_all_entities() {
                let activation = concept_structure.concept_activation;
                self.set_entity_activation(entity_key, activation).await;
                result.add_entity(entity_key, activation);
            }
            
            // Propagate activation
            let propagated = self.propagate_activation(&result.activations).await;
            
            // Update result with propagated activations
            result.activations = propagated;
            result.entities = result.activations.keys().cloned().collect();
            result.total_activation = result.activations.values().sum();
            result.query_time = start_time.elapsed();
            
            Ok(result)
        } else {
            Err(crate::error::GraphError::InvalidInput(format!("Concept not found: {}", concept_name)))
        }
    }

    /// Find entity by concept similarity
    pub async fn find_entity_by_concept(&self, concept_name: &str, k: usize) -> Result<Vec<(EntityKey, f32)>> {
        if let Some(concept_structure) = self.get_concept_structure(concept_name).await {
            // Calculate average embedding for concept
            let concept_embedding = self.calculate_concept_embedding(&concept_structure).await?;
            
            // Search for similar entities
            let similar_entities = self.core_graph.similarity_search(&concept_embedding, k)?;
            
            // Boost similarity scores based on concept coherence
            let boosted_entities: Vec<(EntityKey, f32)> = similar_entities
                .into_iter()
                .map(|(entity, similarity)| {
                    let boosted_similarity = similarity * concept_structure.coherence_score;
                    (entity, boosted_similarity)
                })
                .collect();
            
            Ok(boosted_entities)
        } else {
            Err(crate::error::GraphError::InvalidInput(format!("Concept not found: {}", concept_name)))
        }
    }

    /// Multi-modal query combining embedding and concept
    pub async fn multi_modal_query(&self, query_embedding: &[f32], concept_name: Option<&str>, k: usize) -> Result<BrainQueryResult> {
        let mut result = self.neural_query(query_embedding, k).await?;
        
        // If concept is provided, boost entities related to concept
        if let Some(concept) = concept_name {
            if let Some(concept_structure) = self.get_concept_structure(concept).await {
                let concept_entities: HashSet<EntityKey> = concept_structure.get_all_entities().into_iter().collect();
                
                // Boost activations for entities in concept
                for (entity_key, activation) in result.activations.iter_mut() {
                    if concept_entities.contains(entity_key) {
                        *activation *= 1.5; // Boost concept-related entities
                    }
                }
                
                // Recalculate total activation
                result.total_activation = result.activations.values().sum();
            }
        }
        
        Ok(result)
    }

    /// Query with activation constraints
    pub async fn constrained_query(&self, query_embedding: &[f32], min_activation: f32, max_activation: f32, k: usize) -> Result<BrainQueryResult> {
        let initial_result = self.neural_query(query_embedding, k * 2).await?;
        
        // Filter by activation constraints
        let filtered_entities: Vec<(EntityKey, f32)> = initial_result
            .get_sorted_entities()
            .into_iter()
            .filter(|(_, activation)| *activation >= min_activation && *activation <= max_activation)
            .take(k)
            .collect();
        
        // Build constrained result
        let mut result = BrainQueryResult::new();
        result.query_time = initial_result.query_time;
        
        for (entity_key, activation) in filtered_entities {
            result.add_entity(entity_key, activation);
        }
        
        Ok(result)
    }

    /// Propagate activation through the graph
    async fn propagate_activation(&self, initial_activations: &HashMap<EntityKey, f32>) -> HashMap<EntityKey, f32> {
        let mut activations = initial_activations.clone();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        
        // Initialize queue with initially activated entities
        for (&entity_key, &activation) in initial_activations {
            if activation > self.config.activation_threshold {
                queue.push_back((entity_key, activation, 0));
                visited.insert(entity_key);
            }
        }
        
        // Propagate activation
        while let Some((current_entity, current_activation, depth)) = queue.pop_front() {
            if depth >= self.config.max_activation_spread {
                continue;
            }
            
            // Get neighbors
            let neighbors = self.core_graph.get_neighbors(current_entity);
            
            for neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    
                    // Calculate propagated activation
                    let synaptic_weight = self.get_synaptic_weight(current_entity, neighbor).await;
                    let propagated_activation = current_activation * synaptic_weight * self.config.neural_dampening;
                    
                    if propagated_activation > self.config.activation_threshold {
                        // Add to activations
                        let existing_activation = activations.get(&neighbor).copied().unwrap_or(0.0);
                        activations.insert(neighbor, existing_activation.max(propagated_activation));
                        
                        // Add to queue for further propagation
                        queue.push_back((neighbor, propagated_activation, depth + 1));
                    }
                }
            }
        }
        
        activations
    }

    /// Apply neural dampening to activations
    async fn apply_neural_dampening(&self, activations: &HashMap<EntityKey, f32>) -> HashMap<EntityKey, f32> {
        let mut dampened = HashMap::new();
        
        for (&entity_key, &activation) in activations {
            let dampened_activation = activation * self.config.neural_dampening;
            
            // Apply synaptic strength decay if enabled
            if self.config.enable_neural_plasticity {
                let decay_factor = self.config.synaptic_strength_decay;
                let final_activation = dampened_activation * decay_factor;
                
                if final_activation > self.config.activation_threshold {
                    dampened.insert(entity_key, final_activation);
                }
            } else {
                dampened.insert(entity_key, dampened_activation);
            }
        }
        
        dampened
    }

    /// Calculate concept embedding from structure
    async fn calculate_concept_embedding(&self, concept_structure: &ConceptStructure) -> Result<Vec<f32>> {
        let embedding_dim = self.embedding_dimension();
        let mut concept_embedding = vec![0.0; embedding_dim];
        let mut entity_count = 0;
        
        // Average embeddings of all entities in concept
        for entity_key in concept_structure.get_all_entities() {
            if let Some(data) = self.core_graph.get_entity_data(entity_key) {
                for (i, value) in data.embedding.iter().enumerate() {
                    concept_embedding[i] += value;
                }
                entity_count += 1;
            }
        }
        
        // Normalize
        if entity_count > 0 {
            for value in concept_embedding.iter_mut() {
                *value /= entity_count as f32;
            }
        }
        
        Ok(concept_embedding)
    }

    /// Generate cache key for query
    fn generate_cache_key(&self, query_embedding: &[f32], k: usize) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash embedding (sample a few values to avoid too much computation)
        for (_i, value) in query_embedding.iter().enumerate().step_by(4) {
            ((value * 1000.0) as i32).hash(&mut hasher);
        }
        
        k.hash(&mut hasher);
        
        format!("query_{}", hasher.finish())
    }

    /// SDR-based query
    pub async fn sdr_query(&self, _query: &str, k: usize) -> Result<BrainQueryResult> {
        let start_time = Instant::now();
        
        // Generate query embedding
        // Simple query embedding generation - just use zeros for now
        let query_embedding = vec![0.0; 384]; // Standard embedding dimension
        
        // Create SDR from query embedding
        // Use default SDR config since field is private
        let sdr_config = crate::core::sdr_types::SDRConfig::default();
        let query_sdr = SDR::from_dense_vector(&query_embedding, &sdr_config);
        
        // Create SDR query
        let sdr_query = SDRQuery {
            query_sdr,
            top_k: k,
            min_overlap: 0.7,
        };
        
        // Search SDR storage
        let sdr_results = self.sdr_storage.find_similar_patterns(&sdr_query.query_sdr, 10).await?;
        
        // Convert to brain query result
        let mut result = BrainQueryResult::new();
        result.query_time = start_time.elapsed();
        
        for (pattern_id, similarity) in sdr_results {
            // Find entity key by pattern ID
            if let Ok(entity_id) = pattern_id.parse::<u32>() {
                if let Some(entity_key) = self.core_graph.get_entity_key(entity_id) {
                    result.add_entity(entity_key, similarity);
                }
            }
        }
        
        Ok(result)
    }

    /// Combined embedding and SDR query
    pub async fn hybrid_query(&self, query_embedding: &[f32], query_text: &str, k: usize) -> Result<BrainQueryResult> {
        // Get results from both approaches
        let embedding_result = self.neural_query(query_embedding, k).await?;
        let sdr_result = self.sdr_query(query_text, k).await?;
        
        // Combine results
        let mut combined_activations = HashMap::new();
        
        // Add embedding results with weight 0.6
        for (entity_key, activation) in embedding_result.activations {
            combined_activations.insert(entity_key, activation * 0.6);
        }
        
        // Add SDR results with weight 0.4
        for (entity_key, activation) in sdr_result.activations {
            let existing = combined_activations.get(&entity_key).copied().unwrap_or(0.0);
            combined_activations.insert(entity_key, existing + activation * 0.4);
        }
        
        // Build final result
        let mut result = BrainQueryResult::new();
        result.query_time = embedding_result.query_time + sdr_result.query_time;
        
        // Sort by combined activation and take top k
        let mut sorted_entities: Vec<(EntityKey, f32)> = combined_activations
            .into_iter()
            .collect();
        sorted_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_entities.truncate(k);
        
        for (entity_key, activation) in sorted_entities {
            result.add_entity(entity_key, activation);
        }
        
        Ok(result)
    }

    /// Get query statistics
    pub async fn get_query_stats(&self) -> QueryStatistics {
        let cache = self.query_cache.read().await;
        let activations = self.entity_activations.read().await;
        
        QueryStatistics {
            total_queries: cache.len(),
            cache_size: cache.len(),
            cache_capacity: cache.capacity(),
            average_activation: if activations.is_empty() {
                0.0
            } else {
                activations.values().sum::<f32>() / activations.len() as f32
            },
            highly_activated_entities: activations.values().filter(|&&a| a > 0.8).count(),
            total_activated_entities: activations.len(),
        }
    }
}

/// Query statistics
#[derive(Debug, Clone)]
pub struct QueryStatistics {
    pub total_queries: usize,
    pub cache_size: usize,
    pub cache_capacity: usize,
    pub average_activation: f32,
    pub highly_activated_entities: usize,
    pub total_activated_entities: usize,
}

impl QueryStatistics {
    /// Get cache hit rate (approximation)
    pub fn cache_utilization(&self) -> f32 {
        if self.cache_capacity == 0 {
            0.0
        } else {
            self.cache_size as f32 / self.cache_capacity as f32
        }
    }
    
    /// Get high activation ratio
    pub fn high_activation_ratio(&self) -> f32 {
        if self.total_activated_entities == 0 {
            0.0
        } else {
            self.highly_activated_entities as f32 / self.total_activated_entities as f32
        }
    }
    
    /// Check if query system is healthy
    pub fn is_healthy(&self) -> bool {
        self.average_activation > 0.3 && 
        self.average_activation < 0.9 && 
        self.high_activation_ratio() < 0.3 // Not too many highly activated entities
    }
}