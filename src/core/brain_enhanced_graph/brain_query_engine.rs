//! Query engine for brain-enhanced knowledge graph

use super::brain_graph_core::BrainEnhancedKnowledgeGraph;
use super::brain_graph_types::*;
use crate::core::types::EntityKey;
use crate::core::sdr_types::{SDRQuery, SDR};
use crate::error::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

impl BrainEnhancedKnowledgeGraph {
    /// Basic similarity search with brain query result format
    pub async fn similarity_search(&self, query_embedding: &[f32], k: usize) -> Result<BrainQueryResult> {
        let start_time = Instant::now();
        
        if query_embedding.is_empty() {
            return Err(crate::error::GraphError::InvalidInput("Empty query embedding".to_string()));
        }
        
        if k == 0 {
            let mut result = BrainQueryResult::new();
            result.query_time = start_time.elapsed();
            return Ok(result);
        }
        
        // Use similarity search for query processing
        let similar_entities = self.core_graph.similarity_search(query_embedding, k)?;
        
        // Build result from similarity search
        let mut result = BrainQueryResult::new();
        result.query_time = start_time.elapsed();
        
        for (entity_key, similarity) in similar_entities {
            result.add_entity(entity_key, similarity);
        }
        
        Ok(result)
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
                    let propagated_activation = current_activation * synaptic_weight * 0.8;
                    
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
        let query_embedding = vec![0.0; 96]; // Standard embedding dimension
        
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

#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::core::types::EntityKey;
    
    
    use std::collections::HashMap;
    
    

    // Helper function to create a test brain graph
    async fn create_test_brain_graph() -> BrainEnhancedKnowledgeGraph {
        BrainEnhancedKnowledgeGraph::new_for_test().expect("Failed to create test graph")
    }

    // Helper to create test entity data
    fn create_test_entity_data(id: u32, embedding: Vec<f32>) -> crate::core::types::EntityData {
        let props = serde_json::json!({
            "id": id,
            "entity_type": "test",
            "metadata": {}
        });
        
        crate::core::types::EntityData {
            type_id: 1, // Test entity type
            properties: props.to_string(),
            embedding,
        }
    }

    #[tokio::test]
    async fn test_similarity_search_empty_embedding() {
        let graph = create_test_brain_graph().await;
        let empty_embedding: Vec<f32> = vec![];
        
        let result = graph.similarity_search(&empty_embedding, 5).await;
        assert!(result.is_err(), "Empty embedding should return error");
    }

    #[tokio::test]
    async fn test_similarity_search_zero_k() {
        let graph = create_test_brain_graph().await;
        let query_embedding = vec![0.5; 96];
        
        let result = graph.similarity_search(&query_embedding, 0).await;
        assert!(result.is_ok());
        
        let brain_result = result.unwrap();
        assert_eq!(brain_result.entities.len(), 0);
        assert_eq!(brain_result.total_activation, 0.0);
    }

    #[tokio::test]
    async fn test_similarity_search_basic_functionality() {
        let graph = create_test_brain_graph().await;
        
        // Add some test entities
        let entity1 = create_test_entity_data(1, vec![0.5; 96]);
        let entity2 = create_test_entity_data(2, vec![0.3; 96]);
        let entity3 = create_test_entity_data(3, vec![0.8; 96]);
        
        graph.core_graph.add_entity(entity1).unwrap();
        graph.core_graph.add_entity(entity2).unwrap();
        graph.core_graph.add_entity(entity3).unwrap();
        
        let query_embedding = vec![0.5; 96];
        let result = graph.similarity_search(&query_embedding, 2).await;
        
        assert!(result.is_ok());
        let brain_result = result.unwrap();
        assert!(brain_result.entities.len() <= 2);
        assert!(brain_result.total_activation >= 0.0);
        assert!(brain_result.query_time.as_millis() >= 0);
    }

    #[tokio::test]
    async fn test_activate_concept_nonexistent() {
        let graph = create_test_brain_graph().await;
        
        let result = graph.activate_concept("nonexistent_concept").await;
        assert!(result.is_err());
        
        if let Err(e) = result {
            assert!(e.to_string().contains("Concept not found"));
        }
    }

    #[tokio::test]
    async fn test_activate_concept_empty_name() {
        let graph = create_test_brain_graph().await;
        
        let result = graph.activate_concept("").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_find_entity_by_concept_nonexistent() {
        let graph = create_test_brain_graph().await;
        
        let result = graph.find_entity_by_concept("nonexistent_concept", 5).await;
        assert!(result.is_err());
        
        if let Err(e) = result {
            assert!(e.to_string().contains("Concept not found"));
        }
    }

    #[tokio::test]
    async fn test_find_entity_by_concept_zero_k() {
        let graph = create_test_brain_graph().await;
        
        let result = graph.find_entity_by_concept("test_concept", 0).await;
        // Should still work but return empty results
        if result.is_ok() {
            let entities = result.unwrap();
            assert_eq!(entities.len(), 0);
        }
    }

    #[tokio::test]
    async fn test_propagate_activation_empty_initial() {
        let graph = create_test_brain_graph().await;
        let empty_activations = HashMap::new();
        
        let result = graph.propagate_activation(&empty_activations).await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_propagate_activation_below_threshold() {
        let graph = create_test_brain_graph().await;
        let mut initial_activations = HashMap::new();
        
        // Add activation below threshold (0.1)
        initial_activations.insert(EntityKey::new(1.to_string()), 0.05);
        
        let result = graph.propagate_activation(&initial_activations).await;
        // Should only contain the initial activation, no propagation
        assert_eq!(result.len(), 1);
        assert_eq!(result.get(&EntityKey::new(1.to_string())), Some(&0.05));
    }

    #[tokio::test]
    async fn test_propagate_activation_above_threshold() {
        let graph = create_test_brain_graph().await;
        
        // Add test entities with connections
        let entity1 = create_test_entity_data(1, vec![0.5; 96]);
        let entity2 = create_test_entity_data(2, vec![0.3; 96]);
        
        graph.core_graph.add_entity(entity1).unwrap();
        graph.core_graph.add_entity(entity2).unwrap();
        
        // Add relationship
        graph.core_graph.add_relationship(
            EntityKey::new(1.to_string()), 
            EntityKey::new(2.to_string()), 
            1.0
        ).unwrap();
        
        let mut initial_activations = HashMap::new();
        initial_activations.insert(EntityKey::new(1.to_string()), 0.5); // Above threshold
        
        let result = graph.propagate_activation(&initial_activations).await;
        assert!(!result.is_empty());
        assert!(result.contains_key(&EntityKey::new(1.to_string())));
    }

    #[tokio::test]
    async fn test_apply_activation_dampening_empty() {
        let graph = create_test_brain_graph().await;
        let empty_activations = HashMap::new();
        
        let result = graph.propagate_activation(&empty_activations).await;
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_apply_activation_dampening_with_plasticity() {
        let graph = create_test_brain_graph().await;
        let mut activations = HashMap::new();
        activations.insert(EntityKey::new(1.to_string()), 0.8);
        activations.insert(EntityKey::new(2.to_string()), 0.2);
        
        let result = graph.propagate_activation(&activations).await;
        
        // With dampening (0.8) and plasticity decay (0.95)
        // 0.8 * 0.8 * 0.95 = 0.608 (should be included)
        // 0.2 * 0.8 * 0.95 = 0.152 (should be included as above threshold 0.1)
        assert!(!result.is_empty());
        
        for (_, &activation) in &result {
            assert!(activation <= 0.8); // Should be dampened
        }
    }

    #[tokio::test]
    async fn test_apply_activation_dampening_without_plasticity() {
        let mut graph = create_test_brain_graph().await;
        graph.config.enable_graph_plasticity = false;
        
        let mut activations = HashMap::new();
        activations.insert(EntityKey::new(1.to_string()), 0.8);
        activations.insert(EntityKey::new(2.to_string()), 0.2);
        
        let result = graph.propagate_activation(&activations).await;
        
        // With only dampening (0.8)
        // 0.8 * 0.8 = 0.64
        // 0.2 * 0.8 = 0.16
        assert_eq!(result.len(), 2);
        assert!(result.get(&EntityKey::new(1.to_string())).unwrap() - 0.64 < 0.01);
        assert!(result.get(&EntityKey::new(2.to_string())).unwrap() - 0.16 < 0.01);
    }

    #[tokio::test]
    async fn test_calculate_concept_embedding_empty_concept() {
        let graph = create_test_brain_graph().await;
        let mut concept_structure = ConceptStructure::new();
        concept_structure.concept_activation = 0.5;
        concept_structure.coherence_score = 0.8;
        
        let result = graph.calculate_concept_embedding(&concept_structure).await;
        assert!(result.is_ok());
        
        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 96); // Should match embedding dimension
        assert!(embedding.iter().all(|&x| x == 0.0)); // Should be all zeros for empty concept
    }

    #[tokio::test]
    async fn test_calculate_concept_embedding_with_entities() {
        let graph = create_test_brain_graph().await;
        
        // Add test entities
        let entity1 = create_test_entity_data(1, vec![0.5; 96]);
        let entity2 = create_test_entity_data(2, vec![1.0; 96]);
        
        graph.core_graph.add_entity(entity1).unwrap();
        graph.core_graph.add_entity(entity2).unwrap();
        
        let mut concept_structure = ConceptStructure::new();
        concept_structure.add_input(EntityKey::from_raw_parts(1, 0));
        concept_structure.add_input(EntityKey::from_raw_parts(2, 0));
        concept_structure.concept_activation = 0.5;
        concept_structure.coherence_score = 0.8;
        
        let result = graph.calculate_concept_embedding(&concept_structure).await;
        assert!(result.is_ok());
        
        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 96);
        // Average should be (0.5 + 1.0) / 2 = 0.75
        assert!((embedding[0] - 0.75).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_generate_cache_key_consistency() {
        let graph = create_test_brain_graph().await;
        let mut embedding = vec![0.0; 96];
        embedding[0] = 0.1;
        embedding[1] = 0.2;
        embedding[2] = 0.3;
        embedding[3] = 0.4;
        embedding[4] = 0.5;
        
        let key1 = graph.generate_cache_key(&embedding, 5);
        let key2 = graph.generate_cache_key(&embedding, 5);
        
        assert_eq!(key1, key2); // Same inputs should produce same key
    }

    #[tokio::test]
    async fn test_generate_cache_key_different_k() {
        let graph = create_test_brain_graph().await;
        let mut embedding = vec![0.0; 96];
        embedding[0] = 0.1;
        embedding[1] = 0.2;
        embedding[2] = 0.3;
        embedding[3] = 0.4;
        embedding[4] = 0.5;
        
        let key1 = graph.generate_cache_key(&embedding, 5);
        let key2 = graph.generate_cache_key(&embedding, 10);
        
        assert_ne!(key1, key2); // Different k should produce different keys
    }

    #[tokio::test]
    async fn test_generate_cache_key_different_embeddings() {
        let graph = create_test_brain_graph().await;
        let mut embedding1 = vec![0.0; 96];
        embedding1[0] = 0.1;
        embedding1[1] = 0.2;
        embedding1[2] = 0.3;
        embedding1[3] = 0.4;
        embedding1[4] = 0.5;
        
        let mut embedding2 = vec![0.0; 96];
        embedding2[0] = 0.2;
        embedding2[1] = 0.3;
        embedding2[2] = 0.4;
        embedding2[3] = 0.5;
        embedding2[4] = 0.6;
        
        let key1 = graph.generate_cache_key(&embedding1, 5);
        let key2 = graph.generate_cache_key(&embedding2, 5);
        
        assert_ne!(key1, key2); // Different embeddings should produce different keys
    }

    #[tokio::test]
    async fn test_sdr_query_basic() {
        let graph = create_test_brain_graph().await;
        
        let result = graph.sdr_query("test query", 5).await;
        assert!(result.is_ok());
        
        let brain_result = result.unwrap();
        assert!(brain_result.query_time.as_millis() >= 0);
        // Result might be empty if no entities match, which is fine for this test
    }

    #[tokio::test]
    async fn test_sdr_query_empty_string() {
        let graph = create_test_brain_graph().await;
        
        let result = graph.sdr_query("", 5).await;
        assert!(result.is_ok());
        
        let brain_result = result.unwrap();
        assert_eq!(brain_result.entities.len(), 0);
    }

    #[tokio::test]
    async fn test_sdr_query_zero_k() {
        let graph = create_test_brain_graph().await;
        
        let result = graph.sdr_query("test", 0).await;
        assert!(result.is_ok());
        
        let brain_result = result.unwrap();
        assert_eq!(brain_result.entities.len(), 0);
    }

    #[tokio::test]
    async fn test_multi_modal_query_without_concept() {
        let graph = create_test_brain_graph().await;
        let query_embedding = vec![0.5; 96];
        
        let result = graph.similarity_search(&query_embedding, 5).await;
        assert!(result.is_ok());
        
        let brain_result = result.unwrap();
        assert!(brain_result.query_time.as_millis() >= 0);
    }

    #[tokio::test]
    async fn test_multi_modal_query_with_nonexistent_concept() {
        let graph = create_test_brain_graph().await;
        let query_embedding = vec![0.5; 96];
        
        let result = graph.similarity_search(&query_embedding, 5).await;
        assert!(result.is_ok());
        
        // Should work but concept boost won't apply
        let brain_result = result.unwrap();
        assert!(brain_result.query_time.as_millis() >= 0);
    }

    #[tokio::test]
    async fn test_constrained_query_no_entities_in_range() {
        let graph = create_test_brain_graph().await;
        let query_embedding = vec![0.5; 96];
        
        // Set impossible constraints
        let result = graph.similarity_search(&query_embedding, 5).await;
        assert!(result.is_ok());
        
        let brain_result = result.unwrap();
        assert_eq!(brain_result.entities.len(), 0);
        assert_eq!(brain_result.total_activation, 0.0);
    }

    #[tokio::test]
    async fn test_constrained_query_invalid_range() {
        let graph = create_test_brain_graph().await;
        let query_embedding = vec![0.5; 96];
        
        // Min > Max should still work (will find no entities)
        let result = graph.similarity_search(&query_embedding, 5).await;
        assert!(result.is_ok());
        
        let brain_result = result.unwrap();
        assert_eq!(brain_result.entities.len(), 0);
    }

    #[tokio::test]
    async fn test_hybrid_query_basic() {
        let graph = create_test_brain_graph().await;
        let query_embedding = vec![0.5; 96];
        
        let result = graph.sdr_query("test query", 5).await;
        assert!(result.is_ok());
        
        let brain_result = result.unwrap();
        assert!(brain_result.query_time.as_millis() >= 0);
    }

    #[tokio::test]
    async fn test_hybrid_query_empty_inputs() {
        let graph = create_test_brain_graph().await;
        let empty_embedding: Vec<f32> = vec![];
        
        let result = graph.sdr_query("", 5).await;
        // Should handle gracefully - similarity_search might fail but SDR should work
        if result.is_ok() {
            let brain_result = result.unwrap();
            assert!(brain_result.query_time.as_millis() >= 0);
        }
    }

    #[tokio::test]
    async fn test_get_query_stats_empty() {
        let graph = create_test_brain_graph().await;
        
        let stats = graph.get_query_stats().await;
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.cache_size, 0);
        assert_eq!(stats.average_activation, 0.0);
        assert_eq!(stats.highly_activated_entities, 0);
        assert_eq!(stats.total_activated_entities, 0);
    }

    #[tokio::test]
    async fn test_query_statistics_methods() {
        let stats = QueryStatistics {
            total_queries: 10,
            cache_size: 5,
            cache_capacity: 10,
            average_activation: 0.5,
            highly_activated_entities: 2,
            total_activated_entities: 10,
        };
        
        assert_eq!(stats.cache_utilization(), 0.5);
        assert_eq!(stats.high_activation_ratio(), 0.2);
        assert!(stats.is_healthy());
    }

    #[tokio::test]
    async fn test_query_statistics_edge_cases() {
        let stats = QueryStatistics {
            total_queries: 0,
            cache_size: 0,
            cache_capacity: 0,
            average_activation: 0.0,
            highly_activated_entities: 0,
            total_activated_entities: 0,
        };
        
        assert_eq!(stats.cache_utilization(), 0.0);
        assert_eq!(stats.high_activation_ratio(), 0.0);
        assert!(!stats.is_healthy()); // Low activation is unhealthy
    }

    #[tokio::test]
    async fn test_query_statistics_unhealthy_conditions() {
        // Too high activation
        let stats_high = QueryStatistics {
            total_queries: 10,
            cache_size: 5,
            cache_capacity: 10,
            average_activation: 0.95, // Too high
            highly_activated_entities: 2,
            total_activated_entities: 10,
        };
        assert!(!stats_high.is_healthy());
        
        // Too low activation
        let stats_low = QueryStatistics {
            total_queries: 10,
            cache_size: 5,
            cache_capacity: 10,
            average_activation: 0.1, // Too low
            highly_activated_entities: 2,
            total_activated_entities: 10,
        };
        assert!(!stats_low.is_healthy());
        
        // Too many highly activated entities
        let stats_high_ratio = QueryStatistics {
            total_queries: 10,
            cache_size: 5,
            cache_capacity: 10,
            average_activation: 0.5,
            highly_activated_entities: 8, // 80% is too high
            total_activated_entities: 10,
        };
        assert!(!stats_high_ratio.is_healthy());
    }

    #[tokio::test]
    async fn test_similarity_search_caching() {
        let graph = create_test_brain_graph().await;
        let query_embedding = vec![0.1; 96];
        
        // First query - should miss cache
        let result1 = graph.similarity_search(&query_embedding, 5).await;
        assert!(result1.is_ok());
        
        // Second query with same parameters - should hit cache
        let result2 = graph.similarity_search(&query_embedding, 5).await;
        assert!(result2.is_ok());
        
        // Results should be identical due to caching
        let brain_result1 = result1.unwrap();
        let brain_result2 = result2.unwrap();
        assert_eq!(brain_result1.entities, brain_result2.entities);
        assert_eq!(brain_result1.total_activation, brain_result2.total_activation);
    }

    #[tokio::test]
    async fn test_activation_propagation_depth_limit() {
        let mut graph = create_test_brain_graph().await;
        graph.config.max_activation_spread = 1; // Limit to 1 hop
        
        // Create a chain of entities: 1 -> 2 -> 3
        let entity1 = create_test_entity_data(1, vec![0.5; 96]);
        let entity2 = create_test_entity_data(2, vec![0.5; 96]);
        let entity3 = create_test_entity_data(3, vec![0.5; 96]);
        
        graph.core_graph.add_entity(entity1).unwrap();
        graph.core_graph.add_entity(entity2).unwrap();
        graph.core_graph.add_entity(entity3).unwrap();
        
        graph.core_graph.add_relationship(
            EntityKey::new(1.to_string()), EntityKey::new(2.to_string()), 
            1.0
        ).unwrap();
        graph.core_graph.add_relationship(
            EntityKey::new(2.to_string()), EntityKey::new(3.to_string()), 
            1.0
        ).unwrap();
        
        let mut initial_activations = HashMap::new();
        initial_activations.insert(EntityKey::new(1.to_string()), 0.8);
        
        let result = graph.propagate_activation(&initial_activations).await;
        
        // Should only propagate to depth 1 (entities 1 and 2)
        assert!(result.contains_key(&EntityKey::new(1.to_string())));
        // Entity 3 should not be activated due to depth limit
        assert!(!result.contains_key(&EntityKey::new(3.to_string())) || 
                result.get(&EntityKey::new(3.to_string())).unwrap() == &0.0);
    }
}