//! Query system for knowledge graph

use super::graph_core::{KnowledgeGraph, MAX_QUERY_TIME};
use crate::core::types::{EntityKey, EntityData, QueryResult, ContextEntity};
use crate::error::{GraphError, Result};
use crate::embedding::similarity::cosine_similarity;
use std::time::Instant;
use std::collections::HashMap;

impl KnowledgeGraph {
    /// Perform complex query with context entities and relationships
    pub fn query(&self, query_embedding: &[f32], context_entities: &[ContextEntity], k: usize) -> Result<QueryResult> {
        let start_time = Instant::now();
        
        // Validate query embedding
        if query_embedding.len() != self.embedding_dim {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.embedding_dim,
                actual: query_embedding.len(),
            });
        }
        
        // Step 1: Find similar entities
        let similar_entities = self.similarity_search(query_embedding, k * 2)?; // Get more candidates
        
        // Step 2: Build context entity map
        let mut context_map = HashMap::new();
        for context_entity in context_entities {
            context_map.insert(context_entity.id, context_entity);
        }
        
        // Step 3: Score entities based on similarity and context
        let mut scored_entities = Vec::new();
        for (entity_key, similarity) in similar_entities {
            let context_score = self.calculate_context_score(entity_key, &context_map);
            let combined_score = similarity * 0.7 + context_score * 0.3; // Weighted combination
            
            if let Some((meta, data)) = self.get_entity(entity_key) {
                scored_entities.push((entity_key, combined_score, meta, data));
            }
        }
        
        // Step 4: Sort by combined score and take top k
        scored_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_entities.truncate(k);
        
        // Step 5: Collect relationships for result entities
        let mut result_relationships = Vec::new();
        for (entity_key, _, _, _) in &scored_entities {
            let relationships = self.get_entity_relationships(*entity_key);
            result_relationships.extend(relationships);
        }
        
        // Step 6: Build query result
        let entities: Vec<ContextEntity> = scored_entities
            .into_iter()
            .map(|(key, score, _, data)| {
                // Convert to ContextEntity
                let neighbors = self.get_neighbors(key);
                ContextEntity {
                    id: key,
                    similarity: score,
                    neighbors,
                    properties: format!("{:?}", data.properties), // Convert properties to string
                }
            })
            .collect();
        
        // Calculate overall confidence based on top scores
        let confidence = if entities.is_empty() {
            0.0
        } else {
            entities.iter().map(|e| e.similarity).sum::<f32>() / entities.len() as f32
        };
        
        let result = QueryResult {
            entities,
            relationships: result_relationships,
            confidence,
            query_time_ms: start_time.elapsed().as_millis() as u64,
        };
        
        // Check if query took too long
        if start_time.elapsed() > MAX_QUERY_TIME {
            #[cfg(debug_assertions)]
            log::warn!("Query took longer than expected: {:.2}ms", start_time.elapsed().as_millis());
        }
        
        Ok(result)
    }

    /// Simple query without context
    pub fn simple_query(&self, query_embedding: &[f32], k: usize) -> Result<QueryResult> {
        self.query(query_embedding, &[], k)
    }

    /// Query with entity filtering
    pub fn query_filtered<F>(&self, query_embedding: &[f32], k: usize, filter: F) -> Result<QueryResult>
    where
        F: Fn(EntityKey, &EntityData) -> bool,
    {
        let start_time = Instant::now();
        
        // Get similarity results
        let similar_entities = self.similarity_search(query_embedding, k * 3)?; // Get more candidates
        
        // Filter entities
        let mut filtered_entities = Vec::new();
        for (entity_key, similarity) in similar_entities {
            if let Some((_, data)) = self.get_entity(entity_key) {
                if filter(entity_key, &data) {
                    let neighbors = self.get_neighbors(entity_key);
                    filtered_entities.push(ContextEntity {
                        id: entity_key,
                        similarity,
                        neighbors,
                        properties: format!("{:?}", data.properties),
                    });
                }
            }
            
            if filtered_entities.len() >= k {
                break;
            }
        }
        
        // Collect relationships
        let mut result_relationships = Vec::new();
        for entity in &filtered_entities {
            let relationships = self.get_entity_relationships(entity.id);
            result_relationships.extend(relationships);
        }
        
        // Calculate confidence
        let confidence = if filtered_entities.is_empty() {
            0.0
        } else {
            filtered_entities.iter().map(|e| e.similarity).sum::<f32>() / filtered_entities.len() as f32
        };
        
        Ok(QueryResult {
            entities: filtered_entities,
            relationships: result_relationships,
            confidence,
            query_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Query by entity type/category
    pub fn query_by_type(&self, query_embedding: &[f32], entity_type: &str, k: usize) -> Result<QueryResult> {
        self.query_filtered(query_embedding, k, |_, data| {
            data.properties.contains(entity_type)
        })
    }

    /// Query entities within a specific property range
    pub fn query_by_property_range(&self, query_embedding: &[f32], property: &str, min_value: f32, max_value: f32, k: usize) -> Result<QueryResult> {
        self.query_filtered(query_embedding, k, |_, data| {
            // Parse the properties string as JSON and look for the property
            if let Ok(props) = serde_json::from_str::<serde_json::Value>(&data.properties) {
                if let Some(value) = props.get(property) {
                    if let Some(num) = value.as_f64() {
                        return num as f32 >= min_value && num as f32 <= max_value;
                    }
                }
            }
            false
        })
    }

    /// Multi-step query (query, then query neighbors)
    pub fn multi_step_query(&self, query_embedding: &[f32], steps: usize, k: usize) -> Result<QueryResult> {
        let start_time = Instant::now();
        
        // Step 1: Initial query
        let initial_results = self.similarity_search(query_embedding, k)?;
        let mut all_entities = HashMap::new();
        let mut all_relationships = Vec::new();
        
        // Add initial results
        for (entity_key, _) in &initial_results {
            if let Some((_, data)) = self.get_entity(*entity_key) {
                all_entities.insert(*entity_key, data);
            }
        }
        
        // Step 2: Expand through neighbors
        let mut current_entities: Vec<EntityKey> = initial_results.into_iter().map(|(k, _)| k).collect();
        
        for step in 1..steps {
            let mut next_entities = Vec::new();
            
            for &entity_key in &current_entities {
                let neighbors = self.get_neighbors(entity_key);
                
                for neighbor in neighbors {
                    if !all_entities.contains_key(&neighbor) {
                        if let Some((_, data)) = self.get_entity(neighbor) {
                            // Score neighbor based on similarity to query
                            if let Some(neighbor_embedding) = self.get_entity_embedding(neighbor) {
                                let similarity = cosine_similarity(query_embedding, &neighbor_embedding);
                                if similarity > 0.5 { // Threshold for relevance
                                    all_entities.insert(neighbor, data);
                                    next_entities.push(neighbor);
                                }
                            }
                        }
                    }
                }
                
                // Collect relationships
                let relationships = self.get_entity_relationships(entity_key);
                all_relationships.extend(relationships);
            }
            
            current_entities = next_entities;
            
            // Limit expansion
            if all_entities.len() > k * (step + 1) {
                break;
            }
        }
        
        // Convert to result format
        let entities: Vec<ContextEntity> = all_entities.into_iter()
            .map(|(key, data)| {
                let neighbors = self.get_neighbors(key);
                ContextEntity {
                    id: key,
                    similarity: 0.0, // Will be calculated separately if needed
                    neighbors,
                    properties: data.properties.to_string(),
                }
            })
            .collect();
        
        Ok(QueryResult {
            entities,
            relationships: all_relationships,
            confidence: 1.0, // Default confidence
            query_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Query entities that are connected to a specific entity
    pub fn query_connected_entities(&self, query_embedding: &[f32], connected_to: EntityKey, k: usize) -> Result<QueryResult> {
        let start_time = Instant::now();
        
        // Get neighbors of the connected entity
        let neighbors = self.get_neighbors(connected_to);
        
        // Filter similarity results to only include connected entities
        let similar_entities = self.similarity_search(query_embedding, k * 2)?;
        
        let mut filtered_entities = Vec::new();
        for (entity_key, similarity) in similar_entities {
            if neighbors.contains(&entity_key) {
                if let Some((_, data)) = self.get_entity(entity_key) {
                    let entity_neighbors = self.get_neighbors(entity_key);
                    filtered_entities.push(ContextEntity {
                        id: entity_key,
                        similarity,
                        neighbors: entity_neighbors,
                        properties: format!("{:?}", data.properties),
                    });
                }
            }
            
            if filtered_entities.len() >= k {
                break;
            }
        }
        
        // Collect relationships
        let mut result_relationships = Vec::new();
        for entity in &filtered_entities {
            let relationships = self.get_entity_relationships(entity.id);
            result_relationships.extend(relationships);
        }
        
        // Calculate confidence
        let confidence = if filtered_entities.is_empty() {
            0.0
        } else {
            filtered_entities.iter().map(|e| e.similarity).sum::<f32>() / filtered_entities.len() as f32
        };
        
        Ok(QueryResult {
            entities: filtered_entities,
            relationships: result_relationships,
            confidence,
            query_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Query entities within a specific graph distance
    pub fn query_within_distance(&self, query_embedding: &[f32], center_entity: EntityKey, max_distance: usize, k: usize) -> Result<QueryResult> {
        let start_time = Instant::now();
        
        // Find entities within distance
        let entities_within_distance = self.find_entities_within_distance(center_entity, max_distance);
        
        // Calculate similarities and filter
        let mut scored_entities = Vec::new();
        for entity_key in entities_within_distance {
            if let Some(embedding) = self.get_entity_embedding(entity_key) {
                let similarity = cosine_similarity(query_embedding, &embedding);
                if let Some((_, data)) = self.get_entity(entity_key) {
                    scored_entities.push((entity_key, similarity, data));
                }
            }
        }
        
        // Sort by similarity and take top k
        scored_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored_entities.truncate(k);
        
        // Collect relationships
        let mut result_relationships = Vec::new();
        for (entity_key, _, _) in &scored_entities {
            let relationships = self.get_entity_relationships(*entity_key);
            result_relationships.extend(relationships);
        }
        
        let entities: Vec<ContextEntity> = scored_entities
            .iter()
            .map(|(key, similarity, data)| {
                let neighbors = self.get_neighbors(*key);
                ContextEntity {
                    id: *key,
                    similarity: *similarity,
                    neighbors,
                    properties: data.properties.clone(),
                }
            })
            .collect();
        
        // Calculate confidence from scored entities
        let confidence = if scored_entities.is_empty() {
            0.0
        } else {
            scored_entities.iter().map(|(_, sim, _)| sim).sum::<f32>() / scored_entities.len() as f32
        };
        
        Ok(QueryResult {
            entities,
            relationships: result_relationships,
            confidence,
            query_time_ms: start_time.elapsed().as_millis() as u64,
        })
    }

    /// Calculate context score for an entity
    fn calculate_context_score(&self, entity_key: EntityKey, context_map: &HashMap<EntityKey, &ContextEntity>) -> f32 {
        let mut score = 0.0;
        
        // Get entity's neighbors
        let neighbors = self.get_neighbors(entity_key);
        
        // Score based on connections to context entities
        for neighbor in neighbors {
            if let Some(context_entity) = context_map.get(&neighbor) {
                score += context_entity.similarity;
            }
        }
        
        // Normalize score
        if context_map.is_empty() {
            0.0
        } else {
            score / context_map.len() as f32
        }
    }

    /// Batch query multiple embeddings
    pub fn batch_query(&self, query_embeddings: &[Vec<f32>], k: usize) -> Result<Vec<QueryResult>> {
        let mut results = Vec::with_capacity(query_embeddings.len());
        
        for query_embedding in query_embeddings {
            let result = self.simple_query(query_embedding, k)?;
            results.push(result);
        }
        
        Ok(results)
    }

    /// Get query statistics
    pub fn get_query_stats(&self) -> QueryStats {
        let entity_count = self.entity_count();
        let relationship_count = self.relationship_count();
        let (cache_size, cache_capacity, cache_hit_rate) = self.cache_stats();
        
        QueryStats {
            entity_count,
            relationship_count: relationship_count as usize,
            cache_size,
            cache_capacity,
            cache_hit_rate,
            average_degree: if entity_count > 0 {
                (relationship_count as f64 * 2.0) / entity_count as f64
            } else {
                0.0
            },
        }
    }

    /// Validate query parameters
    pub fn validate_query_params(&self, query_embedding: &[f32], k: usize) -> Result<()> {
        // Validate embedding dimension
        if query_embedding.len() != self.embedding_dim {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.embedding_dim,
                actual: query_embedding.len(),
            });
        }
        
        // Validate k parameter
        if k == 0 {
            return Err(GraphError::InvalidInput("k must be greater than 0".to_string()));
        }
        
        if k > self.entity_count() {
            return Err(GraphError::InvalidInput(
                format!("k ({}) cannot be greater than entity count ({})", k, self.entity_count())
            ));
        }
        
        Ok(())
    }

    /// Explain query (for debugging)
    pub fn explain_query(&self, query_embedding: &[f32], k: usize) -> Result<QueryExplanation> {
        let start_time = Instant::now();
        
        // Validate parameters
        self.validate_query_params(query_embedding, k)?;
        
        // Get similarity results with explanations
        let similar_entities = self.similarity_search(query_embedding, k)?;
        
        let mut explanations = Vec::new();
        for (entity_key, similarity) in similar_entities {
            if let Some((_, data)) = self.get_entity(entity_key) {
                let degree = self.get_entity_degree(entity_key);
                let explanation = EntityExplanation {
                    entity_key,
                    similarity_score: similarity,
                    degree,
                    properties_count: data.properties.len(),
                };
                explanations.push(explanation);
            }
        }
        
        Ok(QueryExplanation {
            query_time: start_time.elapsed(),
            entities_examined: self.entity_count(),
            results_returned: explanations.len(),
            explanations,
        })
    }
}

/// Query statistics
#[derive(Debug, Clone)]
pub struct QueryStats {
    pub entity_count: usize,
    pub relationship_count: usize,
    pub cache_size: usize,
    pub cache_capacity: usize,
    pub cache_hit_rate: f64,
    pub average_degree: f64,
}

/// Query explanation for debugging
#[derive(Debug, Clone)]
pub struct QueryExplanation {
    pub query_time: std::time::Duration,
    pub entities_examined: usize,
    pub results_returned: usize,
    pub explanations: Vec<EntityExplanation>,
}

/// Entity explanation for debugging
#[derive(Debug, Clone)]
pub struct EntityExplanation {
    pub entity_key: EntityKey,
    pub similarity_score: f32,
    pub degree: usize,
    pub properties_count: usize,
}

impl QueryStats {
    /// Check if graph is well-suited for queries
    pub fn is_query_friendly(&self) -> bool {
        self.entity_count > 0 && 
        self.average_degree > 1.0 && 
        self.cache_hit_rate > 0.5
    }
    
    /// Get graph density
    pub fn graph_density(&self) -> f64 {
        if self.entity_count <= 1 {
            return 0.0;
        }
        
        let max_relationships = self.entity_count * (self.entity_count - 1);
        self.relationship_count as f64 / max_relationships as f64
    }
    
    /// Check if caching is effective
    pub fn is_caching_effective(&self) -> bool {
        self.cache_hit_rate > 0.3 && self.cache_size > 0
    }
}