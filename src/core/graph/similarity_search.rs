//! Similarity search operations for knowledge graph

use super::graph_core::{KnowledgeGraph, MAX_SIMILARITY_SEARCH_TIME};
use crate::core::types::{EntityKey};
use crate::storage::lru_cache::QueryCacheKey;
use crate::trace_function;
use crate::error::{GraphError, Result};
use std::time::Instant;

impl KnowledgeGraph {
    /// Perform similarity search with intelligent index selection
    pub fn similarity_search(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(EntityKey, f32)>> {
        let _trace = if let Some(profiler) = &self.runtime_profiler {
            Some(trace_function!(profiler, "similarity_search", query_embedding.len(), k))
        } else {
            None
        };
        
        let start_time = Instant::now();
        
        // Validate query embedding
        if query_embedding.len() != self.embedding_dim {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.embedding_dim,
                actual: query_embedding.len(),
            });
        }
        
        // Check cache first
        let cache_key = QueryCacheKey::new(query_embedding, k, 16);
        {
            let mut cache = self.similarity_cache.write();
            if let Some(cached_results) = cache.get(&cache_key) {
                // Convert cached u32 results back to EntityKey
                let entity_id_map = self.entity_id_map.read();
                let converted_results = cached_results.iter()
                    .filter_map(|(id, score)| {
                        entity_id_map.get(id).map(|key| (*key, *score))
                    })
                    .collect();
                return Ok(converted_results);
            }
        }
        
        // Choose index based on graph size and k
        let entity_count = self.entity_count();
        let results = if entity_count < 1000 {
            // For small graphs, use flat index
            self.similarity_search_flat(query_embedding, k)?
        } else if k <= 10 {
            // For small k, use HNSW for speed
            self.similarity_search_hnsw(query_embedding, k)?
        } else if k >= entity_count / 2 {
            // For large k, use LSH for efficiency
            self.similarity_search_lsh(query_embedding, k)?
        } else {
            // For medium k, use spatial index
            self.similarity_search_spatial(query_embedding, k)?
        };
        
        // Cache results (convert EntityKey to u32 for caching)
        {
            let entity_id_map = self.entity_id_map.read();
            let cacheable_results: Vec<(u32, f32)> = results.iter()
                .filter_map(|(key, score)| {
                    // Find the u32 ID for this EntityKey
                    entity_id_map.iter()
                        .find(|(_, k)| *k == key)
                        .map(|(id, _)| (*id, *score))
                })
                .collect();
            let mut cache = self.similarity_cache.write();
            cache.insert(cache_key, cacheable_results);
        }
        
        // Check if search took too long
        if start_time.elapsed() > MAX_SIMILARITY_SEARCH_TIME {
            #[cfg(debug_assertions)]
            log::warn!("Similarity search took longer than expected: {:.2}ms", start_time.elapsed().as_millis());
        }
        
        Ok(results)
    }

    /// Parallel similarity search for large queries
    pub fn similarity_search_parallel(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(EntityKey, f32)>> {
        let start_time = Instant::now();
        
        // Validate query embedding
        if query_embedding.len() != self.embedding_dim {
            return Err(GraphError::InvalidEmbeddingDimension {
                expected: self.embedding_dim,
                actual: query_embedding.len(),
            });
        }
        
        // Check cache first
        let cache_key = QueryCacheKey::new(query_embedding, k, 16);
        {
            let mut cache = self.similarity_cache.write();
            if let Some(cached_results) = cache.get(&cache_key) {
                // Convert cached u32 results back to EntityKey
                let entity_id_map = self.entity_id_map.read();
                let converted_results = cached_results.iter()
                    .filter_map(|(id, score)| {
                        entity_id_map.get(id).map(|key| (*key, *score))
                    })
                    .collect();
                return Ok(converted_results);
            }
        }
        
        // Use parallel search for large datasets
        let entity_count = self.entity_count();
        let results = if entity_count > 10000 {
            self.similarity_search_parallel_impl(query_embedding, k)?
        } else {
            // Fall back to regular search for smaller datasets
            self.similarity_search(query_embedding, k)?
        };
        
        // Cache results (convert EntityKey to u32 for caching)
        {
            let entity_id_map = self.entity_id_map.read();
            let cacheable_results: Vec<(u32, f32)> = results.iter()
                .filter_map(|(key, score)| {
                    // Find the u32 ID for this EntityKey
                    entity_id_map.iter()
                        .find(|(_, k)| *k == key)
                        .map(|(id, _)| (*id, *score))
                })
                .collect();
            let mut cache = self.similarity_cache.write();
            cache.insert(cache_key, cacheable_results);
        }
        
        #[cfg(debug_assertions)]
        log::trace!("Parallel similarity search completed in {:.2}ms", start_time.elapsed().as_millis());
        
        Ok(results)
    }

    /// Similarity search using flat index
    fn similarity_search_flat(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(EntityKey, f32)>> {
        let flat_index = self.flat_index.read();
        let results = flat_index.k_nearest_neighbors(query_embedding, k);
        
        // Convert u32 to EntityKey
        let entity_id_map = self.entity_id_map.read();
        let converted_results = results.into_iter()
            .filter_map(|(id, score)| {
                // Find the EntityKey for this u32 ID
                entity_id_map.get(&id).map(|key| (*key, score))
            })
            .collect();
        
        Ok(converted_results)
    }

    /// Similarity search using HNSW index
    fn similarity_search_hnsw(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(EntityKey, f32)>> {
        let hnsw_index = self.hnsw_index.read();
        let results = hnsw_index.search(query_embedding, k);
        
        // Convert u32 to EntityKey
        let entity_id_map = self.entity_id_map.read();
        let converted_results = results.into_iter()
            .filter_map(|(id, score)| {
                // Find the EntityKey for this u32 ID
                entity_id_map.get(&id).map(|key| (*key, score))
            })
            .collect();
        
        Ok(converted_results)
    }

    /// Similarity search using LSH index
    fn similarity_search_lsh(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(EntityKey, f32)>> {
        let lsh_index = self.lsh_index.read();
        let results = lsh_index.search(query_embedding, k);
        
        // Convert u32 to EntityKey
        let entity_id_map = self.entity_id_map.read();
        let converted_results = results.into_iter()
            .filter_map(|(id, score)| {
                // Find the EntityKey for this u32 ID
                entity_id_map.get(&id).map(|key| (*key, score))
            })
            .collect();
        
        Ok(converted_results)
    }

    /// Similarity search using spatial index
    fn similarity_search_spatial(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(EntityKey, f32)>> {
        let spatial_index = self.spatial_index.read();
        let results = spatial_index.k_nearest_neighbors(query_embedding, k);
        
        // Convert u32 to EntityKey
        let entity_id_map = self.entity_id_map.read();
        let converted_results = results.into_iter()
            .filter_map(|(id, score)| {
                // Find the EntityKey for this u32 ID
                entity_id_map.get(&id).map(|key| (*key, score))
            })
            .collect();
        
        Ok(converted_results)
    }

    /// Parallel similarity search implementation
    fn similarity_search_parallel_impl(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(EntityKey, f32)>> {
        use rayon::prelude::*;
        
        // Get all entity keys
        let entity_keys = self.get_all_entity_keys();
        
        // Parallel computation of similarities
        let mut similarities: Vec<(EntityKey, f32)> = entity_keys
            .par_iter()
            .filter_map(|&key| {
                // Get entity embedding
                if let Some(embedding) = self.get_entity_embedding(key) {
                    // Calculate similarity
                    let similarity = self.calculate_cosine_similarity(query_embedding, &embedding);
                    Some((key, similarity))
                } else {
                    None
                }
            })
            .collect();
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top k results
        similarities.truncate(k);
        Ok(similarities)
    }

    /// Similarity search with threshold
    pub fn similarity_search_threshold(&self, query_embedding: &[f32], threshold: f32) -> Result<Vec<(EntityKey, f32)>> {
        // Start with a large k and filter
        let large_k = std::cmp::min(1000, self.entity_count());
        let results = self.similarity_search(query_embedding, large_k)?;
        
        // Filter by threshold
        let filtered_results: Vec<(EntityKey, f32)> = results
            .into_iter()
            .filter(|(_, similarity)| *similarity >= threshold)
            .collect();
        
        Ok(filtered_results)
    }

    /// Multi-query similarity search
    pub fn similarity_search_multi(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<(EntityKey, f32)>>> {
        let mut results = Vec::with_capacity(queries.len());
        
        for query in queries {
            let query_results = self.similarity_search(query, k)?;
            results.push(query_results);
        }
        
        Ok(results)
    }

    /// Similarity search with entity filtering
    pub fn similarity_search_filtered<F>(&self, query_embedding: &[f32], k: usize, filter: F) -> Result<Vec<(EntityKey, f32)>>
    where
        F: Fn(EntityKey) -> bool,
    {
        // Get a larger set of candidates
        let candidate_k = std::cmp::min(k * 5, self.entity_count());
        let candidates = self.similarity_search(query_embedding, candidate_k)?;
        
        // Filter and take top k
        let filtered_results: Vec<(EntityKey, f32)> = candidates
            .into_iter()
            .filter(|(key, _)| filter(*key))
            .take(k)
            .collect();
        
        Ok(filtered_results)
    }

    /// Calculate cosine similarity between two vectors
    fn calculate_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (magnitude_a * magnitude_b)
    }

    /// Calculate euclidean distance between two vectors
    fn calculate_euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }
        
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Find most similar entity
    pub fn find_most_similar(&self, query_embedding: &[f32]) -> Result<Option<(EntityKey, f32)>> {
        let results = self.similarity_search(query_embedding, 1)?;
        Ok(results.into_iter().next())
    }

    /// Find least similar entity
    pub fn find_least_similar(&self, query_embedding: &[f32]) -> Result<Option<(EntityKey, f32)>> {
        // Get all entities and find the one with minimum similarity
        let entity_keys = self.get_all_entity_keys();
        let mut min_similarity = f32::INFINITY;
        let mut least_similar = None;
        
        for key in entity_keys {
            if let Some(embedding) = self.get_entity_embedding(key) {
                let similarity = self.calculate_cosine_similarity(query_embedding, &embedding);
                if similarity < min_similarity {
                    min_similarity = similarity;
                    least_similar = Some((key, similarity));
                }
            }
        }
        
        Ok(least_similar)
    }

    /// Get average similarity of query to all entities
    pub fn get_average_similarity(&self, query_embedding: &[f32]) -> Result<f32> {
        let entity_keys = self.get_all_entity_keys();
        let mut total_similarity = 0.0;
        let mut count = 0;
        
        for key in entity_keys {
            if let Some(embedding) = self.get_entity_embedding(key) {
                let similarity = self.calculate_cosine_similarity(query_embedding, &embedding);
                total_similarity += similarity;
                count += 1;
            }
        }
        
        if count == 0 {
            Ok(0.0)
        } else {
            Ok(total_similarity / count as f32)
        }
    }

    /// Get similarity statistics
    pub fn get_similarity_stats(&self, query_embedding: &[f32]) -> Result<SimilarityStats> {
        let entity_keys = self.get_all_entity_keys();
        let mut similarities = Vec::new();
        
        for key in entity_keys {
            if let Some(embedding) = self.get_entity_embedding(key) {
                let similarity = self.calculate_cosine_similarity(query_embedding, &embedding);
                similarities.push(similarity);
            }
        }
        
        if similarities.is_empty() {
            return Ok(SimilarityStats::default());
        }
        
        similarities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let count = similarities.len();
        let min = similarities[0];
        let max = similarities[count - 1];
        let median = if count % 2 == 0 {
            (similarities[count / 2 - 1] + similarities[count / 2]) / 2.0
        } else {
            similarities[count / 2]
        };
        let mean = similarities.iter().sum::<f32>() / count as f32;
        
        // Calculate standard deviation
        let variance = similarities.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / count as f32;
        let std_dev = variance.sqrt();
        
        Ok(SimilarityStats {
            count,
            min,
            max,
            mean,
            median,
            std_dev,
        })
    }

    /// Quantized similarity search (fallback for memory efficiency)
    pub fn similarity_search_quantized(&self, query_embedding: &[f32], k: usize) -> Result<Vec<(EntityKey, f32)>> {
        // Quantize query embedding
        let quantizer = self.quantizer.read();
        let quantized_query = quantizer.encode(query_embedding)?;
        
        // Get all entity keys
        let entity_keys = self.get_all_entity_keys();
        let mut similarities = Vec::new();
        
        // Calculate similarities using quantized embeddings
        let embedding_bank = self.embedding_bank.read();
        let entity_store = self.entity_store.read();
        
        for key in entity_keys {
            if let Some(meta) = entity_store.get(key) {
                let offset = meta.embedding_offset as usize;
                let quantized_size = quantizer.num_subspaces();
                
                if offset + quantized_size <= embedding_bank.len() {
                    let quantized_embedding = &embedding_bank[offset..offset + quantized_size];
                    
                    // Calculate quantized similarity (simplified)
                    let similarity = self.calculate_quantized_similarity(&quantized_query, quantized_embedding);
                    similarities.push((key, similarity));
                }
            }
        }
        
        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Return top k results
        similarities.truncate(k);
        Ok(similarities)
    }

    /// Calculate similarity between quantized embeddings
    fn calculate_quantized_similarity(&self, a: &[u8], b: &[u8]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        // Simple Hamming distance-based similarity
        let mut matches = 0;
        for (byte_a, byte_b) in a.iter().zip(b.iter()) {
            if byte_a == byte_b {
                matches += 1;
            }
        }
        
        matches as f32 / a.len() as f32
    }
}

/// Similarity search statistics
#[derive(Debug, Clone)]
pub struct SimilarityStats {
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
    pub std_dev: f32,
}

impl Default for SimilarityStats {
    fn default() -> Self {
        Self {
            count: 0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
        }
    }
}

impl SimilarityStats {
    /// Check if similarities are well-distributed
    pub fn is_well_distributed(&self) -> bool {
        self.std_dev > 0.1 && (self.max - self.min) > 0.5
    }
    
    /// Get similarity range
    pub fn range(&self) -> f32 {
        self.max - self.min
    }
    
    /// Check if query is an outlier (very different from all entities)
    pub fn is_outlier_query(&self) -> bool {
        self.max < 0.3 && self.mean < 0.2
    }
    
    /// Check if query is very similar to many entities
    pub fn is_common_query(&self) -> bool {
        self.mean > 0.7 && self.median > 0.7
    }
}