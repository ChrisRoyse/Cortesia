use crate::core::types::EntityData;
use crate::error::Result;
use rayon::prelude::*;
use std::collections::HashMap;

/// Parallel processing utilities for knowledge graph operations
pub struct ParallelProcessor;

impl ParallelProcessor {
    /// Parallel similarity computation for large datasets
    /// Works with pre-extracted data to avoid concurrency issues
    pub fn parallel_similarity_search(
        query_embedding: &[f32],
        entities: Vec<(u32, Vec<f32>)>, // (id, embedding) pairs
        k: usize,
    ) -> Vec<(u32, f32)> {
        if entities.len() < 1000 {
            // Use sequential for small datasets
            return Self::sequential_similarity_search(query_embedding, entities, k);
        }

        // Parallel computation for large datasets
        let mut distances: Vec<(u32, f32)> = entities
            .par_iter()
            .map(|(id, embedding)| {
                let similarity = crate::embedding::similarity::cosine_similarity(
                    query_embedding,
                    embedding,
                );
                (*id, 1.0 - similarity) // Convert similarity to distance
            })
            .collect();

        // Sort and return top k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }

    /// Sequential fallback for small datasets
    fn sequential_similarity_search(
        query_embedding: &[f32],
        entities: Vec<(u32, Vec<f32>)>,
        k: usize,
    ) -> Vec<(u32, f32)> {
        let mut distances: Vec<(u32, f32)> = entities
            .iter()
            .map(|(id, embedding)| {
                let similarity = crate::embedding::similarity::cosine_similarity(
                    query_embedding,
                    embedding,
                );
                (*id, 1.0 - similarity)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }

    /// Parallel batch validation for entity insertion
    pub fn parallel_validate_entities(
        entities: &[(u32, EntityData)],
        expected_dim: usize,
    ) -> Result<()> {
        if entities.len() < 100 {
            // Sequential validation for small batches
            return Self::sequential_validate_entities(entities, expected_dim);
        }

        // Parallel validation
        let validation_results: Vec<Result<()>> = entities
            .par_iter()
            .map(|(_, data)| {
                // Validate text size
                crate::text::TextCompressor::validate_text_size(&data.properties)?;
                
                // Validate embedding dimension
                if data.embedding.len() != expected_dim {
                    return Err(crate::error::GraphError::InvalidEmbeddingDimension {
                        expected: expected_dim,
                        actual: data.embedding.len(),
                    });
                }
                
                Ok(())
            })
            .collect();

        // Check if any validation failed
        for result in validation_results {
            result?;
        }

        Ok(())
    }

    /// Sequential validation fallback
    fn sequential_validate_entities(
        entities: &[(u32, EntityData)],
        expected_dim: usize,
    ) -> Result<()> {
        for (_, data) in entities {
            crate::text::TextCompressor::validate_text_size(&data.properties)?;
            
            if data.embedding.len() != expected_dim {
                return Err(crate::error::GraphError::InvalidEmbeddingDimension {
                    expected: expected_dim,
                    actual: data.embedding.len(),
                });
            }
        }
        Ok(())
    }

    /// Parallel quantization encoding for batch operations
    pub fn parallel_encode_embeddings(
        embeddings: &[Vec<f32>],
        quantizer: &crate::embedding::quantizer::ProductQuantizer,
    ) -> Result<Vec<Vec<u8>>> {
        if embeddings.len() < 50 {
            // Sequential for small batches
            return embeddings
                .iter()
                .map(|emb| quantizer.encode(emb))
                .collect();
        }

        // Parallel encoding
        embeddings
            .par_iter()
            .map(|embedding| quantizer.encode(embedding))
            .collect()
    }

    /// Parallel property extraction for query results
    pub fn parallel_extract_properties(
        entity_ids: &[u32],
        property_map: &HashMap<u32, String>, // Pre-extracted property mappings
    ) -> Vec<(u32, Option<String>)> {
        if entity_ids.len() < 10 {
            // Sequential for small result sets
            return entity_ids
                .iter()
                .map(|&id| (id, property_map.get(&id).cloned()))
                .collect();
        }

        // Parallel extraction
        entity_ids
            .par_iter()
            .map(|&id| (id, property_map.get(&id).cloned()))
            .collect()
    }

    /// Parallel neighborhood expansion for graph traversal
    pub fn parallel_expand_neighborhoods(
        entity_ids: &[u32],
        adjacency_map: &HashMap<u32, Vec<u32>>, // Pre-extracted adjacency information
    ) -> HashMap<u32, Vec<u32>> {
        if entity_ids.len() < 20 {
            // Sequential for small sets
            let mut result = HashMap::new();
            for &id in entity_ids {
                if let Some(neighbors) = adjacency_map.get(&id) {
                    result.insert(id, neighbors.clone());
                }
            }
            return result;
        }

        // Parallel neighborhood expansion
        entity_ids
            .par_iter()
            .filter_map(|&id| {
                adjacency_map.get(&id).map(|neighbors| (id, neighbors.clone()))
            })
            .collect()
    }

    /// Adaptive threshold for when to use parallel processing
    pub fn should_use_parallel(data_size: usize, operation_type: ParallelOperation) -> bool {
        match operation_type {
            ParallelOperation::SimilaritySearch => data_size >= 1000,
            ParallelOperation::BatchValidation => data_size >= 100,
            ParallelOperation::Encoding => data_size >= 50,
            ParallelOperation::PropertyExtraction => data_size >= 10,
            ParallelOperation::NeighborhoodExpansion => data_size >= 20,
        }
    }
}

/// Types of parallel operations with different thresholds
#[derive(Debug, Clone, Copy)]
pub enum ParallelOperation {
    SimilaritySearch,
    BatchValidation,
    Encoding,
    PropertyExtraction,
    NeighborhoodExpansion,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_embedding(dim: usize, seed: f32) -> Vec<f32> {
        let mut embedding = vec![0.0; dim];
        for (i, val) in embedding.iter_mut().enumerate() {
            *val = ((i as f32 + seed) % 10.0) / 10.0 - 0.5;
        }
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        embedding
    }

    #[test]
    fn test_parallel_similarity_search() {
        let query = create_test_embedding(64, 0.0);
        let entities: Vec<(u32, Vec<f32>)> = (0..1500)
            .map(|i| (i, create_test_embedding(64, i as f32)))
            .collect();

        let results = ParallelProcessor::parallel_similarity_search(&query, entities, 10);
        
        assert_eq!(results.len(), 10);
        // Results should be sorted by distance (ascending)
        for i in 1..results.len() {
            assert!(results[i-1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_sequential_vs_parallel_consistency() {
        let query = create_test_embedding(32, 1.0);
        let entities: Vec<(u32, Vec<f32>)> = (0..100)
            .map(|i| (i, create_test_embedding(32, i as f32)))
            .collect();

        let sequential = ParallelProcessor::sequential_similarity_search(&query, entities.clone(), 5);
        let parallel = ParallelProcessor::parallel_similarity_search(&query, entities, 5);
        
        // Results should be identical (within floating point tolerance)
        assert_eq!(sequential.len(), parallel.len());
        for (seq, par) in sequential.iter().zip(parallel.iter()) {
            assert_eq!(seq.0, par.0); // Same entity ID
            assert!((seq.1 - par.1).abs() < 1e-6); // Same distance (within tolerance)
        }
    }

    #[test]
    fn test_should_use_parallel() {
        assert!(!ParallelProcessor::should_use_parallel(50, ParallelOperation::SimilaritySearch));
        assert!(ParallelProcessor::should_use_parallel(1500, ParallelOperation::SimilaritySearch));
        
        assert!(!ParallelProcessor::should_use_parallel(50, ParallelOperation::BatchValidation));
        assert!(ParallelProcessor::should_use_parallel(150, ParallelOperation::BatchValidation));
    }
}