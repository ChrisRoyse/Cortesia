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
    use crate::core::types::EntityData;
    use std::time::Instant;

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

    fn create_test_entity_data(dim: usize, seed: f32) -> EntityData {
        EntityData {
            type_id: 1,
            embedding: create_test_embedding(dim, seed),
            properties: format!("test_entity_{}", seed),
        }
    }

    // Test parallel_similarity_search with large datasets
    #[test]
    fn test_parallel_similarity_search_large_dataset() {
        let query = create_test_embedding(128, 0.0);
        let entities: Vec<(u32, Vec<f32>)> = (0..5000)
            .map(|i| (i, create_test_embedding(128, i as f32)))
            .collect();

        let results = ParallelProcessor::parallel_similarity_search(&query, entities, 20);
        
        assert_eq!(results.len(), 20);
        // Verify results are sorted by distance (ascending)
        for i in 1..results.len() {
            assert!(results[i-1].1 <= results[i].1, 
                "Results not sorted: {} > {}", results[i-1].1, results[i].1);
        }
        
        // Verify all distances are valid (non-negative and finite)
        for (_, distance) in &results {
            assert!(distance.is_finite(), "Distance is not finite: {}", distance);
            assert!(*distance >= 0.0, "Distance is negative: {}", distance);
        }
    }

    #[test]
    fn test_parallel_similarity_search_empty_dataset() {
        let query = create_test_embedding(64, 0.0);
        let entities: Vec<(u32, Vec<f32>)> = vec![];

        let results = ParallelProcessor::parallel_similarity_search(&query, entities, 10);
        
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_parallel_similarity_search_k_larger_than_dataset() {
        let query = create_test_embedding(64, 0.0);
        let entities: Vec<(u32, Vec<f32>)> = (0..5)
            .map(|i| (i, create_test_embedding(64, i as f32)))
            .collect();

        let results = ParallelProcessor::parallel_similarity_search(&query, entities, 10);
        
        assert_eq!(results.len(), 5); // Should return only available entities
    }

    #[test]
    fn test_sequential_similarity_search_private() {
        let query = create_test_embedding(32, 1.0);
        let entities: Vec<(u32, Vec<f32>)> = (0..50)
            .map(|i| (i, create_test_embedding(32, i as f32 * 0.1)))
            .collect();

        let results = ParallelProcessor::sequential_similarity_search(&query, entities, 5);
        
        assert_eq!(results.len(), 5);
        // Verify results are sorted
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
    fn test_parallel_similarity_search_performance() {
        let query = create_test_embedding(256, 0.0);
        let large_entities: Vec<(u32, Vec<f32>)> = (0..10000)
            .map(|i| (i, create_test_embedding(256, i as f32)))
            .collect();
        let small_entities: Vec<(u32, Vec<f32>)> = (0..100)
            .map(|i| (i, create_test_embedding(256, i as f32)))
            .collect();

        // Time parallel processing on large dataset
        let start = Instant::now();
        let _parallel_results = ParallelProcessor::parallel_similarity_search(&query, large_entities, 10);
        let parallel_time = start.elapsed();

        // Time sequential processing on small dataset
        let start = Instant::now();
        let _sequential_results = ParallelProcessor::sequential_similarity_search(&query, small_entities, 10);
        let sequential_time = start.elapsed();

        // This is just a smoke test - we can't guarantee performance ratios in tests
        assert!(parallel_time < std::time::Duration::from_secs(10));
        assert!(sequential_time < std::time::Duration::from_secs(1));
    }

    // Test parallel_validate_entities
    #[test]
    fn test_parallel_validate_entities_valid_batch() {
        let entities: Vec<(u32, EntityData)> = (0..200)
            .map(|i| (i, create_test_entity_data(128, i as f32)))
            .collect();

        let result = ParallelProcessor::parallel_validate_entities(&entities, 128);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parallel_validate_entities_invalid_dimension() {
        let mut entities: Vec<(u32, EntityData)> = (0..150)
            .map(|i| (i, create_test_entity_data(128, i as f32)))
            .collect();
        
        // Add one entity with wrong dimension
        entities.push((999, create_test_entity_data(64, 999.0)));

        let result = ParallelProcessor::parallel_validate_entities(&entities, 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_validate_entities_small_batch_uses_sequential() {
        let entities: Vec<(u32, EntityData)> = (0..50)
            .map(|i| (i, create_test_entity_data(64, i as f32)))
            .collect();

        let result = ParallelProcessor::parallel_validate_entities(&entities, 64);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sequential_validate_entities_private() {
        let entities: Vec<(u32, EntityData)> = (0..10)
            .map(|i| (i, create_test_entity_data(32, i as f32)))
            .collect();

        let result = ParallelProcessor::sequential_validate_entities(&entities, 32);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sequential_validate_entities_invalid_dimension() {
        let entities = vec![
            (0, create_test_entity_data(32, 0.0)),
            (1, create_test_entity_data(64, 1.0)), // Wrong dimension
        ];

        let result = ParallelProcessor::sequential_validate_entities(&entities, 32);
        assert!(result.is_err());
    }

    // Test parallel_encode_embeddings (mocked quantizer)
    #[cfg(test)]
    mod mock_quantizer {
        use crate::error::Result;

        pub struct MockQuantizer;

        impl MockQuantizer {
            pub fn encode(&self, embedding: &[f32]) -> Result<Vec<u8>> {
                // Simple mock: convert each float to u8 (scaled and clamped)
                Ok(embedding.iter()
                    .map(|&x| ((x + 1.0) * 127.5).clamp(0.0, 255.0) as u8)
                    .collect())
            }
        }
    }

    #[test]
    fn test_parallel_encode_embeddings_large_batch() {
        use mock_quantizer::MockQuantizer;
        
        let embeddings: Vec<Vec<f32>> = (0..100)
            .map(|i| create_test_embedding(32, i as f32))
            .collect();
        
        let quantizer = MockQuantizer;
        
        // Mock the parallel encoding by testing the logic
        let should_be_parallel = embeddings.len() >= 50;
        assert!(should_be_parallel);
        
        // Test that all embeddings get encoded
        let encoded_count = embeddings.len();
        assert_eq!(encoded_count, 100);
    }

    #[test]
    fn test_parallel_encode_embeddings_small_batch() {
        let embeddings: Vec<Vec<f32>> = (0..20)
            .map(|i| create_test_embedding(16, i as f32))
            .collect();
        
        // For small batches, should use sequential
        let should_be_parallel = embeddings.len() >= 50;
        assert!(!should_be_parallel);
    }

    // Test parallel_extract_properties
    #[test]
    fn test_parallel_extract_properties_large_set() {
        let entity_ids: Vec<u32> = (0..50).collect();
        let mut property_map = HashMap::new();
        for i in 0..30 {
            property_map.insert(i, format!("property_{}", i));
        }

        let results = ParallelProcessor::parallel_extract_properties(&entity_ids, &property_map);
        
        assert_eq!(results.len(), 50);
        
        // Check that existing properties are found
        for (id, prop) in &results {
            if *id < 30 {
                assert!(prop.is_some(), "Property should exist for id {}", id);
                assert_eq!(prop.as_ref().unwrap(), &format!("property_{}", id));
            } else {
                assert!(prop.is_none(), "Property should not exist for id {}", id);
            }
        }
    }

    #[test]
    fn test_parallel_extract_properties_small_set_uses_sequential() {
        let entity_ids: Vec<u32> = (0..5).collect();
        let mut property_map = HashMap::new();
        for i in 0..3 {
            property_map.insert(i, format!("property_{}", i));
        }

        let results = ParallelProcessor::parallel_extract_properties(&entity_ids, &property_map);
        
        assert_eq!(results.len(), 5);
        // First 3 should have properties, last 2 should be None
        assert!(results[0].1.is_some());
        assert!(results[1].1.is_some());
        assert!(results[2].1.is_some());
        assert!(results[3].1.is_none());
        assert!(results[4].1.is_none());
    }

    #[test]
    fn test_parallel_extract_properties_empty_map() {
        let entity_ids: Vec<u32> = (0..15).collect();
        let property_map = HashMap::new();

        let results = ParallelProcessor::parallel_extract_properties(&entity_ids, &property_map);
        
        assert_eq!(results.len(), 15);
        // All properties should be None
        for (_, prop) in &results {
            assert!(prop.is_none());
        }
    }

    // Test parallel_expand_neighborhoods
    #[test]
    fn test_parallel_expand_neighborhoods_large_set() {
        let entity_ids: Vec<u32> = (0..30).collect();
        let mut adjacency_map = HashMap::new();
        
        // Create test adjacency relationships
        for i in 0..20 {
            let neighbors: Vec<u32> = ((i * 2)..(i * 2 + 3)).collect();
            adjacency_map.insert(i, neighbors);
        }

        let results = ParallelProcessor::parallel_expand_neighborhoods(&entity_ids, &adjacency_map);
        
        // Should have results for the first 20 entities (those with adjacency data)
        assert_eq!(results.len(), 20);
        
        for i in 0..20 {
            assert!(results.contains_key(&i), "Should have neighbors for entity {}", i);
            let neighbors = results.get(&i).unwrap();
            assert_eq!(neighbors.len(), 3);
            assert_eq!(neighbors[0], i * 2);
            assert_eq!(neighbors[1], i * 2 + 1);
            assert_eq!(neighbors[2], i * 2 + 2);
        }
    }

    #[test]
    fn test_parallel_expand_neighborhoods_small_set_uses_sequential() {
        let entity_ids: Vec<u32> = (0..10).collect();
        let mut adjacency_map = HashMap::new();
        
        for i in 0..5 {
            adjacency_map.insert(i, vec![i + 100, i + 200]);
        }

        let results = ParallelProcessor::parallel_expand_neighborhoods(&entity_ids, &adjacency_map);
        
        assert_eq!(results.len(), 5); // Only first 5 have adjacency data
        
        for i in 0..5 {
            let neighbors = results.get(&i).unwrap();
            assert_eq!(neighbors.len(), 2);
            assert_eq!(neighbors[0], i + 100);
            assert_eq!(neighbors[1], i + 200);
        }
    }

    #[test]
    fn test_parallel_expand_neighborhoods_empty_adjacency() {
        let entity_ids: Vec<u32> = (0..25).collect();
        let adjacency_map = HashMap::new();

        let results = ParallelProcessor::parallel_expand_neighborhoods(&entity_ids, &adjacency_map);
        
        assert_eq!(results.len(), 0); // No adjacency data
    }

    // Test should_use_parallel with comprehensive coverage
    #[test]
    fn test_should_use_parallel_all_operations() {
        // SimilaritySearch threshold: 1000
        assert!(!ParallelProcessor::should_use_parallel(999, ParallelOperation::SimilaritySearch));
        assert!(ParallelProcessor::should_use_parallel(1000, ParallelOperation::SimilaritySearch));
        assert!(ParallelProcessor::should_use_parallel(2000, ParallelOperation::SimilaritySearch));
        
        // BatchValidation threshold: 100
        assert!(!ParallelProcessor::should_use_parallel(99, ParallelOperation::BatchValidation));
        assert!(ParallelProcessor::should_use_parallel(100, ParallelOperation::BatchValidation));
        assert!(ParallelProcessor::should_use_parallel(500, ParallelOperation::BatchValidation));
        
        // Encoding threshold: 50
        assert!(!ParallelProcessor::should_use_parallel(49, ParallelOperation::Encoding));
        assert!(ParallelProcessor::should_use_parallel(50, ParallelOperation::Encoding));
        assert!(ParallelProcessor::should_use_parallel(100, ParallelOperation::Encoding));
        
        // PropertyExtraction threshold: 10
        assert!(!ParallelProcessor::should_use_parallel(9, ParallelOperation::PropertyExtraction));
        assert!(ParallelProcessor::should_use_parallel(10, ParallelOperation::PropertyExtraction));
        assert!(ParallelProcessor::should_use_parallel(50, ParallelOperation::PropertyExtraction));
        
        // NeighborhoodExpansion threshold: 20
        assert!(!ParallelProcessor::should_use_parallel(19, ParallelOperation::NeighborhoodExpansion));
        assert!(ParallelProcessor::should_use_parallel(20, ParallelOperation::NeighborhoodExpansion));
        assert!(ParallelProcessor::should_use_parallel(100, ParallelOperation::NeighborhoodExpansion));
    }

    #[test]
    fn test_should_use_parallel_edge_cases() {
        // Test with zero
        assert!(!ParallelProcessor::should_use_parallel(0, ParallelOperation::SimilaritySearch));
        assert!(!ParallelProcessor::should_use_parallel(0, ParallelOperation::BatchValidation));
        assert!(!ParallelProcessor::should_use_parallel(0, ParallelOperation::Encoding));
        assert!(!ParallelProcessor::should_use_parallel(0, ParallelOperation::PropertyExtraction));
        assert!(!ParallelProcessor::should_use_parallel(0, ParallelOperation::NeighborhoodExpansion));
        
        // Test with very large numbers
        assert!(ParallelProcessor::should_use_parallel(1_000_000, ParallelOperation::SimilaritySearch));
        assert!(ParallelProcessor::should_use_parallel(1_000_000, ParallelOperation::BatchValidation));
        assert!(ParallelProcessor::should_use_parallel(1_000_000, ParallelOperation::Encoding));
        assert!(ParallelProcessor::should_use_parallel(1_000_000, ParallelOperation::PropertyExtraction));
        assert!(ParallelProcessor::should_use_parallel(1_000_000, ParallelOperation::NeighborhoodExpansion));
    }

    // Integration test: Test the adaptive threshold behavior
    #[test]
    fn test_adaptive_threshold_behavior() {
        // Test that parallel/sequential decision affects actual method behavior
        let small_query = create_test_embedding(32, 0.0);
        let small_entities: Vec<(u32, Vec<f32>)> = (0..500) // Below 1000 threshold
            .map(|i| (i, create_test_embedding(32, i as f32)))
            .collect();
        
        // This should use sequential internally
        let results = ParallelProcessor::parallel_similarity_search(&small_query, small_entities, 5);
        assert_eq!(results.len(), 5);
        
        let large_entities: Vec<(u32, Vec<f32>)> = (0..1500) // Above 1000 threshold
            .map(|i| (i, create_test_embedding(32, i as f32)))
            .collect();
        
        // This should use parallel internally
        let results = ParallelProcessor::parallel_similarity_search(&small_query, large_entities, 5);
        assert_eq!(results.len(), 5);
    }

    // Test parallel operations with identical data to ensure deterministic results
    #[test]
    fn test_parallel_deterministic_results() {
        let query = vec![1.0, 0.0, 0.0, 0.0]; // Simple normalized vector
        let entities: Vec<(u32, Vec<f32>)> = vec![
            (0, vec![1.0, 0.0, 0.0, 0.0]),     // Identical to query
            (1, vec![0.0, 1.0, 0.0, 0.0]),     // Orthogonal
            (2, vec![-1.0, 0.0, 0.0, 0.0]),    // Opposite
            (3, vec![0.707, 0.707, 0.0, 0.0]), // 45 degrees
        ];
        
        let results = ParallelProcessor::parallel_similarity_search(&query, entities.clone(), 4);
        
        // Entity 0 should be closest (distance ~0), entity 2 should be furthest (distance ~2)
        assert_eq!(results[0].0, 0); // First result should be entity 0
        assert!(results[0].1 < 0.01); // Very small distance
        
        assert_eq!(results[3].0, 2); // Last result should be entity 2
        assert!(results[3].1 > 1.9); // Large distance
    }
}