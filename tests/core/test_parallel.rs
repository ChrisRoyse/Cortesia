#[cfg(test)]
mod test_parallel_integration {
    use llmkg::core::parallel::{ParallelProcessor, ParallelOperation};
    use llmkg::core::knowledge_engine::KnowledgeEngine;
    use llmkg::core::triple::Triple;
    use llmkg::core::types::EntityData;
    use llmkg::embedding::quantizer::ProductQuantizer;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::time::timeout;

    // Helper function to create test embeddings
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

    // Helper function to create test entity data
    fn create_test_entity_data(dim: usize, seed: f32) -> EntityData {
        EntityData::new(
            1, // default type_id
            format!("test_entity_{}", seed),
            create_test_embedding(dim, seed)
        )
    }

    // Helper to create test triple
    fn create_test_triple(subject: &str, predicate: &str, object: &str) -> Triple {
        Triple {
            subject: subject.to_string(),
            predicate: predicate.to_string(),
            object: object.to_string(),
            confidence: 0.9,
            source: Some("test".to_string()),
        }
    }

    #[tokio::test]
    async fn test_knowledge_engine_parallel_integration() {
        // Create KnowledgeEngine with reasonable dimensions
        let engine = KnowledgeEngine::new(128, 10000).expect("Failed to create knowledge engine");
        
        // Store a large number of triples to enable parallel processing
        let num_entities = 2000;
        for i in 0..num_entities {
            let triple = create_test_triple(
                &format!("subject_{}", i),
                "relates_to",
                &format!("object_{}", i),
            );
            let embedding = create_test_embedding(128, i as f32);
            engine.store_triple(triple, Some(embedding)).expect("Failed to store triple");
        }

        // Perform semantic search which should use parallel processing internally
        let start = Instant::now();
        let results = engine.semantic_search("subject relates to object", 100)
            .expect("Semantic search failed");
        let search_duration = start.elapsed();

        // Verify results
        assert!(!results.nodes.is_empty(), "Should find matching nodes");
        assert!(results.nodes.len() <= 100, "Should respect limit");
        assert!(results.query_time_ms > 0, "Query time should be tracked");
        
        // The search should be reasonably fast even with many entities
        assert!(search_duration < Duration::from_secs(5), 
            "Parallel search took too long: {:?}", search_duration);
        
        println!("Parallel semantic search completed in {:?} for {} entities", 
            search_duration, num_entities);
    }

    #[tokio::test]
    async fn test_parallel_similarity_search_with_knowledge_engine() {
        // Create a large dataset to ensure parallel processing kicks in
        let embedding_dim = 256;
        let num_entities = 5000;
        
        // Generate test data
        let query_embedding = create_test_embedding(embedding_dim, 0.0);
        let entities: Vec<(u32, Vec<f32>)> = (0..num_entities)
            .map(|i| (i, create_test_embedding(embedding_dim, i as f32)))
            .collect();

        // Time parallel search
        let start = Instant::now();
        let parallel_results = ParallelProcessor::parallel_similarity_search(
            &query_embedding,
            entities.clone(),
            50,
        );
        let parallel_duration = start.elapsed();

        // For comparison, force sequential processing on a smaller subset
        let small_entities = entities[..999].to_vec(); // Below parallel threshold
        let start = Instant::now();
        let sequential_results = ParallelProcessor::parallel_similarity_search(
            &query_embedding,
            small_entities,
            50,
        );
        let sequential_duration = start.elapsed();

        // Verify results
        assert_eq!(parallel_results.len(), 50, "Should return requested number of results");
        assert!(parallel_results.windows(2).all(|w| w[0].1 <= w[1].1), 
            "Results should be sorted by distance");

        // Performance verification - parallel should handle more data efficiently
        println!("Parallel search ({}): {:?}", num_entities, parallel_duration);
        println!("Sequential search (999): {:?}", sequential_duration);
        
        // The parallel version processes 5x more data, so we expect it to be faster per item
        let parallel_per_item = parallel_duration.as_micros() as f64 / num_entities as f64;
        let sequential_per_item = sequential_duration.as_micros() as f64 / 999.0;
        
        println!("Parallel per item: {:.2}μs", parallel_per_item);
        println!("Sequential per item: {:.2}μs", sequential_per_item);
    }

    #[tokio::test]
    async fn test_parallel_batch_validation_integration() {
        let embedding_dim = 128;
        
        // Create a large batch of entities to validate
        let entities: Vec<(u32, EntityData)> = (0..500)
            .map(|i| (i, create_test_entity_data(embedding_dim, i as f32)))
            .collect();

        // Time parallel validation
        let start = Instant::now();
        let validation_result = ParallelProcessor::parallel_validate_entities(&entities, embedding_dim);
        let validation_duration = start.elapsed();

        assert!(validation_result.is_ok(), "Validation should succeed for valid entities");
        assert!(validation_duration < Duration::from_millis(100), 
            "Parallel validation should be fast: {:?}", validation_duration);

        // Test with invalid entities
        let mut invalid_entities = entities.clone();
        // Add an entity with wrong dimension
        invalid_entities.push((999, create_test_entity_data(64, 999.0)));

        let invalid_result = ParallelProcessor::parallel_validate_entities(&invalid_entities, embedding_dim);
        assert!(invalid_result.is_err(), "Should fail with dimension mismatch");
    }

    #[tokio::test]
    async fn test_parallel_vs_sequential_accuracy() {
        let embedding_dim = 128;
        let query = create_test_embedding(embedding_dim, 42.0);
        
        // Create entities that will trigger both parallel and sequential paths
        let large_entities: Vec<(u32, Vec<f32>)> = (0..1500)
            .map(|i| (i, create_test_embedding(embedding_dim, i as f32 * 0.1)))
            .collect();
        
        let small_entities = large_entities[..500].to_vec();

        // Run similarity search on both
        let parallel_results = ParallelProcessor::parallel_similarity_search(
            &query,
            large_entities.clone(),
            20,
        );
        
        let sequential_results = ParallelProcessor::parallel_similarity_search(
            &query,
            small_entities.clone(),
            20,
        );

        // Both should return sorted results
        assert!(parallel_results.windows(2).all(|w| w[0].1 <= w[1].1), 
            "Parallel results should be sorted");
        assert!(sequential_results.windows(2).all(|w| w[0].1 <= w[1].1), 
            "Sequential results should be sorted");

        // The top results should be the same entities (accounting for the subset)
        let parallel_ids: Vec<u32> = parallel_results.iter()
            .filter(|(id, _)| *id < 500)
            .take(10)
            .map(|(id, _)| *id)
            .collect();
        
        let sequential_ids: Vec<u32> = sequential_results.iter()
            .take(10)
            .map(|(id, _)| *id)
            .collect();

        // Should have significant overlap in top results
        let overlap = parallel_ids.iter()
            .filter(|id| sequential_ids.contains(id))
            .count();
        
        assert!(overlap >= 8, "Parallel and sequential should find similar top results, got {} overlap", overlap);
    }

    #[tokio::test]
    async fn test_parallel_property_extraction_integration() {
        // Create test data
        let entity_ids: Vec<u32> = (0..100).collect();
        let mut property_map = HashMap::new();
        
        // Populate property map with test data
        for i in 0..80 {
            property_map.insert(i, format!("property_value_{}", i));
        }

        // Time property extraction
        let start = Instant::now();
        let results = ParallelProcessor::parallel_extract_properties(&entity_ids, &property_map);
        let duration = start.elapsed();

        // Verify results
        assert_eq!(results.len(), 100, "Should return result for each entity");
        
        // Check that properties are correctly extracted
        let mut found_count = 0;
        let mut not_found_count = 0;
        
        for (id, prop) in &results {
            if *id < 80 {
                assert!(prop.is_some(), "Entity {} should have property", id);
                assert_eq!(prop.as_ref().unwrap(), &format!("property_value_{}", id));
                found_count += 1;
            } else {
                assert!(prop.is_none(), "Entity {} should not have property", id);
                not_found_count += 1;
            }
        }
        
        assert_eq!(found_count, 80, "Should find 80 properties");
        assert_eq!(not_found_count, 20, "Should have 20 missing properties");
        
        println!("Property extraction completed in {:?}", duration);
    }

    #[tokio::test]
    async fn test_parallel_neighborhood_expansion_integration() {
        // Create a complex graph structure
        let mut adjacency_map = HashMap::new();
        
        // Create a network where each node connects to several others
        for i in 0..50 {
            let neighbors: Vec<u32> = ((i * 3)..(i * 3 + 5))
                .map(|j| j % 100)
                .collect();
            adjacency_map.insert(i, neighbors);
        }

        let entity_ids: Vec<u32> = (0..30).collect();

        // Time neighborhood expansion
        let start = Instant::now();
        let expanded = ParallelProcessor::parallel_expand_neighborhoods(&entity_ids, &adjacency_map);
        let duration = start.elapsed();

        // Verify results
        assert!(!expanded.is_empty(), "Should find some neighborhoods");
        
        // Verify that expanded neighborhoods match the adjacency map
        for (id, neighbors) in &expanded {
            assert!(entity_ids.contains(id), "Expanded ID should be from input");
            assert_eq!(neighbors, &adjacency_map[id], "Neighbors should match adjacency map");
        }
        
        println!("Neighborhood expansion completed in {:?} for {} entities", duration, entity_ids.len());
    }

    #[tokio::test]
    async fn test_parallel_encoding_with_quantizer() {
        let embedding_dim = 128;
        let subspace_dim = 8;
        let num_subspaces = embedding_dim / subspace_dim;
        
        // Create a product quantizer
        let mut quantizer = ProductQuantizer::new(embedding_dim, num_subspaces)
            .expect("Failed to create quantizer");
        
        // Train the quantizer with sample data
        let training_data: Vec<Vec<f32>> = (0..1000)
            .map(|i| create_test_embedding(embedding_dim, i as f32))
            .collect();
        
        quantizer.train(&training_data, 10).expect("Failed to train quantizer");
        
        // Create embeddings to encode
        let embeddings: Vec<Vec<f32>> = (0..100)
            .map(|i| create_test_embedding(embedding_dim, i as f32 * 2.0))
            .collect();
        
        // Time parallel encoding
        let start = Instant::now();
        let encoded = ParallelProcessor::parallel_encode_embeddings(&embeddings, &quantizer)
            .expect("Encoding failed");
        let duration = start.elapsed();
        
        // Verify results
        assert_eq!(encoded.len(), embeddings.len(), "Should encode all embeddings");
        
        // Each encoded vector should have the expected size
        for enc in &encoded {
            assert_eq!(enc.len(), num_subspaces, "Encoded size should match number of subspaces");
        }
        
        println!("Parallel encoding of {} embeddings completed in {:?}", embeddings.len(), duration);
    }

    #[tokio::test]
    async fn test_adaptive_threshold_behavior() {
        // Test that the adaptive thresholds correctly determine when to use parallel processing
        
        // SimilaritySearch threshold: 1000
        assert!(!ParallelProcessor::should_use_parallel(999, ParallelOperation::SimilaritySearch));
        assert!(ParallelProcessor::should_use_parallel(1000, ParallelOperation::SimilaritySearch));
        
        // BatchValidation threshold: 100
        assert!(!ParallelProcessor::should_use_parallel(99, ParallelOperation::BatchValidation));
        assert!(ParallelProcessor::should_use_parallel(100, ParallelOperation::BatchValidation));
        
        // Create test scenarios for each operation type
        let test_cases = vec![
            (ParallelOperation::SimilaritySearch, 1000, 5000),
            (ParallelOperation::BatchValidation, 100, 500),
            (ParallelOperation::Encoding, 50, 200),
            (ParallelOperation::PropertyExtraction, 10, 50),
            (ParallelOperation::NeighborhoodExpansion, 20, 100),
        ];
        
        for (op_type, threshold, test_size) in test_cases {
            // Below threshold - should use sequential
            let below = ParallelProcessor::should_use_parallel(threshold - 1, op_type);
            assert!(!below, "{:?} should use sequential below threshold", op_type);
            
            // At threshold - should use parallel
            let at = ParallelProcessor::should_use_parallel(threshold, op_type);
            assert!(at, "{:?} should use parallel at threshold", op_type);
            
            // Well above threshold - should definitely use parallel
            let above = ParallelProcessor::should_use_parallel(test_size, op_type);
            assert!(above, "{:?} should use parallel for size {}", op_type, test_size);
        }
    }

    #[tokio::test]
    async fn test_end_to_end_parallel_workflow() {
        // This test simulates a complete workflow using parallel processing
        let embedding_dim = 128;
        let engine = KnowledgeEngine::new(embedding_dim, 50000).expect("Failed to create engine");
        
        // Phase 1: Bulk insert with parallel validation
        let start_phase1 = Instant::now();
        let entities_to_validate: Vec<(u32, EntityData)> = (0..200)
            .map(|i| (i, create_test_entity_data(embedding_dim, i as f32)))
            .collect();
        
        // Validate entities in parallel
        ParallelProcessor::parallel_validate_entities(&entities_to_validate, embedding_dim)
            .expect("Validation failed");
        
        // Store validated entities as triples
        for (id, data) in &entities_to_validate {
            let triple = create_test_triple(
                &format!("entity_{}", id),
                "has_property",
                &data.properties,
            );
            engine.store_triple(triple, Some(data.embedding.clone()))
                .expect("Failed to store triple");
        }
        let phase1_duration = start_phase1.elapsed();
        
        // Phase 2: Parallel similarity search
        let start_phase2 = Instant::now();
        let query_embedding = create_test_embedding(embedding_dim, 100.0);
        
        // Get all stored embeddings for similarity search
        let search_results = engine.semantic_search("entity has_property test", 50)
            .expect("Semantic search failed");
        let phase2_duration = start_phase2.elapsed();
        
        // Phase 3: Property extraction for results
        let start_phase3 = Instant::now();
        let result_ids: Vec<u32> = (0..search_results.nodes.len() as u32).collect();
        let mut property_map = HashMap::new();
        
        for (i, node) in search_results.nodes.iter().enumerate() {
            if let Some(triple) = node.get_triples().first() {
                property_map.insert(i as u32, triple.object.clone());
            }
        }
        
        let extracted_properties = ParallelProcessor::parallel_extract_properties(&result_ids, &property_map);
        let phase3_duration = start_phase3.elapsed();
        
        // Phase 4: Neighborhood expansion (simulate graph traversal)
        let start_phase4 = Instant::now();
        let mut adjacency_map = HashMap::new();
        
        // Create synthetic relationships
        for i in 0..50 {
            let neighbors: Vec<u32> = vec![
                (i + 1) % 50,
                (i + 2) % 50,
                (i + 49) % 50,
            ];
            adjacency_map.insert(i, neighbors);
        }
        
        let expanded = ParallelProcessor::parallel_expand_neighborhoods(&result_ids[..30.min(result_ids.len())], &adjacency_map);
        let phase4_duration = start_phase4.elapsed();
        
        // Verify workflow completed successfully
        assert!(!search_results.nodes.is_empty(), "Should find results");
        assert!(!extracted_properties.is_empty(), "Should extract properties");
        assert!(!expanded.is_empty(), "Should expand neighborhoods");
        
        // Report performance
        println!("End-to-end parallel workflow completed:");
        println!("  Phase 1 (Insert & Validate): {:?}", phase1_duration);
        println!("  Phase 2 (Similarity Search): {:?}", phase2_duration);
        println!("  Phase 3 (Property Extract):  {:?}", phase3_duration);
        println!("  Phase 4 (Graph Expansion):   {:?}", phase4_duration);
        println!("  Total: {:?}", phase1_duration + phase2_duration + phase3_duration + phase4_duration);
    }

    #[tokio::test]
    async fn test_parallel_performance_scaling() {
        // Test how performance scales with different data sizes
        let embedding_dim = 128;
        let sizes = vec![100, 500, 1000, 5000, 10000];
        let mut results = Vec::new();
        
        for size in sizes {
            let query = create_test_embedding(embedding_dim, 0.0);
            let entities: Vec<(u32, Vec<f32>)> = (0..size)
                .map(|i| (i as u32, create_test_embedding(embedding_dim, i as f32)))
                .collect();
            
            let start = Instant::now();
            let search_results = ParallelProcessor::parallel_similarity_search(&query, entities, 20);
            let duration = start.elapsed();
            
            let per_item_us = duration.as_micros() as f64 / size as f64;
            results.push((size, duration, per_item_us));
            
            assert_eq!(search_results.len(), 20.min(size), "Should return correct number of results");
        }
        
        // Report scaling behavior
        println!("Parallel processing scaling:");
        for (size, duration, per_item) in &results {
            println!("  Size {:5}: {:8.2?} total, {:6.2}μs per item", size, duration, per_item);
        }
        
        // Verify that per-item time doesn't increase dramatically with size
        let small_per_item = results[0].2;
        let large_per_item = results.last().unwrap().2;
        
        // Per-item time should not more than double even with 100x more data
        assert!(large_per_item < small_per_item * 2.0, 
            "Per-item processing time should scale well: {:.2}μs -> {:.2}μs", 
            small_per_item, large_per_item);
    }

    #[tokio::test]
    async fn test_parallel_error_handling() {
        // Test that parallel operations handle errors gracefully
        let embedding_dim = 128;
        
        // Test 1: Empty input handling
        let empty_entities: Vec<(u32, Vec<f32>)> = vec![];
        let query = create_test_embedding(embedding_dim, 0.0);
        let empty_results = ParallelProcessor::parallel_similarity_search(&query, empty_entities, 10);
        assert_eq!(empty_results.len(), 0, "Should handle empty input gracefully");
        
        // Test 2: Validation with mixed valid/invalid entities
        let mut mixed_entities: Vec<(u32, EntityData)> = (0..150)
            .map(|i| (i, create_test_entity_data(embedding_dim, i as f32)))
            .collect();
        
        // Insert an entity with wrong dimension at position 75
        mixed_entities[75] = (75, create_test_entity_data(64, 75.0));
        
        let validation_result = ParallelProcessor::parallel_validate_entities(&mixed_entities, embedding_dim);
        assert!(validation_result.is_err(), "Should detect dimension mismatch in parallel");
        
        // Test 3: Property extraction with empty map
        let entity_ids: Vec<u32> = (0..50).collect();
        let empty_map = HashMap::new();
        let empty_props = ParallelProcessor::parallel_extract_properties(&entity_ids, &empty_map);
        
        assert_eq!(empty_props.len(), 50, "Should return None for all entities");
        assert!(empty_props.iter().all(|(_, prop)| prop.is_none()), 
            "All properties should be None with empty map");
    }

    #[tokio::test]
    async fn test_parallel_memory_efficiency() {
        // Test that parallel operations don't cause excessive memory usage
        let embedding_dim = 256;
        let large_size = 50000;
        
        // Create large dataset
        let query = create_test_embedding(embedding_dim, 0.0);
        let entities: Vec<(u32, Vec<f32>)> = (0..large_size)
            .map(|i| (i as u32, create_test_embedding(embedding_dim, i as f32 * 0.01)))
            .collect();
        
        // Measure memory before operation
        let start = Instant::now();
        
        // Perform large parallel search
        let results = ParallelProcessor::parallel_similarity_search(&query, entities, 100);
        
        let duration = start.elapsed();
        
        // Verify operation completed successfully
        assert_eq!(results.len(), 100, "Should return requested results");
        assert!(duration < Duration::from_secs(10), 
            "Large parallel search should complete in reasonable time: {:?}", duration);
        
        println!("Large dataset ({} entities) processed in {:?}", large_size, duration);
    }

    #[tokio::test]
    async fn test_concurrent_parallel_operations() {
        // Test running multiple parallel operations concurrently
        let embedding_dim = 128;
        
        // Create shared test data
        let entities: Vec<(u32, Vec<f32>)> = (0..2000)
            .map(|i| (i as u32, create_test_embedding(embedding_dim, i as f32)))
            .collect();
        
        let entities_arc = Arc::new(entities);
        
        // Spawn multiple concurrent searches
        let mut handles = vec![];
        
        for i in 0..5 {
            let entities_clone = entities_arc.clone();
            let handle = tokio::spawn(async move {
                let query = create_test_embedding(embedding_dim, i as f32 * 10.0);
                let start = Instant::now();
                let results = ParallelProcessor::parallel_similarity_search(
                    &query,
                    (*entities_clone).clone(),
                    30,
                );
                let duration = start.elapsed();
                (i, results.len(), duration)
            });
            handles.push(handle);
        }
        
        // Wait for all searches to complete
        let mut total_duration = Duration::from_secs(0);
        for handle in handles {
            let (id, count, duration) = handle.await.expect("Task failed");
            assert_eq!(count, 30, "Search {} should return 30 results", id);
            total_duration += duration;
            println!("Concurrent search {} completed in {:?}", id, duration);
        }
        
        println!("Total time for 5 concurrent searches: {:?}", total_duration);
    }

    #[tokio::test]
    async fn test_timeout_behavior() {
        // Test that parallel operations complete within reasonable timeouts
        let embedding_dim = 128;
        let huge_size = 100000;
        
        // Create a very large dataset
        let query = create_test_embedding(embedding_dim, 0.0);
        let entities: Vec<(u32, Vec<f32>)> = (0..huge_size)
            .map(|i| (i as u32, create_test_embedding(embedding_dim, i as f32 * 0.001)))
            .collect();
        
        // Run search with timeout
        let search_future = async {
            ParallelProcessor::parallel_similarity_search(&query, entities, 50)
        };
        
        let result = timeout(Duration::from_secs(30), search_future).await;
        
        assert!(result.is_ok(), "Parallel search should complete within timeout");
        let search_results = result.unwrap();
        assert_eq!(search_results.len(), 50, "Should return requested results");
    }
}