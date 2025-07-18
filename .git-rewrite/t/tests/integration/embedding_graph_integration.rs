// Embedding-Graph Integration Tests
// Tests integration between embeddings and graph structures

use std::time::{Duration, Instant};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::test_infrastructure::*;
use crate::entity::{Entity, EntityKey};
use crate::relationship::{Relationship, RelationshipType};
use crate::knowledge_graph::KnowledgeGraph;
use crate::embedding::{EmbeddingStore, SimilarityMetric, SearchResult};
use crate::embedding::quantization::{ProductQuantizer, ScalarQuantizer};
use crate::embedding::simd::{SimdProcessor, SimdCapability};
use crate::query::{GraphRagEngine, RagParameters, RagContext};

#[cfg(test)]
mod embedding_graph_integration {
    use super::*;

    #[test]
    fn test_embedding_quantization_graph_rag_integration() {
        let mut test_env = IntegrationTestEnvironment::new("embedding_rag_integration");
        
        // Create test scenario with known structure
        let scenario = test_env.data_generator.generate_academic_scenario(
            1000, // papers
            300,  // authors  
            50,   // venues
            128   // embedding dimension
        );
        
        // Build knowledge graph
        let mut kg = KnowledgeGraph::new();
        for entity in scenario.entities.values() {
            kg.add_entity(entity.clone()).unwrap();
        }
        for (source, target, rel) in scenario.relationships {
            kg.add_relationship(source, target, rel).unwrap();
        }
        
        // Set up embedding system with quantization
        let mut embedding_store = EmbeddingStore::new(128);
        let mut quantizer = ProductQuantizer::new(128, 256, 16);
        
        // Train quantizer on embeddings
        let embeddings: Vec<Vec<f32>> = scenario.embeddings.values().cloned().collect();
        
        let train_start = Instant::now();
        quantizer.train(&embeddings, 1000).unwrap();
        let train_time = train_start.elapsed();
        
        println!("Quantizer training time: {:?}", train_time);
        test_env.record_performance("quantizer_train_time", train_time);
        
        // Add quantized embeddings to store
        let quantize_start = Instant::now();
        for (entity_key, embedding) in &scenario.embeddings {
            let quantized = quantizer.quantize(&embedding);
            embedding_store.add_quantized_embedding(*entity_key, quantized).unwrap();
        }
        let quantize_time = quantize_start.elapsed();
        
        println!("Quantization time for {} embeddings: {:?}", 
                scenario.embeddings.len(), quantize_time);
        test_env.record_performance("quantize_time", quantize_time);
        
        // Test 1: RAG query with quantized embeddings
        let query_entity = scenario.central_entities[0];
        let rag_engine = GraphRagEngine::new(&kg, &embedding_store);
        
        let rag_start = Instant::now();
        let context = rag_engine.assemble_context(query_entity, &RagParameters {
            max_context_entities: 15,
            max_graph_depth: 2,
            similarity_threshold: 0.6,
            diversity_factor: 0.3,
            include_relationships: true,
            max_relationships_per_entity: 5,
            relationship_weight_threshold: 0.1,
            temporal_decay_factor: None,
            entity_type_filters: None,
            relationship_type_filters: None,
            scoring_weights: Default::default(),
        }).unwrap();
        let rag_time = rag_start.elapsed();
        
        // Verify context quality despite quantization
        assert_eq!(context.entities.len(), 15);
        assert!(context.entities.contains(&query_entity));
        
        // Compare with unquantized RAG for quality assessment
        let mut unquantized_store = EmbeddingStore::new(128);
        for (entity_key, embedding) in &scenario.embeddings {
            unquantized_store.add_embedding(*entity_key, embedding.clone()).unwrap();
        }
        
        let unquantized_rag = GraphRagEngine::new(&kg, &unquantized_store);
        let unquantized_context = unquantized_rag.assemble_context(query_entity, &RagParameters {
            max_context_entities: 15,
            max_graph_depth: 2, 
            similarity_threshold: 0.6,
            diversity_factor: 0.3,
            include_relationships: true,
            max_relationships_per_entity: 5,
            relationship_weight_threshold: 0.1,
            temporal_decay_factor: None,
            entity_type_filters: None,
            relationship_type_filters: None,
            scoring_weights: Default::default(),
        }).unwrap();
        
        // Quantized results should be similar to unquantized
        let context_overlap = calculate_entity_set_overlap(
            &context.entities.iter().cloned().collect(),
            &unquantized_context.entities.iter().cloned().collect()
        );
        assert!(context_overlap >= 0.7, 
               "Quantization degraded context quality: overlap {}", context_overlap);
        
        println!("RAG context overlap (quantized vs unquantized): {:.2}", context_overlap);
        
        // Test 2: Similarity search accuracy with quantization
        let query_embedding = &scenario.embeddings[&query_entity];
        
        let quantized_search_start = Instant::now();
        let quantized_results = embedding_store.similarity_search_quantized(
            query_embedding, 20, &quantizer
        ).unwrap();
        let quantized_search_time = quantized_search_start.elapsed();
        
        let unquantized_results = unquantized_store.similarity_search(
            query_embedding, 20, SimilarityMetric::Cosine
        ).unwrap();
        
        // Compare top-k results
        let similarity_overlap = calculate_ranked_overlap(
            &quantized_results.iter().map(|r| r.entity).collect::<Vec<_>>(),
            &unquantized_results.iter().map(|r| r.entity).collect::<Vec<_>>(),
            10
        );
        assert!(similarity_overlap >= 0.8,
               "Quantization degraded similarity search: overlap {}", similarity_overlap);
        
        println!("Similarity search overlap @10: {:.2}", similarity_overlap);
        
        // Test 3: Memory efficiency validation  
        let quantized_memory = embedding_store.memory_usage();
        let unquantized_memory = unquantized_store.memory_usage();
        let compression_ratio = unquantized_memory as f64 / quantized_memory as f64;
        
        assert!(compression_ratio >= 10.0,
               "Insufficient compression ratio: {:.2}x", compression_ratio);
        
        println!("Memory compression ratio: {:.2}x", compression_ratio);
        
        // Test 4: Performance comparison
        assert!(rag_time < Duration::from_millis(100),
               "RAG with quantization too slow: {:?}", rag_time);
        
        assert!(quantized_search_time < Duration::from_millis(50),
               "Quantized search too slow: {:?}", quantized_search_time);
        
        test_env.record_metric("context_overlap", context_overlap);
        test_env.record_metric("similarity_overlap", similarity_overlap);
        test_env.record_metric("compression_ratio", compression_ratio);
        test_env.record_performance("rag_time", rag_time);
        test_env.record_performance("quantized_search_time", quantized_search_time);
        
        // Test 5: Reconstruction error
        let mut total_error = 0.0;
        let test_embeddings: Vec<_> = scenario.embeddings.iter().take(100).collect();
        
        for (entity_key, original_embedding) in test_embeddings {
            let quantized = quantizer.quantize(original_embedding);
            let reconstructed = quantizer.reconstruct(&quantized);
            
            let error = euclidean_distance(original_embedding, &reconstructed);
            total_error += error;
        }
        
        let avg_reconstruction_error = total_error / 100.0;
        println!("Average reconstruction error: {:.4}", avg_reconstruction_error);
        
        assert!(avg_reconstruction_error < 0.1,
               "Reconstruction error too high: {}", avg_reconstruction_error);
        
        test_env.record_metric("avg_reconstruction_error", avg_reconstruction_error as f64);
    }
    
    #[test]
    fn test_simd_embedding_integration() {
        let mut test_env = IntegrationTestEnvironment::new("simd_embedding_integration");
        
        // Check SIMD availability
        let simd_capability = SimdProcessor::detect_capability();
        println!("SIMD capability: {:?}", simd_capability);
        
        if matches!(simd_capability, SimdCapability::None) {
            println!("Skipping SIMD test - no SIMD support detected");
            return;
        }
        
        // Create large-scale embedding scenario
        let entity_count = 10000;
        let embedding_dim = 256;
        let query_count = 100;
        
        let scenario = test_env.data_generator.generate_embedding_test_scenario(
            entity_count, embedding_dim, query_count
        );
        
        let mut embedding_store = EmbeddingStore::new(embedding_dim);
        
        // Add all embeddings
        let add_start = Instant::now();
        for (entity_key, embedding) in scenario.embeddings {
            embedding_store.add_embedding(entity_key, embedding).unwrap();
        }
        let add_time = add_start.elapsed();
        
        println!("Added {} embeddings in {:?}", entity_count, add_time);
        test_env.record_performance("embedding_add_time", add_time);
        
        // Enable SIMD processing
        embedding_store.enable_simd(true);
        
        // Test 1: SIMD vs scalar similarity search comparison
        let mut simd_times = Vec::new();
        let mut scalar_times = Vec::new();
        
        for query_embedding in scenario.query_embeddings.iter().take(10) {
            // SIMD implementation
            embedding_store.enable_simd(true);
            let simd_start = Instant::now();
            let simd_results = embedding_store.similarity_search(
                query_embedding, 50, SimilarityMetric::Cosine
            ).unwrap();
            let simd_time = simd_start.elapsed();
            simd_times.push(simd_time);
            
            // Scalar implementation (for validation)
            embedding_store.enable_simd(false);
            let scalar_start = Instant::now();
            let scalar_results = embedding_store.similarity_search(
                query_embedding, 50, SimilarityMetric::Cosine
            ).unwrap();
            let scalar_time = scalar_start.elapsed();
            scalar_times.push(scalar_time);
            
            // Results should be nearly identical
            assert_eq!(simd_results.len(), scalar_results.len());
            
            for (i, (simd_result, scalar_result)) in simd_results.iter()
                .zip(scalar_results.iter()).enumerate() {
                assert_eq!(simd_result.entity, scalar_result.entity,
                          "Entity mismatch at position {}", i);
                
                let distance_diff = (simd_result.distance - scalar_result.distance).abs();
                assert!(distance_diff < 1e-4,
                       "Distance mismatch at position {}: {} vs {}", 
                       i, simd_result.distance, scalar_result.distance);
            }
        }
        
        // Calculate average speedup
        let avg_simd_time = simd_times.iter().sum::<Duration>() / simd_times.len() as u32;
        let avg_scalar_time = scalar_times.iter().sum::<Duration>() / scalar_times.len() as u32;
        let speedup = avg_scalar_time.as_nanos() as f64 / avg_simd_time.as_nanos() as f64;
        
        println!("SIMD speedup: {:.2}x (SIMD: {:?}, Scalar: {:?})", 
                speedup, avg_simd_time, avg_scalar_time);
        
        assert!(speedup > 2.0, "SIMD speedup insufficient: {:.2}x", speedup);
        
        test_env.record_metric("simd_speedup", speedup);
        test_env.record_performance("avg_simd_time", avg_simd_time);
        test_env.record_performance("avg_scalar_time", avg_scalar_time);
        
        // Test 2: Batch processing with SIMD
        embedding_store.enable_simd(true);
        let batch_start = Instant::now();
        let batch_results = embedding_store.batch_similarity_search(
            &scenario.query_embeddings, 20, SimilarityMetric::Cosine
        ).unwrap();
        let batch_time = batch_start.elapsed();
        
        assert_eq!(batch_results.len(), scenario.query_embeddings.len());
        
        // Verify batch results match individual queries
        for (i, query_embedding) in scenario.query_embeddings.iter().enumerate() {
            let individual_results = embedding_store.similarity_search(
                query_embedding, 20, SimilarityMetric::Cosine
            ).unwrap();
            let batch_result = &batch_results[i];
            
            assert_eq!(individual_results.len(), batch_result.len());
            
            for (individual, batch) in individual_results.iter().zip(batch_result.iter()) {
                assert_eq!(individual.entity, batch.entity);
                assert!((individual.distance - batch.distance).abs() < 1e-6);
            }
        }
        
        let avg_batch_time_per_query = batch_time / scenario.query_embeddings.len() as u32;
        println!("Batch processing time: {:?} total, {:?} per query",
                batch_time, avg_batch_time_per_query);
        
        // Batch processing should be more efficient than individual queries
        assert!(avg_batch_time_per_query < avg_simd_time * 0.8,
               "Batch processing not efficient enough");
        
        test_env.record_performance("batch_similarity_time", batch_time);
        test_env.record_performance("avg_batch_time_per_query", avg_batch_time_per_query);
        
        // Test 3: Different similarity metrics with SIMD
        let metrics = vec![
            SimilarityMetric::Cosine,
            SimilarityMetric::Euclidean,
            SimilarityMetric::DotProduct,
        ];
        
        for metric in metrics {
            let metric_start = Instant::now();
            let _ = embedding_store.similarity_search(
                &scenario.query_embeddings[0], 20, metric
            ).unwrap();
            let metric_time = metric_start.elapsed();
            
            println!("{:?} similarity search time: {:?}", metric, metric_time);
            
            assert!(metric_time < Duration::from_millis(10),
                   "{:?} search too slow: {:?}", metric, metric_time);
        }
    }
    
    #[test]
    fn test_embedding_graph_consistency() {
        let mut test_env = IntegrationTestEnvironment::new("embedding_graph_consistency");
        
        // Create scenario where graph structure and embeddings are correlated
        let scenario = test_env.data_generator.generate_correlated_graph_embeddings(
            500, 1000, 128, 0.8 // 80% correlation between graph distance and embedding similarity
        );
        
        // Build systems
        let mut kg = KnowledgeGraph::new();
        for entity in scenario.entities.values() {
            kg.add_entity(entity.clone()).unwrap();
        }
        for (source, target, rel) in scenario.relationships {
            kg.add_relationship(source, target, rel).unwrap();
        }
        
        let mut embedding_store = EmbeddingStore::new(128);
        for (entity_key, embedding) in scenario.embeddings {
            embedding_store.add_embedding(entity_key, embedding).unwrap();
        }
        
        // Test 1: Correlation between graph distance and embedding similarity
        let test_pairs = scenario.test_entity_pairs.iter().take(100);
        let mut correlation_data = Vec::new();
        
        for &(entity1, entity2) in test_pairs {
            // Graph distance
            let graph_distance = kg.shortest_path_length(entity1, entity2)
                .unwrap_or(u32::MAX) as f32;
            
            // Embedding similarity (convert to distance)
            let embedding1 = embedding_store.get_embedding(entity1).unwrap();
            let embedding2 = embedding_store.get_embedding(entity2).unwrap();
            let embedding_distance = euclidean_distance(&embedding1, &embedding2);
            
            if graph_distance < u32::MAX as f32 {
                correlation_data.push((graph_distance, embedding_distance));
            }
        }
        
        let correlation = calculate_correlation(&correlation_data);
        println!("Graph-embedding distance correlation: {:.3}", correlation);
        
        assert!(correlation.abs() >= 0.5,  // Should be moderately correlated
               "Insufficient correlation between graph and embedding distances: {}", correlation);
        
        // Test 2: Consistency in neighborhood queries
        let test_entities: Vec<_> = scenario.entities.keys().take(20).cloned().collect();
        let mut overlap_scores = Vec::new();
        
        for &entity in &test_entities {
            // Graph neighbors (1-hop)
            let graph_neighbors: HashSet<EntityKey> = kg.get_neighbors(entity)
                .into_iter()
                .map(|rel| rel.target())
                .collect();
            
            if graph_neighbors.len() < 5 {
                continue; // Skip entities with too few neighbors
            }
            
            // Embedding neighbors (top-k similar)
            let entity_embedding = embedding_store.get_embedding(entity).unwrap();
            let embedding_neighbors: HashSet<EntityKey> = embedding_store
                .similarity_search(&entity_embedding, graph_neighbors.len() * 2, 
                                 SimilarityMetric::Cosine)
                .unwrap()
                .into_iter()
                .skip(1) // Skip self
                .map(|result| result.entity)
                .collect();
            
            // Calculate overlap
            let overlap = calculate_set_overlap_ratio(&graph_neighbors, &embedding_neighbors);
            overlap_scores.push(overlap);
        }
        
        let avg_overlap = overlap_scores.iter().sum::<f64>() / overlap_scores.len() as f64;
        println!("Average graph/embedding neighborhood overlap: {:.3}", avg_overlap);
        
        assert!(avg_overlap >= 0.2,
               "Low average overlap between graph and embedding neighbors: {}", avg_overlap);
        
        test_env.record_metric("graph_embedding_correlation", correlation);
        test_env.record_metric("avg_neighborhood_overlap", avg_overlap);
        
        // Test 3: Combined RAG query consistency
        let rag_engine = GraphRagEngine::new(&kg, &embedding_store);
        
        // Test that RAG respects both graph structure and embeddings
        for &test_entity in test_entities.iter().take(5) {
            let context = rag_engine.assemble_context(test_entity, &RagParameters {
                max_context_entities: 10,
                max_graph_depth: 2,
                similarity_threshold: 0.5,
                diversity_factor: 0.3,
                include_relationships: true,
                max_relationships_per_entity: 3,
                relationship_weight_threshold: 0.1,
                temporal_decay_factor: None,
                entity_type_filters: None,
                relationship_type_filters: None,
                scoring_weights: Default::default(),
            }).unwrap();
            
            // Check that context includes both graph neighbors and similar embeddings
            let graph_neighbors: HashSet<EntityKey> = kg.get_neighbors(test_entity)
                .into_iter()
                .map(|rel| rel.target())
                .collect();
            
            let context_set: HashSet<EntityKey> = context.entities.iter().cloned().collect();
            let graph_overlap = graph_neighbors.intersection(&context_set).count();
            
            assert!(graph_overlap > 0,
                   "RAG context doesn't include any graph neighbors");
            
            // Check embedding similarity of context entities
            let test_embedding = embedding_store.get_embedding(test_entity).unwrap();
            let mut avg_similarity = 0.0;
            
            for &context_entity in &context.entities {
                if context_entity != test_entity {
                    let context_embedding = embedding_store.get_embedding(context_entity).unwrap();
                    let distance = euclidean_distance(&test_embedding, &context_embedding);
                    avg_similarity += 1.0 / (1.0 + distance);
                }
            }
            
            avg_similarity /= (context.entities.len() - 1) as f32;
            
            assert!(avg_similarity > 0.3,
                   "RAG context entities not similar enough in embedding space");
        }
    }
    
    #[test]
    fn test_embedding_incremental_updates() {
        let mut test_env = IntegrationTestEnvironment::new("embedding_incremental");
        
        // Start with a small graph
        let initial_size = 1000;
        let increment_size = 500;
        let embedding_dim = 128;
        
        let mut kg = KnowledgeGraph::new();
        let mut embedding_store = EmbeddingStore::new(embedding_dim);
        
        // Initial data
        let initial_scenario = test_env.data_generator.generate_academic_scenario(
            initial_size, initial_size / 3, 50, embedding_dim
        );
        
        // Build initial state
        let initial_build_start = Instant::now();
        for entity in initial_scenario.entities.values() {
            kg.add_entity(entity.clone()).unwrap();
        }
        for (source, target, rel) in initial_scenario.relationships {
            kg.add_relationship(source, target, rel).unwrap();
        }
        for (entity_key, embedding) in initial_scenario.embeddings {
            embedding_store.add_embedding(entity_key, embedding).unwrap();
        }
        let initial_build_time = initial_build_start.elapsed();
        
        println!("Initial build time for {} entities: {:?}", 
                initial_size, initial_build_time);
        
        // Test baseline performance
        let test_query = &initial_scenario.central_entities[0];
        let test_embedding = embedding_store.get_embedding(*test_query).unwrap();
        
        let baseline_start = Instant::now();
        let baseline_results = embedding_store.similarity_search(
            &test_embedding, 20, SimilarityMetric::Cosine
        ).unwrap();
        let baseline_time = baseline_start.elapsed();
        
        // Incremental updates
        for batch in 0..3 {
            let increment_scenario = test_env.data_generator.generate_academic_scenario(
                increment_size, increment_size / 3, 10, embedding_dim
            );
            
            let update_start = Instant::now();
            
            // Add new entities and embeddings
            for entity in increment_scenario.entities.values() {
                kg.add_entity(entity.clone()).unwrap();
            }
            for (source, target, rel) in increment_scenario.relationships {
                kg.add_relationship(source, target, rel).unwrap();
            }
            for (entity_key, embedding) in increment_scenario.embeddings {
                embedding_store.add_embedding(entity_key, embedding).unwrap();
            }
            
            let update_time = update_start.elapsed();
            
            println!("Batch {} update time for {} entities: {:?}",
                    batch, increment_size, update_time);
            
            // Test performance after update
            let post_update_start = Instant::now();
            let post_update_results = embedding_store.similarity_search(
                &test_embedding, 20, SimilarityMetric::Cosine
            ).unwrap();
            let post_update_time = post_update_start.elapsed();
            
            // Performance shouldn't degrade significantly
            assert!(post_update_time < baseline_time * 2.0,
                   "Search performance degraded too much after batch {}: {:?} vs {:?}",
                   batch, post_update_time, baseline_time);
            
            // Results should still be meaningful
            let result_overlap = calculate_ranked_overlap(
                &baseline_results.iter().map(|r| r.entity).collect::<Vec<_>>(),
                &post_update_results.iter().map(|r| r.entity).collect::<Vec<_>>(),
                10
            );
            
            // Some overlap expected but not complete (new better matches might appear)
            assert!(result_overlap >= 0.3 && result_overlap <= 0.9,
                   "Unexpected result overlap after batch {}: {}", batch, result_overlap);
            
            test_env.record_performance(&format!("update_time_batch_{}", batch), update_time);
            test_env.record_performance(&format!("search_time_batch_{}", batch), post_update_time);
            test_env.record_metric(&format!("result_overlap_batch_{}", batch), result_overlap);
        }
        
        // Final statistics
        let final_entity_count = kg.entity_count();
        let final_embedding_count = embedding_store.entity_count();
        
        assert_eq!(final_entity_count, initial_size + increment_size * 3);
        assert_eq!(final_embedding_count, final_entity_count);
        
        println!("Final graph size: {} entities, {} embeddings",
                final_entity_count, final_embedding_count);
    }
    
    #[test]
    fn test_multi_modal_embedding_integration() {
        let mut test_env = IntegrationTestEnvironment::new("multi_modal_embedding");
        
        // Create entities with different embedding dimensions (simulating different modalities)
        let text_dim = 768;      // BERT-like
        let image_dim = 2048;    // ResNet-like
        let combined_dim = 256;  // Projected dimension
        
        let entity_count = 500;
        
        // Generate test data
        let mut kg = KnowledgeGraph::new();
        let mut text_embeddings = HashMap::new();
        let mut image_embeddings = HashMap::new();
        let mut combined_embeddings = HashMap::new();
        
        for i in 0..entity_count {
            let key = EntityKey::from_hash(format!("multi_modal_{}", i));
            let entity = Entity::new(key, format!("Multi-modal Entity {}", i))
                .with_attribute("has_text", "true")
                .with_attribute("has_image", if i % 3 == 0 { "false" } else { "true" });
            
            kg.add_entity(entity).unwrap();
            
            // Text embedding (all entities have text)
            let text_emb = test_env.data_generator.generate_embedding(text_dim);
            text_embeddings.insert(key, text_emb);
            
            // Image embedding (only some entities have images)
            if i % 3 != 0 {
                let image_emb = test_env.data_generator.generate_embedding(image_dim);
                image_embeddings.insert(key, image_emb);
            }
            
            // Combined embedding (projected from available modalities)
            let combined_emb = test_env.data_generator.generate_embedding(combined_dim);
            combined_embeddings.insert(key, combined_emb);
        }
        
        // Create separate embedding stores for each modality
        let mut text_store = EmbeddingStore::new(text_dim);
        let mut image_store = EmbeddingStore::new(image_dim);
        let mut combined_store = EmbeddingStore::new(combined_dim);
        
        for (key, emb) in text_embeddings {
            text_store.add_embedding(key, emb).unwrap();
        }
        for (key, emb) in image_embeddings {
            image_store.add_embedding(key, emb).unwrap();
        }
        for (key, emb) in combined_embeddings {
            combined_store.add_embedding(key, emb).unwrap();
        }
        
        // Test multi-modal search
        let test_entity = EntityKey::from_hash("multi_modal_10");
        
        // Search in each modality
        let text_query = text_store.get_embedding(test_entity).unwrap();
        let text_results = text_store.similarity_search(
            &text_query, 20, SimilarityMetric::Cosine
        ).unwrap();
        
        let combined_query = combined_store.get_embedding(test_entity).unwrap();
        let combined_results = combined_store.similarity_search(
            &combined_query, 20, SimilarityMetric::Cosine
        ).unwrap();
        
        // Results should be different but have some overlap
        let modality_overlap = calculate_ranked_overlap(
            &text_results.iter().map(|r| r.entity).collect::<Vec<_>>(),
            &combined_results.iter().map(|r| r.entity).collect::<Vec<_>>(),
            10
        );
        
        println!("Text vs Combined modality result overlap: {:.2}", modality_overlap);
        
        assert!(modality_overlap > 0.2 && modality_overlap < 0.8,
               "Unexpected modality overlap: {}", modality_overlap);
        
        // Test fusion of multi-modal results
        let mut fusion_scores: HashMap<EntityKey, f32> = HashMap::new();
        
        // Weight text results
        for (i, result) in text_results.iter().enumerate() {
            let score = 1.0 / (i as f32 + 1.0);
            *fusion_scores.entry(result.entity).or_insert(0.0) += score * 0.6;
        }
        
        // Weight combined results
        for (i, result) in combined_results.iter().enumerate() {
            let score = 1.0 / (i as f32 + 1.0);
            *fusion_scores.entry(result.entity).or_insert(0.0) += score * 0.4;
        }
        
        // Sort by fused score
        let mut fused_results: Vec<_> = fusion_scores.into_iter().collect();
        fused_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        fused_results.truncate(20);
        
        println!("Fused top result: {:?} with score {:.3}", 
                fused_results[0].0, fused_results[0].1);
        
        test_env.record_metric("modality_overlap", modality_overlap);
        test_env.record_metric("text_embedding_count", text_store.entity_count() as f64);
        test_env.record_metric("image_embedding_count", image_store.entity_count() as f64);
    }
}