// Performance Integration Tests
// End-to-end performance validation across components

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;

use crate::test_infrastructure::*;
use crate::entity::{Entity, EntityKey};
use crate::relationship::{Relationship, RelationshipType};
use crate::knowledge_graph::KnowledgeGraph;
use crate::embedding::{EmbeddingStore, SimilarityMetric};
use crate::embedding::quantization::ProductQuantizer;
use crate::query::{GraphRagEngine, RagParameters};
use crate::storage::MemoryUsage;

#[cfg(test)]
mod performance_integration {
    use super::*;

    #[test]
    fn test_query_latency_integration() {
        let mut test_env = IntegrationTestEnvironment::new("query_latency");
        
        // Test different graph sizes
        let sizes = vec![1000, 5000, 10000, 25000];
        let mut scaling_data = Vec::new();
        
        for &size in &sizes {
            println!("\n=== Testing query latency for {} entities ===", size);
            
            // Generate test graph
            let scenario = test_env.data_generator.generate_performance_graph(
                size, size * 2, 128
            );
            
            // Build complete system
            let build_start = Instant::now();
            let mut kg = KnowledgeGraph::new();
            for entity in scenario.entities {
                kg.add_entity(entity).unwrap();
            }
            for (source, target, rel) in scenario.relationships {
                kg.add_relationship(source, target, rel).unwrap();
            }
            
            let mut embedding_store = EmbeddingStore::new(128);
            for (entity_key, embedding) in scenario.embeddings {
                embedding_store.add_embedding(entity_key, embedding).unwrap();
            }
            let build_time = build_start.elapsed();
            
            println!("Build time: {:?}", build_time);
            test_env.record_performance_for_size(size, "build_time", build_time);
            
            let rag_engine = GraphRagEngine::new(&kg, &embedding_store);
            
            // Test 1: Single-hop queries
            let test_entities: Vec<EntityKey> = kg.get_all_entities()
                .take(100)
                .map(|e| e.key())
                .collect();
            
            let mut single_hop_times = Vec::new();
            for &entity in &test_entities {
                let start = Instant::now();
                let _neighbors = kg.get_neighbors(entity);
                let elapsed = start.elapsed();
                single_hop_times.push(elapsed);
            }
            
            let avg_single_hop = single_hop_times.iter().sum::<Duration>() / single_hop_times.len() as u32;
            
            // Target: < 1ms for single-hop queries
            assert!(avg_single_hop < Duration::from_millis(1),
                   "Single-hop queries too slow for size {}: {:?}", size, avg_single_hop);
            
            println!("Average single-hop query: {:?}", avg_single_hop);
            
            // Test 2: Multi-hop traversal
            let mut multi_hop_times = Vec::new();
            for &entity in test_entities.iter().take(20) {
                let start = Instant::now();
                let _traversal = kg.breadth_first_search(entity, 3, Some(100));
                let elapsed = start.elapsed();
                multi_hop_times.push(elapsed);
            }
            
            let avg_multi_hop = multi_hop_times.iter().sum::<Duration>() / multi_hop_times.len() as u32;
            
            // Should scale reasonably with depth
            assert!(avg_multi_hop < Duration::from_millis(10),
                   "Multi-hop queries too slow for size {}: {:?}", size, avg_multi_hop);
            
            println!("Average multi-hop query (depth 3): {:?}", avg_multi_hop);
            
            // Test 3: Similarity search
            let query_embedding = vec![0.1; 128];
            let mut similarity_times = Vec::new();
            
            for _ in 0..50 {
                let start = Instant::now();
                let _results = embedding_store.similarity_search(
                    &query_embedding, 20, SimilarityMetric::Cosine
                ).unwrap();
                let elapsed = start.elapsed();
                similarity_times.push(elapsed);
            }
            
            let avg_similarity = similarity_times.iter().sum::<Duration>() / similarity_times.len() as u32;
            
            // Target: < 5ms for similarity search
            assert!(avg_similarity < Duration::from_millis(5),
                   "Similarity search too slow for size {}: {:?}", size, avg_similarity);
            
            println!("Average similarity search: {:?}", avg_similarity);
            
            // Test 4: RAG context assembly
            let mut rag_times = Vec::new();
            for &entity in test_entities.iter().take(10) {
                let start = Instant::now();
                let _context = rag_engine.assemble_context(entity, &RagParameters {
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
                let elapsed = start.elapsed();
                rag_times.push(elapsed);
            }
            
            let avg_rag = rag_times.iter().sum::<Duration>() / rag_times.len() as u32;
            
            // Should be reasonable for complex operation
            assert!(avg_rag < Duration::from_millis(50),
                   "RAG assembly too slow for size {}: {:?}", size, avg_rag);
            
            println!("Average RAG context assembly: {:?}", avg_rag);
            
            // Record metrics
            test_env.record_performance_for_size(size, "single_hop_avg", avg_single_hop);
            test_env.record_performance_for_size(size, "multi_hop_avg", avg_multi_hop);
            test_env.record_performance_for_size(size, "similarity_avg", avg_similarity);
            test_env.record_performance_for_size(size, "rag_avg", avg_rag);
            
            // Collect scaling data
            scaling_data.push((size as usize, avg_single_hop));
        }
        
        // Analyze scaling behavior
        println!("\n=== Scaling Analysis ===");
        test_env.analyze_scaling_behavior(&["single_hop_avg", "similarity_avg"]);
        
        // Validate scaling is sub-linear
        let result = PerformanceValidator::validate_scaling(&scaling_data, 1.5);
        assert!(result.is_ok(), "Scaling validation failed: {:?}", result.err());
    }
    
    #[test]
    fn test_memory_efficiency_integration() {
        let mut test_env = IntegrationTestEnvironment::new("memory_efficiency");
        
        let sizes = vec![1000, 5000, 10000];
        
        println!("\n=== Memory Efficiency Testing ===");
        
        for &size in &sizes {
            println!("\nTesting memory usage for {} entities", size);
            
            // Create test scenario
            let scenario = test_env.data_generator.generate_memory_test_scenario(size);
            
            // Measure baseline memory
            let baseline_memory = get_current_memory_usage();
            
            // Build knowledge graph
            let graph_build_start = Instant::now();
            let mut kg = KnowledgeGraph::new();
            for entity in scenario.entities {
                kg.add_entity(entity).unwrap();
            }
            let graph_build_time = graph_build_start.elapsed();
            
            let graph_memory = get_current_memory_usage();
            let graph_overhead = graph_memory.saturating_sub(baseline_memory);
            
            // Add relationships
            let rel_start = Instant::now();
            for (source, target, rel) in scenario.relationships {
                kg.add_relationship(source, target, rel).unwrap();
            }
            let rel_time = rel_start.elapsed();
            
            let relationships_memory = get_current_memory_usage();
            let relationships_overhead = relationships_memory.saturating_sub(graph_memory);
            
            // Add embeddings
            let embed_start = Instant::now();
            let mut embedding_store = EmbeddingStore::new(128);
            for (entity_key, embedding) in scenario.embeddings {
                embedding_store.add_embedding(entity_key, embedding).unwrap();
            }
            let embed_time = embed_start.elapsed();
            
            let embeddings_memory = get_current_memory_usage();
            let embeddings_overhead = embeddings_memory.saturating_sub(relationships_memory);
            
            // Calculate memory per entity
            let total_memory = embeddings_memory.saturating_sub(baseline_memory);
            let memory_per_entity = total_memory / size;
            
            println!("Build times - Graph: {:?}, Relations: {:?}, Embeddings: {:?}",
                    graph_build_time, rel_time, embed_time);
            
            println!("Memory usage:");
            println!("  Graph overhead: {} KB ({} bytes/entity)",
                    graph_overhead / 1024, graph_overhead / size);
            println!("  Relationships overhead: {} KB ({} bytes/rel)",
                    relationships_overhead / 1024, relationships_overhead / scenario.relationship_count);
            println!("  Embeddings overhead: {} KB ({} bytes/entity)",
                    embeddings_overhead / 1024, embeddings_overhead / size);
            println!("  Total: {} KB ({} bytes/entity)",
                    total_memory / 1024, memory_per_entity);
            
            // Target: < 70 bytes per entity total
            assert!(memory_per_entity < 70,
                   "Memory per entity too high for size {}: {} bytes", size, memory_per_entity);
            
            // Verify memory efficiency of components
            let graph_per_entity = graph_overhead / size;
            let relationships_per_rel = relationships_overhead / scenario.relationship_count;
            let embeddings_per_entity = embeddings_overhead / size;
            
            assert!(graph_per_entity < 30, "Graph overhead too high: {} bytes/entity", graph_per_entity);
            assert!(relationships_per_rel < 20, "Relationship overhead too high: {} bytes/rel", relationships_per_rel);
            
            test_env.record_memory_usage(size, "total_per_entity", memory_per_entity);
            test_env.record_memory_usage(size, "graph_per_entity", graph_per_entity);
            test_env.record_memory_usage(size, "embeddings_per_entity", embeddings_per_entity);
            
            // Test memory reporting accuracy
            let reported_kg_memory = kg.estimate_memory_usage();
            let reported_embed_memory = embedding_store.memory_usage();
            
            println!("Reported memory - KG: {} KB, Embeddings: {} KB",
                    reported_kg_memory / 1024, reported_embed_memory / 1024);
            
            // Reported should be within 20% of actual
            let reported_total = reported_kg_memory + reported_embed_memory;
            let accuracy = (reported_total as f64 / total_memory as f64 - 1.0).abs();
            assert!(accuracy < 0.2,
                   "Memory reporting inaccurate: {:.1}% difference", accuracy * 100.0);
        }
    }
    
    #[test]
    fn test_compression_integration() {
        let mut test_env = IntegrationTestEnvironment::new("compression_integration");
        
        println!("\n=== Compression Integration Testing ===");
        
        // Test vector quantization compression
        let embedding_sizes = vec![1000, 5000, 10000];
        let dimensions = vec![64, 128, 256];
        
        for &size in &embedding_sizes {
            for &dim in &dimensions {
                println!("\nTesting compression for {} embeddings of dimension {}", size, dim);
                
                let embeddings = test_env.data_generator.generate_random_embeddings(size, dim);
                
                // Measure uncompressed size
                let uncompressed_size = size * dim * std::mem::size_of::<f32>();
                
                // Train quantizer and compress
                let train_start = Instant::now();
                let mut quantizer = ProductQuantizer::new(dim, 256, dim / 8);
                let embedding_vectors: Vec<Vec<f32>> = embeddings.values().cloned().collect();
                quantizer.train(&embedding_vectors, 1000).unwrap();
                let train_time = train_start.elapsed();
                
                // Measure compressed size
                let compress_start = Instant::now();
                let mut compressed_size = quantizer.memory_usage();
                let mut quantized_data = Vec::new();
                
                for embedding in &embedding_vectors {
                    let quantized = quantizer.quantize(embedding);
                    compressed_size += quantized.len() * std::mem::size_of::<u8>();
                    quantized_data.push(quantized);
                }
                let compress_time = compress_start.elapsed();
                
                let compression_ratio = uncompressed_size as f64 / compressed_size as f64;
                
                println!("Compression results:");
                println!("  Training time: {:?}", train_time);
                println!("  Compression time: {:?}", compress_time);
                println!("  Uncompressed: {} KB", uncompressed_size / 1024);
                println!("  Compressed: {} KB", compressed_size / 1024);
                println!("  Compression ratio: {:.1}x", compression_ratio);
                
                // Target: > 10x compression
                assert!(compression_ratio >= 10.0,
                       "Insufficient compression for size {} dim {}: {:.1}x", 
                       size, dim, compression_ratio);
                
                // Test reconstruction accuracy
                let mut total_error = 0.0;
                let test_count = 100.min(embedding_vectors.len());
                
                for i in 0..test_count {
                    let reconstructed = quantizer.reconstruct(&quantized_data[i]);
                    let error = euclidean_distance(&embedding_vectors[i], &reconstructed);
                    total_error += error;
                }
                
                let avg_error = total_error / test_count as f32;
                
                println!("  Average reconstruction error: {:.4}", avg_error);
                
                // Reconstruction should be reasonably accurate
                assert!(avg_error < 2.0, "Reconstruction error too high: {}", avg_error);
                
                // Test query performance with compressed embeddings
                let mut compressed_store = EmbeddingStore::new(dim);
                compressed_store.set_quantizer(Some(Box::new(quantizer)));
                
                for (i, (entity_key, _)) in embeddings.iter().enumerate() {
                    compressed_store.add_quantized_embedding(*entity_key, quantized_data[i].clone()).unwrap();
                }
                
                let query_embedding = test_env.data_generator.generate_embedding(dim);
                
                let compressed_query_start = Instant::now();
                let _results = compressed_store.similarity_search(
                    &query_embedding, 20, SimilarityMetric::Cosine
                ).unwrap();
                let compressed_query_time = compressed_query_start.elapsed();
                
                println!("  Compressed query time: {:?}", compressed_query_time);
                
                assert!(compressed_query_time < Duration::from_millis(10),
                       "Compressed query too slow: {:?}", compressed_query_time);
                
                test_env.record_compression_ratio(size, dim, compression_ratio);
                test_env.record_reconstruction_error(size, dim, avg_error as f64);
            }
        }
    }
    
    #[test]
    fn test_concurrent_access_integration() {
        let mut test_env = IntegrationTestEnvironment::new("concurrent_access");
        
        println!("\n=== Concurrent Access Testing ===");
        
        // Create shared knowledge graph
        let scenario = test_env.data_generator.generate_concurrent_test_scenario(5000, 10000);
        
        let kg = Arc::new(RwLock::new(KnowledgeGraph::new()));
        
        // Populate graph
        {
            let mut kg_write = kg.write().unwrap();
            for entity in scenario.entities {
                kg_write.add_entity(entity).unwrap();
            }
            for (source, target, rel) in scenario.relationships {
                kg_write.add_relationship(source, target, rel).unwrap();
            }
        }
        
        let embedding_store = Arc::new(RwLock::new(EmbeddingStore::new(128)));
        
        // Populate embeddings
        {
            let mut store_write = embedding_store.write().unwrap();
            for (entity_key, embedding) in scenario.embeddings {
                store_write.add_embedding(entity_key, embedding).unwrap();
            }
        }
        
        // Test concurrent readers
        let reader_count = 10;
        let queries_per_reader = 100;
        
        println!("Testing {} concurrent readers, {} queries each", reader_count, queries_per_reader);
        
        let concurrent_start = Instant::now();
        let mut reader_handles = Vec::new();
        
        for reader_id in 0..reader_count {
            let kg_clone = Arc::clone(&kg);
            let store_clone = Arc::clone(&embedding_store);
            let test_entities = scenario.test_entities.clone();
            
            let handle = thread::spawn(move || {
                let mut query_times = Vec::new();
                let mut operations = Vec::new();
                
                for i in 0..queries_per_reader {
                    let entity = test_entities[i % test_entities.len()];
                    
                    let start = Instant::now();
                    
                    // Read from graph
                    {
                        let kg_read = kg_clone.read().unwrap();
                        let _neighbors = kg_read.get_neighbors(entity);
                        operations.push("graph_read");
                    }
                    
                    // Read from embeddings
                    {
                        let store_read = store_clone.read().unwrap();
                        if let Ok(embedding) = store_read.get_embedding(entity) {
                            let _results = store_read.similarity_search(
                                &embedding, 10, SimilarityMetric::Cosine
                            );
                            operations.push("embedding_search");
                        }
                    }
                    
                    let elapsed = start.elapsed();
                    query_times.push(elapsed);
                }
                
                (reader_id, query_times, operations.len())
            });
            
            reader_handles.push(handle);
        }
        
        // Wait for all readers to complete
        let mut all_times = Vec::new();
        let mut total_operations = 0;
        
        for handle in reader_handles {
            let (reader_id, times, op_count) = handle.join().unwrap();
            
            let avg_time = times.iter().sum::<Duration>() / times.len() as u32;
            let max_time = times.iter().max().unwrap();
            
            println!("Reader {}: avg {:?}, max {:?}, operations: {}",
                    reader_id, avg_time, max_time, op_count);
            
            // Each reader should maintain good performance
            assert!(avg_time < Duration::from_millis(2),
                   "Reader {} too slow: {:?}", reader_id, avg_time);
            
            all_times.extend(times);
            total_operations += op_count;
        }
        
        let total_concurrent_time = concurrent_start.elapsed();
        
        // Overall performance should be reasonable
        let overall_avg = all_times.iter().sum::<Duration>() / all_times.len() as u32;
        let throughput = total_operations as f64 / total_concurrent_time.as_secs_f64();
        
        println!("\nConcurrent access results:");
        println!("  Total time: {:?}", total_concurrent_time);
        println!("  Overall average query: {:?}", overall_avg);
        println!("  Total operations: {}", total_operations);
        println!("  Throughput: {:.2} ops/sec", throughput);
        
        assert!(overall_avg < Duration::from_millis(2),
               "Overall concurrent performance too slow: {:?}", overall_avg);
        
        assert!(throughput > 1000.0,
               "Concurrent throughput too low: {:.2} ops/sec", throughput);
        
        test_env.record_performance("concurrent_avg_time", overall_avg);
        test_env.record_metric("concurrent_reader_count", reader_count as f64);
        test_env.record_metric("concurrent_throughput", throughput);
        
        // Test mixed readers/writers
        println!("\nTesting mixed readers and writers");
        
        let writer_count = 2;
        let writes_per_writer = 50;
        
        let mixed_start = Instant::now();
        let mut mixed_handles = Vec::new();
        
        // Spawn readers
        for reader_id in 0..reader_count / 2 {
            let kg_clone = Arc::clone(&kg);
            let test_entities = scenario.test_entities.clone();
            
            let handle = thread::spawn(move || {
                let mut read_count = 0;
                let start_time = Instant::now();
                
                while start_time.elapsed() < Duration::from_secs(2) {
                    let entity = test_entities[read_count % test_entities.len()];
                    
                    let kg_read = kg_clone.read().unwrap();
                    let _neighbors = kg_read.get_neighbors(entity);
                    drop(kg_read);
                    
                    read_count += 1;
                    thread::sleep(Duration::from_millis(10));
                }
                
                read_count
            });
            
            mixed_handles.push(("reader", handle));
        }
        
        // Spawn writers
        for writer_id in 0..writer_count {
            let kg_clone = Arc::clone(&kg);
            
            let handle = thread::spawn(move || {
                let mut write_times = Vec::new();
                
                for i in 0..writes_per_writer {
                    let entity_key = EntityKey::from_hash(format!("writer_{}_{}", writer_id, i));
                    let entity = Entity::new(entity_key, format!("Written Entity {} {}", writer_id, i));
                    
                    let write_start = Instant::now();
                    let mut kg_write = kg_clone.write().unwrap();
                    kg_write.add_entity(entity).unwrap();
                    drop(kg_write);
                    let write_time = write_start.elapsed();
                    
                    write_times.push(write_time);
                    thread::sleep(Duration::from_millis(20));
                }
                
                write_times
            });
            
            mixed_handles.push(("writer", handle));
        }
        
        // Wait for all threads
        let mut total_reads = 0;
        let mut total_writes = 0;
        
        for (thread_type, handle) in mixed_handles {
            match thread_type {
                "reader" => {
                    let read_count = handle.join().unwrap();
                    total_reads += read_count;
                }
                "writer" => {
                    let write_times = handle.join().unwrap();
                    total_writes += write_times.len();
                    
                    let avg_write = write_times.iter().sum::<Duration>() / write_times.len() as u32;
                    println!("Average write time: {:?}", avg_write);
                    
                    assert!(avg_write < Duration::from_millis(10),
                           "Write time too high under contention: {:?}", avg_write);
                }
            }
        }
        
        let mixed_time = mixed_start.elapsed();
        
        println!("\nMixed access results:");
        println!("  Total reads: {}", total_reads);
        println!("  Total writes: {}", total_writes);
        println!("  Duration: {:?}", mixed_time);
        println!("  Read throughput: {:.2} reads/sec", total_reads as f64 / mixed_time.as_secs_f64());
        println!("  Write throughput: {:.2} writes/sec", total_writes as f64 / mixed_time.as_secs_f64());
    }
    
    #[test]
    fn test_stress_integration() {
        let mut test_env = IntegrationTestEnvironment::new("stress_test");
        
        println!("\n=== Stress Testing ===");
        
        // Create large graph
        let entity_count = 50000;
        let relationship_count = 100000;
        let embedding_dim = 128;
        
        println!("Building stress test graph: {} entities, {} relationships",
                entity_count, relationship_count);
        
        let build_start = Instant::now();
        
        // Use batch building for efficiency
        let mut kg = KnowledgeGraph::new();
        kg.enable_batch_mode(true);
        
        // Add entities in batches
        let batch_size = 1000;
        for batch in 0..(entity_count / batch_size) {
            let mut entities = Vec::new();
            for i in 0..batch_size {
                let idx = batch * batch_size + i;
                let key = EntityKey::from_hash(format!("stress_entity_{}", idx));
                let entity = Entity::new(key, format!("Stress Entity {}", idx))
                    .with_attribute("batch", batch.to_string());
                entities.push(entity);
            }
            
            kg.add_entities_batch(entities).unwrap();
            
            if batch % 10 == 0 {
                println!("  Added {} entities", (batch + 1) * batch_size);
            }
        }
        
        // Add relationships
        let mut rng = rand::thread_rng();
        for i in 0..relationship_count {
            let source_idx = rng.gen_range(0..entity_count);
            let target_idx = rng.gen_range(0..entity_count);
            
            if source_idx != target_idx {
                let source = EntityKey::from_hash(format!("stress_entity_{}", source_idx));
                let target = EntityKey::from_hash(format!("stress_entity_{}", target_idx));
                let rel = Relationship::new(
                    format!("stress_rel_{}", i),
                    RelationshipType::Directed,
                    rng.gen_range(0.1..1.0)
                );
                
                kg.add_relationship(source, target, rel).ok();
            }
            
            if i % 10000 == 0 && i > 0 {
                println!("  Added {} relationships", i);
            }
        }
        
        kg.enable_batch_mode(false);
        let build_time = build_start.elapsed();
        
        println!("Build completed in {:?}", build_time);
        println!("Actual counts - Entities: {}, Relationships: {}",
                kg.entity_count(), kg.relationship_count());
        
        // Stress test queries
        println!("\nRunning stress queries...");
        
        let query_count = 1000;
        let mut query_times = Vec::new();
        
        for i in 0..query_count {
            let entity_idx = rng.gen_range(0..entity_count);
            let entity_key = EntityKey::from_hash(format!("stress_entity_{}", entity_idx));
            
            let query_start = Instant::now();
            
            // Complex query combining multiple operations
            let neighbors = kg.get_neighbors(entity_key);
            let _traversal = kg.breadth_first_search(entity_key, 2, Some(50));
            
            for neighbor in neighbors.iter().take(5) {
                let _second_neighbors = kg.get_neighbors(neighbor.target());
            }
            
            let query_time = query_start.elapsed();
            query_times.push(query_time);
            
            if i % 100 == 0 {
                let avg_so_far = query_times.iter().sum::<Duration>() / query_times.len() as u32;
                println!("  Completed {} queries, avg: {:?}", i, avg_so_far);
            }
        }
        
        // Analyze results
        query_times.sort();
        let avg_query = query_times.iter().sum::<Duration>() / query_times.len() as u32;
        let median_query = query_times[query_times.len() / 2];
        let p95_query = query_times[(query_times.len() as f32 * 0.95) as usize];
        let p99_query = query_times[(query_times.len() as f32 * 0.99) as usize];
        
        println!("\nStress test results:");
        println!("  Build time: {:?}", build_time);
        println!("  Query statistics:");
        println!("    Average: {:?}", avg_query);
        println!("    Median: {:?}", median_query);
        println!("    P95: {:?}", p95_query);
        println!("    P99: {:?}", p99_query);
        
        // Even under stress, queries should complete reasonably
        assert!(p99_query < Duration::from_millis(50),
               "P99 query time too high under stress: {:?}", p99_query);
        
        test_env.record_performance("stress_build_time", build_time);
        test_env.record_performance("stress_avg_query", avg_query);
        test_env.record_performance("stress_p99_query", p99_query);
        test_env.record_metric("stress_entity_count", kg.entity_count() as f64);
        test_env.record_metric("stress_relationship_count", kg.relationship_count() as f64);
    }
}

/// Platform-specific memory measurement
fn get_current_memory_usage() -> u64 {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(status) = fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb * 1024; // Convert KB to bytes
                        }
                    }
                }
            }
        }
    }
    
    #[cfg(target_os = "windows")]
    {
        use winapi::um::processthreadsapi::GetCurrentProcess;
        use winapi::um::psapi::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
        use std::mem;
        
        unsafe {
            let mut pmc: PROCESS_MEMORY_COUNTERS = mem::zeroed();
            let result = GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut pmc as *mut _,
                mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
            );
            
            if result != 0 {
                return pmc.WorkingSetSize as u64;
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    {
        use std::mem;
        use libc::{c_int, c_void, size_t};
        
        #[repr(C)]
        struct TaskBasicInfo {
            virtual_size: u64,
            resident_size: u64,
            resident_size_max: u64,
            user_time: [i64; 2],
            system_time: [i64; 2],
            policy: c_int,
            suspend_count: c_int,
        }
        
        extern "C" {
            fn task_info(
                task: c_int,
                flavor: c_int,
                task_info: *mut c_void,
                task_info_count: *mut size_t,
            ) -> c_int;
            
            fn mach_task_self() -> c_int;
        }
        
        const TASK_BASIC_INFO: c_int = 5;
        
        unsafe {
            let mut info: TaskBasicInfo = mem::zeroed();
            let mut count = mem::size_of::<TaskBasicInfo>() as size_t;
            
            let result = task_info(
                mach_task_self(),
                TASK_BASIC_INFO,
                &mut info as *mut _ as *mut c_void,
                &mut count,
            );
            
            if result == 0 {
                return info.resident_size;
            }
        }
    }
    
    // Fallback for other platforms
    0
}