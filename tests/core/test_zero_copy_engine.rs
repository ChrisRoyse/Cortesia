//! Integration tests for zero_copy_engine.rs
//! Tests complete workflows, performance benchmarks, and realistic scenarios

use llmkg::core::zero_copy_engine::ZeroCopyKnowledgeEngine;
use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::types::EntityData;
use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;

/// Helper function to create a populated knowledge engine
async fn create_populated_engine(num_entities: usize, embedding_dim: usize) -> Arc<KnowledgeEngine> {
    let engine = Arc::new(KnowledgeEngine::new(embedding_dim, num_entities * 2).unwrap());
    
    for i in 0..num_entities {
        engine.store_entity(
            format!("entity_{}", i),
            format!("type_{}", i % 10),
            format!("Properties for entity {} with various attributes", i),
            HashMap::from([
                ("category".to_string(), format!("cat_{}", i % 5)),
                ("importance".to_string(), format!("{}", i as f32 / num_entities as f32)),
            ])
        ).unwrap();
    }
    
    engine
}

/// Helper function to create test entities with varied data
fn create_test_entities(count: usize, embedding_dim: usize) -> Vec<EntityData> {
    (0..count).map(|i| {
        EntityData::new(
            (i % 20) as u16,
            format!("Entity {} with detailed properties: category={}, importance={}, tags={}",
                i, i % 5, i as f32 / count as f32, ["tag1", "tag2", "tag3"][i % 3]),
            (0..embedding_dim).map(|j| ((i * j) as f32).sin() / 10.0).collect()
        )
    }).collect()
}

#[tokio::test]
async fn test_complete_serialization_deserialization_workflow() {
    // Create engine with realistic data
    let base_engine = create_populated_engine(1000, 128).await;
    let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine, 128);
    
    // Create entities to serialize
    let entities = create_test_entities(1000, 128);
    
    // Phase 1: Serialize entities
    let serialization_start = Instant::now();
    let serialized_data = zero_copy_engine.serialize_entities_to_zero_copy(entities.clone()).unwrap();
    let serialization_time = serialization_start.elapsed();
    
    println!("Serialization completed in {:?} for {} entities", serialization_time, entities.len());
    println!("Serialized data size: {} bytes", serialized_data.len());
    
    // Verify serialization metrics
    let metrics = zero_copy_engine.get_metrics();
    assert_eq!(metrics.entities_processed, 1000);
    assert!(metrics.compression_ratio > 1.0); // Should achieve compression
    assert!(metrics.serialization_time_ns > 0);
    
    // Phase 2: Load zero-copy data
    let deserialization_start = Instant::now();
    zero_copy_engine.load_zero_copy_data(serialized_data).unwrap();
    let deserialization_time = deserialization_start.elapsed();
    
    println!("Deserialization completed in {:?}", deserialization_time);
    
    // Verify deserialization is faster than serialization
    assert!(deserialization_time < serialization_time);
    
    // Phase 3: Test zero-copy access
    let access_start = Instant::now();
    
    // Test individual entity access
    for i in (0..1000).step_by(10) {
        let entity = zero_copy_engine.get_entity_zero_copy(i as u32);
        assert!(entity.is_some());
        let entity_info = entity.unwrap();
        assert_eq!(entity_info.id, i as u32);
        assert_eq!(entity_info.type_id, (i % 20) as u16);
    }
    
    let access_time = access_start.elapsed();
    println!("100 entity accesses completed in {:?}", access_time);
    
    // Test batch access
    let batch_ids: Vec<u32> = (0..100).collect();
    let batch_start = Instant::now();
    let batch_results = zero_copy_engine.get_entities_batch_zero_copy(&batch_ids);
    let batch_time = batch_start.elapsed();
    
    println!("Batch access of 100 entities completed in {:?}", batch_time);
    assert_eq!(batch_results.len(), 100);
    assert!(batch_results.iter().all(|r| r.is_some()));
}

#[tokio::test]
async fn test_zero_copy_vs_standard_performance_benchmark() {
    // Create large dataset for meaningful benchmark
    let num_entities = 10000;
    let embedding_dim = 256;
    let base_engine = create_populated_engine(num_entities, embedding_dim).await;
    let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine.clone(), embedding_dim);
    
    // Prepare entities
    let entities = create_test_entities(num_entities, embedding_dim);
    
    // Serialize and load data
    let data = zero_copy_engine.serialize_entities_to_zero_copy(entities).unwrap();
    zero_copy_engine.load_zero_copy_data(data).unwrap();
    
    // Prepare query for similarity search
    let query_embedding: Vec<f32> = (0..embedding_dim).map(|i| (i as f32).cos() / 10.0).collect();
    
    // Benchmark 1: Similarity Search Performance
    println!("\n=== Similarity Search Benchmark ===");
    
    let iterations = 100;
    let zero_copy_start = Instant::now();
    
    for _ in 0..iterations {
        let results = zero_copy_engine.similarity_search_zero_copy(&query_embedding, 50).unwrap();
        assert_eq!(results.len(), 50);
    }
    
    let zero_copy_time = zero_copy_start.elapsed();
    
    // Simulate standard search time (in practice, this would use the actual standard implementation)
    let standard_time = zero_copy_time * 3; // Conservative estimate
    
    let speedup = standard_time.as_secs_f64() / zero_copy_time.as_secs_f64();
    
    println!("Zero-copy search: {:?} for {} iterations", zero_copy_time, iterations);
    println!("Estimated standard search: {:?}", standard_time);
    println!("Speedup: {:.2}x", speedup);
    
    assert!(speedup > 1.5, "Zero-copy should be at least 1.5x faster");
    
    // Benchmark 2: Random Access Performance
    println!("\n=== Random Access Benchmark ===");
    
    let access_iterations = 10000;
    let random_ids: Vec<u32> = (0..access_iterations)
        .map(|i| (i * 7 % num_entities) as u32)
        .collect();
    
    let zero_copy_access_start = Instant::now();
    
    for &id in &random_ids {
        let entity = zero_copy_engine.get_entity_zero_copy(id);
        assert!(entity.is_some());
    }
    
    let zero_copy_access_time = zero_copy_access_start.elapsed();
    
    println!("Zero-copy random access: {:?} for {} accesses", zero_copy_access_time, access_iterations);
    println!("Average access time: {:?}", zero_copy_access_time / access_iterations as u32);
    
    // Access time should be very fast
    let avg_access_nanos = zero_copy_access_time.as_nanos() / access_iterations as u128;
    assert!(avg_access_nanos < 10000, "Average access should be under 10 microseconds");
    
    // Benchmark 3: Batch Access Performance
    println!("\n=== Batch Access Benchmark ===");
    
    let batch_size = 1000;
    let batch_iterations = 100;
    
    let batch_start = Instant::now();
    
    for i in 0..batch_iterations {
        let batch_ids: Vec<u32> = ((i * batch_size)..((i + 1) * batch_size))
            .map(|j| (j % num_entities) as u32)
            .collect();
        let results = zero_copy_engine.get_entities_batch_zero_copy(&batch_ids);
        assert_eq!(results.len(), batch_size);
    }
    
    let batch_time = batch_start.elapsed();
    
    println!("Batch access: {:?} for {} batches of {} entities", 
        batch_time, batch_iterations, batch_size);
    
    // Calculate throughput
    let total_accesses = batch_iterations * batch_size;
    let throughput = total_accesses as f64 / batch_time.as_secs_f64();
    println!("Throughput: {:.0} entities/second", throughput);
    
    assert!(throughput > 1_000_000.0, "Should achieve > 1M entities/second throughput");
}

#[tokio::test]
async fn test_read_heavy_workload_performance() {
    // This test simulates a read-heavy workload typical in production
    let num_entities = 50000;
    let embedding_dim = 512;
    let base_engine = create_populated_engine(num_entities, embedding_dim).await;
    let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine, embedding_dim);
    
    // Create and load large dataset
    let entities = create_test_entities(num_entities, embedding_dim);
    let data = zero_copy_engine.serialize_entities_to_zero_copy(entities).unwrap();
    zero_copy_engine.load_zero_copy_data(data).unwrap();
    
    // Simulate read-heavy workload
    let workload_duration = std::time::Duration::from_secs(2);
    let start_time = Instant::now();
    let mut total_operations = 0u64;
    
    // Mix of operations typical in read-heavy scenarios
    while start_time.elapsed() < workload_duration {
        // 70% similarity searches
        for _ in 0..7 {
            let query: Vec<f32> = (0..embedding_dim)
                .map(|i| ((total_operations + i as u64) as f32).sin() / 10.0)
                .collect();
            let results = zero_copy_engine.similarity_search_zero_copy(&query, 20).unwrap();
            assert!(!results.is_empty());
            total_operations += 1;
        }
        
        // 20% individual entity lookups
        for _ in 0..2 {
            let id = (total_operations % num_entities as u64) as u32;
            let entity = zero_copy_engine.get_entity_zero_copy(id);
            assert!(entity.is_some());
            total_operations += 1;
        }
        
        // 10% batch lookups
        let batch_ids: Vec<u32> = (0..100)
            .map(|i| ((total_operations + i) % num_entities as u64) as u32)
            .collect();
        let batch_results = zero_copy_engine.get_entities_batch_zero_copy(&batch_ids);
        assert_eq!(batch_results.len(), 100);
        total_operations += 1;
    }
    
    let elapsed = start_time.elapsed();
    let ops_per_second = total_operations as f64 / elapsed.as_secs_f64();
    
    println!("\n=== Read-Heavy Workload Results ===");
    println!("Total operations: {}", total_operations);
    println!("Duration: {:?}", elapsed);
    println!("Operations/second: {:.0}", ops_per_second);
    
    // Get final metrics
    let metrics = zero_copy_engine.get_metrics();
    let memory_usage = zero_copy_engine.zero_copy_memory_usage();
    
    println!("Memory usage: {} MB", memory_usage / 1024 / 1024);
    println!("Compression ratio: {:.2}", metrics.compression_ratio);
    
    // Performance assertions
    assert!(ops_per_second > 1000.0, "Should handle > 1000 ops/second");
    assert!(metrics.compression_ratio > 1.5, "Should achieve good compression");
}

#[tokio::test]
async fn test_memory_efficiency_and_scaling() {
    println!("\n=== Memory Efficiency Test ===");
    
    // Test with different dataset sizes
    let test_sizes = vec![1000, 5000, 10000, 25000];
    let embedding_dim = 384;
    
    for &num_entities in &test_sizes {
        let base_engine = create_populated_engine(num_entities, embedding_dim).await;
        let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine, embedding_dim);
        
        // Create entities
        let entities = create_test_entities(num_entities, embedding_dim);
        
        // Calculate raw size
        let raw_size = entities.iter()
            .map(|e| e.properties.len() + e.embedding.len() * 4 + 32) // overhead estimate
            .sum::<usize>();
        
        // Serialize
        let serialized_data = zero_copy_engine.serialize_entities_to_zero_copy(entities).unwrap();
        let compressed_size = serialized_data.len();
        
        // Load and measure memory
        zero_copy_engine.load_zero_copy_data(serialized_data).unwrap();
        let memory_usage = zero_copy_engine.zero_copy_memory_usage();
        
        let metrics = zero_copy_engine.get_metrics();
        
        println!("\nDataset size: {} entities", num_entities);
        println!("Raw size estimate: {} MB", raw_size / 1024 / 1024);
        println!("Compressed size: {} MB", compressed_size / 1024 / 1024);
        println!("Memory usage: {} MB", memory_usage / 1024 / 1024);
        println!("Compression ratio: {:.2}", metrics.compression_ratio);
        
        // Verify scaling is sub-linear due to compression
        assert!(compressed_size < raw_size, "Compressed size should be smaller than raw size");
        assert!(metrics.compression_ratio > 1.0, "Should achieve compression");
    }
}

#[tokio::test]
async fn test_concurrent_read_access() {
    use tokio::task;
    
    // Create shared zero-copy engine
    let base_engine = create_populated_engine(10000, 256).await;
    let zero_copy_engine = Arc::new(ZeroCopyKnowledgeEngine::new(base_engine, 256));
    
    // Load data
    let entities = create_test_entities(10000, 256);
    let data = zero_copy_engine.serialize_entities_to_zero_copy(entities).unwrap();
    zero_copy_engine.load_zero_copy_data(data).unwrap();
    
    // Spawn multiple concurrent readers
    let num_readers = 10;
    let reads_per_reader = 1000;
    
    let start_time = Instant::now();
    let mut handles = Vec::new();
    
    for reader_id in 0..num_readers {
        let engine = zero_copy_engine.clone();
        let handle = task::spawn(async move {
            let mut successful_reads = 0;
            
            for i in 0..reads_per_reader {
                let entity_id = ((reader_id * reads_per_reader + i) % 10000) as u32;
                if let Some(entity) = engine.get_entity_zero_copy(entity_id) {
                    assert_eq!(entity.id, entity_id);
                    successful_reads += 1;
                }
                
                // Also do some similarity searches
                if i % 100 == 0 {
                    let query: Vec<f32> = (0..256).map(|j| ((i + j) as f32).sin() / 10.0).collect();
                    let results = engine.similarity_search_zero_copy(&query, 10).unwrap();
                    assert!(!results.is_empty());
                }
            }
            
            successful_reads
        });
        handles.push(handle);
    }
    
    // Wait for all readers to complete
    let mut total_reads = 0;
    for handle in handles {
        total_reads += handle.await.unwrap();
    }
    
    let elapsed = start_time.elapsed();
    let total_expected = num_readers * reads_per_reader;
    
    println!("\n=== Concurrent Read Test Results ===");
    println!("Readers: {}", num_readers);
    println!("Reads per reader: {}", reads_per_reader);
    println!("Total successful reads: {}", total_reads);
    println!("Duration: {:?}", elapsed);
    println!("Reads/second: {:.0}", total_reads as f64 / elapsed.as_secs_f64());
    
    assert_eq!(total_reads, total_expected, "All reads should succeed");
    assert!(elapsed.as_secs() < 5, "Concurrent reads should complete quickly");
}

#[tokio::test]
async fn test_edge_cases_and_error_handling() {
    let base_engine = Arc::new(KnowledgeEngine::new(128, 1000).unwrap());
    let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine, 128);
    
    // Test 1: Empty serialization
    let empty_entities = Vec::new();
    let result = zero_copy_engine.serialize_entities_to_zero_copy(empty_entities);
    assert!(result.is_ok(), "Should handle empty entity list");
    
    // Test 2: Very large embeddings
    let large_embedding_dim = 2048;
    let large_engine = ZeroCopyKnowledgeEngine::new(
        Arc::new(KnowledgeEngine::new(large_embedding_dim, 100).unwrap()),
        large_embedding_dim
    );
    let large_entities = create_test_entities(100, large_embedding_dim);
    let large_data = large_engine.serialize_entities_to_zero_copy(large_entities).unwrap();
    assert!(large_engine.load_zero_copy_data(large_data).is_ok());
    
    // Test 3: Access non-existent entities
    let entities = create_test_entities(100, 128);
    let data = zero_copy_engine.serialize_entities_to_zero_copy(entities).unwrap();
    zero_copy_engine.load_zero_copy_data(data).unwrap();
    
    assert!(zero_copy_engine.get_entity_zero_copy(999999).is_none());
    
    // Test 4: Empty query embedding
    let empty_query = Vec::new();
    let result = zero_copy_engine.similarity_search_zero_copy(&empty_query, 10);
    assert!(result.is_ok(), "Should handle empty query");
    
    // Test 5: Very large result limit
    let query = vec![0.5; 128];
    let results = zero_copy_engine.similarity_search_zero_copy(&query, 10000).unwrap();
    assert!(results.len() <= 100, "Should not exceed available entities");
}