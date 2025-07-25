//! Integration test for real storage with MMAP, HNSW, and Quantization
//! 
//! This test demonstrates:
//! - Zero-copy MMAP storage with persistence
//! - HNSW indexing for fast similarity search (<10ms)
//! - Product quantization for 8-16x compression
//! - String interning for 50-90% memory savings on duplicates
//! - <1ms serialization overhead per entity

use llmkg::{
    storage::{
        persistent_mmap::PersistentMMapStorage,
        hnsw::HnswIndex,
        quantized_index::QuantizedIndex,
        string_interner::StringInterner,
    },
    core::{
        entity_extractor::{CognitiveEntityExtractor, CognitiveEntity},
        types::{EntityKey, EntityData},
    },
    cognitive::{
        orchestrator::CognitiveOrchestrator,
        attention_manager::AttentionManager,
        working_memory::WorkingMemorySystem,
    },
    monitoring::{
        brain_metrics_collector::BrainMetricsCollector,
        performance::PerformanceMonitor,
    },
};
use std::sync::Arc;
use parking_lot::RwLock;
use std::time::Instant;

#[tokio::test]
async fn test_real_storage_integration() {
    println!("\n=== Real Storage Integration Test ===\n");
    
    // Setup storage components
    let embedding_dim = 384; // MiniLM embedding size
    let mmap_storage = Arc::new(PersistentMMapStorage::new(
        Some("test_storage.db"),
        embedding_dim
    ).expect("Failed to create MMAP storage"));
    
    let string_interner = Arc::new(StringInterner::new());
    let hnsw_index = Arc::new(RwLock::new(HnswIndex::new(embedding_dim)));
    let quantized_index = Arc::new(QuantizedIndex::new(embedding_dim, 8).expect("Failed to create quantized index"));
    
    // Create cognitive components (using test builders)
    let cognitive_orchestrator = Arc::new(llmkg::test_support::builders::build_test_cognitive_orchestrator().await);
    let attention_manager = Arc::new(llmkg::test_support::builders::build_test_attention_manager().await);
    let working_memory = Arc::new(llmkg::test_support::builders::build_test_working_memory().await);
    let metrics_collector = Arc::new(llmkg::test_support::builders::build_test_brain_metrics_collector().await);
    let performance_monitor = Arc::new(llmkg::test_support::builders::build_test_performance_monitor().await);
    
    // Create entity extractor with storage integration
    let extractor = CognitiveEntityExtractor::new(
        cognitive_orchestrator,
        attention_manager,
        working_memory,
        metrics_collector,
        performance_monitor,
    ).with_storage(
        mmap_storage.clone(),
        string_interner.clone(),
        hnsw_index.clone(),
        quantized_index.clone(),
    );
    
    // Train quantizer with sample data
    println!("Training quantizer...");
    let training_samples: Vec<Vec<f32>> = (0..500)
        .map(|i| {
            (0..embedding_dim)
                .map(|j| ((i + j) as f32 * 0.01).sin())
                .collect()
        })
        .collect();
    
    quantized_index.train(&training_samples).expect("Failed to train quantizer");
    
    // Test entity extraction and storage
    println!("\nExtracting and storing entities...");
    let test_texts = vec![
        "Albert Einstein developed the Theory of Relativity in 1905.",
        "Marie Curie won the Nobel Prize in Physics and Chemistry.",
        "The Large Hadron Collider at CERN discovers new particles.",
        "Quantum computing promises exponential speedup for certain algorithms.",
        "GPT models demonstrate emergent capabilities in language understanding.",
    ];
    
    let mut all_entities = Vec::new();
    let mut storage_times = Vec::new();
    
    for (i, text) in test_texts.iter().enumerate() {
        println!("\nProcessing: \"{}\"", text);
        
        let start = Instant::now();
        let entities = extractor.extract_entities(text).await.expect("Failed to extract entities");
        let extraction_time = start.elapsed();
        
        // The storage happens inside extract_entities when storage is configured
        storage_times.push(extraction_time);
        
        println!("  Found {} entities in {:?}", entities.len(), extraction_time);
        for entity in &entities {
            println!("    - {} ({:?}, confidence: {:.2})", entity.name, entity.entity_type, entity.confidence_score);
        }
        
        all_entities.extend(entities);
    }
    
    // Calculate average storage time
    let avg_storage_time = storage_times.iter().map(|d| d.as_micros()).sum::<u128>() / storage_times.len() as u128;
    println!("\nAverage extraction + storage time: {}μs", avg_storage_time);
    assert!(avg_storage_time < 10_000, "Storage should be fast (< 10ms per text)");
    
    // Test similarity search
    println!("\n=== Testing Similarity Search ===");
    
    // Create a query embedding (simulating search for "physics")
    let query_embedding: Vec<f32> = (0..embedding_dim)
        .map(|i| ((i as f32 * 0.02).cos() + (i as f32 * 0.03).sin()) / 2.0)
        .collect();
    
    let search_start = Instant::now();
    let results = extractor.search_similar_entities(&query_embedding, 5).await.expect("Failed to search");
    let search_time = search_start.elapsed();
    
    println!("\nSimilarity search completed in {:?}", search_time);
    assert!(search_time.as_millis() < 10, "Search should be <10ms");
    
    println!("Top {} similar entities:", results.len());
    for entity in &results {
        println!("  - {} ({:?})", entity.name, entity.entity_type);
    }
    
    // Test MMAP persistence
    println!("\n=== Testing Persistence ===");
    let stats_before = mmap_storage.storage_stats();
    println!("Storage stats before sync:");
    println!("  Entities: {}", stats_before.entity_count);
    println!("  Memory usage: {:.2} KB", stats_before.memory_usage_bytes as f64 / 1024.0);
    println!("  Compression ratio: {:.2}:1", stats_before.compression_ratio);
    
    mmap_storage.sync_to_disk().expect("Failed to sync to disk");
    println!("Data synced to disk successfully");
    
    // Test string interning efficiency
    println!("\n=== String Interning Stats ===");
    let interner_stats = string_interner.stats();
    println!("Unique strings: {}", interner_stats.unique_strings);
    println!("Total references: {}", interner_stats.total_references);
    println!("Deduplication ratio: {:.2}:1", interner_stats.deduplication_ratio);
    println!("Memory saved: {} bytes", interner_stats.memory_saved_bytes);
    
    // Test HNSW index stats
    println!("\n=== HNSW Index Stats ===");
    let hnsw = hnsw_index.read();
    let hnsw_stats = hnsw.stats();
    println!("Nodes: {}", hnsw_stats.node_count);
    println!("Max layer: {}", hnsw_stats.max_layer);
    println!("Average connections: {:.2}", hnsw_stats.avg_connections);
    drop(hnsw);
    
    // Test quantized index stats
    println!("\n=== Quantized Index Stats ===");
    let quant_stats = quantized_index.memory_usage();
    println!("Entities: {}", quant_stats.entity_count);
    println!("Compressed embeddings: {} bytes", quant_stats.compressed_embeddings_bytes);
    println!("Compression ratio: {:.2}:1", quant_stats.compression_ratio);
    println!("Bytes per entity: {}", quant_stats.bytes_per_entity);
    
    // Performance assertions
    assert!(stats_before.entity_count >= 5, "Should have stored at least 5 entities");
    assert!(stats_before.compression_ratio > 1.0, "Should achieve compression");
    assert!(interner_stats.deduplication_ratio > 1.0, "Should have string deduplication");
    assert!(hnsw_stats.node_count > 0, "HNSW should have nodes");
    assert!(quant_stats.compression_ratio > 2.0, "Quantization should achieve >2x compression");
    
    println!("\n✓ All storage integration tests passed!");
}

#[tokio::test]
async fn test_storage_performance_at_scale() {
    println!("\n=== Storage Performance at Scale Test ===\n");
    
    let embedding_dim = 96; // Smaller embeddings for speed
    let num_entities = 1000;
    
    // Setup storage
    let mmap_storage = Arc::new(PersistentMMapStorage::new(
        Some("scale_test.db"),
        embedding_dim
    ).expect("Failed to create MMAP storage"));
    
    let hnsw_index = Arc::new(RwLock::new(HnswIndex::new(embedding_dim)));
    let quantized_index = Arc::new(QuantizedIndex::new(embedding_dim, 4).expect("Failed to create quantized index"));
    
    // Train quantizer
    let training_samples: Vec<Vec<f32>> = (0..200)
        .map(|i| {
            (0..embedding_dim)
                .map(|j| ((i * j) as f32 / 100.0).tanh())
                .collect()
        })
        .collect();
    
    quantized_index.train(&training_samples).expect("Failed to train quantizer");
    
    // Measure entity storage performance
    println!("Storing {} entities...", num_entities);
    let mut storage_times = Vec::new();
    
    for i in 0..num_entities {
        let entity_key = EntityKey::from_raw(i as u64);
        let embedding: Vec<f32> = (0..embedding_dim)
            .map(|j| ((i + j) as f32 / 100.0).sin())
            .collect();
        
        let entity_data = EntityData::new(
            (i % 10) as u16,
            format!("Entity_{}", i),
            embedding.clone(),
        );
        
        let start = Instant::now();
        
        // Store in MMAP
        mmap_storage.add_entity(entity_key, &entity_data, &embedding).expect("Failed to add to MMAP");
        
        // Add to HNSW
        let mut hnsw = hnsw_index.write();
        hnsw.insert(i as u32, entity_key, embedding.clone()).expect("Failed to add to HNSW");
        drop(hnsw);
        
        // Add to quantized index
        quantized_index.insert(i as u32, entity_key, embedding).expect("Failed to add to quantized index");
        
        storage_times.push(start.elapsed());
        
        if (i + 1) % 100 == 0 {
            let avg_time = storage_times.iter().map(|d| d.as_micros()).sum::<u128>() / storage_times.len() as u128;
            println!("  {} entities stored (avg: {}μs)", i + 1, avg_time);
        }
    }
    
    let avg_storage_time = storage_times.iter().map(|d| d.as_micros()).sum::<u128>() / storage_times.len() as u128;
    println!("\nAverage storage time per entity: {}μs", avg_storage_time);
    assert!(avg_storage_time < 1000, "Should achieve <1ms per entity storage");
    
    // Test search performance at scale
    println!("\nTesting search performance...");
    let query: Vec<f32> = (0..embedding_dim).map(|i| (i as f32 * 0.05).cos()).collect();
    
    let mut search_times = Vec::new();
    for k in [1, 10, 50, 100] {
        let start = Instant::now();
        
        // HNSW search
        let hnsw = hnsw_index.read();
        let results = hnsw.search(&query, k);
        drop(hnsw);
        
        let search_time = start.elapsed();
        search_times.push(search_time);
        
        println!("  Top-{} search: {:?} (found {} results)", k, search_time, results.len());
        assert!(search_time.as_millis() < 10, "Search should be <10ms even for k={}", k);
    }
    
    // Final stats
    let stats = mmap_storage.storage_stats();
    println!("\nFinal storage statistics:");
    println!("  Total entities: {}", stats.entity_count);
    println!("  Memory usage: {:.2} MB", stats.memory_usage_bytes as f64 / 1024.0 / 1024.0);
    println!("  File size: {:.2} MB", stats.file_size_bytes as f64 / 1024.0 / 1024.0);
    println!("  Compression ratio: {:.2}:1", stats.compression_ratio);
    println!("  Avg bytes per entity: {}", stats.avg_bytes_per_entity);
    
    let raw_size = num_entities * embedding_dim * 4; // 4 bytes per f32
    let actual_size = stats.memory_usage_bytes as usize;
    let efficiency = raw_size as f64 / actual_size as f64;
    println!("\nMemory efficiency: {:.2}x", efficiency);
    assert!(efficiency > 2.0, "Should achieve >2x memory efficiency with quantization");
    
    println!("\n✓ Scale test completed successfully!");
}

// Cleanup test files
#[cfg(test)]
mod cleanup {
    use std::fs;
    
    pub fn cleanup_test_files() {
        let _ = fs::remove_file("test_storage.db");
        let _ = fs::remove_file("scale_test.db");
        let _ = fs::remove_file("llmkg_integrated.db");
    }
    
    #[test]
    fn cleanup() {
        cleanup_test_files();
    }
}