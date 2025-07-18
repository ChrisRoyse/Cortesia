// Memory-Mapped Storage Performance Demo
// Phase 4.2: Persistent Memory-Mapped Storage with Product Quantization

use llmkg::{
    PersistentMMapStorage, 
    EntityKey, 
    EntityData,
    Result,
};
use std::time::Instant;
use std::collections::HashMap;
use std::path::Path;
use std::fs;

fn generate_random_embedding(dimension: usize) -> Vec<f32> {
    (0..dimension).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect()
}

fn create_sample_entity(id: u64, name: &str) -> EntityData {
    let properties = format!(r#"{{"name": "{}", "id": {}, "score": {}}}"#, 
                             name, id, rand::random::<f64>());
    
    EntityData {
        type_id: (id % 100) as u16, // Simple type classification
        properties,
        embedding: vec![], // Will be set separately
    }
}

fn main() -> Result<()> {
    println!("\n🗄️  === PHASE 4.2: MEMORY-MAPPED STORAGE DEMO ===");
    println!("Testing persistent memory-mapped storage with Product Quantization integration\n");

    let dimension = 384;
    let entity_count = 5000;
    let storage_file = "test_storage.db";
    
    // Clean up any existing file
    if Path::new(storage_file).exists() {
        fs::remove_file(storage_file).ok();
    }

    // === Test 1: Create New Persistent Storage ===
    println!("📊 Test 1: Creating New Persistent Storage");
    let start = Instant::now();
    
    let mut storage = PersistentMMapStorage::new(Some(storage_file), dimension)?;
    storage.set_auto_sync(false); // Manual sync for better control
    
    let creation_time = start.elapsed();
    println!("✅ Storage created in {:.2}ms", creation_time.as_millis());

    // === Test 2: Single Entity Operations ===
    println!("\n💾 Test 2: Single Entity Operations");
    
    let entity_key = EntityKey::from_hash("test_entity_1");
    let entity_data = create_sample_entity(1, "Test Entity 1");
    let embedding = generate_random_embedding(dimension);
    
    let start = Instant::now();
    storage.add_entity(entity_key, &entity_data, &embedding)?;
    let single_add_time = start.elapsed();
    
    println!("✅ Single entity add: {:.3}ms", single_add_time.as_micros() as f32 / 1000.0);
    
    // Test retrieval
    let start = Instant::now();
    let retrieved_entity = storage.get_entity(entity_key);
    let retrieval_time = start.elapsed();
    
    println!("✅ Single entity retrieval: {:.3}ms (found: {})", 
             retrieval_time.as_micros() as f32 / 1000.0,
             retrieved_entity.is_some());

    // === Test 3: Batch Operations ===
    println!("\n⚡ Test 3: Batch Entity Operations");
    
    let mut entities_data = Vec::new();
    for i in 2..entity_count + 2 {
        let key = EntityKey::from_hash(&format!("entity_{}", i));
        let data = create_sample_entity(i as u64, &format!("Entity {}", i));
        let embedding = generate_random_embedding(dimension);
        entities_data.push((key, data, embedding));
    }
    
    let start = Instant::now();
    storage.batch_add_entities(&entities_data)?;
    let batch_time = start.elapsed();
    
    println!("✅ Batch add {} entities: {:.2}ms ({:.0} entities/sec)", 
             entity_count, batch_time.as_millis(),
             entity_count as f32 / batch_time.as_secs_f32());

    // === Test 4: Memory Usage Before Persistence ===
    println!("\n📈 Test 4: Memory Usage Analysis (Before Persistence)");
    let stats = storage.storage_stats();
    
    println!("📊 Storage Statistics (In-Memory):");
    println!("  • Total entities: {}", stats.entity_count);
    println!("  • Memory usage: {} KB", stats.memory_usage_bytes / 1024);
    println!("  • Quantized embeddings: {} KB", stats.quantized_embedding_bytes / 1024);
    println!("  • Compression ratio: {:.1}:1", stats.compression_ratio);
    println!("  • Avg bytes per entity: {}", stats.avg_bytes_per_entity);
    println!("  • Quantizer trained: {}", stats.quantizer_trained);
    if stats.quantizer_trained {
        println!("  • Quantizer quality: {:.4}", stats.quantizer_quality);
    }

    // === Test 5: Persistence to Disk ===
    println!("\n💿 Test 5: Persistence to Disk");
    
    let start = Instant::now();
    storage.sync_to_disk()?;
    let sync_time = start.elapsed();
    
    let file_size = fs::metadata(storage_file)?.len();
    let stats_after_sync = storage.storage_stats();
    
    println!("✅ Sync to disk: {:.2}ms", sync_time.as_millis());
    println!("📁 File size: {} KB", file_size / 1024);
    println!("📊 Compression efficiency: {:.1}% of original size", 
             (file_size as f32) / (stats_after_sync.entity_count as f32 * dimension as f32 * 4.0) * 100.0);

    // === Test 6: Load from Disk ===
    println!("\n📂 Test 6: Loading from Disk");
    
    drop(storage); // Close current storage
    
    let start = Instant::now();
    let loaded_storage = PersistentMMapStorage::load(storage_file)?;
    let load_time = start.elapsed();
    
    let loaded_stats = loaded_storage.storage_stats();
    
    println!("✅ Loaded from disk: {:.2}ms", load_time.as_millis());
    println!("📊 Loaded {} entities", loaded_stats.entity_count);
    println!("💾 Memory usage after load: {} KB", loaded_stats.memory_usage_bytes / 1024);

    // === Test 7: Similarity Search on Loaded Data ===
    println!("\n🔍 Test 7: Similarity Search on Loaded Data");
    
    let query = generate_random_embedding(dimension);
    let k = 10;
    
    let start = Instant::now();
    let search_results = loaded_storage.similarity_search(&query, k)?;
    let search_time = start.elapsed();
    
    println!("✅ Similarity search: {:.2}ms (found {} results)", 
             search_time.as_millis(), search_results.len());
    
    println!("🏆 Top {} results:", 3.min(search_results.len()));
    for (i, (entity_key, similarity)) in search_results.iter().take(3).enumerate() {
        println!("  {}. Entity {:?}: similarity = {:.4}", 
                 i + 1, entity_key, similarity);
    }

    // === Test 8: Embedding Reconstruction ===
    println!("\n🔄 Test 8: Embedding Reconstruction");
    
    let test_entity = EntityKey::from_hash("entity_10");
    
    let start = Instant::now();
    let reconstructed = loaded_storage.get_reconstructed_embedding(test_entity)?;
    let reconstruction_time = start.elapsed();
    
    if let Some(reconstructed_embedding) = reconstructed {
        println!("✅ Embedding reconstruction: {:.3}ms", reconstruction_time.as_micros() as f32 / 1000.0);
        println!("📏 Reconstructed dimension: {}", reconstructed_embedding.len());
        
        // Show first few values as example
        if reconstructed_embedding.len() >= 5 {
            println!("🔢 First 5 values: {:.4}, {:.4}, {:.4}, {:.4}, {:.4}", 
                     reconstructed_embedding[0], reconstructed_embedding[1], 
                     reconstructed_embedding[2], reconstructed_embedding[3], 
                     reconstructed_embedding[4]);
        }
    } else {
        println!("❌ Entity not found for reconstruction");
    }

    // === Test 9: Performance Stress Test ===
    println!("\n🏃 Test 9: Performance Stress Test");
    
    let stress_queries = 1000;
    let mut total_search_time = 0u128;
    let mut total_retrieval_time = 0u128;
    
    for i in 0..stress_queries {
        // Random similarity search
        let query = generate_random_embedding(dimension);
        let start = Instant::now();
        let _results = loaded_storage.similarity_search(&query, 5)?;
        total_search_time += start.elapsed().as_micros();
        
        // Random entity retrieval
        let entity_key = EntityKey::from_hash(&format!("entity_{}", (i % entity_count) + 2));
        let start = Instant::now();
        let _entity = loaded_storage.get_entity(entity_key);
        total_retrieval_time += start.elapsed().as_micros();
    }
    
    let avg_search_time = total_search_time as f32 / stress_queries as f32 / 1000.0;
    let avg_retrieval_time = total_retrieval_time as f32 / stress_queries as f32 / 1000.0;
    
    println!("✅ Stress test completed:");
    println!("  • {} similarity searches: avg {:.3}ms each", stress_queries, avg_search_time);
    println!("  • {} entity retrievals: avg {:.3}ms each", stress_queries, avg_retrieval_time);
    println!("  • Total throughput: {:.0} searches/sec, {:.0} retrievals/sec", 
             1000.0 / avg_search_time, 1000.0 / avg_retrieval_time);

    // === Test 10: Storage Compaction ===
    println!("\n🗜️  Test 10: Storage Compaction");
    
    let stats_before = loaded_storage.storage_stats();
    let mut mutable_storage = loaded_storage; // Need mutable reference for compaction
    
    let start = Instant::now();
    mutable_storage.compact()?;
    let compact_time = start.elapsed();
    
    let stats_after = mutable_storage.storage_stats();
    
    println!("✅ Storage compaction: {:.2}ms", compact_time.as_millis());
    println!("📊 Memory before: {} KB, after: {} KB", 
             stats_before.memory_usage_bytes / 1024,
             stats_after.memory_usage_bytes / 1024);

    // === Final Statistics ===
    println!("\n📋 === PHASE 4.2 SUMMARY ===");
    let final_stats = mutable_storage.storage_stats();
    
    println!("✅ Memory-Mapped Storage Features Implemented:");
    println!("  • Persistent file-based storage with custom format");
    println!("  • Integrated Product Quantization for space efficiency");
    println!("  • Batch operations for high-throughput data loading");
    println!("  • Zero-copy entity and embedding access");
    println!("  • Automatic quantizer training and serialization");
    println!("  • Storage compaction and optimization");
    
    println!("\n📊 Performance Results:");
    println!("  • Storage file size: {} KB for {} entities", file_size / 1024, final_stats.entity_count);
    println!("  • Compression ratio: {:.1}:1", final_stats.compression_ratio);
    println!("  • Batch insert: {:.0} entities/sec", entity_count as f32 / batch_time.as_secs_f32());
    println!("  • Load from disk: {:.2}ms", load_time.as_millis());
    println!("  • Avg search time: {:.3}ms", avg_search_time);
    println!("  • Avg retrieval time: {:.3}ms", avg_retrieval_time);
    println!("  • Read operations: {}", final_stats.read_operations);
    println!("  • Write operations: {}", final_stats.write_operations);
    
    println!("\n🎯 Phase 4.2 Completed Successfully!");
    println!("Ready for Phase 4.3: String interning for properties\n");
    
    // Clean up test file
    fs::remove_file(storage_file).ok();
    
    Ok(())
}