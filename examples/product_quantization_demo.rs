// Product Quantization Performance Demo
// Phase 4.1: True Product Quantization Implementation

use llmkg::{
    EmbeddingStore, 
    ProductQuantizer,
    EntityKey,
    error::Result,
};
use std::time::Instant;

fn generate_random_embedding(dimension: usize) -> Vec<f32> {
    (0..dimension).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect()
}

fn main() -> Result<()> {
    println!("\n🧪 === PHASE 4.1: PRODUCT QUANTIZATION DEMO ===");
    println!("Testing true Product Quantization compression with memory optimization\n");

    let dimension = 384; // Common embedding dimension
    let entity_count = 2000;
    let target_compression = 16.0; // 16:1 compression ratio
    
    // === Test 1: Basic Product Quantization ===
    println!("📊 Test 1: Basic Product Quantization Training");
    let start = Instant::now();
    
    let mut quantizer = ProductQuantizer::new_optimized(dimension, target_compression)?;
    
    // Generate training data
    let embeddings: Vec<Vec<f32>> = (0..entity_count)
        .map(|_| generate_random_embedding(dimension))
        .collect();
    
    println!("🔧 Training quantizer on {} embeddings (dimension: {})", entity_count, dimension);
    quantizer.train_adaptive(&embeddings)?;
    
    let training_time = start.elapsed();
    println!("✅ Training completed in {:.2}ms", training_time.as_millis());
    
    // === Test 2: Compression Analysis ===
    println!("\n📈 Test 2: Compression Analysis");
    let stats = quantizer.compression_stats(dimension);
    
    println!("📊 Compression Statistics:");
    println!("  • Original size: {} bytes per embedding", stats.original_bytes);
    println!("  • Compressed size: {} bytes per embedding", stats.compressed_bytes);
    println!("  • Compression ratio: {:.1}:1", stats.compression_ratio);
    println!("  • Memory saved: {} bytes per embedding", stats.memory_saved);
    println!("  • Training quality: {:.4} (lower is better)", stats.training_quality);
    println!("  • Codebook memory: {} KB", stats.codebook_memory / 1024);
    
    // === Test 3: Encoding/Decoding Performance ===
    println!("\n⚡ Test 3: Encoding/Decoding Performance");
    
    let start = Instant::now();
    let codes: Result<Vec<Vec<u8>>> = quantizer.batch_encode(&embeddings);
    let encoding_time = start.elapsed();
    
    let codes = codes?;
    println!("✅ Batch encoding: {:.2}ms ({:.0} embeddings/ms)", 
             encoding_time.as_millis(), 
             entity_count as f32 / encoding_time.as_millis() as f32);
    
    let start = Instant::now();
    let reconstructed: Result<Vec<Vec<f32>>> = quantizer.batch_decode(&codes);
    let decoding_time = start.elapsed();
    
    let reconstructed = reconstructed?;
    println!("✅ Batch decoding: {:.2}ms ({:.0} embeddings/ms)", 
             decoding_time.as_millis(), 
             entity_count as f32 / decoding_time.as_millis() as f32);
    
    // === Test 4: Reconstruction Quality ===
    println!("\n🎯 Test 4: Reconstruction Quality Analysis");
    
    let reconstruction_error = quantizer.compute_reconstruction_error(&embeddings)?;
    println!("📊 Average reconstruction error: {:.6}", reconstruction_error);
    
    // Compare a few original vs reconstructed embeddings
    for i in 0..3.min(embeddings.len()) {
        let original = &embeddings[i];
        let reconstructed = &reconstructed[i];
        
        let mse: f32 = original.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>() / original.len() as f32;
        
        println!("  Sample {}: MSE = {:.6}", i + 1, mse);
    }
    
    // === Test 5: Enhanced EmbeddingStore with Quantization ===
    println!("\n🏪 Test 5: Enhanced EmbeddingStore with Auto-Quantization");
    
    let mut store = EmbeddingStore::new_with_quantization(dimension, target_compression)?;
    store.enable_quantization(100); // Auto-quantize after 100 embeddings
    
    let start = Instant::now();
    
    // Add embeddings with auto-quantization
    for (i, embedding) in embeddings.iter().enumerate() {
        let entity_key = EntityKey::from_hash(&format!("entity_{}", i));
        store.add_embedding_key(entity_key, embedding.clone())?;
        
        if i > 0 && i % 200 == 0 {
            println!("  Added {} embeddings...", i);
        }
    }
    
    let insertion_time = start.elapsed();
    println!("✅ Added {} embeddings with auto-quantization in {:.2}ms", 
             entity_count, insertion_time.as_millis());
    
    // === Test 6: Memory Usage Analysis ===
    println!("\n💾 Test 6: Memory Usage Analysis");
    
    let memory_stats = store.memory_stats();
    println!("📊 Memory Statistics:");
    println!("  • Total entities: {}", store.entity_count());
    println!("  • Quantized percentage: {:.1}%", memory_stats.quantized_percentage);
    println!("  • Original memory: {} KB", memory_stats.original_memory / 1024);
    println!("  • Current memory: {} KB", memory_stats.total_memory / 1024);
    println!("  • Memory saved: {} KB ({:.1}%)", 
             (memory_stats.original_memory - memory_stats.total_memory) / 1024,
             (1.0 - memory_stats.total_memory as f32 / memory_stats.original_memory as f32) * 100.0);
    println!("  • Compression ratio: {:.1}:1", memory_stats.compression_ratio);
    
    // === Test 7: Quantized Similarity Search Performance ===
    println!("\n🔍 Test 7: Quantized Similarity Search Performance");
    
    let query = generate_random_embedding(dimension);
    let k = 10;
    
    let start = Instant::now();
    let results = store.similarity_search(&query, k);
    let search_time = start.elapsed();
    
    println!("✅ Similarity search: {:.2}ms (found {} results)", 
             search_time.as_millis(), results.len());
    
    // Show top results
    println!("🏆 Top {} results:", 3.min(results.len()));
    for (i, result) in results.iter().take(3).enumerate() {
        println!("  {}. Entity {:?}: similarity = {:.4}", 
                 i + 1, result.entity, result.similarity);
    }
    
    // === Test 8: Batch Operations Benchmark ===
    println!("\n⚡ Test 8: Batch Operations Benchmark");
    
    let batch_size = 500;
    let batch_embeddings: Vec<Vec<f32>> = (0..batch_size)
        .map(|_| generate_random_embedding(dimension))
        .collect();
    
    let batch_entities: Vec<(EntityKey, Vec<f32>)> = batch_embeddings.iter()
        .enumerate()
        .map(|(i, emb)| (EntityKey::from_hash(&format!("batch_{}", i)), emb.clone()))
        .collect();
    
    let quantizer = store.get_quantizer();
    let quantizer = quantizer.read();
    let start = Instant::now();
    let batch_codes = quantizer.batch_encode(&batch_embeddings)?;
    let batch_encode_time = start.elapsed();
    
    println!("✅ Batch encode {} embeddings: {:.2}ms ({:.0} embeddings/ms)", 
             batch_size, batch_encode_time.as_millis(),
             batch_size as f32 / batch_encode_time.as_millis() as f32);
    
    // === Summary ===
    println!("\n📋 === PHASE 4.1 SUMMARY ===");
    println!("✅ Product Quantization Features Implemented:");
    println!("  • Optimized quantizer creation with target compression");
    println!("  • Adaptive training algorithm for different dataset sizes");
    println!("  • Batch encoding/decoding operations");
    println!("  • Enhanced memory management with auto-quantization");
    println!("  • Quantized similarity search with maintained quality");
    println!("  • Comprehensive memory and compression statistics");
    
    println!("\n📊 Performance Results:");
    println!("  • Compression ratio: {:.1}:1", stats.compression_ratio);
    println!("  • Memory saved: {:.1}%", 
             (1.0 - memory_stats.total_memory as f32 / memory_stats.original_memory as f32) * 100.0);
    println!("  • Encoding speed: {:.0} embeddings/ms", 
             entity_count as f32 / encoding_time.as_millis() as f32);
    println!("  • Search speed: {:.2}ms for {} entities", 
             search_time.as_millis(), store.entity_count());
    println!("  • Reconstruction quality: {:.6} avg error", reconstruction_error);
    
    println!("\n🎯 Phase 4.1 Completed Successfully!");
    println!("Ready for Phase 4.2: Memory-mapped storage integration\n");
    
    Ok(())
}