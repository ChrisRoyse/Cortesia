// Phase 4.4: Zero-Copy Serialization Performance Demonstration
// Comprehensive demo showcasing the performance benefits of zero-copy serialization

use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::core::zero_copy_engine::{ZeroCopyKnowledgeEngine, BenchmarkResult};
use llmkg::core::types::EntityData;
use llmkg::storage::zero_copy::{ZeroCopySerializer, ZeroCopyDeserializer, ZeroCopyMetrics};
use std::sync::Arc;
use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 LLMKG Zero-Copy Serialization Performance Demo");
    println!("===================================================");
    
    // Initialize with deterministic randomness for reproducible results
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let embedding_dim = 96;
    
    // Test different dataset sizes to show scalability
    let test_sizes = vec![100, 1_000, 10_000, 50_000];
    
    for &size in &test_sizes {
        println!("\n📊 Testing with {} entities", size);
        println!("─".repeat(50));
        
        let results = run_zero_copy_benchmark(size, embedding_dim, &mut rng).await?;
        display_results(&results, size);
    }
    
    // Memory efficiency comparison
    println!("\n💾 Memory Efficiency Analysis");
    println!("═".repeat(50));
    run_memory_efficiency_test(10_000, embedding_dim, &mut rng).await?;
    
    // Real-world scenario simulation
    println!("\n🌍 Real-World Scenario Simulation");
    println!("═".repeat(50));
    run_real_world_simulation(&mut rng).await?;
    
    Ok(())
}

async fn run_zero_copy_benchmark(
    entity_count: usize,
    embedding_dim: usize,
    rng: &mut ChaCha8Rng,
) -> Result<ZeroCopyBenchmarkResults, Box<dyn std::error::Error>> {
    // Create knowledge engine
    let base_engine = Arc::new(KnowledgeEngine::new(embedding_dim)?);
    let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine.clone(), embedding_dim);
    
    // Generate test data
    println!("  🔄 Generating {} test entities...", entity_count);
    let start = Instant::now();
    
    for i in 0..entity_count {
        let entity = generate_realistic_entity(i as u32, embedding_dim, rng);
        base_engine.insert_entity(i as u32, entity).await?;
        
        if i % (entity_count / 10).max(1) == 0 {
            let progress = (i as f64 / entity_count as f64) * 100.0;
            print!("\r  📈 Progress: {:.1}%", progress);
        }
    }
    let generation_time = start.elapsed();
    println!("\r  ✅ Data generation completed in {:.2}ms", generation_time.as_millis());
    
    // Benchmark serialization
    println!("  🔄 Serializing to zero-copy format...");
    let start = Instant::now();
    let serialized_data = zero_copy_engine.serialize_to_zero_copy().await?;
    let serialization_time = start.elapsed();
    println!("  ✅ Serialization completed in {:.2}ms", serialization_time.as_millis());
    
    // Benchmark deserialization
    println!("  🔄 Loading zero-copy data...");
    let start = Instant::now();
    zero_copy_engine.load_zero_copy_data(serialized_data.clone())?;
    let deserialization_time = start.elapsed();
    println!("  ✅ Deserialization completed in {:.2}μs", deserialization_time.as_micros());
    
    // Benchmark query performance
    println!("  🔄 Benchmarking query performance...");
    let query_embedding: Vec<f32> = (0..embedding_dim).map(|i| rng.gen_range(-1.0..1.0)).collect();
    let benchmark_result = zero_copy_engine.benchmark_against_standard(&query_embedding, 100).await?;
    
    // Get metrics
    let metrics = zero_copy_engine.get_metrics();
    
    Ok(ZeroCopyBenchmarkResults {
        entity_count,
        generation_time,
        serialization_time,
        deserialization_time,
        serialized_size: serialized_data.len(),
        benchmark_result,
        metrics,
    })
}

fn display_results(results: &ZeroCopyBenchmarkResults, entity_count: usize) {
    println!("  📈 Performance Results:");
    println!("    • Entities: {}", entity_count);
    println!("    • Serialization: {:.2}ms ({:.0} entities/ms)", 
        results.serialization_time.as_millis(), 
        entity_count as f64 / results.serialization_time.as_millis() as f64);
    println!("    • Deserialization: {:.2}μs ({:.0} entities/μs)", 
        results.deserialization_time.as_micros(),
        entity_count as f64 / results.deserialization_time.as_micros() as f64);
    println!("    • Serialized size: {:.2} MB ({:.1} bytes/entity)", 
        results.serialized_size as f64 / (1024.0 * 1024.0),
        results.serialized_size as f64 / entity_count as f64);
    println!("    • Query speedup: {:.2}x", results.benchmark_result.speedup);
    println!("    • Zero-copy queries: {:.0} ops/sec", results.benchmark_result.zero_copy_ops_per_sec());
    println!("    • Standard queries: {:.0} ops/sec", results.benchmark_result.standard_ops_per_sec());
    
    // Performance ratings
    let serialization_rate = entity_count as f64 / results.serialization_time.as_millis() as f64;
    let query_speedup = results.benchmark_result.speedup;
    
    if serialization_rate > 1000.0 && query_speedup > 5.0 {
        println!("    🏆 Performance Rating: EXCELLENT");
    } else if serialization_rate > 500.0 && query_speedup > 3.0 {
        println!("    🥈 Performance Rating: VERY GOOD");
    } else if serialization_rate > 100.0 && query_speedup > 2.0 {
        println!("    🥉 Performance Rating: GOOD");
    } else {
        println!("    ⚠️  Performance Rating: NEEDS OPTIMIZATION");
    }
}

async fn run_memory_efficiency_test(
    entity_count: usize,
    embedding_dim: usize,
    rng: &mut ChaCha8Rng,
) -> Result<(), Box<dyn std::error::Error>> {
    let base_engine = Arc::new(KnowledgeEngine::new(embedding_dim)?);
    let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine.clone(), embedding_dim);
    
    // Generate entities with varying property sizes
    println!("  🔄 Generating entities with varying property sizes...");
    let mut total_property_size = 0usize;
    let mut total_embedding_size = 0usize;
    
    for i in 0..entity_count {
        let property_size = rng.gen_range(50..500); // 50-500 chars
        let properties = generate_random_properties(property_size, rng);
        total_property_size += properties.len();
        total_embedding_size += embedding_dim * 4; // f32 = 4 bytes
        
        let entity = EntityData {
            type_id: (i % 100) as u16,
            properties,
            embedding: (0..embedding_dim).map(|_| rng.gen_range(-1.0..1.0)).collect(),
        };
        
        base_engine.insert_entity(i as u32, entity).await?;
    }
    
    // Serialize with zero-copy
    let serialized_data = zero_copy_engine.serialize_to_zero_copy().await?;
    zero_copy_engine.load_zero_copy_data(serialized_data.clone())?;
    
    // Calculate memory efficiency
    let raw_data_size = total_property_size + total_embedding_size + (entity_count * 16); // metadata overhead
    let zero_copy_size = zero_copy_engine.zero_copy_memory_usage();
    let compression_ratio = raw_data_size as f64 / zero_copy_size as f64;
    
    println!("  📊 Memory Efficiency Results:");
    println!("    • Raw data size: {:.2} MB", raw_data_size as f64 / (1024.0 * 1024.0));
    println!("    • Zero-copy size: {:.2} MB", zero_copy_size as f64 / (1024.0 * 1024.0));
    println!("    • Compression ratio: {:.2}:1", compression_ratio);
    println!("    • Memory savings: {:.1}%", (1.0 - zero_copy_size as f64 / raw_data_size as f64) * 100.0);
    println!("    • Bytes per entity: {:.1}", zero_copy_size as f64 / entity_count as f64);
    
    // Performance rating for memory efficiency
    if compression_ratio > 5.0 {
        println!("    🏆 Memory Efficiency: EXCELLENT");
    } else if compression_ratio > 3.0 {
        println!("    🥈 Memory Efficiency: VERY GOOD");
    } else if compression_ratio > 2.0 {
        println!("    🥉 Memory Efficiency: GOOD");
    } else {
        println!("    ⚠️  Memory Efficiency: NEEDS IMPROVEMENT");
    }
    
    Ok(())
}

async fn run_real_world_simulation(rng: &mut ChaCha8Rng) -> Result<(), Box<dyn std::error::Error>> {
    let embedding_dim = 384; // Common for modern embedding models
    let base_engine = Arc::new(KnowledgeEngine::new(embedding_dim)?);
    let zero_copy_engine = ZeroCopyKnowledgeEngine::new(base_engine.clone(), embedding_dim);
    
    println!("  🌐 Simulating real-world knowledge graph scenario:");
    println!("    • Entity types: People, Organizations, Locations, Concepts");
    println!("    • Embedding dimension: {}", embedding_dim);
    println!("    • Realistic property distributions");
    
    // Generate different entity types with realistic distributions
    let entity_types = vec![
        ("Person", 40, 100..300),      // 40% of entities, 100-300 chars properties
        ("Organization", 25, 80..250), // 25% of entities
        ("Location", 20, 50..150),     // 20% of entities
        ("Concept", 15, 200..500),     // 15% of entities
    ];
    
    let total_entities = 25_000;
    let mut entity_id = 0u32;
    
    for (entity_type, percentage, property_range) in entity_types {
        let count = (total_entities * percentage / 100).max(1);
        println!("  🔄 Creating {} {} entities...", count, entity_type);
        
        for _ in 0..count {
            let property_size = rng.gen_range(property_range.clone());
            let properties = format!(
                "{{\"type\":\"{}\",\"data\":\"{}\"}}",
                entity_type,
                generate_random_properties(property_size - entity_type.len() - 20, rng)
            );
            
            // Generate realistic embeddings (somewhat clustered by type)
            let base_value = match entity_type {
                "Person" => 0.2,
                "Organization" => 0.4,
                "Location" => 0.6,
                "Concept" => 0.8,
                _ => 0.0,
            };
            
            let embedding: Vec<f32> = (0..embedding_dim)
                .map(|_| base_value + rng.gen_range(-0.3..0.3))
                .collect();
            
            let entity = EntityData {
                type_id: match entity_type {
                    "Person" => 1,
                    "Organization" => 2,
                    "Location" => 3,
                    "Concept" => 4,
                    _ => 0,
                },
                properties,
                embedding,
            };
            
            base_engine.insert_entity(entity_id, entity).await?;
            entity_id += 1;
        }
    }
    
    // Run comprehensive performance test
    println!("  🔄 Running comprehensive performance analysis...");
    
    let start = Instant::now();
    let serialized_data = zero_copy_engine.serialize_to_zero_copy().await?;
    let serialization_time = start.elapsed();
    
    let start = Instant::now();
    zero_copy_engine.load_zero_copy_data(serialized_data.clone())?;
    let deserialization_time = start.elapsed();
    
    // Test multiple query patterns
    let query_patterns = vec![
        ("Person-like", vec![0.2; embedding_dim]),
        ("Organization-like", vec![0.4; embedding_dim]),
        ("Location-like", vec![0.6; embedding_dim]),
        ("Concept-like", vec![0.8; embedding_dim]),
        ("Mixed", (0..embedding_dim).map(|i| (i as f32 / embedding_dim as f32) * 2.0 - 1.0).collect()),
    ];
    
    println!("  📊 Real-World Performance Results:");
    println!("    • Total entities: {}", entity_id);
    println!("    • Serialization: {:.2}ms ({:.0} entities/ms)", 
        serialization_time.as_millis(), 
        entity_id as f64 / serialization_time.as_millis() as f64);
    println!("    • Deserialization: {:.2}μs", deserialization_time.as_micros());
    println!("    • Data size: {:.2} MB", serialized_data.len() as f64 / (1024.0 * 1024.0));
    println!("    • Bytes per entity: {:.1}", serialized_data.len() as f64 / entity_id as f64);
    
    for (pattern_name, query_embedding) in query_patterns {
        let start = Instant::now();
        let results = zero_copy_engine.similarity_search_zero_copy(&query_embedding, 100)?;
        let query_time = start.elapsed();
        
        println!("    • {} query: {:.2}ms, {} results", 
            pattern_name, query_time.as_millis(), results.len());
    }
    
    let metrics = zero_copy_engine.get_metrics();
    println!("    • Compression ratio: {:.2}:1", metrics.compression_ratio);
    println!("    • Memory efficiency: {:.1} bytes/entity", metrics.memory_efficiency_bytes_per_entity());
    
    println!("    🎯 Real-world simulation completed successfully!");
    
    Ok(())
}

fn generate_realistic_entity(id: u32, embedding_dim: usize, rng: &mut ChaCha8Rng) -> EntityData {
    let property_types = vec![
        "name", "description", "category", "tags", "metadata", "attributes"
    ];
    
    let mut properties = format!("{{\"id\":{}", id);
    for prop_type in &property_types {
        if rng.gen_bool(0.7) { // 70% chance to include each property
            let value_length = rng.gen_range(10..100);
            let value = generate_random_text(value_length, rng);
            properties.push_str(&format!(",\"{}\":\"{}\"", prop_type, value));
        }
    }
    properties.push('}');
    
    let embedding: Vec<f32> = (0..embedding_dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    EntityData {
        type_id: rng.gen_range(1..10),
        properties,
        embedding,
    }
}

fn generate_random_properties(length: usize, rng: &mut ChaCha8Rng) -> String {
    generate_random_text(length, rng)
}

fn generate_random_text(length: usize, rng: &mut ChaCha8Rng) -> String {
    const CHARS: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-!?";
    (0..length)
        .map(|_| CHARS[rng.gen_range(0..CHARS.len())] as char)
        .collect()
}

#[derive(Debug)]
struct ZeroCopyBenchmarkResults {
    entity_count: usize,
    generation_time: std::time::Duration,
    serialization_time: std::time::Duration,
    deserialization_time: std::time::Duration,
    serialized_size: usize,
    benchmark_result: BenchmarkResult,
    metrics: ZeroCopyMetrics,
}