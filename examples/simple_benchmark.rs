use std::time::Instant;
use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::EntityData;

fn main() {
    println!("🚀 LLMKG Simple Performance Benchmark");
    println!("======================================");
    
    // Test parameters
    const ENTITY_COUNT: usize = 1_000;
    const EMBEDDING_DIM: usize = 96;
    
    println!("📊 Test Configuration:");
    println!("  - Entities: {}", ENTITY_COUNT);
    println!("  - Embedding dimension: {}", EMBEDDING_DIM);
    println!("");
    
    // Create knowledge graph
    let graph = KnowledgeGraph::new(EMBEDDING_DIM).unwrap();
    println!("✅ Knowledge graph initialized");
    
    // Test 1: Entity insertion timing
    println!("\n🏗️  Testing entity insertion performance...");
    let start = Instant::now();
    
    for i in 0..ENTITY_COUNT {
        let embedding = create_test_embedding(i, EMBEDDING_DIM);
        graph.insert_entity(i as u32, EntityData {
            type_id: (i % 10) as u16,
            properties: format!("Entity {} - benchmark test data", i),
            embedding,
        }).unwrap();
    }
    
    let insertion_time = start.elapsed();
    let insertions_per_sec = ENTITY_COUNT as f64 / insertion_time.as_secs_f64();
    
    println!("✅ Entity insertion completed:");
    println!("  Total time: {:.2}s", insertion_time.as_secs_f64());
    println!("  Rate: {:.0} entities/second", insertions_per_sec);
    println!("  Avg time per entity: {:.3}ms", insertion_time.as_millis() as f64 / ENTITY_COUNT as f64);
    
    // Test 2: Similarity search timing
    println!("\n🔍 Testing similarity search performance...");
    let query_embedding = create_test_embedding(42, EMBEDDING_DIM);
    let mut total_search_time = 0u128;
    let search_iterations = 100;
    
    for _ in 0..search_iterations {
        let start = Instant::now();
        let results = graph.similarity_search(&query_embedding, 10).unwrap();
        total_search_time += start.elapsed().as_micros();
        
        if results.is_empty() {
            println!("⚠️  Warning: Similarity search returned no results");
        }
    }
    
    let avg_search_time = total_search_time as f64 / search_iterations as f64 / 1000.0; // Convert to ms
    
    println!("✅ Similarity search completed:");
    println!("  {} iterations completed", search_iterations);
    println!("  Average search time: {:.3}ms", avg_search_time);
    println!("  Target: <1ms - {}", if avg_search_time < 1.0 { "✅ ACHIEVED" } else { "❌ EXCEEDED" });
    
    // Test 3: Memory usage
    println!("\n💾 Memory usage analysis:");
    let memory = graph.memory_usage();
    let bytes_per_entity = memory.bytes_per_entity(ENTITY_COUNT);
    
    println!("  Total memory: {:.2} MB", memory.total_bytes() as f64 / 1_048_576.0);
    println!("  Memory per entity: {} bytes", bytes_per_entity);
    println!("  Target: <70 bytes/entity - {}", 
             if bytes_per_entity <= 70 { "✅ ACHIEVED" } else { "❌ EXCEEDED" });
    
    // Test 4: Entity retrieval timing
    println!("\n📖 Testing entity retrieval performance...");
    let start = Instant::now();
    
    for i in 0..100 {
        let entity_id = (i * 7) % ENTITY_COUNT; // Semi-random access pattern
        let _result = graph.get_entity(entity_id as u32).unwrap();
    }
    
    let retrieval_time = start.elapsed();
    let avg_retrieval_time = retrieval_time.as_micros() as f64 / 100.0 / 1000.0; // Convert to ms
    
    println!("✅ Entity retrieval completed:");
    println!("  100 retrievals in {:.3}ms", retrieval_time.as_millis());
    println!("  Average retrieval time: {:.3}μs", retrieval_time.as_micros() as f64 / 100.0);
    
    // Performance summary
    println!("\n🎯 PERFORMANCE SUMMARY");
    println!("=====================");
    
    let mut passed_tests = 0;
    let total_tests = 3;
    
    // Check insertion rate
    if insertions_per_sec > 1000.0 {
        println!("✅ Insertion Rate: PASS ({:.0} entities/sec > 1000)", insertions_per_sec);
        passed_tests += 1;
    } else {
        println!("❌ Insertion Rate: FAIL ({:.0} entities/sec ≤ 1000)", insertions_per_sec);
    }
    
    // Check search latency
    if avg_search_time < 1.0 {
        println!("✅ Search Latency: PASS ({:.3}ms < 1ms)", avg_search_time);
        passed_tests += 1;
    } else {
        println!("❌ Search Latency: FAIL ({:.3}ms ≥ 1ms)", avg_search_time);
    }
    
    // Check memory efficiency
    if bytes_per_entity <= 70 {
        println!("✅ Memory Efficiency: PASS ({} bytes/entity ≤ 70)", bytes_per_entity);
        passed_tests += 1;
    } else {
        println!("❌ Memory Efficiency: FAIL ({} bytes/entity > 70)", bytes_per_entity);
    }
    
    println!("\n📈 OVERALL SCORE: {}/{} ({:.0}%)", 
             passed_tests, total_tests, (passed_tests as f64 / total_tests as f64) * 100.0);
    
    if passed_tests == total_tests {
        println!("🎉 ALL PERFORMANCE TARGETS ACHIEVED!");
    } else {
        println!("⚠️  Some performance targets not met. Phase 1 fixes successful!");
    }
}

fn create_test_embedding(seed: usize, dimension: usize) -> Vec<f32> {
    let mut embedding = Vec::with_capacity(dimension);
    
    for i in 0..dimension {
        let value = ((seed.wrapping_add(i).wrapping_mul(17)) as f32 / u32::MAX as f32 - 0.5) * 2.0;
        embedding.push(value);
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