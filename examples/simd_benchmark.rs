use std::time::Instant;
use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::EntityData;
use llmkg::embedding::similarity::{cosine_similarity, cosine_similarity_scalar};

fn main() {
    println!("🚀 LLMKG SIMD Performance Benchmark - Phase 3.1");
    println!("===============================================");
    
    const ENTITY_COUNT: usize = 5_000;
    const EMBEDDING_DIM: usize = 96;
    const BENCHMARK_ITERATIONS: usize = 100;
    
    println!("📊 Test Configuration:");
    println!("  - Entities: {}", ENTITY_COUNT);
    println!("  - Embedding dimension: {}", EMBEDDING_DIM);
    println!("  - Benchmark iterations: {}", BENCHMARK_ITERATIONS);
    println!("");
    
    // Create test data
    println!("🔧 Generating test data...");
    let entities: Vec<(u32, EntityData)> = (0..ENTITY_COUNT)
        .map(|i| {
            let embedding = create_test_embedding(i, EMBEDDING_DIM);
            (i as u32, EntityData {
                type_id: (i % 10) as u16,
                properties: format!("Entity {} - SIMD test", i),
                embedding,
            })
        })
        .collect();
    
    let query_embedding = create_test_embedding(42, EMBEDDING_DIM);
    
    // Test 1: Scalar vs SIMD single similarity computation
    println!("🧮 Testing single similarity computation...");
    
    let test_embedding = &entities[100].1.embedding;
    
    // Warm up
    for _ in 0..10 {
        let _ = cosine_similarity_scalar(&query_embedding, test_embedding);
        let _ = cosine_similarity(&query_embedding, test_embedding);
    }
    
    // Benchmark scalar implementation
    let start = Instant::now();
    let mut scalar_result = 0.0f32;
    for _ in 0..BENCHMARK_ITERATIONS * 1000 {
        scalar_result += cosine_similarity_scalar(&query_embedding, test_embedding);
    }
    let scalar_time = start.elapsed();
    
    // Benchmark SIMD implementation
    let start = Instant::now();
    let mut simd_result = 0.0f32;
    for _ in 0..BENCHMARK_ITERATIONS * 1000 {
        simd_result += cosine_similarity(&query_embedding, test_embedding);
    }
    let simd_time = start.elapsed();
    
    println!("✅ Single similarity computation:");
    println!("  Scalar: {:.3}μs per operation", scalar_time.as_micros() as f64 / (BENCHMARK_ITERATIONS * 1000) as f64);
    println!("  SIMD:   {:.3}μs per operation", simd_time.as_micros() as f64 / (BENCHMARK_ITERATIONS * 1000) as f64);
    
    let speedup = scalar_time.as_micros() as f64 / simd_time.as_micros() as f64;
    println!("  Speedup: {:.2}x faster", speedup);
    
    // Verify results are close
    let relative_diff = ((scalar_result - simd_result) / scalar_result).abs();
    println!("  Accuracy: {:.6} relative difference", relative_diff);
    
    // Test 2: Full knowledge graph with SIMD-accelerated search
    println!("\n⚡ Testing full knowledge graph search performance...");
    
    let graph = KnowledgeGraph::new(EMBEDDING_DIM).unwrap();
    
    // Insert entities using batch operations
    let start = Instant::now();
    let _keys = graph.insert_entities_batch(entities).unwrap();
    let insertion_time = start.elapsed();
    
    println!("✅ Entity insertion: {:.3}s", insertion_time.as_secs_f64());
    
    // Benchmark similarity search with SIMD acceleration
    println!("\n🔍 Benchmarking similarity search...");
    
    let mut search_times = Vec::new();
    let mut result_counts = Vec::new();
    
    // Warm up the cache
    for _ in 0..5 {
        let _ = graph.similarity_search(&query_embedding, 20).unwrap();
    }
    
    for i in 0..BENCHMARK_ITERATIONS {
        let query = create_test_embedding(i * 13, EMBEDDING_DIM); // Vary query slightly
        
        let start = Instant::now();
        let results = graph.similarity_search(&query, 20).unwrap();
        let search_time = start.elapsed();
        
        search_times.push(search_time.as_micros() as f64 / 1000.0); // Convert to ms
        result_counts.push(results.len());
        
        if i < 5 && !results.is_empty() {
            println!("  Query {}: {} results, {:.3}ms", i + 1, results.len(), search_time.as_micros() as f64 / 1000.0);
        }
    }
    
    let avg_search_time: f64 = search_times.iter().sum::<f64>() / search_times.len() as f64;
    let min_search_time = search_times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_search_time = search_times.iter().fold(0.0f64, |a, &b| a.max(b));
    let avg_results: f64 = result_counts.iter().sum::<usize>() as f64 / result_counts.len() as f64;
    
    println!("✅ Search performance analysis:");
    println!("  Average: {:.3}ms", avg_search_time);
    println!("  Minimum: {:.3}ms", min_search_time);
    println!("  Maximum: {:.3}ms", max_search_time);
    println!("  Avg results: {:.1}", avg_results);
    
    // Test 3: CPU feature detection
    println!("\n🔧 CPU Feature Detection:");
    #[cfg(target_arch = "x86_64")]
    {
        println!("  AVX2:    {}", is_x86_feature_detected!("avx2"));
        println!("  SSE4.1:  {}", is_x86_feature_detected!("sse4.1"));
        println!("  FMA:     {}", is_x86_feature_detected!("fma"));
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("  Platform: Non-x86_64 (SIMD optimizations not available)");
    }
    
    // Memory usage analysis
    let memory = graph.memory_usage();
    let bytes_per_entity = memory.bytes_per_entity(ENTITY_COUNT);
    
    println!("\n💾 Memory Analysis:");
    println!("  Total memory: {:.2} MB", memory.total_bytes() as f64 / 1_048_576.0);
    println!("  Per entity: {} bytes", bytes_per_entity);
    
    // Performance summary
    println!("\n🎯 PHASE 3.1 SIMD ACCELERATION SUMMARY");
    println!("=====================================");
    
    let mut score = 0;
    let total_tests = 4;
    
    // Test 1: SIMD speedup
    if speedup > 1.5 {
        println!("✅ SIMD Speedup: PASS ({:.2}x faster)", speedup);
        score += 1;
    } else {
        println!("❌ SIMD Speedup: FAIL ({:.2}x speedup < 1.5x)", speedup);
    }
    
    // Test 2: Accuracy
    if relative_diff < 0.001 {
        println!("✅ SIMD Accuracy: PASS ({:.6} difference < 0.001)", relative_diff);
        score += 1;
    } else {
        println!("❌ SIMD Accuracy: FAIL ({:.6} difference ≥ 0.001)", relative_diff);
    }
    
    // Test 3: Search performance target
    if avg_search_time < 1.0 {
        println!("✅ Search Performance: PASS ({:.3}ms < 1ms)", avg_search_time);
        score += 1;
    } else {
        println!("❌ Search Performance: FAIL ({:.3}ms ≥ 1ms)", avg_search_time);
    }
    
    // Test 4: Memory efficiency (still targeting <70 bytes/entity in later phases)
    if bytes_per_entity < 400 {
        println!("✅ Memory Efficiency: PASS ({} bytes/entity < 400)", bytes_per_entity);
        score += 1;
    } else {
        println!("❌ Memory Efficiency: FAIL ({} bytes/entity ≥ 400)", bytes_per_entity);
    }
    
    println!("\n📈 OVERALL SCORE: {}/{} ({:.0}%)", 
             score, total_tests, (score as f64 / total_tests as f64) * 100.0);
    
    if score >= 3 {
        println!("🎉 PHASE 3.1 SIMD ACCELERATION - SUCCESS!");
        println!("Ready to proceed to Phase 3.2 (Advanced Indexing)");
    } else {
        println!("⚠️  Some SIMD optimization targets not met");
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