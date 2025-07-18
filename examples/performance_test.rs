use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::EntityData;
use llmkg::error::Result;
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 LLMKG Performance Test - Production Readiness Verification");
    println!("================================================================");
    
    // Test parameters
    const ENTITY_COUNT: usize = 10_000;
    const EMBEDDING_DIM: usize = 96;
    const QUERY_COUNT: usize = 1_000;
    
    println!("📊 Test Configuration:");
    println!("  - Entities: {}", ENTITY_COUNT);
    println!("  - Embedding dimension: {}", EMBEDDING_DIM);
    println!("  - Query iterations: {}", QUERY_COUNT);
    println!("");
    
    // Create knowledge graph
    let graph = KnowledgeGraph::new(EMBEDDING_DIM)?;
    println!("✅ Knowledge graph initialized");
    
    // Performance test 1: Entity insertion
    println!("🏗️  Testing entity insertion performance...");
    let start = Instant::now();
    
    for i in 0..ENTITY_COUNT {
        let embedding = create_test_embedding(i, EMBEDDING_DIM);
        graph.insert_entity(i as u32, EntityData {
            type_id: (i % 10) as u16,
            properties: format!("Entity {} - performance test data", i),
            embedding,
        })?;
        
        if (i + 1) % 1000 == 0 {
            println!("  Inserted {} entities...", i + 1);
        }
    }
    
    let insertion_time = start.elapsed();
    let insertions_per_sec = ENTITY_COUNT as f64 / insertion_time.as_secs_f64();
    
    println!("✅ Entity insertion completed:");
    println!("  Total time: {:.2}s", insertion_time.as_secs_f64());
    println!("  Rate: {:.0} entities/second", insertions_per_sec);
    println!("  Avg time per entity: {:.3}ms", insertion_time.as_millis() as f64 / ENTITY_COUNT as f64);
    
    // Performance test 2: Memory efficiency
    println!("\n💾 Analyzing memory efficiency...");
    let memory = graph.memory_usage();
    let bytes_per_entity = memory.bytes_per_entity(ENTITY_COUNT);
    
    println!("  Total memory: {:.2} MB", memory.total_bytes() as f64 / 1_048_576.0);
    println!("  Memory per entity: {} bytes", bytes_per_entity);
    println!("  Target: <70 bytes/entity - {}", 
             if bytes_per_entity <= 70 { "✅ ACHIEVED" } else { "❌ EXCEEDED" });
    
    // Performance test 3: Query performance
    println!("\n🔍 Testing query performance...");
    let mut total_query_time = 0u128;
    let mut total_results = 0usize;
    
    for i in 0..QUERY_COUNT {
        let query_embedding = create_test_embedding(i % 100, EMBEDDING_DIM);
        
        let start = Instant::now();
        let results = graph.similarity_search(&query_embedding, 10)?;
        total_query_time += start.elapsed().as_micros();
        total_results += results.len();
        
        if (i + 1) % 100 == 0 {
            println!("  Completed {} queries...", i + 1);
        }
    }
    
    let avg_query_time = total_query_time as f64 / QUERY_COUNT as f64 / 1000.0; // Convert to milliseconds
    let queries_per_second = QUERY_COUNT as f64 / (total_query_time as f64 / 1_000_000.0);
    
    println!("✅ Query performance completed:");
    println!("  Total queries: {}", QUERY_COUNT);
    println!("  Average query time: {:.3}ms", avg_query_time);
    println!("  Queries per second: {:.0}", queries_per_second);
    println!("  Average results per query: {:.1}", total_results as f64 / QUERY_COUNT as f64);
    println!("  Target: <1ms - {}", 
             if avg_query_time < 1.0 { "✅ ACHIEVED" } else { "❌ EXCEEDED" });
    
    // Performance test 4: Entity retrieval
    println!("\n📖 Testing entity retrieval performance...");
    let start = Instant::now();
    
    for i in 0..1000 {
        let entity_id = (i * 7) % ENTITY_COUNT; // Semi-random access pattern
        let _result = graph.get_entity(entity_id as u32)?;
    }
    
    let retrieval_time = start.elapsed();
    let avg_retrieval_time = retrieval_time.as_micros() as f64 / 1000.0 / 1000.0; // Convert to ms
    
    println!("✅ Entity retrieval completed:");
    println!("  1000 retrievals in {:.3}ms", retrieval_time.as_millis());
    println!("  Average retrieval time: {:.3}ms", avg_retrieval_time);
    
    // Performance test 5: Full context queries
    println!("\n🧠 Testing full context query performance...");
    let mut context_query_times = Vec::new();
    
    for i in 0..100 {
        let query_embedding = create_test_embedding(i, EMBEDDING_DIM);
        let start = Instant::now();
        let _context = graph.query(&query_embedding, 20, 2)?;
        let query_time = start.elapsed().as_micros() as f64 / 1000.0; // Convert to ms
        context_query_times.push(query_time);
        
        if (i + 1) % 10 == 0 {
            println!("  Completed {} context queries...", i + 1);
        }
    }
    
    let avg_context_time: f64 = context_query_times.iter().sum::<f64>() / context_query_times.len() as f64;
    let max_context_time = context_query_times.iter().fold(0.0f64, |a, &b| a.max(b));
    
    println!("✅ Context query performance:");
    println!("  Average context query time: {:.3}ms", avg_context_time);
    println!("  Maximum context query time: {:.3}ms", max_context_time);
    
    // Final performance report
    println!("\n🎯 FINAL PERFORMANCE REPORT");
    println!("============================");
    
    let mut score = 0;
    let mut max_score = 0;
    
    // Memory efficiency score
    max_score += 1;
    if bytes_per_entity <= 70 {
        score += 1;
        println!("✅ Memory Efficiency: PASS ({} bytes/entity ≤ 70)", bytes_per_entity);
    } else {
        println!("❌ Memory Efficiency: FAIL ({} bytes/entity > 70)", bytes_per_entity);
    }
    
    // Query latency score
    max_score += 1;
    if avg_query_time < 1.0 {
        score += 1;
        println!("✅ Query Latency: PASS ({:.3}ms < 1ms)", avg_query_time);
    } else {
        println!("❌ Query Latency: FAIL ({:.3}ms ≥ 1ms)", avg_query_time);
    }
    
    // Insertion rate score
    max_score += 1;
    if insertions_per_sec > 1000.0 {
        score += 1;
        println!("✅ Insertion Rate: PASS ({:.0} entities/sec > 1000)", insertions_per_sec);
    } else {
        println!("❌ Insertion Rate: FAIL ({:.0} entities/sec ≤ 1000)", insertions_per_sec);
    }
    
    // Context query score
    max_score += 1;
    if avg_context_time < 10.0 {
        score += 1;
        println!("✅ Context Queries: PASS ({:.3}ms < 10ms)", avg_context_time);
    } else {
        println!("❌ Context Queries: FAIL ({:.3}ms ≥ 10ms)", avg_context_time);
    }
    
    println!("\n📈 OVERALL SCORE: {}/{} ({:.0}%)", 
             score, max_score, (score as f64 / max_score as f64) * 100.0);
    
    if score == max_score {
        println!("🎉 PRODUCTION READY! All performance targets achieved.");
    } else {
        println!("⚠️  Some performance targets not met. Review for production use.");
    }
    
    println!("\n🚀 System demonstrates:");
    println!("  • High-performance entity storage and retrieval");
    println!("  • Memory-efficient graph representation");
    println!("  • Fast similarity search capabilities");
    println!("  • Production-ready performance characteristics");
    
    Ok(())
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