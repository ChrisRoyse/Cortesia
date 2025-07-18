use std::time::Instant;
use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::EntityData;

fn main() {
    println!("🚀 LLMKG Batch Performance Test - Phase 2.2");
    println!("=============================================");
    
    const ENTITY_COUNT: usize = 10_000;
    const EMBEDDING_DIM: usize = 96;
    const BATCH_SIZE: usize = 1_000;
    
    println!("📊 Test Configuration:");
    println!("  - Total entities: {}", ENTITY_COUNT);
    println!("  - Batch size: {}", BATCH_SIZE);
    println!("  - Embedding dimension: {}", EMBEDDING_DIM);
    println!("");
    
    let graph = KnowledgeGraph::new(EMBEDDING_DIM).unwrap();
    
    // Test 1: Individual insertion (baseline)
    println!("🔄 Testing individual insertion (baseline)...");
    let graph_individual = KnowledgeGraph::new(EMBEDDING_DIM).unwrap();
    let start = Instant::now();
    
    for i in 0..1000 { // Smaller set for comparison
        let embedding = create_test_embedding(i, EMBEDDING_DIM);
        graph_individual.insert_entity(i as u32, EntityData {
            type_id: (i % 10) as u16,
            properties: format!("Entity {} - individual test", i),
            embedding,
        }).unwrap();
    }
    
    let individual_time = start.elapsed();
    let individual_rate = 1000.0 / individual_time.as_secs_f64();
    
    println!("✅ Individual insertion:");
    println!("  1000 entities in {:.3}s", individual_time.as_secs_f64());
    println!("  Rate: {:.0} entities/second", individual_rate);
    
    // Test 2: Batch insertion
    println!("\n⚡ Testing batch insertion...");
    
    let mut total_batch_time = 0u128;
    let mut total_entities_inserted = 0;
    
    for batch_start in (0..ENTITY_COUNT).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(ENTITY_COUNT);
        let batch_entities: Vec<(u32, EntityData)> = (batch_start..batch_end)
            .map(|i| {
                let embedding = create_test_embedding(i, EMBEDDING_DIM);
                (i as u32, EntityData {
                    type_id: (i % 10) as u16,
                    properties: format!("Entity {} - batch test", i),
                    embedding,
                })
            })
            .collect();
        
        let start = Instant::now();
        let _keys = graph.insert_entities_batch(batch_entities).unwrap();
        let batch_time = start.elapsed();
        
        total_batch_time += batch_time.as_micros();
        total_entities_inserted += batch_end - batch_start;
        
        println!("  Batch {}: {} entities in {:.3}ms", 
                 (batch_start / BATCH_SIZE) + 1, 
                 batch_end - batch_start, 
                 batch_time.as_millis());
    }
    
    let batch_time_sec = total_batch_time as f64 / 1_000_000.0;
    let batch_rate = total_entities_inserted as f64 / batch_time_sec;
    
    println!("✅ Batch insertion completed:");
    println!("  {} entities in {:.3}s", total_entities_inserted, batch_time_sec);
    println!("  Rate: {:.0} entities/second", batch_rate);
    
    // Test 3: Similarity search performance after batch loading
    println!("\n🔍 Testing similarity search after batch loading...");
    let query_embedding = create_test_embedding(42, EMBEDDING_DIM);
    let mut search_times = Vec::new();
    
    for _ in 0..50 {
        let start = Instant::now();
        let results = graph.similarity_search(&query_embedding, 20).unwrap();
        let search_time = start.elapsed().as_micros() as f64 / 1000.0; // Convert to ms
        search_times.push(search_time);
        
        if results.is_empty() {
            println!("⚠️  Warning: Search returned no results");
        }
    }
    
    let avg_search_time: f64 = search_times.iter().sum::<f64>() / search_times.len() as f64;
    let max_search_time = search_times.iter().fold(0.0f64, |a, &b| a.max(b));
    
    println!("✅ Similarity search performance:");
    println!("  Average: {:.3}ms", avg_search_time);
    println!("  Maximum: {:.3}ms", max_search_time);
    println!("  Target: <1ms - {}", if avg_search_time < 1.0 { "✅ ACHIEVED" } else { "❌ EXCEEDED" });
    
    // Performance comparison
    println!("\n📈 PERFORMANCE COMPARISON");
    println!("========================");
    
    let speedup = batch_rate / individual_rate;
    println!("📊 Insertion Performance:");
    println!("  Individual: {:.0} entities/sec", individual_rate);
    println!("  Batch:      {:.0} entities/sec", batch_rate);
    println!("  Speedup:    {:.1}x faster", speedup);
    
    if speedup > 2.0 {
        println!("🎉 Batch operations provide significant speedup!");
    } else if speedup > 1.5 {
        println!("✅ Batch operations provide good speedup");
    } else {
        println!("⚠️  Batch operations need optimization");
    }
    
    // Memory usage
    let memory = graph.memory_usage();
    let bytes_per_entity = memory.bytes_per_entity(total_entities_inserted);
    
    println!("\n💾 Memory Efficiency:");
    println!("  Total memory: {:.2} MB", memory.total_bytes() as f64 / 1_048_576.0);
    println!("  Per entity: {} bytes", bytes_per_entity);
    
    // Final score
    println!("\n🎯 PHASE 2.2 BATCH OPERATIONS SUMMARY");
    println!("=====================================");
    
    let mut score = 0;
    let total_tests = 3;
    
    if batch_rate > individual_rate * 1.5 {
        println!("✅ Batch Speedup: PASS ({:.1}x faster)", speedup);
        score += 1;
    } else {
        println!("❌ Batch Speedup: FAIL ({:.1}x speedup < 1.5x)", speedup);
    }
    
    if batch_rate > 5000.0 {
        println!("✅ Batch Throughput: PASS ({:.0} entities/sec > 5000)", batch_rate);
        score += 1;
    } else {
        println!("❌ Batch Throughput: FAIL ({:.0} entities/sec ≤ 5000)", batch_rate);
    }
    
    if avg_search_time < 1.0 {
        println!("✅ Search After Batch: PASS ({:.3}ms < 1ms)", avg_search_time);
        score += 1;
    } else {
        println!("❌ Search After Batch: FAIL ({:.3}ms ≥ 1ms)", avg_search_time);
    }
    
    println!("\n📈 OVERALL SCORE: {}/{} ({:.0}%)", 
             score, total_tests, (score as f64 / total_tests as f64) * 100.0);
    
    if score == total_tests {
        println!("🎉 PHASE 2.2 BATCH OPERATIONS - SUCCESS!");
    } else {
        println!("⚠️  Some batch operation targets not met");
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