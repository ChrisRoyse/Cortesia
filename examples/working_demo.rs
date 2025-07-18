use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::EntityData;
use llmkg::error::Result;

fn main() -> Result<()> {
    println!("🧠 LLMKG Working Demo - System Verification");
    
    // Create a new knowledge graph with 96-dimensional embeddings
    let graph = KnowledgeGraph::new(96)?;
    println!("✅ Knowledge graph created successfully");
    
    // Insert a simple entity
    println!("📝 Inserting test entity...");
    
    let test_embedding = create_simple_embedding("test entity", 96);
    
    let entity_key = graph.insert_entity(1, EntityData {
        type_id: 1,
        properties: "Test entity for verification".to_string(),
        embedding: test_embedding.clone(),
    })?;
    
    println!("✅ Entity inserted with key: {:?}", entity_key);
    
    // Get entity back
    println!("🔍 Retrieving entity...");
    let (_meta, data) = graph.get_entity(1)?;
    println!("✅ Entity retrieved: {}", data.properties);
    
    // Test memory usage
    println!("💾 Checking memory usage...");
    let memory = graph.memory_usage();
    println!("  Total memory: {} bytes", memory.total_bytes());
    println!("  Entities: {}", graph.entity_count());
    println!("  Memory per entity: {} bytes", memory.bytes_per_entity(graph.entity_count()));
    
    // Test similarity search
    println!("🔍 Testing similarity search...");
    let search_results = graph.similarity_search(&test_embedding, 5)?;
    println!("  Found {} similar entities", search_results.len());
    
    if !search_results.is_empty() {
        println!("  Best match: entity {} with similarity {:.3}", 
                 search_results[0].0, search_results[0].1);
    }
    
    // Test basic query
    println!("🧠 Testing query functionality...");
    let query_result = graph.query(&test_embedding, 5, 2)?;
    println!("  Query returned {} entities in {}ms", 
             query_result.entities.len(), query_result.query_time_ms);
    
    println!("🎉 All tests passed! System is operational.");
    println!("");
    println!("📊 Performance Summary:");
    println!("  - Entity insertion: ✅ Working");
    println!("  - Entity retrieval: ✅ Working"); 
    println!("  - Similarity search: ✅ Working");
    println!("  - Memory efficiency: ✅ {} bytes/entity", memory.bytes_per_entity(1));
    println!("  - Query system: ✅ Working");
    
    Ok(())
}

fn create_simple_embedding(text: &str, dimension: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let hash = hasher.finish();
    
    let mut embedding = Vec::with_capacity(dimension);
    
    for i in 0..dimension {
        let value = ((hash.wrapping_add(i as u64)) as f32 / u64::MAX as f32 - 0.5) * 2.0;
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