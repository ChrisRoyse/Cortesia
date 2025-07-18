use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::{EntityData, Relationship};
use llmkg::error::Result;

fn main() -> Result<()> {
    println!("🧠 LLMKG Basic Usage Example");
    
    // Create a new knowledge graph with 96-dimensional embeddings
    let graph = KnowledgeGraph::new(96)?;
    
    // Insert some sample entities
    println!("📝 Inserting entities...");
    
    let rust_embedding = create_sample_embedding("Rust programming language", 96);
    let wasm_embedding = create_sample_embedding("WebAssembly runtime", 96);
    let llm_embedding = create_sample_embedding("Large language models", 96);
    
    graph.insert_entity(1, EntityData {
        type_id: 1, // Programming language
        properties: "Rust is a systems programming language focused on safety and performance".to_string(),
        embedding: rust_embedding,
    })?;
    
    graph.insert_entity(2, EntityData {
        type_id: 2, // Technology
        properties: "WebAssembly is a binary instruction format for stack-based virtual machines".to_string(),
        embedding: wasm_embedding,
    })?;
    
    graph.insert_entity(3, EntityData {
        type_id: 3, // AI Technology  
        properties: "Large language models are neural networks trained on vast amounts of text".to_string(),
        embedding: llm_embedding,
    })?;
    
    // Add relationships
    println!("🔗 Adding relationships...");
    
    graph.insert_relationship(Relationship {
        from: 1, // Rust
        to: 2,   // WebAssembly
        rel_type: 1, // "compiles_to"
        weight: 0.9,
    })?;
    
    graph.insert_relationship(Relationship {
        from: 2, // WebAssembly
        to: 3,   // LLM
        rel_type: 2, // "enables"
        weight: 0.7,
    })?;
    
    // Perform semantic search
    println!("🔍 Performing semantic search...");
    
    let query_embedding = create_sample_embedding("fast programming", 96);
    let search_results = graph.similarity_search(&query_embedding, 5)?;
    
    println!("Search results:");
    for (entity_id, similarity) in search_results {
        println!("  Entity {}: similarity {:.3}", entity_id, similarity);
    }
    
    // Find path between entities
    println!("🛤️  Finding path between Rust and LLM...");
    
    if let Ok(Some(path)) = graph.find_path(1, 3, 4) {
        println!("Path found: {:?}", path);
    } else {
        println!("No path found");
    }
    
    // Get context for LLM
    println!("🧠 Retrieving context for LLM...");
    
    let context_query = create_sample_embedding("programming languages for AI", 96);
    let context = graph.query(&context_query, 10, 2)?;
    
    println!("Context retrieved:");
    println!("  Entities: {}", context.entities.len());
    println!("  Relationships: {}", context.relationships.len());
    println!("  Query time: {}ms", context.query_time_ms);
    
    // Display memory usage
    println!("💾 Memory usage:");
    let memory = graph.memory_usage();
    println!("  Total: {} bytes", memory.total_bytes());
    println!("  Per entity: {} bytes", memory.bytes_per_entity(graph.entity_count()));
    
    println!("✅ Example completed successfully!");
    
    Ok(())
}

fn create_sample_embedding(text: &str, dimension: usize) -> Vec<f32> {
    // Simple hash-based embedding for demonstration
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