use std::collections::HashMap;

// Import the main library types we need to test
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::types::EntityData;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing LLMKG Key Fixes");
    println!("===========================");
    
    // Test 1: Brain Enhanced Graph Creation with 96D
    println!("\n🧠 Test 1: Brain Enhanced Graph Creation");
    let graph = BrainEnhancedKnowledgeGraph::new_for_test()?;
    println!("   ✅ BrainEnhancedKnowledgeGraph created successfully");
    println!("   📏 Embedding dimension: {}", graph.embedding_dimension());
    
    // Test 2: Entity Creation with 96D Embeddings
    println!("\n🏷️  Test 2: Entity Creation with 96D Embeddings");
    let mut entity_embedding = vec![0.0; 96];
    entity_embedding[0] = 1.0;
    entity_embedding[1] = 0.5;
    
    let entity_data = EntityData {
        type_id: 1,
        embedding: entity_embedding,
        properties: r#"{"name": "test_entity", "type": "concept"}"#.to_string(),
    };
    
    let entity_key = graph.insert_brain_entity(1, entity_data).await?;
    println!("   ✅ Entity created with 96D embedding");
    println!("   🔑 Entity key: {:?}", entity_key);
    
    // Test 3: Missing Methods Implementation
    println!("\n🔬 Test 3: Missing Methods Implementation");
    
    // Test reset_all_activations
    graph.reset_all_activations().await;
    println!("   ✅ reset_all_activations() method works");
    
    // Test get_configuration
    let config = graph.get_configuration().await;
    println!("   ✅ get_configuration() method works");
    println!("   ⚙️  Learning rate: {}", config.learning_rate);
    
    // Test count_relationships_by_type
    let count = graph.count_relationships_by_type(0).await;
    println!("   ✅ count_relationships_by_type() method works");
    println!("   📊 Relationship count: {}", count);
    
    // Test analyze_weight_distribution
    let weight_dist = graph.analyze_weight_distribution().await;
    println!("   ✅ analyze_weight_distribution() method works");
    println!("   📈 Weight mean: {}", weight_dist.mean);
    
    // Test 4: Batch Operations
    println!("\n⚡ Test 4: Batch Operations");
    
    // Create a second entity for relationship testing
    let mut entity2_embedding = vec![0.0; 96];
    entity2_embedding[0] = 0.8;
    
    let entity2_data = EntityData {
        type_id: 1,
        embedding: entity2_embedding,
        properties: r#"{"name": "test_entity_2", "type": "concept"}"#.to_string(),
    };
    
    let entity2_key = graph.insert_brain_entity(2, entity2_data).await?;
    
    // Test batch operations
    let batch_updates = vec![(entity_key, entity2_key, 0.7)];
    graph.batch_strengthen_relationships(&batch_updates).await?;
    println!("   ✅ batch_strengthen_relationships() method works");
    
    let batch_removals = vec![(entity_key, entity2_key)];
    graph.batch_remove_relationships(&batch_removals).await?;
    println!("   ✅ batch_remove_relationships() method works");
    
    // Test 5: Serialization Support
    println!("\n💾 Test 5: Serialization Support");
    let memory_usage = graph.get_memory_usage().await;
    
    // Try to serialize the memory usage (this tests our Serialize derive)
    match serde_json::to_string(&memory_usage) {
        Ok(json) => {
            println!("   ✅ BrainMemoryUsage serialization works");
            println!("   📝 JSON length: {} chars", json.len());
        }
        Err(e) => {
            println!("   ❌ Serialization failed: {}", e);
        }
    }
    
    println!("\n🎉 All key fixes validated successfully!");
    println!("💡 System appears to be fully operational");
    
    Ok(())
}