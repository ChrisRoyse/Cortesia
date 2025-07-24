use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::mcp::llm_friendly_server::handlers::{
    handle_generate_graph_query,
    handle_get_stats,
    handle_validate_knowledge,
    handle_neural_importance_scoring,
};
use llmkg::mcp::llm_friendly_server::types::UsageStats;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing that LLMKG compiles and the 4 key tools work...\n");

    // Test 1: Create a KnowledgeEngine
    println!("1. Creating KnowledgeEngine...");
    let engine_result = KnowledgeEngine::new(384, 1000);
    match engine_result {
        Ok(engine) => {
            println!("✓ KnowledgeEngine created successfully");
            let engine = Arc::new(RwLock::new(engine));
            let usage_stats = Arc::new(RwLock::new(UsageStats::default()));

            // Add some test data
            {
                let mut engine_write = engine.write().await;
                match engine_write.add_triple("Einstein", "is", "scientist", 1.0) {
                    Ok(_) => println!("✓ Added test triple"),
                    Err(e) => println!("✗ Failed to add triple: {}", e),
                }
            }

            // Test 2: handle_generate_graph_query
            println!("\n2. Testing handle_generate_graph_query...");
            let params = json!({
                "natural_query": "Find all facts about Einstein"
            });
            
            match handle_generate_graph_query(&engine, &usage_stats, params).await {
                Ok((data, message, suggestions)) => {
                    println!("✓ handle_generate_graph_query works!");
                    println!("  - Has query_type: {}", data.get("query_type").is_some());
                    println!("  - Message length: {}", message.len());
                    println!("  - Suggestions count: {}", suggestions.len());
                },
                Err(e) => println!("✗ handle_generate_graph_query failed: {}", e),
            }

            // Test 3: handle_get_stats
            println!("\n3. Testing handle_get_stats...");
            let params = json!({ "include_details": false });
            
            match handle_get_stats(&engine, &usage_stats, params).await {
                Ok((data, message, suggestions)) => {
                    println!("✓ handle_get_stats works!");
                    println!("  - Has total_triples: {}", data.get("total_triples").is_some());
                    println!("  - Message length: {}", message.len());
                    println!("  - Suggestions count: {}", suggestions.len());
                },
                Err(e) => println!("✗ handle_get_stats failed: {}", e),
            }

            // Test 4: handle_validate_knowledge
            println!("\n4. Testing handle_validate_knowledge...");
            let params = json!({
                "validation_type": "all",
                "fix_issues": false
            });
            
            match handle_validate_knowledge(&engine, &usage_stats, params).await {
                Ok((data, message, suggestions)) => {
                    println!("✓ handle_validate_knowledge works!");
                    println!("  - Has validation_type: {}", data.get("validation_type").is_some());
                    println!("  - Message length: {}", message.len());
                    println!("  - Suggestions count: {}", suggestions.len());
                },
                Err(e) => println!("✗ handle_validate_knowledge failed: {}", e),
            }

            // Test 5: handle_neural_importance_scoring
            println!("\n5. Testing handle_neural_importance_scoring...");
            let params = json!({
                "text": "Albert Einstein was a theoretical physicist who revolutionized physics."
            });
            
            match handle_neural_importance_scoring(&engine, &usage_stats, params).await {
                Ok((data, message, suggestions)) => {
                    println!("✓ handle_neural_importance_scoring works!");
                    println!("  - Has importance_score: {}", data.get("importance_score").is_some());
                    println!("  - Message length: {}", message.len());
                    println!("  - Suggestions count: {}", suggestions.len());
                },
                Err(e) => println!("✗ handle_neural_importance_scoring failed: {}", e),
            }
        },
        Err(e) => {
            println!("✗ Failed to create KnowledgeEngine: {}", e);
            return Err(e.into());
        }
    }

    println!("\n✅ All tests completed!");
    Ok(())
}