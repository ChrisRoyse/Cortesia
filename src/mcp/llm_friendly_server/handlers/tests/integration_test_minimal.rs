//! Minimal integration tests for MCP handlers with real KnowledgeEngine instances
//! This is a simplified version that tests just the core functionality

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::mcp::llm_friendly_server::handlers::{
    handle_generate_graph_query,
    handle_get_stats,
    handle_validate_knowledge,
    handle_neural_importance_scoring,
};
use crate::mcp::llm_friendly_server::types::UsageStats;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Helper function to create a test engine with minimal data
async fn create_minimal_test_engine() -> Arc<RwLock<KnowledgeEngine>> {
    let engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(384, 1000).expect("Failed to create engine")
    ));
    
    // Add minimal test data
    {
        let mut engine_write = engine.write().await;
        engine_write.add_triple("Einstein", "is", "scientist", 1.0)
            .expect("Failed to add triple");
    }
    
    engine
}

#[tokio::test]
async fn test_minimal_generate_graph_query() {
    let engine = create_minimal_test_engine().await;
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    
    // Test basic query generation
    let params = json!({
        "natural_query": "Find all facts about Einstein"
    });
    
    let result = handle_generate_graph_query(&engine, &usage_stats, params).await;
    
    match result {
        Ok((data, message, _suggestions)) => {
            println!("Query generation successful!");
            println!("Data: {:?}", data);
            println!("Message: {}", message);
            
            assert!(data.get("query_type").is_some(), "Missing query_type");
            assert!(data.get("query_params").is_some(), "Missing query_params");
            assert!(!message.is_empty(), "Empty message");
        }
        Err(e) => {
            panic!("Failed to generate query: {}", e);
        }
    }
}

#[tokio::test]
async fn test_minimal_get_stats() {
    let engine = create_minimal_test_engine().await;
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    
    let params = json!({
        "include_details": false
    });
    
    let result = handle_get_stats(&engine, &usage_stats, params).await;
    
    match result {
        Ok((data, message, _)) => {
            println!("Stats retrieval successful!");
            println!("Data: {:?}", data);
            println!("Message: {}", message);
            
            // Verify basic stats exist
            assert!(data.get("total_triples").is_some(), "Missing total_triples");
            assert!(data.get("total_chunks").is_some(), "Missing total_chunks");
            assert!(data.get("total_entities").is_some(), "Missing total_entities");
            
            // Verify we have at least one triple
            let triple_count = data.get("total_triples")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            assert!(triple_count >= 1, "Expected at least 1 triple, got {}", triple_count);
        }
        Err(e) => {
            panic!("Failed to get stats: {}", e);
        }
    }
}

#[tokio::test]
async fn test_minimal_validate_knowledge() {
    let engine = create_minimal_test_engine().await;
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    
    let params = json!({
        "validation_type": "consistency",
        "fix_issues": false
    });
    
    let result = handle_validate_knowledge(&engine, &usage_stats, params).await;
    
    match result {
        Ok((data, message, _)) => {
            println!("Validation successful!");
            println!("Data: {:?}", data);
            println!("Message: {}", message);
            
            assert!(data.get("validation_type").is_some(), "Missing validation_type");
            assert!(data.get("results").is_some(), "Missing results");
        }
        Err(e) => {
            panic!("Failed to validate knowledge: {}", e);
        }
    }
}

#[tokio::test]
async fn test_minimal_neural_importance_scoring() {
    let engine = create_minimal_test_engine().await;
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    
    let params = json!({
        "text": "Albert Einstein was a theoretical physicist."
    });
    
    let result = handle_neural_importance_scoring(&engine, &usage_stats, params).await;
    
    match result {
        Ok((data, message, _)) => {
            println!("Importance scoring successful!");
            println!("Data: {:?}", data);
            println!("Message: {}", message);
            
            // Verify scoring results
            assert!(data.get("importance_score").is_some(), "Missing importance_score");
            assert!(data.get("quality_level").is_some(), "Missing quality_level");
            assert!(data.get("should_store").is_some(), "Missing should_store");
            
            // Verify score is in valid range
            let score = data.get("importance_score")
                .and_then(|v| v.as_f64())
                .unwrap_or(-1.0);
            assert!(score >= 0.0 && score <= 1.0, "Invalid score: {}", score);
        }
        Err(e) => {
            panic!("Failed to score importance: {}", e);
        }
    }
}

#[tokio::test]
async fn test_error_handling() {
    let engine = create_minimal_test_engine().await;
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    
    // Test missing required parameter
    let params = json!({});
    let result = handle_generate_graph_query(&engine, &usage_stats, params).await;
    assert!(result.is_err(), "Should fail with missing parameter");
    
    // Test empty text for importance scoring
    let params = json!({
        "text": ""
    });
    let result = handle_neural_importance_scoring(&engine, &usage_stats, params).await;
    assert!(result.is_err(), "Should fail with empty text");
}

#[tokio::test]
async fn test_concurrent_operations_minimal() {
    let engine = create_minimal_test_engine().await;
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    
    // Clone references for concurrent operations
    let engine1 = engine.clone();
    let engine2 = engine.clone();
    let stats1 = usage_stats.clone();
    let stats2 = usage_stats.clone();
    
    // Run two operations concurrently
    let (r1, r2) = tokio::join!(
        handle_get_stats(&engine1, &stats1, json!({"include_details": false})),
        handle_generate_graph_query(&engine2, &stats2, json!({"natural_query": "test"}))
    );
    
    // Both should succeed
    assert!(r1.is_ok(), "Stats failed: {:?}", r1.err());
    assert!(r2.is_ok(), "Query generation failed: {:?}", r2.err());
}