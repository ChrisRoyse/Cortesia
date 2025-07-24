//! Integration tests for MCP handlers
//! This file tests the actual compiled Rust code end-to-end

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

/// Helper to create test engine with sample data
async fn setup_test_engine() -> (Arc<RwLock<KnowledgeEngine>>, Arc<RwLock<UsageStats>>) {
    let engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(384, 1000).expect("Failed to create engine")
    ));
    
    // Add test data
    {
        let mut engine_write = engine.write().await;
        
        // Add triples
        engine_write.add_triple("Einstein", "is", "scientist", 1.0).unwrap();
        engine_write.add_triple("Einstein", "invented", "relativity", 1.0).unwrap();
        engine_write.add_triple("Newton", "is", "scientist", 1.0).unwrap();
        engine_write.add_triple("Newton", "discovered", "gravity", 1.0).unwrap();
        
        // Add chunks
        engine_write.add_knowledge_chunk(
            "Einstein's Theory",
            "Albert Einstein developed the theory of relativity.",
            Some("physics"),
            Some("test")
        ).unwrap();
    }
    
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    (engine, usage_stats)
}

#[tokio::test]
async fn test_generate_graph_query_real() {
    let (engine, usage_stats) = setup_test_engine().await;
    
    println!("Testing generate_graph_query with real engine...");
    
    // Test 1: Valid query
    let params = json!({
        "natural_query": "Find all facts about Einstein"
    });
    
    match handle_generate_graph_query(&engine, &usage_stats, params).await {
        Ok((data, message, suggestions)) => {
            println!("✓ Query generation successful!");
            println!("  Query type: {:?}", data.get("query_type"));
            println!("  Message: {}", message);
            println!("  Suggestions: {} provided", suggestions.len());
            
            assert!(data.get("query_type").is_some());
            assert!(data.get("query_params").is_some());
            assert!(data.get("executable").unwrap().as_bool().unwrap());
        }
        Err(e) => panic!("Query generation failed: {}", e),
    }
    
    // Test 2: Empty query should fail
    let params = json!({
        "natural_query": ""
    });
    
    match handle_generate_graph_query(&engine, &usage_stats, params).await {
        Ok(_) => panic!("Empty query should have failed"),
        Err(e) => {
            println!("✓ Empty query correctly rejected: {}", e);
            assert!(e.contains("empty"));
        }
    }
}

#[tokio::test]
async fn test_get_stats_real() {
    let (engine, usage_stats) = setup_test_engine().await;
    
    println!("Testing get_stats with real engine...");
    
    // Test without details
    let params = json!({
        "include_details": false
    });
    
    match handle_get_stats(&engine, &usage_stats, params).await {
        Ok((data, message, _)) => {
            println!("✓ Stats retrieved successfully!");
            println!("  Total triples: {:?}", data.get("total_triples"));
            println!("  Total chunks: {:?}", data.get("total_chunks"));
            println!("  Total entities: {:?}", data.get("total_entities"));
            
            let triples = data.get("total_triples").unwrap().as_u64().unwrap();
            let chunks = data.get("total_chunks").unwrap().as_u64().unwrap();
            
            assert_eq!(triples, 4, "Expected 4 triples");
            assert_eq!(chunks, 1, "Expected 1 chunk");
        }
        Err(e) => panic!("Stats retrieval failed: {}", e),
    }
    
    // Test with details
    let params = json!({
        "include_details": true
    });
    
    match handle_get_stats(&engine, &usage_stats, params).await {
        Ok((data, _, _)) => {
            println!("✓ Detailed stats retrieved!");
            assert!(data.get("top_entities").is_some());
            assert!(data.get("predicate_distribution").is_some());
        }
        Err(e) => panic!("Detailed stats failed: {}", e),
    }
}

#[tokio::test]
async fn test_validate_knowledge_real() {
    let (engine, usage_stats) = setup_test_engine().await;
    
    println!("Testing validate_knowledge with real engine...");
    
    // Test standard validation
    let params = json!({
        "validation_type": "all",
        "fix_issues": false
    });
    
    match handle_validate_knowledge(&engine, &usage_stats, params).await {
        Ok((data, message, _)) => {
            println!("✓ Validation completed!");
            println!("  Message: {}", message);
            
            let results = data.get("results").unwrap().as_object().unwrap();
            assert!(results.contains_key("consistency"));
            assert!(results.contains_key("conflicts"));
            assert!(results.contains_key("quality"));
            assert!(results.contains_key("completeness"));
            
            let summary = data.get("summary").unwrap();
            assert!(summary.get("is_valid").is_some());
        }
        Err(e) => panic!("Validation failed: {}", e),
    }
    
    // Test comprehensive validation
    let params = json!({
        "validation_type": "all",
        "scope": "comprehensive",
        "include_metrics": true
    });
    
    match handle_validate_knowledge(&engine, &usage_stats, params).await {
        Ok((data, _, _)) => {
            println!("✓ Comprehensive validation completed!");
            assert!(data.get("quality_metrics").is_some());
        }
        Err(e) => panic!("Comprehensive validation failed: {}", e),
    }
}

#[tokio::test]
async fn test_neural_importance_scoring_real() {
    let (engine, usage_stats) = setup_test_engine().await;
    
    println!("Testing neural_importance_scoring with real engine...");
    
    // Test important content
    let params = json!({
        "text": "Albert Einstein revolutionized physics with his theory of relativity.",
        "context": "physics discoveries"
    });
    
    match handle_neural_importance_scoring(&engine, &usage_stats, params).await {
        Ok((data, message, suggestions)) => {
            println!("✓ Importance scoring completed!");
            println!("  Score: {:?}", data.get("importance_score"));
            println!("  Quality: {:?}", data.get("quality_level"));
            println!("  Should store: {:?}", data.get("should_store"));
            
            let score = data.get("importance_score").unwrap().as_f64().unwrap();
            assert!(score > 0.0 && score <= 1.0);
            assert!(data.get("should_store").unwrap().as_bool().unwrap());
            assert!(!suggestions.is_empty());
        }
        Err(e) => panic!("Importance scoring failed: {}", e),
    }
    
    // Test trivial content
    let params = json!({
        "text": "The cat sat."
    });
    
    match handle_neural_importance_scoring(&engine, &usage_stats, params).await {
        Ok((data, _, _)) => {
            println!("✓ Trivial content scored correctly");
            let score = data.get("importance_score").unwrap().as_f64().unwrap();
            assert!(score < 0.5, "Trivial content should have low score");
        }
        Err(e) => panic!("Trivial scoring failed: {}", e),
    }
}

#[tokio::test]
async fn test_full_workflow_real() {
    let (engine, usage_stats) = setup_test_engine().await;
    
    println!("Testing full workflow with real engine...");
    
    // 1. Get initial stats
    let stats_result = handle_get_stats(&engine, &usage_stats, json!({})).await.unwrap();
    let initial_triples = stats_result.0.get("total_triples").unwrap().as_u64().unwrap();
    println!("Initial triples: {}", initial_triples);
    
    // 2. Score new content
    let score_result = handle_neural_importance_scoring(
        &engine,
        &usage_stats,
        json!({
            "text": "Marie Curie discovered radioactivity and won two Nobel prizes.",
            "context": "scientific achievements"
        })
    ).await.unwrap();
    
    let should_store = score_result.0.get("should_store").unwrap().as_bool().unwrap();
    assert!(should_store, "Important content should be marked for storage");
    
    // 3. Add the content (simulating the workflow)
    {
        let mut engine_write = engine.write().await;
        engine_write.add_triple("Marie Curie", "discovered", "radioactivity", 1.0).unwrap();
    }
    
    // 4. Generate query to find it
    let query_result = handle_generate_graph_query(
        &engine,
        &usage_stats,
        json!({
            "natural_query": "Find facts about Marie Curie"
        })
    ).await.unwrap();
    
    assert!(query_result.0.get("executable").unwrap().as_bool().unwrap());
    
    // 5. Validate the knowledge base
    let validation_result = handle_validate_knowledge(
        &engine,
        &usage_stats,
        json!({
            "validation_type": "all"
        })
    ).await.unwrap();
    
    let is_valid = validation_result.0
        .get("summary").unwrap()
        .get("is_valid").unwrap()
        .as_bool().unwrap();
    assert!(is_valid, "Knowledge base should be valid");
    
    // 6. Check final stats
    let final_stats = handle_get_stats(&engine, &usage_stats, json!({})).await.unwrap();
    let final_triples = final_stats.0.get("total_triples").unwrap().as_u64().unwrap();
    assert_eq!(final_triples, initial_triples + 1, "Should have added 1 triple");
    
    println!("✓ Full workflow completed successfully!");
}

#[tokio::test]
async fn test_concurrent_access_real() {
    let (engine, usage_stats) = setup_test_engine().await;
    
    println!("Testing concurrent access with real engine...");
    
    // Run multiple operations concurrently
    let handles: Vec<_> = (0..4).map(|i| {
        let engine = engine.clone();
        let stats = usage_stats.clone();
        
        tokio::spawn(async move {
            match i {
                0 => handle_get_stats(&engine, &stats, json!({})).await,
                1 => handle_generate_graph_query(&engine, &stats, json!({"natural_query": "test"})).await,
                2 => handle_validate_knowledge(&engine, &stats, json!({"validation_type": "consistency"})).await,
                3 => handle_neural_importance_scoring(&engine, &stats, json!({"text": "test"})).await,
                _ => unreachable!(),
            }
        })
    }).collect();
    
    // All should complete successfully
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.await {
            Ok(Ok(_)) => println!("✓ Task {} completed", i),
            Ok(Err(e)) => panic!("Task {} failed: {}", i, e),
            Err(e) => panic!("Task {} panicked: {}", i, e),
        }
    }
    
    // Verify usage stats tracked operations
    let usage = usage_stats.read().await;
    assert!(usage.queries_executed >= 1, "Should have tracked query operations");
    
    println!("✓ Concurrent access successful!");
}