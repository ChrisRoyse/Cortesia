//! Integration tests for MCP handlers with real KnowledgeEngine instances

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::mcp::llm_friendly_server::handlers::{
    handle_generate_graph_query,
    handle_get_stats,
    handle_validate_knowledge,
    handle_neural_importance_scoring,
};
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::test_support::test_utils::{create_test_engine, create_test_stats};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::RwLock;

// Test helper functions are now imported from test_utils

#[tokio::test]
async fn test_generate_graph_query_integration() {
    let engine = create_test_engine(true).await.expect("Failed to create test engine");
    let usage_stats = create_test_stats();
    
    // Test 1: Basic query generation
    let params = json!({
        "natural_query": "Find all facts about Einstein"
    });
    
    let result = handle_generate_graph_query(&engine, &usage_stats, params).await;
    assert!(result.is_ok(), "Failed to generate query: {:?}", result.err());
    
    let (data, message, _suggestions) = result.unwrap();
    assert!(data.get("query_type").is_some());
    assert!(data.get("query_params").is_some());
    assert!(data.get("executable").unwrap().as_bool().unwrap());
    assert!(!message.is_empty());
    assert!(!_suggestions.is_empty());
    
    // Test 2: Complex query
    let params = json!({
        "natural_query": "Show connections between Einstein and Newton"
    });
    
    let result = handle_generate_graph_query(&engine, &usage_stats, params).await;
    assert!(result.is_ok());
    
    let (data, _, _) = result.unwrap();
    let query_type = data.get("query_type").unwrap().as_str().unwrap();
    assert!(query_type == "path_query" || query_type == "hybrid_search");
    
    // Test 3: Empty query should fail
    let params = json!({
        "natural_query": ""
    });
    
    let result = handle_generate_graph_query(&engine, &usage_stats, params).await;
    assert!(result.is_err());
    assert!(result.err().unwrap().contains("empty"));
}

#[tokio::test]
async fn test_get_stats_integration() {
    let engine = create_test_engine(true).await.expect("Failed to create test engine");
    let usage_stats = create_test_stats();
    
    // Test 1: Basic stats without details
    let params = json!({
        "include_details": false
    });
    
    let result = handle_get_stats(&engine, &usage_stats, params).await;
    assert!(result.is_ok(), "Failed to get stats: {:?}", result.err());
    
    let (data, message, _) = result.unwrap();
    
    // Verify basic stats structure
    assert!(data.get("total_triples").is_some());
    assert!(data.get("total_chunks").is_some());
    assert!(data.get("total_entities").is_some());
    assert!(data.get("memory_usage").is_some());
    
    // Verify counts match our test data
    let triple_count = data.get("total_triples").unwrap().as_u64().unwrap();
    assert_eq!(triple_count, 5, "Expected 5 triples");
    
    let chunk_count = data.get("total_chunks").unwrap().as_u64().unwrap();
    assert_eq!(chunk_count, 2, "Expected 2 chunks");
    
    let entity_count = data.get("total_entities").unwrap().as_u64().unwrap();
    assert!(entity_count >= 4, "Expected at least 4 entities");
    
    assert!(!message.is_empty());
    
    // Test 2: Detailed stats
    let params = json!({
        "include_details": true
    });
    
    let result = handle_get_stats(&engine, &usage_stats, params).await;
    assert!(result.is_ok());
    
    let (data, _, _) = result.unwrap();
    assert!(data.get("category_breakdown").is_some());
    assert!(data.get("top_entities").is_some());
    assert!(data.get("predicate_distribution").is_some());
    
    // Verify top entities include our test data
    let top_entities = data.get("top_entities").unwrap().as_array().unwrap();
    let entity_names: Vec<&str> = top_entities.iter()
        .filter_map(|e| e.get("entity").and_then(|v| v.as_str()))
        .collect();
    assert!(entity_names.contains(&"Einstein"));
    assert!(entity_names.contains(&"Newton"));
}

#[tokio::test]
async fn test_validate_knowledge_integration() {
    let engine = create_test_engine(true).await.expect("Failed to create test engine");
    let usage_stats = create_test_stats();
    
    // Test 1: Validate all knowledge (standard mode)
    let params = json!({
        "validation_type": "all",
        "fix_issues": false
    });
    
    let result = handle_validate_knowledge(&engine, &usage_stats, params).await;
    assert!(result.is_ok(), "Failed to validate knowledge: {:?}", result.err());
    
    let (data, message, _suggestions) = result.unwrap();
    
    // Verify validation results structure
    assert!(data.get("validation_type").is_some());
    assert!(data.get("results").is_some());
    assert!(data.get("summary").is_some());
    assert!(!message.is_empty());
    
    let results = data.get("results").unwrap().as_object().unwrap();
    assert!(results.contains_key("consistency"));
    assert!(results.contains_key("conflicts"));
    assert!(results.contains_key("quality"));
    assert!(results.contains_key("completeness"));
    
    // Test 2: Validate specific entity
    let params = json!({
        "validation_type": "consistency",
        "entity": "Einstein"
    });
    
    let result = handle_validate_knowledge(&engine, &usage_stats, params).await;
    assert!(result.is_ok());
    
    let (data, _, _) = result.unwrap();
    let results = data.get("results").unwrap();
    assert!(results.get("consistency").is_some());
    
    // Test 3: Comprehensive validation with metrics
    let params = json!({
        "validation_type": "all",
        "scope": "comprehensive",
        "include_metrics": true,
        "quality_threshold": 0.5,
        "importance_threshold": 0.3
    });
    
    let result = handle_validate_knowledge(&engine, &usage_stats, params).await;
    assert!(result.is_ok());
    
    let (data, _, suggestions) = result.unwrap();
    assert!(data.get("quality_metrics").is_some());
    assert!(!_suggestions.is_empty());
}

#[tokio::test]
async fn test_neural_importance_scoring_integration() {
    let engine = create_test_engine(true).await.expect("Failed to create test engine");
    let usage_stats = create_test_stats();
    
    // Test 1: Score important content
    let params = json!({
        "text": "Albert Einstein revolutionized physics with his theory of relativity, changing our understanding of space and time.",
        "context": "physics discoveries"
    });
    
    let result = handle_neural_importance_scoring(&engine, &usage_stats, params).await;
    assert!(result.is_ok(), "Failed to score importance: {:?}", result.err());
    
    let (data, message, _suggestions) = result.unwrap();
    
    // Verify scoring results
    assert!(data.get("importance_score").is_some());
    assert!(data.get("quality_level").is_some());
    assert!(data.get("should_store").is_some());
    assert!(data.get("complexity_analysis").is_some());
    assert!(data.get("salience_features").is_some());
    
    let importance_score = data.get("importance_score").unwrap().as_f64().unwrap();
    assert!(importance_score > 0.0 && importance_score <= 1.0);
    
    let should_store = data.get("should_store").unwrap().as_bool().unwrap();
    assert!(should_store, "Important content should be marked for storage");
    
    assert!(!message.is_empty());
    assert!(!_suggestions.is_empty());
    
    // Test 2: Score less important content
    let params = json!({
        "text": "The cat sat on the mat."
    });
    
    let result = handle_neural_importance_scoring(&engine, &usage_stats, params).await;
    assert!(result.is_ok());
    
    let (data, _, _) = result.unwrap();
    let importance_score = data.get("importance_score").unwrap().as_f64().unwrap();
    assert!(importance_score < 0.5, "Simple content should have low importance");
    
    // Test 3: Empty text should fail
    let params = json!({
        "text": ""
    });
    
    let result = handle_neural_importance_scoring(&engine, &usage_stats, params).await;
    assert!(result.is_err());
    assert!(result.err().unwrap().contains("Empty"));
    
    // Test 4: With rich context
    let params = json!({
        "text": "Quantum mechanics describes the behavior of matter at atomic scales.",
        "context": "This is part of a comprehensive physics textbook chapter on modern physics theories."
    });
    
    let result = handle_neural_importance_scoring(&engine, &usage_stats, params).await;
    assert!(result.is_ok());
    
    let (data, _, _) = result.unwrap();
    let context_score = data.get("context_relevance").unwrap().as_f64().unwrap();
    assert!(context_score > 0.0, "Context should influence scoring");
}

#[tokio::test]
async fn test_full_workflow_integration() {
    let engine = create_test_engine(true).await.expect("Failed to create test engine");
    let usage_stats = create_test_stats();
    
    // Step 1: Check initial stats
    let params = json!({ "include_details": true });
    let stats_before = handle_get_stats(&engine, &usage_stats, params.clone()).await.unwrap();
    let initial_triples = stats_before.0.get("total_triples").unwrap().as_u64().unwrap();
    
    // Step 2: Score new content for importance
    let new_content = "Marie Curie was a pioneering physicist and chemist who conducted groundbreaking research on radioactivity.";
    let score_params = json!({
        "text": new_content,
        "context": "Scientific discoveries"
    });
    
    let score_result = handle_neural_importance_scoring(&engine, &usage_stats, score_params).await.unwrap();
    let should_store = score_result.0.get("should_store").unwrap().as_bool().unwrap();
    assert!(should_store, "Important scientific content should be marked for storage");
    
    // Step 3: Actually add the content (simulating the workflow)
    {
        let mut engine_write = engine.write().await;
        engine_write.add_triple("Marie Curie", "is", "scientist", 1.0).unwrap();
        engine_write.add_triple("Marie Curie", "discovered", "radioactivity", 1.0).unwrap();
    }
    
    // Step 4: Generate query to find the new content
    let query_params = json!({
        "natural_query": "Find information about Marie Curie"
    });
    
    let query_result = handle_generate_graph_query(&engine, &usage_stats, query_params).await.unwrap();
    assert!(query_result.0.get("executable").unwrap().as_bool().unwrap());
    
    // Step 5: Validate the updated knowledge base
    let validate_params = json!({
        "validation_type": "all",
        "scope": "comprehensive",
        "include_metrics": true
    });
    
    let validation_result = handle_validate_knowledge(&engine, &usage_stats, validate_params).await.unwrap();
    let valid = validation_result.0.get("summary").unwrap().get("is_valid").unwrap().as_bool().unwrap();
    assert!(valid, "Knowledge base should be valid after additions");
    
    // Step 6: Check final stats
    let stats_after = handle_get_stats(&engine, &usage_stats, params).await.unwrap();
    let final_triples = stats_after.0.get("total_triples").unwrap().as_u64().unwrap();
    assert_eq!(final_triples, initial_triples + 2, "Should have added 2 new triples");
    
    // Verify usage stats were updated
    let usage = usage_stats.read().await;
    assert!(usage.queries_executed > 0, "Usage stats should track operations");
}

#[tokio::test]
async fn test_error_handling_integration() {
    let engine = create_test_engine(true).await.expect("Failed to create test engine");
    let usage_stats = create_test_stats();
    
    // Test various error conditions
    
    // 1. Missing required parameters
    let params = json!({});
    let result = handle_generate_graph_query(&engine, &usage_stats, params).await;
    assert!(result.is_err());
    assert!(result.err().unwrap().contains("Missing required"));
    
    // 2. Invalid validation type
    let params = json!({
        "validation_type": "invalid_type"
    });
    let result = handle_validate_knowledge(&engine, &usage_stats, params).await;
    assert!(result.is_err());
    assert!(result.err().unwrap().contains("Invalid validation_type"));
    
    // 3. Text too long for importance scoring
    let long_text = "x".repeat(50001);
    let params = json!({
        "text": long_text
    });
    let result = handle_neural_importance_scoring(&engine, &usage_stats, params).await;
    assert!(result.is_err());
    
    // 4. Invalid JSON types
    let params = json!({
        "include_details": "not_a_boolean"
    });
    let result = handle_get_stats(&engine, &usage_stats, params).await;
    assert!(result.is_ok()); // Should use default value
    let (data, _, _) = result.unwrap();
    assert!(data.get("total_triples").is_some());
}

#[tokio::test]
async fn test_concurrent_operations() {
    let engine = create_test_engine(true).await.expect("Failed to create test engine");
    let usage_stats = create_test_stats();
    
    // Run multiple operations concurrently
    let engine_clone1 = engine.clone();
    let engine_clone2 = engine.clone();
    let engine_clone3 = engine.clone();
    let engine_clone4 = engine.clone();
    
    let stats_clone1 = usage_stats.clone();
    let stats_clone2 = usage_stats.clone();
    let stats_clone3 = usage_stats.clone();
    let stats_clone4 = usage_stats.clone();
    
    let (r1, r2, r3, r4) = tokio::join!(
        handle_get_stats(&engine_clone1, &stats_clone1, json!({"include_details": false})),
        handle_generate_graph_query(&engine_clone2, &stats_clone2, json!({"natural_query": "Find all scientists"})),
        handle_validate_knowledge(&engine_clone3, &stats_clone3, json!({"validation_type": "consistency"})),
        handle_neural_importance_scoring(&engine_clone4, &stats_clone4, json!({"text": "Test content"}))
    );
    
    // All operations should succeed
    assert!(r1.is_ok(), "Stats failed: {:?}", r1.err());
    assert!(r2.is_ok(), "Query generation failed: {:?}", r2.err());
    assert!(r3.is_ok(), "Validation failed: {:?}", r3.err());
    assert!(r4.is_ok(), "Importance scoring failed: {:?}", r4.err());
    
    // Verify usage stats tracked all operations
    let usage = usage_stats.read().await;
    assert!(usage.queries_executed >= 4, "Should have tracked at least 4 operations");
}