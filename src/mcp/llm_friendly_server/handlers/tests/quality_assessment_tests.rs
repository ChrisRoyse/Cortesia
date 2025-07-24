//! Tests for quality assessment consolidation

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::core::triple::Triple;
use crate::mcp::llm_friendly_server::handlers::advanced::handle_validate_knowledge;
use crate::mcp::llm_friendly_server::types::UsageStats;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::json;

#[tokio::test]
async fn test_enhanced_validate_knowledge_comprehensive() {
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    
    // Add test data
    let ke = engine.write().await;
    ke.store_triple(Triple::with_metadata("Einstein".to_string(), "is".to_string(), "physicist".to_string(), 1.0, None).unwrap(), None).unwrap();
    ke.store_triple(Triple::with_metadata("Einstein".to_string(), "developed".to_string(), "relativity".to_string(), 0.9, None).unwrap(), None).unwrap();
    ke.store_triple(Triple::with_metadata("Einstein".to_string(), "born_in".to_string(), "1879".to_string(), 0.8, None).unwrap(), None).unwrap();
    ke.store_triple(Triple::with_metadata("Einstein".to_string(), "born_in".to_string(), "1878".to_string(), 0.7, None).unwrap(), None).unwrap(); // Conflict
    drop(ke);
    
    // Test comprehensive validation (includes quality metrics)
    let params = json!({
        "validation_type": "all",
        "scope": "comprehensive",
        "include_metrics": true
    });
    
    let result = handle_validate_knowledge(&engine, &usage_stats, params).await;
    assert!(result.is_ok());
    
    let (data, message, _) = result.unwrap();
    
    // Should include quality metrics
    assert!(data.get("quality_metrics").is_some());
    assert!(data["quality_metrics"].get("importance_scores").is_some());
    assert!(data["quality_metrics"].get("content_quality").is_some());
    assert!(data["quality_metrics"].get("knowledge_density").is_some());
    assert!(data["quality_metrics"].get("neural_assessment").is_some());
    
    // Should still include standard validation
    assert!(data.get("consistency").is_some());
    assert!(data.get("conflicts").is_some());
    assert!(data.get("quality").is_some());
}

#[tokio::test]
async fn test_validate_with_importance_scoring() {
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    
    // Add test knowledge chunk
    let ke = engine.write().await;
    ke.store_chunk(
        "Einstein's theory of general relativity revolutionized our understanding of gravity.".to_string(),
        None
    ).unwrap();
    drop(ke);
    
    // Test with importance scoring enabled
    let params = json!({
        "validation_type": "quality",
        "scope": "comprehensive",
        "include_metrics": true,
        "importance_threshold": 0.7
    });
    
    let result = handle_validate_knowledge(&engine, &usage_stats, params).await;
    assert!(result.is_ok());
    
    let (data, _, _) = result.unwrap();
    
    // Should include importance analysis
    assert!(data["quality_metrics"]["importance_scores"].is_array());
    let scores = data["quality_metrics"]["importance_scores"].as_array().unwrap();
    assert!(!scores.is_empty());
    
    // Each score should have entity and importance
    for score in scores {
        assert!(score.get("entity").is_some());
        assert!(score.get("importance").is_some());
        assert!(score.get("quality_level").is_some());
    }
}

#[tokio::test]
async fn test_validate_knowledge_density() {
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    
    // Add varying density of knowledge
    let ke = engine.write().await;
    
    // High density around Einstein
    for i in 0..10 {
        ke.store_triple(Triple::with_metadata("Einstein".to_string(), format!("fact_{}", i), format!("value_{}", i), 0.9, None).unwrap(), None).unwrap();
    }
    
    // Low density around Newton
    ke.store_triple(Triple::with_metadata("Newton".to_string(), "discovered".to_string(), "gravity".to_string(), 0.9, None).unwrap(), None).unwrap();
    
    drop(ke);
    
    // Test density analysis
    let params = json!({
        "validation_type": "quality",
        "scope": "comprehensive",
        "include_metrics": true
    });
    
    let result = handle_validate_knowledge(&engine, &usage_stats, params).await;
    assert!(result.is_ok());
    
    let (data, _, _) = result.unwrap();
    
    // Should include density metrics
    let density = &data["quality_metrics"]["knowledge_density"];
    assert!(density.get("average_connections").is_some());
    assert!(density.get("density_distribution").is_some());
    assert!(density.get("highly_connected_entities").is_some());
    assert!(density.get("isolated_entities").is_some());
}

#[tokio::test]
async fn test_neural_quality_assessment() {
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    
    // Add knowledge with varying quality
    let ke = engine.write().await;
    ke.store_chunk(
        "Einstein developed the theory of relativity which describes spacetime curvature.".to_string(),
        None
    ).unwrap();
    
    ke.store_chunk(
        "stuff things whatever".to_string(),
        None
    ).unwrap();
    drop(ke);
    
    // Test neural assessment
    let params = json!({
        "validation_type": "quality",
        "scope": "comprehensive",
        "neural_features": true
    });
    
    let result = handle_validate_knowledge(&engine, &usage_stats, params).await;
    assert!(result.is_ok());
    
    let (data, _, _) = result.unwrap();
    
    // Should include neural assessment
    let neural = &data["quality_metrics"]["neural_assessment"];
    assert!(neural.get("salience_scores").is_some());
    assert!(neural.get("coherence_scores").is_some());
    assert!(neural.get("content_recommendations").is_some());
}

#[tokio::test]
async fn test_backward_compatibility() {
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    
    // Test that old validate_knowledge calls still work
    let params = json!({
        "validation_type": "all"
        // No scope parameter - should default to standard
    });
    
    let result = handle_validate_knowledge(&engine, &usage_stats, params).await;
    assert!(result.is_ok());
    
    let (data, _, _) = result.unwrap();
    
    // Should include standard validation results
    assert!(data.get("consistency").is_some());
    assert!(data.get("conflicts").is_some());
    assert!(data.get("quality").is_some());
    assert!(data.get("completeness").is_some());
    
    // Should NOT include comprehensive metrics by default
    assert!(data.get("quality_metrics").is_none());
}

#[tokio::test]
async fn test_quality_thresholds() {
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
    let usage_stats = Arc::new(RwLock::new(UsageStats::default()));
    
    // Add mixed quality data
    let ke = engine.write().await;
    ke.store_triple(Triple::with_metadata("GoodEntity".to_string(), "has".to_string(), "quality".to_string(), 0.95, None).unwrap(), None).unwrap();
    ke.store_triple(Triple::with_metadata("BadEntity".to_string(), "has".to_string(), "issues".to_string(), 0.3, None).unwrap(), None).unwrap();
    drop(ke);
    
    // Test with quality threshold
    let params = json!({
        "validation_type": "quality",
        "scope": "comprehensive",
        "quality_threshold": 0.8
    });
    
    let result = handle_validate_knowledge(&engine, &usage_stats, params).await;
    assert!(result.is_ok());
    
    let (data, _, _) = result.unwrap();
    
    // Should identify entities below threshold
    assert!(data["quality_metrics"].get("below_threshold_entities").is_some());
    let below = data["quality_metrics"]["below_threshold_entities"].as_array().unwrap();
    assert!(below.iter().any(|e| e["entity"] == "BadEntity"));
}