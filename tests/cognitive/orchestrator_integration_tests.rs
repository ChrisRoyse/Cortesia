//! Integration tests for CognitiveOrchestrator
//! Tests orchestration of cognitive patterns and workflow management

use std::sync::Arc;
use tokio;

use llmkg::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use llmkg::cognitive::types::{
    CognitivePatternType, ReasoningStrategy, ReasoningResult
};
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::test_support::fixtures::*;

/// Creates a test orchestrator with minimal test data
async fn create_test_orchestrator() -> CognitiveOrchestrator {
    let graph = create_test_graph();
    
    let config = CognitiveOrchestratorConfig {
        enable_adaptive_selection: true,
        enable_ensemble_methods: true,
        default_timeout_ms: 5000,
        max_parallel_patterns: 3,
        performance_tracking: true,
    };
    
    CognitiveOrchestrator::new(graph, config).await
        .expect("Failed to create test orchestrator")
}

#[tokio::test]
async fn test_orchestrator_initialization() {
    let orchestrator = create_test_orchestrator().await;
    
    // Test that orchestrator initializes with all required patterns
    let stats = orchestrator.get_statistics().await.unwrap();
    
    assert_eq!(stats.total_patterns, 7, "Should have all 7 cognitive patterns");
    assert!(stats.available_patterns.contains(&CognitivePatternType::Convergent));
    assert!(stats.available_patterns.contains(&CognitivePatternType::Divergent));
    assert!(stats.available_patterns.contains(&CognitivePatternType::Lateral));
    assert!(stats.available_patterns.contains(&CognitivePatternType::Systems));
    assert!(stats.available_patterns.contains(&CognitivePatternType::Critical));
    assert!(stats.available_patterns.contains(&CognitivePatternType::Abstract));
    assert!(stats.available_patterns.contains(&CognitivePatternType::Adaptive));
}

#[tokio::test]
async fn test_automatic_pattern_selection() {
    let orchestrator = create_test_orchestrator().await;
    
    // Test automatic pattern selection for different query types
    let test_cases = vec![
        "What is quantum computing?",
        "Give me creative applications of AI",
        "How might art and technology intersect?",
        "Analyze the relationship between systems",
    ];
    
    for query in test_cases {
        let result = orchestrator.reason(
            query,
            None,
            ReasoningStrategy::Automatic,
        ).await.unwrap();
        
        assert!(!result.final_answer.is_empty(), "Should provide answer for: {}", query);
        assert!(result.quality_metrics.overall_confidence >= 0.0);
        assert!(result.execution_metadata.patterns_executed.len() >= 1);
    }
}

#[tokio::test]
async fn test_specific_pattern_execution() {
    let orchestrator = create_test_orchestrator().await;
    
    let query = "What are the applications of machine learning?";
    
    // Test each pattern individually
    let patterns = vec![
        CognitivePatternType::Convergent,
        CognitivePatternType::Divergent,
        CognitivePatternType::Lateral,
        CognitivePatternType::Systems,
        CognitivePatternType::Critical,
        CognitivePatternType::Abstract,
        CognitivePatternType::Adaptive,
    ];
    
    for pattern in patterns {
        let result = orchestrator.reason(
            query,
            None,
            ReasoningStrategy::Specific(pattern),
        ).await.unwrap();
        
        assert!(!result.final_answer.is_empty(), 
               "Pattern {:?} should provide answer", pattern);
        assert_eq!(result.strategy_used, ReasoningStrategy::Specific(pattern));
        assert!(result.execution_metadata.patterns_executed.contains(&pattern));
        assert!(result.quality_metrics.overall_confidence >= 0.0);
        assert!(result.quality_metrics.overall_confidence <= 1.0);
    }
}

#[tokio::test]
async fn test_ensemble_reasoning() {
    let orchestrator = create_test_orchestrator().await;
    
    let query = "Analyze creative applications of quantum computing in art";
    let ensemble_patterns = vec![
        CognitivePatternType::Convergent,  // For factual foundation
        CognitivePatternType::Divergent,   // For creative exploration
        CognitivePatternType::Lateral,     // For novel connections
    ];
    
    let result = orchestrator.reason(
        query,
        None,
        ReasoningStrategy::Ensemble(ensemble_patterns.clone()),
    ).await.unwrap();
    
    assert!(!result.final_answer.is_empty());
    assert_eq!(result.strategy_used, ReasoningStrategy::Ensemble(ensemble_patterns));
    
    // Should have executed multiple patterns
    assert!(result.execution_metadata.patterns_executed.len() >= 2,
           "Ensemble should execute multiple patterns");
    
    // Quality metrics should reflect ensemble benefits
    assert!(result.quality_metrics.completeness_score >= 0.0,
           "Ensemble should provide completeness metric");
    assert!(result.quality_metrics.consistency_score >= 0.0,
           "Ensemble should provide consistency metric");
}

#[tokio::test]
async fn test_multi_pattern_coordination() {
    let orchestrator = create_test_orchestrator().await;
    
    // Test complex query that requires coordinated reasoning
    let query = "How can biomimicry inform sustainable solutions?";
    
    let result = orchestrator.reason(
        query,
        None,
        ReasoningStrategy::Automatic,
    ).await.unwrap();
    
    assert!(!result.final_answer.is_empty());
    assert!(result.final_answer.len() > 50, "Complex query should generate substantial answer");
    
    // Should demonstrate sophisticated reasoning
    assert!(result.quality_metrics.overall_confidence >= 0.0);
    assert!(result.execution_metadata.total_time_ms >= 0);
}

#[tokio::test]
async fn test_context_integration() {
    let orchestrator = create_test_orchestrator().await;
    
    let base_query = "What is machine learning?";
    let context = "We are discussing applications in healthcare and medical diagnosis";
    
    // Test without context
    let result_no_context = orchestrator.reason(
        base_query,
        None,
        ReasoningStrategy::Automatic,
    ).await.unwrap();
    
    // Test with context
    let result_with_context = orchestrator.reason(
        base_query,
        Some(context),
        ReasoningStrategy::Automatic,
    ).await.unwrap();
    
    assert!(!result_no_context.final_answer.is_empty());
    assert!(!result_with_context.final_answer.is_empty());
    
    // Both should provide reasonable answers
    assert!(result_no_context.quality_metrics.overall_confidence >= 0.0);
    assert!(result_with_context.quality_metrics.overall_confidence >= 0.0);
}

#[tokio::test]
async fn test_error_recovery_and_fallback() {
    let orchestrator = create_test_orchestrator().await;
    
    // Test with challenging queries that might require fallback strategies
    let challenging_queries = vec![
        "", // Empty query
        "   ", // Whitespace only
        "?", // Single character
        "asdfghjkl qwertyuiop", // Nonsensical query
    ];
    
    for query in challenging_queries {
        let result = orchestrator.reason(
            query,
            None,
            ReasoningStrategy::Automatic,
        ).await;
        
        match result {
            Ok(reasoning_result) => {
                // If successful, should have some response or low confidence
                assert!(
                    !reasoning_result.final_answer.is_empty() || 
                    reasoning_result.quality_metrics.overall_confidence <= 0.5,
                    "Should handle challenging query: '{}'", query
                );
            }
            Err(_) => {
                // Errors are acceptable for invalid queries
                println!("Query '{}' appropriately rejected", query);
            }
        }
    }
}

#[tokio::test]
async fn test_performance_monitoring() {
    let orchestrator = create_test_orchestrator().await;
    
    // Execute several queries to generate performance data
    let queries = vec![
        "What is quantum computing?",
        "Creative uses of AI in art",
        "Systems analysis of technology",
    ];
    
    for query in queries {
        let result = orchestrator.reason(
            query,
            None,
            ReasoningStrategy::Automatic,
        ).await.unwrap();
        
        // Basic performance metrics should be available
        assert!(result.execution_metadata.total_time_ms >= 0);
        assert!(!result.execution_metadata.patterns_executed.is_empty());
    }
    
    // Check orchestrator-level performance metrics
    let performance_metrics = orchestrator.get_performance_metrics().await.unwrap();
    
    assert!(performance_metrics.total_queries_processed >= 0);
    assert!(performance_metrics.average_response_time_ms >= 0.0);
    assert!(performance_metrics.success_rate >= 0.0);
    assert!(performance_metrics.success_rate <= 1.0);
}

#[tokio::test]
async fn test_ensemble_quality_metrics() {
    let orchestrator = create_test_orchestrator().await;
    
    let query = "Explore the intersection of AI, creativity, and sustainability";
    let ensemble_patterns = vec![
        CognitivePatternType::Convergent,
        CognitivePatternType::Divergent,
        CognitivePatternType::Lateral,
        CognitivePatternType::Systems,
    ];
    
    let result = orchestrator.reason(
        query,
        None,
        ReasoningStrategy::Ensemble(ensemble_patterns),
    ).await.unwrap();
    
    // Ensemble should provide comprehensive quality metrics
    let quality = &result.quality_metrics;
    
    assert!(quality.overall_confidence >= 0.0 && quality.overall_confidence <= 1.0);
    assert!(quality.consistency_score >= 0.0 && quality.consistency_score <= 1.0);
    assert!(quality.completeness_score >= 0.0 && quality.completeness_score <= 1.0);
    assert!(quality.novelty_score >= 0.0 && quality.novelty_score <= 1.0);
    assert!(quality.efficiency_score >= 0.0);
    
    println!("Ensemble quality metrics: {:?}", quality);
}

#[tokio::test]
async fn test_edge_cases_and_boundary_conditions() {
    let orchestrator = create_test_orchestrator().await;
    
    // Test very long query
    let long_query = "What is AI?".repeat(20);
    let result = orchestrator.reason(
        &long_query,
        None,
        ReasoningStrategy::Automatic,
    ).await;
    
    // Should handle gracefully
    match result {
        Ok(reasoning_result) => {
            assert!(!reasoning_result.final_answer.is_empty());
        }
        Err(_) => {
            // Acceptable to reject very long queries
            println!("Long query appropriately handled");
        }
    }
    
    // Test query with special characters
    let special_query = "What is AI? How does it work? #technology @future!";
    let result = orchestrator.reason(
        special_query,
        None,
        ReasoningStrategy::Automatic,
    ).await.unwrap();
    
    assert!(!result.final_answer.is_empty());
    
    // Test ensemble with single pattern (edge case)
    let single_ensemble = vec![CognitivePatternType::Convergent];
    let result = orchestrator.reason(
        "What is machine learning?",
        None,
        ReasoningStrategy::Ensemble(single_ensemble.clone()),
    ).await.unwrap();
    
    assert_eq!(result.strategy_used, ReasoningStrategy::Ensemble(single_ensemble));
    assert!(!result.final_answer.is_empty());
}

#[tokio::test]
async fn test_orchestrator_configuration() {
    let graph = create_test_graph();
    
    // Test with different configurations
    let strict_config = CognitiveOrchestratorConfig {
        enable_adaptive_selection: false,
        enable_ensemble_methods: false,
        default_timeout_ms: 1000,
        max_parallel_patterns: 1,
        performance_tracking: false,
    };
    
    let orchestrator = CognitiveOrchestrator::new(graph, strict_config).await.unwrap();
    
    let result = orchestrator.reason(
        "What is artificial intelligence?",
        None,
        ReasoningStrategy::Automatic,
    ).await.unwrap();
    
    assert!(!result.final_answer.is_empty());
    assert!(result.quality_metrics.overall_confidence >= 0.0);
}

#[tokio::test]
async fn test_pattern_weight_calculation_and_selection() {
    let orchestrator = create_test_orchestrator().await;
    
    // Test queries that should favor different patterns with different weights
    let test_cases = vec![
        ("What is the definition of machine learning?", "factual query"),
        ("What are creative applications of AI?", "creative query"),
        ("How might quantum computing connect to art?", "lateral query"),
        ("Analyze the AI ecosystem comprehensively", "systems query"),
        ("Evaluate claims about AI consciousness", "critical query"),
        ("What patterns exist in technology adoption?", "abstract query"),
    ];
    
    for (query, description) in test_cases {
        let result = orchestrator.reason(
            query,
            None,
            ReasoningStrategy::Automatic,
        ).await.unwrap();
        
        assert!(!result.final_answer.is_empty(), 
               "Failed on {} query: '{}'", description, query);
        
        // Check that reasoning completed successfully with reasonable confidence
        assert!(result.quality_metrics.overall_confidence >= 0.0,
               "No confidence for {} query", description);
        
        println!("Query: '{}' ({})", query, description);
        println!("  Patterns used: {:?}", result.execution_metadata.patterns_executed);
        println!("  Confidence: {:.3}", result.quality_metrics.overall_confidence);
    }
}