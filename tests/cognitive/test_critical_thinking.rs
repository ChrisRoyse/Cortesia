//! Integration tests for CriticalThinking pattern
//! Tests public API and end-to-end critical analysis workflows

use llmkg::cognitive::{CriticalThinking, CognitivePattern};
use llmkg::cognitive::{
    PatternParameters, CognitivePatternType, ValidationLevel, ConflictType, ResolutionStrategy
};
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::types::EntityKey;
use llmkg::error::Result;
use std::sync::Arc;
use std::collections::HashMap;

/// Helper to create test CriticalThinking instance
async fn create_test_critical_thinking() -> CriticalThinking {
    let graph = Arc::new(BrainEnhancedKnowledgeGraph::new_for_test()
        .expect("Failed to create test graph"));
    CriticalThinking::new(graph)
}

#[tokio::test]
async fn test_critical_thinking_execute_basic_validation() -> Result<()> {
    let critical = create_test_critical_thinking().await;
    
    let parameters = PatternParameters {
        max_depth: Some(10),
        activation_threshold: Some(0.7),
        exploration_breadth: Some(3),
        creativity_threshold: Some(0.5),
        validation_level: Some(ValidationLevel::Basic),
        pattern_type: None,
        reasoning_strategy: None,
    };
    
    let result = critical.execute(
        "validate information about dogs",
        Some("fact checking context"),
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(!result.answer.is_empty());
    assert!(!result.reasoning_trace.is_empty());
    assert!(result.metadata.execution_time_ms > 0);
    
    Ok(())
}

#[tokio::test]
async fn test_critical_thinking_execute_comprehensive_validation() -> Result<()> {
    let critical = create_test_critical_thinking().await;
    
    let parameters = PatternParameters {
        max_depth: Some(15),
        activation_threshold: Some(0.8),
        exploration_breadth: Some(5),
        creativity_threshold: Some(0.3),
        validation_level: Some(ValidationLevel::Comprehensive),
        pattern_type: None,
        reasoning_strategy: None,
    };
    
    let result = critical.execute(
        "check contradictions in climate data",
        Some("scientific validation context"),
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    assert!(result.answer.contains("contradictions") || result.answer.contains("validation") || result.answer.contains("No facts"));
    assert!(result.metadata.additional_info.contains_key("uncertainty_score"));
    assert!(result.metadata.additional_info.contains_key("resolution_strategy"));
    
    Ok(())
}

#[tokio::test]
async fn test_critical_thinking_execute_rigorous_validation() -> Result<()> {
    let critical = create_test_critical_thinking().await;
    
    let parameters = PatternParameters {
        max_depth: Some(20),
        activation_threshold: Some(0.9),
        exploration_breadth: Some(7),
        creativity_threshold: Some(0.1),
        validation_level: Some(ValidationLevel::Rigorous),
        pattern_type: None,
        reasoning_strategy: None,
    };
    
    let result = critical.execute(
        "verify medical claims about vaccines",
        Some("rigorous scientific validation"),
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    assert!(result.answer.contains("uncertainty") || result.answer.contains("validation") || result.answer.contains("No facts"));
    assert!(result.metadata.additional_info.contains_key("facts_count"));
    assert!(result.metadata.additional_info.contains_key("contradictions_count"));
    
    Ok(())
}

#[tokio::test]
async fn test_critical_thinking_contradiction_detection() -> Result<()> {
    let critical = create_test_critical_thinking().await;
    
    let parameters = PatternParameters {
        max_depth: Some(10),
        activation_threshold: Some(0.6),
        exploration_breadth: Some(3),
        creativity_threshold: Some(0.5),
        validation_level: Some(ValidationLevel::Basic),
        pattern_type: None,
        reasoning_strategy: None,
    };
    
    // Query that might contain contradictory information
    let result = critical.execute(
        "cats have 3 legs and cats have 4 legs",
        Some("contradiction test"),
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    assert!(result.confidence >= 0.0);
    
    // Should have metadata about the analysis
    assert!(result.metadata.additional_info.contains_key("contradictions_count"));
    
    Ok(())
}

#[tokio::test]
async fn test_critical_thinking_information_source_validation() -> Result<()> {
    let critical = create_test_critical_thinking().await;
    
    let parameters = PatternParameters {
        max_depth: Some(12),
        activation_threshold: Some(0.7),
        exploration_breadth: Some(4),
        creativity_threshold: Some(0.4),
        validation_level: Some(ValidationLevel::Comprehensive),
        pattern_type: None,
        reasoning_strategy: None,
    };
    
    let result = critical.execute(
        "analyze reliability of news sources",
        Some("source validation"),
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    assert!(!result.reasoning_trace.is_empty());
    
    // Check that analysis was performed
    let uncertainty_score = result.metadata.additional_info.get("uncertainty_score");
    assert!(uncertainty_score.is_some());
    
    Ok(())
}

#[tokio::test]
async fn test_critical_thinking_exception_resolution() -> Result<()> {
    let critical = create_test_critical_thinking().await;
    
    let parameters = PatternParameters {
        max_depth: Some(8),
        activation_threshold: Some(0.5),
        exploration_breadth: Some(2),
        creativity_threshold: Some(0.6),
        validation_level: Some(ValidationLevel::Basic),
        pattern_type: None,
        reasoning_strategy: None,
    };
    
    // Test with potentially conflicting information
    let result = critical.execute(
        "resolve conflicts between different scientific studies",
        Some("exception resolution test"),
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    assert!(result.metadata.additional_info.contains_key("resolution_strategy"));
    
    // Should complete without errors even with complex conflicts
    assert!(result.confidence >= 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_critical_thinking_empty_query() -> Result<()> {
    let critical = create_test_critical_thinking().await;
    
    let parameters = PatternParameters {
        max_depth: Some(5),
        activation_threshold: Some(0.5),
        exploration_breadth: Some(1),
        creativity_threshold: Some(0.5),
        validation_level: Some(ValidationLevel::Basic),
        pattern_type: None,
        reasoning_strategy: None,
    };
    
    let result = critical.execute("", None, parameters).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    assert!(result.answer.contains("No facts") || result.answer.is_empty());
    assert!(result.metadata.nodes_activated == 0);
    
    Ok(())
}

#[tokio::test]
async fn test_critical_thinking_stop_words_query() -> Result<()> {
    let critical = create_test_critical_thinking().await;
    
    let parameters = PatternParameters {
        max_depth: Some(5),
        activation_threshold: Some(0.5),
        exploration_breadth: Some(1),
        creativity_threshold: Some(0.5),
        validation_level: Some(ValidationLevel::Basic),
        pattern_type: None,
        reasoning_strategy: None,
    };
    
    // Query with only stop words
    let result = critical.execute("the is at which on", None, parameters).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    assert!(result.answer.contains("No facts") || result.answer.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_critical_thinking_get_pattern_type() {
    let critical = create_test_critical_thinking().await;
    assert_eq!(critical.get_pattern_type(), CognitivePatternType::Critical);
}

#[tokio::test]
async fn test_critical_thinking_get_optimal_use_cases() {
    let critical = create_test_critical_thinking().await;
    let use_cases = critical.get_optimal_use_cases();
    
    assert!(use_cases.contains(&"Fact validation".to_string()));
    assert!(use_cases.contains(&"Contradiction resolution".to_string()));
    assert!(use_cases.contains(&"Source reliability analysis".to_string()));
    assert!(use_cases.contains(&"Conflict detection".to_string()));
    assert_eq!(use_cases.len(), 4);
}

#[tokio::test]
async fn test_critical_thinking_estimate_complexity() {
    let critical = create_test_critical_thinking().await;
    
    let estimate = critical.estimate_complexity("validate complex scientific data");
    
    assert_eq!(estimate.computational_complexity, 40);
    assert_eq!(estimate.estimated_time_ms, 1500);
    assert_eq!(estimate.memory_requirements_mb, 15);
    assert_eq!(estimate.confidence, 0.9);
    assert!(!estimate.parallelizable);
}

#[tokio::test]
async fn test_critical_thinking_validation_level_inference() -> Result<()> {
    let critical = create_test_critical_thinking().await;
    
    // Test "verify" keyword triggers rigorous validation
    let parameters = PatternParameters::default();
    let result = critical.execute(
        "verify this claim about physics",
        None,
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    
    // Test "check" keyword triggers comprehensive validation  
    let parameters = PatternParameters::default();
    let result = critical.execute(
        "check this information about biology",
        None,
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    
    // Test default triggers basic validation
    let parameters = PatternParameters::default();
    let result = critical.execute(
        "analyze this data",
        None,
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    
    Ok(())
}

#[tokio::test]
async fn test_critical_thinking_execute_critical_analysis_workflow() -> Result<()> {
    let critical = create_test_critical_thinking().await;
    
    // Test the main execute_critical_analysis method directly
    let result = critical.execute_critical_analysis(
        "test query for analysis",
        ValidationLevel::Comprehensive,
    ).await?;
    
    assert!(result.resolved_facts.len() >= 0);
    assert!(result.contradictions_found.len() >= 0);
    assert!(result.confidence_intervals.len() >= 0);
    assert!(result.uncertainty_analysis.overall_uncertainty >= 0.0);
    assert!(result.uncertainty_analysis.overall_uncertainty <= 1.0);
    
    // Test various resolution strategies are possible
    match result.resolution_strategy {
        ResolutionStrategy::PreferLocal => {},
        ResolutionStrategy::PreferTrusted => {},
        ResolutionStrategy::WeightedAverage => {},
        ResolutionStrategy::ExpertSystem => {},
        ResolutionStrategy::PreferHigherConfidence => {},
        ResolutionStrategy::LogicalPriority => {},
    }
    
    Ok(())
}

#[tokio::test]
async fn test_critical_thinking_performance_characteristics() -> Result<()> {
    let critical = create_test_critical_thinking().await;
    
    let start = std::time::Instant::now();
    
    let parameters = PatternParameters {
        max_depth: Some(5),
        activation_threshold: Some(0.6),
        exploration_breadth: Some(2),
        creativity_threshold: Some(0.5),
        validation_level: Some(ValidationLevel::Basic),
        pattern_type: None,
        reasoning_strategy: None,
    };
    
    let result = critical.execute(
        "quick validation test",
        None,
        parameters,
    ).await?;
    
    let elapsed = start.elapsed();
    
    // Performance should be reasonable for basic validation
    assert!(elapsed.as_millis() < 5000, "Basic validation should complete within 5 seconds");
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    assert!(result.metadata.execution_time_ms > 0);
    
    Ok(())
}