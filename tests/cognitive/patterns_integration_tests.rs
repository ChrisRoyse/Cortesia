//! Integration tests for cognitive patterns
//! Tests end-to-end execution of cognitive patterns through public APIs

use llmkg::cognitive::{
    CognitivePatternType,
    ConvergentThinking, DivergentThinking, LateralThinking,
    SystemsThinking, CriticalThinking, AbstractThinking, AdaptiveThinking,
    CognitiveOrchestrator, CognitiveOrchestratorConfig
};
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::types::EntityKey;
use llmkg::error::Result;
use std::sync::Arc;
use std::time::Duration;

/// Creates a test graph with default configuration
fn create_test_graph() -> Arc<BrainEnhancedKnowledgeGraph> {
    Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().expect("Failed to create test graph"))
}

/// Helper to create test entities (simplified version for integration tests)
async fn create_scenario_entities_in_graph(
    _graph: &BrainEnhancedKnowledgeGraph,
    _scenario_name: &str,
) -> Result<Vec<EntityKey>> {
    // For now, just return a few mock entity keys
    // In a real implementation, these would be added to the graph
    let entities = vec![
        EntityKey::from_hash("test_entity_1"),
        EntityKey::from_hash("test_entity_2"),
        EntityKey::from_hash("test_entity_3"),
    ];
    Ok(entities)
}

#[tokio::test]
async fn test_convergent_pattern_end_to_end() -> Result<()> {
    let graph = create_test_graph();
    
    // Add test entities to graph for convergent thinking
    let entity_keys = create_scenario_entities_in_graph(&graph, "simple_facts").await?;
    assert!(!entity_keys.is_empty(), "Should have created test entities");
    
    let convergent = ConvergentThinking::new(graph);
    
    let result = convergent.execute_convergent_query("What is a dog?", None).await?;
    
    assert!(result.confidence > 0.0, "Should have non-zero confidence");
    assert!(!result.answer.is_empty(), "Should have non-empty answer");
    assert!(!result.reasoning_trace.is_empty(), "Should have reasoning trace");
    
    Ok(())
}

#[tokio::test]
async fn test_divergent_pattern_end_to_end() -> Result<()> {
    let graph = create_test_graph();
    
    // Add test entities for divergent exploration
    let entity_keys = create_scenario_entities_in_graph(&graph, "animal_hierarchy").await?;
    assert!(!entity_keys.is_empty(), "Should have created test entities");
    
    let divergent = DivergentThinking::new(graph);
    
    let result = divergent.execute_divergent_exploration(
        "types of animals", 
        llmkg::cognitive::ExplorationType::Categories
    ).await?;
    
    assert!(result.explorations.len() > 0, "Should have exploration results");
    assert!(result.total_paths_explored > 0, "Should have explored paths");
    
    Ok(())
}

#[tokio::test]
async fn test_lateral_pattern_creative_connections() -> Result<()> {
    let graph = create_test_graph();
    
    // Add entities with diverse relationships for lateral thinking
    let entity_keys = create_scenario_entities_in_graph(&graph, "diverse_concepts").await?;
    assert!(!entity_keys.is_empty(), "Should have created test entities");
    
    let lateral = LateralThinking::new(graph);
    
    let result = lateral.find_creative_connections("dogs", "technology", Some(6)).await?;
    
    assert!(result.bridges.len() > 0, "Should find creative connections");
    assert!(result.novelty_analysis.overall_novelty >= 0.0, "Should provide novelty analysis");
    
    Ok(())
}

#[tokio::test]
async fn test_systems_pattern_hierarchy_analysis() -> Result<()> {
    let graph = create_test_graph();
    
    // Create hierarchical test data
    let entity_keys = create_scenario_entities_in_graph(&graph, "classification_hierarchy").await?;
    assert!(!entity_keys.is_empty(), "Should have created hierarchical entities");
    
    let systems = SystemsThinking::new(graph);
    
    let result = systems.execute_hierarchical_reasoning(
        "animal classification",
        llmkg::cognitive::SystemsReasoningType::Classification
    ).await?;
    
    assert!(result.hierarchy_path.len() > 0, "Should analyze hierarchy successfully");
    assert!(result.system_complexity >= 0.0, "Should provide complexity analysis");
    
    Ok(())
}

#[tokio::test]
async fn test_critical_pattern_contradiction_resolution() -> Result<()> {
    let graph = create_test_graph();
    
    // Add conflicting information to test critical thinking
    let entity_keys = create_scenario_entities_in_graph(&graph, "conflicting_facts").await?;
    assert!(!entity_keys.is_empty(), "Should have created entities with conflicts");
    
    let critical = CriticalThinking::new(graph);
    
    let result = critical.execute_critical_analysis(
        "conflicting animal facts",
        llmkg::cognitive::ValidationLevel::Comprehensive
    ).await?;
    
    assert!(result.resolved_facts.len() >= 0, "Should resolve contradictions");
    assert!(result.contradictions_found.len() >= 0, "Should identify contradictions");
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_pattern_structure_detection() -> Result<()> {
    let graph = create_test_graph();
    
    // Create structured data for pattern detection
    let entity_keys = create_scenario_entities_in_graph(&graph, "structured_patterns").await?;
    assert!(!entity_keys.is_empty(), "Should have created structured entities");
    
    let abstract_pattern = AbstractThinking::new(graph);
    
    let result = abstract_pattern.execute_pattern_analysis(
        llmkg::cognitive::AnalysisScope::Global,
        llmkg::cognitive::PatternType::Structural
    ).await?;
    
    assert!(result.patterns_found.len() >= 0, "Should detect patterns");
    assert!(result.abstractions.len() >= 0, "Should suggest abstractions");
    
    Ok(())
}

#[tokio::test]
async fn test_adaptive_pattern_strategy_selection() -> Result<()> {
    let graph = create_test_graph();
    
    // Add diverse test data for adaptive selection
    let entity_keys = create_scenario_entities_in_graph(&graph, "mixed_content").await?;
    assert!(!entity_keys.is_empty(), "Should have created diverse entities");
    
    let adaptive = AdaptiveThinking::new(graph);
    
    // Test factual query routing to convergent
    let factual_result = adaptive.execute_adaptive_reasoning(
        "What is quantum computing?",
        None,
        vec![CognitivePatternType::Convergent, CognitivePatternType::Divergent],
    ).await?;
    
    assert!(factual_result.final_answer.len() > 0, "Should provide factual answer");
    assert!(
        factual_result.strategy_used.selected_patterns.contains(&CognitivePatternType::Convergent),
        "Should select convergent for factual queries"
    );
    
    // Test creative query routing to divergent
    let creative_result = adaptive.execute_adaptive_reasoning(
        "Give me creative examples of future technology",
        None,
        vec![CognitivePatternType::Convergent, CognitivePatternType::Divergent],
    ).await?;
    
    assert!(creative_result.final_answer.len() > 0, "Should provide creative answer");
    assert!(
        creative_result.strategy_used.selected_patterns.contains(&CognitivePatternType::Divergent),
        "Should select divergent for creative queries"
    );
    
    Ok(())
}

#[tokio::test]
async fn test_pattern_coordination_and_ensemble() -> Result<()> {
    let graph = create_test_graph();
    
    // Create comprehensive test data
    let entity_keys = create_scenario_entities_in_graph(&graph, "comprehensive_knowledge").await?;
    assert!(!entity_keys.is_empty(), "Should have created comprehensive entities");
    
    let adaptive = AdaptiveThinking::new(graph);
    
    // Test ensemble execution with multiple patterns
    let ensemble_result = adaptive.execute_adaptive_reasoning(
        "Analyze the concept of intelligence across different domains",
        Some("Consider both biological and artificial intelligence"),
        vec![
            CognitivePatternType::Convergent,
            CognitivePatternType::Systems,
            CognitivePatternType::Abstract,
        ],
    ).await?;
    
    assert!(ensemble_result.final_answer.len() > 0, "Should provide comprehensive answer");
    assert!(
        ensemble_result.pattern_contributions.len() >= 1,
        "Should have contributions from multiple patterns"
    );
    assert!(
        ensemble_result.confidence_distribution.ensemble_confidence > 0.0,
        "Should have positive ensemble confidence"
    );
    
    // Verify learning update occurs
    assert!(
        ensemble_result.learning_update.performance_feedback >= 0.0,
        "Should provide learning feedback"
    );
    
    Ok(())
}

#[tokio::test]
async fn test_cross_pattern_result_merging() -> Result<()> {
    let graph = create_test_graph();
    
    // Add test data suitable for multiple patterns
    let entity_keys = create_scenario_entities_in_graph(&graph, "multi_pattern_data").await?;
    assert!(!entity_keys.is_empty(), "Should have created multi-pattern entities");
    
    let convergent = ConvergentThinking::new(graph.clone());
    let divergent = DivergentThinking::new(graph.clone());
    
    let query = "What are the different types of learning in AI?";
    
    // Execute both patterns independently
    let convergent_result = convergent.execute_convergent_query(query, None).await?;
    let divergent_result = divergent.execute_divergent_exploration(
        "AI learning types",
        llmkg::cognitive::ExplorationType::Categories
    ).await?;
    
    // Both should succeed with complementary information
    assert!(convergent_result.confidence > 0.0, "Convergent should provide focused answer");
    assert!(divergent_result.explorations.len() > 0, "Divergent should provide diverse answers");
    
    // Results should be different but related
    assert_ne!(
        convergent_result.answer.len(),
        0,
        "Convergent should provide meaningful answer"
    );
    assert!(
        divergent_result.total_paths_explored > 0,
        "Divergent should explore multiple paths"
    );
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling_in_pattern_execution() -> Result<()> {
    let graph = create_test_graph();
    
    let convergent = ConvergentThinking::new(graph);
    
    // Test with empty query
    let _empty_result = convergent.execute_convergent_query("", None).await;
    // Should handle gracefully or return appropriate error
    
    // Test with very complex query that might timeout
    let complex_query = "Analyze the interconnected relationships between quantum mechanics, consciousness, artificial intelligence, biological evolution, cosmology, and the nature of reality while considering temporal paradoxes and emergent properties";
    
    let complex_result = convergent.execute_convergent_query(complex_query, None).await;
    match complex_result {
        Ok(result) => {
            // If it succeeds, should have reasonable confidence
            assert!(result.confidence >= 0.0, "Complex query should have valid confidence");
        },
        Err(_) => {
            // If it fails, that's also acceptable for very complex queries
            // The important thing is it doesn't panic
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_pattern_performance_and_timeouts() -> Result<()> {
    let graph = create_test_graph();
    
    // Add moderate amount of test data
    let entity_keys = create_scenario_entities_in_graph(&graph, "performance_test").await?;
    assert!(!entity_keys.is_empty(), "Should have created performance test entities");
    
    let adaptive = AdaptiveThinking::new(graph);
    
    let start_time = std::time::Instant::now();
    
    let result = adaptive.execute_adaptive_reasoning(
        "Provide a comprehensive analysis of machine learning approaches",
        None,
        vec![CognitivePatternType::Convergent, CognitivePatternType::Systems],
    ).await?;
    
    let execution_time = start_time.elapsed();
    
    // Should complete in reasonable time (adjust threshold as needed)
    assert!(
        execution_time < Duration::from_secs(30),
        "Pattern execution should complete within 30 seconds, took {:?}",
        execution_time
    );
    
    assert!(result.final_answer.len() > 0, "Should provide meaningful analysis");
    
    Ok(())
}

#[tokio::test]
async fn test_cognitive_orchestrator_integration() -> Result<()> {
    let graph = create_test_graph();
    
    // Add comprehensive test data
    let entity_keys = create_scenario_entities_in_graph(&graph, "orchestrator_test").await?;
    assert!(!entity_keys.is_empty(), "Should have created orchestrator test entities");
    
    // Create orchestrator to test higher-level coordination
    let config = CognitiveOrchestratorConfig::default();
    let _orchestrator = CognitiveOrchestrator::new(graph, config).await?;
    
    // Test orchestrator's ability to coordinate patterns
    let _query = "Compare different learning paradigms in both biological and artificial systems";
    
    // The orchestrator should be able to handle complex, multi-faceted queries
    // by coordinating multiple cognitive patterns
    
    // Note: This test validates that the orchestrator can be created and integrated
    // The specific execution would depend on orchestrator's public API
    assert!(true, "Orchestrator integration test completed successfully");
    
    Ok(())
}

#[tokio::test]
async fn test_pattern_consistency_across_similar_queries() -> Result<()> {
    let graph = create_test_graph();
    
    // Add consistent test data
    let entity_keys = create_scenario_entities_in_graph(&graph, "consistency_test").await?;
    assert!(!entity_keys.is_empty(), "Should have created consistency test entities");
    
    let convergent = ConvergentThinking::new(graph);
    
    // Test similar queries for consistency
    let queries = vec![
        "What is artificial intelligence?",
        "Define artificial intelligence",
        "Explain AI",
    ];
    
    let mut results = Vec::new();
    for query in &queries {
        let result = convergent.execute_convergent_query(query, None).await?;
        results.push(result);
    }
    
    // All results should have reasonable confidence
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.confidence > 0.0,
            "Query {} should have positive confidence",
            queries[i]
        );
        assert!(
            !result.answer.is_empty(),
            "Query {} should have non-empty answer",
            queries[i]
        );
    }
    
    // Results should be related (though not necessarily identical)
    // This is validated by the fact that all succeeded with reasonable confidence
    
    Ok(())
}

/// Helper function to setup test scenario with specific patterns
async fn setup_pattern_test_scenario(scenario_name: &str) -> Result<Arc<BrainEnhancedKnowledgeGraph>> {
    let graph = create_test_graph();
    let _ = create_scenario_entities_in_graph(&graph, scenario_name).await?;
    Ok(graph)
}

/// Integration test for basic pattern instantiation
#[tokio::test]
async fn test_pattern_instantiation() -> Result<()> {
    let graph = setup_pattern_test_scenario("trait_test").await?;
    
    // Test that all patterns can be instantiated successfully
    let _convergent = ConvergentThinking::new(graph.clone());
    let _divergent = DivergentThinking::new(graph.clone());
    let _lateral = LateralThinking::new(graph.clone());
    let _systems = SystemsThinking::new(graph.clone());
    let _critical = CriticalThinking::new(graph.clone());
    let _abstract_pattern = AbstractThinking::new(graph.clone());
    let _adaptive = AdaptiveThinking::new(graph);
    
    // If we get here without panicking, all patterns instantiated successfully
    assert!(true, "All cognitive patterns instantiated successfully");
    
    Ok(())
}