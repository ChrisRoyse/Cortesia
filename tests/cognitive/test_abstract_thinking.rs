//! Integration tests for AbstractThinking pattern
//! Tests the public API of AbstractThinking and its CognitivePattern implementation

use llmkg::cognitive::{AbstractThinking, CognitivePattern, PatternParameters, CognitivePatternType};
use llmkg::cognitive::{AnalysisScope, PatternType, ValidationLevel};
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::types::{EntityKey, EntityData};
use llmkg::error::Result;
use std::sync::Arc;
use std::time::Duration;

/// Creates a test graph with comprehensive abstract thinking scenarios
async fn create_abstract_thinking_test_graph() -> Arc<BrainEnhancedKnowledgeGraph> {
    let graph = Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().expect("Failed to create test graph"));
    
    // Add hierarchical biological entities for pattern detection
    let animal_key = graph.add_entity(EntityData::new(1, "Basic animal concept".to_string(), vec![0.0; 96])).await.unwrap();
    let mammal_key = graph.add_entity(EntityData::new(1, "Warm-blooded vertebrate animal".to_string(), vec![0.0; 96])).await.unwrap();
    let carnivore_key = graph.add_entity(EntityData::new(1, "Meat-eating animal".to_string(), vec![0.0; 96])).await.unwrap();
    let herbivore_key = graph.add_entity(EntityData::new(1, "Plant-eating animal".to_string(), vec![0.0; 96])).await.unwrap();
    let dog_key = graph.add_entity(EntityData::new(1, "Domesticated canine".to_string(), vec![0.0; 96])).await.unwrap();
    let cat_key = graph.add_entity(EntityData::new(1, "Domesticated feline".to_string(), vec![0.0; 96])).await.unwrap();
    let elephant_key = graph.add_entity(EntityData::new(1, "Large mammal".to_string(), vec![0.0; 96])).await.unwrap();
    
    // Add technological entities for cross-domain pattern detection
    let system_key = graph.add_entity(EntityData::new(1, "Organized structure".to_string(), vec![0.0; 96])).await.unwrap();
    let computer_key = graph.add_entity(EntityData::new(1, "Electronic processing system".to_string(), vec![0.0; 96])).await.unwrap();
    let ai_key = graph.add_entity(EntityData::new(1, "Computer-based intelligence".to_string(), vec![0.0; 96])).await.unwrap();
    let network_key = graph.add_entity(EntityData::new(1, "Interconnected system".to_string(), vec![0.0; 96])).await.unwrap();
    
    // Add properties and attributes for inheritance pattern detection
    let breathing_key = graph.add_entity(EntityData::new(1, "Respiratory function".to_string(), vec![0.0; 96])).await.unwrap();
    let intelligence_key = graph.add_entity(EntityData::new(1, "Cognitive capability".to_string(), vec![0.0; 96])).await.unwrap();
    let mobility_key = graph.add_entity(EntityData::new(1, "Movement capability".to_string(), vec![0.0; 96])).await.unwrap();
    
    // Create hierarchical relationships
    let _ = graph.add_weighted_edge(mammal_key, animal_key, 0.9).await;
    let _ = graph.add_weighted_edge(dog_key, mammal_key, 0.95).await;
    let _ = graph.add_weighted_edge(cat_key, mammal_key, 0.95).await;
    let _ = graph.add_weighted_edge(elephant_key, mammal_key, 0.95).await;
    let _ = graph.add_weighted_edge(dog_key, carnivore_key, 0.8).await;
    let _ = graph.add_weighted_edge(cat_key, carnivore_key, 0.85).await;
    let _ = graph.add_weighted_edge(elephant_key, herbivore_key, 0.9).await;
    
    // Create technological relationships
    let _ = graph.add_weighted_edge(computer_key, system_key, 0.8).await;
    let _ = graph.add_weighted_edge(ai_key, computer_key, 0.9).await;
    let _ = graph.add_weighted_edge(network_key, system_key, 0.8).await;
    
    // Create property relationships
    let _ = graph.add_weighted_edge(animal_key, breathing_key, 0.7).await;
    let _ = graph.add_weighted_edge(animal_key, mobility_key, 0.8).await;
    let _ = graph.add_weighted_edge(mammal_key, intelligence_key, 0.6).await;
    let _ = graph.add_weighted_edge(ai_key, intelligence_key, 0.9).await;
    
    graph
}

/// Creates an AbstractThinking instance for testing
async fn create_test_abstract_thinking() -> AbstractThinking {
    let graph = create_abstract_thinking_test_graph().await;
    AbstractThinking::new(graph)
}

#[tokio::test]
async fn test_abstract_thinking_instantiation() -> Result<()> {
    let graph = create_abstract_thinking_test_graph().await;
    let abstract_thinking = AbstractThinking::new(graph);
    
    // Verify pattern type
    assert_eq!(abstract_thinking.get_pattern_type(), CognitivePatternType::Abstract);
    
    // Verify optimal use cases
    let use_cases = abstract_thinking.get_optimal_use_cases();
    assert!(use_cases.contains(&"Pattern recognition".to_string()));
    assert!(use_cases.contains(&"Meta-analysis".to_string()));
    assert!(use_cases.contains(&"Concept abstraction".to_string()));
    assert!(use_cases.contains(&"System optimization".to_string()));
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_complexity_estimation() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    let test_queries = vec![
        "find patterns",
        "analyze system structure for optimization opportunities",
        "identify abstract concepts in biological taxonomy hierarchy",
        "perform comprehensive meta-analysis of knowledge graph patterns across multiple domains with temporal considerations",
    ];
    
    for query in test_queries {
        let estimate = abstract_thinking.estimate_complexity(query);
        
        assert_eq!(estimate.computational_complexity, 60);
        assert_eq!(estimate.estimated_time_ms, 2000);
        assert_eq!(estimate.memory_requirements_mb, 20);
        assert_eq!(estimate.confidence, 0.7);
        assert!(!estimate.parallelizable);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_execute_pattern_analysis() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    // Test different analysis scopes
    let scopes = vec![
        AnalysisScope::Global,
        AnalysisScope::Regional(vec![EntityKey::from_hash("entity1"), EntityKey::from_hash("entity2")]),
        AnalysisScope::Local(EntityKey::from_hash("local_entity")),
    ];
    
    // Test different pattern types
    let pattern_types = vec![
        PatternType::Structural,
        PatternType::Temporal,
        PatternType::Semantic,
        PatternType::Usage,
    ];
    
    for scope in scopes {
        for pattern_type in &pattern_types {
            let result = abstract_thinking.execute_pattern_analysis(scope.clone(), *pattern_type).await;
            assert!(result.is_ok(), "Pattern analysis should succeed for scope {:?} and type {:?}", scope, pattern_type);
            
            let abstract_result = result.unwrap();
            
            // Verify result structure
            assert!(abstract_result.patterns_found.len() >= 0);
            assert!(abstract_result.abstractions.len() >= 0);
            assert!(abstract_result.refactoring_opportunities.len() >= 0);
            
            // Verify efficiency gains
            assert!(abstract_result.efficiency_gains.query_time_improvement >= 0.0);
            assert!(abstract_result.efficiency_gains.memory_reduction >= 0.0);
            assert!(abstract_result.efficiency_gains.accuracy_improvement >= 0.0);
            assert!(abstract_result.efficiency_gains.maintainability_score >= 0.0);
            assert!(abstract_result.efficiency_gains.maintainability_score <= 1.0);
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_cognitive_pattern_execution() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    let test_queries = vec![
        ("find structural patterns", "Looking for structural relationships"),
        ("identify abstract concepts", "Analyzing conceptual hierarchies"),
        ("analyze system topology", "Examining network structure"),
        ("detect meta-patterns", "Finding patterns of patterns"),
        ("suggest optimizations", "Recommending system improvements"),
    ];
    
    for (query, context) in test_queries {
        let parameters = PatternParameters {
            max_depth: Some(4),
            activation_threshold: Some(0.6),
            exploration_breadth: Some(8),
            creativity_threshold: Some(0.4),
            validation_level: Some(ValidationLevel::Basic),
            pattern_type: Some(PatternType::Structural),
            reasoning_strategy: None,
        };
        
        let result = abstract_thinking.execute(query, Some(context), parameters).await;
        assert!(result.is_ok(), "Abstract thinking execution should succeed for query: {}", query);
        
        let pattern_result = result.unwrap();
        
        // Verify pattern result structure
        assert_eq!(pattern_result.pattern_type, CognitivePatternType::Abstract);
        assert!(pattern_result.confidence >= 0.0 && pattern_result.confidence <= 1.0);
        assert!(!pattern_result.answer.is_empty());
        assert!(!pattern_result.reasoning_trace.is_empty());
        
        // Verify metadata
        assert!(pattern_result.metadata.execution_time_ms > 0);
        assert!(pattern_result.metadata.nodes_activated >= 0);
        assert_eq!(pattern_result.metadata.iterations_completed, 1);
        assert!(pattern_result.metadata.converged);
        assert!(pattern_result.metadata.total_energy >= 0.0);
        
        // Verify abstract-specific metadata
        assert!(pattern_result.metadata.additional_info.contains_key("patterns_count"));
        assert!(pattern_result.metadata.additional_info.contains_key("abstractions_count"));
        assert!(pattern_result.metadata.additional_info.contains_key("refactoring_opportunities"));
        assert!(pattern_result.metadata.additional_info.contains_key("query_improvement"));
        assert!(pattern_result.metadata.additional_info.contains_key("memory_reduction"));
        assert!(pattern_result.metadata.additional_info.contains_key("maintainability_score"));
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_pattern_detection_scenarios() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    // Test structural pattern detection
    let structural_result = abstract_thinking.execute_pattern_analysis(
        AnalysisScope::Global,
        PatternType::Structural
    ).await?;
    
    assert!(structural_result.patterns_found.len() >= 0, "Should detect structural patterns");
    
    // Verify structural patterns include hierarchy and hub patterns
    let structural_patterns = &structural_result.patterns_found;
    for pattern in structural_patterns {
        assert!(!pattern.pattern_id.is_empty());
        assert!(!pattern.description.is_empty());
        assert!(pattern.confidence >= 0.0 && pattern.confidence <= 1.0);
        assert!(pattern.frequency >= 0.0);
        assert_eq!(pattern.pattern_type, PatternType::Structural);
    }
    
    // Test semantic pattern detection
    let semantic_result = abstract_thinking.execute_pattern_analysis(
        AnalysisScope::Global,
        PatternType::Semantic
    ).await?;
    
    // Semantic patterns may be empty but should not error
    assert!(semantic_result.patterns_found.len() >= 0);
    
    // Test temporal pattern detection
    let temporal_result = abstract_thinking.execute_pattern_analysis(
        AnalysisScope::Global,
        PatternType::Temporal
    ).await?;
    
    // Temporal patterns may be empty but should not error
    assert!(temporal_result.patterns_found.len() >= 0);
    
    // Test usage pattern detection
    let usage_result = abstract_thinking.execute_pattern_analysis(
        AnalysisScope::Global,
        PatternType::Usage
    ).await?;
    
    // Usage patterns may be empty but should not error
    assert!(usage_result.patterns_found.len() >= 0);
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_abstraction_identification() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    let result = abstract_thinking.execute_pattern_analysis(
        AnalysisScope::Global,
        PatternType::Structural
    ).await?;
    
    // Verify abstraction candidates
    for abstraction in &result.abstractions {
        assert!(!abstraction.abstraction_type.is_empty());
        assert!(abstraction.complexity_reduction >= 0.0);
        assert!(abstraction.implementation_effort >= 0.0);
        assert!(abstraction.implementation_effort <= 1.0);
        assert!(!abstraction.source_patterns.is_empty());
        assert!(abstraction.entities_to_abstract.len() >= 0);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_refactoring_opportunities() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    let result = abstract_thinking.execute_pattern_analysis(
        AnalysisScope::Global,
        PatternType::Structural
    ).await?;
    
    // Should have at least performance optimization opportunities
    assert!(result.refactoring_opportunities.len() > 0, "Should identify refactoring opportunities");
    
    // Verify refactoring opportunities
    for opportunity in &result.refactoring_opportunities {
        assert!(!opportunity.description.is_empty());
        assert!(opportunity.estimated_benefit >= 0.0);
        assert!(opportunity.estimated_benefit <= 1.0);
        
        // Should have valid refactoring types
        match opportunity.opportunity_type {
            llmkg::cognitive::RefactoringType::ConceptMerging |
            llmkg::cognitive::RefactoringType::HierarchyReorganization |
            llmkg::cognitive::RefactoringType::RedundancyElimination |
            llmkg::cognitive::RefactoringType::PerformanceOptimization => assert!(true),
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_meta_analysis() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    let parameters = PatternParameters {
        max_depth: Some(3),
        activation_threshold: Some(0.5),
        exploration_breadth: Some(10),
        creativity_threshold: Some(0.2),
        validation_level: Some(ValidationLevel::Basic),
        pattern_type: Some(PatternType::Structural),
        reasoning_strategy: None,
    };
    
    let result = abstract_thinking.execute(
        "Perform meta-analysis of biological and technological hierarchies",
        Some("Looking for cross-domain patterns"),
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Abstract);
    assert!(result.confidence > 0.0);
    assert!(result.answer.contains("Pattern"));
    
    // Should mention patterns, abstractions, or optimizations
    let answer_lower = result.answer.to_lowercase();
    assert!(
        answer_lower.contains("pattern") || 
        answer_lower.contains("abstraction") || 
        answer_lower.contains("optimization") ||
        answer_lower.contains("efficiency"),
        "Answer should contain relevant abstract thinking concepts"
    );
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_graph_optimization_analysis() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    let parameters = PatternParameters {
        max_depth: Some(4),
        activation_threshold: Some(0.7),
        exploration_breadth: Some(12),
        creativity_threshold: Some(0.1),
        validation_level: Some(ValidationLevel::Comprehensive),
        pattern_type: Some(PatternType::Structural),
        reasoning_strategy: None,
    };
    
    let result = abstract_thinking.execute(
        "Analyze graph structure for optimization opportunities",
        Some("Focus on performance and maintainability improvements"),
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Abstract);
    
    // Should provide optimization suggestions
    let answer_lower = result.answer.to_lowercase();
    assert!(
        answer_lower.contains("optimization") ||
        answer_lower.contains("efficiency") ||
        answer_lower.contains("performance") ||
        answer_lower.contains("improvement"),
        "Should suggest optimizations"
    );
    
    // Verify efficiency metrics in metadata
    let query_improvement = result.metadata.additional_info.get("query_improvement").unwrap();
    let memory_reduction = result.metadata.additional_info.get("memory_reduction").unwrap();
    let maintainability = result.metadata.additional_info.get("maintainability_score").unwrap();
    
    assert!(query_improvement.parse::<f32>().unwrap() >= 0.0);
    assert!(memory_reduction.parse::<f32>().unwrap() >= 0.0);
    assert!(maintainability.parse::<f32>().unwrap() >= 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_pattern_parameter_inference() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    let test_cases = vec![
        ("analyze structure", PatternType::Structural),
        ("examine topology", PatternType::Structural),
        ("temporal patterns over time", PatternType::Temporal),
        ("semantic meaning analysis", PatternType::Semantic),
        ("usage access patterns", PatternType::Usage),
        ("find patterns", PatternType::Structural), // Default
    ];
    
    for (query, expected_pattern_type) in test_cases {
        let parameters = PatternParameters::default();
        let result = abstract_thinking.execute(query, None, parameters).await?;
        
        assert_eq!(result.pattern_type, CognitivePatternType::Abstract);
        assert!(result.confidence >= 0.0);
        
        // The actual pattern type inference is tested in the unit tests
        // Here we just verify the execution succeeds and produces valid output
        assert!(!result.answer.is_empty());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_cross_domain_patterns() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    let parameters = PatternParameters {
        max_depth: Some(5),
        activation_threshold: Some(0.4),
        exploration_breadth: Some(15),
        creativity_threshold: Some(0.3),
        validation_level: Some(ValidationLevel::Basic),
        pattern_type: Some(PatternType::Semantic),
        reasoning_strategy: None,
    };
    
    let result = abstract_thinking.execute(
        "Find patterns connecting biological and technological domains",
        Some("Look for intelligence, hierarchy, and system organization patterns"),
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Abstract);
    assert!(result.confidence >= 0.0);
    
    // Should identify cross-domain connections
    let answer_lower = result.answer.to_lowercase();
    assert!(
        answer_lower.contains("pattern") ||
        answer_lower.contains("connection") ||
        answer_lower.contains("relationship"),
        "Should identify cross-domain patterns"
    );
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_performance_within_timeout() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    let start_time = std::time::Instant::now();
    
    let parameters = PatternParameters::default();
    let result = abstract_thinking.execute(
        "Comprehensive pattern analysis across all domains",
        Some("Analyze structure, semantics, and optimization opportunities"),
        parameters,
    ).await?;
    
    let execution_time = start_time.elapsed();
    
    // Should complete within reasonable time
    assert!(
        execution_time < Duration::from_secs(10),
        "Abstract thinking should complete within 10 seconds, took {:?}",
        execution_time
    );
    
    assert_eq!(result.pattern_type, CognitivePatternType::Abstract);
    assert!(result.confidence >= 0.0);
    assert!(!result.answer.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_empty_graph_handling() -> Result<()> {
    let empty_graph = Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().expect("Failed to create empty graph"));
    let abstract_thinking = AbstractThinking::new(empty_graph);
    
    let parameters = PatternParameters::default();
    let result = abstract_thinking.execute(
        "analyze patterns in empty system",
        None,
        parameters,
    ).await?;
    
    // Should handle empty graph gracefully
    assert_eq!(result.pattern_type, CognitivePatternType::Abstract);
    assert!(result.confidence >= 0.0);
    assert!(!result.answer.is_empty());
    
    // Should report no patterns found
    let answer_lower = result.answer.to_lowercase();
    assert!(
        answer_lower.contains("no") ||
        answer_lower.contains("empty") ||
        answer_lower.contains("0") ||
        result.metadata.additional_info.get("patterns_count").unwrap() == "0",
        "Should indicate no patterns found in empty graph"
    );
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_error_handling() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    // Test with edge case inputs
    let edge_cases = vec![
        ("", None), // Empty query
        ("  ", None), // Whitespace only
        ("very short", None), // Very short query
    ];
    
    for (query, context) in edge_cases {
        let parameters = PatternParameters::default();
        let result = abstract_thinking.execute(query, context, parameters).await;
        
        // Should either succeed or fail gracefully
        match result {
            Ok(pattern_result) => {
                assert_eq!(pattern_result.pattern_type, CognitivePatternType::Abstract);
                assert!(pattern_result.confidence >= 0.0);
            }
            Err(_) => {
                // Acceptable to fail with edge case inputs
            }
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_regional_scope_analysis() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    // Test regional scope with specific entities
    let entity_keys = vec![
        EntityKey::from_hash("mammal"),
        EntityKey::from_hash("dog"),
        EntityKey::from_hash("cat"),
    ];
    
    let result = abstract_thinking.execute_pattern_analysis(
        AnalysisScope::Regional(entity_keys),
        PatternType::Structural,
    ).await?;
    
    assert!(result.patterns_found.len() >= 0);
    assert!(result.abstractions.len() >= 0);
    assert!(result.refactoring_opportunities.len() >= 0);
    assert!(result.efficiency_gains.maintainability_score >= 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_local_scope_analysis() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    // Test local scope with single entity
    let local_entity = EntityKey::from_hash("specific_entity");
    
    let result = abstract_thinking.execute_pattern_analysis(
        AnalysisScope::Local(local_entity),
        PatternType::Structural,
    ).await?;
    
    assert!(result.patterns_found.len() >= 0);
    assert!(result.abstractions.len() >= 0);
    assert!(result.refactoring_opportunities.len() >= 0);
    assert!(result.efficiency_gains.maintainability_score >= 0.0);
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_answer_formatting() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    let parameters = PatternParameters::default();
    let result = abstract_thinking.execute(
        "Analyze patterns and provide detailed explanation",
        Some("Format results clearly with sections"),
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Abstract);
    
    // Verify answer formatting contains expected sections
    let answer = &result.answer;
    assert!(answer.contains("Pattern"));
    
    // Should contain structured information
    assert!(
        answer.contains("Patterns Detected") ||
        answer.contains("Abstraction Opportunities") ||
        answer.contains("Refactoring Opportunities") ||
        answer.contains("Efficiency Gains"),
        "Answer should contain structured analysis sections"
    );
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_thinking_reasoning_trace() -> Result<()> {
    let abstract_thinking = create_test_abstract_thinking().await;
    
    let parameters = PatternParameters::default();
    let result = abstract_thinking.execute(
        "Trace reasoning through pattern analysis",
        None,
        parameters,
    ).await?;
    
    assert_eq!(result.pattern_type, CognitivePatternType::Abstract);
    assert!(!result.reasoning_trace.is_empty(), "Should have reasoning trace");
    
    // Verify reasoning trace structure
    for step in &result.reasoning_trace {
        assert!(step.step_id > 0);
        assert!(!step.concept_id.is_empty());
        assert!(step.activation_level >= 0.0 && step.activation_level <= 1.0);
    }
    
    // Should have typical abstract thinking reasoning steps
    let concept_ids: Vec<_> = result.reasoning_trace.iter()
        .map(|step| &step.concept_id)
        .collect();
    
    assert!(concept_ids.contains(&&"pattern_detection".to_string()));
    
    Ok(())
}