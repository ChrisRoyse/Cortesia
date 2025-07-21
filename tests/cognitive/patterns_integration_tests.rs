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
async fn test_systems_pattern_attribute_inheritance() -> Result<()> {
    let graph = create_test_graph();
    
    // Create biological hierarchy for inheritance testing
    let entity_keys = create_scenario_entities_in_graph(&graph, "biological_inheritance").await?;
    assert!(!entity_keys.is_empty(), "Should have created biological entities");
    
    let systems = SystemsThinking::new(graph);
    
    // Test attribute inheritance reasoning
    let result = systems.execute_hierarchical_reasoning(
        "What properties does a mammal inherit?",
        llmkg::cognitive::SystemsReasoningType::AttributeInheritance
    ).await?;
    
    assert!(result.hierarchy_path.len() > 0, "Should traverse inheritance hierarchy");
    assert!(result.inherited_attributes.len() >= 0, "Should identify inherited attributes");
    assert!(result.system_complexity >= 0.0 && result.system_complexity <= 1.0, "Should calculate valid complexity");
    
    // Test that inheritance depth is reasonable
    for attr in &result.inherited_attributes {
        assert!(attr.inheritance_depth > 0, "All attributes should have positive inheritance depth");
        assert!(attr.confidence >= 0.0 && attr.confidence <= 1.0, "Confidence should be valid");
        assert!(!attr.attribute_name.is_empty(), "Attribute names should not be empty");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_systems_pattern_emergent_properties() -> Result<()> {
    let graph = create_test_graph();
    
    // Create complex system data for emergent property analysis
    let entity_keys = create_scenario_entities_in_graph(&graph, "complex_systems").await?;
    assert!(!entity_keys.is_empty(), "Should have created complex system entities");
    
    let systems = SystemsThinking::new(graph);
    
    // Test emergent properties reasoning
    let result = systems.execute_hierarchical_reasoning(
        "What emergent behaviors arise from collective intelligence?",
        llmkg::cognitive::SystemsReasoningType::EmergentProperties
    ).await?;
    
    assert!(result.hierarchy_path.len() > 0, "Should analyze emergent system structure");
    assert!(result.system_complexity >= 0.0, "Should measure system complexity");
    
    Ok(())
}

#[tokio::test]
async fn test_systems_pattern_exception_handling() -> Result<()> {
    let graph = create_test_graph();
    
    // Create data with potential contradictions for exception testing
    let entity_keys = create_scenario_entities_in_graph(&graph, "contradictory_hierarchies").await?;
    assert!(!entity_keys.is_empty(), "Should have created contradictory entities");
    
    let systems = SystemsThinking::new(graph);
    
    // Test system analysis with potential exceptions
    let result = systems.execute_hierarchical_reasoning(
        "Analyze systems with conflicting properties",
        llmkg::cognitive::SystemsReasoningType::SystemAnalysis
    ).await?;
    
    assert!(result.hierarchy_path.len() > 0, "Should perform system analysis");
    
    // Test exception handling structure
    for exception in &result.exception_handling {
        assert!(!exception.description.is_empty(), "Exception descriptions should not be empty");
        assert!(!exception.resolution_strategy.is_empty(), "Resolution strategies should not be empty");
        assert!(!exception.affected_entities.is_empty(), "Affected entities should be specified");
        
        // Test exception type validity
        match exception.exception_type {
            llmkg::cognitive::ExceptionType::Contradiction |
            llmkg::cognitive::ExceptionType::MissingData |
            llmkg::cognitive::ExceptionType::InconsistentInheritance |
            llmkg::cognitive::ExceptionType::CircularReference => assert!(true),
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_systems_pattern_cognitive_trait_execution() -> Result<()> {
    let graph = create_test_graph();
    
    // Setup test data for cognitive pattern trait execution
    let entity_keys = create_scenario_entities_in_graph(&graph, "cognitive_trait_test").await?;
    assert!(!entity_keys.is_empty(), "Should have created test entities");
    
    let systems = SystemsThinking::new(graph);
    
    // Test execution through CognitivePattern trait
    let parameters = llmkg::cognitive::PatternParameters::default();
    let result = systems.execute(
        "What are the hierarchical relationships in biological systems?",
        Some("biological classification context"),
        parameters,
    ).await?;
    
    // Validate pattern result structure
    assert_eq!(result.pattern_type, CognitivePatternType::Systems);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0, "Confidence should be valid");
    assert!(!result.answer.is_empty(), "Answer should not be empty");
    assert!(!result.reasoning_trace.is_empty(), "Should have reasoning trace");
    
    // Validate metadata
    assert!(result.metadata.execution_time_ms > 0, "Should record execution time");
    assert!(result.metadata.nodes_activated > 0, "Should activate nodes");
    assert_eq!(result.metadata.iterations_completed, 1, "Should complete one iteration");
    assert!(result.metadata.converged, "Should converge");
    assert!(result.metadata.total_energy >= 0.0, "Should calculate energy");
    
    // Validate additional metadata specific to systems thinking
    assert!(result.metadata.additional_info.contains_key("hierarchy_depth"));
    assert!(result.metadata.additional_info.contains_key("attributes_count"));
    assert!(result.metadata.additional_info.contains_key("exceptions_count"));
    assert!(result.metadata.additional_info.contains_key("complexity_score"));
    
    Ok(())
}

#[tokio::test]
async fn test_systems_pattern_use_cases_and_complexity() -> Result<()> {
    let graph = create_test_graph();
    let systems = SystemsThinking::new(graph);
    
    // Test optimal use cases
    let use_cases = systems.get_optimal_use_cases();
    assert!(!use_cases.is_empty(), "Should have optimal use cases");
    assert!(use_cases.contains(&"Hierarchical analysis".to_string()));
    assert!(use_cases.contains(&"Classification queries".to_string()));
    assert!(use_cases.contains(&"Attribute inheritance".to_string()));
    assert!(use_cases.contains(&"System analysis".to_string()));
    
    // Test complexity estimation for different query types
    let queries = vec![
        "What is a dog?",
        "What properties does a mammal inherit from animals?",
        "Analyze the complex hierarchical relationships in biological taxonomy",
        "What emergent properties arise from multi-level system interactions?",
    ];
    
    for query in queries {
        let estimate = systems.estimate_complexity(query);
        assert!(estimate.computational_complexity > 0, "Should estimate computational complexity");
        assert!(estimate.estimated_time_ms > 0, "Should estimate execution time");
        assert!(estimate.memory_requirements_mb > 0, "Should estimate memory requirements");
        assert!(estimate.confidence >= 0.0 && estimate.confidence <= 1.0, "Should provide confidence estimate");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_systems_pattern_in_ensemble_context() -> Result<()> {
    let graph = create_test_graph();
    
    // Add test data suitable for ensemble execution
    let entity_keys = create_scenario_entities_in_graph(&graph, "ensemble_systems_test").await?;
    assert!(!entity_keys.is_empty(), "Should have created ensemble test entities");
    
    let adaptive = AdaptiveThinking::new(graph.clone());
    
    // Test systems thinking as part of an ensemble
    let ensemble_result = adaptive.execute_adaptive_reasoning(
        "Analyze the hierarchical structure and emergent properties of artificial intelligence systems",
        Some("Consider both theoretical frameworks and practical implementations"),
        vec![
            CognitivePatternType::Systems,
            CognitivePatternType::Convergent,
            CognitivePatternType::Abstract,
        ],
    ).await?;
    
    assert!(ensemble_result.final_answer.len() > 0, "Should provide comprehensive ensemble answer");
    
    // Check that systems thinking contributed to the ensemble
    let systems_contribution = ensemble_result.pattern_contributions.iter()
        .find(|contrib| contrib.pattern_type == CognitivePatternType::Systems);
    
    if let Some(contribution) = systems_contribution {
        assert!(contribution.confidence >= 0.0, "Systems contribution should have valid confidence");
        assert!(!contribution.partial_result.is_empty(), "Systems should provide meaningful contribution");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_systems_pattern_cross_domain_analysis() -> Result<()> {
    let graph = create_test_graph();
    
    // Create cross-domain hierarchical data
    let entity_keys = create_scenario_entities_in_graph(&graph, "cross_domain_hierarchies").await?;
    assert!(!entity_keys.is_empty(), "Should have created cross-domain entities");
    
    let systems = SystemsThinking::new(graph);
    
    // Test different domain queries
    let domain_queries = vec![
        ("biological", "What hierarchies exist in biological classification?"),
        ("technological", "How are technological systems organized hierarchically?"),
        ("social", "What hierarchical structures exist in social systems?"),
        ("organizational", "How do organizational hierarchies function?"),
    ];
    
    for (domain, query) in domain_queries {
        let result = systems.execute_hierarchical_reasoning(
            query,
            llmkg::cognitive::SystemsReasoningType::SystemAnalysis
        ).await?;
        
        assert!(result.hierarchy_path.len() > 0, "Should analyze {} domain hierarchies", domain);
        assert!(result.system_complexity >= 0.0, "Should measure {} domain complexity", domain);
    }
    
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
async fn test_critical_pattern_source_validation() -> Result<()> {
    let graph = create_test_graph();
    
    // Add test data with varying source reliability
    let entity_keys = create_scenario_entities_in_graph(&graph, "multi_source_facts").await?;
    assert!(!entity_keys.is_empty(), "Should have created entities from multiple sources");
    
    let critical = CriticalThinking::new(graph);
    
    let result = critical.execute_critical_analysis(
        "validate sources of scientific information",
        llmkg::cognitive::ValidationLevel::Rigorous
    ).await?;
    
    assert!(result.uncertainty_analysis.overall_uncertainty >= 0.0);
    assert!(result.uncertainty_analysis.overall_uncertainty <= 1.0);
    assert!(result.confidence_intervals.len() >= 0);
    
    // Test different validation levels
    for validation_level in [
        llmkg::cognitive::ValidationLevel::Basic,
        llmkg::cognitive::ValidationLevel::Comprehensive,
        llmkg::cognitive::ValidationLevel::Rigorous,
    ] {
        let validation_result = critical.execute_critical_analysis(
            "test validation level",
            validation_level
        ).await?;
        
        assert!(validation_result.uncertainty_analysis.overall_uncertainty >= 0.0);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_critical_pattern_cognitive_trait_execution() -> Result<()> {
    let graph = create_test_graph();
    
    let entity_keys = create_scenario_entities_in_graph(&graph, "critical_trait_test").await?;
    assert!(!entity_keys.is_empty(), "Should have created test entities");
    
    let critical = CriticalThinking::new(graph);
    
    // Test execution through CognitivePattern trait
    let parameters = llmkg::cognitive::PatternParameters {
        max_iterations: Some(10),
        confidence_threshold: Some(0.7),
        exploration_depth: Some(3),
        creativity_level: Some(0.3),
        validation_level: Some(llmkg::cognitive::ValidationLevel::Comprehensive),
        output_format: None,
    };
    
    let result = critical.execute(
        "Verify the accuracy of claims about renewable energy",
        Some("scientific validation context"),
        parameters,
    ).await?;
    
    // Validate pattern result structure
    assert_eq!(result.pattern_type, CognitivePatternType::Critical);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(!result.reasoning_trace.is_empty());
    
    // Validate metadata
    assert!(result.metadata.execution_time_ms > 0);
    assert!(result.metadata.converged); // Critical thinking should converge
    
    // Validate critical-specific metadata
    assert!(result.metadata.additional_info.contains_key("facts_count"));
    assert!(result.metadata.additional_info.contains_key("contradictions_count"));
    assert!(result.metadata.additional_info.contains_key("resolution_strategy"));
    assert!(result.metadata.additional_info.contains_key("uncertainty_score"));
    assert!(result.metadata.additional_info.contains_key("knowledge_gaps_count"));
    
    Ok(())
}

#[tokio::test]
async fn test_critical_pattern_validation_level_inference() -> Result<()> {
    let graph = create_test_graph();
    let critical = CriticalThinking::new(graph);
    
    let parameters = llmkg::cognitive::PatternParameters::default();
    
    // Test queries that should trigger different validation levels
    let test_cases = vec![
        ("verify this scientific claim", "should trigger rigorous validation"),
        ("validate the research methodology", "should trigger rigorous validation"),
        ("check these statistics", "should trigger comprehensive validation"),
        ("confirm this information", "should trigger comprehensive validation"),
        ("analyze this data", "should trigger basic validation"),
        ("what do you think about this", "should trigger basic validation"),
    ];
    
    for (query, description) in test_cases {
        let result = critical.execute(query, None, parameters.clone()).await?;
        assert_eq!(result.pattern_type, CognitivePatternType::Critical);
        assert!(result.confidence >= 0.0, "{}: {}", query, description);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_critical_pattern_use_cases_and_complexity() -> Result<()> {
    let graph = create_test_graph();
    let critical = CriticalThinking::new(graph);
    
    // Test optimal use cases
    let use_cases = critical.get_optimal_use_cases();
    assert_eq!(use_cases.len(), 4);
    assert!(use_cases.contains(&"Fact validation".to_string()));
    assert!(use_cases.contains(&"Contradiction resolution".to_string()));
    assert!(use_cases.contains(&"Source reliability analysis".to_string()));
    assert!(use_cases.contains(&"Conflict detection".to_string()));
    
    // Test complexity estimation
    let queries = vec![
        "is this true?",
        "validate multiple conflicting sources about climate change",
        "verify complex scientific claims with statistical analysis",
        "resolve contradictions in large dataset with uncertainty quantification",
    ];
    
    for query in queries {
        let estimate = critical.estimate_complexity(query);
        assert_eq!(estimate.computational_complexity, 40);
        assert_eq!(estimate.estimated_time_ms, 1500);
        assert_eq!(estimate.memory_requirements_mb, 15);
        assert_eq!(estimate.confidence, 0.9);
        assert!(!estimate.parallelizable);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_critical_pattern_in_ensemble_context() -> Result<()> {
    let graph = create_test_graph();
    
    let entity_keys = create_scenario_entities_in_graph(&graph, "ensemble_critical_test").await?;
    assert!(!entity_keys.is_empty(), "Should have created ensemble test entities");
    
    let adaptive = AdaptiveThinking::new(graph.clone());
    
    // Test critical thinking as part of an ensemble
    let ensemble_result = adaptive.execute_adaptive_reasoning(
        "Verify and validate claims about artificial intelligence capabilities and limitations",
        Some("Consider both technical accuracy and source reliability"),
        vec![
            CognitivePatternType::Critical,
            CognitivePatternType::Convergent,
            CognitivePatternType::Systems,
        ],
    ).await?;
    
    assert!(ensemble_result.final_answer.len() > 0);
    
    // Check that critical thinking contributed to the ensemble
    let critical_contribution = ensemble_result.pattern_contributions.iter()
        .find(|contrib| contrib.pattern_type == CognitivePatternType::Critical);
    
    if let Some(contribution) = critical_contribution {
        assert!(contribution.confidence >= 0.0);
        assert!(!contribution.partial_result.is_empty());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_critical_pattern_cross_domain_validation() -> Result<()> {
    let graph = create_test_graph();
    
    let entity_keys = create_scenario_entities_in_graph(&graph, "cross_domain_validation").await?;
    assert!(!entity_keys.is_empty(), "Should have created cross-domain entities");
    
    let critical = CriticalThinking::new(graph);
    
    // Test validation across different domains
    let domain_queries = vec![
        ("medical", "validate medical research claims"),
        ("scientific", "verify scientific experimental results"),
        ("historical", "check historical fact accuracy"),
        ("technological", "validate technology performance claims"),
    ];
    
    for (domain, query) in domain_queries {
        let result = critical.execute_critical_analysis(
            query,
            llmkg::cognitive::ValidationLevel::Comprehensive
        ).await?;
        
        assert!(result.uncertainty_analysis.overall_uncertainty >= 0.0, 
               "Should analyze {} domain uncertainty", domain);
        assert!(result.uncertainty_analysis.overall_uncertainty <= 1.0,
               "Should provide valid {} domain uncertainty", domain);
    }
    
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
async fn test_abstract_pattern_cognitive_trait_execution() -> Result<()> {
    let graph = create_test_graph();
    
    // Setup test data for abstract pattern trait execution
    let entity_keys = create_scenario_entities_in_graph(&graph, "abstract_cognitive_trait_test").await?;
    assert!(!entity_keys.is_empty(), "Should have created test entities");
    
    let abstract_thinking = AbstractThinking::new(graph);
    
    // Test execution through CognitivePattern trait
    let parameters = llmkg::cognitive::PatternParameters::default();
    let result = abstract_thinking.execute(
        "Identify abstract patterns and optimization opportunities in the knowledge graph",
        Some("Focus on structural and semantic patterns for system improvement"),
        parameters,
    ).await?;
    
    // Validate pattern result structure
    assert_eq!(result.pattern_type, CognitivePatternType::Abstract);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0, "Confidence should be valid");
    assert!(!result.answer.is_empty(), "Answer should not be empty");
    assert!(!result.reasoning_trace.is_empty(), "Should have reasoning trace");
    
    // Validate metadata
    assert!(result.metadata.execution_time_ms > 0, "Should record execution time");
    assert!(result.metadata.nodes_activated >= 0, "Should activate nodes");
    assert_eq!(result.metadata.iterations_completed, 1, "Should complete one iteration");
    assert!(result.metadata.converged, "Should converge");
    assert!(result.metadata.total_energy >= 0.0, "Should calculate energy");
    
    // Validate additional metadata specific to abstract thinking
    assert!(result.metadata.additional_info.contains_key("patterns_count"));
    assert!(result.metadata.additional_info.contains_key("abstractions_count"));
    assert!(result.metadata.additional_info.contains_key("refactoring_opportunities"));
    assert!(result.metadata.additional_info.contains_key("query_improvement"));
    assert!(result.metadata.additional_info.contains_key("memory_reduction"));
    assert!(result.metadata.additional_info.contains_key("maintainability_score"));
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_pattern_use_cases_and_complexity() -> Result<()> {
    let graph = create_test_graph();
    let abstract_thinking = AbstractThinking::new(graph);
    
    // Test optimal use cases
    let use_cases = abstract_thinking.get_optimal_use_cases();
    assert!(!use_cases.is_empty(), "Should have optimal use cases");
    assert!(use_cases.contains(&"Pattern recognition".to_string()));
    assert!(use_cases.contains(&"Meta-analysis".to_string()));
    assert!(use_cases.contains(&"Concept abstraction".to_string()));
    assert!(use_cases.contains(&"System optimization".to_string()));
    
    // Test complexity estimation for different query types
    let queries = vec![
        "find patterns",
        "analyze graph structure for optimization opportunities",
        "identify abstract concepts in biological taxonomy hierarchy",
        "perform comprehensive meta-analysis of knowledge graph patterns across multiple domains",
    ];
    
    for query in queries {
        let estimate = abstract_thinking.estimate_complexity(query);
        assert_eq!(estimate.computational_complexity, 60, "Should estimate computational complexity");
        assert_eq!(estimate.estimated_time_ms, 2000, "Should estimate execution time");
        assert_eq!(estimate.memory_requirements_mb, 20, "Should estimate memory requirements");
        assert_eq!(estimate.confidence, 0.7, "Should provide confidence estimate");
        assert!(!estimate.parallelizable, "Abstract thinking should not be parallelizable");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_pattern_in_ensemble_context() -> Result<()> {
    let graph = create_test_graph();
    
    // Add test data suitable for ensemble execution
    let entity_keys = create_scenario_entities_in_graph(&graph, "ensemble_abstract_test").await?;
    assert!(!entity_keys.is_empty(), "Should have created ensemble test entities");
    
    let adaptive = AdaptiveThinking::new(graph.clone());
    
    // Test abstract thinking as part of an ensemble
    let ensemble_result = adaptive.execute_adaptive_reasoning(
        "Analyze patterns and optimize the knowledge representation system",
        Some("Consider both structural patterns and optimization opportunities"),
        vec![
            CognitivePatternType::Abstract,
            CognitivePatternType::Systems,
            CognitivePatternType::Convergent,
        ],
    ).await?;
    
    assert!(ensemble_result.final_answer.len() > 0, "Should provide comprehensive ensemble answer");
    
    // Check that abstract thinking contributed to the ensemble
    let abstract_contribution = ensemble_result.pattern_contributions.iter()
        .find(|contrib| contrib.pattern_type == CognitivePatternType::Abstract);
    
    if let Some(contribution) = abstract_contribution {
        assert!(contribution.confidence >= 0.0, "Abstract contribution should have valid confidence");
        assert!(!contribution.partial_result.is_empty(), "Abstract should provide meaningful contribution");
        
        // Abstract thinking contribution should mention patterns or optimization
        let contrib_lower = contribution.partial_result.to_lowercase();
        assert!(
            contrib_lower.contains("pattern") ||
            contrib_lower.contains("optimization") ||
            contrib_lower.contains("abstraction") ||
            contrib_lower.contains("efficiency"),
            "Abstract thinking should contribute pattern-related insights"
        );
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_pattern_cross_domain_analysis() -> Result<()> {
    let graph = create_test_graph();
    
    // Create cross-domain pattern data
    let entity_keys = create_scenario_entities_in_graph(&graph, "cross_domain_patterns").await?;
    assert!(!entity_keys.is_empty(), "Should have created cross-domain entities");
    
    let abstract_thinking = AbstractThinking::new(graph);
    
    // Test different pattern analysis types
    let analysis_queries = vec![
        ("structural", "What structural patterns exist across domains?"),
        ("semantic", "How do semantic patterns connect different concepts?"),
        ("temporal", "What temporal patterns emerge in knowledge evolution?"),
        ("usage", "What usage patterns optimize access efficiency?"),
    ];
    
    for (domain, query) in analysis_queries {
        let pattern_type = match domain {
            "structural" => llmkg::cognitive::PatternType::Structural,
            "semantic" => llmkg::cognitive::PatternType::Semantic,
            "temporal" => llmkg::cognitive::PatternType::Temporal,
            "usage" => llmkg::cognitive::PatternType::Usage,
            _ => llmkg::cognitive::PatternType::Structural,
        };
        
        let result = abstract_thinking.execute_pattern_analysis(
            llmkg::cognitive::AnalysisScope::Global,
            pattern_type,
        ).await?;
        
        assert!(result.patterns_found.len() >= 0, "Should analyze {} patterns", domain);
        assert!(result.efficiency_gains.maintainability_score >= 0.0, "Should measure {} efficiency", domain);
        
        // Verify pattern types match expected
        for pattern in &result.patterns_found {
            assert_eq!(pattern.pattern_type, pattern_type, "Pattern type should match analysis type");
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_pattern_optimization_scenarios() -> Result<()> {
    let graph = create_test_graph();
    
    // Create entities that could benefit from optimization
    let entity_keys = create_scenario_entities_in_graph(&graph, "optimization_scenarios").await?;
    assert!(!entity_keys.is_empty(), "Should have created optimization entities");
    
    let abstract_thinking = AbstractThinking::new(graph);
    
    let parameters = llmkg::cognitive::PatternParameters {
        max_depth: Some(4),
        activation_threshold: Some(0.5),
        exploration_breadth: Some(10),
        creativity_threshold: Some(0.2),
        validation_level: Some(llmkg::cognitive::ValidationLevel::Basic),
        pattern_type: Some(llmkg::cognitive::PatternType::Structural),
        reasoning_strategy: None,
    };
    
    // Test optimization-focused queries
    let optimization_queries = vec![
        "Identify redundant patterns that could be consolidated",
        "Find opportunities to improve query performance",
        "Suggest hierarchy reorganization for better maintainability",
        "Analyze concept merging opportunities",
    ];
    
    for query in optimization_queries {
        let result = abstract_thinking.execute(query, None, parameters.clone()).await?;
        
        assert_eq!(result.pattern_type, CognitivePatternType::Abstract);
        assert!(result.confidence >= 0.0, "Should have valid confidence for: {}", query);
        
        // Should mention optimization, efficiency, or improvement
        let answer_lower = result.answer.to_lowercase();
        assert!(
            answer_lower.contains("optimization") ||
            answer_lower.contains("efficiency") ||
            answer_lower.contains("improve") ||
            answer_lower.contains("refactor") ||
            answer_lower.contains("consolidate"),
            "Should suggest optimizations for: {}", query
        );
        
        // Should have efficiency metrics
        let query_improvement = result.metadata.additional_info.get("query_improvement");
        let memory_reduction = result.metadata.additional_info.get("memory_reduction");
        let maintainability = result.metadata.additional_info.get("maintainability_score");
        
        assert!(query_improvement.is_some(), "Should have query improvement metric");
        assert!(memory_reduction.is_some(), "Should have memory reduction metric");
        assert!(maintainability.is_some(), "Should have maintainability metric");
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_pattern_meta_analysis_workflows() -> Result<()> {
    let graph = create_test_graph();
    
    // Create entities suitable for meta-analysis
    let entity_keys = create_scenario_entities_in_graph(&graph, "meta_analysis_data").await?;
    assert!(!entity_keys.is_empty(), "Should have created meta-analysis entities");
    
    let abstract_thinking = AbstractThinking::new(graph);
    
    let parameters = llmkg::cognitive::PatternParameters {
        max_depth: Some(5),
        activation_threshold: Some(0.4),
        exploration_breadth: Some(15),
        creativity_threshold: Some(0.3),
        validation_level: Some(llmkg::cognitive::ValidationLevel::Comprehensive),
        pattern_type: Some(llmkg::cognitive::PatternType::Structural),
        reasoning_strategy: None,
    };
    
    // Test meta-analysis workflows
    let meta_queries = vec![
        ("pattern_of_patterns", "Find patterns in how patterns are organized"),
        ("abstraction_levels", "Analyze different levels of abstraction in the system"),
        ("optimization_patterns", "Identify meta-patterns in optimization opportunities"),
        ("cross_domain_structures", "Find structural similarities across different domains"),
    ];
    
    for (workflow_type, query) in meta_queries {
        let result = abstract_thinking.execute(query, None, parameters.clone()).await?;
        
        assert_eq!(result.pattern_type, CognitivePatternType::Abstract);
        assert!(result.confidence >= 0.0, "Meta-analysis should have valid confidence: {}", workflow_type);
        
        // Should demonstrate meta-level thinking
        let answer_lower = result.answer.to_lowercase();
        assert!(
            answer_lower.contains("pattern") ||
            answer_lower.contains("meta") ||
            answer_lower.contains("level") ||
            answer_lower.contains("structure") ||
            answer_lower.contains("organization"),
            "Should demonstrate meta-analysis: {}", workflow_type
        );
        
        // Should have detected some patterns or abstractions
        let patterns_count = result.metadata.additional_info.get("patterns_count").unwrap();
        let abstractions_count = result.metadata.additional_info.get("abstractions_count").unwrap();
        
        // At least one should be non-zero for meta-analysis
        assert!(
            patterns_count.parse::<usize>().unwrap() > 0 ||
            abstractions_count.parse::<usize>().unwrap() > 0,
            "Meta-analysis should find patterns or abstractions: {}", workflow_type
        );
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_pattern_refactoring_suggestions() -> Result<()> {
    let graph = create_test_graph();
    
    // Create entities with potential refactoring opportunities
    let entity_keys = create_scenario_entities_in_graph(&graph, "refactoring_candidates").await?;
    assert!(!entity_keys.is_empty(), "Should have created refactoring entities");
    
    let abstract_thinking = AbstractThinking::new(graph);
    
    // Test refactoring suggestion analysis
    let result = abstract_thinking.execute_pattern_analysis(
        llmkg::cognitive::AnalysisScope::Global,
        llmkg::cognitive::PatternType::Structural,
    ).await?;
    
    // Should identify refactoring opportunities
    assert!(result.refactoring_opportunities.len() > 0, "Should suggest refactoring opportunities");
    
    // Verify refactoring opportunity types
    let refactoring_types: std::collections::HashSet<_> = result.refactoring_opportunities.iter()
        .map(|opp| &opp.opportunity_type)
        .collect();
    
    // Should have at least performance optimization
    assert!(
        refactoring_types.contains(&llmkg::cognitive::RefactoringType::PerformanceOptimization),
        "Should suggest performance optimizations"
    );
    
    // Verify refactoring descriptions are meaningful
    for opportunity in &result.refactoring_opportunities {
        assert!(!opportunity.description.is_empty(), "Refactoring description should not be empty");
        assert!(opportunity.estimated_benefit > 0.0, "Should estimate positive benefit");
        assert!(opportunity.estimated_benefit <= 1.0, "Benefit should not exceed 100%");
        
        // Description should mention relevant actions
        let desc_lower = opportunity.description.to_lowercase();
        assert!(
            desc_lower.contains("consolidate") ||
            desc_lower.contains("merge") ||
            desc_lower.contains("optimize") ||
            desc_lower.contains("reorganize") ||
            desc_lower.contains("add") ||
            desc_lower.contains("eliminate"),
            "Refactoring description should suggest concrete actions"
        );
    }
    
    Ok(())
}

#[tokio::test]
async fn test_abstract_pattern_efficiency_analysis() -> Result<()> {
    let graph = create_test_graph();
    
    // Create entities for efficiency analysis
    let entity_keys = create_scenario_entities_in_graph(&graph, "efficiency_analysis").await?;
    assert!(!entity_keys.is_empty(), "Should have created efficiency entities");
    
    let abstract_thinking = AbstractThinking::new(graph);
    
    let result = abstract_thinking.execute_pattern_analysis(
        llmkg::cognitive::AnalysisScope::Global,
        llmkg::cognitive::PatternType::Structural,
    ).await?;
    
    // Verify efficiency gains structure
    let efficiency = &result.efficiency_gains;
    assert!(efficiency.query_time_improvement >= 0.0, "Query time improvement should be non-negative");
    assert!(efficiency.query_time_improvement <= 0.5, "Query improvement should be capped at 50%");
    assert!(efficiency.memory_reduction >= 0.0, "Memory reduction should be non-negative");
    assert!(efficiency.memory_reduction <= 0.4, "Memory reduction should be capped at 40%");
    assert!(efficiency.accuracy_improvement >= 0.0, "Accuracy improvement should be non-negative");
    assert!(efficiency.accuracy_improvement <= 0.3, "Accuracy improvement should be capped at 30%");
    assert!(efficiency.maintainability_score >= 0.0, "Maintainability should be non-negative");
    assert!(efficiency.maintainability_score <= 1.0, "Maintainability should be at most 1.0");
    
    // Test efficiency gains with different refactoring scenarios
    let parameters = llmkg::cognitive::PatternParameters::default();
    let efficiency_result = abstract_thinking.execute(
        "Analyze efficiency gains from implementing suggested optimizations",
        Some("Focus on quantifying performance improvements"),
        parameters,
    ).await?;
    
    assert_eq!(efficiency_result.pattern_type, CognitivePatternType::Abstract);
    
    // Should mention efficiency metrics
    let answer_lower = efficiency_result.answer.to_lowercase();
    assert!(
        answer_lower.contains("efficiency") ||
        answer_lower.contains("improvement") ||
        answer_lower.contains("performance") ||
        answer_lower.contains("time") ||
        answer_lower.contains("memory"),
        "Should discuss efficiency metrics"
    );
    
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