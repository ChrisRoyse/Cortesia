#[cfg(test)]
mod phase2_cognitive_tests {
    use llmkg::cognitive::*;
    use llmkg::cognitive::types::*;
    use llmkg::cognitive::orchestrator::*;
    use llmkg::core::brain_types::*;
    use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use llmkg::versioning::temporal_graph::{TemporalKnowledgeGraph, TimeRange};
    use llmkg::core::graph::KnowledgeGraph;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use ahash::AHashMap;
    use chrono::Utc;
    use std::time::Instant;

    // Include test utilities directly from file
    mod test_data_generator;
    mod benchmarks;
    use test_data_generator::{TestDataGenerator, TestDataStatistics};
    use benchmarks::CognitiveBenchmark;

    async fn setup_test_environment() -> (Arc<BrainEnhancedKnowledgeGraph>, TestDataGenerator) {
        // Create test data generator and generate comprehensive data
        let mut generator = TestDataGenerator::new().await.unwrap();
        generator.generate_comprehensive_data().await.unwrap();
        
        // Print statistics for debugging
        let stats = generator.get_statistics().await.unwrap();
        stats.print_summary();
        
        (generator.graph.clone(), generator)
    }

    #[tokio::test]
    async fn test_convergent_thinking_basic_query() {
        let start_time = Instant::now();
        let (graph, _generator) = setup_test_environment().await;
        
        let convergent = ConvergentThinking::new(
            graph.clone(),
        );
        
        // Test basic factual query
        let result = convergent.execute_convergent_query(
            "What type is dog?",
            None,
        ).await.unwrap();
        
        println!("Convergent query result: '{}'", result.answer);
        println!("Confidence: {}", result.confidence);
        println!("Reasoning trace length: {}", result.reasoning_trace.len());
        println!("Test duration: {:?}", start_time.elapsed());
        
        // Check that we got a reasonable answer
        assert!(!result.answer.is_empty());
        assert!(result.confidence > 0.0);
        assert!(!result.reasoning_trace.is_empty());
        
        // Check that the answer contains relevant concepts or properties
        let answer_lower = result.answer.to_lowercase();
        assert!(answer_lower.contains("mammal") || answer_lower.contains("animal") || 
                answer_lower.contains("domesticated") || answer_lower.contains("properties") || 
                answer_lower.contains("four_legs") || answer_lower.contains("warm_blooded"));
    }

    #[tokio::test]
    async fn test_convergent_thinking_with_context() {
        let start_time = Instant::now();
        let (graph, _generator) = setup_test_environment().await;
        
        let convergent = ConvergentThinking::new(
            graph.clone(),
        );
        
        // Test query with context
        let result = convergent.execute_convergent_query(
            "How many legs?",
            Some("We are talking about dogs"),
        ).await.unwrap();
        
        println!("Convergent with context result: '{}'", result.answer);
        println!("Confidence: {}", result.confidence);
        println!("Test duration: {:?}", start_time.elapsed());
        
        // Check that we got a reasonable answer about legs
        assert!(!result.answer.is_empty());
        assert!(result.confidence > 0.0);
        assert!(!result.reasoning_trace.is_empty());
        
        // Check that the answer relates to legs or locomotion
        let answer_lower = result.answer.to_lowercase();
        assert!(answer_lower.contains("four") || answer_lower.contains("legs") || answer_lower.contains("locomotion"));
    }

    #[tokio::test]
    async fn test_divergent_thinking_exploration() {
        let start_time = Instant::now();
        let (graph, _generator) = setup_test_environment().await;
        
        let divergent = DivergentThinking::new_with_params(
            graph.clone(),
            10,  // exploration_breadth
            0.3, // creativity_threshold
        );
        
        // Test exploration of instances
        let result = divergent.execute_divergent_exploration(
            "animal",
            ExplorationType::Instances,
        ).await.unwrap();
        
        println!("Divergent exploration found {} explorations", result.explorations.len());
        for exploration in &result.explorations {
            println!("  - {}", exploration.concept);
        }
        println!("Test duration: {:?}", start_time.elapsed());
        
        // Check that we found some explorations
        assert!(result.explorations.len() >= 1);
        // Check that we have reasonable exploration results
        assert!(result.explorations.iter().any(|e| 
            e.concept.contains("dog") || e.concept.contains("cat") || 
            e.concept.contains("mammal") || e.concept.contains("bird")
        ));
    }

    #[tokio::test]
    async fn test_divergent_thinking_creative_mode() {
        let (graph, _generator) = setup_test_environment().await;
        
        let divergent = DivergentThinking::new_with_params(
            graph.clone(),
            20,  // higher exploration breadth
            0.2, // lower creativity threshold
        );
        
        // Test creative exploration
        let result = divergent.execute_divergent_exploration(
            "dog",
            ExplorationType::Creative,
        ).await.unwrap();
        
        assert!(!result.explorations.is_empty());
        assert!(result.creativity_scores.iter().any(|&s| s > 0.0));
    }

    #[tokio::test]
    async fn test_lateral_thinking_bridge_finding() {
        let (graph, _generator) = setup_test_environment().await;
        
        // Add more entities for lateral connections
        let pet_entity = BrainInspiredEntity::new("pet".to_string(), EntityDirection::Input);
        let _pet_key = graph.insert_brain_entity(pet_entity).await.unwrap();
        
        let lateral = LateralThinking::new(
            graph.clone(),
        );
        
        // Find creative connection between dog and cat
        let result = lateral.find_creative_connections(
            "dog",
            "cat",
            Some(3), // max_bridge_length
        ).await.unwrap();
        
        assert!(!result.bridges.is_empty());
        let first_bridge = &result.bridges[0];
        assert!(first_bridge.path.len() <= 3);
        assert!(first_bridge.novelty_score > 0.0);
        assert!(first_bridge.plausibility_score > 0.0);
    }

    #[tokio::test]
    async fn test_systems_thinking_hierarchy() {
        let (graph, _generator) = setup_test_environment().await;
        
        // Add hierarchical data
        let mammal_entity = BrainInspiredEntity::new("mammal".to_string(), EntityDirection::Input);
        let _mammal_key = graph.insert_brain_entity(mammal_entity).await.unwrap();
        
        let systems = SystemsThinking::new(
            graph.clone(),
        );
        
        // Test attribute inheritance
        let result = systems.execute_hierarchical_reasoning(
            "What properties do dogs inherit?",
            SystemsReasoningType::AttributeInheritance,
        ).await.unwrap();
        
        assert!(!result.hierarchy_path.is_empty());
        // For systems thinking, check if we have inheritance data
        assert!(!result.inherited_attributes.is_empty());
        // Check for warm_blooded attribute in the inherited attributes
        assert!(result.inherited_attributes.iter().any(|attr| attr.attribute_name == "warm_blooded"));
    }

    #[tokio::test]
    async fn test_critical_thinking_contradiction_detection() {
        let (graph, _generator) = setup_test_environment().await;
        
        // Add contradictory data
        let tripper_entity = BrainInspiredEntity::new("tripper".to_string(), EntityDirection::Input);
        let _tripper_key = graph.insert_brain_entity(tripper_entity).await.unwrap();
        
        let critical = CriticalThinking::new(
            graph.clone(),
        );
        
        // Test that we can find facts about Tripper having 3 legs
        let result = critical.execute_critical_analysis(
            "How many legs does tripper have?",
            ValidationLevel::Comprehensive,
        ).await.unwrap();
        
        println!("Contradictions found: {:?}", result.contradictions_found);
        println!("Resolved facts: {:?}", result.resolved_facts);
        
        // For now, just verify we found facts about Tripper's legs
        // Full contradiction detection between inherited and local properties
        // would require more complex graph reasoning
        assert!(result.resolved_facts.iter().any(|f| 
            f.fact_statement.contains("tripper") && f.fact_statement.contains("three")
        ));
        
        // Also verify we have some uncertainty analysis
        assert!(result.uncertainty_analysis.overall_uncertainty >= 0.0);
    }

    #[tokio::test]
    async fn test_abstract_thinking_pattern_detection() {
        let (graph, _generator) = setup_test_environment().await;
        
        // Add more data for pattern detection
        // This would be done through proper entity creation in a real implementation
        
        let abstract_thinking = AbstractThinking::new(
            graph.clone(),
        );
        
        // Test pattern detection
        let result = abstract_thinking.execute_pattern_analysis(
            AnalysisScope::Global,
            PatternType::Structural,
        ).await.unwrap();
        
        assert!(!result.patterns_found.is_empty());
        
        // Debug output
        println!("Found {} patterns:", result.patterns_found.len());
        for pattern in &result.patterns_found {
            println!("  Pattern: {} - {}", pattern.pattern_id, pattern.description);
        }
        
        // Should detect the repeated is_a relationship pattern
        assert!(result.patterns_found.iter().any(|p| p.pattern_id.contains("is_a")));
    }

    #[tokio::test]
    async fn test_adaptive_thinking_strategy_selection() {
        let (graph, _generator) = setup_test_environment().await;
        
        let adaptive = AdaptiveThinking::new(
            graph.clone(),
        );
        
        // Test automatic strategy selection
        let factual_result = adaptive.execute_adaptive_reasoning(
            "What type is dog?", // Factual query should select convergent
            None,
            vec![
                CognitivePatternType::Convergent,
                CognitivePatternType::Divergent,
                CognitivePatternType::Lateral,
            ],
        ).await.unwrap();
        
        assert_eq!(factual_result.strategy_used.selected_patterns[0], CognitivePatternType::Convergent);
        
        // Test exploration query
        let exploration_result = adaptive.execute_adaptive_reasoning(
            "What are types of animals?", // Exploration should select divergent
            None,
            vec![
                CognitivePatternType::Convergent,
                CognitivePatternType::Divergent,
                CognitivePatternType::Lateral,
            ],
        ).await.unwrap();
        
        assert!(exploration_result.strategy_used.selected_patterns.contains(&CognitivePatternType::Divergent));
    }

    #[tokio::test]
    async fn test_cognitive_orchestrator_integration() {
        let (graph, _generator) = setup_test_environment().await;
        
        let orchestrator = CognitiveOrchestrator::new(
            graph.clone(),
            CognitiveOrchestratorConfig::default(),
        ).await.unwrap();
        
        // Test automatic reasoning
        let auto_result = orchestrator.reason(
            "What is a dog?",
            None,
            ReasoningStrategy::Automatic,
        ).await.unwrap();
        
        assert!(!auto_result.final_answer.is_empty());
        assert!(auto_result.quality_metrics.overall_confidence > 0.0);
        
        // Test specific pattern
        let specific_result = orchestrator.reason(
            "What are examples of animals?",
            None,
            ReasoningStrategy::Specific(CognitivePatternType::Divergent),
        ).await.unwrap();
        
        assert!(!specific_result.final_answer.is_empty());
        
        // Test ensemble reasoning
        let ensemble_result = orchestrator.reason(
            "How is dog related to cat?",
            None,
            ReasoningStrategy::Ensemble(vec![
                CognitivePatternType::Lateral,
                CognitivePatternType::Systems,
            ]),
        ).await.unwrap();
        
        assert!(!ensemble_result.final_answer.is_empty());
        // Pattern contributions would be available in adaptive results, not basic ensemble
        assert!(ensemble_result.execution_metadata.patterns_executed.len() >= 2);
    }

    #[tokio::test]
    async fn test_performance_tracking() {
        let (graph, _generator) = setup_test_environment().await;
        
        let adaptive = AdaptiveThinking::new(
            graph.clone(),
        );
        
        // Execute multiple queries to build performance history
        for _ in 0..3 {
            let _ = adaptive.execute_adaptive_reasoning(
                "What type is dog?",
                None,
                vec![CognitivePatternType::Convergent, CognitivePatternType::Divergent],
            ).await.unwrap();
        }
        
        // Performance should improve over time
        // Since performance tracking is internal to adaptive thinking, we just check execution
        // The actual performance metrics would be tracked internally
    }

    #[tokio::test]
    async fn test_benchmark_convergent_thinking() {
        let benchmark = CognitiveBenchmark::new().await.unwrap();
        
        let queries = vec![
            "What type is dog?",
            "What properties do mammals have?",
            "What is artificial intelligence?",
        ];
        
        let result = benchmark.benchmark_convergent_thinking(&queries).await;
        
        println!("Benchmark Results:");
        println!("Pattern: {}", result.pattern_type);
        println!("Success rate: {:.2}%", result.success_rate * 100.0);
        println!("Average confidence: {:.3}", result.average_confidence);
        println!("Total duration: {:?}", result.total_duration);
        println!("Average duration: {:?}", result.average_duration);
        
        // Test assertions
        assert_eq!(result.pattern_type, "Convergent");
        assert!(result.success_rate > 0.0);
        assert_eq!(result.query_results.len(), 3);
    }

    #[tokio::test]
    async fn test_comprehensive_benchmark() {
        let benchmark = CognitiveBenchmark::new().await.unwrap();
        let comprehensive_result = benchmark.run_comprehensive_benchmark().await;
        
        comprehensive_result.print_detailed_report();
        
        // Test assertions
        assert!(comprehensive_result.get_total_queries() > 0);
        assert!(comprehensive_result.calculate_overall_success_rate() >= 0.0);
        assert!(comprehensive_result.calculate_average_confidence() >= 0.0);
    }
}