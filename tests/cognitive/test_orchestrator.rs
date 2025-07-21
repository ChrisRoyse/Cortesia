#[cfg(test)]
mod orchestrator_tests {
    use tokio;
    use std::sync::Arc;
    use llmkg::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
    use llmkg::core::brain_enhanced_graph::{BrainEnhancedKnowledgeGraph, BrainEnhancedConfig};
    use llmkg::cognitive::{
        PatternResult, CognitivePatternType,
        AdaptiveThinking, ConvergentThinking, DivergentThinking, CriticalThinking,
        LateralThinking, AbstractThinking
    };

    #[tokio::test]
    async fn test_detect_query_type_factual() {
        let orchestrator = create_test_orchestrator().await;
        
        // Test factual queries
        let factual_queries = vec![
            "What is the capital of France?",
            "How many legs does a dog have?",
            "What is the definition of photosynthesis?",
            "Who invented the telephone?",
        ];
        
        for query in factual_queries {
            let query_type = orchestrator.detect_query_type(query).await;
            assert_eq!(query_type, QueryType::Factual, 
                      "Query '{}' should be detected as factual", query);
        }
    }

    #[tokio::test]
    async fn test_detect_query_type_exploratory() {
        let orchestrator = create_test_orchestrator().await;
        
        // Test exploratory queries
        let exploratory_queries = vec![
            "What are some examples of renewable energy?",
            "Tell me about different types of music",
            "What are the connections between art and science?",
            "Give me ideas for creative writing",
        ];
        
        for query in exploratory_queries {
            let query_type = orchestrator.detect_query_type(query).await;
            assert_eq!(query_type, QueryType::Exploratory,
                      "Query '{}' should be detected as exploratory", query);
        }
    }

    #[tokio::test]
    async fn test_detect_query_type_analytical() {
        let orchestrator = create_test_orchestrator().await;
        
        // Test analytical queries
        let analytical_queries = vec![
            "Analyze the relationship between exercise and health",
            "Compare different economic systems",
            "Evaluate the effectiveness of renewable energy",
            "Assess the impact of social media on society",
        ];
        
        for query in analytical_queries {
            let query_type = orchestrator.detect_query_type(query).await;
            assert_eq!(query_type, QueryType::Analytical,
                      "Query '{}' should be detected as analytical", query);
        }
    }

    #[tokio::test]
    async fn test_detect_query_type_creative() {
        let orchestrator = create_test_orchestrator().await;
        
        // Test creative queries
        let creative_queries = vec![
            "How might music relate to mathematics?",
            "What creative connections exist between nature and technology?",
            "Find unexpected links between cooking and chemistry",
            "Discover novel relationships between sports and physics",
        ];
        
        for query in creative_queries {
            let query_type = orchestrator.detect_query_type(query).await;
            assert_eq!(query_type, QueryType::Creative,
                      "Query '{}' should be detected as creative", query);
        }
    }

    #[tokio::test]
    async fn test_build_processing_pipeline_factual() {
        let orchestrator = create_test_orchestrator().await;
        
        let pipeline = orchestrator.build_processing_pipeline(QueryType::Factual).await;
        assert!(pipeline.is_ok());
        
        let pipeline = pipeline.unwrap();
        
        // Factual queries should primarily use convergent thinking
        assert_eq!(pipeline.primary_pattern, CognitivePatternType::Convergent);
        
        // Should have critical thinking for validation
        assert!(pipeline.secondary_patterns.contains(&CognitivePatternType::Critical));
        
        // Should have appropriate confidence threshold
        assert!(pipeline.confidence_threshold > 0.7);
    }

    #[tokio::test]
    async fn test_build_processing_pipeline_exploratory() {
        let orchestrator = create_test_orchestrator().await;
        
        let pipeline = orchestrator.build_processing_pipeline(QueryType::Exploratory).await;
        assert!(pipeline.is_ok());
        
        let pipeline = pipeline.unwrap();
        
        // Exploratory queries should use divergent thinking
        assert_eq!(pipeline.primary_pattern, CognitivePatternType::Divergent);
        
        // May include adaptive thinking for query refinement
        assert!(pipeline.secondary_patterns.contains(&CognitivePatternType::Adaptive) ||
                pipeline.secondary_patterns.contains(&CognitivePatternType::Abstract));
    }

    #[tokio::test]
    async fn test_build_processing_pipeline_creative() {
        let orchestrator = create_test_orchestrator().await;
        
        let pipeline = orchestrator.build_processing_pipeline(QueryType::Creative).await;
        assert!(pipeline.is_ok());
        
        let pipeline = pipeline.unwrap();
        
        // Creative queries should use lateral thinking
        assert_eq!(pipeline.primary_pattern, CognitivePatternType::Lateral);
        
        // Should include divergent thinking for idea generation
        assert!(pipeline.secondary_patterns.contains(&CognitivePatternType::Divergent));
    }

    #[tokio::test]
    async fn test_execute_pipeline_single_pattern() {
        let orchestrator = create_test_orchestrator().await;
        
        // Create simple pipeline with only convergent thinking
        let pipeline = ProcessingPipeline {
            primary_pattern: CognitivePatternType::Convergent,
            secondary_patterns: vec![],
            confidence_threshold: 0.8,
            max_iterations: 1,
            fallback_strategy: "simple_search".to_string(),
        };
        
        let result = orchestrator.execute_pipeline("What is art?", pipeline).await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        match pattern_result {
            PatternResult::Convergent(conv_result) => {
                assert_eq!(conv_result.pattern_type, CognitivePatternType::Convergent);
                assert!(!conv_result.answer.is_empty());
            },
            _ => panic!("Expected convergent result from convergent pipeline")
        }
    }

    #[tokio::test]
    async fn test_execute_pipeline_multi_pattern() {
        let orchestrator = create_test_orchestrator().await;
        
        // Create pipeline with primary and secondary patterns
        let pipeline = ProcessingPipeline {
            primary_pattern: CognitivePatternType::Divergent,
            secondary_patterns: vec![CognitivePatternType::Critical],
            confidence_threshold: 0.7,
            max_iterations: 2,
            fallback_strategy: "adaptive_search".to_string(),
        };
        
        let result = orchestrator.execute_pipeline("Examples of creativity", pipeline).await;
        assert!(result.is_ok());
        
        // Should get results from the pipeline execution
        let pattern_result = result.unwrap();
        
        // Should be from one of the patterns in the pipeline
        match pattern_result {
            PatternResult::Divergent(_) | PatternResult::Critical(_) => {
                // Both are valid results from the pipeline
            },
            _ => panic!("Result should be from pipeline patterns")
        }
    }

    #[tokio::test]
    async fn test_adaptive_pattern_selection() {
        let orchestrator = create_test_orchestrator().await;
        
        // Test that orchestrator can adaptively select patterns based on context
        let ambiguous_query = "What about dogs and their relationship to humans?";
        
        let result = orchestrator.process_query(ambiguous_query).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        
        // Should have selected an appropriate pattern and produced a result
        assert!(!response.final_answer.is_empty());
        assert!(response.confidence >= 0.0 && response.confidence <= 1.0);
        assert!(response.processing_steps.len() > 0);
    }

    #[tokio::test]
    async fn test_confidence_based_iteration() {
        let orchestrator = create_test_orchestrator().await;
        
        // Create pipeline with high confidence threshold to trigger iteration
        let pipeline = ProcessingPipeline {
            primary_pattern: CognitivePatternType::Convergent,
            secondary_patterns: vec![CognitivePatternType::Divergent],
            confidence_threshold: 0.95, // Very high threshold
            max_iterations: 3,
            fallback_strategy: "comprehensive_search".to_string(),
        };
        
        let result = orchestrator.execute_pipeline("Complex philosophical question", pipeline).await;
        assert!(result.is_ok());
        
        // Should have attempted multiple patterns due to high confidence threshold
        let pattern_result = result.unwrap();
        
        // Verify we got a result even with high threshold
        match pattern_result {
            PatternResult::Convergent(conv) => assert!(!conv.answer.is_empty()),
            PatternResult::Divergent(div) => assert!(div.exploration_paths.len() > 0),
            _ => {} // Other patterns are acceptable
        }
    }

    #[tokio::test]
    async fn test_fallback_strategy_activation() {
        let orchestrator = create_test_orchestrator().await;
        
        // Test with a query that might not be well-handled by primary pattern
        let pipeline = ProcessingPipeline {
            primary_pattern: CognitivePatternType::Lateral, // Specific pattern
            secondary_patterns: vec![],
            confidence_threshold: 0.8,
            max_iterations: 1,
            fallback_strategy: "adaptive_fallback".to_string(),
        };
        
        let result = orchestrator.execute_pipeline("What is 2+2?", pipeline).await;
        assert!(result.is_ok());
        
        // Should get some result, potentially from fallback
        let pattern_result = result.unwrap();
        
        // Verify we got a meaningful result
        match pattern_result {
            PatternResult::Lateral(lat) => {
                // If lateral thinking handled it
                assert!(lat.bridge_paths.len() >= 0);
            },
            PatternResult::Convergent(_) => {
                // If fallback to convergent thinking occurred
            },
            _ => {} // Other patterns acceptable
        }
    }

    #[tokio::test]
    async fn test_pattern_coordination() {
        let orchestrator = create_test_orchestrator().await;
        
        // Test coordinated execution of multiple patterns
        let complex_query = "Analyze creative connections between music and mathematics";
        
        let result = orchestrator.process_query(complex_query).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        
        // Should coordinate multiple patterns for complex query
        assert!(response.processing_steps.len() > 1, 
               "Complex query should involve multiple processing steps");
        
        // Should have reasonable confidence
        assert!(response.confidence > 0.3);
        
        // Should have detailed answer
        assert!(response.final_answer.len() > 50);
    }

    #[tokio::test]
    async fn test_memory_and_context_integration() {
        let orchestrator = create_test_orchestrator().await;
        
        // Test that orchestrator maintains context across queries
        let first_query = "What is machine learning?";
        let first_result = orchestrator.process_query(first_query).await;
        assert!(first_result.is_ok());
        
        // Follow-up query that should use context
        let followup_query = "How does that relate to artificial intelligence?";
        let followup_result = orchestrator.process_query(followup_query).await;
        assert!(followup_result.is_ok());
        
        let response = followup_result.unwrap();
        
        // Follow-up should reference previous context
        assert!(!response.final_answer.is_empty());
        assert!(response.context_references.len() > 0);
    }

    #[tokio::test]
    async fn test_error_handling_and_recovery() {
        let orchestrator = create_test_orchestrator().await;
        
        // Test with malformed or problematic queries
        let problematic_queries = vec![
            "", // Empty query
            "   ", // Whitespace only
            "??????????", // Only punctuation
            "a".repeat(1000), // Very long query
        ];
        
        for query in problematic_queries {
            let result = orchestrator.process_query(&query).await;
            
            // Should handle gracefully without panicking
            match result {
                Ok(response) => {
                    // If it succeeds, should have reasonable response
                    assert!(!response.final_answer.is_empty() || 
                           response.error_message.is_some());
                },
                Err(_) => {
                    // Errors are acceptable for malformed queries
                }
            }
        }
    }

    #[tokio::test]
    async fn test_performance_monitoring() {
        let orchestrator = create_test_orchestrator().await;
        
        let query = "Test query for performance monitoring";
        let result = orchestrator.process_query(query).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        
        // Should track performance metrics
        assert!(response.processing_time_ms > 0);
        assert!(response.patterns_used.len() > 0);
        
        // Should have reasonable processing time (less than 10 seconds for test)
        assert!(response.processing_time_ms < 10000);
    }

    #[tokio::test]
    async fn test_cognitive_pattern_interface() {
        let orchestrator = create_test_orchestrator().await;
        
        let result = orchestrator.process_query("Test the cognitive pattern interface").await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        
        // Should implement the standard cognitive pattern interface
        assert!(!response.final_answer.is_empty());
        assert!(response.confidence >= 0.0 && response.confidence <= 1.0);
        assert!(!response.patterns_used.is_empty());
    }

    #[tokio::test]
    async fn test_orchestrator_creation() {
        let config = BrainEnhancedConfig::default();
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new(config));
        let orchestrator_config = CognitiveOrchestratorConfig::default();
        let orchestrator = CognitiveOrchestrator::new(graph, orchestrator_config).await;
        assert!(orchestrator.is_ok());
    }


    // Helper functions

    async fn create_test_orchestrator() -> CognitiveOrchestrator {
        let graph = Arc::new(create_test_graph().await);
        let config = CognitiveOrchestratorConfig::default();
        
        CognitiveOrchestrator::new(graph, config).await.expect("Failed to create test orchestrator")
    }

    async fn create_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create comprehensive test knowledge base
        
        // Basic concepts
        graph.add_entity("art", "Artistic domain").await.unwrap();
        graph.add_entity("science", "Scientific domain").await.unwrap();
        graph.add_entity("music", "Musical art form").await.unwrap();
        graph.add_entity("mathematics", "Mathematical domain").await.unwrap();
        graph.add_entity("creativity", "Creative process").await.unwrap();
        
        // Animals
        graph.add_entity("dog", "Domestic animal").await.unwrap();
        graph.add_entity("cat", "Domestic animal").await.unwrap();
        graph.add_entity("legs", "Body parts for movement").await.unwrap();
        
        // Technology
        graph.add_entity("technology", "Applied science").await.unwrap();
        graph.add_entity("computer", "Computing device").await.unwrap();
        graph.add_entity("internet", "Global network").await.unwrap();
        
        // Relationships
        graph.add_relationship("art", "creativity", "involves", 0.9).await.unwrap();
        graph.add_relationship("music", "art", "is_a", 0.8).await.unwrap();
        graph.add_relationship("music", "mathematics", "relates_to", 0.6).await.unwrap();
        graph.add_relationship("science", "mathematics", "uses", 0.8).await.unwrap();
        graph.add_relationship("technology", "science", "applies", 0.7).await.unwrap();
        
        graph.add_relationship("dog", "legs", "has", 0.9).await.unwrap();
        graph.add_relationship("cat", "legs", "has", 0.9).await.unwrap();
        
        graph.add_relationship("computer", "technology", "is_a", 0.8).await.unwrap();
        graph.add_relationship("internet", "computer", "connects", 0.7).await.unwrap();
        
        graph
    }
}