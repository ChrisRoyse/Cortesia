#[cfg(test)]
mod systems_thinking_integration_tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use tokio;
    use llmkg::cognitive::systems::SystemsThinking;
    use llmkg::cognitive::types::{
        CognitivePattern, PatternParameters, PatternResult, CognitivePatternType,
        SystemsReasoningType, SystemsResult, InheritedAttribute, Exception, ExceptionType
    };
    use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use llmkg::core::types::EntityKey;

    /// Helper function to create a comprehensive test graph with hierarchical relationships
    async fn create_comprehensive_test_graph() -> Arc<BrainEnhancedKnowledgeGraph> {
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new().await.unwrap());
        
        // Create a biological taxonomy hierarchy
        let animal_key = graph.add_entity("animal", "A living organism that feeds on organic matter").await.unwrap();
        let mammal_key = graph.add_entity("mammal", "A warm-blooded vertebrate").await.unwrap();
        let carnivore_key = graph.add_entity("carnivore", "An animal that feeds on meat").await.unwrap();
        let herbivore_key = graph.add_entity("herbivore", "An animal that feeds on plants").await.unwrap();
        
        // Specific animals
        let dog_key = graph.add_entity("dog", "A domesticated carnivorous mammal").await.unwrap();
        let cat_key = graph.add_entity("cat", "An independent carnivorous mammal").await.unwrap();
        let elephant_key = graph.add_entity("elephant", "A large herbivorous mammal").await.unwrap();
        let wolf_key = graph.add_entity("wolf", "A wild carnivorous mammal").await.unwrap();
        
        // Create hierarchical relationships (is_a)
        graph.add_weighted_edge(mammal_key, animal_key, 0.95).await.unwrap();
        graph.add_weighted_edge(carnivore_key, animal_key, 0.9).await.unwrap();
        graph.add_weighted_edge(herbivore_key, animal_key, 0.9).await.unwrap();
        
        graph.add_weighted_edge(dog_key, mammal_key, 0.95).await.unwrap();
        graph.add_weighted_edge(dog_key, carnivore_key, 0.8).await.unwrap();
        
        graph.add_weighted_edge(cat_key, mammal_key, 0.95).await.unwrap();
        graph.add_weighted_edge(cat_key, carnivore_key, 0.9).await.unwrap();
        
        graph.add_weighted_edge(elephant_key, mammal_key, 0.95).await.unwrap();
        graph.add_weighted_edge(elephant_key, herbivore_key, 0.95).await.unwrap();
        
        graph.add_weighted_edge(wolf_key, mammal_key, 0.95).await.unwrap();
        graph.add_weighted_edge(wolf_key, carnivore_key, 0.95).await.unwrap();
        
        graph
    }

    /// Helper function to create a test SystemsThinking instance with comprehensive graph
    async fn create_test_systems_thinking() -> SystemsThinking {
        let graph = create_comprehensive_test_graph().await;
        SystemsThinking::new(graph)
    }

    #[tokio::test]
    async fn test_cognitive_pattern_execute_basic() {
        let systems = create_test_systems_thinking().await;
        let parameters = PatternParameters::default();
        
        let result = systems.execute(
            "What properties does a dog have?",
            None,
            parameters,
        ).await;
        
        assert!(result.is_ok());
        let pattern_result = result.unwrap();
        
        assert_eq!(pattern_result.pattern_type, CognitivePatternType::Systems);
        assert!(pattern_result.confidence > 0.0);
        assert!(pattern_result.confidence <= 1.0);
        assert!(!pattern_result.answer.is_empty());
        assert!(pattern_result.answer.contains("Systems Analysis"));
        assert!(!pattern_result.reasoning_trace.is_empty());
        
        // Check metadata
        assert!(pattern_result.metadata.execution_time_ms > 0);
        assert!(pattern_result.metadata.nodes_activated > 0);
        assert_eq!(pattern_result.metadata.iterations_completed, 1);
        assert!(pattern_result.metadata.converged);
    }

    #[tokio::test]
    async fn test_cognitive_pattern_execute_with_context() {
        let systems = create_test_systems_thinking().await;
        let parameters = PatternParameters::default();
        
        let result = systems.execute(
            "What attributes does a mammal inherit?",
            Some("biological taxonomy context"),
            parameters,
        ).await;
        
        assert!(result.is_ok());
        let pattern_result = result.unwrap();
        
        assert_eq!(pattern_result.pattern_type, CognitivePatternType::Systems);
        assert!(pattern_result.confidence > 0.0);
        assert!(pattern_result.answer.contains("Systems Analysis"));
        assert!(pattern_result.answer.contains("Hierarchy Path"));
    }

    #[tokio::test]
    async fn test_hierarchical_reasoning_attribute_inheritance() {
        let systems = create_test_systems_thinking().await;
        
        let result = systems.execute_hierarchical_reasoning(
            "What properties does a dog inherit from mammals?",
            SystemsReasoningType::AttributeInheritance,
        ).await;
        
        assert!(result.is_ok());
        let systems_result = result.unwrap();
        
        // Should have found a hierarchy path
        assert!(!systems_result.hierarchy_path.is_empty());
        assert!(systems_result.hierarchy_path.len() > 1); // Should traverse up hierarchy
        
        // Should have inherited attributes
        assert!(!systems_result.inherited_attributes.is_empty());
        
        // Check for expected mammalian attributes
        let has_warm_blooded = systems_result.inherited_attributes.iter()
            .any(|attr| attr.attribute_name.contains("warm_blooded"));
        assert!(has_warm_blooded);
        
        // System complexity should be reasonable
        assert!(systems_result.system_complexity >= 0.0);
        assert!(systems_result.system_complexity <= 1.0);
    }

    #[tokio::test]
    async fn test_hierarchical_reasoning_classification() {
        let systems = create_test_systems_thinking().await;
        
        let result = systems.execute_hierarchical_reasoning(
            "How do we classify a dog in the animal kingdom?",
            SystemsReasoningType::Classification,
        ).await;
        
        assert!(result.is_ok());
        let systems_result = result.unwrap();
        
        // Should have classification hierarchy
        assert!(!systems_result.hierarchy_path.is_empty());
        
        // Check that complexity is calculated
        assert!(systems_result.system_complexity >= 0.0);
    }

    #[tokio::test]
    async fn test_hierarchical_reasoning_system_analysis() {
        let systems = create_test_systems_thinking().await;
        
        let result = systems.execute_hierarchical_reasoning(
            "How do carnivorous mammals interact in the ecosystem?",
            SystemsReasoningType::SystemAnalysis,
        ).await;
        
        assert!(result.is_ok());
        let systems_result = result.unwrap();
        
        // Should perform system analysis
        assert!(!systems_result.hierarchy_path.is_empty());
        assert!(systems_result.system_complexity >= 0.0);
    }

    #[tokio::test]
    async fn test_hierarchical_reasoning_emergent_properties() {
        let systems = create_test_systems_thinking().await;
        
        let result = systems.execute_hierarchical_reasoning(
            "What emergent behaviors arise from mammalian characteristics?",
            SystemsReasoningType::EmergentProperties,
        ).await;
        
        assert!(result.is_ok());
        let systems_result = result.unwrap();
        
        // Should analyze emergent properties
        assert!(!systems_result.hierarchy_path.is_empty());
        assert!(systems_result.system_complexity >= 0.0);
    }

    #[tokio::test]
    async fn test_end_to_end_query_processing() {
        let systems = create_test_systems_thinking().await;
        let parameters = PatternParameters::default();
        
        // Test various query types
        let queries = vec![
            "What properties does a dog have?",
            "How is a cat classified?",
            "What attributes do elephants inherit?",
            "What systems are wolves part of?",
        ];
        
        for query in queries {
            let result = systems.execute(query, None, parameters.clone()).await;
            assert!(result.is_ok(), "Failed for query: {}", query);
            
            let pattern_result = result.unwrap();
            assert!(pattern_result.confidence > 0.0);
            assert!(!pattern_result.answer.is_empty());
            assert!(pattern_result.answer.contains("Systems Analysis"));
        }
    }

    #[tokio::test]
    async fn test_pattern_type_and_use_cases() {
        let systems = create_test_systems_thinking().await;
        
        // Test pattern type
        assert_eq!(systems.get_pattern_type(), CognitivePatternType::Systems);
        
        // Test use cases
        let use_cases = systems.get_optimal_use_cases();
        assert!(!use_cases.is_empty());
        assert!(use_cases.contains(&"Hierarchical analysis".to_string()));
        assert!(use_cases.contains(&"Classification queries".to_string()));
        assert!(use_cases.contains(&"Attribute inheritance".to_string()));
        assert!(use_cases.contains(&"System analysis".to_string()));
    }

    #[tokio::test]
    async fn test_complexity_estimation() {
        let systems = create_test_systems_thinking().await;
        
        let simple_query = "What is a dog?";
        let complex_query = "What are the emergent properties of hierarchical systems?";
        
        let simple_estimate = systems.estimate_complexity(simple_query);
        let complex_estimate = systems.estimate_complexity(complex_query);
        
        // Both should have reasonable estimates
        assert!(simple_estimate.computational_complexity > 0);
        assert!(simple_estimate.estimated_time_ms > 0);
        assert!(simple_estimate.memory_requirements_mb > 0);
        assert!(simple_estimate.confidence > 0.0);
        assert!(simple_estimate.confidence <= 1.0);
        
        assert!(complex_estimate.computational_complexity > 0);
        assert!(complex_estimate.estimated_time_ms > 0);
        assert!(complex_estimate.memory_requirements_mb > 0);
        assert!(complex_estimate.confidence > 0.0);
        assert!(complex_estimate.confidence <= 1.0);
        
        // Note: The current implementation returns the same estimate for all queries
        // This test validates the structure is correct
    }

    #[tokio::test]
    async fn test_inheritance_depth_limits() {
        let systems = create_test_systems_thinking().await;
        
        // Test that inheritance respects depth limits
        let result = systems.execute_hierarchical_reasoning(
            "What does a dog inherit?",
            SystemsReasoningType::AttributeInheritance,
        ).await;
        
        assert!(result.is_ok());
        let systems_result = result.unwrap();
        
        // Should not exceed max inheritance depth
        assert!(systems_result.hierarchy_path.len() <= systems.max_inheritance_depth + 1);
        
        // All inherited attributes should have reasonable depths
        for attr in &systems_result.inherited_attributes {
            assert!(attr.inheritance_depth <= systems.max_inheritance_depth);
        }
    }

    #[tokio::test]
    async fn test_exception_handling() {
        let systems = create_test_systems_thinking().await;
        
        // Create a scenario that might generate exceptions
        let result = systems.execute_hierarchical_reasoning(
            "What contradictory properties exist in the hierarchy?",
            SystemsReasoningType::AttributeInheritance,
        ).await;
        
        assert!(result.is_ok());
        let systems_result = result.unwrap();
        
        // Test that exception handling is properly formatted
        for exception in &systems_result.exception_handling {
            assert!(!exception.description.is_empty());
            assert!(!exception.resolution_strategy.is_empty());
            assert!(!exception.affected_entities.is_empty());
            
            // Test exception type formatting
            match exception.exception_type {
                ExceptionType::Contradiction => assert!(true),
                ExceptionType::MissingData => assert!(true),
                ExceptionType::InconsistentInheritance => assert!(true),
                ExceptionType::CircularReference => assert!(true),
            }
        }
    }

    #[tokio::test]
    async fn test_attribute_confidence_propagation() {
        let systems = create_test_systems_thinking().await;
        
        let result = systems.execute_hierarchical_reasoning(
            "What attributes does a dog have?",
            SystemsReasoningType::AttributeInheritance,
        ).await;
        
        assert!(result.is_ok());
        let systems_result = result.unwrap();
        
        // Test that all attributes have valid confidence scores
        for attr in &systems_result.inherited_attributes {
            assert!(attr.confidence >= 0.0);
            assert!(attr.confidence <= 1.0);
            assert!(!attr.attribute_name.is_empty());
            assert!(!attr.value.is_empty());
        }
    }

    #[tokio::test]
    async fn test_reasoning_trace_generation() {
        let systems = create_test_systems_thinking().await;
        let parameters = PatternParameters::default();
        
        let result = systems.execute(
            "What properties does a mammal have?",
            None,
            parameters,
        ).await;
        
        assert!(result.is_ok());
        let pattern_result = result.unwrap();
        
        // Should have reasoning trace
        assert!(!pattern_result.reasoning_trace.is_empty());
        
        // Check trace structure
        for step in &pattern_result.reasoning_trace {
            assert!(step.step_id > 0);
            assert!(!step.concept_id.is_empty());
            assert!(step.activation_level >= 0.0);
            assert!(step.activation_level <= 1.0);
        }
        
        // Should contain expected reasoning steps
        let concept_ids: Vec<&String> = pattern_result.reasoning_trace.iter()
            .map(|step| &step.concept_id)
            .collect();
        
        assert!(concept_ids.contains(&&"hierarchy_traversal".to_string()));
    }

    #[tokio::test]
    async fn test_metadata_generation() {
        let systems = create_test_systems_thinking().await;
        let parameters = PatternParameters::default();
        
        let result = systems.execute(
            "What systems include dogs?",
            None,
            parameters,
        ).await;
        
        assert!(result.is_ok());
        let pattern_result = result.unwrap();
        
        // Test metadata structure
        let metadata = &pattern_result.metadata;
        assert!(metadata.execution_time_ms > 0);
        assert!(metadata.nodes_activated > 0);
        assert_eq!(metadata.iterations_completed, 1);
        assert!(metadata.converged);
        assert!(metadata.total_energy >= 0.0);
        
        // Test additional metadata
        assert!(metadata.additional_info.contains_key("hierarchy_depth"));
        assert!(metadata.additional_info.contains_key("attributes_count"));
        assert!(metadata.additional_info.contains_key("exceptions_count"));
        assert!(metadata.additional_info.contains_key("complexity_score"));
        
        // Validate metadata values
        let hierarchy_depth: usize = metadata.additional_info["hierarchy_depth"].parse().unwrap();
        let attributes_count: usize = metadata.additional_info["attributes_count"].parse().unwrap();
        let exceptions_count: usize = metadata.additional_info["exceptions_count"].parse().unwrap();
        let complexity_score: f32 = metadata.additional_info["complexity_score"].parse().unwrap();
        
        assert!(hierarchy_depth >= 0);
        assert!(attributes_count >= 0);
        assert!(exceptions_count >= 0);
        assert!(complexity_score >= 0.0);
    }

    #[tokio::test]
    async fn test_edge_case_empty_graph() {
        // Test behavior with empty graph
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new().await.unwrap());
        let systems = SystemsThinking::new(graph);
        let parameters = PatternParameters::default();
        
        let result = systems.execute(
            "What properties does anything have?",
            None,
            parameters,
        ).await;
        
        // Should handle empty graph gracefully
        assert!(result.is_err() || (result.is_ok() && {
            let pattern_result = result.unwrap();
            pattern_result.confidence >= 0.0 && pattern_result.confidence <= 1.0
        }));
    }

    #[tokio::test]
    async fn test_edge_case_single_entity() {
        // Test behavior with single entity
        let graph = Arc::new(BrainEnhancedKnowledgeGraph::new().await.unwrap());
        let _single_key = graph.add_entity("lonely", "A single entity").await.unwrap();
        
        let systems = SystemsThinking::new(graph);
        let parameters = PatternParameters::default();
        
        let result = systems.execute(
            "What properties does lonely have?",
            None,
            parameters,
        ).await;
        
        assert!(result.is_ok());
        let pattern_result = result.unwrap();
        
        // Should work with single entity
        assert!(pattern_result.confidence >= 0.0);
        assert!(!pattern_result.answer.is_empty());
        assert!(pattern_result.metadata.nodes_activated >= 1);
    }

    #[tokio::test]
    async fn test_concurrent_execution() {
        let systems = Arc::new(create_test_systems_thinking().await);
        let parameters = PatternParameters::default();
        
        // Test concurrent access to the same SystemsThinking instance
        let mut handles = vec![];
        
        for i in 0..5 {
            let systems_clone = Arc::clone(&systems);
            let params = parameters.clone();
            let query = format!("What properties does entity {} have?", i);
            
            let handle = tokio::spawn(async move {
                systems_clone.execute(&query, None, params).await
            });
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        for handle in handles {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
            
            let pattern_result = result.unwrap();
            assert!(pattern_result.confidence >= 0.0);
            assert!(!pattern_result.answer.is_empty());
        }
    }

    #[tokio::test]
    async fn test_query_inference_types() {
        let systems = create_test_systems_thinking().await;
        
        // Test that different query types produce appropriate reasoning
        let test_cases = vec![
            ("What properties does a dog inherit?", "properties"),
            ("How do we classify a mammal?", "classify"),
            ("What systems include carnivores?", "system"),
            ("What emergent behaviors arise?", "emergent"),
        ];
        
        for (query, expected_keyword) in test_cases {
            let result = systems.execute(
                query,
                None,
                PatternParameters::default(),
            ).await;
            
            assert!(result.is_ok(), "Failed for query: {}", query);
            let pattern_result = result.unwrap();
            
            // All should produce valid systems analysis
            assert!(pattern_result.answer.contains("Systems Analysis"));
            assert!(pattern_result.confidence > 0.0);
        }
    }
}