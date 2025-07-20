#[cfg(test)]
mod critical_tests {
    use tokio;
    use crate::cognitive::critical::CriticalThinking;
    use crate::cognitive::types::{
        PatternResult, CriticalResult, ValidationLevel, FactInfo, 
        Contradiction, ConfidenceInterval, CognitivePatternType
    };
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;

    #[tokio::test]
    async fn test_are_contradictory_direct_conflict() {
        let thinking = create_test_critical_thinking().await;
        
        // Create contradictory facts about the same entity
        let fact1 = FactInfo {
            entity: "tripper".to_string(),
            property: "legs".to_string(),
            value: "3".to_string(),
            confidence: 0.8,
            source: "user_input".to_string(),
        };
        
        let fact2 = FactInfo {
            entity: "tripper".to_string(),
            property: "legs".to_string(),
            value: "4".to_string(),
            confidence: 0.7,
            source: "neural_query".to_string(),
        };
        
        assert!(thinking.are_contradictory(&fact1, &fact2), 
                "Facts with different values for same property should be contradictory");
    }

    #[tokio::test]
    async fn test_are_contradictory_non_conflicts() {
        let thinking = create_test_critical_thinking().await;
        
        // Facts about different entities
        let fact1 = FactInfo {
            entity: "dog".to_string(),
            property: "legs".to_string(),
            value: "4".to_string(),
            confidence: 0.9,
            source: "neural_query".to_string(),
        };
        
        let fact2 = FactInfo {
            entity: "cat".to_string(),
            property: "legs".to_string(),
            value: "4".to_string(),
            confidence: 0.9,
            source: "neural_query".to_string(),
        };
        
        assert!(!thinking.are_contradictory(&fact1, &fact2), 
                "Facts about different entities should not be contradictory");
        
        // Different properties of same entity
        let fact3 = FactInfo {
            entity: "dog".to_string(),
            property: "color".to_string(),
            value: "brown".to_string(),
            confidence: 0.8,
            source: "user_input".to_string(),
        };
        
        assert!(!thinking.are_contradictory(&fact1, &fact3), 
                "Different properties should not be contradictory");
    }

    #[tokio::test]
    async fn test_calculate_source_reliability() {
        let thinking = create_test_critical_thinking().await;
        
        // Test known source reliability scores
        assert_eq!(thinking.calculate_source_reliability("neural_query"), 0.9);
        assert_eq!(thinking.calculate_source_reliability("user_input"), 0.7);
        assert_eq!(thinking.calculate_source_reliability("external_api"), 0.6);
        assert_eq!(thinking.calculate_source_reliability("unknown_source"), 0.5);
    }

    #[tokio::test]
    async fn test_contradiction_detection_and_resolution() {
        // Create a graph with contradictory information
        let graph = create_contradictory_test_graph().await;
        let thinking = CriticalThinking::new(graph);
        
        // Execute critical analysis on conflicting data
        let result = thinking.execute("How many legs does Tripper have?").await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        match pattern_result {
            PatternResult::Critical(crit_result) => {
                // Should detect the contradiction
                assert!(crit_result.contradictions_found.len() > 0, 
                       "Should detect contradictions in test data");
                
                // Should resolve to higher confidence fact
                assert!(crit_result.validated_facts.len() > 0, 
                       "Should have validated facts after resolution");
                
                // Check that resolution picked the more reliable source
                let resolved_fact = &crit_result.validated_facts[0];
                assert!(resolved_fact.confidence > 0.7, 
                       "Resolved fact should have high confidence");
                
                assert!(crit_result.overall_confidence > 0.0 && crit_result.overall_confidence <= 1.0,
                       "Overall confidence should be in valid range");
            },
            _ => panic!("Expected CriticalResult")
        }
    }

    #[tokio::test]
    async fn test_source_validation_and_uncertainty() {
        let thinking = create_test_critical_thinking().await;
        
        // Create facts from different reliability sources
        let high_reliability_facts = vec![
            FactInfo {
                entity: "concept".to_string(),
                property: "verified".to_string(),
                value: "true".to_string(),
                confidence: 0.9,
                source: "neural_query".to_string(),
            }
        ];
        
        let low_reliability_facts = vec![
            FactInfo {
                entity: "concept".to_string(),
                property: "uncertain".to_string(),
                value: "maybe".to_string(),
                confidence: 0.6,
                source: "user_input".to_string(),
            }
        ];
        
        // Test confidence intervals for high reliability source
        let high_intervals = thinking.validate_information_sources(high_reliability_facts).await;
        assert!(high_intervals.is_ok());
        
        let high_result = high_intervals.unwrap();
        let high_interval = &high_result[0].confidence_interval;
        let high_range = high_interval.upper_bound - high_interval.lower_bound;
        
        // Test confidence intervals for low reliability source
        let low_intervals = thinking.validate_information_sources(low_reliability_facts).await;
        assert!(low_intervals.is_ok());
        
        let low_result = low_intervals.unwrap();
        let low_interval = &low_result[0].confidence_interval;
        let low_range = low_interval.upper_bound - low_interval.lower_bound;
        
        // Low reliability source should have wider confidence intervals
        assert!(low_range > high_range, 
               "Low reliability source should have wider confidence interval: {} vs {}", 
               low_range, high_range);
    }

    #[tokio::test]
    async fn test_uncertainty_analysis() {
        let graph = create_mixed_reliability_graph().await;
        let thinking = CriticalThinking::new(graph);
        
        let result = thinking.execute_critical_analysis(
            "What properties does the mixed entity have?", 
            ValidationLevel::Thorough
        ).await;
        
        assert!(result.is_ok());
        let crit_result = result.unwrap();
        
        // Should have uncertainty analysis
        assert!(crit_result.uncertainty_analysis.overall_uncertainty >= 0.0);
        assert!(crit_result.uncertainty_analysis.overall_uncertainty <= 1.0);
        
        // Should track source reliability distribution
        assert!(crit_result.uncertainty_analysis.source_reliability_distribution.len() > 0);
    }

    #[tokio::test]
    async fn test_validation_levels() {
        let graph = create_test_graph().await;
        let thinking = CriticalThinking::new(graph);
        
        // Test basic validation
        let basic_result = thinking.execute_critical_analysis(
            "Test query", ValidationLevel::Basic
        ).await;
        assert!(basic_result.is_ok());
        
        // Test thorough validation
        let thorough_result = thinking.execute_critical_analysis(
            "Test query", ValidationLevel::Thorough
        ).await;
        assert!(thorough_result.is_ok());
        
        // Thorough should generally have more detailed analysis
        let basic = basic_result.unwrap();
        let thorough = thorough_result.unwrap();
        
        // This is a heuristic - thorough validation might find more issues
        assert!(thorough.uncertainty_analysis.confidence_intervals.len() >= 
                basic.uncertainty_analysis.confidence_intervals.len());
    }

    #[tokio::test]
    async fn test_inhibitory_logic_application() {
        let graph = create_conflict_test_graph().await;
        let thinking = CriticalThinking::new(graph);
        
        // Test that inhibitory logic correctly resolves conflicts
        let contradictions = vec![
            Contradiction {
                fact1: create_test_fact("entity", "prop", "value1", 0.8, "source1"),
                fact2: create_test_fact("entity", "prop", "value2", 0.6, "source2"),
                conflict_type: "direct_contradiction".to_string(),
                resolution_confidence: 0.0, // Will be calculated
            }
        ];
        
        let resolved = thinking.apply_inhibitory_logic(contradictions).await;
        assert!(resolved.is_ok());
        
        let resolution = resolved.unwrap();
        assert!(resolution.len() > 0, "Should resolve contradictions");
        
        // Should prefer higher confidence fact
        assert_eq!(resolution[0].value, "value1");
        assert!(resolution[0].confidence > 0.6);
    }

    #[tokio::test]
    async fn test_cognitive_pattern_interface() {
        let thinking = create_test_critical_thinking().await;
        
        let result = thinking.execute("Validate this information").await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        match pattern_result {
            PatternResult::Critical(crit_result) => {
                assert_eq!(crit_result.pattern_type, CognitivePatternType::Critical);
                assert!(crit_result.overall_confidence >= 0.0 && crit_result.overall_confidence <= 1.0);
                assert!(crit_result.validation_level == ValidationLevel::Basic || 
                       crit_result.validation_level == ValidationLevel::Thorough);
            },
            _ => panic!("Expected CriticalResult from critical thinking")
        }
    }

    #[tokio::test]
    async fn test_confidence_interval_calculation() {
        let thinking = create_test_critical_thinking().await;
        
        // Test confidence interval calculation for different scenarios
        let fact = create_test_fact("test", "property", "value", 0.7, "neural_query");
        let interval = thinking.calculate_confidence_interval(&fact).await;
        
        assert!(interval.lower_bound <= fact.confidence, 
               "Lower bound should not exceed original confidence");
        assert!(interval.upper_bound >= fact.confidence, 
               "Upper bound should not be below original confidence");
        assert!(interval.lower_bound >= 0.0 && interval.upper_bound <= 1.0,
               "Confidence interval should be within [0,1]");
    }

    // Helper functions

    async fn create_test_critical_thinking() -> CriticalThinking {
        let graph = create_test_graph().await;
        CriticalThinking::new(graph)
    }

    async fn create_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("test_entity", "Test entity").await.unwrap();
        graph.add_entity("property1", "Test property").await.unwrap();
        graph.add_entity("value1", "Test value").await.unwrap();
        
        graph.add_relationship("test_entity", "property1", "has", 0.8).await.unwrap();
        graph.add_relationship("property1", "value1", "equals", 0.9).await.unwrap();
        
        graph
    }

    async fn create_contradictory_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create entity with contradictory information
        graph.add_entity("tripper", "A test dog").await.unwrap();
        graph.add_entity("legs_3", "Has 3 legs").await.unwrap();
        graph.add_entity("legs_4", "Has 4 legs").await.unwrap();
        graph.add_entity("dog_type", "Is a dog").await.unwrap();
        
        // Contradictory relationships - Tripper has both 3 and 4 legs
        graph.add_relationship("tripper", "legs_3", "has_property", 0.8).await.unwrap();
        graph.add_relationship("tripper", "legs_4", "has_property", 0.7).await.unwrap();
        graph.add_relationship("tripper", "dog_type", "is_a", 0.9).await.unwrap();
        
        graph
    }

    async fn create_mixed_reliability_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("mixed_entity", "Entity with mixed source data").await.unwrap();
        graph.add_entity("reliable_property", "From reliable source").await.unwrap();
        graph.add_entity("unreliable_property", "From unreliable source").await.unwrap();
        
        // Mix of high and low reliability connections
        graph.add_relationship("mixed_entity", "reliable_property", "has", 0.9).await.unwrap();
        graph.add_relationship("mixed_entity", "unreliable_property", "has", 0.4).await.unwrap();
        
        graph
    }

    async fn create_conflict_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("conflict_entity", "Entity with conflicts").await.unwrap();
        graph.add_entity("conflicting_prop", "Property with conflicts").await.unwrap();
        
        graph.add_relationship("conflict_entity", "conflicting_prop", "has", 0.5).await.unwrap();
        
        graph
    }

    fn create_test_fact(entity: &str, property: &str, value: &str, confidence: f32, source: &str) -> FactInfo {
        FactInfo {
            entity: entity.to_string(),
            property: property.to_string(),
            value: value.to_string(),
            confidence,
            source: source.to_string(),
        }
    }
}