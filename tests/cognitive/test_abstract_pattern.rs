#[cfg(test)]
mod abstract_pattern_tests {
    use tokio;
    use crate::cognitive::abstract_pattern::AbstractThinking;
    use crate::cognitive::types::{
        PatternResult, AbstractResult, DetectedPattern, AbstractionCandidate,
        RefactoringOpportunity, EfficiencyAnalysis, AnalysisScope, PatternType,
        CognitivePatternType
    };
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;

    #[tokio::test]
    async fn test_estimate_efficiency_gains_basic() {
        let thinking = create_test_abstract_thinking().await;
        
        // Test with various refactoring opportunities
        let opportunities = vec![
            RefactoringOpportunity {
                opportunity_type: "merge_concepts".to_string(),
                description: "Merge redundant employee concepts".to_string(),
                estimated_query_time_improvement: 0.2,
                estimated_memory_reduction: 0.15,
                estimated_accuracy_gain: 0.1,
                estimated_maintainability_gain: 0.25,
                implementation_complexity: "medium".to_string(),
            },
            RefactoringOpportunity {
                opportunity_type: "add_abstraction".to_string(),
                description: "Create Employment abstraction".to_string(),
                estimated_query_time_improvement: 0.3,
                estimated_memory_reduction: 0.1,
                estimated_accuracy_gain: 0.2,
                estimated_maintainability_gain: 0.3,
                implementation_complexity: "high".to_string(),
            },
        ];
        
        let analysis = thinking.estimate_efficiency_gains(&opportunities);
        
        // Check that gains are calculated correctly
        assert!(analysis.query_time_improvement > 0.0, "Should have query time improvement");
        assert!(analysis.memory_efficiency_gain > 0.0, "Should have memory efficiency gain");
        assert!(analysis.accuracy_improvement > 0.0, "Should have accuracy improvement");
        assert!(analysis.maintainability_improvement > 0.0, "Should have maintainability improvement");
        
        // Verify calculations (sum of individual improvements)
        assert!((analysis.query_time_improvement - 0.5).abs() < 0.01, 
               "Query time should be sum: {}", analysis.query_time_improvement);
    }

    #[tokio::test]
    async fn test_estimate_efficiency_gains_edge_cases() {
        let thinking = create_test_abstract_thinking().await;
        
        // Test with empty opportunities
        let empty_opportunities = vec![];
        let empty_analysis = thinking.estimate_efficiency_gains(&empty_opportunities);
        
        assert_eq!(empty_analysis.query_time_improvement, 0.0);
        assert_eq!(empty_analysis.memory_efficiency_gain, 0.0);
        assert_eq!(empty_analysis.accuracy_improvement, 0.0);
        assert_eq!(empty_analysis.maintainability_improvement, 0.0);
        
        // Test with zero benefit opportunities
        let zero_opportunities = vec![
            RefactoringOpportunity {
                opportunity_type: "no_benefit".to_string(),
                description: "No actual benefit".to_string(),
                estimated_query_time_improvement: 0.0,
                estimated_memory_reduction: 0.0,
                estimated_accuracy_gain: 0.0,
                estimated_maintainability_gain: 0.0,
                implementation_complexity: "low".to_string(),
            },
        ];
        
        let zero_analysis = thinking.estimate_efficiency_gains(&zero_opportunities);
        assert_eq!(zero_analysis.query_time_improvement, 0.0);
    }

    #[tokio::test]
    async fn test_identify_abstractions_happy_path() {
        let thinking = create_test_abstract_thinking().await;
        
        // Create patterns that meet criteria for abstraction
        let patterns = vec![
            DetectedPattern {
                pattern_id: "employee_pattern".to_string(),
                pattern_type: PatternType::Structural,
                frequency: 15,
                confidence: 0.85,
                entities_involved: vec!["employee1".to_string(), "employee2".to_string(), "company".to_string()],
                relationships_involved: vec!["works_for".to_string()],
                description: "Employee-company relationship pattern".to_string(),
                discovered_at: chrono::Utc::now(),
            },
            DetectedPattern {
                pattern_id: "high_freq_pattern".to_string(),
                pattern_type: PatternType::Temporal,
                frequency: 25,
                confidence: 0.9,
                entities_involved: vec!["event1".to_string(), "event2".to_string()],
                relationships_involved: vec!["precedes".to_string()],
                description: "High frequency temporal pattern".to_string(),
                discovered_at: chrono::Utc::now(),
            },
        ];
        
        let candidates = thinking.identify_abstractions(patterns, 10, 0.8).await;
        assert!(candidates.is_ok());
        
        let abstractions = candidates.unwrap();
        assert_eq!(abstractions.len(), 2, "Should identify both patterns as candidates");
        
        // Check first abstraction
        assert_eq!(abstractions[0].pattern_id, "employee_pattern");
        assert!(abstractions[0].proposed_abstraction_name.contains("Employee") || 
                abstractions[0].proposed_abstraction_name.contains("Employment"));
    }

    #[tokio::test]
    async fn test_identify_abstractions_edge_cases() {
        let thinking = create_test_abstract_thinking().await;
        
        // Test with patterns below thresholds
        let low_patterns = vec![
            DetectedPattern {
                pattern_id: "low_freq".to_string(),
                pattern_type: PatternType::Structural,
                frequency: 5, // Below threshold of 10
                confidence: 0.9,
                entities_involved: vec!["entity1".to_string()],
                relationships_involved: vec!["rel1".to_string()],
                description: "Low frequency pattern".to_string(),
                discovered_at: chrono::Utc::now(),
            },
            DetectedPattern {
                pattern_id: "low_conf".to_string(),
                pattern_type: PatternType::Structural,
                frequency: 15,
                confidence: 0.7, // Below threshold of 0.8
                entities_involved: vec!["entity2".to_string()],
                relationships_involved: vec!["rel2".to_string()],
                description: "Low confidence pattern".to_string(),
                discovered_at: chrono::Utc::now(),
            },
        ];
        
        let candidates = thinking.identify_abstractions(low_patterns, 10, 0.8).await;
        assert!(candidates.is_ok());
        
        let abstractions = candidates.unwrap();
        assert_eq!(abstractions.len(), 0, "Should filter out patterns below thresholds");
        
        // Test with empty input
        let empty_candidates = thinking.identify_abstractions(vec![], 10, 0.8).await;
        assert!(empty_candidates.is_ok());
        assert_eq!(empty_candidates.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_infer_analysis_parameters() {
        let thinking = create_test_abstract_thinking().await;
        
        // Test structural analysis keywords
        let (scope, pattern_type) = thinking.infer_analysis_parameters(
            "analyze the graph structure for patterns"
        ).await;
        
        assert_eq!(scope, AnalysisScope::Global);
        assert_eq!(pattern_type, PatternType::Structural);
        
        // Test temporal analysis keywords
        let (scope, pattern_type) = thinking.infer_analysis_parameters(
            "find temporal patterns in recent data"
        ).await;
        
        assert_eq!(pattern_type, PatternType::Temporal);
        
        // Test local scope keywords
        let (scope, pattern_type) = thinking.infer_analysis_parameters(
            "local analysis of specific nodes"
        ).await;
        
        assert_eq!(scope, AnalysisScope::Local);
    }

    #[tokio::test]
    async fn test_infer_analysis_parameters_edge_cases() {
        let thinking = create_test_abstract_thinking().await;
        
        // Test with no keywords (should trigger default)
        let (scope, pattern_type) = thinking.infer_analysis_parameters(
            "some random text without keywords"
        ).await;
        
        // Should use default values
        assert_eq!(scope, AnalysisScope::Global);
        assert_eq!(pattern_type, PatternType::Structural);
        
        // Test empty query
        let (scope, pattern_type) = thinking.infer_analysis_parameters("").await;
        assert_eq!(scope, AnalysisScope::Global);
        assert_eq!(pattern_type, PatternType::Structural);
        
        // Test conflicting terms (should prioritize based on implementation)
        let (scope, pattern_type) = thinking.infer_analysis_parameters(
            "structural temporal global local analysis"
        ).await;
        
        // Should resolve to one of each (implementation dependent)
        assert!(matches!(scope, AnalysisScope::Global | AnalysisScope::Local));
        assert!(matches!(pattern_type, PatternType::Structural | PatternType::Temporal));
    }

    #[tokio::test]
    async fn test_end_to_end_analysis_workflow() {
        // Create predictable test graph with clear patterns
        let graph = create_predictable_test_graph().await;
        let thinking = AbstractThinking::new(graph);
        
        // Execute main analysis
        let result = thinking.execute("analyze the graph structure for patterns").await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        match pattern_result {
            PatternResult::Abstract(abstract_result) => {
                // Verify detected patterns
                assert!(abstract_result.detected_patterns.len() > 0, 
                       "Should detect patterns in test graph");
                
                // Check for expected employment pattern
                let has_employment_pattern = abstract_result.detected_patterns.iter()
                    .any(|p| p.description.contains("employee") || p.description.contains("works_for"));
                assert!(has_employment_pattern, "Should detect employee-company pattern");
                
                // Verify abstraction candidates
                assert!(abstract_result.abstraction_candidates.len() > 0, 
                       "Should propose abstraction candidates");
                
                // Check for Employment abstraction
                let has_employment_abstraction = abstract_result.abstraction_candidates.iter()
                    .any(|a| a.proposed_abstraction_name.contains("Employment"));
                assert!(has_employment_abstraction, "Should propose Employment abstraction");
                
                // Verify refactoring opportunities
                assert!(abstract_result.refactoring_opportunities.len() > 0, 
                       "Should suggest refactoring opportunities");
                
                // Verify efficiency gains are positive
                assert!(abstract_result.efficiency_gains.query_time_improvement > 0.0, 
                       "Should have positive efficiency gains");
                
                assert_eq!(abstract_result.pattern_type, CognitivePatternType::Abstract);
            },
            _ => panic!("Expected AbstractResult")
        }
    }

    #[tokio::test]
    async fn test_mocked_detector_integration() {
        // Test with mock detector returning specific patterns
        let graph = create_test_graph().await;
        let thinking = AbstractThinking::new(graph);
        
        // Create mock patterns to simulate detector output
        let mock_patterns = vec![
            DetectedPattern {
                pattern_id: "mock_pattern_1".to_string(),
                pattern_type: PatternType::Structural,
                frequency: 20,
                confidence: 0.9,
                entities_involved: vec!["node_a".to_string(), "node_b".to_string()],
                relationships_involved: vec!["connects".to_string()],
                description: "Mock structural pattern".to_string(),
                discovered_at: chrono::Utc::now(),
            },
        ];
        
        // Test abstraction identification with known input
        let candidates = thinking.identify_abstractions(mock_patterns.clone(), 15, 0.85).await;
        assert!(candidates.is_ok());
        
        let abstractions = candidates.unwrap();
        assert_eq!(abstractions.len(), 1, "Should process mock pattern");
        assert_eq!(abstractions[0].pattern_id, "mock_pattern_1");
        
        // Test refactoring suggestions
        let opportunities = thinking.suggest_refactoring(abstractions).await;
        assert!(opportunities.is_ok());
        
        let refactoring = opportunities.unwrap();
        assert!(refactoring.len() > 0, "Should generate refactoring opportunities");
    }

    #[tokio::test]
    async fn test_suggest_refactoring() {
        let thinking = create_test_abstract_thinking().await;
        
        // Create abstraction candidates to process
        let candidates = vec![
            AbstractionCandidate {
                pattern_id: "employee_pattern".to_string(),
                proposed_abstraction_name: "Employment".to_string(),
                abstraction_description: "Employment relationship abstraction".to_string(),
                entities_to_abstract: vec!["employee1".to_string(), "employee2".to_string()],
                expected_benefits: "Reduced redundancy, improved queries".to_string(),
                implementation_complexity: "medium".to_string(),
            },
        ];
        
        let opportunities = thinking.suggest_refactoring(candidates).await;
        assert!(opportunities.is_ok());
        
        let refactoring = opportunities.unwrap();
        assert!(refactoring.len() > 0, "Should generate refactoring opportunities");
        
        // Check refactoring opportunity structure
        let first_op = &refactoring[0];
        assert!(!first_op.opportunity_type.is_empty());
        assert!(!first_op.description.is_empty());
        assert!(first_op.estimated_query_time_improvement >= 0.0);
        assert!(first_op.estimated_memory_reduction >= 0.0);
    }

    #[tokio::test]
    async fn test_cognitive_pattern_interface() {
        let thinking = create_test_abstract_thinking().await;
        
        let result = thinking.execute("analyze patterns for abstraction opportunities").await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        match pattern_result {
            PatternResult::Abstract(abstract_result) => {
                assert_eq!(abstract_result.pattern_type, CognitivePatternType::Abstract);
                assert!(abstract_result.analysis_scope == AnalysisScope::Global || 
                       abstract_result.analysis_scope == AnalysisScope::Local);
                assert!(!abstract_result.analysis_summary.is_empty());
            },
            _ => panic!("Expected AbstractResult from abstract thinking")
        }
    }

    // Helper functions

    async fn create_test_abstract_thinking() -> AbstractThinking {
        let graph = create_test_graph().await;
        AbstractThinking::new(graph)
    }

    async fn create_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create basic test structure
        graph.add_entity("concept1", "Test concept").await.unwrap();
        graph.add_entity("concept2", "Another concept").await.unwrap();
        graph.add_entity("relation1", "Test relation").await.unwrap();
        
        graph.add_relationship("concept1", "concept2", "connects", 0.8).await.unwrap();
        graph.add_relationship("concept2", "relation1", "has", 0.7).await.unwrap();
        
        graph
    }

    async fn create_predictable_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create clear employee-company pattern
        graph.add_entity("employee1", "First employee").await.unwrap();
        graph.add_entity("employee2", "Second employee").await.unwrap();
        graph.add_entity("employee3", "Third employee").await.unwrap();
        graph.add_entity("company", "The company").await.unwrap();
        graph.add_entity("department", "Department").await.unwrap();
        
        // Create repeating pattern: employees work for company
        graph.add_relationship("employee1", "company", "works_for", 0.9).await.unwrap();
        graph.add_relationship("employee2", "company", "works_for", 0.9).await.unwrap();
        graph.add_relationship("employee3", "company", "works_for", 0.9).await.unwrap();
        
        // Additional structure
        graph.add_relationship("employee1", "department", "belongs_to", 0.8).await.unwrap();
        graph.add_relationship("employee2", "department", "belongs_to", 0.8).await.unwrap();
        graph.add_relationship("department", "company", "part_of", 0.9).await.unwrap();
        
        graph
    }
}