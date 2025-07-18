#[cfg(test)]
mod phase1_cognitive_tests {
    use llmkg::cognitive::{
        ConvergentThinking, DivergentThinking, CriticalThinking,
        SystemsThinking, LateralThinking, AbstractThinking, AdaptiveThinking,
        CognitiveOrchestrator, CognitiveOrchestratorConfig, CognitivePatternType,
        ReasoningStrategy
    };
    use llmkg::cognitive::types::*;
    use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use llmkg::core::brain_types::*;
    use llmkg::core::types::{EntityKey, AttributeValue};
    use std::sync::Arc;
    use std::time::Instant;

    /// Helper to create a minimal test graph with real data
    async fn create_minimal_test_graph() -> Arc<BrainEnhancedKnowledgeGraph> {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Add minimal test data - just enough to test
        let mut entity_keys = vec![];
        
        // Create a few entities
        let ai_key = graph.create_brain_entity(
            "artificial_intelligence".to_string(),
            EntityDirection::Input
        ).await.unwrap();
        
        let ml_key = graph.create_brain_entity(
            "machine_learning".to_string(),
            EntityDirection::Gate
        ).await.unwrap();
        
        let dl_key = graph.create_brain_entity(
            "deep_learning".to_string(),
            EntityDirection::Gate
        ).await.unwrap();
        
        // Create relationships
        graph.create_brain_relationship(
            ai_key,
            ml_key,
            RelationType::IsA,
            0.9
        ).await.unwrap();
        
        graph.create_brain_relationship(
            ml_key,
            dl_key,
            RelationType::IsA,
            0.95
        ).await.unwrap();
        
        Arc::new(graph)
    }

    // Test 1: Convergent Thinking - Basic functionality
    #[tokio::test]
    async fn test_convergent_basic_functionality() {
        // Step 1: Create real graph
        let graph = create_minimal_test_graph().await;
        
        // Step 2: Create convergent thinking instance
        let convergent = ConvergentThinking::new(graph.clone());
        
        // Step 3: Test basic query
        let result = convergent.execute_convergent_query(
            "What is machine learning?",
            None
        ).await;
        
        // Step 4: Verify result
        assert!(result.is_ok(), "Query should succeed");
        let result = result.unwrap();
        assert!(!result.answer.is_empty(), "Should return an answer");
        assert!(result.confidence > 0.0, "Should have confidence score");
    }

    // Test 2: Convergent Thinking - Edge case with deep hierarchy
    #[tokio::test]
    async fn test_convergent_edge_case_deep_hierarchy() {
        let graph = create_minimal_test_graph().await;
        let convergent = ConvergentThinking::new(graph.clone());
        
        // Test with very specific query requiring traversal
        let result = convergent.execute_convergent_query(
            "What is the root concept of deep learning?",
            Some("Looking for the highest level category")
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        // Should find AI as the root
        assert!(result.reasoning_trace.len() > 1, "Should show traversal path");
    }

    // Test 3: Divergent Thinking - Basic functionality
    #[tokio::test]
    async fn test_divergent_basic_functionality() {
        let graph = create_minimal_test_graph().await;
        let divergent = DivergentThinking::new(graph.clone());
        
        let result = divergent.execute(
            "What are related to machine learning?",
            None,
            PatternParameters::default()
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.answer.is_empty(), "Should find suggestions");
        assert!(result.confidence > 0.0);
    }

    // Test 4: Critical Thinking - Basic contradiction detection
    #[tokio::test]
    async fn test_critical_basic_functionality() {
        let graph = create_minimal_test_graph().await;
        
        // Add contradictory data
        let ml_key = graph.get_entity_by_name("machine_learning").await.unwrap();
        let ai_key = graph.get_entity_by_name("artificial_intelligence").await.unwrap();
        
        // Add reverse relationship (contradiction)
        graph.create_brain_relationship(
            ml_key,
            ai_key,
            RelationType::IsA,
            0.3  // Low confidence
        ).await.unwrap();
        
        let critical = CriticalThinking::new(graph.clone());
        let result = critical.execute_critical_analysis(
            "Is machine learning a type of AI or vice versa?",
            None
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.contradictions.is_empty(), "Should detect contradiction");
    }

    // Test 5: Orchestrator Integration - Verify it can select patterns
    #[tokio::test]
    async fn test_orchestrator_pattern_selection() {
        let graph = create_minimal_test_graph().await;
        let config = CognitiveOrchestratorConfig::default();
        let orchestrator = CognitiveOrchestrator::new(graph.clone(), config).await.unwrap();
        
        // Test automatic pattern selection
        let result = orchestrator.reason(
            "What is deep learning?",
            None,
            ReasoningStrategy::Automatic
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.pattern_used, CognitivePatternType::Convergent);
    }

    // ===== CONVERGENT THINKING TESTS (4 total) =====
    
    // Test 6: Convergent - Performance stress test
    #[tokio::test]
    async fn test_convergent_performance_stress() {
        let graph = create_minimal_test_graph().await;
        
        // Add more data for stress test
        for i in 0..100 {
            graph.create_brain_entity(
                format!("concept_{}", i),
                EntityDirection::Gate
            ).await.unwrap();
        }
        
        let convergent = ConvergentThinking::new(graph.clone());
        let start = Instant::now();
        
        let result = convergent.execute_convergent_query(
            "Find the most specific AI concept",
            None
        ).await;
        
        let duration = start.elapsed();
        assert!(result.is_ok());
        assert!(duration.as_millis() < 500, "Query took too long: {:?}", duration);
    }

    // Test 7: Convergent - Integration with graph features
    #[tokio::test]
    async fn test_convergent_integration() {
        let graph = create_minimal_test_graph().await;
        
        // Test with activation states
        let ml_key = graph.get_entity_by_name("machine_learning").await.unwrap();
        graph.update_activation(ml_key, 0.9).await.unwrap();
        
        let convergent = ConvergentThinking::new(graph.clone());
        let result = convergent.execute_convergent_query(
            "What has high activation?",
            None
        ).await;
        
        assert!(result.is_ok());
        // Should find the activated entity
        let result = result.unwrap();
        assert!(result.reasoning_trace.iter().any(|r| r.contains("machine_learning")));
    }

    // ===== DIVERGENT THINKING TESTS (4 total) =====

    // Test 8: Divergent - Edge case with limited connections
    #[tokio::test]
    async fn test_divergent_edge_case_limited() {
        let graph = create_minimal_test_graph().await;
        
        // Create isolated entity
        let isolated = graph.create_brain_entity(
            "isolated_concept".to_string(),
            EntityDirection::Output
        ).await.unwrap();
        
        let divergent = DivergentThinking::new(graph.clone());
        let result = divergent.execute_divergent_exploration(
            "What connects to isolated_concept?",
            None
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        // Should handle gracefully with few or no connections
        assert!(result.suggestions.len() <= 1);
    }

    // Test 9: Divergent - Performance with broad exploration
    #[tokio::test]
    async fn test_divergent_performance() {
        let graph = create_minimal_test_graph().await;
        
        // Add interconnected concepts
        let mut keys = vec![];
        for i in 0..20 {
            let key = graph.create_brain_entity(
                format!("concept_{}", i),
                EntityDirection::Gate
            ).await.unwrap();
            keys.push(key);
        }
        
        // Create mesh of connections
        for i in 0..keys.len() {
            for j in i+1..keys.len().min(i+3) {
                graph.create_brain_relationship(
                    keys[i],
                    keys[j],
                    RelationType::RelatedTo,
                    0.7
                ).await.unwrap();
            }
        }
        
        let divergent = DivergentThinking::new_with_params(
            graph.clone(),
            50, // exploration_breadth
            0.3, // creativity_threshold
            0.5  // novelty_weight
        );
        
        let start = Instant::now();
        let result = divergent.execute_divergent_exploration(
            "Explore all concepts",
            None
        ).await;
        
        assert!(result.is_ok());
        assert!(start.elapsed().as_millis() < 1000);
        assert!(result.unwrap().suggestions.len() >= 10);
    }

    // Test 10: Divergent - Integration with creativity scoring
    #[tokio::test]
    async fn test_divergent_creativity_integration() {
        let graph = create_minimal_test_graph().await;
        let divergent = DivergentThinking::new(graph.clone());
        
        let result = divergent.execute_divergent_exploration(
            "Creative connections from AI",
            Some("Looking for unexpected relationships")
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        // Check creativity scores are applied
        assert!(result.creativity_score > 0.0);
        assert!(result.exploration_depth > 0);
    }

    // ===== CRITICAL THINKING TESTS (4 total) =====

    // Test 11: Critical - Edge case with no contradictions
    #[tokio::test]
    async fn test_critical_edge_case_no_contradictions() {
        let graph = create_minimal_test_graph().await;
        let critical = CriticalThinking::new(graph.clone());
        
        let result = critical.execute_critical_analysis(
            "Is AI a subset of ML?",
            None
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        // Should handle case with no contradictions
        assert!(result.contradictions.is_empty());
        assert!(result.confidence >= 0.0);
    }

    // Test 12: Critical - Performance with many contradictions
    #[tokio::test]
    async fn test_critical_performance() {
        let graph = create_minimal_test_graph().await;
        
        // Add multiple contradictory relationships
        let ai_key = graph.get_entity_by_name("artificial_intelligence").await.unwrap();
        let ml_key = graph.get_entity_by_name("machine_learning").await.unwrap();
        let dl_key = graph.get_entity_by_name("deep_learning").await.unwrap();
        
        // Create contradictory cycles
        graph.create_brain_relationship(dl_key, ml_key, RelationType::IsA, 0.2).await.unwrap();
        graph.create_brain_relationship(ml_key, dl_key, RelationType::IsA, 0.3).await.unwrap();
        
        let critical = CriticalThinking::new(graph.clone());
        let start = Instant::now();
        
        let result = critical.execute_critical_analysis(
            "Analyze all relationships",
            None
        ).await;
        
        assert!(result.is_ok());
        assert!(start.elapsed().as_millis() < 200);
    }

    // Test 13: Critical - Integration with validation
    #[tokio::test]
    async fn test_critical_validation_integration() {
        let graph = create_minimal_test_graph().await;
        let critical = CriticalThinking::new(graph.clone());
        
        let result = critical.execute_critical_analysis(
            "Validate the AI knowledge hierarchy",
            Some("Check for consistency")
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.validation_results.is_empty());
    }

    // ===== SYSTEMS THINKING TESTS (4 total) =====

    // Test 14: Systems - Basic hierarchy traversal
    #[tokio::test]
    async fn test_systems_basic_functionality() {
        let graph = create_minimal_test_graph().await;
        let systems = SystemsThinking::new(graph.clone());
        
        let result = systems.execute_hierarchical_reasoning(
            "Show the hierarchy of deep learning",
            None
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.hierarchy_levels > 0);
        assert!(!result.inheritance_chain.is_empty());
    }

    // Test 15: Systems - Edge case with circular hierarchy
    #[tokio::test]
    async fn test_systems_edge_case_circular() {
        let graph = create_minimal_test_graph().await;
        
        // Create circular reference
        let ai_key = graph.get_entity_by_name("artificial_intelligence").await.unwrap();
        let dl_key = graph.get_entity_by_name("deep_learning").await.unwrap();
        graph.create_brain_relationship(ai_key, dl_key, RelationType::IsA, 0.1).await.unwrap();
        
        let systems = SystemsThinking::new(graph.clone());
        let result = systems.execute_hierarchical_reasoning(
            "Trace complete hierarchy",
            None
        ).await;
        
        assert!(result.is_ok());
        // Should handle circular references gracefully
    }

    // Test 16: Systems - Performance with deep hierarchy
    #[tokio::test]
    async fn test_systems_performance_deep() {
        let graph = create_minimal_test_graph().await;
        
        // Create deep hierarchy
        let mut prev_key = graph.get_entity_by_name("deep_learning").await.unwrap();
        for i in 0..10 {
            let key = graph.create_brain_entity(
                format!("level_{}", i),
                EntityDirection::Gate
            ).await.unwrap();
            graph.create_brain_relationship(key, prev_key, RelationType::IsA, 0.9).await.unwrap();
            prev_key = key;
        }
        
        let systems = SystemsThinking::new(graph.clone());
        let start = Instant::now();
        
        let result = systems.execute_hierarchical_reasoning(
            "Find complete hierarchy of level_9",
            None
        ).await;
        
        assert!(result.is_ok());
        assert!(start.elapsed().as_millis() < 150);
        assert!(result.unwrap().hierarchy_levels >= 10);
    }

    // Test 17: Systems - Integration with inheritance
    #[tokio::test]
    async fn test_systems_inheritance_integration() {
        let graph = create_minimal_test_graph().await;
        
        // Add properties to test inheritance
        let ai_key = graph.get_entity_by_name("artificial_intelligence").await.unwrap();
        graph.add_entity_property(ai_key, "computational", AttributeValue::Boolean(true)).await.unwrap();
        
        let systems = SystemsThinking::new(graph.clone());
        let result = systems.execute_hierarchical_reasoning(
            "What properties does deep learning inherit?",
            None
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.inherited_properties.is_empty());
    }

    // ===== LATERAL THINKING TESTS (4 total) =====

    // Test 18: Lateral - Basic bridge finding
    #[tokio::test]
    async fn test_lateral_basic_functionality() {
        let graph = create_minimal_test_graph().await;
        
        // Add bridge concept
        let bridge = graph.create_brain_entity("neural_networks".to_string(), EntityDirection::Gate).await.unwrap();
        let ai_key = graph.get_entity_by_name("artificial_intelligence").await.unwrap();
        let dl_key = graph.get_entity_by_name("deep_learning").await.unwrap();
        
        graph.create_brain_relationship(ai_key, bridge, RelationType::RelatedTo, 0.7).await.unwrap();
        graph.create_brain_relationship(bridge, dl_key, RelationType::RelatedTo, 0.8).await.unwrap();
        
        let lateral = LateralThinking::new(graph.clone());
        let result = lateral.find_creative_connections(
            "artificial_intelligence",
            "deep_learning",
            None
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.bridge_concepts.is_empty());
    }

    // Test 19: Lateral - Edge case with no bridges
    #[tokio::test]
    async fn test_lateral_edge_case_no_bridges() {
        let graph = create_minimal_test_graph().await;
        
        // Create disconnected concepts
        let isolated1 = graph.create_brain_entity("isolated1".to_string(), EntityDirection::Output).await.unwrap();
        let isolated2 = graph.create_brain_entity("isolated2".to_string(), EntityDirection::Output).await.unwrap();
        
        let lateral = LateralThinking::new(graph.clone());
        let result = lateral.find_creative_connections(
            "isolated1",
            "isolated2",
            None
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.bridge_concepts.len(), 0);
    }

    // Test 20: Lateral - Performance with many paths
    #[tokio::test]
    async fn test_lateral_performance() {
        let graph = create_minimal_test_graph().await;
        
        // Create network with many possible paths
        for i in 0..10 {
            let key = graph.create_brain_entity(
                format!("bridge_{}", i),
                EntityDirection::Gate
            ).await.unwrap();
            
            let ai_key = graph.get_entity_by_name("artificial_intelligence").await.unwrap();
            let ml_key = graph.get_entity_by_name("machine_learning").await.unwrap();
            
            graph.create_brain_relationship(ai_key, key, RelationType::RelatedTo, 0.5).await.unwrap();
            graph.create_brain_relationship(key, ml_key, RelationType::RelatedTo, 0.5).await.unwrap();
        }
        
        let lateral = LateralThinking::new_with_params(
            graph.clone(),
            0.3, // novelty_threshold
            5,   // max_bridge_length
            1.5  // creativity_boost
        );
        
        let start = Instant::now();
        let result = lateral.find_creative_connections(
            "artificial_intelligence",
            "machine_learning",
            None
        ).await;
        
        assert!(result.is_ok());
        assert!(start.elapsed().as_millis() < 300);
    }

    // Test 21: Lateral - Integration with creativity boost
    #[tokio::test]
    async fn test_lateral_creativity_integration() {
        let graph = create_minimal_test_graph().await;
        
        let lateral = LateralThinking::new_with_params(
            graph.clone(),
            0.2,
            3,
            2.0 // High creativity boost
        );
        
        let result = lateral.find_creative_connections(
            "artificial_intelligence",
            "deep_learning",
            Some("Find unexpected connections")
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.creativity_score > 0.5);
    }

    // ===== ABSTRACT THINKING TESTS (4 total) =====

    // Test 22: Abstract - Basic pattern detection
    #[tokio::test]
    async fn test_abstract_basic_functionality() {
        let graph = create_minimal_test_graph().await;
        let abstract_thinking = AbstractThinking::new(graph.clone());
        
        let result = abstract_thinking.execute_pattern_analysis(
            "Find patterns in AI concepts",
            None
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.patterns.is_empty());
        assert!(result.abstraction_level > 0);
    }

    // Test 23: Abstract - Edge case with no patterns
    #[tokio::test]
    async fn test_abstract_edge_case_no_patterns() {
        let graph = create_minimal_test_graph().await;
        
        // Create random unrelated entities
        for i in 0..5 {
            graph.create_brain_entity(
                format!("random_{}", i),
                EntityDirection::Output
            ).await.unwrap();
        }
        
        let abstract_thinking = AbstractThinking::new(graph.clone());
        let result = abstract_thinking.execute_pattern_analysis(
            "Find patterns in random concepts",
            None
        ).await;
        
        assert!(result.is_ok());
        // Should handle gracefully when no patterns exist
    }

    // Test 24: Abstract - Performance with complex patterns
    #[tokio::test]
    async fn test_abstract_performance() {
        let graph = create_minimal_test_graph().await;
        
        // Create structured patterns
        for i in 0..3 {
            for j in 0..3 {
                let entity = graph.create_brain_entity(
                    format!("pattern_{}_{}", i, j),
                    EntityDirection::Gate
                ).await.unwrap();
                
                if j > 0 {
                    let prev = graph.get_entity_by_name(&format!("pattern_{}_{}", i, j-1)).await.unwrap();
                    graph.create_brain_relationship(prev, entity, RelationType::Sequence, 0.9).await.unwrap();
                }
            }
        }
        
        let abstract_thinking = AbstractThinking::new(graph.clone());
        let start = Instant::now();
        
        let result = abstract_thinking.execute_pattern_analysis(
            "Analyze all patterns",
            None
        ).await;
        
        assert!(result.is_ok());
        assert!(start.elapsed().as_millis() < 400);
    }

    // Test 25: Abstract - Integration with meta-analysis
    #[tokio::test]
    async fn test_abstract_meta_integration() {
        let graph = create_minimal_test_graph().await;
        let abstract_thinking = AbstractThinking::new(graph.clone());
        
        let result = abstract_thinking.execute_pattern_analysis(
            "Meta-analysis of AI knowledge structure",
            Some("Look for higher-order patterns")
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.meta_patterns.len() > 0);
    }

    // ===== ADAPTIVE THINKING TESTS (4 total) =====

    // Test 26: Adaptive - Basic strategy selection
    #[tokio::test]
    async fn test_adaptive_basic_functionality() {
        let graph = create_minimal_test_graph().await;
        let adaptive = AdaptiveThinking::new(graph.clone());
        
        let result = adaptive.execute_adaptive_reasoning(
            "What is machine learning?",
            None
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.strategy_used.selected_patterns.is_empty());
        assert!(result.confidence_distribution.mean_confidence > 0.0);
    }

    // Test 27: Adaptive - Edge case with ambiguous query
    #[tokio::test]
    async fn test_adaptive_edge_case_ambiguous() {
        let graph = create_minimal_test_graph().await;
        let adaptive = AdaptiveThinking::new(graph.clone());
        
        let result = adaptive.execute_adaptive_reasoning(
            "Tell me about it",
            None
        ).await;
        
        assert!(result.is_ok());
        // Should handle ambiguous queries with low confidence
        let result = result.unwrap();
        assert!(result.confidence_distribution.mean_confidence < 0.5);
    }

    // Test 28: Adaptive - Performance with pattern switching
    #[tokio::test]
    async fn test_adaptive_performance() {
        let graph = create_minimal_test_graph().await;
        let adaptive = AdaptiveThinking::new(graph.clone());
        
        let queries = vec![
            "What is AI?",                    // Convergent
            "Explore AI applications",         // Divergent
            "Compare AI and ML",              // Critical
            "Show AI hierarchy",              // Systems
            "Connect AI to cooking",          // Lateral
            "Find patterns in AI",            // Abstract
        ];
        
        let start = Instant::now();
        for query in queries {
            let result = adaptive.execute_adaptive_reasoning(query, None).await;
            assert!(result.is_ok());
        }
        
        let total_time = start.elapsed();
        assert!(total_time.as_millis() < 1000, "Pattern switching took too long");
    }

    // Test 29: Adaptive - Integration with ensemble
    #[tokio::test]
    async fn test_adaptive_ensemble_integration() {
        let graph = create_minimal_test_graph().await;
        let adaptive = AdaptiveThinking::new(graph.clone());
        
        let result = adaptive.execute_adaptive_reasoning(
            "Comprehensive analysis of deep learning",
            Some("Use multiple perspectives")
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        // Should use multiple patterns for comprehensive analysis
        assert!(result.strategy_used.selected_patterns.len() > 1);
        assert!(result.confidence_distribution.mean_confidence > 0.0);
    }
}