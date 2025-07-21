#[cfg(test)]
mod convergent_tests {
    use std::collections::HashMap;
    use tokio;
    use llmkg::cognitive::convergent::ConvergentThinking;
    use llmkg::cognitive::{PatternResult, ConvergentResult, CognitivePatternType};
    use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;

    // NOTE: Tests for calculate_concept_relevance have been moved to src/cognitive/convergent.rs
    // in the #[cfg(test)] module where they can access the private method directly.

    // NOTE: Tests for extract_target_concept have been moved to src/cognitive/convergent.rs
    // in the #[cfg(test)] module where they can access the private method directly.

    #[tokio::test]
    async fn test_end_to_end_query_and_answer() {
        // Create a test graph with known structure
        let graph = create_test_graph().await;
        let thinking = ConvergentThinking::new(graph); // Uses default max_depth and beam_width
        
        // Execute query about dog properties
        let result = thinking.execute("What is a dog?").await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        match pattern_result {
            PatternResult::Convergent(conv_result) => {
                assert!(conv_result.confidence > 0.5, "Confidence should be reasonable: {}", conv_result.confidence);
                assert!(conv_result.answer.contains("Mammal") || conv_result.answer.contains("mammal"));
                assert!(conv_result.reasoning_path.len() > 0, "Should have reasoning path");
            },
            _ => panic!("Expected ConvergentResult")
        }
    }

    #[tokio::test]
    async fn test_beam_search_pruning() {
        // Test that beam search correctly prunes less relevant paths
        let graph = create_pruning_test_graph().await;
        let thinking = ConvergentThinking::new(graph); // Uses default beam_width
        
        let result = thinking.execute_convergent_query("What connects to start?", Some("start")).await;
        assert!(result.is_ok());
        
        let conv_result = result.unwrap();
        // Should find the more fruitful path despite initial lower activation
        assert!(conv_result.answer.contains("chain_end"), "Should follow longer chain: {}", conv_result.answer);
    }

    // NOTE: Tests for focused_propagation have been moved to src/cognitive/convergent.rs
    // in the #[cfg(test)] module where they can access the private method directly.

    // Helper functions for creating test data

    async fn create_test_convergent_thinking() -> ConvergentThinking {
        let graph = create_test_graph().await;
        ConvergentThinking::new(graph)
    }

    async fn create_test_graph() -> std::sync::Arc<BrainEnhancedKnowledgeGraph> {
        use llmkg::core::types::EntityData;
        let graph = std::sync::Arc::new(BrainEnhancedKnowledgeGraph::new(128).unwrap());
        
        // Create basic animal hierarchy
        graph.add_entity(EntityData::new(1, "Animal concept".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(1, "Classification concept".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(1, "Property concept".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(1, "Relationship concept".to_string(), vec![0.0; 128])).await.unwrap();
        
        // Add relationships
        graph.add_relationship("dog", "mammal", "is_a", 0.9).await.unwrap();
        graph.add_relationship("mammal", "warm_blooded", "has_property", 0.8).await.unwrap();
        graph.add_relationship("dog", "pet", "can_be", 0.7).await.unwrap();
        
        graph
    }

    async fn create_pruning_test_graph() -> std::sync::Arc<BrainEnhancedKnowledgeGraph> {
        use llmkg::core::types::EntityData;
        let graph = std::sync::Arc::new(BrainEnhancedKnowledgeGraph::new(128).unwrap());
        
        // Create a graph where beam search pruning matters
        graph.add_entity(EntityData::new(1, "Starting point".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(1, "High activation but no connections".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(1, "Lower initial but leads somewhere".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(1, "Final destination with valuable info".to_string(), vec![0.0; 128])).await.unwrap();
        
        // High activation path that leads nowhere
        graph.add_relationship("start", "dead_end", "strong_link", 0.9).await.unwrap();
        
        // Lower activation path that continues
        graph.add_relationship("start", "chain_middle", "weak_link", 0.4).await.unwrap();
        graph.add_relationship("chain_middle", "chain_end", "continues_to", 0.8).await.unwrap();
        
        graph
    }

    #[tokio::test]
    async fn test_cognitive_pattern_interface() {
        // Test that ConvergentThinking properly implements CognitivePattern trait
        let thinking = create_test_convergent_thinking().await;
        
        let result = thinking.execute("What is a mammal?").await;
        assert!(result.is_ok());
        
        // Verify the result format matches interface expectations
        let pattern_result = result.unwrap();
        match pattern_result {
            PatternResult::Convergent(conv_result) => {
                assert!(!conv_result.answer.is_empty());
                assert!(conv_result.confidence >= 0.0 && conv_result.confidence <= 1.0);
                assert_eq!(conv_result.pattern_type, CognitivePatternType::Convergent);
            },
            _ => panic!("Expected ConvergentResult from convergent thinking")
        }
    }

    #[tokio::test]
    async fn test_activation_propagation_limits() {
        let graph = create_deep_test_graph().await;
        let thinking = ConvergentThinking::new(graph, 2, 2); // Limited depth
        
        let result = thinking.focused_propagation("root", 2).await;
        assert!(result.is_ok());
        
        let activations = result.unwrap();
        
        // Should not propagate beyond max_depth
        assert!(activations.contains_key("level1"));
        assert!(activations.contains_key("level2"));
        assert!(!activations.contains_key("level3"), "Should not exceed max depth");
    }


    async fn create_deep_test_graph() -> std::sync::Arc<BrainEnhancedKnowledgeGraph> {
        use llmkg::core::types::EntityData;
        let graph = std::sync::Arc::new(BrainEnhancedKnowledgeGraph::new(128).unwrap());
        
        // Create a deep chain to test depth limits
        graph.add_entity(EntityData::new(1, "Starting point".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(1, "First level".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(1, "Second level".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(1, "Third level - should not be reached".to_string(), vec![0.0; 128])).await.unwrap();
        
        graph.add_relationship("root", "level1", "connects", 0.8).await.unwrap();
        graph.add_relationship("level1", "level2", "connects", 0.8).await.unwrap();
        graph.add_relationship("level2", "level3", "connects", 0.8).await.unwrap();
        
        graph
    }


}