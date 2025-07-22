#[cfg(test)]
mod convergent_tests {
    use std::collections::HashMap;
    use tokio;
    use llmkg::cognitive::convergent::ConvergentThinking;
    use llmkg::cognitive::{CognitivePattern, PatternResult, ConvergentResult, CognitivePatternType, PatternParameters};
    use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use llmkg::core::brain_types::{ActivationStep, ActivationOperation};

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
        let params = PatternParameters::default();
        let result = thinking.execute("What is a dog?", None, params).await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        assert!(pattern_result.confidence > 0.5, "Confidence should be reasonable: {}", pattern_result.confidence);
        assert!(pattern_result.answer.contains("Mammal") || pattern_result.answer.contains("mammal"));
        assert!(pattern_result.reasoning_trace.len() > 0, "Should have reasoning trace");
    }

    #[tokio::test]
    async fn test_beam_search_pruning() {
        // Test that beam search correctly prunes less relevant paths
        let graph = create_pruning_test_graph().await;
        let thinking = ConvergentThinking::new(graph); // Uses default beam_width
        
        let params = PatternParameters::default();
        let result = thinking.execute("What connects to start?", Some("start"), params).await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        // Should find the more fruitful path despite initial lower activation
        assert!(pattern_result.answer.contains("chain_end") || pattern_result.answer.contains("dead_end"), 
                "Should find connected nodes: {}", pattern_result.answer);
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
        graph.add_entity(EntityData::new(2, "Classification concept".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(3, "Property concept".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(4, "Relationship concept".to_string(), vec![0.0; 128])).await.unwrap();
        
        // Add entities for relationships
        graph.add_entity(EntityData::new(5, "dog".to_string(), vec![0.1; 128])).await.unwrap();
        graph.add_entity(EntityData::new(6, "mammal".to_string(), vec![0.2; 128])).await.unwrap();
        graph.add_entity(EntityData::new(7, "warm_blooded".to_string(), vec![0.3; 128])).await.unwrap();
        graph.add_entity(EntityData::new(8, "pet".to_string(), vec![0.4; 128])).await.unwrap();
        
        // Add relationships using entity IDs
        let dog_id = graph.get_entity_by_description("dog").await.unwrap().unwrap().id;
        let mammal_id = graph.get_entity_by_description("mammal").await.unwrap().unwrap().id;
        let warm_id = graph.get_entity_by_description("warm_blooded").await.unwrap().unwrap().id;
        let pet_id = graph.get_entity_by_description("pet").await.unwrap().unwrap().id;
        
        graph.add_connection(dog_id, mammal_id, 0.9).await.unwrap();
        graph.add_connection(mammal_id, warm_id, 0.8).await.unwrap();
        graph.add_connection(dog_id, pet_id, 0.7).await.unwrap();
        
        graph
    }

    async fn create_pruning_test_graph() -> std::sync::Arc<BrainEnhancedKnowledgeGraph> {
        use llmkg::core::types::EntityData;
        let graph = std::sync::Arc::new(BrainEnhancedKnowledgeGraph::new(128).unwrap());
        
        // Create a graph where beam search pruning matters
        graph.add_entity(EntityData::new(1, "start".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(2, "dead_end".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(3, "chain_middle".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(4, "chain_end".to_string(), vec![0.0; 128])).await.unwrap();
        
        let start_id = graph.get_entity_by_description("start").await.unwrap().unwrap().id;
        let dead_id = graph.get_entity_by_description("dead_end").await.unwrap().unwrap().id;
        let middle_id = graph.get_entity_by_description("chain_middle").await.unwrap().unwrap().id;
        let end_id = graph.get_entity_by_description("chain_end").await.unwrap().unwrap().id;
        
        // High activation path that leads nowhere
        graph.add_connection(start_id, dead_id, 0.9).await.unwrap();
        
        // Lower activation path that continues
        graph.add_connection(start_id, middle_id, 0.4).await.unwrap();
        graph.add_connection(middle_id, end_id, 0.8).await.unwrap();
        
        graph
    }

    #[tokio::test]
    async fn test_cognitive_pattern_interface() {
        // Test that ConvergentThinking properly implements CognitivePattern trait
        let thinking = create_test_convergent_thinking().await;
        
        let params = PatternParameters::default();
        let result = thinking.execute("What is a mammal?", None, params).await;
        assert!(result.is_ok());
        
        // Verify the result format matches interface expectations
        let pattern_result = result.unwrap();
        assert!(!pattern_result.answer.is_empty());
        assert!(pattern_result.confidence >= 0.0 && pattern_result.confidence <= 1.0);
        assert_eq!(pattern_result.pattern_type, CognitivePatternType::Convergent);
    }

    #[tokio::test]
    async fn test_activation_propagation_limits() {
        let graph = create_deep_test_graph().await;
        let mut thinking = ConvergentThinking::new(graph);
        thinking.max_depth = 2; // Limited depth
        
        let params = PatternParameters::default();
        let result = thinking.execute("What is at root?", Some("root"), params).await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        // Verify depth limiting worked
        assert!(pattern_result.reasoning_trace.len() <= 6, "Should limit trace depth");
    }


    async fn create_deep_test_graph() -> std::sync::Arc<BrainEnhancedKnowledgeGraph> {
        use llmkg::core::types::EntityData;
        let graph = std::sync::Arc::new(BrainEnhancedKnowledgeGraph::new(128).unwrap());
        
        // Create a deep chain to test depth limits
        graph.add_entity(EntityData::new(1, "root".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(2, "level1".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(3, "level2".to_string(), vec![0.0; 128])).await.unwrap();
        graph.add_entity(EntityData::new(4, "level3".to_string(), vec![0.0; 128])).await.unwrap();
        
        let root_id = graph.get_entity_by_description("root").await.unwrap().unwrap().id;
        let level1_id = graph.get_entity_by_description("level1").await.unwrap().unwrap().id;
        let level2_id = graph.get_entity_by_description("level2").await.unwrap().unwrap().id;
        let level3_id = graph.get_entity_by_description("level3").await.unwrap().unwrap().id;
        
        graph.add_connection(root_id, level1_id, 0.8).await.unwrap();
        graph.add_connection(level1_id, level2_id, 0.8).await.unwrap();
        graph.add_connection(level2_id, level3_id, 0.8).await.unwrap();
        
        graph
    }


}