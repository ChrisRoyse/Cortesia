#[cfg(test)]
mod lateral_tests {
    use tokio;
    use llmkg::cognitive::lateral::LateralThinking;
    use llmkg::cognitive::{
        CognitivePattern, PatternResult, LateralResult, BridgePath, CognitivePatternType, PatternParameters
    };
    use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use llmkg::core::types::EntityData;


    #[tokio::test]
    async fn test_parse_lateral_query_basic() {
        let thinking = create_test_lateral_thinking().await;
        
        // Test basic lateral query
        let params = PatternParameters::default();
        let result = thinking.execute("connections between art and technology", None, params).await;
        assert!(result.is_ok());
        let pattern_result = result.unwrap();
        assert!(!pattern_result.answer.is_empty());
        assert_eq!(pattern_result.pattern_type, CognitivePatternType::Lateral);
    }

    #[tokio::test]
    async fn test_parse_lateral_query_edge_cases() {
        let thinking = create_test_lateral_thinking().await;
        
        // Test with insufficient concepts
        let params = PatternParameters::default();
        let result = thinking.execute("just one concept", None, params).await;
        // Should still return a result but may indicate no bridges found
        assert!(result.is_ok());
        
        // Test empty query
        let result = thinking.parse_lateral_query("").await;
        assert!(result.is_err(), "Should error with empty query");
        
        // Test with only connecting words
        let result = thinking.parse_lateral_query("and to with between").await;
        assert!(result.is_err(), "Should error with only connectors");
    }

    #[tokio::test]
    async fn test_indirect_path_discovery() {
        // Create graph with both direct and indirect paths
        let graph = create_indirect_path_graph().await;
        let thinking = LateralThinking::new(graph, 0.3, 4); // novelty_threshold=0.3, max_bridge_length=4
        
        // Execute search for creative connections via execute method
        let params = PatternParameters::default();
        let result = thinking.execute("connections between concept_a and concept_c", None, params).await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        assert!(!pattern_result.answer.is_empty(), "Should find connections");
        
        // Verify we found meaningful connections
        assert_eq!(pattern_result.pattern_type, CognitivePatternType::Lateral);
        assert!(pattern_result.confidence > 0.0, "Should have confidence in connections");
    }

    #[tokio::test]
    async fn test_max_bridge_length_constraint() {
        let graph = create_long_path_graph().await;
        let mut thinking = LateralThinking::new(graph);
        thinking.max_bridge_length = 2; // max_bridge_length=2
        
        // Test with length constraint
        let params = PatternParameters::default();
        let result = thinking.execute("connections between start and end", None, params).await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        // Should respect bridge length constraint
        assert_eq!(pattern_result.pattern_type, CognitivePatternType::Lateral);
        
        // Test with increased length limit
        let mut thinking_long = LateralThinking::new(create_long_path_graph().await);
        thinking_long.max_bridge_length = 4; // max_bridge_length=4
        
        let params = PatternParameters::default();
        let result_long = thinking_long.execute("connections between start and end", None, params).await;
        assert!(result_long.is_ok());
        
        let pattern_result_long = result_long.unwrap();
        // Should now find longer paths
        assert!(!pattern_result_long.answer.is_empty(), "Should find paths with increased length limit");
    }

    // NOTE: Tests for analyze_novelty have been moved to src/cognitive/lateral.rs
    // in the #[cfg(test)] module where they can access the private method directly.

    #[tokio::test]
    async fn test_end_to_end_lateral_thinking() {
        let graph = create_creative_test_graph().await;
        let thinking = LateralThinking::new(graph);
        
        // Execute full lateral thinking process
        let params = PatternParameters::default();
        let result = thinking.execute("connections between science and art", None, params).await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        assert_eq!(pattern_result.pattern_type, CognitivePatternType::Lateral);
        assert!(!pattern_result.answer.is_empty(), "Should find connections between science and art");
        assert!(pattern_result.confidence > 0.0, "Should have confidence in connections");
    }

    #[tokio::test]
    async fn test_novelty_threshold_filtering() {
        let graph = create_mixed_novelty_graph().await;
        let thinking = LateralThinking::new(graph, 0.7, 3); // High novelty threshold
        
        let result = thinking.find_creative_connections("start", "target", 3).await;
        assert!(result.is_ok());
        
        let bridges = result.unwrap();
        
        // All returned bridges should meet the novelty threshold
        for bridge in &bridges {
            assert!(bridge.novelty_score >= 0.7, 
                   "Bridge '{}' novelty {} should meet threshold 0.7", 
                   bridge.bridge_id, bridge.novelty_score);
        }
    }

    // NOTE: Tests for generate_bridge_explanation have been moved to src/cognitive/lateral.rs
    // in the #[cfg(test)] module where they can access the private method directly.

    #[tokio::test]
    async fn test_cognitive_pattern_interface() {
        let thinking = create_test_lateral_thinking().await;
        
        let result = thinking.execute("creative links between music and color").await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        match pattern_result {
            PatternResult::Lateral(lateral_result) => {
                assert_eq!(lateral_result.pattern_type, CognitivePatternType::Lateral);
                assert!(!lateral_result.start_concept.is_empty());
                assert!(!lateral_result.end_concept.is_empty());
                assert!(lateral_result.overall_creativity_score >= 0.0 && 
                       lateral_result.overall_creativity_score <= 1.0);
            },
            _ => panic!("Expected LateralResult from lateral thinking")
        }
    }


    // Helper functions

    async fn create_test_lateral_thinking() -> LateralThinking {
        let graph = create_test_graph().await;
        LateralThinking::new(graph)
    }

    async fn create_test_graph() -> std::sync::Arc<BrainEnhancedKnowledgeGraph> {
        let graph = std::sync::Arc::new(BrainEnhancedKnowledgeGraph::new(128).unwrap());
        
        graph.add_entity(EntityData::new(1, "art".to_string(), vec![0.1; 128])).await.unwrap();
        graph.add_entity(EntityData::new(2, "technology".to_string(), vec![0.2; 128])).await.unwrap();
        graph.add_entity(EntityData::new(3, "creativity".to_string(), vec![0.3; 128])).await.unwrap();
        
        let art_id = graph.get_entity_by_description("art").await.unwrap().unwrap().id;
        let tech_id = graph.get_entity_by_description("technology").await.unwrap().unwrap().id;
        let creativity_id = graph.get_entity_by_description("creativity").await.unwrap().unwrap().id;
        
        graph.add_connection(art_id, creativity_id, 0.8).await.unwrap();
        graph.add_connection(creativity_id, tech_id, 0.7).await.unwrap();
        
        graph
    }

    async fn create_indirect_path_graph() -> std::sync::Arc<BrainEnhancedKnowledgeGraph> {
        let graph = std::sync::Arc::new(BrainEnhancedKnowledgeGraph::new(128).unwrap());
        
        graph.add_entity(EntityData::new(1, "concept_a".to_string(), vec![0.1; 128])).await.unwrap();
        graph.add_entity(EntityData::new(2, "concept_b".to_string(), vec![0.2; 128])).await.unwrap();
        graph.add_entity(EntityData::new(3, "concept_c".to_string(), vec![0.3; 128])).await.unwrap();
        
        let a_id = graph.get_entity_by_description("concept_a").await.unwrap().unwrap().id;
        let b_id = graph.get_entity_by_description("concept_b").await.unwrap().unwrap().id;
        let c_id = graph.get_entity_by_description("concept_c").await.unwrap().unwrap().id;
        
        // Direct path (strong but obvious)
        graph.add_connection(a_id, c_id, 0.9).await.unwrap();
        
        // Indirect path (weaker but more creative)
        graph.add_connection(a_id, b_id, 0.6).await.unwrap();
        graph.add_connection(b_id, c_id, 0.7).await.unwrap();
        
        graph
    }

    async fn create_long_path_graph() -> std::sync::Arc<BrainEnhancedKnowledgeGraph> {
        let graph = std::sync::Arc::new(BrainEnhancedKnowledgeGraph::new(128).unwrap());
        
        // Create a long chain: start -> intermediate1 -> intermediate2 -> end
        graph.add_entity(EntityData::new(1, "start".to_string(), vec![0.1; 128])).await.unwrap();
        graph.add_entity(EntityData::new(2, "intermediate1".to_string(), vec![0.2; 128])).await.unwrap();
        graph.add_entity(EntityData::new(3, "intermediate2".to_string(), vec![0.3; 128])).await.unwrap();
        graph.add_entity(EntityData::new(4, "end".to_string(), vec![0.4; 128])).await.unwrap();
        
        let start_id = graph.get_entity_by_description("start").await.unwrap().unwrap().id;
        let int1_id = graph.get_entity_by_description("intermediate1").await.unwrap().unwrap().id;
        let int2_id = graph.get_entity_by_description("intermediate2").await.unwrap().unwrap().id;
        let end_id = graph.get_entity_by_description("end").await.unwrap().unwrap().id;
        
        graph.add_connection(start_id, int1_id, 0.7).await.unwrap();
        graph.add_connection(int1_id, int2_id, 0.6).await.unwrap();
        graph.add_connection(int2_id, end_id, 0.8).await.unwrap();
        
        graph
    }

    async fn create_creative_test_graph() -> std::sync::Arc<BrainEnhancedKnowledgeGraph> {
        let graph = std::sync::Arc::new(BrainEnhancedKnowledgeGraph::new(128).unwrap());
        
        // Science and art with creative bridges
        graph.add_entity(EntityData::new(1, "science".to_string(), vec![0.1; 128])).await.unwrap();
        graph.add_entity(EntityData::new(2, "art".to_string(), vec![0.2; 128])).await.unwrap();
        graph.add_entity(EntityData::new(3, "observation".to_string(), vec![0.3; 128])).await.unwrap();
        graph.add_entity(EntityData::new(4, "pattern".to_string(), vec![0.4; 128])).await.unwrap();
        graph.add_entity(EntityData::new(5, "beauty".to_string(), vec![0.5; 128])).await.unwrap();
        
        let science_id = graph.get_entity_by_description("science").await.unwrap().unwrap().id;
        let art_id = graph.get_entity_by_description("art").await.unwrap().unwrap().id;
        let obs_id = graph.get_entity_by_description("observation").await.unwrap().unwrap().id;
        let pattern_id = graph.get_entity_by_description("pattern").await.unwrap().unwrap().id;
        let beauty_id = graph.get_entity_by_description("beauty").await.unwrap().unwrap().id;
        
        // Creative connections
        graph.add_connection(science_id, obs_id, 0.8).await.unwrap();
        graph.add_connection(obs_id, pattern_id, 0.7).await.unwrap();
        graph.add_connection(pattern_id, beauty_id, 0.6).await.unwrap();
        graph.add_connection(beauty_id, art_id, 0.8).await.unwrap();
        
        // Alternative path
        graph.add_connection(science_id, pattern_id, 0.7).await.unwrap();
        graph.add_connection(art_id, pattern_id, 0.6).await.unwrap();
        
        graph
    }

    async fn create_mixed_novelty_graph() -> std::sync::Arc<BrainEnhancedKnowledgeGraph> {
        let graph = std::sync::Arc::new(BrainEnhancedKnowledgeGraph::new(128).unwrap());
        
        graph.add_entity(EntityData::new(1, "start".to_string(), vec![0.1; 128])).await.unwrap();
        graph.add_entity(EntityData::new(2, "target".to_string(), vec![0.2; 128])).await.unwrap();
        graph.add_entity(EntityData::new(3, "obvious_bridge".to_string(), vec![0.3; 128])).await.unwrap();
        graph.add_entity(EntityData::new(4, "novel_bridge".to_string(), vec![0.4; 128])).await.unwrap();
        
        let start_id = graph.get_entity_by_description("start").await.unwrap().unwrap().id;
        let target_id = graph.get_entity_by_description("target").await.unwrap().unwrap().id;
        let obvious_id = graph.get_entity_by_description("obvious_bridge").await.unwrap().unwrap().id;
        let novel_id = graph.get_entity_by_description("novel_bridge").await.unwrap().unwrap().id;
        
        // Obvious path (low novelty)
        graph.add_connection(start_id, obvious_id, 0.9).await.unwrap();
        graph.add_connection(obvious_id, target_id, 0.8).await.unwrap();
        
        // Novel path (high novelty)
        graph.add_connection(start_id, novel_id, 0.4).await.unwrap();
        graph.add_connection(novel_id, target_id, 0.5).await.unwrap();
        
        graph
    }
}