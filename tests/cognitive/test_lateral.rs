#[cfg(test)]
mod lateral_tests {
    use tokio;
    use llmkg::cognitive::lateral::LateralThinking;
    use llmkg::cognitive::types::{
        PatternResult, LateralResult, BridgePath, CognitivePatternType
    };
    use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;


    #[tokio::test]
    async fn test_parse_lateral_query_basic() {
        let thinking = create_test_lateral_thinking().await;
        
        // Test "between" connector
        let result = thinking.parse_lateral_query("connections between art and technology").await;
        assert!(result.is_ok());
        let (concept1, concept2) = result.unwrap();
        assert_eq!(concept1, "art");
        assert_eq!(concept2, "technology");
        
        // Test "and" connector
        let result = thinking.parse_lateral_query("art and technology").await;
        assert!(result.is_ok());
        let (concept1, concept2) = result.unwrap();
        assert_eq!(concept1, "art");
        assert_eq!(concept2, "technology");
        
        // Test "to" connector
        let result = thinking.parse_lateral_query("art to technology").await;
        assert!(result.is_ok());
        let (concept1, concept2) = result.unwrap();
        assert_eq!(concept1, "art");
        assert_eq!(concept2, "technology");
        
        // Test "with" connector
        let result = thinking.parse_lateral_query("art with technology").await;
        assert!(result.is_ok());
        let (concept1, concept2) = result.unwrap();
        assert_eq!(concept1, "art");
        assert_eq!(concept2, "technology");
    }

    #[tokio::test]
    async fn test_parse_lateral_query_edge_cases() {
        let thinking = create_test_lateral_thinking().await;
        
        // Test with fewer than two concepts
        let result = thinking.parse_lateral_query("just one concept").await;
        assert!(result.is_err(), "Should error with insufficient concepts");
        
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
        
        // Execute search for creative connections
        let result = thinking.find_creative_connections("concept_a", "concept_c", 3).await;
        assert!(result.is_ok());
        
        let bridges = result.unwrap();
        assert!(bridges.len() >= 2, "Should find both direct and indirect paths: {:?}", bridges);
        
        // Should find direct path A -> C
        let has_direct = bridges.iter().any(|b| 
            b.bridge_concepts.is_empty() && 
            b.start_concept == "concept_a" && 
            b.end_concept == "concept_c"
        );
        
        // Should find indirect path A -> B -> C
        let has_indirect = bridges.iter().any(|b| 
            b.bridge_concepts.contains(&"concept_b".to_string()) &&
            b.start_concept == "concept_a" && 
            b.end_concept == "concept_c"
        );
        
        assert!(has_direct, "Should find direct path");
        assert!(has_indirect, "Should find indirect path through bridge");
        
        // Verify creativity scoring - indirect should have higher novelty
        let indirect_path = bridges.iter().find(|b| 
            b.bridge_concepts.contains(&"concept_b".to_string())
        ).unwrap();
        
        let direct_path = bridges.iter().find(|b| 
            b.bridge_concepts.is_empty()
        ).unwrap();
        
        assert!(indirect_path.novelty_score > direct_path.novelty_score, 
               "Indirect path should have higher novelty: {} vs {}", 
               indirect_path.novelty_score, direct_path.novelty_score);
    }

    #[tokio::test]
    async fn test_max_bridge_length_constraint() {
        let graph = create_long_path_graph().await;
        let thinking = LateralThinking::new(graph, 0.2, 2); // max_bridge_length=2
        
        // Test with length constraint
        let result = thinking.find_creative_connections("start", "end", 2).await;
        assert!(result.is_ok());
        
        let bridges = result.unwrap();
        // Should not find the long path A -> B -> C -> D with max_length=2
        assert!(bridges.is_empty() || bridges.iter().all(|b| b.bridge_concepts.len() <= 2),
               "Should respect max bridge length constraint");
        
        // Test with increased length limit
        let thinking_long = LateralThinking::new(
            create_long_path_graph().await, 0.2, 4
        ); // max_bridge_length=4
        
        let result_long = thinking_long.find_creative_connections("start", "end", 4).await;
        assert!(result_long.is_ok());
        
        let bridges_long = result_long.unwrap();
        // Should now find the longer path
        assert!(bridges_long.len() > 0, "Should find paths with increased length limit");
        
        let has_long_path = bridges_long.iter().any(|b| b.bridge_concepts.len() >= 2);
        assert!(has_long_path, "Should find longer bridge paths");
    }

    #[tokio::test]
    async fn test_analyze_novelty() {
        let thinking = create_test_lateral_thinking().await;
        
        // Create test bridge paths with different characteristics
        let conventional_bridge = BridgePath {
            bridge_id: "conventional".to_string(),
            start_concept: "art".to_string(),
            end_concept: "design".to_string(),
            bridge_concepts: vec!["visual".to_string()], // obvious connection
            novelty_score: 0.2,
            plausibility_score: 0.9,
            creativity_score: 0.0, // Will be calculated
            explanation: "Art connects to design through visual elements".to_string(),
            connection_strength: 0.8,
        };
        
        let novel_bridge = BridgePath {
            bridge_id: "novel".to_string(),
            start_concept: "music".to_string(),
            end_concept: "mathematics".to_string(),
            bridge_concepts: vec!["harmony".to_string(), "ratios".to_string()], // creative connection
            novelty_score: 0.8,
            plausibility_score: 0.6,
            creativity_score: 0.0, // Will be calculated
            explanation: "Music connects to mathematics through harmonic ratios".to_string(),
            connection_strength: 0.5,
        };
        
        let bridges = vec![conventional_bridge, novel_bridge];
        let analyzed = thinking.analyze_novelty(bridges).await;
        assert!(analyzed.is_ok());
        
        let analyzed_bridges = analyzed.unwrap();
        assert_eq!(analyzed_bridges.len(), 2);
        
        // Novel bridge should score higher on creativity
        let novel = analyzed_bridges.iter().find(|b| b.bridge_id == "novel").unwrap();
        let conventional = analyzed_bridges.iter().find(|b| b.bridge_id == "conventional").unwrap();
        
        assert!(novel.creativity_score > conventional.creativity_score,
               "Novel bridge should have higher creativity score: {} vs {}", 
               novel.creativity_score, conventional.creativity_score);
    }

    #[tokio::test]
    async fn test_end_to_end_lateral_thinking() {
        let graph = create_creative_test_graph().await;
        let thinking = LateralThinking::new(graph, 0.3, 3);
        
        // Execute full lateral thinking process
        let result = thinking.execute("connections between science and art").await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        match pattern_result {
            PatternResult::Lateral(lateral_result) => {
                assert_eq!(lateral_result.pattern_type, CognitivePatternType::Lateral);
                assert_eq!(lateral_result.start_concept, "science");
                assert_eq!(lateral_result.end_concept, "art");
                
                assert!(lateral_result.bridge_paths.len() > 0, "Should find bridge paths");
                
                // Check for meaningful connections
                let has_creative_bridge = lateral_result.bridge_paths.iter()
                    .any(|b| b.novelty_score > 0.4);
                assert!(has_creative_bridge, "Should find creative connections");
                
                // Verify paths are ranked by creativity
                if lateral_result.bridge_paths.len() > 1 {
                    let first = &lateral_result.bridge_paths[0];
                    let second = &lateral_result.bridge_paths[1];
                    assert!(first.creativity_score >= second.creativity_score,
                           "Paths should be ranked by creativity");
                }
            },
            _ => panic!("Expected LateralResult")
        }
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

    #[tokio::test]
    async fn test_bridge_explanation_generation() {
        let thinking = create_test_lateral_thinking().await;
        
        // Test explanation generation for different bridge types
        let simple_bridge = BridgePath {
            bridge_id: "simple".to_string(),
            start_concept: "dog".to_string(),
            end_concept: "loyalty".to_string(),
            bridge_concepts: vec!["companion".to_string()],
            novelty_score: 0.4,
            plausibility_score: 0.8,
            creativity_score: 0.6,
            explanation: "".to_string(), // Will be generated
            connection_strength: 0.7,
        };
        
        let explanation = thinking.generate_bridge_explanation(&simple_bridge).await;
        assert!(explanation.is_ok());
        
        let generated_explanation = explanation.unwrap();
        assert!(!generated_explanation.is_empty(), "Should generate explanation");
        assert!(generated_explanation.contains("dog") && generated_explanation.contains("loyalty"),
               "Explanation should mention both concepts");
        assert!(generated_explanation.contains("companion"),
               "Explanation should mention bridge concept");
    }

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
        LateralThinking::new(graph, 0.3, 3)
    }

    async fn create_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("art", "Artistic concept").await.unwrap();
        graph.add_entity("technology", "Technology concept").await.unwrap();
        graph.add_entity("creativity", "Bridge concept").await.unwrap();
        
        graph.add_relationship("art", "creativity", "involves", 0.8).await.unwrap();
        graph.add_relationship("creativity", "technology", "drives", 0.7).await.unwrap();
        
        graph
    }

    async fn create_indirect_path_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("concept_a", "Start concept").await.unwrap();
        graph.add_entity("concept_b", "Bridge concept").await.unwrap();
        graph.add_entity("concept_c", "End concept").await.unwrap();
        
        // Direct path (strong but obvious)
        graph.add_relationship("concept_a", "concept_c", "direct_link", 0.9).await.unwrap();
        
        // Indirect path (weaker but more creative)
        graph.add_relationship("concept_a", "concept_b", "connects_to", 0.6).await.unwrap();
        graph.add_relationship("concept_b", "concept_c", "leads_to", 0.7).await.unwrap();
        
        graph
    }

    async fn create_long_path_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create a long chain: start -> intermediate1 -> intermediate2 -> end
        graph.add_entity("start", "Starting point").await.unwrap();
        graph.add_entity("intermediate1", "First bridge").await.unwrap();
        graph.add_entity("intermediate2", "Second bridge").await.unwrap();
        graph.add_entity("end", "End point").await.unwrap();
        
        graph.add_relationship("start", "intermediate1", "step1", 0.7).await.unwrap();
        graph.add_relationship("intermediate1", "intermediate2", "step2", 0.6).await.unwrap();
        graph.add_relationship("intermediate2", "end", "step3", 0.8).await.unwrap();
        
        graph
    }

    async fn create_creative_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Science and art with creative bridges
        graph.add_entity("science", "Scientific domain").await.unwrap();
        graph.add_entity("art", "Artistic domain").await.unwrap();
        graph.add_entity("observation", "Bridge concept 1").await.unwrap();
        graph.add_entity("pattern", "Bridge concept 2").await.unwrap();
        graph.add_entity("beauty", "Bridge concept 3").await.unwrap();
        
        // Creative connections
        graph.add_relationship("science", "observation", "uses", 0.8).await.unwrap();
        graph.add_relationship("observation", "pattern", "reveals", 0.7).await.unwrap();
        graph.add_relationship("pattern", "beauty", "creates", 0.6).await.unwrap();
        graph.add_relationship("beauty", "art", "inspires", 0.8).await.unwrap();
        
        // Alternative path
        graph.add_relationship("science", "pattern", "discovers", 0.7).await.unwrap();
        graph.add_relationship("art", "pattern", "expresses", 0.6).await.unwrap();
        
        graph
    }

    async fn create_mixed_novelty_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("start", "Start").await.unwrap();
        graph.add_entity("target", "Target").await.unwrap();
        graph.add_entity("obvious_bridge", "Obvious connection").await.unwrap();
        graph.add_entity("novel_bridge", "Novel connection").await.unwrap();
        
        // Obvious path (low novelty)
        graph.add_relationship("start", "obvious_bridge", "common", 0.9).await.unwrap();
        graph.add_relationship("obvious_bridge", "target", "expected", 0.8).await.unwrap();
        
        // Novel path (high novelty)
        graph.add_relationship("start", "novel_bridge", "surprising", 0.4).await.unwrap();
        graph.add_relationship("novel_bridge", "target", "creative", 0.5).await.unwrap();
        
        graph
    }
}