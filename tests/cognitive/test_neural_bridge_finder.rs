#[cfg(test)]
mod neural_bridge_finder_tests {
    use tokio;
    use llmkg::cognitive::neural_bridge_finder::NeuralBridgeFinder;
    use llmkg::cognitive::types::BridgePath;
    use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;

    #[tokio::test]
    async fn test_calculate_path_novelty_diverse_concepts() {
        let finder = create_test_bridge_finder().await;
        
        // Create a long path with diverse concepts
        let diverse_path = vec![
            "art".to_string(),
            "symmetry".to_string(), 
            "mathematics".to_string(),
            "physics".to_string(),
            "universe".to_string(),
        ];
        
        let novelty_score = finder.calculate_path_novelty(&diverse_path).await;
        assert!(novelty_score > 0.7, "Diverse long path should have high novelty: {}", novelty_score);
        
        // Create a short path with similar concepts
        let similar_path = vec![
            "dog".to_string(),
            "canine".to_string(),
        ];
        
        let low_novelty = finder.calculate_path_novelty(&similar_path).await;
        assert!(low_novelty < 0.5, "Similar short path should have low novelty: {}", low_novelty);
        
        // Verify diverse path scores higher
        assert!(novelty_score > low_novelty, 
               "Diverse path should score higher than similar path: {} vs {}", 
               novelty_score, low_novelty);
    }

    #[tokio::test]
    async fn test_calculate_path_plausibility_connection_weights() {
        let graph = create_weighted_test_graph().await;
        let finder = NeuralBridgeFinder::new(graph, 3, 0.3);
        
        // Test path with high-weight connections
        let strong_path = vec![
            "node_a".to_string(),
            "node_b".to_string(),
            "node_c".to_string(),
        ];
        
        let high_plausibility = finder.calculate_path_plausibility(&strong_path).await;
        assert!(high_plausibility.is_ok());
        assert!(high_plausibility.unwrap() > 0.7, 
               "Strong connections should have high plausibility");
        
        // Test path with weak connections
        let weak_path = vec![
            "node_a".to_string(),
            "weak_node".to_string(),
            "node_c".to_string(),
        ];
        
        let low_plausibility = finder.calculate_path_plausibility(&weak_path).await;
        assert!(low_plausibility.is_ok());
        assert!(low_plausibility.unwrap() < 0.5, 
               "Weak connections should have low plausibility");
    }

    #[tokio::test]
    async fn test_calculate_path_plausibility_edge_cases() {
        let graph = create_test_graph().await;
        let finder = NeuralBridgeFinder::new(graph, 3, 0.3);
        
        // Test path with missing connection
        let broken_path = vec![
            "art".to_string(),
            "nonexistent_node".to_string(),
            "physics".to_string(),
        ];
        
        let broken_plausibility = finder.calculate_path_plausibility(&broken_path).await;
        assert!(broken_plausibility.is_ok());
        assert!(broken_plausibility.unwrap() < 0.3, 
               "Path with missing connections should have very low plausibility");
        
        // Test single node path
        let single_path = vec!["art".to_string()];
        let single_plausibility = finder.calculate_path_plausibility(&single_path).await;
        assert!(single_plausibility.is_ok());
        assert_eq!(single_plausibility.unwrap(), 1.0, 
                  "Single node path should have perfect plausibility");
    }

    #[tokio::test]
    async fn test_end_to_end_bridge_finding() {
        // Create graph with known creative bridge path
        let graph = create_art_physics_graph().await;
        let finder = NeuralBridgeFinder::new(graph, 4, 0.2);
        
        // Find bridges between Art and Physics
        let result = finder.find_creative_bridges("Art", "Physics").await;
        assert!(result.is_ok());
        
        let bridges = result.unwrap();
        assert!(bridges.len() > 0, "Should find bridge paths between Art and Physics");
        
        // Verify expected bridge path exists
        let expected_bridge = bridges.iter().find(|b| {
            b.bridge_concepts.contains(&"Symmetry".to_string()) &&
            b.start_concept == "Art" &&
            b.end_concept == "Physics"
        });
        
        assert!(expected_bridge.is_some(), 
               "Should find Art -> Symmetry -> Physics bridge: {:?}", 
               bridges.iter().map(|b| &b.bridge_concepts).collect::<Vec<_>>());
        
        let bridge = expected_bridge.unwrap();
        
        // Verify bridge properties
        assert!(bridge.novelty_score > 0.0 && bridge.novelty_score <= 1.0, 
               "Novelty score should be in valid range: {}", bridge.novelty_score);
        assert!(bridge.plausibility_score > 0.0 && bridge.plausibility_score <= 1.0, 
               "Plausibility score should be in valid range: {}", bridge.plausibility_score);
        assert!(!bridge.explanation.is_empty(), "Should have explanation");
        assert!(bridge.explanation.contains("Symmetry"), 
               "Explanation should mention bridge concept: {}", bridge.explanation);
    }

    #[tokio::test]
    async fn test_max_length_constraint() {
        let graph = create_long_chain_graph().await;
        
        // Test with restrictive length limit
        let finder_short = NeuralBridgeFinder::new(graph.clone(), 2, 0.2);
        let result_short = finder_short.find_creative_bridges_with_length("Start", "End", 2).await;
        assert!(result_short.is_ok());
        
        let bridges_short = result_short.unwrap();
        // Should not find the long path with length limit of 2
        assert!(bridges_short.is_empty(), 
               "Should not find bridges exceeding length limit");
        
        // Test with permissive length limit
        let finder_long = NeuralBridgeFinder::new(graph, 4, 0.2);
        let result_long = finder_long.find_creative_bridges_with_length("Start", "End", 4).await;
        assert!(result_long.is_ok());
        
        let bridges_long = result_long.unwrap();
        // Should now find the long path
        assert!(bridges_long.len() > 0, "Should find bridges with increased length limit");
        
        // Verify path length constraint is respected
        for bridge in &bridges_long {
            assert!(bridge.bridge_concepts.len() <= 4, 
                   "Bridge path should respect length constraint: {:?}", bridge.bridge_concepts);
        }
    }

    #[tokio::test]
    async fn test_neural_pathfinding_with_length() {
        let graph = create_pathfinding_test_graph().await;
        let finder = NeuralBridgeFinder::new(graph, 3, 0.2);
        
        // Test pathfinding between known entities
        let paths = finder.neural_pathfinding_with_length("source", "destination", 3).await;
        assert!(paths.is_ok());
        
        let found_paths = paths.unwrap();
        assert!(found_paths.len() > 0, "Should find paths between source and destination");
        
        // Verify all paths respect length limit
        for path in &found_paths {
            assert!(path.len() <= 4, "Path length should respect limit (including endpoints): {:?}", path);
        }
        
        // Verify paths are valid (connected)
        for path in &found_paths {
            assert!(path.len() >= 2, "Path should have at least source and destination");
            assert_eq!(path[0], "source", "Path should start with source");
            assert_eq!(path[path.len()-1], "destination", "Path should end with destination");
        }
    }

    #[tokio::test]
    async fn test_evaluate_bridge_creativity() {
        let finder = create_test_bridge_finder().await;
        
        // Create test paths with different characteristics
        let short_creative_path = vec!["music".to_string(), "mathematics".to_string()];
        let long_obvious_path = vec![
            "dog".to_string(), 
            "animal".to_string(), 
            "mammal".to_string(), 
            "pet".to_string()
        ];
        
        // Evaluate creativity of short creative path
        let creative_bridge = finder.evaluate_bridge_creativity(
            "music", "mathematics", short_creative_path.clone(), 0.3
        ).await;
        assert!(creative_bridge.is_ok());
        
        let creative_result = creative_bridge.unwrap();
        assert!(creative_result.is_some(), "Creative path should meet threshold");
        
        let bridge = creative_result.unwrap();
        assert!(bridge.novelty_score > 0.3, "Creative bridge should have reasonable novelty");
        
        // Evaluate long obvious path
        let obvious_bridge = finder.evaluate_bridge_creativity(
            "dog", "pet", long_obvious_path.clone(), 0.7 // High threshold
        ).await;
        assert!(obvious_bridge.is_ok());
        
        // Obvious path might not meet high creativity threshold
        let obvious_result = obvious_bridge.unwrap();
        if obvious_result.is_some() {
            let obvious = obvious_result.unwrap();
            assert!(obvious.novelty_score < bridge.novelty_score, 
                   "Obvious path should have lower novelty than creative path");
        }
    }

    #[tokio::test]
    async fn test_creativity_threshold_filtering() {
        let graph = create_mixed_creativity_graph().await;
        let finder = NeuralBridgeFinder::new(graph, 3, 0.8); // High creativity threshold
        
        let result = finder.find_creative_bridges("start", "end").await;
        assert!(result.is_ok());
        
        let bridges = result.unwrap();
        
        // All returned bridges should meet the creativity threshold
        for bridge in &bridges {
            let creativity = bridge.novelty_score * 0.6 + bridge.plausibility_score * 0.4;
            assert!(creativity >= 0.8, 
                   "Bridge should meet creativity threshold: {} (novelty: {}, plausibility: {})", 
                   creativity, bridge.novelty_score, bridge.plausibility_score);
        }
    }

    #[tokio::test]
    async fn test_bridge_explanation_generation() {
        let finder = create_test_bridge_finder().await;
        
        // Test explanation for different bridge types
        let simple_bridge = BridgePath {
            bridge_id: "test_bridge".to_string(),
            start_concept: "music".to_string(),
            end_concept: "emotion".to_string(),
            bridge_concepts: vec!["rhythm".to_string()],
            novelty_score: 0.6,
            plausibility_score: 0.8,
            creativity_score: 0.7,
            explanation: "".to_string(),
            connection_strength: 0.75,
        };
        
        let explanation = finder.generate_explanation(&simple_bridge).await;
        assert!(!explanation.is_empty(), "Should generate non-empty explanation");
        assert!(explanation.contains("music") && explanation.contains("emotion"), 
               "Explanation should mention both endpoints");
        assert!(explanation.contains("rhythm"), 
               "Explanation should mention bridge concept");
    }

    #[tokio::test]
    async fn test_concept_discovery_and_mapping() {
        let graph = create_concept_graph().await;
        let finder = NeuralBridgeFinder::new(graph, 3, 0.3);
        
        // Test that concepts are correctly found in graph
        let art_entity = finder.find_concept_entity("art").await;
        assert!(art_entity.is_ok());
        assert!(art_entity.unwrap().is_some(), "Should find 'art' concept in graph");
        
        let missing_entity = finder.find_concept_entity("nonexistent").await;
        assert!(missing_entity.is_ok());
        assert!(missing_entity.unwrap().is_none(), "Should not find nonexistent concept");
        
        // Test fuzzy matching for close concepts
        let fuzzy_entity = finder.find_concept_entity("artistic").await;
        assert!(fuzzy_entity.is_ok());
        // Depending on implementation, might find "art" as close match
    }

    // Helper functions

    async fn create_test_bridge_finder() -> NeuralBridgeFinder {
        let graph = create_test_graph().await;
        NeuralBridgeFinder::new(graph, 3, 0.3)
    }

    async fn create_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create basic test concepts
        graph.add_entity("art", "Artistic domain").await.unwrap();
        graph.add_entity("physics", "Physics domain").await.unwrap();
        graph.add_entity("symmetry", "Bridge concept").await.unwrap();
        graph.add_entity("music", "Musical concept").await.unwrap();
        graph.add_entity("mathematics", "Mathematical concept").await.unwrap();
        
        // Add relationships
        graph.add_relationship("art", "symmetry", "exhibits", 0.7).await.unwrap();
        graph.add_relationship("symmetry", "physics", "fundamental_to", 0.8).await.unwrap();
        graph.add_relationship("music", "mathematics", "based_on", 0.6).await.unwrap();
        
        graph
    }

    async fn create_weighted_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("node_a", "Start node").await.unwrap();
        graph.add_entity("node_b", "Middle node").await.unwrap();
        graph.add_entity("node_c", "End node").await.unwrap();
        graph.add_entity("weak_node", "Weakly connected").await.unwrap();
        
        // Strong connections
        graph.add_relationship("node_a", "node_b", "strong", 0.9).await.unwrap();
        graph.add_relationship("node_b", "node_c", "strong", 0.8).await.unwrap();
        
        // Weak connections
        graph.add_relationship("node_a", "weak_node", "weak", 0.2).await.unwrap();
        graph.add_relationship("weak_node", "node_c", "weak", 0.1).await.unwrap();
        
        graph
    }

    async fn create_art_physics_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create Art -> Symmetry -> Physics bridge path
        graph.add_entity("Art", "Artistic domain").await.unwrap();
        graph.add_entity("Symmetry", "Mathematical/Artistic concept").await.unwrap();
        graph.add_entity("Physics", "Physics domain").await.unwrap();
        graph.add_entity("Beauty", "Aesthetic concept").await.unwrap();
        
        // Primary bridge path
        graph.add_relationship("Art", "Symmetry", "expresses", 0.7).await.unwrap();
        graph.add_relationship("Symmetry", "Physics", "governs", 0.8).await.unwrap();
        
        // Alternative path
        graph.add_relationship("Art", "Beauty", "creates", 0.8).await.unwrap();
        graph.add_relationship("Beauty", "Symmetry", "involves", 0.6).await.unwrap();
        
        graph
    }

    async fn create_long_chain_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create: Start -> Bridge1 -> Bridge2 -> Bridge3 -> End
        graph.add_entity("Start", "Starting concept").await.unwrap();
        graph.add_entity("Bridge1", "First bridge").await.unwrap();
        graph.add_entity("Bridge2", "Second bridge").await.unwrap();
        graph.add_entity("Bridge3", "Third bridge").await.unwrap();
        graph.add_entity("End", "End concept").await.unwrap();
        
        graph.add_relationship("Start", "Bridge1", "connects", 0.7).await.unwrap();
        graph.add_relationship("Bridge1", "Bridge2", "connects", 0.6).await.unwrap();
        graph.add_relationship("Bridge2", "Bridge3", "connects", 0.7).await.unwrap();
        graph.add_relationship("Bridge3", "End", "connects", 0.8).await.unwrap();
        
        graph
    }

    async fn create_pathfinding_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("source", "Source").await.unwrap();
        graph.add_entity("intermediate1", "Path 1").await.unwrap();
        graph.add_entity("intermediate2", "Path 2").await.unwrap();
        graph.add_entity("destination", "Destination").await.unwrap();
        
        // Multiple paths to destination
        graph.add_relationship("source", "intermediate1", "path1", 0.8).await.unwrap();
        graph.add_relationship("intermediate1", "destination", "path1", 0.7).await.unwrap();
        
        graph.add_relationship("source", "intermediate2", "path2", 0.6).await.unwrap();
        graph.add_relationship("intermediate2", "destination", "path2", 0.9).await.unwrap();
        
        graph
    }

    async fn create_mixed_creativity_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("start", "Start").await.unwrap();
        graph.add_entity("end", "End").await.unwrap();
        graph.add_entity("obvious", "Obvious bridge").await.unwrap();
        graph.add_entity("creative", "Creative bridge").await.unwrap();
        
        // Obvious path (high plausibility, low novelty)
        graph.add_relationship("start", "obvious", "common", 0.9).await.unwrap();
        graph.add_relationship("obvious", "end", "expected", 0.8).await.unwrap();
        
        // Creative path (moderate plausibility, high novelty)
        graph.add_relationship("start", "creative", "surprising", 0.5).await.unwrap();
        graph.add_relationship("creative", "end", "novel", 0.6).await.unwrap();
        
        graph
    }

    async fn create_concept_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("art", "Art concept").await.unwrap();
        graph.add_entity("artistic", "Artistic concept").await.unwrap();
        graph.add_entity("creativity", "Creativity concept").await.unwrap();
        
        graph.add_relationship("art", "artistic", "related", 0.9).await.unwrap();
        graph.add_relationship("art", "creativity", "involves", 0.8).await.unwrap();
        
        graph
    }
}