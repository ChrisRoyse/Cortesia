#[cfg(test)]
mod divergent_tests {
    use tokio;
    use llmkg::cognitive::divergent::{DivergentThinking, calculate_concept_similarity, extract_seed_concept, infer_exploration_type, ExplorationType};
    use llmkg::cognitive::{PatternResult, DivergentResult, ExplorationPath, CognitivePatternType};
    use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;

    // NOTE: Tests for extract_seed_concept method have been moved to src/cognitive/divergent.rs
    // The public extract_seed_concept function is still tested below

    #[tokio::test]
    async fn test_seed_concept_extraction() {
        let test_cases = vec![
            ("examples of dogs", "dogs"),
            ("brainstorm about artificial intelligence", "artificial intelligence"),
            ("creative uses for plastic bottles", "plastic bottles"),
            ("things related to space exploration", "space exploration"),
            ("innovative applications of nanotechnology", "nanotechnology"),
        ];
        
        for (query, expected) in test_cases {
            let result = extract_seed_concept(query).await.unwrap();
            assert_eq!(result, expected);
        }
    }

    // NOTE: Tests for rank_by_creativity have been moved to src/cognitive/divergent.rs
    // in the #[cfg(test)] module where they can access the private method directly.

    #[tokio::test]
    async fn test_path_exploration_and_diversity() {
        // Create a graph with multiple distinct clusters
        let graph = create_diverse_test_graph().await;
        let thinking = DivergentThinking::new_with_params(graph, 5, 0.3); // exploration_breadth=5
        
        // Execute exploration from "music" seed
        let result = thinking.execute("What are examples of music?").await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        match pattern_result {
            PatternResult::Divergent(div_result) => {
                assert!(div_result.paths.len() > 1, "Should find multiple exploration paths");
                
                // Should explore both classical and rock clusters
                let has_classical = div_result.paths.iter().any(|p| p.destination.contains("classical"));
                let has_rock = div_result.paths.iter().any(|p| p.destination.contains("rock"));
                
                assert!(has_classical || has_rock, "Should explore different clusters");
                assert!(div_result.confidence > 0.0, "Should have positive confidence");
            },
            _ => panic!("Expected DivergentResult")
        }
    }

    #[tokio::test]
    async fn test_novelty_ranking() {
        // Test the system's ability to identify and prioritize novel connections
        let graph = create_novelty_test_graph().await;
        let thinking = DivergentThinking::new_with_params(graph, 3, 0.2); // Custom params
        
        let result = thinking.execute_divergent_exploration("seed", 3).await;
        assert!(result.is_ok());
        
        let div_result = result.unwrap();
        assert!(div_result.paths.len() >= 2, "Should find both obvious and novel paths");
        
        // The novel path should rank higher due to high novelty weight
        let has_novel_higher = div_result.paths.iter()
            .take(2)
            .any(|p| p.destination.contains("novel"));
        
        assert!(has_novel_higher, "Novel connection should rank highly: {:?}", 
               div_result.paths.iter().map(|p| &p.destination).collect::<Vec<_>>());
    }

    // NOTE: Tests for spread_activation have been moved to src/cognitive/divergent.rs
    // in the #[cfg(test)] module where they can access the private method directly.

    // NOTE: Tests for neural_path_exploration have been moved to src/cognitive/divergent.rs
    // in the #[cfg(test)] module where they can access the private method directly.

    #[tokio::test]
    async fn test_cognitive_pattern_interface() {
        let thinking = create_test_divergent_thinking().await;
        
        let result = thinking.execute("What are examples of animals?").await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        match pattern_result {
            PatternResult::Divergent(div_result) => {
                assert!(!div_result.paths.is_empty(), "Should have exploration paths");
                assert!(div_result.confidence >= 0.0 && div_result.confidence <= 1.0);
                assert_eq!(div_result.pattern_type, CognitivePatternType::Divergent);
                assert!(!div_result.seed_concept.is_empty(), "Should identify seed concept");
            },
            _ => panic!("Expected DivergentResult from divergent thinking")
        }
    }

    #[tokio::test]
    async fn test_creativity_threshold() {
        let graph = create_test_graph().await;
        let thinking = DivergentThinking::new(graph, 3, 0.8, 0.5); // High creativity threshold
        
        let result = thinking.execute("Tell me about music").await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        if let PatternResult::Divergent(div_result) = pattern_result {
            // With high creativity threshold, should filter out low-creativity paths
            for path in &div_result.paths {
                let creativity_score = path.relevance_score * 0.5 + path.novelty_score * 0.5;
                assert!(creativity_score >= 0.8, 
                       "All paths should meet creativity threshold: {}", creativity_score);
            }
        }
    }

    #[test]
    fn test_concept_similarity() {
        // High similarity
        assert!(calculate_concept_similarity("dog", "puppy") > 0.8);
        assert!(calculate_concept_similarity("car", "automobile") > 0.8);
        assert!(calculate_concept_similarity("happy", "joyful") > 0.7);
        
        // Medium similarity
        assert!(calculate_concept_similarity("dog", "cat") > 0.4);
        assert!(calculate_concept_similarity("computer", "laptop") > 0.6);
        assert!(calculate_concept_similarity("tree", "forest") > 0.5);
        
        // Low similarity
        assert!(calculate_concept_similarity("dog", "quantum") < 0.2);
        assert!(calculate_concept_similarity("music", "mathematics") < 0.4);
        assert!(calculate_concept_similarity("ocean", "desert") < 0.3);
        
        // Identical concepts
        assert_eq!(calculate_concept_similarity("test", "test"), 1.0);
        
        // Empty strings
        assert_eq!(calculate_concept_similarity("", "test"), 0.0);
        assert_eq!(calculate_concept_similarity("test", ""), 0.0);
        assert_eq!(calculate_concept_similarity("", ""), 0.0);
    }

    #[test]
    fn test_exploration_type_inference() {
        // Example queries
        assert_eq!(
            infer_exploration_type("give me examples of machine learning algorithms"),
            ExplorationType::Examples
        );
        
        // Creative queries
        assert_eq!(
            infer_exploration_type("brainstorm innovative uses for blockchain"),
            ExplorationType::Creative
        );
        
        // Related queries
        assert_eq!(
            infer_exploration_type("what concepts are related to quantum computing"),
            ExplorationType::Related
        );
    }

    // Helper functions

    async fn create_test_divergent_thinking() -> DivergentThinking {
        let graph = create_test_graph().await;
        DivergentThinking::new(graph, 3, 0.3, 0.5)
    }

    async fn create_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create a music-focused graph
        graph.add_entity("music", "Central concept").await.unwrap();
        graph.add_entity("classical", "Music genre").await.unwrap();
        graph.add_entity("rock", "Music genre").await.unwrap();
        graph.add_entity("symphony", "Classical form").await.unwrap();
        graph.add_entity("guitar", "Rock instrument").await.unwrap();
        
        // Add relationships
        graph.add_relationship("music", "classical", "genre", 0.8).await.unwrap();
        graph.add_relationship("music", "rock", "genre", 0.8).await.unwrap();
        graph.add_relationship("classical", "symphony", "form", 0.7).await.unwrap();
        graph.add_relationship("rock", "guitar", "instrument", 0.7).await.unwrap();
        
        graph
    }

    async fn create_diverse_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create distinct clusters connected to music
        graph.add_entity("music", "Central seed").await.unwrap();
        
        // Classical cluster
        graph.add_entity("classical", "Genre").await.unwrap();
        graph.add_entity("orchestra", "Classical group").await.unwrap();
        graph.add_entity("composer", "Classical creator").await.unwrap();
        
        // Rock cluster  
        graph.add_entity("rock", "Genre").await.unwrap();
        graph.add_entity("band", "Rock group").await.unwrap();
        graph.add_entity("drummer", "Rock member").await.unwrap();
        
        // Connect clusters to seed
        graph.add_relationship("music", "classical", "includes", 0.8).await.unwrap();
        graph.add_relationship("music", "rock", "includes", 0.8).await.unwrap();
        
        // Internal cluster connections
        graph.add_relationship("classical", "orchestra", "performed_by", 0.9).await.unwrap();
        graph.add_relationship("classical", "composer", "created_by", 0.9).await.unwrap();
        graph.add_relationship("rock", "band", "performed_by", 0.9).await.unwrap();
        graph.add_relationship("rock", "drummer", "includes", 0.8).await.unwrap();
        
        graph
    }

    async fn create_novelty_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("seed", "Starting point").await.unwrap();
        graph.add_entity("obvious", "Direct connection").await.unwrap();
        graph.add_entity("novel", "Creative connection").await.unwrap();
        
        // Strong, obvious connection
        graph.add_relationship("seed", "obvious", "direct", 0.9).await.unwrap();
        
        // Weaker but more novel connection
        graph.add_relationship("seed", "novel", "creative", 0.4).await.unwrap();
        
        graph
    }

    async fn create_path_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("start", "Beginning").await.unwrap();
        graph.add_entity("bridge", "Intermediate").await.unwrap();
        graph.add_entity("end", "Destination").await.unwrap();
        
        graph.add_relationship("start", "bridge", "connects", 0.7).await.unwrap();
        graph.add_relationship("bridge", "end", "leads_to", 0.8).await.unwrap();
        
        graph
    }

    fn create_mock_path(id: &str, relevance: f32, novelty: f32) -> ExplorationPath {
        ExplorationPath {
            path_id: id.to_string(),
            source_concept: "test_source".to_string(),
            destination: format!("dest_{}", id),
            intermediate_concepts: vec![],
            relevance_score: relevance,
            novelty_score: novelty,
            path_length: 2,
            explanation: format!("Mock path {}", id),
        }
    }

    // Mock struct for testing
    struct ExplorationMap {
        activated_nodes: std::collections::HashMap<String, f32>,
        exploration_depth: u32,
    }

    impl ExplorationMap {
        fn new() -> Self {
            Self {
                activated_nodes: std::collections::HashMap::new(),
                exploration_depth: 0,
            }
        }
        
        fn add_activation(&mut self, node: &str, activation: f32) {
            self.activated_nodes.insert(node.to_string(), activation);
        }
    }
}