#[cfg(test)]
mod convergent_tests {
    use std::collections::HashMap;
    use tokio;
    use crate::cognitive::convergent::ConvergentThinking;
    use crate::cognitive::types::{PatternResult, ConvergentResult, CognitivePatternType};
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;

    #[tokio::test]
    async fn test_calculate_concept_relevance_hierarchical() {
        // Test hierarchical relationship relevance
        let thinking = create_test_convergent_thinking().await;
        
        // Test hierarchical match - golden retriever is a type of dog
        let relevance = thinking.calculate_concept_relevance("golden retriever", "dog");
        assert!(relevance > 0.8, "Hierarchical relevance should be high: {}", relevance);
        
        // Test exact match
        let exact = thinking.calculate_concept_relevance("dog", "dog");
        assert_eq!(exact, 1.0, "Exact match should return 1.0");
        
        // Test unrelated concepts
        let unrelated = thinking.calculate_concept_relevance("dog", "computer");
        assert!(unrelated < 0.2, "Unrelated concepts should have low relevance: {}", unrelated);
    }

    #[tokio::test]
    async fn test_calculate_concept_relevance_semantic() {
        let thinking = create_test_convergent_thinking().await;
        
        // Test semantic field matching - dog and pet are in same semantic field
        let semantic = thinking.calculate_concept_relevance("dog", "pet");
        assert!(semantic > 0.5, "Semantic field relevance should be moderate-high: {}", semantic);
        
        // Test lexical similarity
        let lexical = thinking.calculate_concept_relevance("canine", "dog");
        assert!(lexical > 0.4, "Lexically similar concepts should have moderate relevance: {}", lexical);
    }

    #[tokio::test]
    async fn test_extract_target_concept_basic() {
        let thinking = create_test_convergent_thinking().await;
        
        // Test basic "what are" questions
        let result = thinking.extract_target_concept("what are the properties of a dog").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "dog");
        
        // Test "how many" questions
        let result = thinking.extract_target_concept("how many legs does a cat have").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "cat");
    }

    #[tokio::test]
    async fn test_extract_target_concept_edge_cases() {
        let thinking = create_test_convergent_thinking().await;
        
        // Test query with only stop words
        let result = thinking.extract_target_concept("the and or but").await;
        assert!(result.is_err(), "Stop words only should return error");
        
        // Test empty query
        let result = thinking.extract_target_concept("").await;
        assert!(result.is_err(), "Empty query should return error");
        
        // Test query with no recognizable concepts
        let result = thinking.extract_target_concept("xyz abc def").await;
        assert!(result.is_err(), "Unrecognizable concepts should return error");
    }

    #[tokio::test]
    async fn test_end_to_end_query_and_answer() {
        // Create a test graph with known structure
        let graph = create_test_graph().await;
        let thinking = ConvergentThinking::new(graph, 3, 2); // max_depth=3, beam_width=2
        
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
        let thinking = ConvergentThinking::new(graph, 2, 1); // beam_width=1 for strict pruning
        
        let result = thinking.execute_convergent_query("What connects to start?", "start").await;
        assert!(result.is_ok());
        
        let conv_result = result.unwrap();
        // Should find the more fruitful path despite initial lower activation
        assert!(conv_result.answer.contains("chain_end"), "Should follow longer chain: {}", conv_result.answer);
    }

    #[tokio::test]
    async fn test_focused_propagation() {
        let graph = create_test_graph().await;
        let thinking = ConvergentThinking::new(graph, 2, 3);
        
        // Test propagation from a known start point
        let activation_map = thinking.focused_propagation("dog", 2).await;
        assert!(activation_map.is_ok());
        
        let activations = activation_map.unwrap();
        assert!(activations.len() > 1, "Should propagate to multiple nodes");
        
        // Should include "mammal" with high activation
        let mammal_activation = activations.get("mammal").unwrap_or(&0.0);
        assert!(*mammal_activation > 0.5, "Mammal should be highly activated");
    }

    // Helper functions for creating test data

    async fn create_test_convergent_thinking() -> ConvergentThinking {
        let graph = create_test_graph().await;
        ConvergentThinking::new(graph, 3, 2)
    }

    async fn create_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create basic animal hierarchy
        graph.add_entity("dog", "Animal concept").await.unwrap();
        graph.add_entity("mammal", "Classification concept").await.unwrap();
        graph.add_entity("warm_blooded", "Property concept").await.unwrap();
        graph.add_entity("pet", "Relationship concept").await.unwrap();
        
        // Add relationships
        graph.add_relationship("dog", "mammal", "is_a", 0.9).await.unwrap();
        graph.add_relationship("mammal", "warm_blooded", "has_property", 0.8).await.unwrap();
        graph.add_relationship("dog", "pet", "can_be", 0.7).await.unwrap();
        
        graph
    }

    async fn create_pruning_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create a graph where beam search pruning matters
        graph.add_entity("start", "Starting point").await.unwrap();
        graph.add_entity("dead_end", "High activation but no connections").await.unwrap();
        graph.add_entity("chain_middle", "Lower initial but leads somewhere").await.unwrap();
        graph.add_entity("chain_end", "Final destination with valuable info").await.unwrap();
        
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

    async fn create_deep_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create a deep chain to test depth limits
        graph.add_entity("root", "Starting point").await.unwrap();
        graph.add_entity("level1", "First level").await.unwrap();
        graph.add_entity("level2", "Second level").await.unwrap();
        graph.add_entity("level3", "Third level - should not be reached").await.unwrap();
        
        graph.add_relationship("root", "level1", "connects", 0.8).await.unwrap();
        graph.add_relationship("level1", "level2", "connects", 0.8).await.unwrap();
        graph.add_relationship("level2", "level3", "connects", 0.8).await.unwrap();
        
        graph
    }
}