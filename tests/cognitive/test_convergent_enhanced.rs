#[cfg(test)]
mod convergent_enhanced_tests {
    use std::collections::BinaryHeap;
    use tokio;
    use crate::cognitive::convergent_enhanced::EnhancedConvergentThinking;
    use crate::cognitive::convergent_enhanced::BeamSearchNode;
    use crate::cognitive::types::{PatternResult, ConvergentResult, CognitivePatternType};
    use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;

    #[tokio::test]
    async fn test_beam_search_node_ordering() {
        // Test that BeamSearchNode correctly implements ordering for BinaryHeap
        let mut heap = BinaryHeap::new();
        
        // Add nodes with different scores
        let node1 = BeamSearchNode {
            entity_key: "entity1".to_string(),
            cumulative_score: 0.5,
            path_score: 0.5,
            depth: 1,
            path: vec!["start".to_string(), "entity1".to_string()],
            attention_weight: 0.8,
        };
        
        let node2 = BeamSearchNode {
            entity_key: "entity2".to_string(),
            cumulative_score: 0.9, // Higher score
            path_score: 0.9,
            depth: 1,
            path: vec!["start".to_string(), "entity2".to_string()],
            attention_weight: 0.9,
        };
        
        let node3 = BeamSearchNode {
            entity_key: "entity3".to_string(),
            cumulative_score: 0.2, // Lower score
            path_score: 0.2,
            depth: 1,
            path: vec!["start".to_string(), "entity3".to_string()],
            attention_weight: 0.3,
        };
        
        heap.push(node1);
        heap.push(node2);
        heap.push(node3);
        
        // Verify ordering - highest score should come first
        let first = heap.pop().unwrap();
        assert_eq!(first.cumulative_score, 0.9, "Highest score node should be first");
        assert_eq!(first.entity_key, "entity2");
        
        let second = heap.pop().unwrap();
        assert_eq!(second.cumulative_score, 0.5, "Medium score node should be second");
        
        let third = heap.pop().unwrap();
        assert_eq!(third.cumulative_score, 0.2, "Lowest score node should be last");
    }

    #[tokio::test]
    async fn test_compute_cosine_similarity_basic() {
        let thinking = create_test_enhanced_convergent().await;
        
        // Test with known vectors
        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![1.0, 0.0, 0.0];
        let similarity = thinking.compute_cosine_similarity(&vec1, &vec2);
        assert!((similarity - 1.0).abs() < 0.001, "Identical vectors should have similarity 1.0");
        
        // Test orthogonal vectors
        let vec3 = vec![1.0, 0.0, 0.0];
        let vec4 = vec![0.0, 1.0, 0.0];
        let orthogonal_sim = thinking.compute_cosine_similarity(&vec3, &vec4);
        assert!((orthogonal_sim - 0.0).abs() < 0.001, "Orthogonal vectors should have similarity 0.0");
        
        // Test opposite vectors
        let vec5 = vec![1.0, 0.0, 0.0];
        let vec6 = vec![-1.0, 0.0, 0.0];
        let opposite_sim = thinking.compute_cosine_similarity(&vec5, &vec6);
        assert!((opposite_sim + 1.0).abs() < 0.001, "Opposite vectors should have similarity -1.0");
    }

    #[tokio::test]
    async fn test_compute_cosine_similarity_edge_cases() {
        let thinking = create_test_enhanced_convergent().await;
        
        // Test with zero vectors
        let zero_vec = vec![0.0, 0.0, 0.0];
        let normal_vec = vec![1.0, 1.0, 1.0];
        let zero_sim = thinking.compute_cosine_similarity(&zero_vec, &normal_vec);
        assert!(zero_sim.is_nan() || zero_sim == 0.0, "Zero vector similarity should be NaN or 0");
        
        // Test with vectors of different lengths
        let short_vec = vec![1.0, 0.0];
        let long_vec = vec![1.0, 0.0, 0.0, 1.0];
        // This should either handle gracefully or return error
        let diff_len_sim = thinking.compute_cosine_similarity(&short_vec, &long_vec);
        // Implementation dependent - should not panic
        
        // Test with very small vectors
        let tiny_vec1 = vec![1e-10, 0.0, 0.0];
        let tiny_vec2 = vec![1e-10, 0.0, 0.0];
        let tiny_sim = thinking.compute_cosine_similarity(&tiny_vec1, &tiny_vec2);
        // Should handle numerical precision issues gracefully
        assert!(!tiny_sim.is_infinite(), "Should handle tiny vectors without infinity");
    }

    #[tokio::test]
    async fn test_neural_guided_search_path() {
        // Create test graph with two possible paths - one semantically relevant, one not
        let graph = create_neural_guidance_test_graph().await;
        let thinking = EnhancedConvergentThinking::new(graph);
        
        // Mock neural server to prefer semantic path
        // In real implementation, this would involve mocking the neural server
        // For this test, we assume the neural guidance correctly identifies relevant paths
        
        let result = thinking.execute_advanced_convergent_query("What is the nature of artistic beauty?").await;
        assert!(result.is_ok());
        
        let conv_result = result.unwrap();
        
        // Verify that neural guidance influenced the path selection
        assert!(conv_result.confidence > 0.6, "Neural guidance should produce confident results");
        
        // Check that the answer includes semantically relevant concepts
        let answer_lower = conv_result.answer.to_lowercase();
        assert!(answer_lower.contains("beauty") || answer_lower.contains("aesthetic") || answer_lower.contains("harmony"),
               "Answer should include semantically relevant concepts: {}", conv_result.answer);
        
        // Verify reasoning path shows neural guidance was used
        assert!(conv_result.reasoning_path.len() > 0, "Should have reasoning path");
        assert!(conv_result.neural_guidance_used, "Should indicate neural guidance was used");
    }

    #[tokio::test]
    async fn test_confidence_calibration_and_history() {
        let graph = create_confidence_test_graph().await;
        let thinking = EnhancedConvergentThinking::new(graph);
        
        // Execute a query that should produce high initial confidence
        let result = thinking.execute_advanced_convergent_query("What connects directly to the central node?").await;
        assert!(result.is_ok());
        
        let conv_result = result.unwrap();
        
        // Check confidence calibration
        assert!(conv_result.confidence > 0.0 && conv_result.confidence <= 1.0, 
               "Confidence should be in valid range after calibration: {}", conv_result.confidence);
        
        // Verify calibration made confidence more conservative
        assert!(conv_result.raw_confidence >= conv_result.confidence, 
               "Calibrated confidence should be more conservative than raw");
        
        // Check that reasoning history is updated
        assert!(conv_result.reasoning_trace.query.contains("central node"), 
               "Reasoning trace should capture the query");
        assert!(conv_result.reasoning_trace.neural_steps.len() > 0, 
               "Should have neural processing steps");
        assert!(conv_result.reasoning_trace.final_confidence == conv_result.confidence, 
               "Trace should record final confidence");
        
        // Verify history tracking
        let history_length = thinking.get_reasoning_history_length().await;
        assert!(history_length > 0, "Should have history entries after execution");
    }

    #[tokio::test]
    async fn test_neural_query_understanding() {
        let thinking = create_test_enhanced_convergent().await;
        
        // Test complex query understanding
        let complex_query = "What are the fundamental mathematical principles underlying the aesthetic beauty found in Renaissance art?";
        
        let understanding = thinking.neural_query_understanding(complex_query).await;
        assert!(understanding.is_ok());
        
        let query_analysis = understanding.unwrap();
        
        // Verify key concepts were identified
        assert!(query_analysis.primary_concepts.len() > 0, "Should identify primary concepts");
        assert!(query_analysis.semantic_intent.len() > 0, "Should determine semantic intent");
        
        // Check for expected concept extraction
        let concepts_str = query_analysis.primary_concepts.join(" ").to_lowercase();
        assert!(concepts_str.contains("mathematical") || concepts_str.contains("aesthetic") || 
                concepts_str.contains("beauty") || concepts_str.contains("renaissance") || 
                concepts_str.contains("art"), 
               "Should extract key concepts from query: {:?}", query_analysis.primary_concepts);
    }

    #[tokio::test]
    async fn test_build_semantic_context() {
        let graph = create_semantic_context_graph().await;
        let thinking = EnhancedConvergentThinking::new(graph);
        
        // Test semantic context building for art-related query
        let context = thinking.build_semantic_context("Renaissance art", vec!["art".to_string(), "renaissance".to_string()]).await;
        assert!(context.is_ok());
        
        let semantic_context = context.unwrap();
        
        // Verify context enrichment
        assert!(semantic_context.associated_concepts.len() > 0, "Should find associated concepts");
        assert!(semantic_context.temporal_context.is_some(), "Should establish temporal context");
        
        // Check that relevant concepts were included
        let associated_str = semantic_context.associated_concepts.join(" ").to_lowercase();
        assert!(associated_str.contains("beauty") || associated_str.contains("painting") || 
                associated_str.contains("sculpture"), 
               "Should include art-related associated concepts: {:?}", semantic_context.associated_concepts);
    }

    #[tokio::test]
    async fn test_beam_search_with_neural_guidance() {
        let graph = create_beam_search_test_graph().await;
        let thinking = EnhancedConvergentThinking::new(graph);
        
        // Test that beam search with neural guidance finds better paths than simple search
        let result = thinking.neural_guided_beam_search(
            vec!["start_concept".to_string()], 
            "target concept",
            3, // beam_width
            4  // max_depth
        ).await;
        
        assert!(result.is_ok());
        let search_result = result.unwrap();
        
        // Verify beam search results
        assert!(search_result.best_paths.len() > 0, "Should find paths to target");
        assert!(search_result.final_nodes.len() <= 3, "Should respect beam width");
        
        // Check that paths are ranked by neural guidance scores
        if search_result.best_paths.len() > 1 {
            let first_score = search_result.best_paths[0].cumulative_score;
            let second_score = search_result.best_paths[1].cumulative_score;
            assert!(first_score >= second_score, "Paths should be ranked by score");
        }
        
        // Verify neural guidance influenced selection
        assert!(search_result.neural_guidance_applied, "Should indicate neural guidance was used");
    }

    #[tokio::test]
    async fn test_attention_weighted_propagation() {
        let graph = create_attention_test_graph().await;
        let thinking = EnhancedConvergentThinking::new(graph);
        
        // Test that attention weights properly influence propagation
        let propagation = thinking.attention_weighted_propagation(
            "focus_concept",
            vec!["related_a".to_string(), "related_b".to_string(), "distractor".to_string()],
            0.8 // attention_strength
        ).await;
        
        assert!(propagation.is_ok());
        let weighted_activations = propagation.unwrap();
        
        // Verify attention weighting worked
        assert!(weighted_activations.len() > 0, "Should have weighted activations");
        
        // Related concepts should have higher activation than distractors
        let related_activation = weighted_activations.get("related_a").unwrap_or(&0.0);
        let distractor_activation = weighted_activations.get("distractor").unwrap_or(&0.0);
        
        assert!(related_activation > distractor_activation, 
               "Related concepts should have higher activation than distractors: {} vs {}", 
               related_activation, distractor_activation);
    }

    #[tokio::test]
    async fn test_cognitive_pattern_interface() {
        let thinking = create_test_enhanced_convergent().await;
        
        let result = thinking.execute("What is the relationship between mathematics and music?").await;
        assert!(result.is_ok());
        
        let pattern_result = result.unwrap();
        match pattern_result {
            PatternResult::Convergent(conv_result) => {
                assert_eq!(conv_result.pattern_type, CognitivePatternType::Convergent);
                assert!(conv_result.confidence >= 0.0 && conv_result.confidence <= 1.0);
                assert!(!conv_result.answer.is_empty());
                assert!(conv_result.neural_guidance_used, "Enhanced version should use neural guidance");
                assert!(conv_result.reasoning_path.len() > 0, "Should have detailed reasoning path");
            },
            _ => panic!("Expected ConvergentResult from enhanced convergent thinking")
        }
    }

    #[tokio::test]
    async fn test_multi_modal_context_integration() {
        let graph = create_multi_modal_graph().await;
        let thinking = EnhancedConvergentThinking::new(graph);
        
        // Test integration of multiple modalities (text, visual, auditory concepts)
        let result = thinking.execute_advanced_convergent_query("How does visual art relate to musical harmony?").await;
        assert!(result.is_ok());
        
        let conv_result = result.unwrap();
        
        // Should integrate concepts from different modalities
        let answer_lower = conv_result.answer.to_lowercase();
        assert!(answer_lower.contains("visual") || answer_lower.contains("musical") || 
                answer_lower.contains("harmony") || answer_lower.contains("pattern"),
               "Should integrate multi-modal concepts: {}", conv_result.answer);
        
        // Should have high confidence for cross-modal connections
        assert!(conv_result.confidence > 0.5, 
               "Multi-modal integration should be confident: {}", conv_result.confidence);
    }

    // Helper functions

    async fn create_test_enhanced_convergent() -> EnhancedConvergentThinking {
        let graph = create_test_graph().await;
        EnhancedConvergentThinking::new(graph)
    }

    async fn create_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create basic test structure
        graph.add_entity("art", "Artistic domain").await.unwrap();
        graph.add_entity("beauty", "Aesthetic concept").await.unwrap();
        graph.add_entity("harmony", "Musical/visual concept").await.unwrap();
        graph.add_entity("mathematics", "Mathematical domain").await.unwrap();
        
        graph.add_relationship("art", "beauty", "expresses", 0.8).await.unwrap();
        graph.add_relationship("beauty", "harmony", "involves", 0.7).await.unwrap();
        graph.add_relationship("harmony", "mathematics", "based_on", 0.6).await.unwrap();
        
        graph
    }

    async fn create_neural_guidance_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create paths where semantic relevance matters
        graph.add_entity("artistic_beauty", "Query concept").await.unwrap();
        graph.add_entity("aesthetic_theory", "Semantically relevant").await.unwrap();
        graph.add_entity("random_concept", "Semantically irrelevant").await.unwrap();
        graph.add_entity("harmony", "Aesthetically relevant").await.unwrap();
        graph.add_entity("proportion", "Mathematical beauty").await.unwrap();
        
        // Relevant semantic path
        graph.add_relationship("artistic_beauty", "aesthetic_theory", "relates_to", 0.6).await.unwrap();
        graph.add_relationship("aesthetic_theory", "harmony", "involves", 0.8).await.unwrap();
        graph.add_relationship("harmony", "proportion", "based_on", 0.7).await.unwrap();
        
        // Irrelevant but direct path
        graph.add_relationship("artistic_beauty", "random_concept", "mentioned_with", 0.9).await.unwrap();
        
        graph
    }

    async fn create_confidence_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("central_node", "Central concept").await.unwrap();
        graph.add_entity("direct_connection", "Directly connected").await.unwrap();
        graph.add_entity("secondary_connection", "Secondary connection").await.unwrap();
        
        // Strong direct connection
        graph.add_relationship("central_node", "direct_connection", "directly_connects", 0.9).await.unwrap();
        graph.add_relationship("direct_connection", "secondary_connection", "leads_to", 0.7).await.unwrap();
        
        graph
    }

    async fn create_semantic_context_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Renaissance art context
        graph.add_entity("renaissance", "Historical period").await.unwrap();
        graph.add_entity("art", "Artistic domain").await.unwrap();
        graph.add_entity("painting", "Art form").await.unwrap();
        graph.add_entity("sculpture", "Art form").await.unwrap();
        graph.add_entity("beauty", "Aesthetic concept").await.unwrap();
        graph.add_entity("perspective", "Artistic technique").await.unwrap();
        
        graph.add_relationship("renaissance", "art", "period_of", 0.9).await.unwrap();
        graph.add_relationship("art", "painting", "includes", 0.8).await.unwrap();
        graph.add_relationship("art", "sculpture", "includes", 0.8).await.unwrap();
        graph.add_relationship("art", "beauty", "expresses", 0.7).await.unwrap();
        graph.add_relationship("renaissance", "perspective", "developed", 0.8).await.unwrap();
        
        graph
    }

    async fn create_beam_search_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Create branching paths for beam search
        graph.add_entity("start_concept", "Starting point").await.unwrap();
        graph.add_entity("path_a_1", "Path A step 1").await.unwrap();
        graph.add_entity("path_a_2", "Path A step 2").await.unwrap();
        graph.add_entity("path_b_1", "Path B step 1").await.unwrap();
        graph.add_entity("path_b_2", "Path B step 2").await.unwrap();
        graph.add_entity("target_concept", "Target").await.unwrap();
        
        // Multiple paths to target
        graph.add_relationship("start_concept", "path_a_1", "connects", 0.8).await.unwrap();
        graph.add_relationship("path_a_1", "path_a_2", "connects", 0.7).await.unwrap();
        graph.add_relationship("path_a_2", "target_concept", "reaches", 0.9).await.unwrap();
        
        graph.add_relationship("start_concept", "path_b_1", "connects", 0.6).await.unwrap();
        graph.add_relationship("path_b_1", "path_b_2", "connects", 0.8).await.unwrap();
        graph.add_relationship("path_b_2", "target_concept", "reaches", 0.8).await.unwrap();
        
        graph
    }

    async fn create_attention_test_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        graph.add_entity("focus_concept", "Focus of attention").await.unwrap();
        graph.add_entity("related_a", "Related concept A").await.unwrap();
        graph.add_entity("related_b", "Related concept B").await.unwrap();
        graph.add_entity("distractor", "Distractor concept").await.unwrap();
        
        // Related concepts should get more attention
        graph.add_relationship("focus_concept", "related_a", "closely_related", 0.8).await.unwrap();
        graph.add_relationship("focus_concept", "related_b", "somewhat_related", 0.6).await.unwrap();
        graph.add_relationship("focus_concept", "distractor", "weakly_related", 0.2).await.unwrap();
        
        graph
    }

    async fn create_multi_modal_graph() -> BrainEnhancedKnowledgeGraph {
        let mut graph = BrainEnhancedKnowledgeGraph::new().await;
        
        // Visual modality
        graph.add_entity("visual_art", "Visual domain").await.unwrap();
        graph.add_entity("color", "Visual element").await.unwrap();
        graph.add_entity("composition", "Visual structure").await.unwrap();
        
        // Auditory modality
        graph.add_entity("musical_harmony", "Auditory domain").await.unwrap();
        graph.add_entity("frequency", "Sound element").await.unwrap();
        graph.add_entity("rhythm", "Temporal structure").await.unwrap();
        
        // Cross-modal connections
        graph.add_entity("pattern", "Abstract structure").await.unwrap();
        graph.add_entity("proportion", "Mathematical relationship").await.unwrap();
        
        // Visual connections
        graph.add_relationship("visual_art", "color", "uses", 0.9).await.unwrap();
        graph.add_relationship("visual_art", "composition", "has", 0.8).await.unwrap();
        graph.add_relationship("composition", "pattern", "exhibits", 0.7).await.unwrap();
        
        // Auditory connections
        graph.add_relationship("musical_harmony", "frequency", "based_on", 0.9).await.unwrap();
        graph.add_relationship("musical_harmony", "rhythm", "includes", 0.8).await.unwrap();
        graph.add_relationship("rhythm", "pattern", "is_a", 0.8).await.unwrap();
        
        // Cross-modal bridges
        graph.add_relationship("pattern", "proportion", "follows", 0.7).await.unwrap();
        graph.add_relationship("color", "frequency", "analogous_to", 0.5).await.unwrap();
        
        graph
    }
}