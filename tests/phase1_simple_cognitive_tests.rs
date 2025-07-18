#[cfg(test)]
mod phase1_simple_cognitive_tests {
    use llmkg::cognitive::{
        ConvergentThinking, DivergentThinking, CriticalThinking,
        SystemsThinking, LateralThinking, AbstractThinking, AdaptiveThinking,
        CognitiveOrchestrator, CognitiveOrchestratorConfig,
        ReasoningStrategy, CognitivePattern
    };
    use llmkg::cognitive::types::*;
    use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use llmkg::core::brain_types::*;
    use std::sync::Arc;

    // Simple test to verify convergent thinking works
    #[tokio::test]
    async fn test_convergent_thinking_works() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Add minimal data
        let ai_key = graph.create_brain_entity(
            "artificial_intelligence".to_string(),
            EntityDirection::Input
        ).await.unwrap();
        
        let ml_key = graph.create_brain_entity(
            "machine_learning".to_string(),
            EntityDirection::Gate
        ).await.unwrap();
        
        graph.create_brain_relationship(
            ai_key,
            ml_key,
            RelationType::IsA,
            0.9
        ).await.unwrap();
        
        let convergent = ConvergentThinking::new(Arc::new(graph));
        
        // Test basic query
        let result = convergent.execute_convergent_query(
            "What is machine learning?",
            None
        ).await;
        
        // Just verify it doesn't crash and returns something
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.answer.is_empty());
        println!("Convergent answer: {}", result.answer);
    }

    // Simple test to verify divergent thinking works with trait
    #[tokio::test]
    async fn test_divergent_thinking_trait() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Add some data
        graph.create_brain_entity("concept1".to_string(), EntityDirection::Input).await.unwrap();
        graph.create_brain_entity("concept2".to_string(), EntityDirection::Input).await.unwrap();
        
        let divergent = DivergentThinking::new(Arc::new(graph));
        
        // Use the trait method
        let result = divergent.execute(
            "explore concepts",
            None,
            PatternParameters::default()
        ).await;
        
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.pattern_type, CognitivePatternType::Divergent);
        println!("Divergent result: {}", result.answer);
    }

    // Simple orchestrator test
    #[tokio::test]
    async fn test_orchestrator_basic() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Add minimal data
        graph.create_brain_entity("test".to_string(), EntityDirection::Input).await.unwrap();
        
        let config = CognitiveOrchestratorConfig::default();
        let orchestrator = CognitiveOrchestrator::new(Arc::new(graph), config).await.unwrap();
        
        // Test with automatic strategy
        let result = orchestrator.reason(
            "What is test?",
            None,
            ReasoningStrategy::Automatic
        ).await;
        
        assert!(result.is_ok());
        println!("Orchestrator selected pattern: {:?}", result.unwrap().pattern_used);
    }
}