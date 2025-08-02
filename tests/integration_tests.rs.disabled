//! Integration tests for LLMKG system core functionality

use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::text::{StringNormalizer, HeuristicImportanceScorer, GraphStructurePredictor};
use llmkg::core::types::EntityData;

/// Test basic knowledge graph functionality
#[tokio::test]
async fn test_brain_enhanced_knowledge_graph_creation() {
    let graph = BrainEnhancedKnowledgeGraph::new(128).unwrap();
    assert_eq!(graph.entity_count(), 0);
}

/// Test text processing components work independently
#[tokio::test]
async fn test_text_processing_components() {
    // Test StringNormalizer
    let normalizer = StringNormalizer::new();
    let result = normalizer.normalize("MIXED case Text!").unwrap();
    assert_eq!(result, "mixed case text");
    
    // Test HeuristicImportanceScorer
    let scorer = HeuristicImportanceScorer::new();
    let score = scorer.calculate_importance("important concept", None);
    assert!(score > 0.0);
    
    // Test GraphStructurePredictor
    let predictor = GraphStructurePredictor::new("basic".to_string());
    let operations = predictor.predict_structure("test content").await.unwrap();
    // Should return some operations for any non-empty content
    assert!(!operations.is_empty());
}

/// Test basic graph operations
#[tokio::test]
async fn test_basic_graph_operations() {
    let graph = BrainEnhancedKnowledgeGraph::new(128).unwrap();
    
    // Add an entity
    let entity_data = EntityData::new(1, "test entity".to_string(), vec![0.5; 128]);
    let entity_key = graph.add_entity(entity_data).await.unwrap();
    
    // Verify entity was added
    assert_eq!(graph.entity_count(), 1);
    
    // Test entity retrieval
    let retrieved = graph.get_entity(entity_key).await;
    assert!(retrieved.is_some());
}