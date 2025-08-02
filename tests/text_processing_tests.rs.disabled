//! Text Processing Component Tests
//! 
//! Tests for the text processing components that replace pattern recognition functionality

use llmkg::text::{StringNormalizer, HeuristicImportanceScorer, GraphStructurePredictor};
use llmkg::text::structure_predictor::GraphOperation;
use llmkg::text::importance::GraphMetrics;

/// Test StringNormalizer functionality
#[tokio::test]
async fn test_string_normalizer_basic() {
    let normalizer = StringNormalizer::new();
    
    // Test basic normalization
    let result = normalizer.normalize("Complex Text Input!").unwrap();
    assert_eq!(result, "complex text input");
    
    // Test with mixed case
    let result = normalizer.normalize("MiXeD cAsE").unwrap();
    assert_eq!(result, "mixed case");
    
    // Test with special characters
    let result = normalizer.normalize("Hello, World! How are you?").unwrap();
    assert_eq!(result, "hello world how are you");
    
    // Test with numbers
    let result = normalizer.normalize("Test 123 Numbers").unwrap();
    assert_eq!(result, "test 123 numbers");
}

/// Test StringNormalizer edge cases
#[tokio::test]
async fn test_string_normalizer_edge_cases() {
    let normalizer = StringNormalizer::new();
    
    // Test empty string
    let result = normalizer.normalize("").unwrap();
    assert_eq!(result, "");
    
    // Test whitespace only
    let result = normalizer.normalize("   \t\n  ").unwrap();
    assert_eq!(result, "");
    
    // Test single character
    let result = normalizer.normalize("A").unwrap();
    assert_eq!(result, "a");
    
    // Test Unicode characters
    let result = normalizer.normalize("Héllo Wörld").unwrap();
    assert_eq!(result, "hello world");
}

/// Test HeuristicImportanceScorer basic functionality
#[tokio::test]
async fn test_importance_scorer_basic() {
    let scorer = HeuristicImportanceScorer::new();
    
    // Test with simple text
    let score = scorer.calculate_importance("simple text", None);
    assert!(score > 0.0);
    assert!(score <= 1.0);
    
    // Test with important keywords
    let score = scorer.calculate_importance("artificial intelligence machine learning", None);
    assert!(score > 0.5); // Should be relatively high
    
    // Test with common words
    let score = scorer.calculate_importance("the and or but", None);
    assert!(score < 0.5); // Should be relatively low
}

/// Test HeuristicImportanceScorer with context
#[tokio::test]
async fn test_importance_scorer_with_context() {
    let scorer = HeuristicImportanceScorer::new();
    
    // Test with context
    let mut context = std::collections::HashMap::new();
    context.insert("domain".to_string(), "technology".to_string());
    
    let graph_metrics = GraphMetrics::default();
    let score = scorer.calculate_importance("pattern recognition", Some(graph_metrics));
    assert!(score > 0.0);
    assert!(score <= 1.0);
    
    // Test without context for comparison
    let _score_no_context = scorer.calculate_importance("pattern recognition", None);
    assert!(score > 0.0);
    assert!(score <= 1.0);
}

/// Test HeuristicImportanceScorer edge cases
#[tokio::test]
async fn test_importance_scorer_edge_cases() {
    let scorer = HeuristicImportanceScorer::new();
    
    // Test empty string
    let score = scorer.calculate_importance("", None);
    assert_eq!(score, 0.0);
    
    // Test single character
    let score = scorer.calculate_importance("a", None);
    assert!(score > 0.0);
    
    // Test very long text
    let long_text = "word ".repeat(1000);
    let score = scorer.calculate_importance(&long_text, None);
    assert!(score > 0.0);
    assert!(score <= 1.0);
}

/// Test GraphStructurePredictor basic functionality
#[tokio::test]
async fn test_structure_predictor_basic() {
    let predictor = GraphStructurePredictor::new("basic".to_string());
    
    // Test basic structure prediction
    let operations = predictor.predict_structure("Machine learning system").await.unwrap();
    assert!(!operations.is_empty());
    
    // Verify operation types
    for operation in &operations {
        match operation {
            GraphOperation::CreateNode { id, node_type, properties: _ } => {
                assert!(!id.is_empty());
                assert!(!node_type.is_empty());
            },
            GraphOperation::CreateEdge { from, to, relationship, .. } => {
                assert!(!from.is_empty());
                assert!(!to.is_empty());
                assert!(!relationship.is_empty());
            },
            _ => {} // Other operations are valid too
        }
    }
}

/// Test GraphStructurePredictor with different strategies
#[tokio::test]
async fn test_structure_predictor_strategies() {
    // Test basic strategy
    let predictor_basic = GraphStructurePredictor::new("basic".to_string());
    let operations_basic = predictor_basic.predict_structure("Artificial intelligence research").await.unwrap();
    assert!(!operations_basic.is_empty());
    
    // Test enhanced strategy
    let predictor_enhanced = GraphStructurePredictor::new("enhanced".to_string());
    let operations_enhanced = predictor_enhanced.predict_structure("Artificial intelligence research").await.unwrap();
    assert!(!operations_enhanced.is_empty());
    
    // Enhanced should typically produce more operations
    assert!(operations_enhanced.len() >= operations_basic.len());
}

/// Test GraphStructurePredictor edge cases
#[tokio::test]
async fn test_structure_predictor_edge_cases() {
    let predictor = GraphStructurePredictor::new("basic".to_string());
    
    // Test empty string
    let operations = predictor.predict_structure("").await.unwrap();
    assert!(operations.is_empty());
    
    // Test single word
    let operations = predictor.predict_structure("word").await.unwrap();
    assert!(!operations.is_empty());
    
    // Test very short text
    let operations = predictor.predict_structure("AI").await.unwrap();
    assert!(!operations.is_empty());
}

/// Test GraphStructurePredictor with complex text
#[tokio::test]
async fn test_structure_predictor_complex() {
    let predictor = GraphStructurePredictor::new("enhanced".to_string());
    
    let complex_text = "The machine learning architecture consists of multiple layers including \
                       input layers, hidden layers, and output layers. Each layer contains \
                       nodes that are connected to nodes in adjacent layers through \
                       weighted connections. The training process involves backpropagation \
                       to adjust these weights based on the error between predicted and \
                       actual outputs.";
    
    let operations = predictor.predict_structure(complex_text).await.unwrap();
    
    // Should extract multiple entities and relationships
    assert!(operations.len() > 5);
    
    // Should have both entities and relationships
    let has_entities = operations.iter().any(|op| matches!(op, GraphOperation::CreateNode { .. }));
    let has_relations = operations.iter().any(|op| matches!(op, GraphOperation::CreateEdge { .. }));
    
    assert!(has_entities);
    assert!(has_relations);
}

/// Test component integration
#[tokio::test]
async fn test_component_integration() {
    let normalizer = StringNormalizer::new();
    let scorer = HeuristicImportanceScorer::new();
    let predictor = GraphStructurePredictor::new("basic".to_string());
    
    let raw_text = "ARTIFICIAL INTELLIGENCE and Machine Learning!";
    
    // Step 1: Normalize
    let normalized = normalizer.normalize(raw_text).unwrap();
    assert_eq!(normalized, "artificial intelligence and machine learning");
    
    // Step 2: Score importance
    let importance = scorer.calculate_importance(&normalized, None);
    assert!(importance > 0.5); // Should be important
    
    // Step 3: Extract structure
    let operations = predictor.predict_structure(&normalized).await.unwrap();
    assert!(!operations.is_empty());
    
    // Verify the pipeline works end-to-end
    assert!(operations.len() >= 2); // Should extract at least a few concepts
}

/// Test performance with large inputs
#[tokio::test]
async fn test_performance_large_inputs() {
    let normalizer = StringNormalizer::new();
    let scorer = HeuristicImportanceScorer::new();
    let predictor = GraphStructurePredictor::new("basic".to_string());
    
    // Create large input
    let large_text = "Machine learning is a powerful technology. ".repeat(100);
    
    // All operations should complete in reasonable time
    let start = std::time::Instant::now();
    
    let normalized = normalizer.normalize(&large_text).unwrap();
    let _importance = scorer.calculate_importance(&normalized, None);
    let _operations = predictor.predict_structure(&normalized).await.unwrap();
    
    let duration = start.elapsed();
    
    // Should complete in under 5 seconds for this size
    assert!(duration.as_secs() < 5);
}

/// Test error handling in text processing components
#[tokio::test]
async fn test_error_handling() {
    let normalizer = StringNormalizer::new();
    let scorer = HeuristicImportanceScorer::new();
    let predictor = GraphStructurePredictor::new("basic".to_string());
    
    // Test with various problematic inputs
    let problematic_inputs = vec![
        "",
        "   ",
        "\t\n\r",
        "\0",
        "🚀🌟💡", // Emojis
        "Θεός", // Greek text
        "中文", // Chinese text
    ];
    
    for input in problematic_inputs {
        // All operations should handle these gracefully
        let norm_result = normalizer.normalize(input);
        assert!(norm_result.is_ok());
        
        let normalized = norm_result.unwrap();
        let _score = scorer.calculate_importance(&normalized, None);
        
        let struct_result = predictor.predict_structure(&normalized).await;
        assert!(struct_result.is_ok());
    }
}

/// Test concurrent usage of text processing components
#[tokio::test]
async fn test_concurrent_usage() {
    use std::sync::Arc;
    
    let normalizer = Arc::new(StringNormalizer::new());
    let scorer = Arc::new(HeuristicImportanceScorer::new());
    let predictor = Arc::new(GraphStructurePredictor::new("basic".to_string()));
    
    let mut handles = Vec::new();
    
    // Spawn multiple tasks using the same components
    for i in 0..10 {
        let normalizer = normalizer.clone();
        let scorer = scorer.clone();
        let predictor = predictor.clone();
        
        let handle = tokio::spawn(async move {
            let text = format!("Test text number {i}");
            let normalized = normalizer.normalize(&text).unwrap();
            let _score = scorer.calculate_importance(&normalized, None);
            let operations = predictor.predict_structure(&normalized).await.unwrap();
            operations.len()
        });
        
        handles.push(handle);
    }
    
    // Wait for all tasks and verify they completed successfully
    let mut total_operations = 0;
    for handle in handles {
        let result = handle.await.unwrap();
        total_operations += result;
    }
    
    // Should have processed operations from all tasks
    assert!(total_operations > 0);
}