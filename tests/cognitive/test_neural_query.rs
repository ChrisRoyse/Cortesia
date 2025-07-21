//! Tests for the neural query processor (public interface)
//! Private method tests are in src/cognitive/neural_query.rs

use llmkg::cognitive::neural_query::*;
use llmkg::cognitive::types::*;
use llmkg::graph::Graph;
use std::sync::Arc;

#[cfg(test)]
mod tests {
    use super::*;
    
    // Helper function to create a test processor
    fn create_test_processor() -> NeuralQueryProcessor {
        let graph = Arc::new(Graph::new());
        NeuralQueryProcessor::new(graph)
    }
    
    #[test]
    fn test_public_query_intent_identification() {
        let processor = create_test_processor();
        let test_cases = vec![
            ("What is quantum computing?", QueryIntent::Factual),
            ("What makes up an atom?", QueryIntent::Compositional),
            ("How does photosynthesis relate to solar panels?", QueryIntent::Relational),
            ("Why is the sky blue?", QueryIntent::Causal),
            ("What if we had no gravity?", QueryIntent::Counterfactual),
            ("Which is better, solar or wind energy?", QueryIntent::Comparative),
        ];
        
        for (query, expected_intent) in test_cases {
            let intent = processor.extract_question_type(query);
            assert_eq!(
                intent, expected_intent,
                "Failed for query: {}",
                query
            );
        }
    }

    #[test]
    fn test_query_understanding() {
        let processor = create_test_processor();
        let context = QueryContext::default();
        
        let result = processor.understand_query(
            "How does the brain process visual information from the eyes?",
            Some(&context)
        ).unwrap();
        
        // Check that we got some understanding
        assert!(!result.concepts.is_empty());
        assert!(matches!(result.intent, QueryIntent::Relational));
    }
    
    #[test]
    fn test_find_entities_in_query() {
        let processor = create_test_processor();
        let test_cases = vec![
            "How does machine learning relate to artificial intelligence?",
            "What is the connection between DNA and heredity?",
            "Examples of sustainable energy sources",
        ];
        
        for query in test_cases {
            let entities = processor.find_entities_in_query(query);
            // Since we're working with an empty graph, we won't find entities,
            // but we're testing that the method works
            assert_eq!(entities.len(), 0);
        }
    }

    #[test]
    fn test_get_related_concepts() {
        let processor = create_test_processor();
        
        // Test that the method works, even if it returns empty for now
        let concepts = processor.get_related_concepts("test_entity", 2);
        assert_eq!(concepts.len(), 0); // Empty graph, so no related concepts
    }
}