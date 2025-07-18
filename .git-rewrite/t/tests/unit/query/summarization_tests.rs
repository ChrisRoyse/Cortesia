//! Summarization Unit Tests

use crate::unit::*;
use crate::unit::test_utils::*;
use crate::query::summarization::*;

#[cfg(test)]
mod summarization_tests {
    use super::*;

    #[test]
    fn test_graph_summarization() {
        let graph = create_test_graph(50, 100);
        let summarizer = GraphSummarizer::new();
        
        let summary = summarizer.summarize(&graph, 10).unwrap();
        
        assert!(summary.entities.len() <= 10);
        assert!(summary.relationships.len() <= summary.entities.len() * 2);
        assert!(summary.coverage_score > 0.0);
    }
}