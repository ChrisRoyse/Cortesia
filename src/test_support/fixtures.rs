//! Common test fixtures for LLMKG tests

use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use std::sync::Arc;

/// Creates a test graph with default configuration
pub fn create_test_graph() -> Arc<BrainEnhancedKnowledgeGraph> {
    Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().expect("Failed to create test graph"))
}

/// Creates a test graph with custom embedding dimension
pub fn create_test_graph_with_dim(embedding_dim: usize) -> Arc<BrainEnhancedKnowledgeGraph> {
    Arc::new(BrainEnhancedKnowledgeGraph::new(embedding_dim).expect("Failed to create test graph"))
}