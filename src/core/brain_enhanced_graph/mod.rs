//! Brain-enhanced knowledge graph with neural processing capabilities
//!
//! This module provides a brain-enhanced knowledge graph that combines traditional
//! graph storage with brain-inspired processing patterns including neural activation,
//! concept formation, and adaptive learning.

pub mod brain_graph_types;
pub mod brain_graph_core;
pub mod brain_entity_manager;
pub mod brain_query_engine;
pub mod brain_relationship_manager;
pub mod brain_advanced_ops;
pub mod brain_analytics;

#[cfg(test)]
pub mod test_helpers;

// Re-export main types and functionality
pub use brain_graph_types::*;
pub use brain_graph_core::{BrainEnhancedKnowledgeGraph, BrainMemoryUsage};
pub use brain_entity_manager::EntityStatistics;
pub use brain_query_engine::QueryStatistics;
pub use brain_relationship_manager::{RelationshipStatistics, RelationshipPattern};
pub use brain_advanced_ops::{EntityRole, SplitCriteria, OptimizationResult};
pub use brain_analytics::{ConceptUsageStats, GraphPatternAnalysis};