//! Knowledge graph implementation with modular architecture
//!
//! This module provides a comprehensive knowledge graph implementation with
//! support for entity management, relationship tracking, similarity search,
//! path finding, and advanced query capabilities.

pub mod graph_core;
pub mod entity_operations;
pub mod relationship_operations;
pub mod path_finding;
pub mod similarity_search;
pub mod query_system;
pub mod compatibility;

// Re-export main types and functionality
pub use graph_core::{
    KnowledgeGraph, 
    MemoryUsage, 
    MemoryBreakdown,
    MAX_INSERTION_TIME,
    MAX_QUERY_TIME,
    MAX_SIMILARITY_SEARCH_TIME,
};

pub use entity_operations::EntityStats;
pub use relationship_operations::RelationshipStats;
pub use path_finding::PathStats;
pub use similarity_search::SimilarityStats;
pub use query_system::{QueryStats, QueryExplanation, EntityExplanation, AdvancedQueryResult};