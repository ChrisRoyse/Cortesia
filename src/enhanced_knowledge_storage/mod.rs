//! Enhanced Knowledge Storage System
//! 
//! This module implements the enhanced knowledge storage system that solves
//! RAG context fragmentation problems through hierarchical knowledge organization,
//! intelligent processing using small language models, and semantic retrieval.

pub mod model_management;
pub mod knowledge_processing;
pub mod hierarchical_storage;
pub mod retrieval_system;
pub mod types;
pub mod logging;
pub mod production;
#[cfg(feature = "ai")]
pub mod ai_components;

// Re-export commonly used types and traits
pub use types::*;
pub use model_management::*;
pub use production::caching::*;

// Integration tests
#[cfg(test)]
mod simple_integration_test;
#[cfg(test)]
mod demonstration;
// #[cfg(test)]
// mod final_test; // Removed: Tests internal implementation details