//! Unified memory integration system
//!
//! This module provides a comprehensive memory integration system that unifies
//! working memory, SDR storage, and brain-enhanced knowledge graph into a
//! coherent memory architecture.

pub mod types;
pub mod hierarchy;
pub mod coordinator;
pub mod retrieval;
pub mod consolidation;
pub mod system;

// Re-export main types and systems
pub use types::*;
pub use hierarchy::{MemoryHierarchy, ItemStats, LevelStatistics};
pub use coordinator::MemoryCoordinator;
pub use retrieval::MemoryRetrieval;
pub use consolidation::MemoryConsolidation;
pub use system::UnifiedMemorySystem;