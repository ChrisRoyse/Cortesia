//! Hierarchical Storage System
//! 
//! Implements layered knowledge storage that organizes information in
//! hierarchical layers: Document → Section → Paragraph → Sentence → Entity.

pub mod knowledge_layers;
pub mod semantic_links;
pub mod hierarchical_index;
pub mod storage_engine;
pub mod types;

// Re-export public interface
pub use knowledge_layers::*;
pub use semantic_links::*;
pub use hierarchical_index::*;
pub use storage_engine::*;
pub use types::*;