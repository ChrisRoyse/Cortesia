//! Knowledge Processing System
//! 
//! AI-powered knowledge processing pipeline using small language models for
//! advanced entity extraction, relationship mapping, and semantic analysis.

pub mod intelligent_processor;
pub mod entity_extractor;
pub mod relationship_mapper;
pub mod semantic_chunker;
pub mod context_analyzer;
pub mod types;

// Re-export public interface
pub use intelligent_processor::*;
pub use entity_extractor::*;
pub use relationship_mapper::*;
pub use semantic_chunker::*;
pub use context_analyzer::*;
pub use types::*;