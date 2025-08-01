//! AI Components
//! 
//! Production-ready AI/ML components that replace mock implementations with 
//! real transformer models, embeddings, and intelligent processing capabilities.

pub mod real_entity_extractor;
pub mod real_semantic_chunker;  
pub mod real_reasoning_engine;
pub mod ai_model_backend;
pub mod caching_layer;
pub mod performance_monitor;
pub mod types;

// Re-export commonly used types and traits
pub use types::*;
pub use real_entity_extractor::*;
pub use real_semantic_chunker::*;
pub use real_reasoning_engine::*;
pub use ai_model_backend::*;
pub use caching_layer::*;
pub use performance_monitor::*;