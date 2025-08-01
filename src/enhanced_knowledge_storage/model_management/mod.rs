//! Model Management System
//! 
//! Handles resource-constrained loading, caching, and optimization of small language models
//! for enhanced knowledge processing. Implements LRU eviction and memory management.

pub mod resource_manager;
pub mod model_loader;
pub mod model_cache;
pub mod model_registry;

// Re-export public interface
pub use resource_manager::*;
pub use model_loader::*;
pub use model_cache::*;
pub use model_registry::*;