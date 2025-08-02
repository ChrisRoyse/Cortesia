//! Enhanced Knowledge Storage Benchmarks Module
//!
//! Organizes benchmark suites for different components of the enhanced knowledge storage system.

pub mod model_loading;
pub mod knowledge_processing;
pub mod retrieval_performance;
pub mod baseline_comparison;

// Re-export benchmark functions
pub use model_loading::*;
pub use knowledge_processing::*;
pub use retrieval_performance::*;
pub use baseline_comparison::*;