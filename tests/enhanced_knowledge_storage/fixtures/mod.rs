//! Test Fixtures for Enhanced Knowledge Storage System
//! 
//! Provides realistic test data and scenarios for comprehensive testing.

pub mod sample_documents;
pub mod expected_extractions;
pub mod performance_benchmarks;
pub mod model_configurations;

// Re-export commonly used fixtures
pub use sample_documents::*;
pub use expected_extractions::*;
pub use performance_benchmarks::*;
pub use model_configurations::*;