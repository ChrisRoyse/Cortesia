//! Phase 4: Self-Organization & Learning Systems
//! 
//! This module implements adaptive learning mechanisms that enhance the existing
//! cognitive architecture with biological learning principles.

pub mod hebbian;
pub mod homeostasis;
pub mod optimization_agent;
pub mod adaptive_learning;
pub mod phase4_integration;
pub mod types;
pub mod neural_pattern_detection;
pub mod parameter_tuning;
pub mod meta_learning;

pub use hebbian::*;
pub use homeostasis::*;
pub use optimization_agent::*;
pub use adaptive_learning::*;
pub use phase4_integration::*;
pub use types::*;
pub use neural_pattern_detection::*;
pub use parameter_tuning::*;
pub use meta_learning::*;