//! Phase 4 Cognitive Integration
//!
//! This module provides the interface between cognitive reasoning capabilities
//! and Phase 4 learning systems. It enables cognitive patterns to benefit from
//! continuous learning while maintaining backward compatibility with Phase 3.

pub mod types;
pub mod orchestrator;
pub mod interface;
pub mod performance;
pub mod adaptation;
pub mod system;

// Re-export main types and system
pub use types::*;
pub use orchestrator::LearningEnhancedOrchestrator;
pub use interface::CognitiveLearningInterface;
pub use performance::CognitivePerformanceTracker;
pub use adaptation::{AdaptationEngine, PatternAdaptationEngine};
pub use system::{Phase4CognitiveSystem, EnhancedCognitiveResult, LearningFeedback};

// Re-export for backward compatibility
pub use system::Phase4CognitiveSystem as Phase4IntegratedCognitiveSystem;