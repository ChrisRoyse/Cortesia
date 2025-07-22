//! Phase 4 Learning System Integration
//!
//! This module provides the complete Phase 4 learning system that integrates
//! Hebbian learning, synaptic homeostasis, graph optimization, and adaptive
//! learning while maintaining seamless integration with Phase 3 cognitive systems.

pub mod types;
pub mod emergency;
pub mod coordination;
pub mod performance;
pub mod config;
pub mod system;

// Re-export main types and system
pub use types::*;
pub use emergency::{EmergencyProtocols, EmergencyType};
pub use coordination::LearningCoordinator;
pub use performance::Phase4PerformanceTracker;
pub use config::{Phase4Config, IntegrationDepth, PerformanceTargets, SafetyConstraints, ResourceLimits};
pub use system::Phase4LearningSystem;

// Re-export for backward compatibility
pub use system::Phase4LearningSystem as Phase4IntegratedLearningSystem;