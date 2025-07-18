//! Adaptive Learning System
//!
//! This module provides adaptive learning capabilities that integrate with
//! cognitive systems to continuously improve performance based on feedback
//! and system monitoring.

pub mod types;
pub mod monitoring;
pub mod feedback;
pub mod scheduler;
pub mod system;

// Re-export main types and system
pub use types::*;
pub use monitoring::PerformanceMonitor;
pub use feedback::FeedbackAggregator;
pub use scheduler::LearningScheduler;
pub use system::{AdaptiveLearningSystem, AdaptiveLearningResult, EmergencyAdaptationResult, SystemStatus};

// Re-export for backward compatibility
pub use system::AdaptiveLearningSystem as AdaptiveLearningEngine;