//! Production Integration Framework
//! 
//! This module provides the production-ready integration framework that seamlessly
//! combines all AI components into a cohesive, scalable system designed to solve
//! RAG context fragmentation problems in production environments.

pub mod system_orchestrator;
pub mod config;
pub mod monitoring;
pub mod error_handling;
#[cfg(feature = "api")]
pub mod api_layer;
pub mod caching;
#[cfg(feature = "ai")]
pub mod ai_integration;

#[cfg(test)]
pub mod integration_tests;

// Re-export main public interfaces
pub use system_orchestrator::*;
pub use config::{ProductionConfig, Environment, ScalingConfig, MonitoringConfig, 
                ErrorHandlingConfig, LogLevel as ConfigLogLevel, DashboardConfig as ConfigDashboard};
pub use monitoring::{PerformanceMonitor, MonitoringError, CurrentPerformanceMetrics, 
                    PerformanceReport, TimeRange, MemoryUsage};
pub use error_handling::{SystemErrorHandler, ErrorHandlingError};
#[cfg(feature = "api")]
pub use api_layer::*;
pub use caching::*;
#[cfg(feature = "ai")]
pub use ai_integration::{EntityExtractionAdapter, SemanticChunkingAdapter, ReasoningAdapter};