//! Production Integration Framework
//! 
//! This module provides the production-ready integration framework that seamlessly
//! combines all AI components into a cohesive, scalable system designed to solve
//! RAG context fragmentation problems in production environments.

pub mod system_orchestrator;
pub mod config;
pub mod monitoring;
pub mod error_handling;
pub mod api_layer;
pub mod caching;
pub mod deployment;

// Re-export main public interfaces
pub use system_orchestrator::*;
pub use config::*;
pub use monitoring::*;
pub use error_handling::*;
pub use api_layer::*;
pub use caching::*;