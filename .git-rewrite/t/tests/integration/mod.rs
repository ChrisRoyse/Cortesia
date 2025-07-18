// Integration Test Suite Module
// Main entry point for all integration tests

pub mod test_infrastructure;
pub mod graph_storage_integration;
pub mod embedding_graph_integration;
pub mod wasm_integration;
pub mod mcp_integration;
pub mod performance_integration;

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::{HashMap, HashSet};

// Re-export common test utilities
pub use test_infrastructure::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_suite_initialization() {
        let env = IntegrationTestEnvironment::new("integration_init");
        assert!(env.is_initialized());
        println!("Integration test suite initialized successfully");
    }
}