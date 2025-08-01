//! Enhanced Knowledge Storage System Test Suite
//! 
//! Main test file that orchestrates all enhanced knowledge storage system tests.
//! This file serves as the entry point for running the comprehensive test suite.

mod enhanced_knowledge_storage;
mod enhanced_knowledge_integration;

// Re-export test modules for easy access
pub use enhanced_knowledge_storage::*;
pub use enhanced_knowledge_integration::*;

// Integration with main test runner
#[cfg(test)]
mod test_runner {
    use super::*;
    
    /// Smoke test to verify test infrastructure is working
    #[tokio::test]
    async fn test_infrastructure_smoke_test() {
        // This should pass immediately to verify our test setup works
        assert!(true, "Test infrastructure is functional");
    }
    
    /// Test that our fixtures load correctly
    #[test]
    fn test_fixtures_load_correctly() {
        let samples = enhanced_knowledge_storage::fixtures::DocumentSamples::get_simple_samples();
        assert!(!samples.is_empty(), "Should have sample documents");
        
        let (name, content) = &samples[0];
        assert!(!name.is_empty(), "Sample should have a name");
        assert!(!content.is_empty(), "Sample should have content");
    }
    
    /// Test that mock framework initializes correctly
    #[test]
    fn test_mock_framework_initialization() {
        let mock_backend = enhanced_knowledge_storage::mocks::create_mock_model_backend_with_standard_models();
        // If this compiles and runs, our mock framework is working
        assert!(true, "Mock framework initialized successfully");
    }
}

// Test configuration and utilities
pub mod test_config {
    use std::time::Duration;
    
    /// Default timeout for async tests
    pub const DEFAULT_TEST_TIMEOUT: Duration = Duration::from_secs(30);
    
    /// Memory limits for testing
    pub const TEST_MEMORY_LIMIT_SMALL: u64 = 500_000_000;  // 500MB
    pub const TEST_MEMORY_LIMIT_MEDIUM: u64 = 1_000_000_000; // 1GB
    pub const TEST_MEMORY_LIMIT_LARGE: u64 = 2_000_000_000;  // 2GB
    
    /// Performance benchmarks
    pub const SIMPLE_TASK_MAX_TIME_MS: u64 = 100;
    pub const MEDIUM_TASK_MAX_TIME_MS: u64 = 500;
    pub const COMPLEX_TASK_MAX_TIME_MS: u64 = 2000;
}