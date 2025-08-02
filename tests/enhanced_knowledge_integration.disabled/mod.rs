// Enhanced Knowledge Storage Integration Tests
//
// This module provides integration tests that validate the enhanced knowledge storage system
// works end-to-end with actual public APIs and working components.

pub mod basic_functionality_tests;
pub mod system_integration_tests;
pub mod end_to_end_document_processing;
pub mod multi_hop_reasoning_integration;
pub mod performance_load_tests;

/// Create test content for integration testing
pub fn create_test_knowledge_content() -> &'static str {
    r#"
    Albert Einstein (1879-1955) was a theoretical physicist who developed 
    the theory of relativity. His famous equation E=mc² revolutionized our 
    understanding of energy and mass. Einstein received the Nobel Prize in 
    Physics in 1921 for his work on the photoelectric effect.
    
    His theories led to numerous technological advances including GPS satellites,
    which must account for relativistic effects to maintain accuracy.
    "#
}

/// Setup basic test environment
pub async fn setup_basic_test_environment() {
    // Initialize logging for tests
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Debug)
        .is_test(true)
        .try_init();
}