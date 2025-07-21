/// Tests for the cognitive module
/// 
/// This module contains comprehensive tests for the cognitive system,
/// including unit tests for individual components and integration tests
/// for testing public APIs and component interactions.

// Integration test modules (testing public APIs only)
pub mod attention_integration_tests;
pub mod patterns_integration_tests; 
pub mod orchestrator_integration_tests;

// Unit test modules for individual components
pub mod test_abstract_thinking;
pub mod test_adaptive;
pub mod test_attention_manager;
pub mod test_convergent;
pub mod test_critical_thinking;
pub mod test_divergent;
pub mod test_lateral;
pub mod test_neural_bridge_finder;
pub mod test_neural_query;
pub mod test_orchestrator;
pub mod test_phase3_integration;
pub mod test_systems_thinking;
pub mod test_utils;
pub mod test_working_memory;
pub mod test_inhibitory_logic;
pub mod test_unified_memory;

// Property-based test modules
pub mod property_tests;

// Performance and benchmark tests
pub mod performance_tests;