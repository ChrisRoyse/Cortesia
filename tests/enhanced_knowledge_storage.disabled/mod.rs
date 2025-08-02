//! Enhanced Knowledge Storage System Test Suite
//! 
//! Comprehensive test coverage for the enhanced knowledge storage system
//! following London School TDD methodology with heavy mocking and behavior verification.

pub mod acceptance;
pub mod unit;
pub mod integration;
pub mod fixtures;
pub mod mocks;
pub mod mock_system_verification;
pub mod simple_mock_verification;
pub mod comprehensive_mock_validation_test;

// Re-export commonly used test utilities
pub use fixtures::*;
pub use mocks::*;