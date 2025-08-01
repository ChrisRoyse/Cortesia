//! Enhanced Knowledge Storage System Test Suite
//! 
//! Comprehensive test coverage for the enhanced knowledge storage system
//! following London School TDD methodology with heavy mocking and behavior verification.

pub mod acceptance;
pub mod unit;
pub mod integration;
pub mod fixtures;
pub mod mocks;

// Re-export commonly used test utilities
pub use fixtures::*;
pub use mocks::*;