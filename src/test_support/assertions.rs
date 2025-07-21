//! Custom assertions for cognitive module tests

use crate::core::types::EntityKey;
use crate::core::entity_compat; // Import for EntityKey::from_hash
use std::collections::HashMap;

/// Custom assertions for cognitive tests
pub trait CognitiveAssertions {
    /// Asserts that attention is focused on the given target with at least the minimum weight
    fn assert_attention_focused_on(&self, target: &EntityKey, min_weight: f32);
    
    /// Asserts that the total weight is approximately the expected value
    fn assert_total_weight_approximately(&self, expected: f32, tolerance: f32);
    
    /// Asserts that weights are distributed evenly with given tolerance
    fn assert_weights_distributed_evenly(&self, tolerance: f32);
}

impl CognitiveAssertions for HashMap<EntityKey, f32> {
    fn assert_attention_focused_on(&self, target: &EntityKey, min_weight: f32) {
        let weight = self.get(target).copied().unwrap_or(0.0);
        assert!(
            weight >= min_weight,
            "Expected attention weight for {:?} to be at least {}, but was {}",
            target, min_weight, weight
        );
    }
    
    fn assert_total_weight_approximately(&self, expected: f32, tolerance: f32) {
        let total: f32 = self.values().sum();
        assert!(
            (total - expected).abs() <= tolerance,
            "Expected total weight to be approximately {} (±{}), but was {}",
            expected, tolerance, total
        );
    }
    
    fn assert_weights_distributed_evenly(&self, tolerance: f32) {
        if self.is_empty() {
            return;
        }
        
        let expected = 1.0 / self.len() as f32;
        for (key, &weight) in self {
            assert!(
                (weight - expected).abs() <= tolerance,
                "Expected weight for {:?} to be approximately {} (±{}), but was {}",
                key, expected, tolerance, weight
            );
        }
    }
}

/// Assertions for verifying pattern results
pub trait PatternAssertions {
    /// Asserts that confidence is within expected range
    fn assert_confidence_in_range(&self, min: f32, max: f32);
    
    /// Asserts that the result contains expected content
    fn assert_contains_content(&self, expected: &str);
}

/// Helper macro for asserting async results
#[macro_export]
macro_rules! assert_ok {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(e) => panic!("Expected Ok, got Err: {:?}", e),
        }
    };
}

/// Helper macro for asserting specific error types
#[macro_export]
macro_rules! assert_err_type {
    ($expr:expr, $err_type:path) => {
        match $expr {
            Err(e) if matches!(e, $err_type(_)) => e,
            Ok(_) => panic!("Expected Err of type {}, got Ok", stringify!($err_type)),
            Err(e) => panic!("Expected Err of type {}, got different error: {:?}", stringify!($err_type), e),
        }
    };
}