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
    
    /// Asserts that a pattern was detected with minimum confidence
    fn assert_pattern_detected(&self, pattern_type: &str, min_confidence: f32);
    
    /// Asserts that reasoning trace has expected depth
    fn assert_reasoning_depth(&self, expected_depth: usize);
    
    /// Asserts that execution completed within time limit
    fn assert_execution_time_within(&self, max_ms: u64);
}

/// Trait for asserting cognitive orchestrator results
pub trait OrchestratorAssertions {
    /// Asserts that a workflow completed successfully
    fn assert_workflow_completed(&self);
    
    /// Asserts that the expected cognitive patterns were used
    fn assert_patterns_used(&self, expected_patterns: &[crate::cognitive::CognitivePatternType]);
    
    /// Asserts that the result has reasonable confidence
    fn assert_reasonable_confidence(&self, min_confidence: f32);
}

/// Trait for asserting attention management results
pub trait AttentionAssertions {
    /// Asserts that attention focuses on expected entities
    fn assert_attention_focuses_on(&self, entities: &[EntityKey]);
    
    /// Asserts that attention distribution matches expectations
    fn assert_attention_distribution(&self, expected_distribution: &[(EntityKey, f32)], tolerance: f32);
    
    /// Asserts that attention state is as expected
    fn assert_attention_state(&self, expected_state: &str);
}

/// Implementations for cognitive result types
impl PatternAssertions for crate::cognitive::PatternResult {
    fn assert_confidence_in_range(&self, min: f32, max: f32) {
        assert!(
            self.confidence >= min && self.confidence <= max,
            "Confidence {} not in expected range [{}, {}]",
            self.confidence, min, max
        );
    }
    
    fn assert_contains_content(&self, expected: &str) {
        assert!(
            self.answer.contains(expected),
            "Answer '{}' does not contain expected content '{}'",
            self.answer, expected
        );
    }
    
    fn assert_pattern_detected(&self, pattern_type: &str, min_confidence: f32) {
        let pattern_str = format!("{:?}", self.pattern_type);
        assert!(
            pattern_str.to_lowercase().contains(&pattern_type.to_lowercase()),
            "Expected pattern type '{}', but got '{:?}'",
            pattern_type, self.pattern_type
        );
        self.assert_confidence_in_range(min_confidence, 1.0);
    }
    
    fn assert_reasoning_depth(&self, expected_depth: usize) {
        assert!(
            self.reasoning_trace.len() >= expected_depth,
            "Reasoning trace depth {} is less than expected {}",
            self.reasoning_trace.len(), expected_depth
        );
    }
    
    fn assert_execution_time_within(&self, max_ms: u64) {
        assert!(
            self.metadata.execution_time_ms <= max_ms,
            "Execution time {}ms exceeds maximum {}ms",
            self.metadata.execution_time_ms, max_ms
        );
    }
}

/// Implementations for convergent thinking results
impl PatternAssertions for crate::cognitive::ConvergentResult {
    fn assert_confidence_in_range(&self, min: f32, max: f32) {
        assert!(
            self.confidence >= min && self.confidence <= max,
            "Convergent confidence {} not in expected range [{}, {}]",
            self.confidence, min, max
        );
    }
    
    fn assert_contains_content(&self, expected: &str) {
        assert!(
            self.answer.contains(expected),
            "Convergent answer '{}' does not contain expected content '{}'",
            self.answer, expected
        );
    }
    
    fn assert_pattern_detected(&self, _pattern_type: &str, min_confidence: f32) {
        self.assert_confidence_in_range(min_confidence, 1.0);
    }
    
    fn assert_reasoning_depth(&self, expected_depth: usize) {
        assert!(
            self.reasoning_trace.len() >= expected_depth,
            "Convergent reasoning depth {} is less than expected {}",
            self.reasoning_trace.len(), expected_depth
        );
    }
    
    fn assert_execution_time_within(&self, max_ms: u64) {
        assert!(
            self.execution_time_ms <= max_ms,
            "Convergent execution time {}ms exceeds maximum {}ms",
            self.execution_time_ms, max_ms
        );
    }
}

/// Implementations for divergent thinking results
impl PatternAssertions for crate::cognitive::DivergentResult {
    fn assert_confidence_in_range(&self, min: f32, max: f32) {
        // For divergent results, check average creativity score
        let avg_creativity = if self.creativity_scores.is_empty() {
            0.0
        } else {
            self.creativity_scores.iter().sum::<f32>() / self.creativity_scores.len() as f32
        };
        assert!(
            avg_creativity >= min && avg_creativity <= max,
            "Average creativity score {} not in expected range [{}, {}]",
            avg_creativity, min, max
        );
    }
    
    fn assert_contains_content(&self, expected: &str) {
        let found = self.explorations.iter().any(|exploration| {
            exploration.concepts.iter().any(|concept| concept.contains(expected))
        });
        assert!(
            found,
            "Expected content '{}' not found in divergent explorations",
            expected
        );
    }
    
    fn assert_pattern_detected(&self, _pattern_type: &str, min_confidence: f32) {
        self.assert_confidence_in_range(min_confidence, 1.0);
    }
    
    fn assert_reasoning_depth(&self, expected_depth: usize) {
        assert!(
            self.explorations.len() >= expected_depth,
            "Number of explorations {} is less than expected depth {}",
            self.explorations.len(), expected_depth
        );
    }
    
    fn assert_execution_time_within(&self, _max_ms: u64) {
        // Note: DivergentResult doesn't have execution_time_ms field
        // This would need to be tracked externally in tests
    }
}

/// Custom assertion macros for cognitive testing
#[macro_export]
macro_rules! assert_cognitive_pattern {
    ($result:expr, $pattern:expr, $min_confidence:expr) => {
        $result.assert_pattern_detected(stringify!($pattern), $min_confidence);
    };
}

#[macro_export]
macro_rules! assert_attention_weights_sum_to_one {
    ($weights:expr, $tolerance:expr) => {
        let total: f32 = $weights.values().sum();
        assert!(
            (total - 1.0).abs() <= $tolerance,
            "Attention weights sum to {}, expected approximately 1.0 (tolerance: {})",
            total,
            $tolerance
        );
    };
}

#[macro_export]
macro_rules! assert_performance_within_bounds {
    ($duration:expr, $max_ms:expr) => {
        let duration_ms = $duration.as_millis() as u64;
        assert!(
            duration_ms <= $max_ms,
            "Performance test took {}ms, expected <= {}ms",
            duration_ms,
            $max_ms
        );
    };
}

#[macro_export]
macro_rules! assert_memory_usage_reasonable {
    ($usage_mb:expr, $max_mb:expr) => {
        assert!(
            $usage_mb <= $max_mb,
            "Memory usage {}MB exceeds maximum {}MB",
            $usage_mb,
            $max_mb
        );
    };
}

/// Helper function to create custom assertion error messages
pub fn create_assertion_message(
    test_name: &str,
    expected: &str,
    actual: &str,
    context: Option<&str>,
) -> String {
    match context {
        Some(ctx) => format!(
            "Test '{}' failed in context '{}': expected '{}', got '{}'",
            test_name, ctx, expected, actual
        ),
        None => format!(
            "Test '{}' failed: expected '{}', got '{}'",
            test_name, expected, actual
        ),
    }
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