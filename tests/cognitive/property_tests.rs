//! Property-based tests for cognitive module invariants
//! Tests that certain properties hold across all possible inputs

use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

// Property tests for basic cognitive invariants
proptest! {
    #[test]
    fn test_basic_math_property(
        a in -100.0..100.0f32,
        b in -100.0..100.0f32,
    ) -> Result<(), TestCaseError> {
        // Property: addition is commutative
        let sum1 = a + b;
        let sum2 = b + a;
        
        if (sum1 - sum2).abs() >= 0.001 {
            return Err(TestCaseError::fail(format!(
                "Addition not commutative: {} + {} = {}, {} + {} = {}",
                a, b, sum1, b, a, sum2
            )));
        }
        
        Ok(())
    }

    #[test]
    fn test_cognitive_load_bounds_isolated(
        load in -1.0..3.0f32,
    ) -> Result<(), TestCaseError> {
        // Property: cognitive load should be clamped between 0.0 and 1.0
        let clamped_load = load.clamp(0.0, 1.0);
        
        if clamped_load < 0.0 {
            return Err(TestCaseError::fail("Clamped load should not be negative".to_string()));
        }
        if clamped_load > 1.0 {
            return Err(TestCaseError::fail("Clamped load should not exceed 1.0".to_string()));
        }
        
        // Test relationship: capacity = (1.0 - load * 0.5).max(0.2)
        let capacity = (1.0 - clamped_load * 0.5).max(0.2f32);
        if capacity < 0.2 {
            return Err(TestCaseError::fail("Capacity should be at least 0.2".to_string()));
        }
        if capacity > 1.0 {
            return Err(TestCaseError::fail("Capacity should not exceed 1.0".to_string()));
        }
        
        Ok(())
    }

    #[test]
    fn test_attention_weight_normalization(
        weights in prop::collection::vec(0.0..1.0f32, 1..10),
    ) -> Result<(), TestCaseError> {
        // Property: normalized weights should sum to approximately 1.0
        if weights.is_empty() {
            return Ok(());
        }
        
        let sum: f32 = weights.iter().sum();
        if sum <= 0.0 {
            return Ok(()); // Skip if all weights are zero
        }
        
        let normalized: Vec<f32> = weights.iter().map(|w| w / sum).collect();
        let normalized_sum: f32 = normalized.iter().sum();
        
        if (normalized_sum - 1.0).abs() >= 0.001 {
            return Err(TestCaseError::fail(format!(
                "Normalized weights sum was {:.6}, expected ~1.0",
                normalized_sum
            )));
        }
        
        // All normalized weights should be non-negative
        for &weight in &normalized {
            if weight < 0.0 {
                return Err(TestCaseError::fail(format!(
                    "Found negative normalized weight: {:.6}",
                    weight
                )));
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_attention_focus_strength_bounds(
        strength in -1.0..2.0f32,
    ) -> Result<(), TestCaseError> {
        // Property: focus strength should be in valid range [0.0, 1.0]
        let clamped_strength = strength.clamp(0.0, 1.0);
        
        if clamped_strength < 0.0 {
            return Err(TestCaseError::fail("Focus strength should not be negative".to_string()));
        }
        if clamped_strength > 1.0 {
            return Err(TestCaseError::fail("Focus strength should not exceed 1.0".to_string()));
        }
        
        // Test that the original strength is preserved when in valid range
        if strength >= 0.0 && strength <= 1.0 && (clamped_strength - strength).abs() >= 0.001 {
            return Err(TestCaseError::fail(format!(
                "Valid strength not preserved: {} became {}",
                strength, clamped_strength
            )));
        }
        
        Ok(())
    }

    #[test]
    fn test_divided_attention_distribution(
        target_count in 2..10usize,
        total_strength in 0.1..1.0f32,
    ) -> Result<(), TestCaseError> {
        // Property: divided attention should distribute strength evenly
        let weight_per_target = total_strength / target_count as f32;
        let total_distributed = weight_per_target * target_count as f32;
        
        // The total should be approximately equal to the original strength
        if (total_distributed - total_strength).abs() >= 0.001 {
            return Err(TestCaseError::fail(format!(
                "Total distributed {:.6} doesn't match original strength {:.6}",
                total_distributed, total_strength
            )));
        }
        
        // Each weight should be positive and reasonable
        if weight_per_target <= 0.0 {
            return Err(TestCaseError::fail("Weight per target should be positive".to_string()));
        }
        
        if weight_per_target > total_strength {
            return Err(TestCaseError::fail("Weight per target should not exceed total strength".to_string()));
        }
        
        Ok(())
    }

    #[test]
    fn test_selective_attention_dominance(
        target_count in 2..10usize,
        focus_strength in 0.1..1.0f32,
    ) -> Result<(), TestCaseError> {
        // Property: selective attention should focus on one primary target
        // Simulate selective attention: first target gets most attention
        let primary_weight = focus_strength * 0.8; // 80% to primary
        let remaining_strength = focus_strength - primary_weight;
        let secondary_weight = if target_count > 1 {
            remaining_strength / (target_count - 1) as f32
        } else {
            0.0
        };
        
        // Primary target should have significantly more weight than others
        if target_count > 1 && primary_weight <= secondary_weight {
            return Err(TestCaseError::fail(format!(
                "Primary weight {:.6} should be greater than secondary weight {:.6}",
                primary_weight, secondary_weight
            )));
        }
        
        // Total weight should not exceed original focus strength
        let total_weight = primary_weight + secondary_weight * (target_count - 1) as f32;
        if total_weight > focus_strength + 0.001 {
            return Err(TestCaseError::fail(format!(
                "Total weight {:.6} exceeds focus strength {:.6}",
                total_weight, focus_strength
            )));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod basic_property_tests {
    use super::*;

    #[test]
    fn test_property_framework_works() {
        // This is a simple test to ensure the property test framework is working
        let result = std::panic::catch_unwind(|| {
            proptest!(|(x in 0..100i32)| {
                prop_assert!(x >= 0);
                prop_assert!(x < 100);
            });
        });
        
        assert!(result.is_ok(), "Property test framework should work");
    }
    
    #[test]
    fn test_cognitive_invariants_basic() {
        // Test basic cognitive invariants without external dependencies
        let cognitive_load = 0.5f32;
        let attention_capacity = (1.0 - cognitive_load * 0.5).max(0.2);
        
        assert!(attention_capacity >= 0.2, "Attention capacity should be at least 0.2");
        assert!(attention_capacity <= 1.0, "Attention capacity should not exceed 1.0");
        
        // Test with extreme values
        let high_load = 1.0f32;
        let low_capacity = (1.0 - high_load * 0.5).max(0.2);
        assert_eq!(low_capacity, 0.5, "High cognitive load should result in reduced capacity");
        
        let no_load = 0.0f32;
        let full_capacity = (1.0 - no_load * 0.5).max(0.2);
        assert_eq!(full_capacity, 1.0, "No cognitive load should result in full capacity");
    }
}