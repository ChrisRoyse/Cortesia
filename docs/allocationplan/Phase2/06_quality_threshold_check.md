# Task 06: Quality Threshold Check

## Metadata
- **Micro-Phase**: 2.6
- **Duration**: 15-20 minutes
- **Dependencies**: Task 01 (quality_gate_config), Task 05 (validated_fact_structure)
- **Output**: `src/quality_integration/threshold_checker.rs`

## Description
Implement the ThresholdChecker that validates ValidatedFacts against QualityGateConfig thresholds. This is a critical component that determines if facts meet minimum quality requirements for allocation.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality_integration::{QualityGateConfig, ValidatedFact, FactContent, ConfidenceComponents};

    #[test]
    fn test_threshold_checker_creation() {
        let config = QualityGateConfig::default();
        let checker = ThresholdChecker::new(config);
        assert_eq!(checker.config.min_confidence_for_allocation, 0.8);
    }
    
    #[test]
    fn test_confidence_threshold_check() {
        let config = QualityGateConfig::default(); // min_confidence = 0.8
        let checker = ThresholdChecker::new(config);
        
        let fact_content = FactContent::new("High confidence fact");
        let high_confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let validated_fact = ValidatedFact::new(fact_content, high_confidence);
        
        let result = checker.check_confidence_threshold(&validated_fact);
        assert!(result.passed);
        assert!(result.score >= 0.8);
        
        let low_confidence = ConfidenceComponents::new(0.7, 0.6, 0.75);
        let low_fact = ValidatedFact::new(FactContent::new("Low confidence"), low_confidence);
        let low_result = checker.check_confidence_threshold(&low_fact);
        assert!(!low_result.passed);
    }
    
    #[test]
    fn test_entity_confidence_check() {
        let config = QualityGateConfig::default(); // min_entity_confidence = 0.75
        let checker = ThresholdChecker::new(config);
        
        let fact_content = FactContent::new("Entity test");
        let good_entity_confidence = ConfidenceComponents::new(0.8, 0.8, 0.7);
        let validated_fact = ValidatedFact::new(fact_content, good_entity_confidence);
        
        let result = checker.check_entity_threshold(&validated_fact);
        assert!(result.passed);
        
        let bad_entity_confidence = ConfidenceComponents::new(0.9, 0.6, 0.8);
        let bad_fact = ValidatedFact::new(FactContent::new("Bad entity"), bad_entity_confidence);
        let bad_result = checker.check_entity_threshold(&bad_fact);
        assert!(!bad_result.passed);
    }
    
    #[test]
    fn test_ambiguity_threshold_check() {
        let config = QualityGateConfig::default(); // max_ambiguity_count = 0
        let checker = ThresholdChecker::new(config);
        
        let fact_content = FactContent::new("Clear fact");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let validated_fact = ValidatedFact::new(fact_content, confidence);
        
        let result = checker.check_ambiguity_threshold(&validated_fact);
        assert!(result.passed);
        
        let mut ambiguous_fact = validated_fact.clone();
        ambiguous_fact.add_ambiguity("Unclear reference".to_string());
        let ambiguous_result = checker.check_ambiguity_threshold(&ambiguous_fact);
        assert!(!ambiguous_result.passed);
    }
    
    #[test]
    fn test_comprehensive_threshold_check() {
        let config = QualityGateConfig::default();
        let checker = ThresholdChecker::new(config);
        
        let fact_content = FactContent::new("Perfect fact");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let mut validated_fact = ValidatedFact::new(fact_content, confidence);
        validated_fact.mark_fully_validated();
        
        let result = checker.check_all_thresholds(&validated_fact);
        assert!(result.passed);
        assert_eq!(result.failed_checks.len(), 0);
        
        // Test with multiple failures
        let bad_confidence = ConfidenceComponents::new(0.5, 0.6, 0.7);
        let mut bad_fact = ValidatedFact::new(FactContent::new("Bad fact"), bad_confidence);
        bad_fact.add_ambiguity("Issue 1".to_string());
        bad_fact.add_ambiguity("Issue 2".to_string());
        
        let bad_result = checker.check_all_thresholds(&bad_fact);
        assert!(!bad_result.passed);
        assert!(bad_result.failed_checks.len() > 1);
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use crate::quality_integration::{QualityGateConfig, ValidatedFact};

/// Result of a threshold check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdCheckResult {
    /// Whether the check passed
    pub passed: bool,
    
    /// The actual score that was checked
    pub score: f32,
    
    /// The threshold that was applied
    pub threshold: f32,
    
    /// Type of check performed
    pub check_type: String,
    
    /// Optional failure reason
    pub failure_reason: Option<String>,
    
    /// Timestamp of the check
    pub checked_at: u64,
}

/// Comprehensive result of all threshold checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveThresholdResult {
    /// Whether all checks passed
    pub passed: bool,
    
    /// Individual check results
    pub check_results: Vec<ThresholdCheckResult>,
    
    /// List of failed check types
    pub failed_checks: Vec<String>,
    
    /// Overall quality score
    pub overall_score: f32,
    
    /// Timestamp of the comprehensive check
    pub checked_at: u64,
}

/// Validates facts against quality thresholds
#[derive(Debug, Clone)]
pub struct ThresholdChecker {
    /// Configuration defining the thresholds
    pub config: QualityGateConfig,
}

impl ThresholdChecker {
    /// Create a new threshold checker with configuration
    pub fn new(config: QualityGateConfig) -> Self {
        Self { config }
    }
    
    /// Check if fact meets overall confidence threshold
    pub fn check_confidence_threshold(&self, fact: &ValidatedFact) -> ThresholdCheckResult {
        let score = fact.confidence.overall_confidence();
        let threshold = self.config.min_confidence_for_allocation;
        let passed = score >= threshold;
        
        ThresholdCheckResult {
            passed,
            score,
            threshold,
            check_type: "overall_confidence".to_string(),
            failure_reason: if !passed {
                Some(format!("Overall confidence {} below threshold {}", score, threshold))
            } else {
                None
            },
            checked_at: current_timestamp(),
        }
    }
    
    /// Check if fact meets entity confidence threshold
    pub fn check_entity_threshold(&self, fact: &ValidatedFact) -> ThresholdCheckResult {
        let score = fact.confidence.entity_confidence;
        let threshold = self.config.min_entity_confidence;
        let passed = score >= threshold;
        
        ThresholdCheckResult {
            passed,
            score,
            threshold,
            check_type: "entity_confidence".to_string(),
            failure_reason: if !passed {
                Some(format!("Entity confidence {} below threshold {}", score, threshold))
            } else {
                None
            },
            checked_at: current_timestamp(),
        }
    }
    
    /// Check if fact meets ambiguity threshold
    pub fn check_ambiguity_threshold(&self, fact: &ValidatedFact) -> ThresholdCheckResult {
        let score = fact.ambiguity_count() as f32;
        let threshold = self.config.max_ambiguity_count as f32;
        let passed = score <= threshold;
        
        ThresholdCheckResult {
            passed,
            score,
            threshold,
            check_type: "ambiguity_count".to_string(),
            failure_reason: if !passed {
                Some(format!("Ambiguity count {} exceeds threshold {}", score, threshold))
            } else {
                None
            },
            checked_at: current_timestamp(),
        }
    }
    
    /// Check if fact meets minimum confidence in all components
    pub fn check_minimum_component_threshold(&self, fact: &ValidatedFact) -> ThresholdCheckResult {
        let score = fact.confidence.minimum_confidence();
        let threshold = self.config.min_entity_confidence; // Use entity threshold as minimum
        let passed = score >= threshold;
        
        ThresholdCheckResult {
            passed,
            score,
            threshold,
            check_type: "minimum_component_confidence".to_string(),
            failure_reason: if !passed {
                Some(format!("Minimum component confidence {} below threshold {}", score, threshold))
            } else {
                None
            },
            checked_at: current_timestamp(),
        }
    }
    
    /// Check if all validation stages are required and completed
    pub fn check_validation_completeness(&self, fact: &ValidatedFact) -> ThresholdCheckResult {
        let passed = if self.config.require_all_validations {
            fact.is_syntax_validated() && 
            fact.is_entity_validated() && 
            fact.is_semantic_validated()
        } else {
            true // Not required
        };
        
        let score = if passed { 1.0 } else { 0.0 };
        
        ThresholdCheckResult {
            passed,
            score,
            threshold: 1.0,
            check_type: "validation_completeness".to_string(),
            failure_reason: if !passed {
                Some("Not all required validation stages completed".to_string())
            } else {
                None
            },
            checked_at: current_timestamp(),
        }
    }
    
    /// Perform all threshold checks
    pub fn check_all_thresholds(&self, fact: &ValidatedFact) -> ComprehensiveThresholdResult {
        let mut check_results = Vec::new();
        let mut failed_checks = Vec::new();
        
        // Perform all individual checks
        let confidence_result = self.check_confidence_threshold(fact);
        if !confidence_result.passed {
            failed_checks.push(confidence_result.check_type.clone());
        }
        check_results.push(confidence_result);
        
        let entity_result = self.check_entity_threshold(fact);
        if !entity_result.passed {
            failed_checks.push(entity_result.check_type.clone());
        }
        check_results.push(entity_result);
        
        let ambiguity_result = self.check_ambiguity_threshold(fact);
        if !ambiguity_result.passed {
            failed_checks.push(ambiguity_result.check_type.clone());
        }
        check_results.push(ambiguity_result);
        
        let minimum_result = self.check_minimum_component_threshold(fact);
        if !minimum_result.passed {
            failed_checks.push(minimum_result.check_type.clone());
        }
        check_results.push(minimum_result);
        
        let validation_result = self.check_validation_completeness(fact);
        if !validation_result.passed {
            failed_checks.push(validation_result.check_type.clone());
        }
        check_results.push(validation_result);
        
        let passed = failed_checks.is_empty();
        let overall_score = fact.quality_score();
        
        ComprehensiveThresholdResult {
            passed,
            check_results,
            failed_checks,
            overall_score,
            checked_at: current_timestamp(),
        }
    }
    
    /// Update the configuration
    pub fn update_config(&mut self, config: QualityGateConfig) {
        self.config = config;
    }
    
    /// Get a summary of current thresholds
    pub fn get_threshold_summary(&self) -> std::collections::HashMap<String, f32> {
        let mut summary = std::collections::HashMap::new();
        summary.insert("min_confidence_for_allocation".to_string(), self.config.min_confidence_for_allocation);
        summary.insert("min_entity_confidence".to_string(), self.config.min_entity_confidence);
        summary.insert("max_ambiguity_count".to_string(), self.config.max_ambiguity_count as f32);
        summary.insert("require_all_validations".to_string(), if self.config.require_all_validations { 1.0 } else { 0.0 });
        summary
    }
}

/// Get current timestamp in seconds since epoch
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl Default for ThresholdChecker {
    fn default() -> Self {
        Self::new(QualityGateConfig::default())
    }
}
```

## Verification Steps
1. Create ThresholdChecker structure with QualityGateConfig
2. Implement individual threshold checking methods
3. Add comprehensive checking that combines all thresholds
4. Implement detailed result structures with failure reasons
5. Ensure all tests pass with proper error handling

## Success Criteria
- [ ] ThresholdChecker struct compiles without errors
- [ ] Individual threshold checks work correctly
- [ ] Comprehensive checking combines all validations
- [ ] Detailed failure reasons provided
- [ ] All tests pass with comprehensive coverage