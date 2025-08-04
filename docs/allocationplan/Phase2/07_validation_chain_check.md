# Task 07: Validation Chain Check

## Metadata
- **Micro-Phase**: 2.7
- **Duration**: 15-20 minutes
- **Dependencies**: Task 05 (validated_fact_structure)
- **Output**: `src/quality_integration/validation_chain.rs`

## Description
Create the ValidationChainChecker that verifies all validation stages have been completed in the correct order. This ensures the quality gate pipeline maintains integrity and completeness before allocation.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality_integration::{ValidatedFact, FactContent, ConfidenceComponents, ValidationStatus};

    #[test]
    fn test_validation_chain_checker_creation() {
        let checker = ValidationChainChecker::new();
        assert_eq!(checker.required_stages.len(), 3);
        assert!(checker.required_stages.contains(&ValidationStage::Syntax));
        assert!(checker.required_stages.contains(&ValidationStage::Entity));
        assert!(checker.required_stages.contains(&ValidationStage::Semantic));
    }
    
    #[test]
    fn test_stage_completion_check() {
        let checker = ValidationChainChecker::new();
        
        let fact_content = FactContent::new("Test validation stages");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let mut validated_fact = ValidatedFact::new(fact_content, confidence);
        
        // Test initial state
        let result = checker.check_stage_completion(&validated_fact);
        assert!(!result.all_stages_complete);
        assert_eq!(result.completed_stages.len(), 0);
        assert_eq!(result.missing_stages.len(), 3);
        
        // Test progressive completion
        validated_fact.mark_syntax_validated();
        let result = checker.check_stage_completion(&validated_fact);
        assert_eq!(result.completed_stages.len(), 1);
        assert_eq!(result.missing_stages.len(), 2);
        
        validated_fact.mark_entity_validated();
        validated_fact.mark_semantic_validated();
        validated_fact.mark_fully_validated();
        
        let result = checker.check_stage_completion(&validated_fact);
        assert!(result.all_stages_complete);
        assert_eq!(result.completed_stages.len(), 3);
        assert_eq!(result.missing_stages.len(), 0);
    }
    
    #[test]
    fn test_stage_order_validation() {
        let checker = ValidationChainChecker::new();
        
        let fact_content = FactContent::new("Test stage order");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let mut validated_fact = ValidatedFact::new(fact_content, confidence);
        
        // Complete stages in correct order
        validated_fact.mark_syntax_validated();
        validated_fact.mark_entity_validated();
        validated_fact.mark_semantic_validated();
        
        let result = checker.check_stage_order(&validated_fact);
        assert!(result.correct_order);
        assert!(result.order_violations.is_empty());
        
        // Test with violated order (simulate by checking metadata directly)
        let mut bad_fact = ValidatedFact::new(FactContent::new("Bad order"), confidence);
        bad_fact.validation_metadata.completed_stages.insert("semantic".to_string());
        bad_fact.validation_metadata.completed_stages.insert("entity".to_string());
        // Missing syntax stage
        
        let bad_result = checker.check_stage_order(&bad_fact);
        assert!(!bad_result.correct_order);
        assert!(!bad_result.order_violations.is_empty());
    }
    
    #[test]
    fn test_timing_validation() {
        let checker = ValidationChainChecker::new();
        
        let fact_content = FactContent::new("Test timing");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let mut validated_fact = ValidatedFact::new(fact_content, confidence);
        
        // Add timing information
        validated_fact.record_stage_duration("syntax", 100);
        validated_fact.record_stage_duration("entity", 150);
        validated_fact.record_stage_duration("semantic", 200);
        
        let result = checker.check_timing_requirements(&validated_fact);
        assert!(result.within_limits);
        assert_eq!(result.total_processing_time, 450);
        
        // Test with excessive timing
        validated_fact.record_stage_duration("syntax", 30000); // 30 seconds
        let slow_result = checker.check_timing_requirements(&validated_fact);
        assert!(!slow_result.within_limits);
        assert!(slow_result.timing_violations.len() > 0);
    }
    
    #[test]
    fn test_comprehensive_chain_validation() {
        let checker = ValidationChainChecker::new();
        
        let fact_content = FactContent::new("Complete validation test");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let mut validated_fact = ValidatedFact::new(fact_content, confidence);
        
        // Complete all stages properly
        validated_fact.mark_syntax_validated();
        validated_fact.mark_entity_validated();
        validated_fact.mark_semantic_validated();
        validated_fact.mark_fully_validated();
        
        validated_fact.record_stage_duration("syntax", 50);
        validated_fact.record_stage_duration("entity", 75);
        validated_fact.record_stage_duration("semantic", 100);
        
        let result = checker.validate_complete_chain(&validated_fact);
        assert!(result.chain_valid);
        assert!(result.stage_result.all_stages_complete);
        assert!(result.order_result.correct_order);
        assert!(result.timing_result.within_limits);
        assert!(result.validation_issues.is_empty());
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, HashMap};
use crate::quality_integration::ValidatedFact;

/// Represents different validation stages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ValidationStage {
    Syntax,
    Entity,
    Semantic,
    Quality,
    Final,
}

impl ValidationStage {
    /// Get the stage name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            ValidationStage::Syntax => "syntax",
            ValidationStage::Entity => "entity",
            ValidationStage::Semantic => "semantic",
            ValidationStage::Quality => "quality",
            ValidationStage::Final => "final",
        }
    }
    
    /// Get stage order number
    pub fn order(&self) -> usize {
        match self {
            ValidationStage::Syntax => 1,
            ValidationStage::Entity => 2,
            ValidationStage::Semantic => 3,
            ValidationStage::Quality => 4,
            ValidationStage::Final => 5,
        }
    }
}

/// Result of stage completion check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageCompletionResult {
    /// Whether all required stages are complete
    pub all_stages_complete: bool,
    
    /// List of completed stages
    pub completed_stages: Vec<ValidationStage>,
    
    /// List of missing stages
    pub missing_stages: Vec<ValidationStage>,
    
    /// Completion percentage (0.0-1.0)
    pub completion_percentage: f32,
    
    /// Timestamp of the check
    pub checked_at: u64,
}

/// Result of stage order validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageOrderResult {
    /// Whether stages were completed in correct order
    pub correct_order: bool,
    
    /// List of order violations
    pub order_violations: Vec<String>,
    
    /// Expected vs actual order
    pub expected_order: Vec<ValidationStage>,
    pub actual_order: Vec<ValidationStage>,
    
    /// Timestamp of the check
    pub checked_at: u64,
}

/// Result of timing requirements check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingResult {
    /// Whether all stages completed within time limits
    pub within_limits: bool,
    
    /// Total processing time in milliseconds
    pub total_processing_time: u64,
    
    /// Individual stage timings
    pub stage_timings: HashMap<String, u64>,
    
    /// List of timing violations
    pub timing_violations: Vec<String>,
    
    /// Timestamp of the check
    pub checked_at: u64,
}

/// Comprehensive validation chain result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationChainResult {
    /// Whether the entire validation chain is valid
    pub chain_valid: bool,
    
    /// Stage completion results
    pub stage_result: StageCompletionResult,
    
    /// Stage order results
    pub order_result: StageOrderResult,
    
    /// Timing results
    pub timing_result: TimingResult,
    
    /// Overall validation issues
    pub validation_issues: Vec<String>,
    
    /// Overall quality score for the chain
    pub chain_quality_score: f32,
    
    /// Timestamp of the comprehensive check
    pub checked_at: u64,
}

/// Validates the completion and order of validation stages
#[derive(Debug, Clone)]
pub struct ValidationChainChecker {
    /// Required validation stages
    pub required_stages: HashSet<ValidationStage>,
    
    /// Maximum allowed time per stage (milliseconds)
    pub max_stage_time: HashMap<ValidationStage, u64>,
    
    /// Maximum total processing time (milliseconds)
    pub max_total_time: u64,
}

impl ValidationChainChecker {
    /// Create a new validation chain checker with default requirements
    pub fn new() -> Self {
        let mut required_stages = HashSet::new();
        required_stages.insert(ValidationStage::Syntax);
        required_stages.insert(ValidationStage::Entity);
        required_stages.insert(ValidationStage::Semantic);
        
        let mut max_stage_time = HashMap::new();
        max_stage_time.insert(ValidationStage::Syntax, 5000);    // 5 seconds
        max_stage_time.insert(ValidationStage::Entity, 10000);   // 10 seconds
        max_stage_time.insert(ValidationStage::Semantic, 15000); // 15 seconds
        max_stage_time.insert(ValidationStage::Quality, 5000);   // 5 seconds
        max_stage_time.insert(ValidationStage::Final, 2000);     // 2 seconds
        
        Self {
            required_stages,
            max_stage_time,
            max_total_time: 30000, // 30 seconds total
        }
    }
    
    /// Check if all required stages are completed
    pub fn check_stage_completion(&self, fact: &ValidatedFact) -> StageCompletionResult {
        let mut completed_stages = Vec::new();
        let mut missing_stages = Vec::new();
        
        for &stage in &self.required_stages {
            if fact.validation_metadata.completed_stages.contains(stage.as_str()) {
                completed_stages.push(stage);
            } else {
                missing_stages.push(stage);
            }
        }
        
        let all_stages_complete = missing_stages.is_empty();
        let completion_percentage = completed_stages.len() as f32 / self.required_stages.len() as f32;
        
        StageCompletionResult {
            all_stages_complete,
            completed_stages,
            missing_stages,
            completion_percentage,
            checked_at: current_timestamp(),
        }
    }
    
    /// Check if stages were completed in correct order
    pub fn check_stage_order(&self, fact: &ValidatedFact) -> StageOrderResult {
        let expected_order = vec![
            ValidationStage::Syntax,
            ValidationStage::Entity,
            ValidationStage::Semantic,
        ];
        
        let mut actual_order = Vec::new();
        let mut order_violations = Vec::new();
        
        // Build actual order from completed stages
        for stage in &expected_order {
            if fact.validation_metadata.completed_stages.contains(stage.as_str()) {
                actual_order.push(*stage);
            }
        }
        
        // Check for order violations
        let mut correct_order = true;
        for (i, &stage) in actual_order.iter().enumerate() {
            if i < expected_order.len() && stage != expected_order[i] {
                correct_order = false;
                order_violations.push(format!(
                    "Expected {} at position {}, found {}",
                    expected_order[i].as_str(),
                    i + 1,
                    stage.as_str()
                ));
            }
        }
        
        // Check for skipped stages
        for i in 0..actual_order.len().saturating_sub(1) {
            let current_order = actual_order[i].order();
            let next_order = actual_order[i + 1].order();
            
            if next_order != current_order + 1 {
                correct_order = false;
                order_violations.push(format!(
                    "Skipped validation stage between {} and {}",
                    actual_order[i].as_str(),
                    actual_order[i + 1].as_str()
                ));
            }
        }
        
        StageOrderResult {
            correct_order,
            order_violations,
            expected_order,
            actual_order,
            checked_at: current_timestamp(),
        }
    }
    
    /// Check timing requirements for validation stages
    pub fn check_timing_requirements(&self, fact: &ValidatedFact) -> TimingResult {
        let stage_timings = fact.validation_metadata.stage_durations.clone();
        let total_processing_time: u64 = stage_timings.values().sum();
        
        let mut timing_violations = Vec::new();
        let mut within_limits = true;
        
        // Check individual stage timings
        for (stage_name, &duration) in &stage_timings {
            if let Some(stage) = self.stage_from_string(stage_name) {
                if let Some(&max_time) = self.max_stage_time.get(&stage) {
                    if duration > max_time {
                        within_limits = false;
                        timing_violations.push(format!(
                            "Stage {} took {}ms, exceeds limit of {}ms",
                            stage_name, duration, max_time
                        ));
                    }
                }
            }
        }
        
        // Check total time
        if total_processing_time > self.max_total_time {
            within_limits = false;
            timing_violations.push(format!(
                "Total processing time {}ms exceeds limit of {}ms",
                total_processing_time, self.max_total_time
            ));
        }
        
        TimingResult {
            within_limits,
            total_processing_time,
            stage_timings,
            timing_violations,
            checked_at: current_timestamp(),
        }
    }
    
    /// Perform comprehensive validation chain check
    pub fn validate_complete_chain(&self, fact: &ValidatedFact) -> ValidationChainResult {
        let stage_result = self.check_stage_completion(fact);
        let order_result = self.check_stage_order(fact);
        let timing_result = self.check_timing_requirements(fact);
        
        let mut validation_issues = Vec::new();
        
        // Collect all issues
        if !stage_result.all_stages_complete {
            validation_issues.push(format!(
                "Missing {} validation stages",
                stage_result.missing_stages.len()
            ));
        }
        
        if !order_result.correct_order {
            validation_issues.extend(order_result.order_violations.clone());
        }
        
        if !timing_result.within_limits {
            validation_issues.extend(timing_result.timing_violations.clone());
        }
        
        let chain_valid = stage_result.all_stages_complete && 
                         order_result.correct_order && 
                         timing_result.within_limits;
        
        // Calculate chain quality score
        let completion_score = stage_result.completion_percentage;
        let order_score = if order_result.correct_order { 1.0 } else { 0.5 };
        let timing_score = if timing_result.within_limits { 1.0 } else { 0.7 };
        let chain_quality_score = (completion_score * 0.5 + order_score * 0.3 + timing_score * 0.2).max(0.0);
        
        ValidationChainResult {
            chain_valid,
            stage_result,
            order_result,
            timing_result,
            validation_issues,
            chain_quality_score,
            checked_at: current_timestamp(),
        }
    }
    
    /// Update timing limits
    pub fn update_timing_limits(&mut self, stage: ValidationStage, max_time: u64) {
        self.max_stage_time.insert(stage, max_time);
    }
    
    /// Update total time limit
    pub fn update_total_time_limit(&mut self, max_time: u64) {
        self.max_total_time = max_time;
    }
    
    /// Convert string to ValidationStage
    fn stage_from_string(&self, stage_name: &str) -> Option<ValidationStage> {
        match stage_name {
            "syntax" => Some(ValidationStage::Syntax),
            "entity" => Some(ValidationStage::Entity),
            "semantic" => Some(ValidationStage::Semantic),
            "quality" => Some(ValidationStage::Quality),
            "final" => Some(ValidationStage::Final),
            _ => None,
        }
    }
}

/// Get current timestamp in seconds since epoch
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl Default for ValidationChainChecker {
    fn default() -> Self {
        Self::new()
    }
}
```

## Verification Steps
1. Create ValidationChainChecker with stage tracking capabilities
2. Implement stage completion checking with missing stage detection
3. Add stage order validation to ensure proper sequence
4. Implement timing requirements checking with violation detection
5. Ensure comprehensive validation combines all aspects correctly

## Success Criteria
- [ ] ValidationChainChecker struct compiles without errors
- [ ] Stage completion checking works accurately
- [ ] Stage order validation detects violations correctly
- [ ] Timing requirements properly enforced
- [ ] All tests pass with comprehensive coverage