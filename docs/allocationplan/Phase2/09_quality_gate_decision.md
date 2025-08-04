# Task 09: Quality Gate Decision

## Metadata
- **Micro-Phase**: 2.9
- **Duration**: 15-20 minutes
- **Dependencies**: Task 01 (quality_gate_config), Task 06 (threshold_checker), Task 07 (validation_chain), Task 08 (ambiguity_detection)
- **Output**: `src/quality_integration/quality_gate_decision.rs`

## Description
Create the QualityGateDecision engine that combines all quality checks (thresholds, validation chain, ambiguities) to make the final allocation decision. This is the central decision-making component of the quality gate system.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality_integration::{
        QualityGateConfig, ValidatedFact, FactContent, ConfidenceComponents,
        ThresholdChecker, ValidationChainChecker, AmbiguityDetector
    };

    #[test]
    fn test_quality_gate_decision_engine_creation() {
        let config = QualityGateConfig::default();
        let engine = QualityGateDecisionEngine::new(config);
        assert!(engine.is_enabled);
        assert_eq!(engine.decision_strategy, DecisionStrategy::Conservative);
    }
    
    #[test]
    fn test_perfect_fact_passes_gate() {
        let config = QualityGateConfig::default();
        let engine = QualityGateDecisionEngine::new(config);
        
        // Create a high-quality fact
        let fact_content = FactContent::new("The African elephant weighs up to 6 tons");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let mut validated_fact = ValidatedFact::new(fact_content, confidence);
        
        // Mark as fully validated
        validated_fact.mark_syntax_validated();
        validated_fact.mark_entity_validated();
        validated_fact.mark_semantic_validated();
        validated_fact.mark_fully_validated();
        
        // Add reasonable timing
        validated_fact.record_stage_duration("syntax", 100);
        validated_fact.record_stage_duration("entity", 150);
        validated_fact.record_stage_duration("semantic", 200);
        
        let decision = engine.make_allocation_decision(&validated_fact);
        assert_eq!(decision.decision, AllocationDecision::Approve);
        assert!(decision.overall_quality_score > 0.8);
        assert!(decision.passed_checks.len() >= 3);
        assert!(decision.failed_checks.is_empty());
    }
    
    #[test]
    fn test_low_confidence_fact_rejected() {
        let config = QualityGateConfig::default();
        let engine = QualityGateDecisionEngine::new(config);
        
        // Create a low-confidence fact
        let fact_content = FactContent::new("Something happened somewhere");
        let low_confidence = ConfidenceComponents::new(0.5, 0.4, 0.6);
        let validated_fact = ValidatedFact::new(fact_content, low_confidence);
        
        let decision = engine.make_allocation_decision(&validated_fact);
        assert_eq!(decision.decision, AllocationDecision::Reject);
        assert!(decision.overall_quality_score < 0.8);
        assert!(!decision.failed_checks.is_empty());
    }
    
    #[test]
    fn test_ambiguous_fact_handling() {
        let config = QualityGateConfig::default();
        let engine = QualityGateDecisionEngine::new(config);
        
        // Create a fact with multiple ambiguities
        let fact_content = FactContent::new("It was recently found that they often do that thing");
        let confidence = ConfidenceComponents::new(0.85, 0.8, 0.82);
        let mut validated_fact = ValidatedFact::new(fact_content, confidence);
        
        // Mark as validated but add ambiguities
        validated_fact.mark_fully_validated();
        validated_fact.add_ambiguity("Unclear pronoun 'it'".to_string());
        validated_fact.add_ambiguity("Vague temporal 'recently'".to_string());
        validated_fact.add_ambiguity("Unspecified 'thing'".to_string());
        
        let decision = engine.make_allocation_decision(&validated_fact);
        // With default config (max_ambiguity_count = 0), this should be rejected
        assert_eq!(decision.decision, AllocationDecision::Reject);
        assert!(decision.rejection_reasons.iter().any(|r| r.contains("ambiguity")));
    }
    
    #[test]
    fn test_incomplete_validation_chain() {
        let config = QualityGateConfig::default();
        let engine = QualityGateDecisionEngine::new(config);
        
        // Create fact with incomplete validation
        let fact_content = FactContent::new("Complete validation test");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let mut validated_fact = ValidatedFact::new(fact_content, confidence);
        
        // Only partially validate
        validated_fact.mark_syntax_validated();
        // Missing entity and semantic validation
        
        let decision = engine.make_allocation_decision(&validated_fact);
        assert_eq!(decision.decision, AllocationDecision::Reject);
        assert!(decision.rejection_reasons.iter().any(|r| r.contains("validation")));
    }
    
    #[test]
    fn test_decision_strategy_variants() {
        let config = QualityGateConfig::default();
        
        // Test conservative strategy
        let mut conservative_engine = QualityGateDecisionEngine::new(config.clone());
        conservative_engine.set_decision_strategy(DecisionStrategy::Conservative);
        
        // Test permissive strategy
        let mut permissive_engine = QualityGateDecisionEngine::new(config);
        permissive_engine.set_decision_strategy(DecisionStrategy::Permissive);
        
        // Create a borderline fact
        let fact_content = FactContent::new("The elephant is large");
        let borderline_confidence = ConfidenceComponents::new(0.78, 0.76, 0.79);
        let mut validated_fact = ValidatedFact::new(fact_content, borderline_confidence);
        validated_fact.mark_fully_validated();
        
        let conservative_decision = conservative_engine.make_allocation_decision(&validated_fact);
        let permissive_decision = permissive_engine.make_allocation_decision(&validated_fact);
        
        // Conservative should be more strict
        assert!(conservative_decision.overall_quality_score <= permissive_decision.overall_quality_score);
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::quality_integration::{
    QualityGateConfig, ValidatedFact, ThresholdChecker, ValidationChainChecker, 
    AmbiguityDetector, ComprehensiveThresholdResult, ValidationChainResult,
    ComprehensiveAmbiguityResult
};

/// Final allocation decision
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationDecision {
    /// Approve for allocation to neuromorphic system
    Approve,
    /// Reject - does not meet quality requirements
    Reject,
    /// Pending - requires manual review
    ManualReview,
}

/// Decision strategy for borderline cases
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecisionStrategy {
    /// Strict requirements - err on side of rejection
    Conservative,
    /// Moderate requirements - balanced approach
    Balanced,
    /// Relaxed requirements - err on side of approval
    Permissive,
}

/// Comprehensive quality gate decision result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGateDecisionResult {
    /// Final allocation decision
    pub decision: AllocationDecision,
    
    /// Overall quality score (0.0-1.0)
    pub overall_quality_score: f32,
    
    /// List of checks that passed
    pub passed_checks: Vec<String>,
    
    /// List of checks that failed
    pub failed_checks: Vec<String>,
    
    /// Detailed rejection reasons if rejected
    pub rejection_reasons: Vec<String>,
    
    /// Individual check results
    pub threshold_result: Option<ComprehensiveThresholdResult>,
    pub validation_result: Option<ValidationChainResult>,
    pub ambiguity_result: Option<ComprehensiveAmbiguityResult>,
    
    /// Decision confidence (0.0-1.0)
    pub decision_confidence: f32,
    
    /// Strategy used for decision
    pub strategy_used: DecisionStrategy,
    
    /// Processing metadata
    pub processing_metadata: DecisionMetadata,
    
    /// Timestamp of decision
    pub decided_at: u64,
}

/// Metadata about the decision process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionMetadata {
    /// Time spent on each check (milliseconds)
    pub check_durations: HashMap<String, u64>,
    
    /// Total decision processing time
    pub total_processing_time: u64,
    
    /// Configuration hash used
    pub config_hash: u64,
    
    /// Fact content hash
    pub fact_hash: u64,
}

/// Main quality gate decision engine
#[derive(Debug, Clone)]
pub struct QualityGateDecisionEngine {
    /// Configuration for quality requirements
    pub config: QualityGateConfig,
    
    /// Threshold checking component
    pub threshold_checker: ThresholdChecker,
    
    /// Validation chain checker
    pub validation_checker: ValidationChainChecker,
    
    /// Ambiguity detection component
    pub ambiguity_detector: AmbiguityDetector,
    
    /// Decision strategy
    pub decision_strategy: DecisionStrategy,
    
    /// Whether the engine is enabled
    pub is_enabled: bool,
    
    /// Strategy-specific quality thresholds
    pub strategy_thresholds: HashMap<DecisionStrategy, f32>,
}

impl QualityGateDecisionEngine {
    /// Create a new quality gate decision engine
    pub fn new(config: QualityGateConfig) -> Self {
        let threshold_checker = ThresholdChecker::new(config.clone());
        let validation_checker = ValidationChainChecker::new();
        let ambiguity_detector = AmbiguityDetector::new();
        
        let mut strategy_thresholds = HashMap::new();
        strategy_thresholds.insert(DecisionStrategy::Conservative, 0.85);
        strategy_thresholds.insert(DecisionStrategy::Balanced, 0.75);
        strategy_thresholds.insert(DecisionStrategy::Permissive, 0.65);
        
        Self {
            config,
            threshold_checker,
            validation_checker,
            ambiguity_detector,
            decision_strategy: DecisionStrategy::Conservative,
            is_enabled: true,
            strategy_thresholds,
        }
    }
    
    /// Make the final allocation decision for a validated fact
    pub fn make_allocation_decision(&self, fact: &ValidatedFact) -> QualityGateDecisionResult {
        if !self.is_enabled {
            return QualityGateDecisionResult {
                decision: AllocationDecision::ManualReview,
                overall_quality_score: 0.0,
                passed_checks: vec!["engine_disabled".to_string()],
                failed_checks: Vec::new(),
                rejection_reasons: vec!["Quality gate engine is disabled".to_string()],
                threshold_result: None,
                validation_result: None,
                ambiguity_result: None,
                decision_confidence: 0.0,
                strategy_used: self.decision_strategy,
                processing_metadata: DecisionMetadata {
                    check_durations: HashMap::new(),
                    total_processing_time: 0,
                    config_hash: 0,
                    fact_hash: fact.content.content_hash(),
                },
                decided_at: current_timestamp(),
            };
        }
        
        let start_time = std::time::Instant::now();
        let mut check_durations = HashMap::new();
        let mut passed_checks = Vec::new();
        let mut failed_checks = Vec::new();
        let mut rejection_reasons = Vec::new();
        
        // Perform threshold checks
        let threshold_start = std::time::Instant::now();
        let threshold_result = self.threshold_checker.check_all_thresholds(fact);
        check_durations.insert("thresholds".to_string(), threshold_start.elapsed().as_millis() as u64);
        
        if threshold_result.passed {
            passed_checks.push("threshold_validation".to_string());
        } else {
            failed_checks.push("threshold_validation".to_string());
            for failed_check in &threshold_result.failed_checks {
                rejection_reasons.push(format!("Threshold failure: {}", failed_check));
            }
        }
        
        // Perform validation chain checks
        let validation_start = std::time::Instant::now();
        let validation_result = self.validation_checker.validate_complete_chain(fact);
        check_durations.insert("validation_chain".to_string(), validation_start.elapsed().as_millis() as u64);
        
        if validation_result.chain_valid {
            passed_checks.push("validation_chain".to_string());
        } else {
            failed_checks.push("validation_chain".to_string());
            for issue in &validation_result.validation_issues {
                rejection_reasons.push(format!("Validation issue: {}", issue));
            }
        }
        
        // Perform ambiguity detection
        let ambiguity_start = std::time::Instant::now();
        let ambiguity_result = self.ambiguity_detector.detect_all_ambiguities(fact);
        check_durations.insert("ambiguity_detection".to_string(), ambiguity_start.elapsed().as_millis() as u64);
        
        if !ambiguity_result.should_reject {
            passed_checks.push("ambiguity_check".to_string());
        } else {
            failed_checks.push("ambiguity_check".to_string());
            rejection_reasons.push(format!(
                "Too many ambiguities: {} detected (max: {})",
                ambiguity_result.total_ambiguity_count,
                self.ambiguity_detector.max_allowed_ambiguities
            ));
        }
        
        // Calculate overall quality score
        let overall_quality_score = self.calculate_overall_quality_score(
            &threshold_result,
            &validation_result,
            &ambiguity_result,
            fact
        );
        
        // Make final decision based on strategy
        let (decision, decision_confidence) = self.make_final_decision(
            &threshold_result,
            &validation_result,
            &ambiguity_result,
            overall_quality_score,
            &failed_checks
        );
        
        let total_processing_time = start_time.elapsed().as_millis() as u64;
        
        QualityGateDecisionResult {
            decision,
            overall_quality_score,
            passed_checks,
            failed_checks,
            rejection_reasons,
            threshold_result: Some(threshold_result),
            validation_result: Some(validation_result),
            ambiguity_result: Some(ambiguity_result),
            decision_confidence,
            strategy_used: self.decision_strategy,
            processing_metadata: DecisionMetadata {
                check_durations,
                total_processing_time,
                config_hash: self.calculate_config_hash(),
                fact_hash: fact.content.content_hash(),
            },
            decided_at: current_timestamp(),
        }
    }
    
    /// Calculate overall quality score from all components
    fn calculate_overall_quality_score(
        &self,
        threshold_result: &ComprehensiveThresholdResult,
        validation_result: &ValidationChainResult,
        ambiguity_result: &ComprehensiveAmbiguityResult,
        fact: &ValidatedFact
    ) -> f32 {
        let threshold_score = if threshold_result.passed { threshold_result.overall_score } else { 0.0 };
        let validation_score = validation_result.chain_quality_score;
        let ambiguity_score = if ambiguity_result.should_reject { 0.0 } else { 
            1.0 - (ambiguity_result.total_ambiguity_count as f32 * 0.1).min(0.8)
        };
        let fact_quality_score = fact.quality_score();
        
        // Weighted combination based on strategy
        let weights = match self.decision_strategy {
            DecisionStrategy::Conservative => (0.3, 0.3, 0.3, 0.1),
            DecisionStrategy::Balanced => (0.25, 0.25, 0.25, 0.25),
            DecisionStrategy::Permissive => (0.2, 0.2, 0.2, 0.4),
        };
        
        (threshold_score * weights.0 + 
         validation_score * weights.1 + 
         ambiguity_score * weights.2 + 
         fact_quality_score * weights.3).max(0.0).min(1.0)
    }
    
    /// Make the final decision based on all checks and strategy
    fn make_final_decision(
        &self,
        threshold_result: &ComprehensiveThresholdResult,
        validation_result: &ValidationChainResult,
        ambiguity_result: &ComprehensiveAmbiguityResult,
        overall_score: f32,
        failed_checks: &[String]
    ) -> (AllocationDecision, f32) {
        let required_threshold = self.strategy_thresholds.get(&self.decision_strategy)
            .copied().unwrap_or(0.75);
        
        // Conservative strategy: any failure is rejection
        if self.decision_strategy == DecisionStrategy::Conservative {
            if !failed_checks.is_empty() {
                return (AllocationDecision::Reject, 0.9);
            }
        }
        
        // Check critical failures (always reject)
        if !threshold_result.passed && threshold_result.failed_checks.len() > 2 {
            return (AllocationDecision::Reject, 0.95);
        }
        
        if !validation_result.chain_valid && validation_result.validation_issues.len() > 1 {
            return (AllocationDecision::Reject, 0.95);
        }
        
        // Score-based decision
        if overall_score >= required_threshold {
            let confidence = (overall_score - required_threshold) / (1.0 - required_threshold) * 0.5 + 0.5;
            (AllocationDecision::Approve, confidence)
        } else if overall_score >= required_threshold - 0.1 && failed_checks.len() <= 1 {
            // Borderline case - consider manual review
            match self.decision_strategy {
                DecisionStrategy::Conservative => (AllocationDecision::ManualReview, 0.6),
                DecisionStrategy::Balanced => (AllocationDecision::ManualReview, 0.7),
                DecisionStrategy::Permissive => (AllocationDecision::Approve, 0.6),
            }
        } else {
            let confidence = 0.8 + (required_threshold - overall_score) / required_threshold * 0.2;
            (AllocationDecision::Reject, confidence.min(0.95))
        }
    }
    
    /// Set the decision strategy
    pub fn set_decision_strategy(&mut self, strategy: DecisionStrategy) {
        self.decision_strategy = strategy;
    }
    
    /// Update strategy threshold
    pub fn set_strategy_threshold(&mut self, strategy: DecisionStrategy, threshold: f32) {
        self.strategy_thresholds.insert(strategy, threshold.clamp(0.0, 1.0));
    }
    
    /// Enable or disable the engine
    pub fn set_enabled(&mut self, enabled: bool) {
        self.is_enabled = enabled;
    }
    
    /// Update configuration and propagate to components
    pub fn update_config(&mut self, config: QualityGateConfig) {
        self.config = config.clone();
        self.threshold_checker.update_config(config);
    }
    
    /// Calculate hash of current configuration
    fn calculate_config_hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        
        // Hash key configuration values
        self.config.min_confidence_for_allocation.to_bits().hash(&mut hasher);
        self.config.min_entity_confidence.to_bits().hash(&mut hasher);
        self.config.max_ambiguity_count.hash(&mut hasher);
        self.config.require_all_validations.hash(&mut hasher);
        (self.decision_strategy as u8).hash(&mut hasher);
        
        hasher.finish()
    }
    
    /// Get processing statistics
    pub fn get_processing_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("strategy".to_string(), format!("{:?}", self.decision_strategy));
        stats.insert("enabled".to_string(), self.is_enabled.to_string());
        stats.insert("threshold_count".to_string(), "5".to_string()); // Number of threshold checks
        stats.insert("validation_stages".to_string(), "3".to_string()); // Number of validation stages
        stats.insert("ambiguity_rules".to_string(), self.ambiguity_detector.detection_rules.len().to_string());
        stats
    }
}

/// Get current timestamp in seconds since epoch
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl Default for QualityGateDecisionEngine {
    fn default() -> Self {
        Self::new(QualityGateConfig::default())
    }
}
```

## Verification Steps
1. Create QualityGateDecisionEngine that integrates all quality components
2. Implement comprehensive decision logic with multiple strategies
3. Add detailed result structures with processing metadata
4. Implement strategy-based scoring and threshold management
5. Ensure all integration tests pass with realistic scenarios

## Success Criteria
- [ ] QualityGateDecisionEngine compiles without errors
- [ ] Decision logic correctly integrates all quality checks
- [ ] Strategy variants produce appropriately different results
- [ ] Processing metadata accurately tracks performance
- [ ] All tests pass with comprehensive coverage