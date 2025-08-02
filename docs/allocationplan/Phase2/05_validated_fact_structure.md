# Task 05: Validated Fact Structure

## Metadata
- **Micro-Phase**: 2.5
- **Duration**: 15-20 minutes
- **Dependencies**: Task 03 (fact_content_structure), Task 04 (confidence_components)
- **Output**: `src/quality_integration/validated_fact.rs`

## Description
Create the ValidatedFact structure that combines FactContent with ConfidenceComponents and validation status. This represents a fact that has passed through the quality gate pipeline.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::quality_integration::{FactContent, ConfidenceComponents};

    #[test]
    fn test_validated_fact_creation() {
        let fact_content = FactContent::new("Elephants are mammals");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let validated_fact = ValidatedFact::new(fact_content, confidence);
        
        assert_eq!(validated_fact.content.text, "Elephants are mammals");
        assert_eq!(validated_fact.confidence.syntax_confidence, 0.9);
        assert_eq!(validated_fact.validation_status, ValidationStatus::Pending);
    }
    
    #[test]
    fn test_validation_status_transitions() {
        let fact_content = FactContent::new("Test fact");
        let confidence = ConfidenceComponents::new(0.8, 0.8, 0.8);
        let mut validated_fact = ValidatedFact::new(fact_content, confidence);
        
        validated_fact.mark_syntax_validated();
        assert!(validated_fact.is_syntax_validated());
        
        validated_fact.mark_entity_validated();
        assert!(validated_fact.is_entity_validated());
        
        validated_fact.mark_semantic_validated();
        assert!(validated_fact.is_semantic_validated());
        
        validated_fact.mark_fully_validated();
        assert_eq!(validated_fact.validation_status, ValidationStatus::FullyValidated);
    }
    
    #[test]
    fn test_quality_requirements_check() {
        let fact_content = FactContent::new("High quality fact");
        let high_confidence = ConfidenceComponents::new(0.95, 0.92, 0.90);
        let validated_fact = ValidatedFact::new(fact_content, high_confidence);
        
        assert!(validated_fact.meets_quality_requirements(0.85));
        assert!(!validated_fact.meets_quality_requirements(0.95));
    }
    
    #[test]
    fn test_ambiguity_tracking() {
        let fact_content = FactContent::new("Ambiguous fact");
        let confidence = ConfidenceComponents::new(0.8, 0.8, 0.8);
        let mut validated_fact = ValidatedFact::new(fact_content, confidence);
        
        validated_fact.add_ambiguity("Entity type unclear".to_string());
        validated_fact.add_ambiguity("Temporal reference vague".to_string());
        
        assert_eq!(validated_fact.ambiguity_count(), 2);
        assert!(validated_fact.has_ambiguities());
    }
    
    #[test]
    fn test_allocation_readiness() {
        let fact_content = FactContent::new("Ready for allocation");
        let confidence = ConfidenceComponents::new(0.9, 0.85, 0.88);
        let mut validated_fact = ValidatedFact::new(fact_content, confidence);
        
        validated_fact.mark_fully_validated();
        assert!(validated_fact.is_allocation_ready(0.8, 0));
        
        validated_fact.add_ambiguity("Minor issue".to_string());
        assert!(!validated_fact.is_allocation_ready(0.8, 0));
        assert!(validated_fact.is_allocation_ready(0.8, 1));
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use crate::quality_integration::{FactContent, ConfidenceComponents};

/// Validation status tracking for facts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Initial state - no validation performed
    Pending,
    /// Syntax validation completed
    SyntaxValidated,
    /// Entity validation completed
    EntityValidated,
    /// Semantic validation completed
    SemanticValidated,
    /// All validation stages completed
    FullyValidated,
    /// Validation failed at some stage
    Failed,
}

/// A fact that has been processed through quality validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedFact {
    /// The original fact content
    pub content: FactContent,
    
    /// Confidence scores for different aspects
    pub confidence: ConfidenceComponents,
    
    /// Current validation status
    pub validation_status: ValidationStatus,
    
    /// List of identified ambiguities
    pub ambiguities: Vec<String>,
    
    /// Timestamp when validation was completed
    pub validated_at: Option<u64>,
    
    /// Additional validation metadata
    pub validation_metadata: ValidationMetadata,
}

/// Additional metadata about the validation process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    /// Which validation stages were completed
    pub completed_stages: HashSet<String>,
    
    /// Time spent in each validation stage (milliseconds)
    pub stage_durations: std::collections::HashMap<String, u64>,
    
    /// Quality gate configuration used
    pub config_hash: Option<u64>,
}

impl ValidatedFact {
    /// Create a new validated fact
    pub fn new(content: FactContent, confidence: ConfidenceComponents) -> Self {
        Self {
            content,
            confidence,
            validation_status: ValidationStatus::Pending,
            ambiguities: Vec::new(),
            validated_at: None,
            validation_metadata: ValidationMetadata {
                completed_stages: HashSet::new(),
                stage_durations: std::collections::HashMap::new(),
                config_hash: None,
            },
        }
    }
    
    /// Mark syntax validation as completed
    pub fn mark_syntax_validated(&mut self) {
        self.validation_metadata.completed_stages.insert("syntax".to_string());
        if self.validation_status == ValidationStatus::Pending {
            self.validation_status = ValidationStatus::SyntaxValidated;
        }
    }
    
    /// Mark entity validation as completed
    pub fn mark_entity_validated(&mut self) {
        self.validation_metadata.completed_stages.insert("entity".to_string());
        if self.validation_status == ValidationStatus::SyntaxValidated {
            self.validation_status = ValidationStatus::EntityValidated;
        }
    }
    
    /// Mark semantic validation as completed
    pub fn mark_semantic_validated(&mut self) {
        self.validation_metadata.completed_stages.insert("semantic".to_string());
        if self.validation_status == ValidationStatus::EntityValidated {
            self.validation_status = ValidationStatus::SemanticValidated;
        }
    }
    
    /// Mark as fully validated
    pub fn mark_fully_validated(&mut self) {
        self.validation_status = ValidationStatus::FullyValidated;
        self.validated_at = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
        );
    }
    
    /// Mark validation as failed
    pub fn mark_failed(&mut self, reason: String) {
        self.validation_status = ValidationStatus::Failed;
        self.ambiguities.push(format!("FAILURE: {}", reason));
    }
    
    /// Check if syntax validation is completed
    pub fn is_syntax_validated(&self) -> bool {
        self.validation_metadata.completed_stages.contains("syntax")
    }
    
    /// Check if entity validation is completed
    pub fn is_entity_validated(&self) -> bool {
        self.validation_metadata.completed_stages.contains("entity")
    }
    
    /// Check if semantic validation is completed
    pub fn is_semantic_validated(&self) -> bool {
        self.validation_metadata.completed_stages.contains("semantic")
    }
    
    /// Check if fact meets quality requirements
    pub fn meets_quality_requirements(&self, threshold: f32) -> bool {
        self.confidence.minimum_confidence() >= threshold
    }
    
    /// Add an ambiguity to the fact
    pub fn add_ambiguity(&mut self, ambiguity: String) {
        self.ambiguities.push(ambiguity);
    }
    
    /// Get the number of ambiguities
    pub fn ambiguity_count(&self) -> usize {
        self.ambiguities.len()
    }
    
    /// Check if fact has any ambiguities
    pub fn has_ambiguities(&self) -> bool {
        !self.ambiguities.is_empty()
    }
    
    /// Check if fact is ready for allocation
    pub fn is_allocation_ready(&self, min_confidence: f32, max_ambiguities: usize) -> bool {
        self.validation_status == ValidationStatus::FullyValidated &&
        self.meets_quality_requirements(min_confidence) &&
        self.ambiguity_count() <= max_ambiguities
    }
    
    /// Get overall quality score (0.0-1.0)
    pub fn quality_score(&self) -> f32 {
        let confidence_score = self.confidence.overall_confidence();
        let validation_score = match self.validation_status {
            ValidationStatus::FullyValidated => 1.0,
            ValidationStatus::SemanticValidated => 0.8,
            ValidationStatus::EntityValidated => 0.6,
            ValidationStatus::SyntaxValidated => 0.4,
            ValidationStatus::Pending => 0.2,
            ValidationStatus::Failed => 0.0,
        };
        let ambiguity_penalty = (self.ambiguity_count() as f32 * 0.1).min(0.5);
        
        (confidence_score * 0.6 + validation_score * 0.4 - ambiguity_penalty).max(0.0)
    }
    
    /// Record validation stage duration
    pub fn record_stage_duration(&mut self, stage: &str, duration_ms: u64) {
        self.validation_metadata.stage_durations.insert(stage.to_string(), duration_ms);
    }
}

impl PartialEq for ValidatedFact {
    fn eq(&self, other: &Self) -> bool {
        self.content == other.content &&
        self.validation_status == other.validation_status
    }
}
```

## Verification Steps
1. Create ValidatedFact structure combining FactContent and ConfidenceComponents
2. Implement validation status tracking with stage progression
3. Add ambiguity detection and tracking capabilities
4. Implement allocation readiness checking logic
5. Ensure comprehensive test coverage passes

## Success Criteria
- [ ] ValidatedFact struct compiles without errors
- [ ] Validation status transitions work correctly
- [ ] Quality requirements checking functions properly
- [ ] Ambiguity tracking and counting accurate
- [ ] All tests pass with comprehensive coverage