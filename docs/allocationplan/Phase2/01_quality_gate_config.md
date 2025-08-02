# Task 01: Quality Gate Configuration Structure

## Metadata
- **Micro-Phase**: 2.1
- **Duration**: 15 minutes
- **Dependencies**: None
- **Output**: `src/quality_integration/quality_gate_config.rs`

## Description
Create the configuration structure that defines quality thresholds for Phase 0A integration. This establishes the minimum quality requirements for facts entering the allocation engine.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_gate_config_defaults() {
        let config = QualityGateConfig::default();
        assert_eq!(config.min_confidence_for_allocation, 0.8);
        assert_eq!(config.require_all_validations, true);
        assert_eq!(config.max_ambiguity_count, 0);
        assert_eq!(config.min_entity_confidence, 0.75);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = QualityGateConfig::default();
        assert!(config.validate().is_ok());
        
        config.min_confidence_for_allocation = 1.5; // Invalid
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_config_builder() {
        let config = QualityGateConfig::new()
            .with_min_confidence(0.9)
            .with_max_ambiguity(2);
        assert_eq!(config.min_confidence_for_allocation, 0.9);
        assert_eq!(config.max_ambiguity_count, 2);
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};

/// Configuration for quality gate thresholds
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QualityGateConfig {
    /// Minimum confidence score required for allocation (0.0-1.0)
    pub min_confidence_for_allocation: f32,
    
    /// Whether all validation stages must be completed
    pub require_all_validations: bool,
    
    /// Maximum number of unresolved ambiguities allowed
    pub max_ambiguity_count: usize,
    
    /// Minimum entity confidence score required
    pub min_entity_confidence: f32,
}

impl Default for QualityGateConfig {
    fn default() -> Self {
        Self {
            min_confidence_for_allocation: 0.8,
            require_all_validations: true,
            max_ambiguity_count: 0,
            min_entity_confidence: 0.75,
        }
    }
}

impl QualityGateConfig {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_min_confidence(mut self, confidence: f32) -> Self {
        self.min_confidence_for_allocation = confidence.clamp(0.0, 1.0);
        self
    }
    
    pub fn with_max_ambiguity(mut self, count: usize) -> Self {
        self.max_ambiguity_count = count;
        self
    }
    
    pub fn validate(&self) -> Result<(), String> {
        if self.min_confidence_for_allocation < 0.0 || 
           self.min_confidence_for_allocation > 1.0 {
            return Err("Confidence must be between 0.0 and 1.0".to_string());
        }
        if self.min_entity_confidence < 0.0 || 
           self.min_entity_confidence > 1.0 {
            return Err("Entity confidence must be between 0.0 and 1.0".to_string());
        }
        Ok(())
    }
}
```

## Verification Steps
1. Create the configuration structure with all required fields
2. Implement Default trait with sensible defaults
3. Add builder methods for configuration
4. Implement validation logic
5. Ensure all tests pass

## Success Criteria
- [ ] Struct compiles without errors
- [ ] Default values match specification
- [ ] Validation catches invalid values
- [ ] Builder pattern works correctly
- [ ] All tests pass