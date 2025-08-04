# Task 04: Confidence Components

## Metadata
- **Micro-Phase**: 2.4
- **Duration**: 15 minutes
- **Dependencies**: None
- **Output**: `src/quality_integration/confidence_components.rs`

## Description
Create the ConfidenceComponents structure that breaks down confidence scores into syntax, entity, and semantic components for detailed quality tracking.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_components_creation() {
        let components = ConfidenceComponents::new(0.9, 0.85, 0.88);
        assert_eq!(components.syntax_confidence, 0.9);
        assert_eq!(components.entity_confidence, 0.85);
        assert_eq!(components.semantic_confidence, 0.88);
    }
    
    #[test]
    fn test_confidence_clamping() {
        let components = ConfidenceComponents::new(1.5, -0.1, 0.5);
        assert_eq!(components.syntax_confidence, 1.0);
        assert_eq!(components.entity_confidence, 0.0);
        assert_eq!(components.semantic_confidence, 0.5);
    }
    
    #[test]
    fn test_overall_confidence_calculation() {
        let components = ConfidenceComponents::new(0.9, 0.8, 0.7);
        let overall = components.overall_confidence();
        assert!((overall - 0.8).abs() < 0.01); // Average of 0.9, 0.8, 0.7
    }
    
    #[test]
    fn test_minimum_confidence() {
        let components = ConfidenceComponents::new(0.9, 0.6, 0.8);
        assert_eq!(components.minimum_confidence(), 0.6);
    }
    
    #[test]
    fn test_meets_threshold() {
        let components = ConfidenceComponents::new(0.9, 0.85, 0.88);
        assert!(components.meets_threshold(0.8));
        assert!(!components.meets_threshold(0.9));
    }
}
```

## Implementation
```rust
use serde::{Deserialize, Serialize};

/// Breakdown of confidence scores by component
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConfidenceComponents {
    /// Confidence in syntactic parsing (0.0-1.0)
    pub syntax_confidence: f32,
    
    /// Confidence in entity recognition (0.0-1.0)
    pub entity_confidence: f32,
    
    /// Confidence in semantic understanding (0.0-1.0)
    pub semantic_confidence: f32,
    
    /// Optional relationship confidence (0.0-1.0)
    pub relationship_confidence: Option<f32>,
    
    /// Optional temporal confidence (0.0-1.0)
    pub temporal_confidence: Option<f32>,
}

impl ConfidenceComponents {
    /// Create new confidence components with required values
    pub fn new(syntax: f32, entity: f32, semantic: f32) -> Self {
        Self {
            syntax_confidence: syntax.clamp(0.0, 1.0),
            entity_confidence: entity.clamp(0.0, 1.0),
            semantic_confidence: semantic.clamp(0.0, 1.0),
            relationship_confidence: None,
            temporal_confidence: None,
        }
    }
    
    /// Add relationship confidence
    pub fn with_relationship(mut self, confidence: f32) -> Self {
        self.relationship_confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }
    
    /// Add temporal confidence
    pub fn with_temporal(mut self, confidence: f32) -> Self {
        self.temporal_confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }
    
    /// Calculate overall confidence as weighted average
    pub fn overall_confidence(&self) -> f32 {
        let mut sum = self.syntax_confidence + 
                     self.entity_confidence + 
                     self.semantic_confidence;
        let mut count = 3.0;
        
        if let Some(rel) = self.relationship_confidence {
            sum += rel;
            count += 1.0;
        }
        
        if let Some(temp) = self.temporal_confidence {
            sum += temp;
            count += 1.0;
        }
        
        sum / count
    }
    
    /// Get the minimum confidence across all components
    pub fn minimum_confidence(&self) -> f32 {
        let mut min = self.syntax_confidence
            .min(self.entity_confidence)
            .min(self.semantic_confidence);
        
        if let Some(rel) = self.relationship_confidence {
            min = min.min(rel);
        }
        
        if let Some(temp) = self.temporal_confidence {
            min = min.min(temp);
        }
        
        min
    }
    
    /// Check if all components meet a threshold
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.minimum_confidence() >= threshold
    }
    
    /// Get a breakdown of components as a vector
    pub fn as_vector(&self) -> Vec<f32> {
        let mut v = vec![
            self.syntax_confidence,
            self.entity_confidence,
            self.semantic_confidence,
        ];
        
        if let Some(rel) = self.relationship_confidence {
            v.push(rel);
        }
        
        if let Some(temp) = self.temporal_confidence {
            v.push(temp);
        }
        
        v
    }
}

impl Default for ConfidenceComponents {
    fn default() -> Self {
        Self::new(0.5, 0.5, 0.5)
    }
}
```

## Verification Steps
1. Create ConfidenceComponents structure with required fields
2. Implement confidence clamping to [0, 1]
3. Add overall confidence calculation
4. Implement threshold checking
5. Ensure all tests pass

## Success Criteria
- [ ] ConfidenceComponents struct compiles
- [ ] Value clamping works correctly
- [ ] Overall confidence calculation accurate
- [ ] Threshold checking functional
- [ ] All tests pass