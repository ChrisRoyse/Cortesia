# Task 39: Scoring Framework Design

## Metadata
- **Micro-Phase**: 2.39
- **Duration**: 18 minutes
- **Dependencies**: Tasks 34-38 (hierarchy detection)
- **Output**: `src/scoring/scoring_framework.rs`

## Description
Design the multi-factor scoring framework that evaluates allocation candidates based on semantic similarity, property compatibility, structural fit, and confidence scores.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_score_creation() {
        let score = AllocationScore::new(0.8, 0.7, 0.9, 0.85);
        assert_eq!(score.semantic_similarity, 0.8);
        assert_eq!(score.property_compatibility, 0.7);
        assert_eq!(score.structural_fit, 0.9);
        assert_eq!(score.confidence, 0.85);
        assert!((score.total - 0.8125).abs() < 0.001); // Weighted average
    }
    
    #[test]
    fn test_scoring_weights_configuration() {
        let weights = ScoringWeights::new()
            .semantic_weight(0.5)
            .property_weight(0.3)
            .structural_weight(0.15)
            .confidence_weight(0.05);
        
        assert!((weights.total() - 1.0).abs() < 0.001);
        assert!(weights.is_valid());
    }
    
    #[test]
    fn test_allocation_scorer_basic() {
        let scorer = AllocationScorer::new(ScoringWeights::default());
        let concept = create_test_concept("elephant");
        let parent = create_test_hierarchy_node("animal");
        let hierarchy = create_test_hierarchy();
        
        let score = scorer.score_allocation(&concept, &parent, &hierarchy);
        assert!(score.total >= 0.0 && score.total <= 1.0);
        assert!(score.breakdown.len() >= 4);
    }
    
    #[test]
    fn test_parallel_scoring_performance() {
        let scorer = AllocationScorer::new(ScoringWeights::default());
        let concepts: Vec<_> = (0..100).map(|i| create_test_concept(&format!("concept_{}", i))).collect();
        
        let start = std::time::Instant::now();
        let scores: Vec<_> = concepts.iter()
            .map(|c| scorer.score_allocation(c, &create_test_hierarchy_node("root"), &create_test_hierarchy()))
            .collect();
        let elapsed = start.elapsed();
        
        assert_eq!(scores.len(), 100);
        assert!(elapsed < std::time::Duration::from_millis(100)); // <1ms per concept
    }
}
```

## Implementation
```rust
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::hierarchy::{ExtractedConcept, HierarchyNode, Hierarchy};

/// Comprehensive allocation score with component breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationScore {
    /// Overall weighted score (0.0-1.0)
    pub total: f32,
    
    /// Semantic similarity component
    pub semantic_similarity: f32,
    
    /// Property compatibility component
    pub property_compatibility: f32,
    
    /// Structural fit component
    pub structural_fit: f32,
    
    /// Confidence component
    pub confidence: f32,
    
    /// Detailed breakdown by factor
    pub breakdown: HashMap<String, f32>,
    
    /// Human-readable justification
    pub justification: String,
}

impl AllocationScore {
    pub fn new(semantic: f32, property: f32, structural: f32, confidence: f32) -> Self {
        let weights = ScoringWeights::default();
        let total = weights.semantic * semantic +
                   weights.property * property +
                   weights.structural * structural +
                   weights.confidence * confidence;
        
        let mut breakdown = HashMap::new();
        breakdown.insert("semantic".to_string(), semantic);
        breakdown.insert("property".to_string(), property);
        breakdown.insert("structural".to_string(), structural);
        breakdown.insert("confidence".to_string(), confidence);
        
        Self {
            total: total.clamp(0.0, 1.0),
            semantic_similarity: semantic,
            property_compatibility: property,
            structural_fit: structural,
            confidence,
            breakdown,
            justification: format!("Score: {:.3} (semantic: {:.2}, property: {:.2}, structural: {:.2}, confidence: {:.2})", 
                                 total, semantic, property, structural, confidence),
        }
    }
}

/// Configurable weights for scoring components
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ScoringWeights {
    pub semantic: f32,
    pub property: f32,
    pub structural: f32,
    pub confidence: f32,
}

impl ScoringWeights {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn semantic_weight(mut self, weight: f32) -> Self {
        self.semantic = weight.clamp(0.0, 1.0);
        self
    }
    
    pub fn property_weight(mut self, weight: f32) -> Self {
        self.property = weight.clamp(0.0, 1.0);
        self
    }
    
    pub fn structural_weight(mut self, weight: f32) -> Self {
        self.structural = weight.clamp(0.0, 1.0);
        self
    }
    
    pub fn confidence_weight(mut self, weight: f32) -> Self {
        self.confidence = weight.clamp(0.0, 1.0);
        self
    }
    
    pub fn total(&self) -> f32 {
        self.semantic + self.property + self.structural + self.confidence
    }
    
    pub fn is_valid(&self) -> bool {
        (self.total() - 1.0).abs() < 0.001
    }
    
    pub fn normalize(&mut self) {
        let total = self.total();
        if total > 0.0 {
            self.semantic /= total;
            self.property /= total;
            self.structural /= total;
            self.confidence /= total;
        }
    }
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            semantic: 0.4,
            property: 0.3,
            structural: 0.2,
            confidence: 0.1,
        }
    }
}

/// Main allocation scorer
pub struct AllocationScorer {
    weights: ScoringWeights,
}

impl AllocationScorer {
    pub fn new(weights: ScoringWeights) -> Self {
        Self { weights }
    }
    
    pub fn score_allocation(
        &self,
        concept: &ExtractedConcept,
        parent: &HierarchyNode,
        hierarchy: &Hierarchy,
    ) -> AllocationScore {
        // Compute component scores (implementation details in subsequent tasks)
        let semantic = self.compute_semantic_similarity(concept, parent);
        let property = self.compute_property_compatibility(concept, parent);
        let structural = self.compute_structural_fit(concept, parent, hierarchy);
        let confidence = concept.confidence;
        
        AllocationScore::new(semantic, property, structural, confidence)
    }
    
    fn compute_semantic_similarity(&self, _concept: &ExtractedConcept, _parent: &HierarchyNode) -> f32 {
        // Placeholder - implemented in task 40
        0.8
    }
    
    fn compute_property_compatibility(&self, _concept: &ExtractedConcept, _parent: &HierarchyNode) -> f32 {
        // Placeholder - implemented in task 41
        0.7
    }
    
    fn compute_structural_fit(&self, _concept: &ExtractedConcept, _parent: &HierarchyNode, _hierarchy: &Hierarchy) -> f32 {
        // Placeholder - implemented in task 42
        0.9
    }
}

// Test helpers
fn create_test_concept(name: &str) -> ExtractedConcept {
    ExtractedConcept {
        name: name.to_string(),
        confidence: 0.8,
        properties: HashMap::new(),
        concept_type: crate::hierarchy::ConceptType::Entity,
        proposed_parent: None,
        source_span: (0, name.len()),
        evidence: vec![],
    }
}

fn create_test_hierarchy_node(name: &str) -> HierarchyNode {
    HierarchyNode {
        name: name.to_string(),
        properties: HashMap::new(),
        children: vec![],
        parent: None,
    }
}

fn create_test_hierarchy() -> Hierarchy {
    Hierarchy::new()
}
```

## Verification Steps
1. Create AllocationScore structure with component breakdown
2. Implement configurable ScoringWeights with validation
3. Create AllocationScorer with pluggable components
4. Add parallel processing capability
5. Ensure all tests pass

## Success Criteria
- [ ] AllocationScore compiles and functions correctly
- [ ] ScoringWeights validation works
- [ ] Parallel scoring <1ms per concept
- [ ] Component breakdown accurate
- [ ] All tests pass