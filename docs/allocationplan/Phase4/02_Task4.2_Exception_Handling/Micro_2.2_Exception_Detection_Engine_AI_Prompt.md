# AI Prompt: Micro Phase 2.2 - Exception Detection Engine

You are tasked with implementing intelligent detection of property exceptions in inheritance hierarchies. Your goal is to create `src/exceptions/detector.rs` with an automated system that detects when local properties override inherited values.

## Your Task
Implement the `ExceptionDetector` struct with detection algorithms, similarity analysis, pattern recognition, and confidence scoring to automatically identify meaningful property exceptions.

## Specific Requirements
1. Create `src/exceptions/detector.rs` with ExceptionDetector struct
2. Implement multiple detection strategies (strict, similarity-based, pattern-based)
3. Add similarity analysis to prevent false positives from minor differences
4. Implement pattern recognition to learn common exception scenarios
5. Provide confidence scoring for detected exceptions
6. Support configurable sensitivity levels
7. Integrate with ExceptionStore for automatic exception creation

## Expected Code Structure
```rust
use crate::exceptions::store::{Exception, ExceptionSource, ExceptionStore};
use crate::hierarchy::tree::InheritanceHierarchy;
use crate::hierarchy::node::NodeId;
use crate::properties::value::PropertyValue;
use crate::properties::resolver::PropertyResolver;

pub struct ExceptionDetector {
    similarity_threshold: f32,
    confidence_threshold: f32,
    detection_strategy: DetectionStrategy,
    pattern_learner: PatternLearner,
}

#[derive(Debug, Clone)]
pub enum DetectionStrategy {
    Strict,         // Any difference is an exception
    Similarity,     // Based on similarity threshold
    Pattern,        // Based on learned patterns
    Hybrid,         // Combination of strategies
}

impl ExceptionDetector {
    pub fn new() -> Self;
    
    pub fn detect_exceptions(
        &self,
        hierarchy: &InheritanceHierarchy,
        resolver: &PropertyResolver,
        node: NodeId
    ) -> Vec<DetectedExceptionCandidate>;
    
    pub fn analyze_property_override(
        &self,
        inherited_value: &PropertyValue,
        local_value: &PropertyValue,
        property_name: &str,
        node: NodeId
    ) -> Option<DetectedExceptionCandidate>;
    
    pub fn learn_from_feedback(
        &mut self,
        candidate: &DetectedExceptionCandidate,
        was_valid_exception: bool
    );
}

#[derive(Debug, Clone)]
pub struct DetectedExceptionCandidate {
    pub node: NodeId,
    pub property_name: String,
    pub inherited_value: PropertyValue,
    pub local_value: PropertyValue,
    pub confidence: f32,
    pub reason: String,
    pub detection_method: DetectionMethod,
}

#[derive(Debug, Clone)]
pub enum DetectionMethod {
    ValueDifference,
    TypeMismatch,
    PatternMatch,
    SemanticDifference,
}
```

## Success Criteria (You must verify these)
- [ ] Detects 100% of explicit property overrides correctly
- [ ] Similarity threshold prevents false positives for minor differences
- [ ] Pattern recognition improves accuracy over time through learning
- [ ] Confidence scoring reflects actual exception likelihood
- [ ] Performance suitable for real-time detection
- [ ] Configurable sensitivity prevents both false positives and negatives
- [ ] Integration with hierarchy and property systems works seamlessly

## Test Requirements
```rust
#[test]
fn test_strict_detection() {
    let detector = ExceptionDetector::new();
    let hierarchy = create_test_hierarchy();
    let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
    
    // Should detect any property override
    let candidates = detector.detect_exceptions(&hierarchy, &resolver, test_node);
    assert!(!candidates.is_empty());
}

#[test]
fn test_similarity_threshold() {
    let mut detector = ExceptionDetector::new();
    detector.set_similarity_threshold(0.9);
    
    // Minor differences should not trigger exceptions
    let candidate = detector.analyze_property_override(
        &PropertyValue::String("color".to_string()),
        &PropertyValue::String("colour".to_string()), // Minor spelling difference
        "description",
        NodeId(1)
    );
    
    assert!(candidate.is_none() || candidate.unwrap().confidence < 0.5);
}

#[test]
fn test_pattern_learning() {
    let mut detector = ExceptionDetector::new();
    
    // Train with feedback
    let candidate = create_test_candidate();
    detector.learn_from_feedback(&candidate, true);
    
    // Should improve detection for similar patterns
    let similar_candidate = detector.analyze_property_override(
        &PropertyValue::Boolean(true),
        &PropertyValue::Boolean(false),
        "can_fly",
        NodeId(2)
    );
    
    assert!(similar_candidate.is_some());
    assert!(similar_candidate.unwrap().confidence > 0.7);
}
```

## File to Create
Create exactly this file: `src/exceptions/detector.rs`

## When Complete
Respond with "MICRO PHASE 2.2 COMPLETE" and a brief summary of your implementation.