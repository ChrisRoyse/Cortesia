# Micro Phase 2.2: Exception Detection Engine

**Estimated Time**: 45 minutes
**Dependencies**: Micro 2.1 (Exception Data Structures)
**Objective**: Implement intelligent detection of property exceptions in inheritance hierarchies

## Task Description

Create an automated system that can detect when local properties override inherited values and determine if these should be treated as exceptions with configurable sensitivity.

## Deliverables

Create `src/exceptions/detector.rs` with:

1. **ExceptionDetector struct**: Core detection engine
2. **Detection algorithms**: Various strategies for identifying exceptions
3. **Similarity analysis**: Determine if differences are meaningful
4. **Pattern recognition**: Learn common exception patterns
5. **Confidence scoring**: Rate how certain we are about exceptions

## Success Criteria

- [ ] Detects 100% of explicit property overrides
- [ ] Similarity threshold prevents false positives for minor differences
- [ ] Pattern recognition improves accuracy over time
- [ ] Confidence scores correlate with human judgment
- [ ] Detection runs in < 1ms per node
- [ ] Memory usage scales linearly with patterns

## Implementation Requirements

```rust
pub struct ExceptionDetector {
    similarity_threshold: f32,
    confidence_threshold: f32,
    patterns: Vec<ExceptionPattern>,
    statistics: DetectionStatistics,
}

#[derive(Debug, Clone)]
pub struct ExceptionPattern {
    property_regex: Regex,
    inherited_pattern: ValuePattern,
    actual_pattern: ValuePattern,
    reason_template: String,
    confidence_modifier: f32,
}

#[derive(Debug, Clone)]
pub enum ValuePattern {
    Exact(PropertyValue),
    Range(f64, f64),
    Regex(String),
    Type(String),
    Any,
}

impl ExceptionDetector {
    pub fn new(similarity_threshold: f32, confidence_threshold: f32) -> Self;
    
    pub fn detect_node_exceptions(
        &self,
        node: &InheritanceNode,
        inherited_properties: &HashMap<String, PropertyValue>
    ) -> Vec<(String, Exception)>;
    
    pub fn calculate_similarity(&self, value1: &PropertyValue, value2: &PropertyValue) -> f32;
    
    pub fn learn_pattern(&mut self, exceptions: &[(String, Exception)]);
    
    pub fn infer_exception_reason(&self, property: &str, inherited: &PropertyValue, actual: &PropertyValue) -> String;
    
    pub fn calculate_confidence(&self, property: &str, inherited: &PropertyValue, actual: &PropertyValue) -> f32;
}

#[derive(Debug, Default)]
pub struct DetectionStatistics {
    pub total_comparisons: AtomicU64,
    pub exceptions_detected: AtomicU64,
    pub false_positives: AtomicU64,
    pub patterns_learned: AtomicU64,
}
```

## Test Requirements

Must pass exception detection tests:
```rust
#[test]
fn test_basic_exception_detection() {
    let detector = ExceptionDetector::new(0.8, 0.7);
    
    let mut node = InheritanceNode::new(NodeId(1), "Penguin");
    node.local_properties.insert("can_fly".to_string(), PropertyValue::Boolean(false));
    
    let mut inherited = HashMap::new();
    inherited.insert("can_fly".to_string(), PropertyValue::Boolean(true));
    
    let exceptions = detector.detect_node_exceptions(&node, &inherited);
    
    assert_eq!(exceptions.len(), 1);
    let (prop, exception) = &exceptions[0];
    assert_eq!(prop, "can_fly");
    assert_eq!(exception.inherited_value, PropertyValue::Boolean(true));
    assert_eq!(exception.actual_value, PropertyValue::Boolean(false));
    assert!(exception.confidence > 0.9);
}

#[test]
fn test_similarity_thresholding() {
    let detector = ExceptionDetector::new(0.9, 0.7); // High similarity threshold
    
    let val1 = PropertyValue::String("color".to_string());
    let val2 = PropertyValue::String("colour".to_string()); // Minor spelling difference
    
    let similarity = detector.calculate_similarity(&val1, &val2);
    assert!(similarity > 0.9); // Should be considered very similar
    
    // Should NOT detect as exception due to high similarity
    let mut node = InheritanceNode::new(NodeId(1), "Node");
    node.local_properties.insert("spelling".to_string(), val2.clone());
    
    let mut inherited = HashMap::new();
    inherited.insert("spelling".to_string(), val1);
    
    let exceptions = detector.detect_node_exceptions(&node, &inherited);
    assert!(exceptions.is_empty()); // No exception due to similarity threshold
}

#[test]
fn test_pattern_learning() {
    let mut detector = ExceptionDetector::new(0.8, 0.7);
    
    // Teach it about common "flightless bird" pattern
    let training_exceptions = vec![
        ("can_fly".to_string(), Exception {
            inherited_value: PropertyValue::Boolean(true),
            actual_value: PropertyValue::Boolean(false),
            reason: "Flightless bird".to_string(),
            source: ExceptionSource::Explicit,
            created_at: Instant::now(),
            confidence: 1.0,
        }),
    ];
    
    detector.learn_pattern(&training_exceptions);
    
    // Test pattern recognition
    let reason = detector.infer_exception_reason(
        "can_fly",
        &PropertyValue::Boolean(true),
        &PropertyValue::Boolean(false)
    );
    
    assert!(reason.contains("flightless") || reason.contains("bird"));
}

#[test]
fn test_confidence_scoring() {
    let detector = ExceptionDetector::new(0.8, 0.7);
    
    // High confidence: clear boolean override
    let conf1 = detector.calculate_confidence(
        "can_fly",
        &PropertyValue::Boolean(true),
        &PropertyValue::Boolean(false)
    );
    
    // Lower confidence: similar strings
    let conf2 = detector.calculate_confidence(
        "color",
        &PropertyValue::String("red".to_string()),
        &PropertyValue::String("crimson".to_string())
    );
    
    assert!(conf1 > conf2);
    assert!(conf1 > 0.9);
    assert!(conf2 < 0.8);
}

#[test]
fn test_detection_performance() {
    let detector = ExceptionDetector::new(0.8, 0.7);
    
    let node = create_large_node_with_properties(1000);
    let inherited = create_inherited_properties(1000);
    
    let start = Instant::now();
    for _ in 0..100 {
        detector.detect_node_exceptions(&node, &inherited);
    }
    let elapsed = start.elapsed();
    
    let per_detection = elapsed / 100;
    assert!(per_detection < Duration::from_millis(1)); // < 1ms per detection
}
```

## File Location
`src/exceptions/detector.rs`

## Next Micro Phase
After completion, proceed to Micro 2.3: Exception Application System