# AI Prompt: Micro Phase 2.4 - Exception Pattern Learning

You are tasked with implementing machine learning capabilities for exception pattern recognition. Your goal is to create `src/exceptions/learner.rs` with a system that learns from exception patterns to improve future detection.

## Your Task
Implement the `PatternLearner` struct that learns from exception feedback to improve detection accuracy over time using pattern recognition and statistical analysis.

## Specific Requirements
1. Create `src/exceptions/learner.rs` with PatternLearner struct
2. Implement pattern recognition algorithms for common exception scenarios
3. Add statistical analysis for exception probability prediction
4. Support feedback learning from user corrections
5. Maintain pattern database for future reference
6. Provide confidence scoring based on learned patterns

## Expected Code Structure
```rust
use crate::exceptions::store::{Exception, ExceptionSource};
use crate::hierarchy::node::NodeId;
use crate::properties::value::PropertyValue;
use std::collections::HashMap;

pub struct PatternLearner {
    pattern_database: HashMap<PatternKey, PatternStats>,
    feedback_history: Vec<FeedbackEntry>,
    learning_rate: f32,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct PatternKey {
    property_name: String,
    value_type_from: String,
    value_type_to: String,
    similarity_bucket: u8,
}

#[derive(Debug, Clone)]
struct PatternStats {
    true_positives: u32,
    false_positives: u32,
    confidence: f32,
    last_updated: std::time::Instant,
}

impl PatternLearner {
    pub fn new() -> Self;
    
    pub fn learn_pattern(
        &mut self,
        inherited_value: &PropertyValue,
        actual_value: &PropertyValue,
        property_name: &str,
        was_exception: bool
    );
    
    pub fn predict_exception_probability(
        &self,
        inherited_value: &PropertyValue,
        actual_value: &PropertyValue,
        property_name: &str
    ) -> f32;
    
    pub fn get_pattern_insights(&self) -> Vec<PatternInsight>;
}
```

## Success Criteria
- [ ] Learns from exception patterns effectively
- [ ] Improves detection accuracy over time
- [ ] Provides meaningful probability predictions
- [ ] Maintains efficient pattern database

## Test Requirements
```rust
#[test]
fn test_pattern_learning() {
    let mut learner = PatternLearner::new();
    
    // Train with known exception patterns
    learner.learn_pattern(
        &PropertyValue::Boolean(true),
        &PropertyValue::Boolean(false),
        "can_fly",
        true
    );
    
    // Test prediction
    let probability = learner.predict_exception_probability(
        &PropertyValue::Boolean(true),
        &PropertyValue::Boolean(false),
        "can_fly"
    );
    
    assert!(probability > 0.5);
}
```

## File to Create
Create exactly this file: `src/exceptions/learner.rs`

## When Complete
Respond with "MICRO PHASE 2.4 COMPLETE" and a brief summary of your implementation.