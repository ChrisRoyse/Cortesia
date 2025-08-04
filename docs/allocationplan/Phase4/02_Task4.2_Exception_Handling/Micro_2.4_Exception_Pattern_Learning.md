# Micro Phase 2.4: Exception Pattern Learning

**Estimated Time**: 40 minutes
**Dependencies**: Micro 2.3 (Exception Application System)
**Objective**: Implement AI system that learns common exception patterns to improve detection accuracy

## Task Description

Create a machine learning component that analyzes historical exceptions to identify patterns and improve future exception detection. The system learns from user feedback and successful detections to build a knowledge base of common exception scenarios.

This component uses lightweight pattern recognition algorithms that can run locally without external AI services, focusing on property name patterns, value type patterns, and contextual relationships.

## Deliverables

Create `src/exceptions/pattern_learner.rs` with:

1. **PatternLearner struct**: Core ML component for pattern recognition
2. **Pattern extraction**: Extract features from exceptions for learning
3. **Pattern matching**: Match new scenarios against learned patterns
4. **Feedback integration**: Learn from user corrections and confirmations
5. **Pattern persistence**: Save and load learned patterns

## Success Criteria

- [ ] Learns patterns from minimum 10 training examples
- [ ] Pattern matching accuracy improves over time (>80% after 100 examples)
- [ ] Predicts exception likelihood with calibrated confidence scores
- [ ] Integrates user feedback to refine patterns
- [ ] Pattern storage scales to 10,000+ patterns efficiently
- [ ] Learning process completes in < 5ms per example

## Implementation Requirements

```rust
pub struct PatternLearner {
    patterns: Vec<LearnedPattern>,
    feature_extractors: Vec<Box<dyn FeatureExtractor>>,
    classifier: SimpleClassifier,
    feedback_buffer: VecDeque<FeedbackExample>,
    learning_stats: LearningStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedPattern {
    pattern_id: Uuid,
    feature_vector: Vec<f32>,
    exception_likelihood: f32,
    reason_template: String,
    confidence_score: f32,
    usage_count: u32,
    success_rate: f32,
    created_at: SystemTime,
    last_used: SystemTime,
}

pub trait FeatureExtractor: Send + Sync {
    fn extract_features(&self, context: &ExceptionContext) -> Vec<f32>;
    fn feature_names(&self) -> Vec<String>;
    fn feature_count(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct ExceptionContext {
    property_name: String,
    inherited_value: PropertyValue,
    actual_value: PropertyValue,
    node_type: String,
    inheritance_depth: u32,
    sibling_properties: HashMap<String, PropertyValue>,
}

pub struct SimpleClassifier {
    weights: Vec<f32>,
    bias: f32,
    learning_rate: f32,
    regularization: f32,
}

#[derive(Debug, Clone)]
pub struct FeedbackExample {
    context: ExceptionContext,
    predicted_likelihood: f32,
    actual_outcome: bool, // true if it was indeed an exception
    user_feedback: Option<UserFeedback>,
    timestamp: SystemTime,
}

#[derive(Debug, Clone)]
pub enum UserFeedback {
    Correct,
    Incorrect,
    PartiallyCorrect(f32), // 0.0-1.0 correctness score
    BetterReason(String),
}

impl PatternLearner {
    pub fn new() -> Self;
    
    pub fn learn_from_exception(&mut self, exception: &Exception, context: &ExceptionContext);
    
    pub fn predict_exception_likelihood(&self, context: &ExceptionContext) -> f32;
    
    pub fn suggest_exception_reason(&self, context: &ExceptionContext) -> Option<String>;
    
    pub fn add_user_feedback(&mut self, context: ExceptionContext, feedback: UserFeedback);
    
    pub fn process_feedback_batch(&mut self) -> usize;
    
    pub fn get_best_matching_patterns(&self, context: &ExceptionContext, limit: usize) -> Vec<&LearnedPattern>;
    
    pub fn save_patterns(&self, path: &Path) -> Result<(), Box<dyn Error>>;
    
    pub fn load_patterns(&mut self, path: &Path) -> Result<(), Box<dyn Error>>;
    
    pub fn get_learning_statistics(&self) -> LearningStatistics;
}

// Feature extractors
pub struct PropertyNameExtractor;
pub struct ValueTypeExtractor;
pub struct ValueSimilarityExtractor;
pub struct ContextualExtractor;

impl FeatureExtractor for PropertyNameExtractor {
    fn extract_features(&self, context: &ExceptionContext) -> Vec<f32>;
    fn feature_names(&self) -> Vec<String>;
    fn feature_count(&self) -> usize;
}

#[derive(Debug, Default)]
pub struct LearningStatistics {
    pub patterns_learned: AtomicUsize,
    pub total_predictions: AtomicU64,
    pub correct_predictions: AtomicU64,
    pub feedback_examples: AtomicUsize,
    pub avg_learning_time_micros: AtomicU64,
    pub pattern_usage_distribution: HashMap<Uuid, u32>,
}
```

## Test Requirements

Must pass pattern learning tests:
```rust
#[test]
fn test_basic_pattern_learning() {
    let mut learner = PatternLearner::new();
    
    // Train on penguin examples
    for i in 0..20 {
        let context = ExceptionContext {
            property_name: "can_fly".to_string(),
            inherited_value: PropertyValue::Boolean(true),
            actual_value: PropertyValue::Boolean(false),
            node_type: "Penguin".to_string(),
            inheritance_depth: 2,
            sibling_properties: HashMap::new(),
        };
        
        let exception = Exception {
            inherited_value: PropertyValue::Boolean(true),
            actual_value: PropertyValue::Boolean(false),
            reason: "Flightless bird".to_string(),
            source: ExceptionSource::Explicit,
            created_at: Instant::now(),
            confidence: 1.0,
        };
        
        learner.learn_from_exception(&exception, &context);
    }
    
    // Test prediction on new penguin
    let test_context = ExceptionContext {
        property_name: "can_fly".to_string(),
        inherited_value: PropertyValue::Boolean(true),
        actual_value: PropertyValue::Boolean(false),
        node_type: "Emperor_Penguin".to_string(),
        inheritance_depth: 3,
        sibling_properties: HashMap::new(),
    };
    
    let likelihood = learner.predict_exception_likelihood(&test_context);
    assert!(likelihood > 0.8); // Should recognize similar pattern
    
    let reason = learner.suggest_exception_reason(&test_context);
    assert!(reason.is_some());
    assert!(reason.unwrap().to_lowercase().contains("flight"));
}

#[test]
fn test_feature_extraction() {
    let context = ExceptionContext {
        property_name: "can_fly".to_string(),
        inherited_value: PropertyValue::Boolean(true),
        actual_value: PropertyValue::Boolean(false),
        node_type: "Penguin".to_string(),
        inheritance_depth: 2,
        sibling_properties: {
            let mut props = HashMap::new();
            props.insert("habitat".to_string(), PropertyValue::String("antarctica".to_string()));
            props.insert("diet".to_string(), PropertyValue::String("fish".to_string()));
            props
        },
    };
    
    let name_extractor = PropertyNameExtractor;
    let type_extractor = ValueTypeExtractor;
    let similarity_extractor = ValueSimilarityExtractor;
    let context_extractor = ContextualExtractor;
    
    let name_features = name_extractor.extract_features(&context);
    let type_features = type_extractor.extract_features(&context);
    let similarity_features = similarity_extractor.extract_features(&context);
    let context_features = context_extractor.extract_features(&context);
    
    assert!(!name_features.is_empty());
    assert!(!type_features.is_empty());
    assert!(!similarity_features.is_empty());
    assert!(!context_features.is_empty());
    
    // All features should be normalized (0.0-1.0)
    for &feature in &name_features {
        assert!(feature >= 0.0 && feature <= 1.0);
    }
}

#[test]
fn test_user_feedback_integration() {
    let mut learner = PatternLearner::new();
    
    let context = ExceptionContext {
        property_name: "color".to_string(),
        inherited_value: PropertyValue::String("red".to_string()),
        actual_value: PropertyValue::String("crimson".to_string()),
        node_type: "Apple".to_string(),
        inheritance_depth: 1,
        sibling_properties: HashMap::new(),
    };
    
    // Initial prediction (should be uncertain)
    let initial_likelihood = learner.predict_exception_likelihood(&context);
    
    // User says this is NOT an exception (too similar)
    learner.add_user_feedback(context.clone(), UserFeedback::Incorrect);
    learner.process_feedback_batch();
    
    // Prediction should be lower now
    let updated_likelihood = learner.predict_exception_likelihood(&context);
    assert!(updated_likelihood < initial_likelihood);
}

#[test]
fn test_pattern_persistence() {
    let mut learner1 = PatternLearner::new();
    
    // Train some patterns
    for i in 0..10 {
        let context = ExceptionContext {
            property_name: format!("prop_{}", i),
            inherited_value: PropertyValue::String("default".to_string()),
            actual_value: PropertyValue::String(format!("special_{}", i)),
            node_type: "TestNode".to_string(),
            inheritance_depth: 1,
            sibling_properties: HashMap::new(),
        };
        
        let exception = Exception {
            inherited_value: PropertyValue::String("default".to_string()),
            actual_value: PropertyValue::String(format!("special_{}", i)),
            reason: "Test override".to_string(),
            source: ExceptionSource::Detected,
            created_at: Instant::now(),
            confidence: 0.8,
        };
        
        learner1.learn_from_exception(&exception, &context);
    }
    
    // Save patterns
    let temp_path = std::env::temp_dir().join("test_patterns.json");
    learner1.save_patterns(&temp_path).expect("Failed to save patterns");
    
    // Load into new learner
    let mut learner2 = PatternLearner::new();
    learner2.load_patterns(&temp_path).expect("Failed to load patterns");
    
    // Test that patterns were preserved
    let test_context = ExceptionContext {
        property_name: "prop_5".to_string(),
        inherited_value: PropertyValue::String("default".to_string()),
        actual_value: PropertyValue::String("special_5".to_string()),
        node_type: "TestNode".to_string(),
        inheritance_depth: 1,
        sibling_properties: HashMap::new(),
    };
    
    let likelihood = learner2.predict_exception_likelihood(&test_context);
    assert!(likelihood > 0.5); // Should recognize learned pattern
    
    // Cleanup
    std::fs::remove_file(temp_path).ok();
}

#[test]
fn test_learning_performance() {
    let mut learner = PatternLearner::new();
    
    let start = Instant::now();
    
    // Learn from 1000 examples
    for i in 0..1000 {
        let context = ExceptionContext {
            property_name: format!("prop_{}", i % 10),
            inherited_value: PropertyValue::String("default".to_string()),
            actual_value: PropertyValue::String(format!("value_{}", i)),
            node_type: format!("Type_{}", i % 5),
            inheritance_depth: (i % 5) as u32,
            sibling_properties: HashMap::new(),
        };
        
        let exception = Exception {
            inherited_value: PropertyValue::String("default".to_string()),
            actual_value: PropertyValue::String(format!("value_{}", i)),
            reason: "Generated".to_string(),
            source: ExceptionSource::Detected,
            created_at: Instant::now(),
            confidence: 0.8,
        };
        
        learner.learn_from_exception(&exception, &context);
    }
    
    let elapsed = start.elapsed();
    let avg_per_example = elapsed / 1000;
    
    assert!(avg_per_example < Duration::from_millis(5)); // < 5ms per example
    
    let stats = learner.get_learning_statistics();
    assert_eq!(stats.patterns_learned.load(Ordering::Relaxed), 1000);
}
```

## File Location
`src/exceptions/pattern_learner.rs`

## Next Micro Phase
After completion, proceed to Micro 2.5: Exception Storage Optimization