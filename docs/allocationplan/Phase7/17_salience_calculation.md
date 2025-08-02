# Micro Task 17: Salience Calculation

**Priority**: MEDIUM  
**Estimated Time**: 25 minutes  
**Dependencies**: 16_focus_switching.md completed  
**Skills Required**: Perceptual modeling, salience algorithms, feature detection

## Objective

Implement bottom-up salience calculation that automatically detects attention-worthy stimuli based on novelty, contrast, motion, and other perceptual features that capture attention involuntarily.

## Context

Salience calculation models the bottom-up attention capture mechanism that operates independently of current goals. It identifies stimuli that stand out from their context due to perceptual properties like high contrast, sudden appearance, motion, or deviation from expected patterns.

## Specifications

### Core Salience Components

1. **SalienceCalculator struct**
   - Multi-feature salience computation
   - Temporal salience tracking
   - Context-relative salience
   - Feature combination strategies

2. **SalienceFeatures struct**
   - Visual contrast measures
   - Temporal change detection
   - Novelty assessment
   - Pattern deviation scoring

3. **SalienceMap struct**
   - Spatial salience distribution
   - Temporal salience evolution
   - Feature contribution tracking
   - Attention capture probabilities

### Performance Requirements

- Salience calculation latency < 10ms
- Support real-time salience updates
- Handle 100+ simultaneous stimuli
- Adaptive feature weighting
- Memory efficient continuous operation

## Implementation Guide

### Step 1: Core Salience Types

```rust
// File: src/cognitive/attention/salience.rs

use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::core::types::{EntityId, NodeId};

#[derive(Debug, Clone)]
pub struct SalienceFeatures {
    pub intensity: f32,           // Absolute stimulus strength (0.0-1.0)
    pub contrast: f32,           // Relative to background (0.0-1.0) 
    pub novelty: f32,            // Deviation from expectation (0.0-1.0)
    pub motion: f32,             // Rate of change (0.0-1.0)
    pub size: f32,               // Relative size importance (0.0-1.0)
    pub color_pop_out: f32,      // Color distinctiveness (0.0-1.0)
    pub onset: f32,              // Sudden appearance (0.0-1.0)
    pub semantic_surprise: f32,   // Unexpected meaning (0.0-1.0)
}

#[derive(Debug, Clone)]
pub struct SalienceContext {
    pub background_activity: f32,
    pub adaptation_level: f32,
    pub attention_allocation: HashMap<EntityId, f32>,
    pub temporal_expectation: f32,
    pub noise_level: f32,
}

#[derive(Debug, Clone)]
pub struct SalienceRecord {
    pub entity_id: EntityId,
    pub total_salience: f32,
    pub feature_contributions: SalienceFeatures,
    pub computed_at: Instant,
    pub context: SalienceContext,
    pub capture_probability: f32,
}

#[derive(Debug)]
pub struct SalienceCalculator {
    feature_weights: HashMap<String, f32>,
    adaptation_tracker: AdaptationTracker,
    novelty_detector: NoveltyDetector,
    contrast_analyzer: ContrastAnalyzer,
    temporal_processor: TemporalProcessor,
    salience_history: HashMap<EntityId, Vec<SalienceRecord>>,
    combination_strategy: CombinationStrategy,
}

#[derive(Debug, Clone)]
pub enum CombinationStrategy {
    Linear,
    MaxPooling,
    WeightedSum { normalization: bool },
    NonLinearSigmoid { steepness: f32 },
    CompetitiveWTA { threshold: f32 },
}

#[derive(Debug)]
pub struct AdaptationTracker {
    baseline_levels: HashMap<String, f32>,
    adaptation_rates: HashMap<String, f32>,
    last_update: Instant,
}

#[derive(Debug)]
pub struct NoveltyDetector {
    expectation_model: HashMap<EntityId, ExpectationRecord>,
    surprise_threshold: f32,
    learning_rate: f32,
}

#[derive(Debug, Clone)]
pub struct ExpectationRecord {
    pub expected_features: SalienceFeatures,
    pub confidence: f32,
    pub last_observed: Instant,
    pub observation_count: usize,
}

#[derive(Debug)]
pub struct ContrastAnalyzer {
    background_model: BackgroundModel,
    contrast_kernels: Vec<ContrastKernel>,
}

#[derive(Debug)]
pub struct BackgroundModel {
    mean_intensity: f32,
    variance: f32,
    dominant_features: SalienceFeatures,
    update_rate: f32,
}

#[derive(Debug)]
pub struct ContrastKernel {
    pub feature_type: String,
    pub weight: f32,
    pub threshold: f32,
}
```

### Step 2: Salience Calculation Implementation

```rust
impl SalienceCalculator {
    pub fn new() -> Self {
        Self {
            feature_weights: Self::initialize_feature_weights(),
            adaptation_tracker: AdaptationTracker::new(),
            novelty_detector: NoveltyDetector::new(),
            contrast_analyzer: ContrastAnalyzer::new(),
            temporal_processor: TemporalProcessor::new(),
            salience_history: HashMap::new(),
            combination_strategy: CombinationStrategy::WeightedSum { normalization: true },
        }
    }
    
    fn initialize_feature_weights() -> HashMap<String, f32> {
        let mut weights = HashMap::new();
        
        // Weights based on psychological research
        weights.insert("intensity".to_string(), 0.15);
        weights.insert("contrast".to_string(), 0.25);
        weights.insert("novelty".to_string(), 0.20);
        weights.insert("motion".to_string(), 0.15);
        weights.insert("size".to_string(), 0.10);
        weights.insert("color_pop_out".to_string(), 0.08);
        weights.insert("onset".to_string(), 0.12);
        weights.insert("semantic_surprise".to_string(), 0.15);
        
        weights
    }
    
    pub fn calculate_salience(
        &mut self,
        entity_id: EntityId,
        stimulus_properties: &StimulusProperties,
        context: &SalienceContext
    ) -> SalienceRecord {
        
        // Extract basic features
        let mut features = self.extract_basic_features(stimulus_properties);
        
        // Calculate contrast relative to background
        features.contrast = self.contrast_analyzer.calculate_contrast(&features, context);
        
        // Detect novelty
        features.novelty = self.novelty_detector.calculate_novelty(entity_id, &features);
        
        // Calculate motion salience
        features.motion = self.temporal_processor.calculate_motion(entity_id, &features);
        
        // Calculate onset salience
        features.onset = self.temporal_processor.calculate_onset(entity_id);
        
        // Apply adaptation effects
        self.adaptation_tracker.apply_adaptation(&mut features);
        
        // Combine features into total salience
        let total_salience = self.combine_features(&features);
        
        // Calculate capture probability
        let capture_probability = self.calculate_capture_probability(total_salience, context);
        
        let record = SalienceRecord {
            entity_id,
            total_salience,
            feature_contributions: features,
            computed_at: Instant::now(),
            context: context.clone(),
            capture_probability,
        };
        
        // Store in history
        self.store_salience_record(&record);
        
        record
    }
    
    fn extract_basic_features(&self, properties: &StimulusProperties) -> SalienceFeatures {
        SalienceFeatures {
            intensity: properties.intensity.clamp(0.0, 1.0),
            contrast: 0.0, // Will be calculated later
            novelty: 0.0,  // Will be calculated later
            motion: 0.0,   // Will be calculated later
            size: properties.size_ratio.clamp(0.0, 1.0),
            color_pop_out: self.calculate_color_pop_out(properties),
            onset: 0.0,    // Will be calculated later
            semantic_surprise: properties.semantic_unexpectedness.clamp(0.0, 1.0),
        }
    }
    
    fn calculate_color_pop_out(&self, properties: &StimulusProperties) -> f32 {
        // Simplified color pop-out calculation
        // In practice, this would use more sophisticated color space analysis
        let color_distance = properties.color_distinctiveness;
        let saturation_bonus = properties.color_saturation * 0.3;
        
        (color_distance + saturation_bonus).clamp(0.0, 1.0)
    }
    
    fn combine_features(&self, features: &SalienceFeatures) -> f32 {
        match &self.combination_strategy {
            CombinationStrategy::Linear => {
                self.linear_combination(features)
            },
            CombinationStrategy::MaxPooling => {
                self.max_pooling_combination(features)
            },
            CombinationStrategy::WeightedSum { normalization } => {
                self.weighted_sum_combination(features, *normalization)
            },
            CombinationStrategy::NonLinearSigmoid { steepness } => {
                self.sigmoid_combination(features, *steepness)
            },
            CombinationStrategy::CompetitiveWTA { threshold } => {
                self.winner_take_all_combination(features, *threshold)
            },
        }
    }
    
    fn weighted_sum_combination(&self, features: &SalienceFeatures, normalize: bool) -> f32 {
        let feature_values = vec![
            ("intensity", features.intensity),
            ("contrast", features.contrast),
            ("novelty", features.novelty),
            ("motion", features.motion),
            ("size", features.size),
            ("color_pop_out", features.color_pop_out),
            ("onset", features.onset),
            ("semantic_surprise", features.semantic_surprise),
        ];
        
        let weighted_sum: f32 = feature_values.iter()
            .map(|(name, value)| {
                let weight = self.feature_weights.get(*name).unwrap_or(&1.0);
                value * weight
            })
            .sum();
        
        if normalize {
            let total_weights: f32 = self.feature_weights.values().sum();
            weighted_sum / total_weights
        } else {
            weighted_sum.clamp(0.0, 1.0)
        }
    }
    
    fn max_pooling_combination(&self, features: &SalienceFeatures) -> f32 {
        vec![
            features.intensity,
            features.contrast,
            features.novelty,
            features.motion,
            features.size,
            features.color_pop_out,
            features.onset,
            features.semantic_surprise,
        ].into_iter()
            .fold(0.0f32, |max, val| max.max(val))
    }
    
    fn sigmoid_combination(&self, features: &SalienceFeatures, steepness: f32) -> f32 {
        let linear_sum = self.weighted_sum_combination(features, true);
        1.0 / (1.0 + (-steepness * (linear_sum - 0.5)).exp())
    }
    
    fn winner_take_all_combination(&self, features: &SalienceFeatures, threshold: f32) -> f32 {
        let max_feature = self.max_pooling_combination(features);
        
        if max_feature > threshold {
            max_feature
        } else {
            // If no feature is dominant, use weighted average of top features
            let mut sorted_features = vec![
                features.intensity,
                features.contrast,
                features.novelty,
                features.motion,
                features.onset,
            ];
            
            sorted_features.sort_by(|a, b| b.partial_cmp(a).unwrap());
            
            // Average of top 3 features
            sorted_features.iter().take(3).sum::<f32>() / 3.0
        }
    }
    
    fn calculate_capture_probability(&self, salience: f32, context: &SalienceContext) -> f32 {
        // Base probability from salience
        let base_probability = salience;
        
        // Adjust for current attention allocation
        let attention_competition = context.attention_allocation.values().sum::<f32>();
        let competition_factor = 1.0 / (1.0 + attention_competition);
        
        // Adjust for background noise
        let noise_factor = 1.0 - (context.noise_level * 0.3);
        
        // Adjust for adaptation level
        let adaptation_factor = 1.0 - (context.adaptation_level * 0.2);
        
        (base_probability * competition_factor * noise_factor * adaptation_factor).clamp(0.0, 1.0)
    }
    
    fn store_salience_record(&mut self, record: &SalienceRecord) {
        let history = self.salience_history.entry(record.entity_id).or_insert_with(Vec::new);
        history.push(record.clone());
        
        // Keep history manageable
        if history.len() > 50 {
            history.drain(..25);
        }
    }
}

#[derive(Debug, Clone)]
pub struct StimulusProperties {
    pub intensity: f32,
    pub size_ratio: f32,
    pub color_distinctiveness: f32,
    pub color_saturation: f32,
    pub semantic_unexpectedness: f32,
    pub position: (f32, f32),
    pub timestamp: Instant,
}
```

### Step 3: Specialized Feature Processors

```rust
impl NoveltyDetector {
    fn new() -> Self {
        Self {
            expectation_model: HashMap::new(),
            surprise_threshold: 0.3,
            learning_rate: 0.1,
        }
    }
    
    fn calculate_novelty(&mut self, entity_id: EntityId, features: &SalienceFeatures) -> f32 {
        if let Some(expectation) = self.expectation_model.get_mut(&entity_id) {
            // Calculate surprise as deviation from expectation
            let surprise = self.calculate_surprise(features, expectation);
            
            // Update expectation with new observation
            self.update_expectation(expectation, features);
            
            surprise
        } else {
            // First observation is always novel
            let expectation = ExpectationRecord {
                expected_features: features.clone(),
                confidence: 0.1,
                last_observed: Instant::now(),
                observation_count: 1,
            };
            
            self.expectation_model.insert(entity_id, expectation);
            
            0.8 // High novelty for first observation
        }
    }
    
    fn calculate_surprise(&self, observed: &SalienceFeatures, expected: &ExpectationRecord) -> f32 {
        let intensity_diff = (observed.intensity - expected.expected_features.intensity).abs();
        let size_diff = (observed.size - expected.expected_features.size).abs();
        let semantic_diff = (observed.semantic_surprise - expected.expected_features.semantic_surprise).abs();
        
        let total_diff = (intensity_diff + size_diff + semantic_diff) / 3.0;
        
        // Weight by expectation confidence
        let confidence_weighted_surprise = total_diff * expected.confidence;
        
        confidence_weighted_surprise.clamp(0.0, 1.0)
    }
    
    fn update_expectation(&mut self, expectation: &mut ExpectationRecord, observed: &SalienceFeatures) {
        let learning_rate = self.learning_rate / (1.0 + expectation.observation_count as f32 * 0.1);
        
        // Update expected features with exponential moving average
        expectation.expected_features.intensity = 
            (1.0 - learning_rate) * expectation.expected_features.intensity + 
            learning_rate * observed.intensity;
        
        expectation.expected_features.size = 
            (1.0 - learning_rate) * expectation.expected_features.size + 
            learning_rate * observed.size;
        
        expectation.expected_features.semantic_surprise = 
            (1.0 - learning_rate) * expectation.expected_features.semantic_surprise + 
            learning_rate * observed.semantic_surprise;
        
        // Increase confidence with more observations
        expectation.confidence = (expectation.confidence + 0.1).min(1.0);
        expectation.observation_count += 1;
        expectation.last_observed = Instant::now();
    }
}

impl ContrastAnalyzer {
    fn new() -> Self {
        Self {
            background_model: BackgroundModel::new(),
            contrast_kernels: Self::create_contrast_kernels(),
        }
    }
    
    fn create_contrast_kernels() -> Vec<ContrastKernel> {
        vec![
            ContrastKernel {
                feature_type: "intensity".to_string(),
                weight: 1.0,
                threshold: 0.2,
            },
            ContrastKernel {
                feature_type: "size".to_string(),
                weight: 0.8,
                threshold: 0.3,
            },
            ContrastKernel {
                feature_type: "color".to_string(),
                weight: 0.9,
                threshold: 0.25,
            },
        ]
    }
    
    fn calculate_contrast(&mut self, features: &SalienceFeatures, context: &SalienceContext) -> f32 {
        // Update background model
        self.background_model.update(features, context);
        
        let mut total_contrast = 0.0;
        let mut total_weight = 0.0;
        
        for kernel in &self.contrast_kernels {
            let feature_value = match kernel.feature_type.as_str() {
                "intensity" => features.intensity,
                "size" => features.size,
                "color" => features.color_pop_out,
                _ => 0.0,
            };
            
            let background_value = match kernel.feature_type.as_str() {
                "intensity" => self.background_model.mean_intensity,
                "size" => self.background_model.dominant_features.size,
                "color" => self.background_model.dominant_features.color_pop_out,
                _ => 0.0,
            };
            
            let contrast = (feature_value - background_value).abs();
            
            if contrast > kernel.threshold {
                total_contrast += contrast * kernel.weight;
                total_weight += kernel.weight;
            }
        }
        
        if total_weight > 0.0 {
            (total_contrast / total_weight).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

impl BackgroundModel {
    fn new() -> Self {
        Self {
            mean_intensity: 0.5,
            variance: 0.1,
            dominant_features: SalienceFeatures {
                intensity: 0.5,
                contrast: 0.0,
                novelty: 0.0,
                motion: 0.1,
                size: 0.3,
                color_pop_out: 0.2,
                onset: 0.0,
                semantic_surprise: 0.1,
            },
            update_rate: 0.05,
        }
    }
    
    fn update(&mut self, features: &SalienceFeatures, context: &SalienceContext) {
        // Exponential moving average update
        let rate = self.update_rate;
        
        self.mean_intensity = (1.0 - rate) * self.mean_intensity + rate * features.intensity;
        self.dominant_features.size = (1.0 - rate) * self.dominant_features.size + rate * features.size;
        self.dominant_features.color_pop_out = (1.0 - rate) * self.dominant_features.color_pop_out + rate * features.color_pop_out;
        
        // Update variance estimate
        let intensity_diff = features.intensity - self.mean_intensity;
        self.variance = (1.0 - rate) * self.variance + rate * (intensity_diff * intensity_diff);
    }
}

impl AdaptationTracker {
    fn new() -> Self {
        Self {
            baseline_levels: HashMap::new(),
            adaptation_rates: HashMap::new(),
            last_update: Instant::now(),
        }
    }
    
    fn apply_adaptation(&mut self, features: &mut SalienceFeatures) {
        let elapsed = self.last_update.elapsed().as_secs_f32();
        
        // Apply adaptation to each feature
        features.intensity = self.adapt_feature("intensity", features.intensity, elapsed);
        features.contrast = self.adapt_feature("contrast", features.contrast, elapsed);
        features.motion = self.adapt_feature("motion", features.motion, elapsed);
        
        self.last_update = Instant::now();
    }
    
    fn adapt_feature(&mut self, feature_name: &str, current_value: f32, elapsed_time: f32) -> f32 {
        let baseline = self.baseline_levels.entry(feature_name.to_string()).or_insert(current_value);
        let adaptation_rate = self.adaptation_rates.entry(feature_name.to_string()).or_insert(0.1);
        
        // Calculate adaptation effect
        let adaptation_factor = 1.0 - (*adaptation_rate * elapsed_time).min(0.9);
        
        // Update baseline
        *baseline = 0.95 * *baseline + 0.05 * current_value;
        
        // Apply adaptation
        let adapted_value = current_value * adaptation_factor;
        
        adapted_value.clamp(0.0, 1.0)
    }
}

#[derive(Debug)]
pub struct TemporalProcessor {
    previous_observations: HashMap<EntityId, SalienceFeatures>,
    onset_tracker: HashMap<EntityId, Instant>,
}

impl TemporalProcessor {
    fn new() -> Self {
        Self {
            previous_observations: HashMap::new(),
            onset_tracker: HashMap::new(),
        }
    }
    
    fn calculate_motion(&mut self, entity_id: EntityId, current_features: &SalienceFeatures) -> f32 {
        if let Some(previous) = self.previous_observations.get(&entity_id) {
            let intensity_change = (current_features.intensity - previous.intensity).abs();
            let size_change = (current_features.size - previous.size).abs();
            
            let motion_magnitude = (intensity_change + size_change) / 2.0;
            
            // Store current as previous for next calculation
            self.previous_observations.insert(entity_id, current_features.clone());
            
            motion_magnitude.clamp(0.0, 1.0)
        } else {
            // First observation, no motion
            self.previous_observations.insert(entity_id, current_features.clone());
            0.0
        }
    }
    
    fn calculate_onset(&mut self, entity_id: EntityId) -> f32 {
        if let Some(onset_time) = self.onset_tracker.get(&entity_id) {
            // Calculate onset salience based on recency
            let age = onset_time.elapsed().as_secs_f32();
            let onset_salience = (-age / 2.0).exp(); // 2-second half-life
            
            onset_salience.clamp(0.0, 1.0)
        } else {
            // First detection, strong onset
            self.onset_tracker.insert(entity_id, Instant::now());
            1.0
        }
    }
}
```

## File Locations

- `src/cognitive/attention/salience.rs` - Main implementation
- `src/cognitive/attention/mod.rs` - Module exports
- `tests/cognitive/attention/salience_tests.rs` - Test implementation

## Success Criteria

- [ ] SalienceCalculator compiles and runs correctly
- [ ] All feature calculations implemented properly
- [ ] Novelty detection working with adaptation
- [ ] Contrast analysis relative to background
- [ ] Temporal processing for motion and onset
- [ ] Thread-safe concurrent access
- [ ] All tests pass:
  - Feature extraction accuracy
  - Novelty detection behavior
  - Contrast calculation correctness
  - Temporal processing functionality

## Test Requirements

```rust
#[test]
fn test_basic_salience_calculation() {
    let mut calculator = SalienceCalculator::new();
    
    let stimulus = StimulusProperties {
        intensity: 0.8,
        size_ratio: 0.6,
        color_distinctiveness: 0.7,
        color_saturation: 0.9,
        semantic_unexpectedness: 0.3,
        position: (0.5, 0.5),
        timestamp: Instant::now(),
    };
    
    let context = SalienceContext {
        background_activity: 0.3,
        adaptation_level: 0.2,
        attention_allocation: HashMap::new(),
        temporal_expectation: 0.5,
        noise_level: 0.1,
    };
    
    let record = calculator.calculate_salience(EntityId(1), &stimulus, &context);
    
    assert!(record.total_salience > 0.0);
    assert!(record.total_salience <= 1.0);
    assert!(record.capture_probability > 0.0);
}

#[test]
fn test_novelty_detection() {
    let mut calculator = SalienceCalculator::new();
    
    let context = SalienceContext::default();
    
    // First observation should be novel
    let stimulus1 = StimulusProperties {
        intensity: 0.5,
        size_ratio: 0.4,
        color_distinctiveness: 0.3,
        color_saturation: 0.5,
        semantic_unexpectedness: 0.2,
        position: (0.5, 0.5),
        timestamp: Instant::now(),
    };
    
    let record1 = calculator.calculate_salience(EntityId(1), &stimulus1, &context);
    let initial_novelty = record1.feature_contributions.novelty;
    
    // Second similar observation should be less novel
    let stimulus2 = StimulusProperties {
        intensity: 0.52, // Slightly different
        size_ratio: 0.41,
        color_distinctiveness: 0.31,
        color_saturation: 0.51,
        semantic_unexpectedness: 0.21,
        position: (0.5, 0.5),
        timestamp: Instant::now(),
    };
    
    let record2 = calculator.calculate_salience(EntityId(1), &stimulus2, &context);
    let second_novelty = record2.feature_contributions.novelty;
    
    assert!(second_novelty < initial_novelty);
}

#[test]
fn test_contrast_calculation() {
    let mut calculator = SalienceCalculator::new();
    
    let context = SalienceContext {
        background_activity: 0.2, // Low background
        ..Default::default()
    };
    
    // High intensity stimulus against low background
    let high_contrast_stimulus = StimulusProperties {
        intensity: 0.9,
        size_ratio: 0.8,
        color_distinctiveness: 0.7,
        color_saturation: 0.8,
        semantic_unexpectedness: 0.1,
        position: (0.5, 0.5),
        timestamp: Instant::now(),
    };
    
    let record = calculator.calculate_salience(EntityId(1), &high_contrast_stimulus, &context);
    
    // Should have high contrast due to intensity difference
    assert!(record.feature_contributions.contrast > 0.5);
}

#[test]
fn test_motion_detection() {
    let mut calculator = SalienceCalculator::new();
    
    let context = SalienceContext::default();
    
    // First observation
    let stimulus1 = StimulusProperties {
        intensity: 0.5,
        size_ratio: 0.3,
        color_distinctiveness: 0.4,
        color_saturation: 0.5,
        semantic_unexpectedness: 0.2,
        position: (0.3, 0.3),
        timestamp: Instant::now(),
    };
    
    let record1 = calculator.calculate_salience(EntityId(1), &stimulus1, &context);
    
    // Second observation with significant change
    let stimulus2 = StimulusProperties {
        intensity: 0.8, // Changed significantly
        size_ratio: 0.7, // Changed significantly
        color_distinctiveness: 0.4,
        color_saturation: 0.5,
        semantic_unexpectedness: 0.2,
        position: (0.7, 0.7), // Position change (not directly measured in this simple test)
        timestamp: Instant::now(),
    };
    
    let record2 = calculator.calculate_salience(EntityId(1), &stimulus2, &context);
    
    // Should detect motion due to changes
    assert!(record2.feature_contributions.motion > 0.2);
}

#[test]
fn test_onset_salience() {
    let mut calculator = SalienceCalculator::new();
    
    let context = SalienceContext::default();
    
    let stimulus = StimulusProperties {
        intensity: 0.6,
        size_ratio: 0.4,
        color_distinctiveness: 0.3,
        color_saturation: 0.5,
        semantic_unexpectedness: 0.2,
        position: (0.5, 0.5),
        timestamp: Instant::now(),
    };
    
    // First appearance should have high onset salience
    let record = calculator.calculate_salience(EntityId(1), &stimulus, &context);
    
    assert!(record.feature_contributions.onset > 0.8);
}
```

## Quality Gates

- [ ] Feature calculations mathematically sound
- [ ] Novelty detection shows adaptation over time
- [ ] Contrast calculation relative to background
- [ ] Performance < 10ms for salience calculation
- [ ] Thread-safe concurrent access verified
- [ ] Memory efficient for continuous operation

## Next Task

Upon completion, proceed to **18_attention_tests.md**