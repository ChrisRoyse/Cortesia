# Micro Task 15: Attention Weighting

**Priority**: HIGH  
**Estimated Time**: 35 minutes  
**Dependencies**: 14_working_memory.md completed  
**Skills Required**: Mathematical modeling, attention algorithms, weight calculation

## Objective

Implement sophisticated attention weight calculation algorithms that combine top-down goals, bottom-up salience, and contextual factors to determine optimal attention allocation.

## Context

Attention weighting is the computational mechanism that determines how attention is distributed across multiple competing targets. This system combines multiple factors including goal relevance, stimulus salience, recency, novelty, and contextual importance to compute optimal attention weights.

## Specifications

### Core Weighting Components

1. **AttentionWeightCalculator struct**
   - Multi-factor weight computation
   - Dynamic weight normalization
   - Context-sensitive adjustments
   - Temporal weight smoothing

2. **WeightingFactors struct**
   - Goal relevance scoring
   - Salience computation
   - Recency and frequency effects
   - Novelty detection

3. **WeightingStrategy enum**
   - Linear combination
   - Multiplicative combination
   - Sigmoid activation
   - Competitive normalization

### Performance Requirements

- Weight calculation latency < 5ms
- Support 20+ simultaneous targets
- Smooth weight transitions
- Numerical stability maintained
- Real-time adaptation to context changes

## Implementation Guide

### Step 1: Core Weighting Types

```rust
// File: src/cognitive/attention/weighting.rs

use std::collections::HashMap;
use std::time::{Duration, Instant};
use crate::core::types::{EntityId, NodeId};
use crate::cognitive::attention::{AttentionSource, AttentionTarget};

#[derive(Debug, Clone)]
pub struct WeightingFactors {
    pub goal_relevance: f32,      // 0.0 to 1.0
    pub stimulus_salience: f32,   // 0.0 to 1.0
    pub recency: f32,            // 0.0 to 1.0 (recent = higher)
    pub frequency: f32,          // 0.0 to 1.0 (frequent = higher)
    pub novelty: f32,            // 0.0 to 1.0 (novel = higher)
    pub context_relevance: f32,  // 0.0 to 1.0
    pub persistence: f32,        // 0.0 to 1.0
    pub uncertainty: f32,        // 0.0 to 1.0 (uncertain = higher)
}

#[derive(Debug, Clone)]
pub enum WeightingStrategy {
    Linear { weights: Vec<f32> },
    Multiplicative { exponents: Vec<f32> },
    Sigmoid { steepness: f32, threshold: f32 },
    Competitive { competition_strength: f32 },
    Adaptive { learning_rate: f32 },
}

#[derive(Debug)]
pub struct AttentionWeightCalculator {
    strategy: WeightingStrategy,
    factor_weights: HashMap<String, f32>,
    context_state: ContextState,
    temporal_smoother: TemporalSmoother,
    weight_history: Vec<WeightSnapshot>,
    normalization_method: NormalizationMethod,
}

#[derive(Debug, Clone)]
pub struct ContextState {
    pub current_goal: Option<String>,
    pub task_difficulty: f32,
    pub cognitive_load: f32,
    pub environmental_noise: f32,
    pub time_pressure: f32,
    pub domain_context: String,
}

#[derive(Debug)]
pub struct TemporalSmoother {
    smoothing_factor: f32,
    previous_weights: HashMap<EntityId, f32>,
    update_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct WeightSnapshot {
    pub timestamp: Instant,
    pub entity_weights: HashMap<EntityId, f32>,
    pub total_weight: f32,
    pub context: ContextState,
}

#[derive(Debug, Clone)]
pub enum NormalizationMethod {
    SoftMax { temperature: f32 },
    Linear,
    WinnerTakeAll { threshold: f32 },
    ProportionalToSum,
}
```

### Step 2: Weight Calculation Implementation

```rust
impl AttentionWeightCalculator {
    pub fn new(strategy: WeightingStrategy) -> Self {
        let mut factor_weights = HashMap::new();
        
        // Default factor importance weights
        factor_weights.insert("goal_relevance".to_string(), 0.3);
        factor_weights.insert("stimulus_salience".to_string(), 0.25);
        factor_weights.insert("recency".to_string(), 0.15);
        factor_weights.insert("frequency".to_string(), 0.1);
        factor_weights.insert("novelty".to_string(), 0.1);
        factor_weights.insert("context_relevance".to_string(), 0.1);
        
        Self {
            strategy,
            factor_weights,
            context_state: ContextState::default(),
            temporal_smoother: TemporalSmoother::new(0.7),
            weight_history: Vec::new(),
            normalization_method: NormalizationMethod::SoftMax { temperature: 1.0 },
        }
    }
    
    pub fn calculate_attention_weights(
        &mut self, 
        targets: &[AttentionTarget],
        context: &ContextState
    ) -> HashMap<EntityId, f32> {
        
        // Update context state
        self.context_state = context.clone();
        
        // Calculate raw weights for each target
        let mut raw_weights = HashMap::new();
        
        for target in targets {
            let factors = self.extract_weighting_factors(target, context);
            let raw_weight = self.compute_raw_weight(&factors);
            raw_weights.insert(target.entity_id, raw_weight);
        }
        
        // Apply strategy-specific processing
        let processed_weights = self.apply_weighting_strategy(&raw_weights);
        
        // Apply temporal smoothing
        let smoothed_weights = self.temporal_smoother.smooth_weights(&processed_weights);
        
        // Normalize weights
        let normalized_weights = self.normalize_weights(&smoothed_weights);
        
        // Store snapshot for history
        self.store_weight_snapshot(&normalized_weights);
        
        normalized_weights
    }
    
    fn extract_weighting_factors(&self, target: &AttentionTarget, context: &ContextState) -> WeightingFactors {
        let now = Instant::now();
        
        // Goal relevance based on target source and current goals
        let goal_relevance = self.calculate_goal_relevance(target, context);
        
        // Stimulus salience from attention source
        let stimulus_salience = match &target.source {
            AttentionSource::BottomUp { stimulus_strength, .. } => *stimulus_strength,
            AttentionSource::TopDown { priority, .. } => *priority * 0.8,
            AttentionSource::Maintenance { importance } => *importance * 0.6,
        };
        
        // Recency calculation (higher for more recent)
        let recency = {
            let age = target.last_accessed.elapsed().as_secs_f32();
            (-age / 30.0).exp() // 30 second half-life
        };
        
        // Frequency based on access count (mock implementation)
        let frequency = (target.attention_strength * 10.0).min(1.0);
        
        // Novelty from source information
        let novelty = match &target.source {
            AttentionSource::BottomUp { novelty, .. } => *novelty,
            _ => 0.5, // Default moderate novelty
        };
        
        // Context relevance
        let context_relevance = self.calculate_context_relevance(target, context);
        
        // Persistence based on how long attention has been maintained
        let persistence = {
            let duration = target.created_at.elapsed().as_secs_f32();
            1.0 - (-duration / 60.0).exp() // Increases with time
        };
        
        // Uncertainty based on attention strength variance
        let uncertainty = 1.0 - target.attention_strength; // Lower strength = higher uncertainty
        
        WeightingFactors {
            goal_relevance,
            stimulus_salience,
            recency,
            frequency,
            novelty,
            context_relevance,
            persistence,
            uncertainty,
        }
    }
    
    fn calculate_goal_relevance(&self, target: &AttentionTarget, context: &ContextState) -> f32 {
        match (&target.source, &context.current_goal) {
            (AttentionSource::TopDown { goal, .. }, Some(current_goal)) => {
                // Calculate semantic similarity between goals (simplified)
                if goal.contains(current_goal) || current_goal.contains(goal) {
                    0.9
                } else {
                    0.3
                }
            },
            (AttentionSource::TopDown { .. }, None) => 0.6,
            _ => 0.4, // Bottom-up or maintenance sources have moderate goal relevance
        }
    }
    
    fn calculate_context_relevance(&self, target: &AttentionTarget, context: &ContextState) -> f32 {
        // Context relevance based on current cognitive state
        let base_relevance = match &target.source {
            AttentionSource::TopDown { .. } => 0.7,
            AttentionSource::BottomUp { .. } => {
                // Higher relevance if low cognitive load (can process bottom-up signals)
                1.0 - context.cognitive_load
            },
            AttentionSource::Maintenance { .. } => 0.5,
        };
        
        // Adjust for environmental factors
        let noise_adjustment = 1.0 - (context.environmental_noise * 0.3);
        let pressure_adjustment = if context.time_pressure > 0.7 {
            match &target.source {
                AttentionSource::TopDown { .. } => 1.2, // Boost goal-directed under pressure
                _ => 0.8, // Reduce others
            }
        } else {
            1.0
        };
        
        (base_relevance * noise_adjustment * pressure_adjustment).clamp(0.0, 1.0)
    }
    
    fn compute_raw_weight(&self, factors: &WeightingFactors) -> f32 {
        let factor_values = vec![
            ("goal_relevance", factors.goal_relevance),
            ("stimulus_salience", factors.stimulus_salience),
            ("recency", factors.recency),
            ("frequency", factors.frequency),
            ("novelty", factors.novelty),
            ("context_relevance", factors.context_relevance),
        ];
        
        let weighted_sum: f32 = factor_values.iter()
            .map(|(name, value)| {
                let weight = self.factor_weights.get(*name).unwrap_or(&1.0);
                value * weight
            })
            .sum();
        
        // Add uncertainty bonus (attention to uncertain things)
        let uncertainty_bonus = factors.uncertainty * 0.1;
        
        // Add persistence bonus (maintain ongoing attention)
        let persistence_bonus = factors.persistence * 0.05;
        
        (weighted_sum + uncertainty_bonus + persistence_bonus).clamp(0.0, 1.0)
    }
}
```

### Step 3: Strategy Application and Normalization

```rust
impl AttentionWeightCalculator {
    fn apply_weighting_strategy(&self, raw_weights: &HashMap<EntityId, f32>) -> HashMap<EntityId, f32> {
        match &self.strategy {
            WeightingStrategy::Linear { weights } => {
                self.apply_linear_strategy(raw_weights, weights)
            },
            WeightingStrategy::Multiplicative { exponents } => {
                self.apply_multiplicative_strategy(raw_weights, exponents)
            },
            WeightingStrategy::Sigmoid { steepness, threshold } => {
                self.apply_sigmoid_strategy(raw_weights, *steepness, *threshold)
            },
            WeightingStrategy::Competitive { competition_strength } => {
                self.apply_competitive_strategy(raw_weights, *competition_strength)
            },
            WeightingStrategy::Adaptive { learning_rate } => {
                self.apply_adaptive_strategy(raw_weights, *learning_rate)
            },
        }
    }
    
    fn apply_linear_strategy(&self, raw_weights: &HashMap<EntityId, f32>, _weights: &[f32]) -> HashMap<EntityId, f32> {
        // Simple linear pass-through (weights could be used for different linear combinations)
        raw_weights.clone()
    }
    
    fn apply_sigmoid_strategy(&self, raw_weights: &HashMap<EntityId, f32>, steepness: f32, threshold: f32) -> HashMap<EntityId, f32> {
        raw_weights.iter()
            .map(|(entity_id, weight)| {
                let sigmoid_weight = 1.0 / (1.0 + ((-steepness * (weight - threshold)).exp()));
                (*entity_id, sigmoid_weight)
            })
            .collect()
    }
    
    fn apply_competitive_strategy(&self, raw_weights: &HashMap<EntityId, f32>, competition_strength: f32) -> HashMap<EntityId, f32> {
        let max_weight = raw_weights.values().fold(0.0f32, |max, &w| max.max(w));
        
        raw_weights.iter()
            .map(|(entity_id, weight)| {
                // Winner-take-all competition
                let relative_strength = weight / max_weight.max(0.001);
                let competitive_weight = relative_strength.powf(competition_strength);
                (*entity_id, competitive_weight)
            })
            .collect()
    }
    
    fn apply_adaptive_strategy(&self, raw_weights: &HashMap<EntityId, f32>, learning_rate: f32) -> HashMap<EntityId, f32> {
        // Adaptive strategy adjusts weights based on recent history
        let mut adaptive_weights = raw_weights.clone();
        
        if let Some(last_snapshot) = self.weight_history.last() {
            for (entity_id, current_weight) in adaptive_weights.iter_mut() {
                if let Some(previous_weight) = last_snapshot.entity_weights.get(entity_id) {
                    // Exponential moving average
                    *current_weight = (1.0 - learning_rate) * previous_weight + learning_rate * *current_weight;
                }
            }
        }
        
        adaptive_weights
    }
    
    fn normalize_weights(&self, weights: &HashMap<EntityId, f32>) -> HashMap<EntityId, f32> {
        match &self.normalization_method {
            NormalizationMethod::SoftMax { temperature } => {
                self.softmax_normalize(weights, *temperature)
            },
            NormalizationMethod::Linear => {
                self.linear_normalize(weights)
            },
            NormalizationMethod::WinnerTakeAll { threshold } => {
                self.winner_take_all_normalize(weights, *threshold)
            },
            NormalizationMethod::ProportionalToSum => {
                self.proportional_normalize(weights)
            },
        }
    }
    
    fn softmax_normalize(&self, weights: &HashMap<EntityId, f32>, temperature: f32) -> HashMap<EntityId, f32> {
        // Apply temperature scaling and softmax
        let scaled_weights: Vec<f32> = weights.values()
            .map(|w| (w / temperature).exp())
            .collect();
        
        let sum: f32 = scaled_weights.iter().sum();
        
        weights.keys()
            .zip(scaled_weights.iter())
            .map(|(entity_id, scaled_weight)| {
                (*entity_id, scaled_weight / sum)
            })
            .collect()
    }
    
    fn linear_normalize(&self, weights: &HashMap<EntityId, f32>) -> HashMap<EntityId, f32> {
        let sum: f32 = weights.values().sum();
        let max_weight = weights.values().fold(0.0f32, |max, &w| max.max(w));
        
        weights.iter()
            .map(|(entity_id, weight)| {
                let normalized = if sum > 0.0 { weight / sum } else { 1.0 / weights.len() as f32 };
                (*entity_id, normalized)
            })
            .collect()
    }
    
    fn winner_take_all_normalize(&self, weights: &HashMap<EntityId, f32>, threshold: f32) -> HashMap<EntityId, f32> {
        let max_weight = weights.values().fold(0.0f32, |max, &w| max.max(w));
        let winner_threshold = max_weight * threshold;
        
        weights.iter()
            .map(|(entity_id, weight)| {
                let normalized_weight = if *weight >= winner_threshold { 1.0 } else { 0.0 };
                (*entity_id, normalized_weight)
            })
            .collect()
    }
    
    fn proportional_normalize(&self, weights: &HashMap<EntityId, f32>) -> HashMap<EntityId, f32> {
        let total: f32 = weights.values().sum();
        
        if total > 0.0 {
            weights.iter()
                .map(|(entity_id, weight)| (*entity_id, weight / total))
                .collect()
        } else {
            // Equal distribution if all weights are zero
            let equal_weight = 1.0 / weights.len() as f32;
            weights.keys()
                .map(|entity_id| (*entity_id, equal_weight))
                .collect()
        }
    }
    
    fn store_weight_snapshot(&mut self, weights: &HashMap<EntityId, f32>) {
        let snapshot = WeightSnapshot {
            timestamp: Instant::now(),
            entity_weights: weights.clone(),
            total_weight: weights.values().sum(),
            context: self.context_state.clone(),
        };
        
        self.weight_history.push(snapshot);
        
        // Keep history manageable
        if self.weight_history.len() > 100 {
            self.weight_history.drain(..50);
        }
    }
}

impl TemporalSmoother {
    fn new(smoothing_factor: f32) -> Self {
        Self {
            smoothing_factor: smoothing_factor.clamp(0.0, 1.0),
            previous_weights: HashMap::new(),
            update_threshold: 0.05,
        }
    }
    
    fn smooth_weights(&mut self, current_weights: &HashMap<EntityId, f32>) -> HashMap<EntityId, f32> {
        let mut smoothed_weights = HashMap::new();
        
        for (entity_id, current_weight) in current_weights {
            let smoothed_weight = if let Some(previous_weight) = self.previous_weights.get(entity_id) {
                // Exponential moving average
                let smoothed = self.smoothing_factor * previous_weight + (1.0 - self.smoothing_factor) * current_weight;
                
                // Only update if change is significant
                if (smoothed - previous_weight).abs() > self.update_threshold {
                    smoothed
                } else {
                    *previous_weight
                }
            } else {
                *current_weight
            };
            
            smoothed_weights.insert(*entity_id, smoothed_weight);
        }
        
        // Update previous weights
        self.previous_weights = smoothed_weights.clone();
        
        smoothed_weights
    }
}

impl Default for ContextState {
    fn default() -> Self {
        Self {
            current_goal: None,
            task_difficulty: 0.5,
            cognitive_load: 0.3,
            environmental_noise: 0.2,
            time_pressure: 0.1,
            domain_context: "general".to_string(),
        }
    }
}
```

## File Locations

- `src/cognitive/attention/weighting.rs` - Main implementation
- `src/cognitive/attention/mod.rs` - Module exports
- `tests/cognitive/attention/weighting_tests.rs` - Test implementation

## Success Criteria

- [ ] AttentionWeightCalculator compiles and runs
- [ ] Multi-factor weight computation working correctly
- [ ] All normalization methods implemented
- [ ] Temporal smoothing prevents oscillations
- [ ] Strategy pattern working for different algorithms
- [ ] Thread-safe concurrent access
- [ ] All tests pass:
  - Weight calculation accuracy
  - Normalization correctness
  - Temporal smoothing behavior
  - Strategy application logic

## Test Requirements

```rust
#[test]
fn test_weight_calculation_accuracy() {
    let mut calculator = AttentionWeightCalculator::new(
        WeightingStrategy::Linear { weights: vec![1.0] }
    );
    
    let targets = vec![
        AttentionTarget {
            entity_id: EntityId(1),
            attention_strength: 0.8,
            source: AttentionSource::TopDown { goal: "important".into(), priority: 0.9 },
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            persistence: Duration::from_secs(30),
            decay_rate: 0.1,
        },
        AttentionTarget {
            entity_id: EntityId(2),
            attention_strength: 0.4,
            source: AttentionSource::BottomUp { stimulus_strength: 0.3, novelty: 0.5 },
            created_at: Instant::now(),
            last_accessed: Instant::now(),
            persistence: Duration::from_secs(30),
            decay_rate: 0.1,
        },
    ];
    
    let context = ContextState {
        current_goal: Some("important".into()),
        ..Default::default()
    };
    
    let weights = calculator.calculate_attention_weights(&targets, &context);
    
    // Target 1 should have higher weight due to goal alignment
    let weight1 = weights.get(&EntityId(1)).unwrap();
    let weight2 = weights.get(&EntityId(2)).unwrap();
    
    assert!(weight1 > weight2);
    
    // Weights should sum to approximately 1.0
    let total_weight: f32 = weights.values().sum();
    assert!((total_weight - 1.0).abs() < 0.01);
}

#[test]
fn test_softmax_normalization() {
    let calculator = AttentionWeightCalculator::new(
        WeightingStrategy::Linear { weights: vec![1.0] }
    );
    
    let mut raw_weights = HashMap::new();
    raw_weights.insert(EntityId(1), 0.8);
    raw_weights.insert(EntityId(2), 0.6);
    raw_weights.insert(EntityId(3), 0.4);
    
    let normalized = calculator.softmax_normalize(&raw_weights, 1.0);
    
    // Check that weights sum to 1.0
    let total: f32 = normalized.values().sum();
    assert!((total - 1.0).abs() < 0.001);
    
    // Check that relative ordering is preserved
    let weight1 = normalized.get(&EntityId(1)).unwrap();
    let weight2 = normalized.get(&EntityId(2)).unwrap();
    let weight3 = normalized.get(&EntityId(3)).unwrap();
    
    assert!(weight1 > weight2);
    assert!(weight2 > weight3);
}

#[test]
fn test_temporal_smoothing() {
    let mut smoother = TemporalSmoother::new(0.7);
    
    let mut weights1 = HashMap::new();
    weights1.insert(EntityId(1), 0.8);
    weights1.insert(EntityId(2), 0.2);
    
    let smoothed1 = smoother.smooth_weights(&weights1);
    
    // First call should return input weights
    assert_eq!(smoothed1.get(&EntityId(1)), Some(&0.8));
    
    let mut weights2 = HashMap::new();
    weights2.insert(EntityId(1), 0.4);
    weights2.insert(EntityId(2), 0.6);
    
    let smoothed2 = smoother.smooth_weights(&weights2);
    
    // Second call should be smoothed
    let smoothed_weight1 = smoothed2.get(&EntityId(1)).unwrap();
    assert!(*smoothed_weight1 > 0.4); // Should be between 0.4 and 0.8
    assert!(*smoothed_weight1 < 0.8);
}

#[test]
fn test_competitive_strategy() {
    let calculator = AttentionWeightCalculator::new(
        WeightingStrategy::Competitive { competition_strength: 2.0 }
    );
    
    let mut raw_weights = HashMap::new();
    raw_weights.insert(EntityId(1), 0.8);
    raw_weights.insert(EntityId(2), 0.7);
    raw_weights.insert(EntityId(3), 0.3);
    
    let competitive_weights = calculator.apply_competitive_strategy(&raw_weights, 2.0);
    
    // Highest weight should be enhanced
    let weight1 = competitive_weights.get(&EntityId(1)).unwrap();
    let weight2 = competitive_weights.get(&EntityId(2)).unwrap();
    let weight3 = competitive_weights.get(&EntityId(3)).unwrap();
    
    // Competition should amplify differences
    assert!(weight1 > &0.8);
    assert!(weight3 < &0.3);
}
```

## Quality Gates

- [ ] Weight calculations numerically stable
- [ ] All normalization methods sum to 1.0
- [ ] Temporal smoothing prevents oscillations
- [ ] Performance < 5ms for 20+ targets
- [ ] Thread-safe concurrent access verified
- [ ] No division by zero or NaN values

## Next Task

Upon completion, proceed to **16_focus_switching.md**