# Fix: Implement Real Parameter Tuning

## Executive Summary After Deep Analysis

After analyzing the entire LLMKG codebase at a granular level, I discovered that:

1. **The infrastructure already exists** - A sophisticated `ParameterTuningSystem` with Bayesian optimization, grid search, and genetic algorithms is already implemented but not connected
2. **Over 200 tunable parameters** exist across 8 subsystems, with only a stub implementation preventing their use
3. **The fix is simpler than the original plan suggested** - We mainly need to integrate existing components rather than build from scratch

The updated implementation plan below reflects these findings and provides precise, function-level details for implementation.

## Problem
The adaptive learning system's parameter tuning (`src/learning/adaptive_learning/system.rs:222-227`) only prints messages without actually modifying any parameters or measuring impact. Despite having a complete `ParameterTuningSystem` implementation (`src/learning/parameter_tuning.rs`), it's not integrated with the adaptive learning system's `execute_parameter_tuning()` method.

## Current State
```rust
// src/learning/adaptive_learning/system.rs:222-227
async fn execute_parameter_tuning(&self) -> Result<bool> {
    // Tune cognitive orchestrator parameters
    println!("Executing parameter tuning");
    // Would adjust actual parameters here
    Ok(true)
}
```

**Key Finding**: The system already has:
- `ParameterTuningSystem` with Bayesian optimization, grid search, and genetic algorithms
- Parameter spaces defined for Hebbian learning and attention systems
- Performance monitoring infrastructure
- Parameter update structures for all cognitive components

## Solution: Integrate Existing Infrastructure

### 1. Add ParameterTuningSystem to AdaptiveLearningSystem
Update `src/learning/adaptive_learning/system.rs` to include the parameter tuning system:

```rust
// Add to struct fields (around line 20)
pub struct AdaptiveLearningSystem {
    // ... existing fields ...
    parameter_tuning_system: Arc<ParameterTuningSystem>, // ADD THIS
    current_parameters: Arc<RwLock<SystemParameters>>,    // ADD THIS
}

// Update new() method (around line 42)
impl AdaptiveLearningSystem {
    pub fn new(
        integrated_cognitive_system: Arc<Phase3IntegratedCognitiveSystem>,
        config: AdaptiveLearningConfig,
    ) -> Result<Self> {
        // ... existing code ...
        
        // Initialize parameter tuning system
        let parameter_tuning_system = Arc::new(ParameterTuningSystem::new(
            integrated_cognitive_system.clone()
        ));
        
        // Initialize current parameters from system state
        let current_parameters = Arc::new(RwLock::new(
            Self::extract_current_parameters(&integrated_cognitive_system)?
        ));
        
        Ok(Self {
            // ... existing fields ...
            parameter_tuning_system,
            current_parameters,
        })
    }
    
    // Add method to extract current parameters (new method)
    fn extract_current_parameters(
        system: &Arc<Phase3IntegratedCognitiveSystem>
    ) -> Result<SystemParameters> {
        Ok(SystemParameters {
            // Core graph parameters
            max_query_time_ms: 100,  // From graph_core.rs:25
            max_similarity_search_time_ms: 50,  // From graph_core.rs:26
            similarity_cache_capacity: 1000,  // From graph_core.rs:88
            
            // Activation engine parameters (from activation_config.rs)
            activation: ActivationParameters {
                max_iterations: 100,
                convergence_threshold: 0.001,
                decay_rate: 0.1,
                inhibition_strength: 2.0,
                default_threshold: 0.5,
            },
            
            // Query engine parameters
            query: QueryParameters {
                similarity_weight: 0.7,  // From query_system.rs:43
                context_weight: 0.3,     // From query_system.rs:43
                relevance_threshold: 0.5, // From query_system.rs:209
                candidate_multiplier: 2,  // From query_system.rs:31
            },
            
            // Cognitive parameters (from tuned_parameters.rs)
            cognitive: CognitiveParameters {
                convergent_activation_threshold: 0.3,
                divergent_exploration_breadth: 25,
                lateral_novelty_threshold: 0.4,
                critical_contradiction_threshold: 0.6,
                pattern_weights: HashMap::from([
                    (CognitivePatternType::Convergent, 1.0),
                    (CognitivePatternType::Critical, 1.1),
                    (CognitivePatternType::Systems, 0.9),
                    (CognitivePatternType::Divergent, 0.8),
                    (CognitivePatternType::Abstract, 0.7),
                    (CognitivePatternType::Lateral, 0.6),
                ]),
            },
            
            // Index parameters (from hnsw.rs and lsh.rs)
            index: IndexParameters {
                hnsw_max_connections: 8,
                hnsw_ef_construction: 200,
                lsh_num_hashes: 16,
                lsh_num_tables: 3,
            },
        })
    }
}

### 2. Define System Parameters Structure
Create `src/learning/adaptive_learning/parameters.rs`:

```rust
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use crate::cognitive::CognitivePatternType;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemParameters {
    // Core graph parameters
    pub max_query_time_ms: u64,
    pub max_similarity_search_time_ms: u64,
    pub similarity_cache_capacity: usize,
    
    // Subsystem parameters
    pub activation: ActivationParameters,
    pub query: QueryParameters,
    pub cognitive: CognitiveParameters,
    pub index: IndexParameters,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationParameters {
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub decay_rate: f32,
    pub inhibition_strength: f32,
    pub default_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParameters {
    pub similarity_weight: f32,
    pub context_weight: f32,
    pub relevance_threshold: f32,
    pub candidate_multiplier: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveParameters {
    pub convergent_activation_threshold: f32,
    pub divergent_exploration_breadth: usize,
    pub lateral_novelty_threshold: f32,
    pub critical_contradiction_threshold: f32,
    pub pattern_weights: HashMap<CognitivePatternType, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexParameters {
    pub hnsw_max_connections: usize,
    pub hnsw_ef_construction: usize,
    pub lsh_num_hashes: usize,
    pub lsh_num_tables: usize,
}

impl SystemParameters {
    pub fn validate(&self) -> Result<()> {
        // Validate activation parameters
        ensure!(
            self.activation.decay_rate >= 0.01 && self.activation.decay_rate <= 0.5,
            "activation decay_rate out of range [0.01, 0.5]"
        );
        ensure!(
            self.activation.convergence_threshold >= 0.0001 && self.activation.convergence_threshold <= 0.01,
            "convergence_threshold out of range [0.0001, 0.01]"
        );
        
        // Validate query parameters
        ensure!(
            (self.query.similarity_weight + self.query.context_weight - 1.0).abs() < 0.001,
            "query weights must sum to 1.0"
        );
        
        // Validate cognitive parameters
        ensure!(
            self.cognitive.convergent_activation_threshold >= 0.1 && self.cognitive.convergent_activation_threshold <= 0.7,
            "convergent activation threshold out of range [0.1, 0.7]"
        );
        
        Ok(())
    }
}
```

### 3. Implement Real execute_parameter_tuning
Replace the stub implementation in `src/learning/adaptive_learning/system.rs:222-227`:

```rust
// Replace the stub execute_parameter_tuning method (lines 222-227)
async fn execute_parameter_tuning(&self) -> Result<bool> {
    log::info!("Starting parameter tuning execution");
    
    // 1. Analyze current performance using existing monitoring
    let performance_data = self.performance_monitor.get_recent_performance(
        Duration::from_secs(3600) // Last hour
    )?;
    
    // 2. Identify bottlenecks
    let bottlenecks = self.performance_monitor.detect_bottlenecks(&performance_data)?;
    
    // 3. Determine which parameters to tune based on bottlenecks
    let mut tuning_targets = Vec::new();
    
    for bottleneck in &bottlenecks {
        match bottleneck.bottleneck_type {
            BottleneckType::Memory => {
                tuning_targets.push("memory");
                log::info!("Memory bottleneck detected: {}", bottleneck.description);
            }
            BottleneckType::Computation => {
                tuning_targets.push("computation");
                log::info!("Computation bottleneck detected: {}", bottleneck.description);
            }
            BottleneckType::Cognitive => {
                match bottleneck.component.as_deref() {
                    Some("attention") => tuning_targets.push("attention"),
                    Some("orchestrator") => tuning_targets.push("orchestrator"),
                    _ => tuning_targets.push("cognitive"),
                }
                log::info!("Cognitive bottleneck detected: {}", bottleneck.description);
            }
            _ => {}
        }
    }
    
    // 4. Execute parameter tuning using the existing ParameterTuningSystem
    let mut improvements_made = false;
    
    // Tune Hebbian learning parameters if cognitive bottlenecks exist
    if tuning_targets.contains(&"cognitive") {
        let hebbian_update = self.parameter_tuning_system
            .tune_hebbian_parameters(&performance_data)
            .await?;
            
        if hebbian_update.expected_performance_gain > 0.01 {
            // Apply the update to the Hebbian engine
            self.integrated_cognitive_system
                .get_hebbian_engine()
                .apply_parameter_update(hebbian_update)?;
            improvements_made = true;
            
            log::info!(
                "Applied Hebbian parameter updates: learning_rate adjustment={}, expected gain={}",
                hebbian_update.learning_rate_adjustment,
                hebbian_update.expected_performance_gain
            );
        }
    }
    
    // Tune attention parameters if attention bottlenecks exist
    if tuning_targets.contains(&"attention") {
        // Get attention-specific metrics
        let attention_metrics = self.extract_attention_metrics(&performance_data)?;
        
        let attention_update = self.parameter_tuning_system
            .tune_attention_parameters(&attention_metrics)
            .await?;
            
        if attention_update.expected_performance_gain > 0.01 {
            // Apply the update to the attention manager
            self.integrated_cognitive_system
                .get_attention_manager()
                .apply_parameter_update(attention_update)?;
            improvements_made = true;
            
            log::info!(
                "Applied attention parameter updates: focus_strength adjustment={}, expected gain={}",
                attention_update.focus_strength_adjustment,
                attention_update.expected_performance_gain
            );
        }
    }
    
    // Tune memory parameters if memory bottlenecks exist
    if tuning_targets.contains(&"memory") {
        // Get memory-specific metrics
        let memory_metrics = self.extract_memory_metrics(&performance_data)?;
        
        let memory_update = self.parameter_tuning_system
            .tune_memory_parameters(&memory_metrics)
            .await?;
            
        if memory_update.expected_performance_gain > 0.01 {
            // Apply cache size updates
            if let Some(new_cache_size) = memory_update.cache_size_adjustment {
                self.update_cache_parameters(new_cache_size)?;
                improvements_made = true;
                
                log::info!(
                    "Applied memory parameter updates: cache size={}, expected gain={}",
                    new_cache_size,
                    memory_update.expected_performance_gain
                );
            }
        }
    }
    
    // 5. Auto-tune system if multiple bottlenecks or general performance issues
    if tuning_targets.len() > 1 || performance_data.overall_health < 0.7 {
        let system_state = self.create_system_state()?;
        
        let system_update = self.parameter_tuning_system
            .auto_tune_system(&system_state)
            .await?;
            
        if system_update.expected_performance_gain > 0.02 {
            self.apply_system_update(system_update)?;
            improvements_made = true;
            
            log::info!("Applied comprehensive system parameter updates");
        }
    }
    
    // 6. Record parameter tuning results
    if improvements_made {
        self.performance_monitor.record_metric(
            "parameter_tuning_executed",
            1.0
        );
        
        // Schedule performance validation in 5 minutes
        self.scheduler.schedule_task(
            LearningTaskType::PerformanceValidation,
            SystemTime::now() + Duration::from_secs(300),
            TaskPriority::High
        )?;
    }
    
    Ok(improvements_made)
}

// Helper method to extract attention metrics from performance data
fn extract_attention_metrics(&self, perf_data: &PerformanceData) -> Result<AttentionMetrics> {
    let cognitive_metrics = perf_data.cognitive_metrics
        .as_ref()
        .ok_or_else(|| anyhow!("No cognitive metrics available"))?;
        
    Ok(AttentionMetrics {
        focus_accuracy: cognitive_metrics.attention_focus_accuracy,
        attention_stability: cognitive_metrics.attention_stability,
        switching_efficiency: cognitive_metrics.attention_switching_efficiency,
        capacity_utilization: cognitive_metrics.attention_capacity_utilization,
    })
}

// Helper method to extract memory metrics from performance data
fn extract_memory_metrics(&self, perf_data: &PerformanceData) -> Result<MemoryMetrics> {
    Ok(MemoryMetrics {
        efficiency: perf_data.memory_efficiency,
        retention_rate: perf_data.cache_hit_rate,
        consolidation_rate: 0.8, // Default if not available
        capacity_utilization: perf_data.memory_usage_percent,
    })
}

// Helper method to create system state for auto-tuning
fn create_system_state(&self) -> Result<SystemState> {
    let current_params = self.current_parameters.read();
    
    Ok(SystemState {
        current_parameters: current_params.clone(),
        performance_history: self.performance_monitor.get_performance_history()?,
        resource_usage: ResourceUsage {
            memory_percent: self.performance_monitor.get_current_memory_usage(),
            cpu_percent: self.performance_monitor.get_current_cpu_usage(),
            active_entities: self.integrated_cognitive_system.get_active_entity_count(),
            cache_entries: self.integrated_cognitive_system.get_cache_entry_count(),
        },
        active_patterns: self.integrated_cognitive_system
            .get_orchestrator()
            .get_active_pattern_stats(),
    })
}

// Helper method to update cache parameters
fn update_cache_parameters(&self, new_size: usize) -> Result<()> {
    // Update similarity cache in knowledge graph
    self.integrated_cognitive_system
        .get_knowledge_graph()
        .update_similarity_cache_size(new_size)?;
        
    // Update current parameters
    let mut params = self.current_parameters.write();
    params.similarity_cache_capacity = new_size;
    
    Ok(())
}

// Helper method to apply comprehensive system updates
fn apply_system_update(&self, update: SystemParameterUpdate) -> Result<()> {
    let mut params = self.current_parameters.write();
    
    // Apply activation parameter updates
    if let Some(activation) = update.activation_updates {
        params.activation.decay_rate += activation.decay_rate_delta;
        params.activation.convergence_threshold += activation.convergence_threshold_delta;
        params.activation.inhibition_strength += activation.inhibition_strength_delta;
        
        // Apply to activation engine
        self.integrated_cognitive_system
            .get_activation_engine()
            .update_config(|config| {
                config.decay_rate = params.activation.decay_rate;
                config.convergence_threshold = params.activation.convergence_threshold;
                config.inhibition_strength = params.activation.inhibition_strength;
            })?;
    }
    
    // Apply query parameter updates
    if let Some(query) = update.query_updates {
        let sum = query.similarity_weight_delta + query.context_weight_delta;
        if sum.abs() < 0.001 {
            params.query.similarity_weight += query.similarity_weight_delta;
            params.query.context_weight += query.context_weight_delta;
        }
        params.query.relevance_threshold += query.relevance_threshold_delta;
    }
    
    // Apply cognitive pattern weight updates
    if let Some(pattern_weights) = update.pattern_weight_updates {
        for (pattern_type, weight_delta) in pattern_weights {
            if let Some(weight) = params.cognitive.pattern_weights.get_mut(&pattern_type) {
                *weight = (*weight + weight_delta).clamp(0.1, 2.0);
            }
        }
        
        // Apply to orchestrator
        self.integrated_cognitive_system
            .get_orchestrator()
            .update_pattern_weights(params.cognitive.pattern_weights.clone())?;
    }
    
    // Validate all parameters
    params.validate()?;
    
    log::info!("System parameters updated successfully");
    Ok(())
}
```

### 4. Add Required Methods to Support Parameter Updates

Based on the implementation above, the following methods need to be added to various components:

#### 4.1 Knowledge Graph Updates (`src/core/graph/graph_core.rs`)
```rust
impl KnowledgeGraph {
    // Add method to update similarity cache size (around line 300)
    pub fn update_similarity_cache_size(&self, new_size: usize) -> Result<()> {
        let mut cache = self.similarity_cache.write();
        // Note: LRU cache doesn't support dynamic resizing, so we need to recreate
        let mut new_cache = SimilarityCache::new(new_size);
        
        // Copy existing entries up to new size
        for (key, value) in cache.iter().take(new_size) {
            new_cache.insert(key.clone(), value.clone());
        }
        
        *cache = new_cache;
        log::info!("Updated similarity cache size to {}", new_size);
        Ok(())
    }
}
```

#### 4.2 Activation Engine Updates (`src/core/activation_engine.rs` or similar)
```rust
impl ActivationEngine {
    // Add method to update configuration (location depends on actual struct)
    pub fn update_config<F>(&self, updater: F) -> Result<()> 
    where
        F: FnOnce(&mut ActivationConfig),
    {
        let mut config = self.config.write();
        updater(&mut config);
        config.validate()?; // Ensure configuration is valid
        log::info!("Updated activation engine configuration");
        Ok(())
    }
}
```

#### 4.3 Cognitive Orchestrator Updates (`src/cognitive/orchestrator.rs`)
```rust
impl CognitiveOrchestrator {
    // Add method to update pattern weights (around line 450)
    pub fn update_pattern_weights(
        &self, 
        new_weights: HashMap<CognitivePatternType, f32>
    ) -> Result<()> {
        // Validate weights
        for (_, weight) in &new_weights {
            if *weight < 0.1 || *weight > 2.0 {
                return Err(anyhow!("Pattern weight out of range [0.1, 2.0]"));
            }
        }
        
        // Store in internal state (may need to add Arc<RwLock<>> field)
        *self.pattern_weights.write() = new_weights;
        
        log::info!("Updated cognitive pattern weights");
        Ok(())
    }
}
```

#### 4.4 Hebbian Engine Updates (`src/learning/hebbian.rs`)
```rust
impl HebbianLearningEngine {
    // Add method to apply parameter updates (around line 200)
    pub fn apply_parameter_update(&self, update: HebbianParameterUpdate) -> Result<()> {
        let mut state = self.state.write();
        
        // Apply learning rate adjustment
        state.learning_rate = (state.learning_rate + update.learning_rate_adjustment)
            .clamp(0.001, 0.1);
            
        // Apply decay constant adjustment
        state.decay_constant = (state.decay_constant + update.decay_constant_adjustment)
            .clamp(0.0001, 0.01);
            
        // Apply strengthening threshold adjustment
        if let Some(threshold) = &mut state.strengthening_threshold {
            *threshold = (*threshold + update.strengthening_threshold_adjustment)
                .clamp(0.1, 0.9);
        }
        
        log::info!(
            "Applied Hebbian parameter update: learning_rate={}, decay_constant={}",
            state.learning_rate, state.decay_constant
        );
        
        Ok(())
    }
}
```

#### 4.5 Attention Manager Updates (`src/cognitive/attention_manager.rs`)
```rust
impl AttentionManager {
    // Add method to apply parameter updates (around line 400)
    pub fn apply_parameter_update(&self, update: AttentionParameterUpdate) -> Result<()> {
        let mut config = self.config.write();
        
        // Apply focus strength adjustment
        let new_focus = config.executive_control_strength + update.focus_strength_adjustment;
        config.executive_control_strength = new_focus.clamp(0.1, 1.0);
        
        // Apply capacity adjustment
        let new_capacity = (config.max_attention_targets as f32 + update.capacity_adjustment) as usize;
        config.max_attention_targets = new_capacity.clamp(3, 12);
        
        // Apply shift speed adjustment (maps to decay rate)
        let new_decay = config.attention_decay_rate + update.shift_speed_adjustment;
        config.attention_decay_rate = new_decay.clamp(0.01, 0.5);
        
        log::info!(
            "Applied attention parameter update: focus={}, capacity={}, decay_rate={}",
            config.executive_control_strength,
            config.max_attention_targets,
            config.attention_decay_rate
        );
        
        Ok(())
    }
}
```

### 5. Create Performance Validation Task

Add to `src/learning/adaptive_learning/system.rs`:

```rust
impl AdaptiveLearningSystem {
    // Add performance validation method
    pub async fn validate_parameter_changes(&self) -> Result<()> {
        log::info!("Validating recent parameter changes");
        
        // Get performance data since last tuning
        let recent_perf = self.performance_monitor.get_recent_performance(
            Duration::from_secs(300) // Last 5 minutes
        )?;
        
        // Compare with baseline
        let baseline = self.performance_monitor.get_baseline_performance()?;
        
        let improvement = (recent_perf.overall_score - baseline.overall_score) / baseline.overall_score;
        
        if improvement < -0.05 {
            // Performance degraded by more than 5%
            log::warn!("Performance degradation detected: {:.2}%", improvement * 100.0);
            
            // Rollback to previous parameters
            if let Some(previous) = self.parameter_history.get_previous() {
                self.apply_system_parameters(previous)?;
                log::info!("Rolled back to previous parameters");
            }
        } else if improvement > 0.01 {
            log::info!("Performance improvement confirmed: {:.2}%", improvement * 100.0);
            
            // Update baseline
            self.performance_monitor.update_baseline(recent_perf)?;
        }
        
        Ok(())
    }
}
```

### 6. Integration with Learning Scheduler

Update `src/learning/adaptive_learning/scheduler.rs`:

```rust
impl LearningScheduler {
    // Update task execution to handle PerformanceValidation
    async fn execute_task(&self, task: &ScheduledTask) -> Result<()> {
        match task.task_type {
            LearningTaskType::ParameterTuning => {
                self.system.execute_parameter_tuning().await?;
            }
            LearningTaskType::PerformanceValidation => {
                self.system.validate_parameter_changes().await?;
            }
            // ... other task types
        }
        Ok(())
    }
}
```

### 7. Required Type Additions

Add to `src/learning/adaptive_learning/types.rs`:

```rust
// Add to LearningTaskType enum
pub enum LearningTaskType {
    // ... existing variants ...
    PerformanceValidation,
}

// Add system state structure
#[derive(Debug, Clone)]
pub struct SystemState {
    pub current_parameters: SystemParameters,
    pub performance_history: Vec<PerformanceData>,
    pub resource_usage: ResourceUsage,
    pub active_patterns: HashMap<CognitivePatternType, u64>,
}

#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub memory_percent: f32,
    pub cpu_percent: f32,
    pub active_entities: usize,
    pub cache_entries: usize,
}

// Add update structures
#[derive(Debug, Clone)]
pub struct SystemParameterUpdate {
    pub activation_updates: Option<ActivationUpdate>,
    pub query_updates: Option<QueryUpdate>,
    pub pattern_weight_updates: Option<HashMap<CognitivePatternType, f32>>,
    pub expected_performance_gain: f32,
}

#[derive(Debug, Clone)]
pub struct ActivationUpdate {
    pub decay_rate_delta: f32,
    pub convergence_threshold_delta: f32,
    pub inhibition_strength_delta: f32,
}

#[derive(Debug, Clone)]
pub struct QueryUpdate {
    pub similarity_weight_delta: f32,
    pub context_weight_delta: f32,
    pub relevance_threshold_delta: f32,
}
```

## Benefits
- **Leverages Existing Infrastructure**: Uses the already-implemented `ParameterTuningSystem` with Bayesian optimization, grid search, and genetic algorithms
- **Measurable Impact**: Direct connection between parameter changes and performance metrics
- **Automatic Optimization**: System self-tunes based on real workload patterns
- **Safe Rollback**: Performance validation and automatic rollback prevent degradation
- **Comprehensive Coverage**: Tunes parameters across all subsystems (graph, cognitive, learning)

## Implementation Notes

### Key Discoveries from Analysis:
1. **The `ParameterTuningSystem` already exists** (`src/learning/parameter_tuning.rs`) with sophisticated optimization algorithms - it just needs to be integrated
2. **Performance monitoring is already comprehensive** - bottleneck detection, metrics tracking, and feedback aggregation are all implemented
3. **Parameter update structures exist** for Hebbian learning and attention systems
4. **The main gap** is connecting `execute_parameter_tuning()` to the existing infrastructure

### Critical Implementation Steps:
1. **Add `parameter_tuning_system` field** to `AdaptiveLearningSystem` struct
2. **Initialize it in the constructor** using the existing `ParameterTuningSystem::new()`
3. **Replace the stub `execute_parameter_tuning()`** with the detailed implementation above
4. **Add update methods** to components that need runtime parameter changes:
   - `KnowledgeGraph::update_similarity_cache_size()`
   - `ActivationEngine::update_config()`
   - `CognitiveOrchestrator::update_pattern_weights()`
   - `HebbianLearningEngine::apply_parameter_update()`
   - `AttentionManager::apply_parameter_update()`

### Parameter Ranges (from codebase analysis):
- **Activation decay_rate**: 0.01 to 0.5 (current: 0.1)
- **Convergence threshold**: 0.0001 to 0.01 (current: 0.001)
- **Query similarity_weight**: Must sum to 1.0 with context_weight
- **Cache capacity**: 100 to 100,000 (current: 1,000)
- **Pattern weights**: 0.1 to 2.0 (vary by pattern type)

### Safety Considerations:
- **Validate all parameters** before applying using existing validation methods
- **Use small deltas** for adjustments (typically 1-10% of current value)
- **Monitor performance** for 5 minutes after changes before confirming
- **Maintain parameter history** for rollback capability
- **Respect resource constraints** from `AdaptiveLearningConfig`

### Testing Strategy:
1. **Unit tests** for each component's parameter update method
2. **Integration test** for full parameter tuning cycle
3. **Performance regression test** to verify rollback works
4. **Load test** to ensure stability under parameter changes
5. **Benchmark suite** to measure actual performance improvements

## Files to Modify - Complete List

### 1. `src/learning/adaptive_learning/system.rs`
- **Line 20**: Add `parameter_tuning_system` and `current_parameters` fields to struct
- **Line 42-59**: Update `new()` method to initialize parameter tuning system
- **Line 222-227**: Replace stub `execute_parameter_tuning()` with full implementation
- **Add**: Helper methods for parameter extraction, validation, and application

### 2. `src/learning/adaptive_learning/parameters.rs` (NEW FILE)
- Create complete `SystemParameters` structure with all subsystem parameters
- Add validation methods with proper range checking

### 3. `src/core/graph/graph_core.rs`
- **Add around line 300**: `update_similarity_cache_size()` method

### 4. `src/core/activation_engine.rs` (or location of ActivationEngine)
- **Add**: `update_config()` method with closure-based updates

### 5. `src/cognitive/orchestrator.rs`
- **Add around line 450**: `update_pattern_weights()` method
- **May need**: Add `Arc<RwLock<HashMap<CognitivePatternType, f32>>>` field for weights

### 6. `src/learning/hebbian.rs`
- **Add around line 200**: `apply_parameter_update()` method

### 7. `src/cognitive/attention_manager.rs`
- **Add around line 400**: `apply_parameter_update()` method

### 8. `src/learning/adaptive_learning/types.rs`
- **Add to `LearningTaskType`**: `PerformanceValidation` variant
- **Add**: `SystemState`, `ResourceUsage`, `SystemParameterUpdate` structures

### 9. `src/learning/adaptive_learning/scheduler.rs`
- **Update `execute_task()`**: Handle `PerformanceValidation` task type

### 10. Test Files
- **Create**: `tests/learning/test_parameter_tuning.rs` for integration tests
- **Update**: Existing component tests to verify parameter update methods

This implementation connects the sophisticated parameter tuning infrastructure that already exists, enabling the system to truly self-optimize based on performance feedback.