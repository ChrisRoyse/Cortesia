# Critical Components Fix Plan: Phase 07

## Executive Summary

**Current State**: 4 critical components with severe quality issues
**Target**: 100% functional compliance with standardized quality gates
**Timeline**: 3 weeks (21 days) intensive development
**Success Criteria**: All components pass quality gates with >95% reliability

## Critical Component Analysis

### 1. Spike Pattern Mathematical Edge Cases (42.1% → 85%+ Coverage)

**Component**: `C:\code\LLMKG\crates\neuromorphic-core\src\ttfs_concept\spike_pattern.rs`
**Current Issues**: Mathematical edge cases causing NaN/infinity in complexity calculations
**Risk Level**: CRITICAL - Core TTFS encoding foundation

#### Mathematical Formula Fixes Required

**1.1 Temporal Entropy Calculation (Lines 125-149)**
```rust
// CURRENT PROBLEMATIC FORMULA:
entropy / 2.3 // Normalize by ln(10) - WRONG

// FIXED FORMULA:
entropy / (10.0_f32).ln() // Correct normalization

// EDGE CASE HANDLING:
fn calculate_temporal_entropy_fixed(events: &[SpikeEvent]) -> f32 {
    if events.is_empty() { return 0.0; }
    
    let total_duration = events.iter()
        .map(|e| e.timestamp.as_millis())
        .max()
        .unwrap_or(1).max(1) as f32; // Prevent zero division
    
    let bin_count = 10.0;
    let mut bins = vec![0u32; 10];
    
    for event in events {
        let normalized_time = event.timestamp.as_millis() as f32 / total_duration;
        let bin = (normalized_time * (bin_count - 1.0)) as usize;
        bins[bin.min(9)] += 1; // Clamp to valid range
    }
    
    let total = events.len() as f32;
    let entropy = bins.iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f32 / total;
            if p > 0.0 { -p * p.ln() } else { 0.0 } // Handle zero probability
        })
        .sum::<f32>();
    
    if entropy.is_finite() {
        entropy / bin_count.ln() // Proper normalization
    } else {
        0.0 // Fallback for edge cases
    }
}
```

**1.2 Frequency Variance Fix (Lines 112-123)**
```rust
// EDGE CASE: Empty events, identical frequencies, extreme values
fn calculate_frequency_variance_fixed(events: &[SpikeEvent]) -> f32 {
    if events.len() < 2 { return 0.0; }
    
    let frequencies: Vec<f32> = events.iter()
        .map(|e| e.frequency.max(0.0).min(1000.0)) // Clamp to reasonable range
        .collect();
    
    let mean_freq = frequencies.iter().sum::<f32>() / frequencies.len() as f32;
    let variance = frequencies.iter()
        .map(|&freq| (freq - mean_freq).powi(2))
        .sum::<f32>() / frequencies.len() as f32;
    
    let std_dev = variance.sqrt();
    if std_dev.is_finite() && std_dev > 0.0 {
        (std_dev / 100.0).min(1.0) // Normalize and cap at 1.0
    } else {
        0.0
    }
}
```

**1.3 Complexity Calculation Enhancement (Lines 87-110)**
```rust
fn calculate_complexity_fixed(events: &[SpikeEvent]) -> f32 {
    if events.len() < 2 { return 0.0; }
    
    // Component 1: Neuron diversity (0.0-1.0)
    let unique_neurons = events.iter()
        .map(|e| e.neuron_id)
        .collect::<std::collections::HashSet<_>>()
        .len();
    let neuron_diversity = if events.len() > 0 {
        (unique_neurons as f32 / events.len() as f32).min(1.0)
    } else { 0.0 };
    
    // Component 2: Frequency diversity (0.0-1.0)  
    let frequency_variance = Self::calculate_frequency_variance_fixed(events);
    
    // Component 3: Temporal entropy (0.0-1.0)
    let temporal_entropy = Self::calculate_temporal_entropy_fixed(events);
    
    // Weighted combination with safety checks
    let components = [neuron_diversity, frequency_variance, temporal_entropy];
    if components.iter().all(|&x| x.is_finite()) {
        let complexity = (neuron_diversity * 0.4) + 
                        (frequency_variance * 0.3) + 
                        (temporal_entropy * 0.3);
        complexity.clamp(0.0, 1.0)
    } else {
        0.0 // Safe fallback
    }
}
```

#### Test Coverage Enhancement
**New Test File**: `C:\code\LLMKG\crates\neuromorphic-core\tests\spike_pattern_edge_cases.rs`

```rust
#[test]
fn test_mathematical_edge_cases() {
    // Test 1: Zero duration handling
    let events = vec![SpikeEvent {
        neuron_id: 1,
        timestamp: Duration::ZERO,
        amplitude: 1.0,
        frequency: 50.0,
    }];
    let pattern = SpikePattern::new(events);
    assert!(pattern.complexity.is_finite());
    assert!(!pattern.complexity.is_nan());
    
    // Test 2: Identical timestamps
    let events = vec![
        SpikeEvent { neuron_id: 1, timestamp: Duration::from_millis(10), amplitude: 1.0, frequency: 50.0 },
        SpikeEvent { neuron_id: 2, timestamp: Duration::from_millis(10), amplitude: 1.0, frequency: 50.0 },
    ];
    let pattern = SpikePattern::new(events);
    assert_eq!(pattern.density, 2.0 / 10.0);
    
    // Test 3: Extreme frequency values
    let events = vec![
        SpikeEvent { neuron_id: 1, timestamp: Duration::from_millis(1), amplitude: 1.0, frequency: 0.0 },
        SpikeEvent { neuron_id: 2, timestamp: Duration::from_millis(2), amplitude: 1.0, frequency: f32::INFINITY },
    ];
    let pattern = SpikePattern::new(events);
    assert!(pattern.complexity >= 0.0 && pattern.complexity <= 1.0);
}

#[test]
fn test_entropy_boundary_conditions() {
    // All spikes at start
    let events = (0..10).map(|i| SpikeEvent {
        neuron_id: i,
        timestamp: Duration::from_millis(1),
        amplitude: 1.0,
        frequency: 50.0,
    }).collect();
    let pattern = SpikePattern::new(events);
    assert!(pattern.complexity < 0.5); // Low temporal diversity
    
    // Uniformly distributed spikes
    let events = (0..10).map(|i| SpikeEvent {
        neuron_id: i,
        timestamp: Duration::from_millis(i as u64 * 10),
        amplitude: 1.0,
        frequency: 50.0,
    }).collect();
    let pattern = SpikePattern::new(events);
    assert!(pattern.complexity > 0.3); // Higher temporal diversity
}

#[test]
fn test_large_dataset_performance() {
    let events: Vec<SpikeEvent> = (0..10000).map(|i| SpikeEvent {
        neuron_id: i % 1000,
        timestamp: Duration::from_millis(i as u64),
        amplitude: (i as f32 / 10000.0),
        frequency: 50.0 + (i % 100) as f32,
    }).collect();
    
    let start = std::time::Instant::now();
    let pattern = SpikePattern::new(events);
    let duration = start.elapsed();
    
    assert!(duration < Duration::from_millis(100)); // Performance requirement
    assert!(pattern.complexity.is_finite());
    assert!(pattern.density > 0.0);
}
```

### 2. Memory Consolidation State Transitions (44.4% → 75%+ Coverage)

**Component**: `C:\code\LLMKG\crates\temporal-memory\src\consolidation\mod.rs`
**Current Issues**: Incomplete state machine transitions, race conditions in parallel mode
**Risk Level**: CRITICAL - Data integrity foundation

#### State Machine Definition
```rust
// File: C:\code\LLMKG\crates\temporal-memory\src\consolidation\state_machine.rs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsolidationState {
    // Initial state
    Pending,
    
    // Active processing states  
    AnalyzingDivergence,
    DetectingConflicts,
    ResolvingConflicts,
    MergingConcepts,
    UpdatingReferences,
    
    // Terminal states
    Consolidated,
    Failed(ConsolidationError),
    RolledBack,
}

#[derive(Debug, Clone)]
pub struct ConsolidationStateMachine {
    current_state: ConsolidationState,
    transition_log: Vec<StateTransition>,
    start_time: Instant,
    config: ConsolidationConfig,
}

#[derive(Debug, Clone)]
pub struct StateTransition {
    from: ConsolidationState,
    to: ConsolidationState,
    timestamp: Instant,
    trigger: TransitionTrigger,
    metadata: Option<String>,
}

#[derive(Debug, Clone)]
pub enum TransitionTrigger {
    StartConsolidation,
    DivergenceDetected(f32),
    ConflictsFound(usize),
    ConflictsResolved,
    MergeCompleted,
    ReferencesUpdated,
    ErrorOccurred(String),
    RollbackInitiated,
}

impl ConsolidationStateMachine {
    pub fn new(config: ConsolidationConfig) -> Self {
        Self {
            current_state: ConsolidationState::Pending,
            transition_log: Vec::new(),
            start_time: Instant::now(),
            config,
        }
    }
    
    pub fn transition(&mut self, trigger: TransitionTrigger) -> Result<ConsolidationState, StateError> {
        let new_state = self.calculate_next_state(&trigger)?;
        
        let transition = StateTransition {
            from: self.current_state,
            to: new_state,
            timestamp: Instant::now(),
            trigger,
            metadata: None,
        };
        
        self.transition_log.push(transition);
        self.current_state = new_state;
        
        Ok(new_state)
    }
    
    fn calculate_next_state(&self, trigger: &TransitionTrigger) -> Result<ConsolidationState, StateError> {
        use ConsolidationState::*;
        use TransitionTrigger::*;
        
        match (self.current_state, trigger) {
            (Pending, StartConsolidation) => Ok(AnalyzingDivergence),
            (AnalyzingDivergence, DivergenceDetected(_)) => Ok(DetectingConflicts),
            (DetectingConflicts, ConflictsFound(count)) => {
                if *count > self.config.max_conflicts {
                    Ok(Failed(ConsolidationError::TooManyConflicts))
                } else {
                    Ok(ResolvingConflicts)
                }
            }
            (DetectingConflicts, ConflictsResolved) => Ok(MergingConcepts),
            (ResolvingConflicts, ConflictsResolved) => Ok(MergingConcepts),
            (MergingConcepts, MergeCompleted) => Ok(UpdatingReferences),
            (UpdatingReferences, ReferencesUpdated) => Ok(Consolidated),
            (_, ErrorOccurred(_)) => Ok(Failed(ConsolidationError::ProcessingError)),
            (Failed(_), RollbackInitiated) => Ok(RolledBack),
            _ => Err(StateError::InvalidTransition {
                from: self.current_state,
                trigger: trigger.clone(),
            }),
        }
    }
    
    pub fn is_terminal(&self) -> bool {
        matches!(self.current_state, ConsolidationState::Consolidated | 
                ConsolidationState::Failed(_) | 
                ConsolidationState::RolledBack)
    }
    
    pub fn get_progress(&self) -> f32 {
        use ConsolidationState::*;
        match self.current_state {
            Pending => 0.0,
            AnalyzingDivergence => 0.1,
            DetectingConflicts => 0.3,
            ResolvingConflicts => 0.5,
            MergingConcepts => 0.7,
            UpdatingReferences => 0.9,
            Consolidated => 1.0,
            Failed(_) | RolledBack => 0.0,
        }
    }
}
```

#### Enhanced Consolidation Engine
```rust
// Enhanced consolidation with state machine integration
impl ConsolidationEngine {
    pub async fn consolidate_with_state_tracking(
        &self,
        parent: &mut MemoryBranch,
        child: &MemoryBranch,
    ) -> Result<ConsolidationResult, ConsolidationError> {
        let mut state_machine = ConsolidationStateMachine::new(self.config.clone());
        
        // Start consolidation
        state_machine.transition(TransitionTrigger::StartConsolidation)?;
        
        // Phase 1: Analyze divergence
        let divergence = self.calculate_divergence(parent, child).await?;
        state_machine.transition(TransitionTrigger::DivergenceDetected(divergence))?;
        
        // Phase 2: Detect conflicts
        let conflicts = self.detect_conflicts(parent, child).await?;
        if conflicts.is_empty() {
            state_machine.transition(TransitionTrigger::ConflictsResolved)?;
        } else {
            state_machine.transition(TransitionTrigger::ConflictsFound(conflicts.len()))?;
            
            // Phase 3: Resolve conflicts
            let resolutions = self.resolve_conflicts(&conflicts, parent, child).await?;
            state_machine.transition(TransitionTrigger::ConflictsResolved)?;
        }
        
        // Phase 4: Merge concepts
        let merged_concepts = self.merge_concepts(parent, child).await?;
        state_machine.transition(TransitionTrigger::MergeCompleted)?;
        
        // Phase 5: Update references
        self.update_references(parent, &merged_concepts).await?;
        state_machine.transition(TransitionTrigger::ReferencesUpdated)?;
        
        Ok(ConsolidationResult {
            merged_concepts,
            conflicts,
            resolutions: Vec::new(),
            final_divergence: self.calculate_divergence(parent, child).await?,
            success: true,
            state_log: state_machine.transition_log,
        })
    }
}
```

### 3. Allocation Engine Quality Gates Definition

**Component**: `C:\code\LLMKG\crates\snn-allocation-engine\src\quality_gates.rs` (NEW)
**Current Issues**: No standardized quality metrics, inconsistent allocation decisions
**Risk Level**: HIGH - Resource allocation accuracy

#### Quality Gate Configuration
```rust
// File: C:\code\LLMKG\crates\snn-allocation-engine\src\quality_gates.rs
#[derive(Debug, Clone)]
pub struct AllocationQualityGates {
    pub performance_thresholds: PerformanceThresholds,
    pub accuracy_requirements: AccuracyRequirements,
    pub resource_limits: ResourceLimits,
    pub consistency_checks: ConsistencyChecks,
}

#[derive(Debug, Clone)]
pub struct PerformanceThresholds {
    /// Maximum allocation time per concept (ms)
    pub max_allocation_time_ms: u64,
    
    /// Maximum memory usage per allocation (MB)
    pub max_memory_usage_mb: f64,
    
    /// Minimum throughput (allocations/second)
    pub min_throughput: f64,
    
    /// Maximum CPU utilization (0.0-1.0)
    pub max_cpu_utilization: f64,
}

#[derive(Debug, Clone)]
pub struct AccuracyRequirements {
    /// Minimum semantic similarity threshold
    pub min_semantic_similarity: f32,
    
    /// Maximum acceptable false positive rate
    pub max_false_positive_rate: f32,
    
    /// Minimum recall for concept retrieval
    pub min_recall: f32,
    
    /// Maximum allocation variance between runs
    pub max_allocation_variance: f32,
}

#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum cortical columns per allocation
    pub max_columns_per_allocation: usize,
    
    /// Maximum neural connections per concept
    pub max_connections_per_concept: usize,
    
    /// Maximum spike patterns per second
    pub max_spike_patterns_per_sec: usize,
    
    /// Memory pool size limit (bytes)
    pub memory_pool_limit_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct ConsistencyChecks {
    /// Enable deterministic allocation verification
    pub enable_deterministic_check: bool,
    
    /// Enable cross-validation with multiple algorithms
    pub enable_cross_validation: bool,
    
    /// Enable allocation rollback on quality failure
    pub enable_quality_rollback: bool,
    
    /// Minimum quality score threshold (0.0-1.0)
    pub min_quality_score: f32,
}

impl Default for AllocationQualityGates {
    fn default() -> Self {
        Self {
            performance_thresholds: PerformanceThresholds {
                max_allocation_time_ms: 50,
                max_memory_usage_mb: 10.0,
                min_throughput: 100.0,
                max_cpu_utilization: 0.8,
            },
            accuracy_requirements: AccuracyRequirements {
                min_semantic_similarity: 0.75,
                max_false_positive_rate: 0.05,
                min_recall: 0.90,
                max_allocation_variance: 0.1,
            },
            resource_limits: ResourceLimits {
                max_columns_per_allocation: 1000,
                max_connections_per_concept: 10000,
                max_spike_patterns_per_sec: 50000,
                memory_pool_limit_bytes: 100 * 1024 * 1024, // 100MB
            },
            consistency_checks: ConsistencyChecks {
                enable_deterministic_check: true,
                enable_cross_validation: false,
                enable_quality_rollback: true,
                min_quality_score: 0.8,
            },
        }
    }
}

#[derive(Debug)]
pub struct QualityGateResult {
    pub passed: bool,
    pub score: f32,
    pub failed_checks: Vec<QualityCheck>,
    pub performance_metrics: PerformanceMetrics,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum QualityCheck {
    PerformanceTimeout { actual_ms: u64, limit_ms: u64 },
    MemoryExceeded { actual_mb: f64, limit_mb: f64 },
    ThroughputBelowLimit { actual: f64, required: f64 },
    SemanticSimilarityLow { actual: f32, required: f32 },
    HighFalsePositiveRate { actual: f32, limit: f32 },
    LowRecall { actual: f32, required: f32 },
    HighAllocationVariance { actual: f32, limit: f32 },
    ResourceLimitExceeded(String),
    ConsistencyFailure(String),
}

impl AllocationQualityGates {
    pub fn evaluate_allocation(
        &self,
        allocation_result: &AllocationResult,
        performance_metrics: &PerformanceMetrics,
    ) -> QualityGateResult {
        let mut failed_checks = Vec::new();
        let mut score_components = Vec::new();
        
        // Performance checks
        if performance_metrics.allocation_time_ms > self.performance_thresholds.max_allocation_time_ms {
            failed_checks.push(QualityCheck::PerformanceTimeout {
                actual_ms: performance_metrics.allocation_time_ms,
                limit_ms: self.performance_thresholds.max_allocation_time_ms,
            });
        } else {
            score_components.push(0.25); // 25% of score
        }
        
        if performance_metrics.memory_usage_mb > self.performance_thresholds.max_memory_usage_mb {
            failed_checks.push(QualityCheck::MemoryExceeded {
                actual_mb: performance_metrics.memory_usage_mb,
                limit_mb: self.performance_thresholds.max_memory_usage_mb,
            });
        } else {
            score_components.push(0.20); // 20% of score
        }
        
        // Accuracy checks
        if allocation_result.semantic_similarity < self.accuracy_requirements.min_semantic_similarity {
            failed_checks.push(QualityCheck::SemanticSimilarityLow {
                actual: allocation_result.semantic_similarity,
                required: self.accuracy_requirements.min_semantic_similarity,
            });
        } else {
            score_components.push(0.30); // 30% of score
        }
        
        // Resource limit checks
        if allocation_result.allocated_columns.len() > self.resource_limits.max_columns_per_allocation {
            failed_checks.push(QualityCheck::ResourceLimitExceeded(
                format!("Too many columns allocated: {} > {}", 
                       allocation_result.allocated_columns.len(),
                       self.resource_limits.max_columns_per_allocation)
            ));
        } else {
            score_components.push(0.15); // 15% of score
        }
        
        // Consistency checks
        if self.consistency_checks.enable_deterministic_check {
            if let Some(variance) = allocation_result.allocation_variance {
                if variance > self.accuracy_requirements.max_allocation_variance {
                    failed_checks.push(QualityCheck::HighAllocationVariance {
                        actual: variance,
                        limit: self.accuracy_requirements.max_allocation_variance,
                    });
                } else {
                    score_components.push(0.10); // 10% of score
                }
            }
        }
        
        let score = score_components.iter().sum::<f32>();
        let passed = failed_checks.is_empty() && score >= self.consistency_checks.min_quality_score;
        
        let recommendations = self.generate_recommendations(&failed_checks);
        
        QualityGateResult {
            passed,
            score,
            failed_checks,
            performance_metrics: performance_metrics.clone(),
            recommendations,
        }
    }
    
    fn generate_recommendations(&self, failed_checks: &[QualityCheck]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        for check in failed_checks {
            match check {
                QualityCheck::PerformanceTimeout { .. } => {
                    recommendations.push("Consider optimizing allocation algorithm or using parallel processing".to_string());
                }
                QualityCheck::MemoryExceeded { .. } => {
                    recommendations.push("Implement memory pooling or reduce allocation batch size".to_string());
                }
                QualityCheck::SemanticSimilarityLow { .. } => {
                    recommendations.push("Improve TTFS encoding accuracy or adjust similarity thresholds".to_string());
                }
                QualityCheck::ResourceLimitExceeded(_) => {
                    recommendations.push("Implement resource management or increase limits if justified".to_string());
                }
                QualityCheck::HighAllocationVariance { .. } => {
                    recommendations.push("Ensure deterministic random seeds or improve algorithm stability".to_string());
                }
                _ => {}
            }
        }
        
        recommendations
    }
}
```

### 4. Query Processor Spreading Activation Standardization

**Component**: `C:\code\LLMKG\crates\query-processor\src\spreading_activation.rs` (NEW)
**Current Issues**: Inconsistent activation algorithms, no standardized decay functions
**Risk Level**: HIGH - Query accuracy and performance

#### Standardized Spreading Activation Framework
```rust
// File: C:\code\LLMKG\crates\query-processor\src\spreading_activation.rs
#[derive(Debug, Clone)]
pub struct SpreadingActivationConfig {
    /// Initial activation strength for seed nodes
    pub initial_activation: f32,
    
    /// Decay rate per iteration (0.0-1.0)
    pub decay_rate: f32,
    
    /// Minimum activation threshold
    pub activation_threshold: f32,
    
    /// Maximum iterations before convergence
    pub max_iterations: usize,
    
    /// Convergence threshold (change per iteration)
    pub convergence_threshold: f32,
    
    /// Spreading strategy
    pub spreading_strategy: SpreadingStrategy,
    
    /// Enable lateral inhibition
    pub enable_lateral_inhibition: bool,
    
    /// Inhibition strength (0.0-1.0)
    pub inhibition_strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpreadingStrategy {
    /// Classic spreading activation
    Classic,
    
    /// Distance-weighted spreading
    DistanceWeighted,
    
    /// Frequency-based spreading
    FrequencyBased,
    
    /// Adaptive spreading (adjusts based on graph structure)
    Adaptive,
    
    /// Neuromorphic TTFS-based spreading
    Neuromorphic,
}

#[derive(Debug, Clone)]
pub struct ActivationState {
    /// Node activations (node_id -> activation_level)
    activations: HashMap<NodeId, f32>,
    
    /// Previous iteration activations for convergence detection
    previous_activations: HashMap<NodeId, f32>,
    
    /// Activation history for tracing
    history: Vec<ActivationSnapshot>,
    
    /// Current iteration
    iteration: usize,
}

#[derive(Debug, Clone)]
pub struct ActivationSnapshot {
    pub iteration: usize,
    pub timestamp: Instant,
    pub active_nodes: usize,
    pub total_activation: f32,
    pub max_activation: f32,
    pub convergence_delta: f32,
}

#[derive(Debug)]
pub struct SpreadingActivationEngine {
    config: SpreadingActivationConfig,
    graph: Arc<KnowledgeGraph>,
    performance_monitor: Arc<PerformanceMonitor>,
}

impl SpreadingActivationEngine {
    pub fn new(
        config: SpreadingActivationConfig,
        graph: Arc<KnowledgeGraph>,
        performance_monitor: Arc<PerformanceMonitor>,
    ) -> Self {
        Self {
            config,
            graph,
            performance_monitor,
        }
    }
    
    pub async fn spread_activation(
        &self,
        seed_nodes: &[NodeId],
        seed_strengths: &[f32],
    ) -> Result<ActivationState, SpreadingError> {
        let start_time = Instant::now();
        
        // Initialize activation state
        let mut state = ActivationState::new();
        for (node_id, strength) in seed_nodes.iter().zip(seed_strengths.iter()) {
            state.set_activation(*node_id, *strength);
        }
        
        // Record initial snapshot
        state.record_snapshot();
        
        // Main spreading loop
        for iteration in 0..self.config.max_iterations {
            state.iteration = iteration;
            
            // Store previous state for convergence detection
            state.store_previous_state();
            
            // Perform one spreading iteration
            self.spread_iteration(&mut state).await?;
            
            // Apply decay
            self.apply_decay(&mut state);
            
            // Apply lateral inhibition if enabled
            if self.config.enable_lateral_inhibition {
                self.apply_lateral_inhibition(&mut state).await?;
            }
            
            // Record snapshot
            state.record_snapshot();
            
            // Check convergence
            if self.check_convergence(&state) {
                break;
            }
        }
        
        // Record performance metrics
        let duration = start_time.elapsed();
        self.performance_monitor.record_spreading_metrics(SpreadingMetrics {
            duration,
            iterations: state.iteration,
            active_nodes: state.activations.len(),
            total_activation: state.total_activation(),
            convergence_achieved: state.iteration < self.config.max_iterations,
        });
        
        Ok(state)
    }
    
    async fn spread_iteration(&self, state: &mut ActivationState) -> Result<(), SpreadingError> {
        let mut new_activations = HashMap::new();
        
        // For each active node, spread activation to neighbors
        for (&node_id, &activation) in &state.activations {
            if activation < self.config.activation_threshold {
                continue;
            }
            
            let neighbors = self.graph.get_neighbors(node_id).await?;
            
            for neighbor in neighbors {
                let edge_weight = self.graph.get_edge_weight(node_id, neighbor.id).await?;
                let spread_amount = self.calculate_spread_amount(
                    activation,
                    edge_weight,
                    neighbor.distance,
                    &neighbor.properties,
                );
                
                *new_activations.entry(neighbor.id).or_insert(0.0) += spread_amount;
            }
        }
        
        // Apply new activations based on strategy
        match self.config.spreading_strategy {
            SpreadingStrategy::Classic => {
                self.apply_classic_spreading(state, &new_activations);
            }
            SpreadingStrategy::DistanceWeighted => {
                self.apply_distance_weighted_spreading(state, &new_activations).await?;
            }
            SpreadingStrategy::Neuromorphic => {
                self.apply_neuromorphic_spreading(state, &new_activations).await?;
            }
            _ => {
                return Err(SpreadingError::UnsupportedStrategy(self.config.spreading_strategy));
            }
        }
        
        Ok(())
    }
    
    fn calculate_spread_amount(
        &self,
        source_activation: f32,
        edge_weight: f32,
        distance: f32,
        neighbor_properties: &NodeProperties,
    ) -> f32 {
        match self.config.spreading_strategy {
            SpreadingStrategy::Classic => {
                source_activation * edge_weight * 0.1 // 10% spreading factor
            }
            SpreadingStrategy::DistanceWeighted => {
                let distance_factor = 1.0 / (1.0 + distance);
                source_activation * edge_weight * distance_factor * 0.1
            }
            SpreadingStrategy::FrequencyBased => {
                let frequency_boost = neighbor_properties.access_frequency.unwrap_or(1.0);
                source_activation * edge_weight * frequency_boost.ln().max(1.0) * 0.1
            }
            SpreadingStrategy::Neuromorphic => {
                // Use TTFS-based spreading with spike timing
                let spike_factor = self.calculate_spike_timing_factor(neighbor_properties);
                source_activation * edge_weight * spike_factor * 0.1
            }
            _ => source_activation * edge_weight * 0.1,
        }
    }
    
    fn apply_decay(&self, state: &mut ActivationState) {
        for activation in state.activations.values_mut() {
            *activation *= 1.0 - self.config.decay_rate;
            
            // Remove very low activations to improve performance
            if *activation < self.config.activation_threshold * 0.01 {
                *activation = 0.0;
            }
        }
        
        // Remove zero activations
        state.activations.retain(|_, &mut v| v > 0.0);
    }
    
    async fn apply_lateral_inhibition(&self, state: &mut ActivationState) -> Result<(), SpreadingError> {
        let mut inhibition_effects = HashMap::new();
        
        // Calculate inhibition effects between competing nodes
        for (&node_id, &activation) in &state.activations {
            let competitors = self.graph.get_competing_nodes(node_id).await?;
            
            for competitor in competitors {
                let inhibition = activation * self.config.inhibition_strength * 0.1;
                *inhibition_effects.entry(competitor).or_insert(0.0) += inhibition;
            }
        }
        
        // Apply inhibition
        for (node_id, inhibition) in inhibition_effects {
            if let Some(activation) = state.activations.get_mut(&node_id) {
                *activation = (*activation - inhibition).max(0.0);
            }
        }
        
        Ok(())
    }
    
    fn check_convergence(&self, state: &ActivationState) -> bool {
        if state.iteration == 0 {
            return false;
        }
        
        let mut total_change = 0.0;
        let mut node_count = 0;
        
        for (&node_id, &current_activation) in &state.activations {
            let previous_activation = state.previous_activations.get(&node_id).unwrap_or(&0.0);
            total_change += (current_activation - previous_activation).abs();
            node_count += 1;
        }
        
        // Check nodes that were active in previous iteration but not current
        for (&node_id, &previous_activation) in &state.previous_activations {
            if !state.activations.contains_key(&node_id) {
                total_change += previous_activation;
                node_count += 1;
            }
        }
        
        if node_count == 0 {
            return true;
        }
        
        let average_change = total_change / node_count as f32;
        average_change < self.config.convergence_threshold
    }
}

impl Default for SpreadingActivationConfig {
    fn default() -> Self {
        Self {
            initial_activation: 1.0,
            decay_rate: 0.1,
            activation_threshold: 0.01,
            max_iterations: 50,
            convergence_threshold: 0.001,
            spreading_strategy: SpreadingStrategy::Classic,
            enable_lateral_inhibition: true,
            inhibition_strength: 0.2,
        }
    }
}
```

## Performance Benchmarks

### 1. Spike Pattern Performance Targets
```rust
// File: C:\code\LLMKG\benches\spike_pattern_performance.rs
#[bench]
fn bench_spike_pattern_creation_1k_events(b: &mut Bencher) {
    let events: Vec<SpikeEvent> = (0..1000).map(|i| SpikeEvent {
        neuron_id: i % 100,
        timestamp: Duration::from_micros(i as u64 * 10),
        amplitude: (i as f32 / 1000.0),
        frequency: 50.0 + (i % 50) as f32,
    }).collect();
    
    b.iter(|| {
        let pattern = SpikePattern::new(events.clone());
        black_box(pattern.complexity);
    });
}

// Target: < 500μs for 1k events
// Target: < 5ms for 10k events
// Target: < 100ms for 100k events
```

### 2. Memory Consolidation Performance Targets
```rust
// File: C:\code\LLMKG\benches\consolidation_performance.rs
#[bench]  
fn bench_consolidation_100_concepts(b: &mut Bencher) {
    let mut parent = create_test_branch_with_concepts(100);
    let child = create_test_branch_with_concepts(50);
    let engine = ConsolidationEngine::new(ConsolidationConfig::default());
    
    b.iter(|| {
        let result = engine.consolidate_branches(&mut parent, &child);
        black_box(result);
    });
}

// Target: < 50ms for 100 concepts
// Target: < 500ms for 1k concepts  
// Target: < 5s for 10k concepts
```

### 3. Allocation Engine Performance Targets
```rust
// Target: < 10ms per allocation
// Target: > 1000 allocations/second throughput
// Target: < 10MB memory usage per allocation
// Target: > 95% allocation accuracy
```

### 4. Query Processor Performance Targets  
```rust
// Target: < 100ms for complex queries
// Target: < 50 iterations for convergence
// Target: > 90% query intent accuracy
// Target: < 20MB memory usage per query
```

## Implementation Timeline

### Week 1: Mathematical Edge Cases & State Machines (Days 1-7)
- **Day 1-2**: Fix spike pattern mathematical formulas
- **Day 3-4**: Implement comprehensive spike pattern tests
- **Day 5-6**: Design and implement consolidation state machine
- **Day 7**: Integration testing and validation

### Week 2: Quality Gates & Spreading Activation (Days 8-14)
- **Day 8-10**: Implement allocation engine quality gates
- **Day 11-12**: Standardize spreading activation algorithms
- **Day 13-14**: Performance optimization and benchmarking

### Week 3: Integration & Validation (Days 15-21)
- **Day 15-17**: Full system integration testing
- **Day 18-19**: Performance regression testing
- **Day 20-21**: Documentation and final validation

## Success Criteria Checklist

### Critical Component Health Metrics
- [ ] Spike pattern: 0% NaN/infinity results in 10k random test cases
- [ ] Memory consolidation: 100% state transition coverage, 0% deadlocks
- [ ] Allocation engine: All quality gates pass with >95% consistency
- [ ] Query processor: <5% variance in spreading activation results

### Performance Compliance
- [ ] All performance benchmarks meet target thresholds
- [ ] Memory usage within specified limits
- [ ] Concurrent operation safety verified
- [ ] Error recovery mechanisms functional

### Test Coverage Achievement
- [ ] Spike pattern: 85%+ line coverage
- [ ] Memory consolidation: 75%+ line coverage  
- [ ] Allocation engine: 70%+ line coverage
- [ ] Query processor: 70%+ line coverage

This comprehensive plan addresses all critical component issues with precise mathematical fixes, standardized quality gates, performance benchmarks, and complete state machine definitions. Implementation will result in 100% reliable critical component functionality.