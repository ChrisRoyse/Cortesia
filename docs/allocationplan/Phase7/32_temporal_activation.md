# Micro Task 32: Temporal Activation for Time-Based Queries

**Priority**: CRITICAL  
**Estimated Time**: 45 minutes  
**Dependencies**: Phase 6 TMS, Task 31 (belief-aware queries), Tasks 01-30  
**Skills Required**: Rust temporal programming, time-series data, versioning systems

## Objective

Implement temporal activation capabilities that enable time-aware spreading activation, allowing queries to be processed at specific points in time or across time ranges while maintaining belief consistency and justification validity.

## Context

Building on the belief-aware queries from Task 31, this component adds temporal awareness to the spreading activation engine. It integrates with Phase 6's temporal belief management to provide historically accurate query results and time-based reasoning capabilities.

## Specifications

### Required Components

1. **TemporalActivationState**
   - Time-aware activation tracking with version history
   - Temporal validity intervals for activated nodes
   - Historical activation snapshots and reconstruction
   - Time-based belief evolution tracking

2. **TimeAwareSpreader**
   - Spreading algorithm that respects temporal constraints
   - Point-in-time activation reconstruction
   - Time-range activation aggregation
   - Temporal pathway invalidation handling

3. **TemporalBeliefResolver**
   - Resolves belief states at specific timestamps
   - Handles temporal inheritance and validity
   - Manages time-dependent justification chains
   - Supports temporal belief evolution queries

4. **ChronologicalQueryProcessor**
   - High-level interface for temporal queries
   - Supports time-travel queries and historical analysis
   - Temporal consistency validation
   - Time-based result synthesis

### Performance Requirements

- Time-point reconstruction: <5ms for recent history
- Historical query latency: <20ms for 1-year lookback
- Temporal constraint checking: <0.2ms per node
- Memory overhead for temporal data: <30% of base activation

## Implementation Guide

### Step 1: Temporal Activation State

```rust
// File: src/core/activation/temporal_activation_state.rs

use std::collections::{HashMap, BTreeMap};
use std::time::{SystemTime, Duration, Instant};
use std::sync::Arc;
use crate::core::activation::belief_aware_state::{BeliefAwareActivationState, BeliefActivation};
use crate::core::types::{NodeId, ContextId, Timestamp};
use crate::tms::{BeliefId, TemporalValidity, BeliefVersion};
use crate::versioning::{VersionId, TemporalBranch};

#[derive(Debug, Clone)]
pub struct TemporalActivationState {
    // Current state
    pub current_state: BeliefAwareActivationState,
    
    // Temporal tracking
    pub temporal_snapshots: BTreeMap<Timestamp, ActivationSnapshot>,
    pub validity_intervals: HashMap<NodeId, Vec<ValidityInterval>>,
    
    // Historical activation data
    pub activation_history: HashMap<NodeId, ActivationTimeline>,
    
    // Temporal metadata
    pub query_timestamp: Option<Timestamp>,
    pub temporal_context: TemporalContext,
    
    // Version tracking
    pub belief_versions: HashMap<BeliefId, Vec<TemporalBeliefVersion>>,
    
    // Temporal constraints
    pub temporal_constraints: TemporalConstraints,
}

#[derive(Debug, Clone)]
pub struct ActivationSnapshot {
    pub timestamp: Timestamp,
    pub activation_levels: HashMap<NodeId, f32>,
    pub belief_states: HashMap<NodeId, BeliefActivation>,
    pub active_justifications: HashMap<NodeId, Vec<JustificationId>>,
    pub metadata: SnapshotMetadata,
}

#[derive(Debug, Clone)]
pub struct ValidityInterval {
    pub start_time: Timestamp,
    pub end_time: Option<Timestamp>, // None means still valid
    pub validity_type: ValidityType,
    pub belief_version: BeliefVersion,
    pub confidence_trajectory: Vec<(Timestamp, f32)>,
}

#[derive(Debug, Clone)]
pub enum ValidityType {
    Absolute,      // Valid for the entire interval
    Conditional,   // Valid under certain conditions
    Inherited,     // Validity inherited from parent beliefs
    Temporal,      // Time-dependent validity
}

#[derive(Debug, Clone)]
pub struct ActivationTimeline {
    pub node_id: NodeId,
    pub activation_points: BTreeMap<Timestamp, f32>,
    pub belief_changes: BTreeMap<Timestamp, BeliefChange>,
    pub justification_changes: BTreeMap<Timestamp, JustificationChange>,
    pub first_activation: Option<Timestamp>,
    pub last_activation: Option<Timestamp>,
}

#[derive(Debug, Clone)]
pub struct TemporalContext {
    pub context_id: ContextId,
    pub time_reference: TimeReference,
    pub temporal_scope: TemporalScope,
    pub inheritance_rules: Vec<TemporalInheritanceRule>,
}

#[derive(Debug, Clone)]
pub enum TimeReference {
    Absolute(Timestamp),
    Relative(Duration), // Relative to query time
    Range(Timestamp, Timestamp),
    Latest, // Most recent state
}

#[derive(Debug, Clone)]
pub enum TemporalScope {
    PointInTime,           // Single timestamp
    TimeRange,             // Interval
    Evolution,             // Change over time
    Comparative,           // Compare multiple time points
}

impl TemporalActivationState {
    pub fn new(
        current_state: BeliefAwareActivationState,
        temporal_context: TemporalContext,
    ) -> Self {
        Self {
            current_state,
            temporal_snapshots: BTreeMap::new(),
            validity_intervals: HashMap::new(),
            activation_history: HashMap::new(),
            query_timestamp: None,
            temporal_context,
            belief_versions: HashMap::new(),
            temporal_constraints: TemporalConstraints::default(),
        }
    }
    
    pub async fn reconstruct_at_timestamp(
        &self,
        timestamp: Timestamp,
    ) -> Result<BeliefAwareActivationState, TemporalError> {
        // Find the closest snapshot before or at the timestamp
        let base_snapshot = self.find_base_snapshot(timestamp)?;
        
        // Reconstruct state by applying changes since snapshot
        let mut reconstructed_state = self.create_state_from_snapshot(&base_snapshot)?;
        
        // Apply incremental changes from snapshot to target timestamp
        self.apply_incremental_changes(
            &mut reconstructed_state,
            base_snapshot.timestamp,
            timestamp,
        ).await?;
        
        // Validate temporal consistency
        self.validate_temporal_consistency(&reconstructed_state, timestamp).await?;
        
        Ok(reconstructed_state)
    }
    
    pub async fn add_temporal_activation(
        &mut self,
        node_id: NodeId,
        activation: f32,
        timestamp: Timestamp,
        belief_context: TemporalBeliefContext,
    ) -> Result<(), TemporalError> {
        // Validate temporal constraints
        self.validate_temporal_activation(node_id, timestamp, &belief_context).await?;
        
        // Update activation timeline
        self.update_activation_timeline(node_id, activation, timestamp).await?;
        
        // Update validity intervals
        self.update_validity_intervals(node_id, timestamp, &belief_context).await?;
        
        // Update belief versions
        self.update_belief_versions(node_id, timestamp, &belief_context).await?;
        
        // Create snapshot if needed
        if self.should_create_snapshot(timestamp) {
            self.create_snapshot(timestamp).await?;
        }
        
        Ok(())
    }
    
    pub async fn query_activation_at_time(
        &self,
        node_id: NodeId,
        timestamp: Timestamp,
    ) -> Result<Option<TemporalActivationResult>, TemporalError> {
        // Check if node was valid at the given time
        if !self.is_node_valid_at_time(node_id, timestamp).await? {
            return Ok(None);
        }
        
        // Get activation level at timestamp
        let activation_level = self.get_activation_at_timestamp(node_id, timestamp).await?;
        
        // Get belief state at timestamp
        let belief_state = self.get_belief_state_at_timestamp(node_id, timestamp).await?;
        
        // Get justifications valid at timestamp
        let justifications = self.get_valid_justifications_at_time(node_id, timestamp).await?;
        
        Ok(Some(TemporalActivationResult {
            node_id,
            timestamp,
            activation_level,
            belief_state,
            justifications,
            validity_info: self.get_validity_info_at_time(node_id, timestamp).await?,
        }))
    }
    
    pub async fn trace_activation_evolution(
        &self,
        node_id: NodeId,
        start_time: Timestamp,
        end_time: Timestamp,
    ) -> Result<ActivationEvolution, TemporalError> {
        let timeline = self.activation_history.get(&node_id)
            .ok_or(TemporalError::NodeNotFound(node_id))?;
        
        // Extract activation points in the time range
        let activation_points: Vec<_> = timeline.activation_points
            .range(start_time..=end_time)
            .map(|(t, a)| (*t, *a))
            .collect();
        
        // Extract belief changes in the time range
        let belief_changes: Vec<_> = timeline.belief_changes
            .range(start_time..=end_time)
            .map(|(t, c)| (*t, c.clone()))
            .collect();
        
        // Calculate evolution metrics
        let evolution_metrics = self.calculate_evolution_metrics(
            &activation_points,
            &belief_changes,
        ).await?;
        
        Ok(ActivationEvolution {
            node_id,
            time_range: (start_time, end_time),
            activation_points,
            belief_changes,
            evolution_metrics,
            trend_analysis: self.analyze_activation_trend(&activation_points).await?,
        })
    }
    
    async fn find_base_snapshot(&self, timestamp: Timestamp) -> Result<&ActivationSnapshot, TemporalError> {
        // Find the latest snapshot before or at the timestamp
        self.temporal_snapshots
            .range(..=timestamp)
            .next_back()
            .map(|(_, snapshot)| snapshot)
            .ok_or(TemporalError::NoBaseSnapshot(timestamp))
    }
    
    async fn apply_incremental_changes(
        &self,
        state: &mut BeliefAwareActivationState,
        from_timestamp: Timestamp,
        to_timestamp: Timestamp,
    ) -> Result<(), TemporalError> {
        // Apply all changes that occurred between timestamps
        for (node_id, timeline) in &self.activation_history {
            // Get changes in the time window
            let changes: Vec<_> = timeline.activation_points
                .range(from_timestamp..=to_timestamp)
                .collect();
            
            // Apply latest change for each node
            if let Some((timestamp, activation)) = changes.last() {
                // Check if the belief was valid at target timestamp
                if self.is_node_valid_at_time(*node_id, to_timestamp).await? {
                    let belief_context = self.reconstruct_belief_context_at_time(
                        *node_id, 
                        to_timestamp
                    ).await?;
                    
                    state.set_belief_activation(*node_id, *activation, belief_context)?;
                }
            }
        }
        
        Ok(())
    }
    
    async fn validate_temporal_consistency(
        &self,
        state: &BeliefAwareActivationState,
        timestamp: Timestamp,
    ) -> Result<(), TemporalError> {
        // Validate that all active beliefs were valid at the timestamp
        for (node_id, belief_activation) in &state.belief_activations {
            if !self.is_node_valid_at_time(*node_id, timestamp).await? {
                return Err(TemporalError::InconsistentTemporalState {
                    node_id: *node_id,
                    timestamp,
                    reason: "Node not valid at timestamp".to_string(),
                });
            }
            
            // Validate justification chains were valid at timestamp
            for belief_id in &belief_activation.supporting_beliefs {
                if !self.is_belief_valid_at_time(*belief_id, timestamp).await? {
                    return Err(TemporalError::InconsistentTemporalState {
                        node_id: *node_id,
                        timestamp,
                        reason: format!("Supporting belief {:?} not valid at timestamp", belief_id),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    async fn is_node_valid_at_time(
        &self,
        node_id: NodeId,
        timestamp: Timestamp,
    ) -> Result<bool, TemporalError> {
        let intervals = self.validity_intervals.get(&node_id)
            .ok_or(TemporalError::NodeNotFound(node_id))?;
        
        for interval in intervals {
            if timestamp >= interval.start_time {
                if let Some(end_time) = interval.end_time {
                    if timestamp <= end_time {
                        return Ok(true);
                    }
                } else {
                    // No end time means still valid
                    return Ok(true);
                }
            }
        }
        
        Ok(false)
    }
}

#[derive(Debug, Clone)]
pub struct TemporalActivationResult {
    pub node_id: NodeId,
    pub timestamp: Timestamp,
    pub activation_level: f32,
    pub belief_state: BeliefActivation,
    pub justifications: Vec<JustificationId>,
    pub validity_info: ValidityInfo,
}

#[derive(Debug, Clone)]
pub struct ActivationEvolution {
    pub node_id: NodeId,
    pub time_range: (Timestamp, Timestamp),
    pub activation_points: Vec<(Timestamp, f32)>,
    pub belief_changes: Vec<(Timestamp, BeliefChange)>,
    pub evolution_metrics: EvolutionMetrics,
    pub trend_analysis: TrendAnalysis,
}

#[derive(Debug, Clone)]
pub struct EvolutionMetrics {
    pub total_changes: usize,
    pub average_activation: f32,
    pub peak_activation: f32,
    pub activation_variance: f32,
    pub stability_score: f32,
}
```

### Step 2: Time-Aware Spreader

```rust
// File: src/core/activation/time_aware_spreader.rs

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::core::activation::temporal_activation_state::{TemporalActivationState, TemporalContext};
use crate::core::activation::tms_constrained_spreader::TMSConstrainedSpreader;
use crate::core::graph::Graph;
use crate::core::types::{NodeId, Timestamp};
use crate::tms::{TruthMaintenanceSystem, TemporalBeliefGraph};

pub struct TimeAwareSpreader {
    // Base TMS-constrained spreader
    pub base_spreader: TMSConstrainedSpreader,
    
    // Temporal components
    pub temporal_belief_graph: Arc<RwLock<TemporalBeliefGraph>>,
    pub temporal_constraints: TemporalSpreadingConstraints,
    
    // Time-aware parameters
    pub temporal_decay_rate: f32,
    pub historical_weight_factor: f32,
    pub max_temporal_depth: Duration,
    
    // Caching for performance
    pub temporal_cache: HashMap<(Timestamp, NodeId), f32>,
    pub validity_cache: HashMap<(NodeId, Timestamp), bool>,
}

#[derive(Debug, Clone)]
pub struct TemporalSpreadingConstraints {
    pub max_historical_lookback: Duration,
    pub temporal_consistency_required: bool,
    pub allow_anachronistic_propagation: bool,
    pub temporal_inheritance_depth: usize,
}

impl TimeAwareSpreader {
    pub fn new(
        tms: Arc<RwLock<TruthMaintenanceSystem>>,
        temporal_belief_graph: Arc<RwLock<TemporalBeliefGraph>>,
    ) -> Self {
        Self {
            base_spreader: TMSConstrainedSpreader::new(tms),
            temporal_belief_graph,
            temporal_constraints: TemporalSpreadingConstraints::default(),
            temporal_decay_rate: 0.95,
            historical_weight_factor: 0.8,
            max_temporal_depth: Duration::from_secs(365 * 24 * 3600), // 1 year
            temporal_cache: HashMap::new(),
            validity_cache: HashMap::new(),
        }
    }
    
    pub async fn spread_at_timestamp(
        &mut self,
        initial_activations: HashMap<NodeId, f32>,
        graph: &Graph,
        timestamp: Timestamp,
        temporal_context: TemporalContext,
    ) -> Result<TemporalActivationState, TemporalSpreadingError> {
        // Reconstruct graph state at timestamp
        let historical_graph = self.reconstruct_graph_at_timestamp(graph, timestamp).await?;
        
        // Filter initial activations based on temporal validity
        let valid_activations = self.filter_temporally_valid_activations(
            initial_activations,
            timestamp,
        ).await?;
        
        // Create temporal belief context for the timestamp
        let belief_context = self.create_temporal_belief_context(
            timestamp,
            &temporal_context,
        ).await?;
        
        // Perform base spreading with temporal constraints
        let base_result = self.base_spreader.spread_with_beliefs(
            valid_activations,
            &historical_graph,
            belief_context,
        ).await?;
        
        // Create temporal activation state
        let mut temporal_state = TemporalActivationState::new(
            base_result,
            temporal_context,
        );
        
        // Set query timestamp
        temporal_state.query_timestamp = Some(timestamp);
        
        // Populate historical data for activated nodes
        self.populate_historical_data(&mut temporal_state, timestamp).await?;
        
        Ok(temporal_state)
    }
    
    pub async fn spread_over_time_range(
        &mut self,
        initial_activations: HashMap<NodeId, f32>,
        graph: &Graph,
        start_time: Timestamp,
        end_time: Timestamp,
        time_resolution: Duration,
        temporal_context: TemporalContext,
    ) -> Result<TemporalRangeResult, TemporalSpreadingError> {
        let mut time_results = HashMap::new();
        let mut current_time = start_time;
        
        while current_time <= end_time {
            let result = self.spread_at_timestamp(
                initial_activations.clone(),
                graph,
                current_time,
                temporal_context.clone(),
            ).await?;
            
            time_results.insert(current_time, result);
            current_time += time_resolution;
        }
        
        // Analyze temporal patterns across the range
        let pattern_analysis = self.analyze_temporal_patterns(&time_results).await?;
        
        // Calculate temporal metrics
        let temporal_metrics = self.calculate_temporal_metrics(&time_results).await?;
        
        Ok(TemporalRangeResult {
            time_results,
            pattern_analysis,
            temporal_metrics,
            time_range: (start_time, end_time),
        })
    }
    
    pub async fn spread_with_temporal_inheritance(
        &mut self,
        initial_activations: HashMap<NodeId, f32>,
        graph: &Graph,
        query_time: Timestamp,
        inheritance_rules: Vec<TemporalInheritanceRule>,
    ) -> Result<TemporalActivationState, TemporalSpreadingError> {
        // Apply temporal inheritance to initial activations
        let inherited_activations = self.apply_temporal_inheritance(
            initial_activations,
            query_time,
            &inheritance_rules,
        ).await?;
        
        // Create temporal context with inheritance
        let temporal_context = TemporalContext {
            context_id: ContextId::new(),
            time_reference: TimeReference::Absolute(query_time),
            temporal_scope: TemporalScope::PointInTime,
            inheritance_rules,
        };
        
        // Perform temporal spreading
        self.spread_at_timestamp(
            inherited_activations,
            graph,
            query_time,
            temporal_context,
        ).await
    }
    
    async fn reconstruct_graph_at_timestamp(
        &self,
        graph: &Graph,
        timestamp: Timestamp,
    ) -> Result<Graph, TemporalSpreadingError> {
        let temporal_graph_guard = self.temporal_belief_graph.read().await;
        
        // Get graph snapshot at timestamp
        let historical_snapshot = temporal_graph_guard
            .get_graph_snapshot_at_time(timestamp)
            .await?;
        
        if let Some(snapshot) = historical_snapshot {
            Ok(snapshot.to_graph())
        } else {
            // If no snapshot, reconstruct from closest and apply changes
            let closest_snapshot = temporal_graph_guard
                .get_closest_snapshot_before(timestamp)
                .await?;
            
            if let Some(base_snapshot) = closest_snapshot {
                let mut reconstructed = base_snapshot.to_graph();
                
                // Apply incremental changes
                let changes = temporal_graph_guard
                    .get_changes_since(base_snapshot.timestamp, timestamp)
                    .await?;
                
                for change in changes {
                    reconstructed.apply_temporal_change(change)?;
                }
                
                Ok(reconstructed)
            } else {
                // Fallback to current graph (may not be historically accurate)
                Ok(graph.clone())
            }
        }
    }
    
    async fn filter_temporally_valid_activations(
        &self,
        activations: HashMap<NodeId, f32>,
        timestamp: Timestamp,
    ) -> Result<HashMap<NodeId, f32>, TemporalSpreadingError> {
        let mut valid_activations = HashMap::new();
        
        let temporal_graph_guard = self.temporal_belief_graph.read().await;
        
        for (node_id, activation) in activations {
            // Check if node existed at timestamp
            if temporal_graph_guard.node_existed_at_time(node_id, timestamp).await? {
                // Check if node was valid (not retracted) at timestamp
                if temporal_graph_guard.node_valid_at_time(node_id, timestamp).await? {
                    valid_activations.insert(node_id, activation);
                }
            }
        }
        
        Ok(valid_activations)
    }
    
    async fn apply_temporal_inheritance(
        &self,
        activations: HashMap<NodeId, f32>,
        query_time: Timestamp,
        inheritance_rules: &[TemporalInheritanceRule],
    ) -> Result<HashMap<NodeId, f32>, TemporalSpreadingError> {
        let mut inherited_activations = activations.clone();
        
        for rule in inheritance_rules {
            match rule {
                TemporalInheritanceRule::DecayOverTime { decay_rate, .. } => {
                    // Apply time-based decay to activations
                    for (node_id, activation) in &mut inherited_activations {
                        let age = query_time.duration_since(
                            self.get_node_creation_time(*node_id).await?
                        )?;
                        
                        let decay_factor = decay_rate.powf(age.as_secs_f32() / 86400.0); // Daily decay
                        *activation *= decay_factor;
                    }
                }
                
                TemporalInheritanceRule::InheritFromAncestors { depth, weight } => {
                    // Inherit activation from temporal ancestors
                    let ancestors = self.find_temporal_ancestors(
                        &activations.keys().cloned().collect(),
                        query_time,
                        *depth,
                    ).await?;
                    
                    for (ancestor_node, ancestor_activation) in ancestors {
                        let inherited_value = ancestor_activation * weight;
                        *inherited_activations.entry(ancestor_node).or_insert(0.0) += inherited_value;
                    }
                }
                
                TemporalInheritanceRule::ContextualInheritance { context_filter, .. } => {
                    // Apply context-specific inheritance logic
                    self.apply_contextual_inheritance(
                        &mut inherited_activations,
                        query_time,
                        context_filter,
                    ).await?;
                }
            }
        }
        
        Ok(inherited_activations)
    }
    
    async fn populate_historical_data(
        &self,
        temporal_state: &mut TemporalActivationState,
        query_timestamp: Timestamp,
    ) -> Result<(), TemporalSpreadingError> {
        let temporal_graph_guard = self.temporal_belief_graph.read().await;
        
        for node_id in temporal_state.current_state.base_state.activated_nodes() {
            // Get activation history for node
            let history = temporal_graph_guard
                .get_activation_history(node_id)
                .await?;
            
            if let Some(timeline) = history {
                temporal_state.activation_history.insert(node_id, timeline);
            }
            
            // Get validity intervals
            let validity_intervals = temporal_graph_guard
                .get_validity_intervals(node_id)
                .await?;
            
            temporal_state.validity_intervals.insert(node_id, validity_intervals);
            
            // Get belief version history
            if let Some(belief_activation) = temporal_state.current_state.get_belief_activation(node_id) {
                for belief_id in &belief_activation.supporting_beliefs {
                    let versions = temporal_graph_guard
                        .get_belief_version_history(*belief_id)
                        .await?;
                    
                    temporal_state.belief_versions.insert(*belief_id, versions);
                }
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TemporalRangeResult {
    pub time_results: HashMap<Timestamp, TemporalActivationState>,
    pub pattern_analysis: TemporalPatternAnalysis,
    pub temporal_metrics: TemporalMetrics,
    pub time_range: (Timestamp, Timestamp),
}

#[derive(Debug, Clone)]
pub struct TemporalPatternAnalysis {
    pub activation_trends: HashMap<NodeId, ActivationTrend>,
    pub belief_evolution_patterns: Vec<BeliefEvolutionPattern>,
    pub temporal_correlations: Vec<TemporalCorrelation>,
    pub anomaly_detection: Vec<TemporalAnomaly>,
}

#[derive(Debug, Clone)]
pub enum ActivationTrend {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
    Irregular,
}
```

### Step 3: Temporal Belief Resolver

```rust
// File: src/core/temporal/temporal_belief_resolver.rs

use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::core::types::{NodeId, Timestamp};
use crate::tms::{BeliefId, BeliefState, JustificationChain, TemporalBeliefGraph};

pub struct TemporalBeliefResolver {
    // Temporal belief graph
    pub temporal_graph: Arc<RwLock<TemporalBeliefGraph>>,
    
    // Resolution configuration
    pub resolution_strategy: ResolutionStrategy,
    pub temporal_tolerance: Duration,
    pub max_inheritance_depth: usize,
    
    // Caching for performance
    pub belief_cache: HashMap<(BeliefId, Timestamp), ResolvedBelief>,
    pub justification_cache: HashMap<(NodeId, Timestamp), Vec<JustificationChain>>,
}

#[derive(Debug, Clone)]
pub enum ResolutionStrategy {
    Strict,      // Only exact timestamp matches
    Interpolated, // Interpolate between known states
    NearestValid, // Use nearest valid state
    Inherited,   // Use temporal inheritance rules
}

#[derive(Debug, Clone)]
pub struct ResolvedBelief {
    pub belief_id: BeliefId,
    pub node_id: NodeId,
    pub timestamp: Timestamp,
    pub belief_state: BeliefState,
    pub confidence: f32,
    pub justifications: Vec<JustificationChain>,
    pub resolution_method: ResolutionMethod,
    pub temporal_validity: TemporalValidity,
}

#[derive(Debug, Clone)]
pub enum ResolutionMethod {
    Exact,                    // Found exact timestamp match
    Interpolated(f32),        // Interpolated with given weight
    Inherited(BeliefId),      // Inherited from another belief
    Default(String),          // Used default value with reason
}

impl TemporalBeliefResolver {
    pub fn new(temporal_graph: Arc<RwLock<TemporalBeliefGraph>>) -> Self {
        Self {
            temporal_graph,
            resolution_strategy: ResolutionStrategy::Interpolated,
            temporal_tolerance: Duration::from_secs(3600), // 1 hour
            max_inheritance_depth: 5,
            belief_cache: HashMap::new(),
            justification_cache: HashMap::new(),
        }
    }
    
    pub async fn resolve_belief_at_time(
        &mut self,
        belief_id: BeliefId,
        timestamp: Timestamp,
    ) -> Result<ResolvedBelief, ResolutionError> {
        // Check cache first
        if let Some(cached) = self.belief_cache.get(&(belief_id, timestamp)) {
            return Ok(cached.clone());
        }
        
        let temporal_graph_guard = self.temporal_graph.read().await;
        
        // Try exact timestamp match first
        if let Some(exact_belief) = temporal_graph_guard
            .get_belief_at_exact_time(belief_id, timestamp)
            .await? {
            
            let resolved = ResolvedBelief {
                belief_id,
                node_id: exact_belief.node_id,
                timestamp,
                belief_state: exact_belief.state,
                confidence: exact_belief.confidence,
                justifications: exact_belief.justifications,
                resolution_method: ResolutionMethod::Exact,
                temporal_validity: exact_belief.validity,
            };
            
            self.belief_cache.insert((belief_id, timestamp), resolved.clone());
            return Ok(resolved);
        }
        
        // Apply resolution strategy for non-exact matches
        let resolved = match self.resolution_strategy {
            ResolutionStrategy::Strict => {
                return Err(ResolutionError::NoExactMatch(belief_id, timestamp));
            }
            
            ResolutionStrategy::Interpolated => {
                self.resolve_by_interpolation(belief_id, timestamp, &*temporal_graph_guard).await?
            }
            
            ResolutionStrategy::NearestValid => {
                self.resolve_by_nearest_valid(belief_id, timestamp, &*temporal_graph_guard).await?
            }
            
            ResolutionStrategy::Inherited => {
                self.resolve_by_inheritance(belief_id, timestamp, &*temporal_graph_guard).await?
            }
        };
        
        self.belief_cache.insert((belief_id, timestamp), resolved.clone());
        Ok(resolved)
    }
    
    pub async fn resolve_justifications_at_time(
        &mut self,
        node_id: NodeId,
        timestamp: Timestamp,
    ) -> Result<Vec<JustificationChain>, ResolutionError> {
        // Check cache
        if let Some(cached) = self.justification_cache.get(&(node_id, timestamp)) {
            return Ok(cached.clone());
        }
        
        let temporal_graph_guard = self.temporal_graph.read().await;
        
        // Get all justification chains that were valid at the timestamp
        let mut valid_justifications = Vec::new();
        
        let all_justifications = temporal_graph_guard
            .get_all_justifications_for_node(node_id)
            .await?;
        
        for justification in all_justifications {
            if self.is_justification_valid_at_time(&justification, timestamp, &*temporal_graph_guard).await? {
                valid_justifications.push(justification);
            }
        }
        
        // Sort by strength and confidence
        valid_justifications.sort_by(|a, b| {
            let score_a = a.total_strength * a.min_confidence();
            let score_b = b.total_strength * b.min_confidence();
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        self.justification_cache.insert((node_id, timestamp), valid_justifications.clone());
        Ok(valid_justifications)
    }
    
    pub async fn resolve_belief_evolution(
        &mut self,
        belief_id: BeliefId,
        start_time: Timestamp,
        end_time: Timestamp,
        resolution: Duration,
    ) -> Result<BeliefEvolution, ResolutionError> {
        let mut evolution_points = Vec::new();
        let mut current_time = start_time;
        
        while current_time <= end_time {
            match self.resolve_belief_at_time(belief_id, current_time).await {
                Ok(resolved) => {
                    evolution_points.push((current_time, resolved));
                }
                Err(ResolutionError::NoExactMatch(_, _)) => {
                    // Skip points where belief didn't exist
                }
                Err(e) => return Err(e),
            }
            
            current_time += resolution;
        }
        
        // Analyze evolution patterns
        let patterns = self.analyze_evolution_patterns(&evolution_points).await?;
        
        Ok(BeliefEvolution {
            belief_id,
            time_range: (start_time, end_time),
            evolution_points,
            patterns,
            stability_metrics: self.calculate_stability_metrics(&evolution_points).await?,
        })
    }
    
    async fn resolve_by_interpolation(
        &self,
        belief_id: BeliefId,
        timestamp: Timestamp,
        temporal_graph: &TemporalBeliefGraph,
    ) -> Result<ResolvedBelief, ResolutionError> {
        // Find the closest beliefs before and after timestamp
        let before = temporal_graph
            .get_closest_belief_before(belief_id, timestamp)
            .await?;
        
        let after = temporal_graph
            .get_closest_belief_after(belief_id, timestamp)
            .await?;
        
        match (before, after) {
            (Some(before_belief), Some(after_belief)) => {
                // Interpolate between the two beliefs
                let total_duration = after_belief.timestamp.duration_since(before_belief.timestamp)?;
                let elapsed_duration = timestamp.duration_since(before_belief.timestamp)?;
                
                let interpolation_weight = elapsed_duration.as_secs_f32() / total_duration.as_secs_f32();
                let interpolation_weight = interpolation_weight.clamp(0.0, 1.0);
                
                // Interpolate confidence
                let interpolated_confidence = before_belief.confidence * (1.0 - interpolation_weight) +
                    after_belief.confidence * interpolation_weight;
                
                // Use the belief state from the closer timestamp
                let (belief_state, base_justifications) = if interpolation_weight < 0.5 {
                    (before_belief.state.clone(), before_belief.justifications.clone())
                } else {
                    (after_belief.state.clone(), after_belief.justifications.clone())
                };
                
                Ok(ResolvedBelief {
                    belief_id,
                    node_id: before_belief.node_id, // Should be same for both
                    timestamp,
                    belief_state,
                    confidence: interpolated_confidence,
                    justifications: base_justifications,
                    resolution_method: ResolutionMethod::Interpolated(interpolation_weight),
                    temporal_validity: self.compute_interpolated_validity(
                        &before_belief,
                        &after_belief,
                        interpolation_weight,
                    ).await?,
                })
            }
            
            (Some(before_belief), None) => {
                // Only have before belief, use with decay
                let age = timestamp.duration_since(before_belief.timestamp)?;
                let decay_factor = self.calculate_temporal_decay(age);
                
                Ok(ResolvedBelief {
                    belief_id,
                    node_id: before_belief.node_id,
                    timestamp,
                    belief_state: before_belief.state.clone(),
                    confidence: before_belief.confidence * decay_factor,
                    justifications: before_belief.justifications.clone(),
                    resolution_method: ResolutionMethod::Interpolated(decay_factor),
                    temporal_validity: before_belief.validity.clone(),
                })
            }
            
            (None, Some(after_belief)) => {
                // Only have after belief, use with reduced confidence
                let anticipation_penalty = 0.7; // Reduce confidence for future beliefs
                
                Ok(ResolvedBelief {
                    belief_id,
                    node_id: after_belief.node_id,
                    timestamp,
                    belief_state: after_belief.state.clone(),
                    confidence: after_belief.confidence * anticipation_penalty,
                    justifications: after_belief.justifications.clone(),
                    resolution_method: ResolutionMethod::Interpolated(anticipation_penalty),
                    temporal_validity: after_belief.validity.clone(),
                })
            }
            
            (None, None) => {
                Err(ResolutionError::BeliefNeverExisted(belief_id))
            }
        }
    }
    
    async fn resolve_by_inheritance(
        &self,
        belief_id: BeliefId,
        timestamp: Timestamp,
        temporal_graph: &TemporalBeliefGraph,
    ) -> Result<ResolvedBelief, ResolutionError> {
        // Find parent beliefs that might provide inheritance
        let inheritance_candidates = temporal_graph
            .find_inheritance_candidates(belief_id, timestamp)
            .await?;
        
        for candidate in inheritance_candidates {
            if let Ok(inherited_belief) = self.apply_inheritance_rules(
                belief_id,
                &candidate,
                timestamp,
                temporal_graph,
            ).await {
                return Ok(inherited_belief);
            }
        }
        
        Err(ResolutionError::NoInheritanceAvailable(belief_id, timestamp))
    }
    
    fn calculate_temporal_decay(&self, age: Duration) -> f32 {
        // Exponential decay based on age
        let decay_rate = 0.99; // Per day
        let days = age.as_secs_f32() / 86400.0;
        decay_rate.powf(days).max(0.1) // Minimum 10% retention
    }
}

#[derive(Debug, Clone)]
pub struct BeliefEvolution {
    pub belief_id: BeliefId,
    pub time_range: (Timestamp, Timestamp),
    pub evolution_points: Vec<(Timestamp, ResolvedBelief)>,
    pub patterns: EvolutionPatterns,
    pub stability_metrics: StabilityMetrics,
}

#[derive(Debug, Clone)]
pub struct EvolutionPatterns {
    pub trend: EvolutionTrend,
    pub volatility: f32,
    pub change_points: Vec<Timestamp>,
    pub confidence_trajectory: Vec<(Timestamp, f32)>,
}

#[derive(Debug, Clone)]
pub enum EvolutionTrend {
    Stable,
    IncreasingConfidence,
    DecreasingConfidence,
    Cyclical,
    Chaotic,
}
```

### Step 4: Chronological Query Processor

```rust
// File: src/core/query/chronological_query_processor.rs

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::core::activation::temporal_activation_state::TemporalActivationState;
use crate::core::activation::time_aware_spreader::TimeAwareSpreader;
use crate::core::temporal::temporal_belief_resolver::TemporalBeliefResolver;
use crate::core::query::intent::{QueryIntent, TemporalQueryIntent};

pub struct ChronologicalQueryProcessor {
    // Core components
    pub time_aware_spreader: TimeAwareSpreader,
    pub belief_resolver: TemporalBeliefResolver,
    
    // Query configuration
    pub default_time_tolerance: Duration,
    pub max_historical_depth: Duration,
    pub temporal_cache_size: usize,
    
    // Performance optimization
    pub enable_temporal_caching: bool,
    pub prefetch_related_times: bool,
}

#[derive(Debug, Clone)]
pub struct ChronologicalQueryResult {
    // Core results
    pub temporal_activations: TemporalActivationState,
    
    // Time-specific analysis
    pub temporal_analysis: TemporalAnalysis,
    pub historical_context: HistoricalContext,
    
    // Query metadata
    pub query_timestamp: Timestamp,
    pub temporal_scope: TemporalScope,
    pub resolution_quality: ResolutionQuality,
    
    // Recommendations
    pub temporal_insights: Vec<TemporalInsight>,
    pub related_time_periods: Vec<RelatedTimePeriod>,
}

impl ChronologicalQueryProcessor {
    pub async fn process_temporal_query(
        &mut self,
        query_intent: &QueryIntent,
        temporal_intent: &TemporalQueryIntent,
        graph: &Graph,
    ) -> Result<ChronologicalQueryResult, TemporalQueryError> {
        match temporal_intent {
            TemporalQueryIntent::PointInTime { timestamp } => {
                self.process_point_in_time_query(query_intent, *timestamp, graph).await
            }
            
            TemporalQueryIntent::TimeRange { start, end, resolution } => {
                self.process_time_range_query(query_intent, *start, *end, *resolution, graph).await
            }
            
            TemporalQueryIntent::Evolution { entity, start, end } => {
                self.process_evolution_query(query_intent, *entity, *start, *end, graph).await
            }
            
            TemporalQueryIntent::Comparative { timestamps } => {
                self.process_comparative_query(query_intent, timestamps.clone(), graph).await
            }
        }
    }
    
    async fn process_point_in_time_query(
        &mut self,
        query_intent: &QueryIntent,
        timestamp: Timestamp,
        graph: &Graph,
    ) -> Result<ChronologicalQueryResult, TemporalQueryError> {
        // Extract initial activations from query
        let initial_activations = self.extract_query_activations(query_intent).await?;
        
        // Create temporal context
        let temporal_context = TemporalContext {
            context_id: ContextId::new(),
            time_reference: TimeReference::Absolute(timestamp),
            temporal_scope: TemporalScope::PointInTime,
            inheritance_rules: self.get_default_inheritance_rules(),
        };
        
        // Perform temporal spreading
        let temporal_activations = self.time_aware_spreader.spread_at_timestamp(
            initial_activations,
            graph,
            timestamp,
            temporal_context,
        ).await?;
        
        // Analyze temporal aspects
        let temporal_analysis = self.analyze_temporal_aspects(
            &temporal_activations,
            timestamp,
        ).await?;
        
        // Gather historical context
        let historical_context = self.gather_historical_context(
            &temporal_activations,
            timestamp,
        ).await?;
        
        // Generate insights
        let temporal_insights = self.generate_temporal_insights(
            &temporal_activations,
            &temporal_analysis,
        ).await?;
        
        // Find related time periods
        let related_time_periods = self.find_related_time_periods(
            &temporal_activations,
            timestamp,
        ).await?;
        
        Ok(ChronologicalQueryResult {
            temporal_activations,
            temporal_analysis,
            historical_context,
            query_timestamp: timestamp,
            temporal_scope: TemporalScope::PointInTime,
            resolution_quality: self.assess_resolution_quality(&temporal_activations).await?,
            temporal_insights,
            related_time_periods,
        })
    }
    
    async fn process_time_range_query(
        &mut self,
        query_intent: &QueryIntent,
        start_time: Timestamp,
        end_time: Timestamp,
        resolution: Duration,
        graph: &Graph,
    ) -> Result<ChronologicalQueryResult, TemporalQueryError> {
        // Extract initial activations
        let initial_activations = self.extract_query_activations(query_intent).await?;
        
        // Create temporal context for range
        let temporal_context = TemporalContext {
            context_id: ContextId::new(),
            time_reference: TimeReference::Range(start_time, end_time),
            temporal_scope: TemporalScope::TimeRange,
            inheritance_rules: self.get_default_inheritance_rules(),
        };
        
        // Perform range spreading
        let range_result = self.time_aware_spreader.spread_over_time_range(
            initial_activations,
            graph,
            start_time,
            end_time,
            resolution,
            temporal_context,
        ).await?;
        
        // Synthesize range results into single temporal activation state
        let synthesized_state = self.synthesize_range_results(&range_result).await?;
        
        // Analyze temporal patterns across the range
        let temporal_analysis = self.analyze_range_patterns(&range_result).await?;
        
        // Create historical context for the entire range
        let historical_context = self.create_range_historical_context(&range_result).await?;
        
        // Generate range-specific insights
        let temporal_insights = self.generate_range_insights(&range_result).await?;
        
        Ok(ChronologicalQueryResult {
            temporal_activations: synthesized_state,
            temporal_analysis,
            historical_context,
            query_timestamp: start_time, // Use start as representative timestamp
            temporal_scope: TemporalScope::TimeRange,
            resolution_quality: self.assess_range_resolution_quality(&range_result).await?,
            temporal_insights,
            related_time_periods: Vec::new(), // TODO: Implement for ranges
        })
    }
}
```

## File Locations

- `src/core/activation/temporal_activation_state.rs` - Temporal activation state
- `src/core/activation/time_aware_spreader.rs` - Time-aware spreading algorithm
- `src/core/temporal/temporal_belief_resolver.rs` - Temporal belief resolution
- `src/core/query/chronological_query_processor.rs` - High-level temporal query processor
- `tests/temporal/temporal_activation_tests.rs` - Comprehensive test suite

## Success Criteria

- [ ] TemporalActivationState handles time-based activation tracking
- [ ] TimeAwareSpreader performs historical spreading activation correctly
- [ ] TemporalBeliefResolver resolves beliefs at any timestamp
- [ ] ChronologicalQueryProcessor supports all temporal query types
- [ ] Historical reconstruction works within 5ms for recent queries
- [ ] Time-range queries scale efficiently with resolution
- [ ] All tests pass:
  - Point-in-time query processing
  - Time-range spreading activation
  - Temporal belief resolution
  - Historical state reconstruction
  - Temporal consistency validation
  - Evolution pattern analysis

## Test Requirements

```rust
#[test]
async fn test_temporal_activation_reconstruction() {
    let temporal_graph = setup_temporal_graph().await;
    let state = setup_temporal_activation_state().await;
    
    let past_timestamp = SystemTime::now() - Duration::from_secs(3600); // 1 hour ago
    
    let reconstructed = state.reconstruct_at_timestamp(past_timestamp)
        .await.unwrap();
    
    // Verify activation levels are historically accurate
    assert!(reconstructed.base_state.activated_nodes().len() > 0);
    
    // Verify temporal consistency
    for (node_id, belief_activation) in &reconstructed.belief_activations {
        assert!(state.is_node_valid_at_time(*node_id, past_timestamp).await.unwrap());
    }
}

#[test]
async fn test_time_aware_spreading() {
    let tms = setup_test_tms().await;
    let temporal_graph = setup_temporal_graph().await;
    let mut spreader = TimeAwareSpreader::new(tms, temporal_graph);
    
    let initial_activations = HashMap::from([
        (NodeId(1), 0.8),
        (NodeId(2), 0.6),
    ]);
    
    let timestamp = SystemTime::now() - Duration::from_secs(1800); // 30 minutes ago
    let temporal_context = TemporalContext::default();
    
    let result = spreader.spread_at_timestamp(
        initial_activations,
        &test_graph(),
        timestamp,
        temporal_context,
    ).await.unwrap();
    
    assert!(result.query_timestamp.is_some());
    assert_eq!(result.query_timestamp.unwrap(), timestamp);
    assert!(!result.activation_history.is_empty());
}

#[test]
async fn test_temporal_belief_resolution() {
    let temporal_graph = setup_temporal_graph().await;
    let mut resolver = TemporalBeliefResolver::new(temporal_graph);
    
    let belief_id = BeliefId(1);
    let timestamp = SystemTime::now() - Duration::from_secs(7200); // 2 hours ago
    
    let resolved = resolver.resolve_belief_at_time(belief_id, timestamp)
        .await.unwrap();
    
    assert_eq!(resolved.belief_id, belief_id);
    assert_eq!(resolved.timestamp, timestamp);
    assert!(resolved.confidence > 0.0);
    assert!(!resolved.justifications.is_empty());
}

#[test]
async fn test_evolution_tracking() {
    let temporal_graph = setup_temporal_graph().await;
    let state = setup_temporal_activation_state().await;
    
    let node_id = NodeId(1);
    let start_time = SystemTime::now() - Duration::from_secs(86400); // 1 day ago
    let end_time = SystemTime::now();
    
    let evolution = state.trace_activation_evolution(node_id, start_time, end_time)
        .await.unwrap();
    
    assert_eq!(evolution.node_id, node_id);
    assert!(!evolution.activation_points.is_empty());
    assert!(evolution.evolution_metrics.total_changes > 0);
}
```

## Quality Gates

- [ ] Historical reconstruction latency <5ms for 1-hour lookback
- [ ] Time-range queries scale linearly with time resolution
- [ ] Temporal belief resolution accuracy >95% for interpolated values
- [ ] Memory usage for temporal data stays within 30% overhead
- [ ] Temporal consistency validation catches all inconsistencies
- [ ] Evolution pattern analysis identifies trends correctly

## Next Task

Upon completion, proceed to **33_context_switching.md**