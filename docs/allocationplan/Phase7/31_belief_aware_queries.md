# Micro Task 31: Belief-Aware Queries with TMS Integration

**Priority**: CRITICAL  
**Estimated Time**: 55 minutes  
**Dependencies**: Phase 6 TMS, Tasks 01-30 (complete spreading activation system)  
**Skills Required**: Rust async programming, belief systems, query processing

## Objective

Integrate the Truth Maintenance System (TMS) with the spreading activation query engine to enable belief-aware queries that consider justification chains, confidence levels, and truth maintenance constraints during activation propagation.

## Context

This task bridges the gap between pure activation spreading and intelligent belief reasoning. It ensures that query results are not just semantically related but also logically justified and consistent with the system's belief state.

## Specifications

### Required Components

1. **BeliefAwareActivationState**
   - Extends basic ActivationState with belief context
   - Tracks justification strength for each activated node
   - Maintains belief revision history during activation
   - Supports confidence-weighted activation propagation

2. **TMSConstrainedSpreader**
   - Spreader algorithm that respects TMS constraints
   - Blocks activation along retracted justification paths
   - Applies belief confidence as activation multipliers
   - Handles context switching during propagation

3. **JustificationPathTracer**
   - Traces belief justification chains during activation
   - Maps activation paths to logical inference chains
   - Identifies weak justification links
   - Supports belief chain reconstruction

4. **BeliefQueryProcessor**
   - High-level interface for belief-aware queries
   - Integrates query intent with belief context
   - Handles multi-context belief queries
   - Provides justification explanations

### Performance Requirements

- Belief constraint checking: <0.5ms per node
- Justification tracing overhead: <20% additional latency
- Context switching: <1ms between belief contexts
- Confidence calculation: <0.1ms per activated node

## Implementation Guide

### Step 1: Belief-Aware Activation State

```rust
// File: src/core/activation/belief_aware_state.rs

use std::collections::HashMap;
use std::sync::Arc;
use crate::core::activation::state::{ActivationState, ActivationFrame};
use crate::tms::{BeliefId, JustificationId, BeliefStatus, ConfidenceLevel};
use crate::core::types::{NodeId, ContextId};

#[derive(Debug, Clone)]
pub struct BeliefAwareActivationState {
    // Base activation state
    pub base_state: ActivationState,
    
    // Belief-specific tracking
    pub belief_activations: HashMap<NodeId, BeliefActivation>,
    
    // Active justification chains
    pub active_justifications: HashMap<NodeId, Vec<JustificationChain>>,
    
    // Current belief context
    pub belief_context: ContextId,
    
    // TMS constraints for this activation
    pub tms_constraints: TMSConstraints,
    
    // Belief revision tracking
    pub revision_history: Vec<BeliefRevision>,
}

#[derive(Debug, Clone)]
pub struct BeliefActivation {
    // Standard activation level
    pub activation_level: f32,
    
    // Belief confidence influencing activation
    pub confidence_level: ConfidenceLevel,
    
    // Justification strength
    pub justification_strength: f32,
    
    // Belief status (IN/OUT in TMS terms)
    pub belief_status: BeliefStatus,
    
    // Source beliefs supporting this activation
    pub supporting_beliefs: Vec<BeliefId>,
    
    // Activation timestamp
    pub activation_timestamp: std::time::Instant,
}

#[derive(Debug, Clone)]
pub struct JustificationChain {
    pub chain_id: JustificationId,
    pub belief_path: Vec<BeliefId>,
    pub confidence_path: Vec<f32>,
    pub total_strength: f32,
    pub is_valid: bool,
}

impl BeliefAwareActivationState {
    pub fn new(belief_context: ContextId, tms_constraints: TMSConstraints) -> Self {
        Self {
            base_state: ActivationState::new(),
            belief_activations: HashMap::new(),
            active_justifications: HashMap::new(),
            belief_context,
            tms_constraints,
            revision_history: Vec::new(),
        }
    }
    
    pub fn set_belief_activation(
        &mut self, 
        node: NodeId, 
        activation: f32, 
        belief_context: BeliefActivationContext
    ) -> Result<(), BeliefActivationError> {
        // Check TMS constraints before allowing activation
        if !self.tms_constraints.allows_activation(node, &belief_context)? {
            return Err(BeliefActivationError::TMSConstraintViolation(node));
        }
        
        // Calculate justification-weighted activation
        let weighted_activation = self.calculate_justified_activation(
            activation, 
            &belief_context
        )?;
        
        // Update base activation
        self.base_state.set_activation(node, weighted_activation);
        
        // Update belief-specific activation
        self.belief_activations.insert(node, BeliefActivation {
            activation_level: weighted_activation,
            confidence_level: belief_context.confidence,
            justification_strength: belief_context.justification_strength,
            belief_status: belief_context.status,
            supporting_beliefs: belief_context.supporting_beliefs.clone(),
            activation_timestamp: std::time::Instant::now(),
        });
        
        // Trace justification chains
        self.trace_justification_chains(node, &belief_context)?;
        
        Ok(())
    }
    
    pub fn get_belief_activation(&self, node: NodeId) -> Option<&BeliefActivation> {
        self.belief_activations.get(&node)
    }
    
    fn calculate_justified_activation(
        &self, 
        base_activation: f32, 
        context: &BeliefActivationContext
    ) -> Result<f32, BeliefActivationError> {
        // Weight activation by belief confidence
        let confidence_multiplier = match context.confidence {
            ConfidenceLevel::High => 1.0,
            ConfidenceLevel::Medium => 0.8,
            ConfidenceLevel::Low => 0.6,
            ConfidenceLevel::Uncertain => 0.4,
        };
        
        // Weight by justification strength
        let justification_multiplier = context.justification_strength.clamp(0.1, 1.0);
        
        // Combine weights
        let weighted_activation = base_activation * confidence_multiplier * justification_multiplier;
        
        Ok(weighted_activation.clamp(0.0, 1.0))
    }
    
    fn trace_justification_chains(
        &mut self, 
        node: NodeId, 
        context: &BeliefActivationContext
    ) -> Result<(), BeliefActivationError> {
        // Build justification chain for this activation
        let mut chain = JustificationChain {
            chain_id: JustificationId::new(),
            belief_path: context.supporting_beliefs.clone(),
            confidence_path: Vec::new(),
            total_strength: context.justification_strength,
            is_valid: true,
        };
        
        // Calculate confidence path
        for belief_id in &context.supporting_beliefs {
            let confidence = self.get_belief_confidence(*belief_id)?;
            chain.confidence_path.push(confidence);
        }
        
        // Validate chain consistency
        chain.is_valid = self.validate_justification_chain(&chain)?;
        
        // Store chain
        self.active_justifications
            .entry(node)
            .or_insert_with(Vec::new)
            .push(chain);
        
        Ok(())
    }
    
    fn validate_justification_chain(&self, chain: &JustificationChain) -> Result<bool, BeliefActivationError> {
        // Check if all beliefs in chain are currently believed (IN status)
        for belief_id in &chain.belief_path {
            if !self.is_belief_currently_in(*belief_id)? {
                return Ok(false);
            }
        }
        
        // Check for circular dependencies
        if self.has_circular_dependency(&chain.belief_path)? {
            return Ok(false);
        }
        
        // Check confidence threshold
        let min_confidence = chain.confidence_path.iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&0.0);
        
        Ok(*min_confidence >= self.tms_constraints.min_confidence_threshold)
    }
}
```

### Step 2: TMS-Constrained Spreader

```rust
// File: src/core/activation/tms_constrained_spreader.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::core::activation::belief_aware_state::BeliefAwareActivationState;
use crate::core::graph::Graph;
use crate::tms::{TruthMaintenanceSystem, BeliefContext};

pub struct TMSConstrainedSpreader {
    // Base spreading parameters
    pub decay_rate: f32,
    pub activation_threshold: f32,
    pub max_iterations: usize,
    
    // TMS integration
    pub tms: Arc<RwLock<TruthMaintenanceSystem>>,
    
    // Belief-specific parameters
    pub confidence_threshold: f32,
    pub justification_threshold: f32,
    
    // Context management
    pub current_context: Option<BeliefContext>,
    pub context_switching_enabled: bool,
}

impl TMSConstrainedSpreader {
    pub fn new(tms: Arc<RwLock<TruthMaintenanceSystem>>) -> Self {
        Self {
            decay_rate: 0.9,
            activation_threshold: 0.001,
            max_iterations: 100,
            tms,
            confidence_threshold: 0.5,
            justification_threshold: 0.3,
            current_context: None,
            context_switching_enabled: true,
        }
    }
    
    pub async fn spread_with_beliefs(
        &mut self,
        initial_activations: HashMap<NodeId, f32>,
        graph: &Graph,
        belief_context: BeliefContext,
    ) -> Result<BeliefAwareActivationState, ActivationError> {
        // Set belief context
        self.current_context = Some(belief_context.clone());
        
        // Get TMS constraints for this context
        let tms_guard = self.tms.read().await;
        let tms_constraints = tms_guard.get_constraints_for_context(&belief_context).await?;
        drop(tms_guard);
        
        // Initialize belief-aware state
        let mut state = BeliefAwareActivationState::new(
            belief_context.context_id, 
            tms_constraints
        );
        
        // Set initial activations with belief context
        for (node, activation) in initial_activations {
            let belief_activation_context = self.create_belief_activation_context(
                node, 
                activation, 
                &belief_context
            ).await?;
            
            state.set_belief_activation(node, activation, belief_activation_context)?;
        }
        
        // Spread activation with TMS constraints
        for iteration in 0..self.max_iterations {
            let changes = self.spread_one_iteration(&mut state, graph).await?;
            
            // Check for convergence
            if changes < 0.001 {
                break;
            }
            
            // Handle belief revisions that occurred during spreading
            self.handle_belief_revisions_during_spreading(&mut state).await?;
        }
        
        Ok(state)
    }
    
    async fn spread_one_iteration(
        &mut self,
        state: &mut BeliefAwareActivationState,
        graph: &Graph,
    ) -> Result<f32, ActivationError> {
        let mut new_activations = HashMap::new();
        let mut total_change = 0.0;
        
        // Get currently activated nodes
        let activated_nodes = state.base_state.activated_nodes();
        
        for &source_node in &activated_nodes {
            let source_activation = state.base_state.get_activation(source_node);
            let source_belief = state.get_belief_activation(source_node);
            
            // Skip if belief is not currently supported
            if let Some(belief) = source_belief {
                if belief.belief_status != BeliefStatus::In {
                    continue;
                }
            }
            
            // Get neighbors
            let neighbors = graph.get_neighbors(source_node)?;
            
            for neighbor in neighbors {
                // Check TMS constraints for this propagation
                if !self.can_propagate_between_nodes(source_node, neighbor, state).await? {
                    continue;
                }
                
                // Calculate belief-aware propagation
                let edge_weight = graph.get_edge_weight(source_node, neighbor)?;
                let belief_multiplier = self.calculate_belief_propagation_multiplier(
                    source_node, 
                    neighbor, 
                    state
                ).await?;
                
                let propagated_activation = source_activation * edge_weight * 
                    belief_multiplier * self.decay_rate;
                
                if propagated_activation > self.activation_threshold {
                    *new_activations.entry(neighbor).or_insert(0.0) += propagated_activation;
                }
            }
        }
        
        // Apply new activations
        for (node, activation) in new_activations {
            let old_activation = state.base_state.get_activation(node);
            let change = (activation - old_activation).abs();
            total_change += change;
            
            // Create belief context for new activation
            let belief_context = self.create_propagated_belief_context(
                node, 
                activation, 
                state
            ).await?;
            
            state.set_belief_activation(node, activation, belief_context)?;
        }
        
        Ok(total_change)
    }
    
    async fn can_propagate_between_nodes(
        &self,
        source: NodeId,
        target: NodeId,
        state: &BeliefAwareActivationState,
    ) -> Result<bool, ActivationError> {
        let tms_guard = self.tms.read().await;
        
        // Check if there's a valid justification path
        let has_valid_justification = tms_guard
            .has_valid_justification_between(source, target, &state.belief_context)
            .await?;
        
        // Check if target belief is retracted
        let target_belief_status = tms_guard
            .get_belief_status(target, &state.belief_context)
            .await?;
        
        Ok(has_valid_justification && target_belief_status != BeliefStatus::Out)
    }
    
    async fn calculate_belief_propagation_multiplier(
        &self,
        source: NodeId,
        target: NodeId,
        state: &BeliefAwareActivationState,
    ) -> Result<f32, ActivationError> {
        let tms_guard = self.tms.read().await;
        
        // Get justification strength between nodes
        let justification_strength = tms_guard
            .get_justification_strength(source, target, &state.belief_context)
            .await?
            .unwrap_or(0.5);
        
        // Get source belief confidence
        let source_confidence = if let Some(belief) = state.get_belief_activation(source) {
            match belief.confidence_level {
                ConfidenceLevel::High => 1.0,
                ConfidenceLevel::Medium => 0.8,
                ConfidenceLevel::Low => 0.6,
                ConfidenceLevel::Uncertain => 0.4,
            }
        } else {
            0.5
        };
        
        // Combine justification strength and confidence
        Ok(justification_strength * source_confidence)
    }
    
    async fn handle_belief_revisions_during_spreading(
        &mut self,
        state: &mut BeliefAwareActivationState,
    ) -> Result<(), ActivationError> {
        let tms_guard = self.tms.read().await;
        
        // Check for any belief revisions that occurred
        let recent_revisions = tms_guard
            .get_recent_revisions(&state.belief_context)
            .await?;
        
        for revision in recent_revisions {
            // Handle retracted beliefs
            if revision.revision_type == RevisionType::Retraction {
                // Remove activation from retracted beliefs
                if state.belief_activations.contains_key(&revision.belief_node) {
                    state.base_state.set_activation(revision.belief_node, 0.0);
                    state.belief_activations.remove(&revision.belief_node);
                }
                
                // Mark dependent activations for recalculation
                self.mark_dependent_activations_for_recalc(
                    revision.belief_node, 
                    state
                ).await?;
            }
            
            // Handle new beliefs
            if revision.revision_type == RevisionType::Addition {
                // The new belief might enable new activation paths
                self.recalculate_activation_paths_through(
                    revision.belief_node, 
                    state
                ).await?;
            }
        }
        
        Ok(())
    }
}
```

### Step 3: Justification Path Tracer

```rust
// File: src/core/activation/justification_tracer.rs

use std::collections::{HashMap, HashSet, VecDeque};
use crate::core::types::NodeId;
use crate::tms::{BeliefId, JustificationId, JustificationChain};

#[derive(Debug, Clone)]
pub struct JustificationPathTracer {
    // Trace configuration
    pub max_depth: usize,
    pub min_confidence: f32,
    
    // Traced paths
    pub traced_paths: HashMap<NodeId, Vec<JustificationPath>>,
    
    // Path statistics
    pub path_statistics: PathStatistics,
}

#[derive(Debug, Clone)]
pub struct JustificationPath {
    pub path_id: JustificationId,
    pub nodes: Vec<NodeId>,
    pub beliefs: Vec<BeliefId>,
    pub confidences: Vec<f32>,
    pub total_strength: f32,
    pub depth: usize,
    pub is_circular: bool,
    pub weak_links: Vec<WeakLink>,
}

#[derive(Debug, Clone)]
pub struct WeakLink {
    pub from_node: NodeId,
    pub to_node: NodeId,
    pub confidence: f32,
    pub justification_strength: f32,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone)]
pub enum RiskLevel {
    Low,      // Strong justification
    Medium,   // Moderate justification
    High,     // Weak justification
    Critical, // Very weak or circular
}

impl JustificationPathTracer {
    pub fn new() -> Self {
        Self {
            max_depth: 10,
            min_confidence: 0.1,
            traced_paths: HashMap::new(),
            path_statistics: PathStatistics::new(),
        }
    }
    
    pub async fn trace_justification_paths(
        &mut self,
        activation_state: &BeliefAwareActivationState,
        tms: &TruthMaintenanceSystem,
    ) -> Result<JustificationTraceResult, TracingError> {
        let mut trace_result = JustificationTraceResult::new();
        
        // Trace paths for all activated nodes
        for (&node_id, belief_activation) in &activation_state.belief_activations {
            let paths = self.trace_paths_for_node(
                node_id, 
                belief_activation, 
                activation_state, 
                tms
            ).await?;
            
            self.traced_paths.insert(node_id, paths.clone());
            trace_result.add_node_paths(node_id, paths);
        }
        
        // Analyze path quality
        self.analyze_path_quality(&mut trace_result).await?;
        
        // Identify weak links
        self.identify_weak_links(&mut trace_result).await?;
        
        Ok(trace_result)
    }
    
    async fn trace_paths_for_node(
        &self,
        target_node: NodeId,
        belief_activation: &BeliefActivation,
        state: &BeliefAwareActivationState,
        tms: &TruthMaintenanceSystem,
    ) -> Result<Vec<JustificationPath>, TracingError> {
        let mut paths = Vec::new();
        let mut visited = HashSet::new();
        let mut path_queue = VecDeque::new();
        
        // Initialize with supporting beliefs
        for &belief_id in &belief_activation.supporting_beliefs {
            if let Some(belief_node) = tms.get_belief_node(belief_id).await? {
                path_queue.push_back(JustificationPath {
                    path_id: JustificationId::new(),
                    nodes: vec![belief_node, target_node],
                    beliefs: vec![belief_id],
                    confidences: vec![tms.get_belief_confidence(belief_id).await?],
                    total_strength: belief_activation.justification_strength,
                    depth: 1,
                    is_circular: false,
                    weak_links: Vec::new(),
                });
            }
        }
        
        // Breadth-first traversal of justification chains
        while let Some(mut current_path) = path_queue.pop_front() {
            if current_path.depth >= self.max_depth {
                paths.push(current_path);
                continue;
            }
            
            let current_node = current_path.nodes[0];
            
            // Check for cycles
            if visited.contains(&current_node) {
                current_path.is_circular = true;
                paths.push(current_path);
                continue;
            }
            visited.insert(current_node);
            
            // Get justifications for current node
            let justifications = tms.get_justifications_for_node(
                current_node, 
                &state.belief_context
            ).await?;
            
            if justifications.is_empty() {
                // Reached a base belief
                paths.push(current_path);
            } else {
                // Extend path with each justification
                for justification in justifications {
                    let mut extended_path = current_path.clone();
                    
                    // Add justification to path
                    extended_path.nodes.insert(0, justification.antecedent_node);
                    extended_path.beliefs.insert(0, justification.belief_id);
                    extended_path.confidences.insert(0, justification.confidence);
                    extended_path.depth += 1;
                    
                    // Update total strength
                    extended_path.total_strength *= justification.strength;
                    
                    // Check if this creates a weak link
                    if justification.confidence < 0.5 || justification.strength < 0.3 {
                        extended_path.weak_links.push(WeakLink {
                            from_node: justification.antecedent_node,
                            to_node: current_node,
                            confidence: justification.confidence,
                            justification_strength: justification.strength,
                            risk_level: self.calculate_risk_level(
                                justification.confidence, 
                                justification.strength
                            ),
                        });
                    }
                    
                    path_queue.push_back(extended_path);
                }
            }
        }
        
        // Filter paths by minimum confidence
        paths.retain(|path| {
            path.confidences.iter().all(|&c| c >= self.min_confidence)
        });
        
        Ok(paths)
    }
    
    async fn analyze_path_quality(
        &mut self,
        trace_result: &mut JustificationTraceResult,
    ) -> Result<(), TracingError> {
        for (node_id, paths) in &trace_result.node_paths {
            let mut node_quality = NodePathQuality::new(*node_id);
            
            for path in paths {
                // Calculate path strength
                let path_strength = self.calculate_path_strength(path);
                
                // Calculate path reliability
                let path_reliability = self.calculate_path_reliability(path);
                
                // Identify critical dependencies
                let critical_deps = self.identify_critical_dependencies(path);
                
                node_quality.add_path_analysis(PathAnalysis {
                    path_id: path.path_id,
                    strength: path_strength,
                    reliability: path_reliability,
                    critical_dependencies: critical_deps,
                    weakness_score: self.calculate_weakness_score(path),
                });
            }
            
            trace_result.quality_analysis.insert(*node_id, node_quality);
        }
        
        Ok(())
    }
    
    fn calculate_path_strength(&self, path: &JustificationPath) -> f32 {
        // Strength is the product of all confidences in the path
        path.confidences.iter().product::<f32>() * path.total_strength
    }
    
    fn calculate_path_reliability(&self, path: &JustificationPath) -> f32 {
        if path.is_circular {
            return 0.0; // Circular paths are unreliable
        }
        
        // Reliability decreases with path length and weak links
        let length_penalty = 0.9_f32.powi(path.depth as i32);
        let weak_link_penalty = 0.8_f32.powi(path.weak_links.len() as i32);
        
        length_penalty * weak_link_penalty
    }
    
    fn calculate_risk_level(&self, confidence: f32, strength: f32) -> RiskLevel {
        let combined_score = confidence * strength;
        
        match combined_score {
            s if s >= 0.8 => RiskLevel::Low,
            s if s >= 0.6 => RiskLevel::Medium,
            s if s >= 0.3 => RiskLevel::High,
            _ => RiskLevel::Critical,
        }
    }
    
    async fn identify_weak_links(
        &mut self,
        trace_result: &mut JustificationTraceResult,
    ) -> Result<(), TracingError> {
        let mut all_weak_links = Vec::new();
        
        // Collect all weak links across all paths
        for paths in trace_result.node_paths.values() {
            for path in paths {
                all_weak_links.extend(path.weak_links.clone());
            }
        }
        
        // Sort by risk level and frequency
        let mut link_frequency = HashMap::new();
        for link in &all_weak_links {
            let key = (link.from_node, link.to_node);
            *link_frequency.entry(key).or_insert(0) += 1;
        }
        
        // Identify most problematic links
        let mut critical_links = Vec::new();
        for link in all_weak_links {
            let frequency = link_frequency[&(link.from_node, link.to_node)];
            
            if matches!(link.risk_level, RiskLevel::Critical | RiskLevel::High) 
                && frequency > 1 {
                critical_links.push(link);
            }
        }
        
        trace_result.critical_weak_links = critical_links;
        
        Ok(())
    }
}

#[derive(Debug)]
pub struct JustificationTraceResult {
    pub node_paths: HashMap<NodeId, Vec<JustificationPath>>,
    pub quality_analysis: HashMap<NodeId, NodePathQuality>,
    pub critical_weak_links: Vec<WeakLink>,
    pub overall_reliability: f32,
    pub recommendation: PathRecommendation,
}

#[derive(Debug)]
pub enum PathRecommendation {
    Reliable,                           // All paths are strong
    CautionWeakLinks(Vec<WeakLink>),   // Some weak links identified
    Unreliable(String),                // Significant reliability issues
    RequiresRevision(Vec<NodeId>),     // Beliefs need revision
}
```

### Step 4: Belief Query Processor

```rust
// File: src/core/query/belief_query_processor.rs

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::core::activation::belief_aware_state::BeliefAwareActivationState;
use crate::core::activation::tms_constrained_spreader::TMSConstrainedSpreader;
use crate::core::activation::justification_tracer::{JustificationPathTracer, JustificationTraceResult};
use crate::core::query::intent::{QueryIntent, QueryContext};
use crate::tms::{TruthMaintenanceSystem, BeliefContext};

pub struct BeliefQueryProcessor {
    // Core components
    pub spreader: TMSConstrainedSpreader,
    pub justification_tracer: JustificationPathTracer,
    pub tms: Arc<RwLock<TruthMaintenanceSystem>>,
    
    // Query configuration
    pub default_confidence_threshold: f32,
    pub max_results: usize,
    pub include_justifications: bool,
    
    // Active belief contexts
    pub active_contexts: HashMap<String, BeliefContext>,
}

#[derive(Debug, Clone)]
pub struct BeliefQueryResult {
    // Basic query results
    pub activated_nodes: Vec<ActivatedNode>,
    pub activation_summary: ActivationSummary,
    
    // Belief-specific results
    pub belief_analysis: BeliefAnalysis,
    pub justification_trace: Option<JustificationTraceResult>,
    
    // Context information
    pub belief_context: BeliefContext,
    pub context_switches: Vec<ContextSwitch>,
    
    // Quality metrics
    pub reliability_score: f32,
    pub confidence_score: f32,
    pub justification_strength: f32,
}

#[derive(Debug, Clone)]
pub struct BeliefAnalysis {
    pub total_beliefs_activated: usize,
    pub high_confidence_beliefs: usize,
    pub questionable_beliefs: usize,
    pub belief_consistency: f32,
    pub revision_suggestions: Vec<RevisionSuggestion>,
}

impl BeliefQueryProcessor {
    pub fn new(tms: Arc<RwLock<TruthMaintenanceSystem>>) -> Self {
        Self {
            spreader: TMSConstrainedSpreader::new(tms.clone()),
            justification_tracer: JustificationPathTracer::new(),
            tms,
            default_confidence_threshold: 0.5,
            max_results: 100,
            include_justifications: true,
            active_contexts: HashMap::new(),
        }
    }
    
    pub async fn process_belief_query(
        &mut self,
        query_intent: &QueryIntent,
        belief_context: Option<BeliefContext>,
        graph: &Graph,
    ) -> Result<BeliefQueryResult, BeliefQueryError> {
        // Determine belief context
        let context = if let Some(ctx) = belief_context {
            ctx
        } else {
            self.infer_belief_context_from_query(query_intent).await?
        };
        
        // Extract initial activations from query
        let initial_activations = self.extract_query_activations(query_intent).await?;
        
        // Validate query against TMS constraints
        self.validate_query_against_tms(&initial_activations, &context).await?;
        
        // Perform belief-aware spreading activation
        let activation_state = self.spreader.spread_with_beliefs(
            initial_activations,
            graph,
            context.clone(),
        ).await?;
        
        // Trace justification paths if requested
        let justification_trace = if self.include_justifications {
            let tms_guard = self.tms.read().await;
            Some(self.justification_tracer.trace_justification_paths(
                &activation_state,
                &*tms_guard,
            ).await?)
        } else {
            None
        };
        
        // Analyze belief consistency
        let belief_analysis = self.analyze_belief_consistency(&activation_state).await?;
        
        // Calculate quality metrics
        let reliability_score = self.calculate_reliability_score(
            &activation_state,
            &justification_trace,
        ).await?;
        
        let confidence_score = self.calculate_confidence_score(&activation_state).await?;
        
        let justification_strength = self.calculate_justification_strength(
            &activation_state
        ).await?;
        
        // Build result
        Ok(BeliefQueryResult {
            activated_nodes: self.extract_activated_nodes(&activation_state),
            activation_summary: self.create_activation_summary(&activation_state),
            belief_analysis,
            justification_trace,
            belief_context: context,
            context_switches: Vec::new(), // TODO: Track context switches
            reliability_score,
            confidence_score,
            justification_strength,
        })
    }
    
    pub async fn process_multi_context_query(
        &mut self,
        query_intent: &QueryIntent,
        contexts: Vec<BeliefContext>,
        graph: &Graph,
    ) -> Result<MultiContextBeliefResult, BeliefQueryError> {
        let mut context_results = HashMap::new();
        
        // Process query in each context
        for context in contexts {
            let result = self.process_belief_query(
                query_intent,
                Some(context.clone()),
                graph,
            ).await?;
            
            context_results.insert(context.context_id, result);
        }
        
        // Analyze context consistency
        let consistency_analysis = self.analyze_context_consistency(
            &context_results
        ).await?;
        
        // Synthesize final recommendation
        let synthesis = self.synthesize_multi_context_results(
            &context_results,
            &consistency_analysis,
        ).await?;
        
        Ok(MultiContextBeliefResult {
            context_results,
            consistency_analysis,
            synthesis,
        })
    }
    
    async fn validate_query_against_tms(
        &self,
        activations: &HashMap<NodeId, f32>,
        context: &BeliefContext,
    ) -> Result<(), BeliefQueryError> {
        let tms_guard = self.tms.read().await;
        
        // Check if query concepts are consistent with current beliefs
        for &node_id in activations.keys() {
            let belief_status = tms_guard.get_belief_status(node_id, context).await?;
            
            if belief_status == BeliefStatus::Out {
                return Err(BeliefQueryError::QueryConceptRetracted(node_id));
            }
        }
        
        // Check for potential conflicts the query might create
        let potential_conflicts = tms_guard.check_potential_conflicts(
            activations.keys().cloned().collect(),
            context,
        ).await?;
        
        if !potential_conflicts.is_empty() {
            return Err(BeliefQueryError::PotentialConflicts(potential_conflicts));
        }
        
        Ok(())
    }
    
    async fn analyze_belief_consistency(
        &self,
        state: &BeliefAwareActivationState,
    ) -> Result<BeliefAnalysis, BeliefQueryError> {
        let mut high_confidence = 0;
        let mut questionable = 0;
        let mut revision_suggestions = Vec::new();
        
        for (node_id, belief_activation) in &state.belief_activations {
            match belief_activation.confidence_level {
                ConfidenceLevel::High => high_confidence += 1,
                ConfidenceLevel::Low | ConfidenceLevel::Uncertain => {
                    questionable += 1;
                    
                    if belief_activation.justification_strength < 0.3 {
                        revision_suggestions.push(RevisionSuggestion {
                            node_id: *node_id,
                            suggestion_type: SuggestionType::WeakJustification,
                            description: format!(
                                "Node {} has weak justification ({})",
                                node_id.0, belief_activation.justification_strength
                            ),
                            recommended_action: RecommendedAction::StrengthenJustification,
                        });
                    }
                }
                _ => {}
            }
        }
        
        // Calculate overall consistency
        let total_beliefs = state.belief_activations.len();
        let consistency = if total_beliefs > 0 {
            high_confidence as f32 / total_beliefs as f32
        } else {
            1.0
        };
        
        Ok(BeliefAnalysis {
            total_beliefs_activated: total_beliefs,
            high_confidence_beliefs: high_confidence,
            questionable_beliefs: questionable,
            belief_consistency: consistency,
            revision_suggestions,
        })
    }
    
    async fn calculate_reliability_score(
        &self,
        state: &BeliefAwareActivationState,
        justification_trace: &Option<JustificationTraceResult>,
    ) -> Result<f32, BeliefQueryError> {
        if let Some(trace) = justification_trace {
            Ok(trace.overall_reliability)
        } else {
            // Fallback calculation based on activation state
            let total_activations = state.belief_activations.len() as f32;
            let high_confidence_count = state.belief_activations.values()
                .filter(|b| matches!(b.confidence_level, ConfidenceLevel::High))
                .count() as f32;
            
            Ok(if total_activations > 0.0 {
                high_confidence_count / total_activations
            } else {
                1.0
            })
        }
    }
}
```

## File Locations

- `src/core/activation/belief_aware_state.rs` - Belief-aware activation state
- `src/core/activation/tms_constrained_spreader.rs` - TMS-constrained spreader
- `src/core/activation/justification_tracer.rs` - Justification path tracer
- `src/core/query/belief_query_processor.rs` - High-level belief query processor
- `tests/activation/belief_aware_tests.rs` - Comprehensive test suite

## Success Criteria

- [ ] BeliefAwareActivationState compiles and handles TMS constraints
- [ ] TMSConstrainedSpreader respects belief retraction and justification
- [ ] JustificationPathTracer identifies weak links and circular dependencies
- [ ] BeliefQueryProcessor provides comprehensive belief analysis
- [ ] Multi-context queries work correctly
- [ ] Performance targets met (belief constraint checking <0.5ms)
- [ ] All tests pass:
  - Basic belief-aware activation
  - TMS constraint enforcement
  - Justification path tracing
  - Multi-context consistency
  - Circular dependency detection
  - Weak link identification

## Test Requirements

```rust
#[test]
async fn test_belief_aware_activation() {
    let tms = setup_test_tms().await;
    let mut state = BeliefAwareActivationState::new(
        ContextId(1), 
        TMSConstraints::default()
    );
    
    let belief_context = BeliefActivationContext {
        confidence: ConfidenceLevel::High,
        justification_strength: 0.8,
        status: BeliefStatus::In,
        supporting_beliefs: vec![BeliefId(1), BeliefId(2)],
    };
    
    state.set_belief_activation(NodeId(1), 0.7, belief_context).unwrap();
    
    let activation = state.get_belief_activation(NodeId(1)).unwrap();
    assert_eq!(activation.confidence_level, ConfidenceLevel::High);
    assert!(activation.justification_strength > 0.5);
}

#[test]
async fn test_tms_constraint_enforcement() {
    let tms = setup_test_tms().await;
    let mut spreader = TMSConstrainedSpreader::new(tms.clone());
    
    // Add a retracted belief to TMS
    {
        let mut tms_guard = tms.write().await;
        tms_guard.retract_belief(BeliefId(5)).await.unwrap();
    }
    
    let initial_activations = HashMap::from([
        (NodeId(1), 0.8),
        (NodeId(5), 0.6), // This should be blocked
    ]);
    
    let result = spreader.spread_with_beliefs(
        initial_activations,
        &test_graph(),
        BeliefContext::default(),
    ).await.unwrap();
    
    // Node 5 should not be activated due to retraction
    assert_eq!(result.base_state.get_activation(NodeId(5)), 0.0);
}

#[test]
async fn test_justification_path_tracing() {
    let tms = setup_test_tms().await;
    let mut tracer = JustificationPathTracer::new();
    
    let state = setup_test_belief_state().await;
    
    let trace_result = tracer.trace_justification_paths(&state, &*tms.read().await)
        .await.unwrap();
    
    assert!(!trace_result.node_paths.is_empty());
    
    // Check for weak link identification
    if !trace_result.critical_weak_links.is_empty() {
        assert!(trace_result.critical_weak_links[0].confidence < 0.5);
    }
}
```

## Quality Gates

- [ ] TMS constraint checking overhead <20%
- [ ] Justification tracing scales to 1000+ node paths
- [ ] Belief consistency validation completes in <10ms
- [ ] Multi-context queries handle 10+ contexts efficiently
- [ ] Memory usage scales linearly with activated beliefs
- [ ] Circular dependency detection is reliable

## Next Task

Upon completion, proceed to **32_temporal_activation.md**