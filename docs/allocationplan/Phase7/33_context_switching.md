# Micro Task 33: Context Switching for Multi-Context Processing

**Priority**: CRITICAL  
**Estimated Time**: 40 minutes  
**Dependencies**: Phase 6 TMS, Tasks 31-32 (belief-aware and temporal queries)  
**Skills Required**: Rust async programming, context management, concurrent processing

## Objective

Implement dynamic context switching capabilities that allow the spreading activation engine to seamlessly transition between different belief contexts, enabling multi-perspective reasoning and handling of conflicting or incompatible belief sets.

## Context

Building on the temporal and belief-aware query capabilities, this component enables the system to maintain and switch between multiple belief contexts during activation spreading. This is crucial for handling scenarios where different assumptions lead to different conclusions, supporting hypothetical reasoning and multi-agent perspectives.

## Specifications

### Required Components

1. **ContextManager**
   - Manages multiple active belief contexts
   - Handles context creation, switching, and cleanup
   - Maintains context isolation and independence
   - Supports context inheritance and branching

2. **ContextAwareSpreader**
   - Spreading algorithm that respects context boundaries
   - Dynamic context switching during propagation
   - Context-specific activation tracking
   - Cross-context comparison and synthesis

3. **ContextSwitchOrchestrator**
   - Orchestrates context switches based on triggers
   - Manages context switch timing and coordination
   - Handles context merge and split operations
   - Optimizes context resource allocation

4. **MultiContextQueryProcessor**
   - High-level interface for multi-context queries
   - Aggregates results across contexts
   - Identifies context-dependent differences
   - Provides context synthesis and recommendations

### Performance Requirements

- Context switch latency: <1ms between contexts
- Context isolation overhead: <15% per additional context
- Multi-context query processing: <10ms for 5 contexts
- Memory usage: Linear scaling with number of active contexts

## Implementation Guide

### Step 1: Context Manager

```rust
// File: src/core/context/context_manager.rs

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use std::time::{Instant, Duration};
use crate::core::types::{NodeId, ContextId};
use crate::tms::{BeliefContext, AssumptionSet, BeliefId};
use crate::core::activation::belief_aware_state::BeliefAwareActivationState;

#[derive(Debug, Clone)]
pub struct ContextManager {
    // Active contexts
    pub active_contexts: Arc<RwLock<HashMap<ContextId, ManagedContext>>>,
    
    // Context relationships
    pub context_hierarchy: Arc<RwLock<ContextHierarchy>>,
    
    // Context lifecycle management
    pub context_lifecycle: Arc<Mutex<ContextLifecycle>>,
    
    // Configuration
    pub max_active_contexts: usize,
    pub context_timeout: Duration,
    pub enable_context_gc: bool,
}

#[derive(Debug, Clone)]
pub struct ManagedContext {
    // Context identity
    pub context_id: ContextId,
    pub parent_context: Option<ContextId>,
    pub child_contexts: HashSet<ContextId>,
    
    // Belief context
    pub belief_context: BeliefContext,
    pub assumption_set: AssumptionSet,
    
    // State management
    pub activation_state: Option<BeliefAwareActivationState>,
    pub is_active: bool,
    pub last_used: Instant,
    
    // Context metadata
    pub creation_time: Instant,
    pub usage_count: usize,
    pub priority: ContextPriority,
    
    // Resource tracking
    pub memory_usage: usize,
    pub cpu_time: Duration,
}

#[derive(Debug, Clone)]
pub struct ContextHierarchy {
    // Tree structure for context relationships
    pub root_contexts: HashSet<ContextId>,
    pub parent_map: HashMap<ContextId, ContextId>,
    pub children_map: HashMap<ContextId, HashSet<ContextId>>,
    
    // Context groups
    pub context_groups: HashMap<String, HashSet<ContextId>>,
    pub incompatible_contexts: HashMap<ContextId, HashSet<ContextId>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ContextPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl ContextManager {
    pub fn new() -> Self {
        Self {
            active_contexts: Arc::new(RwLock::new(HashMap::new())),
            context_hierarchy: Arc::new(RwLock::new(ContextHierarchy::new())),
            context_lifecycle: Arc::new(Mutex::new(ContextLifecycle::new())),
            max_active_contexts: 10,
            context_timeout: Duration::from_secs(300), // 5 minutes
            enable_context_gc: true,
        }
    }
    
    pub async fn create_context(
        &mut self,
        belief_context: BeliefContext,
        parent_context: Option<ContextId>,
        priority: ContextPriority,
    ) -> Result<ContextId, ContextError> {
        let context_id = ContextId::new();
        
        // Create managed context
        let mut managed_context = ManagedContext {
            context_id,
            parent_context,
            child_contexts: HashSet::new(),
            belief_context,
            assumption_set: AssumptionSet::new(),
            activation_state: None,
            is_active: false,
            last_used: Instant::now(),
            creation_time: Instant::now(),
            usage_count: 0,
            priority,
            memory_usage: 0,
            cpu_time: Duration::ZERO,
        };
        
        // Inherit assumptions from parent if specified
        if let Some(parent_id) = parent_context {
            self.inherit_from_parent(&mut managed_context, parent_id).await?;
        }
        
        // Check context limits
        self.enforce_context_limits().await?;
        
        // Add to active contexts
        {
            let mut contexts = self.active_contexts.write().await;
            contexts.insert(context_id, managed_context);
        }
        
        // Update hierarchy
        self.update_context_hierarchy(context_id, parent_context).await?;
        
        // Track lifecycle
        {
            let mut lifecycle = self.context_lifecycle.lock().await;
            lifecycle.track_context_creation(context_id);
        }
        
        Ok(context_id)
    }
    
    pub async fn activate_context(
        &mut self,
        context_id: ContextId,
    ) -> Result<ContextActivationResult, ContextError> {
        let mut contexts = self.active_contexts.write().await;
        
        let context = contexts.get_mut(&context_id)
            .ok_or(ContextError::ContextNotFound(context_id))?;
        
        // Check if context is compatible with currently active contexts
        let active_context_ids: Vec<_> = contexts.values()
            .filter(|c| c.is_active)
            .map(|c| c.context_id)
            .collect();
        
        self.validate_context_compatibility(context_id, &active_context_ids).await?;
        
        // Deactivate incompatible contexts if needed
        let deactivated = self.deactivate_incompatible_contexts(
            context_id, 
            &mut contexts
        ).await?;
        
        // Activate the context
        context.is_active = true;
        context.last_used = Instant::now();
        context.usage_count += 1;
        
        Ok(ContextActivationResult {
            activated_context: context_id,
            deactivated_contexts: deactivated,
            activation_time: Instant::now(),
        })
    }
    
    pub async fn switch_context(
        &mut self,
        from_context: ContextId,
        to_context: ContextId,
        transfer_state: bool,
    ) -> Result<ContextSwitchResult, ContextError> {
        let switch_start = Instant::now();
        
        // Validate contexts exist
        {
            let contexts = self.active_contexts.read().await;
            if !contexts.contains_key(&from_context) {
                return Err(ContextError::ContextNotFound(from_context));
            }
            if !contexts.contains_key(&to_context) {
                return Err(ContextError::ContextNotFound(to_context));
            }
        }
        
        // Prepare for switch
        let switch_state = if transfer_state {
            Some(self.extract_transferable_state(from_context).await?)
        } else {
            None
        };
        
        // Perform atomic switch
        let (deactivation_result, activation_result) = {
            let mut contexts = self.active_contexts.write().await;
            
            // Deactivate source context
            let from_ctx = contexts.get_mut(&from_context)
                .ok_or(ContextError::ContextNotFound(from_context))?;
            from_ctx.is_active = false;
            
            // Activate target context
            let to_ctx = contexts.get_mut(&to_context)
                .ok_or(ContextError::ContextNotFound(to_context))?;
            to_ctx.is_active = true;
            to_ctx.last_used = Instant::now();
            to_ctx.usage_count += 1;
            
            (from_context, to_context)
        };
        
        // Transfer state if requested
        if let Some(state) = switch_state {
            self.apply_transferred_state(to_context, state).await?;
        }
        
        let switch_duration = switch_start.elapsed();
        
        // Track switch metrics
        {
            let mut lifecycle = self.context_lifecycle.lock().await;
            lifecycle.track_context_switch(from_context, to_context, switch_duration);
        }
        
        Ok(ContextSwitchResult {
            from_context,
            to_context,
            switch_duration,
            state_transferred: transfer_state,
        })
    }
    
    pub async fn branch_context(
        &mut self,
        parent_context: ContextId,
        branch_assumptions: AssumptionSet,
        branch_name: Option<String>,
    ) -> Result<ContextId, ContextError> {
        // Get parent context
        let parent = {
            let contexts = self.active_contexts.read().await;
            contexts.get(&parent_context)
                .ok_or(ContextError::ContextNotFound(parent_context))?
                .clone()
        };
        
        // Create branched belief context
        let mut branched_belief_context = parent.belief_context.clone();
        branched_belief_context.add_assumptions(branch_assumptions.clone())?;
        
        // Create new context as child
        let branch_id = self.create_context(
            branched_belief_context,
            Some(parent_context),
            parent.priority,
        ).await?;
        
        // Update branch with additional assumptions
        {
            let mut contexts = self.active_contexts.write().await;
            let branch_context = contexts.get_mut(&branch_id)
                .ok_or(ContextError::ContextNotFound(branch_id))?;
            branch_context.assumption_set = branch_assumptions;
        }
        
        // Update hierarchy to mark as branch
        {
            let mut hierarchy = self.context_hierarchy.write().await;
            hierarchy.children_map
                .entry(parent_context)
                .or_insert_with(HashSet::new)
                .insert(branch_id);
            
            // Add to named group if provided
            if let Some(name) = branch_name {
                hierarchy.context_groups
                    .entry(name)
                    .or_insert_with(HashSet::new)
                    .insert(branch_id);
            }
        }
        
        Ok(branch_id)
    }
    
    pub async fn merge_contexts(
        &mut self,
        context_ids: Vec<ContextId>,
        merge_strategy: MergeStrategy,
    ) -> Result<ContextId, ContextError> {
        if context_ids.len() < 2 {
            return Err(ContextError::InsufficientContextsForMerge);
        }
        
        // Validate all contexts exist and are compatible for merging
        let contexts_to_merge = {
            let contexts = self.active_contexts.read().await;
            let mut result = Vec::new();
            
            for &context_id in &context_ids {
                let context = contexts.get(&context_id)
                    .ok_or(ContextError::ContextNotFound(context_id))?;
                result.push(context.clone());
            }
            
            result
        };
        
        // Check merge compatibility
        self.validate_merge_compatibility(&contexts_to_merge, &merge_strategy).await?;
        
        // Create merged belief context
        let merged_belief_context = self.create_merged_belief_context(
            &contexts_to_merge,
            &merge_strategy,
        ).await?;
        
        // Determine merge priority
        let merge_priority = contexts_to_merge.iter()
            .map(|c| c.priority.clone())
            .max()
            .unwrap_or(ContextPriority::Medium);
        
        // Create merged context
        let merged_id = self.create_context(
            merged_belief_context,
            None, // Merged contexts are typically root contexts
            merge_priority,
        ).await?;
        
        // Transfer aggregated state
        self.transfer_merged_state(&context_ids, merged_id, &merge_strategy).await?;
        
        // Deactivate source contexts
        for context_id in &context_ids {
            self.deactivate_context(*context_id).await?;
        }
        
        Ok(merged_id)
    }
    
    async fn inherit_from_parent(
        &self,
        child_context: &mut ManagedContext,
        parent_id: ContextId,
    ) -> Result<(), ContextError> {
        let contexts = self.active_contexts.read().await;
        let parent = contexts.get(&parent_id)
            .ok_or(ContextError::ContextNotFound(parent_id))?;
        
        // Inherit assumptions
        child_context.assumption_set = parent.assumption_set.clone();
        
        // Inherit compatible beliefs
        child_context.belief_context = parent.belief_context.create_child_context()?;
        
        Ok(())
    }
    
    async fn enforce_context_limits(&mut self) -> Result<(), ContextError> {
        let mut contexts = self.active_contexts.write().await;
        
        // Check if we're at the limit
        if contexts.len() >= self.max_active_contexts {
            // Find least recently used, lowest priority context to evict
            let mut candidates: Vec<_> = contexts.values().collect();
            candidates.sort_by_key(|c| (c.priority.clone(), c.last_used));
            
            if let Some(evict_candidate) = candidates.first() {
                let evict_id = evict_candidate.context_id;
                self.cleanup_context(evict_id, &mut contexts).await?;
            }
        }
        
        Ok(())
    }
    
    async fn cleanup_context(
        &self,
        context_id: ContextId,
        contexts: &mut HashMap<ContextId, ManagedContext>,
    ) -> Result<(), ContextError> {
        // Remove from active contexts
        if let Some(context) = contexts.remove(&context_id) {
            // Clean up child contexts first
            for child_id in &context.child_contexts {
                self.cleanup_context(*child_id, contexts).await?;
            }
            
            // Update parent's child list
            if let Some(parent_id) = context.parent_context {
                if let Some(parent) = contexts.get_mut(&parent_id) {
                    parent.child_contexts.remove(&context_id);
                }
            }
        }
        
        // Clean up hierarchy
        {
            let mut hierarchy = self.context_hierarchy.write().await;
            hierarchy.cleanup_context(context_id);
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ContextActivationResult {
    pub activated_context: ContextId,
    pub deactivated_contexts: Vec<ContextId>,
    pub activation_time: Instant,
}

#[derive(Debug, Clone)]
pub struct ContextSwitchResult {
    pub from_context: ContextId,
    pub to_context: ContextId,
    pub switch_duration: Duration,
    pub state_transferred: bool,
}

#[derive(Debug, Clone)]
pub enum MergeStrategy {
    Union,                    // Combine all beliefs
    Intersection,             // Only common beliefs
    Weighted(HashMap<ContextId, f32>), // Weight contexts differently
    ConflictResolution,       // Resolve conflicts using TMS
}

#[derive(Debug)]
pub struct ContextLifecycle {
    pub context_creation_times: HashMap<ContextId, Instant>,
    pub context_switch_history: Vec<ContextSwitch>,
    pub context_usage_stats: HashMap<ContextId, ContextUsageStats>,
}

#[derive(Debug, Clone)]
pub struct ContextSwitch {
    pub from_context: ContextId,
    pub to_context: ContextId,
    pub timestamp: Instant,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct ContextUsageStats {
    pub total_activations: usize,
    pub total_active_time: Duration,
    pub last_activation: Instant,
    pub switch_count: usize,
}
```

### Step 2: Context-Aware Spreader

```rust
// File: src/core/activation/context_aware_spreader.rs

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::core::context::context_manager::{ContextManager, ContextId};
use crate::core::activation::tms_constrained_spreader::TMSConstrainedSpreader;
use crate::core::activation::belief_aware_state::BeliefAwareActivationState;
use crate::core::types::NodeId;

pub struct ContextAwareSpreader {
    // Context management
    pub context_manager: Arc<RwLock<ContextManager>>,
    
    // Base spreader
    pub base_spreader: TMSConstrainedSpreader,
    
    // Context-specific configuration
    pub context_isolation_level: IsolationLevel,
    pub enable_cross_context_inference: bool,
    pub context_switch_threshold: f32,
    
    // Performance optimization
    pub context_cache: HashMap<ContextId, CachedContextState>,
    pub enable_context_prefetching: bool,
}

#[derive(Debug, Clone)]
pub enum IsolationLevel {
    Strict,      // Complete isolation between contexts
    Permeable,   // Allow some cross-context influence
    Shared,      // Share compatible beliefs across contexts
}

#[derive(Debug, Clone)]
pub struct CachedContextState {
    pub context_id: ContextId,
    pub cached_activations: HashMap<NodeId, f32>,
    pub cache_timestamp: std::time::Instant,
    pub cache_validity: std::time::Duration,
}

#[derive(Debug, Clone)]
pub struct ContextAwareActivationResult {
    // Results per context
    pub context_results: HashMap<ContextId, BeliefAwareActivationState>,
    
    // Cross-context analysis
    pub context_comparisons: Vec<ContextComparison>,
    pub consensus_analysis: ConsensusAnalysis,
    
    // Context switching information
    pub context_switches_performed: Vec<ContextSwitch>,
    pub active_contexts: Vec<ContextId>,
    
    // Performance metrics
    pub total_processing_time: std::time::Duration,
    pub context_overhead: std::time::Duration,
}

impl ContextAwareSpreader {
    pub fn new(
        context_manager: Arc<RwLock<ContextManager>>,
        base_spreader: TMSConstrainedSpreader,
    ) -> Self {
        Self {
            context_manager,
            base_spreader,
            context_isolation_level: IsolationLevel::Permeable,
            enable_cross_context_inference: true,
            context_switch_threshold: 0.1,
            context_cache: HashMap::new(),
            enable_context_prefetching: false,
        }
    }
    
    pub async fn spread_across_contexts(
        &mut self,
        initial_activations: HashMap<NodeId, f32>,
        graph: &Graph,
        context_ids: Vec<ContextId>,
    ) -> Result<ContextAwareActivationResult, ContextSpreadingError> {
        let processing_start = std::time::Instant::now();
        let mut context_results = HashMap::new();
        let mut context_switches = Vec::new();
        
        // Process each context
        for &context_id in &context_ids {
            let context_start = std::time::Instant::now();
            
            // Activate context
            let activation_result = {
                let mut manager = self.context_manager.write().await;
                manager.activate_context(context_id).await?
            };
            
            // Track context switches
            if !activation_result.deactivated_contexts.is_empty() {
                for &deactivated in &activation_result.deactivated_contexts {
                    context_switches.push(ContextSwitch {
                        from_context: deactivated,
                        to_context: context_id,
                        timestamp: context_start,
                        duration: context_start.elapsed(),
                    });
                }
            }
            
            // Get context-specific belief context
            let belief_context = {
                let manager = self.context_manager.read().await;
                let contexts = manager.active_contexts.read().await;
                contexts.get(&context_id)
                    .ok_or(ContextSpreadingError::ContextNotFound(context_id))?
                    .belief_context.clone()
            };
            
            // Perform spreading in this context
            let activation_result = self.base_spreader.spread_with_beliefs(
                initial_activations.clone(),
                graph,
                belief_context,
            ).await?;
            
            context_results.insert(context_id, activation_result);
        }
        
        // Analyze cross-context patterns
        let context_comparisons = self.compare_contexts(&context_results).await?;
        let consensus_analysis = self.analyze_consensus(&context_results).await?;
        
        let total_time = processing_start.elapsed();
        let context_overhead = self.calculate_context_overhead(&context_switches);
        
        Ok(ContextAwareActivationResult {
            context_results,
            context_comparisons,
            consensus_analysis,
            context_switches_performed: context_switches,
            active_contexts: context_ids,
            total_processing_time: total_time,
            context_overhead,
        })
    }
    
    pub async fn spread_with_dynamic_switching(
        &mut self,
        initial_activations: HashMap<NodeId, f32>,
        graph: &Graph,
        switching_criteria: SwitchingCriteria,
    ) -> Result<ContextAwareActivationResult, ContextSpreadingError> {
        let mut current_context = switching_criteria.initial_context;
        let mut context_results = HashMap::new();
        let mut context_switches = Vec::new();
        let mut iteration = 0;
        
        loop {
            // Perform spreading in current context
            let belief_context = self.get_context_belief_context(current_context).await?;
            
            let activation_result = self.base_spreader.spread_with_beliefs(
                initial_activations.clone(),
                graph,
                belief_context,
            ).await?;
            
            context_results.insert(current_context, activation_result.clone());
            
            // Check if we should switch contexts
            let switch_decision = self.evaluate_context_switch(
                &activation_result,
                &switching_criteria,
                iteration,
            ).await?;
            
            match switch_decision {
                SwitchDecision::Stay => break,
                SwitchDecision::SwitchTo(new_context) => {
                    let switch_start = std::time::Instant::now();
                    
                    // Perform context switch
                    let switch_result = {
                        let mut manager = self.context_manager.write().await;
                        manager.switch_context(current_context, new_context, true).await?
                    };
                    
                    context_switches.push(ContextSwitch {
                        from_context: current_context,
                        to_context: new_context,
                        timestamp: switch_start,
                        duration: switch_result.switch_duration,
                    });
                    
                    current_context = new_context;
                }
                SwitchDecision::Branch(branch_assumptions) => {
                    // Create new context branch
                    let branch_context = {
                        let mut manager = self.context_manager.write().await;
                        manager.branch_context(
                            current_context,
                            branch_assumptions,
                            Some(format!("branch_{}", iteration)),
                        ).await?
                    };
                    
                    current_context = branch_context;
                }
            }
            
            iteration += 1;
            
            // Prevent infinite switching
            if iteration > switching_criteria.max_iterations {
                break;
            }
        }
        
        // Analyze results
        let context_comparisons = self.compare_contexts(&context_results).await?;
        let consensus_analysis = self.analyze_consensus(&context_results).await?;
        
        Ok(ContextAwareActivationResult {
            context_results,
            context_comparisons,
            consensus_analysis,
            context_switches_performed: context_switches,
            active_contexts: vec![current_context],
            total_processing_time: std::time::Duration::ZERO, // TODO: Calculate
            context_overhead: self.calculate_context_overhead(&context_switches),
        })
    }
    
    async fn compare_contexts(
        &self,
        context_results: &HashMap<ContextId, BeliefAwareActivationState>,
    ) -> Result<Vec<ContextComparison>, ContextSpreadingError> {
        let mut comparisons = Vec::new();
        let context_ids: Vec<_> = context_results.keys().cloned().collect();
        
        // Compare each pair of contexts
        for i in 0..context_ids.len() {
            for j in (i + 1)..context_ids.len() {
                let context_a = context_ids[i];
                let context_b = context_ids[j];
                
                let state_a = &context_results[&context_a];
                let state_b = &context_results[&context_b];
                
                let comparison = self.compare_activation_states(
                    context_a,
                    state_a,
                    context_b,
                    state_b,
                ).await?;
                
                comparisons.push(comparison);
            }
        }
        
        Ok(comparisons)
    }
    
    async fn analyze_consensus(
        &self,
        context_results: &HashMap<ContextId, BeliefAwareActivationState>,
    ) -> Result<ConsensusAnalysis, ContextSpreadingError> {
        let mut consensus_nodes = HashMap::new();
        let mut conflicting_nodes = HashMap::new();
        
        // Find nodes that appear in multiple contexts
        let mut node_appearances = HashMap::new();
        
        for (context_id, state) in context_results {
            for node_id in state.base_state.activated_nodes() {
                node_appearances
                    .entry(node_id)
                    .or_insert_with(Vec::new)
                    .push(*context_id);
            }
        }
        
        // Analyze consensus and conflicts
        for (node_id, contexts) in node_appearances {
            if contexts.len() > 1 {
                // Get activation levels across contexts
                let mut activations = Vec::new();
                for &context_id in &contexts {
                    let activation = context_results[&context_id]
                        .base_state
                        .get_activation(node_id);
                    activations.push((context_id, activation));
                }
                
                // Calculate consensus metrics
                let mean_activation = activations.iter().map(|(_, a)| a).sum::<f32>() / activations.len() as f32;
                let variance = activations.iter()
                    .map(|(_, a)| (a - mean_activation).powi(2))
                    .sum::<f32>() / activations.len() as f32;
                
                if variance < 0.01 {
                    // High consensus
                    consensus_nodes.insert(node_id, ConsensusInfo {
                        node_id,
                        consensus_level: 1.0 - variance,
                        mean_activation,
                        participating_contexts: contexts.clone(),
                        confidence: self.calculate_consensus_confidence(&activations),
                    });
                } else {
                    // Conflict detected
                    conflicting_nodes.insert(node_id, ConflictInfo {
                        node_id,
                        conflict_level: variance,
                        activation_range: (
                            activations.iter().map(|(_, a)| a).fold(f32::INFINITY, |a, &b| a.min(b)),
                            activations.iter().map(|(_, a)| a).fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                        ),
                        conflicting_contexts: contexts.clone(),
                        resolution_strategy: self.suggest_conflict_resolution(&activations),
                    });
                }
            }
        }
        
        Ok(ConsensusAnalysis {
            consensus_nodes,
            conflicting_nodes,
            overall_consensus_score: self.calculate_overall_consensus(&consensus_nodes, &conflicting_nodes),
            consensus_summary: self.generate_consensus_summary(&consensus_nodes, &conflicting_nodes),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ContextComparison {
    pub context_a: ContextId,
    pub context_b: ContextId,
    pub similarity_score: f32,
    pub common_activations: Vec<NodeId>,
    pub unique_to_a: Vec<NodeId>,
    pub unique_to_b: Vec<NodeId>,
    pub activation_differences: HashMap<NodeId, f32>,
}

#[derive(Debug, Clone)]
pub struct ConsensusAnalysis {
    pub consensus_nodes: HashMap<NodeId, ConsensusInfo>,
    pub conflicting_nodes: HashMap<NodeId, ConflictInfo>,
    pub overall_consensus_score: f32,
    pub consensus_summary: ConsensusSummary,
}

#[derive(Debug, Clone)]
pub struct ConsensusInfo {
    pub node_id: NodeId,
    pub consensus_level: f32,
    pub mean_activation: f32,
    pub participating_contexts: Vec<ContextId>,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct ConflictInfo {
    pub node_id: NodeId,
    pub conflict_level: f32,
    pub activation_range: (f32, f32),
    pub conflicting_contexts: Vec<ContextId>,
    pub resolution_strategy: ConflictResolutionStrategy,
}

#[derive(Debug, Clone)]
pub enum ConflictResolutionStrategy {
    UseHighestActivation,
    AverageActivations,
    UseHighestConfidenceContext,
    RequireManualResolution,
}
```

### Step 3: Context Switch Orchestrator

```rust
// File: src/core/context/context_switch_orchestrator.rs

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use std::time::{Instant, Duration};
use crate::core::context::context_manager::{ContextManager, ContextId};

pub struct ContextSwitchOrchestrator {
    // Context management
    pub context_manager: Arc<RwLock<ContextManager>>,
    
    // Switch triggers
    pub switch_triggers: Arc<Mutex<Vec<SwitchTrigger>>>,
    
    // Switch queue
    pub switch_queue: Arc<Mutex<VecDeque<ScheduledSwitch>>>,
    
    // Orchestration config
    pub max_concurrent_switches: usize,
    pub switch_batch_size: usize,
    pub switch_timeout: Duration,
    
    // Performance tracking
    pub switch_metrics: Arc<Mutex<SwitchMetrics>>,
}

#[derive(Debug, Clone)]
pub enum SwitchTrigger {
    // Activation-based triggers
    ActivationThreshold {
        node_id: NodeId,
        threshold: f32,
        direction: ThresholdDirection,
    },
    
    // Belief-based triggers
    BeliefRevision {
        belief_id: BeliefId,
        revision_type: RevisionType,
    },
    
    // Time-based triggers
    TemporalEvent {
        timestamp: Timestamp,
        event_type: TemporalEventType,
    },
    
    // Conflict-based triggers
    ConflictDetected {
        conflict_severity: ConflictSeverity,
        affected_nodes: Vec<NodeId>,
    },
    
    // Resource-based triggers
    ResourcePressure {
        resource_type: ResourceType,
        pressure_level: f32,
    },
}

#[derive(Debug, Clone)]
pub struct ScheduledSwitch {
    pub switch_id: SwitchId,
    pub from_context: ContextId,
    pub to_context: ContextId,
    pub trigger: SwitchTrigger,
    pub priority: SwitchPriority,
    pub scheduled_time: Instant,
    pub deadline: Option<Instant>,
    pub dependencies: Vec<SwitchId>,
    pub state_transfer: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SwitchPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
    Emergency = 5,
}

impl ContextSwitchOrchestrator {
    pub fn new(context_manager: Arc<RwLock<ContextManager>>) -> Self {
        Self {
            context_manager,
            switch_triggers: Arc::new(Mutex::new(Vec::new())),
            switch_queue: Arc::new(Mutex::new(VecDeque::new())),
            max_concurrent_switches: 3,
            switch_batch_size: 5,
            switch_timeout: Duration::from_millis(100),
            switch_metrics: Arc::new(Mutex::new(SwitchMetrics::new())),
        }
    }
    
    pub async fn register_trigger(&mut self, trigger: SwitchTrigger) -> Result<TriggerId, OrchestratorError> {
        let trigger_id = TriggerId::new();
        
        let mut triggers = self.switch_triggers.lock().await;
        triggers.push(trigger);
        
        Ok(trigger_id)
    }
    
    pub async fn schedule_switch(
        &mut self,
        from_context: ContextId,
        to_context: ContextId,
        trigger: SwitchTrigger,
        priority: SwitchPriority,
        state_transfer: bool,
    ) -> Result<SwitchId, OrchestratorError> {
        let switch_id = SwitchId::new();
        
        let scheduled_switch = ScheduledSwitch {
            switch_id,
            from_context,
            to_context,
            trigger,
            priority,
            scheduled_time: Instant::now(),
            deadline: None,
            dependencies: Vec::new(),
            state_transfer,
        };
        
        // Validate switch is possible
        self.validate_scheduled_switch(&scheduled_switch).await?;
        
        // Add to queue in priority order
        let mut queue = self.switch_queue.lock().await;
        
        // Find insertion position based on priority
        let insertion_pos = queue.iter()
            .position(|s| s.priority < scheduled_switch.priority)
            .unwrap_or(queue.len());
        
        queue.insert(insertion_pos, scheduled_switch);
        
        Ok(switch_id)
    }
    
    pub async fn execute_pending_switches(&mut self) -> Result<SwitchExecutionResult, OrchestratorError> {
        let mut executed_switches = Vec::new();
        let mut failed_switches = Vec::new();
        let execution_start = Instant::now();
        
        // Process switches in batches
        for _batch in 0..self.switch_batch_size {
            let switch_to_execute = {
                let mut queue = self.switch_queue.lock().await;
                queue.pop_front()
            };
            
            if let Some(scheduled_switch) = switch_to_execute {
                match self.execute_switch(scheduled_switch.clone()).await {
                    Ok(execution_result) => {
                        executed_switches.push(execution_result);
                        
                        // Update metrics
                        let mut metrics = self.switch_metrics.lock().await;
                        metrics.record_successful_switch(
                            scheduled_switch.switch_id,
                            execution_start.elapsed(),
                        );
                    }
                    Err(e) => {
                        failed_switches.push((scheduled_switch.switch_id, e));
                        
                        // Update metrics
                        let mut metrics = self.switch_metrics.lock().await;
                        metrics.record_failed_switch(scheduled_switch.switch_id);
                    }
                }
            } else {
                // No more switches to execute
                break;
            }
        }
        
        Ok(SwitchExecutionResult {
            executed_switches,
            failed_switches,
            execution_time: execution_start.elapsed(),
            batch_size: self.switch_batch_size,
        })
    }
    
    pub async fn orchestrate_optimal_switching(
        &mut self,
        optimization_goal: OptimizationGoal,
        context_constraints: ContextConstraints,
    ) -> Result<OptimalSwitchingPlan, OrchestratorError> {
        // Analyze current context landscape
        let context_analysis = self.analyze_context_landscape().await?;
        
        // Generate switching plan based on goal
        let switching_plan = match optimization_goal {
            OptimizationGoal::MinimizeLatency => {
                self.generate_latency_optimal_plan(&context_analysis, &context_constraints).await?
            }
            OptimizationGoal::MaximizeCoverage => {
                self.generate_coverage_optimal_plan(&context_analysis, &context_constraints).await?
            }
            OptimizationGoal::BalanceResourceUsage => {
                self.generate_resource_balanced_plan(&context_analysis, &context_constraints).await?
            }
            OptimizationGoal::MinimizeConflicts => {
                self.generate_conflict_minimal_plan(&context_analysis, &context_constraints).await?
            }
        };
        
        // Validate plan feasibility
        self.validate_switching_plan(&switching_plan).await?;
        
        // Schedule switches according to plan
        for planned_switch in &switching_plan.switches {
            self.schedule_switch(
                planned_switch.from_context,
                planned_switch.to_context,
                planned_switch.trigger.clone(),
                planned_switch.priority,
                planned_switch.state_transfer,
            ).await?;
        }
        
        Ok(switching_plan)
    }
    
    async fn execute_switch(
        &mut self,
        scheduled_switch: ScheduledSwitch,
    ) -> Result<SwitchExecutionResult, OrchestratorError> {
        let execution_start = Instant::now();
        
        // Check dependencies
        self.verify_switch_dependencies(&scheduled_switch).await?;
        
        // Execute the switch
        let switch_result = {
            let mut manager = self.context_manager.write().await;
            manager.switch_context(
                scheduled_switch.from_context,
                scheduled_switch.to_context,
                scheduled_switch.state_transfer,
            ).await?
        };
        
        // Update context utilization metrics
        self.update_context_utilization_metrics(
            scheduled_switch.from_context,
            scheduled_switch.to_context,
            execution_start.elapsed(),
        ).await?;
        
        Ok(SwitchExecutionResult {
            executed_switches: vec![ExecutedSwitch {
                switch_id: scheduled_switch.switch_id,
                from_context: switch_result.from_context,
                to_context: switch_result.to_context,
                execution_time: execution_start.elapsed(),
                state_transferred: switch_result.state_transferred,
            }],
            failed_switches: Vec::new(),
            execution_time: execution_start.elapsed(),
            batch_size: 1,
        })
    }
    
    async fn analyze_context_landscape(&self) -> Result<ContextLandscapeAnalysis, OrchestratorError> {
        let manager = self.context_manager.read().await;
        let contexts = manager.active_contexts.read().await;
        
        let mut active_contexts = Vec::new();
        let mut resource_usage = HashMap::new();
        let mut context_relationships = HashMap::new();
        
        for (context_id, context) in contexts.iter() {
            if context.is_active {
                active_contexts.push(*context_id);
                resource_usage.insert(*context_id, ResourceUsage {
                    memory: context.memory_usage,
                    cpu_time: context.cpu_time,
                    last_used: context.last_used,
                });
            }
            
            // Analyze relationships
            let relationships = ContextRelationships {
                parent: context.parent_context,
                children: context.child_contexts.clone(),
                incompatible_contexts: self.find_incompatible_contexts(*context_id).await?,
            };
            context_relationships.insert(*context_id, relationships);
        }
        
        Ok(ContextLandscapeAnalysis {
            active_contexts,
            resource_usage,
            context_relationships,
            total_contexts: contexts.len(),
            landscape_complexity: self.calculate_landscape_complexity(&contexts),
        })
    }
}

#[derive(Debug, Clone)]
pub struct OptimalSwitchingPlan {
    pub switches: Vec<PlannedSwitch>,
    pub optimization_goal: OptimizationGoal,
    pub estimated_execution_time: Duration,
    pub expected_resource_usage: ResourceUsage,
    pub confidence_score: f32,
}

#[derive(Debug, Clone)]
pub enum OptimizationGoal {
    MinimizeLatency,
    MaximizeCoverage,
    BalanceResourceUsage,
    MinimizeConflicts,
}
```

### Step 4: Multi-Context Query Processor

```rust
// File: src/core/query/multi_context_query_processor.rs

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::core::context::context_manager::{ContextManager, ContextId};
use crate::core::activation::context_aware_spreader::{ContextAwareSpreader, ContextAwareActivationResult};
use crate::core::query::intent::{QueryIntent, MultiContextQueryIntent};

pub struct MultiContextQueryProcessor {
    // Core components
    pub context_manager: Arc<RwLock<ContextManager>>,
    pub context_aware_spreader: ContextAwareSpreader,
    
    // Query configuration
    pub default_context_selection: ContextSelectionStrategy,
    pub result_aggregation_method: AggregationMethod,
    pub conflict_resolution_strategy: ConflictResolutionStrategy,
    
    // Performance optimization
    pub enable_parallel_processing: bool,
    pub context_result_caching: bool,
    pub max_concurrent_contexts: usize,
}

#[derive(Debug, Clone)]
pub enum ContextSelectionStrategy {
    All,                        // Use all available contexts
    Relevant(f32),              // Use contexts with relevance > threshold
    TopK(usize),               // Use top K most relevant contexts
    Manual(Vec<ContextId>),    // Manually specified contexts
    Dynamic,                   // Dynamically select based on query
}

#[derive(Debug, Clone)]
pub enum AggregationMethod {
    Union,                     // Combine all results
    Intersection,              // Only common results
    Weighted(HashMap<ContextId, f32>), // Weight contexts differently
    ConsensusFiltered(f32),    // Filter by consensus threshold
    ConflictAware,             // Account for conflicts in aggregation
}

#[derive(Debug, Clone)]
pub struct MultiContextQueryResult {
    // Individual context results
    pub context_results: HashMap<ContextId, BeliefAwareActivationState>,
    
    // Aggregated results
    pub aggregated_result: AggregatedActivationResult,
    
    // Analysis
    pub context_analysis: MultiContextAnalysis,
    pub consensus_report: ConsensusReport,
    pub conflict_report: ConflictReport,
    
    // Recommendations
    pub context_recommendations: Vec<ContextRecommendation>,
    pub query_refinement_suggestions: Vec<QueryRefinementSuggestion>,
    
    // Metadata
    pub processing_metadata: ProcessingMetadata,
}

impl MultiContextQueryProcessor {
    pub fn new(
        context_manager: Arc<RwLock<ContextManager>>,
        context_aware_spreader: ContextAwareSpreader,
    ) -> Self {
        Self {
            context_manager,
            context_aware_spreader,
            default_context_selection: ContextSelectionStrategy::Relevant(0.5),
            result_aggregation_method: AggregationMethod::ConsensusFiltered(0.7),
            conflict_resolution_strategy: ConflictResolutionStrategy::UseHighestConfidenceContext,
            enable_parallel_processing: true,
            context_result_caching: true,
            max_concurrent_contexts: 5,
        }
    }
    
    pub async fn process_multi_context_query(
        &mut self,
        query_intent: &QueryIntent,
        multi_context_intent: &MultiContextQueryIntent,
        graph: &Graph,
    ) -> Result<MultiContextQueryResult, MultiContextQueryError> {
        // Select contexts based on strategy
        let selected_contexts = self.select_contexts(
            query_intent,
            multi_context_intent,
        ).await?;
        
        // Extract initial activations
        let initial_activations = self.extract_query_activations(query_intent).await?;
        
        // Process query across selected contexts
        let context_aware_result = self.context_aware_spreader.spread_across_contexts(
            initial_activations,
            graph,
            selected_contexts.clone(),
        ).await?;
        
        // Aggregate results
        let aggregated_result = self.aggregate_context_results(
            &context_aware_result.context_results,
        ).await?;
        
        // Analyze contexts
        let context_analysis = self.analyze_multi_context_results(
            &context_aware_result,
        ).await?;
        
        // Generate consensus and conflict reports
        let consensus_report = self.generate_consensus_report(
            &context_aware_result.consensus_analysis,
        ).await?;
        
        let conflict_report = self.generate_conflict_report(
            &context_aware_result.consensus_analysis,
        ).await?;
        
        // Generate recommendations
        let context_recommendations = self.generate_context_recommendations(
            &context_analysis,
            &consensus_report,
            &conflict_report,
        ).await?;
        
        let query_refinement_suggestions = self.generate_query_refinement_suggestions(
            query_intent,
            &context_analysis,
        ).await?;
        
        Ok(MultiContextQueryResult {
            context_results: context_aware_result.context_results,
            aggregated_result,
            context_analysis,
            consensus_report,
            conflict_report,
            context_recommendations,
            query_refinement_suggestions,
            processing_metadata: ProcessingMetadata {
                total_contexts_processed: selected_contexts.len(),
                processing_time: context_aware_result.total_processing_time,
                context_overhead: context_aware_result.context_overhead,
                cache_hits: 0, // TODO: Track cache usage
            },
        })
    }
    
    async fn select_contexts(
        &self,
        query_intent: &QueryIntent,
        multi_context_intent: &MultiContextQueryIntent,
    ) -> Result<Vec<ContextId>, MultiContextQueryError> {
        let strategy = multi_context_intent.context_selection
            .as_ref()
            .unwrap_or(&self.default_context_selection);
        
        match strategy {
            ContextSelectionStrategy::All => {
                self.get_all_active_contexts().await
            }
            
            ContextSelectionStrategy::Relevant(threshold) => {
                self.get_relevant_contexts(query_intent, *threshold).await
            }
            
            ContextSelectionStrategy::TopK(k) => {
                self.get_top_k_contexts(query_intent, *k).await
            }
            
            ContextSelectionStrategy::Manual(context_ids) => {
                Ok(context_ids.clone())
            }
            
            ContextSelectionStrategy::Dynamic => {
                self.dynamically_select_contexts(query_intent).await
            }
        }
    }
    
    async fn aggregate_context_results(
        &self,
        context_results: &HashMap<ContextId, BeliefAwareActivationState>,
    ) -> Result<AggregatedActivationResult, MultiContextQueryError> {
        match &self.result_aggregation_method {
            AggregationMethod::Union => {
                self.aggregate_by_union(context_results).await
            }
            
            AggregationMethod::Intersection => {
                self.aggregate_by_intersection(context_results).await
            }
            
            AggregationMethod::Weighted(weights) => {
                self.aggregate_by_weights(context_results, weights).await
            }
            
            AggregationMethod::ConsensusFiltered(threshold) => {
                self.aggregate_by_consensus(context_results, *threshold).await
            }
            
            AggregationMethod::ConflictAware => {
                self.aggregate_conflict_aware(context_results).await
            }
        }
    }
    
    async fn aggregate_by_consensus(
        &self,
        context_results: &HashMap<ContextId, BeliefAwareActivationState>,
        threshold: f32,
    ) -> Result<AggregatedActivationResult, MultiContextQueryError> {
        let mut aggregated_activations = HashMap::new();
        let mut consensus_scores = HashMap::new();
        
        // Find all unique nodes across contexts
        let mut all_nodes = HashSet::new();
        for state in context_results.values() {
            all_nodes.extend(state.base_state.activated_nodes());
        }
        
        // Calculate consensus for each node
        for node_id in all_nodes {
            let mut activations = Vec::new();
            let mut contexts_with_node = Vec::new();
            
            for (context_id, state) in context_results {
                let activation = state.base_state.get_activation(node_id);
                if activation > 0.0 {
                    activations.push(activation);
                    contexts_with_node.push(*context_id);
                }
            }
            
            if !activations.is_empty() {
                // Calculate consensus metrics
                let mean_activation = activations.iter().sum::<f32>() / activations.len() as f32;
                let variance = activations.iter()
                    .map(|a| (a - mean_activation).powi(2))
                    .sum::<f32>() / activations.len() as f32;
                
                let consensus_score = 1.0 - variance.sqrt() / mean_activation;
                
                // Include node if consensus meets threshold
                if consensus_score >= threshold {
                    aggregated_activations.insert(node_id, mean_activation);
                    consensus_scores.insert(node_id, consensus_score);
                }
            }
        }
        
        Ok(AggregatedActivationResult {
            final_activations: aggregated_activations,
            aggregation_method: self.result_aggregation_method.clone(),
            consensus_scores: Some(consensus_scores),
            confidence_scores: self.calculate_aggregated_confidence(context_results).await?,
            metadata: AggregationMetadata {
                total_nodes_considered: all_nodes.len(),
                nodes_included: aggregated_activations.len(),
                consensus_threshold: Some(threshold),
            },
        })
    }
}

#[derive(Debug, Clone)]
pub struct AggregatedActivationResult {
    pub final_activations: HashMap<NodeId, f32>,
    pub aggregation_method: AggregationMethod,
    pub consensus_scores: Option<HashMap<NodeId, f32>>,
    pub confidence_scores: HashMap<NodeId, f32>,
    pub metadata: AggregationMetadata,
}

#[derive(Debug, Clone)]
pub struct MultiContextAnalysis {
    pub context_diversity: f32,
    pub activation_overlap: f32,
    pub conflict_density: f32,
    pub consensus_strength: f32,
    pub context_efficiency: HashMap<ContextId, f32>,
}

#[derive(Debug, Clone)]
pub struct ConsensusReport {
    pub overall_consensus_level: f32,
    pub high_consensus_nodes: Vec<NodeId>,
    pub moderate_consensus_nodes: Vec<NodeId>,
    pub low_consensus_nodes: Vec<NodeId>,
    pub consensus_trends: Vec<ConsensusTrend>,
}

#[derive(Debug, Clone)]
pub struct ConflictReport {
    pub total_conflicts: usize,
    pub severe_conflicts: Vec<ConflictDescription>,
    pub moderate_conflicts: Vec<ConflictDescription>,
    pub conflict_patterns: Vec<ConflictPattern>,
    pub resolution_recommendations: Vec<ConflictResolutionRecommendation>,
}
```

## File Locations

- `src/core/context/context_manager.rs` - Context lifecycle management
- `src/core/activation/context_aware_spreader.rs` - Context-aware spreading algorithm
- `src/core/context/context_switch_orchestrator.rs` - Context switching orchestration
- `src/core/query/multi_context_query_processor.rs` - High-level multi-context queries
- `tests/context/context_switching_tests.rs` - Comprehensive test suite

## Success Criteria

- [ ] ContextManager handles context lifecycle correctly
- [ ] ContextAwareSpreader isolates contexts properly
- [ ] ContextSwitchOrchestrator optimizes switching performance
- [ ] MultiContextQueryProcessor aggregates results accurately
- [ ] Context switch latency <1ms between contexts
- [ ] Multi-context queries scale to 10+ contexts efficiently
- [ ] All tests pass:
  - Context creation and cleanup
  - Context isolation and switching
  - Multi-context result aggregation
  - Conflict detection and resolution
  - Resource usage optimization
  - Context inheritance and branching

## Test Requirements

```rust
#[test]
async fn test_context_creation_and_management() {
    let mut manager = ContextManager::new();
    
    let belief_context = BeliefContext::default();
    let context_id = manager.create_context(
        belief_context,
        None,
        ContextPriority::Medium,
    ).await.unwrap();
    
    assert_ne!(context_id, ContextId(0));
    
    let contexts = manager.active_contexts.read().await;
    assert!(contexts.contains_key(&context_id));
}

#[test]
async fn test_context_switching() {
    let mut manager = ContextManager::new();
    
    let context_a = manager.create_context(
        BeliefContext::default(),
        None,
        ContextPriority::Medium,
    ).await.unwrap();
    
    let context_b = manager.create_context(
        BeliefContext::default(),
        None,
        ContextPriority::High,
    ).await.unwrap();
    
    // Activate context A
    manager.activate_context(context_a).await.unwrap();
    
    // Switch to context B
    let switch_result = manager.switch_context(context_a, context_b, true)
        .await.unwrap();
    
    assert_eq!(switch_result.from_context, context_a);
    assert_eq!(switch_result.to_context, context_b);
    assert!(switch_result.switch_duration < Duration::from_millis(10));
}

#[test]
async fn test_multi_context_spreading() {
    let context_manager = Arc::new(RwLock::new(ContextManager::new()));
    let base_spreader = setup_test_spreader().await;
    let mut context_spreader = ContextAwareSpreader::new(context_manager.clone(), base_spreader);
    
    // Create multiple contexts
    let mut context_ids = Vec::new();
    for i in 0..3 {
        let context_id = {
            let mut manager = context_manager.write().await;
            manager.create_context(
                BeliefContext::default(),
                None,
                ContextPriority::Medium,
            ).await.unwrap()
        };
        context_ids.push(context_id);
    }
    
    let initial_activations = HashMap::from([
        (NodeId(1), 0.8),
        (NodeId(2), 0.6),
    ]);
    
    let result = context_spreader.spread_across_contexts(
        initial_activations,
        &test_graph(),
        context_ids.clone(),
    ).await.unwrap();
    
    assert_eq!(result.context_results.len(), 3);
    assert!(!result.context_comparisons.is_empty());
}

#[test]
async fn test_context_consensus_analysis() {
    let context_manager = Arc::new(RwLock::new(ContextManager::new()));
    let base_spreader = setup_test_spreader().await;
    let context_spreader = ContextAwareSpreader::new(context_manager, base_spreader);
    
    // Create test results with known consensus patterns
    let mut context_results = HashMap::new();
    
    // Context 1: High activation for node 1
    let mut state1 = setup_test_belief_state().await;
    state1.base_state.set_activation(NodeId(1), 0.8);
    context_results.insert(ContextId(1), state1);
    
    // Context 2: Similar activation for node 1
    let mut state2 = setup_test_belief_state().await;
    state2.base_state.set_activation(NodeId(1), 0.75);
    context_results.insert(ContextId(2), state2);
    
    let consensus = context_spreader.analyze_consensus(&context_results)
        .await.unwrap();
    
    assert!(consensus.consensus_nodes.contains_key(&NodeId(1)));
    assert!(consensus.overall_consensus_score > 0.8);
}
```

## Quality Gates

- [ ] Context switch latency consistently <1ms
- [ ] Memory isolation between contexts maintained
- [ ] Multi-context processing scales linearly with context count
- [ ] Consensus analysis accuracy >90% for known patterns
- [ ] Context cleanup prevents memory leaks
- [ ] Conflict resolution strategies work correctly

## Next Task

Upon completion, proceed to **34_justification_paths.md**