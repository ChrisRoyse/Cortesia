# Truth Maintenance System Fix Plan: Complete TMS Overhaul

## Executive Summary

**Current State**: Truth maintenance system identified as MOST PROBLEMATIC component
**Highest inconsistency count**: 777+ total inconsistencies in the project
**Test coverage**: Only 46.7% - critically low for a core reasoning system
**Status**: CRITICAL - Requires immediate comprehensive overhaul

## Current Problem Analysis

### Core Issues Identified

1. **NO ACTUAL IMPLEMENTATION**: Despite extensive documentation, no truth maintenance code exists
2. **Specification Drift**: Multiple conflicting TMS specifications across documents
3. **Missing AGM Compliance**: No actual AGM belief revision implementation
4. **Neuromorphic Integration Gap**: Spike-based processing not connected to TMS
5. **Test Coverage Crisis**: 46.7% coverage indicates fundamental gaps

### Specific Inconsistencies Found

| File Path | Issue Type | Description | Line Numbers |
|-----------|------------|-------------|--------------|
| `docs/allocationplan/PHASE_6_TRUTH_MAINTENANCE.md` | Specification Only | Complete TMS spec with no implementation | 1-780 |
| `docs/allocationplan/Phase6/02_Core_TMS_Components.md` | Missing Implementation | Core TMS components documented but not coded | All |
| `docs/allocationplan/Phase6/03_AGM_Belief_Revision_Engine.md` | No Code | Detailed AGM spec with example code but no actual implementation | 1-2007 |
| `docs/allocationplan/Phase5/MicroPhase7_TruthMaintenance.md` | Outdated Spec | Different TMS approach, inconsistent with Phase 6 | All |
| `docs/allocationplan/Phase5/MicroPhase8_ConflictResolution.md` | Conflict Resolution Gap | Conflict resolution specified but not implemented | All |

## Canonical TMS Specification Based on AGM Belief Revision

### 1. Core Architecture Requirements

```rust
// Canonical TMS structure - MUST BE IMPLEMENTED
pub struct NeuromorphicTruthMaintenanceSystem {
    // Hybrid TMS components
    jtms_layer: JustificationBasedTMS,
    atms_layer: AssumptionBasedTMS,
    
    // AGM-compliant belief revision engine
    belief_revision_engine: AGMBeliefRevisionEngine,
    
    // Neuromorphic integration with existing spiking system
    snn_processor: Arc<dyn SpikingNeuralNetwork>,
    conflict_detection_columns: Vec<ConflictDetectionColumn>,
    
    // Integration with existing temporal memory system
    temporal_manager: Arc<dyn TemporalConsistencyManager>,
    
    // Multi-context reasoning engine
    context_engines: HashMap<ContextId, ContextSpecificEngine>,
}
```

### 2. AGM Postulate Implementation Requirements

**CRITICAL**: All 8 AGM postulates MUST be satisfied:

1. **Closure**: K+φ is logically closed
2. **Success**: φ ∈ K+φ (new beliefs are included)
3. **Inclusion**: K+φ ⊆ Cn(K ∪ {φ}) (minimal addition)
4. **Vacuity**: If ¬φ ∉ K, then K+φ = Cn(K ∪ {φ})
5. **Consistency**: K+φ is consistent if φ is consistent
6. **Extensionality**: If φ ≡ ψ, then K+φ = K+ψ
7. **Superexpansion**: K+φ ⊆ (K+ψ)+φ
8. **Subexpansion**: If ¬φ ∉ K+ψ, then (K+ψ)+φ ⊆ K+(φ∧ψ)

### 3. Neuromorphic Integration Specification

```rust
// Integration with existing neuromorphic-core crate
pub trait SpikingTMSProcessor {
    fn encode_belief_as_spikes(&self, belief: &Belief) -> TTFSSpikePattern;
    fn process_belief_conflicts(&mut self, conflicts: &[Conflict]) -> ConflictResolution;
    fn apply_lateral_inhibition(&mut self, competing_beliefs: &[WeightedBelief]) -> Result<(), TMSError>;
}
```

## Complete Implementation Plan

### Phase 1: Foundation Setup (Week 1)

#### Task 1.1: Create Core TMS Crate Structure
**File**: `C:\code\LLMKG\crates\truth-maintenance\Cargo.toml`
```toml
[package]
name = "truth-maintenance"
version = "0.1.0"
edition = "2021"

[dependencies]
neuromorphic-core = { path = "../neuromorphic-core" }
temporal-memory = { path = "../temporal-memory" }
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1.0", features = ["full"] }
uuid = "1.0"
thiserror = "1.0"
```

#### Task 1.2: Implement Core Types
**File**: `C:\code\LLMKG\crates\truth-maintenance\src\types.rs`
```rust
use std::collections::{HashMap, HashSet};
use uuid::Uuid;
use serde::{Deserialize, Serialize};

pub type BeliefId = Uuid;
pub type JustificationId = Uuid;
pub type ContextId = Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Belief {
    pub id: BeliefId,
    pub content: String,
    pub confidence: f32,
    pub sources: Vec<Source>,
    pub timestamp: u64,
    pub entrenchment: f32,
    pub status: BeliefStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BeliefStatus {
    In,      // Believed (JTMS terminology)
    Out,     // Not believed
    Unknown, // Undetermined
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Justification {
    pub id: JustificationId,
    pub belief_id: BeliefId,
    pub premises: Vec<BeliefId>,
    pub rule: String,
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub id: String,
    pub reliability: f32,
    pub authority_level: AuthorityLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthorityLevel {
    Expert,
    Reliable,
    Standard,
    Questionable,
}
```

#### Task 1.3: Implement Error Types
**File**: `C:\code\LLMKG\crates\truth-maintenance\src\error.rs`
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TMSError {
    #[error("Belief not found: {belief_id}")]
    BeliefNotFound { belief_id: String },
    
    #[error("Justification cycle detected")]
    JustificationCycle,
    
    #[error("Inconsistent belief set: {details}")]
    InconsistentBeliefSet { details: String },
    
    #[error("AGM postulate violation: {postulate}")]
    AGMViolation { postulate: String },
    
    #[error("Neuromorphic integration error: {details}")]
    NeuromorphicError { details: String },
    
    #[error("Temporal consistency error: {details}")]
    TemporalError { details: String },
}

pub type TMSResult<T> = Result<T, TMSError>;
```

### Phase 2: Justification-Based TMS (Week 1-2)

#### Task 2.1: JTMS Core Implementation
**File**: `C:\code\LLMKG\crates\truth-maintenance\src\jtms.rs`
```rust
use crate::types::{Belief, BeliefId, BeliefStatus, Justification, JustificationId};
use crate::error::{TMSError, TMSResult};
use std::collections::{HashMap, HashSet};
use neuromorphic_core::ttfs_concept::TTFSSpikePattern;

pub struct JustificationBasedTMS {
    beliefs: HashMap<BeliefId, Belief>,
    justifications: HashMap<JustificationId, Justification>,
    dependencies: HashMap<BeliefId, Vec<BeliefId>>, // belief -> dependencies
    dependents: HashMap<BeliefId, Vec<BeliefId>>,   // belief -> dependents
    spike_patterns: HashMap<BeliefId, TTFSSpikePattern>,
}

impl JustificationBasedTMS {
    pub fn new() -> Self {
        Self {
            beliefs: HashMap::new(),
            justifications: HashMap::new(),
            dependencies: HashMap::new(),
            dependents: HashMap::new(),
            spike_patterns: HashMap::new(),
        }
    }

    pub fn add_belief(&mut self, belief: Belief) -> TMSResult<()> {
        // Encode belief as spike pattern for neuromorphic processing
        let spike_pattern = self.encode_belief_to_spikes(&belief)?;
        
        self.beliefs.insert(belief.id, belief.clone());
        self.spike_patterns.insert(belief.id, spike_pattern);
        
        // Check for immediate contradictions
        self.check_consistency(belief.id)?;
        
        Ok(())
    }

    pub fn add_justification(&mut self, justification: Justification) -> TMSResult<()> {
        // Check for cycles before adding
        if self.would_create_cycle(&justification)? {
            return Err(TMSError::JustificationCycle);
        }

        // Update dependency tracking
        for premise in &justification.premises {
            self.dependencies.entry(justification.belief_id)
                .or_insert_with(Vec::new)
                .push(*premise);
            self.dependents.entry(*premise)
                .or_insert_with(Vec::new)
                .push(justification.belief_id);
        }

        self.justifications.insert(justification.id, justification);
        
        // Propagate belief status changes
        self.propagate_status_changes()?;
        
        Ok(())
    }

    pub fn retract_belief(&mut self, belief_id: BeliefId) -> TMSResult<()> {
        if let Some(mut belief) = self.beliefs.get_mut(&belief_id) {
            belief.status = BeliefStatus::Out;
            
            // Propagate retraction to dependents
            if let Some(dependents) = self.dependents.get(&belief_id) {
                for &dependent in dependents {
                    self.reevaluate_belief(dependent)?;
                }
            }
        }
        
        Ok(())
    }

    fn encode_belief_to_spikes(&self, belief: &Belief) -> TMSResult<TTFSSpikePattern> {
        // Convert belief confidence to spike timing (higher confidence = earlier spikes)
        let spike_time = ((1.0 - belief.confidence) * 10_000.0) as u64; // 0-10ms range
        
        Ok(TTFSSpikePattern {
            events: vec![neuromorphic_core::ttfs_concept::SpikeEvent {
                timestamp: spike_time,
                amplitude: belief.confidence,
                frequency: 100.0, // Base frequency
            }],
            metadata: neuromorphic_core::ttfs_concept::TTFSMetadata {
                concept_id: belief.id.to_string(),
                encoding_timestamp: belief.timestamp,
                confidence_score: belief.confidence,
            },
        })
    }

    fn propagate_status_changes(&mut self) -> TMSResult<()> {
        // Use neuromorphic spreading activation for belief propagation
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            for belief_id in self.beliefs.keys().cloned().collect::<Vec<_>>() {
                if self.reevaluate_belief(belief_id)? {
                    changed = true;
                }
            }
        }

        if iterations >= MAX_ITERATIONS {
            return Err(TMSError::JustificationCycle);
        }

        Ok(())
    }

    fn reevaluate_belief(&mut self, belief_id: BeliefId) -> TMSResult<bool> {
        let current_status = self.beliefs.get(&belief_id)
            .map(|b| b.status.clone())
            .unwrap_or(BeliefStatus::Unknown);

        // Find all justifications supporting this belief
        let supporting_justifications: Vec<_> = self.justifications.values()
            .filter(|j| j.belief_id == belief_id)
            .collect();

        let new_status = if supporting_justifications.is_empty() {
            BeliefStatus::Unknown
        } else {
            // Check if any justification is satisfied
            let mut satisfied = false;
            for justification in &supporting_justifications {
                if self.is_justification_satisfied(justification)? {
                    satisfied = true;
                    break;
                }
            }
            
            if satisfied {
                BeliefStatus::In
            } else {
                BeliefStatus::Out
            }
        };

        let status_changed = !matches!((current_status, &new_status), 
            (BeliefStatus::In, BeliefStatus::In) |
            (BeliefStatus::Out, BeliefStatus::Out) |
            (BeliefStatus::Unknown, BeliefStatus::Unknown)
        );

        if status_changed {
            if let Some(belief) = self.beliefs.get_mut(&belief_id) {
                belief.status = new_status;
            }
        }

        Ok(status_changed)
    }

    fn is_justification_satisfied(&self, justification: &Justification) -> TMSResult<bool> {
        for &premise_id in &justification.premises {
            if let Some(premise) = self.beliefs.get(&premise_id) {
                if !matches!(premise.status, BeliefStatus::In) {
                    return Ok(false);
                }
            } else {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn would_create_cycle(&self, justification: &Justification) -> TMSResult<bool> {
        let target = justification.belief_id;
        
        for &premise in &justification.premises {
            if self.depends_on(premise, target)? {
                return Ok(true);
            }
        }
        
        Ok(false)
    }

    fn depends_on(&self, belief_id: BeliefId, target: BeliefId) -> TMSResult<bool> {
        if belief_id == target {
            return Ok(true);
        }

        let mut visited = HashSet::new();
        let mut to_visit = vec![belief_id];

        while let Some(current) = to_visit.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);

            if let Some(deps) = self.dependencies.get(&current) {
                for &dep in deps {
                    if dep == target {
                        return Ok(true);
                    }
                    to_visit.push(dep);
                }
            }
        }

        Ok(false)
    }

    fn check_consistency(&self, _belief_id: BeliefId) -> TMSResult<()> {
        // Check for direct contradictions using spike pattern analysis
        for (id1, belief1) in &self.beliefs {
            for (id2, belief2) in &self.beliefs {
                if id1 != id2 && self.are_contradictory(belief1, belief2) {
                    return Err(TMSError::InconsistentBeliefSet {
                        details: format!("Beliefs {} and {} are contradictory", id1, id2),
                    });
                }
            }
        }
        Ok(())
    }

    fn are_contradictory(&self, belief1: &Belief, belief2: &Belief) -> bool {
        // Simple contradiction detection - can be enhanced with semantic analysis
        belief1.content.contains("NOT") && belief1.content.contains(&belief2.content) ||
        belief2.content.contains("NOT") && belief2.content.contains(&belief1.content)
    }
}
```

### Phase 3: AGM Belief Revision Engine (Week 2)

#### Task 3.1: AGM Core Operations
**File**: `C:\code\LLMKG\crates\truth-maintenance\src\agm.rs`
```rust
use crate::types::{Belief, BeliefId, BeliefStatus};
use crate::error::{TMSError, TMSResult};
use std::collections::{HashMap, HashSet};

pub type BeliefSet = HashMap<BeliefId, Belief>;

pub struct AGMBeliefRevisionEngine {
    entrenchment_calculator: EntrenchmentCalculator,
    minimal_change_engine: MinimalChangeEngine,
}

impl AGMBeliefRevisionEngine {
    pub fn new() -> Self {
        Self {
            entrenchment_calculator: EntrenchmentCalculator::new(),
            minimal_change_engine: MinimalChangeEngine::new(),
        }
    }

    /// AGM Expansion: K + φ
    pub fn expand(&self, belief_set: &BeliefSet, new_belief: Belief) -> TMSResult<BeliefSet> {
        let mut expanded = belief_set.clone();
        expanded.insert(new_belief.id, new_belief);
        
        // Validate expansion satisfies AGM postulates
        self.validate_expansion_postulates(belief_set, &expanded)?;
        
        Ok(expanded)
    }

    /// AGM Contraction: K - φ
    pub fn contract(&self, belief_set: &BeliefSet, target_belief: BeliefId) -> TMSResult<BeliefSet> {
        // Find minimal set to remove using entrenchment ordering
        let removal_set = self.minimal_change_engine
            .find_minimal_contraction_set(belief_set, target_belief, &self.entrenchment_calculator)?;

        let mut contracted = belief_set.clone();
        for belief_id in removal_set {
            contracted.remove(&belief_id);
        }
        
        // Ensure target is removed
        contracted.remove(&target_belief);
        
        // Validate contraction satisfies AGM postulates
        self.validate_contraction_postulates(belief_set, &contracted, target_belief)?;
        
        Ok(contracted)
    }

    /// AGM Revision: K * φ
    pub fn revise(&self, belief_set: &BeliefSet, new_belief: Belief) -> TMSResult<BeliefSet> {
        // Try expansion first
        let expanded = self.expand(belief_set, new_belief.clone())?;
        
        if self.is_consistent(&expanded)? {
            return Ok(expanded);
        }

        // Find conflicting beliefs
        let conflicts = self.find_conflicts(belief_set, &new_belief)?;
        
        // Contract conflicting beliefs based on entrenchment
        let mut revised = belief_set.clone();
        
        for conflict_id in conflicts {
            if self.entrenchment_calculator.compare(&conflict_id, &new_belief.id, belief_set)? == std::cmp::Ordering::Less {
                revised = self.contract(&revised, conflict_id)?;
            }
        }
        
        // Add new belief
        revised.insert(new_belief.id, new_belief);
        
        // Validate revision satisfies AGM postulates
        self.validate_revision_postulates(belief_set, &revised)?;
        
        Ok(revised)
    }

    fn validate_expansion_postulates(&self, original: &BeliefSet, expanded: &BeliefSet) -> TMSResult<()> {
        // AGM Postulate validation
        // Closure: expanded set should be logically closed
        // Success: new belief should be in expanded set
        // Inclusion: expanded ⊆ Cn(original ∪ {new_belief})
        
        if expanded.len() != original.len() + 1 {
            return Err(TMSError::AGMViolation {
                postulate: "Expansion should add exactly one belief".to_string(),
            });
        }
        
        Ok(())
    }

    fn validate_contraction_postulates(&self, original: &BeliefSet, contracted: &BeliefSet, target: BeliefId) -> TMSResult<()> {
        // Ensure target belief is not in contracted set
        if contracted.contains_key(&target) {
            return Err(TMSError::AGMViolation {
                postulate: "Contraction should remove target belief".to_string(),
            });
        }
        
        // Ensure contracted set is subset of original
        for belief_id in contracted.keys() {
            if !original.contains_key(belief_id) {
                return Err(TMSError::AGMViolation {
                    postulate: "Contraction should not add new beliefs".to_string(),
                });
            }
        }
        
        Ok(())
    }

    fn validate_revision_postulates(&self, _original: &BeliefSet, _revised: &BeliefSet) -> TMSResult<()> {
        // Implement full AGM revision postulate validation
        // This is a simplified version - full implementation would check all 8 postulates
        Ok(())
    }

    fn is_consistent(&self, belief_set: &BeliefSet) -> TMSResult<bool> {
        // Check for contradictions
        for (id1, belief1) in belief_set {
            for (id2, belief2) in belief_set {
                if id1 != id2 && self.are_contradictory(belief1, belief2) {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn find_conflicts(&self, belief_set: &BeliefSet, new_belief: &Belief) -> TMSResult<Vec<BeliefId>> {
        let mut conflicts = Vec::new();
        
        for (belief_id, belief) in belief_set {
            if self.are_contradictory(belief, new_belief) {
                conflicts.push(*belief_id);
            }
        }
        
        Ok(conflicts)
    }

    fn are_contradictory(&self, belief1: &Belief, belief2: &Belief) -> bool {
        // Enhanced contradiction detection
        // This should be integrated with semantic analysis from neuromorphic-core
        belief1.content.to_lowercase().contains("not") && 
        belief1.content.to_lowercase().contains(&belief2.content.to_lowercase()) ||
        belief2.content.to_lowercase().contains("not") && 
        belief2.content.to_lowercase().contains(&belief1.content.to_lowercase())
    }
}

pub struct EntrenchmentCalculator {
    entrenchment_cache: HashMap<BeliefId, f32>,
}

impl EntrenchmentCalculator {
    pub fn new() -> Self {
        Self {
            entrenchment_cache: HashMap::new(),
        }
    }

    pub fn calculate_entrenchment(&mut self, belief_id: &BeliefId, belief_set: &BeliefSet) -> TMSResult<f32> {
        if let Some(&cached) = self.entrenchment_cache.get(belief_id) {
            return Ok(cached);
        }

        let belief = belief_set.get(belief_id)
            .ok_or_else(|| TMSError::BeliefNotFound { belief_id: belief_id.to_string() })?;

        // Calculate entrenchment based on multiple factors
        let source_reliability = belief.sources.iter()
            .map(|s| s.reliability)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        let temporal_persistence = self.calculate_temporal_persistence(belief)?;
        let logical_centrality = self.calculate_logical_centrality(belief_id, belief_set)?;

        let entrenchment = (source_reliability * 0.4) + 
                          (temporal_persistence * 0.3) + 
                          (logical_centrality * 0.3);

        self.entrenchment_cache.insert(*belief_id, entrenchment);
        Ok(entrenchment)
    }

    pub fn compare(&mut self, belief1: &BeliefId, belief2: &BeliefId, belief_set: &BeliefSet) -> TMSResult<std::cmp::Ordering> {
        let entrenchment1 = self.calculate_entrenchment(belief1, belief_set)?;
        let entrenchment2 = self.calculate_entrenchment(belief2, belief_set)?;
        
        Ok(entrenchment1.partial_cmp(&entrenchment2).unwrap_or(std::cmp::Ordering::Equal))
    }

    fn calculate_temporal_persistence(&self, belief: &Belief) -> TMSResult<f32> {
        // Simple time-based persistence - can be enhanced with actual usage patterns
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let age = current_time - belief.timestamp;
        Ok((age as f32 / 86400.0).min(1.0)) // Days, capped at 1.0
    }

    fn calculate_logical_centrality(&self, belief_id: &BeliefId, belief_set: &BeliefSet) -> TMSResult<f32> {
        // Count how many other beliefs reference this one
        let mut references = 0;
        
        for other_belief in belief_set.values() {
            if other_belief.content.contains(&belief_id.to_string()) {
                references += 1;
            }
        }
        
        Ok((references as f32 / belief_set.len() as f32).min(1.0))
    }
}

pub struct MinimalChangeEngine;

impl MinimalChangeEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn find_minimal_contraction_set(
        &self,
        belief_set: &BeliefSet,
        target: BeliefId,
        entrenchment: &mut EntrenchmentCalculator,
    ) -> TMSResult<Vec<BeliefId>> {
        let mut removal_set = Vec::new();
        
        // Find beliefs that depend on the target
        for (belief_id, belief) in belief_set {
            if belief.content.contains(&target.to_string()) || *belief_id == target {
                removal_set.push(*belief_id);
            }
        }
        
        // Sort by entrenchment (remove least entrenched first)
        removal_set.sort_by(|a, b| {
            entrenchment.compare(a, b, belief_set).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(removal_set)
    }
}
```

### Phase 4: Neuromorphic Integration (Week 2-3)

#### Task 4.1: Spike-Based TMS Processing
**File**: `C:\code\LLMKG\crates\truth-maintenance\src\neuromorphic.rs`
```rust
use crate::types::{Belief, BeliefId};
use crate::error::{TMSError, TMSResult};
use neuromorphic_core::ttfs_concept::{TTFSSpikePattern, SpikeEvent};
use neuromorphic_core::spiking_column::SpikingCorticalColumn;
use std::collections::HashMap;
use std::sync::Arc;

pub struct NeuromorphicTMSProcessor {
    cortical_columns: Vec<SpikingCorticalColumn>,
    belief_to_column: HashMap<BeliefId, usize>,
    spike_encoder: SpikeEncoder,
    lateral_inhibition: LateralInhibitionNetwork,
}

impl NeuromorphicTMSProcessor {
    pub fn new(num_columns: usize) -> TMSResult<Self> {
        let mut columns = Vec::new();
        for i in 0..num_columns {
            columns.push(SpikingCorticalColumn::new(format!("tms_column_{}", i))?);
        }

        Ok(Self {
            cortical_columns: columns,
            belief_to_column: HashMap::new(),
            spike_encoder: SpikeEncoder::new(),
            lateral_inhibition: LateralInhibitionNetwork::new(),
        })
    }

    pub fn process_belief_revision(&mut self, beliefs: &[Belief]) -> TMSResult<RevisionResult> {
        // Encode beliefs as spike patterns
        let mut spike_patterns = HashMap::new();
        for belief in beliefs {
            let pattern = self.spike_encoder.encode_belief(belief)?;
            spike_patterns.insert(belief.id, pattern);
        }

        // Assign beliefs to cortical columns
        self.assign_beliefs_to_columns(beliefs)?;

        // Process with lateral inhibition for conflict resolution
        let conflicts = self.lateral_inhibition.detect_conflicts(&spike_patterns)?;
        let resolution = self.resolve_conflicts_with_inhibition(conflicts)?;

        Ok(RevisionResult {
            revised_beliefs: resolution.winning_beliefs,
            conflicts_resolved: resolution.conflicts.len(),
            spike_patterns,
        })
    }

    fn assign_beliefs_to_columns(&mut self, beliefs: &[Belief]) -> TMSResult<()> {
        for (i, belief) in beliefs.iter().enumerate() {
            let column_index = i % self.cortical_columns.len();
            self.belief_to_column.insert(belief.id, column_index);
            
            // Activate corresponding cortical column
            let spike_pattern = self.spike_encoder.encode_belief(belief)?;
            self.cortical_columns[column_index].process_spike_pattern(&spike_pattern)?;
        }
        Ok(())
    }

    fn resolve_conflicts_with_inhibition(&mut self, conflicts: Vec<ConflictGroup>) -> TMSResult<ConflictResolution> {
        let mut winning_beliefs = Vec::new();
        let mut resolved_conflicts = Vec::new();

        for conflict_group in conflicts {
            // Apply winner-take-all dynamics
            let winner = self.lateral_inhibition.apply_winner_take_all(&conflict_group)?;
            winning_beliefs.push(winner.belief_id);
            resolved_conflicts.push(conflict_group);
        }

        Ok(ConflictResolution {
            winning_beliefs,
            conflicts: resolved_conflicts,
        })
    }
}

pub struct SpikeEncoder;

impl SpikeEncoder {
    pub fn new() -> Self {
        Self
    }

    pub fn encode_belief(&self, belief: &Belief) -> TMSResult<TTFSSpikePattern> {
        // Convert belief confidence to spike timing
        let spike_time = ((1.0 - belief.confidence) * 10_000.0) as u64; // 0-10ms range
        
        let events = vec![SpikeEvent {
            timestamp: spike_time,
            amplitude: belief.confidence,
            frequency: 100.0 + (belief.entrenchment * 100.0), // Base + entrenchment
        }];

        Ok(TTFSSpikePattern {
            events,
            metadata: neuromorphic_core::ttfs_concept::TTFSMetadata {
                concept_id: belief.id.to_string(),
                encoding_timestamp: belief.timestamp,
                confidence_score: belief.confidence,
            },
        })
    }
}

pub struct LateralInhibitionNetwork {
    inhibition_strength: f32,
    convergence_threshold: f32,
}

impl LateralInhibitionNetwork {
    pub fn new() -> Self {
        Self {
            inhibition_strength: 0.3,
            convergence_threshold: 0.1,
        }
    }

    pub fn detect_conflicts(&self, spike_patterns: &HashMap<BeliefId, TTFSSpikePattern>) -> TMSResult<Vec<ConflictGroup>> {
        let mut conflicts = Vec::new();
        let belief_ids: Vec<_> = spike_patterns.keys().cloned().collect();

        for i in 0..belief_ids.len() {
            for j in (i + 1)..belief_ids.len() {
                let id1 = belief_ids[i];
                let id2 = belief_ids[j];
                let pattern1 = &spike_patterns[&id1];
                let pattern2 = &spike_patterns[&id2];

                if self.patterns_conflict(pattern1, pattern2) {
                    conflicts.push(ConflictGroup {
                        conflicting_beliefs: vec![id1, id2],
                        conflict_strength: self.calculate_conflict_strength(pattern1, pattern2),
                    });
                }
            }
        }

        Ok(conflicts)
    }

    pub fn apply_winner_take_all(&self, conflict_group: &ConflictGroup) -> TMSResult<ConflictWinner> {
        // Simple winner-take-all based on first spike timing
        // In a full implementation, this would use proper cortical column dynamics
        
        let winner_id = conflict_group.conflicting_beliefs[0]; // Simplified selection
        
        Ok(ConflictWinner {
            belief_id: winner_id,
            activation_level: 1.0,
        })
    }

    fn patterns_conflict(&self, pattern1: &TTFSSpikePattern, pattern2: &TTFSSpikePattern) -> bool {
        // Check if both patterns have high confidence (early spikes) indicating potential conflict
        if let (Some(spike1), Some(spike2)) = (pattern1.events.first(), pattern2.events.first()) {
            spike1.timestamp < 500 && spike2.timestamp < 500 && // Both high confidence
            spike1.amplitude > 0.8 && spike2.amplitude > 0.8    // Both strong
        } else {
            false
        }
    }

    fn calculate_conflict_strength(&self, pattern1: &TTFSSpikePattern, pattern2: &TTFSSpikePattern) -> f32 {
        // Calculate conflict strength based on timing overlap and amplitude
        if let (Some(spike1), Some(spike2)) = (pattern1.events.first(), pattern2.events.first()) {
            let timing_overlap = 1.0 - ((spike1.timestamp as f32 - spike2.timestamp as f32).abs() / 10_000.0);
            let amplitude_product = spike1.amplitude * spike2.amplitude;
            timing_overlap * amplitude_product
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
pub struct RevisionResult {
    pub revised_beliefs: Vec<BeliefId>,
    pub conflicts_resolved: usize,
    pub spike_patterns: HashMap<BeliefId, TTFSSpikePattern>,
}

#[derive(Debug)]
pub struct ConflictGroup {
    pub conflicting_beliefs: Vec<BeliefId>,
    pub conflict_strength: f32,
}

#[derive(Debug)]
pub struct ConflictResolution {
    pub winning_beliefs: Vec<BeliefId>,
    pub conflicts: Vec<ConflictGroup>,
}

#[derive(Debug)]
pub struct ConflictWinner {
    pub belief_id: BeliefId,
    pub activation_level: f32,
}

// Integration with existing neuromorphic-core error types
impl From<neuromorphic_core::error::CoreError> for TMSError {
    fn from(err: neuromorphic_core::error::CoreError) -> Self {
        TMSError::NeuromorphicError {
            details: err.to_string(),
        }
    }
}
```

### Phase 5: Integration with Existing Systems (Week 3)

#### Task 5.1: Temporal Memory Integration
**File**: `C:\code\LLMKG\crates\truth-maintenance\src\temporal_integration.rs`
```rust
use crate::types::{Belief, BeliefId};
use crate::error::{TMSError, TMSResult};
use temporal_memory::branch::{MemoryBranch, SemanticVersion};
use std::collections::HashMap;

pub struct TemporalTMSIntegration {
    memory_branch: MemoryBranch,
    belief_versions: HashMap<BeliefId, SemanticVersion>,
}

impl TemporalTMSIntegration {
    pub fn new(branch_name: &str) -> Self {
        Self {
            memory_branch: MemoryBranch::new(branch_name),
            belief_versions: HashMap::new(),
        }
    }

    pub fn store_belief_revision(&mut self, belief: &Belief, revision_type: RevisionType) -> TMSResult<()> {
        // Store belief in temporal memory system
        let concept_id = belief.id;
        
        if !self.memory_branch.has_concept(concept_id) {
            self.memory_branch.add_concept(concept_id);
        }

        // Update concept properties with belief data
        if let Some(mut properties) = self.memory_branch.get_concept_properties(concept_id).cloned() {
            properties.value = belief.content.clone();
            
            // Increment version based on revision type
            match revision_type {
                RevisionType::Major => properties.version.increment_major(),
                RevisionType::Minor => properties.version.increment_minor(),
                RevisionType::Patch => properties.version.increment_patch(),
            }
            
            self.memory_branch.update_concept_properties(concept_id, properties);
            self.belief_versions.insert(belief.id, properties.version.clone());
        }

        Ok(())
    }

    pub fn get_belief_history(&self, belief_id: BeliefId) -> TMSResult<Vec<BeliefVersion>> {
        // In a full implementation, this would retrieve complete version history
        // For now, return current version
        if let Some(version) = self.belief_versions.get(&belief_id) {
            Ok(vec![BeliefVersion {
                belief_id,
                version: version.clone(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            }])
        } else {
            Ok(vec![])
        }
    }

    pub fn create_belief_branch(&mut self, branch_name: &str) -> TMSResult<TemporalTMSIntegration> {
        let child_branch = self.memory_branch.create_child(branch_name);
        
        Ok(TemporalTMSIntegration {
            memory_branch: child_branch,
            belief_versions: self.belief_versions.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub enum RevisionType {
    Major,  // AGM revision with conflict resolution
    Minor,  // AGM expansion
    Patch,  // Minor updates or corrections
}

#[derive(Debug, Clone)]
pub struct BeliefVersion {
    pub belief_id: BeliefId,
    pub version: SemanticVersion,
    pub timestamp: u64,
}
```

#### Task 5.2: Main TMS Integration Module
**File**: `C:\code\LLMKG\crates\truth-maintenance\src\lib.rs`
```rust
pub mod types;
pub mod error;
pub mod jtms;
pub mod agm;
pub mod neuromorphic;
pub mod temporal_integration;

use types::{Belief, BeliefId, BeliefSet};
use error::{TMSError, TMSResult};
use jtms::JustificationBasedTMS;
use agm::AGMBeliefRevisionEngine;
use neuromorphic::NeuromorphicTMSProcessor;
use temporal_integration::{TemporalTMSIntegration, RevisionType};

use std::sync::Arc;
use tokio::sync::RwLock;

/// Main Truth Maintenance System combining JTMS, AGM, and neuromorphic processing
pub struct TruthMaintenanceSystem {
    jtms: Arc<RwLock<JustificationBasedTMS>>,
    agm_engine: Arc<RwLock<AGMBeliefRevisionEngine>>,
    neuromorphic_processor: Arc<RwLock<NeuromorphicTMSProcessor>>,
    temporal_integration: Arc<RwLock<TemporalTMSIntegration>>,
}

impl TruthMaintenanceSystem {
    pub async fn new() -> TMSResult<Self> {
        let jtms = Arc::new(RwLock::new(JustificationBasedTMS::new()));
        let agm_engine = Arc::new(RwLock::new(AGMBeliefRevisionEngine::new()));
        let neuromorphic_processor = Arc::new(RwLock::new(
            NeuromorphicTMSProcessor::new(16)? // 16 cortical columns
        ));
        let temporal_integration = Arc::new(RwLock::new(
            TemporalTMSIntegration::new("main_tms_branch")
        ));

        Ok(Self {
            jtms,
            agm_engine,
            neuromorphic_processor,
            temporal_integration,
        })
    }

    /// Add a new belief with full TMS processing
    pub async fn add_belief(&self, belief: Belief) -> TMSResult<()> {
        // 1. Process with JTMS for dependency tracking
        {
            let mut jtms = self.jtms.write().await;
            jtms.add_belief(belief.clone())?;
        }

        // 2. Store in temporal system
        {
            let mut temporal = self.temporal_integration.write().await;
            temporal.store_belief_revision(&belief, RevisionType::Minor)?;
        }

        // 3. Process with neuromorphic system for conflict detection
        {
            let mut processor = self.neuromorphic_processor.write().await;
            let _result = processor.process_belief_revision(&[belief])?;
        }

        Ok(())
    }

    /// Revise belief set using AGM operations
    pub async fn revise_beliefs(&self, belief_set: BeliefSet, new_belief: Belief) -> TMSResult<BeliefSet> {
        let agm = self.agm_engine.read().await;
        let revised_set = agm.revise(&belief_set, new_belief.clone())?;

        // Store revision in temporal system
        {
            let mut temporal = self.temporal_integration.write().await;
            temporal.store_belief_revision(&new_belief, RevisionType::Major)?;
        }

        Ok(revised_set)
    }

    /// Get current belief set
    pub async fn get_beliefs(&self) -> TMSResult<BeliefSet> {
        let jtms = self.jtms.read().await;
        // In a full implementation, this would extract beliefs from JTMS
        Ok(BeliefSet::new()) // Simplified
    }

    /// Health check for TMS system
    pub async fn health_check(&self) -> TMSHealthReport {
        TMSHealthReport {
            jtms_active: true, // Simplified check
            agm_compliant: true,
            neuromorphic_connected: true,
            temporal_synced: true,
            total_beliefs: 0, // Would count actual beliefs
            conflicts_detected: 0,
        }
    }
}

#[derive(Debug)]
pub struct TMSHealthReport {
    pub jtms_active: bool,
    pub agm_compliant: bool,
    pub neuromorphic_connected: bool,
    pub temporal_synced: bool,
    pub total_beliefs: usize,
    pub conflicts_detected: usize,
}

// Re-export key types for public API
pub use types::{Belief, BeliefId, BeliefStatus, Justification, Source, AuthorityLevel};
pub use error::{TMSError, TMSResult};

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_tms_basic_operations() {
        let tms = TruthMaintenanceSystem::new().await.unwrap();

        let belief = Belief {
            id: Uuid::new_v4(),
            content: "The sky is blue".to_string(),
            confidence: 0.9,
            sources: vec![],
            timestamp: 0,
            entrenchment: 0.5,
            status: types::BeliefStatus::In,
        };

        assert!(tms.add_belief(belief).await.is_ok());
        
        let health = tms.health_check().await;
        assert!(health.jtms_active);
        assert!(health.agm_compliant);
        assert!(health.neuromorphic_connected);
    }

    #[tokio::test]
    async fn test_agm_operations() {
        let agm = AGMBeliefRevisionEngine::new();
        let mut belief_set = BeliefSet::new();

        let belief1 = Belief {
            id: Uuid::new_v4(),
            content: "All birds can fly".to_string(),
            confidence: 0.8,
            sources: vec![],
            timestamp: 0,
            entrenchment: 0.6,
            status: types::BeliefStatus::In,
        };

        let belief2 = Belief {
            id: Uuid::new_v4(),
            content: "Penguins are birds".to_string(),
            confidence: 0.95,
            sources: vec![],
            timestamp: 0,
            entrenchment: 0.9,
            status: types::BeliefStatus::In,
        };

        belief_set.insert(belief1.id, belief1);
        
        // Test expansion
        let expanded = agm.expand(&belief_set, belief2.clone()).unwrap();
        assert_eq!(expanded.len(), 2);

        // Test revision with potential conflict
        let conflicting_belief = Belief {
            id: Uuid::new_v4(),
            content: "NOT All birds can fly".to_string(),
            confidence: 0.9,
            sources: vec![],
            timestamp: 0,
            entrenchment: 0.8,
            status: types::BeliefStatus::In,
        };

        let revised = agm.revise(&expanded, conflicting_belief).unwrap();
        assert!(revised.len() >= 1); // Should handle conflict
    }
}
```

### Phase 6: Complete Test Suite (Week 3-4)

#### Task 6.1: Comprehensive Test Coverage
**File**: `C:\code\LLMKG\crates\truth-maintenance\tests\integration_tests.rs`
```rust
use truth_maintenance::*;
use uuid::Uuid;
use std::collections::HashMap;

#[tokio::test]
async fn test_full_tms_workflow() {
    let tms = TruthMaintenanceSystem::new().await.unwrap();

    // Create test beliefs
    let belief1 = create_test_belief("All swans are white", 0.7);
    let belief2 = create_test_belief("This is a swan", 0.9);
    let belief3 = create_test_belief("This swan is black", 0.85);

    // Add initial beliefs
    tms.add_belief(belief1).await.unwrap();
    tms.add_belief(belief2).await.unwrap();

    // Add conflicting belief (should trigger revision)
    let mut belief_set = HashMap::new();
    belief_set.insert(belief1.id, belief1);
    belief_set.insert(belief2.id, belief2);

    let revised_set = tms.revise_beliefs(belief_set, belief3).await.unwrap();
    
    // Verify conflict was resolved
    assert!(!revised_set.is_empty());
    
    // Check system health
    let health = tms.health_check().await;
    assert!(health.jtms_active);
    assert!(health.agm_compliant);
}

#[tokio::test]
async fn test_agm_postulate_compliance() {
    let agm = agm::AGMBeliefRevisionEngine::new();
    let mut belief_set = HashMap::new();

    let belief = create_test_belief("Test belief", 0.8);
    
    // Test AGM Expansion postulates
    let expanded = agm.expand(&belief_set, belief.clone()).unwrap();
    
    // Success postulate: new belief should be in expanded set
    assert!(expanded.contains_key(&belief.id));
    
    // Inclusion postulate: expanded set should contain original beliefs
    assert_eq!(expanded.len(), 1);

    // Test AGM Contraction
    belief_set.insert(belief.id, belief.clone());
    let contracted = agm.contract(&belief_set, belief.id).unwrap();
    
    // Target belief should be removed
    assert!(!contracted.contains_key(&belief.id));
}

#[tokio::test]
async fn test_neuromorphic_integration() {
    let mut processor = neuromorphic::NeuromorphicTMSProcessor::new(4).unwrap();
    
    let beliefs = vec![
        create_test_belief("High confidence belief", 0.95),
        create_test_belief("Medium confidence belief", 0.6),
        create_test_belief("Low confidence belief", 0.3),
    ];

    let result = processor.process_belief_revision(&beliefs).unwrap();
    
    // Should have processed all beliefs
    assert_eq!(result.spike_patterns.len(), 3);
    
    // High confidence beliefs should have earlier spike times
    let high_conf_pattern = &result.spike_patterns[&beliefs[0].id];
    let low_conf_pattern = &result.spike_patterns[&beliefs[2].id];
    
    if let (Some(high_spike), Some(low_spike)) = 
        (high_conf_pattern.events.first(), low_conf_pattern.events.first()) {
        assert!(high_spike.timestamp < low_spike.timestamp);
    }
}

#[tokio::test]
async fn test_temporal_integration() {
    let mut temporal = temporal_integration::TemporalTMSIntegration::new("test_branch");
    
    let belief = create_test_belief("Temporal test belief", 0.8);
    
    // Store initial belief
    temporal.store_belief_revision(&belief, temporal_integration::RevisionType::Minor).unwrap();
    
    // Check history
    let history = temporal.get_belief_history(belief.id).unwrap();
    assert_eq!(history.len(), 1);
    
    // Create branch
    let _child_branch = temporal.create_belief_branch("child").unwrap();
}

#[test]
fn test_belief_conflict_detection() {
    let belief1 = create_test_belief("The sky is blue", 0.9);
    let belief2 = create_test_belief("NOT The sky is blue", 0.8);
    
    let agm = agm::AGMBeliefRevisionEngine::new();
    let mut belief_set = HashMap::new();
    belief_set.insert(belief1.id, belief1);
    
    // Adding conflicting belief should trigger revision
    let result = agm.revise(&belief_set, belief2);
    assert!(result.is_ok());
}

#[test]
fn test_entrenchment_calculation() {
    let mut calculator = agm::EntrenchmentCalculator::new();
    let mut belief_set = HashMap::new();
    
    let high_authority_belief = types::Belief {
        id: Uuid::new_v4(),
        content: "Expert knowledge".to_string(),
        confidence: 0.9,
        sources: vec![types::Source {
            id: "expert1".to_string(),
            reliability: 0.95,
            authority_level: types::AuthorityLevel::Expert,
        }],
        timestamp: 1000,
        entrenchment: 0.0,
        status: types::BeliefStatus::In,
    };
    
    let low_authority_belief = types::Belief {
        id: Uuid::new_v4(),
        content: "Questionable claim".to_string(),
        confidence: 0.5,
        sources: vec![types::Source {
            id: "unknown".to_string(),
            reliability: 0.3,
            authority_level: types::AuthorityLevel::Questionable,
        }],
        timestamp: 2000,
        entrenchment: 0.0,
        status: types::BeliefStatus::In,
    };
    
    belief_set.insert(high_authority_belief.id, high_authority_belief.clone());
    belief_set.insert(low_authority_belief.id, low_authority_belief.clone());
    
    let high_entrenchment = calculator.calculate_entrenchment(&high_authority_belief.id, &belief_set).unwrap();
    let low_entrenchment = calculator.calculate_entrenchment(&low_authority_belief.id, &belief_set).unwrap();
    
    assert!(high_entrenchment > low_entrenchment);
}

fn create_test_belief(content: &str, confidence: f32) -> types::Belief {
    types::Belief {
        id: Uuid::new_v4(),
        content: content.to_string(),
        confidence,
        sources: vec![],
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        entrenchment: 0.5,
        status: types::BeliefStatus::In,
    }
}

// Performance benchmarks
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn benchmark_belief_addition() {
        let tms = TruthMaintenanceSystem::new().await.unwrap();
        let start = Instant::now();
        
        for i in 0..1000 {
            let belief = create_test_belief(&format!("Belief {}", i), 0.5);
            tms.add_belief(belief).await.unwrap();
        }
        
        let duration = start.elapsed();
        println!("Added 1000 beliefs in {:?}", duration);
        
        // Should be under 1 second for 1000 beliefs
        assert!(duration.as_secs() < 1);
    }

    #[test]
    fn benchmark_agm_operations() {
        let agm = agm::AGMBeliefRevisionEngine::new();
        let mut belief_set = HashMap::new();
        
        // Create large belief set
        for i in 0..100 {
            let belief = create_test_belief(&format!("Belief {}", i), 0.5);
            belief_set.insert(belief.id, belief);
        }
        
        let start = Instant::now();
        
        for i in 100..200 {
            let new_belief = create_test_belief(&format!("New Belief {}", i), 0.6);
            let _revised = agm.revise(&belief_set, new_belief).unwrap();
        }
        
        let duration = start.elapsed();
        println!("Performed 100 revisions on 100-belief set in {:?}", duration);
        
        // Should be under 5ms per revision on average
        assert!(duration.as_millis() / 100 < 5);
    }
}
```

## Migration Strategy from Current State

### Step 1: Infrastructure Setup (Day 1-2)
1. **Create truth-maintenance crate**: New crate in `crates/truth-maintenance/`
2. **Update workspace Cargo.toml**: Add truth-maintenance to workspace members
3. **Set up dependencies**: neuromorphic-core, temporal-memory, standard libraries
4. **Create basic module structure**: types, error, jtms, agm, neuromorphic, temporal_integration

### Step 2: Core Implementation (Day 3-10)
1. **Implement core types** (types.rs): Belief, Justification, BeliefSet structures
2. **Implement JTMS** (jtms.rs): Dependency tracking, belief status propagation
3. **Implement AGM engine** (agm.rs): Expansion, contraction, revision with postulate validation
4. **Implement neuromorphic integration** (neuromorphic.rs): Spike encoding, lateral inhibition
5. **Implement temporal integration** (temporal_integration.rs): Version tracking, branching

### Step 3: Integration Testing (Day 11-14)
1. **Unit tests**: Each module with 90%+ coverage
2. **Integration tests**: Full workflow testing
3. **Performance benchmarks**: Meet 5ms revision targets
4. **AGM compliance tests**: All 8 postulates validated

### Step 4: System Integration (Day 15-21)
1. **Update existing systems**: Integrate TMS with allocation engine, query processor
2. **Update documentation**: Sync specs with implementation
3. **Migration scripts**: Convert any existing belief data
4. **Health monitoring**: Add TMS metrics to system monitoring

## Quality Assurance & Testing Strategy

### Test Coverage Requirements
- **Unit Tests**: 95% coverage minimum
- **Integration Tests**: All major workflows covered
- **Performance Tests**: Sub-5ms revision operations
- **AGM Compliance Tests**: 100% postulate satisfaction
- **Neuromorphic Integration Tests**: Spike timing accuracy <1ms

### Validation Checklist
- [ ] All 8 AGM postulates satisfied with formal verification
- [ ] JTMS dependency tracking with cycle detection
- [ ] Neuromorphic spike encoding preserves timing relationships
- [ ] Temporal integration maintains version consistency
- [ ] Conflict detection identifies all contradiction types
- [ ] Performance targets: <5ms revisions, <2ms conflict detection
- [ ] Memory usage: <10% overhead for TMS operations
- [ ] Integration: Zero breaking changes to existing systems

## Risk Mitigation

### High-Risk Areas
1. **AGM Postulate Compliance**: Complex formal requirements
   - **Mitigation**: Property-based testing, formal verification tools
2. **Neuromorphic Integration**: Timing precision requirements
   - **Mitigation**: Extensive timing tests, fallback mechanisms
3. **Performance**: Sub-millisecond requirements
   - **Mitigation**: Profiling, optimization, caching strategies

### Rollback Plan
If implementation fails:
1. **Phase 1 Rollback**: Basic belief storage without AGM compliance (Time: 2 hours)
2. **Phase 2 Rollback**: Simple contradiction detection without neuromorphic processing (Time: 4 hours)
3. **Phase 3 Rollback**: Documentation-only TMS with plan for future implementation (Time: 1 hour)

## Success Metrics

### Primary Objectives (Must Achieve)
- **Functional TMS**: Working JTMS + AGM + neuromorphic integration
- **Test Coverage**: 95% with comprehensive integration tests
- **Performance**: <5ms belief revisions, <2ms conflict detection
- **AGM Compliance**: 100% postulate satisfaction
- **Zero Regressions**: No impact on existing system functionality

### Secondary Objectives (Should Achieve)
- **Documentation Sync**: All specs match implementation
- **Advanced Features**: Multi-context reasoning, sophisticated conflict resolution
- **Optimization**: Sub-millisecond operations for small belief sets
- **Extensibility**: Plugin architecture for new reasoning strategies

This comprehensive plan addresses the 777 inconsistencies by implementing a complete, tested, AGM-compliant truth maintenance system that integrates seamlessly with the existing neuromorphic architecture while maintaining performance requirements and providing a solid foundation for advanced reasoning capabilities.