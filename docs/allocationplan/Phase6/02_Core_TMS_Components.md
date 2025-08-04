# Phase 6.2: Core TMS Components Implementation

**Duration**: 4-5 hours  
**Complexity**: High  
**Dependencies**: Phase 6.1 Foundation Setup

## Micro-Tasks Overview

This phase implements the core hybrid JTMS-ATMS architecture with neuromorphic integration.

---

## Task 6.2.1: Implement Justification-Based TMS (JTMS) Layer

**Estimated Time**: 75 minutes  
**Complexity**: High  
**AI Task**: Create the JTMS core implementation

**Prompt for AI:**
```
Create `src/truth_maintenance/jtms.rs` implementing JustificationBasedTMS:
1. Implement the spiking dependency graph for belief networks
2. Create belief state management with spike patterns
3. Implement dependency-directed backtracking for conflict resolution
4. Add belief propagation using spreading activation
5. Integrate with neuromorphic spike timing patterns

Key components:
- SpikingDependencyGraph for belief relationships
- BeliefNode management with IN/OUT states
- Justification strength via synaptic weights
- Propagation algorithms using spike timing
- Minimal culprit set calculation for contradictions

Performance requirements:
- Propagation latency <3ms for typical networks
- Support for >10,000 belief nodes
- Memory efficient belief state representation

Code Example from existing codebase pattern (Phase 1 CorticalColumn):
```rust
// Similar atomic state management from Phase 1:
pub struct CorticalColumn {
    id: ColumnId,
    state: AtomicColumnState,
    created_at: SystemTime,
    last_transition: RwLock<SystemTime>,
    transition_count: AtomicU64,
}

impl CorticalColumn {
    pub fn try_transition_to(&self, target: ColumnState) -> Result<(), StateTransitionError> {
        let current = self.current_state();
        self.state.try_transition(current, target)?;
        
        // Update metadata
        *self.last_transition.write() = SystemTime::now();
        self.transition_count.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
}
```

Expected implementation for JTMS:
```rust
// src/truth_maintenance/jtms.rs - Enhanced with concrete neuromorphic spike patterns
use crate::types::{BeliefId, BeliefNode, BeliefStatus, JustificationId, Justification, SpikePattern};
use crate::errors::{TMSError, RevisionError};
use crate::config::TMSConfig;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

/// TTFS (Time-To-First-Spike) belief confidence encoding
/// Maps belief confidence to spike timing: high confidence = early spike
#[derive(Debug, Clone)]
pub struct TTFSBeliefEncoder {
    /// Range of spike times (min for max confidence, max for zero confidence)
    pub spike_time_range: (Duration, Duration), // (100μs, 10ms)
}

impl TTFSBeliefEncoder {
    pub fn new(config: &TMSConfig) -> Self {
        let (min_time, max_time) = config.ttfs_range();
        Self {
            spike_time_range: (min_time, max_time),
        }
    }
    
    /// Encode belief confidence as TTFS spike pattern
    /// Example: confidence=0.9 → spike_time≈190μs, confidence=0.1 → spike_time≈9.01ms
    pub fn encode_belief_confidence(&self, confidence: f64) -> TTFSPattern {
        let (min_time, max_time) = self.spike_time_range;
        let range_duration = max_time - min_time;
        
        // Higher confidence = earlier spike time (shorter TTFS)
        let spike_time = min_time + range_duration.mul_f64(1.0 - confidence);
        
        TTFSPattern {
            first_spike_time: spike_time,
            confidence_level: confidence,
            spike_intensity: (confidence * 255.0) as u8,
            neural_population: 1, // Single neuron for simple beliefs
        }
    }
    
    /// Decode TTFS pattern back to confidence level
    pub fn decode_spike_confidence(&self, pattern: &TTFSPattern) -> f64 {
        let (min_time, max_time) = self.spike_time_range;
        let range_duration = max_time - min_time;
        
        if range_duration.is_zero() {
            return 1.0; // Default to max confidence if no range
        }
        
        let spike_offset = pattern.first_spike_time.saturating_sub(min_time);
        let normalized_offset = spike_offset.as_secs_f64() / range_duration.as_secs_f64();
        
        1.0 - normalized_offset.clamp(0.0, 1.0)
    }
}

/// TTFS spike pattern for belief representation
#[derive(Debug, Clone)]
pub struct TTFSPattern {
    pub first_spike_time: Duration,
    pub confidence_level: f64,
    pub spike_intensity: u8,
    pub neural_population: u32,
}

/// Lateral inhibition network for conflict resolution
#[derive(Debug)]
pub struct LateralInhibitionProcessor {
    /// Inhibitory connections between competing beliefs
    inhibitory_connections: Arc<RwLock<HashMap<BeliefId, Vec<InhibitoryLink>>>>,
    /// Strength of lateral inhibition
    inhibition_strength: f64,
    /// Decay time constant for inhibition
    decay_time_constant: Duration,
}

/// Inhibitory connection between two competing beliefs
#[derive(Debug, Clone)]
pub struct InhibitoryLink {
    pub target_belief: BeliefId,
    pub inhibition_strength: f64,
    pub synaptic_delay: Duration,
    pub last_activation: Option<SystemTime>,
}

impl LateralInhibitionProcessor {
    pub fn new(config: &TMSConfig) -> Self {
        Self {
            inhibitory_connections: Arc::new(RwLock::new(HashMap::new())),
            inhibition_strength: config.lateral_inhibition_strength,
            decay_time_constant: Duration::from_millis(config.inhibition_decay_time_ms),
        }
    }
    
    /// Apply lateral inhibition for conflicting beliefs
    /// Example: If belief_a and belief_b conflict, suppress the weaker one
    pub async fn apply_lateral_inhibition(
        &self,
        belief_a: &BeliefNode,
        belief_b: &BeliefNode,
    ) -> Result<WinnerTakeAllResult, TMSError> {
        // Determine which belief should win based on confidence and evidence
        let winner = if belief_a.confidence > belief_b.confidence {
            belief_a.clone()
        } else if belief_b.confidence > belief_a.confidence {
            belief_b.clone()
        } else {
            // Equal confidence - use additional criteria
            if belief_a.justifications.len() >= belief_b.justifications.len() {
                belief_a.clone()
            } else {
                belief_b.clone()
            }
        };
        
        let loser = if winner.id == belief_a.id {
            belief_b.clone()
        } else {
            belief_a.clone()
        };
        
        // Create inhibitory connection from winner to loser
        let inhibitory_link = InhibitoryLink {
            target_belief: loser.id,
            inhibition_strength: self.inhibition_strength,
            synaptic_delay: Duration::from_micros(100), // 100μs synaptic delay
            last_activation: Some(SystemTime::now()),
        };
        
        // Record the inhibitory connection
        {
            let mut connections = self.inhibitory_connections.write().await;
            connections.entry(winner.id)
                .or_insert_with(Vec::new)
                .push(inhibitory_link);
        }
        
        debug!(
            "Lateral inhibition: belief {} (confidence: {:.3}) suppresses belief {} (confidence: {:.3})",
            winner.id, winner.confidence, loser.id, loser.confidence
        );
        
        Ok(WinnerTakeAllResult {
            winning_belief: winner.id,
            suppressed_beliefs: vec![loser.id],
            inhibition_strength: self.inhibition_strength,
            resolution_time: SystemTime::now(),
        })
    }
    
    /// Process winner-take-all dynamics for multiple competing beliefs
    pub async fn winner_take_all(
        &self,
        competing_beliefs: &[BeliefNode],
    ) -> Result<WinnerTakeAllResult, TMSError> {
        if competing_beliefs.is_empty() {
            return Err(TMSError::Integration("No competing beliefs provided".to_string()));
        }
        
        if competing_beliefs.len() == 1 {
            return Ok(WinnerTakeAllResult {
                winning_belief: competing_beliefs[0].id,
                suppressed_beliefs: Vec::new(),
                inhibition_strength: 0.0,
                resolution_time: SystemTime::now(),
            });
        }
        
        // Find the belief with highest confidence (winner)
        let winner = competing_beliefs.iter()
            .max_by(|a, b| a.confidence.partial_cmp(&b.confidence)
                           .unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        
        // Suppress all other beliefs
        let suppressed_beliefs: Vec<BeliefId> = competing_beliefs.iter()
            .filter(|belief| belief.id != winner.id)
            .map(|belief| belief.id)
            .collect();
        
        // Create inhibitory connections from winner to all losers
        {
            let mut connections = self.inhibitory_connections.write().await;
            let winner_connections = connections.entry(winner.id).or_insert_with(Vec::new);
            
            for &suppressed_id in &suppressed_beliefs {
                winner_connections.push(InhibitoryLink {
                    target_belief: suppressed_id,
                    inhibition_strength: self.inhibition_strength,
                    synaptic_delay: Duration::from_micros(100),
                    last_activation: Some(SystemTime::now()),
                });
            }
        }
        
        info!(
            "Winner-take-all: belief {} wins over {} competitors with confidence {:.3}",
            winner.id, suppressed_beliefs.len(), winner.confidence
        );
        
        Ok(WinnerTakeAllResult {
            winning_belief: winner.id,
            suppressed_beliefs,
            inhibition_strength: self.inhibition_strength,
            resolution_time: SystemTime::now(),
        })
    }
}

/// Result of winner-take-all competition
#[derive(Debug, Clone)]
pub struct WinnerTakeAllResult {
    pub winning_belief: BeliefId,
    pub suppressed_beliefs: Vec<BeliefId>,
    pub inhibition_strength: f64,
    pub resolution_time: SystemTime,
}

#[derive(Debug)]
pub struct JustificationBasedTMS {
    config: Arc<TMSConfig>,
    beliefs: Arc<RwLock<HashMap<BeliefId, BeliefNode>>>,
    justifications: Arc<RwLock<HashMap<JustificationId, Justification>>>,
    dependency_graph: Arc<Mutex<SpikingDependencyGraph>>,
    ttfs_encoder: TTFSBeliefEncoder,
    inhibition_processor: LateralInhibitionProcessor,
    cortical_columns: Arc<RwLock<HashMap<u32, CorticalColumnState>>>,
    metrics: Arc<super::metrics::TMSHealthMetrics>,
}

/// State of a cortical column processing beliefs
#[derive(Debug, Clone)]
pub struct CorticalColumnState {
    pub column_id: u32,
    pub current_belief: Option<BeliefId>,
    pub activation_level: f64,
    pub last_spike_time: Option<SystemTime>,
    pub refractory_until: Option<SystemTime>,
    pub inhibitory_input: f64,
}

impl JustificationBasedTMS {
    pub async fn new(config: Arc<TMSConfig>) -> Result<Self, TMSError> {
        info!("Initializing Justification-Based TMS with neuromorphic integration");
        
        // Initialize cortical columns
        let mut cortical_columns = HashMap::new();
        for i in 0..config.cortical_column_count {
            cortical_columns.insert(i as u32, CorticalColumnState {
                column_id: i as u32,
                current_belief: None,
                activation_level: 0.0,
                last_spike_time: None,
                refractory_until: None,
                inhibitory_input: 0.0,
            });
        }
        
        Ok(Self {
            config: config.clone(),
            beliefs: Arc::new(RwLock::new(HashMap::new())),
            justifications: Arc::new(RwLock::new(HashMap::new())),
            dependency_graph: Arc::new(Mutex::new(SpikingDependencyGraph::new())),
            ttfs_encoder: TTFSBeliefEncoder::new(&config),
            inhibition_processor: LateralInhibitionProcessor::new(&config),
            cortical_columns: Arc::new(RwLock::new(cortical_columns)),
            metrics: Arc::new(super::metrics::TMSHealthMetrics::new()),
        })
    }
    
    /// Add a belief to the TMS with neuromorphic spike pattern processing
    pub async fn add_belief(&self, mut belief: BeliefNode) -> Result<(), TMSError> {
        let start_time = Instant::now();
        
        // Generate TTFS spike pattern for belief confidence
        let ttfs_pattern = self.ttfs_encoder.encode_belief_confidence(belief.confidence);
        
        // Validate and enhance spike pattern
        belief.spike_pattern = self.enhance_spike_pattern(
            &belief.spike_pattern, 
            &ttfs_pattern,
        )?;
        
        // Assign to cortical column based on content hash and confidence
        let column_id = self.assign_cortical_column(&belief).await?;
        
        // Check for conflicts with existing beliefs
        let conflicts = self.detect_spike_pattern_conflicts(&belief).await?;
        
        if !conflicts.is_empty() {
            // Apply lateral inhibition for conflict resolution
            let mut competing_beliefs = vec![belief.clone()];
            for conflict_id in conflicts {
                if let Some(conflict_belief) = self.get_belief(conflict_id).await? {
                    competing_beliefs.push(conflict_belief);
                }
            }
            
            let winner_result = self.inhibition_processor
                .winner_take_all(&competing_beliefs).await?;
            
            // Update metrics
            self.metrics.record_lateral_inhibition(
                winner_result.winning_belief.into(),
                &winner_result.suppressed_beliefs.iter()
                    .map(|id| u32::from(*id))
                    .collect::<Vec<_>>()
            );
            
            if winner_result.winning_belief != belief.id {
                // This belief was suppressed
                belief.status = BeliefStatus::OUT;
                warn!("Belief {} suppressed by lateral inhibition", belief.id);
            } else {
                belief.status = BeliefStatus::IN;
                info!("Belief {} won lateral inhibition competition", belief.id);
            }
        } else {
            // No conflicts - determine status based on justifications
            belief.status = self.determine_initial_status(&belief).await?;
        }
        
        // Update cortical column state
        self.update_cortical_column(column_id, &belief).await?;
        
        // Update data structures
        {
            let mut beliefs = self.beliefs.write().unwrap();
            beliefs.insert(belief.id, belief.clone());
        }
        
        // Update dependency graph with enhanced spike timing
        {
            let mut graph = self.dependency_graph.lock().await;
            graph.add_node(belief.id, belief.spike_pattern.clone());
        }
        
        // Propagate belief status changes using spike timing
        self.propagate_with_spike_timing(belief.id).await?;
        
        // Record performance metrics
        self.metrics.record_revision(start_time.elapsed());
        self.metrics.record_ttfs_encoding(
            start_time.elapsed(),
            self.ttfs_encoder.decode_spike_confidence(&ttfs_pattern)
        );
        
        debug!(
            "Added belief {} with status {:?}, confidence {:.3}, column {}",
            belief.id, belief.status, belief.confidence, column_id
        );
        
        Ok(())
    }
    
    /// Enhance spike pattern with TTFS encoding
    fn enhance_spike_pattern(
        &self,
        original: &SpikePattern,
        ttfs_pattern: &TTFSPattern,
    ) -> Result<SpikePattern, TMSError> {
        let mut enhanced = original.clone();
        
        // Add TTFS timing to spike pattern
        enhanced.ttfs_values.insert(0, ttfs_pattern.first_spike_time.as_secs_f64() * 1000.0);
        enhanced.strength = ttfs_pattern.confidence_level;
        
        // Ensure temporal consistency
        enhanced.ttfs_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(enhanced)
    }
    
    /// Assign belief to cortical column based on content and confidence
    async fn assign_cortical_column(&self, belief: &BeliefNode) -> Result<u32, TMSError> {
        let content_hash = self.hash_belief_content(&belief.content);
        let confidence_factor = (belief.confidence * 1000.0) as u32;
        
        // Combine content hash and confidence to select column
        let preferred_column = (content_hash + confidence_factor) % self.config.cortical_column_count as u32;
        
        // Find available column near preferred location
        let columns = self.cortical_columns.read().await;
        
        // Check if preferred column is available (not in refractory period)
        if let Some(column_state) = columns.get(&preferred_column) {
            if self.is_column_available(column_state) {
                return Ok(preferred_column);
            }
        }
        
        // Find next available column
        for offset in 1..self.config.cortical_column_count {
            let candidate = (preferred_column + offset as u32) % self.config.cortical_column_count as u32;
            if let Some(column_state) = columns.get(&candidate) {
                if self.is_column_available(column_state) {
                    return Ok(candidate);
                }
            }
        }
        
        // All columns busy - use preferred anyway (will handle competition)
        Ok(preferred_column)
    }
    
    /// Check if cortical column is available for new belief
    fn is_column_available(&self, column_state: &CorticalColumnState) -> bool {
        let now = SystemTime::now();
        
        // Check refractory period
        if let Some(refractory_until) = column_state.refractory_until {
            if now < refractory_until {
                return false;
            }
        }
        
        // Check if already processing a belief
        column_state.current_belief.is_none()
    }
    
    /// Update cortical column state with new belief
    async fn update_cortical_column(
        &self,
        column_id: u32,
        belief: &BeliefNode,
    ) -> Result<(), TMSError> {
        let mut columns = self.cortical_columns.write().await;
        
        if let Some(column_state) = columns.get_mut(&column_id) {
            column_state.current_belief = Some(belief.id);
            column_state.activation_level = belief.confidence;
            column_state.last_spike_time = Some(SystemTime::now());
            
            // Set refractory period
            let refractory_duration = self.config.refractory_period();
            column_state.refractory_until = Some(
                SystemTime::now() + refractory_duration
            );
        }
        
        Ok(())
    }
    
    /// Simple hash function for belief content
    fn hash_belief_content(&self, content: &str) -> u32 {
        // Simple hash - in production, use a proper hash function
        content.bytes().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32))
    }
    
    /// Propagate status changes using spike timing dynamics
    async fn propagate_with_spike_timing(&self, belief_id: BeliefId) -> Result<(), TMSError> {
        let mut propagation_queue = VecDeque::new();
        let mut visited = HashSet::new();
        
        propagation_queue.push_back(belief_id);
        
        while let Some(current_id) = propagation_queue.pop_front() {
            if visited.contains(&current_id) {
                continue;
            }
            visited.insert(current_id);
            
            // Get dependent beliefs with spike timing delays
            let dependents_with_timing = {
                let graph = self.dependency_graph.lock().await;
                graph.get_dependents_with_timing(current_id)
            };
            
            for (dependent_id, propagation_delay) in dependents_with_timing {
                // Schedule propagation based on spike timing
                tokio::time::sleep(propagation_delay).await;
                
                if self.update_belief_status_with_timing(dependent_id).await? {
                    propagation_queue.push_back(dependent_id);
                }
            }
        }
        
        Ok(())
    }
    
    /// Update belief status based on justifications with spike timing
    async fn update_belief_status_with_timing(&self, belief_id: BeliefId) -> Result<bool, TMSError> {
        let justifications = self.get_belief_justifications(belief_id).await;
        let new_status = self.compute_status_from_spike_patterns(&justifications).await?;
        
        let mut beliefs = self.beliefs.write().unwrap();
        if let Some(belief) = beliefs.get_mut(&belief_id) {
            let changed = belief.status != new_status;
            
            if changed {
                belief.status = new_status;
                belief.last_updated = SystemTime::now();
                
                // Update spike pattern to reflect new status
                let new_confidence = match new_status {
                    BeliefStatus::IN => belief.confidence.max(0.7), // Boost confidence when IN
                    BeliefStatus::OUT => belief.confidence.min(0.3), // Reduce when OUT
                    BeliefStatus::UNKNOWN => 0.5, // Neutral confidence
                };
                
                let updated_ttfs = self.ttfs_encoder.encode_belief_confidence(new_confidence);
                belief.spike_pattern = self.enhance_spike_pattern(
                    &belief.spike_pattern,
                    &updated_ttfs,
                )?;
                
                debug!(
                    "Updated belief {} status to {:?} with confidence {:.3}",
                    belief_id, new_status, new_confidence
                );
            }
            
            Ok(changed)
        } else {
            Ok(false)
        }
    }
    
    /// Compute belief status from spike patterns of justifications
    async fn compute_status_from_spike_patterns(
        &self,
        justifications: &[Justification],
    ) -> Result<BeliefStatus, TMSError> {
        if justifications.is_empty() {
            return Ok(BeliefStatus::UNKNOWN);
        }
        
        let mut total_strength = 0.0;
        let mut supporting_strength = 0.0;
        
        for justification in justifications {
            let strength = justification.spike_encoding.synaptic_weight;
            total_strength += strength;
            
            // Check if justification supports or opposes the belief
            if strength > 0.5 {
                supporting_strength += strength;
            }
        }
        
        if total_strength == 0.0 {
            return Ok(BeliefStatus::UNKNOWN);
        }
        
        let support_ratio = supporting_strength / total_strength;
        
        if support_ratio > 0.7 {
            Ok(BeliefStatus::IN)
        } else if support_ratio < 0.3 {
            Ok(BeliefStatus::OUT)
        } else {
            Ok(BeliefStatus::UNKNOWN)
        }
    }
    
    /// Detect spike pattern conflicts with existing beliefs
    async fn detect_spike_pattern_conflicts(
        &self,
        new_belief: &BeliefNode,
    ) -> Result<Vec<BeliefId>, TMSError> {
        let beliefs = self.beliefs.read().unwrap();
        let mut conflicts = Vec::new();
        
        for (existing_id, existing_belief) in beliefs.iter() {
            if existing_id == &new_belief.id {
                continue;
            }
            
            // Check for spike pattern conflicts
            if new_belief.spike_pattern.conflicts_with(&existing_belief.spike_pattern) {
                // Additional semantic conflict check
                if self.beliefs_semantically_conflict(new_belief, existing_belief) {
                    conflicts.push(*existing_id);
                }
            }
        }
        
        Ok(conflicts)
    }
    
    /// Check if two beliefs semantically conflict
    fn beliefs_semantically_conflict(
        &self,
        belief_a: &BeliefNode,
        belief_b: &BeliefNode,
    ) -> bool {
        // Simple heuristic: beliefs with similar content but different truth values
        // In production, this would use more sophisticated semantic analysis
        
        let content_a = belief_a.content.to_lowercase();
        let content_b = belief_b.content.to_lowercase();
        
        // Check for negation patterns
        let negation_patterns = ["not ", "no ", "isn't ", "aren't ", "won't ", "can't "];
        
        for pattern in &negation_patterns {
            if (content_a.contains(pattern) && !content_b.contains(pattern)) ||
               (!content_a.contains(pattern) && content_b.contains(pattern)) {
                // Check if they're about the same subject
                let subject_a = self.extract_subject(&content_a);
                let subject_b = self.extract_subject(&content_b);
                
                if self.subjects_similar(&subject_a, &subject_b) {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// Extract subject from belief content (simplified)
    fn extract_subject(&self, content: &str) -> String {
        // Very simple subject extraction - first few words
        content.split_whitespace()
            .take(3)
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    /// Check if two subjects are similar
    fn subjects_similar(&self, subject_a: &str, subject_b: &str) -> bool {
        // Simple similarity check - edit distance or common words
        let words_a: HashSet<&str> = subject_a.split_whitespace().collect();
        let words_b: HashSet<&str> = subject_b.split_whitespace().collect();
        
        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();
        
        if union == 0 {
            return false;
        }
        
        let similarity = intersection as f64 / union as f64;
        similarity > 0.5 // Threshold for similarity
    }
    
    /// Get belief by ID
    async fn get_belief(&self, belief_id: BeliefId) -> Result<Option<BeliefNode>, TMSError> {
        let beliefs = self.beliefs.read().unwrap();
        Ok(beliefs.get(&belief_id).cloned())
    }
}

/// Spiking dependency graph for neuromorphic belief relationships
#[derive(Debug)]
pub struct SpikingDependencyGraph {
    nodes: HashMap<BeliefId, NodeData>,
    edges: HashMap<BeliefId, Vec<(BeliefId, Duration)>>, // adjacency with delays
    reverse_edges: HashMap<BeliefId, Vec<(BeliefId, Duration)>>, // reverse adjacency
    spike_propagation_speed: f64, // m/s - speed of spike propagation
}

#[derive(Debug, Clone)]
struct NodeData {
    spike_pattern: SpikePattern,
    last_spike_time: f64,
    activation_level: f64,
}

impl SpikingDependencyGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            reverse_edges: HashMap::new(),
            spike_propagation_speed: 1.0, // 1 m/s for neural computation
        }
    }
    
    pub fn add_node(&mut self, belief_id: BeliefId, spike_pattern: SpikePattern) {
        self.nodes.insert(belief_id, NodeData {
            spike_pattern,
            last_spike_time: 0.0,
            activation_level: 0.0,
        });
        self.edges.entry(belief_id).or_insert_with(Vec::new);
        self.reverse_edges.entry(belief_id).or_insert_with(Vec::new);
    }
    
    pub fn add_edge(&mut self, from: BeliefId, to: BeliefId, weight: f64) {
        // Calculate propagation delay based on synaptic strength
        // Stronger connections have shorter delays
        let base_delay = Duration::from_micros(100); // 100μs base synaptic delay
        let weight_factor = 1.0 / weight.max(0.1); // Inverse relationship
        let propagation_delay = base_delay.mul_f64(weight_factor);
        
        self.edges.entry(from).or_default().push((to, propagation_delay));
        self.reverse_edges.entry(to).or_default().push((from, propagation_delay));
    }
    
    pub fn get_dependents(&self, belief_id: BeliefId) -> Vec<BeliefId> {
        self.edges.get(&belief_id)
            .map(|deps| deps.iter().map(|(id, _)| *id).collect())
            .unwrap_or_default()
    }
    
    pub fn get_dependents_with_timing(&self, belief_id: BeliefId) -> Vec<(BeliefId, Duration)> {
        self.edges.get(&belief_id).cloned().unwrap_or_default()
    }
    
    /// Propagate activation using spike timing dynamics
    pub fn propagate_activation(&mut self, from: BeliefId, current_time: f64) -> Vec<BeliefId> {
        let mut activated = Vec::new();
        
        if let Some(dependents) = self.edges.get(&from).cloned() {
            for dependent_id in dependents {
                if let Some(node_data) = self.nodes.get_mut(&dependent_id) {
                    // Calculate spike-based activation
                    let spike_delay = self.calculate_spike_delay(&node_data.spike_pattern);
                    let arrival_time = current_time + spike_delay;
                    
                    // Update activation level
                    node_data.activation_level += node_data.spike_pattern.strength;
                    node_data.last_spike_time = arrival_time;
                    
                    // Check if threshold is reached
                    if node_data.activation_level > 0.7 { // threshold
                        activated.push(dependent_id);
                    }
                }
            }
        }
        
        activated
    }
    
    fn calculate_spike_delay(&self, pattern: &SpikePattern) -> f64 {
        // Use first spike time as propagation delay
        pattern.spikes.get(0).copied().unwrap_or(1.0)
    }
}
```
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small: 100 beliefs, 50 justifications, targets 1ms operations
- Medium: 1,000 beliefs, 500 justifications, targets 3ms operations  
- Large: 10,000 beliefs, 5,000 justifications, targets 5ms operations
- Stress: 100,000 beliefs, 50,000 justifications, validates scalability

**Validation Scenarios:**
1. Happy path: Well-formed belief networks with valid dependency chains
2. Edge cases: Empty networks, single beliefs, circular dependency detection
3. Error cases: Invalid spike patterns, corrupted dependencies, memory exhaustion
4. Performance: Belief networks sized to test latency/throughput targets

**Synthetic Data Generator:**
```rust
pub fn generate_jtms_test_data(scale: TestScale, seed: u64) -> JTMSTestDataSet {
    let mut rng = StdRng::seed_from_u64(seed);
    JTMSTestDataSet {
        beliefs: generate_belief_network(scale.belief_count, &mut rng),
        justifications: generate_justification_chains(scale.justification_count, &mut rng),
        spike_patterns: generate_neuromorphic_patterns(scale.pattern_count, &mut rng),
        dependency_graphs: generate_dependency_structures(scale.complexity, &mut rng),
    }
}
```

**Performance Benchmarking:**
- Tool: criterion.rs with 100 iterations
- Metrics: p50/p95/p99 latency, operations/second
- Duration: 30 seconds per benchmark
- Statistical: confidence intervals, regression detection

**Success Criteria:**
- JTMS maintains belief dependencies with 100% accuracy across dependency chains of up to 1000 nodes
- Propagation algorithms complete in <3ms for networks with 10,000 belief nodes (measured via benchmarks)
- Dependency-directed backtracking resolves 100% of detectable contradictions within 10ms
- Spike pattern timing preservation validated with <1ms deviation from original patterns across 1000 test cases
- Memory usage scales linearly with O(n) complexity, using <100MB for 10,000 beliefs (measured via memory profiler)

**Error Recovery Procedures:**
1. **Dependency Graph Corruption**:
   - Detect: Inconsistent dependency chains or missing belief references during graph traversal
   - Action: Rebuild dependency graph from stored justifications and validate all relationships
   - Retry: Implement incremental graph repair starting from leaf nodes working toward roots

2. **Performance Target Failures**:
   - Detect: Propagation latency exceeds 3ms or backtracking takes >10ms during benchmarks
   - Action: Implement simplified propagation algorithm with optimized data structures
   - Retry: Add caching layer for frequently accessed beliefs and parallel processing for independent chains

3. **Memory Exhaustion**:
   - Detect: Memory usage exceeds 100MB for 10,000 beliefs or non-linear growth patterns
   - Action: Implement belief garbage collection and compress inactive belief representations
   - Retry: Use memory-mapped storage for large belief sets and lazy loading for unused beliefs

**Rollback Procedure:**
- Time limit: 10 minutes maximum rollback time
- Steps: [1] disable JTMS layer completely [2] implement minimal belief storage without dependencies [3] add basic propagation logic
- Validation: Verify basic belief operations work without dependency tracking and system remains stable

---

## Task 6.2.2: Implement Assumption-Based TMS (ATMS) Layer

**Estimated Time**: 70 minutes  
**Complexity**: High  
**AI Task**: Create the ATMS multi-context system

**Prompt for AI:**
```
Create `src/truth_maintenance/atms.rs` implementing AssumptionBasedTMS:
1. Implement NeuralContext management with parallel processing
2. Create assumption set handling with cortical column assignment
3. Implement context splitting for inconsistency resolution
4. Add parallel context reasoning with async processing
5. Integrate with neuromorphic cortical architecture

Key components:
- Multiple contexts as parallel spike trains
- Assumption sets with neuromorphic encoding
- Context consistency validation
- Automatic context splitting on contradictions
- Result aggregation across contexts

Performance requirements:
- Context switch time <1ms
- Support for >100 parallel contexts
- Consistent result aggregation
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small: 10 contexts, 20 assumptions, targets 0.5ms operations
- Medium: 100 contexts, 200 assumptions, targets 1ms operations  
- Large: 1,000 contexts, 2,000 assumptions, targets 3ms operations
- Stress: 10,000 contexts, 20,000 assumptions, validates scalability

**Validation Scenarios:**
1. Happy path: Consistent assumption sets with valid parallel contexts
2. Edge cases: Empty contexts, single assumptions, context splitting scenarios
3. Error cases: Inconsistent assumptions, context corruption, parallel access conflicts
4. Performance: Context sets sized to test switching/processing targets

**Synthetic Data Generator:**
```rust
pub fn generate_atms_test_data(scale: TestScale, seed: u64) -> ATMSTestDataSet {
    let mut rng = StdRng::seed_from_u64(seed);
    ATMSTestDataSet {
        contexts: generate_parallel_contexts(scale.context_count, &mut rng),
        assumptions: generate_assumption_sets(scale.assumption_count, &mut rng),
        consistency_checks: generate_consistency_scenarios(scale.complexity, &mut rng),
        workload_patterns: generate_parallel_workloads(scale.parallelism, &mut rng),
    }
}
```

**Performance Benchmarking:**
- Tool: criterion.rs with 100 iterations
- Metrics: p50/p95/p99 latency, operations/second
- Duration: 30 seconds per benchmark
- Statistical: confidence intervals, regression detection

**Success Criteria:**
- ATMS maintains logical consistency across >100 parallel contexts with <0.1% context corruption rate
- Context switching completes within <1ms target (measured via high-resolution timers)
- Automatic context splitting resolves 100% of detectable inconsistencies within 5ms
- Parallel processing achieves >80% CPU core utilization scaling with 4-16 cores (measured via system monitors)
- Result aggregation maintains logical consistency with 100% validation success across 10,000 test cases

**Error Recovery Procedures:**
1. **Context Corruption**:
   - Detect: Context validation fails or corruption rate exceeds 0.1% during consistency checks
   - Action: Implement context checkpointing and rollback to last known good state
   - Retry: Rebuild corrupted contexts from base assumptions and re-derive beliefs step by step

2. **Context Switch Performance Issues**:
   - Detect: Context switching latency exceeds 1ms target measured via high-resolution timers
   - Action: Implement context pre-loading and copy-on-write strategies to reduce switch overhead
   - Retry: Use lock-free data structures and minimize context state that needs copying

3. **Parallel Processing Failures**:
   - Detect: CPU utilization falls below 80% or context splitting fails to resolve inconsistencies
   - Action: Implement dynamic work stealing and adjust parallelism based on workload characteristics
   - Retry: Add context affinity tracking and reduce inter-context communication overhead

**Rollback Procedure:**
- Time limit: 8 minutes maximum rollback time
- Steps: [1] disable parallel context processing [2] implement single-context ATMS mode [3] add basic context management
- Validation: Verify single-context operations work correctly and can handle basic assumption-based reasoning

---

## Task 6.2.3: Create Spiking Dependency Graph

**Estimated Time**: 60 minutes  
**Complexity**: High  
**AI Task**: Implement neuromorphic dependency tracking

**Prompt for AI:**
```
Create `src/truth_maintenance/spiking_dependency.rs`:
1. Implement graph structure using spike timing relationships
2. Create propagation algorithms with temporal dynamics
3. Add cycle detection using spike trace analysis
4. Implement graph traversal with spike patterns
5. Integrate with TTFS encoding from neuromorphic system

Key features:
- Nodes represent beliefs with spike patterns
- Edges represent dependencies with synaptic weights
- Propagation follows spike timing dynamics
- Cycle detection prevents infinite loops
- Integration with existing TTFS encoding

Technical requirements:
- Support for directed acyclic belief graphs
- Efficient cycle detection algorithms
- Spike pattern preservation during propagation
- Memory efficient adjacency representation
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small: 100 nodes, 200 edges, targets 0.5ms operations
- Medium: 1,000 nodes, 2,000 edges, targets 1ms operations  
- Large: 10,000 nodes, 20,000 edges, targets 2ms operations
- Stress: 100,000 nodes, 200,000 edges, validates scalability

**Validation Scenarios:**
1. Happy path: Acyclic dependency graphs with valid spike patterns
2. Edge cases: Empty graphs, single nodes, complex cyclic structures
3. Error cases: Corrupted edges, invalid spike timings, memory constraints
4. Performance: Graph structures sized to test traversal/detection targets

**Synthetic Data Generator:**
```rust
pub fn generate_dependency_test_data(scale: TestScale, seed: u64) -> DependencyTestDataSet {
    let mut rng = StdRng::seed_from_u64(seed);
    DependencyTestDataSet {
        graphs: generate_spiking_graphs(scale.node_count, scale.edge_count, &mut rng),
        spike_sequences: generate_neuromorphic_sequences(scale.sequence_count, &mut rng),
        cycle_scenarios: generate_cyclic_patterns(scale.complexity, &mut rng),
        timing_patterns: generate_spike_timing_data(scale.pattern_count, &mut rng),
    }
}
```

**Performance Benchmarking:**
- Tool: criterion.rs with 100 iterations
- Metrics: p50/p95/p99 latency, operations/second
- Duration: 30 seconds per benchmark
- Statistical: confidence intervals, regression detection

**Success Criteria:**
- Dependency graph represents 100% of belief relationships with zero missing or incorrect edges (validated via graph traversal)
- Propagation timing matches neuromorphic patterns within 5% deviation measured across 1000 spike sequences
- Cycle detection identifies 100% of circular dependencies in <2ms for graphs with up to 5000 nodes
- Performance scales sub-linearly with O(n log n) complexity for graphs up to 50,000 nodes (benchmarked)
- Spike timing preservation shows <1ms drift from original patterns over 1000 propagation cycles

**Error Recovery Procedures:**
1. **Graph Structure Corruption**:
   - Detect: Missing edges, incorrect relationships, or graph traversal inconsistencies
   - Action: Implement graph validation and reconstruction from stored edge lists
   - Retry: Rebuild graph incrementally with edge verification at each step

2. **Spike Timing Drift**:
   - Detect: Propagation timing deviates >5% from neuromorphic patterns or drift exceeds 1ms
   - Action: Implement spike timing calibration and synchronization mechanisms
   - Retry: Use hardware-accurate timing simulation and compensate for accumulated drift

3. **Cycle Detection Performance**:
   - Detect: Cycle detection takes >2ms for graphs with 5000 nodes or fails to detect cycles
   - Action: Implement optimized cycle detection using Tarjan's algorithm with early termination
   - Retry: Add cycle prevention during graph construction and maintain cycle-free invariants

**Rollback Procedure:**
- Time limit: 7 minutes maximum rollback time
- Steps: [1] disable spiking features and use basic graph representation [2] implement simple dependency tracking [3] add basic cycle detection
- Validation: Verify graph operations work without neuromorphic features and basic dependencies are maintained

---

## Task 6.2.4: Implement Belief State Management

**Estimated Time**: 50 minutes  
**Complexity**: Medium  
**AI Task**: Create belief state tracking system

**Prompt for AI:**
```
Create `src/truth_maintenance/belief_state.rs`:
1. Implement BeliefState enum with IN/OUT/UNKNOWN states
2. Create state transition logic with spike-based triggers
3. Add confidence tracking using spike frequency
4. Implement state persistence and recovery
5. Integrate with temporal versioning system

State management features:
- Atomic state transitions
- Confidence-weighted state determination
- State history tracking
- Recovery from inconsistent states
- Integration with belief versioning

Performance requirements:
- State transitions <0.5ms
- Concurrent state access without contention
- Memory efficient state representation
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small: 100 belief states, 50 transitions, targets 0.1ms operations
- Medium: 1,000 belief states, 500 transitions, targets 0.3ms operations  
- Large: 10,000 belief states, 5,000 transitions, targets 0.5ms operations
- Stress: 100,000 belief states, 50,000 transitions, validates scalability

**Validation Scenarios:**
1. Happy path: Valid AGM-compliant state transitions with spike evidence
2. Edge cases: Unknown states, simultaneous transitions, state conflicts
3. Error cases: Invalid transitions, corrupted states, concurrent access violations
4. Performance: State sets sized to test transition/persistence targets

**Synthetic Data Generator:**
```rust
pub fn generate_belief_state_test_data(scale: TestScale, seed: u64) -> BeliefStateTestDataSet {
    let mut rng = StdRng::seed_from_u64(seed);
    BeliefStateTestDataSet {
        states: generate_belief_state_sequences(scale.state_count, &mut rng),
        transitions: generate_agm_transitions(scale.transition_count, &mut rng),
        confidence_data: generate_spike_evidence(scale.evidence_count, &mut rng),
        concurrent_scenarios: generate_thread_access_patterns(scale.thread_count, &mut rng),
    }
}
```

**Performance Benchmarking:**
- Tool: criterion.rs with 100 iterations
- Metrics: p50/p95/p99 latency, operations/second
- Duration: 30 seconds per benchmark
- Statistical: confidence intervals, regression detection

**Success Criteria:**
- Belief state transitions follow AGM rules with 100% correctness across 10,000 justification scenarios
- Confidence tracking correlates with spike evidence at >95% accuracy (measured via statistical analysis)
- State persistence recovers 100% of belief states correctly after simulated system failures
- Concurrent access by >100 threads maintains consistency with zero state corruption (validated by stress testing)
- State transitions complete in <0.5ms (measured via microsecond-precision timers)

**Error Recovery Procedures:**
1. **AGM Rule Violations**:
   - Detect: State transitions that violate AGM postulates during correctness validation
   - Action: Implement AGM rule enforcement with validation before each state change
   - Retry: Rollback invalid transitions and re-apply using compliant AGM operations

2. **Confidence Correlation Failures**:
   - Detect: Confidence tracking accuracy falls below 95% correlation with spike evidence
   - Action: Implement confidence recalibration using statistical learning from spike patterns
   - Retry: Add confidence smoothing and outlier detection to improve correlation accuracy

3. **Concurrent Access Corruption**:
   - Detect: State corruption during concurrent access by multiple threads
   - Action: Implement optimistic locking with conflict detection and retry mechanisms
   - Retry: Use atomic operations and lock-free data structures for high-frequency state updates

**Rollback Procedure:**
- Time limit: 6 minutes maximum rollback time
- Steps: [1] disable concurrent access and use single-threaded state management [2] implement basic belief states without AGM compliance [3] add simple confidence tracking
- Validation: Verify basic belief state operations work in single-threaded mode and states persist correctly

---

## Task 6.2.5: Create Context Management System

**Estimated Time**: 65 minutes  
**Complexity**: High  
**AI Task**: Implement multi-context reasoning framework

**Prompt for AI:**
```
Create `src/truth_maintenance/context_manager.rs`:
1. Implement ContextManager for parallel context handling
2. Create context creation and lifecycle management
3. Add context compatibility checking
4. Implement context merging and splitting algorithms
5. Integrate with cortical column assignment

Context management features:
- Dynamic context creation based on assumptions
- Automatic context lifecycle management
- Compatibility analysis for assumption sets
- Efficient context switching mechanisms
- Resource management for context overhead

Technical requirements:
- Support for hierarchical context relationships
- Efficient context comparison algorithms
- Memory pooling for context reuse
- Thread-safe context operations
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small: 10 contexts, 5 hierarchies, targets 0.5ms operations
- Medium: 100 contexts, 20 hierarchies, targets 1ms operations  
- Large: 1,000 contexts, 100 hierarchies, targets 2ms operations
- Stress: 10,000 contexts, 500 hierarchies, validates scalability

**Validation Scenarios:**
1. Happy path: Well-formed context hierarchies with compatible assumptions
2. Edge cases: Empty contexts, deep hierarchies, context merging scenarios
3. Error cases: Incompatible contexts, memory exhaustion, lifecycle failures
4. Performance: Context sets sized to test management/switching targets

**Synthetic Data Generator:**
```rust
pub fn generate_context_test_data(scale: TestScale, seed: u64) -> ContextTestDataSet {
    let mut rng = StdRng::seed_from_u64(seed);
    ContextTestDataSet {
        contexts: generate_context_hierarchies(scale.context_count, &mut rng),
        assumptions: generate_assumption_compatibility_sets(scale.assumption_count, &mut rng),
        lifecycle_scenarios: generate_context_lifecycle_patterns(scale.scenario_count, &mut rng),
        switching_patterns: generate_context_switch_workloads(scale.switch_count, &mut rng),
    }
}
```

**Performance Benchmarking:**
- Tool: criterion.rs with 100 iterations
- Metrics: p50/p95/p99 latency, operations/second
- Duration: 30 seconds per benchmark
- Statistical: confidence intervals, regression detection

**Success Criteria:**
- Context manager successfully handles >100 parallel contexts with <2% management overhead
- Context lifecycle shows zero memory leaks over 24-hour stress test (validated via memory monitoring)
- Compatibility checking identifies conflicts with >99% accuracy and <1% false positive rate
- Context switching achieves <1ms latency target with 95th percentile <2ms (measured via latency distribution)
- Memory overhead remains <10% of total system memory usage for context management operations

**Error Recovery Procedures:**
1. **Context Management Overhead**:
   - Detect: Management overhead exceeds 2% or context operations become bottleneck
   - Action: Implement context pooling and lazy context initialization to reduce overhead
   - Retry: Use lock-free context queues and batch context operations to improve efficiency

2. **Memory Leaks**:
   - Detect: Memory monitoring shows growth over 24-hour stress test or context cleanup failures
   - Action: Implement reference counting and automatic context garbage collection
   - Retry: Add memory profiling hooks and debug leaked context references systematically

3. **Compatibility Check Failures**:
   - Detect: Conflict detection accuracy below 99% or false positive rate above 1%
   - Action: Implement statistical conflict analysis and machine learning-based conflict prediction
   - Retry: Add conflict pattern recognition and improve checking algorithms based on false positive analysis

**Rollback Procedure:**
- Time limit: 9 minutes maximum rollback time
- Steps: [1] disable parallel context management [2] implement basic single-context operations [3] add simple context switching without optimization
- Validation: Verify basic context operations work without parallel processing and no memory leaks in simple scenarios

---

## Task 6.2.6: Implement TMS Integration Layer

**Estimated Time**: 45 minutes  
**Complexity**: Medium  
**AI Task**: Create unified TMS interface

**Prompt for AI:**
```
Create `src/truth_maintenance/integration.rs`:
1. Implement unified interface combining JTMS and ATMS
2. Create routing logic for choosing appropriate TMS layer
3. Add result coordination between TMS layers
4. Implement fallback mechanisms for layer failures
5. Integrate with neuromorphic processing pipeline

Integration features:
- Transparent switching between JTMS and ATMS
- Coordination for hybrid reasoning tasks
- Performance monitoring for both layers
- Error handling and recovery mechanisms
- Seamless neuromorphic integration

Performance requirements:
- Routing decisions <0.1ms
- Layer coordination overhead <5%
- Fallback switching <2ms
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small: 5 TMS layers, 20 routing scenarios, targets 0.1ms operations
- Medium: 10 TMS layers, 100 routing scenarios, targets 0.5ms operations  
- Large: 20 TMS layers, 500 routing scenarios, targets 1ms operations
- Stress: 50 TMS layers, 2000 routing scenarios, validates scalability

**Validation Scenarios:**
1. Happy path: Optimal routing with consistent results across layers
2. Edge cases: Layer failures, conflicting results, routing ambiguity
3. Error cases: Complete layer failure, coordination failures, API breaks
4. Performance: Workloads sized to test routing/coordination targets

**Synthetic Data Generator:**
```rust
pub fn generate_integration_test_data(scale: TestScale, seed: u64) -> IntegrationTestDataSet {
    let mut rng = StdRng::seed_from_u64(seed);
    IntegrationTestDataSet {
        routing_scenarios: generate_workload_patterns(scale.scenario_count, &mut rng),
        layer_configurations: generate_tms_layer_setups(scale.layer_count, &mut rng),
        failure_modes: generate_failure_scenarios(scale.failure_count, &mut rng),
        coordination_tests: generate_result_coordination_cases(scale.coordination_count, &mut rng),
    }
}
```

**Performance Benchmarking:**
- Tool: criterion.rs with 100 iterations
- Metrics: p50/p95/p99 latency, operations/second
- Duration: 30 seconds per benchmark
- Statistical: confidence intervals, regression detection

**Success Criteria:**
- Integration layer exposes 100% of TMS functionality through unified API with zero breaking changes
- Routing selects optimal TMS layer with >98% accuracy based on workload characteristics
- Result coordination maintains logical consistency across layers with 100% validation success
- Fallback mechanisms handle 100% of failure modes with <5ms recovery time
- Performance overhead <5% compared to direct layer access (measured via comparative benchmarks)

**Error Recovery Procedures:**
1. **API Compatibility Issues**:
   - Detect: Breaking changes in unified API or functionality gaps compared to individual layers
   - Action: Implement API versioning and backward compatibility adapters
   - Retry: Add gradual migration path and deprecation warnings for API changes

2. **Routing Accuracy Problems**:
   - Detect: Workload routing accuracy falls below 98% or incorrect layer selection
   - Action: Implement machine learning-based routing with workload pattern recognition
   - Retry: Add routing decision logging and improve routing algorithms based on actual performance data

3. **Result Coordination Failures**:
   - Detect: Logical inconsistency across layers or validation failures during result coordination
   - Action: Implement conflict detection and resolution mechanisms for inter-layer results
   - Retry: Add result verification checkpoints and automatic retry with alternative routing

**Rollback Procedure:**
- Time limit: 7 minutes maximum rollback time
- Steps: [1] disable unified API and expose layers directly [2] implement basic routing without optimization [3] add simple result coordination
- Validation: Verify individual TMS layers work independently and basic integration functionality operates correctly

---

## Task 6.2.7: Create TMS Factory and Builder Pattern

**Estimated Time**: 35 minutes  
**Complexity**: Low  
**AI Task**: Implement TMS construction and configuration

**Prompt for AI:**
```
Create `src/truth_maintenance/factory.rs`:
1. Implement TMSBuilder with fluent configuration interface
2. Create factory methods for different TMS configurations
3. Add validation for TMS configuration combinations
4. Implement async initialization with proper error handling
5. Integrate with dependency injection framework

Builder features:
- Fluent configuration interface
- Validation of configuration constraints
- Support for custom component injection
- Async initialization with progress tracking
- Integration with existing factory patterns

Technical requirements:
- Type-safe configuration building
- Comprehensive validation before construction
- Resource initialization ordering
- Error recovery during initialization
```

**Test Data Specification:**

**Benchmark Dataset:**
- Small: 10 configurations, 5 build patterns, targets 0.1s operations
- Medium: 50 configurations, 20 build patterns, targets 0.5s operations  
- Large: 200 configurations, 100 build patterns, targets 1s operations
- Stress: 1000 configurations, 500 build patterns, validates scalability

**Validation Scenarios:**
1. Happy path: Valid configurations with successful TMS instantiation
2. Edge cases: Minimal configurations, complex configurations, builder chaining
3. Error cases: Invalid configurations, initialization failures, timeout scenarios
4. Performance: Configuration sets sized to test validation/initialization targets

**Synthetic Data Generator:**
```rust
pub fn generate_factory_test_data(scale: TestScale, seed: u64) -> FactoryTestDataSet {
    let mut rng = StdRng::seed_from_u64(seed);
    FactoryTestDataSet {
        configurations: generate_tms_configurations(scale.config_count, &mut rng),
        build_patterns: generate_builder_scenarios(scale.pattern_count, &mut rng),
        validation_cases: generate_constraint_violations(scale.violation_count, &mut rng),
        initialization_scenarios: generate_async_init_patterns(scale.init_count, &mut rng),
    }
}
```

**Performance Benchmarking:**
- Tool: criterion.rs with 100 iterations
- Metrics: p50/p95/p99 latency, operations/second
- Duration: 30 seconds per benchmark
- Statistical: confidence intervals, regression detection

**Success Criteria:**
- Builder enforces type safety with 100% compile-time validation preventing invalid configurations
- Configuration validation catches 100% of constraint violations with specific error descriptions
- Async initialization completes within 2 seconds with timeout protection preventing hangs
- Error messages include specific corrective actions for 100% of error conditions
- Factory pattern matches existing codebase patterns with >95% similarity (validated via code review)

**Error Recovery Procedures:**
1. **Type Safety Violations**:
   - Detect: Compilation errors from invalid configurations or runtime type mismatches
   - Action: Implement stronger type constraints and compile-time configuration validation
   - Retry: Use phantom types and const generics to enforce configuration validity at compile time

2. **Initialization Timeouts**:
   - Detect: Async initialization takes longer than 2 seconds or hangs indefinitely
   - Action: Implement initialization with timeout and progress monitoring
   - Retry: Break initialization into stages with individual timeouts and provide detailed progress information

3. **Configuration Validation Gaps**:
   - Detect: Invalid configurations pass validation or unclear error messages
   - Action: Implement comprehensive validation rules and detailed error reporting
   - Retry: Add configuration examples and validation assistance with suggested corrections

**Rollback Procedure:**
- Time limit: 5 minutes maximum rollback time
- Steps: [1] implement basic factory with minimal validation [2] remove type safety enforcement temporarily [3] use synchronous initialization
- Validation: Verify basic TMS instances can be created and configured using simple factory pattern without advanced features

---

## Validation Checklist

- [ ] JTMS layer correctly maintains belief dependencies
- [ ] ATMS layer handles multiple contexts efficiently
- [ ] Spiking dependency graph preserves timing relationships
- [ ] Belief state management meets performance targets
- [ ] Context management prevents resource leaks
- [ ] Integration layer provides unified interface
- [ ] Factory pattern enables flexible TMS construction
- [ ] All components pass unit and integration tests
- [ ] Performance benchmarks meet target metrics
- [ ] Memory usage remains within specified limits

## Next Phase

Upon completion, proceed to **Phase 6.3: AGM Belief Revision Engine** for implementing belief revision capabilities.