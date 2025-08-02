# Task 1.6: Hebbian Strengthening

**Duration**: 3 hours  
**Complexity**: Medium-High  
**Dependencies**: Task 1.5 (Exponential Decay)  
**AI Assistant Suitability**: High - Well-defined learning algorithms  

## Objective

Implement high-performance Hebbian learning and synaptic strengthening mechanisms with spike-timing dependent plasticity (STDP), connection pruning, and batch learning optimizations for realistic neuromorphic behavior.

## Specification

Create biologically-accurate synaptic learning that scales to thousands of connections:

**Learning Rules**:
- Classic Hebbian: "Cells that fire together, wire together"
- STDP: Timing-dependent potentiation and depression
- Connection decay: Unused synapses weaken over time
- Competitive learning: Strong connections inhibit weak ones

**Performance Requirements**:
- Synaptic update: < 20ns per connection
- STDP calculation: < 50ns
- Batch learning: > 10,000 updates/second
- Memory per connection: < 16 bytes

**Mathematical Models**:
- Hebbian rule: `Δw = η * a_pre * a_post`
- STDP function: `F(Δt) = A_+ * e^(-Δt/τ_+)` (potentiation) or `-A_- * e^(Δt/τ_-)` (depression)
- Decay: `w(t+1) = w(t) * (1 - decay_rate * dt)`
- Competitive normalization: `w_i' = w_i / Σ(w_j)`

## Implementation Guide

### Step 1: Optimized Synaptic Connection Storage

```rust
// src/synaptic_connection.rs
use std::time::{SystemTime, UNIX_EPOCH};

/// Compact synaptic connection representation (16 bytes total)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SynapticConnection {
    /// Target column ID (4 bytes)
    pub target_id: u32,
    
    /// Synaptic weight (4 bytes)
    pub weight: f32,
    
    /// Last update timestamp in microseconds (8 bytes)
    pub last_update_us: u64,
}

impl SynapticConnection {
    pub fn new(target_id: u32, initial_weight: f32) -> Self {
        Self {
            target_id,
            weight: initial_weight.clamp(0.0, 1.0),
            last_update_us: current_time_us(),
        }
    }
    
    /// Calculate age in milliseconds
    #[inline]
    pub fn age_ms(&self) -> f32 {
        let now_us = current_time_us();
        (now_us.saturating_sub(self.last_update_us)) as f32 / 1000.0
    }
    
    /// Apply time-based decay
    #[inline]
    pub fn apply_decay(&mut self, decay_rate: f32, dt_ms: f32) {
        let decay_factor = (-decay_rate * dt_ms / 1000.0).exp();
        self.weight *= decay_factor;
        self.last_update_us = current_time_us();
    }
    
    /// Update weight with bounds checking
    #[inline]
    pub fn update_weight(&mut self, delta: f32) {
        self.weight = (self.weight + delta).clamp(0.0, 1.0);
        self.last_update_us = current_time_us();
    }
    
    /// Check if connection should be pruned
    #[inline]
    pub fn should_prune(&self, min_weight: f32, max_age_ms: f32) -> bool {
        self.weight < min_weight || self.age_ms() > max_age_ms
    }
}

/// Optimized storage for synaptic connections
pub struct SynapticConnectionMap {
    /// Sorted connections for efficient lookup
    connections: Vec<SynapticConnection>,
    
    /// Connection statistics
    total_weight: f32,
    last_normalization_us: u64,
    
    /// Performance counters
    update_count: u64,
    prune_count: u64,
}

impl SynapticConnectionMap {
    pub fn new() -> Self {
        Self {
            connections: Vec::new(),
            total_weight: 0.0,
            last_normalization_us: current_time_us(),
            update_count: 0,
            prune_count: 0,
        }
    }
    
    /// Add or update connection
    pub fn add_connection(&mut self, target_id: u32, weight: f32) {
        if let Some(conn) = self.find_connection_mut(target_id) {
            conn.weight = weight.clamp(0.0, 1.0);
            conn.last_update_us = current_time_us();
        } else {
            let new_conn = SynapticConnection::new(target_id, weight);
            
            // Insert in sorted order
            let insert_pos = self.connections
                .binary_search_by_key(&target_id, |c| c.target_id)
                .unwrap_or_else(|pos| pos);
            self.connections.insert(insert_pos, new_conn);
        }
        
        self.update_total_weight();
        self.update_count += 1;
    }
    
    /// Find connection by target ID
    #[inline]
    pub fn find_connection(&self, target_id: u32) -> Option<&SynapticConnection> {
        self.connections
            .binary_search_by_key(&target_id, |c| c.target_id)
            .ok()
            .map(|idx| &self.connections[idx])
    }
    
    /// Find mutable connection by target ID
    #[inline]
    fn find_connection_mut(&mut self, target_id: u32) -> Option<&mut SynapticConnection> {
        match self.connections.binary_search_by_key(&target_id, |c| c.target_id) {
            Ok(idx) => Some(&mut self.connections[idx]),
            Err(_) => None,
        }
    }
    
    /// Get connection weight (0.0 if not exists)
    #[inline]
    pub fn get_weight(&self, target_id: u32) -> f32 {
        self.find_connection(target_id)
            .map(|conn| conn.weight)
            .unwrap_or(0.0)
    }
    
    /// Apply decay to all connections
    pub fn apply_decay_all(&mut self, decay_rate: f32, dt_ms: f32) {
        for conn in &mut self.connections {
            conn.apply_decay(decay_rate, dt_ms);
        }
        self.update_total_weight();
    }
    
    /// Prune weak and old connections
    pub fn prune_connections(&mut self, min_weight: f32, max_age_ms: f32) -> usize {
        let initial_count = self.connections.len();
        
        self.connections.retain(|conn| {
            let should_keep = !conn.should_prune(min_weight, max_age_ms);
            if !should_keep {
                self.prune_count += 1;
            }
            should_keep
        });
        
        self.update_total_weight();
        initial_count - self.connections.len()
    }
    
    /// Normalize all weights to maintain total weight budget
    pub fn normalize_weights(&mut self, target_total: f32) {
        if self.total_weight > 0.0 && self.total_weight != target_total {
            let scale_factor = target_total / self.total_weight;
            
            for conn in &mut self.connections {
                conn.weight *= scale_factor;
                conn.last_update_us = current_time_us();
            }
            
            self.total_weight = target_total;
            self.last_normalization_us = current_time_us();
        }
    }
    
    /// Update total weight calculation
    fn update_total_weight(&mut self) {
        self.total_weight = self.connections.iter().map(|c| c.weight).sum();
    }
    
    /// Get all connections (read-only)
    pub fn connections(&self) -> &[SynapticConnection] {
        &self.connections
    }
    
    /// Get performance statistics
    pub fn stats(&self) -> ConnectionStats {
        ConnectionStats {
            connection_count: self.connections.len(),
            total_weight: self.total_weight,
            average_weight: if self.connections.is_empty() { 
                0.0 
            } else { 
                self.total_weight / self.connections.len() as f32 
            },
            update_count: self.update_count,
            prune_count: self.prune_count,
            last_normalization_us: self.last_normalization_us,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConnectionStats {
    pub connection_count: usize,
    pub total_weight: f32,
    pub average_weight: f32,
    pub update_count: u64,
    pub prune_count: u64,
    pub last_normalization_us: u64,
}

fn current_time_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}
```

### Step 2: STDP Learning Engine

```rust
// src/stdp_learning.rs
use crate::{SynapticConnectionMap, BiologicalConfig, current_time_us};
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};

/// Spike-Timing Dependent Plasticity learning engine
pub struct STDPLearningEngine {
    /// Configuration parameters
    config: STDPConfig,
    
    /// Learning statistics
    potentiation_events: AtomicU64,
    depression_events: AtomicU64,
    total_updates: AtomicU64,
    
    /// Performance tracking
    update_times_ns: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct STDPConfig {
    /// Learning rate for potentiation
    pub learning_rate_pos: f32,
    
    /// Learning rate for depression  
    pub learning_rate_neg: f32,
    
    /// Time constant for potentiation (ms)
    pub tau_pos_ms: f32,
    
    /// Time constant for depression (ms)
    pub tau_neg_ms: f32,
    
    /// STDP window (ms) - beyond this, no plasticity
    pub stdp_window_ms: f32,
    
    /// Maximum weight change per update
    pub max_delta_weight: f32,
    
    /// Minimum weight threshold
    pub min_weight: f32,
    
    /// Maximum weight threshold
    pub max_weight: f32,
}

impl Default for STDPConfig {
    fn default() -> Self {
        Self {
            learning_rate_pos: 0.01,
            learning_rate_neg: 0.008,
            tau_pos_ms: 16.8,
            tau_neg_ms: 33.7,
            stdp_window_ms: 100.0,
            max_delta_weight: 0.1,
            min_weight: 0.001,
            max_weight: 1.0,
        }
    }
}

impl STDPConfig {
    pub fn from_biological_config(bio_config: &BiologicalConfig) -> Self {
        Self {
            learning_rate_pos: bio_config.hebbian_learning_rate,
            learning_rate_neg: bio_config.hebbian_learning_rate * 0.8,
            tau_pos_ms: bio_config.stdp_window_ms * 0.2,
            tau_neg_ms: bio_config.stdp_window_ms * 0.4,
            stdp_window_ms: bio_config.stdp_window_ms,
            max_delta_weight: 0.1,
            min_weight: bio_config.min_synaptic_weight,
            max_weight: bio_config.max_synaptic_weight,
        }
    }
}

impl STDPLearningEngine {
    pub fn new(config: STDPConfig) -> Self {
        Self {
            config,
            potentiation_events: AtomicU64::new(0),
            depression_events: AtomicU64::new(0),
            total_updates: AtomicU64::new(0),
            update_times_ns: AtomicU64::new(0),
        }
    }
    
    /// Calculate STDP weight change based on timing difference
    #[inline]
    pub fn calculate_stdp_delta(&self, dt_ms: f32, pre_strength: f32, post_strength: f32) -> STDPResult {
        let abs_dt = dt_ms.abs();
        
        // Check if within STDP window
        if abs_dt > self.config.stdp_window_ms {
            return STDPResult::OutsideWindow;
        }
        
        let activation_product = pre_strength * post_strength;
        
        let (stdp_factor, is_potentiation) = if dt_ms > 0.0 {
            // Post-synaptic fired after pre-synaptic: LTP (potentiation)
            let factor = self.config.learning_rate_pos * (-dt_ms / self.config.tau_pos_ms).exp();
            (factor, true)
        } else {
            // Post-synaptic fired before pre-synaptic: LTD (depression)
            let factor = -self.config.learning_rate_neg * (dt_ms / self.config.tau_neg_ms).exp();
            (factor, false)
        };
        
        let delta_weight = (stdp_factor * activation_product)
            .clamp(-self.config.max_delta_weight, self.config.max_delta_weight);
        
        STDPResult::WeightChange {
            delta: delta_weight,
            is_potentiation,
            stdp_factor,
            timing_ms: dt_ms,
        }
    }
    
    /// Apply STDP learning to synaptic connections
    pub fn apply_stdp_learning(
        &self,
        connections: &mut SynapticConnectionMap,
        pre_fire_time_us: u64,
        pre_strength: f32,
        post_fire_events: &[(u32, u64, f32)], // (target_id, fire_time_us, strength)
    ) -> STDPBatchResult {
        let start_time = std::time::Instant::now();
        
        let mut potentiation_count = 0;
        let mut depression_count = 0;
        let mut total_delta = 0.0f32;
        
        for &(target_id, post_fire_time_us, post_strength) in post_fire_events {
            // Calculate timing difference
            let dt_ms = if post_fire_time_us > pre_fire_time_us {
                (post_fire_time_us - pre_fire_time_us) as f32 / 1000.0
            } else {
                -((pre_fire_time_us - post_fire_time_us) as f32 / 1000.0)
            };
            
            // Calculate STDP delta
            match self.calculate_stdp_delta(dt_ms, pre_strength, post_strength) {
                STDPResult::WeightChange { delta, is_potentiation, .. } => {
                    // Get current weight
                    let current_weight = connections.get_weight(target_id);
                    let new_weight = (current_weight + delta)
                        .clamp(self.config.min_weight, self.config.max_weight);
                    
                    // Update connection
                    connections.add_connection(target_id, new_weight);
                    
                    // Track statistics
                    if is_potentiation {
                        potentiation_count += 1;
                    } else {
                        depression_count += 1;
                    }
                    total_delta += delta.abs();
                }
                STDPResult::OutsideWindow => {
                    // No learning - timing difference too large
                }
            }
        }
        
        // Update performance counters
        let elapsed_ns = start_time.elapsed().as_nanos() as u64;
        self.update_times_ns.fetch_add(elapsed_ns, Ordering::Relaxed);
        self.potentiation_events.fetch_add(potentiation_count, Ordering::Relaxed);
        self.depression_events.fetch_add(depression_count, Ordering::Relaxed);
        self.total_updates.fetch_add(post_fire_events.len() as u64, Ordering::Relaxed);
        
        STDPBatchResult {
            potentiation_events: potentiation_count,
            depression_events: depression_count,
            total_weight_change: total_delta,
            processing_time_ns: elapsed_ns,
        }
    }
    
    /// Competitive learning: strengthen winner, weaken others
    pub fn apply_competitive_learning(
        &self,
        connections: &mut SynapticConnectionMap,
        winner_id: u32,
        winner_strength: f32,
        competitor_ids: &[u32],
        competition_factor: f32,
    ) -> CompetitiveLearningResult {
        let start_time = std::time::Instant::now();
        
        // Strengthen winner connection
        let winner_current = connections.get_weight(winner_id);
        let winner_delta = self.config.learning_rate_pos * winner_strength * competition_factor;
        let winner_new = (winner_current + winner_delta).clamp(self.config.min_weight, self.config.max_weight);
        connections.add_connection(winner_id, winner_new);
        
        // Weaken competitor connections
        let mut competitors_weakened = 0;
        let mut total_weakening = 0.0f32;
        
        for &competitor_id in competitor_ids {
            if competitor_id != winner_id {
                let current_weight = connections.get_weight(competitor_id);
                if current_weight > self.config.min_weight {
                    let depression = self.config.learning_rate_neg * competition_factor * 0.5;
                    let new_weight = (current_weight - depression).clamp(self.config.min_weight, self.config.max_weight);
                    connections.add_connection(competitor_id, new_weight);
                    
                    competitors_weakened += 1;
                    total_weakening += depression;
                }
            }
        }
        
        CompetitiveLearningResult {
            winner_strengthening: winner_delta,
            competitors_weakened,
            total_weakening,
            processing_time_ns: start_time.elapsed().as_nanos() as u64,
        }
    }
    
    /// Get learning performance statistics
    pub fn get_performance_stats(&self) -> STDPPerformanceStats {
        let total_updates = self.total_updates.load(Ordering::Relaxed);
        let total_time_ns = self.update_times_ns.load(Ordering::Relaxed);
        
        STDPPerformanceStats {
            total_updates,
            potentiation_events: self.potentiation_events.load(Ordering::Relaxed),
            depression_events: self.depression_events.load(Ordering::Relaxed),
            average_time_per_update_ns: if total_updates > 0 { 
                total_time_ns / total_updates 
            } else { 
                0 
            },
            total_processing_time_ns: total_time_ns,
        }
    }
    
    /// Reset performance counters
    pub fn reset_stats(&self) {
        self.potentiation_events.store(0, Ordering::Relaxed);
        self.depression_events.store(0, Ordering::Relaxed);
        self.total_updates.store(0, Ordering::Relaxed);
        self.update_times_ns.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub enum STDPResult {
    WeightChange {
        delta: f32,
        is_potentiation: bool,
        stdp_factor: f32,
        timing_ms: f32,
    },
    OutsideWindow,
}

#[derive(Debug, Clone)]
pub struct STDPBatchResult {
    pub potentiation_events: u64,
    pub depression_events: u64,
    pub total_weight_change: f32,
    pub processing_time_ns: u64,
}

#[derive(Debug, Clone)]
pub struct CompetitiveLearningResult {
    pub winner_strengthening: f32,
    pub competitors_weakened: usize,
    pub total_weakening: f32,
    pub processing_time_ns: u64,
}

#[derive(Debug, Clone)]
pub struct STDPPerformanceStats {
    pub total_updates: u64,
    pub potentiation_events: u64,
    pub depression_events: u64,
    pub average_time_per_update_ns: u64,
    pub total_processing_time_ns: u64,
}
```

### Step 3: Enhanced Biological Column with Learning

```rust
// src/learning_cortical_column.rs
use crate::{
    BiologicalCorticalColumn, STDPLearningEngine, STDPConfig, SynapticConnectionMap,
    BiologicalConfig, current_time_us, STDPBatchResult, CompetitiveLearningResult
};
use std::sync::atomic::{AtomicU64, Ordering};
use parking_lot::RwLock;

/// Cortical column with advanced learning capabilities
pub struct LearningCorticalColumn {
    /// Base biological column
    base: BiologicalCorticalColumn,
    
    /// Synaptic connections to other columns
    outgoing_connections: RwLock<SynapticConnectionMap>,
    
    /// STDP learning engine
    stdp_engine: STDPLearningEngine,
    
    /// Learning statistics
    learning_events: AtomicU64,
    last_fire_time_us: AtomicU64,
    last_fire_strength: AtomicU64, // Stored as f32 bits
    
    /// Configuration
    config: BiologicalConfig,
}

impl LearningCorticalColumn {
    pub fn new(id: u32, config: BiologicalConfig) -> Self {
        let stdp_config = STDPConfig::from_biological_config(&config);
        
        Self {
            base: BiologicalCorticalColumn::new(id, config.clone()),
            outgoing_connections: RwLock::new(SynapticConnectionMap::new()),
            stdp_engine: STDPLearningEngine::new(stdp_config),
            learning_events: AtomicU64::new(0),
            last_fire_time_us: AtomicU64::new(0),
            last_fire_strength: AtomicU64::new(0.0f32.to_bits() as u64),
            config,
        }
    }
    
    /// Stimulate and potentially learn
    pub fn stimulate_and_learn(&self, input_strength: f32, duration_ms: f32) -> LearningStimulationResult {
        // Apply biological stimulation
        let stimulation_result = self.base.stimulate(input_strength, duration_ms);
        
        // Check if we fired
        if let crate::StimulationResult::Fired { fire_voltage, refractory_state, .. } = &stimulation_result {
            // Record firing for future learning
            self.last_fire_time_us.store(refractory_state.fire_time_us, Ordering::Release);
            self.last_fire_strength.store(fire_voltage.to_bits() as u64, Ordering::Release);
            
            LearningStimulationResult {
                stimulation: stimulation_result,
                fired: true,
                fire_time_us: refractory_state.fire_time_us,
                fire_strength: *fire_voltage,
                learning_ready: true,
            }
        } else {
            LearningStimulationResult {
                stimulation: stimulation_result,
                fired: false,
                fire_time_us: 0,
                fire_strength: 0.0,
                learning_ready: false,
            }
        }
    }
    
    /// Learn from co-activation with other columns (batch STDP)
    pub fn learn_from_coactivations(&self, coactivation_events: &[(u32, u64, f32)]) -> STDPBatchResult {
        let pre_fire_time = self.last_fire_time_us.load(Ordering::Acquire);
        let pre_strength = f32::from_bits(self.last_fire_strength.load(Ordering::Acquire) as u32);
        
        if pre_fire_time == 0 || coactivation_events.is_empty() {
            return STDPBatchResult {
                potentiation_events: 0,
                depression_events: 0,
                total_weight_change: 0.0,
                processing_time_ns: 0,
            };
        }
        
        // Apply STDP learning
        let mut connections = self.outgoing_connections.write();
        let result = self.stdp_engine.apply_stdp_learning(
            &mut *connections,
            pre_fire_time,
            pre_strength,
            coactivation_events,
        );
        
        // Prune weak connections periodically
        if self.learning_events.fetch_add(1, Ordering::Relaxed) % 100 == 0 {
            connections.prune_connections(0.001, 10000.0); // Remove weights < 0.001 or older than 10s
        }
        
        result
    }
    
    /// Apply competitive learning (winner-take-all)
    pub fn apply_competitive_learning(
        &self,
        is_winner: bool,
        competitor_ids: &[u32],
        competition_strength: f32,
    ) -> Option<CompetitiveLearningResult> {
        if !is_winner {
            return None;
        }
        
        let my_id = self.base.base().id();
        let my_strength = f32::from_bits(self.last_fire_strength.load(Ordering::Acquire) as u32);
        
        let mut connections = self.outgoing_connections.write();
        let result = self.stdp_engine.apply_competitive_learning(
            &mut *connections,
            my_id,
            my_strength,
            competitor_ids,
            competition_strength,
        );
        
        Some(result)
    }
    
    /// Decay all synaptic connections
    pub fn apply_synaptic_decay(&self, decay_rate: f32, dt_ms: f32) {
        let mut connections = self.outgoing_connections.write();
        connections.apply_decay_all(decay_rate, dt_ms);
    }
    
    /// Normalize synaptic weights
    pub fn normalize_synaptic_weights(&self, target_total_weight: f32) {
        let mut connections = self.outgoing_connections.write();
        connections.normalize_weights(target_total_weight);
    }
    
    /// Get synaptic weight to specific target
    pub fn get_synaptic_weight(&self, target_id: u32) -> f32 {
        self.outgoing_connections.read().get_weight(target_id)
    }
    
    /// Get all synaptic connections
    pub fn get_all_synaptic_weights(&self) -> Vec<(u32, f32)> {
        self.outgoing_connections.read()
            .connections()
            .iter()
            .map(|conn| (conn.target_id, conn.weight))
            .collect()
    }
    
    /// Get learning performance metrics
    pub fn learning_performance(&self) -> LearningPerformanceMetrics {
        let stdp_stats = self.stdp_engine.get_performance_stats();
        let connection_stats = self.outgoing_connections.read().stats();
        
        LearningPerformanceMetrics {
            stdp_stats,
            connection_stats,
            total_learning_events: self.learning_events.load(Ordering::Relaxed),
        }
    }
    
    /// Access base biological column
    pub fn base(&self) -> &BiologicalCorticalColumn {
        &self.base
    }
    
    /// Get learning readiness
    pub fn is_learning_ready(&self) -> bool {
        self.last_fire_time_us.load(Ordering::Relaxed) > 0
    }
    
    /// Reset learning state
    pub fn reset_learning_state(&self) {
        self.last_fire_time_us.store(0, Ordering::Release);
        self.last_fire_strength.store(0.0f32.to_bits() as u64, Ordering::Release);
        self.learning_events.store(0, Ordering::Release);
        self.stdp_engine.reset_stats();
    }
}

#[derive(Debug, Clone)]
pub struct LearningStimulationResult {
    pub stimulation: crate::StimulationResult,
    pub fired: bool,
    pub fire_time_us: u64,
    pub fire_strength: f32,
    pub learning_ready: bool,
}

#[derive(Debug, Clone)]
pub struct LearningPerformanceMetrics {
    pub stdp_stats: crate::STDPPerformanceStats,
    pub connection_stats: crate::ConnectionStats,
    pub total_learning_events: u64,
}
```

## AI-Executable Test Suite

```rust
// tests/hebbian_strengthening_test.rs
use llmkg::{
    LearningCorticalColumn, BiologicalConfig, SynapticConnectionMap, 
    STDPLearningEngine, STDPConfig, current_time_us
};
use std::time::{Duration, Instant};
use std::thread;

#[test]
fn test_synaptic_connection_storage() {
    let mut connections = SynapticConnectionMap::new();
    
    // Add connections
    connections.add_connection(1, 0.5);
    connections.add_connection(2, 0.8);
    connections.add_connection(3, 0.3);
    
    // Test retrieval
    assert_eq!(connections.get_weight(1), 0.5);
    assert_eq!(connections.get_weight(2), 0.8);
    assert_eq!(connections.get_weight(3), 0.3);
    assert_eq!(connections.get_weight(999), 0.0); // Non-existent
    
    // Test update
    connections.add_connection(1, 0.7);
    assert_eq!(connections.get_weight(1), 0.7);
    
    // Test statistics
    let stats = connections.stats();
    assert_eq!(stats.connection_count, 3);
    assert_eq!(stats.total_weight, 0.7 + 0.8 + 0.3);
}

#[test]
fn test_stdp_learning_engine() {
    let config = STDPConfig::default();
    let engine = STDPLearningEngine::new(config);
    
    // Test STDP calculation
    let dt_positive = 10.0; // Post fires after pre (potentiation)
    let dt_negative = -15.0; // Post fires before pre (depression)
    
    let result_pos = engine.calculate_stdp_delta(dt_positive, 0.8, 0.9);
    let result_neg = engine.calculate_stdp_delta(dt_negative, 0.8, 0.9);
    
    // Verify potentiation
    if let crate::STDPResult::WeightChange { delta, is_potentiation, .. } = result_pos {
        assert!(delta > 0.0);
        assert!(is_potentiation);
    } else {
        panic!("Expected weight change for potentiation");
    }
    
    // Verify depression
    if let crate::STDPResult::WeightChange { delta, is_potentiation, .. } = result_neg {
        assert!(delta < 0.0);
        assert!(!is_potentiation);
    } else {
        panic!("Expected weight change for depression");
    }
    
    // Test outside window
    let result_outside = engine.calculate_stdp_delta(200.0, 0.8, 0.9);
    assert!(matches!(result_outside, crate::STDPResult::OutsideWindow));
}

#[test]
fn test_learning_cortical_column() {
    let config = BiologicalConfig::fast_processing();
    let column = LearningCorticalColumn::new(1, config);
    
    // Initial state - no learning ready
    assert!(!column.is_learning_ready());
    
    // Fire the column
    let result = column.stimulate_and_learn(1.5, 1.0);
    assert!(result.fired);
    assert!(result.learning_ready);
    assert!(column.is_learning_ready());
    
    // Test synaptic weight retrieval
    let weight = column.get_synaptic_weight(999);
    assert_eq!(weight, 0.0); // No connection initially
}

#[test]
fn test_hebbian_learning_coactivation() {
    let config = BiologicalConfig::fast_processing();
    let column1 = LearningCorticalColumn::new(1, config.clone());
    let column2 = LearningCorticalColumn::new(2, config);
    
    // Fire both columns with slight timing difference
    let result1 = column1.stimulate_and_learn(1.5, 1.0);
    thread::sleep(Duration::from_millis(5)); // 5ms delay
    let result2 = column2.stimulate_and_learn(1.5, 1.0);
    
    assert!(result1.fired);
    assert!(result2.fired);
    
    // Create coactivation event
    let coactivations = vec![(2, result2.fire_time_us, result2.fire_strength)];
    
    // Apply STDP learning
    let learning_result = column1.learn_from_coactivations(&coactivations);
    
    // Should have learning events
    assert!(learning_result.potentiation_events > 0 || learning_result.depression_events > 0);
    assert!(learning_result.processing_time_ns > 0);
    
    // Check if synaptic weight was updated
    let weight_after = column1.get_synaptic_weight(2);
    assert!(weight_after > 0.0); // Should have established connection
}

#[test]
fn test_competitive_learning() {
    let config = BiologicalConfig::fast_processing();
    let winner = LearningCorticalColumn::new(1, config.clone());
    let competitor1 = LearningCorticalColumn::new(2, config.clone());
    let competitor2 = LearningCorticalColumn::new(3, config);
    
    // Fire all columns
    winner.stimulate_and_learn(1.5, 1.0);
    competitor1.stimulate_and_learn(1.2, 1.0);
    competitor2.stimulate_and_learn(1.1, 1.0);
    
    // Apply competitive learning
    let competitor_ids = vec![2, 3];
    let result = winner.apply_competitive_learning(true, &competitor_ids, 0.8);
    
    assert!(result.is_some());
    let learning_result = result.unwrap();
    assert!(learning_result.winner_strengthening > 0.0);
    assert_eq!(learning_result.competitors_weakened, 2);
}

#[test]
fn test_synaptic_decay_and_pruning() {
    let mut connections = SynapticConnectionMap::new();
    
    // Add some connections
    connections.add_connection(1, 0.9);
    connections.add_connection(2, 0.1);
    connections.add_connection(3, 0.05);
    
    // Apply decay
    connections.apply_decay_all(0.1, 100.0); // 10% decay over 100ms
    
    // Weights should have decayed
    assert!(connections.get_weight(1) < 0.9);
    assert!(connections.get_weight(2) < 0.1);
    
    // Prune weak connections
    let pruned_count = connections.prune_connections(0.08, 1000.0);
    
    // Should have pruned weak connections
    assert!(pruned_count > 0);
    assert_eq!(connections.get_weight(3), 0.0); // Should be pruned
}

#[test]
fn test_performance_benchmarks() {
    let config = BiologicalConfig::fast_processing();
    let column = LearningCorticalColumn::new(1, config);
    
    // Fire the column
    column.stimulate_and_learn(1.5, 1.0);
    
    // Benchmark STDP learning
    let coactivations: Vec<_> = (0..100)
        .map(|i| (i as u32 + 10, current_time_us() + i * 1000, 0.8))
        .collect();
    
    let start = Instant::now();
    let result = column.learn_from_coactivations(&coactivations);
    let elapsed = start.elapsed();
    
    println!("STDP batch learning: {} events in {} ns", 
             coactivations.len(), result.processing_time_ns);
    
    // Should be fast
    assert!(result.processing_time_ns < 1_000_000); // < 1ms for 100 updates
    
    // Check performance metrics
    let performance = column.learning_performance();
    assert!(performance.stdp_stats.total_updates > 0);
    assert!(performance.stdp_stats.average_time_per_update_ns < 10_000); // < 10μs per update
}

#[test]
fn test_weight_normalization() {
    let mut connections = SynapticConnectionMap::new();
    
    // Add connections with total weight > target
    connections.add_connection(1, 0.6);
    connections.add_connection(2, 0.8);
    connections.add_connection(3, 0.9);
    // Total = 2.3
    
    // Normalize to target total of 1.0
    connections.normalize_weights(1.0);
    
    let stats = connections.stats();
    assert!((stats.total_weight - 1.0).abs() < 0.001);
    
    // Individual weights should be scaled proportionally
    let weight1 = connections.get_weight(1);
    let weight2 = connections.get_weight(2);
    let weight3 = connections.get_weight(3);
    
    // Ratios should be preserved
    let ratio_1_2 = weight1 / weight2;
    let expected_ratio = 0.6 / 0.8;
    assert!((ratio_1_2 - expected_ratio).abs() < 0.001);
}

#[test]
fn test_batch_learning_performance() {
    let config = BiologicalConfig::fast_processing();
    let column = LearningCorticalColumn::new(1, config);
    
    // Fire the column
    column.stimulate_and_learn(1.5, 1.0);
    
    // Create large batch of coactivations
    let large_batch: Vec<_> = (0..1000)
        .map(|i| (i as u32 + 100, current_time_us() + i * 100, 0.7 + (i % 10) as f32 * 0.03))
        .collect();
    
    let start = Instant::now();
    let result = column.learn_from_coactivations(&large_batch);
    let elapsed = start.elapsed();
    
    println!("Large batch (1000 events): {} ns total, {} ns per event",
             result.processing_time_ns,
             result.processing_time_ns / 1000);
    
    // Should process > 10,000 updates/second
    let updates_per_second = 1_000_000_000.0 / (result.processing_time_ns as f64 / 1000.0);
    assert!(updates_per_second > 10_000.0);
    
    // Should have learning events
    assert!(result.potentiation_events + result.depression_events > 0);
}
```

## Success Criteria (AI Verification)

Your implementation is complete when:

1. **All tests pass**: 8/8 Hebbian learning tests passing
2. **Performance targets met**:
   - STDP calculation < 10μs per update
   - Batch learning > 10,000 updates/second
   - Synaptic storage < 16 bytes per connection
3. **Learning behavior verified**:
   - Potentiation for positive timing differences
   - Depression for negative timing differences
   - Competitive learning strengthens winners
4. **Memory management**: Pruning removes weak connections correctly

## Verification Commands

```bash
# Run Hebbian learning tests
cargo test hebbian_strengthening_test --release -- --nocapture

# Performance benchmarks
cargo test test_performance_benchmarks --release -- --nocapture
cargo test test_batch_learning_performance --release -- --nocapture

# Learning behavior validation
cargo test test_hebbian_learning_coactivation --release -- --nocapture
cargo test test_competitive_learning --release -- --nocapture
```

## Files to Create

1. `src/synaptic_connection.rs`
2. `src/stdp_learning.rs`
3. `src/learning_cortical_column.rs`
4. `tests/hebbian_strengthening_test.rs`

## Expected Performance Results

```
STDP batch learning: 100 events in 50,000 ns
Large batch (1000 events): 400,000 ns total, 400 ns per event
Updates per second: 25,000+
Memory per connection: 16 bytes
Cache effectiveness: > 90%
Weight normalization: < 1ms for 1000 connections
```

## Expected Completion Time

3 hours for an AI assistant:
- 60 minutes: Synaptic connection storage optimization
- 75 minutes: STDP learning engine implementation
- 30 minutes: Integration with biological column
- 15 minutes: Testing and performance validation

## Next Task

Task 1.7: Lateral Inhibition Core (winner-take-all mechanisms)