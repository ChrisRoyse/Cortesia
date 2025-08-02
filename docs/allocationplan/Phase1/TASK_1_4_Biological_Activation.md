# Task 1.4: Biological Activation

**Duration**: 3 hours  
**Complexity**: Medium-High  
**Dependencies**: Task 1.3 (Thread Safety Tests)  
**AI Assistant Suitability**: High - Well-defined biological modeling  

## Objective

Implement biologically-inspired activation dynamics for cortical columns, including exponential decay, membrane potential simulation, and Hebbian learning principles. This creates the neuromorphic foundation for realistic neural behavior.

## Specification

Implement activation dynamics that mirror real cortical neurons:

**Biological Properties**:
- Exponential decay with configurable time constants (tau)
- Membrane potential simulation with threshold firing
- Refractory period enforcement (absolute + relative)
- Hebbian-style connection strengthening
- Spike-timing dependent behavior

**Mathematical Models**:
- Activation decay: `A(t) = A₀ * e^(-t/τ)`
- Membrane potential: `V(t) = V_rest + (V_input - V_rest) * (1 - e^(-t/τ_mem))`
- Hebbian learning: `Δw = η * a_pre * a_post * f(|t_pre - t_post|)`
- Refractory scaling: `threshold = threshold_base * (1 + refractory_factor)`

**Performance Requirements**:
- Decay calculation: < 10ns
- Membrane update: < 15ns  
- Hebbian update: < 20ns
- Temporal precision: 1μs accuracy

## Implementation Guide

### Step 1: Biological Constants and Configuration

```rust
// src/biological_config.rs
use std::time::Duration;

/// Biologically-inspired configuration for cortical column activation
#[derive(Debug, Clone)]
pub struct BiologicalConfig {
    /// Membrane time constant (typical: 10-20ms for cortical neurons)
    pub membrane_tau_ms: f32,
    
    /// Activation decay time constant (how long activation persists)
    pub activation_tau_ms: f32,
    
    /// Resting membrane potential (normalized: 0.0)
    pub resting_potential: f32,
    
    /// Firing threshold (normalized: 0.7-0.9)
    pub firing_threshold: f32,
    
    /// Absolute refractory period (neuron cannot fire)
    pub absolute_refractory_ms: f32,
    
    /// Relative refractory period (higher threshold)
    pub relative_refractory_ms: f32,
    
    /// Hebbian learning rate
    pub hebbian_learning_rate: f32,
    
    /// STDP time window (spike-timing dependent plasticity)
    pub stdp_window_ms: f32,
    
    /// Maximum synaptic weight
    pub max_synaptic_weight: f32,
    
    /// Minimum synaptic weight  
    pub min_synaptic_weight: f32,
}

impl Default for BiologicalConfig {
    fn default() -> Self {
        Self {
            membrane_tau_ms: 15.0,      // Typical cortical neuron
            activation_tau_ms: 100.0,   // Slower decay for concepts
            resting_potential: 0.0,
            firing_threshold: 0.8,
            absolute_refractory_ms: 2.0,
            relative_refractory_ms: 10.0,
            hebbian_learning_rate: 0.01,
            stdp_window_ms: 20.0,
            max_synaptic_weight: 1.0,
            min_synaptic_weight: 0.0,
        }
    }
}

impl BiologicalConfig {
    /// Cortical neuron configuration (realistic biology)
    pub fn cortical_neuron() -> Self {
        Self {
            membrane_tau_ms: 12.0,
            activation_tau_ms: 80.0,
            firing_threshold: 0.75,
            absolute_refractory_ms: 1.5,
            relative_refractory_ms: 8.0,
            hebbian_learning_rate: 0.008,
            stdp_window_ms: 15.0,
            ..Default::default()
        }
    }
    
    /// Fast processing configuration (optimized for speed)
    pub fn fast_processing() -> Self {
        Self {
            membrane_tau_ms: 5.0,
            activation_tau_ms: 50.0,
            firing_threshold: 0.85,
            absolute_refractory_ms: 0.5,
            relative_refractory_ms: 3.0,
            hebbian_learning_rate: 0.02,
            stdp_window_ms: 10.0,
            ..Default::default()
        }
    }
}
```

### Step 2: Membrane Potential Simulation

```rust
// src/membrane_potential.rs
use crate::BiologicalConfig;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Simulates biological membrane potential with realistic dynamics
pub struct MembranePotential {
    /// Current membrane voltage (encoded as f32 bits)
    voltage: AtomicU32,
    
    /// Last update timestamp (microseconds)
    last_update_us: AtomicU64,
    
    /// Target voltage (what we're decaying toward)
    target_voltage: AtomicU32,
    
    /// Configuration
    config: BiologicalConfig,
}

impl MembranePotential {
    pub fn new(config: BiologicalConfig) -> Self {
        let now_us = current_time_us();
        
        Self {
            voltage: AtomicU32::new(config.resting_potential.to_bits()),
            last_update_us: AtomicU64::new(now_us),
            target_voltage: AtomicU32::new(config.resting_potential.to_bits()),
            config,
        }
    }
    
    /// Get current membrane potential (with decay applied)
    pub fn current_voltage(&self) -> f32 {
        self.update_voltage_decay();
        f32::from_bits(self.voltage.load(Ordering::Acquire))
    }
    
    /// Apply input stimulus to membrane
    pub fn apply_input(&self, input_voltage: f32, duration_ms: f32) {
        // Clamp input to reasonable range
        let clamped_input = input_voltage.clamp(-2.0, 2.0);
        
        // Set new target voltage
        let new_target = self.config.resting_potential + 
                        (clamped_input - self.config.resting_potential) * 0.8;
        self.target_voltage.store(new_target.to_bits(), Ordering::Release);
        
        // Update timestamp
        self.last_update_us.store(current_time_us(), Ordering::Release);
    }
    
    /// Check if membrane has reached firing threshold
    pub fn check_firing_threshold(&self) -> bool {
        self.current_voltage() >= self.config.firing_threshold
    }
    
    /// Reset membrane after firing (refractory period)
    pub fn fire_and_reset(&self) -> FireResult {
        let pre_fire_voltage = self.current_voltage();
        
        // Reset to below resting potential (hyperpolarization)
        let reset_voltage = self.config.resting_potential - 0.1;
        self.voltage.store(reset_voltage.to_bits(), Ordering::Release);
        self.target_voltage.store(self.config.resting_potential.to_bits(), Ordering::Release);
        
        let fire_time_us = current_time_us();
        self.last_update_us.store(fire_time_us, Ordering::Release);
        
        FireResult {
            pre_fire_voltage,
            fire_time_us,
            reset_voltage,
        }
    }
    
    /// Update voltage based on exponential decay toward target
    fn update_voltage_decay(&self) {
        let now_us = current_time_us();
        let last_us = self.last_update_us.load(Ordering::Acquire);
        let dt_ms = (now_us - last_us) as f32 / 1000.0;
        
        // Only update if significant time has passed (>0.01ms)
        if dt_ms < 0.01 {
            return;
        }
        
        let current_v = f32::from_bits(self.voltage.load(Ordering::Acquire));
        let target_v = f32::from_bits(self.target_voltage.load(Ordering::Acquire));
        
        // Exponential decay: V(t) = V_target + (V_current - V_target) * e^(-t/τ)
        let tau = self.config.membrane_tau_ms;
        let decay_factor = (-dt_ms / tau).exp();
        let new_voltage = target_v + (current_v - target_v) * decay_factor;
        
        // Update voltage and timestamp atomically
        self.voltage.store(new_voltage.to_bits(), Ordering::Release);
        self.last_update_us.store(now_us, Ordering::Release);
    }
    
    /// Get decay time constant
    pub fn tau_ms(&self) -> f32 {
        self.config.membrane_tau_ms
    }
}

#[derive(Debug, Clone)]
pub struct FireResult {
    pub pre_fire_voltage: f32,
    pub fire_time_us: u64,
    pub reset_voltage: f32,
}

fn current_time_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as u64
}
```

### Step 3: Refractory Period Management

```rust
// src/refractory_period.rs
use crate::{BiologicalConfig, current_time_us};
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};

/// Manages refractory periods with biological accuracy
pub struct RefractoryPeriodManager {
    /// Last firing time (microseconds)
    last_fire_time_us: AtomicU64,
    
    /// Number of spikes in recent window
    recent_spike_count: AtomicU32,
    
    /// Configuration
    config: BiologicalConfig,
}

impl RefractoryPeriodManager {
    pub fn new(config: BiologicalConfig) -> Self {
        Self {
            last_fire_time_us: AtomicU64::new(0),
            recent_spike_count: AtomicU32::new(0),
            config,
        }
    }
    
    /// Check if neuron can fire (not in refractory period)
    pub fn can_fire(&self) -> bool {
        let now_us = current_time_us();
        let last_fire_us = self.last_fire_time_us.load(Ordering::Acquire);
        
        if last_fire_us == 0 {
            return true; // Never fired before
        }
        
        let time_since_fire_ms = (now_us - last_fire_us) as f32 / 1000.0;
        
        // Absolute refractory period - cannot fire at all
        if time_since_fire_ms < self.config.absolute_refractory_ms {
            return false;
        }
        
        // Relative refractory period - can fire but threshold is higher
        // This is handled by increasing the firing threshold
        true
    }
    
    /// Get current firing threshold (adjusted for refractory period)
    pub fn current_firing_threshold(&self) -> f32 {
        let now_us = current_time_us();
        let last_fire_us = self.last_fire_time_us.load(Ordering::Acquire);
        
        if last_fire_us == 0 {
            return self.config.firing_threshold;
        }
        
        let time_since_fire_ms = (now_us - last_fire_us) as f32 / 1000.0;
        
        // During relative refractory period, threshold is elevated
        if time_since_fire_ms < self.config.relative_refractory_ms {
            let refractory_factor = 1.0 - (time_since_fire_ms / self.config.relative_refractory_ms);
            let threshold_increase = refractory_factor * 0.3; // Up to 30% increase
            return self.config.firing_threshold + threshold_increase;
        }
        
        self.config.firing_threshold
    }
    
    /// Record a firing event
    pub fn record_firing(&self) -> RefractoryState {
        let now_us = current_time_us();
        let previous_fire_us = self.last_fire_time_us.swap(now_us, Ordering::AcqRel);
        
        // Update spike count
        let spike_count = self.recent_spike_count.fetch_add(1, Ordering::Relaxed);
        
        // Calculate inter-spike interval
        let inter_spike_interval_ms = if previous_fire_us > 0 {
            (now_us - previous_fire_us) as f32 / 1000.0
        } else {
            f32::INFINITY
        };
        
        RefractoryState {
            fire_time_us: now_us,
            inter_spike_interval_ms,
            spike_count: spike_count + 1,
            in_absolute_refractory: true,
            in_relative_refractory: true,
        }
    }
    
    /// Reset refractory state (for testing or initialization)
    pub fn reset(&self) {
        self.last_fire_time_us.store(0, Ordering::Release);
        self.recent_spike_count.store(0, Ordering::Release);
    }
    
    /// Get time since last firing
    pub fn time_since_last_fire_ms(&self) -> f32 {
        let now_us = current_time_us();
        let last_fire_us = self.last_fire_time_us.load(Ordering::Acquire);
        
        if last_fire_us == 0 {
            f32::INFINITY
        } else {
            (now_us - last_fire_us) as f32 / 1000.0
        }
    }
    
    /// Check for adaptation (frequency-dependent threshold changes)
    pub fn adaptation_factor(&self) -> f32 {
        let spike_count = self.recent_spike_count.load(Ordering::Relaxed);
        
        // Higher spike counts increase threshold (spike frequency adaptation)
        let adaptation = (spike_count as f32 * 0.02).min(0.2); // Max 20% increase
        1.0 + adaptation
    }
}

#[derive(Debug, Clone)]
pub struct RefractoryState {
    pub fire_time_us: u64,
    pub inter_spike_interval_ms: f32,
    pub spike_count: u32,
    pub in_absolute_refractory: bool,
    pub in_relative_refractory: bool,
}
```

### Step 4: Hebbian Learning and Synaptic Plasticity

```rust
// src/hebbian_learning.rs
use crate::{BiologicalConfig, current_time_us};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::collections::HashMap;
use parking_lot::RwLock;

/// Manages Hebbian learning and synaptic plasticity
pub struct HebbianLearningManager {
    /// Synaptic weights to other columns
    synaptic_weights: RwLock<HashMap<u32, SynapticConnection>>,
    
    /// Last activation time for STDP
    last_activation_us: AtomicU64,
    
    /// Activation strength at last firing
    last_activation_strength: AtomicU32,
    
    /// Configuration
    config: BiologicalConfig,
}

#[derive(Debug, Clone)]
struct SynapticConnection {
    weight: f32,
    last_update_us: u64,
    update_count: u32,
    potentiation_events: u32,
    depression_events: u32,
}

impl HebbianLearningManager {
    pub fn new(config: BiologicalConfig) -> Self {
        Self {
            synaptic_weights: RwLock::new(HashMap::new()),
            last_activation_us: AtomicU64::new(0),
            last_activation_strength: AtomicU32::new(0.0_f32.to_bits()),
            config,
        }
    }
    
    /// Record activation for STDP calculations
    pub fn record_activation(&self, strength: f32) {
        let now_us = current_time_us();
        self.last_activation_us.store(now_us, Ordering::Release);
        self.last_activation_strength.store(strength.to_bits(), Ordering::Release);
    }
    
    /// Update synaptic weight based on Hebbian learning
    pub fn update_synaptic_weight(
        &self,
        target_column_id: u32,
        target_activation_time_us: u64,
        target_strength: f32,
    ) -> HebbianUpdateResult {
        let now_us = current_time_us();
        let my_activation_us = self.last_activation_us.load(Ordering::Acquire);
        let my_strength = f32::from_bits(self.last_activation_strength.load(Ordering::Acquire));
        
        if my_activation_us == 0 {
            return HebbianUpdateResult::NoUpdate("No recent activation".to_string());
        }
        
        // Calculate timing difference (STDP)
        let dt_ms = if target_activation_time_us > my_activation_us {
            (target_activation_time_us - my_activation_us) as f32 / 1000.0
        } else {
            -((my_activation_us - target_activation_time_us) as f32 / 1000.0)
        };
        
        // STDP window check
        if dt_ms.abs() > self.config.stdp_window_ms {
            return HebbianUpdateResult::OutsideWindow(dt_ms);
        }
        
        // Calculate weight change
        let stdp_factor = self.calculate_stdp_factor(dt_ms);
        let activation_product = my_strength * target_strength;
        let weight_delta = self.config.hebbian_learning_rate * activation_product * stdp_factor;
        
        // Update synaptic connection
        let mut weights = self.synaptic_weights.write();
        let connection = weights.entry(target_column_id).or_insert(SynapticConnection {
            weight: 0.5, // Initial weight
            last_update_us: now_us,
            update_count: 0,
            potentiation_events: 0,
            depression_events: 0,
        });
        
        let old_weight = connection.weight;
        connection.weight = (connection.weight + weight_delta)
            .clamp(self.config.min_synaptic_weight, self.config.max_synaptic_weight);
        connection.last_update_us = now_us;
        connection.update_count += 1;
        
        if weight_delta > 0.0 {
            connection.potentiation_events += 1;
        } else if weight_delta < 0.0 {
            connection.depression_events += 1;
        }
        
        HebbianUpdateResult::Updated {
            old_weight,
            new_weight: connection.weight,
            weight_delta,
            stdp_factor,
            timing_ms: dt_ms,
        }
    }
    
    /// Get synaptic weight to target column
    pub fn get_synaptic_weight(&self, target_column_id: u32) -> f32 {
        self.synaptic_weights.read()
            .get(&target_column_id)
            .map(|conn| conn.weight)
            .unwrap_or(0.0)
    }
    
    /// Get all synaptic connections
    pub fn get_all_connections(&self) -> HashMap<u32, f32> {
        self.synaptic_weights.read()
            .iter()
            .map(|(&id, conn)| (id, conn.weight))
            .collect()
    }
    
    /// Calculate STDP factor based on timing
    fn calculate_stdp_factor(&self, dt_ms: f32) -> f32 {
        // Classic STDP curve: A+ * exp(-|Δt|/τ+) for potentiation (dt > 0)
        //                     -A- * exp(-|Δt|/τ-) for depression (dt < 0)
        
        let tau_plus = self.config.stdp_window_ms * 0.6; // 60% of window for potentiation
        let tau_minus = self.config.stdp_window_ms * 0.4; // 40% of window for depression
        
        if dt_ms > 0.0 {
            // Post-synaptic fired after pre-synaptic: potentiation
            (-(dt_ms / tau_plus)).exp()
        } else {
            // Post-synaptic fired before pre-synaptic: depression
            -0.5 * (-(-dt_ms / tau_minus)).exp()
        }
    }
    
    /// Decay all synaptic weights over time
    pub fn apply_synaptic_decay(&self, decay_rate: f32) {
        let now_us = current_time_us();
        let mut weights = self.synaptic_weights.write();
        
        for connection in weights.values_mut() {
            let time_since_update_ms = (now_us - connection.last_update_us) as f32 / 1000.0;
            
            // Apply exponential decay
            let decay_factor = (-decay_rate * time_since_update_ms / 1000.0).exp();
            connection.weight *= decay_factor;
            
            // Remove very weak connections
            if connection.weight < 0.01 {
                connection.weight = 0.0;
            }
        }
        
        // Remove zero-weight connections
        weights.retain(|_, conn| conn.weight > 0.0);
    }
}

#[derive(Debug, Clone)]
pub enum HebbianUpdateResult {
    Updated {
        old_weight: f32,
        new_weight: f32,
        weight_delta: f32,
        stdp_factor: f32,
        timing_ms: f32,
    },
    NoUpdate(String),
    OutsideWindow(f32),
}
```

### Step 5: Integrated Biological Cortical Column

```rust
// src/biological_cortical_column.rs
use crate::{
    BiologicalConfig, MembranePotential, RefractoryPeriodManager, 
    HebbianLearningManager, EnhancedCorticalColumn, ColumnState
};
use std::sync::Arc;

/// Cortical column with full biological dynamics
pub struct BiologicalCorticalColumn {
    /// Base column functionality
    base_column: EnhancedCorticalColumn,
    
    /// Membrane potential simulation
    membrane: MembranePotential,
    
    /// Refractory period management
    refractory: RefractoryPeriodManager,
    
    /// Hebbian learning
    hebbian: HebbianLearningManager,
    
    /// Configuration
    config: BiologicalConfig,
}

impl BiologicalCorticalColumn {
    pub fn new(id: u32, config: BiologicalConfig) -> Self {
        Self {
            base_column: EnhancedCorticalColumn::new(id),
            membrane: MembranePotential::new(config.clone()),
            refractory: RefractoryPeriodManager::new(config.clone()),
            hebbian: HebbianLearningManager::new(config.clone()),
            config,
        }
    }
    
    /// Stimulate the column with biological input
    pub fn stimulate(&self, input_strength: f32, duration_ms: f32) -> StimulationResult {
        // Apply input to membrane
        self.membrane.apply_input(input_strength, duration_ms);
        
        // Check if we can fire
        if !self.refractory.can_fire() {
            return StimulationResult::RefractoryBlock {
                time_remaining_ms: self.config.absolute_refractory_ms - 
                                  self.refractory.time_since_last_fire_ms(),
            };
        }
        
        // Check firing threshold (adjusted for refractory period)
        let threshold = self.refractory.current_firing_threshold() * 
                       self.refractory.adaptation_factor();
        let membrane_voltage = self.membrane.current_voltage();
        
        if membrane_voltage >= threshold {
            // Fire!
            let fire_result = self.membrane.fire_and_reset();
            let refractory_state = self.refractory.record_firing();
            
            // Update activation level
            let _ = self.base_column.try_activate_with_level(membrane_voltage);
            
            // Record for Hebbian learning
            self.hebbian.record_activation(membrane_voltage);
            
            StimulationResult::Fired {
                fire_voltage: fire_result.pre_fire_voltage,
                threshold_used: threshold,
                refractory_state,
            }
        } else {
            StimulationResult::SubThreshold {
                membrane_voltage,
                threshold_required: threshold,
                deficit: threshold - membrane_voltage,
            }
        }
    }
    
    /// Learn from co-activation with another column
    pub fn learn_from_coactivation(&self, other_column: &BiologicalCorticalColumn) -> crate::HebbianUpdateResult {
        let other_id = other_column.base_column.id();
        let other_activation_time = other_column.refractory.last_fire_time_us.load(std::sync::atomic::Ordering::Acquire);
        let other_strength = other_column.base_column.activation_level();
        
        self.hebbian.update_synaptic_weight(other_id, other_activation_time, other_strength)
    }
    
    /// Get current biological state
    pub fn biological_state(&self) -> BiologicalState {
        BiologicalState {
            membrane_voltage: self.membrane.current_voltage(),
            firing_threshold: self.refractory.current_firing_threshold(),
            time_since_fire_ms: self.refractory.time_since_last_fire_ms(),
            can_fire: self.refractory.can_fire(),
            synaptic_connections: self.hebbian.get_all_connections(),
        }
    }
    
    /// Access base column
    pub fn base(&self) -> &EnhancedCorticalColumn {
        &self.base_column
    }
}

#[derive(Debug, Clone)]
pub enum StimulationResult {
    Fired {
        fire_voltage: f32,
        threshold_used: f32,
        refractory_state: crate::RefractoryState,
    },
    SubThreshold {
        membrane_voltage: f32,
        threshold_required: f32,
        deficit: f32,
    },
    RefractoryBlock {
        time_remaining_ms: f32,
    },
}

#[derive(Debug, Clone)]
pub struct BiologicalState {
    pub membrane_voltage: f32,
    pub firing_threshold: f32,
    pub time_since_fire_ms: f32,
    pub can_fire: bool,
    pub synaptic_connections: std::collections::HashMap<u32, f32>,
}
```

## AI-Executable Test Suite

```rust
// tests/biological_activation_test.rs
use llmkg::{BiologicalCorticalColumn, BiologicalConfig, StimulationResult};
use std::thread;
use std::time::Duration;

#[test]
fn test_membrane_potential_decay() {
    let config = BiologicalConfig::default();
    let column = BiologicalCorticalColumn::new(1, config);
    
    // Apply strong stimulus
    column.stimulate(1.0, 5.0);
    let initial_voltage = column.biological_state().membrane_voltage;
    
    // Wait for decay
    thread::sleep(Duration::from_millis(50));
    let decayed_voltage = column.biological_state().membrane_voltage;
    
    // Should have decayed
    assert!(decayed_voltage < initial_voltage);
    assert!(decayed_voltage > 0.0); // But not to zero yet
}

#[test]
fn test_firing_threshold() {
    let config = BiologicalConfig::default();
    let column = BiologicalCorticalColumn::new(1, config);
    
    // Weak stimulus should not fire
    let result = column.stimulate(0.5, 1.0);
    assert!(matches!(result, StimulationResult::SubThreshold { .. }));
    
    // Strong stimulus should fire
    let result = column.stimulate(1.5, 1.0);
    assert!(matches!(result, StimulationResult::Fired { .. }));
}

#[test]
fn test_refractory_period() {
    let config = BiologicalConfig::default();
    let column = BiologicalCorticalColumn::new(1, config);
    
    // Fire the neuron
    let result = column.stimulate(1.5, 1.0);
    assert!(matches!(result, StimulationResult::Fired { .. }));
    
    // Immediate re-stimulation should be blocked
    let result = column.stimulate(1.5, 1.0);
    assert!(matches!(result, StimulationResult::RefractoryBlock { .. }));
    
    // Wait for refractory period to end
    thread::sleep(Duration::from_millis(5));
    let result = column.stimulate(1.5, 1.0);
    // Should either fire or be sub-threshold (depending on decay)
    assert!(!matches!(result, StimulationResult::RefractoryBlock { .. }));
}

#[test]
fn test_hebbian_learning() {
    let config = BiologicalConfig::default();
    let column1 = BiologicalCorticalColumn::new(1, config.clone());
    let column2 = BiologicalCorticalColumn::new(2, config);
    
    // Initial connection should be weak or non-existent
    let initial_weight = column1.biological_state().synaptic_connections.get(&2).copied().unwrap_or(0.0);
    
    // Stimulate both columns to fire
    column1.stimulate(1.5, 1.0);
    thread::sleep(Duration::from_millis(1)); // Small delay
    column2.stimulate(1.5, 1.0);
    
    // Learn from co-activation
    let learn_result = column1.learn_from_coactivation(&column2);
    assert!(matches!(learn_result, crate::HebbianUpdateResult::Updated { .. }));
    
    // Connection should have strengthened
    let final_weight = column1.biological_state().synaptic_connections.get(&2).copied().unwrap_or(0.0);
    assert!(final_weight > initial_weight);
}

#[test]
fn test_biological_timing_precision() {
    let config = BiologicalConfig::fast_processing();
    let column = BiologicalCorticalColumn::new(1, config);
    
    // Measure timing precision
    let start = std::time::Instant::now();
    column.stimulate(1.5, 1.0);
    let fire_time = start.elapsed();
    
    // Should be very fast (< 1ms)
    assert!(fire_time < Duration::from_millis(1));
    
    // Biological state should be consistent
    let state = column.biological_state();
    assert!(state.membrane_voltage >= 0.0);
    assert!(state.firing_threshold > 0.0);
    assert!(state.time_since_fire_ms >= 0.0);
}

#[test]
fn test_performance_biological_operations() {
    let config = BiologicalConfig::fast_processing();
    let column = BiologicalCorticalColumn::new(1, config);
    
    // Benchmark membrane updates
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        column.stimulate(0.5, 0.1); // Sub-threshold stimulation
    }
    let elapsed = start.elapsed();
    
    let ns_per_stimulation = elapsed.as_nanos() / 1000;
    println!("Biological stimulation: {} ns", ns_per_stimulation);
    
    // Should be fast (< 100ns per operation)
    assert!(ns_per_stimulation < 1000); // Allow margin for slower systems
}
```

## Success Criteria (AI Verification)

Your implementation is complete when:

1. **All tests pass**: 5/5 biological tests passing
2. **Performance targets met**:
   - Membrane update < 100ns
   - Stimulation processing < 1000ns  
   - Timing precision verified
3. **Biological accuracy**:
   - Exponential decay follows τ = 15ms ± 10%
   - Refractory periods enforced correctly
   - Hebbian learning increases weights
4. **Mathematical correctness**: Decay calculations match expected exponential curves

## Verification Commands

```bash
# Run biological tests
cargo test biological_activation_test --release -- --nocapture

# Performance verification
cargo test test_performance_biological_operations --release -- --nocapture

# Extended biological behavior test
cargo test --release -- --ignored biological_extended
```

## Files to Create

1. `src/biological_config.rs`
2. `src/membrane_potential.rs`  
3. `src/refractory_period.rs`
4. `src/hebbian_learning.rs`
5. `src/biological_cortical_column.rs`
6. `tests/biological_activation_test.rs`

## Expected Completion Time

3 hours for an AI assistant:
- 45 minutes: Biological configuration and membrane potential
- 60 minutes: Refractory period management  
- 45 minutes: Hebbian learning implementation
- 30 minutes: Integration and testing

## Next Task

Task 1.5: Exponential Decay (optimize mathematical calculations)