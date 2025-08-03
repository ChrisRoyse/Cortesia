# Phase 1 Foundation Types Specification

**Version**: 1.0  
**Date**: 2025-08-02  
**Purpose**: Comprehensive standardization of all core types used across Phase 1 tasks (1.1-1.14)

## Overview

This document defines all foundational types, utility functions, configuration structures, and error types used across Phase 1 tasks. It serves as the authoritative reference to ensure consistency and prevent undefined type errors throughout the implementation.

## Core Type Definitions

### 1. Identifier Types

#### ColumnId
Primary identifier for cortical columns throughout the system.

```rust
/// Unique identifier for cortical columns
/// - u32 provides 4.3 billion unique columns
/// - Efficient for spatial indexing operations
/// - Compatible with grid coordinate calculations
pub type ColumnId = u32;

impl ColumnId {
    /// Maximum valid column ID
    pub const MAX: Self = u32::MAX;
    
    /// Create from grid coordinates (x, y, z, dimensions)
    pub fn from_grid_coords(x: u32, y: u32, z: u32, width: u32, height: u32) -> Self {
        x + y * width + z * width * height
    }
    
    /// Convert to grid coordinates given dimensions
    pub fn to_grid_coords(self, width: u32, height: u32) -> (u32, u32, u32) {
        let z = self / (width * height);
        let remainder = self % (width * height);
        let y = remainder / width;
        let x = remainder % width;
        (x, y, z)
    }
}
```

#### NodeId and ConceptId
Distinguish between different entity types in the system.

```rust
/// Identifier for spatial indexing nodes (KD-tree, etc.)
pub type NodeId = u64;

/// Identifier for concepts in the knowledge system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConceptId(pub u64);

impl ConceptId {
    pub fn new() -> Self {
        Self(current_time_us())
    }
    
    pub fn from_string(s: &str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        Self(hasher.finish())
    }
    
    pub fn as_u64(self) -> u64 {
        self.0
    }
}
```

### 2. State Management Types

#### ColumnState
Complete state machine for cortical columns with validation.

```rust
/// Cortical column states with biological transitions
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColumnState {
    /// Ready for allocation (resting state)
    Available = 0,
    /// Currently processing input (excited state)
    Activated = 1,
    /// In lateral inhibition competition
    Competing = 2,
    /// Successfully allocated to a concept
    Allocated = 3,
    /// Temporarily unavailable after firing (refractory period)
    Refractory = 4,
}

impl ColumnState {
    /// Convert from u8 with validation
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Available),
            1 => Some(Self::Activated),
            2 => Some(Self::Competing),
            3 => Some(Self::Allocated),
            4 => Some(Self::Refractory),
            _ => None,
        }
    }
    
    /// Check if transition is biologically valid
    pub fn is_valid_transition(&self, to: Self) -> bool {
        use ColumnState::*;
        matches!(
            (self, to),
            // Normal activation path
            (Available, Activated) |
            (Activated, Competing) |
            (Competing, Allocated) |
            // Competition failure - back to available
            (Competing, Available) |
            // Allocation completion - enter refractory
            (Allocated, Refractory) |
            // Recovery from refractory
            (Refractory, Available) |
            // Emergency reset transitions
            (Activated, Available) |
            (Allocated, Available) |
            (Refractory, Activated) // For forced reactivation
        )
    }
    
    /// Get all valid next states
    pub fn valid_transitions(&self) -> &'static [ColumnState] {
        use ColumnState::*;
        match self {
            Available => &[Activated],
            Activated => &[Competing, Available],
            Competing => &[Allocated, Available],
            Allocated => &[Refractory, Available],
            Refractory => &[Available, Activated],
        }
    }
    
    /// Check if state allows input processing
    pub fn can_receive_input(&self) -> bool {
        matches!(self, ColumnState::Available | ColumnState::Activated)
    }
    
    /// Check if state is in active processing
    pub fn is_active(&self) -> bool {
        matches!(self, ColumnState::Activated | ColumnState::Competing | ColumnState::Allocated)
    }
}

impl Default for ColumnState {
    fn default() -> Self {
        Self::Available
    }
}

impl std::fmt::Display for ColumnState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColumnState::Available => write!(f, "Available"),
            ColumnState::Activated => write!(f, "Activated"),
            ColumnState::Competing => write!(f, "Competing"),
            ColumnState::Allocated => write!(f, "Allocated"),
            ColumnState::Refractory => write!(f, "Refractory"),
        }
    }
}
```

### 3. Spatial Types

#### Position3D
3D spatial coordinates for cortical grid topology.

```rust
/// 3D position in physical space (micrometers)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Position3D {
    /// Create new position
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    
    /// Origin position (0, 0, 0)
    pub const fn origin() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
    
    /// Calculate Euclidean distance to another position
    pub fn distance_to(&self, other: &Position3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
    
    /// Calculate squared distance (faster for comparisons)
    pub fn distance_squared_to(&self, other: &Position3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }
    
    /// Calculate Manhattan distance
    pub fn manhattan_distance_to(&self, other: &Position3D) -> f32 {
        (self.x - other.x).abs() + (self.y - other.y).abs() + (self.z - other.z).abs()
    }
    
    /// Check if position is within radius of another position
    pub fn is_within_radius(&self, other: &Position3D, radius: f32) -> bool {
        self.distance_squared_to(other) <= radius * radius
    }
    
    /// Linear interpolation between two positions
    pub fn lerp(&self, other: &Position3D, t: f32) -> Position3D {
        Position3D::new(
            self.x + t * (other.x - self.x),
            self.y + t * (other.y - self.y),
            self.z + t * (other.z - self.z),
        )
    }
    
    /// Add another position (vector addition)
    pub fn add(&self, other: &Position3D) -> Position3D {
        Position3D::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
    
    /// Subtract another position (vector subtraction)
    pub fn sub(&self, other: &Position3D) -> Position3D {
        Position3D::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
    
    /// Scale by a factor
    pub fn scale(&self, factor: f32) -> Position3D {
        Position3D::new(self.x * factor, self.y * factor, self.z * factor)
    }
    
    /// Get magnitude (distance from origin)
    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
    
    /// Normalize to unit vector
    pub fn normalize(&self) -> Position3D {
        let mag = self.magnitude();
        if mag > 0.0 {
            Position3D::new(self.x / mag, self.y / mag, self.z / mag)
        } else {
            *self
        }
    }
}

impl std::ops::Add for Position3D {
    type Output = Self;
    
    fn add(self, other: Self) -> Self {
        self.add(&other)
    }
}

impl std::ops::Sub for Position3D {
    type Output = Self;
    
    fn sub(self, other: Self) -> Self {
        self.sub(&other)
    }
}

impl std::ops::Mul<f32> for Position3D {
    type Output = Self;
    
    fn mul(self, factor: f32) -> Self {
        self.scale(factor)
    }
}
```

## Utility Functions

### Time Functions

```rust
/// Get current time in microseconds since Unix epoch
/// Provides precise timing for performance measurement and biological simulation
pub fn current_time_us() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// Get current time in nanoseconds since Unix epoch
/// For high-precision performance measurements
pub fn current_time_ns() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Convert milliseconds to microseconds
pub fn ms_to_us(ms: f32) -> u64 {
    (ms * 1000.0) as u64
}

/// Convert microseconds to milliseconds
pub fn us_to_ms(us: u64) -> f32 {
    us as f32 / 1000.0
}

/// Convert nanoseconds to microseconds
pub fn ns_to_us(ns: u64) -> u64 {
    ns / 1000
}

/// Convert microseconds to nanoseconds
pub fn us_to_ns(us: u64) -> u64 {
    us * 1000
}

/// High-precision sleep for timing-critical operations
pub fn precision_sleep_us(microseconds: u64) {
    use std::time::Duration;
    
    if microseconds > 0 {
        std::thread::sleep(Duration::from_micros(microseconds));
    }
}

/// Calculate time difference safely (handles wraparound)
pub fn time_diff_us(later: u64, earlier: u64) -> u64 {
    later.saturating_sub(earlier)
}
```

### Conversion Utilities

```rust
/// Convert between floating-point and atomic representation
pub mod atomic_float {
    use std::sync::atomic::{AtomicU32, Ordering};
    
    /// Atomically load f32 from AtomicU32
    pub fn load_f32(atomic: &AtomicU32, ordering: Ordering) -> f32 {
        f32::from_bits(atomic.load(ordering))
    }
    
    /// Atomically store f32 to AtomicU32
    pub fn store_f32(atomic: &AtomicU32, value: f32, ordering: Ordering) {
        atomic.store(value.to_bits(), ordering);
    }
    
    /// Atomic compare-and-swap for f32
    pub fn compare_exchange_f32(
        atomic: &AtomicU32,
        current: f32,
        new: f32,
        success: Ordering,
        failure: Ordering,
    ) -> Result<f32, f32> {
        match atomic.compare_exchange(current.to_bits(), new.to_bits(), success, failure) {
            Ok(bits) => Ok(f32::from_bits(bits)),
            Err(bits) => Err(f32::from_bits(bits)),
        }
    }
}

/// Mathematical utility functions
pub mod math_utils {
    /// Fast exponential approximation for decay calculations
    pub fn fast_exp(x: f32) -> f32 {
        if x >= 0.0 {
            return 1.0;
        }
        
        // Use lookup table for common decay values
        const LOOKUP_SIZE: usize = 1024;
        const MAX_X: f32 = 10.0;
        
        static LOOKUP_TABLE: std::sync::LazyLock<[f32; LOOKUP_SIZE]> = std::sync::LazyLock::new(|| {
            let mut table = [0.0f32; LOOKUP_SIZE];
            for i in 0..LOOKUP_SIZE {
                let x = (i as f32 / LOOKUP_SIZE as f32) * MAX_X;
                table[i] = (-x).exp();
            }
            table
        });
        
        let abs_x = -x;
        if abs_x >= MAX_X {
            return 0.0;
        }
        
        let index = (abs_x / MAX_X * LOOKUP_SIZE as f32) as usize;
        if index >= LOOKUP_SIZE {
            return 0.0;
        }
        
        LOOKUP_TABLE[index]
    }
    
    /// Clamp value to range [min, max]
    pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
        if value < min {
            min
        } else if value > max {
            max
        } else {
            value
        }
    }
    
    /// Linear interpolation
    pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + t * (b - a)
    }
    
    /// Normalize value from range [old_min, old_max] to [new_min, new_max]
    pub fn normalize_range(value: f32, old_min: f32, old_max: f32, new_min: f32, new_max: f32) -> f32 {
        let normalized = (value - old_min) / (old_max - old_min);
        lerp(new_min, new_max, normalized)
    }
}
```

## Configuration Structures

### BiologicalConfig
Configuration for biological simulation parameters.

```rust
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
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.membrane_tau_ms <= 0.0 {
            return Err("membrane_tau_ms must be positive".to_string());
        }
        if self.activation_tau_ms <= 0.0 {
            return Err("activation_tau_ms must be positive".to_string());
        }
        if self.firing_threshold <= 0.0 || self.firing_threshold > 2.0 {
            return Err("firing_threshold must be in range (0.0, 2.0]".to_string());
        }
        if self.absolute_refractory_ms < 0.0 {
            return Err("absolute_refractory_ms must be non-negative".to_string());
        }
        if self.relative_refractory_ms < self.absolute_refractory_ms {
            return Err("relative_refractory_ms must be >= absolute_refractory_ms".to_string());
        }
        if self.hebbian_learning_rate < 0.0 || self.hebbian_learning_rate > 1.0 {
            return Err("hebbian_learning_rate must be in range [0.0, 1.0]".to_string());
        }
        if self.max_synaptic_weight <= self.min_synaptic_weight {
            return Err("max_synaptic_weight must be > min_synaptic_weight".to_string());
        }
        Ok(())
    }
}
```

### InhibitionConfig
Configuration for lateral inhibition mechanisms.

```rust
/// Configuration for lateral inhibition network
#[derive(Debug, Clone)]
pub struct InhibitionConfig {
    /// Inhibition strength (0.0 = no inhibition, 1.0 = complete suppression)
    pub inhibition_strength: f32,
    
    /// Inhibition radius in spatial units (micrometers)
    pub inhibition_radius: f32,
    
    /// Maximum number of competing columns
    pub max_competitors: usize,
    
    /// Convergence threshold for inhibition settling
    pub convergence_threshold: f32,
    
    /// Maximum iterations for inhibition convergence
    pub max_iterations: usize,
    
    /// Inhibition decay time constant (ms)
    pub inhibition_tau_ms: f32,
    
    /// Winner-take-all threshold
    pub winner_threshold: f32,
    
    /// Minimum activation required to participate in competition
    pub min_competition_activation: f32,
    
    /// Gaussian falloff sigma for distance-based inhibition
    pub distance_sigma: f32,
    
    /// Enable adaptive inhibition strength
    pub adaptive_inhibition: bool,
}

impl Default for InhibitionConfig {
    fn default() -> Self {
        Self {
            inhibition_strength: 0.8,
            inhibition_radius: 200.0,
            max_competitors: 50,
            convergence_threshold: 0.001,
            max_iterations: 100,
            inhibition_tau_ms: 5.0,
            winner_threshold: 0.7,
            min_competition_activation: 0.1,
            distance_sigma: 100.0,
            adaptive_inhibition: true,
        }
    }
}

impl InhibitionConfig {
    /// Configuration for strong competition (high selectivity)
    pub fn strong_competition() -> Self {
        Self {
            inhibition_strength: 0.9,
            winner_threshold: 0.8,
            max_competitors: 25,
            convergence_threshold: 0.0005,
            ..Default::default()
        }
    }
    
    /// Configuration for weak competition (more distributed activation)
    pub fn weak_competition() -> Self {
        Self {
            inhibition_strength: 0.5,
            winner_threshold: 0.5,
            max_competitors: 100,
            convergence_threshold: 0.005,
            ..Default::default()
        }
    }
    
    /// Fast processing configuration
    pub fn fast_processing() -> Self {
        Self {
            max_iterations: 50,
            convergence_threshold: 0.01,
            inhibition_tau_ms: 2.0,
            adaptive_inhibition: false,
            ..Default::default()
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.inhibition_strength < 0.0 || self.inhibition_strength > 1.0 {
            return Err("inhibition_strength must be in range [0.0, 1.0]".to_string());
        }
        if self.inhibition_radius <= 0.0 {
            return Err("inhibition_radius must be positive".to_string());
        }
        if self.max_competitors == 0 {
            return Err("max_competitors must be positive".to_string());
        }
        if self.convergence_threshold <= 0.0 {
            return Err("convergence_threshold must be positive".to_string());
        }
        if self.max_iterations == 0 {
            return Err("max_iterations must be positive".to_string());
        }
        if self.winner_threshold < 0.0 || self.winner_threshold > 1.0 {
            return Err("winner_threshold must be in range [0.0, 1.0]".to_string());
        }
        Ok(())
    }
}
```

### ScalabilityConfig
Configuration for performance tuning and scalability.

```rust
/// Configuration for performance tuning and scalability
#[derive(Debug, Clone)]
pub struct ScalabilityConfig {
    /// Number of worker threads for parallel processing
    pub worker_threads: usize,
    
    /// Batch size for bulk operations
    pub batch_size: usize,
    
    /// Maximum queue size for allocation requests
    pub max_queue_size: usize,
    
    /// Cache size for neighbor lookups
    pub neighbor_cache_size: usize,
    
    /// Enable SIMD acceleration where available
    pub enable_simd: bool,
    
    /// Prefetch distance for spatial queries
    pub prefetch_distance: usize,
    
    /// Memory pool sizes
    pub column_pool_size: usize,
    pub connection_pool_size: usize,
    
    /// Performance monitoring configuration
    pub enable_performance_monitoring: bool,
    pub monitoring_sample_rate: f32, // 0.0 to 1.0
    
    /// Adaptive scaling configuration
    pub enable_adaptive_scaling: bool,
    pub scale_up_threshold: f32,     // CPU utilization threshold to scale up
    pub scale_down_threshold: f32,   // CPU utilization threshold to scale down
    
    /// Memory management
    pub enable_memory_compression: bool,
    pub gc_trigger_threshold: f32,   // Memory usage threshold to trigger cleanup
}

impl Default for ScalabilityConfig {
    fn default() -> Self {
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        
        Self {
            worker_threads: cpu_count,
            batch_size: 64,
            max_queue_size: 10_000,
            neighbor_cache_size: 1_000,
            enable_simd: true,
            prefetch_distance: 8,
            column_pool_size: 100_000,
            connection_pool_size: 1_000_000,
            enable_performance_monitoring: true,
            monitoring_sample_rate: 0.1, // 10% sampling
            enable_adaptive_scaling: true,
            scale_up_threshold: 0.8,     // 80% CPU
            scale_down_threshold: 0.3,   // 30% CPU
            enable_memory_compression: false,
            gc_trigger_threshold: 0.9,   // 90% memory usage
        }
    }
}

impl ScalabilityConfig {
    /// High-performance configuration
    pub fn high_performance() -> Self {
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        
        Self {
            worker_threads: cpu_count * 2, // Oversubscribe for I/O bound tasks
            batch_size: 128,
            max_queue_size: 50_000,
            neighbor_cache_size: 10_000,
            enable_simd: true,
            prefetch_distance: 16,
            column_pool_size: 1_000_000,
            connection_pool_size: 10_000_000,
            monitoring_sample_rate: 0.05, // Reduced overhead
            enable_memory_compression: true,
            ..Default::default()
        }
    }
    
    /// Memory-optimized configuration
    pub fn memory_optimized() -> Self {
        Self {
            worker_threads: 2,
            batch_size: 32,
            max_queue_size: 1_000,
            neighbor_cache_size: 100,
            column_pool_size: 10_000,
            connection_pool_size: 100_000,
            enable_memory_compression: true,
            gc_trigger_threshold: 0.7, // More aggressive cleanup
            monitoring_sample_rate: 0.01, // Minimal monitoring
            ..Default::default()
        }
    }
    
    /// Development/testing configuration
    pub fn development() -> Self {
        Self {
            worker_threads: 1,
            batch_size: 16,
            max_queue_size: 100,
            neighbor_cache_size: 50,
            column_pool_size: 1_000,
            connection_pool_size: 10_000,
            enable_performance_monitoring: true,
            monitoring_sample_rate: 1.0, // Full monitoring for debugging
            enable_adaptive_scaling: false,
            enable_memory_compression: false,
            ..Default::default()
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.worker_threads == 0 {
            return Err("worker_threads must be positive".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be positive".to_string());
        }
        if self.max_queue_size == 0 {
            return Err("max_queue_size must be positive".to_string());
        }
        if self.monitoring_sample_rate < 0.0 || self.monitoring_sample_rate > 1.0 {
            return Err("monitoring_sample_rate must be in range [0.0, 1.0]".to_string());
        }
        if self.scale_up_threshold <= self.scale_down_threshold {
            return Err("scale_up_threshold must be > scale_down_threshold".to_string());
        }
        if self.gc_trigger_threshold < 0.0 || self.gc_trigger_threshold > 1.0 {
            return Err("gc_trigger_threshold must be in range [0.0, 1.0]".to_string());
        }
        Ok(())
    }
}
```

## Error Types

### StateTransitionError
Complete error handling for state machine operations.

```rust
/// Errors that can occur during column state transitions
#[derive(Debug, Clone, thiserror::Error)]
pub enum StateTransitionError {
    #[error("Invalid transition from {from:?} to {to:?}")]
    InvalidTransition { from: ColumnState, to: ColumnState },
    
    #[error("State mismatch: expected {expected:?}, found {actual:?}")]
    StateMismatch { expected: ColumnState, actual: ColumnState },
    
    #[error("Concurrent modification detected during transition")]
    ConcurrentModification,
    
    #[error("Transition timeout after {timeout_ms}ms")]
    TransitionTimeout { timeout_ms: u64 },
    
    #[error("Column {column_id} is in invalid state")]
    InvalidColumnState { column_id: ColumnId },
    
    #[error("Column {column_id} is locked by another operation")]
    ColumnLocked { column_id: ColumnId },
    
    #[error("Maximum retry attempts ({max_retries}) exceeded")]
    MaxRetriesExceeded { max_retries: u32 },
    
    #[error("Column {column_id} not found in grid")]
    ColumnNotFound { column_id: ColumnId },
    
    #[error("State validation failed: {reason}")]
    ValidationFailed { reason: String },
}

impl StateTransitionError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            StateTransitionError::ConcurrentModification |
            StateTransitionError::StateMismatch { .. } |
            StateTransitionError::ColumnLocked { .. }
        )
    }
    
    /// Check if error indicates a system-level problem
    pub fn is_system_error(&self) -> bool {
        matches!(
            self,
            StateTransitionError::TransitionTimeout { .. } |
            StateTransitionError::MaxRetriesExceeded { .. } |
            StateTransitionError::ColumnNotFound { .. }
        )
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            StateTransitionError::InvalidTransition { .. } => ErrorSeverity::Warning,
            StateTransitionError::StateMismatch { .. } => ErrorSeverity::Info,
            StateTransitionError::ConcurrentModification => ErrorSeverity::Info,
            StateTransitionError::TransitionTimeout { .. } => ErrorSeverity::Error,
            StateTransitionError::InvalidColumnState { .. } => ErrorSeverity::Error,
            StateTransitionError::ColumnLocked { .. } => ErrorSeverity::Warning,
            StateTransitionError::MaxRetriesExceeded { .. } => ErrorSeverity::Error,
            StateTransitionError::ColumnNotFound { .. } => ErrorSeverity::Error,
            StateTransitionError::ValidationFailed { .. } => ErrorSeverity::Warning,
        }
    }
}
```

### AllocationError
Errors for allocation engine operations.

```rust
/// Errors that can occur during concept allocation
#[derive(Debug, Clone, thiserror::Error)]
pub enum AllocationError {
    #[error("No suitable column found for concept {concept_id}")]
    NoSuitableColumn { concept_id: ConceptId },
    
    #[error("Allocation queue is full (capacity: {capacity})")]
    QueueFull { capacity: usize },
    
    #[error("Neural network inference failed: {reason}")]
    NeuralInferenceFailed { reason: String },
    
    #[error("Spatial indexing error: {reason}")]
    SpatialIndexingError { reason: String },
    
    #[error("Lateral inhibition convergence failed after {iterations} iterations")]
    InhibitionConvergenceFailed { iterations: usize },
    
    #[error("Winner-take-all selection failed: no clear winner")]
    WinnerSelectionFailed,
    
    #[error("Concept {concept_id} already allocated at position {position:?}")]
    ConceptAlreadyAllocated { concept_id: ConceptId, position: Position3D },
    
    #[error("Memory allocation failed: {reason}")]
    MemoryAllocationFailed { reason: String },
    
    #[error("Performance threshold exceeded: {metric} = {value} > {threshold}")]
    PerformanceThresholdExceeded { metric: String, value: f64, threshold: f64 },
    
    #[error("Grid topology error: {reason}")]
    GridTopologyError { reason: String },
    
    #[error("Configuration validation error: {reason}")]
    ConfigurationError { reason: String },
}

impl AllocationError {
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            AllocationError::QueueFull { .. } |
            AllocationError::InhibitionConvergenceFailed { .. } |
            AllocationError::WinnerSelectionFailed |
            AllocationError::MemoryAllocationFailed { .. }
        )
    }
    
    /// Get recommended retry delay in milliseconds
    pub fn retry_delay_ms(&self) -> u64 {
        match self {
            AllocationError::QueueFull { .. } => 100,
            AllocationError::InhibitionConvergenceFailed { .. } => 50,
            AllocationError::WinnerSelectionFailed => 10,
            AllocationError::MemoryAllocationFailed { .. } => 1000,
            _ => 0,
        }
    }
}
```

### ErrorSeverity
Classification of error severity levels.

```rust
/// Error severity levels for logging and handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Informational - normal operation variations
    Info,
    /// Warning - potential issues that don't affect functionality
    Warning,
    /// Error - operation failed but system remains stable
    Error,
    /// Critical - system stability at risk
    Critical,
}

impl ErrorSeverity {
    /// Check if error should trigger system shutdown
    pub fn requires_shutdown(&self) -> bool {
        matches!(self, ErrorSeverity::Critical)
    }
    
    /// Check if error should be logged
    pub fn should_log(&self) -> bool {
        *self >= ErrorSeverity::Warning
    }
    
    /// Get log level string
    pub fn log_level(&self) -> &'static str {
        match self {
            ErrorSeverity::Info => "INFO",
            ErrorSeverity::Warning => "WARN",
            ErrorSeverity::Error => "ERROR",
            ErrorSeverity::Critical => "CRITICAL",
        }
    }
}

impl std::fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.log_level())
    }
}
```

## Performance Types

### Metrics and Statistics

```rust
/// Performance metrics for allocation operations
#[derive(Debug, Clone, Default)]
pub struct AllocationMetrics {
    /// Total number of allocations attempted
    pub total_allocations: u64,
    
    /// Number of successful allocations
    pub successful_allocations: u64,
    
    /// Number of failed allocations
    pub failed_allocations: u64,
    
    /// Total time spent in allocation (nanoseconds)
    pub total_allocation_time_ns: u64,
    
    /// Total time spent in neural inference (nanoseconds)
    pub total_inference_time_ns: u64,
    
    /// Total time spent in spatial queries (nanoseconds)
    pub total_spatial_time_ns: u64,
    
    /// Total time spent in inhibition (nanoseconds)
    pub total_inhibition_time_ns: u64,
    
    /// Peak queue size observed
    pub peak_queue_size: usize,
    
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
}

impl AllocationMetrics {
    /// Calculate average allocation time
    pub fn average_allocation_time_ns(&self) -> u64 {
        if self.total_allocations > 0 {
            self.total_allocation_time_ns / self.total_allocations
        } else {
            0
        }
    }
    
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_allocations > 0 {
            self.successful_allocations as f64 / self.total_allocations as f64
        } else {
            0.0
        }
    }
    
    /// Calculate throughput (allocations per second)
    pub fn throughput_per_second(&self) -> f64 {
        if self.total_allocation_time_ns > 0 {
            (self.total_allocations as f64 * 1_000_000_000.0) / self.total_allocation_time_ns as f64
        } else {
            0.0
        }
    }
    
    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Default::default();
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Current memory usage in bytes
    pub current_usage_bytes: usize,
    
    /// Peak memory usage in bytes
    pub peak_usage_bytes: usize,
    
    /// Number of memory allocations
    pub allocation_count: u64,
    
    /// Number of memory deallocations
    pub deallocation_count: u64,
    
    /// Memory pool statistics
    pub pool_usage: f32, // 0.0 to 1.0
    
    /// Cache hit rate
    pub cache_hit_rate: f32, // 0.0 to 1.0
}

impl MemoryStats {
    /// Calculate memory efficiency
    pub fn efficiency(&self) -> f32 {
        if self.peak_usage_bytes > 0 {
            self.current_usage_bytes as f32 / self.peak_usage_bytes as f32
        } else {
            1.0
        }
    }
    
    /// Check if memory usage is within acceptable range
    pub fn is_healthy(&self, max_usage_bytes: usize) -> bool {
        self.current_usage_bytes <= max_usage_bytes && 
        self.pool_usage <= 0.9 && 
        self.cache_hit_rate >= 0.7
    }
}

/// Benchmark result for performance testing
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Test name
    pub name: String,
    
    /// Number of operations performed
    pub operations: u64,
    
    /// Total time elapsed (nanoseconds)
    pub total_time_ns: u64,
    
    /// Average time per operation (nanoseconds)
    pub avg_time_ns: u64,
    
    /// Minimum time observed (nanoseconds)
    pub min_time_ns: u64,
    
    /// Maximum time observed (nanoseconds)
    pub max_time_ns: u64,
    
    /// Standard deviation (nanoseconds)
    pub std_dev_ns: f64,
    
    /// Memory usage during benchmark
    pub memory_usage: MemoryStats,
}

impl BenchmarkResult {
    /// Calculate operations per second
    pub fn ops_per_second(&self) -> f64 {
        if self.total_time_ns > 0 {
            (self.operations as f64 * 1_000_000_000.0) / self.total_time_ns as f64
        } else {
            0.0
        }
    }
    
    /// Check if benchmark meets performance target
    pub fn meets_target(&self, target_ops_per_second: f64) -> bool {
        self.ops_per_second() >= target_ops_per_second
    }
    
    /// Calculate P99 latency estimate (assuming normal distribution)
    pub fn p99_latency_ns(&self) -> u64 {
        (self.avg_time_ns as f64 + 2.33 * self.std_dev_ns) as u64
    }
}
```

## Constants and Type Aliases

```rust
/// Common constants used throughout Phase 1
pub mod constants {
    /// Default grid dimensions
    pub const DEFAULT_GRID_WIDTH: u32 = 100;
    pub const DEFAULT_GRID_HEIGHT: u32 = 100;
    pub const DEFAULT_GRID_DEPTH: u32 = 6;
    
    /// Physical dimensions
    pub const DEFAULT_COLUMN_SPACING_UM: f32 = 50.0;
    pub const DEFAULT_CONNECTION_RADIUS_UM: f32 = 300.0;
    
    /// Performance targets
    pub const TARGET_ALLOCATION_THROUGHPUT: f64 = 500.0; // allocations/second
    pub const TARGET_P99_LATENCY_MS: f32 = 20.0; // milliseconds
    pub const TARGET_MEMORY_PER_COLUMN_BYTES: usize = 512; // bytes
    
    /// Biological parameters
    pub const CORTICAL_NEURON_TAU_MS: f32 = 15.0;
    pub const DEFAULT_FIRING_THRESHOLD: f32 = 0.8;
    pub const DEFAULT_REFRACTORY_PERIOD_MS: f32 = 2.0;
    
    /// System limits
    pub const MAX_COLUMNS_PER_GRID: u32 = 10_000_000; // 10M columns
    pub const MAX_CONNECTIONS_PER_COLUMN: usize = 1000;
    pub const MAX_QUEUE_SIZE: usize = 100_000;
    
    /// Neural network parameters
    pub const MLP_INPUT_SIZE: usize = 512;
    pub const LSTM_INPUT_SIZE: usize = 512;
    pub const TCN_INPUT_SIZE: usize = 512;
    
    /// Time precision
    pub const TIME_PRECISION_US: u64 = 1; // 1 microsecond precision
    pub const PERFORMANCE_SAMPLE_INTERVAL_MS: u64 = 100; // 100ms sampling
}

/// Type aliases for commonly used types
pub type AllocationResult<T> = Result<T, AllocationError>;
pub type StateResult<T> = Result<T, StateTransitionError>;
pub type ConfigValidationResult = Result<(), String>;

/// Feature vector type for neural networks
pub type FeatureVector = Vec<f32>;

/// Connection strength type
pub type ConnectionStrength = f32;

/// Timestamp type for high-precision timing
pub type Timestamp = u64;

/// Distance type for spatial calculations
pub type Distance = f32;

/// Activation level type for neural activation
pub type ActivationLevel = f32;
```

## Usage Examples

### Basic Column Operations

```rust
use foundation_types::*;

// Create a new column
let column_id = ColumnId::from_grid_coords(10, 20, 3, 100, 100);
let mut state = ColumnState::Available;

// Validate and perform state transition
if state.is_valid_transition(ColumnState::Activated) {
    state = ColumnState::Activated;
    println!("Column {} activated", column_id);
}

// Check current capabilities
if state.can_receive_input() {
    println!("Column can receive input");
}
```

### Spatial Operations

```rust
use foundation_types::*;

// Create positions
let pos1 = Position3D::new(100.0, 200.0, 50.0);
let pos2 = Position3D::new(150.0, 180.0, 60.0);

// Calculate distance
let distance = pos1.distance_to(&pos2);
println!("Distance: {:.2} μm", distance);

// Check proximity
if pos1.is_within_radius(&pos2, 100.0) {
    println!("Positions are within inhibition radius");
}
```

### Configuration Management

```rust
use foundation_types::*;

// Create biological configuration
let bio_config = BiologicalConfig::cortical_neuron();
if let Err(e) = bio_config.validate() {
    eprintln!("Configuration error: {}", e);
}

// Create scalability configuration
let scale_config = ScalabilityConfig::high_performance();
println!("Using {} worker threads", scale_config.worker_threads);
```

### Performance Monitoring

```rust
use foundation_types::*;

// Create metrics tracking
let mut metrics = AllocationMetrics::default();

// Simulate allocation timing
let start_time = current_time_ns();
// ... perform allocation ...
let end_time = current_time_ns();

metrics.total_allocations += 1;
metrics.total_allocation_time_ns += end_time - start_time;

println!("Average allocation time: {} ns", metrics.average_allocation_time_ns());
println!("Throughput: {:.1} allocations/second", metrics.throughput_per_second());
```

## Integration Notes

### Thread Safety
- All atomic types use appropriate memory ordering
- Configuration structures are immutable after creation
- Error types implement `Send + Sync` for multi-threaded use

### Memory Management
- Position3D is `Copy` for efficient passing
- Large configuration structures use `Arc` for sharing
- Metrics use atomic operations for lock-free updates

### Performance Considerations
- Fast math utilities use lookup tables for exponential functions
- Time functions are optimized for minimal overhead
- Type conversions are zero-cost where possible

### Compatibility
- All types use standard Rust conventions
- Error types integrate with `thiserror` for ergonomic error handling
- Configuration validation prevents runtime errors

## Version History

- **v1.0** (2025-08-02): Initial specification with all Phase 1 types defined
- Comprehensive type definitions for Tasks 1.1-1.14
- Complete error handling framework
- Performance monitoring types
- Configuration management system

## Dependencies

This specification requires the following external crates:
- `thiserror` for error handling
- `std::sync` for atomic operations
- `std::time` for timing functions

All types are designed to be self-contained and minimize external dependencies while providing comprehensive functionality for Phase 1 implementation.