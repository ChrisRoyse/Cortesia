# Task 15: TTFS Encoder Base Structure

## Metadata
- **Micro-Phase**: 2.15
- **Duration**: 20-25 minutes
- **Dependencies**: Task 13 (neuromorphic_concept)
- **Output**: `src/ttfs_encoding/ttfs_encoder_base.rs`

## Description
Create the base TTFS encoder structure and configuration system that provides the foundation for neuromorphic concept encoding. This establishes the core architecture for transforming concepts into spike patterns with biological accuracy.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttfs_encoding::ConceptId;
    use std::time::Duration;

    #[test]
    fn test_ttfs_encoder_creation() {
        let config = TTFSConfig::default();
        let encoder = TTFSEncoder::new(config);
        
        assert_eq!(encoder.neuron_count(), 1024);
        assert_eq!(encoder.max_spike_time(), Duration::from_millis(10));
        assert_eq!(encoder.refractory_period(), Duration::from_micros(100));
        assert!(encoder.simd_enabled());
    }
    
    #[test]
    fn test_ttfs_config_validation() {
        let mut config = TTFSConfig::default();
        assert!(config.validate().is_ok());
        
        // Test invalid configurations
        config.neuron_count = 0;
        assert!(config.validate().is_err());
        
        config = TTFSConfig::default();
        config.refractory_period = Duration::from_nanos(50); // Too short
        assert!(config.validate().is_err());
        
        config = TTFSConfig::default();
        config.max_spike_time = Duration::from_secs(1); // Too long
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_encoder_state_management() {
        let config = TTFSConfig::default();
        let mut encoder = TTFSEncoder::new(config);
        
        assert!(encoder.is_ready());
        assert_eq!(encoder.encoding_count(), 0);
        
        // Simulate encoding operation
        encoder.begin_encoding();
        assert!(!encoder.is_ready());
        
        encoder.complete_encoding();
        assert!(encoder.is_ready());
        assert_eq!(encoder.encoding_count(), 1);
    }
    
    #[test]
    fn test_neuron_allocation() {
        let config = TTFSConfig::default();
        let mut encoder = TTFSEncoder::new(config);
        
        let concept_id = ConceptId::new("test_concept");
        let allocated_neurons = encoder.allocate_neurons(&concept_id, 64);
        
        assert_eq!(allocated_neurons.len(), 64);
        assert!(allocated_neurons.iter().all(|&id| id.0 < 1024));
        
        // Verify no duplicates
        let mut sorted = allocated_neurons.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), allocated_neurons.len());
    }
    
    #[test]
    fn test_timing_configuration() {
        let mut config = TTFSConfig::default();
        config.timing_precision = TimingPrecision::Nanosecond;
        config.max_spike_time = Duration::from_millis(5);
        
        let encoder = TTFSEncoder::new(config);
        assert_eq!(encoder.timing_resolution_ns(), 1);
        assert_eq!(encoder.max_spike_time(), Duration::from_millis(5));
    }
    
    #[test]
    fn test_performance_settings() {
        let mut config = TTFSConfig::high_performance();
        let encoder = TTFSEncoder::new(config);
        
        assert!(encoder.simd_enabled());
        assert!(encoder.parallel_processing_enabled());
        assert_eq!(encoder.cache_size(), 10000);
        
        config = TTFSConfig::low_latency();
        let encoder = TTFSEncoder::new(config);
        
        assert!(encoder.is_optimized_for_latency());
        assert!(encoder.max_spike_time() < Duration::from_millis(1));
    }
    
    #[test]
    fn test_biological_parameters() {
        let config = TTFSConfig::biological_accurate();
        let encoder = TTFSEncoder::new(config);
        
        assert_eq!(encoder.refractory_period(), Duration::from_micros(100));
        assert!(encoder.membrane_dynamics_enabled());
        assert!(encoder.adaptation_enabled());
        
        let bio_params = encoder.biological_parameters();
        assert!(bio_params.resting_potential > -80.0);
        assert!(bio_params.threshold_potential > -50.0);
        assert!(bio_params.leak_conductance > 0.0);
    }
    
    #[test]
    fn test_encoder_metrics() {
        let config = TTFSConfig::default();
        let encoder = TTFSEncoder::new(config);
        
        let metrics = encoder.performance_metrics();
        assert_eq!(metrics.total_encodings, 0);
        assert_eq!(metrics.failed_encodings, 0);
        assert!(metrics.average_encoding_time.is_zero());
        assert!(metrics.cache_hit_rate >= 0.0);
    }
}
```

## Implementation
```rust
use crate::ttfs_encoding::{ConceptId, NeuronId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Timing precision settings for TTFS encoding
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TimingPrecision {
    /// Nanosecond precision (1ns resolution)
    Nanosecond,
    /// Microsecond precision (1μs resolution)
    Microsecond,
    /// 100-nanosecond precision (100ns resolution)
    HundredNanosecond,
}

impl TimingPrecision {
    /// Get resolution in nanoseconds
    pub fn resolution_ns(&self) -> u64 {
        match self {
            TimingPrecision::Nanosecond => 1,
            TimingPrecision::HundredNanosecond => 100,
            TimingPrecision::Microsecond => 1000,
        }
    }
}

/// Encoding optimization strategy
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Optimize for minimal latency (<1ms)
    LowLatency,
    /// Optimize for high throughput
    HighThroughput,
    /// Optimize for biological accuracy
    BiologicalAccuracy,
    /// Balanced optimization
    Balanced,
}

/// Biological parameters for neuromorphic encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalParameters {
    /// Resting membrane potential (mV)
    pub resting_potential: f32,
    /// Threshold potential for spike generation (mV)
    pub threshold_potential: f32,
    /// Membrane leak conductance (nS)
    pub leak_conductance: f32,
    /// Membrane capacitance (pF)
    pub membrane_capacitance: f32,
    /// Spike adaptation rate
    pub adaptation_rate: f32,
    /// Noise variance
    pub noise_variance: f32,
}

impl Default for BiologicalParameters {
    fn default() -> Self {
        Self {
            resting_potential: -70.0,
            threshold_potential: -50.0,
            leak_conductance: 10.0,
            membrane_capacitance: 100.0,
            adaptation_rate: 0.02,
            noise_variance: 0.1,
        }
    }
}

/// Performance metrics for TTFS encoder
#[derive(Debug, Clone, Default)]
pub struct EncoderMetrics {
    /// Total number of encoding operations
    pub total_encodings: u64,
    /// Number of failed encoding operations
    pub failed_encodings: u64,
    /// Average encoding time
    pub average_encoding_time: Duration,
    /// Cache hit rate (0.0-1.0)
    pub cache_hit_rate: f32,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
    /// SIMD acceleration usage rate
    pub simd_usage_rate: f32,
}

/// Configuration for TTFS encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTFSConfig {
    /// Number of neurons available for encoding
    pub neuron_count: usize,
    
    /// Maximum spike time for any pattern
    pub max_spike_time: Duration,
    
    /// Refractory period between spikes
    pub refractory_period: Duration,
    
    /// Timing precision for spike events
    pub timing_precision: TimingPrecision,
    
    /// Optimization strategy
    pub optimization_strategy: OptimizationStrategy,
    
    /// Enable SIMD acceleration
    pub enable_simd: bool,
    
    /// Enable parallel processing
    pub enable_parallel: bool,
    
    /// Cache size for pattern storage
    pub cache_size: usize,
    
    /// Biological parameters
    pub biological_params: BiologicalParameters,
    
    /// Enable membrane dynamics simulation
    pub enable_membrane_dynamics: bool,
    
    /// Enable spike frequency adaptation
    pub enable_adaptation: bool,
    
    /// Maximum neurons per concept
    pub max_neurons_per_concept: usize,
    
    /// Encoding timeout
    pub encoding_timeout: Duration,
}

impl Default for TTFSConfig {
    fn default() -> Self {
        Self {
            neuron_count: 1024,
            max_spike_time: Duration::from_millis(10),
            refractory_period: Duration::from_micros(100),
            timing_precision: TimingPrecision::HundredNanosecond,
            optimization_strategy: OptimizationStrategy::Balanced,
            enable_simd: true,
            enable_parallel: true,
            cache_size: 1000,
            biological_params: BiologicalParameters::default(),
            enable_membrane_dynamics: true,
            enable_adaptation: true,
            max_neurons_per_concept: 256,
            encoding_timeout: Duration::from_millis(10),
        }
    }
}

impl TTFSConfig {
    /// Create high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            optimization_strategy: OptimizationStrategy::HighThroughput,
            enable_simd: true,
            enable_parallel: true,
            cache_size: 10000,
            timing_precision: TimingPrecision::Microsecond,
            enable_membrane_dynamics: false,
            enable_adaptation: false,
            ..Default::default()
        }
    }
    
    /// Create low-latency configuration
    pub fn low_latency() -> Self {
        Self {
            optimization_strategy: OptimizationStrategy::LowLatency,
            max_spike_time: Duration::from_micros(500),
            timing_precision: TimingPrecision::HundredNanosecond,
            enable_simd: true,
            enable_parallel: true,
            cache_size: 5000,
            max_neurons_per_concept: 64,
            encoding_timeout: Duration::from_millis(1),
            ..Default::default()
        }
    }
    
    /// Create biologically accurate configuration
    pub fn biological_accurate() -> Self {
        Self {
            optimization_strategy: OptimizationStrategy::BiologicalAccuracy,
            timing_precision: TimingPrecision::Nanosecond,
            refractory_period: Duration::from_micros(100),
            enable_membrane_dynamics: true,
            enable_adaptation: true,
            biological_params: BiologicalParameters::default(),
            ..Default::default()
        }
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.neuron_count == 0 {
            return Err("Neuron count must be greater than zero".to_string());
        }
        
        if self.neuron_count > 1_000_000 {
            return Err("Neuron count too large (max: 1,000,000)".to_string());
        }
        
        if self.refractory_period < Duration::from_nanos(100) {
            return Err("Refractory period too short (min: 100ns)".to_string());
        }
        
        if self.refractory_period > Duration::from_millis(10) {
            return Err("Refractory period too long (max: 10ms)".to_string());
        }
        
        if self.max_spike_time > Duration::from_millis(100) {
            return Err("Maximum spike time too long (max: 100ms)".to_string());
        }
        
        if self.max_neurons_per_concept > self.neuron_count {
            return Err("Max neurons per concept exceeds total neuron count".to_string());
        }
        
        if self.cache_size > 1_000_000 {
            return Err("Cache size too large (max: 1,000,000)".to_string());
        }
        
        // Validate biological parameters
        let bio = &self.biological_params;
        if bio.resting_potential >= bio.threshold_potential {
            return Err("Resting potential must be less than threshold potential".to_string());
        }
        
        if bio.leak_conductance <= 0.0 {
            return Err("Leak conductance must be positive".to_string());
        }
        
        if bio.membrane_capacitance <= 0.0 {
            return Err("Membrane capacitance must be positive".to_string());
        }
        
        Ok(())
    }
}

/// Encoder state for tracking operations
#[derive(Debug, Clone, PartialEq)]
pub enum EncoderState {
    /// Ready for new encoding operations
    Ready,
    /// Currently encoding a pattern
    Encoding,
    /// Error state requiring reset
    Error(String),
    /// Maintenance mode (cache cleanup, etc.)
    Maintenance,
}

/// Main TTFS encoder structure
#[derive(Debug)]
pub struct TTFSEncoder {
    /// Configuration parameters
    config: TTFSConfig,
    
    /// Current encoder state
    state: EncoderState,
    
    /// Performance metrics
    metrics: EncoderMetrics,
    
    /// Neuron allocation map
    neuron_allocations: HashMap<ConceptId, Vec<NeuronId>>,
    
    /// Available neurons pool
    available_neurons: Vec<NeuronId>,
    
    /// Last encoding timestamp
    last_encoding: Option<Instant>,
    
    /// Encoding operation counter
    encoding_counter: u64,
    
    /// Random number generator seed for reproducibility
    rng_seed: u64,
}

impl TTFSEncoder {
    /// Create new TTFS encoder with configuration
    pub fn new(config: TTFSConfig) -> Self {
        // Validate configuration
        config.validate().expect("Invalid TTFS configuration");
        
        // Initialize available neurons
        let available_neurons: Vec<NeuronId> = (0..config.neuron_count)
            .map(NeuronId)
            .collect();
        
        Self {
            config,
            state: EncoderState::Ready,
            metrics: EncoderMetrics::default(),
            neuron_allocations: HashMap::new(),
            available_neurons,
            last_encoding: None,
            encoding_counter: 0,
            rng_seed: 42, // Default seed for reproducibility
        }
    }
    
    /// Get total neuron count
    pub fn neuron_count(&self) -> usize {
        self.config.neuron_count
    }
    
    /// Get maximum spike time
    pub fn max_spike_time(&self) -> Duration {
        self.config.max_spike_time
    }
    
    /// Get refractory period
    pub fn refractory_period(&self) -> Duration {
        self.config.refractory_period
    }
    
    /// Check if SIMD is enabled
    pub fn simd_enabled(&self) -> bool {
        self.config.enable_simd
    }
    
    /// Check if parallel processing is enabled
    pub fn parallel_processing_enabled(&self) -> bool {
        self.config.enable_parallel
    }
    
    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.config.cache_size
    }
    
    /// Check if encoder is ready for operations
    pub fn is_ready(&self) -> bool {
        matches!(self.state, EncoderState::Ready)
    }
    
    /// Get current encoding count
    pub fn encoding_count(&self) -> u64 {
        self.encoding_counter
    }
    
    /// Get timing resolution in nanoseconds
    pub fn timing_resolution_ns(&self) -> u64 {
        self.config.timing_precision.resolution_ns()
    }
    
    /// Check if optimized for latency
    pub fn is_optimized_for_latency(&self) -> bool {
        matches!(self.config.optimization_strategy, OptimizationStrategy::LowLatency)
    }
    
    /// Check if membrane dynamics are enabled
    pub fn membrane_dynamics_enabled(&self) -> bool {
        self.config.enable_membrane_dynamics
    }
    
    /// Check if adaptation is enabled
    pub fn adaptation_enabled(&self) -> bool {
        self.config.enable_adaptation
    }
    
    /// Get biological parameters
    pub fn biological_parameters(&self) -> &BiologicalParameters {
        &self.config.biological_params
    }
    
    /// Get performance metrics
    pub fn performance_metrics(&self) -> &EncoderMetrics {
        &self.metrics
    }
    
    /// Begin encoding operation
    pub fn begin_encoding(&mut self) {
        if !self.is_ready() {
            panic!("Encoder not ready for encoding operation");
        }
        
        self.state = EncoderState::Encoding;
        self.last_encoding = Some(Instant::now());
    }
    
    /// Complete encoding operation
    pub fn complete_encoding(&mut self) {
        if !matches!(self.state, EncoderState::Encoding) {
            panic!("No encoding operation in progress");
        }
        
        self.state = EncoderState::Ready;
        self.encoding_counter += 1;
        
        // Update metrics
        if let Some(start_time) = self.last_encoding {
            let encoding_time = start_time.elapsed();
            self.update_average_encoding_time(encoding_time);
        }
    }
    
    /// Allocate neurons for a concept
    pub fn allocate_neurons(&mut self, concept_id: &ConceptId, count: usize) -> Vec<NeuronId> {
        if count > self.available_neurons.len() {
            panic!("Not enough available neurons for allocation");
        }
        
        if count > self.config.max_neurons_per_concept {
            panic!("Requested neuron count exceeds maximum per concept");
        }
        
        // Use deterministic allocation based on concept ID for reproducibility
        let mut allocated = Vec::with_capacity(count);
        let concept_hash = self.hash_concept_id(concept_id);
        
        for i in 0..count {
            let index = (concept_hash + i as u64) % self.available_neurons.len() as u64;
            allocated.push(self.available_neurons[index as usize]);
        }
        
        // Remove allocated neurons from available pool
        for neuron in &allocated {
            self.available_neurons.retain(|&n| n != *neuron);
        }
        
        // Store allocation
        self.neuron_allocations.insert(concept_id.clone(), allocated.clone());
        
        allocated
    }
    
    /// Release neurons back to available pool
    pub fn release_neurons(&mut self, concept_id: &ConceptId) {
        if let Some(allocated) = self.neuron_allocations.remove(concept_id) {
            self.available_neurons.extend(allocated);
            // Keep available neurons sorted for deterministic behavior
            self.available_neurons.sort_by_key(|n| n.0);
        }
    }
    
    /// Get current encoder state
    pub fn state(&self) -> &EncoderState {
        &self.state
    }
    
    /// Reset encoder to ready state
    pub fn reset(&mut self) {
        self.state = EncoderState::Ready;
        self.neuron_allocations.clear();
        self.available_neurons = (0..self.config.neuron_count).map(NeuronId).collect();
    }
    
    /// Update configuration (requires reset)
    pub fn update_config(&mut self, new_config: TTFSConfig) -> Result<(), String> {
        new_config.validate()?;
        self.config = new_config;
        self.reset();
        Ok(())
    }
    
    /// Set random seed for reproducible behavior
    pub fn set_random_seed(&mut self, seed: u64) {
        self.rng_seed = seed;
    }
    
    /// Get available neuron count
    pub fn available_neuron_count(&self) -> usize {
        self.available_neurons.len()
    }
    
    /// Get allocated concepts
    pub fn allocated_concepts(&self) -> Vec<&ConceptId> {
        self.neuron_allocations.keys().collect()
    }
    
    /// Check if concept has allocated neurons
    pub fn has_allocation(&self, concept_id: &ConceptId) -> bool {
        self.neuron_allocations.contains_key(concept_id)
    }
    
    /// Get neuron allocation for concept
    pub fn get_allocation(&self, concept_id: &ConceptId) -> Option<&[NeuronId]> {
        self.neuron_allocations.get(concept_id).map(|v| v.as_slice())
    }
    
    // Helper methods
    
    fn hash_concept_id(&self, concept_id: &ConceptId) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        concept_id.hash(&mut hasher);
        self.rng_seed.hash(&mut hasher);
        hasher.finish()
    }
    
    fn update_average_encoding_time(&mut self, new_time: Duration) {
        let total_encodings = self.encoding_counter;
        if total_encodings == 0 {
            self.metrics.average_encoding_time = new_time;
        } else {
            let current_avg_nanos = self.metrics.average_encoding_time.as_nanos() as u64;
            let new_time_nanos = new_time.as_nanos() as u64;
            let new_avg_nanos = (current_avg_nanos * (total_encodings - 1) + new_time_nanos) / total_encodings;
            self.metrics.average_encoding_time = Duration::from_nanos(new_avg_nanos);
        }
    }
}

/// Error types for TTFS encoder operations
#[derive(Debug, Clone, PartialEq)]
pub enum TTFSEncoderError {
    /// Configuration validation failed
    InvalidConfiguration(String),
    /// Insufficient neurons available
    InsufficientNeurons { requested: usize, available: usize },
    /// Encoding timeout
    EncodingTimeout,
    /// Encoder in wrong state
    InvalidState(String),
    /// Memory allocation failed
    MemoryAllocation(String),
    /// SIMD operation failed
    SimdError(String),
}

impl std::fmt::Display for TTFSEncoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TTFSEncoderError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            TTFSEncoderError::InsufficientNeurons { requested, available } => {
                write!(f, "Insufficient neurons: requested {}, available {}", requested, available)
            }
            TTFSEncoderError::EncodingTimeout => {
                write!(f, "Encoding operation timed out")
            }
            TTFSEncoderError::InvalidState(msg) => {
                write!(f, "Invalid encoder state: {}", msg)
            }
            TTFSEncoderError::MemoryAllocation(msg) => {
                write!(f, "Memory allocation failed: {}", msg)
            }
            TTFSEncoderError::SimdError(msg) => {
                write!(f, "SIMD operation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for TTFSEncoderError {}

/// Result type for TTFS encoder operations
pub type TTFSResult<T> = Result<T, TTFSEncoderError>;
```

## Verification Steps
1. Create TTFSConfig structure with validation
2. Implement TTFSEncoder with state management
3. Add neuron allocation and release mechanisms
4. Implement performance metrics tracking
5. Add comprehensive error handling
6. Ensure all tests pass

## Success Criteria
- [ ] TTFSEncoder struct compiles without errors
- [ ] Configuration validation prevents invalid states
- [ ] Neuron allocation works deterministically
- [ ] State management functions correctly
- [ ] Performance metrics are tracked accurately
- [ ] All tests pass with biological accuracy requirements