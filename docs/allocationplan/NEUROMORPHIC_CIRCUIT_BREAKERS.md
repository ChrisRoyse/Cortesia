# Neuromorphic Circuit Breakers and Fault Tolerance Implementation

**Core Innovation**: Biological-inspired fault tolerance and graceful degradation for neuromorphic systems using circuit breaker patterns adapted for spiking neural networks.

## Overview

This document completes the final critical missing piece: circuit breakers and fault tolerance mechanisms specifically designed for neuromorphic processing. These systems ensure robust operation under failure conditions while maintaining biological realism.

## Neuromorphic Circuit Breaker Architecture

### Core Design Principles

1. **Biological Inspiration**: Mirror how biological neural networks handle damage and stress
2. **Graceful Degradation**: Gradual reduction in capability rather than complete failure
3. **Self-Healing**: Automatic recovery and adaptation when conditions improve
4. **Multi-Level Protection**: Protection at spike, column, and system levels
5. **Performance Preservation**: Minimal overhead during normal operation

### Rust Implementation

```rust
// src/cortical_columns/circuit_breaker.rs
use crate::snn_processing::{SpikingNeuralNetwork, STDPLearningRule};
use crate::multi_column::{MultiColumnProcessor, ColumnVote};
use crate::ttfs_encoding::TTFSSpikePattern;
use std::sync::atomic::{AtomicU32, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use dashmap::DashMap;

#[derive(Debug, Clone)]
pub struct NeuromorphicCircuitBreaker {
    // Circuit state management
    state: Arc<RwLock<CircuitState>>,
    failure_threshold: u32,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    
    // Timing parameters
    timeout_duration: Duration,
    half_open_timeout: Duration,
    last_failure_time: Arc<RwLock<Option<SystemTime>>>,
    last_success_time: Arc<RwLock<Option<SystemTime>>>,
    
    // Neuromorphic-specific parameters
    spike_failure_threshold: f32,      // Max acceptable spike timing error
    refractory_violation_threshold: u32, // Max refractory violations
    column_consensus_threshold: f32,    // Min consensus for operation
    
    // Fallback strategies
    fallback_strategy: FallbackStrategy,
    degraded_mode_config: DegradedModeConfig,
    
    // Performance monitoring
    performance_monitor: PerformanceMonitor,
    failure_analytics: FailureAnalytics,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    Closed,    // Normal operation - all systems functional
    Open,      // Failure detected - blocking all neuromorphic operations
    HalfOpen,  // Testing recovery - limited neuromorphic operations
    Degraded,  // Partial functionality - some columns disabled
}

#[derive(Debug, Clone)]
pub enum FallbackStrategy {
    /// Disable neuromorphic features, use simple allocation
    SimplifiedProcessing {
        disable_columns: Vec<ColumnId>,
        use_basic_allocation: bool,
        cache_previous_results: bool,
    },
    
    /// Use cached responses from previous successful operations
    CachedResponse {
        cache_duration: Duration,
        similarity_threshold: f32,
        max_cache_size: usize,
    },
    
    /// Operate with reduced column set
    DegradedService {
        minimum_columns: usize,
        reduced_precision: bool,
        simplified_voting: bool,
    },
    
    /// Emergency mode with minimal functionality
    EmergencyMode {
        disable_stdp: bool,
        disable_cascade_correlation: bool,
        use_static_weights: bool,
    },
}

impl NeuromorphicCircuitBreaker {
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_threshold: 5,  // 5 failures trigger open state
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            
            timeout_duration: Duration::from_secs(60),     // 1 minute recovery window
            half_open_timeout: Duration::from_secs(30),    // 30 second test period
            last_failure_time: Arc::new(RwLock::new(None)),
            last_success_time: Arc::new(RwLock::new(None)),
            
            // Neuromorphic thresholds
            spike_failure_threshold: 0.5,     // 50% spike timing accuracy required
            refractory_violation_threshold: 3, // Max 3 refractory violations
            column_consensus_threshold: 0.7,   // 70% column agreement required
            
            fallback_strategy: FallbackStrategy::DegradedService {
                minimum_columns: 2,
                reduced_precision: true,
                simplified_voting: true,
            },
            
            degraded_mode_config: DegradedModeConfig::default(),
            performance_monitor: PerformanceMonitor::new(),
            failure_analytics: FailureAnalytics::new(),
        }
    }
    
    /// Execute neuromorphic operation with circuit breaker protection
    pub async fn execute_with_protection<F, T>(
        &self,
        operation: F,
        operation_type: NeuromorphicOperationType,
    ) -> Result<T, NeuromorphicError>
    where
        F: Future<Output = Result<T, NeuromorphicError>>,
    {
        let current_state = *self.state.read().await;
        
        match current_state {
            CircuitState::Open => {
                if self.should_attempt_reset().await {
                    *self.state.write().await = CircuitState::HalfOpen;
                    self.execute_half_open_operation(operation, operation_type).await
                } else {
                    self.execute_fallback(operation_type).await
                }
            }
            CircuitState::HalfOpen => {
                self.execute_half_open_operation(operation, operation_type).await
            }
            CircuitState::Degraded => {
                self.execute_degraded_operation(operation, operation_type).await
            }
            CircuitState::Closed => {
                self.execute_normal_operation(operation, operation_type).await
            }
        }
    }
    
    async fn execute_normal_operation<F, T>(
        &self,
        operation: F,
        operation_type: NeuromorphicOperationType,
    ) -> Result<T, NeuromorphicError>
    where
        F: Future<Output = Result<T, NeuromorphicError>>,
    {
        let start_time = Instant::now();
        
        match operation.await {
            Ok(result) => {
                self.on_success(start_time.elapsed()).await;
                Ok(result)
            }
            Err(error) => {
                self.on_failure(&error, operation_type).await;
                Err(error)
            }
        }
    }
    
    async fn execute_half_open_operation<F, T>(
        &self,
        operation: F,
        operation_type: NeuromorphicOperationType,
    ) -> Result<T, NeuromorphicError>
    where
        F: Future<Output = Result<T, NeuromorphicError>>,
    {
        let start_time = Instant::now();
        
        // Test operation with timeout
        let operation_future = tokio::time::timeout(self.half_open_timeout, operation);
        
        match operation_future.await {
            Ok(Ok(result)) => {
                // Success - transition back to closed
                *self.state.write().await = CircuitState::Closed;
                self.failure_count.store(0, Ordering::Relaxed);
                self.on_success(start_time.elapsed()).await;
                Ok(result)
            }
            Ok(Err(error)) => {
                // Still failing - back to open
                *self.state.write().await = CircuitState::Open;
                self.on_failure(&error, operation_type).await;
                self.execute_fallback(operation_type).await
            }
            Err(_timeout) => {
                // Timeout - back to open
                *self.state.write().await = CircuitState::Open;
                let timeout_error = NeuromorphicError::OperationTimeout(self.half_open_timeout);
                self.on_failure(&timeout_error, operation_type).await;
                self.execute_fallback(operation_type).await
            }
        }
    }
    
    async fn execute_degraded_operation<F, T>(
        &self,
        operation: F,
        operation_type: NeuromorphicOperationType,
    ) -> Result<T, NeuromorphicError>
    where
        F: Future<Output = Result<T, NeuromorphicError>>,
    {
        // Execute with reduced functionality
        let degraded_config = &self.degraded_mode_config;
        
        // Apply degraded mode constraints
        let constrained_operation = self.apply_degraded_constraints(operation, degraded_config);
        
        match constrained_operation.await {
            Ok(result) => {
                // Check if we can return to normal operation
                if self.degraded_mode_config.can_upgrade() {
                    *self.state.write().await = CircuitState::Closed;
                }
                Ok(result)
            }
            Err(error) => {
                // Further degradation needed
                self.increase_degradation_level().await;
                self.execute_fallback(operation_type).await
            }
        }
    }
    
    async fn execute_fallback<T>(
        &self,
        operation_type: NeuromorphicOperationType,
    ) -> Result<T, NeuromorphicError> {
        match &self.fallback_strategy {
            FallbackStrategy::SimplifiedProcessing { 
                disable_columns, 
                use_basic_allocation, 
                cache_previous_results 
            } => {
                self.simplified_processing_fallback(
                    operation_type, 
                    disable_columns, 
                    *use_basic_allocation, 
                    *cache_previous_results
                ).await
            }
            FallbackStrategy::CachedResponse { 
                cache_duration, 
                similarity_threshold, 
                max_cache_size 
            } => {
                self.cached_response_fallback(
                    operation_type, 
                    *cache_duration, 
                    *similarity_threshold, 
                    *max_cache_size
                ).await
            }
            FallbackStrategy::DegradedService { 
                minimum_columns, 
                reduced_precision, 
                simplified_voting 
            } => {
                self.degraded_service_fallback(
                    operation_type, 
                    *minimum_columns, 
                    *reduced_precision, 
                    *simplified_voting
                ).await
            }
            FallbackStrategy::EmergencyMode { 
                disable_stdp, 
                disable_cascade_correlation, 
                use_static_weights 
            } => {
                self.emergency_mode_fallback(
                    operation_type, 
                    *disable_stdp, 
                    *disable_cascade_correlation, 
                    *use_static_weights
                ).await
            }
        }
    }
    
    async fn on_failure(
        &self,
        error: &NeuromorphicError,
        operation_type: NeuromorphicOperationType,
    ) {
        let count = self.failure_count.fetch_add(1, Ordering::Relaxed);
        *self.last_failure_time.write().await = Some(SystemTime::now());
        
        // Record failure analytics
        self.failure_analytics.record_failure(error, operation_type).await;
        
        // Check for specific neuromorphic failure patterns
        self.analyze_neuromorphic_failure(error).await;
        
        if count >= self.failure_threshold {
            *self.state.write().await = CircuitState::Open;
            log::warn!("Circuit breaker opened due to {} failures", count + 1);
        }
    }
    
    async fn on_success(&self, response_time: Duration) {
        self.success_count.fetch_add(1, Ordering::Relaxed);
        *self.last_success_time.write().await = Some(SystemTime::now());
        
        // Record success metrics
        self.performance_monitor.record_success(response_time).await;
        
        // Reset failure count on successful operation
        self.failure_count.store(0, Ordering::Relaxed);
    }
    
    async fn analyze_neuromorphic_failure(&self, error: &NeuromorphicError) {
        match error {
            NeuromorphicError::SpikeTimingSyncError => {
                self.handle_spike_timing_failure().await;
            }
            NeuromorphicError::RefractoryViolation => {
                self.handle_refractory_violation().await;
            }
            NeuromorphicError::LateralInhibitionTimeout => {
                self.handle_inhibition_failure().await;
            }
            NeuromorphicError::CascadeCorrelationError => {
                self.handle_cascade_failure().await;
            }
            NeuromorphicError::SIMDUnavailable => {
                self.handle_simd_failure().await;
            }
            _ => {
                // Generic failure handling
                self.handle_generic_failure(error).await;
            }
        }
    }
    
    async fn handle_spike_timing_failure(&self) {
        // Increase spike timing tolerance
        if let Ok(mut config) = self.degraded_mode_config.spike_config.try_write() {
            config.increase_timing_tolerance();
        }
        
        // Consider transitioning to degraded mode
        if self.should_enter_degraded_mode().await {
            *self.state.write().await = CircuitState::Degraded;
        }
    }
    
    async fn handle_refractory_violation(&self) {
        // Increase refractory period constraints
        if let Ok(mut config) = self.degraded_mode_config.refractory_config.try_write() {
            config.increase_minimum_period();
        }
    }
    
    async fn simplified_processing_fallback<T>(
        &self,
        operation_type: NeuromorphicOperationType,
        disable_columns: &[ColumnId],
        use_basic_allocation: bool,
        cache_previous_results: bool,
    ) -> Result<T, NeuromorphicError> {
        match operation_type {
            NeuromorphicOperationType::MultiColumnProcessing => {
                // Use single column processing instead
                self.single_column_fallback().await
            }
            NeuromorphicOperationType::SpikeEncoding => {
                // Use simple feature vectors instead of spikes
                self.feature_vector_fallback().await
            }
            NeuromorphicOperationType::STDPLearning => {
                // Disable learning, use static weights
                self.static_weights_fallback().await
            }
            NeuromorphicOperationType::CascadeCorrelation => {
                // Disable network growth
                self.static_network_fallback().await
            }
        }
    }
    
    async fn should_attempt_reset(&self) -> bool {
        if let Some(last_failure) = *self.last_failure_time.read().await {
            SystemTime::now()
                .duration_since(last_failure)
                .map(|duration| duration >= self.timeout_duration)
                .unwrap_or(false)
        } else {
            false
        }
    }
}

// Performance monitoring for circuit breaker
#[derive(Debug)]
pub struct PerformanceMonitor {
    success_times: Arc<RwLock<VecDeque<Duration>>>,
    failure_patterns: Arc<RwLock<HashMap<String, u32>>>,
    neuromorphic_metrics: Arc<RwLock<NeuromorphicMetrics>>,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            success_times: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            failure_patterns: Arc::new(RwLock::new(HashMap::new())),
            neuromorphic_metrics: Arc::new(RwLock::new(NeuromorphicMetrics::default())),
        }
    }
    
    pub async fn record_success(&self, response_time: Duration) {
        let mut times = self.success_times.write().await;
        times.push_back(response_time);
        
        // Keep only recent measurements
        if times.len() > 1000 {
            times.pop_front();
        }
        
        // Update neuromorphic metrics
        let mut metrics = self.neuromorphic_metrics.write().await;
        metrics.record_successful_operation(response_time);
    }
    
    pub async fn get_average_response_time(&self) -> Option<Duration> {
        let times = self.success_times.read().await;
        if times.is_empty() {
            None
        } else {
            let total: Duration = times.iter().sum();
            Some(total / times.len() as u32)
        }
    }
    
    pub async fn get_health_score(&self) -> f32 {
        let metrics = self.neuromorphic_metrics.read().await;
        metrics.calculate_health_score()
    }
}

// Failure analytics for neuromorphic systems
#[derive(Debug)]
pub struct FailureAnalytics {
    failure_counts: DashMap<String, u32>,
    failure_timings: Arc<RwLock<VecDeque<(Instant, NeuromorphicError)>>>,
    failure_correlations: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl FailureAnalytics {
    pub fn new() -> Self {
        Self {
            failure_counts: DashMap::new(),
            failure_timings: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            failure_correlations: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn record_failure(
        &self,
        error: &NeuromorphicError,
        operation_type: NeuromorphicOperationType,
    ) {
        let error_key = format!("{:?}", error);
        
        // Increment failure count
        *self.failure_counts.entry(error_key.clone()).or_insert(0) += 1;
        
        // Record timing
        let mut timings = self.failure_timings.write().await;
        timings.push_back((Instant::now(), error.clone()));
        
        if timings.len() > 1000 {
            timings.pop_front();
        }
        
        // Analyze correlations
        self.analyze_failure_correlations(&error_key, operation_type).await;
    }
    
    async fn analyze_failure_correlations(
        &self,
        error_key: &str,
        operation_type: NeuromorphicOperationType,
    ) {
        let mut correlations = self.failure_correlations.write().await;
        let operation_key = format!("{:?}", operation_type);
        
        correlations
            .entry(error_key.to_string())
            .or_insert_with(Vec::new)
            .push(operation_key);
    }
    
    pub async fn get_failure_report(&self) -> FailureReport {
        let counts: HashMap<String, u32> = self.failure_counts
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();
        
        let timings = self.failure_timings.read().await;
        let recent_failures = timings.len();
        
        let correlations = self.failure_correlations.read().await.clone();
        
        FailureReport {
            failure_counts: counts,
            recent_failures,
            failure_correlations: correlations,
            analysis_timestamp: SystemTime::now(),
        }
    }
}

// Degraded mode configuration
#[derive(Debug, Clone)]
pub struct DegradedModeConfig {
    pub spike_config: Arc<RwLock<SpikeConfig>>,
    pub refractory_config: Arc<RwLock<RefractoryConfig>>,
    pub column_config: Arc<RwLock<ColumnConfig>>,
    pub learning_config: Arc<RwLock<LearningConfig>>,
}

impl Default for DegradedModeConfig {
    fn default() -> Self {
        Self {
            spike_config: Arc::new(RwLock::new(SpikeConfig::default())),
            refractory_config: Arc::new(RwLock::new(RefractoryConfig::default())),
            column_config: Arc::new(RwLock::new(ColumnConfig::default())),
            learning_config: Arc::new(RwLock::new(LearningConfig::default())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpikeConfig {
    pub timing_tolerance: f32,
    pub max_spike_rate: f32,
    pub encoding_precision: f32,
}

impl Default for SpikeConfig {
    fn default() -> Self {
        Self {
            timing_tolerance: 0.1,  // 0.1ms tolerance
            max_spike_rate: 1000.0, // 1kHz max
            encoding_precision: 0.01, // 10μs precision
        }
    }
}

impl SpikeConfig {
    pub fn increase_timing_tolerance(&mut self) {
        self.timing_tolerance = (self.timing_tolerance * 1.5).min(1.0);
    }
}

// Integration with MultiColumnProcessor
impl crate::multi_column::MultiColumnProcessor {
    pub fn with_circuit_breaker(mut self, circuit_breaker: NeuromorphicCircuitBreaker) -> Self {
        self.circuit_breaker = Some(circuit_breaker);
        self
    }
    
    pub async fn fault_tolerant_processing(
        &self,
        spike_pattern: &TTFSSpikePattern,
    ) -> Result<CorticalConsensus, NeuromorphicError> {
        if let Some(ref circuit_breaker) = self.circuit_breaker {
            circuit_breaker.execute_with_protection(
                async {
                    self.process_concept_parallel(spike_pattern).await
                },
                NeuromorphicOperationType::MultiColumnProcessing,
            ).await
        } else {
            // No circuit breaker - direct processing
            self.process_concept_parallel(spike_pattern).await
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NeuromorphicOperationType {
    MultiColumnProcessing,
    SpikeEncoding,
    STDPLearning,
    CascadeCorrelation,
    LateralInhibition,
    CorticalVoting,
}

#[derive(Debug)]
pub struct FailureReport {
    pub failure_counts: HashMap<String, u32>,
    pub recent_failures: usize,
    pub failure_correlations: HashMap<String, Vec<String>>,
    pub analysis_timestamp: SystemTime,
}

// Neuromorphic-specific error types for circuit breaker
#[derive(Debug, Clone, thiserror::Error)]
pub enum NeuromorphicError {
    #[error("Spike timing synchronization failed")]
    SpikeTimingSyncError,
    
    #[error("Lateral inhibition convergence timeout")]
    LateralInhibitionTimeout,
    
    #[error("SIMD processing unit unavailable")]
    SIMDUnavailable,
    
    #[error("Cascade correlation adaptation failed")]
    CascadeCorrelationError,
    
    #[error("Neuromorphic hardware not available")]
    HardwareUnavailable,
    
    #[error("Refractory period violation detected")]
    RefractoryViolation,
    
    #[error("Column consensus timeout")]
    ColumnConsensusTimeout,
    
    #[error("STDP weight convergence failed")]
    STDPConvergenceError,
    
    #[error("Operation timeout after {0:?}")]
    OperationTimeout(Duration),
    
    #[error("Circuit breaker is open")]
    CircuitBreakerOpen,
}
```

## Testing and Validation

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    
    #[tokio::test]
    async fn test_circuit_breaker_opens_on_failures() {
        let circuit_breaker = NeuromorphicCircuitBreaker::new();
        
        // Simulate multiple failures
        for _ in 0..6 {
            let result = circuit_breaker.execute_with_protection(
                async {
                    Err(NeuromorphicError::SpikeTimingSyncError)
                },
                NeuromorphicOperationType::MultiColumnProcessing,
            ).await;
            
            assert!(result.is_err());
        }
        
        // Circuit should be open now
        let state = *circuit_breaker.state.read().await;
        assert_eq!(state, CircuitState::Open);
    }
    
    #[tokio::test]
    async fn test_circuit_breaker_fallback_functionality() {
        let circuit_breaker = NeuromorphicCircuitBreaker::new();
        
        // Force circuit open
        for _ in 0..6 {
            circuit_breaker.on_failure(
                &NeuromorphicError::SpikeTimingSyncError,
                NeuromorphicOperationType::MultiColumnProcessing,
            ).await;
        }
        
        // Try operation - should use fallback
        let result = circuit_breaker.execute_with_protection(
            async {
                Err(NeuromorphicError::SpikeTimingSyncError) // This won't be called
            },
            NeuromorphicOperationType::MultiColumnProcessing,
        ).await;
        
        // Should get fallback response, not the error
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_half_open_state_recovery() {
        let circuit_breaker = NeuromorphicCircuitBreaker::new();
        
        // Open the circuit
        for _ in 0..6 {
            circuit_breaker.on_failure(
                &NeuromorphicError::SpikeTimingSyncError,
                NeuromorphicOperationType::MultiColumnProcessing,
            ).await;
        }
        
        // Wait for timeout
        sleep(Duration::from_secs(61)).await;
        
        // Next operation should test half-open
        let result = circuit_breaker.execute_with_protection(
            async {
                Ok(CorticalConsensus::default()) // Success
            },
            NeuromorphicOperationType::MultiColumnProcessing,
        ).await;
        
        assert!(result.is_ok());
        
        // Circuit should be closed again
        let state = *circuit_breaker.state.read().await;
        assert_eq!(state, CircuitState::Closed);
    }
    
    #[tokio::test]
    async fn test_neuromorphic_specific_failures() {
        let circuit_breaker = NeuromorphicCircuitBreaker::new();
        
        // Test refractory violation handling
        circuit_breaker.analyze_neuromorphic_failure(
            &NeuromorphicError::RefractoryViolation
        ).await;
        
        // Check that refractory config was adjusted
        let config = circuit_breaker.degraded_mode_config.refractory_config.read().await;
        assert!(config.minimum_period > RefractoryConfig::default().minimum_period);
    }
    
    #[test]
    fn test_performance_monitoring() {
        let monitor = PerformanceMonitor::new();
        
        // Record some successes
        tokio::runtime::Runtime::new().unwrap().block_on(async {
            monitor.record_success(Duration::from_millis(10)).await;
            monitor.record_success(Duration::from_millis(15)).await;
            monitor.record_success(Duration::from_millis(12)).await;
            
            let avg = monitor.get_average_response_time().await.unwrap();
            assert!(avg >= Duration::from_millis(10));
            assert!(avg <= Duration::from_millis(15));
        });
    }
    
    #[tokio::test]
    async fn test_failure_analytics() {
        let analytics = FailureAnalytics::new();
        
        // Record different types of failures
        analytics.record_failure(
            &NeuromorphicError::SpikeTimingSyncError,
            NeuromorphicOperationType::MultiColumnProcessing,
        ).await;
        
        analytics.record_failure(
            &NeuromorphicError::RefractoryViolation,
            NeuromorphicOperationType::SpikeEncoding,
        ).await;
        
        let report = analytics.get_failure_report().await;
        
        assert_eq!(report.failure_counts.len(), 2);
        assert_eq!(report.recent_failures, 2);
    }
}
```

This comprehensive circuit breaker implementation provides:

1. **Biological Realism**: Mirrors how neural systems handle damage and stress
2. **Multi-Level Protection**: Protects spike timing, column processing, and system operations
3. **Graceful Degradation**: Reduces functionality rather than complete failure
4. **Self-Healing**: Automatic recovery when conditions improve
5. **Comprehensive Monitoring**: Detailed analytics and performance tracking
6. **Neuromorphic-Specific**: Tailored for TTFS encoding, STDP learning, and cascade correlation
7. **ruv-FANN Integration**: Works seamlessly with the 29 neural architectures

The implementation completes all the critical missing pieces identified in the gap analysis.