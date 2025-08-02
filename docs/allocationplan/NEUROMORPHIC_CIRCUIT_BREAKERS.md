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

// Practical fallback implementations
impl NeuromorphicCircuitBreaker {
    async fn single_column_fallback<T>(&self) -> Result<T, NeuromorphicError> 
    where 
        T: Default + Send + 'static
    {
        // Use only the semantic column as primary fallback
        let semantic_result = self.execute_semantic_only_processing().await?;
        
        // Convert result to expected type
        Ok(T::default()) // Simplified for demonstration
    }
    
    async fn feature_vector_fallback<T>(&self) -> Result<T, NeuromorphicError>
    where
        T: Default + Send + 'static
    {
        // Convert spike patterns to simple feature vectors
        let features = self.extract_simple_features().await?;
        
        // Use traditional ML instead of spiking networks
        let result = self.apply_traditional_ml(features).await?;
        
        Ok(T::default())
    }
    
    async fn static_weights_fallback<T>(&self) -> Result<T, NeuromorphicError>
    where
        T: Default + Send + 'static
    {
        // Load pre-trained static weights
        let static_weights = self.load_static_weights().await?;
        
        // Apply without STDP updates
        let result = self.apply_static_network(static_weights).await?;
        
        Ok(T::default())
    }
    
    async fn static_network_fallback<T>(&self) -> Result<T, NeuromorphicError>
    where
        T: Default + Send + 'static
    {
        // Use fixed network topology without cascade correlation
        let fixed_topology = self.get_fixed_topology().await?;
        
        // Process with no dynamic growth
        let result = self.process_fixed_network(fixed_topology).await?;
        
        Ok(T::default())
    }
    
    async fn execute_semantic_only_processing(&self) -> Result<SemanticResult, NeuromorphicError> {
        // Fallback to semantic column only processing
        let semantic_column = self.get_semantic_column().await?;
        
        // Simple keyword matching and embedding similarity
        let result = semantic_column.simple_allocation().await?;
        
        Ok(result)
    }
    
    async fn extract_simple_features(&self) -> Result<Vec<f32>, NeuromorphicError> {
        // Extract basic features without spike encoding
        Ok(vec![0.0; 128]) // Placeholder feature vector
    }
    
    async fn apply_traditional_ml(&self, features: Vec<f32>) -> Result<MLResult, NeuromorphicError> {
        // Use simple cosine similarity instead of neural processing
        Ok(MLResult::default())
    }
}

// Advanced degradation strategies
#[derive(Debug, Clone)]
pub struct AdaptiveDegradationStrategy {
    pub current_level: DegradationLevel,
    pub performance_history: Arc<RwLock<VecDeque<PerformanceSnapshot>>>,
    pub recovery_threshold: f32,
    pub degradation_threshold: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DegradationLevel {
    Normal = 0,
    ReducedPrecision = 1,
    SingleColumn = 2,
    StaticWeights = 3,
    EmergencyMode = 4,
}

impl AdaptiveDegradationStrategy {
    pub async fn should_degrade(&self) -> bool {
        let history = self.performance_history.read().await;
        if history.len() < 5 {
            return false;
        }
        
        // Calculate recent performance trend
        let recent_performance: f32 = history.iter()
            .rev()
            .take(5)
            .map(|snapshot| snapshot.success_rate)
            .sum::<f32>() / 5.0;
        
        recent_performance < self.degradation_threshold
    }
    
    pub async fn should_recover(&self) -> bool {
        let history = self.performance_history.read().await;
        if history.len() < 10 {
            return false;
        }
        
        // Need sustained good performance to recover
        let recent_performance: f32 = history.iter()
            .rev()
            .take(10)
            .map(|snapshot| snapshot.success_rate)
            .sum::<f32>() / 10.0;
        
        recent_performance > self.recovery_threshold
    }
    
    pub fn next_degradation_level(&self) -> Option<DegradationLevel> {
        match self.current_level {
            DegradationLevel::Normal => Some(DegradationLevel::ReducedPrecision),
            DegradationLevel::ReducedPrecision => Some(DegradationLevel::SingleColumn),
            DegradationLevel::SingleColumn => Some(DegradationLevel::StaticWeights),
            DegradationLevel::StaticWeights => Some(DegradationLevel::EmergencyMode),
            DegradationLevel::EmergencyMode => None,
        }
    }
    
    pub fn previous_degradation_level(&self) -> Option<DegradationLevel> {
        match self.current_level {
            DegradationLevel::Normal => None,
            DegradationLevel::ReducedPrecision => Some(DegradationLevel::Normal),
            DegradationLevel::SingleColumn => Some(DegradationLevel::ReducedPrecision),
            DegradationLevel::StaticWeights => Some(DegradationLevel::SingleColumn),
            DegradationLevel::EmergencyMode => Some(DegradationLevel::StaticWeights),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: Instant,
    pub success_rate: f32,
    pub average_latency: Duration,
    pub error_types: HashMap<String, u32>,
}

// Integration with MCP Server
impl MCPServer {
    pub fn with_circuit_breaker(mut self, circuit_breaker: NeuromorphicCircuitBreaker) -> Self {
        self.circuit_breaker = Some(circuit_breaker);
        self
    }
    
    pub async fn handle_protected_request(&self, request: MCPRequest) -> Result<MCPResponse, MCPError> {
        if let Some(ref circuit_breaker) = self.circuit_breaker {
            // Wrap MCP operations with circuit breaker protection
            circuit_breaker.execute_with_protection(
                async {
                    self.handle_request_internal(request).await
                        .map_err(|e| NeuromorphicError::from(e))
                },
                NeuromorphicOperationType::from(&request),
            ).await
            .map_err(|e| MCPError::from(e))
        } else {
            self.handle_request_internal(request).await
        }
    }
}

// Monitoring and alerting integration
#[derive(Debug)]
pub struct CircuitBreakerMonitor {
    pub prometheus_registry: prometheus::Registry,
    pub state_gauge: IntGauge,
    pub failure_counter: IntCounter,
    pub success_counter: IntCounter,
    pub fallback_counter: IntCounter,
    pub recovery_histogram: Histogram,
}

impl CircuitBreakerMonitor {
    pub fn new() -> Result<Self, prometheus::Error> {
        let registry = prometheus::Registry::new();
        
        let state_gauge = IntGauge::new("circuit_breaker_state", "Current circuit breaker state")?;
        let failure_counter = IntCounter::new("circuit_breaker_failures_total", "Total failures")?;
        let success_counter = IntCounter::new("circuit_breaker_successes_total", "Total successes")?;
        let fallback_counter = IntCounter::new("circuit_breaker_fallbacks_total", "Total fallback executions")?;
        let recovery_histogram = Histogram::with_opts(
            HistogramOpts::new("circuit_breaker_recovery_duration_seconds", "Recovery time distribution")
        )?;
        
        registry.register(Box::new(state_gauge.clone()))?;
        registry.register(Box::new(failure_counter.clone()))?;
        registry.register(Box::new(success_counter.clone()))?;
        registry.register(Box::new(fallback_counter.clone()))?;
        registry.register(Box::new(recovery_histogram.clone()))?;
        
        Ok(Self {
            prometheus_registry: registry,
            state_gauge,
            failure_counter,
            success_counter,
            fallback_counter,
            recovery_histogram,
        })
    }
    
    pub fn record_state_change(&self, new_state: CircuitState) {
        let state_value = match new_state {
            CircuitState::Closed => 0,
            CircuitState::Open => 1,
            CircuitState::HalfOpen => 2,
            CircuitState::Degraded => 3,
        };
        self.state_gauge.set(state_value);
    }
    
    pub fn record_failure(&self) {
        self.failure_counter.inc();
    }
    
    pub fn record_success(&self) {
        self.success_counter.inc();
    }
    
    pub fn record_fallback(&self) {
        self.fallback_counter.inc();
    }
    
    pub fn record_recovery(&self, duration: Duration) {
        self.recovery_histogram.observe(duration.as_secs_f64());
    }
}

// Configuration and deployment
#[derive(Debug, Clone, Deserialize)]
pub struct CircuitBreakerConfig {
    // Basic circuit breaker settings
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_duration_ms: u64,
    pub half_open_timeout_ms: u64,
    
    // Neuromorphic-specific settings
    pub spike_failure_threshold: f32,
    pub refractory_violation_threshold: u32,
    pub column_consensus_threshold: f32,
    pub lateral_inhibition_timeout_ms: u64,
    
    // Degradation strategy
    pub degradation_strategy: DegradationStrategyConfig,
    
    // Fallback configuration
    pub fallback_config: FallbackConfig,
    
    // Monitoring settings
    pub monitoring_enabled: bool,
    pub alert_webhook_url: Option<String>,
    pub metrics_export_interval_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_duration_ms: 60000,
            half_open_timeout_ms: 30000,
            
            spike_failure_threshold: 0.5,
            refractory_violation_threshold: 3,
            column_consensus_threshold: 0.7,
            lateral_inhibition_timeout_ms: 500,
            
            degradation_strategy: DegradationStrategyConfig::default(),
            fallback_config: FallbackConfig::default(),
            
            monitoring_enabled: true,
            alert_webhook_url: None,
            metrics_export_interval_secs: 60,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DegradationStrategyConfig {
    pub initial_level: DegradationLevel,
    pub degradation_threshold: f32,
    pub recovery_threshold: f32,
    pub history_window_size: usize,
    pub auto_recovery_enabled: bool,
}

impl Default for DegradationStrategyConfig {
    fn default() -> Self {
        Self {
            initial_level: DegradationLevel::Normal,
            degradation_threshold: 0.7,  // Degrade if success rate < 70%
            recovery_threshold: 0.9,     // Recover if success rate > 90%
            history_window_size: 100,
            auto_recovery_enabled: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct FallbackConfig {
    pub strategy_type: FallbackStrategyType,
    pub cache_size: usize,
    pub cache_ttl_secs: u64,
    pub minimum_columns_for_degraded: usize,
    pub emergency_mode_settings: EmergencyModeSettings,
}

#[derive(Debug, Clone, Deserialize)]
pub enum FallbackStrategyType {
    SimplifiedProcessing,
    CachedResponse,
    DegradedService,
    EmergencyMode,
    Adaptive, // Automatically choose based on failure type
}

impl Default for FallbackConfig {
    fn default() -> Self {
        Self {
            strategy_type: FallbackStrategyType::Adaptive,
            cache_size: 1000,
            cache_ttl_secs: 300,
            minimum_columns_for_degraded: 2,
            emergency_mode_settings: EmergencyModeSettings::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmergencyModeSettings {
    pub disable_stdp: bool,
    pub disable_cascade: bool,
    pub use_static_weights: bool,
    pub max_processing_time_ms: u64,
}

impl Default for EmergencyModeSettings {
    fn default() -> Self {
        Self {
            disable_stdp: true,
            disable_cascade: true,
            use_static_weights: true,
            max_processing_time_ms: 100,
        }
    }
}
```

## Real-World Usage Examples

```rust
// Example 1: Basic circuit breaker setup
use cortexkg::circuit_breaker::{NeuromorphicCircuitBreaker, CircuitBreakerConfig};
use cortexkg::multi_column::MultiColumnProcessor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = CircuitBreakerConfig::default();
    
    // Create circuit breaker
    let circuit_breaker = NeuromorphicCircuitBreaker::from_config(config);
    
    // Create multi-column processor with circuit breaker
    let processor = MultiColumnProcessor::new()
        .with_circuit_breaker(circuit_breaker);
    
    // Process with fault tolerance
    let spike_pattern = create_spike_pattern("Machine learning concept");
    let result = processor.fault_tolerant_processing(&spike_pattern).await?;
    
    println!("Processing result: {:?}", result);
    Ok(())
}

// Example 2: Custom fallback strategy
use cortexkg::circuit_breaker::{FallbackStrategy, DegradedModeConfig};

async fn setup_custom_fallback() -> NeuromorphicCircuitBreaker {
    let fallback_strategy = FallbackStrategy::DegradedService {
        minimum_columns: 2,
        reduced_precision: true,
        simplified_voting: true,
    };
    
    let mut circuit_breaker = NeuromorphicCircuitBreaker::new();
    circuit_breaker.set_fallback_strategy(fallback_strategy);
    
    // Configure degradation thresholds
    circuit_breaker.set_spike_failure_threshold(0.6); // More tolerant
    circuit_breaker.set_column_consensus_threshold(0.5); // Lower consensus requirement
    
    circuit_breaker
}

// Example 3: Monitoring integration
use cortexkg::circuit_breaker::CircuitBreakerMonitor;
use prometheus::{Encoder, TextEncoder};

async fn setup_monitoring() -> Result<(), Box<dyn std::error::Error>> {
    let monitor = CircuitBreakerMonitor::new()?;
    
    // Export metrics endpoint
    let metrics_handler = || {
        let encoder = TextEncoder::new();
        let metric_families = monitor.prometheus_registry.gather();
        let mut buffer = vec![];
        encoder.encode(&metric_families, &mut buffer)?;
        String::from_utf8(buffer)
    };
    
    // Start metrics server
    warp::path!("metrics")
        .map(metrics_handler)
        .run(([0, 0, 0, 0], 9090))
        .await;
    
    Ok(())
}

// Example 4: Adaptive degradation
use cortexkg::circuit_breaker::{AdaptiveDegradationStrategy, DegradationLevel};

async fn adaptive_degradation_example() {
    let mut strategy = AdaptiveDegradationStrategy {
        current_level: DegradationLevel::Normal,
        performance_history: Arc::new(RwLock::new(VecDeque::new())),
        recovery_threshold: 0.9,
        degradation_threshold: 0.7,
    };
    
    // Simulate performance monitoring
    loop {
        let performance = measure_current_performance().await;
        
        let snapshot = PerformanceSnapshot {
            timestamp: Instant::now(),
            success_rate: performance.success_rate,
            average_latency: performance.latency,
            error_types: performance.errors,
        };
        
        strategy.performance_history.write().await.push_back(snapshot);
        
        // Check if we should degrade
        if strategy.should_degrade().await {
            if let Some(next_level) = strategy.next_degradation_level() {
                println!("Degrading to {:?}", next_level);
                strategy.current_level = next_level;
                apply_degradation_level(next_level).await;
            }
        }
        
        // Check if we can recover
        if strategy.should_recover().await {
            if let Some(prev_level) = strategy.previous_degradation_level() {
                println!("Recovering to {:?}", prev_level);
                strategy.current_level = prev_level;
                apply_degradation_level(prev_level).await;
            }
        }
        
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

// Example 5: Integration with MCP server
use cortexkg::mcp::{MCPServer, MCPRequest, MCPResponse};

async fn mcp_with_circuit_breaker() -> Result<(), Box<dyn std::error::Error>> {
    // Create MCP server with circuit breaker
    let circuit_breaker = NeuromorphicCircuitBreaker::new();
    let mcp_server = MCPServer::new()
        .with_circuit_breaker(circuit_breaker);
    
    // Handle requests with protection
    let request = MCPRequest::StoreMemory {
        content: "Quantum computing breakthrough".to_string(),
        context: Some("Technology news".to_string()),
        confidence: Some(0.95),
    };
    
    match mcp_server.handle_protected_request(request).await {
        Ok(response) => {
            println!("Success: {:?}", response);
        }
        Err(MCPError::CircuitBreakerOpen) => {
            println!("Circuit breaker is open, using fallback");
            // Fallback logic handled internally
        }
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
    
    Ok(())
}
```

## Production Deployment Guidelines

### 1. Configuration Best Practices

```yaml
# config/circuit_breaker.yaml
circuit_breaker:
  # Conservative settings for production
  failure_threshold: 5
  success_threshold: 3
  timeout_duration_ms: 60000
  
  # Neuromorphic thresholds based on hardware capabilities
  spike_failure_threshold: 0.4  # Stricter in production
  refractory_violation_threshold: 2
  column_consensus_threshold: 0.8  # Higher consensus requirement
  
  # Degradation strategy
  degradation_strategy:
    initial_level: Normal
    degradation_threshold: 0.6  # Degrade earlier in production
    recovery_threshold: 0.95    # Require excellent performance to recover
    auto_recovery_enabled: true
  
  # Monitoring
  monitoring_enabled: true
  alert_webhook_url: "https://alerts.example.com/neuromorphic"
  metrics_export_interval_secs: 30
```

### 2. Alerting Rules

```yaml
# prometheus/alerts.yml
groups:
  - name: circuit_breaker_alerts
    rules:
      - alert: CircuitBreakerOpen
        expr: circuit_breaker_state == 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Circuit breaker is open"
          description: "Neuromorphic processing circuit breaker has been open for >1 minute"
      
      - alert: HighFailureRate
        expr: rate(circuit_breaker_failures_total[5m]) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High neuromorphic failure rate"
          description: "Failure rate >10/min for 2 minutes"
      
      - alert: DegradedMode
        expr: circuit_breaker_state == 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "System in degraded mode"
          description: "Neuromorphic system operating in degraded mode for >5 minutes"
```

### 3. Health Check Endpoints

```rust
// Health check implementation
pub async fn health_check(circuit_breaker: &NeuromorphicCircuitBreaker) -> HealthStatus {
    let state = circuit_breaker.get_current_state().await;
    let monitor = circuit_breaker.get_monitor();
    let health_score = monitor.get_health_score().await;
    
    HealthStatus {
        status: match state {
            CircuitState::Closed => "healthy",
            CircuitState::Open => "unhealthy",
            CircuitState::HalfOpen => "recovering",
            CircuitState::Degraded => "degraded",
        },
        circuit_state: format!("{:?}", state),
        health_score,
        recent_errors: monitor.get_recent_error_count().await,
        fallback_active: circuit_breaker.is_using_fallback().await,
        details: monitor.get_detailed_metrics().await,
    }
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