# Phase 1 Error Handling Framework

**Purpose**: Standardize error handling patterns across all Phase 1 neuromorphic tasks  
**Scope**: Consistent Result<T, E> patterns, error recovery strategies, and failure scenarios  
**Goal**: Production-ready error handling with comprehensive recovery mechanisms

## Error Handling Principles

### 1. Consistent Result Types
All fallible operations must return `Result<T, E>` where:
- `T` is the success type  
- `E` is a specific, descriptive error type (never `Box<dyn Error>`)
- Error types implement `std::error::Error + Send + Sync + 'static`

### 2. Error Type Hierarchy
```rust
// Core error types for Phase 1
pub enum NeuromorphicError {
    State(StateTransitionError),
    Biological(BiologicalError),
    Spatial(SpatialError),
    Neural(NeuralError),
    Allocation(AllocationError),
    System(SystemError),
}

// Specific error types
#[derive(thiserror::Error, Debug)]
pub enum StateTransitionError {
    #[error("Invalid transition from {from:?} to {to:?}")]
    InvalidTransition { from: ColumnState, to: ColumnState },
    
    #[error("State mismatch: expected {expected:?}, found {actual:?}")]
    StateMismatch { expected: ColumnState, actual: ColumnState },
    
    #[error("Concurrent modification detected during transition")]
    ConcurrentModification,
    
    #[error("Transition timeout after {timeout_ms}ms")]
    TransitionTimeout { timeout_ms: u64 },
}

#[derive(thiserror::Error, Debug)]
pub enum BiologicalError {
    #[error("Invalid membrane potential: {value} (range: {min}-{max})")]
    InvalidMembranePotential { value: f32, min: f32, max: f32 },
    
    #[error("Firing threshold not met: {potential} < {threshold}")]
    ThresholdNotMet { potential: f32, threshold: f32 },
    
    #[error("Column in refractory period for {remaining_ms}ms")]
    RefractoryPeriod { remaining_ms: f32 },
    
    #[error("Invalid synaptic weight: {weight} (must be 0.0-1.0)")]
    InvalidSynapticWeight { weight: f32 },
}

#[derive(thiserror::Error, Debug)]
pub enum SpatialError {
    #[error("Invalid coordinates: ({x}, {y}, {z}) outside grid bounds")]
    OutOfBounds { x: i32, y: i32, z: i32 },
    
    #[error("Column not found at position ({x}, {y}, {z})")]
    ColumnNotFound { x: i32, y: i32, z: i32 },
    
    #[error("Spatial index not built - call build_index() first")]
    IndexNotBuilt,
    
    #[error("Query timeout after {timeout_us}μs")]
    QueryTimeout { timeout_us: u64 },
}

#[derive(thiserror::Error, Debug)]
pub enum NeuralError {
    #[error("Neural network not initialized: {network_type}")]
    NetworkNotInitialized { network_type: String },
    
    #[error("Inference failed: {reason}")]
    InferenceFailed { reason: String },
    
    #[error("Invalid input size: got {got}, expected {expected}")]
    InvalidInputSize { got: usize, expected: usize },
    
    #[error("Memory allocation failed: requested {bytes} bytes")]
    MemoryAllocationFailed { bytes: usize },
}

#[derive(thiserror::Error, Debug)]
pub enum AllocationError {
    #[error("No suitable columns found for concept")]
    NoSuitableColumns,
    
    #[error("Allocation timeout after {timeout_ms}ms")]
    AllocationTimeout { timeout_ms: u64 },
    
    #[error("Maximum allocations exceeded: {current}/{max}")]
    MaxAllocationsExceeded { current: usize, max: usize },
    
    #[error("Concept already allocated at position ({x}, {y}, {z})")]
    ConceptAlreadyAllocated { x: f32, y: f32, z: f32 },
}

#[derive(thiserror::Error, Debug)]
pub enum SystemError {
    #[error("Initialization failed: {component}")]
    InitializationFailed { component: String },
    
    #[error("Resource exhausted: {resource}")]
    ResourceExhausted { resource: String },
    
    #[error("Health check failed: {details}")]
    HealthCheckFailed { details: String },
    
    #[error("Performance degradation detected: {metric} = {value}")]
    PerformanceDegradation { metric: String, value: String },
}
```

### 3. Error Recovery Strategies

#### State Transition Recovery
```rust
impl CorticalColumn {
    pub fn try_activate_with_retry(&self) -> Result<(), StateTransitionError> {
        for attempt in 0..3 {
            match self.try_activate() {
                Ok(()) => return Ok(()),
                Err(StateTransitionError::ConcurrentModification) => {
                    // Brief backoff before retry
                    std::thread::sleep(Duration::from_nanos(100 * (1 << attempt)));
                    continue;
                },
                Err(e) => return Err(e), // Don't retry other errors
            }
        }
        Err(StateTransitionError::ConcurrentModification)
    }
    
    pub fn force_reset_if_stuck(&self) -> Result<(), StateTransitionError> {
        // Emergency recovery for stuck states
        match self.current_state() {
            ColumnState::Available => Ok(()), // Already good
            _ => {
                // Force reset using atomic operations
                self.atomic_state.force_reset();
                Ok(())
            }
        }
    }
}
```

#### Biological Process Recovery
```rust
impl BiologicalCorticalColumn {
    pub fn stimulate_with_fallback(&self, strength: f32, duration: f32) -> Result<StimulationResult, BiologicalError> {
        match self.stimulate(strength, duration) {
            Ok(result) => Ok(result),
            Err(BiologicalError::RefractoryPeriod { remaining_ms }) => {
                // Wait for refractory period to end
                std::thread::sleep(Duration::from_millis(remaining_ms as u64));
                self.stimulate(strength, duration)
            },
            Err(BiologicalError::ThresholdNotMet { potential, threshold }) => {
                // Try with adjusted strength
                let adjusted_strength = strength * (threshold / potential).min(2.0);
                self.stimulate(adjusted_strength, duration)
            },
            Err(e) => Err(e),
        }
    }
}
```

#### Spatial Query Recovery
```rust
impl SpatialIndexer {
    pub fn find_neighbors_with_fallback(&self, query: &SpatialQuery) -> Result<Vec<u32>, SpatialError> {
        match self.find_neighbors(query) {
            Ok(neighbors) => Ok(neighbors),
            Err(SpatialError::IndexNotBuilt) => {
                // Rebuild index and retry
                self.build_index()?;
                self.find_neighbors(query)
            },
            Err(SpatialError::QueryTimeout { .. }) => {
                // Try simplified query
                let simplified = query.with_reduced_scope();
                self.find_neighbors(&simplified)
            },
            Err(e) => Err(e),
        }
    }
}
```

### 4. Error Propagation Patterns

#### Chain of Results
```rust
pub fn full_allocation_pipeline(&self, concept: &Concept) -> Result<AllocationResult, NeuromorphicError> {
    // Each step can fail, errors bubble up with context
    let neural_result = self.neural_inference(&concept.features)
        .map_err(NeuromorphicError::Neural)?;
    
    let spatial_candidates = self.find_spatial_candidates(&concept.position)
        .map_err(NeuromorphicError::Spatial)?;
    
    let inhibition_result = self.apply_lateral_inhibition(&spatial_candidates)
        .map_err(|e| match e {
            InhibitionError::Convergence => NeuromorphicError::System(
                SystemError::PerformanceDegradation {
                    metric: "inhibition_convergence".to_string(),
                    value: "failed".to_string(),
                }
            ),
            _ => NeuromorphicError::Allocation(AllocationError::NoSuitableColumns),
        })?;
    
    let winner = self.select_winner(&inhibition_result)
        .map_err(NeuromorphicError::Allocation)?;
    
    Ok(AllocationResult::Success { winner })
}
```

#### Error Context Enhancement
```rust
use anyhow::{Context, Result as AnyhowResult};

pub fn allocate_with_context(&self, concept: &Concept) -> AnyhowResult<AllocationResult> {
    let neural_result = self.neural_inference(&concept.features)
        .with_context(|| format!("Neural inference failed for concept {}", concept.id))?;
    
    let spatial_candidates = self.find_spatial_candidates(&concept.position)
        .with_context(|| format!("Spatial search failed at position {:?}", concept.position))?;
    
    // ... rest of pipeline with context
}
```

### 5. Error Logging and Telemetry

#### Structured Error Logging
```rust
use tracing::{error, warn, info};

impl AllocationEngine {
    fn handle_allocation_error(&self, error: &AllocationError, concept_id: &str) {
        match error {
            AllocationError::NoSuitableColumns => {
                warn!(
                    concept_id = %concept_id,
                    error = %error,
                    "No suitable columns found - may need grid expansion"
                );
            },
            AllocationError::AllocationTimeout { timeout_ms } => {
                error!(
                    concept_id = %concept_id,
                    timeout_ms = %timeout_ms,
                    error = %error,
                    "Allocation timeout - performance issue detected"
                );
                // Trigger performance monitoring
                self.performance_monitor.record_timeout(*timeout_ms);
            },
            _ => {
                error!(
                    concept_id = %concept_id,
                    error = %error,
                    "Allocation failed"
                );
            }
        }
    }
}
```

#### Error Metrics Collection
```rust
pub struct ErrorMetrics {
    pub state_transition_errors: AtomicU64,
    pub biological_errors: AtomicU64,
    pub spatial_errors: AtomicU64,
    pub neural_errors: AtomicU64,
    pub allocation_errors: AtomicU64,
    pub system_errors: AtomicU64,
}

impl ErrorMetrics {
    pub fn record_error(&self, error: &NeuromorphicError) {
        match error {
            NeuromorphicError::State(_) => {
                self.state_transition_errors.fetch_add(1, Ordering::Relaxed);
            },
            NeuromorphicError::Biological(_) => {
                self.biological_errors.fetch_add(1, Ordering::Relaxed);
            },
            // ... etc
        }
    }
}
```

### 6. Testing Error Scenarios

#### Error Injection for Testing
```rust
#[cfg(test)]
pub struct ErrorInjector {
    inject_state_errors: AtomicBool,
    inject_spatial_errors: AtomicBool,
    inject_neural_errors: AtomicBool,
}

#[cfg(test)]
impl ErrorInjector {
    pub fn new() -> Self {
        Self {
            inject_state_errors: AtomicBool::new(false),
            inject_spatial_errors: AtomicBool::new(false),
            inject_neural_errors: AtomicBool::new(false),
        }
    }
    
    pub fn enable_state_errors(&self) {
        self.inject_state_errors.store(true, Ordering::Relaxed);
    }
    
    pub fn should_inject_state_error(&self) -> bool {
        self.inject_state_errors.load(Ordering::Relaxed)
    }
}
```

#### Comprehensive Error Tests
```rust
#[cfg(test)]
mod error_tests {
    use super::*;
    
    #[test]
    fn test_state_transition_error_recovery() {
        let column = CorticalColumn::new(1);
        let injector = ErrorInjector::new();
        injector.enable_state_errors();
        
        // Test retry mechanism
        let result = column.try_activate_with_retry();
        assert!(result.is_ok(), "Retry mechanism should handle transient errors");
    }
    
    #[test]
    fn test_error_propagation_chain() {
        let engine = AllocationEngine::new_with_error_injection();
        let concept = Concept::new("test");
        
        // Test that specific errors propagate correctly
        match engine.allocate_concept(&concept) {
            Err(NeuromorphicError::Neural(NeuralError::NetworkNotInitialized { .. })) => {
                // Expected error path
            },
            other => panic!("Unexpected result: {:?}", other),
        }
    }
    
    #[test]
    fn test_error_metrics_collection() {
        let metrics = ErrorMetrics::new();
        let error = NeuromorphicError::State(StateTransitionError::InvalidTransition {
            from: ColumnState::Available,
            to: ColumnState::Allocated,
        });
        
        metrics.record_error(&error);
        assert_eq!(metrics.state_transition_errors.load(Ordering::Relaxed), 1);
    }
}
```

### 7. Performance Error Handling

#### Timeout Management
```rust
pub struct TimeoutConfig {
    pub state_transition_timeout_ns: u64,
    pub neural_inference_timeout_ms: u64,
    pub spatial_query_timeout_us: u64,
    pub allocation_timeout_ms: u64,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            state_transition_timeout_ns: 1_000, // 1μs
            neural_inference_timeout_ms: 10,    // 10ms
            spatial_query_timeout_us: 100,      // 100μs
            allocation_timeout_ms: 50,          // 50ms
        }
    }
}
```

#### Circuit Breaker Pattern
```rust
pub struct CircuitBreaker {
    failure_count: AtomicU32,
    last_failure_time: AtomicU64,
    failure_threshold: u32,
    recovery_timeout_ms: u64,
    state: AtomicU8, // 0=Closed, 1=Open, 2=HalfOpen
}

impl CircuitBreaker {
    pub fn call<T, E, F>(&self, operation: F) -> Result<T, E>
    where
        F: FnOnce() -> Result<T, E>,
        E: std::error::Error,
    {
        match self.state.load(Ordering::Acquire) {
            0 => { // Closed - allow operation
                match operation() {
                    Ok(result) => {
                        self.failure_count.store(0, Ordering::Relaxed);
                        Ok(result)
                    },
                    Err(e) => {
                        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                        if failures >= self.failure_threshold {
                            self.open_circuit();
                        }
                        Err(e)
                    }
                }
            },
            1 => { // Open - fail fast
                Err(/* CircuitBreakerOpen error */)
            },
            2 => { // Half-open - try once
                match operation() {
                    Ok(result) => {
                        self.close_circuit();
                        Ok(result)
                    },
                    Err(e) => {
                        self.open_circuit();
                        Err(e)
                    }
                }
            },
            _ => unreachable!(),
        }
    }
}
```

## Implementation Checklist

### For Each Task:
- [ ] Define specific error types with descriptive messages
- [ ] Implement retry mechanisms for transient failures
- [ ] Add timeout handling for all operations
- [ ] Include error context and logging
- [ ] Write comprehensive error scenario tests
- [ ] Document error recovery strategies
- [ ] Add performance monitoring for error rates

### System-Wide:
- [ ] Consistent error type hierarchy
- [ ] Error metrics collection
- [ ] Circuit breaker implementation
- [ ] Structured error logging
- [ ] Error injection for testing
- [ ] Documentation of error handling patterns

## Success Criteria

1. **Consistency**: All Result types follow the same patterns
2. **Recoverability**: Transient errors can be recovered automatically
3. **Observability**: All errors are logged with appropriate context
4. **Performance**: Error handling doesn't impact critical path performance
5. **Testability**: Error scenarios are comprehensively tested
6. **Production-Ready**: Error handling is suitable for production deployment

This framework ensures that Phase 1 components have robust, consistent error handling that enables reliable operation in production environments.