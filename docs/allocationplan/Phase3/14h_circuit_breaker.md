# Task 14h: Implement Circuit Breaker Pattern

**Time**: 7 minutes (1.5 min read, 4 min implement, 1.5 min verify)
**Dependencies**: 14g_error_metrics.md
**Stage**: Inheritance System

## Objective
Add circuit breaker pattern to prevent cascading failures in inheritance operations.

## Implementation
Create `src/inheritance/circuit_breaker.rs`:

```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use crate::inheritance::error_types::InheritanceError;

#[derive(Debug, Clone)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing fast
    HalfOpen, // Testing if service recovered
}

pub struct CircuitBreaker {
    state: Arc<RwLock<CircuitBreakerState>>,
    config: CircuitBreakerConfig,
}

#[derive(Debug)]
struct CircuitBreakerState {
    current_state: CircuitState,
    failure_count: u32,
    last_failure_time: Option<Instant>,
    last_success_time: Option<Instant>,
    consecutive_successes: u32,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub timeout_duration: Duration,
    pub success_threshold: u32,
    pub half_open_timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            timeout_duration: Duration::from_secs(60),
            success_threshold: 3,
            half_open_timeout: Duration::from_secs(30),
        }
    }
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitBreakerState {
                current_state: CircuitState::Closed,
                failure_count: 0,
                last_failure_time: None,
                last_success_time: None,
                consecutive_successes: 0,
            })),
            config,
        }
    }

    pub async fn call<T, F, Fut>(&self, operation: F) -> Result<T, CircuitBreakerError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<T, InheritanceError>>,
    {
        // Check if circuit is open
        if self.is_open().await {
            return Err(CircuitBreakerError::CircuitOpen);
        }

        let start_time = Instant::now();
        
        match operation().await {
            Ok(result) => {
                self.record_success().await;
                Ok(result)
            }
            Err(error) => {
                self.record_failure().await;
                Err(CircuitBreakerError::OperationFailed(error))
            }
        }
    }

    async fn is_open(&self) -> bool {
        let state = self.state.read().await;
        
        match state.current_state {
            CircuitState::Open => {
                // Check if timeout has passed
                if let Some(last_failure) = state.last_failure_time {
                    if last_failure.elapsed() >= self.config.timeout_duration {
                        drop(state);
                        self.transition_to_half_open().await;
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            }
            CircuitState::HalfOpen => {
                // Allow limited requests in half-open state
                false
            }
            CircuitState::Closed => false,
        }
    }

    async fn record_success(&self) {
        let mut state = self.state.write().await;
        
        match state.current_state {
            CircuitState::HalfOpen => {
                state.consecutive_successes += 1;
                if state.consecutive_successes >= self.config.success_threshold {
                    state.current_state = CircuitState::Closed;
                    state.failure_count = 0;
                    state.consecutive_successes = 0;
                }
            }
            CircuitState::Closed => {
                state.failure_count = 0;
            }
            CircuitState::Open => {
                // Should not happen, but reset if it does
                state.current_state = CircuitState::Closed;
                state.failure_count = 0;
            }
        }
        
        state.last_success_time = Some(Instant::now());
    }

    async fn record_failure(&self) {
        let mut state = self.state.write().await;
        
        state.failure_count += 1;
        state.last_failure_time = Some(Instant::now());
        state.consecutive_successes = 0;
        
        match state.current_state {
            CircuitState::Closed => {
                if state.failure_count >= self.config.failure_threshold {
                    state.current_state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                state.current_state = CircuitState::Open;
            }
            CircuitState::Open => {
                // Already open, just update failure time
            }
        }
    }

    async fn transition_to_half_open(&self) {
        let mut state = self.state.write().await;
        state.current_state = CircuitState::HalfOpen;
        state.consecutive_successes = 0;
    }

    pub async fn get_state(&self) -> CircuitState {
        self.state.read().await.current_state.clone()
    }

    pub async fn get_stats(&self) -> CircuitBreakerStats {
        let state = self.state.read().await;
        
        CircuitBreakerStats {
            current_state: state.current_state.clone(),
            failure_count: state.failure_count,
            consecutive_successes: state.consecutive_successes,
            time_since_last_failure: state.last_failure_time.map(|t| t.elapsed()),
            time_since_last_success: state.last_success_time.map(|t| t.elapsed()),
        }
    }

    pub async fn force_open(&self) {
        let mut state = self.state.write().await;
        state.current_state = CircuitState::Open;
        state.last_failure_time = Some(Instant::now());
    }

    pub async fn force_close(&self) {
        let mut state = self.state.write().await;
        state.current_state = CircuitState::Closed;
        state.failure_count = 0;
        state.consecutive_successes = 0;
    }
}

#[derive(Debug)]
pub enum CircuitBreakerError {
    CircuitOpen,
    OperationFailed(InheritanceError),
}

impl std::fmt::Display for CircuitBreakerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitBreakerError::CircuitOpen => write!(f, "Circuit breaker is open"),
            CircuitBreakerError::OperationFailed(err) => write!(f, "Operation failed: {}", err),
        }
    }
}

impl std::error::Error for CircuitBreakerError {}

#[derive(Debug)]
pub struct CircuitBreakerStats {
    pub current_state: CircuitState,
    pub failure_count: u32,
    pub consecutive_successes: u32,
    pub time_since_last_failure: Option<Duration>,
    pub time_since_last_success: Option<Duration>,
}
```

## Success Criteria
- Circuit breaker prevents cascading failures
- State transitions work correctly
- Statistics are accurately tracked

## Next Task
14i_error_aggregation.md