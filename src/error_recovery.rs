use crate::error::{GraphError, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use std::sync::Arc;

/// Circuit breaker implementation for error handling
pub struct CircuitBreaker {
    failure_threshold: u32,
    reset_timeout: Duration,
    failure_count: Arc<RwLock<u32>>,
    last_failure: Arc<RwLock<Option<Instant>>>,
    state: Arc<RwLock<CircuitBreakerState>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitBreakerState {
    Closed,     // Normal operation
    Open,       // Failing fast
    HalfOpen,   // Testing if service recovered
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failure_threshold,
            reset_timeout,
            failure_count: Arc::new(RwLock::new(0)),
            last_failure: Arc::new(RwLock::new(None)),
            state: Arc::new(RwLock::new(CircuitBreakerState::Closed)),
        }
    }
    
    pub async fn execute<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let state = self.state.read().await;
        
        match *state {
            CircuitBreakerState::Open => {
                // Check if we should transition to half-open
                if let Some(last_failure) = *self.last_failure.read().await {
                    if last_failure.elapsed() > self.reset_timeout {
                        drop(state);
                        let mut state = self.state.write().await;
                        *state = CircuitBreakerState::HalfOpen;
                        drop(state);
                        return self.execute_with_monitoring(operation).await;
                    }
                }
                
                Err(GraphError::OperationTimeout("Circuit breaker is open".to_string()))
            }
            CircuitBreakerState::HalfOpen | CircuitBreakerState::Closed => {
                drop(state);
                self.execute_with_monitoring(operation).await
            }
        }
    }
    
    async fn execute_with_monitoring<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let result = operation();
        
        match result {
            Ok(value) => {
                // Reset failure count on success
                let mut failure_count = self.failure_count.write().await;
                *failure_count = 0;
                
                // Transition to closed state if we were half-open
                let mut state = self.state.write().await;
                if *state == CircuitBreakerState::HalfOpen {
                    *state = CircuitBreakerState::Closed;
                }
                
                Ok(value)
            }
            Err(error) => {
                // Increment failure count
                let mut failure_count = self.failure_count.write().await;
                *failure_count += 1;
                
                // Update last failure time
                let mut last_failure = self.last_failure.write().await;
                *last_failure = Some(Instant::now());
                
                // Check if we should open the circuit
                if *failure_count >= self.failure_threshold {
                    let mut state = self.state.write().await;
                    *state = CircuitBreakerState::Open;
                }
                
                Err(error)
            }
        }
    }
    
    pub async fn get_state(&self) -> CircuitBreakerState {
        self.state.read().await.clone()
    }
    
    pub async fn reset(&self) {
        let mut failure_count = self.failure_count.write().await;
        *failure_count = 0;
        
        let mut last_failure = self.last_failure.write().await;
        *last_failure = None;
        
        let mut state = self.state.write().await;
        *state = CircuitBreakerState::Closed;
    }
}

/// Retry policy for operations
pub struct RetryPolicy {
    max_attempts: u32,
    base_delay: Duration,
    max_delay: Duration,
    backoff_multiplier: f64,
}

impl RetryPolicy {
    pub fn new(max_attempts: u32, base_delay: Duration, max_delay: Duration) -> Self {
        Self {
            max_attempts,
            base_delay,
            max_delay,
            backoff_multiplier: 2.0,
        }
    }
    
    pub async fn execute<F, Fut, T>(&self, operation: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut last_error = None;
        let mut delay = self.base_delay;
        
        for attempt in 1..=self.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = Some(error);
                    
                    if attempt < self.max_attempts {
                        if self.should_retry(&last_error.as_ref().unwrap()) {
                            tokio::time::sleep(delay).await;
                            delay = std::cmp::min(
                                Duration::from_millis((delay.as_millis() as f64 * self.backoff_multiplier) as u64),
                                self.max_delay,
                            );
                        } else {
                            break;
                        }
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| GraphError::OperationTimeout("All retry attempts exhausted".to_string())))
    }
    
    fn should_retry(&self, error: &GraphError) -> bool {
        match error {
            GraphError::OperationTimeout(_) => true,
            GraphError::DatabaseConnectionError(_) => true,
            GraphError::TransactionError(_) => true,
            GraphError::ResourceExhausted { .. } => true,
            GraphError::IndexCorruption => false,
            GraphError::ConsistencyViolation(_) => false,
            GraphError::SecurityViolation(_) => false,
            _ => true,
        }
    }
}

/// Health check system for monitoring component health
pub struct HealthChecker {
    checks: HashMap<String, Box<dyn HealthCheck + Send + Sync>>,
    check_interval: Duration,
}

pub trait HealthCheck {
    fn name(&self) -> &str;
    fn check(&self) -> Box<dyn std::future::Future<Output = HealthStatus> + Send + Unpin>;
}

#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub healthy: bool,
    pub message: String,
    pub details: HashMap<String, String>,
}

impl HealthChecker {
    pub fn new(check_interval: Duration) -> Self {
        Self {
            checks: HashMap::new(),
            check_interval,
        }
    }
    
    pub fn add_check(&mut self, check: Box<dyn HealthCheck + Send + Sync>) {
        self.checks.insert(check.name().to_string(), check);
    }
    
    pub async fn check_all(&self) -> HashMap<String, HealthStatus> {
        let mut results = HashMap::new();
        
        for (name, check) in &self.checks {
            let status = tokio::time::timeout(
                Duration::from_secs(10),
                check.check(),
            ).await;
            
            let health_status = match status {
                Ok(status) => status,
                Err(_) => HealthStatus {
                    healthy: false,
                    message: "Health check timeout".to_string(),
                    details: HashMap::new(),
                },
            };
            
            results.insert(name.clone(), health_status);
        }
        
        results
    }
    
    pub async fn check_specific(&self, name: &str) -> Option<HealthStatus> {
        if let Some(check) = self.checks.get(name) {
            let status = tokio::time::timeout(
                Duration::from_secs(10),
                check.check(),
            ).await;
            
            match status {
                Ok(status) => Some(status),
                Err(_) => Some(HealthStatus {
                    healthy: false,
                    message: "Health check timeout".to_string(),
                    details: HashMap::new(),
                }),
            }
        } else {
            None
        }
    }
}

/// Error recovery coordinator
pub struct ErrorRecoveryCoordinator {
    circuit_breakers: HashMap<String, CircuitBreaker>,
    retry_policies: HashMap<String, RetryPolicy>,
    health_checker: HealthChecker,
}

impl ErrorRecoveryCoordinator {
    pub fn new() -> Self {
        Self {
            circuit_breakers: HashMap::new(),
            retry_policies: HashMap::new(),
            health_checker: HealthChecker::new(Duration::from_secs(30)),
        }
    }
    
    pub fn add_circuit_breaker(&mut self, name: String, breaker: CircuitBreaker) {
        self.circuit_breakers.insert(name, breaker);
    }
    
    pub fn add_retry_policy(&mut self, name: String, policy: RetryPolicy) {
        self.retry_policies.insert(name, policy);
    }
    
    pub fn add_health_check(&mut self, check: Box<dyn HealthCheck + Send + Sync>) {
        self.health_checker.add_check(check);
    }
    
    pub async fn execute_with_recovery<F, Fut, T>(&self, component: &str, operation: F) -> Result<T>
    where
        F: Fn() -> Fut + Clone,
        Fut: std::future::Future<Output = Result<T>>,
    {
        // First try with circuit breaker
        if let Some(breaker) = self.circuit_breakers.get(component) {
            let breaker_result = breaker.execute(|| {
                // Execute with retry policy if available
                if let Some(policy) = self.retry_policies.get(component) {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            policy.execute(operation.clone()).await
                        })
                    })
                } else {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async {
                            operation().await
                        })
                    })
                }
            }).await;
            
            return breaker_result;
        }
        
        // Fall back to retry policy only
        if let Some(policy) = self.retry_policies.get(component) {
            return policy.execute(operation).await;
        }
        
        // No recovery mechanism, execute directly
        operation().await
    }
    
    pub async fn get_system_health(&self) -> HashMap<String, HealthStatus> {
        self.health_checker.check_all().await
    }
}

impl Default for ErrorRecoveryCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

