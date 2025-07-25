//! Production-Ready Error Recovery System
//!
//! Provides comprehensive error recovery mechanisms with retry logic,
//! circuit breaker patterns, and graceful degradation for all operations.

use crate::error::{GraphError, Result};
use std::future::Future;
use std::sync::{Arc, atomic::{AtomicU32, AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use dashmap::DashMap;

/// Configuration for retry behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub recovery_timeout: Duration,
    pub success_threshold: u32,
    pub timeout: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            success_threshold: 3,
            timeout: Duration::from_secs(10),
        }
    }
}

/// Circuit breaker for operation protection
pub struct CircuitBreaker {
    state: std::sync::RwLock<CircuitState>,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    last_failure_time: AtomicU64,
    config: CircuitBreakerConfig,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: std::sync::RwLock::new(CircuitState::Closed),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            last_failure_time: AtomicU64::new(0),
            config,
        }
    }

    pub fn can_execute(&self) -> bool {
        let state = *self.state.read().unwrap();
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                let now = Instant::now().elapsed().as_secs();
                let last_failure = self.last_failure_time.load(Ordering::Relaxed);
                if now - last_failure > self.config.recovery_timeout.as_secs() {
                    self.transition_to_half_open();
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    pub fn record_success(&self) {
        let state = *self.state.read().unwrap();
        match state {
            CircuitState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                if success_count >= self.config.success_threshold {
                    self.transition_to_closed();
                }
            }
            CircuitState::Closed => {
                self.failure_count.store(0, Ordering::Relaxed);
            }
            _ => {}
        }
    }

    pub fn record_failure(&self) {
        let failure_count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.last_failure_time.store(
            Instant::now().elapsed().as_secs(),
            Ordering::Relaxed,
        );

        if failure_count >= self.config.failure_threshold {
            self.transition_to_open();
        }
    }

    fn transition_to_closed(&self) {
        *self.state.write().unwrap() = CircuitState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
    }

    fn transition_to_open(&self) {
        *self.state.write().unwrap() = CircuitState::Open;
        self.success_count.store(0, Ordering::Relaxed);
    }

    fn transition_to_half_open(&self) {
        *self.state.write().unwrap() = CircuitState::HalfOpen;
        self.success_count.store(0, Ordering::Relaxed);
    }

    pub fn get_state(&self) -> CircuitState {
        *self.state.read().unwrap()
    }
}

/// Error recovery manager with circuit breakers and retry logic
pub struct ErrorRecoveryManager {
    circuit_breakers: DashMap<String, Arc<CircuitBreaker>>,
    retry_configs: DashMap<String, RetryConfig>,
    operation_stats: DashMap<String, OperationStats>,
}

#[derive(Debug, Default)]
pub struct OperationStats {
    pub total_attempts: AtomicU64,
    pub successful_attempts: AtomicU64,
    pub failed_attempts: AtomicU64,
    pub circuit_breaker_trips: AtomicU64,
    pub total_retry_attempts: AtomicU64,
    pub avg_execution_time_ms: AtomicU64,
}

impl ErrorRecoveryManager {
    pub fn new() -> Self {
        Self {
            circuit_breakers: DashMap::new(),
            retry_configs: DashMap::new(),
            operation_stats: DashMap::new(),
        }
    }

    /// Configure retry behavior for a specific operation
    pub fn configure_retry(&self, operation: &str, config: RetryConfig) {
        self.retry_configs.insert(operation.to_string(), config);
    }

    /// Configure circuit breaker for a specific operation
    pub fn configure_circuit_breaker(&self, operation: &str, config: CircuitBreakerConfig) {
        let circuit_breaker = Arc::new(CircuitBreaker::new(config));
        self.circuit_breakers.insert(operation.to_string(), circuit_breaker);
    }

    /// Execute operation with comprehensive error recovery
    pub async fn execute_with_recovery<T, F, Fut>(
        &self,
        operation_name: &str,
        operation: F,
    ) -> Result<T>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: Future<Output = Result<T>> + Send,
        T: Send,
    {
        let start_time = Instant::now();
        
        // Get or create operation stats
        let stats = self.operation_stats
            .entry(operation_name.to_string())
            .or_insert_with(OperationStats::default);
        
        stats.total_attempts.fetch_add(1, Ordering::Relaxed);

        // Check circuit breaker
        if let Some(circuit_breaker) = self.circuit_breakers.get(operation_name) {
            if !circuit_breaker.can_execute() {
                stats.circuit_breaker_trips.fetch_add(1, Ordering::Relaxed);
                return Err(GraphError::ResourceExhausted {
                    resource: format!("Circuit breaker open for operation: {}", operation_name),
                });
            }
        }

        // Get retry configuration
        let retry_config = self.retry_configs
            .get(operation_name)
            .map(|config| config.clone())
            .unwrap_or_default();

        let mut last_error = None;
        let mut attempt = 0;

        while attempt <= retry_config.max_retries {
            match self.execute_with_timeout(operation(), Duration::from_secs(30)).await {
                Ok(result) => {
                    // Record success
                    stats.successful_attempts.fetch_add(1, Ordering::Relaxed);
                    if let Some(circuit_breaker) = self.circuit_breakers.get(operation_name) {
                        circuit_breaker.record_success();
                    }
                    
                    // Update execution time
                    let execution_time = start_time.elapsed().as_millis() as u64;
                    self.update_avg_execution_time(&stats, execution_time);
                    
                    return Ok(result);
                }
                Err(error) => {
                    last_error = Some(error.clone());
                    attempt += 1;
                    stats.total_retry_attempts.fetch_add(1, Ordering::Relaxed);

                    // Check if error is retryable
                    if !self.is_retryable_error(&error) || attempt > retry_config.max_retries {
                        break;
                    }

                    // Calculate delay with exponential backoff and jitter
                    let delay = self.calculate_delay(&retry_config, attempt);
                    sleep(delay).await;
                }
            }
        }

        // Record failure
        stats.failed_attempts.fetch_add(1, Ordering::Relaxed);
        if let Some(circuit_breaker) = self.circuit_breakers.get(operation_name) {
            circuit_breaker.record_failure();
        }

        Err(last_error.unwrap_or_else(|| {
            GraphError::ProcessingError("Unknown error during retry".to_string())
        }))
    }

    async fn execute_with_timeout<T>(
        &self,
        operation: impl Future<Output = Result<T>>,
        timeout: Duration,
    ) -> Result<T> {
        match tokio::time::timeout(timeout, operation).await {
            Ok(result) => result,
            Err(_) => Err(GraphError::QueryTimeout {
                timeout_ms: timeout.as_millis() as u64,
            }),
        }
    }

    fn is_retryable_error(&self, error: &GraphError) -> bool {
        match error {
            // Retryable errors
            GraphError::QueryTimeout { .. } |
            GraphError::DatabaseConnectionError(_) |
            GraphError::TransactionError(_) |
            GraphError::ResourceExhausted { .. } => true,
            
            // Non-retryable errors
            GraphError::EntityNotFound { .. } |
            GraphError::InvalidInput(_) |
            GraphError::InvalidConfiguration(_) |
            GraphError::SecurityViolation(_) => false,
            
            // Default to retryable for unknown errors
            _ => true,
        }
    }

    fn calculate_delay(&self, config: &RetryConfig, attempt: u32) -> Duration {
        let base_delay = config.base_delay_ms as f64;
        let multiplier = config.backoff_multiplier.powi(attempt as i32 - 1);
        let delay_ms = (base_delay * multiplier).min(config.max_delay_ms as f64);
        
        // Add jitter
        let jitter = delay_ms * config.jitter_factor * (rand::random::<f64>() - 0.5);
        let final_delay = (delay_ms + jitter).max(0.0) as u64;
        
        Duration::from_millis(final_delay)
    }

    fn update_avg_execution_time(&self, stats: &OperationStats, new_time: u64) {
        let current_avg = stats.avg_execution_time_ms.load(Ordering::Relaxed);
        let total_attempts = stats.total_attempts.load(Ordering::Relaxed);
        
        let new_avg = if total_attempts == 1 {
            new_time
        } else {
            (current_avg * (total_attempts - 1) + new_time) / total_attempts
        };
        
        stats.avg_execution_time_ms.store(new_avg, Ordering::Relaxed);
    }

    /// Get operation statistics
    pub fn get_operation_stats(&self, operation_name: &str) -> Option<HashMap<String, u64>> {
        self.operation_stats.get(operation_name).map(|stats| {
            let mut map = HashMap::new();
            map.insert("total_attempts".to_string(), stats.total_attempts.load(Ordering::Relaxed));
            map.insert("successful_attempts".to_string(), stats.successful_attempts.load(Ordering::Relaxed));
            map.insert("failed_attempts".to_string(), stats.failed_attempts.load(Ordering::Relaxed));
            map.insert("circuit_breaker_trips".to_string(), stats.circuit_breaker_trips.load(Ordering::Relaxed));
            map.insert("total_retry_attempts".to_string(), stats.total_retry_attempts.load(Ordering::Relaxed));
            map.insert("avg_execution_time_ms".to_string(), stats.avg_execution_time_ms.load(Ordering::Relaxed));
            map
        })
    }

    /// Get circuit breaker status
    pub fn get_circuit_breaker_status(&self, operation_name: &str) -> Option<CircuitState> {
        self.circuit_breakers.get(operation_name).map(|cb| cb.get_state())
    }

    /// Get all operation statistics
    pub fn get_all_stats(&self) -> HashMap<String, HashMap<String, u64>> {
        let mut all_stats = HashMap::new();
        for item in self.operation_stats.iter() {
            if let Some(stats) = self.get_operation_stats(item.key()) {
                all_stats.insert(item.key().clone(), stats);
            }
        }
        all_stats
    }

    /// Reset statistics for an operation
    pub fn reset_operation_stats(&self, operation_name: &str) {
        if let Some(stats) = self.operation_stats.get(operation_name) {
            stats.total_attempts.store(0, Ordering::Relaxed);
            stats.successful_attempts.store(0, Ordering::Relaxed);
            stats.failed_attempts.store(0, Ordering::Relaxed);
            stats.circuit_breaker_trips.store(0, Ordering::Relaxed);
            stats.total_retry_attempts.store(0, Ordering::Relaxed);
            stats.avg_execution_time_ms.store(0, Ordering::Relaxed);
        }
    }

    /// Health check for error recovery system
    pub fn health_check(&self) -> HashMap<String, serde_json::Value> {
        let mut health = HashMap::new();
        
        // Overall system health
        let total_operations = self.operation_stats.len();
        let mut healthy_operations = 0;
        let mut degraded_operations = 0;
        let mut failed_operations = 0;
        
        for item in self.operation_stats.iter() {
            let stats = item.value();
            let total = stats.total_attempts.load(Ordering::Relaxed);
            let successful = stats.successful_attempts.load(Ordering::Relaxed);
            let _failed = stats.failed_attempts.load(Ordering::Relaxed);
            
            if total == 0 {
                continue;
            }
            
            let success_rate = successful as f64 / total as f64;
            
            if success_rate >= 0.95 {
                healthy_operations += 1;
            } else if success_rate >= 0.80 {
                degraded_operations += 1;
            } else {
                failed_operations += 1;
            }
        }
        
        health.insert("total_operations".to_string(), serde_json::json!(total_operations));
        health.insert("healthy_operations".to_string(), serde_json::json!(healthy_operations));
        health.insert("degraded_operations".to_string(), serde_json::json!(degraded_operations));
        health.insert("failed_operations".to_string(), serde_json::json!(failed_operations));
        
        // Circuit breaker status
        let mut circuit_breaker_status = HashMap::new();
        for item in self.circuit_breakers.iter() {
            circuit_breaker_status.insert(
                item.key().clone(),
                serde_json::json!(format!("{:?}", item.value().get_state()))
            );
        }
        health.insert("circuit_breakers".to_string(), serde_json::json!(circuit_breaker_status));
        
        health
    }
}

/// Default configurations for common operations
impl ErrorRecoveryManager {
    pub fn configure_defaults(&self) {
        // Storage operations
        self.configure_retry("store_fact", RetryConfig {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 2000,
            backoff_multiplier: 1.5,
            jitter_factor: 0.1,
        });
        
        self.configure_circuit_breaker("store_fact", CircuitBreakerConfig {
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
            success_threshold: 2,
            timeout: Duration::from_secs(10),
        });
        
        // Query operations  
        self.configure_retry("find_facts", RetryConfig {
            max_retries: 2,
            base_delay_ms: 50,
            max_delay_ms: 1000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.15,
        });
        
        self.configure_circuit_breaker("find_facts", CircuitBreakerConfig {
            failure_threshold: 8,
            recovery_timeout: Duration::from_secs(20),
            success_threshold: 3,
            timeout: Duration::from_secs(5),
        });
        
        // Advanced operations
        self.configure_retry("hybrid_search", RetryConfig {
            max_retries: 2,
            base_delay_ms: 200,
            max_delay_ms: 3000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.2,
        });
        
        self.configure_circuit_breaker("hybrid_search", CircuitBreakerConfig {
            failure_threshold: 3,
            recovery_timeout: Duration::from_secs(60),
            success_threshold: 2,
            timeout: Duration::from_secs(15),
        });
        
        // Graph analysis operations
        self.configure_retry("analyze_graph", RetryConfig {
            max_retries: 1,
            base_delay_ms: 500,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.25,
        });
        
        self.configure_circuit_breaker("analyze_graph", CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_secs(120),
            success_threshold: 1,
            timeout: Duration::from_secs(30),
        });
    }
}

impl Clone for ErrorRecoveryManager {
    fn clone(&self) -> Self {
        Self {
            circuit_breakers: self.circuit_breakers.clone(),
            retry_configs: self.retry_configs.clone(),
            operation_stats: DashMap::new(), // Reset stats for cloned instance
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_retry_mechanism() {
        let manager = ErrorRecoveryManager::new();
        manager.configure_retry("test_op", RetryConfig {
            max_retries: 2,
            base_delay_ms: 10,
            max_delay_ms: 100,
            backoff_multiplier: 2.0,
            jitter_factor: 0.0,
        });

        let attempt_count = Arc::new(AtomicU32::new(0));
        let result = manager.execute_with_recovery("test_op", || {
            let count = attempt_count.fetch_add(1, Ordering::Relaxed) + 1;
            async move {
                if count < 3 {
                    Err(GraphError::TransactionError("Temporary failure".to_string()))
                } else {
                    Ok("Success".to_string())
                }
            }
        }).await;

        assert!(result.is_ok());
        assert_eq!(attempt_count.load(Ordering::Relaxed), 3);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            recovery_timeout: Duration::from_millis(100),
            success_threshold: 1,
            timeout: Duration::from_secs(1),
        };
        
        let circuit_breaker = CircuitBreaker::new(config);
        
        // Initially closed
        assert_eq!(circuit_breaker.get_state(), CircuitState::Closed);
        assert!(circuit_breaker.can_execute());
        
        // Trigger failures to open circuit
        circuit_breaker.record_failure();
        circuit_breaker.record_failure();
        
        assert_eq!(circuit_breaker.get_state(), CircuitState::Open);
        assert!(!circuit_breaker.can_execute());
        
        // Wait for recovery timeout
        sleep(Duration::from_millis(150)).await;
        
        // Should transition to half-open
        assert!(circuit_breaker.can_execute());
        assert_eq!(circuit_breaker.get_state(), CircuitState::HalfOpen);
        
        // Record success to close circuit
        circuit_breaker.record_success();
        assert_eq!(circuit_breaker.get_state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_operation_stats() {
        let manager = ErrorRecoveryManager::new();
        
        // Execute successful operation
        let _result = manager.execute_with_recovery("test_stats", || async {
            Ok("Success")
        }).await;
        
        let stats = manager.get_operation_stats("test_stats").unwrap();
        assert_eq!(stats["total_attempts"], 1);
        assert_eq!(stats["successful_attempts"], 1);
        assert_eq!(stats["failed_attempts"], 0);
    }
}