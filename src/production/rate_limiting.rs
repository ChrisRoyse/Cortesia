//! Production Rate Limiting and Resource Management System
//!
//! Provides comprehensive rate limiting, resource management, and system protection
//! mechanisms to prevent resource exhaustion and ensure system stability.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{RwLock, Semaphore};
use serde::{Serialize, Deserialize};
use dashmap::DashMap;
use crate::error::{GraphError, Result};

/// Rate limiting algorithm types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    TokenBucket,
    SlidingWindow,
    FixedWindow,
    LeakyBucket,
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_second: u32,
    pub burst_capacity: u32,
    pub window_size_seconds: u64,
    pub algorithm: RateLimitAlgorithm,
    pub enabled: bool,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 100,
            burst_capacity: 200,
            window_size_seconds: 60,
            algorithm: RateLimitAlgorithm::TokenBucket,
            enabled: true,
        }
    }
}

/// Resource limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_bytes: u64,
    pub max_cpu_percent: f32,
    pub max_concurrent_operations: u32,
    pub max_database_connections: u32,
    pub max_request_size_bytes: u64,
    pub max_response_size_bytes: u64,
    pub operation_timeout_seconds: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: 2_000_000_000, // 2GB
            max_cpu_percent: 80.0,
            max_concurrent_operations: 1000,
            max_database_connections: 100,
            max_request_size_bytes: 10_000_000, // 10MB
            max_response_size_bytes: 50_000_000, // 50MB
            operation_timeout_seconds: 30,
        }
    }
}

/// Token bucket for rate limiting
pub struct TokenBucket {
    tokens: AtomicU32,
    capacity: u32,
    refill_rate: u32, // tokens per second
    last_refill: AtomicU64,
}

impl TokenBucket {
    pub fn new(capacity: u32, refill_rate: u32) -> Self {
        Self {
            tokens: AtomicU32::new(capacity),
            capacity,
            refill_rate,
            last_refill: AtomicU64::new(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64
            ),
        }
    }

    pub fn try_consume(&self, tokens: u32) -> bool {
        self.refill_tokens();
        
        loop {
            let current_tokens = self.tokens.load(Ordering::Relaxed);
            if current_tokens < tokens {
                return false;
            }
            
            if self.tokens.compare_exchange_weak(
                current_tokens,
                current_tokens - tokens,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ).is_ok() {
                return true;
            }
        }
    }

    fn refill_tokens(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let last_refill = self.last_refill.load(Ordering::Relaxed);
        let elapsed_ms = now.saturating_sub(last_refill);
        
        if elapsed_ms > 0 {
            let tokens_to_add = (elapsed_ms * self.refill_rate as u64) / 1000;
            if tokens_to_add > 0 {
                let current_tokens = self.tokens.load(Ordering::Relaxed);
                let new_tokens = (current_tokens + tokens_to_add as u32).min(self.capacity);
                
                if self.tokens.compare_exchange_weak(
                    current_tokens,
                    new_tokens,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ).is_ok() {
                    self.last_refill.store(now, Ordering::Relaxed);
                }
            }
        }
    }

    pub fn available_tokens(&self) -> u32 {
        self.refill_tokens();
        self.tokens.load(Ordering::Relaxed)
    }
}

/// Sliding window rate limiter
pub struct SlidingWindowRateLimiter {
    requests: Arc<RwLock<Vec<u64>>>,
    window_size: Duration,
    max_requests: u32,
}

impl SlidingWindowRateLimiter {
    pub fn new(window_size: Duration, max_requests: u32) -> Self {
        Self {
            requests: Arc::new(RwLock::new(Vec::new())),
            window_size,
            max_requests,
        }
    }

    pub async fn try_consume(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let cutoff_time = now - self.window_size.as_millis() as u64;
        
        let mut requests = self.requests.write().await;
        
        // Remove old requests
        requests.retain(|&timestamp| timestamp > cutoff_time);
        
        // Check if we can add a new request
        if requests.len() < self.max_requests as usize {
            requests.push(now);
            true
        } else {
            false
        }
    }

    pub async fn current_count(&self) -> u32 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let cutoff_time = now - self.window_size.as_millis() as u64;
        
        let requests = self.requests.read().await;
        requests.iter().filter(|&&timestamp| timestamp > cutoff_time).count() as u32
    }
}

/// Resource usage tracker
#[derive(Debug, Default)]
pub struct ResourceUsage {
    pub memory_bytes: AtomicU64,
    pub cpu_percent: AtomicU32, // Stored as integer percentage * 100
    pub active_operations: AtomicU32,
    pub database_connections: AtomicU32,
    pub peak_memory_bytes: AtomicU64,
    pub peak_operations: AtomicU32,
}

impl ResourceUsage {
    pub fn set_memory_usage(&self, bytes: u64) {
        self.memory_bytes.store(bytes, Ordering::Relaxed);
        
        // Update peak if necessary
        loop {
            let current_peak = self.peak_memory_bytes.load(Ordering::Relaxed);
            if bytes <= current_peak {
                break;
            }
            
            if self.peak_memory_bytes.compare_exchange_weak(
                current_peak,
                bytes,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ).is_ok() {
                break;
            }
        }
    }

    pub fn set_cpu_usage(&self, percent: f32) {
        let percent_int = (percent * 100.0) as u32;
        self.cpu_percent.store(percent_int, Ordering::Relaxed);
    }

    pub fn increment_operations(&self) -> u32 {
        let current = self.active_operations.fetch_add(1, Ordering::Relaxed) + 1;
        
        // Update peak if necessary
        loop {
            let current_peak = self.peak_operations.load(Ordering::Relaxed);
            if current <= current_peak {
                break;
            }
            
            if self.peak_operations.compare_exchange_weak(
                current_peak,
                current,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ).is_ok() {
                break;
            }
        }
        
        current
    }

    pub fn decrement_operations(&self) -> u32 {
        self.active_operations.fetch_sub(1, Ordering::Relaxed).saturating_sub(1)
    }

    pub fn get_memory_usage(&self) -> u64 {
        self.memory_bytes.load(Ordering::Relaxed)
    }

    pub fn get_cpu_usage(&self) -> f32 {
        self.cpu_percent.load(Ordering::Relaxed) as f32 / 100.0
    }

    pub fn get_active_operations(&self) -> u32 {
        self.active_operations.load(Ordering::Relaxed)
    }
}

/// Comprehensive rate limiting and resource management system
pub struct RateLimitingManager {
    // Rate limiting
    global_rate_limiter: TokenBucket,
    operation_rate_limiters: DashMap<String, Arc<dyn RateLimiterTrait + Send + Sync>>,
    user_rate_limiters: DashMap<String, Arc<dyn RateLimiterTrait + Send + Sync>>,
    
    // Resource management
    resource_usage: Arc<ResourceUsage>,
    resource_limits: Arc<RwLock<ResourceLimits>>,
    operation_semaphore: Arc<Semaphore>,
    connection_pool: Arc<Semaphore>,
    
    // Configuration
    rate_limit_configs: DashMap<String, RateLimitConfig>,
    
    // Statistics
    rate_limit_stats: DashMap<String, RateLimitStats>,
    resource_stats: Arc<RwLock<ResourceStats>>,
}

trait RateLimiterTrait {
    fn try_consume(&self, tokens: u32) -> Result<bool>;
    fn available_capacity(&self) -> u32;
}

impl RateLimiterTrait for TokenBucket {
    fn try_consume(&self, tokens: u32) -> Result<bool> {
        Ok(self.try_consume(tokens))
    }

    fn available_capacity(&self) -> u32 {
        self.available_tokens()
    }
}

#[derive(Debug, Default)]
pub struct RateLimitStats {
    pub total_requests: AtomicU64,
    pub allowed_requests: AtomicU64,
    pub denied_requests: AtomicU64,
    pub last_reset: AtomicU64,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ResourceStats {
    pub peak_memory_usage: u64,
    pub peak_cpu_usage: f32,
    pub peak_operations: u32,
    pub total_operations: u64,
    pub resource_limit_violations: u64,
    pub connection_pool_exhaustions: u64,
}

impl RateLimitingManager {
    pub fn new(global_config: RateLimitConfig, resource_limits: ResourceLimits) -> Self {
        let operation_semaphore = Arc::new(Semaphore::new(resource_limits.max_concurrent_operations as usize));
        let connection_pool = Arc::new(Semaphore::new(resource_limits.max_database_connections as usize));
        
        Self {
            global_rate_limiter: TokenBucket::new(
                global_config.burst_capacity,
                global_config.requests_per_second,
            ),
            operation_rate_limiters: DashMap::new(),
            user_rate_limiters: DashMap::new(),
            
            resource_usage: Arc::new(ResourceUsage::default()),
            resource_limits: Arc::new(RwLock::new(resource_limits)),
            operation_semaphore,
            connection_pool,
            
            rate_limit_configs: DashMap::new(),
            rate_limit_stats: DashMap::new(),
            resource_stats: Arc::new(RwLock::new(ResourceStats::default())),
        }
    }

    /// Check if a request should be allowed based on rate limits
    pub async fn check_rate_limit(&self, operation: &str, user_id: Option<&str>) -> Result<()> {
        // Check global rate limit
        if !self.global_rate_limiter.try_consume(1) {
            self.record_rate_limit_denial("global").await;
            return Err(GraphError::ResourceExhausted {
                resource: "Global rate limit exceeded".to_string(),
            });
        }

        // Check operation-specific rate limit
        if let Some(op_limiter) = self.operation_rate_limiters.get(operation) {
            if !op_limiter.try_consume(1)? {
                self.record_rate_limit_denial(operation).await;
                return Err(GraphError::ResourceExhausted {
                    resource: format!("Rate limit exceeded for operation: {}", operation),
                });
            }
        }

        // Check user-specific rate limit
        if let Some(user_id) = user_id {
            if let Some(user_limiter) = self.user_rate_limiters.get(user_id) {
                if !user_limiter.try_consume(1)? {
                    self.record_rate_limit_denial(&format!("user_{}", user_id)).await;
                    return Err(GraphError::ResourceExhausted {
                        resource: format!("Rate limit exceeded for user: {}", user_id),
                    });
                }
            }
        }

        self.record_rate_limit_allowed("global").await;
        Ok(())
    }

    /// Check resource limits before executing an operation
    pub async fn check_resource_limits(&self, estimated_memory: Option<u64>) -> Result<()> {
        let limits = self.resource_limits.read().await;
        
        // Check memory usage
        if let Some(memory_needed) = estimated_memory {
            let current_memory = self.resource_usage.get_memory_usage();
            if current_memory + memory_needed > limits.max_memory_bytes {
                return Err(GraphError::ResourceExhausted {
                    resource: format!("Memory limit would be exceeded: {} + {} > {}",
                                    current_memory, memory_needed, limits.max_memory_bytes),
                });
            }
        }

        // Check CPU usage
        let current_cpu = self.resource_usage.get_cpu_usage();
        if current_cpu > limits.max_cpu_percent {
            return Err(GraphError::ResourceExhausted {
                resource: format!("CPU limit exceeded: {}% > {}%", 
                                current_cpu, limits.max_cpu_percent),
            });
        }

        // Check concurrent operations
        let active_ops = self.resource_usage.get_active_operations();
        if active_ops >= limits.max_concurrent_operations {
            return Err(GraphError::ResourceExhausted {
                resource: format!("Maximum concurrent operations exceeded: {} >= {}",
                                active_ops, limits.max_concurrent_operations),
            });
        }

        Ok(())
    }

    /// Acquire operation permit (blocks if necessary)
    pub async fn acquire_operation_permit(&self) -> Result<OperationPermit> {
        let permit = self.operation_semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| GraphError::ResourceExhausted {
                resource: "Failed to acquire operation permit".to_string(),
            })?;

        let current_ops = self.resource_usage.increment_operations();
        
        Ok(OperationPermit {
            _permit: permit,
            resource_usage: self.resource_usage.clone(),
            operation_count: current_ops,
        })
    }

    /// Try to acquire operation permit (non-blocking)
    pub fn try_acquire_operation_permit(&self) -> Result<OperationPermit> {
        let permit = self.operation_semaphore
            .clone()
            .try_acquire_owned()
            .map_err(|_| GraphError::ResourceExhausted {
                resource: "No operation permits available".to_string(),
            })?;

        let current_ops = self.resource_usage.increment_operations();
        
        Ok(OperationPermit {
            _permit: permit,
            resource_usage: self.resource_usage.clone(),
            operation_count: current_ops,
        })
    }

    /// Acquire database connection from pool
    pub async fn acquire_db_connection(&self) -> Result<DatabaseConnection> {
        let permit = self.connection_pool
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| GraphError::DatabaseConnectionError(
                "Failed to acquire database connection".to_string()
            ))?;

        self.resource_usage.database_connections.fetch_add(1, Ordering::Relaxed);
        
        Ok(DatabaseConnection {
            _permit: permit,
            resource_usage: self.resource_usage.clone(),
        })
    }

    /// Configure rate limiting for a specific operation
    pub fn configure_operation_rate_limit(&self, operation: &str, config: RateLimitConfig) {
        if !config.enabled {
            self.operation_rate_limiters.remove(operation);
            return;
        }

        let limiter: Arc<dyn RateLimiterTrait + Send + Sync> = match config.algorithm {
            RateLimitAlgorithm::TokenBucket => {
                Arc::new(TokenBucket::new(config.burst_capacity, config.requests_per_second))
            }
            _ => {
                // For other algorithms, fall back to token bucket for now
                Arc::new(TokenBucket::new(config.burst_capacity, config.requests_per_second))
            }
        };

        self.operation_rate_limiters.insert(operation.to_string(), limiter);
        self.rate_limit_configs.insert(operation.to_string(), config);
    }

    /// Configure rate limiting for a specific user
    pub fn configure_user_rate_limit(&self, user_id: &str, config: RateLimitConfig) {
        if !config.enabled {
            self.user_rate_limiters.remove(user_id);
            return;
        }

        let limiter: Arc<dyn RateLimiterTrait + Send + Sync> = Arc::new(
            TokenBucket::new(config.burst_capacity, config.requests_per_second)
        );

        self.user_rate_limiters.insert(user_id.to_string(), limiter);
    }

    /// Update resource limits
    pub async fn update_resource_limits(&self, limits: ResourceLimits) {
        *self.resource_limits.write().await = limits;
    }

    /// Get current resource usage
    pub async fn get_resource_usage(&self) -> HashMap<String, serde_json::Value> {
        let mut usage = HashMap::new();
        let limits = self.resource_limits.read().await;
        
        usage.insert("memory_bytes".to_string(), 
                    serde_json::json!(self.resource_usage.get_memory_usage()));
        usage.insert("memory_limit_bytes".to_string(), 
                    serde_json::json!(limits.max_memory_bytes));
        usage.insert("cpu_percent".to_string(), 
                    serde_json::json!(self.resource_usage.get_cpu_usage()));
        usage.insert("cpu_limit_percent".to_string(), 
                    serde_json::json!(limits.max_cpu_percent));
        usage.insert("active_operations".to_string(), 
                    serde_json::json!(self.resource_usage.get_active_operations()));
        usage.insert("max_operations".to_string(), 
                    serde_json::json!(limits.max_concurrent_operations));
        usage.insert("database_connections".to_string(), 
                    serde_json::json!(self.resource_usage.database_connections.load(Ordering::Relaxed)));
        usage.insert("max_connections".to_string(), 
                    serde_json::json!(limits.max_database_connections));
        
        usage
    }

    /// Get rate limiting statistics  
    pub async fn get_rate_limit_stats(&self) -> HashMap<String, HashMap<String, u64>> {
        let mut all_stats = HashMap::new();
        
        for item in self.rate_limit_stats.iter() {
            let stats = item.value();
            let mut stat_map = HashMap::new();
            
            stat_map.insert("total_requests".to_string(), 
                          stats.total_requests.load(Ordering::Relaxed));
            stat_map.insert("allowed_requests".to_string(), 
                          stats.allowed_requests.load(Ordering::Relaxed));
            stat_map.insert("denied_requests".to_string(), 
                          stats.denied_requests.load(Ordering::Relaxed));
            stat_map.insert("last_reset".to_string(), 
                          stats.last_reset.load(Ordering::Relaxed));
            
            all_stats.insert(item.key().clone(), stat_map);
        }
        
        all_stats
    }

    async fn record_rate_limit_allowed(&self, identifier: &str) {
        let stats = self.rate_limit_stats
            .entry(identifier.to_string())
            .or_insert_with(RateLimitStats::default);
        
        stats.total_requests.fetch_add(1, Ordering::Relaxed);
        stats.allowed_requests.fetch_add(1, Ordering::Relaxed);
    }

    async fn record_rate_limit_denial(&self, identifier: &str) {
        let stats = self.rate_limit_stats
            .entry(identifier.to_string())
            .or_insert_with(RateLimitStats::default);
        
        stats.total_requests.fetch_add(1, Ordering::Relaxed);
        stats.denied_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Health check for rate limiting system
    pub async fn health_check(&self) -> HashMap<String, serde_json::Value> {
        let mut health = HashMap::new();
        
        let usage = self.get_resource_usage().await;
        let stats = self.get_rate_limit_stats().await;
        
        // Check if system is healthy
        let limits = self.resource_limits.read().await;
        let memory_usage = self.resource_usage.get_memory_usage();
        let cpu_usage = self.resource_usage.get_cpu_usage();
        let active_ops = self.resource_usage.get_active_operations();
        
        let memory_health = if memory_usage > limits.max_memory_bytes * 9 / 10 {
            "critical"
        } else if memory_usage > limits.max_memory_bytes * 8 / 10 {
            "warning"  
        } else {
            "healthy"
        };
        
        let cpu_health = if cpu_usage > limits.max_cpu_percent * 0.9 {
            "critical"
        } else if cpu_usage > limits.max_cpu_percent * 0.8 {
            "warning"
        } else {
            "healthy"
        };
        
        let ops_health = if active_ops > limits.max_concurrent_operations * 9 / 10 {
            "critical"
        } else if active_ops > limits.max_concurrent_operations * 8 / 10 {
            "warning"
        } else {
            "healthy"
        };
        
        health.insert("memory_health".to_string(), serde_json::json!(memory_health));
        health.insert("cpu_health".to_string(), serde_json::json!(cpu_health));
        health.insert("operations_health".to_string(), serde_json::json!(ops_health));
        health.insert("resource_usage".to_string(), serde_json::json!(usage));
        health.insert("rate_limit_stats".to_string(), serde_json::json!(stats));
        
        // Global rate limiter status
        health.insert("global_tokens_available".to_string(), 
                     serde_json::json!(self.global_rate_limiter.available_tokens()));
        
        health
    }
}

impl Clone for RateLimitingManager {
    fn clone(&self) -> Self {
        // Create new instances with the same configuration but reset state
        let global_config = RateLimitConfig {
            requests_per_second: self.global_rate_limiter.capacity,
            burst_capacity: self.global_rate_limiter.capacity,
            window_size_seconds: 60,
            algorithm: RateLimitAlgorithm::TokenBucket,
            enabled: true,
        };
        
        let resource_limits = ResourceLimits::default();
        Self::new(global_config, resource_limits)
    }
}

/// RAII guard for operation permits
pub struct OperationPermit {
    _permit: tokio::sync::OwnedSemaphorePermit,
    resource_usage: Arc<ResourceUsage>,
    operation_count: u32,
}

impl Drop for OperationPermit {
    fn drop(&mut self) {
        self.resource_usage.decrement_operations();
    }
}

/// RAII guard for database connections
pub struct DatabaseConnection {
    _permit: tokio::sync::OwnedSemaphorePermit,
    resource_usage: Arc<ResourceUsage>,
}

impl Drop for DatabaseConnection {
    fn drop(&mut self) {
        self.resource_usage.database_connections.fetch_sub(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_token_bucket() {
        let bucket = TokenBucket::new(10, 5); // 10 capacity, 5 tokens/sec
        
        // Should be able to consume initial tokens
        assert!(bucket.try_consume(5));
        assert!(bucket.try_consume(5));
        
        // Should fail when empty
        assert!(!bucket.try_consume(1));
        
        // Wait for refill
        sleep(Duration::from_millis(1100)).await;
        
        // Should have tokens again
        assert!(bucket.try_consume(5));
    }

    #[tokio::test]
    async fn test_rate_limiting_manager() {
        let config = RateLimitConfig {
            requests_per_second: 10,
            burst_capacity: 20,
            ..Default::default()
        };
        
        let limits = ResourceLimits::default();
        let manager = RateLimitingManager::new(config, limits);
        
        // Should allow initial requests
        assert!(manager.check_rate_limit("test_op", None).await.is_ok());
        
        // Configure operation rate limit
        manager.configure_operation_rate_limit("test_op", RateLimitConfig {
            requests_per_second: 1,
            burst_capacity: 1,
            ..Default::default()
        });
        
        // Should allow first request
        assert!(manager.check_rate_limit("test_op", None).await.is_ok());
        
        // Should deny second immediate request
        assert!(manager.check_rate_limit("test_op", None).await.is_err());
    }

    #[tokio::test]
    async fn test_resource_limits() {
        let config = RateLimitConfig::default();
        let limits = ResourceLimits {
            max_concurrent_operations: 2,
            ..Default::default()
        };
        
        let manager = RateLimitingManager::new(config, limits);
        
        // Should be able to acquire permits
        let _permit1 = manager.acquire_operation_permit().await.unwrap();
        let _permit2 = manager.acquire_operation_permit().await.unwrap();
        
        // Third permit should fail immediately
        assert!(manager.try_acquire_operation_permit().is_err());
    }

    #[tokio::test]
    async fn test_resource_usage_tracking() {
        let usage = ResourceUsage::default();
        
        usage.set_memory_usage(1000);
        assert_eq!(usage.get_memory_usage(), 1000);
        
        usage.set_cpu_usage(50.5);
        assert_eq!(usage.get_cpu_usage(), 50.5);
        
        let ops1 = usage.increment_operations();
        let ops2 = usage.increment_operations();
        assert_eq!(ops1, 1);
        assert_eq!(ops2, 2);
        
        let ops3 = usage.decrement_operations();
        assert_eq!(ops3, 1);
    }
}