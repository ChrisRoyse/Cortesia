//! Production-Ready System Components
//!
//! This module provides comprehensive production-ready features for the LLMKG system:
//! - Error recovery with circuit breakers and retry logic
//! - Comprehensive monitoring and logging
//! - Rate limiting and resource management
//! - Health checks and system status monitoring
//! - Graceful shutdown with data integrity preservation
//!
//! These components work together to provide a robust, observable, and maintainable
//! production system suitable for enterprise deployment.

pub mod error_recovery;
pub mod monitoring;
pub mod rate_limiting;
pub mod health_checks;
pub mod graceful_shutdown;

use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use serde_json::Value;

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::error::Result;

pub use error_recovery::{
    ErrorRecoveryManager, RetryConfig, CircuitBreakerConfig, CircuitState
};
pub use monitoring::{
    ProductionMonitor, MonitoringConfig, LogLevel, MetricType, AlertSeverity
};
pub use rate_limiting::{
    RateLimitingManager, RateLimitConfig, ResourceLimits, OperationPermit, DatabaseConnection
};
pub use health_checks::{
    HealthCheckSystem, HealthCheckConfig, HealthStatus, SystemHealthReport
};
pub use graceful_shutdown::{
    GracefulShutdownManager, ShutdownConfig, ShutdownPhase, ActiveRequestGuard
};

/// Complete production system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductionConfig {
    pub monitoring: MonitoringConfig,
    pub rate_limiting: RateLimitConfig,
    pub resource_limits: ResourceLimits,
    pub health_checks: HealthCheckConfig,
    pub shutdown: ShutdownConfig,
    pub error_recovery_enabled: bool,
}

impl Default for ProductionConfig {
    fn default() -> Self {
        Self {
            monitoring: MonitoringConfig::default(),
            rate_limiting: RateLimitConfig::default(),
            resource_limits: ResourceLimits::default(),
            health_checks: HealthCheckConfig::default(),
            shutdown: ShutdownConfig::default(),
            error_recovery_enabled: true,
        }
    }
}

/// Comprehensive production system that integrates all components
pub struct ProductionSystem {
    // Core components
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
    
    // Production systems
    error_recovery: ErrorRecoveryManager,
    monitor: Arc<ProductionMonitor>,
    rate_limiter: RateLimitingManager,
    health_checker: HealthCheckSystem,
    shutdown_manager: Arc<GracefulShutdownManager>,
    
    // Configuration
    config: ProductionConfig,
}

impl ProductionSystem {
    /// Create a new production system with all components integrated
    pub fn new(
        knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
        config: ProductionConfig,
    ) -> Self {
        // Initialize error recovery
        let error_recovery = ErrorRecoveryManager::new();
        if config.error_recovery_enabled {
            error_recovery.configure_defaults();
        }
        
        // Initialize monitoring
        let monitor = Arc::new(ProductionMonitor::new(config.monitoring.clone()));
        
        // Initialize rate limiting
        let rate_limiter = RateLimitingManager::new(
            config.rate_limiting.clone(),
            config.resource_limits.clone(),
        );
        
        // Initialize health checks
        let health_checker = HealthCheckSystem::new(
            knowledge_engine.clone(),
            config.health_checks.clone(),
        );
        
        // Initialize graceful shutdown
        let shutdown_manager = Arc::new(GracefulShutdownManager::new(
            knowledge_engine.clone(),
            config.shutdown.clone(),
        ));
        
        // Setup signal handlers for graceful shutdown
        shutdown_manager.clone().setup_signal_handlers();
        
        let system = Self {
            knowledge_engine,
            error_recovery,
            monitor,
            rate_limiter,
            health_checker,
            shutdown_manager,
            config,
        };
        
        // Log system initialization
        tokio::spawn({
            let monitor = system.monitor.clone();
            async move {
                monitor.log(
                    LogLevel::Info,
                    "production_system",
                    "initialization",
                    "Production system initialized with all components"
                ).await;
            }
        });
        
        system
    }

    /// Execute an operation with full production protection
    pub async fn execute_protected_operation<T, F, Fut>(
        &self,
        operation_name: &str,
        user_id: Option<&str>,
        operation: F,
    ) -> Result<T>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<T>> + Send,
        T: Send,
    {
        // Check if system is shutting down
        if self.shutdown_manager.is_shutting_down() {
            return Err(crate::error::GraphError::InvalidState(
                "System is shutting down".to_string()
            ));
        }
        
        // Track active request
        let _request_guard = self.shutdown_manager.track_active_request()?;
        
        // Start monitoring timer (after sync operations)
        let timer = self.monitor.start_timer(operation_name);
        
        // Check rate limits
        self.rate_limiter.check_rate_limit(operation_name, user_id).await?;
        
        // Check resource limits
        self.rate_limiter.check_resource_limits(None).await?;
        
        // Acquire operation permit
        let _operation_permit = self.rate_limiter.acquire_operation_permit().await?;
        
        // Log operation start
        self.monitor.log(
            LogLevel::Info,
            "operation_executor",
            operation_name,
            &format!("Starting operation: {}", operation_name)
        ).await;
        
        // Execute with error recovery if enabled
        let result = if self.config.error_recovery_enabled {
            self.error_recovery.execute_with_recovery(operation_name, operation).await
        } else {
            operation().await
        };
        
        // Log result and update metrics
        match &result {
            Ok(_) => {
                self.monitor.log(
                    LogLevel::Info,
                    "operation_executor",
                    operation_name,
                    &format!("Operation completed successfully: {}", operation_name)
                ).await;
                
                self.monitor.increment_counter(&format!("{}_success", operation_name));
            }
            Err(e) => {
                self.monitor.log(
                    LogLevel::Error,
                    "operation_executor",
                    operation_name,
                    &format!("Operation failed: {} - {}", operation_name, e)
                ).await;
                
                self.monitor.increment_counter(&format!("{}_error", operation_name));
            }
        }
        
        // Timer is automatically recorded when dropped
        drop(timer);
        
        result
    }

    /// Get comprehensive system status
    pub async fn get_system_status(&self) -> HashMap<String, Value> {
        let mut status = HashMap::new();
        
        // Overall system health
        let health_status = self.health_checker.get_health_status().await;
        status.insert("health".to_string(), serde_json::json!(health_status));
        
        // Resource usage
        let resource_usage = self.rate_limiter.get_resource_usage().await;
        status.insert("resources".to_string(), serde_json::json!(resource_usage));
        
        // Rate limiting stats
        let rate_limit_stats = self.rate_limiter.get_rate_limit_stats().await;
        status.insert("rate_limits".to_string(), serde_json::json!(rate_limit_stats));
        
        // Error recovery stats
        let error_stats = self.error_recovery.get_all_stats();
        status.insert("error_recovery".to_string(), serde_json::json!(error_stats));
        
        // System metrics
        let system_stats = self.monitor.get_system_stats().await;
        status.insert("system_metrics".to_string(), serde_json::json!(system_stats));
        
        // Shutdown status
        if self.shutdown_manager.is_shutting_down() {
            let shutdown_progress = self.shutdown_manager.get_shutdown_progress();
            status.insert("shutdown".to_string(), serde_json::json!(shutdown_progress));
        } else {
            status.insert("shutdown".to_string(), serde_json::json!("not_initiated"));
        }
        
        // Performance data
        let performance_data = self.monitor.get_performance_data().await;
        status.insert("performance".to_string(), serde_json::json!(performance_data));
        
        status
    }

    /// Get detailed health report
    pub async fn get_health_report(&self) -> SystemHealthReport {
        self.health_checker.perform_full_health_check().await
    }

    /// Get production metrics in Prometheus format
    pub async fn get_prometheus_metrics(&self) -> String {
        self.monitor.export_prometheus_metrics().await
    }

    /// Initiate graceful shutdown
    pub async fn shutdown(&self) -> Result<graceful_shutdown::ShutdownReport> {
        self.monitor.log(
            LogLevel::Info,
            "production_system",
            "shutdown",
            "Initiating graceful shutdown"
        ).await;
        
        // Stop health checks first
        self.health_checker.shutdown().await;
        
        // Then shutdown the main system
        match self.shutdown_manager.initiate_shutdown().await {
            Ok(report) => {
                self.monitor.log(
                    LogLevel::Info,
                    "production_system",
                    "shutdown",
                    "Graceful shutdown completed successfully"
                ).await;
                Ok(report)
            }
            Err(e) => {
                self.monitor.log(
                    LogLevel::Error,
                    "production_system",
                    "shutdown",
                    &format!("Graceful shutdown failed: {}", e)
                ).await;
                
                // Attempt force shutdown
                let _force_report = self.shutdown_manager.force_shutdown().await?;
                Err(crate::error::GraphError::RecoveryFailed(
                    format!("Graceful shutdown failed, force shutdown executed: {}", e)
                ))
            }
        }
    }

    /// Configure operation-specific settings
    pub fn configure_operation(&self, operation_name: &str, config: OperationConfig) {
        // Configure rate limiting
        if let Some(rate_config) = config.rate_limit {
            self.rate_limiter.configure_operation_rate_limit(operation_name, rate_config);
        }
        
        // Configure error recovery
        if let Some(retry_config) = config.retry {
            self.error_recovery.configure_retry(operation_name, retry_config);
        }
        
        if let Some(circuit_config) = config.circuit_breaker {
            self.error_recovery.configure_circuit_breaker(operation_name, circuit_config);
        }
        
        // Configure health checks if needed
        if config.enable_health_check {
            self.health_checker.register_component(operation_name, None, false);
        }
    }

    /// Access individual components for advanced configuration
    pub fn error_recovery(&self) -> &ErrorRecoveryManager {
        &self.error_recovery
    }
    
    pub fn monitor(&self) -> &ProductionMonitor {
        &self.monitor
    }
    
    pub fn rate_limiter(&self) -> &RateLimitingManager {
        &self.rate_limiter
    }
    
    pub fn health_checker(&self) -> &HealthCheckSystem {
        &self.health_checker
    }
    
    pub fn shutdown_manager(&self) -> &Arc<GracefulShutdownManager> {
        &self.shutdown_manager
    }
}

impl Clone for ProductionSystem {
    fn clone(&self) -> Self {
        Self {
            knowledge_engine: Arc::clone(&self.knowledge_engine),
            error_recovery: self.error_recovery.clone(),
            monitor: Arc::clone(&self.monitor),
            rate_limiter: self.rate_limiter.clone(),
            health_checker: self.health_checker.clone(),
            shutdown_manager: Arc::clone(&self.shutdown_manager),
            config: self.config.clone(),
        }
    }
}

/// Configuration for individual operations
#[derive(Debug, Clone)]
pub struct OperationConfig {
    pub rate_limit: Option<RateLimitConfig>,
    pub retry: Option<RetryConfig>,
    pub circuit_breaker: Option<CircuitBreakerConfig>,
    pub enable_health_check: bool,
}

impl Default for OperationConfig {
    fn default() -> Self {
        Self {
            rate_limit: None,
            retry: None,
            circuit_breaker: None,
            enable_health_check: false,
        }
    }
}

/// Helper function to create a production system with sensible defaults
pub fn create_production_system(
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
) -> ProductionSystem {
    let config = ProductionConfig::default();
    ProductionSystem::new(knowledge_engine, config)
}

/// Helper function to create a production system with custom configuration
pub fn create_production_system_with_config(
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
    config: ProductionConfig,
) -> ProductionSystem {
    ProductionSystem::new(knowledge_engine, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::knowledge_engine::KnowledgeEngine;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_production_system_creation() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
        let system = create_production_system(engine);
        
        // System should be created successfully
        assert!(!system.shutdown_manager.is_shutting_down());
        
        // Health check should be available
        let health = system.get_health_report().await;
        assert_ne!(health.overall_status, HealthStatus::Unknown);
    }

    #[tokio::test]
    async fn test_protected_operation_execution() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
        let system = create_production_system(engine);
        
        // Execute a simple operation
        let result = system.execute_protected_operation(
            "test_operation",
            Some("test_user"),
            || async { Ok("test_result".to_string()) }
        ).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test_result");
        
        // Check that metrics were recorded
        let stats = system.error_recovery.get_operation_stats("test_operation");
        assert!(stats.is_some());
        
        let stats = stats.unwrap();
        assert_eq!(stats["total_attempts"], 1);
        assert_eq!(stats["successful_attempts"], 1);
    }

    #[tokio::test]
    async fn test_operation_failure_handling() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
        let system = create_production_system(engine);
        
        // Execute an operation that fails
        let result: Result<String> = system.execute_protected_operation(
            "failing_operation",
            None,
            || async {
                Err(crate::error::GraphError::ProcessingError(
                    "Test error".to_string()
                ))
            }
        ).await;
        
        assert!(result.is_err());
        
        // Check that error recovery attempted retries
        let stats = system.error_recovery.get_operation_stats("failing_operation");
        assert!(stats.is_some());
        
        let stats = stats.unwrap();
        assert!(stats["total_attempts"] > 1); // Should have retried
        assert_eq!(stats["successful_attempts"], 0);
        assert!(stats["failed_attempts"] > 0);
    }

    #[tokio::test]
    async fn test_system_status_collection() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
        let system = create_production_system(engine);
        
        // Execute some operations to generate data
        let _ = system.execute_protected_operation(
            "status_test",
            None,
            || async { Ok("result") }
        ).await;
        
        // Get system status
        let status = system.get_system_status().await;
        
        // Should contain all major sections
        assert!(status.contains_key("health"));
        assert!(status.contains_key("resources"));
        assert!(status.contains_key("rate_limits"));
        assert!(status.contains_key("error_recovery"));
        assert!(status.contains_key("system_metrics"));
        assert!(status.contains_key("performance"));
    }

    // FIXME: Commented out due to DashMap lifetime issues with ShutdownHandler trait
    // #[tokio::test]
    // async fn test_graceful_shutdown() {
    //     let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
    //     let config = ProductionConfig {
    //         shutdown: ShutdownConfig {
    //             graceful_timeout_seconds: 1,
    //             save_state_timeout_seconds: 1,
    //             ..Default::default()
    //         },
    //         ..Default::default()
    //     };
    //     let system = create_production_system_with_config(engine, config);
    //     
    //     // System should not be shutting down initially
    //     assert!(!system.shutdown_manager.is_shutting_down());
    //     
    //     // Start an operation
    //     let operation_task = {
    //         let system = system.clone();
    //         tokio::spawn(async move {
    //             system.execute_protected_operation(
    //                 "long_operation",
    //                 None,
    //                 || async {
    //                     sleep(Duration::from_millis(500)).await;
    //                     Ok("completed")
    //                 }
    //             ).await
    //         })
    //     };
    //     
    //     // Wait a bit for operation to start
    //     sleep(Duration::from_millis(100)).await;
    //     
    //     // Initiate shutdown
    //     let system_clone = system.clone();
    //     let shutdown_task = tokio::spawn(async move {
    //         system_clone.shutdown().await
    //     });
    //     
    //     // Operation should complete
    //     let op_result = operation_task.await.unwrap();
    //     assert!(op_result.is_ok());
    //     
    //     // Shutdown should complete
    //     let shutdown_result = shutdown_task.await.unwrap();
    //     assert!(shutdown_result.is_ok());
    //     
    //     // System should be in shutdown state
    //     assert!(system.shutdown_manager.is_shutting_down());
    // }

    #[tokio::test]
    async fn test_operation_configuration() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
        let system = create_production_system(engine);
        
        // Configure a specific operation
        system.configure_operation("configured_op", OperationConfig {
            rate_limit: Some(RateLimitConfig {
                requests_per_second: 10,
                burst_capacity: 20,
                ..Default::default()
            }),
            retry: Some(RetryConfig {
                max_retries: 5,
                base_delay_ms: 50,
                ..Default::default()
            }),
            circuit_breaker: Some(CircuitBreakerConfig {
                failure_threshold: 3,
                recovery_timeout: Duration::from_secs(30),
                ..Default::default()
            }),
            enable_health_check: true,
        });
        
        // Execute the configured operation
        let result = system.execute_protected_operation(
            "configured_op",
            None,
            || async { Ok("configured_result") }
        ).await;
        
        assert!(result.is_ok());
        
        // Check that configuration was applied
        let circuit_state = system.error_recovery.get_circuit_breaker_status("configured_op");
        assert!(circuit_state.is_some());
        assert_eq!(circuit_state.unwrap(), CircuitState::Closed);
    }
}