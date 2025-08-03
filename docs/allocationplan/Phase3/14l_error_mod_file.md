# Task 14l: Create Error Handling Module Declaration

**Time**: 3 minutes
**Dependencies**: 14k_error_dashboard.md
**Stage**: Inheritance System

## Objective
Create module declarations for the complete error handling system.

## Implementation
Update `src/inheritance/mod.rs` to include error handling modules:

```rust
pub mod hierarchy_types;
pub mod hierarchy_manager;
pub mod property_types;
pub mod property_inheritance_engine;
pub mod property_exceptions;
pub mod cache;

// Error handling modules
pub mod error_types;
pub mod retry_logic;
pub mod error_logging;
pub mod error_recovery;
pub mod error_metrics;
pub mod circuit_breaker;
pub mod error_aggregation;
pub mod error_notification;
pub mod error_dashboard;

// Re-export main types
pub use hierarchy_types::*;
pub use hierarchy_manager::InheritanceHierarchyManager;
pub use property_types::*;
pub use property_inheritance_engine::PropertyInheritanceEngine;
pub use property_exceptions::PropertyExceptionHandler;
pub use cache::InheritanceCacheManager;

// Re-export error handling types
pub use error_types::{InheritanceError, PropertyError, ErrorContext, InheritanceErrorWithContext};
pub use error_recovery::{ErrorRecoveryManager, RecoveryResult, RecoveryError};
pub use error_metrics::{ErrorMetrics, ErrorMetricsCollector};
pub use circuit_breaker::{CircuitBreaker, CircuitState, CircuitBreakerConfig};
pub use error_aggregation::{ErrorAggregator, ErrorPattern, SeverityLevel};
pub use error_notification::{ErrorNotificationSystem, ErrorNotification, NotificationType};
pub use error_dashboard::{ErrorDashboard, DashboardData, SystemHealthStatus};
```

Create comprehensive error handling factory:

```rust
// Add to the same file
use std::sync::Arc;

/// Factory for creating a complete error handling system
pub struct InheritanceErrorHandlingSystem {
    pub error_handler: Arc<error_types::DefaultErrorHandler>,
    pub retry_executor: Arc<retry_logic::RetryExecutor>,
    pub logger: Arc<error_logging::InheritanceErrorLogger>,
    pub recovery_manager: Arc<error_recovery::ErrorRecoveryManager>,
    pub metrics_collector: Arc<error_metrics::ErrorMetricsCollector>,
    pub circuit_breaker: Arc<circuit_breaker::CircuitBreaker>,
    pub error_aggregator: Arc<error_aggregation::ErrorAggregator>,
    pub notification_system: Arc<error_notification::ErrorNotificationSystem>,
    pub dashboard: Arc<error_dashboard::ErrorDashboard>,
}

impl InheritanceErrorHandlingSystem {
    pub fn new(component_name: &str) -> (Self, tokio::sync::mpsc::UnboundedReceiver<error_notification::ErrorNotification>) {
        let error_handler = Arc::new(error_types::DefaultErrorHandler::new());
        let retry_executor = Arc::new(retry_logic::RetryExecutor::new(error_handler.clone()));
        let logger = Arc::new(error_logging::InheritanceErrorLogger::new(component_name));
        let recovery_manager = Arc::new(error_recovery::ErrorRecoveryManager::new(logger.clone()));
        let metrics_collector = Arc::new(error_metrics::ErrorMetricsCollector::new());
        
        let circuit_config = circuit_breaker::CircuitBreakerConfig::default();
        let circuit_breaker = Arc::new(circuit_breaker::CircuitBreaker::new(circuit_config));
        
        let aggregation_config = error_aggregation::AggregationConfig::default();
        let error_aggregator = Arc::new(error_aggregation::ErrorAggregator::new(aggregation_config));
        
        let notification_config = error_notification::NotificationConfig::default();
        let (notification_system, notification_receiver) = error_notification::ErrorNotificationSystem::new(notification_config);
        let notification_system = Arc::new(notification_system);
        
        let mut dashboard = error_dashboard::ErrorDashboard::new(
            metrics_collector.clone(),
            error_aggregator.clone(),
        );
        dashboard.add_circuit_breaker("inheritance_operations".to_string(), circuit_breaker.clone());
        let dashboard = Arc::new(dashboard);

        let system = Self {
            error_handler,
            retry_executor,
            logger,
            recovery_manager,
            metrics_collector,
            circuit_breaker,
            error_aggregator,
            notification_system,
            dashboard,
        };

        (system, notification_receiver)
    }

    pub async fn handle_error(&self, error: InheritanceErrorWithContext) -> Result<(), InheritanceError> {
        // Log the error
        self.logger.log_inheritance_error(&error);
        
        // Record metrics
        self.metrics_collector.record_error(&format!("{:?}", error.error)).await;
        
        // Aggregate for pattern detection
        self.error_aggregator.record_error(&error).await;
        
        // Send notification if needed
        self.notification_system.notify_error(&error).await;
        
        // Attempt recovery
        match self.recovery_manager.attempt_recovery(&error).await {
            Ok(_) => Ok(()),
            Err(_) => Err(error.error),
        }
    }
}
```

## Success Criteria
- All error modules are properly declared
- Factory creates complete error handling system
- Module re-exports are accessible

## Next Task
15a_validation_rules.md