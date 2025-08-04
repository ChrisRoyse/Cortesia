# Task 14j: Implement Error Notification System

**Time**: 5 minutes (1 min read, 3 min implement, 1 min verify)
**Dependencies**: 14i_error_aggregation.md
**Stage**: Inheritance System

## Objective
Create notification system for critical errors and patterns.

## Implementation
Create `src/inheritance/error_notification.rs`:

```rust
use std::sync::Arc;
use tokio::sync::mpsc;
use serde::{Deserialize, Serialize};
use crate::inheritance::error_types::*;
use crate::inheritance::error_aggregation::{ErrorPattern, SeverityLevel};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorNotification {
    pub notification_type: NotificationType,
    pub severity: SeverityLevel,
    pub title: String,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationType {
    SingleError,
    ErrorPattern,
    SystemHealth,
    CircuitBreakerTripped,
}

pub struct ErrorNotificationSystem {
    notification_sender: mpsc::UnboundedSender<ErrorNotification>,
    config: NotificationConfig,
}

#[derive(Debug, Clone)]
pub struct NotificationConfig {
    pub enable_notifications: bool,
    pub notify_on_critical: bool,
    pub notify_on_patterns: bool,
    pub notify_on_circuit_breaker: bool,
    pub rate_limit_minutes: i64,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enable_notifications: true,
            notify_on_critical: true,
            notify_on_patterns: true,
            notify_on_circuit_breaker: true,
            rate_limit_minutes: 5,
        }
    }
}

impl ErrorNotificationSystem {
    pub fn new(config: NotificationConfig) -> (Self, mpsc::UnboundedReceiver<ErrorNotification>) {
        let (sender, receiver) = mpsc::unbounded_channel();
        
        let system = Self {
            notification_sender: sender,
            config,
        };
        
        (system, receiver)
    }

    pub async fn notify_error(&self, error: &InheritanceErrorWithContext) {
        if !self.config.enable_notifications {
            return;
        }

        let severity = self.determine_error_severity(&error.error);
        
        if matches!(severity, SeverityLevel::Critical | SeverityLevel::High) && self.config.notify_on_critical {
            let notification = ErrorNotification {
                notification_type: NotificationType::SingleError,
                severity,
                title: "Critical Inheritance Error".to_string(),
                message: format!("Error in operation '{}': {}", error.context.operation, error.error),
                timestamp: chrono::Utc::now(),
                metadata: self.build_error_metadata(error),
            };

            let _ = self.notification_sender.send(notification);
        }
    }

    pub async fn notify_pattern(&self, pattern: &ErrorPattern) {
        if !self.config.enable_notifications || !self.config.notify_on_patterns {
            return;
        }

        let notification = ErrorNotification {
            notification_type: NotificationType::ErrorPattern,
            severity: pattern.severity_level.clone(),
            title: "Error Pattern Detected".to_string(),
            message: format!(
                "Pattern detected: {} (frequency: {}, window: {} minutes)",
                pattern.error_sequence.join(" -> "),
                pattern.frequency,
                pattern.time_window_minutes
            ),
            timestamp: chrono::Utc::now(),
            metadata: self.build_pattern_metadata(pattern),
        };

        let _ = self.notification_sender.send(notification);
    }

    pub async fn notify_circuit_breaker_tripped(&self, component: &str) {
        if !self.config.enable_notifications || !self.config.notify_on_circuit_breaker {
            return;
        }

        let notification = ErrorNotification {
            notification_type: NotificationType::CircuitBreakerTripped,
            severity: SeverityLevel::High,
            title: "Circuit Breaker Tripped".to_string(),
            message: format!("Circuit breaker for '{}' has been tripped due to repeated failures", component),
            timestamp: chrono::Utc::now(),
            metadata: {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("component".to_string(), component.to_string());
                metadata
            },
        };

        let _ = self.notification_sender.send(notification);
    }

    pub async fn notify_system_health(&self, health_status: &str, details: &str) {
        if !self.config.enable_notifications {
            return;
        }

        let severity = match health_status {
            "degraded" => SeverityLevel::Medium,
            "unhealthy" => SeverityLevel::High,
            "critical" => SeverityLevel::Critical,
            _ => SeverityLevel::Low,
        };

        let notification = ErrorNotification {
            notification_type: NotificationType::SystemHealth,
            severity,
            title: format!("System Health: {}", health_status),
            message: details.to_string(),
            timestamp: chrono::Utc::now(),
            metadata: {
                let mut metadata = std::collections::HashMap::new();
                metadata.insert("health_status".to_string(), health_status.to_string());
                metadata
            },
        };

        let _ = self.notification_sender.send(notification);
    }

    fn determine_error_severity(&self, error: &InheritanceError) -> SeverityLevel {
        match error {
            InheritanceError::CycleDetected { .. } => SeverityLevel::Critical,
            InheritanceError::DatabaseError(_) => SeverityLevel::High,
            InheritanceError::MaxDepthExceeded(_) => SeverityLevel::Medium,
            InheritanceError::CacheError(_) => SeverityLevel::Low,
            InheritanceError::ConceptNotFound(_) => SeverityLevel::Medium,
            InheritanceError::PropertyNotFound { .. } => SeverityLevel::Low,
            InheritanceError::ValidationError(_) => SeverityLevel::Medium,
            InheritanceError::ConfigurationError(_) => SeverityLevel::High,
        }
    }

    fn build_error_metadata(&self, error: &InheritanceErrorWithContext) -> std::collections::HashMap<String, String> {
        let mut metadata = std::collections::HashMap::new();
        
        metadata.insert("trace_id".to_string(), error.context.trace_id.clone());
        metadata.insert("operation".to_string(), error.context.operation.clone());
        
        if let Some(concept_id) = &error.context.concept_id {
            metadata.insert("concept_id".to_string(), concept_id.clone());
        }
        
        if let Some(property_name) = &error.context.property_name {
            metadata.insert("property_name".to_string(), property_name.clone());
        }
        
        if let Some(suggestion) = &error.recovery_suggestion {
            metadata.insert("recovery_suggestion".to_string(), suggestion.clone());
        }
        
        metadata
    }

    fn build_pattern_metadata(&self, pattern: &ErrorPattern) -> std::collections::HashMap<String, String> {
        let mut metadata = std::collections::HashMap::new();
        
        metadata.insert("frequency".to_string(), pattern.frequency.to_string());
        metadata.insert("time_window".to_string(), pattern.time_window_minutes.to_string());
        metadata.insert("affected_concepts".to_string(), pattern.affected_concepts.join(","));
        
        metadata
    }
}

// Helper for consuming notifications
pub struct NotificationConsumer {
    receiver: mpsc::UnboundedReceiver<ErrorNotification>,
}

impl NotificationConsumer {
    pub fn new(receiver: mpsc::UnboundedReceiver<ErrorNotification>) -> Self {
        Self { receiver }
    }

    pub async fn start_processing(mut self) {
        while let Some(notification) = self.receiver.recv().await {
            self.process_notification(notification).await;
        }
    }

    async fn process_notification(&self, notification: ErrorNotification) {
        // In a real implementation, this would send to external systems
        // For now, just log the notification
        match notification.severity {
            SeverityLevel::Critical => {
                tracing::error!(
                    title = %notification.title,
                    message = %notification.message,
                    notification_type = ?notification.notification_type,
                    "Critical error notification"
                );
            }
            SeverityLevel::High => {
                tracing::warn!(
                    title = %notification.title,
                    message = %notification.message,
                    notification_type = ?notification.notification_type,
                    "High severity notification"
                );
            }
            _ => {
                tracing::info!(
                    title = %notification.title,
                    message = %notification.message,
                    notification_type = ?notification.notification_type,
                    "Notification"
                );
            }
        }
    }
}
```

## Success Criteria
- Notifications are sent for critical errors
- Pattern notifications work correctly
- Rate limiting prevents spam

## Next Task
14k_error_dashboard.md