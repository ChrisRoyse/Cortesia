//! Logging Configuration
//! 
//! Production-ready logging initialization and configuration for the enhanced
//! knowledge storage system using the tracing ecosystem.

use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
    fmt,
};

/// Initialize structured logging for the enhanced knowledge storage system
pub fn init_logging() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Create a subscriber with structured JSON logging for structured environments
    // and pretty formatting for development
    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_level(true)
        .with_file(true)
        .with_line_number(true)
        .with_thread_names(true)
        .compact();

    // Set up environment filter with sensible defaults
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| {
            // Default log levels for different components
            EnvFilter::new(
                "info,\
                 llmkg::enhanced_knowledge_storage=debug,\
                 llmkg::enhanced_knowledge_storage::model_management=info,\
                 llmkg::enhanced_knowledge_storage::knowledge_processing=info,\
                 llmkg::enhanced_knowledge_storage::retrieval_system=info,\
                 llmkg::enhanced_knowledge_storage::hierarchical_storage=info"
            )
        });

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .try_init()?;

    tracing::info!("Enhanced Knowledge Storage logging initialized");
    Ok(())
}

/// Initialize JSON logging for production environments
pub fn init_json_logging() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let json_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_level(true)
        .with_file(true)
        .with_line_number(true);

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(json_layer)
        .try_init()?;

    tracing::info!("Enhanced Knowledge Storage JSON logging initialized");
    Ok(())
}

/// Logging macros with context
#[macro_export]
macro_rules! log_operation_start {
    ($operation:expr, $context:expr) => {
        tracing::info!(
            operation = $operation,
            context = ?$context,
            "Starting operation"
        );
    };
}

#[macro_export]
macro_rules! log_operation_success {
    ($operation:expr, $duration:expr, $metrics:expr) => {
        tracing::info!(
            operation = $operation,
            duration_ms = $duration.as_millis(),
            metrics = ?$metrics,
            "Operation completed successfully"
        );
    };
}

#[macro_export]
macro_rules! log_operation_error {
    ($operation:expr, $error:expr, $context:expr) => {
        tracing::error!(
            operation = $operation,
            error = %$error,
            context = ?$context,
            "Operation failed"
        );
    };
}

#[macro_export]
macro_rules! log_performance_metric {
    ($metric_name:expr, $value:expr, $unit:expr) => {
        tracing::debug!(
            metric = $metric_name,
            value = $value,
            unit = $unit,
            "Performance metric"
        );
    };
}

/// Common logging fields for structured logging
pub struct LogContext {
    pub request_id: Option<String>,
    pub user_id: Option<String>,
    pub operation: String,
    pub component: String,
}

impl LogContext {
    pub fn new(operation: impl Into<String>, component: impl Into<String>) -> Self {
        Self {
            request_id: None,
            user_id: None,
            operation: operation.into(),
            component: component.into(),
        }
    }

    pub fn with_request_id(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }
}

impl std::fmt::Debug for LogContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogContext")
            .field("request_id", &self.request_id)
            .field("user_id", &self.user_id)
            .field("operation", &self.operation)
            .field("component", &self.component)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_context_creation() {
        let context = LogContext::new("test_operation", "test_component");
        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.component, "test_component");
        assert!(context.request_id.is_none());
        assert!(context.user_id.is_none());
    }

    #[test]
    fn test_log_context_with_ids() {
        let context = LogContext::new("test", "component")
            .with_request_id("req_123")
            .with_user_id("user_456");
        
        assert_eq!(context.request_id, Some("req_123".to_string()));
        assert_eq!(context.user_id, Some("user_456".to_string()));
    }
}