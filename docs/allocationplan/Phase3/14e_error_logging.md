# Task 14e: Implement Error Logging

**Time**: 5 minutes
**Dependencies**: 14d_retry_logic.md
**Stage**: Inheritance System

## Objective
Add comprehensive error logging and monitoring.

## Implementation
Create `src/inheritance/error_logging.rs`:

```rust
use tracing::{error, warn, info, debug};
use serde_json::json;
use crate::inheritance::error_types::*;

pub struct InheritanceErrorLogger {
    component_name: String,
    enable_detailed_logging: bool,
}

impl InheritanceErrorLogger {
    pub fn new(component_name: &str) -> Self {
        Self {
            component_name: component_name.to_string(),
            enable_detailed_logging: true,
        }
    }

    pub fn log_inheritance_error(&self, error: &InheritanceErrorWithContext) {
        let log_entry = json!({
            "component": self.component_name,
            "error_type": format!("{:?}", error.error),
            "operation": error.context.operation,
            "concept_id": error.context.concept_id,
            "property_name": error.context.property_name,
            "timestamp": error.context.timestamp,
            "trace_id": error.context.trace_id,
            "metadata": error.context.metadata,
            "recovery_suggestion": error.recovery_suggestion
        });

        match &error.error {
            InheritanceError::CycleDetected { .. } => {
                error!(
                    target: "inheritance_system",
                    error = %error.error,
                    trace_id = %error.context.trace_id,
                    "Critical inheritance cycle detected"
                );
            }
            InheritanceError::MaxDepthExceeded(_) => {
                warn!(
                    target: "inheritance_system",
                    error = %error.error,
                    trace_id = %error.context.trace_id,
                    "Inheritance depth limit exceeded"
                );
            }
            InheritanceError::DatabaseError(_) => {
                error!(
                    target: "inheritance_system",
                    error = %error.error,
                    trace_id = %error.context.trace_id,
                    "Database operation failed"
                );
            }
            InheritanceError::CacheError(_) => {
                warn!(
                    target: "inheritance_system",
                    error = %error.error,
                    trace_id = %error.context.trace_id,
                    "Cache operation failed"
                );
            }
            _ => {
                info!(
                    target: "inheritance_system",
                    error = %error.error,
                    trace_id = %error.context.trace_id,
                    "Inheritance operation error"
                );
            }
        }

        if self.enable_detailed_logging {
            debug!(
                target: "inheritance_system",
                log_entry = %log_entry,
                "Detailed error information"
            );
        }
    }

    pub fn log_property_error(&self, error: &PropertyError, context: &ErrorContext) {
        let log_entry = json!({
            "component": self.component_name,
            "error_type": format!("{:?}", error),
            "operation": context.operation,
            "concept_id": context.concept_id,
            "property_name": context.property_name,
            "timestamp": context.timestamp,
            "trace_id": context.trace_id,
            "metadata": context.metadata
        });

        match error {
            PropertyError::InheritanceConflict { .. } => {
                warn!(
                    target: "property_system",
                    error = %error,
                    trace_id = %context.trace_id,
                    "Property inheritance conflict detected"
                );
            }
            PropertyError::TypeMismatch { .. } => {
                error!(
                    target: "property_system",
                    error = %error,
                    trace_id = %context.trace_id,
                    "Property type mismatch"
                );
            }
            _ => {
                info!(
                    target: "property_system",
                    error = %error,
                    trace_id = %context.trace_id,
                    "Property operation error"
                );
            }
        }

        if self.enable_detailed_logging {
            debug!(
                target: "property_system",
                log_entry = %log_entry,
                "Detailed property error information"
            );
        }
    }

    pub fn log_recovery_attempt(&self, trace_id: &str, recovery_action: &str) {
        info!(
            target: "inheritance_system",
            trace_id = %trace_id,
            recovery_action = %recovery_action,
            "Attempting error recovery"
        );
    }

    pub fn log_retry_attempt(&self, trace_id: &str, attempt: u32, delay_ms: u64) {
        info!(
            target: "inheritance_system",
            trace_id = %trace_id,
            attempt = attempt,
            delay_ms = delay_ms,
            "Retrying operation"
        );
    }
}
```

## Success Criteria
- Error logging captures all relevant context
- Log levels are appropriate for error severity
- Structured logging enables analysis

## Next Task
14f_error_recovery_strategies.md