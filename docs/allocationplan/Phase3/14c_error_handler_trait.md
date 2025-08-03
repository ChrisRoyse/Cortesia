# Task 14c: Create Error Handler Trait

**Time**: 5 minutes
**Dependencies**: 14b_error_context.md
**Stage**: Inheritance System

## Objective
Define trait for standardized error handling across the inheritance system.

## Implementation
Add to `src/inheritance/error_types.rs`:

```rust
use async_trait::async_trait;

#[async_trait]
pub trait ErrorHandler: Send + Sync {
    async fn handle_inheritance_error(&self, error: InheritanceErrorWithContext) -> ErrorHandlingResult;
    async fn handle_property_error(&self, error: PropertyError, context: ErrorContext) -> ErrorHandlingResult;
    async fn should_retry(&self, error: &InheritanceError) -> bool;
    async fn get_retry_delay(&self, attempt: u32) -> std::time::Duration;
}

#[derive(Debug)]
pub enum ErrorHandlingResult {
    Retry { delay: std::time::Duration },
    Fail { final_error: String },
    Recover { recovered_value: Option<String> },
    Ignore,
}

pub struct DefaultErrorHandler {
    max_retries: u32,
    base_delay_ms: u64,
}

impl DefaultErrorHandler {
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
        }
    }

    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    pub fn with_base_delay(mut self, delay_ms: u64) -> Self {
        self.base_delay_ms = delay_ms;
        self
    }
}

#[async_trait]
impl ErrorHandler for DefaultErrorHandler {
    async fn handle_inheritance_error(&self, error: InheritanceErrorWithContext) -> ErrorHandlingResult {
        match &error.error {
            InheritanceError::DatabaseError(_) => {
                ErrorHandlingResult::Retry { 
                    delay: std::time::Duration::from_millis(self.base_delay_ms) 
                }
            }
            InheritanceError::CycleDetected { .. } => {
                ErrorHandlingResult::Fail { 
                    final_error: format!("Cycle detection: {}", error.error) 
                }
            }
            InheritanceError::CacheError(_) => {
                ErrorHandlingResult::Recover { recovered_value: None }
            }
            _ => ErrorHandlingResult::Fail { 
                final_error: error.error.to_string() 
            }
        }
    }

    async fn handle_property_error(&self, error: PropertyError, context: ErrorContext) -> ErrorHandlingResult {
        match error {
            PropertyError::ResolutionError(_) => {
                ErrorHandlingResult::Retry { 
                    delay: std::time::Duration::from_millis(self.base_delay_ms) 
                }
            }
            PropertyError::InheritanceConflict { .. } => {
                ErrorHandlingResult::Recover { recovered_value: Some("default".to_string()) }
            }
            _ => ErrorHandlingResult::Fail { 
                final_error: error.to_string() 
            }
        }
    }

    async fn should_retry(&self, error: &InheritanceError) -> bool {
        matches!(error, 
            InheritanceError::DatabaseError(_) | 
            InheritanceError::CacheError(_))
    }

    async fn get_retry_delay(&self, attempt: u32) -> std::time::Duration {
        let delay_ms = self.base_delay_ms * 2_u64.pow(attempt.min(5));
        std::time::Duration::from_millis(delay_ms)
    }
}
```

## Success Criteria
- Error handler trait is properly defined
- Default implementation handles common cases

## Next Task
14d_retry_logic.md