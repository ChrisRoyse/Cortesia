# Task 14d: Implement Retry Logic

**Time**: 7 minutes
**Dependencies**: 14c_error_handler_trait.md
**Stage**: Inheritance System

## Objective
Create retry mechanism for transient errors in inheritance operations.

## Implementation
Create `src/inheritance/retry_logic.rs`:

```rust
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use crate::inheritance::error_types::*;

pub struct RetryExecutor {
    error_handler: Arc<dyn ErrorHandler>,
    max_attempts: u32,
}

impl RetryExecutor {
    pub fn new(error_handler: Arc<dyn ErrorHandler>) -> Self {
        Self {
            error_handler,
            max_attempts: 3,
        }
    }

    pub fn with_max_attempts(mut self, max_attempts: u32) -> Self {
        self.max_attempts = max_attempts;
        self
    }

    pub async fn execute_with_retry<T, F, Fut>(
        &self,
        operation: F,
        context: ErrorContext,
    ) -> Result<T, InheritanceErrorWithContext>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<T, InheritanceError>> + Send,
        T: Send,
    {
        let mut attempt = 0;
        let mut last_error = None;

        while attempt < self.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let should_retry = self.error_handler.should_retry(&error).await;
                    
                    if !should_retry || attempt >= self.max_attempts - 1 {
                        let error_with_context = InheritanceErrorWithContext::new(error, context);
                        return Err(error_with_context);
                    }

                    let delay = self.error_handler.get_retry_delay(attempt).await;
                    sleep(delay).await;
                    
                    last_error = Some(error);
                    attempt += 1;
                }
            }
        }

        // This should never be reached due to the logic above, but just in case
        let final_error = last_error.unwrap_or(InheritanceError::ConfigurationError(
            "Retry logic error".to_string()
        ));
        Err(InheritanceErrorWithContext::new(final_error, context))
    }

    pub async fn execute_property_operation_with_retry<T, F, Fut>(
        &self,
        operation: F,
        context: ErrorContext,
    ) -> Result<T, PropertyError>
    where
        F: Fn() -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<T, PropertyError>> + Send,
        T: Send,
    {
        let mut attempt = 0;
        let mut last_error = None;

        while attempt < self.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let handling_result = self.error_handler.handle_property_error(error.clone(), context.clone()).await;
                    
                    match handling_result {
                        ErrorHandlingResult::Retry { delay } => {
                            if attempt >= self.max_attempts - 1 {
                                return Err(error);
                            }
                            sleep(delay).await;
                            attempt += 1;
                        }
                        ErrorHandlingResult::Fail { .. } => {
                            return Err(error);
                        }
                        ErrorHandlingResult::Recover { .. } => {
                            // For now, just fail - recovery would need more context
                            return Err(error);
                        }
                        ErrorHandlingResult::Ignore => {
                            // Return a default or continue - would need type-specific handling
                            return Err(error);
                        }
                    }
                    
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or(PropertyError::ResolutionError(
            "Retry logic error".to_string()
        )))
    }
}

// Helper macro for retry operations
#[macro_export]
macro_rules! retry_inheritance_operation {
    ($executor:expr, $operation:expr, $context:expr) => {
        $executor.execute_with_retry(|| async { $operation }, $context).await
    };
}
```

## Success Criteria
- Retry logic handles transient failures correctly
- Exponential backoff is implemented
- Maximum attempts are respected

## Next Task
14e_error_logging.md