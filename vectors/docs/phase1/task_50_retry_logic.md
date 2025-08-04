# Task 50: Implement Intelligent Retry Logic for Transient Failures

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 49 (Error reporting)
**Input Files:** `C:\code\LLMKG\vectors\tantivy_search\src\lib.rs`, error handling modules

## Complete Context (For AI with ZERO Knowledge)

You are implementing **intelligent retry logic for a Rust search system** that handles transient failures during file operations, network requests, and database operations. Transient failures in search systems typically include:
- File system busy errors (Windows file locks, concurrent access)
- Index writer lock contention (multiple processes accessing same index)
- Memory pressure causing temporary allocation failures
- Network timeouts during distributed operations

**What is Intelligent Retry Logic?** A system that automatically retries failed operations using strategies like exponential backoff, jitter, circuit breakers, and failure classification to distinguish between retryable and permanent failures.

**This Task:** Creates a comprehensive retry framework with configurable strategies, backoff algorithms, and integration hooks for existing search operations.

## Exact Steps (6 minutes implementation)

### Step 1: Create retry logic module (4 minutes)
Create file: `C:\code\LLMKG\vectors\tantivy_search\src\retry_logic.rs`

```rust
//! Intelligent retry logic for handling transient failures
use std::time::{Duration, Instant};
use std::fmt;
use anyhow::{Result, Error};
use rand::Rng;
use tokio::time::sleep;

/// Configurable retry strategy for different failure types
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub backoff_strategy: BackoffStrategy,
    pub jitter: bool,
    pub timeout: Option<Duration>,
}

#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed,
    /// Linear increase: delay = base_delay * attempt
    Linear,
    /// Exponential backoff: delay = base_delay * 2^attempt
    Exponential,
    /// Custom function for calculating delay
    Custom(fn(u32, Duration) -> Duration),
}

/// Classification of failures for retry decisions
#[derive(Debug, Clone, PartialEq)]
pub enum FailureType {
    /// Transient failures that should be retried
    Transient,
    /// Permanent failures that should not be retried
    Permanent,
    /// Rate-limited operations that need longer delays
    RateLimited,
    /// Resource exhaustion that needs exponential backoff
    ResourceExhausted,
}

/// Retry execution context and statistics
#[derive(Debug)]
pub struct RetryContext {
    pub attempt: u32,
    pub total_elapsed: Duration,
    pub last_error: Option<Error>,
    pub failure_type: Option<FailureType>,
    pub start_time: Instant,
}

/// High-level retry executor
pub struct RetryExecutor {
    config: RetryConfig,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_strategy: BackoffStrategy::Exponential,
            jitter: true,
            timeout: Some(Duration::from_secs(300)), // 5 minute total timeout
        }
    }
}

impl RetryConfig {
    /// Create retry config optimized for file operations
    pub fn for_file_operations() -> Self {
        Self {
            max_attempts: 5,
            base_delay: Duration::from_millis(50),
            max_delay: Duration::from_secs(5),
            backoff_strategy: BackoffStrategy::Exponential,
            jitter: true,
            timeout: Some(Duration::from_secs(60)),
        }
    }

    /// Create retry config optimized for index operations
    pub fn for_index_operations() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(10),
            backoff_strategy: BackoffStrategy::Linear,
            jitter: false,
            timeout: Some(Duration::from_secs(120)),
        }
    }

    /// Create retry config optimized for network operations
    pub fn for_network_operations() -> Self {
        Self {
            max_attempts: 4,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(60),
            backoff_strategy: BackoffStrategy::Exponential,
            jitter: true,
            timeout: Some(Duration::from_secs(300)),
        }
    }
}

impl RetryExecutor {
    /// Create new retry executor with configuration
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Create retry executor with default configuration
    pub fn default() -> Self {
        Self::new(RetryConfig::default())
    }

    /// Execute operation with retry logic
    pub async fn execute<F, R, E>(&self, operation: F) -> Result<R>
    where
        F: Fn() -> Result<R, E>,
        E: Into<Error> + fmt::Debug + Clone,
    {
        let mut context = RetryContext {
            attempt: 0,
            total_elapsed: Duration::ZERO,
            last_error: None,
            failure_type: None,
            start_time: Instant::now(),
        };

        loop {
            context.attempt += 1;
            context.total_elapsed = context.start_time.elapsed();

            // Check timeout
            if let Some(timeout) = self.config.timeout {
                if context.total_elapsed >= timeout {
                    return Err(anyhow::anyhow!(
                        "Operation timed out after {:?} with {} attempts",
                        timeout,
                        context.attempt - 1
                    ));
                }
            }

            // Execute operation
            match operation() {
                Ok(result) => return Ok(result),
                Err(error) => {
                    let error: Error = error.into();
                    context.failure_type = Some(self.classify_error(&error));
                    context.last_error = Some(error.clone());

                    // Check if we should retry
                    if !self.should_retry(&context) {
                        return Err(error);
                    }

                    // Calculate delay and wait
                    if context.attempt < self.config.max_attempts {
                        let delay = self.calculate_delay(context.attempt, &context);
                        sleep(delay).await;
                    }
                }
            }

            // Max attempts reached
            if context.attempt >= self.config.max_attempts {
                let final_error = context.last_error.unwrap_or_else(|| {
                    anyhow::anyhow!("Max retry attempts ({}) exceeded", self.config.max_attempts)
                });
                return Err(final_error);
            }
        }
    }

    /// Execute operation synchronously with retry logic
    pub fn execute_sync<F, R, E>(&self, operation: F) -> Result<R>
    where
        F: Fn() -> Result<R, E>,
        E: Into<Error> + fmt::Debug + Clone,
    {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(self.execute(operation))
    }

    /// Classify error to determine retry strategy
    fn classify_error(&self, error: &Error) -> FailureType {
        let error_msg = error.to_string().to_lowercase();

        // File system errors
        if error_msg.contains("permission denied") || 
           error_msg.contains("access denied") ||
           error_msg.contains("file not found") {
            return FailureType::Permanent;
        }

        // Resource contention
        if error_msg.contains("resource busy") ||
           error_msg.contains("lock") ||
           error_msg.contains("would block") {
            return FailureType::Transient;
        }

        // Memory issues
        if error_msg.contains("out of memory") ||
           error_msg.contains("allocation failed") {
            return FailureType::ResourceExhausted;
        }

        // Network issues
        if error_msg.contains("timeout") ||
           error_msg.contains("connection") ||
           error_msg.contains("network") {
            return FailureType::Transient;
        }

        // Rate limiting
        if error_msg.contains("rate limit") ||
           error_msg.contains("throttle") ||
           error_msg.contains("too many requests") {
            return FailureType::RateLimited;
        }

        // Default to transient for unknown errors
        FailureType::Transient
    }

    /// Determine if operation should be retried based on context
    fn should_retry(&self, context: &RetryContext) -> bool {
        match context.failure_type {
            Some(FailureType::Permanent) => false,
            Some(FailureType::Transient) => true,
            Some(FailureType::RateLimited) => true,
            Some(FailureType::ResourceExhausted) => true,
            None => true, // Default to retry
        }
    }

    /// Calculate delay before next retry attempt
    fn calculate_delay(&self, attempt: u32, context: &RetryContext) -> Duration {
        let base_delay = match context.failure_type {
            Some(FailureType::RateLimited) => self.config.base_delay * 2, // Longer delays for rate limits
            Some(FailureType::ResourceExhausted) => self.config.base_delay * 3, // Even longer for resource exhaustion
            _ => self.config.base_delay,
        };

        let calculated_delay = match &self.config.backoff_strategy {
            BackoffStrategy::Fixed => base_delay,
            BackoffStrategy::Linear => base_delay * attempt,
            BackoffStrategy::Exponential => {
                let exp_delay = base_delay * (2_u32.pow(attempt.saturating_sub(1)));
                std::cmp::min(exp_delay, self.config.max_delay)
            },
            BackoffStrategy::Custom(calc_fn) => calc_fn(attempt, base_delay),
        };

        // Apply jitter to prevent thundering herd
        if self.config.jitter {
            let jitter_range = calculated_delay.as_millis() as f64 * 0.1; // 10% jitter
            let jitter = rand::thread_rng().gen_range(-jitter_range..=jitter_range);
            let jittered_ms = (calculated_delay.as_millis() as f64 + jitter).max(0.0) as u64;
            Duration::from_millis(jittered_ms)
        } else {
            calculated_delay
        }
    }
}

/// Convenience trait for adding retry logic to any operation
pub trait Retryable<T, E> {
    fn with_retry(self, config: RetryConfig) -> impl std::future::Future<Output = Result<T>>;
    fn with_default_retry(self) -> impl std::future::Future<Output = Result<T>>;
}

impl<F, T, E> Retryable<T, E> for F
where
    F: Fn() -> Result<T, E> + Send + Sync,
    T: Send,
    E: Into<Error> + fmt::Debug + Clone + Send,
{
    async fn with_retry(self, config: RetryConfig) -> Result<T> {
        let executor = RetryExecutor::new(config);
        executor.execute(self).await
    }

    async fn with_default_retry(self) -> Result<T> {
        let executor = RetryExecutor::default();
        executor.execute(self).await
    }
}

/// Specialized retry functions for common operations
pub mod specialized {
    use super::*;
    use std::path::Path;
    use std::fs;

    /// Retry file read operations with appropriate configuration
    pub async fn retry_file_read<P: AsRef<Path>>(path: P) -> Result<String> {
        let path = path.as_ref().to_path_buf();
        let operation = || -> Result<String, std::io::Error> {
            fs::read_to_string(&path)
        };

        let config = RetryConfig::for_file_operations();
        let executor = RetryExecutor::new(config);
        executor.execute(operation).await
    }

    /// Retry file write operations with appropriate configuration
    pub async fn retry_file_write<P: AsRef<Path>>(path: P, contents: String) -> Result<()> {
        let path = path.as_ref().to_path_buf();
        let operation = || -> Result<(), std::io::Error> {
            fs::write(&path, &contents)
        };

        let config = RetryConfig::for_file_operations();
        let executor = RetryExecutor::new(config);
        executor.execute(operation).await
    }

    /// Retry index operations with lock contention handling
    pub async fn retry_index_operation<F, R, E>(operation: F) -> Result<R>
    where
        F: Fn() -> Result<R, E>,
        E: Into<Error> + fmt::Debug + Clone,
    {
        let config = RetryConfig::for_index_operations();
        let executor = RetryExecutor::new(config);
        executor.execute(operation).await
    }
}
```

### Step 2: Add comprehensive retry logic tests (2 minutes)
Create file: `C:\code\LLMKG\vectors\tantivy_search\tests\retry_logic_tests.rs`

```rust
//! Retry logic comprehensive tests
use tantivy_search::retry_logic::*;
use tantivy_search::retry_logic::specialized::*;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use std::fs;

#[tokio::test]
async fn test_successful_operation_no_retry() -> anyhow::Result<()> {
    let executor = RetryExecutor::default();
    let call_count = Arc::new(Mutex::new(0));
    let call_count_clone = Arc::clone(&call_count);

    let result = executor.execute(|| {
        let mut count = call_count_clone.lock().unwrap();
        *count += 1;
        Ok::<i32, std::io::Error>(42)
    }).await?;

    assert_eq!(result, 42);
    assert_eq!(*call_count.lock().unwrap(), 1);
    Ok(())
}

#[tokio::test]
async fn test_retry_with_eventual_success() -> anyhow::Result<()> {
    let executor = RetryExecutor::new(RetryConfig {
        max_attempts: 3,
        base_delay: Duration::from_millis(10),
        ..Default::default()
    });

    let call_count = Arc::new(Mutex::new(0));
    let call_count_clone = Arc::clone(&call_count);

    let result = executor.execute(|| {
        let mut count = call_count_clone.lock().unwrap();
        *count += 1;
        
        if *count < 3 {
            Err(std::io::Error::new(std::io::ErrorKind::Other, "Transient failure"))
        } else {
            Ok::<String, std::io::Error>("Success".to_string())
        }
    }).await?;

    assert_eq!(result, "Success");
    assert_eq!(*call_count.lock().unwrap(), 3);
    Ok(())
}

#[tokio::test]
async fn test_max_attempts_exceeded() {
    let executor = RetryExecutor::new(RetryConfig {
        max_attempts: 2,
        base_delay: Duration::from_millis(1),
        ..Default::default()
    });

    let call_count = Arc::new(Mutex::new(0));
    let call_count_clone = Arc::clone(&call_count);

    let result = executor.execute(|| {
        let mut count = call_count_clone.lock().unwrap();
        *count += 1;
        Err::<(), std::io::Error>(std::io::Error::new(std::io::ErrorKind::Other, "Always fails"))
    }).await;

    assert!(result.is_err());
    assert_eq!(*call_count.lock().unwrap(), 2);
}

#[tokio::test]
async fn test_permanent_failure_no_retry() {
    let executor = RetryExecutor::default();
    let call_count = Arc::new(Mutex::new(0));
    let call_count_clone = Arc::clone(&call_count);

    let result = executor.execute(|| {
        let mut count = call_count_clone.lock().unwrap();
        *count += 1;
        Err::<(), std::io::Error>(std::io::Error::new(std::io::ErrorKind::PermissionDenied, "Permission denied"))
    }).await;

    assert!(result.is_err());
    assert_eq!(*call_count.lock().unwrap(), 1); // Should not retry permanent failures
}

#[test]
fn test_backoff_strategies() {
    let base_delay = Duration::from_millis(100);
    
    // Fixed backoff
    let config = RetryConfig {
        backoff_strategy: BackoffStrategy::Fixed,
        base_delay,
        max_delay: Duration::from_secs(10),
        jitter: false,
        ..Default::default()
    };
    let executor = RetryExecutor::new(config);
    assert_eq!(executor.calculate_delay(1, &RetryContext {
        attempt: 1,
        total_elapsed: Duration::ZERO,
        last_error: None,
        failure_type: Some(FailureType::Transient),
        start_time: Instant::now(),
    }), base_delay);

    // Linear backoff
    let config = RetryConfig {
        backoff_strategy: BackoffStrategy::Linear,
        base_delay,
        max_delay: Duration::from_secs(10),
        jitter: false,
        ..Default::default()
    };
    let executor = RetryExecutor::new(config);
    assert_eq!(executor.calculate_delay(3, &RetryContext {
        attempt: 3,
        total_elapsed: Duration::ZERO,
        last_error: None,
        failure_type: Some(FailureType::Transient),
        start_time: Instant::now(),
    }), base_delay * 3);

    // Exponential backoff
    let config = RetryConfig {
        backoff_strategy: BackoffStrategy::Exponential,
        base_delay,
        max_delay: Duration::from_secs(10),
        jitter: false,
        ..Default::default()
    };
    let executor = RetryExecutor::new(config);
    assert_eq!(executor.calculate_delay(3, &RetryContext {
        attempt: 3,
        total_elapsed: Duration::ZERO,
        last_error: None,
        failure_type: Some(FailureType::Transient),
        start_time: Instant::now(),
    }), base_delay * 4); // 2^(3-1) = 4
}

#[test]
fn test_error_classification() {
    let executor = RetryExecutor::default();
    
    // Test permanent errors
    let perm_error = anyhow::anyhow!("Permission denied");
    assert_eq!(executor.classify_error(&perm_error), FailureType::Permanent);
    
    // Test transient errors
    let trans_error = anyhow::anyhow!("Resource busy");
    assert_eq!(executor.classify_error(&trans_error), FailureType::Transient);
    
    // Test rate limited errors
    let rate_error = anyhow::anyhow!("Rate limit exceeded");
    assert_eq!(executor.classify_error(&rate_error), FailureType::RateLimited);
    
    // Test resource exhaustion
    let resource_error = anyhow::anyhow!("Out of memory");
    assert_eq!(executor.classify_error(&resource_error), FailureType::ResourceExhausted);
}

#[tokio::test]
async fn test_specialized_file_operations() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let test_file = temp_dir.path().join("retry_test.txt");
    let test_content = "Retry test content";

    // Test retry file write
    retry_file_write(&test_file, test_content.to_string()).await?;

    // Test retry file read
    let read_content = retry_file_read(&test_file).await?;
    assert_eq!(read_content, test_content);

    Ok(())
}

#[tokio::test]
async fn test_retryable_trait() -> anyhow::Result<()> {
    let call_count = Arc::new(Mutex::new(0));
    let call_count_clone = Arc::clone(&call_count);

    let operation = || {
        let mut count = call_count_clone.lock().unwrap();
        *count += 1;
        if *count < 2 {
            Err(std::io::Error::new(std::io::ErrorKind::Other, "Fail once"))
        } else {
            Ok("Success".to_string())
        }
    };

    let result = operation.with_default_retry().await?;
    assert_eq!(result, "Success");
    assert_eq!(*call_count.lock().unwrap(), 2);

    Ok(())
}

#[tokio::test]
async fn test_timeout_handling() {
    let executor = RetryExecutor::new(RetryConfig {
        max_attempts: 10,
        base_delay: Duration::from_millis(100),
        timeout: Some(Duration::from_millis(250)), // Short timeout
        ..Default::default()
    });

    let start_time = Instant::now();
    let result = executor.execute(|| {
        Err::<(), std::io::Error>(std::io::Error::new(std::io::ErrorKind::Other, "Always fails"))
    }).await;

    assert!(result.is_err());
    assert!(start_time.elapsed() >= Duration::from_millis(250));
    assert!(start_time.elapsed() < Duration::from_millis(400)); // Should timeout before many retries
}

#[test]
fn test_retry_config_presets() {
    let file_config = RetryConfig::for_file_operations();
    assert_eq!(file_config.max_attempts, 5);
    assert_eq!(file_config.base_delay, Duration::from_millis(50));

    let index_config = RetryConfig::for_index_operations();
    assert_eq!(index_config.max_attempts, 3);
    assert_eq!(index_config.base_delay, Duration::from_millis(200));

    let network_config = RetryConfig::for_network_operations();
    assert_eq!(network_config.max_attempts, 4);
    assert_eq!(network_config.base_delay, Duration::from_millis(500));
}

#[tokio::test]
async fn test_jitter_application() {
    let config = RetryConfig {
        backoff_strategy: BackoffStrategy::Fixed,
        base_delay: Duration::from_millis(100),
        jitter: true,
        ..Default::default()
    };
    let executor = RetryExecutor::new(config);

    // Calculate multiple delays and ensure they vary due to jitter
    let delays: Vec<Duration> = (0..10)
        .map(|_| executor.calculate_delay(1, &RetryContext {
            attempt: 1,
            total_elapsed: Duration::ZERO,
            last_error: None,
            failure_type: Some(FailureType::Transient),
            start_time: Instant::now(),
        }))
        .collect();

    // With jitter, not all delays should be identical
    let all_same = delays.windows(2).all(|w| w[0] == w[1]);
    assert!(!all_same, "Jitter should cause delay variation");
}
```

## Verification Steps (2 minutes)

### Verify 1: Compilation succeeds
```bash
cd C:\code\LLMKG\vectors\tantivy_search
cargo check
```

### Verify 2: Retry logic tests pass
```bash
cargo test retry_logic_tests
```
**Expected output:**
```
running 11 tests
test retry_logic_tests::test_successful_operation_no_retry ... ok
test retry_logic_tests::test_retry_with_eventual_success ... ok
test retry_logic_tests::test_max_attempts_exceeded ... ok
test retry_logic_tests::test_permanent_failure_no_retry ... ok
test retry_logic_tests::test_backoff_strategies ... ok
test retry_logic_tests::test_error_classification ... ok
test retry_logic_tests::test_specialized_file_operations ... ok
test retry_logic_tests::test_retryable_trait ... ok
test retry_logic_tests::test_timeout_handling ... ok
test retry_logic_tests::test_retry_config_presets ... ok
test retry_logic_tests::test_jitter_application ... ok

test result: ok. 11 passed; 0 failed
```

### Verify 3: Add module export
Add to `C:\code\LLMKG\vectors\tantivy_search\src\lib.rs`:
```rust
pub mod retry_logic;
```

## Success Validation Checklist
- [ ] File `retry_logic.rs` completely implemented with intelligent retry strategies
- [ ] File `retry_logic_tests.rs` created with 11+ comprehensive tests  
- [ ] Command `cargo check` completes without errors
- [ ] Command `cargo test retry_logic_tests` passes all tests
- [ ] Multiple backoff strategies implemented (fixed, linear, exponential, custom)
- [ ] Error classification system distinguishes transient vs permanent failures
- [ ] Specialized retry functions for file, index, and network operations
- [ ] Jitter and timeout handling prevent thundering herd problems
- [ ] Integration trait allows easy adoption by existing components
- [ ] Configurable retry policies for different operation types

## Context for Task 51
Task 51 will implement the circuit breaker pattern to complement the retry logic, providing protection against cascading failures by temporarily disabling operations when failure rates exceed thresholds, and automatically recovering when the service becomes healthy again.
