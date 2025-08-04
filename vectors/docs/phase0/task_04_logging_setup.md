# Task 04: Setup Logging and Tracing Configuration

## Context
You are completing the environment setup portion of Phase 0. Tasks 01-03 created the project structure, modules, and Windows configuration. Now you need to implement proper logging and tracing for debugging and performance monitoring.

## Objective
Implement comprehensive logging and tracing setup with Windows-compatible configuration, structured logging, and performance monitoring capabilities.

## Requirements
1. Setup tracing-subscriber with Windows-compatible formatting
2. Create different log levels for different components
3. Add performance tracing for benchmarks
4. Setup file-based logging for Windows
5. Create logging utilities for other modules to use

## Implementation for setup.rs (extend existing WindowsSetup)
```rust
use tracing::{info, debug, Level};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use std::fs::OpenOptions;
use std::io;

impl WindowsSetup {
    /// Setup comprehensive logging for Windows
    fn setup_logging() -> Result<()> {
        // Create logs directory
        std::fs::create_dir_all("logs")?;
        
        // Setup file appender for Windows
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("logs/llmkg.log")?;
        
        // Setup layered logging
        let file_layer = fmt::layer()
            .with_writer(file)
            .with_ansi(false) // No ANSI colors in file
            .with_target(true)
            .with_thread_ids(true)
            .with_file(true)
            .with_line_number(true);
        
        let stdout_layer = fmt::layer()
            .with_writer(io::stdout)
            .with_ansi(true) // Colors for console
            .with_target(false)
            .compact();
        
        // Environment filter with defaults
        let env_filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| {
                EnvFilter::new("debug,tantivy=info,lancedb=info")
            });
        
        tracing_subscriber::registry()
            .with(env_filter)
            .with(file_layer)
            .with(stdout_layer)
            .init();
        
        info!("Logging system initialized successfully");
        debug!("Debug logging enabled");
        
        Ok(())
    }
    
    /// Setup performance tracing for benchmarks
    pub fn setup_performance_tracing() -> Result<()> {
        // Additional setup for performance monitoring
        info!("Performance tracing enabled");
        Ok(())
    }
    
    /// Verify logging works correctly
    pub fn test_logging() -> Result<()> {
        use tracing::{error, warn, info, debug, trace};
        
        error!("Test error message");
        warn!("Test warning message");
        info!("Test info message");
        debug!("Test debug message");
        trace!("Test trace message");
        
        info!("Logging test completed");
        Ok(())
    }
}
```

## Logging Utilities Module (add to lib.rs)
```rust
/// Logging utilities for consistent formatting across modules
pub mod logging {
    use tracing::{info, debug, error};
    
    pub fn log_operation_start(operation: &str, details: &str) {
        info!("Starting {}: {}", operation, details);
    }
    
    pub fn log_operation_success(operation: &str, duration_ms: u64) {
        info!("Completed {} in {}ms", operation, duration_ms);
    }
    
    pub fn log_operation_error(operation: &str, error: &str) {
        error!("Failed {}: {}", operation, error);
    }
    
    pub fn log_performance_metric(metric: &str, value: f64, unit: &str) {
        info!("Performance: {} = {:.2} {}", metric, value, unit);
    }
}
```

## Implementation Steps
1. Extend WindowsSetup::setup_logging() with comprehensive logging
2. Add file-based logging to logs/ directory
3. Setup layered logging (console + file)
4. Create environment-based log filtering
5. Add performance tracing utilities
6. Create logging utilities module
7. Test all logging levels and outputs
8. Verify log files are created correctly on Windows

## Success Criteria
- [ ] Tracing-subscriber setup works on Windows
- [ ] Both console and file logging are active
- [ ] Different log levels work correctly (error, warn, info, debug, trace)
- [ ] Log files are created in logs/ directory
- [ ] Environment filtering works (RUST_LOG variable)
- [ ] Performance tracing utilities are available
- [ ] Logging utilities module is accessible
- [ ] All logging tests pass

## Test Command
```bash
# Set log level and test
set RUST_LOG=debug
cargo test test_logging
cargo run --example logging_test  # If you create an example
```

## Verification
Check that logs/llmkg.log is created and contains structured log entries with timestamps, thread IDs, and proper formatting.

## Time Estimate
10 minutes

## Next Task
Task 05: Test Tantivy basic functionality on Windows with special character handling.