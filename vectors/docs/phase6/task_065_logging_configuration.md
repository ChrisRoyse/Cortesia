# Task 065: Set up Logging Configuration

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates structured logging configuration with multiple output formats, log levels and filtering, performance logging, and error tracking integration for comprehensive system monitoring.

## Project Structure
```
src/logging/
├── mod.rs              <- Logging module entry point
├── config.rs           <- Logging configuration
├── formatters.rs       <- Custom log formatters
├── filters.rs          <- Log filtering logic
└── performance.rs      <- Performance logging utilities
```

## Task Description
Create a comprehensive logging system that provides structured logging with JSON and text formats, configurable log levels, performance metrics tracking, error correlation, and integration with the validation system for detailed monitoring and debugging.

## Requirements
1. Configure structured logging with multiple output formats (JSON, text, structured)
2. Implement configurable log levels and filtering by module
3. Add performance logging with timing and metrics
4. Create error tracking with correlation IDs
5. Support Windows-specific logging requirements

## Expected File Content/Code Structure

### Main Logging Module (`src/logging/mod.rs`)
```rust
//! Comprehensive logging system for LLMKG validation
//! 
//! Provides structured logging with multiple output formats, performance tracking,
//! and error correlation for the validation system.

pub mod config;
pub mod formatters;
pub mod filters;
pub mod performance;

use anyhow::Result;
use tracing::{info, warn, error};
use tracing_subscriber::{
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter, Registry
};
use tracing_appender::{non_blocking, rolling};
use std::path::Path;

pub use config::LoggingConfig;
pub use formatters::{JsonFormatter, StructuredFormatter};
pub use filters::ValidationFilter;
pub use performance::{PerformanceLogger, TimingGuard};

/// Initialize the logging system with the specified configuration
pub fn initialize_logging(config: &LoggingConfig) -> Result<()> {
    // Create log directory if it doesn't exist
    if let Some(parent) = Path::new(&config.log_file).parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    // Create file appender with rotation
    let file_appender = rolling::daily(&config.log_directory, &config.log_file_prefix);
    let (non_blocking_file, _guard) = non_blocking(file_appender);
    
    // Create console appender
    let (non_blocking_console, _console_guard) = non_blocking(std::io::stdout());
    
    // Build the subscriber
    let registry = Registry::default();
    
    // Add file layer with JSON formatting
    let file_layer = fmt::layer()
        .json()
        .with_writer(non_blocking_file)
        .with_span_events(FmtSpan::CLOSE)
        .with_current_span(true)
        .with_thread_ids(true)
        .with_thread_names(true);
    
    // Add console layer with structured formatting
    let console_layer = fmt::layer()
        .with_writer(non_blocking_console)
        .with_span_events(FmtSpan::CLOSE)
        .with_ansi(config.enable_colors)
        .with_target(true)
        .with_thread_ids(config.show_thread_ids)
        .with_line_number(config.show_line_numbers)
        .with_file(config.show_file_names);
    
    // Create environment filter
    let env_filter = create_env_filter(config)?;
    
    // Initialize the subscriber
    registry
        .with(env_filter)
        .with(file_layer)
        .with(console_layer)
        .init();
    
    info!(
        config = ?config,
        "Logging system initialized successfully"
    );
    
    Ok(())
}

/// Create environment filter based on configuration
fn create_env_filter(config: &LoggingConfig) -> Result<EnvFilter> {
    let mut filter = EnvFilter::new(&config.default_level);
    
    // Add module-specific filters
    for (module, level) in &config.module_filters {
        filter = filter.add_directive(format!("{}={}", module, level).parse()?);
    }
    
    // Add validation-specific filters
    if config.enable_validation_logging {
        filter = filter.add_directive("llmkg_validation=debug".parse()?);
    }
    
    if config.enable_performance_logging {
        filter = filter.add_directive("llmkg_validation::performance=trace".parse()?);
    }
    
    Ok(filter)
}

/// Log validation start with correlation ID
pub fn log_validation_start(validation_id: &str, test_count: usize) {
    info!(
        validation_id = validation_id,
        test_count = test_count,
        event = "validation_start",
        "Starting validation run"
    );
}

/// Log validation completion with results
pub fn log_validation_complete(
    validation_id: &str, 
    success: bool, 
    duration_ms: u64,
    accuracy: f64
) {
    if success {
        info!(
            validation_id = validation_id,
            duration_ms = duration_ms,
            accuracy = accuracy,
            event = "validation_complete",
            "Validation completed successfully"
        );
    } else {
        error!(
            validation_id = validation_id,
            duration_ms = duration_ms,
            accuracy = accuracy,
            event = "validation_failed",
            "Validation failed"
        );
    }
}

/// Log query execution with performance metrics
pub fn log_query_execution(
    query_id: &str,
    query: &str,
    query_type: &str,
    duration_ms: u64,
    result_count: usize,
    success: bool
) {
    if success {
        info!(
            query_id = query_id,
            query = query,
            query_type = query_type,
            duration_ms = duration_ms,
            result_count = result_count,
            event = "query_executed",
            "Query executed successfully"
        );
    } else {
        warn!(
            query_id = query_id,
            query = query,
            query_type = query_type,
            duration_ms = duration_ms,
            event = "query_failed",
            "Query execution failed"
        );
    }
}

/// Log system performance metrics
pub fn log_system_metrics(
    memory_usage_mb: u64,
    cpu_usage_percent: f64,
    active_queries: usize,
    index_size_mb: u64
) {
    info!(
        memory_usage_mb = memory_usage_mb,
        cpu_usage_percent = cpu_usage_percent,
        active_queries = active_queries,
        index_size_mb = index_size_mb,
        event = "system_metrics",
        "System performance metrics"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_logging_initialization() -> Result<()> {
        let temp_dir = tempdir()?;
        let config = LoggingConfig {
            log_directory: temp_dir.path().to_string_lossy().to_string(),
            log_file_prefix: "test".to_string(),
            default_level: "info".to_string(),
            ..Default::default()
        };
        
        initialize_logging(&config)?;
        
        // Test that logging works
        info!("Test log message");
        
        Ok(())
    }
}
```

### Logging Configuration (`src/logging/config.rs`)
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Directory for log files
    pub log_directory: String,
    
    /// Prefix for log file names
    pub log_file_prefix: String,
    
    /// Main log file name
    pub log_file: String,
    
    /// Default log level (trace, debug, info, warn, error)
    pub default_level: String,
    
    /// Module-specific log levels
    pub module_filters: HashMap<String, String>,
    
    /// Enable validation-specific logging
    pub enable_validation_logging: bool,
    
    /// Enable performance logging
    pub enable_performance_logging: bool,
    
    /// Enable error tracking
    pub enable_error_tracking: bool,
    
    /// Log failed queries for debugging
    pub log_failed_queries: bool,
    
    /// Enable colored output in console
    pub enable_colors: bool,
    
    /// Show thread IDs in logs
    pub show_thread_ids: bool,
    
    /// Show line numbers in logs
    pub show_line_numbers: bool,
    
    /// Show file names in logs
    pub show_file_names: bool,
    
    /// Maximum log file size in MB before rotation
    pub max_file_size_mb: u64,
    
    /// Number of rotated log files to keep
    pub max_files_to_keep: usize,
    
    /// Enable structured JSON logging to file
    pub enable_json_logging: bool,
    
    /// Enable plain text logging to console
    pub enable_console_logging: bool,
    
    /// Windows-specific configuration
    pub windows_config: WindowsLoggingConfig,
}

/// Windows-specific logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsLoggingConfig {
    /// Enable Windows Event Log integration
    pub enable_event_log: bool,
    
    /// Event log source name
    pub event_log_source: String,
    
    /// Handle Unicode in log messages
    pub handle_unicode: bool,
    
    /// Normalize path separators in logs
    pub normalize_paths: bool,
    
    /// Enable Windows performance counters logging
    pub enable_performance_counters: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        let mut module_filters = HashMap::new();
        module_filters.insert("llmkg_validation".to_string(), "debug".to_string());
        module_filters.insert("tantivy".to_string(), "warn".to_string());
        module_filters.insert("lancedb".to_string(), "warn".to_string());
        module_filters.insert("tokio".to_string(), "warn".to_string());
        
        Self {
            log_directory: "./logs".to_string(),
            log_file_prefix: "validation".to_string(),
            log_file: "./logs/validation.log".to_string(),
            default_level: "info".to_string(),
            module_filters,
            enable_validation_logging: true,
            enable_performance_logging: true,
            enable_error_tracking: true,
            log_failed_queries: true,
            enable_colors: true,
            show_thread_ids: true,
            show_line_numbers: true,
            show_file_names: true,
            max_file_size_mb: 100,
            max_files_to_keep: 10,
            enable_json_logging: true,
            enable_console_logging: true,
            windows_config: WindowsLoggingConfig::default(),
        }
    }
}

impl Default for WindowsLoggingConfig {
    fn default() -> Self {
        Self {
            enable_event_log: false, // Disabled by default to avoid permissions issues
            event_log_source: "LLMKG-Validation".to_string(),
            handle_unicode: true,
            normalize_paths: true,
            enable_performance_counters: true,
        }
    }
}

/// Load logging configuration from TOML file
pub fn load_config_from_file(path: &str) -> anyhow::Result<LoggingConfig> {
    let content = std::fs::read_to_string(path)?;
    let config: LoggingConfig = toml::from_str(&content)?;
    Ok(config)
}

/// Save logging configuration to TOML file
pub fn save_config_to_file(config: &LoggingConfig, path: &str) -> anyhow::Result<()> {
    let content = toml::to_string_pretty(config)?;
    std::fs::write(path, content)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_default_config() {
        let config = LoggingConfig::default();
        assert_eq!(config.default_level, "info");
        assert!(config.enable_validation_logging);
        assert!(config.enable_performance_logging);
    }
    
    #[test]
    fn test_config_serialization() -> anyhow::Result<()> {
        let config = LoggingConfig::default();
        let temp_file = NamedTempFile::new()?;
        
        save_config_to_file(&config, temp_file.path().to_str().unwrap())?;
        let loaded_config = load_config_from_file(temp_file.path().to_str().unwrap())?;
        
        assert_eq!(config.default_level, loaded_config.default_level);
        assert_eq!(config.enable_validation_logging, loaded_config.enable_validation_logging);
        
        Ok(())
    }
}
```

### Performance Logging Utilities (`src/logging/performance.rs`)
```rust
use std::time::{Duration, Instant};
use tracing::{info, debug, warn};
use serde::{Deserialize, Serialize};

/// Performance logger for tracking system metrics
#[derive(Debug)]
pub struct PerformanceLogger {
    start_time: Instant,
    last_checkpoint: Instant,
}

impl PerformanceLogger {
    /// Create a new performance logger
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            start_time: now,
            last_checkpoint: now,
        }
    }
    
    /// Record a timing checkpoint
    pub fn checkpoint(&mut self, operation: &str) {
        let now = Instant::now();
        let since_last = now.duration_since(self.last_checkpoint);
        let total_elapsed = now.duration_since(self.start_time);
        
        debug!(
            operation = operation,
            checkpoint_duration_ms = since_last.as_millis(),
            total_elapsed_ms = total_elapsed.as_millis(),
            event = "performance_checkpoint",
            "Performance checkpoint recorded"
        );
        
        self.last_checkpoint = now;
    }
    
    /// Log final performance summary
    pub fn finish(&self, operation: &str) {
        let total_elapsed = self.start_time.elapsed();
        
        info!(
            operation = operation,
            total_duration_ms = total_elapsed.as_millis(),
            event = "performance_complete",
            "Performance logging complete"
        );
    }
}

/// RAII timing guard that automatically logs duration on drop
pub struct TimingGuard {
    operation: String,
    start_time: Instant,
    log_on_slow: Option<Duration>,
}

impl TimingGuard {
    /// Create a new timing guard
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            start_time: Instant::now(),
            log_on_slow: None,
        }
    }
    
    /// Create timing guard that warns if operation takes longer than threshold
    pub fn with_slow_threshold(operation: impl Into<String>, threshold: Duration) -> Self {
        Self {
            operation: operation.into(),
            start_time: Instant::now(),
            log_on_slow: Some(threshold),
        }
    }
    
    /// Get elapsed time without dropping the guard
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }
}

impl Drop for TimingGuard {
    fn drop(&mut self) {
        let elapsed = self.start_time.elapsed();
        
        if let Some(threshold) = self.log_on_slow {
            if elapsed > threshold {
                warn!(
                    operation = %self.operation,
                    duration_ms = elapsed.as_millis(),
                    threshold_ms = threshold.as_millis(),
                    event = "slow_operation",
                    "Operation exceeded expected duration"
                );
            }
        }
        
        debug!(
            operation = %self.operation,
            duration_ms = elapsed.as_millis(),
            event = "operation_complete",
            "Operation timing recorded"
        );
    }
}

/// System performance metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub memory_usage_mb: u64,
    pub cpu_usage_percent: f64,
    pub disk_usage_mb: u64,
    pub active_threads: usize,
    pub open_file_handles: usize,
}

/// Performance metrics collector
pub struct MetricsCollector {
    #[cfg(windows)]
    performance_counter: Option<WindowsPerformanceCounter>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            #[cfg(windows)]
            performance_counter: WindowsPerformanceCounter::new().ok(),
        }
    }
    
    /// Collect current system metrics
    pub fn collect_metrics(&self) -> anyhow::Result<SystemMetrics> {
        let timestamp = chrono::Utc::now();
        
        // Get memory usage
        let memory_usage_mb = self.get_memory_usage()?;
        
        // Get CPU usage
        let cpu_usage_percent = self.get_cpu_usage()?;
        
        // Get disk usage
        let disk_usage_mb = self.get_disk_usage()?;
        
        // Get thread count
        let active_threads = self.get_thread_count()?;
        
        // Get file handle count
        let open_file_handles = self.get_file_handle_count()?;
        
        Ok(SystemMetrics {
            timestamp,
            memory_usage_mb,
            cpu_usage_percent,
            disk_usage_mb,
            active_threads,
            open_file_handles,
        })
    }
    
    /// Log collected metrics
    pub fn log_metrics(&self) -> anyhow::Result<()> {
        let metrics = self.collect_metrics()?;
        
        info!(
            timestamp = %metrics.timestamp,
            memory_usage_mb = metrics.memory_usage_mb,
            cpu_usage_percent = metrics.cpu_usage_percent,
            disk_usage_mb = metrics.disk_usage_mb,
            active_threads = metrics.active_threads,
            open_file_handles = metrics.open_file_handles,
            event = "system_metrics",
            "System performance metrics collected"
        );
        
        // Warn on high resource usage
        if metrics.memory_usage_mb > 1024 {
            warn!(
                memory_usage_mb = metrics.memory_usage_mb,
                event = "high_memory_usage",
                "High memory usage detected"
            );
        }
        
        if metrics.cpu_usage_percent > 80.0 {
            warn!(
                cpu_usage_percent = metrics.cpu_usage_percent,
                event = "high_cpu_usage", 
                "High CPU usage detected"
            );
        }
        
        Ok(())
    }
    
    // Platform-specific implementations
    #[cfg(windows)]
    fn get_memory_usage(&self) -> anyhow::Result<u64> {
        use windows::Win32::System::ProcessStatus::GetProcessMemoryInfo;
        use windows::Win32::Foundation::GetCurrentProcess;
        // Implementation using Windows APIs
        Ok(0) // Placeholder
    }
    
    #[cfg(not(windows))]
    fn get_memory_usage(&self) -> anyhow::Result<u64> {
        // Unix implementation using /proc/self/status
        Ok(0) // Placeholder
    }
    
    fn get_cpu_usage(&self) -> anyhow::Result<f64> {
        // Platform-specific CPU usage implementation
        Ok(0.0) // Placeholder
    }
    
    fn get_disk_usage(&self) -> anyhow::Result<u64> {
        // Platform-specific disk usage implementation
        Ok(0) // Placeholder
    }
    
    fn get_thread_count(&self) -> anyhow::Result<usize> {
        // Platform-specific thread count implementation
        Ok(1) // Placeholder
    }
    
    fn get_file_handle_count(&self) -> anyhow::Result<usize> {
        // Platform-specific file handle count implementation
        Ok(0) // Placeholder
    }
}

#[cfg(windows)]
struct WindowsPerformanceCounter {
    // Windows-specific performance counter implementation
}

#[cfg(windows)]
impl WindowsPerformanceCounter {
    fn new() -> anyhow::Result<Self> {
        Ok(Self {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_timing_guard() {
        let _guard = TimingGuard::new("test_operation");
        thread::sleep(Duration::from_millis(10));
        // Guard automatically logs on drop
    }
    
    #[test]
    fn test_performance_logger() {
        let mut logger = PerformanceLogger::new();
        
        thread::sleep(Duration::from_millis(5));
        logger.checkpoint("step_1");
        
        thread::sleep(Duration::from_millis(5));
        logger.checkpoint("step_2");
        
        logger.finish("test_operation");
    }
    
    #[test]
    fn test_metrics_collector() -> anyhow::Result<()> {
        let collector = MetricsCollector::new();
        let metrics = collector.collect_metrics()?;
        
        assert!(metrics.timestamp <= chrono::Utc::now());
        
        Ok(())
    }
}
```

### Custom Log Formatters (`src/logging/formatters.rs`)
```rust
use serde_json::{json, Value};
use tracing::{Event, Subscriber};
use tracing_subscriber::fmt::{format::Writer, FormatEvent, FormatFields};
use tracing_subscriber::registry::LookupSpan;
use std::fmt;

/// Custom JSON formatter for structured logging
pub struct JsonFormatter;

impl<S, N> FormatEvent<S, N> for JsonFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let metadata = event.metadata();
        
        let mut json_event = json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "level": metadata.level().to_string(),
            "target": metadata.target(),
            "module": metadata.module_path(),
            "file": metadata.file(),
            "line": metadata.line(),
        });
        
        // Add span information
        if let Some(span) = ctx.lookup_current() {
            json_event["span"] = json!({
                "name": span.name(),
                "target": span.metadata().target(),
            });
        }
        
        // Add event fields
        let mut field_visitor = JsonFieldVisitor::new();
        event.record(&mut field_visitor);
        
        if !field_visitor.fields.is_empty() {
            json_event["fields"] = Value::Object(field_visitor.fields);
        }
        
        writeln!(writer, "{}", json_event.to_string())
    }
}

/// Field visitor for JSON formatting
struct JsonFieldVisitor {
    fields: serde_json::Map<String, Value>,
}

impl JsonFieldVisitor {
    fn new() -> Self {
        Self {
            fields: serde_json::Map::new(),
        }
    }
}

impl tracing::field::Visit for JsonFieldVisitor {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn fmt::Debug) {
        self.fields.insert(
            field.name().to_string(),
            Value::String(format!("{:?}", value)),
        );
    }
    
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.fields.insert(
            field.name().to_string(),
            Value::String(value.to_string()),
        );
    }
    
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        self.fields.insert(
            field.name().to_string(),
            Value::Number(value.into()),
        );
    }
    
    fn record_f64(&mut self, field: &tracing::field::Field, value: f64) {
        if let Some(number) = serde_json::Number::from_f64(value) {
            self.fields.insert(
                field.name().to_string(),
                Value::Number(number),
            );
        }
    }
    
    fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
        self.fields.insert(
            field.name().to_string(),
            Value::Bool(value),
        );
    }
}

/// Custom structured formatter for human-readable logs
pub struct StructuredFormatter;

impl<S, N> FormatEvent<S, N> for StructuredFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        let metadata = event.metadata();
        
        // Write timestamp and level
        write!(
            writer,
            "{} {:5} ",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.3f"),
            metadata.level().to_string()
        )?;
        
        // Write span context if available
        if let Some(span) = ctx.lookup_current() {
            write!(writer, "[{}] ", span.name())?;
        }
        
        // Write target
        write!(writer, "{}: ", metadata.target())?;
        
        // Write event message and fields
        ctx.format_fields(writer.by_ref(), event)?;
        
        // Write location info
        if let (Some(file), Some(line)) = (metadata.file(), metadata.line()) {
            write!(writer, " ({}:{})", file, line)?;
        }
        
        writeln!(writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_test::traced_test;
    
    #[traced_test]
    #[test]
    fn test_json_formatter() {
        tracing::info!(
            event = "test_event",
            count = 42,
            success = true,
            "Test message"
        );
        
        // Test that JSON formatter doesn't panic
        // Actual output validation would require capturing the writer
    }
    
    #[traced_test]
    #[test]
    fn test_structured_formatter() {
        tracing::warn!(
            operation = "test_operation",
            duration_ms = 150,
            "Operation completed with warning"
        );
        
        // Test that structured formatter doesn't panic
    }
}
```

## Success Criteria
- Logging system initializes successfully with configuration
- Multiple output formats (JSON, structured text) work correctly
- Log levels and filtering function as expected
- Performance logging captures timing and metrics accurately
- Error tracking with correlation IDs works properly
- Windows-specific logging features are implemented
- Log rotation and cleanup work correctly
- Integration with validation system provides comprehensive monitoring

## Time Limit
10 minutes maximum