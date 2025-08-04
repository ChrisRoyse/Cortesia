# Task 066: Create Error Handling Utilities

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This creates common error types and handling utilities, error conversion utilities, contextual error reporting, recovery strategies, and integration with anyhow for robust error management throughout the validation system.

## Project Structure
```
src/error/
├── mod.rs              <- Error module entry point
├── types.rs            <- Common error types
├── conversion.rs       <- Error conversion utilities
├── context.rs          <- Contextual error reporting
├── recovery.rs         <- Error recovery strategies
└── validation.rs       <- Validation-specific errors
```

## Task Description
Create a comprehensive error handling system that provides structured error types, automatic error conversion, contextual error reporting, recovery strategies, and integration with the anyhow crate for consistent error handling throughout the validation system.

## Requirements
1. Define common error types for different failure modes
2. Implement error conversion utilities for third-party library errors
3. Create contextual error reporting with detailed diagnostics
4. Implement recovery strategies for common error scenarios
5. Integrate with anyhow for ergonomic error handling

## Expected File Content/Code Structure

### Main Error Module (`src/error/mod.rs`)
```rust
//! Comprehensive error handling system for LLMKG validation
//! 
//! Provides structured error types, contextual reporting, and recovery strategies
//! for robust error handling throughout the validation system.

pub mod types;
pub mod conversion;
pub mod context;
pub mod recovery;
pub mod validation;

use anyhow::{Context as AnyhowContext, Result};
use thiserror::Error;
use std::fmt;

pub use types::*;
pub use conversion::*;
pub use context::*;
pub use recovery::*;
pub use validation::*;

/// Main error type for the validation system
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("I/O operation failed: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("JSON parsing failed: {0}")]
    Json(#[from] serde_json::Error),
    
    #[error("TOML parsing failed: {0}")]
    Toml(#[from] toml::de::Error),
    
    #[error("Search engine error: {0}")]
    SearchEngine(#[from] SearchEngineError),
    
    #[error("Validation failed: {0}")]
    ValidationFailed(#[from] ValidationFailure),
    
    #[error("Configuration error: {0}")]
    Configuration(#[from] ConfigurationError),
    
    #[error("Performance benchmark failed: {0}")]
    Performance(#[from] PerformanceError),
    
    #[error("Test data generation failed: {0}")]
    TestData(#[from] TestDataError),
    
    #[error("System resource error: {0}")]
    SystemResource(#[from] SystemResourceError),
    
    #[error("Windows-specific error: {0}")]
    WindowsSpecific(#[from] WindowsError),
    
    #[error("Unknown error: {0}")]
    Unknown(String),
}

/// Result type alias for validation operations
pub type ValidationResult<T> = Result<T, ValidationError>;

/// Context-aware error reporting trait
pub trait ErrorContext<T> {
    /// Add context to an error with additional information
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
    
    /// Add static context to an error
    fn context(self, msg: &'static str) -> Result<T>;
    
    /// Add validation context with test case information
    fn validation_context(self, test_id: &str, query: &str) -> Result<T>;
    
    /// Add performance context with timing information
    fn performance_context(self, operation: &str, duration_ms: u64) -> Result<T>;
}

impl<T, E> ErrorContext<T> for std::result::Result<T, E>
where
    E: Into<ValidationError>,
{
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| e.into()).with_context(f)
    }
    
    fn context(self, msg: &'static str) -> Result<T> {
        self.map_err(|e| e.into()).context(msg)
    }
    
    fn validation_context(self, test_id: &str, query: &str) -> Result<T> {
        self.with_context(|| {
            format!("Validation failed for test '{}' with query '{}'", test_id, query)
        })
    }
    
    fn performance_context(self, operation: &str, duration_ms: u64) -> Result<T> {
        self.with_context(|| {
            format!("Performance issue in '{}' (took {}ms)", operation, duration_ms)
        })
    }
}

/// Error recovery utilities
pub struct ErrorRecovery;

impl ErrorRecovery {
    /// Attempt to recover from a validation error
    pub fn recover_validation_error(error: &ValidationError) -> Option<RecoveryAction> {
        match error {
            ValidationError::Io(_) => Some(RecoveryAction::RetryWithBackoff),
            ValidationError::SearchEngine(SearchEngineError::IndexCorrupted) => {
                Some(RecoveryAction::RebuildIndex)
            }
            ValidationError::SystemResource(SystemResourceError::OutOfMemory) => {
                Some(RecoveryAction::ReduceMemoryUsage)
            }
            ValidationError::WindowsSpecific(WindowsError::PathTooLong) => {
                Some(RecoveryAction::ShortenPaths)
            }
            _ => None,
        }
    }
    
    /// Execute a recovery action
    pub async fn execute_recovery(action: RecoveryAction) -> Result<()> {
        match action {
            RecoveryAction::RetryWithBackoff => {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            RecoveryAction::RebuildIndex => {
                tracing::warn!("Attempting to rebuild corrupted index");
                // Index rebuilding logic would go here
            }
            RecoveryAction::ReduceMemoryUsage => {
                tracing::warn!("Attempting to reduce memory usage");
                // Memory cleanup logic would go here
            }
            RecoveryAction::ShortenPaths => {
                tracing::warn!("Attempting to shorten file paths");
                // Path shortening logic would go here
            }
        }
        Ok(())
    }
}

/// Recovery actions that can be taken for different error types
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    RetryWithBackoff,
    RebuildIndex,
    ReduceMemoryUsage,
    ShortenPaths,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorSeverity {
    /// Critical errors that prevent system operation
    Critical,
    /// High severity errors that may cause significant problems
    High,
    /// Medium severity errors that cause minor issues
    Medium,
    /// Low severity errors that are mostly informational
    Low,
}

impl ValidationError {
    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ValidationError::SystemResource(_) => ErrorSeverity::Critical,
            ValidationError::SearchEngine(SearchEngineError::IndexCorrupted) => ErrorSeverity::Critical,
            ValidationError::ValidationFailed(_) => ErrorSeverity::High,
            ValidationError::Performance(_) => ErrorSeverity::Medium,
            ValidationError::Configuration(_) => ErrorSeverity::Medium,
            ValidationError::WindowsSpecific(_) => ErrorSeverity::Low,
            _ => ErrorSeverity::Medium,
        }
    }
    
    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        ErrorRecovery::recover_validation_error(self).is_some()
    }
    
    /// Get detailed error information for debugging
    pub fn debug_info(&self) -> ErrorDebugInfo {
        ErrorDebugInfo {
            error_type: format!("{:?}", self),
            severity: self.severity(),
            recoverable: self.is_recoverable(),
            timestamp: chrono::Utc::now(),
            context: self.to_string(),
        }
    }
}

/// Detailed error information for debugging
#[derive(Debug, Clone)]
pub struct ErrorDebugInfo {
    pub error_type: String,
    pub severity: ErrorSeverity,
    pub recoverable: bool,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub context: String,
}

/// Macro for creating contextual errors
#[macro_export]
macro_rules! validation_error {
    ($msg:expr) => {
        anyhow::anyhow!($msg)
    };
    ($fmt:expr, $($arg:tt)*) => {
        anyhow::anyhow!($fmt, $($arg)*)
    };
}

/// Macro for adding validation context
#[macro_export]
macro_rules! validation_context {
    ($result:expr, $test_id:expr, $query:expr) => {
        $result.validation_context($test_id, $query)
    };
}

/// Macro for adding performance context
#[macro_export]
macro_rules! performance_context {
    ($result:expr, $operation:expr, $duration:expr) => {
        $result.performance_context($operation, $duration)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_severity() {
        let error = ValidationError::SystemResource(SystemResourceError::OutOfMemory);
        assert_eq!(error.severity(), ErrorSeverity::Critical);
        assert!(error.is_recoverable());
    }
    
    #[test]
    fn test_error_context() {
        let result: std::result::Result<(), std::io::Error> = Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "test error"
        ));
        
        let with_context = result.validation_context("test_001", "test query");
        assert!(with_context.is_err());
    }
    
    #[test]
    fn test_debug_info() {
        let error = ValidationError::Unknown("test error".to_string());
        let debug_info = error.debug_info();
        
        assert_eq!(debug_info.severity, ErrorSeverity::Medium);
        assert!(!debug_info.recoverable);
    }
    
    #[tokio::test]
    async fn test_error_recovery() {
        let action = RecoveryAction::RetryWithBackoff;
        let result = ErrorRecovery::execute_recovery(action).await;
        assert!(result.is_ok());
    }
}
```

### Error Types (`src/error/types.rs`)
```rust
use thiserror::Error;
use std::path::PathBuf;

/// Search engine specific errors
#[derive(Error, Debug)]
pub enum SearchEngineError {
    #[error("Index is corrupted and cannot be read")]
    IndexCorrupted,
    
    #[error("Index not found at path: {path}")]
    IndexNotFound { path: PathBuf },
    
    #[error("Failed to create index: {reason}")]
    IndexCreationFailed { reason: String },
    
    #[error("Query parsing failed: {query}")]
    QueryParsingFailed { query: String },
    
    #[error("Search operation timed out after {timeout_ms}ms")]
    SearchTimeout { timeout_ms: u64 },
    
    #[error("Tantivy error: {0}")]
    Tantivy(String),
    
    #[error("LanceDB error: {0}")]
    LanceDb(String),
    
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    VectorDimensionMismatch { expected: usize, actual: usize },
}

/// Validation specific errors
#[derive(Error, Debug)]
pub enum ValidationFailure {
    #[error("Accuracy requirement not met: {actual}% < {required}%")]
    AccuracyRequirementNotMet { actual: f64, required: f64 },
    
    #[error("Test case failed: {test_id} - {reason}")]
    TestCaseFailed { test_id: String, reason: String },
    
    #[error("Expected {expected} results, got {actual}")]
    ResultCountMismatch { expected: usize, actual: usize },
    
    #[error("Missing required content: {content}")]
    MissingRequiredContent { content: String },
    
    #[error("Found forbidden content: {content}")]
    ForbiddenContentFound { content: String },
    
    #[error("Ground truth data is invalid: {reason}")]
    InvalidGroundTruth { reason: String },
    
    #[error("Validation timeout after {timeout_ms}ms")]
    ValidationTimeout { timeout_ms: u64 },
}

/// Configuration related errors
#[derive(Error, Debug)]
pub enum ConfigurationError {
    #[error("Configuration file not found: {path}")]
    FileNotFound { path: PathBuf },
    
    #[error("Invalid configuration: {field} = {value}")]
    InvalidValue { field: String, value: String },
    
    #[error("Missing required configuration: {field}")]
    MissingRequired { field: String },
    
    #[error("Configuration validation failed: {reason}")]
    ValidationFailed { reason: String },
    
    #[error("Incompatible configuration version: {version}")]
    IncompatibleVersion { version: String },
}

/// Performance related errors
#[derive(Error, Debug)]
pub enum PerformanceError {
    #[error("Performance target not met: {metric} = {actual} (target: {target})")]
    TargetNotMet {
        metric: String,
        actual: f64,
        target: f64,
    },
    
    #[error("Benchmark failed: {benchmark} - {reason}")]
    BenchmarkFailed { benchmark: String, reason: String },
    
    #[error("Measurement error: {reason}")]
    MeasurementError { reason: String },
    
    #[error("Resource exhaustion: {resource}")]
    ResourceExhaustion { resource: String },
    
    #[error("Performance regression detected: {regression}%")]
    RegressionDetected { regression: f64 },
}

/// Test data generation errors
#[derive(Error, Debug)]
pub enum TestDataError {
    #[error("Failed to generate test data: {reason}")]
    GenerationFailed { reason: String },
    
    #[error("Invalid test data format: {format}")]
    InvalidFormat { format: String },
    
    #[error("Test data size limit exceeded: {size} > {limit}")]
    SizeLimitExceeded { size: u64, limit: u64 },
    
    #[error("Duplicate test case ID: {id}")]
    DuplicateId { id: String },
    
    #[error("Test data validation failed: {reason}")]
    ValidationFailed { reason: String },
}

/// System resource errors
#[derive(Error, Debug)]
pub enum SystemResourceError {
    #[error("Out of memory: requested {requested} bytes")]
    OutOfMemory { requested: u64 },
    
    #[error("Disk space exhausted: {available} bytes available")]
    DiskSpaceExhausted { available: u64 },
    
    #[error("Too many open files: {count} (limit: {limit})")]
    TooManyOpenFiles { count: usize, limit: usize },
    
    #[error("CPU usage too high: {usage}%")]
    HighCpuUsage { usage: f64 },
    
    #[error("Network operation failed: {reason}")]
    NetworkError { reason: String },
    
    #[error("Permission denied: {operation}")]
    PermissionDenied { operation: String },
}

/// Windows-specific errors
#[derive(Error, Debug)]
pub enum WindowsError {
    #[error("Path too long: {path} ({length} > 260 characters)")]
    PathTooLong { path: String, length: usize },
    
    #[error("Invalid Unicode in filename: {filename}")]
    InvalidUnicode { filename: String },
    
    #[error("Windows API error: {function} returned {code}")]
    ApiError { function: String, code: u32 },
    
    #[error("Registry operation failed: {operation}")]
    RegistryError { operation: String },
    
    #[error("Service operation failed: {service}")]
    ServiceError { service: String },
    
    #[error("Performance counter error: {counter}")]
    PerformanceCounterError { counter: String },
}

/// File system errors
#[derive(Error, Debug)]
pub enum FileSystemError {
    #[error("File not found: {path}")]
    FileNotFound { path: PathBuf },
    
    #[error("Directory not found: {path}")]
    DirectoryNotFound { path: PathBuf },
    
    #[error("Access denied: {path}")]
    AccessDenied { path: PathBuf },
    
    #[error("File already exists: {path}")]
    FileAlreadyExists { path: PathBuf },
    
    #[error("Invalid path: {path}")]
    InvalidPath { path: String },
    
    #[error("File system full")]
    FileSystemFull,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_search_engine_error_display() {
        let error = SearchEngineError::IndexNotFound {
            path: PathBuf::from("/test/path"),
        };
        assert!(error.to_string().contains("/test/path"));
    }
    
    #[test]
    fn test_validation_failure_display() {
        let error = ValidationFailure::AccuracyRequirementNotMet {
            actual: 95.5,
            required: 100.0,
        };
        assert!(error.to_string().contains("95.5%"));
        assert!(error.to_string().contains("100%"));
    }
    
    #[test]
    fn test_windows_error_display() {
        let error = WindowsError::PathTooLong {
            path: "very/long/path".to_string(),
            length: 300,
        };
        assert!(error.to_string().contains("300"));
        assert!(error.to_string().contains("260"));
    }
}
```

### Error Conversion Utilities (`src/error/conversion.rs`)
```rust
use crate::error::types::*;
use crate::error::ValidationError;

/// Convert Tantivy errors to our error types
impl From<tantivy::TantivyError> for ValidationError {
    fn from(error: tantivy::TantivyError) -> Self {
        let search_error = match error {
            tantivy::TantivyError::IndexAlreadyExists(_) => {
                SearchEngineError::IndexCreationFailed {
                    reason: "Index already exists".to_string(),
                }
            }
            tantivy::TantivyError::PathDoesNotExist(path) => {
                SearchEngineError::IndexNotFound { path }
            }
            tantivy::TantivyError::FileCorrupted(_) => {
                SearchEngineError::IndexCorrupted
            }
            _ => SearchEngineError::Tantivy(error.to_string()),
        };
        
        ValidationError::SearchEngine(search_error)
    }
}

/// Convert tokio errors to our error types
impl From<tokio::time::error::Elapsed> for ValidationError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        ValidationError::Performance(PerformanceError::BenchmarkFailed {
            benchmark: "timeout".to_string(),
            reason: "Operation timed out".to_string(),
        })
    }
}

/// Convert system time errors
impl From<std::time::SystemTimeError> for ValidationError {
    fn from(error: std::time::SystemTimeError) -> Self {
        ValidationError::Performance(PerformanceError::MeasurementError {
            reason: format!("System time error: {}", error),
        })
    }
}

/// Convert regex errors
impl From<regex::Error> for ValidationError {
    fn from(error: regex::Error) -> Self {
        ValidationError::TestData(TestDataError::GenerationFailed {
            reason: format!("Regex error: {}", error),
        })
    }
}

/// Convert Windows API errors
#[cfg(windows)]
impl From<windows::core::Error> for ValidationError {
    fn from(error: windows::core::Error) -> Self {
        ValidationError::WindowsSpecific(WindowsError::ApiError {
            function: "unknown".to_string(),
            code: error.code().0 as u32,
        })
    }
}

/// Utility functions for error conversion
pub struct ErrorConverter;

impl ErrorConverter {
    /// Convert any error to a validation error with context
    pub fn to_validation_error<E: std::fmt::Display>(
        error: E,
        context: &str,
    ) -> ValidationError {
        ValidationError::Unknown(format!("{}: {}", context, error))
    }
    
    /// Convert I/O errors with file context
    pub fn io_error_with_path(error: std::io::Error, path: &std::path::Path) -> ValidationError {
        match error.kind() {
            std::io::ErrorKind::NotFound => {
                ValidationError::from(FileSystemError::FileNotFound {
                    path: path.to_path_buf(),
                })
            }
            std::io::ErrorKind::PermissionDenied => {
                ValidationError::from(FileSystemError::AccessDenied {
                    path: path.to_path_buf(),
                })
            }
            std::io::ErrorKind::AlreadyExists => {
                ValidationError::from(FileSystemError::FileAlreadyExists {
                    path: path.to_path_buf(),
                })
            }
            _ => ValidationError::Io(error),
        }
    }
    
    /// Convert performance measurement errors
    pub fn performance_error(
        metric: &str,
        actual: f64,
        target: f64,
    ) -> ValidationError {
        ValidationError::Performance(PerformanceError::TargetNotMet {
            metric: metric.to_string(),
            actual,
            target,
        })
    }
    
    /// Convert validation test failures
    pub fn test_failure(test_id: &str, reason: &str) -> ValidationError {
        ValidationError::ValidationFailed(ValidationFailure::TestCaseFailed {
            test_id: test_id.to_string(),
            reason: reason.to_string(),
        })
    }
}

/// Trait for adding conversion context
pub trait ErrorConversion<T> {
    /// Convert to validation error with context
    fn to_validation_error(self, context: &str) -> Result<T, ValidationError>;
    
    /// Convert I/O error with file path context
    fn with_path_context(self, path: &std::path::Path) -> Result<T, ValidationError>;
    
    /// Convert to test failure
    fn as_test_failure(self, test_id: &str) -> Result<T, ValidationError>;
}

impl<T, E> ErrorConversion<T> for Result<T, E>
where
    E: std::fmt::Display,
{
    fn to_validation_error(self, context: &str) -> Result<T, ValidationError> {
        self.map_err(|e| ErrorConverter::to_validation_error(e, context))
    }
    
    fn with_path_context(self, path: &std::path::Path) -> Result<T, ValidationError> {
        self.map_err(|e| {
            if let Ok(io_error) = format!("{}", e).parse::<std::io::Error>() {
                ErrorConverter::io_error_with_path(io_error, path)
            } else {
                ErrorConverter::to_validation_error(e, &format!("Path: {}", path.display()))
            }
        })
    }
    
    fn as_test_failure(self, test_id: &str) -> Result<T, ValidationError> {
        self.map_err(|e| ErrorConverter::test_failure(test_id, &e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_io_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "test");
        let path = PathBuf::from("/test/path");
        let validation_error = ErrorConverter::io_error_with_path(io_error, &path);
        
        match validation_error {
            ValidationError::from(FileSystemError::FileNotFound { path: error_path }) => {
                assert_eq!(error_path, path);
            }
            _ => panic!("Expected FileNotFound error"),
        }
    }
    
    #[test]
    fn test_performance_error_conversion() {
        let error = ErrorConverter::performance_error("latency", 150.0, 100.0);
        
        match error {
            ValidationError::Performance(PerformanceError::TargetNotMet {
                metric,
                actual,
                target,
            }) => {
                assert_eq!(metric, "latency");
                assert_eq!(actual, 150.0);
                assert_eq!(target, 100.0);
            }
            _ => panic!("Expected TargetNotMet error"),
        }
    }
    
    #[test]
    fn test_error_conversion_trait() {
        let result: Result<i32, &str> = Err("test error");
        let converted = result.to_validation_error("test context");
        
        assert!(converted.is_err());
        assert!(converted.unwrap_err().to_string().contains("test context"));
        assert!(converted.unwrap_err().to_string().contains("test error"));
    }
}
```

### Contextual Error Reporting (`src/error/context.rs`)
```rust
use crate::error::{ValidationError, ErrorSeverity};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Contextual error information for detailed reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Unique error correlation ID
    pub correlation_id: String,
    
    /// Timestamp when error occurred
    pub timestamp: DateTime<Utc>,
    
    /// Error severity level
    pub severity: ErrorSeverity,
    
    /// Operation that was being performed
    pub operation: String,
    
    /// Test case ID if applicable
    pub test_id: Option<String>,
    
    /// Query that caused the error if applicable
    pub query: Option<String>,
    
    /// File path where error occurred if applicable
    pub file_path: Option<String>,
    
    /// Additional context data
    pub metadata: HashMap<String, String>,
    
    /// Stack trace or call chain
    pub stack_trace: Option<String>,
    
    /// Recovery suggestions
    pub recovery_suggestions: Vec<String>,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            correlation_id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            severity: ErrorSeverity::Medium,
            operation: operation.into(),
            test_id: None,
            query: None,
            file_path: None,
            metadata: HashMap::new(),
            stack_trace: None,
            recovery_suggestions: Vec::new(),
        }
    }
    
    /// Set the severity level
    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }
    
    /// Add test case context
    pub fn with_test_case(mut self, test_id: impl Into<String>) -> Self {
        self.test_id = Some(test_id.into());
        self
    }
    
    /// Add query context
    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }
    
    /// Add file path context
    pub fn with_file_path(mut self, path: impl Into<String>) -> Self {
        self.file_path = Some(path.into());
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// Add recovery suggestion
    pub fn with_recovery_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.recovery_suggestions.push(suggestion.into());
        self
    }
    
    /// Capture stack trace
    pub fn with_stack_trace(mut self) -> Self {
        // In a real implementation, you might use backtrace crate
        self.stack_trace = Some("Stack trace not implemented".to_string());
        self
    }
    
    /// Generate a detailed error report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str(&format!("Error Report\n"));
        report.push_str(&format!("============\n"));
        report.push_str(&format!("Correlation ID: {}\n", self.correlation_id));
        report.push_str(&format!("Timestamp: {}\n", self.timestamp.format("%Y-%m-%d %H:%M:%S UTC")));
        report.push_str(&format!("Severity: {:?}\n", self.severity));
        report.push_str(&format!("Operation: {}\n", self.operation));
        
        if let Some(test_id) = &self.test_id {
            report.push_str(&format!("Test Case: {}\n", test_id));
        }
        
        if let Some(query) = &self.query {
            report.push_str(&format!("Query: {}\n", query));
        }
        
        if let Some(file_path) = &self.file_path {
            report.push_str(&format!("File Path: {}\n", file_path));
        }
        
        if !self.metadata.is_empty() {
            report.push_str("\nMetadata:\n");
            for (key, value) in &self.metadata {
                report.push_str(&format!("  {}: {}\n", key, value));
            }
        }
        
        if !self.recovery_suggestions.is_empty() {
            report.push_str("\nRecovery Suggestions:\n");
            for suggestion in &self.recovery_suggestions {
                report.push_str(&format!("  - {}\n", suggestion));
            }
        }
        
        if let Some(stack_trace) = &self.stack_trace {
            report.push_str(&format!("\nStack Trace:\n{}\n", stack_trace));
        }
        
        report
    }
}

/// Error reporter for collecting and managing error contexts
pub struct ErrorReporter {
    contexts: Vec<ErrorContext>,
}

impl ErrorReporter {
    /// Create a new error reporter
    pub fn new() -> Self {
        Self {
            contexts: Vec::new(),
        }
    }
    
    /// Report an error with context
    pub fn report_error(&mut self, error: &ValidationError, context: ErrorContext) {
        let mut context = context.with_severity(error.severity());
        
        // Add error-specific recovery suggestions
        match error {
            ValidationError::SearchEngine(_) => {
                context = context.with_recovery_suggestion("Check index integrity and rebuild if necessary");
            }
            ValidationError::SystemResource(_) => {
                context = context.with_recovery_suggestion("Free up system resources and retry");
            }
            ValidationError::WindowsSpecific(_) => {
                context = context.with_recovery_suggestion("Check Windows-specific configuration");
            }
            _ => {}
        }
        
        self.contexts.push(context);
        
        // Log the error
        match error.severity() {
            ErrorSeverity::Critical => {
                tracing::error!(
                    correlation_id = %context.correlation_id,
                    error = %error,
                    "Critical error reported"
                );
            }
            ErrorSeverity::High => {
                tracing::error!(
                    correlation_id = %context.correlation_id,
                    error = %error,
                    "High severity error reported"
                );
            }
            ErrorSeverity::Medium => {
                tracing::warn!(
                    correlation_id = %context.correlation_id,
                    error = %error,
                    "Medium severity error reported"
                );
            }
            ErrorSeverity::Low => {
                tracing::info!(
                    correlation_id = %context.correlation_id,
                    error = %error,
                    "Low severity error reported"
                );
            }
        }
    }
    
    /// Get all error contexts
    pub fn get_contexts(&self) -> &[ErrorContext] {
        &self.contexts
    }
    
    /// Get errors by severity
    pub fn get_errors_by_severity(&self, severity: ErrorSeverity) -> Vec<&ErrorContext> {
        self.contexts
            .iter()
            .filter(|ctx| ctx.severity == severity)
            .collect()
    }
    
    /// Generate a summary report
    pub fn generate_summary_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("Error Summary Report\n");
        report.push_str("===================\n\n");
        
        let total_errors = self.contexts.len();
        let critical_count = self.get_errors_by_severity(ErrorSeverity::Critical).len();
        let high_count = self.get_errors_by_severity(ErrorSeverity::High).len();
        let medium_count = self.get_errors_by_severity(ErrorSeverity::Medium).len();
        let low_count = self.get_errors_by_severity(ErrorSeverity::Low).len();
        
        report.push_str(&format!("Total Errors: {}\n", total_errors));
        report.push_str(&format!("Critical: {}\n", critical_count));
        report.push_str(&format!("High: {}\n", high_count));
        report.push_str(&format!("Medium: {}\n", medium_count));
        report.push_str(&format!("Low: {}\n\n", low_count));
        
        // Add details for critical and high severity errors
        for context in self.get_errors_by_severity(ErrorSeverity::Critical) {
            report.push_str(&format!("CRITICAL - {}: {}\n", 
                context.correlation_id, context.operation));
        }
        
        for context in self.get_errors_by_severity(ErrorSeverity::High) {
            report.push_str(&format!("HIGH - {}: {}\n", 
                context.correlation_id, context.operation));
        }
        
        report
    }
    
    /// Clear all error contexts
    pub fn clear(&mut self) {
        self.contexts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::types::ValidationFailure;
    
    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext::new("test_operation")
            .with_severity(ErrorSeverity::High)
            .with_test_case("test_001")
            .with_query("test query")
            .with_metadata("key", "value")
            .with_recovery_suggestion("Try again");
        
        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.severity, ErrorSeverity::High);
        assert_eq!(context.test_id, Some("test_001".to_string()));
        assert_eq!(context.query, Some("test query".to_string()));
        assert_eq!(context.metadata.get("key"), Some(&"value".to_string()));
        assert_eq!(context.recovery_suggestions.len(), 1);
    }
    
    #[test]
    fn test_error_reporter() {
        let mut reporter = ErrorReporter::new();
        let error = ValidationError::ValidationFailed(ValidationFailure::TestCaseFailed {
            test_id: "test_001".to_string(),
            reason: "Failed assertion".to_string(),
        });
        let context = ErrorContext::new("validation_test");
        
        reporter.report_error(&error, context);
        
        assert_eq!(reporter.get_contexts().len(), 1);
        assert_eq!(reporter.get_errors_by_severity(ErrorSeverity::High).len(), 1);
    }
    
    #[test]
    fn test_report_generation() {
        let context = ErrorContext::new("test_operation")
            .with_test_case("test_001")
            .with_recovery_suggestion("Try again");
        
        let report = context.generate_report();
        
        assert!(report.contains("Error Report"));
        assert!(report.contains("test_operation"));
        assert!(report.contains("test_001"));
        assert!(report.contains("Try again"));
    }
}
```

## Success Criteria
- Error types cover all major failure modes in the validation system
- Error conversion utilities handle third-party library errors correctly
- Contextual error reporting provides detailed diagnostic information
- Recovery strategies are implemented for common error scenarios
- Integration with anyhow provides ergonomic error handling
- Windows-specific errors are properly handled
- Error correlation IDs enable tracking across system components
- Error severity levels help prioritize issue resolution

## Time Limit
10 minutes maximum