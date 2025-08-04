# Task 015: Comprehensive Validation Error Handling and Recovery

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 005-014, specifically extending the CorrectnessValidator with robust error handling, categorization, recovery mechanisms, and structured reporting with actionable suggestions.

## Project Structure
```
src/
  validation/
    correctness.rs  <- Extend this file
  lib.rs
```

## Task Description
Create the `ValidationErrorHandler` that provides comprehensive error categorization, recovery strategies, structured reporting with actionable suggestions, error aggregation across batch validation, and retry logic with circuit breaker patterns.

## Requirements
1. Add to existing `src/validation/correctness.rs`
2. Implement `ValidationErrorHandler` with error categorization
3. Add structured error reporting with actionable suggestions
4. Include error aggregation and analysis across batch operations
5. Support retry logic with exponential backoff and circuit breakers
6. Add logging integration with structured output
7. Provide recovery strategies for different error types

## Expected Code Structure to Add
```rust
use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorCategory {
    SystemError,      // Infrastructure failures
    DataError,        // Invalid or corrupt data
    ConfigError,      // Configuration issues
    NetworkError,     // Connection problems
    PerformanceError, // Timeout or resource issues
    ValidationError,  // Business logic validation failures
    RetryableError,   // Temporary errors that can be retried
    FatalError,       // Non-recoverable errors
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub id: String,
    pub category: ErrorCategory,
    pub severity: ErrorSeverity,
    pub message: String,
    pub details: String,
    pub timestamp: std::time::SystemTime,
    pub context: ErrorContext,
    pub suggestions: Vec<ActionableSuggestion>,
    pub retry_count: usize,
    pub is_recoverable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub query: Option<String>,
    pub operation: String,
    pub component: String,
    pub file_path: Option<String>,
    pub line_number: Option<usize>,
    pub stack_trace: Option<String>,
    pub system_info: SystemInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub available_memory: u64,
    pub cpu_usage: f64,
    pub disk_space: u64,
    pub network_status: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionableSuggestion {
    pub action: String,
    pub description: String,
    pub priority: SuggestionPriority,
    pub estimated_effort: EffortLevel,
    pub success_probability: f64,
    pub related_docs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionPriority {
    Immediate,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffortLevel {
    Trivial,   // < 5 minutes
    Quick,     // 5-30 minutes
    Moderate,  // 30 minutes - 2 hours
    Significant, // 2-8 hours
    Major,     // > 8 hours
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAggregation {
    pub total_errors: usize,
    pub by_category: HashMap<ErrorCategory, usize>,
    pub by_severity: HashMap<ErrorSeverity, usize>,
    pub most_common_errors: Vec<(String, usize)>,
    pub error_rate: f64,
    pub recovery_rate: f64,
    pub time_range: (std::time::SystemTime, std::time::SystemTime),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub base_delay: Duration,
    pub max_delay: Duration,
    pub backoff_multiplier: f64,
    pub jitter: bool,
    pub retryable_categories: Vec<ErrorCategory>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerState {
    pub is_open: bool,
    pub failure_count: usize,
    pub success_count: usize,
    pub last_failure_time: Option<Instant>,
    pub failure_threshold: usize,
    pub recovery_timeout: Duration,
    pub half_open_max_calls: usize,
}

pub struct ValidationErrorHandler {
    errors: Vec<ValidationError>,
    retry_policy: RetryPolicy,
    circuit_breaker: CircuitBreakerState,
    error_patterns: HashMap<String, ErrorCategory>,
    logger: Box<dyn ErrorLogger>,
    metrics_collector: Box<dyn MetricsCollector>,
}

pub trait ErrorLogger {
    fn log_error(&self, error: &ValidationError);
    fn log_recovery(&self, error_id: &str, strategy: &str, success: bool);
    fn log_aggregation(&self, aggregation: &ErrorAggregation);
}

pub trait MetricsCollector {
    fn increment_error_count(&self, category: &ErrorCategory);
    fn record_recovery_time(&self, duration: Duration);
    fn record_retry_attempt(&self, error_id: &str, attempt: usize);
}

struct DefaultLogger;
struct DefaultMetricsCollector;

impl ErrorLogger for DefaultLogger {
    fn log_error(&self, error: &ValidationError) {
        eprintln!("[ERROR] {}: {} ({})", error.id, error.message, error.category);
    }
    
    fn log_recovery(&self, error_id: &str, strategy: &str, success: bool) {
        eprintln!("[RECOVERY] {} - Strategy: {} - Success: {}", error_id, strategy, success);
    }
    
    fn log_aggregation(&self, aggregation: &ErrorAggregation) {
        eprintln!("[SUMMARY] Total errors: {}, Error rate: {:.2}%", 
                 aggregation.total_errors, aggregation.error_rate * 100.0);
    }
}

impl MetricsCollector for DefaultMetricsCollector {
    fn increment_error_count(&self, _category: &ErrorCategory) {
        // In a real implementation, this would send metrics to a monitoring system
    }
    
    fn record_recovery_time(&self, _duration: Duration) {
        // Record recovery time metrics
    }
    
    fn record_retry_attempt(&self, _error_id: &str, _attempt: usize) {
        // Record retry attempt metrics
    }
}

impl ValidationErrorHandler {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            retry_policy: RetryPolicy::default(),
            circuit_breaker: CircuitBreakerState::default(),
            error_patterns: Self::default_error_patterns(),
            logger: Box::new(DefaultLogger),
            metrics_collector: Box::new(DefaultMetricsCollector),
        }
    }
    
    pub fn with_retry_policy(mut self, policy: RetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }
    
    pub fn handle_error(&mut self, error: ValidationError) -> ErrorHandlingResult {
        let error_id = error.id.clone();
        self.metrics_collector.increment_error_count(&error.category);
        self.logger.log_error(&error);
        
        // Check circuit breaker
        if self.circuit_breaker.is_open {
            if self.should_attempt_recovery() {
                self.circuit_breaker.is_open = false;
                self.circuit_breaker.success_count = 0;
            } else {
                return ErrorHandlingResult::CircuitBreakerOpen {
                    retry_after: self.circuit_breaker.recovery_timeout,
                };
            }
        }
        
        // Determine if error is retryable
        let is_retryable = self.is_retryable(&error);
        let recovery_strategy = self.determine_recovery_strategy(&error);
        
        // Store error for aggregation
        self.errors.push(error.clone());
        
        // Attempt recovery if possible
        if is_retryable && error.retry_count < self.retry_policy.max_attempts {
            let recovery_result = self.attempt_recovery(&error, &recovery_strategy);
            match recovery_result {
                RecoveryResult::Success => {
                    self.circuit_breaker.success_count += 1;
                    self.circuit_breaker.failure_count = 0;
                    self.logger.log_recovery(&error_id, &recovery_strategy.name, true);
                    return ErrorHandlingResult::Recovered {
                        strategy: recovery_strategy,
                        attempts: error.retry_count + 1,
                    };
                },
                RecoveryResult::Retry { delay } => {
                    self.metrics_collector.record_retry_attempt(&error_id, error.retry_count + 1);
                    return ErrorHandlingResult::WillRetry {
                        delay,
                        attempt: error.retry_count + 1,
                        max_attempts: self.retry_policy.max_attempts,
                    };
                },
                RecoveryResult::Failed => {
                    self.circuit_breaker.failure_count += 1;
                    if self.circuit_breaker.failure_count >= self.circuit_breaker.failure_threshold {
                        self.circuit_breaker.is_open = true;
                        self.circuit_breaker.last_failure_time = Some(Instant::now());
                    }
                    self.logger.log_recovery(&error_id, &recovery_strategy.name, false);
                },
            }
        }
        
        ErrorHandlingResult::Failed {
            error,
            suggestions: self.generate_suggestions(&error),
            is_recoverable: is_retryable,
        }
    }
    
    fn is_retryable(&self, error: &ValidationError) -> bool {
        if !error.is_recoverable {
            return false;
        }
        
        self.retry_policy.retryable_categories.contains(&error.category) &&
        error.retry_count < self.retry_policy.max_attempts
    }
    
    fn determine_recovery_strategy(&self, error: &ValidationError) -> RecoveryStrategy {
        match error.category {
            ErrorCategory::NetworkError => RecoveryStrategy {
                name: "network_retry".to_string(),
                description: "Retry with exponential backoff".to_string(),
                actions: vec![
                    "Check network connectivity".to_string(),
                    "Retry request with backoff".to_string(),
                    "Switch to backup endpoint if available".to_string(),
                ],
            },
            ErrorCategory::PerformanceError => RecoveryStrategy {
                name: "performance_optimization".to_string(),
                description: "Reduce load and retry".to_string(),
                actions: vec![
                    "Reduce batch size".to_string(),
                    "Add delay between operations".to_string(),
                    "Check system resources".to_string(),
                ],
            },
            ErrorCategory::DataError => RecoveryStrategy {
                name: "data_validation".to_string(),
                description: "Validate and clean data".to_string(),
                actions: vec![
                    "Validate input data format".to_string(),
                    "Clean or sanitize problematic data".to_string(),
                    "Skip invalid records if allowed".to_string(),
                ],
            },
            ErrorCategory::ConfigError => RecoveryStrategy {
                name: "config_correction".to_string(),
                description: "Fix configuration issues".to_string(),
                actions: vec![
                    "Validate configuration syntax".to_string(),
                    "Check required settings".to_string(),
                    "Reload configuration".to_string(),
                ],
            },
            _ => RecoveryStrategy {
                name: "generic_retry".to_string(),
                description: "Generic retry strategy".to_string(),
                actions: vec![
                    "Wait and retry".to_string(),
                    "Log detailed error information".to_string(),
                ],
            },
        }
    }
    
    fn attempt_recovery(&self, error: &ValidationError, strategy: &RecoveryStrategy) -> RecoveryResult {
        // Calculate retry delay with exponential backoff
        let delay = self.calculate_retry_delay(error.retry_count);
        
        // Simulate recovery attempt (in real implementation, this would perform actual recovery)
        match error.category {
            ErrorCategory::NetworkError => {
                // Simulate network recovery success rate
                if error.retry_count < 3 && rand::random::<f64>() > 0.3 {
                    RecoveryResult::Success
                } else {
                    RecoveryResult::Retry { delay }
                }
            },
            ErrorCategory::PerformanceError => {
                // Performance errors often resolve with delay
                if error.retry_count < 2 {
                    RecoveryResult::Retry { delay: delay * 2 }
                } else {
                    RecoveryResult::Failed
                }
            },
            ErrorCategory::DataError => {
                // Data errors typically don't resolve without intervention
                RecoveryResult::Failed
            },
            ErrorCategory::ConfigError => {
                // Config errors need manual intervention
                RecoveryResult::Failed
            },
            ErrorCategory::RetryableError => {
                if error.retry_count < self.retry_policy.max_attempts - 1 {
                    RecoveryResult::Retry { delay }
                } else {
                    RecoveryResult::Failed
                }
            },
            _ => RecoveryResult::Failed,
        }
    }
    
    fn calculate_retry_delay(&self, attempt: usize) -> Duration {
        let base_delay = self.retry_policy.base_delay.as_millis() as f64;
        let multiplier = self.retry_policy.backoff_multiplier.powi(attempt as i32);
        let delay = base_delay * multiplier;
        
        let delay = delay.min(self.retry_policy.max_delay.as_millis() as f64);
        
        let delay = if self.retry_policy.jitter {
            let jitter = rand::random::<f64>() * 0.1 + 0.95; // 5% jitter
            delay * jitter
        } else {
            delay
        };
        
        Duration::from_millis(delay as u64)
    }
    
    fn should_attempt_recovery(&self) -> bool {
        if let Some(last_failure) = self.circuit_breaker.last_failure_time {
            Instant::now().duration_since(last_failure) > self.circuit_breaker.recovery_timeout
        } else {
            true
        }
    }
    
    fn generate_suggestions(&self, error: &ValidationError) -> Vec<ActionableSuggestion> {
        let mut suggestions = error.suggestions.clone();
        
        // Add category-specific suggestions
        match error.category {
            ErrorCategory::SystemError => {
                suggestions.push(ActionableSuggestion {
                    action: "Check system resources".to_string(),
                    description: "Verify CPU, memory, and disk space availability".to_string(),
                    priority: SuggestionPriority::High,
                    estimated_effort: EffortLevel::Quick,
                    success_probability: 0.7,
                    related_docs: vec!["system-requirements.md".to_string()],
                });
            },
            ErrorCategory::NetworkError => {
                suggestions.push(ActionableSuggestion {
                    action: "Verify network connectivity".to_string(),
                    description: "Check internet connection and firewall settings".to_string(),
                    priority: SuggestionPriority::Immediate,
                    estimated_effort: EffortLevel::Quick,
                    success_probability: 0.8,
                    related_docs: vec!["network-troubleshooting.md".to_string()],
                });
            },
            ErrorCategory::ConfigError => {
                suggestions.push(ActionableSuggestion {
                    action: "Review configuration file".to_string(),
                    description: "Check configuration syntax and required parameters".to_string(),
                    priority: SuggestionPriority::High,
                    estimated_effort: EffortLevel::Moderate,
                    success_probability: 0.9,
                    related_docs: vec!["configuration-guide.md".to_string()],
                });
            },
            _ => {},
        }
        
        suggestions
    }
    
    pub fn aggregate_errors(&self, time_window: Duration) -> ErrorAggregation {
        let now = std::time::SystemTime::now();
        let cutoff = now - time_window;
        
        let recent_errors: Vec<&ValidationError> = self.errors.iter()
            .filter(|e| e.timestamp > cutoff)
            .collect();
        
        let total_errors = recent_errors.len();
        let mut by_category = HashMap::new();
        let mut by_severity = HashMap::new();
        let mut error_messages = HashMap::new();
        
        for error in &recent_errors {
            *by_category.entry(error.category.clone()).or_insert(0) += 1;
            *by_severity.entry(error.severity.clone()).or_insert(0) += 1;
            *error_messages.entry(error.message.clone()).or_insert(0) += 1;
        }
        
        let mut most_common_errors: Vec<(String, usize)> = error_messages.into_iter().collect();
        most_common_errors.sort_by(|a, b| b.1.cmp(&a.1));
        most_common_errors.truncate(10);
        
        let recovered_errors = recent_errors.iter()
            .filter(|e| e.retry_count > 0)
            .count();
        
        let recovery_rate = if total_errors > 0 {
            recovered_errors as f64 / total_errors as f64
        } else {
            0.0
        };
        
        let error_rate = total_errors as f64 / time_window.as_secs() as f64;
        
        let time_range = if recent_errors.is_empty() {
            (now, now)
        } else {
            let earliest = recent_errors.iter()
                .map(|e| e.timestamp)
                .min()
                .unwrap_or(now);
            (earliest, now)
        };
        
        ErrorAggregation {
            total_errors,
            by_category,
            by_severity,
            most_common_errors,
            error_rate,
            recovery_rate,
            time_range,
        }
    }
    
    pub fn clear_old_errors(&mut self, older_than: Duration) {
        let cutoff = std::time::SystemTime::now() - older_than;
        self.errors.retain(|error| error.timestamp > cutoff);
    }
    
    fn default_error_patterns() -> HashMap<String, ErrorCategory> {
        let mut patterns = HashMap::new();
        patterns.insert("connection refused".to_string(), ErrorCategory::NetworkError);
        patterns.insert("timeout".to_string(), ErrorCategory::PerformanceError);
        patterns.insert("out of memory".to_string(), ErrorCategory::SystemError);
        patterns.insert("invalid json".to_string(), ErrorCategory::DataError);
        patterns.insert("missing config".to_string(), ErrorCategory::ConfigError);
        patterns.insert("validation failed".to_string(), ErrorCategory::ValidationError);
        patterns
    }
}

#[derive(Debug, Clone)]
pub struct RecoveryStrategy {
    pub name: String,
    pub description: String,
    pub actions: Vec<String>,
}

#[derive(Debug)]
pub enum RecoveryResult {
    Success,
    Retry { delay: Duration },
    Failed,
}

#[derive(Debug)]
pub enum ErrorHandlingResult {
    Recovered {
        strategy: RecoveryStrategy,
        attempts: usize,
    },
    WillRetry {
        delay: Duration,
        attempt: usize,
        max_attempts: usize,
    },
    Failed {
        error: ValidationError,
        suggestions: Vec<ActionableSuggestion>,
        is_recoverable: bool,
    },
    CircuitBreakerOpen {
        retry_after: Duration,
    },
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
            retryable_categories: vec![
                ErrorCategory::NetworkError,
                ErrorCategory::PerformanceError,
                ErrorCategory::RetryableError,
            ],
        }
    }
}

impl Default for CircuitBreakerState {
    fn default() -> Self {
        Self {
            is_open: false,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            failure_threshold: 5,
            recovery_timeout: Duration::from_secs(60),
            half_open_max_calls: 3,
        }
    }
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::SystemError => write!(f, "System Error"),
            ErrorCategory::DataError => write!(f, "Data Error"),
            ErrorCategory::ConfigError => write!(f, "Configuration Error"),
            ErrorCategory::NetworkError => write!(f, "Network Error"),
            ErrorCategory::PerformanceError => write!(f, "Performance Error"),
            ErrorCategory::ValidationError => write!(f, "Validation Error"),
            ErrorCategory::RetryableError => write!(f, "Retryable Error"),
            ErrorCategory::FatalError => write!(f, "Fatal Error"),
        }
    }
}

impl Default for ValidationErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}
```

## Success Criteria
- ValidationErrorHandler struct compiles without errors
- Error categorization accurately classifies different error types
- Recovery strategies are appropriate for each error category
- Retry logic with exponential backoff works correctly
- Circuit breaker pattern prevents cascading failures
- Error aggregation provides meaningful insights
- Actionable suggestions are relevant and helpful
- Logging integration provides structured output
- Performance impact is minimal even with comprehensive error handling

## Time Limit
10 minutes maximum