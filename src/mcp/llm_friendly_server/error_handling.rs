//! Comprehensive error handling for production-ready LLMKG system

use serde_json::{json, Value};
use std::fmt;

/// Production-ready error types for the LLMKG system
#[derive(Debug, Clone)]
pub enum LlmkgError {
    /// Input validation errors
    ValidationError {
        field: String,
        message: String,
        received_value: Option<String>,
    },
    /// Engine operation errors
    EngineError {
        operation: String,
        cause: String,
        recoverable: bool,
    },
    /// Entity extraction errors
    ExtractionError {
        extraction_type: String,
        input_sample: String,
        cause: String,
    },
    /// Query processing errors
    QueryError {
        query_type: String,
        parameters: Value,
        cause: String,
    },
    /// Storage operation errors
    StorageError {
        operation: String,
        entity_id: Option<String>,
        cause: String,
    },
    /// Authentication/authorization errors
    AuthError {
        operation: String,
        message: String,
    },
    /// Rate limiting errors
    RateLimitError {
        operation: String,
        retry_after_seconds: u64,
    },
    /// Resource limit errors
    ResourceError {
        resource_type: String,
        limit: u64,
        current: u64,
    },
    /// Network/external service errors
    ExternalError {
        service: String,
        operation: String,
        retry_count: u32,
        cause: String,
    },
    /// Internal system errors
    InternalError {
        component: String,
        message: String,
    },
}

impl fmt::Display for LlmkgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmkgError::ValidationError { field, message, received_value } => {
                write!(f, "Validation error in field '{}': {}. Received: {:?}", 
                    field, message, received_value)
            }
            LlmkgError::EngineError { operation, cause, recoverable } => {
                write!(f, "Engine error during '{}': {}. Recoverable: {}", 
                    operation, cause, recoverable)
            }
            LlmkgError::ExtractionError { extraction_type, input_sample, cause } => {
                write!(f, "{} extraction failed for input '{}': {}", 
                    extraction_type, 
                    if input_sample.len() > 50 {
                        format!("{}...", &input_sample[..50])
                    } else {
                        input_sample.clone()
                    },
                    cause)
            }
            LlmkgError::QueryError { query_type, parameters, cause } => {
                write!(f, "Query error for type '{}' with params {}: {}", 
                    query_type, parameters, cause)
            }
            LlmkgError::StorageError { operation, entity_id, cause } => {
                write!(f, "Storage error during '{}' for entity {:?}: {}", 
                    operation, entity_id, cause)
            }
            LlmkgError::AuthError { operation, message } => {
                write!(f, "Authentication error during '{}': {}", operation, message)
            }
            LlmkgError::RateLimitError { operation, retry_after_seconds } => {
                write!(f, "Rate limit exceeded for '{}'. Retry after {} seconds", 
                    operation, retry_after_seconds)
            }
            LlmkgError::ResourceError { resource_type, limit, current } => {
                write!(f, "{} limit exceeded: {}/{}", resource_type, current, limit)
            }
            LlmkgError::ExternalError { service, operation, retry_count, cause } => {
                write!(f, "External service '{}' error during '{}' (attempt {}): {}", 
                    service, operation, retry_count, cause)
            }
            LlmkgError::InternalError { component, message } => {
                write!(f, "Internal error in '{}': {}", component, message)
            }
        }
    }
}

impl std::error::Error for LlmkgError {}

/// Result type alias for LLMKG operations
pub type LlmkgResult<T> = Result<T, LlmkgError>;

/// Handler result type for MCP operations
pub type HandlerResult = std::result::Result<(Value, String, Vec<String>), LlmkgError>;

impl LlmkgError {
    /// Check if the error is recoverable and operation should be retried
    pub fn is_recoverable(&self) -> bool {
        match self {
            LlmkgError::ValidationError { .. } => false,
            LlmkgError::EngineError { recoverable, .. } => *recoverable,
            LlmkgError::ExtractionError { .. } => true, // Can fallback to simpler extraction
            LlmkgError::QueryError { .. } => true, // Can retry with modified params
            LlmkgError::StorageError { .. } => true, // Can retry storage operations
            LlmkgError::AuthError { .. } => false,
            LlmkgError::RateLimitError { .. } => true, // Can retry after delay
            LlmkgError::ResourceError { .. } => false, // Hard limits
            LlmkgError::ExternalError { .. } => true, // External services can recover
            LlmkgError::InternalError { .. } => false, // Internal bugs need fixes
        }
    }

    /// Get suggested retry delay in seconds
    pub fn retry_delay_seconds(&self) -> Option<u64> {
        match self {
            LlmkgError::RateLimitError { retry_after_seconds, .. } => Some(*retry_after_seconds),
            LlmkgError::ExternalError { retry_count, .. } => {
                // Exponential backoff: 2^retry_count seconds, max 300 (5 minutes)
                Some((2_u64.pow(*retry_count)).min(300))
            }
            LlmkgError::ExtractionError { .. } => Some(1), // Quick retry for extraction
            LlmkgError::QueryError { .. } => Some(2), // Short delay for query retry
            LlmkgError::StorageError { .. } => Some(5), // Medium delay for storage
            _ => None,
        }
    }

    /// Convert error to user-friendly JSON response
    pub fn to_json_response(&self) -> Value {
        let error_code = match self {
            LlmkgError::ValidationError { .. } => "VALIDATION_ERROR",
            LlmkgError::EngineError { .. } => "ENGINE_ERROR",
            LlmkgError::ExtractionError { .. } => "EXTRACTION_ERROR",
            LlmkgError::QueryError { .. } => "QUERY_ERROR",
            LlmkgError::StorageError { .. } => "STORAGE_ERROR",
            LlmkgError::AuthError { .. } => "AUTH_ERROR",
            LlmkgError::RateLimitError { .. } => "RATE_LIMIT_ERROR",
            LlmkgError::ResourceError { .. } => "RESOURCE_ERROR",
            LlmkgError::ExternalError { .. } => "EXTERNAL_ERROR",
            LlmkgError::InternalError { .. } => "INTERNAL_ERROR",
        };

        let mut response = json!({
            "error": true,
            "error_code": error_code,
            "message": self.to_string(),
            "recoverable": self.is_recoverable(),
            "timestamp": chrono::Utc::now().to_rfc3339()
        });

        if let Some(retry_delay) = self.retry_delay_seconds() {
            response["retry_after_seconds"] = json!(retry_delay);
        }

        // Add specific error details
        match self {
            LlmkgError::ValidationError { field, received_value, .. } => {
                response["details"] = json!({
                    "field": field,
                    "received_value": received_value
                });
            }
            LlmkgError::RateLimitError { operation, retry_after_seconds } => {
                response["details"] = json!({
                    "operation": operation,
                    "retry_after_seconds": retry_after_seconds
                });
            }
            LlmkgError::ResourceError { resource_type, limit, current } => {
                response["details"] = json!({
                    "resource_type": resource_type,
                    "limit": limit,
                    "current_usage": current
                });
            }
            _ => {}
        }

        response
    }

    /// Get suggested fallback action for graceful degradation
    pub fn suggested_fallback(&self) -> Option<String> {
        match self {
            LlmkgError::ExtractionError { extraction_type, .. } => {
                Some(match extraction_type.as_str() {
                    "entity" => "Using simple keyword-based entity extraction".to_string(),
                    "relationship" => "Using pattern-based relationship extraction".to_string(),
                    _ => "Using simplified extraction methods".to_string(),
                })
            }
            LlmkgError::QueryError { query_type, .. } => {
                Some(match query_type.as_str() {
                    "semantic" => "Falling back to keyword-based search".to_string(),
                    "hybrid" => "Using individual search methods".to_string(),
                    _ => "Using simplified query processing".to_string(),
                })
            }
            LlmkgError::ExternalError { service, .. } => {
                Some(format!("Using cached data or alternative method for {}", service))
            }
            _ => None,
        }
    }
}

/// Convert from legacy string errors to structured errors
impl From<String> for LlmkgError {
    fn from(error: String) -> Self {
        // Try to parse common error patterns
        if error.contains("validation") || error.contains("invalid") {
            LlmkgError::ValidationError {
                field: "unknown".to_string(),
                message: error,
                received_value: None,
            }
        } else if error.contains("query") {
            LlmkgError::QueryError {
                query_type: "unknown".to_string(),
                parameters: json!({}),
                cause: error,
            }
        } else if error.contains("storage") || error.contains("store") {
            LlmkgError::StorageError {
                operation: "unknown".to_string(),
                entity_id: None,
                cause: error,
            }
        } else {
            LlmkgError::InternalError {
                component: "unknown".to_string(),
                message: error,
            }
        }
    }
}

/// Convert from crate::error::Result to LlmkgError
impl From<crate::error::Result<()>> for LlmkgError {
    fn from(_result: crate::error::Result<()>) -> Self {
        LlmkgError::InternalError {
            component: "core".to_string(),
            message: "Core operation failed".to_string(),
        }
    }
}

/// Input validation utilities
pub mod validation {
    use super::*;

    /// Validate string field with length constraints
    pub fn validate_string_field(
        field_name: &str,
        value: Option<&str>,
        required: bool,
        max_length: Option<usize>,
        min_length: Option<usize>,
    ) -> Result<String, LlmkgError> {
        let str_value = if required {
            value.ok_or_else(|| LlmkgError::ValidationError {
                field: field_name.to_string(),
                message: "Required field is missing".to_string(),
                received_value: None,
            })?
        } else {
            value.unwrap_or("")
        };

        if str_value.is_empty() && required {
            return Err(LlmkgError::ValidationError {
                field: field_name.to_string(),
                message: "Field cannot be empty".to_string(),
                received_value: Some("empty string".to_string()),
            });
        }

        if let Some(max_len) = max_length {
            if str_value.len() > max_len {
                return Err(LlmkgError::ValidationError {
                    field: field_name.to_string(),
                    message: format!("Field exceeds maximum length of {} characters", max_len),
                    received_value: Some(format!("{} characters", str_value.len())),
                });
            }
        }

        if let Some(min_len) = min_length {
            if str_value.len() < min_len {
                return Err(LlmkgError::ValidationError {
                    field: field_name.to_string(),
                    message: format!("Field must be at least {} characters", min_len),
                    received_value: Some(format!("{} characters", str_value.len())),
                });
            }
        }

        Ok(str_value.to_string())
    }

    /// Validate numeric field with range constraints
    pub fn validate_numeric_field<T>(
        field_name: &str,
        value: Option<T>,
        required: bool,
        min_value: Option<T>,
        max_value: Option<T>,
    ) -> Result<Option<T>, LlmkgError>
    where
        T: PartialOrd + Copy + fmt::Display,
    {
        let num_value = if required {
            Some(value.ok_or_else(|| LlmkgError::ValidationError {
                field: field_name.to_string(),
                message: "Required numeric field is missing".to_string(),
                received_value: None,
            })?)
        } else {
            value
        };

        if let Some(val) = num_value {
            if let Some(min_val) = min_value {
                if val < min_val {
                    return Err(LlmkgError::ValidationError {
                        field: field_name.to_string(),
                        message: format!("Value must be at least {}", min_val),
                        received_value: Some(val.to_string()),
                    });
                }
            }

            if let Some(max_val) = max_value {
                if val > max_val {
                    return Err(LlmkgError::ValidationError {
                        field: field_name.to_string(),
                        message: format!("Value must be at most {}", max_val),
                        received_value: Some(val.to_string()),
                    });
                }
            }
        }

        Ok(num_value)
    }

    /// Validate enum field with allowed values
    pub fn validate_enum_field(
        field_name: &str,
        value: Option<&str>,
        required: bool,
        allowed_values: &[&str],
    ) -> Result<Option<String>, LlmkgError> {
        let str_value = if required {
            Some(value.ok_or_else(|| LlmkgError::ValidationError {
                field: field_name.to_string(),
                message: "Required field is missing".to_string(),
                received_value: None,
            })?)
        } else {
            value
        };

        if let Some(val) = str_value {
            if !allowed_values.contains(&val) {
                return Err(LlmkgError::ValidationError {
                    field: field_name.to_string(),
                    message: format!("Invalid value. Allowed values: {}", allowed_values.join(", ")),
                    received_value: Some(val.to_string()),
                });
            }
        }

        Ok(str_value.map(|s| s.to_string()))
    }

    /// Sanitize user input to prevent injection attacks
    pub fn sanitize_input(input: &str) -> String {
        input
            .replace(['<', '>', '"', '\'', '&'], "")
            .replace('\0', "")
            .trim()
            .to_string()
    }

    /// Validate and sanitize JSON input
    pub fn validate_json_input(input: &Value, max_depth: usize, max_size: usize) -> Result<Value, LlmkgError> {
        let serialized = serde_json::to_string(input)
            .map_err(|e| LlmkgError::ValidationError {
                field: "json_input".to_string(),
                message: format!("Invalid JSON structure: {}", e),
                received_value: None,
            })?;

        if serialized.len() > max_size {
            return Err(LlmkgError::ValidationError {
                field: "json_input".to_string(),
                message: format!("JSON input exceeds maximum size of {} bytes", max_size),
                received_value: Some(format!("{} bytes", serialized.len())),
            });
        }

        // Check depth recursively
        fn check_depth(value: &Value, current_depth: usize, max_depth: usize) -> bool {
            if current_depth > max_depth {
                return false;
            }
            match value {
                Value::Object(obj) => obj.values().all(|v| check_depth(v, current_depth + 1, max_depth)),
                Value::Array(arr) => arr.iter().all(|v| check_depth(v, current_depth + 1, max_depth)),
                _ => true,
            }
        }

        if !check_depth(input, 0, max_depth) {
            return Err(LlmkgError::ValidationError {
                field: "json_input".to_string(),
                message: format!("JSON input exceeds maximum depth of {}", max_depth),
                received_value: None,
            });
        }

        Ok(input.clone())
    }
}

/// Graceful degradation utilities
pub mod graceful {
    use super::*;

    /// Execute operation with fallback
    pub async fn with_fallback<T, F, Fallback>(
        primary_operation: F,
        fallback_operation: Fallback,
        operation_name: &str,
    ) -> LlmkgResult<T>
    where
        F: std::future::Future<Output = LlmkgResult<T>>,
        Fallback: std::future::Future<Output = LlmkgResult<T>>,
    {
        match primary_operation.await {
            Ok(result) => Ok(result),
            Err(error) => {
                log::warn!("Primary operation '{}' failed: {}. Attempting fallback.", operation_name, error);
                
                if error.is_recoverable() {
                    match fallback_operation.await {
                        Ok(fallback_result) => {
                            log::info!("Fallback operation '{}' succeeded", operation_name);
                            Ok(fallback_result)
                        }
                        Err(fallback_error) => {
                            log::error!("Both primary and fallback operations failed for '{}'. Primary: {}, Fallback: {}", 
                                operation_name, error, fallback_error);
                            Err(error) // Return original error
                        }
                    }
                } else {
                    log::error!("Operation '{}' failed with non-recoverable error: {}", operation_name, error);
                    Err(error)
                }
            }
        }
    }

    /// Execute with retry logic
    pub async fn with_retry<T, F>(
        mut operation: F,
        max_retries: u32,
        operation_name: &str,
    ) -> LlmkgResult<T>
    where
        F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = LlmkgResult<T>> + Send>>,
    {
        let mut last_error = None;
        
        for attempt in 0..=max_retries {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !error.is_recoverable() {
                        log::error!("Operation '{}' failed with non-recoverable error on attempt {}: {}", 
                            operation_name, attempt + 1, error);
                        return Err(error);
                    }
                    
                    if attempt < max_retries {
                        if let Some(delay) = error.retry_delay_seconds() {
                            log::warn!("Operation '{}' failed on attempt {}. Retrying after {} seconds: {}", 
                                operation_name, attempt + 1, delay, error);
                            tokio::time::sleep(tokio::time::Duration::from_secs(delay)).await;
                        } else {
                            log::warn!("Operation '{}' failed on attempt {}. Retrying immediately: {}", 
                                operation_name, attempt + 1, error);
                        }
                    }
                    
                    last_error = Some(error);
                }
            }
        }
        
        let final_error = last_error.unwrap_or_else(|| LlmkgError::InternalError {
            component: "retry_logic".to_string(),
            message: "Unexpected error in retry logic".to_string(),
        });
        
        log::error!("Operation '{}' failed after {} attempts: {}", 
            operation_name, max_retries + 1, final_error);
        Err(final_error)
    }
}

/// Convert LlmkgError back to the legacy string error format for compatibility
impl From<LlmkgError> for String {
    fn from(error: LlmkgError) -> Self {
        error.to_string()
    }
}