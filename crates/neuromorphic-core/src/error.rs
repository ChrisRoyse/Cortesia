//! Error types for neuromorphic core operations
//!
//! Provides comprehensive error handling for all neuromorphic operations
//! with clear error messages and proper error chaining.

use thiserror::Error;

/// Main error type for neuromorphic operations
#[derive(Error, Debug)]
pub enum NeuromorphicError {
    /// Error in spiking column operations
    #[error("Spiking column error: {0}")]
    SpikingColumn(String),

    /// Error in TTFS encoding/decoding
    #[error("TTFS encoding error: {0}")]
    TTFSEncoding(String),

    /// Error in neural branch operations
    #[error("Neural branch error: {0}")]
    NeuralBranch(String),

    /// Error in SIMD operations
    #[error("SIMD backend error: {0}")]
    SimdBackend(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Resource allocation error
    #[error("Resource allocation failed: {0}")]
    ResourceAllocation(String),

    /// Synchronization error
    #[error("Synchronization error: {0}")]
    Synchronization(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Generic error with source
    #[error("{message}")]
    Other {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

/// Result type alias for neuromorphic operations
pub type Result<T> = std::result::Result<T, NeuromorphicError>;

/// Extension trait for adding context to results
pub trait ResultExt<T> {
    /// Add context to an error
    fn context<C>(self, context: C) -> Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static;

    /// Add context lazily (only if error occurs)
    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn context<C>(self, context: C) -> Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static,
    {
        self.map_err(|e| NeuromorphicError::Other {
            message: context.to_string(),
            source: Some(Box::new(e)),
        })
    }

    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        self.map_err(|e| NeuromorphicError::Other {
            message: f().to_string(),
            source: Some(Box::new(e)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = NeuromorphicError::SpikingColumn("test error".to_string());
        assert_eq!(err.to_string(), "Spiking column error: test error");
    }

    #[test]
    fn error_chaining() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err = NeuromorphicError::from(io_err);
        assert!(matches!(err, NeuromorphicError::Io(_)));
    }
}
