# Task 14b: Create PatternSearchError Enum

**Estimated Time: 10 minutes**  
**Lines of Code: ~30**
**Prerequisites: Task 14a (ProximitySearchError) completed**

## Context
Pattern search operations combine multiple search types including proximity, regex, and wildcard searches. This enum provides specific error handling for pattern validation, compilation failures, and integration with proximity search errors.

## Your Task
Create a comprehensive error enum for pattern search operations that integrates with the ProximitySearchError from task 14a.

## Required Implementation

Create/update `src/pattern_search/error.rs`:

```rust
//! Error types for pattern search operations

use thiserror::Error;
use crate::proximity_search::error::ProximitySearchError;

/// Errors that can occur during pattern search operations
#[derive(Error, Debug)]
pub enum PatternSearchError {
    #[error("Invalid regex pattern: '{pattern}'. Compilation error: {reason}")]
    InvalidRegexPattern { pattern: String, reason: String },
    
    #[error("Invalid wildcard pattern: '{pattern}'. {suggestion}")]
    InvalidWildcardPattern { pattern: String, suggestion: String },
    
    #[error("Pattern too complex: '{pattern}'. Maximum complexity: {max_complexity}, actual: {actual_complexity}")]
    PatternTooComplex { 
        pattern: String, 
        max_complexity: usize, 
        actual_complexity: usize 
    },
    
    #[error("Empty pattern provided. Pattern search requires a valid search pattern")]
    EmptyPattern,
    
    #[error("Unsupported pattern type: '{pattern_type}'. Supported types: regex, wildcard, proximity")]
    UnsupportedPatternType { pattern_type: String },
    
    #[error("Pattern search timeout: '{pattern}' exceeded {timeout_ms}ms limit")]
    SearchTimeout { pattern: String, timeout_ms: u64 },
    
    #[error("Pattern search index mismatch: expected '{expected_type}', found '{actual_type}'")]
    IndexTypeMismatch { expected_type: String, actual_type: String },
    
    #[error("Proximity search error in pattern operation: {0}")]
    ProximitySearchError(#[from] ProximitySearchError),
    
    #[error("IO error during pattern search: {0}")]
    IoError(#[from] std::io::Error),
}

impl PatternSearchError {
    /// Create an invalid regex error with helpful suggestion
    pub fn invalid_regex<S: Into<String>>(pattern: S, reason: S) -> Self {
        Self::InvalidRegexPattern {
            pattern: pattern.into(),
            reason: reason.into(),
        }
    }
    
    /// Create a wildcard pattern error with suggestion
    pub fn invalid_wildcard<S: Into<String>>(pattern: S, suggestion: S) -> Self {
        Self::InvalidWildcardPattern {
            pattern: pattern.into(),
            suggestion: suggestion.into(),
        }
    }
    
    /// Create a complexity error with context
    pub fn pattern_too_complex<S: Into<String>>(
        pattern: S, 
        max: usize, 
        actual: usize
    ) -> Self {
        Self::PatternTooComplex {
            pattern: pattern.into(),
            max_complexity: max,
            actual_complexity: actual,
        }
    }
    
    /// Create a timeout error
    pub fn search_timeout<S: Into<String>>(pattern: S, timeout_ms: u64) -> Self {
        Self::SearchTimeout {
            pattern: pattern.into(),
            timeout_ms,
        }
    }
}

/// Result type alias for pattern search operations
pub type PatternSearchResult<T> = Result<T, PatternSearchError>;

/// Combined error type for operations that may involve both pattern and proximity search
#[derive(Error, Debug)]
pub enum CombinedSearchError {
    #[error("Pattern search error: {0}")]
    Pattern(#[from] PatternSearchError),
    
    #[error("Proximity search error: {0}")]
    Proximity(#[from] ProximitySearchError),
}

pub type CombinedSearchResult<T> = Result<T, CombinedSearchError>;
```

## Success Criteria
- [ ] PatternSearchError enum with 9 comprehensive error variants
- [ ] Integration with ProximitySearchError via #[from] attribute
- [ ] Helper methods for common error creation patterns
- [ ] PatternSearchResult type alias for ergonomic error handling
- [ ] CombinedSearchError for operations using both search types
- [ ] Clear, actionable error messages with context
- [ ] File compiles without errors

## Validation
Run `cargo check` - should compile without errors.

## Next Task
Task 14c will add input validation to proximity search methods using the ProximitySearchError enum.