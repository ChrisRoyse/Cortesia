# Task 14a: Create ProximitySearchError Enum

**Estimated Time: 10 minutes**  
**Lines of Code: ~30**
**Prerequisites: Basic search functionality completed**

## Context
The current proximity search system lacks specific error handling for distance validation, empty queries, and index failures. This creates a robust ProximitySearchError enum with thiserror integration for clear, actionable error messages.

## Your Task
Create a dedicated error enum for proximity search operations with comprehensive error variants and descriptive messages.

## Required Implementation

Create/update `src/proximity_search/error.rs`:

```rust
//! Error types for proximity search operations

use thiserror::Error;

/// Errors that can occur during proximity search operations
#[derive(Error, Debug)]
pub enum ProximitySearchError {
    #[error("Invalid proximity distance: {distance}. Distance must be between 1 and {max_distance}")]
    InvalidDistance { distance: u32, max_distance: u32 },
    
    #[error("Empty search query provided. Proximity search requires at least one search term")]
    EmptyQuery,
    
    #[error("Invalid search terms: {terms:?}. All terms must be non-empty and contain valid characters")]
    InvalidSearchTerms { terms: Vec<String> },
    
    #[error("Proximity search index error: {message}")]
    IndexError { message: String },
    
    #[error("Query parsing failed for proximity search: '{query}'. Reason: {reason}")]
    QueryParsingFailed { query: String, reason: String },
    
    #[error("Search execution failed: {operation}. Error: {source}")]
    ExecutionFailed { operation: String, source: String },
    
    #[error("Configuration error in proximity search: {setting} = {value}. {suggestion}")]
    ConfigurationError { 
        setting: String, 
        value: String, 
        suggestion: String 
    },
    
    #[error("IO error during proximity search: {0}")]
    IoError(#[from] std::io::Error),
}

impl ProximitySearchError {
    /// Create an invalid distance error with context
    pub fn invalid_distance(distance: u32, max_allowed: u32) -> Self {
        Self::InvalidDistance { 
            distance, 
            max_distance: max_allowed 
        }
    }
    
    /// Create an index error with descriptive message
    pub fn index_error<S: Into<String>>(message: S) -> Self {
        Self::IndexError { 
            message: message.into() 
        }
    }
    
    /// Create a query parsing error with context
    pub fn query_parsing_failed<S: Into<String>>(query: S, reason: S) -> Self {
        Self::QueryParsingFailed {
            query: query.into(),
            reason: reason.into(),
        }
    }
}

/// Result type alias for proximity search operations
pub type ProximitySearchResult<T> = Result<T, ProximitySearchError>;
```

## Dependencies to Add
Add to `Cargo.toml` if not already present:
```toml
[dependencies]
thiserror = "1.0"
```

## Success Criteria
- [ ] ProximitySearchError enum with 8 comprehensive error variants
- [ ] Clear, actionable error messages with context
- [ ] Helper methods for common error creation patterns
- [ ] ProximitySearchResult type alias for ergonomic error handling
- [ ] File compiles without errors
- [ ] Follows thiserror best practices

## Validation
Run `cargo check` - should compile without errors.

## Next Task
Task 14b will create the PatternSearchError enum that links to ProximitySearchError for composite search operations.