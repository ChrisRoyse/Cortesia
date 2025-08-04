# Task 00i: Create Error Types Module

**Estimated Time: 10 minutes**
**Lines of Code: 30**
**Prerequisites: Task 00h completed**

## Context

Phase 3 tasks need proper error handling. This task creates the error types module that provides clear error messages for search and indexing operations.

## Your Task

Create `src/error.rs` with SearchError and IndexingError enums.

## Required Implementation

Create `src/error.rs` with exactly this content:

```rust
//! Error types for the vector search system

use thiserror::Error;

/// Errors that can occur during search operations
#[derive(Error, Debug)]
pub enum SearchError {
    #[error("Index not found at path: {path}")]
    IndexNotFound { path: String },
    
    #[error("Invalid query syntax: {query}. {reason}")]
    InvalidQuery { query: String, reason: String },
    
    #[error("IO error during search: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Configuration error: {message}")]
    ConfigError { message: String },
}

/// Errors that can occur during indexing operations
#[derive(Error, Debug)]
pub enum IndexingError {
    #[error("Failed to create index at path: {path}. {reason}")]
    IndexCreationFailed { path: String, reason: String },
    
    #[error("Failed to index file: {file}. {reason}")]
    FileIndexingFailed { file: String, reason: String },
    
    #[error("IO error during indexing: {0}")]
    IoError(#[from] std::io::Error),
}
```

## Success Criteria

- [ ] File `src/error.rs` created
- [ ] `SearchError` enum with 4 error variants
- [ ] `IndexingError` enum with 3 error variants
- [ ] Proper `thiserror::Error` derives and error messages
- [ ] `#[from]` attributes for automatic conversion from `std::io::Error`
- [ ] Documentation comment for the module
- [ ] File compiles without errors

## Validation

Run `cargo check` - should compile without errors.

## Next Task

Task 00j will create the lib.rs module structure.