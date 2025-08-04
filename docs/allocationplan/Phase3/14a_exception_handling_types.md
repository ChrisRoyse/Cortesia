# Task 14a: Create Exception Handling Types

**Time**: 5 minutes (1 min read, 3 min implement, 1 min verify)
**Dependencies**: 13j_cache_mod_file.md
**Stage**: Inheritance System

## Objective
Create comprehensive error types for the inheritance system.

## Implementation
Create `src/inheritance/error_types.rs`:

```rust
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum InheritanceError {
    DatabaseError(String),
    CycleDetected { parent: String, child: String },
    MaxDepthExceeded(u32),
    ConceptNotFound(String),
    PropertyNotFound { concept: String, property: String },
    CacheError(String),
    ValidationError(String),
    ConfigurationError(String),
}

impl fmt::Display for InheritanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InheritanceError::DatabaseError(msg) => write!(f, "Database error: {}", msg),
            InheritanceError::CycleDetected { parent, child } => 
                write!(f, "Inheritance cycle detected: {} -> {}", parent, child),
            InheritanceError::MaxDepthExceeded(depth) => 
                write!(f, "Maximum inheritance depth exceeded: {}", depth),
            InheritanceError::ConceptNotFound(id) => 
                write!(f, "Concept not found: {}", id),
            InheritanceError::PropertyNotFound { concept, property } => 
                write!(f, "Property '{}' not found on concept '{}'", property, concept),
            InheritanceError::CacheError(msg) => write!(f, "Cache error: {}", msg),
            InheritanceError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            InheritanceError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl Error for InheritanceError {}

#[derive(Debug)]
pub enum PropertyError {
    InvalidValue(String),
    TypeMismatch { expected: String, actual: String },
    InheritanceConflict { property: String, sources: Vec<String> },
    ExceptionError(String),
    ResolutionError(String),
}

impl fmt::Display for PropertyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyError::InvalidValue(msg) => write!(f, "Invalid property value: {}", msg),
            PropertyError::TypeMismatch { expected, actual } => 
                write!(f, "Type mismatch: expected {}, got {}", expected, actual),
            PropertyError::InheritanceConflict { property, sources } => 
                write!(f, "Inheritance conflict for property '{}' from sources: {:?}", property, sources),
            PropertyError::ExceptionError(msg) => write!(f, "Exception error: {}", msg),
            PropertyError::ResolutionError(msg) => write!(f, "Resolution error: {}", msg),
        }
    }
}

impl Error for PropertyError {}
```

## Success Criteria
- All error types compile without errors
- Error messages are clear and descriptive

## Next Task
14b_error_context.md