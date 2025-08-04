# Task 002: Create Core Data Structures

## Prerequisites
- Task 001 completed: Project structure and empty module files created
- src/boolean.rs, src/cross_chunk.rs, src/validator.rs exist
- Module declarations added to lib.rs

## Required Imports
```rust
// Standard imports for this task
use anyhow::Result;
use std::path::Path;
use tempfile::TempDir;
```

## Context
You are continuing Phase 2 implementation of boolean search functionality. The previous task created the basic file structure. Now you need to define the core data structures that will be used throughout the boolean search system.

This system processes search results at both chunk level (pieces of documents) and document level (complete files). Boolean logic needs to work across both levels.

**CRITICAL**: This task defines the CANONICAL versions of key structs. Other tasks will import these, never redefine them.

## Your Task (10 minutes max)
Define the core data structures needed for boolean search functionality, following TDD principles.

## Success Criteria
1. Write failing tests for `DocumentResult` struct creation
2. Implement `DocumentResult` struct with required fields
3. Write failing tests for `BooleanQueryStructure` enum usage
4. Implement `BooleanQueryStructure` enum with all variants
5. All tests should pass after implementation

## Implementation Steps

### 1. RED: Write failing test for DocumentResult
```rust
// Add to src/boolean.rs
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_document_result_creation() {
        let result = DocumentResult {
            file_path: "test.rs".to_string(),
            content: "test content".to_string(),
            chunks: 2,
            score: 0.95,
        };
        
        assert_eq!(result.file_path, "test.rs");
        assert_eq!(result.chunks, 2);
        assert_eq!(result.score, 0.95);
    }
}
```

### 2. GREEN: Implement DocumentResult
```rust
// Add to src/boolean.rs
#[derive(Debug, Clone)]
pub struct DocumentResult {
    pub file_path: String,
    pub content: String,
    pub chunks: usize,
    pub score: f32,
}
```

### 3. RED: Write failing test for BooleanQueryStructure
```rust
#[test]
fn test_boolean_query_structure() {
    let and_query = BooleanQueryStructure::And(vec!["pub".to_string(), "struct".to_string()]);
    let or_query = BooleanQueryStructure::Or(vec!["fn".to_string(), "impl".to_string()]);
    let not_query = BooleanQueryStructure::Not { 
        include: "pub".to_string(), 
        exclude: "private".to_string() 
    };
    
    // These should compile and match correctly
    match and_query {
        BooleanQueryStructure::And(terms) => assert_eq!(terms.len(), 2),
        _ => panic!("Should be And variant"),
    }
}
```

### 4. GREEN: Implement BooleanQueryStructure
```rust
// Add to src/validator.rs - CANONICAL DEFINITION (other tasks import this)
#[derive(Debug, Clone)]
pub enum BooleanQueryStructure {
    And(Vec<String>),
    Or(Vec<String>),
    Not { include: String, exclude: String },
    Complex(Vec<BooleanQueryStructure>),
}
```

## Validation
1. Run `cargo test` - all tests should pass
2. Run `cargo check` - no compilation errors
3. Verify both `DocumentResult` and `BooleanQueryStructure` are properly defined

## Creates for Future Tasks
- **CANONICAL** `DocumentResult` struct in boolean.rs (other tasks import this)
- **CANONICAL** `BooleanQueryStructure` enum in validator.rs (other tasks import this)
- Basic struct definitions with proper derives

## Exports for Other Tasks
```rust
// From boolean.rs
pub struct DocumentResult { file_path, content, chunks, score }

// From validator.rs  
pub enum BooleanQueryStructure { And, Or, Not, Complex }
```

## Context for Next Task
Next task will create the BooleanSearchEngine constructor and basic setup functionality, importing DocumentResult from this task.