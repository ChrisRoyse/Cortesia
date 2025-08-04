# Task 003: BooleanSearchEngine Constructor

## Prerequisites
- Task 001 completed: Project structure created
- Task 002 completed: Core data structures defined
- BooleanSearchEngine struct exists in boolean.rs
- DocumentResult struct available from boolean.rs

## Required Imports
```rust
// Add to src/boolean.rs for this task
use anyhow::{Result, Context};
use std::path::Path;
use tempfile::TempDir;
use tantivy::{Index, IndexWriter, IndexReader, ReloadPolicy};
use tantivy::schema::{Schema, Field, TEXT, STORED};
use tantivy::query::{QueryParser, Query};
```

## Context
You are implementing the constructor for BooleanSearchEngine, which wraps Tantivy's QueryParser to provide boolean search functionality. The system needs to:

- Load an existing Tantivy index from a file path
- Configure a QueryParser for boolean operations (AND/OR/NOT)
- Set up field mappings for content and raw_content fields
- Handle initialization errors gracefully

The existing codebase has a `SearchEngine` struct that handles basic Tantivy index operations.

## Your Task (10 minutes max)
Implement the BooleanSearchEngine constructor with proper error handling and Tantivy QueryParser configuration.

## Success Criteria
1. Write failing test for `BooleanSearchEngine::new()` method
2. Implement the constructor with proper Tantivy integration
3. Configure QueryParser for boolean operations
4. Handle initialization errors with anyhow::Result
5. Test passes after implementation

## Implementation Steps

### 1. RED: Write failing test
```rust
// Add to src/boolean.rs
use anyhow::Result;
use std::path::Path;
use tempfile::TempDir;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_boolean_search_engine_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        
        // This should fail initially - constructor doesn't exist
        let engine = BooleanSearchEngine::new(&index_path)?;
        
        // Should not crash - basic smoke test
        assert!(std::ptr::addr_of!(engine) as *const _ != std::ptr::null());
        
        Ok(())
    }
}
```

### 2. GREEN: Implement basic constructor
```rust
// Add to src/boolean.rs
use tantivy::query::QueryParser;
use tantivy::Index;
use std::path::Path;
use anyhow::{Result, Context};

pub struct BooleanSearchEngine {
    index: Index,
    query_parser: QueryParser,
}

impl BooleanSearchEngine {
    pub fn new(index_path: &Path) -> Result<Self> {
        // Load existing index
        let index = Index::open_in_dir(index_path)
            .with_context(|| format!("Failed to open index at {:?}", index_path))?;
        
        let schema = index.schema();
        
        // Configure query parser for boolean operations
        let query_parser = QueryParser::for_index(&index, vec![
            schema.get_field("content").context("content field not found")?,
            schema.get_field("raw_content").context("raw_content field not found")?,
        ]);
        
        Ok(Self {
            index,
            query_parser,
        })
    }
}
```

### 3. REFACTOR: Add proper imports and error handling
```rust
// Add proper imports at top of file
use tantivy::query::{QueryParser, Query};
use tantivy::schema::{Field, Schema};
use tantivy::{Index, ReloadPolicy};
use std::path::Path;
use anyhow::{Result, Context};
```

## Error Handling Requirements
- Use `anyhow::Result` for all fallible operations
- Provide meaningful error messages with context
- Handle missing fields gracefully
- Handle index opening failures

## Validation
1. Run `cargo test test_boolean_search_engine_creation` - should pass
2. Test with invalid index path - should return proper error
3. Run `cargo check` - no compilation errors

## Creates for Future Tasks
- Functional BooleanSearchEngine::new() constructor
- Proper Index and QueryParser initialization
- Error handling patterns for future methods

## Exports for Other Tasks
```rust
// From boolean.rs
impl BooleanSearchEngine {
    pub fn new(index_path: &Path) -> Result<Self> { ... }
}
```

## Context for Next Task
Next task will implement basic query parsing functionality that converts string queries into Tantivy Query objects, building on the constructor created here.