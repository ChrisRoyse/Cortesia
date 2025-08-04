# Task 004: Basic Query Parsing Setup

## Prerequisites
- Task 001-003 completed: Project structure and constructor implemented
- BooleanSearchEngine::new() method functional
- QueryParser properly initialized in constructor

## Required Imports
```rust
// Add to src/boolean.rs for this task
use anyhow::{Result, Context};
use tantivy::query::{QueryParser, Query, BooleanQuery};
use tantivy::{Index, ReloadPolicy};
use std::path::Path;
use tempfile::TempDir;
```

## Context
You are implementing basic query parsing functionality for the BooleanSearchEngine. This task sets up the foundation for parsing boolean queries (like "pub AND struct") into Tantivy Query objects.

Tantivy's QueryParser automatically handles boolean logic, but we need to configure it properly and create a method to parse query strings.

## Your Task (10 minutes max)
Implement basic query parsing functionality that converts string queries into Tantivy Query objects.

## Success Criteria
1. Write failing test for query parsing method
2. Implement `parse_query()` method  
3. Configure QueryParser with proper boolean settings
4. Handle parsing errors gracefully
5. Test passes after implementation

## Implementation Steps

### 1. RED: Write failing test
```rust
// Add to src/boolean.rs tests
#[test]
fn test_basic_query_parsing() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    // Create a minimal index for testing (you may need to create this helper)
    create_test_index(&index_path)?;
    
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // This method doesn't exist yet - should fail
    let query = engine.parse_query("test")?;
    
    // Basic smoke test - query should be created
    assert!(std::ptr::addr_of!(query) as *const _ != std::ptr::null());
    
    Ok(())
}

// Helper function to create minimal test index
fn create_test_index(index_path: &Path) -> Result<()> {
    use tantivy::schema::{Schema, TEXT, STORED};
    use tantivy::{Index, doc};
    
    let mut schema_builder = Schema::builder();
    let content_field = schema_builder.add_text_field("content", TEXT | STORED);
    let raw_content_field = schema_builder.add_text_field("raw_content", TEXT | STORED);
    let schema = schema_builder.build();
    
    let index = Index::create_in_dir(index_path, schema.clone())?;
    let mut index_writer = index.writer(50_000_000)?;
    
    // Add a simple test document
    index_writer.add_document(doc!(
        content_field => "test content",
        raw_content_field => "test content"
    ))?;
    
    index_writer.commit()?;
    Ok(())
}
```

### 2. GREEN: Implement parse_query method
```rust
// Add to BooleanSearchEngine impl
use tantivy::query::Query;
use std::sync::Arc;

impl BooleanSearchEngine {
    pub fn parse_query(&self, query_str: &str) -> Result<Box<dyn Query>> {
        let query = self.query_parser
            .parse_query(query_str)
            .with_context(|| format!("Failed to parse query: {}", query_str))?;
        
        Ok(query)
    }
}
```

### 3. REFACTOR: Configure QueryParser for boolean operations
```rust
// Update the constructor to properly configure boolean parsing
impl BooleanSearchEngine {
    pub fn new(index_path: &Path) -> Result<Self> {
        let index = Index::open_in_dir(index_path)
            .with_context(|| format!("Failed to open index at {:?}", index_path))?;
        
        let schema = index.schema();
        
        let mut query_parser = QueryParser::for_index(&index, vec![
            schema.get_field("content").context("content field not found")?,
            schema.get_field("raw_content").context("raw_content field not found")?,
        ]);
        
        // Configure for boolean operations
        query_parser.set_conjunction_by_default(); // AND is default between terms
        
        Ok(Self {
            index,
            query_parser,
        })
    }
}
```

## Expected Query Types
Your implementation should handle:
- Simple terms: "rust"
- AND queries: "pub AND struct" 
- OR queries: "fn OR impl"
- NOT queries: "pub NOT private"

## Validation
1. Run `cargo test test_basic_query_parsing` - should pass
2. Test with invalid query syntax - should return proper error
3. Run `cargo check` - no compilation errors

## Context for Next Task
Next task will implement the search execution functionality to actually run parsed queries against the index.