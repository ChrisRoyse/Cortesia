# Task 005: Search Execution Implementation

## Prerequisites
- Task 001-004 completed: Constructor and query parsing implemented
- BooleanSearchEngine::parse_query() method functional
- Index and QueryParser properly configured

## Required Imports
```rust
// Add to src/boolean.rs for this task
use anyhow::{Result, Context};
use tantivy::collector::TopDocs;
use tantivy::{DocAddress, Document, ReloadPolicy};
use tantivy::query::{QueryParser, Query};
use std::path::Path;
use tempfile::TempDir;
```

## Context
You have BooleanSearchEngine with query parsing capability. Now you need to implement the actual search execution that:

- Takes a parsed query and runs it against the Tantivy index
- Returns search results with scores
- Handles the Tantivy Reader/Searcher pattern correctly
- Converts Tantivy documents back to SearchResult structs

The existing system uses a `SearchResult` struct with fields: `file_path`, `content`, `chunk_index`, `score`.

**CRITICAL**: This task defines the CANONICAL SearchResult struct that other tasks will import.

## Your Task (10 minutes max)
Implement search execution functionality that runs boolean queries and returns SearchResult objects.

## Success Criteria
1. Write failing test for search execution
2. Implement `search_boolean()` method
3. Handle Tantivy Reader/Searcher pattern correctly
4. Convert Tantivy docs to SearchResult structs
5. Test passes with actual search results

## Implementation Steps

### 1. RED: Write failing test
```rust
// Add to src/boolean.rs tests
use std::fs;

#[test]
fn test_search_execution() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    // Create test index with sample data
    create_test_index_with_content(&index_path)?;
    
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // This method doesn't exist yet - should fail
    let results = engine.search_boolean("test")?;
    
    // Should find at least one result
    assert!(!results.is_empty(), "Should find search results");
    assert!(results[0].content.contains("test"));
    
    Ok(())
}

// Helper to create index with actual content
fn create_test_index_with_content(index_path: &Path) -> Result<()> {
    use tantivy::schema::{Schema, TEXT, STORED};
    use tantivy::{Index, doc};
    
    let mut schema_builder = Schema::builder();
    let file_path_field = schema_builder.add_text_field("file_path", TEXT | STORED);
    let content_field = schema_builder.add_text_field("content", TEXT | STORED);
    let raw_content_field = schema_builder.add_text_field("raw_content", TEXT | STORED);
    let chunk_index_field = schema_builder.add_u64_field("chunk_index", STORED);
    let schema = schema_builder.build();
    
    let index = Index::create_in_dir(index_path, schema.clone())?;
    let mut index_writer = index.writer(50_000_000)?;
    
    // Add test documents
    index_writer.add_document(doc!(
        file_path_field => "test1.rs",
        content_field => "pub struct TestStruct { data: String }",
        raw_content_field => "pub struct TestStruct { data: String }",
        chunk_index_field => 0u64
    ))?;
    
    index_writer.add_document(doc!(
        file_path_field => "test2.rs", 
        content_field => "fn test_function() { println!(\"Hello\"); }",
        raw_content_field => "fn test_function() { println!(\"Hello\"); }",
        chunk_index_field => 0u64
    ))?;
    
    index_writer.commit()?;
    Ok(())
}
```

### 2. GREEN: Implement search_boolean method
```rust
// Add to BooleanSearchEngine impl
use tantivy::collector::TopDocs;
use tantivy::{DocAddress, Document};

// Define SearchResult struct - CANONICAL DEFINITION (other tasks import this)
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file_path: String,
    pub content: String,
    pub chunk_index: u64,
    pub score: f32,
}

impl BooleanSearchEngine {
    pub fn search_boolean(&self, query_str: &str) -> Result<Vec<SearchResult>> {
        // Parse the query
        let query = self.parse_query(query_str)?;
        
        // Create reader and searcher
        let reader = self.index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommit)
            .try_into()?;
        let searcher = reader.searcher();
        
        // Execute search
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(100))
            .with_context(|| "Search execution failed")?;
        
        // Convert to SearchResult objects
        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let doc = searcher.doc(doc_address)?;
            let search_result = self.doc_to_search_result(doc, score)?;
            results.push(search_result);
        }
        
        Ok(results)
    }
    
    fn doc_to_search_result(&self, doc: Document, score: f32) -> Result<SearchResult> {
        let schema = self.index.schema();
        
        let file_path_field = schema.get_field("file_path")?;
        let content_field = schema.get_field("content")?;
        let chunk_index_field = schema.get_field("chunk_index")?;
        
        let file_path = doc.get_first(file_path_field)
            .and_then(|v| v.as_text())
            .unwrap_or("unknown")
            .to_string();
            
        let content = doc.get_first(content_field)
            .and_then(|v| v.as_text())
            .unwrap_or("")
            .to_string();
            
        let chunk_index = doc.get_first(chunk_index_field)
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        
        Ok(SearchResult {
            file_path,
            content,
            chunk_index,
            score,
        })
    }
}
```

## Validation
1. Run `cargo test test_search_execution` - should pass
2. Verify search results contain expected fields
3. Test with empty query - should handle gracefully
4. Run `cargo check` - no compilation errors

## Creates for Future Tasks
- search_boolean() method that executes queries and returns results
- **CANONICAL** SearchResult struct (other tasks import this)
- Document to SearchResult conversion logic
- Tantivy Reader/Searcher integration patterns

## Exports for Other Tasks
```rust
// From boolean.rs
pub struct SearchResult { file_path, content, chunk_index, score }

impl BooleanSearchEngine {
    pub fn search_boolean(&self, query_str: &str) -> Result<Vec<SearchResult>> { ... }
}
```

## Context for Next Task
Next task will implement AND logic testing to verify boolean queries work correctly using the search_boolean method and SearchResult struct created here.