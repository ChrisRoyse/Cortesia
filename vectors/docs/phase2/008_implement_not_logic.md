# Task 008: Implement NOT Logic Testing

## Prerequisites
- Task 001-007 completed: AND and OR logic testing verified
- BooleanSearchEngine::search_boolean() method functional
- SearchResult struct available from boolean.rs
- Test infrastructure established

## Required Imports
```rust
// Add to test section of src/boolean.rs
use anyhow::Result;
use tempfile::TempDir;
use tantivy::schema::{Schema, TEXT, STORED};
use tantivy::{Index, doc};
use crate::boolean::{BooleanSearchEngine, SearchResult};
use std::path::Path;
```

## Context
You have verified AND and OR logic work correctly. Now you need to implement and test NOT logic. NOT logic means documents should contain certain terms but EXCLUDE other terms.

For example:
- Query "pub NOT Error" should return documents containing "pub" but NOT containing "Error"
- Documents with both "pub" and "Error" should be excluded
- Documents with only "pub" (no "Error") should be included

## Your Task (10 minutes max)
Implement comprehensive NOT logic testing to verify exclusion queries work correctly.

## Success Criteria
1. Write failing test for NOT logic with precise verification
2. Create test documents with different term combinations
3. Verify NOT queries properly exclude unwanted terms
4. Verify inclusion terms are still required
5. All tests pass after verification

## Implementation Steps

### 1. RED: Write failing NOT logic test
```rust
// Add to src/boolean.rs tests
#[test]
fn test_not_logic_exclusion() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_not_logic_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test NOT logic - should find documents with "pub" but NOT "Error"
    let results = engine.search_boolean("pub NOT Error")?;
    
    // Should find only documents with "pub" but without "Error"
    assert!(!results.is_empty(), "Should find documents with pub but not Error");
    
    // Verify each result has "pub" but not "Error"
    for result in &results {
        let content_lower = result.content.to_lowercase();
        assert!(content_lower.contains("pub"), "All results must contain 'pub': {}", result.content);
        assert!(!content_lower.contains("error"), "No results should contain 'error': {}", result.content);
    }
    
    // Should specifically find "simple_pub.rs" but not "error_handling.rs"
    let file_paths: Vec<&str> = results.iter().map(|r| r.file_path.as_str()).collect();
    assert!(file_paths.iter().any(|path| path.contains("simple_pub.rs")));
    assert!(!file_paths.iter().any(|path| path.contains("error_handling.rs")));
    
    Ok(())
}

fn create_not_logic_test_index(index_path: &Path) -> Result<()> {
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
    
    // Test documents for NOT logic testing
    let test_docs = vec![
        ("simple_pub.rs", "pub struct MyStruct { name: String }"),  // Has pub, no Error - SHOULD MATCH
        ("error_handling.rs", "pub fn test() -> Result<(), Error> { Ok(()) }"),  // Has BOTH pub and Error - SHOULD NOT MATCH
        ("private_struct.rs", "struct InternalData { value: i32 }"),  // No pub, no Error - SHOULD NOT MATCH
        ("error_only.rs", "fn handle_error(e: Error) { eprintln!(\"{}\", e); }"),  // No pub, has Error - SHOULD NOT MATCH
        ("another_pub.rs", "pub mod utils { pub const VALUE: i32 = 42; }"),  // Has pub, no Error - SHOULD MATCH
    ];
    
    for (filename, content) in test_docs {
        index_writer.add_document(doc!(
            file_path_field => filename,
            content_field => content,
            raw_content_field => content,
            chunk_index_field => 0u64
        ))?;
    }
    
    index_writer.commit()?;
    Ok(())
}
```

### 2. GREEN: Add validation helper for NOT logic
```rust
// Add to BooleanSearchEngine impl
impl BooleanSearchEngine {
    /// Validates that all search results contain include terms but not exclude terms
    pub fn validate_not_results(&self, results: &[SearchResult], include_term: &str, exclude_term: &str) -> bool {
        results.iter().all(|result| {
            let content_lower = result.content.to_lowercase();
            let include_lower = include_term.to_lowercase();
            let exclude_lower = exclude_term.to_lowercase();
            
            content_lower.contains(&include_lower) && !content_lower.contains(&exclude_lower)
        })
    }
}
```

### 3. RED: Write edge case tests for NOT logic
```rust
#[test]
fn test_not_logic_edge_cases() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_not_logic_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test NOT with non-existent include term
    let results = engine.search_boolean("nonexistent NOT Error")?;
    assert!(results.is_empty(), "Should return empty for non-existent include term");
    
    // Test NOT with non-existent exclude term (should work like regular search)
    let results = engine.search_boolean("pub NOT nonexistent")?;
    assert!(!results.is_empty(), "Should find pub documents when exclude term doesn't exist");
    
    // Verify all have pub and none have the nonexistent term
    for result in &results {
        assert!(result.content.to_lowercase().contains("pub"));
        assert!(!result.content.to_lowercase().contains("nonexistent"));
    }
    
    Ok(())
}
```

### 4. REFACTOR: Add comprehensive NOT validation test
```rust
#[test]
fn test_not_validation_comprehensive() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_not_logic_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test multiple NOT scenarios
    let test_cases = vec![
        ("pub NOT Error", "pub", "error"),
        ("struct NOT fn", "struct", "fn"),
    ];
    
    for (query, include, exclude) in test_cases {
        let results = engine.search_boolean(query)?;
        
        // Use validation helper
        assert!(engine.validate_not_results(&results, include, exclude),
               "NOT validation failed for query: {}", query);
        
        // Double-check manually
        for result in &results {
            let content_lower = result.content.to_lowercase();
            assert!(content_lower.contains(include), 
                   "Result missing include term '{}': {}", include, result.content);
            assert!(!content_lower.contains(exclude), 
                   "Result contains exclude term '{}': {}", exclude, result.content);
        }
    }
    
    Ok(())
}

#[test]
fn test_not_case_sensitivity() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_not_logic_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test case variations
    let queries = vec!["pub NOT Error", "PUB NOT ERROR", "Pub NOT error"];
    
    for query in queries {
        let results = engine.search_boolean(query)?;
        
        // Should get same results regardless of case
        assert!(!results.is_empty(), "Case variation should still work: {}", query);
        
        // Verify exclusion works with case variations
        for result in &results {
            let content_lower = result.content.to_lowercase();
            assert!(!content_lower.contains("error"), 
                   "Case-insensitive exclusion failed for: {}", query);
        }
    }
    
    Ok(())
}
```

## Validation Checklist
- [ ] NOT queries properly exclude documents with forbidden terms
- [ ] Include terms are still required in results
- [ ] Edge cases handled (non-existent terms)
- [ ] Case sensitivity works correctly
- [ ] Validation helper methods work correctly

## Context for Next Task
Next task will implement nested boolean expression testing to verify complex queries like "(pub AND struct) OR (fn NOT Error)" work correctly.