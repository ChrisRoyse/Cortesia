# Task 006: Implement AND Logic Testing

## Prerequisites
- Task 001-005 completed: Search execution functional
- BooleanSearchEngine::search_boolean() method works
- SearchResult struct available from boolean.rs

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
You have basic search execution working. Now you need to verify that AND logic works correctly. AND logic means ALL specified terms must be present in the results.

For example:
- Query "pub AND struct" should only return documents containing BOTH "pub" AND "struct"
- Documents with only "pub" or only "struct" should NOT be returned

Tantivy's QueryParser handles this automatically, but you need to verify it works and create comprehensive tests.

## Your Task (10 minutes max)
Implement comprehensive AND logic testing to verify boolean queries work correctly.

## Success Criteria
1. Write failing test for AND logic with precise verification
2. Create test documents with different term combinations
3. Verify AND queries return only docs with ALL terms
4. Verify results are accurate (no false positives)
5. All tests pass after verification

## Implementation Steps

### 1. RED: Write failing AND logic test
```rust
// Add to src/boolean.rs tests
#[test]
fn test_and_logic_precision() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    // Create index with specific test cases
    create_and_logic_test_index(&index_path)?;
    
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test AND logic - should find only documents with BOTH terms
    let results = engine.search_boolean("pub AND struct")?;
    
    // Should find exactly one document (file1.rs has both "pub" and "struct")
    assert_eq!(results.len(), 1, "AND query should find exactly one document with both terms");
    assert!(results[0].file_path.contains("file1.rs"));
    assert!(results[0].content.contains("pub") && results[0].content.contains("struct"));
    
    // Verify no false positives
    for result in &results {
        assert!(result.content.contains("pub"), "All results must contain 'pub'");
        assert!(result.content.contains("struct"), "All results must contain 'struct'");
    }
    
    Ok(())
}

fn create_and_logic_test_index(index_path: &Path) -> Result<()> {
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
    
    // Test documents for AND logic
    let test_docs = vec![
        ("file1.rs", "pub struct MyStruct { name: String }"),  // Has BOTH pub AND struct
        ("file2.rs", "fn process() { println!(\"Hello\"); }"),  // Has NEITHER
        ("file3.rs", "pub fn initialize() -> Result<(), Error> { Ok(()) }"),  // Has pub but NOT struct
        ("file4.rs", "struct InternalData { value: i32 }"),  // Has struct but NOT pub
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

### 2. GREEN: Verify AND logic works (should already work with Tantivy)
```rust
// Add additional test cases for edge cases
#[test]
fn test_and_logic_edge_cases() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_and_logic_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test case sensitivity 
    let results = engine.search_boolean("PUB AND STRUCT")?;
    // Should still find results (Tantivy is case-insensitive by default)
    assert!(!results.is_empty(), "Should handle case variations");
    
    // Test AND with non-existent terms
    let results = engine.search_boolean("nonexistent AND alsononexistent")?;
    assert!(results.is_empty(), "Should return empty for non-existent terms");
    
    // Test AND with one valid, one invalid term
    let results = engine.search_boolean("pub AND nonexistent")?;
    assert!(results.is_empty(), "Should return empty when any AND term is missing");
    
    Ok(())
}
```

### 3. REFACTOR: Add helper methods for result validation
```rust
// Add helper methods to BooleanSearchEngine
impl BooleanSearchEngine {
    /// Validates that all search results contain all required terms for AND queries
    pub fn validate_and_results(&self, results: &[SearchResult], terms: &[&str]) -> bool {
        results.iter().all(|result| {
            terms.iter().all(|term| {
                result.content.to_lowercase().contains(&term.to_lowercase())
            })
        })
    }
}

// Use in tests
#[test]
fn test_and_validation_helper() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_and_logic_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    let results = engine.search_boolean("pub AND struct")?;
    
    // Use validation helper
    assert!(engine.validate_and_results(&results, &["pub", "struct"]));
    assert!(!engine.validate_and_results(&results, &["pub", "nonexistent"]));
    
    Ok(())
}
```

## Validation Checklist
- [ ] AND queries return only documents with ALL terms
- [ ] No false positives (documents missing any term)
- [ ] Edge cases handled (case sensitivity, non-existent terms)
- [ ] Helper validation methods work correctly

## Context for Next Task
Next task will implement OR logic testing to verify that OR queries return documents with ANY of the specified terms.