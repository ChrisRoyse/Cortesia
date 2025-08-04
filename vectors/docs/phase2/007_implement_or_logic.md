# Task 007: Implement OR Logic Testing

## Prerequisites
- Task 001-006 completed: AND logic testing verified
- BooleanSearchEngine::search_boolean() method functional
- SearchResult struct available from boolean.rs
- Test infrastructure from previous tasks

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
You have verified AND logic works correctly. Now you need to implement and test OR logic. OR logic means ANY of the specified terms can be present in the results.

For example:
- Query "struct OR fn" should return documents containing EITHER "struct" OR "fn" (or both)
- Documents with only "struct", only "fn", or both should ALL be returned

## Your Task (10 minutes max)
Implement comprehensive OR logic testing to verify boolean queries work correctly.

## Success Criteria
1. Write failing test for OR logic with comprehensive verification
2. Create test documents with different term combinations
3. Verify OR queries return docs with ANY of the terms
4. Verify results include all valid matches
5. All tests pass after verification

## Implementation Steps

### 1. RED: Write failing OR logic test
```rust
// Add to src/boolean.rs tests
#[test]
fn test_or_logic_comprehensive() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_or_logic_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test OR logic - should find documents with EITHER term
    let results = engine.search_boolean("struct OR fn")?;
    
    // Should find 3 documents:
    // - file1.rs (has "struct")
    // - file2.rs (has "fn") 
    // - file3.rs (has both "struct" and "fn")
    assert!(results.len() >= 3, "OR query should find at least 3 documents");
    
    // Verify each result has at least one of the terms
    for result in &results {
        let has_struct = result.content.to_lowercase().contains("struct");
        let has_fn = result.content.to_lowercase().contains("fn");
        assert!(has_struct || has_fn, 
               "Each result must contain at least one OR term: {}", result.content);
    }
    
    // Verify specific expected files are found
    let file_paths: Vec<&str> = results.iter().map(|r| r.file_path.as_str()).collect();
    assert!(file_paths.iter().any(|path| path.contains("struct_only.rs")));
    assert!(file_paths.iter().any(|path| path.contains("fn_only.rs")));
    assert!(file_paths.iter().any(|path| path.contains("both_terms.rs")));
    
    Ok(())
}

fn create_or_logic_test_index(index_path: &Path) -> Result<()> {
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
    
    // Test documents for OR logic testing
    let test_docs = vec![
        ("struct_only.rs", "pub struct MyStruct { name: String }"),  // Has struct only
        ("fn_only.rs", "pub fn process() { println!(\"Hello\"); }"),  // Has fn only
        ("both_terms.rs", "pub struct Data; pub fn initialize() {}"),  // Has BOTH
        ("neither.rs", "pub mod utils { const VALUE: i32 = 42; }"),  // Has NEITHER
        ("impl_only.rs", "impl Display for MyType { }"),  // Has neither struct nor fn
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

### 2. GREEN: Add validation helper for OR logic
```rust
// Add to BooleanSearchEngine impl
impl BooleanSearchEngine {
    /// Validates that all search results contain at least one of the required terms for OR queries
    pub fn validate_or_results(&self, results: &[SearchResult], terms: &[&str]) -> bool {
        results.iter().all(|result| {
            terms.iter().any(|term| {
                result.content.to_lowercase().contains(&term.to_lowercase())
            })
        })
    }
}
```

### 3. RED: Write edge case tests for OR logic
```rust
#[test]
fn test_or_logic_edge_cases() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_or_logic_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test OR with non-existent terms
    let results = engine.search_boolean("nonexistent OR alsononexistent")?;
    assert!(results.is_empty(), "Should return empty for all non-existent terms");
    
    // Test OR with one valid, one invalid term
    let results = engine.search_boolean("struct OR nonexistent")?;
    assert!(!results.is_empty(), "Should find results for valid term in OR");
    
    // Verify only valid term matches are returned
    assert!(engine.validate_or_results(&results, &["struct", "nonexistent"]));
    
    // Test single term OR (should work like regular search)
    let results = engine.search_boolean("struct")?;
    let or_results = engine.search_boolean("struct OR")?; // Malformed, but Tantivy might handle it
    
    Ok(())
}
```

### 4. REFACTOR: Add comprehensive OR validation test
```rust
#[test]
fn test_or_validation_comprehensive() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_or_logic_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test three-way OR
    let results = engine.search_boolean("struct OR fn OR impl")?;
    
    // Should find 4 documents (struct_only, fn_only, both_terms, impl_only)
    assert!(results.len() >= 4, "Three-way OR should find multiple documents");
    
    // Use validation helper
    assert!(engine.validate_or_results(&results, &["struct", "fn", "impl"]));
    
    // Verify no documents without any of the terms
    for result in &results {
        let content_lower = result.content.to_lowercase();
        let has_any = content_lower.contains("struct") || 
                     content_lower.contains("fn") || 
                     content_lower.contains("impl");
        assert!(has_any, "Result should contain at least one OR term: {}", result.content);
    }
    
    Ok(())
}
```

## Validation Checklist
- [ ] OR queries return documents with ANY of the terms
- [ ] All valid matches are included (no false negatives)
- [ ] Edge cases handled (non-existent terms, single term OR)
- [ ] Three-way OR queries work correctly
- [ ] Validation helper methods work correctly

## Context for Next Task
Next task will implement NOT logic testing to verify that NOT queries properly exclude documents containing specified terms.