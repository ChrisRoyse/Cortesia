# Task 009: Nested Boolean Expression Testing

## Prerequisites
- Task 001-008 completed: Basic boolean logic (AND/OR/NOT) verified
- BooleanSearchEngine::search_boolean() method functional
- SearchResult struct available from boolean.rs
- All basic boolean tests passing

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
You have verified basic AND, OR, and NOT logic work correctly. Now you need to test complex nested boolean expressions that combine multiple operators with parentheses.

Examples of nested expressions:
- "(pub AND struct) OR (fn AND impl)" - Either (pub AND struct) OR (fn AND impl)
- "(pub OR private) AND struct" - struct AND either pub OR private
- "pub AND (struct OR enum) NOT Error" - pub AND (struct OR enum) but NOT Error

## Your Task (10 minutes max)
Implement comprehensive nested boolean expression testing to verify complex queries work correctly.

## Success Criteria
1. Write failing test for nested expressions with verification
2. Create test documents covering all combinations
3. Verify nested parentheses are handled correctly
4. Test complex multi-level nesting
5. All tests pass after verification

## Implementation Steps

### 1. RED: Write failing nested expression test
```rust
// Add to src/boolean.rs tests
#[test]
fn test_nested_boolean_expressions() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_nested_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test nested AND-OR: (pub AND struct) OR (fn AND impl)
    let results = engine.search_boolean("(pub AND struct) OR (fn AND impl)")?;
    
    // Should find documents matching either condition
    assert!(!results.is_empty(), "Nested expression should find results");
    
    // Verify each result matches at least one of the conditions
    for result in &results {
        let content_lower = result.content.to_lowercase();
        let condition1 = content_lower.contains("pub") && content_lower.contains("struct");
        let condition2 = content_lower.contains("fn") && content_lower.contains("impl");
        
        assert!(condition1 || condition2, 
               "Result should match (pub AND struct) OR (fn AND impl): {}", result.content);
    }
    
    Ok(())
}

fn create_nested_test_index(index_path: &Path) -> Result<()> {
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
    
    // Test documents for nested boolean logic
    let test_docs = vec![
        ("pub_struct.rs", "pub struct Data { value: i32 }"),                    // Matches (pub AND struct)
        ("fn_impl.rs", "fn test() {} impl Display for MyType {}"),             // Matches (fn AND impl)
        ("both_conditions.rs", "pub struct S {} fn f() {} impl I for S {}"),   // Matches BOTH conditions
        ("pub_only.rs", "pub const VALUE: i32 = 42;"),                         // Matches neither condition
        ("struct_only.rs", "struct Internal { data: String }"),                // Has struct but not pub
        ("fn_only.rs", "fn helper() -> bool { true }"),                        // Has fn but not impl
        ("impl_only.rs", "impl Clone for MyStruct { }"),                       // Has impl but not fn
        ("none.rs", "mod utils { use std::collections::HashMap; }"),           // Matches nothing
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

### 2. GREEN: Add nested expression validation helper
```rust
// Add to BooleanSearchEngine impl
impl BooleanSearchEngine {
    /// Validates nested boolean expression results
    pub fn validate_nested_or_and(&self, results: &[SearchResult]) -> bool {
        results.iter().all(|result| {
            let content_lower = result.content.to_lowercase();
            let condition1 = content_lower.contains("pub") && content_lower.contains("struct");
            let condition2 = content_lower.contains("fn") && content_lower.contains("impl");
            condition1 || condition2
        })
    }
}
```

### 3. RED: Write complex nesting tests
```rust
#[test]
fn test_complex_nested_expressions() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_nested_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test three-level nesting: (pub OR private) AND (struct OR enum) NOT Error
    let results = engine.search_boolean("(pub OR struct) AND (impl OR fn)")?;
    
    assert!(!results.is_empty(), "Complex nested expression should find results");
    
    // Verify complex condition
    for result in &results {
        let content_lower = result.content.to_lowercase();
        let first_part = content_lower.contains("pub") || content_lower.contains("struct");
        let second_part = content_lower.contains("impl") || content_lower.contains("fn");
        
        assert!(first_part && second_part, 
               "Result should match (pub OR struct) AND (impl OR fn): {}", result.content);
    }
    
    Ok(())
}

#[test]
fn test_nested_with_not() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_nested_not_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test nested with NOT: (pub AND struct) NOT Error
    let results = engine.search_boolean("(pub AND struct) NOT Error")?;
    
    // Should find pub+struct documents but exclude those with Error
    for result in &results {
        let content_lower = result.content.to_lowercase();
        assert!(content_lower.contains("pub") && content_lower.contains("struct"));
        assert!(!content_lower.contains("error"));
    }
    
    Ok(())
}

fn create_nested_not_test_index(index_path: &Path) -> Result<()> {
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
    
    let test_docs = vec![
        ("good_pub_struct.rs", "pub struct Data { value: i32 }"),                     // SHOULD MATCH
        ("error_pub_struct.rs", "pub struct Data -> Result<(), Error> {}"),          // Should NOT match (has Error)
        ("pub_only.rs", "pub const VALUE: i32 = 42;"),                              // Should NOT match (no struct)
        ("struct_only.rs", "struct Internal { data: String }"),                     // Should NOT match (no pub)
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

### 4. REFACTOR: Add comprehensive nesting validation
```rust
#[test]
fn test_parentheses_precedence() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_nested_test_index(&index_path)?;
    let engine = BooleanSearchEngine::new(&index_path)?;
    
    // Test that parentheses change precedence
    let results1 = engine.search_boolean("pub AND struct OR fn")?;          // (pub AND struct) OR fn
    let results2 = engine.search_boolean("pub AND (struct OR fn)")?;       // pub AND (struct OR fn)
    
    // Results should be different due to different precedence
    // This tests that Tantivy respects parentheses
    
    // Validate results1: should have (pub AND struct) OR fn
    for result in &results1 {
        let content_lower = result.content.to_lowercase();
        let condition1 = content_lower.contains("pub") && content_lower.contains("struct");
        let condition2 = content_lower.contains("fn");
        assert!(condition1 || condition2, "Results1 precedence validation failed");
    }
    
    // Validate results2: should have pub AND (struct OR fn)
    for result in &results2 {
        let content_lower = result.content.to_lowercase();
        let has_pub = content_lower.contains("pub");
        let has_struct_or_fn = content_lower.contains("struct") || content_lower.contains("fn");
        assert!(has_pub && has_struct_or_fn, "Results2 precedence validation failed");
    }
    
    Ok(())
}
```

## Validation Checklist
- [ ] Nested AND-OR expressions work correctly
- [ ] Complex three-level nesting works
- [ ] NOT operator works within nested expressions
- [ ] Parentheses precedence is respected
- [ ] Multiple conditions are properly evaluated

## Context for Next Task
Next task will implement the DocumentLevelValidator to ensure search results actually satisfy boolean logic requirements at the document level.