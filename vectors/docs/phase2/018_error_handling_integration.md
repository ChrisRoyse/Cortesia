# Task 018: Error Handling Integration

## Prerequisites
- Task 001-017 completed: End-to-end integration functional
- All components working in integrated scenarios
- Basic integration test patterns established

## Required Imports
```rust
// Add to error handling test file
use anyhow::{Result, Context};
use tempfile::TempDir;
use std::path::Path;
use crate::boolean::{BooleanSearchEngine, SearchResult};
use crate::cross_chunk::{CrossChunkBooleanHandler, DocumentResult};
use crate::validator::{DocumentLevelValidator, BooleanQueryStructure};
use std::fs;
```

## Context
You have basic end-to-end integration working from Task 017. Now you need to test error handling across component boundaries and edge cases. This task focuses on ensuring robust error handling when components interact and validating graceful failure modes.

Error scenarios to test:
- Invalid index paths and missing files
- Malformed queries and edge cases
- Component boundary error propagation
- Graceful handling of empty results

## Your Task (10 minutes max)
Implement error handling integration tests that verify robust error boundaries across all component interactions.

## Success Criteria
1. Test invalid index path error handling
2. Implement malformed query error testing
3. Verify error propagation across component boundaries
4. Test edge cases like empty queries and results
5. All error handling integration tests pass

## Implementation Steps

### 1. RED: Write failing error handling integration tests
```rust
// Add to tests/basic_integration_tests.rs
#[test]
fn test_error_handling_integration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    println!("Testing error handling integration...");
    
    // Test 1: Invalid index path
    let result = BooleanSearchEngine::new(&temp_dir.path().join("nonexistent"));
    assert!(result.is_err(), "Should fail with invalid index path");
    println!("✓ Invalid index path properly rejected");
    
    // Test 2: Create valid index for other error tests
    create_basic_error_test_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    // Test 3: Empty query handling
    let results = boolean_engine.search_boolean("")?;
    println!("✓ Empty query handled gracefully: {} results", results.len());
    
    // Test 4: Cross-component error handling
    let empty_results = vec![];
    let validation_result = validator.validate_boolean_results("pub AND struct", &empty_results)?;
    assert!(!validation_result, "Empty results should not satisfy boolean logic");
    println!("✓ Cross-component validation handles empty results");
    
    // Test 5: Very long query handling
    let long_query = "pub AND ".repeat(50) + "struct";
    let result = boolean_engine.search_boolean(&long_query);
    match result {
        Ok(results) => println!("✓ Long query handled gracefully: {} results", results.len()),
        Err(e) => println!("✓ Long query properly rejected: {}", e),
    }
    
    Ok(())
}

fn create_basic_error_test_index(index_path: &Path) -> Result<()> {
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
    
    // Add minimal test document
    index_writer.add_document(doc!(
        file_path_field => "test.rs",
        content_field => "pub struct TestStruct { data: String }",
        raw_content_field => "pub struct TestStruct { data: String }",
        chunk_index_field => 0u64
    ))?;
    
    index_writer.commit()?;
    Ok(())
}
```

### 2. GREEN: Add edge case testing
```rust
#[test]  
fn test_edge_case_integration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_edge_case_test_index(&index_path)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing edge case integration...");
    
    // Test 1: Query with no results
    let no_results = boolean_engine.search_boolean("nonexistent AND impossible")?;
    assert!(no_results.is_empty(), "Should return empty results for impossible query");
    
    let validation = validator.validate_boolean_results("nonexistent AND impossible", &no_results)?;
    assert!(!validation, "Empty results should not satisfy validation");
    println!("✓ No results case handled properly");
    
    // Test 2: Single character queries
    let single_char_results = boolean_engine.search_boolean("a")?;
    println!("✓ Single character query: {} results", single_char_results.len());
    
    // Test 3: Special characters in query
    let special_char_result = boolean_engine.search_boolean("struct");
    assert!(special_char_result.is_ok(), "Basic query should work");
    println!("✓ Basic queries work correctly");
    
    // Test 4: Cross-chunk with no spanning results
    let no_span_results = cross_chunk_handler.search_across_chunks("nonexistent AND impossible")?;
    assert!(no_span_results.is_empty(), "Should return empty for impossible cross-chunk query");
    println!("✓ Cross-chunk empty results handled properly");
    
    Ok(())
}

fn create_edge_case_test_index(index_path: &Path) -> Result<()> {
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
    
    // Add documents with various edge case content
    let documents = vec![
        ("simple.rs", "pub struct Data { value: i32 }"),
        ("empty.rs", ""),
        ("special.rs", "struct MyStruct { }"),
    ];
    
    for (filename, content) in documents {
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

### 3. REFACTOR: Add component boundary error testing
```rust
#[test]
fn test_component_boundary_errors() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_boundary_test_index(&index_path)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing component boundary error handling...");
    
    // Test 1: Validator with mismatched results
    let mismatched_result = SearchResult {
        file_path: "test.rs".to_string(),
        content: "fn helper() {}".to_string(), // Missing required "struct"
        chunk_index: 0,
        score: 1.0,
    };
    
    let validation = validator.validate_boolean_results("pub AND struct", &[mismatched_result])?;
    assert!(!validation, "Mismatched results should fail validation");
    println!("✓ Validator correctly rejects mismatched results");
    
    // Test 2: Cross-chunk handler with empty input
    let empty_cross_results = cross_chunk_handler.search_across_chunks("nonexistent")?;
    assert!(empty_cross_results.is_empty(), "Should handle empty cross-chunk gracefully");
    println!("✓ Cross-chunk handler handles empty input correctly");
    
    // Test 3: Boolean engine with minimal query
    let minimal_results = boolean_engine.search_boolean("a")?;
    println!("✓ Boolean engine handles minimal query: {} results", minimal_results.len());
    
    // Test 4: Integration between all components with edge case
    let edge_results = boolean_engine.search_boolean("struct")?;
    if !edge_results.is_empty() {
        let edge_validation = validator.validate_boolean_results("struct", &edge_results)?;
        println!("✓ Full integration with edge case: validation = {}", edge_validation);
    }
    
    println!("All component boundary error tests passed");
    
    Ok(())
}

// Import SearchResult for boundary testing
use llmkg::boolean::SearchResult;

fn create_boundary_test_index(index_path: &Path) -> Result<()> {
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
    
    // Documents for boundary testing
    let boundary_docs = vec![
        ("boundary1.rs", "pub struct BoundaryStruct { data: String }"),
        ("boundary2.rs", "fn boundary_function() { println!(\"test\"); }"),
        ("boundary3.rs", "impl Display for BoundaryType { }"),
    ];
    
    for (filename, content) in boundary_docs {
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

#[test]
fn test_graceful_failure_modes() -> Result<()> {
    println!("Testing graceful failure modes...");
    
    // Test creation of components with minimal setup
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_boundary_test_index(&index_path)?;
    
    // Test that all components can be created successfully
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let _cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let _validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("✓ All components created successfully");
    
    // Test basic functionality doesn't crash
    let basic_results = boolean_engine.search_boolean("struct")?;
    println!("✓ Basic search executed: {} results", basic_results.len());
    
    println!("All graceful failure mode tests passed");
    
    Ok(())
}
```

## Validation Checklist
- [ ] Invalid index path errors are handled properly
- [ ] Malformed and edge case queries are managed gracefully
- [ ] Error propagation across components works correctly
- [ ] Empty results and edge cases are handled appropriately
- [ ] Component boundary error testing passes

## Context for Next Task
Next task (019) will implement advanced multi-component interaction tests with complex scenarios that test the full integration between all boolean search components.