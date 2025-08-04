# Task 015: Integration Testing

## Prerequisites
- Task 001-014 completed: Performance infrastructure created
- All components functional: BooleanSearchEngine, CrossChunkBooleanHandler, DocumentLevelValidator
- Performance benchmarks established

## Required Imports
```rust
// Add to integration test file
use anyhow::Result;
use tempfile::TempDir;
use std::path::Path;
use crate::boolean::{BooleanSearchEngine, SearchResult};
use crate::cross_chunk::{CrossChunkBooleanHandler, DocumentResult};
use crate::validator::{DocumentLevelValidator, BooleanQueryStructure};
```

## TASK SPLIT NOTICE
This task has been split into multiple smaller tasks for timing compliance:

- **Task 017**: End-to-End Integration (≤10min) - Basic integration workflow testing
- **Task 018**: Error Handling Integration (≤10min) - Error boundary and edge case testing
- **Task 019**: Component Interaction Tests (≤10min) - Advanced multi-component scenarios

Please complete Tasks 017-019 in sequence for full integration testing coverage.

---

## Context
You have implemented all individual components of the boolean search system. Now you need integration tests that verify all components work together correctly in realistic scenarios that simulate real-world usage.

Integration tests should cover:
- End-to-end boolean search workflows
- Integration between BooleanSearchEngine, DocumentLevelValidator, and CrossChunkBooleanHandler
- Real file indexing and searching scenarios
- Error handling across component boundaries

## Your Task (DEPRECATED - USE SPLIT TASKS)
~~Implement comprehensive integration tests that verify the complete boolean search system works correctly.~~

**USE SPLIT TASKS 017-019 INSTEAD**

## Success Criteria
1. Write integration test for complete boolean search workflow
2. Test real file indexing and searching scenarios  
3. Verify error handling across component boundaries
4. Test integration with existing search infrastructure
5. All integration tests pass successfully

## Implementation Steps

### 1. RED: Write failing end-to-end integration test
```rust
// Create tests/boolean_integration_tests.rs
use anyhow::Result;
use std::path::Path;
use tempfile::TempDir;
use std::fs;

// Import all the components we've built
use llmkg::boolean::BooleanSearchEngine;
use llmkg::cross_chunk::CrossChunkBooleanHandler;
use llmkg::validator::DocumentLevelValidator;

#[test]
fn test_end_to_end_boolean_search() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    let docs_path = temp_dir.path().join("docs");
    
    // Create realistic test documents
    create_realistic_test_documents(&docs_path)?;
    
    // Index the documents (assuming we have an indexer)
    index_realistic_documents(&docs_path, &index_path)?;
    
    // Create the boolean search system
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    // Test 1: Simple boolean AND search
    let results = boolean_engine.search_boolean("pub AND struct")?;
    assert!(!results.is_empty(), "Should find pub struct declarations");
    
    // Validate results with DocumentLevelValidator
    let is_valid = validator.validate_boolean_results("pub AND struct", &results)?;
    assert!(is_valid, "Results should satisfy boolean logic");
    
    // Test 2: Cross-chunk boolean search
    let cross_chunk_results = cross_chunk_handler.search_across_chunks("pub AND Display")?;
    assert!(!cross_chunk_results.is_empty(), "Should find results across chunks");
    
    // Test 3: Complex nested query
    let complex_results = cross_chunk_handler.search_across_chunks("(pub AND struct) OR (impl AND Display)")?;
    assert!(!complex_results.is_empty(), "Should handle complex nested queries");
    
    println!("End-to-end test passed: {} simple results, {} cross-chunk results, {} complex results", 
             results.len(), cross_chunk_results.len(), complex_results.len());
    
    Ok(())
}

fn create_realistic_test_documents(docs_path: &Path) -> Result<()> {
    fs::create_dir_all(docs_path)?;
    
    // Document 1: Simple struct definition
    fs::write(docs_path.join("simple_struct.rs"), 
        "pub struct UserData {\n    name: String,\n    email: String,\n}")?;
    
    // Document 2: Large file that will be chunked
    let large_content = format!(
        "// Large Rust file with multiple components\n{}\n\npub struct LargeStruct {{\n    data: Vec<String>,\n    count: usize,\n}}\n\n{}\n\nimpl Display for LargeStruct {{\n    fn fmt(&self, f: &mut Formatter) -> Result {{\n        write!(f, \"LargeStruct with {} items\", self.count)\n    }}\n}}",
        "// ".repeat(200), // Padding to force chunking
        "// ".repeat(200)  // More padding
    );
    fs::write(docs_path.join("large_file.rs"), large_content)?;
    
    // Document 3: Function implementations
    fs::write(docs_path.join("functions.rs"), 
        "fn helper() -> bool { true }\n\nimpl Clone for MyType {\n    fn clone(&self) -> Self { *self }\n}")?;
    
    // Document 4: Error handling code
    fs::write(docs_path.join("error_handling.rs"), 
        "pub fn process() -> Result<(), Error> {\n    // Process with error handling\n    Ok(())\n}")?;
    
    // Document 5: Mixed visibility
    fs::write(docs_path.join("mixed.rs"), 
        "pub struct PublicStruct;\nstruct PrivateStruct;\npub fn public_function() {}\nfn private_function() {}")?;
    
    Ok(())
}

fn index_realistic_documents(docs_path: &Path, index_path: &Path) -> Result<()> {
    // This function should use your existing document indexer
    // For now, create a simplified indexer that works with our test setup
    
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
    
    // Index all files in the docs directory
    for entry in fs::read_dir(docs_path)? {
        let entry = entry?;
        let file_path = entry.path();
        
        if file_path.extension().map_or(false, |ext| ext == "rs") {
            let content = fs::read_to_string(&file_path)?;
            let file_name = file_path.file_name().unwrap().to_string_lossy();
            
            // Simple chunking: split large files into chunks
            let chunks = chunk_content(&content, 500); // 500 char chunks
            
            for (chunk_index, chunk_content) in chunks.into_iter().enumerate() {
                index_writer.add_document(doc!(
                    file_path_field => file_name.to_string(),
                    content_field => chunk_content.clone(),
                    raw_content_field => chunk_content,
                    chunk_index_field => chunk_index as u64
                ))?;
            }
        }
    }
    
    index_writer.commit()?;
    Ok(())
}

fn chunk_content(content: &str, chunk_size: usize) -> Vec<String> {
    if content.len() <= chunk_size {
        return vec![content.to_string()];
    }
    
    let mut chunks = Vec::new();
    let mut start = 0;
    
    while start < content.len() {
        let end = std::cmp::min(start + chunk_size, content.len());
        chunks.push(content[start..end].to_string());
        start = end;
    }
    
    chunks
}
```

### 2. GREEN: Add error handling integration tests
```rust
// Add to tests/boolean_integration_tests.rs
#[test]
fn test_error_handling_integration() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    // Test 1: Invalid index path
    let result = BooleanSearchEngine::new(&temp_dir.path().join("nonexistent"));
    assert!(result.is_err(), "Should fail with invalid index path");
    
    // Test 2: Create valid index for other error tests
    create_simple_test_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    
    // Test 3: Invalid query syntax (if supported)
    let result = boolean_engine.search_boolean("pub AND AND struct");
    // Tantivy might handle this gracefully, so check if it returns results or error
    match result {
        Ok(results) => println!("Query handled gracefully with {} results", results.len()),
        Err(e) => println!("Query properly rejected: {}", e),
    }
    
    // Test 4: Empty query
    let results = boolean_engine.search_boolean("")?;
    // Should handle empty query gracefully
    println!("Empty query returned {} results", results.len());
    
    // Test 5: Very long query
    let long_query = "pub AND ".repeat(100) + "struct";
    let result = boolean_engine.search_boolean(&long_query);
    match result {
        Ok(results) => println!("Long query handled with {} results", results.len()),
        Err(e) => println!("Long query rejected: {}", e),
    }
    
    Ok(())
}

fn create_simple_test_index(index_path: &Path) -> Result<()> {
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

### 3. REFACTOR: Add component interaction tests
```rust
#[test]
fn test_component_interaction() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_interaction_test_index(&index_path)?;
    
    // Create all components
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    // Test 1: BooleanSearchEngine -> DocumentLevelValidator interaction
    let search_results = boolean_engine.search_boolean("pub AND struct")?;
    let validation_result = validator.validate_boolean_results("pub AND struct", &search_results)?;
    assert!(validation_result, "Search results should be valid according to validator");
    
    // Test 2: BooleanSearchEngine -> CrossChunkBooleanHandler interaction
    let chunk_results = cross_chunk_handler.search_across_chunks("pub AND Display")?;
    assert!(!chunk_results.is_empty(), "Cross-chunk handler should find results");
    
    // Test 3: Verify cross-chunk results would also pass validation
    // Convert DocumentResult back to SearchResult for validation
    let search_results_from_chunks: Vec<_> = chunk_results.into_iter().map(|doc_result| {
        SearchResult {
            file_path: doc_result.file_path,
            content: doc_result.content,
            chunk_index: 0,
            score: doc_result.score,
        }
    }).collect();
    
    let chunk_validation = validator.validate_boolean_results("pub AND Display", &search_results_from_chunks)?;
    assert!(chunk_validation, "Cross-chunk results should pass validation");
    
    println!("Component interaction test passed successfully");
    
    Ok(())
}

use crate::boolean::SearchResult; // Make sure to import this

fn create_interaction_test_index(index_path: &Path) -> Result<()> {
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
    
    // Single chunk document
    index_writer.add_document(doc!(
        file_path_field => "single.rs",
        content_field => "pub struct Data { value: i32 }",
        raw_content_field => "pub struct Data { value: i32 }",
        chunk_index_field => 0u64
    ))?;
    
    // Multi-chunk document with terms spanning chunks
    index_writer.add_document(doc!(
        file_path_field => "multi.rs",
        content_field => "pub struct LargeStruct {",
        raw_content_field => "pub struct LargeStruct {",
        chunk_index_field => 0u64
    ))?;
    
    index_writer.add_document(doc!(
        file_path_field => "multi.rs",
        content_field => "    data: Vec<String>,",
        raw_content_field => "    data: Vec<String>,",
        chunk_index_field => 1u64
    ))?;
    
    index_writer.add_document(doc!(
        file_path_field => "multi.rs",
        content_field => "} impl Display for LargeStruct {",
        raw_content_field => "} impl Display for LargeStruct {",
        chunk_index_field => 2u64
    ))?;
    
    index_writer.commit()?;
    Ok(())
}
```

### 4. Add comprehensive integration test suite
```rust
#[test]
fn test_comprehensive_boolean_scenarios() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_comprehensive_test_index(&index_path)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    
    // Comprehensive test scenarios
    let test_scenarios = vec![
        ("Simple AND", "pub AND struct", 1, 3),
        ("Simple OR", "struct OR fn", 1, 10),
        ("Simple NOT", "pub NOT Error", 1, 10),
        ("Complex nested", "(pub AND struct) OR (fn AND impl)", 1, 10),
        ("Three-way AND", "pub AND struct AND impl", 0, 2),
        ("Cross-chunk spanning", "pub AND Display", 1, 5),
    ];
    
    for (name, query, min_results, max_results) in test_scenarios {
        println!("Testing scenario: {}", name);
        
        // Test with regular search
        let regular_results = boolean_engine.search_boolean(query)?;
        
        // Test with cross-chunk search
        let cross_chunk_results = cross_chunk_handler.search_across_chunks(query)?;
        
        // Validate result counts are reasonable
        assert!(regular_results.len() >= min_results && regular_results.len() <= max_results,
               "Scenario '{}' regular search returned {} results, expected {}-{}", 
               name, regular_results.len(), min_results, max_results);
        
        assert!(cross_chunk_results.len() >= min_results && cross_chunk_results.len() <= max_results,
               "Scenario '{}' cross-chunk search returned {} results, expected {}-{}", 
               name, cross_chunk_results.len(), min_results, max_results);
        
        println!("✓ {}: {} regular, {} cross-chunk results", 
                name, regular_results.len(), cross_chunk_results.len());
    }
    
    Ok(())
}

fn create_comprehensive_test_index(index_path: &Path) -> Result<()> {
    // Create a comprehensive test index with various scenarios
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
    
    // Various test documents to cover all scenarios
    let documents = vec![
        ("pub_struct.rs", "pub struct Data { value: i32 }"),
        ("private_struct.rs", "struct Internal { data: String }"),
        ("pub_fn.rs", "pub fn process() { println!(\"test\"); }"),
        ("impl_only.rs", "impl Display for MyType { fn fmt() {} }"),
        ("error_handling.rs", "pub fn test() -> Result<(), Error> { Ok(()) }"),
        ("fn_impl.rs", "fn helper() {} impl Clone for MyType {}"),
    ];
    
    for (filename, content) in documents {
        index_writer.add_document(doc!(
            file_path_field => filename,
            content_field => content,
            raw_content_field => content,
            chunk_index_field => 0u64
        ))?;
    }
    
    // Add a multi-chunk document for cross-chunk testing
    index_writer.add_document(doc!(
        file_path_field => "large.rs",
        content_field => "pub struct LargeStruct {",
        raw_content_field => "pub struct LargeStruct {",
        chunk_index_field => 0u64
    ))?;
    
    index_writer.add_document(doc!(
        file_path_field => "large.rs",
        content_field => "    data: Vec<String>,",
        raw_content_field => "    data: Vec<String>,",
        chunk_index_field => 1u64
    ))?;
    
    index_writer.add_document(doc!(
        file_path_field => "large.rs",
        content_field => "} impl Display for LargeStruct {",
        raw_content_field => "} impl Display for LargeStruct {",
        chunk_index_field => 2u64
    ))?;
    
    index_writer.commit()?;
    Ok(())
}
```

## Validation Checklist
- [ ] End-to-end boolean search workflow works correctly
- [ ] Component integration between all parts functions properly
- [ ] Error handling works across component boundaries
- [ ] Real file indexing and searching scenarios pass
- [ ] All integration test scenarios pass

## Context for Next Task
Next task will implement final validation and cleanup to ensure all functionality meets the Phase 2 requirements and is ready for production use.