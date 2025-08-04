# Task 017: End-to-End Integration

## Prerequisites
- Task 001-016 completed: All individual components tested and validated
- Full system integration ready for end-to-end testing
- All canonical structs available from their respective modules

## Required Imports
```rust
// Add to integration test file or new test module
use anyhow::Result;
use tempfile::TempDir;
use std::path::Path;
use crate::boolean::{BooleanSearchEngine, SearchResult};
use crate::cross_chunk::{CrossChunkBooleanHandler, DocumentResult};
use crate::validator::{DocumentLevelValidator, BooleanQueryStructure};
use std::fs;
```

## Context
You have implemented all individual boolean search components with performance optimizations. Now you need to create basic end-to-end integration tests that verify the complete workflow from real file indexing through boolean search execution.

This task focuses on the core integration workflow:
- Real file creation and indexing
- Basic boolean search execution
- Component integration verification

## Your Task (10 minutes max)
Implement basic end-to-end integration tests that verify the complete boolean search workflow works correctly.

## Success Criteria
1. Create realistic test documents for integration testing
2. Implement file indexing integration test
3. Test basic boolean search workflow end-to-end
4. Verify component integration between main components
5. All basic integration tests pass successfully

## Implementation Steps

### 1. RED: Write failing end-to-end integration test
```rust
// Create tests/basic_integration_tests.rs
use anyhow::Result;
use std::path::Path;
use tempfile::TempDir;
use std::fs;

// Import all the components we've built
use llmkg::boolean::BooleanSearchEngine;
use llmkg::cross_chunk::CrossChunkBooleanHandler;
use llmkg::validator::DocumentLevelValidator;

#[test]
fn test_basic_end_to_end_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    let docs_path = temp_dir.path().join("docs");
    
    // Step 1: Create realistic test documents
    create_basic_test_documents(&docs_path)?;
    
    // Step 2: Index the documents
    index_test_documents(&docs_path, &index_path)?;
    
    // Step 3: Create the boolean search system
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    // Step 4: Test basic boolean search workflow
    let results = boolean_engine.search_boolean("pub AND struct")?;
    assert!(!results.is_empty(), "Should find pub struct declarations");
    
    // Step 5: Validate results with DocumentLevelValidator
    let is_valid = validator.validate_boolean_results("pub AND struct", &results)?;
    assert!(is_valid, "Results should satisfy boolean logic");
    
    // Step 6: Test cross-chunk functionality
    let cross_chunk_results = cross_chunk_handler.search_across_chunks("pub AND Display")?;
    // May be empty if no spanning results, but should not error
    
    println!("Basic end-to-end test passed: {} simple results, {} cross-chunk results", 
             results.len(), cross_chunk_results.len());
    
    Ok(())
}

fn create_basic_test_documents(docs_path: &Path) -> Result<()> {
    fs::create_dir_all(docs_path)?;
    
    // Document 1: Simple struct definition
    fs::write(docs_path.join("simple_struct.rs"), 
        "pub struct UserData {\n    name: String,\n    email: String,\n}")?;
    
    // Document 2: Function implementations  
    fs::write(docs_path.join("functions.rs"), 
        "fn helper() -> bool { true }\n\nimpl Clone for MyType {\n    fn clone(&self) -> Self { *self }\n}")?;
    
    // Document 3: Mixed visibility
    fs::write(docs_path.join("mixed.rs"), 
        "pub struct PublicStruct;\nstruct PrivateStruct;\npub fn public_function() {}\nfn private_function() {}")?;
    
    Ok(())
}
```

### 2. GREEN: Add document indexing integration
```rust
fn index_test_documents(docs_path: &Path, index_path: &Path) -> Result<()> {
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
            
            // Simple chunking for basic integration test
            let chunks = simple_chunk_content(&content, 200);
            
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

fn simple_chunk_content(content: &str, chunk_size: usize) -> Vec<String> {
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

### 3. REFACTOR: Add component integration verification
```rust
#[test]
fn test_component_integration_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_integration_test_index(&index_path)?;
    
    // Create all components
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    // Test workflow: BooleanSearchEngine -> DocumentLevelValidator
    let search_results = boolean_engine.search_boolean("pub AND struct")?;
    let validation_result = validator.validate_boolean_results("pub AND struct", &search_results)?;
    assert!(validation_result, "Search results should be valid according to validator");
    
    // Test workflow: BooleanSearchEngine -> CrossChunkBooleanHandler  
    let chunk_results = cross_chunk_handler.search_across_chunks("pub AND struct")?;
    // Should execute without error (may have zero results)
    
    println!("Component integration test passed: {} search results, {} cross-chunk results", 
             search_results.len(), chunk_results.len());
    
    Ok(())
}

fn create_integration_test_index(index_path: &Path) -> Result<()> {
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
    
    // Basic integration test documents
    let test_documents = vec![
        ("basic.rs", "pub struct Data { value: i32 }"),
        ("functions.rs", "fn process() { println!(\"test\"); }"),
        ("mixed.rs", "pub fn helper() -> bool { true }"),
    ];
    
    for (filename, content) in test_documents {
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
fn test_integration_query_scenarios() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_integration_test_index(&index_path)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    
    // Test basic integration scenarios
    let integration_scenarios = vec![
        ("Simple AND", "pub AND struct"),
        ("Simple OR", "struct OR fn"),
        ("Simple NOT", "pub NOT Error"),
    ];
    
    for (name, query) in integration_scenarios {
        println!("Testing integration scenario: {}", name);
        
        // Test with regular search
        let regular_results = boolean_engine.search_boolean(query)?;
        
        // Test with cross-chunk search (should not error)
        let cross_chunk_results = cross_chunk_handler.search_across_chunks(query)?;
        
        println!("✓ {}: {} regular, {} cross-chunk results", 
                name, regular_results.len(), cross_chunk_results.len());
    }
    
    Ok(())
}
```

## Validation Checklist
- [ ] Realistic test documents are created and indexed
- [ ] Basic boolean search workflow executes end-to-end
- [ ] Component integration between main parts works
- [ ] File indexing integration functions correctly
- [ ] All basic integration test scenarios pass

## Context for Next Task
Next task (018) will focus on error handling integration, testing error boundaries and edge cases across component interactions.