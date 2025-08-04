# Task 019: Component Interaction Tests

## Prerequisites
- Task 001-018 completed: Error handling integration verified
- All components robust and error-resistant
- Integration patterns well-established

## Required Imports
```rust
// Add to component interaction test file
use anyhow::Result;
use tempfile::TempDir;
use std::path::Path;
use crate::boolean::{BooleanSearchEngine, SearchResult};
use crate::cross_chunk::{CrossChunkBooleanHandler, DocumentResult};
use crate::validator::{DocumentLevelValidator, BooleanQueryStructure};
use std::sync::{Arc, Mutex};
use std::thread;
```

## Context
You have basic integration and error handling from Tasks 017-018. Now you need to test advanced multi-component scenarios that verify complex interactions between BooleanSearchEngine, CrossChunkBooleanHandler, and DocumentLevelValidator working together in realistic scenarios.

Advanced interaction scenarios:
- Multi-step workflows combining all components
- Complex nested queries with cross-chunk validation
- Advanced result processing and validation chains
- Realistic multi-component data flows

## Your Task (10 minutes max)
Implement advanced component interaction tests that verify complex multi-component scenarios work correctly together.

## Success Criteria
1. Test multi-step workflows combining all components
2. Implement complex nested query interaction scenarios
3. Verify advanced result validation chains
4. Test realistic multi-component data flows
5. All advanced interaction tests pass successfully

## Implementation Steps

### 1. RED: Write failing advanced interaction tests
```rust
// Create tests/advanced_interaction_tests.rs
use anyhow::Result;
use std::path::Path;
use tempfile::TempDir;

use llmkg::boolean::{BooleanSearchEngine, SearchResult};
use llmkg::cross_chunk::{CrossChunkBooleanHandler, DocumentResult};
use llmkg::validator::DocumentLevelValidator;

#[test]
fn test_advanced_multi_component_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_advanced_interaction_index(&index_path)?;
    
    // Create all components
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing advanced multi-component workflow...");
    
    // Workflow Step 1: Complex boolean search
    let complex_results = boolean_engine.search_boolean("(pub AND struct) OR (impl AND Display)")?;
    println!("Step 1 - Complex search: {} results", complex_results.len());
    
    // Workflow Step 2: Validate complex results
    let complex_validation = validator.validate_boolean_results(
        "(pub AND struct) OR (impl AND Display)", 
        &complex_results
    )?;
    println!("Step 2 - Complex validation: {}", complex_validation);
    
    // Workflow Step 3: Cross-chunk processing
    let cross_chunk_results = cross_chunk_handler.search_across_chunks("pub AND Display")?;
    println!("Step 3 - Cross-chunk search: {} results", cross_chunk_results.len());
    
    // Workflow Step 4: Convert and validate cross-chunk results
    if !cross_chunk_results.is_empty() {
        let converted_results: Vec<SearchResult> = cross_chunk_results.into_iter().map(|doc_result| {
            SearchResult {
                file_path: doc_result.file_path,
                content: doc_result.content,
                chunk_index: 0,
                score: doc_result.score,
            }
        }).collect();
        
        let cross_validation = validator.validate_boolean_results("pub AND Display", &converted_results)?;
        println!("Step 4 - Cross-chunk validation: {}", cross_validation);
    }
    
    println!("✓ Advanced multi-component workflow completed successfully");
    
    Ok(())
}

fn create_advanced_interaction_index(index_path: &Path) -> Result<()> {
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
    
    // Advanced interaction test documents
    let documents = vec![
        // Single chunk documents
        ("single_pub_struct.rs", "pub struct AdvancedData { value: i32 }", 0),
        ("single_impl.rs", "impl Display for AdvancedType { fn fmt() {} }", 0),
        ("single_fn.rs", "pub fn advanced_process() { println!(\"test\"); }", 0),
        
        // Multi-chunk document
        ("multi_chunk.rs", "pub struct LargeAdvanced {", 0),
        ("multi_chunk.rs", "    data: Vec<String>,", 1),
        ("multi_chunk.rs", "} impl Display for LargeAdvanced {", 2),
        ("multi_chunk.rs", "    fn fmt(&self) -> String { String::new() }", 3),
    ];
    
    for (filename, content, chunk_index) in documents {
        index_writer.add_document(doc!(
            file_path_field => filename,
            content_field => content,
            raw_content_field => content,
            chunk_index_field => chunk_index as u64
        ))?;
    }
    
    index_writer.commit()?;
    Ok(())
}
```

### 2. GREEN: Add complex interaction scenarios
```rust
#[test]
fn test_complex_nested_query_interactions() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_nested_query_index(&index_path)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing complex nested query interactions...");
    
    // Complex nested query scenarios
    let nested_scenarios = vec![
        ("Three-way OR", "struct OR fn OR impl"),
        ("Nested AND-OR", "(pub AND struct) OR (fn AND helper)"),
        ("Complex NOT", "(pub OR struct) NOT Error"),
        ("Deep nesting", "((pub AND struct) OR (impl AND Display)) NOT Error"),
    ];
    
    for (scenario_name, query) in nested_scenarios {
        println!("  Testing scenario: {}", scenario_name);
        
        // Test with boolean engine
        let boolean_results = boolean_engine.search_boolean(query)?;
        
        // Test with cross-chunk handler
        let cross_results = cross_chunk_handler.search_across_chunks(query)?;
        
        // Validate boolean results if they exist
        if !boolean_results.is_empty() {
            let validation = validator.validate_boolean_results(query, &boolean_results)?;
            println!("    Boolean: {} results, validation: {}", boolean_results.len(), validation);
        } else {
            println!("    Boolean: 0 results");
        }
        
        // Report cross-chunk results
        println!("    Cross-chunk: {} results", cross_results.len());
    }
    
    println!("✓ All complex nested query interactions completed");
    
    Ok(())
}

fn create_nested_query_index(index_path: &Path) -> Result<()> {
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
    
    // Documents for nested query testing
    let nested_docs = vec![
        ("nested1.rs", "pub struct NestedData { value: i32 }"),
        ("nested2.rs", "fn helper() -> bool { true }"),
        ("nested3.rs", "impl Display for NestedType { }"),
        ("nested4.rs", "pub fn process() -> Result<(), Error> { Ok(()) }"),
        ("nested5.rs", "struct Internal { data: String }"),
        ("nested6.rs", "impl Clone for NestedClone { }"),
    ];
    
    for (filename, content) in nested_docs {
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

### 3. REFACTOR: Add validation chain testing
```rust
#[test]
fn test_validation_chain_interactions() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_validation_chain_index(&index_path)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing validation chain interactions...");
    
    // Chain Test 1: Boolean -> Validation -> Cross-chunk -> Validation
    let query = "pub AND struct";
    
    // Step 1: Boolean search
    let boolean_results = boolean_engine.search_boolean(query)?;
    println!("  Chain step 1 - Boolean search: {} results", boolean_results.len());
    
    // Step 2: Validate boolean results
    if !boolean_results.is_empty() {
        let boolean_validation = validator.validate_boolean_results(query, &boolean_results)?;
        println!("  Chain step 2 - Boolean validation: {}", boolean_validation);
        
        // Step 3: Cross-chunk search
        let cross_results = cross_chunk_handler.search_across_chunks(query)?;
        println!("  Chain step 3 - Cross-chunk search: {} results", cross_results.len());
        
        // Step 4: Validate cross-chunk results (convert first)
        if !cross_results.is_empty() {
            let converted_cross_results: Vec<SearchResult> = cross_results.into_iter().map(|doc_result| {
                SearchResult {
                    file_path: doc_result.file_path,
                    content: doc_result.content,
                    chunk_index: 0,
                    score: doc_result.score,
                }
            }).collect();
            
            let cross_validation = validator.validate_boolean_results(query, &converted_cross_results)?;
            println!("  Chain step 4 - Cross-chunk validation: {}", cross_validation);
        }
    }
    
    println!("✓ Validation chain interactions completed successfully");
    
    Ok(())
}

fn create_validation_chain_index(index_path: &Path) -> Result<()> {
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
    
    // Documents for validation chain testing
    let chain_docs = vec![
        ("chain1.rs", "pub struct ChainData { value: i32 }"),
        ("chain2.rs", "struct PrivateChain { data: String }"),
        ("chain3.rs", "pub fn chain_process() { }"),
        ("chain4.rs", "impl Display for ChainType { }"),
    ];
    
    for (filename, content) in chain_docs {
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
fn test_comprehensive_interaction_scenarios() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_comprehensive_interaction_index(&index_path)?;
    
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let cross_chunk_handler = CrossChunkBooleanHandler::new(boolean_engine.clone());
    let validator = DocumentLevelValidator::new(boolean_engine.clone());
    
    println!("Testing comprehensive interaction scenarios...");
    
    // Comprehensive scenario testing
    let comprehensive_scenarios = vec![
        ("Basic AND with validation", "pub AND struct"),
        ("Basic OR with cross-chunk", "struct OR impl"),
        ("Complex with all components", "(pub AND struct) OR (impl AND Display)"),
    ];
    
    for (scenario_name, query) in comprehensive_scenarios {
        println!("  Comprehensive scenario: {}", scenario_name);
        
        // Full workflow test
        let boolean_results = boolean_engine.search_boolean(query)?;
        let cross_results = cross_chunk_handler.search_across_chunks(query)?;
        
        if !boolean_results.is_empty() {
            let validation = validator.validate_boolean_results(query, &boolean_results)?;
            println!("    Results: {} boolean (valid: {}), {} cross-chunk", 
                    boolean_results.len(), validation, cross_results.len());
        } else {
            println!("    Results: 0 boolean, {} cross-chunk", cross_results.len());
        }
    }
    
    println!("✓ All comprehensive interaction scenarios completed");
    
    Ok(())
}

fn create_comprehensive_interaction_index(index_path: &Path) -> Result<()> {
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
    
    // Comprehensive test documents
    let comp_docs = vec![
        ("comp1.rs", "pub struct ComprehensiveData { value: i32 }"),
        ("comp2.rs", "impl Display for CompType { fn fmt() {} }"),
        ("comp3.rs", "pub fn comprehensive_process() -> bool { true }"),
        ("comp4.rs", "struct InternalComp { data: Vec<String> }"),
    ];
    
    for (filename, content) in comp_docs {
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

## Validation Checklist
- [ ] Multi-step workflows combining all components work correctly
- [ ] Complex nested query interactions are handled properly  
- [ ] Advanced result validation chains function as expected
- [ ] Realistic multi-component data flows execute successfully
- [ ] All advanced interaction test scenarios pass

## Context for Next Task
Next task (020) will focus on logic correctness validation, implementing focused tests specifically for AND/OR/NOT logic validation to ensure 100% boolean logic accuracy.