# Task 22: Complete Component Interaction Testing Implementation

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)  
**Prerequisites:** Task 21 completed (end-to-end integration tests)  
**Input Files:** 
- `C:/code/LLMKG/vectors/tantivy_search/src/schema.rs` (exists with schema functions)
- `C:/code/LLMKG/vectors/tantivy_search/tests/` (directory exists)
- Task 21 integration tests (`integration_workflow.rs`)

## Complete Context (For AI with ZERO Knowledge)

**What is Component Interaction Testing?** Component interaction testing focuses on validating the interfaces and data flow between different modules of a system, ensuring they communicate correctly and handle edge cases gracefully.

**Why This Task is Critical?** After Task 21 validated the complete end-to-end workflow, we now need focused tests on how specific components interact with each other. This catches subtle integration bugs that E2E tests might miss.

**Key Component Interactions to Test:**
1. **Schema ↔ Indexer** - Schema field definitions must match indexer expectations
2. **Indexer ↔ Writer** - Document creation and commit operations
3. **Reader ↔ Searcher** - Query execution and result retrieval  
4. **Query Parser ↔ Index** - Field validation and query construction
5. **Document ↔ Schema** - Field type validation and data integrity

**Real-World Interaction Scenarios:**
- **Field Mismatch:** Schema defines "content" but indexer tries to write to "text"
- **Type Conflicts:** Schema expects u64 but indexer provides String
- **Missing Fields:** Document lacks required schema fields
- **Index Corruption:** Reader tries to read corrupted index data
- **Query Errors:** Parser creates invalid queries for schema
- **Memory Pressure:** Large documents cause writer buffer overflow

**What Makes This Different from E2E Tests?**
- **Focused scope:** Tests individual component pairs, not entire workflow
- **Error injection:** Deliberately introduces invalid data to test error handling
- **Edge cases:** Tests boundary conditions and unusual inputs
- **Performance isolation:** Measures interaction overhead separately

## Exact Steps (6 minutes implementation)

### Step 1: Create component interaction test file (4 minutes)

Create the file `C:/code/LLMKG/vectors/tantivy_search/tests/component_interactions.rs` with this exact content:

```rust
use tantivy_search::*;
use tempfile::TempDir;
use tantivy::{doc, Index, IndexWriter, Searcher, Document};
use tantivy::query::QueryParser;
use tantivy::collector::TopDocs;
use anyhow::Result;
use std::time::Instant;

/// Test schema and indexer interaction - ensuring field compatibility
#[test]
fn test_schema_indexer_field_compatibility() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("test_index");
    
    // Create schema
    let schema = get_schema();
    let index = create_index(&index_path)?;
    
    // Verify schema matches indexer expectations
    assert_eq!(index.schema(), schema, "Created index schema should match get_schema()");
    
    // Test all required fields exist
    let required_fields = [
        "content", "raw_content", "file_path", 
        "chunk_index", "chunk_start", "chunk_end", "has_overlap"
    ];
    
    for field_name in &required_fields {
        let field_result = schema.get_field(field_name);
        assert!(field_result.is_ok(), "Schema must have field: {}", field_name);
    }
    
    // Test field types match expectations
    let content_field = schema.get_field("content")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    let content_entry = schema.get_field_entry(content_field);
    let chunk_index_entry = schema.get_field_entry(chunk_index_field);
    let has_overlap_entry = schema.get_field_entry(has_overlap_field);
    
    // Verify field types
    assert!(matches!(content_entry.field_type(), tantivy::schema::FieldType::Str(_)));
    assert!(matches!(chunk_index_entry.field_type(), tantivy::schema::FieldType::U64(_)));
    assert!(matches!(has_overlap_entry.field_type(), tantivy::schema::FieldType::Bool(_)));
    
    Ok(())
}

/// Test indexer and writer interaction - document creation and commits
#[test]
fn test_indexer_writer_document_operations() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("test_index");
    
    let schema = get_schema();
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    // Get field handles
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    // Test document creation with all fields
    let test_content = "fn test_function() { println!(\"Hello, world!\"); }";
    let doc = doc!(
        content_field => test_content,
        raw_content_field => test_content,
        file_path_field => "/test/path.rs",
        chunk_index_field => 0u64,
        chunk_start_field => 0u64,
        chunk_end_field => test_content.len() as u64,
        has_overlap_field => false
    );
    
    // Test writer accepts document
    writer.add_document(doc.clone())?;
    
    // Test commit operation
    let commit_result = writer.commit();
    assert!(commit_result.is_ok(), "Writer commit should succeed");
    
    // Verify document was actually written
    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.num_docs(), 1, "Should have exactly one document after commit");
    
    Ok(())
}

/// Test reader and searcher interaction - query execution and retrieval
#[test]
fn test_reader_searcher_query_execution() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("test_index");
    
    let schema = get_schema();
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    // Add test documents
    let test_documents = vec![
        ("fn main() {}", "/src/main.rs"),
        ("struct Config {}", "/src/config.rs"),
        ("impl Database {}", "/src/database.rs"),
    ];
    
    let content_field = schema.get_field("content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    let raw_content_field = schema.get_field("raw_content")?;
    
    for (i, (content, path)) in test_documents.iter().enumerate() {
        let doc = doc!(
            content_field => *content,
            raw_content_field => *content,
            file_path_field => *path,
            chunk_index_field => i as u64,
            chunk_start_field => 0u64,
            chunk_end_field => content.len() as u64,
            has_overlap_field => false
        );
        writer.add_document(doc)?;
    }
    writer.commit()?;
    
    // Test reader creation
    let reader = index.reader()?;
    let searcher = reader.searcher();
    
    // Test searcher can access documents
    assert_eq!(searcher.num_docs(), 3, "Should have all test documents");
    
    // Test query execution
    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    let query = query_parser.parse_query("main")?;
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
    
    assert!(!top_docs.is_empty(), "Should find documents matching 'main'");
    
    // Test document retrieval
    let (_, doc_address) = top_docs[0];
    let retrieved_doc = searcher.doc(doc_address)?;
    let retrieved_content = retrieved_doc.get_first(content_field).unwrap().as_text().unwrap();
    assert!(retrieved_content.contains("main"), "Retrieved document should contain 'main'");
    
    Ok(())
}

/// Test query parser and index interaction - field validation
#[test]
fn test_query_parser_index_field_validation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("test_index");
    
    let schema = get_schema();
    let index = create_index(&index_path)?;
    
    // Test query parser with valid fields
    let content_field = schema.get_field("content")?;
    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    
    // Test valid query creation
    let valid_query = query_parser.parse_query("test");
    assert!(valid_query.is_ok(), "Should parse simple query successfully");
    
    let valid_phrase_query = query_parser.parse_query("\"test phrase\"");
    assert!(valid_phrase_query.is_ok(), "Should parse phrase query successfully");
    
    // Test query parser field compatibility
    let file_path_field = schema.get_field("file_path")?;
    let multi_field_parser = QueryParser::for_index(&index, vec![content_field, file_path_field]);
    
    let multi_field_query = multi_field_parser.parse_query("config");
    assert!(multi_field_query.is_ok(), "Should handle multi-field queries");
    
    Ok(())
}

/// Test document and schema interaction - field validation and data integrity
#[test]
fn test_document_schema_data_integrity() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("test_index");
    
    let schema = get_schema();
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    // Get all field handles
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    // Test document with all required fields
    let complete_doc = doc!(
        content_field => "complete test content",
        raw_content_field => "complete test content",
        file_path_field => "/complete/test.rs",
        chunk_index_field => 42u64,
        chunk_start_field => 100u64,
        chunk_end_field => 200u64,
        has_overlap_field => true
    );
    
    writer.add_document(complete_doc)?;
    writer.commit()?;
    
    // Verify data integrity after round-trip
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let retrieved_doc = searcher.doc(tantivy::DocAddress::new(0, 0))?;
    
    // Verify all fields preserved correctly
    assert_eq!(
        retrieved_doc.get_first(content_field).unwrap().as_text().unwrap(),
        "complete test content"
    );
    assert_eq!(
        retrieved_doc.get_first(file_path_field).unwrap().as_text().unwrap(),
        "/complete/test.rs"
    );
    assert_eq!(
        retrieved_doc.get_first(chunk_index_field).unwrap().as_u64().unwrap(),
        42u64
    );
    assert_eq!(
        retrieved_doc.get_first(chunk_start_field).unwrap().as_u64().unwrap(),
        100u64
    );
    assert_eq!(
        retrieved_doc.get_first(chunk_end_field).unwrap().as_u64().unwrap(),
        200u64
    );
    assert_eq!(
        retrieved_doc.get_first(has_overlap_field).unwrap().as_bool().unwrap(),
        true
    );
    
    Ok(())
}

/// Test error handling in component interactions
#[test]
fn test_component_interaction_error_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("test_index");
    
    let schema = get_schema();
    let index = create_index(&index_path)?;
    
    // Test invalid field access
    let invalid_field_result = schema.get_field("nonexistent_field");
    assert!(invalid_field_result.is_err(), "Should fail for nonexistent field");
    
    // Test writer with excessive memory (should handle gracefully)
    let writer_result = index.writer(1); // Very small buffer
    assert!(writer_result.is_ok(), "Should create writer even with small buffer");
    
    // Test empty query parsing
    let content_field = schema.get_field("content")?;
    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    
    let empty_query_result = query_parser.parse_query("");
    // Empty queries might be valid or invalid depending on parser implementation
    // Just ensure it doesn't panic
    
    Ok(())
}

/// Test performance of component interactions
#[test]
fn test_component_interaction_performance() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("test_index");
    
    let schema = get_schema();
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    // Measure schema field lookup performance
    let start_time = Instant::now();
    for _ in 0..1000 {
        let _ = schema.get_field("content")?;
        let _ = schema.get_field("raw_content")?;
        let _ = schema.get_field("file_path")?;
    }
    let field_lookup_time = start_time.elapsed();
    
    // Should be very fast
    assert!(field_lookup_time.as_millis() < 100, "Field lookups should be fast");
    
    // Measure document creation performance
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    let start_time = Instant::now();
    for i in 0..100 {
        let doc = doc!(
            content_field => format!("test content {}", i),
            raw_content_field => format!("test content {}", i),
            file_path_field => format!("/test/file_{}.rs", i),
            chunk_index_field => i as u64,
            chunk_start_field => 0u64,
            chunk_end_field => 20u64,
            has_overlap_field => false
        );
        writer.add_document(doc)?;
    }
    let document_creation_time = start_time.elapsed();
    
    // Document creation should be reasonably fast
    assert!(document_creation_time.as_millis() < 1000, "Document creation should be fast");
    
    // Test commit performance
    let start_time = Instant::now();
    writer.commit()?;
    let commit_time = start_time.elapsed();
    
    // Commit should complete in reasonable time
    assert!(commit_time.as_millis() < 5000, "Commit should complete within 5 seconds");
    
    Ok(())
}
```

### Step 2: Add test configuration (1 minute)

Add to `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml` if not already present:

```toml
[dev-dependencies]
tempfile = "3.8"
anyhow = "1.0"
```

### Step 3: Verify component interactions (1 minute)

Ensure all required functions are exported in `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs`:

```rust
pub mod schema;
pub use schema::{get_schema, create_index, open_or_create_index};
```

## Verification Steps (2 minutes)

```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo test component_interactions
```

**Expected output:**
```
running 7 tests
test test_schema_indexer_field_compatibility ... ok
test test_indexer_writer_document_operations ... ok
test test_reader_searcher_query_execution ... ok
test test_query_parser_index_field_validation ... ok
test test_document_schema_data_integrity ... ok
test test_component_interaction_error_handling ... ok
test test_component_interaction_performance ... ok

test result: ok. 7 passed; 0 failed; 0 ignored
```

## If This Task Fails

### Common Errors and Solutions

**Error 1: "error[E0432]: unresolved import `tantivy::doc`"**
```bash
# Solution: Missing doc macro import
# Add to test file: use tantivy::doc;
cargo check
cargo test component_interactions
```

**Error 2: "Field not found: content"**
```bash
# Solution: Schema not properly initialized
# Verify schema.rs has get_schema() function implemented
cargo test schema
cargo test component_interactions
```

**Error 3: "Index creation failed: directory not found"**
```bash
# Solution: Temporary directory creation issue
# Verify tempfile dependency in Cargo.toml
cargo add --dev tempfile
cargo test component_interactions
```

**Error 4: "Writer commit failed: lock error"**
```bash
# Solution: Multiple writers or existing locks
# Clean up lock files and reduce concurrency
rm -f *.lock
cargo test component_interactions -- --test-threads=1
```

## Troubleshooting Checklist

- [ ] All required imports in test file (tantivy::doc, tantivy::Index, etc.)
- [ ] Schema module exports get_schema function correctly
- [ ] tempfile dependency added to dev-dependencies
- [ ] No existing index lock files in project directory
- [ ] Sufficient system memory for index operations
- [ ] Test files have correct field name references
- [ ] Rust version supports all used tantivy features

## Recovery Procedures

### Schema Field Mismatch
If schema fields don't match test expectations:
1. Check schema.rs field definitions: `let content_field = schema_builder.add_text_field("content", TEXT | STORED);`
2. Verify field names are consistent: "content", "raw_content", "file_path"
3. Check field types match: TEXT for strings, U64 for numbers, BOOL for booleans
4. Run schema tests first: `cargo test schema`

### Index Writer Issues
If writer operations fail:
1. Increase writer memory: `index.writer(100_000_000)`
2. Check disk space: `df -h` or `dir`
3. Verify temp directory permissions
4. Use smaller test datasets for debugging

### Query Parser Problems
If query parsing fails:
1. Verify schema field exists: `schema.get_field("content")`
2. Check field is included in parser: `QueryParser::for_index(&index, vec![field])`
3. Test with simple queries first: `"test"`
4. Avoid special characters in test queries initially

## Success Validation Checklist

- [ ] File `tests/component_interactions.rs` exists with 7 test functions
- [ ] Command `cargo test component_interactions` shows all tests passing
- [ ] Schema-indexer compatibility test validates all 7 required fields
- [ ] Document round-trip preserves all field values correctly
- [ ] Query parser handles both single and multi-field searches
- [ ] Error handling tests complete without panics
- [ ] Performance tests complete within specified time limits

## Files Created For Next Task

After completing this task, you will have:

1. **C:/code/LLMKG/vectors/tantivy_search/tests/component_interactions.rs** - Comprehensive component interaction tests
2. **Validated component interfaces** - Confirmed all components work together correctly
3. **Performance baselines** - Established timing expectations for component operations

**Next Task (Task 23)** will implement a stress testing framework to validate system behavior under heavy load and extreme conditions.

## Context for Task 23

Task 23 will build upon the component interaction validation to create stress tests that push the system to its limits, testing behavior with thousands of documents, concurrent operations, and resource constraints.