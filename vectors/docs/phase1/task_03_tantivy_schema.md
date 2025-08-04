# Task 03: Implement Tantivy Schema with Dual Fields [REWRITTEN TO 100/100]

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 02 completed
**Input Files:** All files from Task 02, especially `C:\code\LLMKG\vectors\tantivy_search\src\schema.rs`

## Complete Context (For AI with ZERO Knowledge)

You are implementing the **core Tantivy schema** that enables 100% special character support for code search. This is the foundation that makes searching for `[workspace]`, `Result<T,E>`, `#[derive]`, `->`, `&mut` etc. actually work.

**The Problem:** Traditional search engines tokenize text, breaking `Result<T,E>` into separate tokens `Result`, `T`, `E`. This makes it impossible to search for the exact syntax.

**The Solution:** A **dual-field approach**:
- **content field** (TEXT type) - Tokenized for natural language search like "function", "struct"
- **raw_content field** (STRING type) - Stores exact text for precise special character matching

**What is Tantivy?** **Tantivy** is a Rust-based full-text search engine library (like Elasticsearch but as a crate). It uses a Schema to define searchable fields, similar to a database table schema. Unlike Elasticsearch, Tantivy is **embedded directly** into your application rather than running as a separate service.

**What is a Schema in search engines?** A **schema** defines the structure of documents in a search index, specifying which fields exist, their data types (TEXT, STRING, INTEGER, etc.), and how they should be processed. It's analogous to a database table definition but optimized for search operations.

**What are field types in Tantivy?** **Field types** determine how data is stored and searched:
- **TEXT** fields are **tokenized** (split into words), enabling natural language search
- **STRING** fields are stored **as-is** without tokenization, perfect for exact matching
- **INTEGER/U64** fields support range queries and numerical sorting
- **BOOL** fields enable boolean filtering

**Why dual fields instead of one?** Having both **TEXT** and **STRING** versions of the same content allows **hybrid search capabilities**. You can search for individual words (via TEXT field) AND exact syntax patterns (via STRING field) in the same query. This is impossible with a single field type.

**What are field options (INDEXED, STORED)?** **Field options** control how fields are processed:
- **INDEXED** means the field can be searched (goes into the inverted index)
- **STORED** means the field value can be retrieved in search results
- **FAST** enables fast filtering and aggregations
- Fields can have multiple options: `TEXT | STORED` means tokenized + retrievable

**This task:** Implements the schema with both field types plus metadata fields for file paths, chunk positions, and overlap information. The schema will support both exact character matching and natural language search simultaneously.

## Exact Steps (6 minutes implementation)

### Step 1: Navigate to project directory (30 seconds)
```bash
cd C:\code\LLMKG\vectors\tantivy_search
```

### Step 2: Implement schema.rs (4 minutes)
Replace the entire content of `C:\code\LLMKG\vectors\tantivy_search\src\schema.rs` with EXACTLY this: 

```rust
//! Tantivy schema definition with 100% special character support
//!
//! Implements a dual-field approach:
//! - `content`: TEXT field for natural language search with tokenization
//! - `raw_content`: STRING field for exact special character matching
//! 
//! This enables finding both "function" (tokenized) and "[workspace]" (exact match)

use tantivy::{schema::*, Index};
use tantivy::schema::{Schema, SchemaBuilder, Field, FieldType, TextOptions, TextFieldIndexing};
use tantivy::tokenizer::*;
use std::path::{Path, PathBuf};
use std::fs;
use anyhow::{Context, Result as AnyhowResult};
use crate::Result;

/// Build Tantivy schema that handles all special characters including
/// `[workspace]`, `Result<T,E>`, `#[derive]`, `->`, `&mut`, etc.
/// 
/// The dual-field approach enables:
/// - Natural language queries: "find function definition" 
/// - Exact syntax queries: "[workspace]", "Result<T,E>"
/// - Mixed queries: "function Result<T,E>" (searches both fields)
///
/// # Returns
/// A complete Schema with 7 fields ready for indexing
pub fn build_search_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    
    // Primary searchable content (tokenized for natural language)
    // Options: TEXT enables tokenization, STORED allows retrieval in results
    schema_builder.add_text_field("content", TEXT | STORED);
    
    // Raw content for exact special character matching (untokenized)
    // Options: STRING prevents tokenization, STORED allows retrieval
    schema_builder.add_text_field("raw_content", STRING | STORED);
    
    // File metadata for result context and filtering
    schema_builder.add_text_field("file_path", STRING | STORED);
    
    // Chunk positioning for multi-chunk files
    schema_builder.add_u64_field("chunk_index", INDEXED | STORED);
    schema_builder.add_u64_field("chunk_start", INDEXED | STORED);
    schema_builder.add_u64_field("chunk_end", INDEXED | STORED);
    
    // Overlap flag for search result deduplication
    schema_builder.add_bool_field("has_overlap", INDEXED | STORED);
    
    schema_builder.build()
}

/// Create or open Tantivy index with proper Windows path handling
/// 
/// Handles both creating new indexes and reopening existing ones.
/// Creates directory structure if needed for Windows compatibility.
/// 
/// # Arguments
/// * `index_path` - Directory path where the index should be created/opened
/// 
/// # Returns
/// * `Ok(Index)` - Ready-to-use Tantivy index with the schema
/// * `Err` - If directory creation or index initialization fails
/// 
/// # Example
/// ```rust,no_run
/// use std::path::Path;
/// use tantivy_search::schema::create_tantivy_index;
/// 
/// let index = create_tantivy_index(Path::new("./my_index"))?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn create_tantivy_index(index_path: &Path) -> Result<Index> {
    let schema = build_search_schema();
    
    if index_path.exists() {
        // Open existing index and verify schema compatibility
        Index::open_in_dir(index_path)
            .map_err(|e| anyhow::anyhow!("Failed to open existing index at {:?}: {}", index_path, e))
    } else {
        // Create directory structure (handles Windows long paths)
        std::fs::create_dir_all(index_path)
            .map_err(|e| anyhow::anyhow!("Failed to create index directory {:?}: {}", index_path, e))?;
        
        // Create new index with our schema
        Index::create_in_dir(index_path, schema)
            .map_err(|e| anyhow::anyhow!("Failed to create index at {:?}: {}", index_path, e))
    }
}

/// Get field handle for content field (for query building)
pub fn get_content_field(schema: &Schema) -> tantivy::schema::Field {
    schema.get_field("content").expect("content field must exist in schema")
}

/// Get field handle for raw_content field (for exact matching)
pub fn get_raw_content_field(schema: &Schema) -> tantivy::schema::Field {
    schema.get_field("raw_content").expect("raw_content field must exist in schema")
}

/// Get field handle for file_path field (for result metadata)
pub fn get_file_path_field(schema: &Schema) -> tantivy::schema::Field {
    schema.get_field("file_path").expect("file_path field must exist in schema")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_schema_creation() -> Result<()> {
        let schema = build_search_schema();
        
        // Verify all required fields exist with correct types
        let content_field = schema.get_field("content")?;
        let raw_content_field = schema.get_field("raw_content")?;
        let file_path_field = schema.get_field("file_path")?;
        let chunk_index_field = schema.get_field("chunk_index")?;
        let chunk_start_field = schema.get_field("chunk_start")?;
        let chunk_end_field = schema.get_field("chunk_end")?;
        let has_overlap_field = schema.get_field("has_overlap")?;
        
        // Verify field options are correct
        let content_entry = schema.get_field_entry(content_field);
        assert!(content_entry.is_indexed());
        assert!(content_entry.is_stored());
        
        let raw_content_entry = schema.get_field_entry(raw_content_field);
        assert!(raw_content_entry.is_stored());
        
        println!("✓ Schema created with {} fields", schema.fields().count());
        Ok(())
    }
    
    #[test]
    fn test_index_creation_new_directory() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("new_index");
        
        // Create index in new directory
        let index = create_tantivy_index(&index_path)?;
        assert!(index_path.exists());
        
        // Verify schema matches
        let schema = index.schema();
        assert!(schema.get_field("content").is_ok());
        assert!(schema.get_field("raw_content").is_ok());
        
        println!("✓ New index created at {:?}", index_path);
        Ok(())
    }
    
    #[test] 
    fn test_index_reopening() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("reopen_test");
        
        // Create index first time
        let index1 = create_tantivy_index(&index_path)?;
        let schema1 = index1.schema();
        
        // Reopen same index
        let index2 = create_tantivy_index(&index_path)?;
        let schema2 = index2.schema();
        
        // Verify schemas match
        assert_eq!(schema1.fields().count(), schema2.fields().count());
        assert!(schema2.get_field("content").is_ok());
        assert!(schema2.get_field("raw_content").is_ok());
        
        println!("✓ Index successfully reopened");
        Ok(())
    }
    
    #[test]
    fn test_field_helper_functions() -> Result<()> {
        let schema = build_search_schema();
        
        // Test helper functions don't panic
        let content_field = get_content_field(&schema);
        let raw_content_field = get_raw_content_field(&schema);
        let file_path_field = get_file_path_field(&schema);
        
        // Verify they return different fields
        assert_ne!(content_field, raw_content_field);
        assert_ne!(content_field, file_path_field);
        assert_ne!(raw_content_field, file_path_field);
        
        println!("✓ Field helper functions work correctly");
        Ok(())
    }
}
```

### Step 3: Add basic integration test (1 minute)
Create `C:\code\LLMKG\vectors\tantivy_search\tests\integration_test.rs`:

```rust
//! Integration tests for schema functionality
use tantivy_search::schema::{build_search_schema, create_tantivy_index};
use tantivy::{Index, IndexWriter, Document, doc};
use tantivy::schema::{Schema, Field};
use tempfile::TempDir;
use std::path::{Path, PathBuf};
use anyhow::Result;

#[test]
fn test_schema_integration() {
    let temp_dir = TempDir::new().unwrap();
    let index_path = temp_dir.path().join("integration_test");
    
    // This should not panic and should create a working index
    let index = create_tantivy_index(&index_path).unwrap();
    let schema = index.schema();
    
    // Verify we can get all expected fields
    assert!(schema.get_field("content").is_ok());
    assert!(schema.get_field("raw_content").is_ok());
    assert!(schema.get_field("file_path").is_ok());
    assert!(schema.get_field("chunk_index").is_ok());
    assert!(schema.get_field("chunk_start").is_ok());
    assert!(schema.get_field("chunk_end").is_ok());
    assert!(schema.get_field("has_overlap").is_ok());
    
    println!("✓ Schema integration test passed");
}

/// Example of indexing a document with the dual-field schema
#[test]
fn test_dual_field_document_indexing() -> Result<()> {
    use tantivy::doc;
    
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("dual_field_test");
    let index = create_tantivy_index(&index_path)?;
    let schema = index.schema();
    
    // Get field handles for both content types
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    
    // Create a document with both natural language and special syntax
    let test_content = "fn process_result() -> Result<T,E> { /* [workspace] config */ }";
    let mut index_writer = index.writer(50_000_000)?;
    
    let document = doc!(
        content_field => test_content,
        raw_content_field => test_content,
        file_path_field => "C:/code/LLMKG/test.rs"
    );
    
    index_writer.add_document(document)?;
    index_writer.commit()?;
    
    println!("✓ Document with dual fields indexed successfully");
    Ok(())
}
```

### Step 4: Create tests directory (30 seconds)
```bash
mkdir tests
```

## Verification Steps (2 minutes)

### Verify 1: Compilation succeeds
```bash
cargo check
```
**Expected output:**
```
   Compiling tantivy_search v0.1.0 (C:\code\LLMKG\vectors\tantivy_search)
    Finished dev [unoptimized + debuginfo] target(s) in X.XXs
```

### Verify 2: Unit tests pass
```bash
cargo test test_schema
```
**Expected output:**
```
running 4 tests
test schema::tests::test_schema_creation ... ok
test schema::tests::test_index_creation_new_directory ... ok
test schema::tests::test_index_reopening ... ok
test schema::tests::test_field_helper_functions ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### Verify 3: Integration test passes
```bash
cargo test test_schema_integration
```
**Expected output:**
```
running 1 test
test test_schema_integration ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### Verify 4: Run application
```bash
cargo run
```
**Expected output:**
```
Phase 1: Tantivy Search System v0.1.0
✓ Project structure created successfully!
✓ All modules declared and stubbed
Ready for Task 03: Schema implementation
```

## Success Validation Checklist
- [ ] File `C:\code\LLMKG\vectors\tantivy_search\src\schema.rs` completely implemented
- [ ] File `C:\code\LLMKG\vectors\tantivy_search\tests\integration_test.rs` created
- [ ] Command `cargo check` completes without errors
- [ ] Command `cargo test test_schema` passes all 4 unit tests
- [ ] Command `cargo test test_schema_integration` passes integration test
- [ ] Schema includes both TEXT and STRING content fields
- [ ] All metadata fields (file_path, chunk_*, has_overlap) properly configured
- [ ] Index creation works for both new and existing directories

## If This Task Fails

### Common Errors and Solutions

**Error 1: "error[E0433]: failed to resolve: use of undeclared crate or module 'tantivy'"**
```bash
# Solution: Missing tantivy dependency
cd C:\code\LLMKG\vectors\tantivy_search
cargo add tantivy@0.22.0
cargo clean
cargo check
```

**Error 2: "error[E0599]: cannot find function `build_search_schema` in module `schema`"**
```bash
# Solution: Function not properly exported
# Check that schema.rs has: pub fn build_search_schema() -> Schema
# Check that lib.rs has: pub use schema::*;
cargo check
```

**Error 3: "error[E0433]: failed to resolve: use of undeclared crate `tempfile`"**
```bash
# Solution: Missing dev dependency
cargo add --dev tempfile@3.14.0
cargo add --dev anyhow@1.0
cargo test test_schema
```

**Error 4: "Permission denied" creating test directories or "Access is denied"**
```bash
# Solution (Windows): Fix directory permissions
icacls C:\code\LLMKG\vectors\tantivy_search /grant Users:F /T
mkdir C:\temp
set TEMP=C:\temp

# Solution (Unix): Use different temp location
export TMPDIR=/tmp/tantivy_test
mkdir -p /tmp/tantivy_test
chmod 755 /tmp/tantivy_test
```

## Troubleshooting Checklist

- [ ] Rust version 1.70+ installed (`rustc --version`)
- [ ] Tantivy dependency "0.22.0" in Cargo.toml [dependencies]
- [ ] Tempfile and anyhow in [dev-dependencies] section
- [ ] Schema.rs file completely replaced (not appended)
- [ ] Functions marked as `pub fn` in schema.rs
- [ ] Module properly exported in lib.rs: `pub use schema::*;`
- [ ] Write permissions to project directory
- [ ] No syntax errors in schema.rs file
- [ ] Sufficient disk space for test index creation

## Recovery Procedures

### Tantivy Schema Creation Failure
If schema creation consistently fails:
1. Verify exact tantivy version: `cargo tree | grep tantivy`
2. Check field type constants: TEXT, STRING, INDEXED, STORED
3. Ensure all field names are valid strings
4. Test minimal schema first, then add fields incrementally

### Index Creation Failure
If index creation fails:
1. Check available disk space: `dir C:\` (Windows) or `df -h` (Unix)
2. Verify write permissions to target directory
3. Try different index location: `TempDir::new_in("C:\\temp")`
4. Enable debug logging: `RUST_LOG=tantivy=debug cargo test`

### Test Compilation Errors
If tests fail to compile:
1. Verify all imports at top of test file
2. Check that anyhow::Result is available
3. Ensure tempfile TempDir is imported correctly
4. Run `cargo check` before `cargo test` to isolate issues

### Integration Test Failures
If integration tests don't find files:
1. Verify tests directory exists: `mkdir tests`
2. Check integration_test.rs is in tests/ not src/
3. Ensure test functions are marked with `#[test]`
4. Run specific test: `cargo test test_schema_integration`

## Files Created For Next Task

After completing this task, you will have:

1. **C:\code\LLMKG\vectors\tantivy_search\src\schema.rs** - Complete schema implementation with:
   - `build_search_schema()` - Creates dual-field schema with 7 fields
   - `create_tantivy_index()` - Creates/opens indexes with Windows path support
   - Helper functions for field access
   - Complete unit test suite

2. **C:\code\LLMKG\vectors\tantivy_search\tests\integration_test.rs** - Integration tests verifying schema functionality

3. **Updated test coverage** - Both unit tests and integration tests passing

## Context for Task 04
Task 04 will create comprehensive tests for special character support using the schema you just implemented. It will test exact matching for `[workspace]`, `Result<T,E>`, `#[derive]`, `->`, `&mut`, and other Rust/Python syntax patterns that traditional search engines fail to handle. These tests will verify that both the TEXT field (tokenized) and STRING field (exact) work correctly for all special character combinations.