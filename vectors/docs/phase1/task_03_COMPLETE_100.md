# Task 03: Implement Tantivy Schema with Dual Fields for Special Character Support

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)  
**Prerequisites:** Task 02 completed (module structure created)  
**Input Files:** 
- `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs` (exists with module declarations)
- `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml` (has tantivy = "0.22.0")

## Complete Context (For AI with ZERO Knowledge)

**What is Tantivy?** Tantivy is a full-text search engine library written in Rust, similar to Apache Lucene but designed for Rust applications. It provides fast indexing and searching of text documents.

**What is a Schema?** In Tantivy, a schema defines the structure of documents in your search index. It specifies what fields each document has and how those fields should be indexed and stored.

**Why Dual Fields?** We're implementing a dual-field approach to handle special characters in code:
- **content field** (TEXT type): Tokenized and processed for normal search queries
- **raw_content field** (STRING type): Stores exact text for special character matching

This dual approach allows searching for programming syntax like `[workspace]`, `Result<T,E>`, `#[derive(Debug)]` exactly as they appear in code, while also supporting normal text search.

## Exact Steps (6 minutes implementation)

### Step 1: Create schema.rs file (2 minutes)

Create the file `C:/code/LLMKG/vectors/tantivy_search/src/schema.rs` with this exact content:

```rust
use tantivy::schema::*;
use tantivy::{Index, Result};
use std::path::Path;

/// Creates the Tantivy schema with dual fields for special character support
pub fn get_schema() -> Schema {
    let mut schema_builder = Schema::builder();
    
    // Primary content field - tokenized for normal search
    // TEXT: Tokenizes the content into searchable terms
    // STORED: Keeps original content for retrieval
    schema_builder.add_text_field("content", TEXT | STORED);
    
    // Raw content field - exact string matching for special characters
    // STRING: No tokenization, exact matching only
    // STORED: Keeps original for retrieval
    schema_builder.add_text_field("raw_content", STRING | STORED);
    
    // File path - where this content came from
    schema_builder.add_text_field("file_path", STRING | STORED);
    
    // Chunk index - position in the original file (for chunked documents)
    schema_builder.add_u64_field("chunk_index", INDEXED | STORED);
    
    // Chunk boundaries for reconstruction
    schema_builder.add_u64_field("chunk_start", INDEXED | STORED);
    schema_builder.add_u64_field("chunk_end", INDEXED | STORED);
    
    // Metadata flag for overlapping chunks
    schema_builder.add_bool_field("has_overlap", INDEXED | STORED);
    
    schema_builder.build()
}

/// Creates a new Tantivy index with our schema at the specified path
pub fn create_index(index_path: &Path) -> Result<Index> {
    let schema = get_schema();
    
    // Ensure directory exists
    std::fs::create_dir_all(index_path)?;
    
    // Create index with schema
    Index::create_in_dir(index_path, schema)
}

/// Opens an existing index or creates a new one
pub fn open_or_create_index(index_path: &Path) -> Result<Index> {
    if index_path.exists() && index_path.join("meta.json").exists() {
        // Index exists, open it
        Index::open_in_dir(index_path)
    } else {
        // Create new index
        create_index(index_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_schema_has_all_fields() {
        let schema = get_schema();
        
        // Verify all required fields exist
        assert!(schema.get_field("content").is_ok());
        assert!(schema.get_field("raw_content").is_ok());
        assert!(schema.get_field("file_path").is_ok());
        assert!(schema.get_field("chunk_index").is_ok());
        assert!(schema.get_field("chunk_start").is_ok());
        assert!(schema.get_field("chunk_end").is_ok());
        assert!(schema.get_field("has_overlap").is_ok());
    }
    
    #[test]
    fn test_create_index() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("test_index");
        
        let index = create_index(&index_path)?;
        
        // Verify index was created with correct schema
        let schema = index.schema();
        assert_eq!(schema.fields().count(), 7);
        
        Ok(())
    }
    
    #[test]
    fn test_dual_field_types() {
        let schema = get_schema();
        
        let content_field = schema.get_field("content").unwrap();
        let raw_content_field = schema.get_field("raw_content").unwrap();
        
        let content_entry = schema.get_field_entry(content_field);
        let raw_entry = schema.get_field_entry(raw_content_field);
        
        // Verify field types match our dual-field strategy
        match content_entry.field_type() {
            FieldType::Str(ref options) => {
                assert!(options.get_indexing_options().is_some());
            }
            _ => panic!("content field should be text type"),
        }
        
        match raw_entry.field_type() {
            FieldType::Str(ref options) => {
                // STRING type has different indexing options
                assert!(options.get_indexing_options().is_some());
            }
            _ => panic!("raw_content field should be string type"),
        }
    }
}
```

### Step 2: Update lib.rs to export schema module (2 minutes)

Add this line to `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs`:

```rust
pub mod schema;
pub use schema::{get_schema, create_index, open_or_create_index};
```

### Step 3: Create integration test (2 minutes)

Create file `C:/code/LLMKG/vectors/tantivy_search/tests/schema_integration.rs`:

```rust
use tantivy_search::*;
use tempfile::TempDir;
use tantivy::{doc, Index, IndexWriter};

#[test]
fn test_special_character_indexing() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let index = create_index(temp_dir.path())?;
    let schema = index.schema();
    
    // Get field handles
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    
    // Create index writer
    let mut writer: IndexWriter = index.writer(50_000_000)?;
    
    // Test special character content
    let special_content = "[workspace]\nmembers = [\"backend\"]\nResult<T, E>";
    
    // Add document with special characters
    writer.add_document(doc!(
        content_field => special_content,
        raw_content_field => special_content,
        file_path_field => "Cargo.toml"
    ))?;
    
    writer.commit()?;
    
    // Verify document was indexed
    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert_eq!(searcher.num_docs(), 1);
    
    Ok(())
}
```

## Verification Steps (2 minutes)

```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo test schema
```

**Expected output:**
```
running 4 tests
test schema::tests::test_schema_has_all_fields ... ok
test schema::tests::test_create_index ... ok
test schema::tests::test_dual_field_types ... ok
test schema_integration::test_special_character_indexing ... ok

test result: ok. 4 passed; 0 failed; 0 ignored
```

## If This Task Fails

### Common Errors and Solutions

**Error 1: "error[E0433]: failed to resolve: use of undeclared type `TempDir`"**
```bash
# Solution: Missing dev dependency
cargo add --dev tempfile
cargo test schema
```

**Error 2: "Permission denied" when creating index directory**
```bash
# Solution (Windows): Fix permissions
icacls C:/code/LLMKG/vectors/tantivy_search /grant Users:F /T

# Solution (Unix): Fix permissions
chmod -R 755 C:/code/LLMKG/vectors/tantivy_search
```

**Error 3: "could not compile `tantivy_search`"**
```bash
# Solution: Version mismatch or cache issue
cargo clean
cargo update
cargo build --release
```

**Error 4: "error[E0432]: unresolved import `tantivy`"**
```bash
# Solution: Dependency not downloaded
cargo fetch
cargo build
```

## Troubleshooting Checklist

- [ ] Rust version 1.70+ installed (`rustc --version`)
- [ ] Tantivy 0.22.0 in Cargo.toml dependencies
- [ ] File path `src/schema.rs` created correctly
- [ ] No syntax errors in code (check with `cargo check`)
- [ ] Internet connection for downloading crates
- [ ] Sufficient disk space for index creation
- [ ] File permissions allow read/write operations

## Recovery Procedures

### Network Failure During Crate Download
If crates.io is unreachable:
1. Check internet: `ping crates.io`
2. Try proxy: `export HTTPS_PROXY=http://proxy:8080`
3. Use vendored deps: `cargo vendor && cargo build --offline`
4. Use mirror: Add to `.cargo/config.toml`:
   ```toml
   [source.crates-io]
   replace-with = "mirror"
   [source.mirror]
   registry = "https://mirrors.ustc.edu.cn/crates.io-index/"
   ```

### Index Creation Failure
If index creation fails:
1. Check disk space: `df -h` (Unix) or `fsutil volume diskfree C:` (Windows)
2. Verify path exists: `ls -la C:/code/LLMKG/vectors/tantivy_search`
3. Test with temp directory: Change path to `/tmp/test_index`
4. Check Tantivy logs: `RUST_LOG=tantivy=debug cargo test`

### Test Execution Failure
If tests fail to run:
1. Run single test: `cargo test test_schema_has_all_fields`
2. Run with output: `cargo test schema -- --nocapture`
3. Check for panics: `RUST_BACKTRACE=1 cargo test`
4. Verify setup: `cargo test --doc`

## Success Validation Checklist

- [ ] File `src/schema.rs` exists with exactly 98 lines
- [ ] Command `cargo test schema` shows 4 tests passing
- [ ] Schema has 7 fields (content, raw_content, file_path, chunk_index, chunk_start, chunk_end, has_overlap)
- [ ] Both TEXT and STRING field types are present
- [ ] Index creation succeeds in temp directory
- [ ] Special characters can be indexed without errors

## Files Created For Next Task

After completing this task, you will have:

1. **C:/code/LLMKG/vectors/tantivy_search/src/schema.rs** - Complete schema implementation with dual fields
2. **C:/code/LLMKG/vectors/tantivy_search/tests/schema_integration.rs** - Integration tests for special characters
3. **Updated lib.rs** - Exports schema module functions

**Next Task (Task 04)** will use this schema to create a comprehensive test suite for all special character patterns like `[workspace]`, `Result<T,E>`, `#[derive(Debug)]`, ensuring they can be indexed and searched correctly.

## Context for Task 04

Task 04 will build upon this schema implementation to create exhaustive tests for special character handling. The dual-field approach established here (content + raw_content) will be validated with real-world code patterns to ensure the search system can handle all Rust syntax correctly.