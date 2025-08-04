# Task 21: Complete End-to-End Integration Test Workflow Implementation

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)  
**Prerequisites:** Task 20 completed (search caching system)  
**Input Files:** 
- `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs` (exists with all modules)
- `C:/code/LLMKG/vectors/tantivy_search/tests/` (directory exists)
- All schema, indexing, search modules from Tasks 01-20

## Complete Context (For AI with ZERO Knowledge)

**What is End-to-End Testing?** End-to-end (E2E) testing validates complete user workflows by testing the entire application stack from start to finish, ensuring all components work together correctly.

**What is Integration Testing?** Integration testing focuses on verifying that different modules or services work correctly when combined, catching issues that unit tests miss.

**Why This Task is Critical?** After implementing 20 individual components (schema, indexing, search, caching), we need to verify they work together as a complete system. This task creates comprehensive tests that simulate real user scenarios.

**The Complete Workflow We're Testing:**
1. **File Discovery** → Scan directories for code files
2. **Content Processing** → Parse and chunk files appropriately  
3. **Indexing** → Store chunks in Tantivy index with metadata
4. **Search** → Query the index with complex queries
5. **Result Processing** → Rank, highlight, and present results
6. **Caching** → Store and retrieve frequent searches efficiently

**Real-World Scenarios We Must Cover:**
- Mixed codebases (Rust + Python + Markdown)
- Large files requiring chunking
- Special characters in code (brackets, generics)
- Incremental updates (adding new files)
- Error handling (malformed files)
- Performance under load (20+ files)
- Cache hit/miss scenarios

## Exact Steps (6 minutes implementation)

### Step 1: Create comprehensive integration test file (4 minutes)

Create the file `C:/code/LLMKG/vectors/tantivy_search/tests/integration_workflow.rs` with this exact content:

```rust
use tantivy_search::*;
use tempfile::TempDir;
use std::fs;
use anyhow::Result;
use std::time::Instant;
use std::path::Path;

#[tokio::test]
async fn test_complete_indexing_and_search_workflow() -> Result<()> {
    // Setup test environment
    let temp_dir = TempDir::new()?;
    let test_files = create_test_codebase(&temp_dir)?;
    
    // 1. Create and configure pipeline
    let schema = get_schema();
    let index_path = temp_dir.path().join("index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    // 2. Index the test codebase
    let mut total_docs = 0;
    for file_path in &test_files {
        let content = fs::read_to_string(file_path)?;
        let doc = create_document(&schema, &content, file_path, 0)?;
        writer.add_document(doc)?;
        total_docs += 1;
    }
    writer.commit()?;
    
    assert!(total_docs > 0, "Should process test files");
    
    // 3. Create search components
    let reader = index.reader()?;
    let searcher = reader.searcher();
    
    assert!(searcher.num_docs() > 0, "Should have indexed documents");
    
    // 4. Test various search scenarios
    test_function_search(&index, &searcher).await?;
    test_content_search(&index, &searcher).await?;
    test_file_path_search(&index, &searcher).await?;
    
    Ok(())
}

fn create_test_codebase(temp_dir: &TempDir) -> Result<Vec<String>> {
    let files = vec![
        ("src/main.rs", r#"
            /// Main application entry point
            fn main() -> Result<(), Box<dyn std::error::Error>> {
                let config = load_configuration()?;
                start_server(config).await
            }
            
            /// Load application configuration
            fn load_configuration() -> Result<Config, ConfigError> {
                Config::from_file("config.toml")
            }
        "#),
        ("src/config.rs", r#"
            use serde::{Deserialize, Serialize};
            
            /// Application configuration structure
            #[derive(Debug, Serialize, Deserialize)]
            pub struct Config {
                pub server_port: u16,
                pub database_url: String,
                pub debug_mode: bool,
            }
            
            impl Config {
                pub fn from_file(path: &str) -> Result<Self, ConfigError> {
                    let content = std::fs::read_to_string(path)?;
                    toml::from_str(&content).map_err(Into::into)
                }
            }
        "#),
        ("src/database.py", r#"
            """Database connection and operations module."""
            
            import asyncpg
            from typing import List, Dict, Any
            
            class DatabaseManager:
                """Manages database connections and operations."""
                
                def __init__(self, connection_url: str):
                    self.connection_url = connection_url
                    self.pool = None
                
                async def connect(self) -> None:
                    """Establish database connection pool."""
                    self.pool = await asyncpg.create_pool(self.connection_url)
                
                async def execute_query(self, query: str, params: List[Any] = None) -> List[Dict]:
                    """Execute a database query and return results."""
                    async with self.pool.acquire() as connection:
                        return await connection.fetch(query, *(params or []))
        "#),
        ("README.md", r#"
            # Test Project
            
            This is a test project for the vector search system.
            It contains sample Rust and Python code to test indexing and search functionality.
            
            ## Features
            
            - Configuration management
            - Database operations
            - Server functionality
            
            ## Usage
            
            Run the application with:
            ```
            cargo run
            ```
        "#),
    ];
    
    fs::create_dir_all(temp_dir.path().join("src"))?;
    
    let mut created_files = Vec::new();
    for (path, content) in files {
        let full_path = temp_dir.path().join(path);
        fs::write(&full_path, content.trim())?;
        created_files.push(full_path.to_string_lossy().to_string());
    }
    
    Ok(created_files)
}

async fn test_function_search(index: &tantivy::Index, searcher: &tantivy::Searcher) -> Result<()> {
    use tantivy::query::QueryParser;
    use tantivy::collector::TopDocs;
    
    let schema = index.schema();
    let content_field = schema.get_field("content")?;
    let query_parser = QueryParser::for_index(index, vec![content_field]);
    
    let query = query_parser.parse_query("load_configuration")?;
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
    
    assert!(!top_docs.is_empty(), "Should find function references");
    
    // Verify result contains the function
    let (_, doc_address) = top_docs[0];
    let retrieved_doc = searcher.doc(doc_address)?;
    let content = retrieved_doc.get_first(content_field).unwrap().as_text().unwrap();
    assert!(content.contains("load_configuration"), "Should find load_configuration function");
    
    Ok(())
}

async fn test_content_search(index: &tantivy::Index, searcher: &tantivy::Searcher) -> Result<()> {
    use tantivy::query::QueryParser;
    use tantivy::collector::TopDocs;
    
    let schema = index.schema();
    let content_field = schema.get_field("content")?;
    let query_parser = QueryParser::for_index(index, vec![content_field]);
    
    // Test searching for configuration-related content
    let query = query_parser.parse_query("configuration")?;
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
    
    assert!(!top_docs.is_empty(), "Should find configuration content");
    
    Ok(())
}

async fn test_file_path_search(index: &tantivy::Index, searcher: &tantivy::Searcher) -> Result<()> {
    use tantivy::query::QueryParser;
    use tantivy::collector::TopDocs;
    
    let schema = index.schema();
    let file_path_field = schema.get_field("file_path")?;
    let query_parser = QueryParser::for_index(index, vec![file_path_field]);
    
    // Test searching by file path
    let query = query_parser.parse_query("main.rs")?;
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
    
    if !top_docs.is_empty() {
        let (_, doc_address) = top_docs[0];
        let retrieved_doc = searcher.doc(doc_address)?;
        let file_path = retrieved_doc.get_first(file_path_field).unwrap().as_text().unwrap();
        assert!(file_path.contains("main.rs"), "Should find main.rs file");
    }
    
    Ok(())
}

fn create_document(schema: &tantivy::Schema, content: &str, file_path: &str, chunk_index: u64) -> Result<tantivy::Document> {
    use tantivy::doc;
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    let chunk_start_field = schema.get_field("chunk_start")?;
    let chunk_end_field = schema.get_field("chunk_end")?;
    let has_overlap_field = schema.get_field("has_overlap")?;
    
    Ok(doc!(
        content_field => content,
        raw_content_field => content,
        file_path_field => file_path,
        chunk_index_field => chunk_index,
        chunk_start_field => 0u64,
        chunk_end_field => content.len() as u64,
        has_overlap_field => false
    ))
}

#[tokio::test]
async fn test_incremental_indexing_workflow() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let initial_files = create_test_codebase(&temp_dir)?;
    
    // Initial indexing
    let schema = get_schema();
    let index_path = temp_dir.path().join("index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    // Index initial files
    let mut initial_docs = 0;
    for file_path in &initial_files {
        let content = fs::read_to_string(file_path)?;
        let doc = create_document(&schema, &content, file_path, 0)?;
        writer.add_document(doc)?;
        initial_docs += 1;
    }
    writer.commit()?;
    
    // Add a new file
    let new_file_path = temp_dir.path().join("src/new_module.rs");
    fs::write(&new_file_path, r#"
        /// New module for testing incremental indexing
        pub fn new_function() -> String {
            "This is a new function".to_string()
        }
    "#)?;
    
    // Index new file
    let new_content = fs::read_to_string(&new_file_path)?;
    let new_doc = create_document(&schema, &new_content, new_file_path.to_str().unwrap(), 0)?;
    writer.add_document(new_doc)?;
    writer.commit()?;
    
    // Verify increased document count
    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert!(searcher.num_docs() > initial_docs, "Should have more documents after incremental update");
    
    Ok(())
}

#[tokio::test]
async fn test_search_performance_measurement() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let test_files = create_test_codebase(&temp_dir)?;
    
    // Setup index
    let schema = get_schema();
    let index_path = temp_dir.path().join("index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    // Index files
    for file_path in &test_files {
        let content = fs::read_to_string(file_path)?;
        let doc = create_document(&schema, &content, file_path, 0)?;
        writer.add_document(doc)?;
    }
    writer.commit()?;
    
    // Performance test
    let reader = index.reader()?;
    let searcher = reader.searcher();
    
    use tantivy::query::QueryParser;
    use tantivy::collector::TopDocs;
    
    let content_field = schema.get_field("content")?;
    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    
    // First search (measure performance)
    let start_time = Instant::now();
    let query = query_parser.parse_query("configuration")?;
    let results = searcher.search(&query, &TopDocs::with_limit(10))?;
    let search_time = start_time.elapsed();
    
    // Basic performance assertions
    assert!(search_time.as_millis() < 1000, "Search should complete within 1 second");
    assert!(searcher.num_docs() > 0, "Should have indexed documents");
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling_and_recovery() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Test with different file types including potentially problematic ones
    fs::create_dir_all(temp_dir.path().join("src"))?;
    fs::write(temp_dir.path().join("src/malformed.rs"), "invalid rust syntax {{{")?;
    fs::write(temp_dir.path().join("src/empty.rs"), "")?;
    fs::write(temp_dir.path().join("src/valid.rs"), "fn test() {}")?;
    
    let schema = get_schema();
    let index_path = temp_dir.path().join("index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    // Process files with error handling
    let mut processed_count = 0;
    let test_files = vec![
        temp_dir.path().join("src/malformed.rs"),
        temp_dir.path().join("src/empty.rs"),
        temp_dir.path().join("src/valid.rs"),
    ];
    
    for file_path in &test_files {
        match fs::read_to_string(file_path) {
            Ok(content) => {
                let doc = create_document(&schema, &content, file_path.to_str().unwrap(), 0)?;
                writer.add_document(doc)?;
                processed_count += 1;
            }
            Err(_) => {
                // Handle file read errors gracefully
                continue;
            }
        }
    }
    
    writer.commit()?;
    
    // Should process at least some files
    assert!(processed_count > 0, "Should process some valid files despite errors");
    
    // Search should still work
    let reader = index.reader()?;
    let searcher = reader.searcher();
    assert!(searcher.num_docs() > 0, "Should have some indexed documents");
    
    Ok(())
}

#[tokio::test]
async fn test_large_codebase_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    
    // Create a larger test codebase
    fs::create_dir_all(temp_dir.path().join("src"))?;
    
    let mut created_files = Vec::new();
    for i in 0..20 {
        let content = format!(r#"
            /// Module {} for testing large codebase handling
            pub mod module{} {{
                use std::collections::HashMap;
                
                /// Function {} that processes data
                pub fn process_data_{}(input: &str) -> Result<String, Box<dyn std::error::Error>> {{
                    let mut map = HashMap::new();
                    map.insert("key_{}", input);
                    Ok(format!("Processed: {{}}", input))
                }}
                
                /// Structure {} for data storage
                pub struct DataStore{} {{
                    pub data: HashMap<String, String>,
                    pub metadata: Vec<String>,
                }}
                
                impl DataStore{} {{
                    pub fn new() -> Self {{
                        Self {{
                            data: HashMap::new(),
                            metadata: Vec::new(),
                        }}
                    }}
                }}
            }}
        "#, i, i, i, i, i, i, i, i);
        
        let file_path = temp_dir.path().join(format!("src/module_{}.rs", i));
        fs::write(&file_path, content)?;
        created_files.push(file_path.to_string_lossy().to_string());
    }
    
    let schema = get_schema();
    let index_path = temp_dir.path().join("index");
    let index = create_index(&index_path)?;
    let mut writer = index.writer(50_000_000)?;
    
    // Index all files
    let mut total_docs = 0;
    for file_path in &created_files {
        let content = fs::read_to_string(file_path)?;
        let doc = create_document(&schema, &content, file_path, 0)?;
        writer.add_document(doc)?;
        total_docs += 1;
    }
    writer.commit()?;
    
    // Should handle large number of files
    assert!(total_docs >= 20, "Should process all test files");
    
    // Search should work efficiently
    let reader = index.reader()?;
    let searcher = reader.searcher();
    
    use tantivy::query::QueryParser;
    use tantivy::collector::TopDocs;
    
    let content_field = schema.get_field("content")?;
    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    
    let start_time = Instant::now();
    let query = query_parser.parse_query("process_data")?;
    let results = searcher.search(&query, &TopDocs::with_limit(10))?;
    let search_time = start_time.elapsed();
    
    assert!(!results.is_empty(), "Should find results in large codebase");
    assert!(search_time.as_millis() < 1000, "Search should be reasonably fast");
    
    Ok(())
}
```

### Step 2: Add required dependencies (1 minute)

Add to `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml` under `[dev-dependencies]`:

```toml
tempfile = "3.8"
anyhow = "1.0"
tokio = { version = "1.0", features = ["full"] }
```

### Step 3: Update lib.rs exports (1 minute)

Ensure `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs` contains:

```rust
pub mod schema;
pub use schema::{get_schema, create_index, open_or_create_index};
```

## Verification Steps (2 minutes)

```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo test integration_workflow
```

**Expected output:**
```
running 6 tests
test test_complete_indexing_and_search_workflow ... ok
test test_incremental_indexing_workflow ... ok
test test_search_performance_measurement ... ok
test test_error_handling_and_recovery ... ok
test test_large_codebase_handling ... ok

test result: ok. 6 passed; 0 failed; 0 ignored
```

## If This Task Fails

### Common Errors and Solutions

**Error 1: "error[E0433]: failed to resolve: use of undeclared type `TempDir`"**
```bash
# Solution: Missing dev dependency
cargo add --dev tempfile
cargo test integration_workflow
```

**Error 2: "error[E0599]: no method named `get_schema` found"**
```bash
# Solution: Schema module not properly exported
# Verify lib.rs contains: pub use schema::{get_schema, create_index};
cargo check
cargo test integration_workflow
```

**Error 3: "Permission denied" creating temporary directories**
```bash
# Solution (Windows): Fix temp directory permissions
set TEMP=C:\temp
mkdir C:\temp
icacls C:\temp /grant Users:F /T

# Solution (Unix): Use different temp location
export TMPDIR=/tmp/tantivy_test
mkdir -p /tmp/tantivy_test
```

**Error 4: "Index creation failed: LockError"**
```bash
# Solution: Clean up existing lock files
find . -name "*.lock" -delete
cargo clean
cargo test integration_workflow
```

## Troubleshooting Checklist

- [ ] Rust version 1.70+ installed (`rustc --version`)
- [ ] All dependencies in Cargo.toml (tantivy, tempfile, anyhow, tokio)
- [ ] Schema module properly exported from lib.rs
- [ ] No syntax errors in integration test file
- [ ] Sufficient disk space for temporary index creation
- [ ] Write permissions in system temp directory
- [ ] No existing index lock files interfering

## Recovery Procedures

### Tantivy Index Creation Failure
If index creation consistently fails:
1. Check available disk space: `df -h` (Unix) or `dir C:\` (Windows)
2. Verify temp directory permissions
3. Try manual temp directory: `TempDir::new_in("/tmp")`
4. Enable Tantivy debug logs: `RUST_LOG=tantivy=debug cargo test`

### Test Timeout Issues
If tests hang or timeout:
1. Reduce test data size in `create_test_codebase`
2. Increase writer memory: `index.writer(10_000_000)`
3. Run single test: `cargo test test_complete_indexing_and_search_workflow`
4. Check system resources: `top` or Task Manager

### Query Parser Errors
If query parsing fails:
1. Verify schema field names match exactly
2. Check field types (TEXT vs STRING)
3. Test with simple queries first: `"test"`
4. Enable query parser debug: Add `println!` statements

## Success Validation Checklist

- [ ] File `tests/integration_workflow.rs` exists with exactly 6 test functions
- [ ] Command `cargo test integration_workflow` shows 6 tests passing
- [ ] All tests complete within 30 seconds total
- [ ] Test creates and searches temporary indexes successfully  
- [ ] Large codebase test (20 files) completes without memory issues
- [ ] Error handling test processes malformed files gracefully
- [ ] Search performance test completes under 1 second per query

## Files Created For Next Task

After completing this task, you will have:

1. **C:/code/LLMKG/vectors/tantivy_search/tests/integration_workflow.rs** - Complete end-to-end integration tests
2. **Updated Cargo.toml** - Added dev dependencies for testing
3. **Verified working system** - All components integrated and tested

**Next Task (Task 22)** will focus on component interaction testing, specifically validating how individual modules communicate and handle edge cases in their interfaces.

## Context for Task 22

Task 22 will build upon this end-to-end validation to create focused interaction tests between specific components like schema-indexer communication, indexer-searcher data flow, and query parser-search engine integration.