# Task 00_2: Create Basic Vector-Search Crate Structure

**Estimated Time:** 6-8 minutes  
**Prerequisites:** Task 00_1 (workspace member added)  
**Dependencies:** Must be completed before Task 00_3

## Objective
Create the basic directory structure and Cargo.toml for the vector-search crate, establishing the foundation module layout that Phase 2 tasks expect.

## Context
You are creating the actual vector-search crate that was added to the workspace in Task 00_1. This crate will provide full-text search capabilities using Tantivy, with AST-based smart chunking for code files. The structure must align with the existing neuromorphic codebase patterns.

## Task Details

### What You Need to Do
1. **Create the crate directory structure:**
   ```
   crates/vector-search/
   ├── Cargo.toml
   ├── src/
   │   ├── lib.rs
   │   ├── indexing/
   │   │   └── mod.rs
   │   ├── search/
   │   │   └── mod.rs
   │   └── chunking/
   │       └── mod.rs
   ```

2. **Create Cargo.toml for the vector-search crate:**
   ```toml
   [package]
   name = "vector-search"
   version.workspace = true
   authors.workspace = true
   edition.workspace = true
   license.workspace = true
   
   [dependencies]
   # Workspace dependencies
   anyhow.workspace = true
   thiserror.workspace = true
   tokio.workspace = true
   tracing.workspace = true
   serde.workspace = true
   serde_json.workspace = true
   uuid.workspace = true
   
   # Vector search specific
   tantivy.workspace = true
   tree-sitter.workspace = true
   tree-sitter-rust.workspace = true
   tree-sitter-python.workspace = true
   walkdir.workspace = true
   regex.workspace = true
   
   [dev-dependencies]
   tokio-test.workspace = true
   tempfile = "3.8.0"
   ```

3. **Create src/lib.rs with basic module structure:**
   ```rust
   //! Vector search system for the neuromorphic neural network
   //! 
   //! Provides full-text search capabilities with AST-based smart chunking
   //! for integration with cortical column allocation and neural processing.
   
   pub mod indexing;
   pub mod search;
   pub mod chunking;
   
   #[cfg(test)]
   pub mod test_utils;
   
   pub use indexing::{DocumentIndexer, IndexingError};
   pub use search::{SearchEngine, SearchError, SearchResult};
   pub use chunking::{SmartChunker, ChunkingError, CodeChunk};
   
   /// Vector search system errors
   #[derive(thiserror::Error, Debug)]
   pub enum VectorSearchError {
       #[error("Indexing error: {0}")]
       Indexing(#[from] IndexingError),
       
       #[error("Search error: {0}")]
       Search(#[from] SearchError),
       
       #[error("Chunking error: {0}")]
       Chunking(#[from] ChunkingError),
   }
   
   pub type Result<T> = std::result::Result<T, VectorSearchError>;
   ```

4. **Create placeholder module files:**
   - `src/indexing/mod.rs`: Basic DocumentIndexer and IndexingError types
   - `src/search/mod.rs`: Basic SearchEngine and SearchError types  
   - `src/chunking/mod.rs`: Basic SmartChunker and ChunkingError types
   - `src/test_utils.rs`: Standardized test utilities with consistent schema definitions

5. **Create standardized test utilities (src/test_utils.rs):**
   This module provides consistent schema definitions and test helpers to prevent code duplication across tests. 
   It includes:
   - `StandardSchema`: Unified schema with file_path, content, raw_content, and chunk_index fields
   - `TestIndexBuilder`: Helper for creating test indices with standard configuration
   - `TestDataGenerator`: Common test data patterns for Rust code, special characters, and large documents
   - `CompilationVerifier`: Helpers to verify crate compilation and test execution

### Expected Output Files
- **Created:** `crates/vector-search/Cargo.toml`
- **Created:** `crates/vector-search/src/lib.rs`
- **Created:** `crates/vector-search/src/indexing/mod.rs`  
- **Created:** `crates/vector-search/src/search/mod.rs`
- **Created:** `crates/vector-search/src/chunking/mod.rs`
- **Created:** `crates/vector-search/src/test_utils.rs`

## Success Criteria
- [ ] Directory structure matches neuromorphic crate patterns
- [ ] Cargo.toml properly references workspace dependencies
- [ ] lib.rs compiles without errors
- [ ] Module structure supports Phase 2 task requirements
- [ ] `cargo check -p vector-search` succeeds
- [ ] Placeholder types exist for all major components
- [ ] Test utilities module compiles and basic tests pass
- [ ] `cargo test -p vector-search test_standard_schema_creation` succeeds

## Module Implementation Details

### src/indexing/mod.rs
```rust
//! Document indexing capabilities for vector search

use thiserror::Error;

#[derive(Error, Debug)]
pub enum IndexingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Tantivy error: {0}")]
    Tantivy(String),
}

/// Document indexer for creating searchable indices
pub struct DocumentIndexer {
    // Implementation in Task 00_4
}

impl DocumentIndexer {
    pub fn new() -> Result<Self, IndexingError> {
        todo!("Implementation in Task 00_4")
    }
}
```

### src/search/mod.rs
```rust
//! Search engine implementation for vector search

use thiserror::Error;

#[derive(Error, Debug)]
pub enum SearchError {
    #[error("Query parsing error: {0}")]
    QueryParsing(String),
    
    #[error("Index error: {0}")]
    Index(String),
}

/// Search result structure
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub score: f32,
    pub content: String,
    pub file_path: String,
}

/// Search engine for querying indexed documents
pub struct SearchEngine {
    // Implementation in Task 00_5
}

impl SearchEngine {
    pub fn new() -> Result<Self, SearchError> {
        todo!("Implementation in Task 00_5")
    }
}
```

### src/chunking/mod.rs
```rust
//! Smart chunking for AST-based document processing

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ChunkingError {
    #[error("AST parsing error: {0}")]
    AstParsing(String),
    
    #[error("Language detection error: {0}")]
    LanguageDetection(String),
}

/// Code chunk with metadata
#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub chunk_type: String,
}

/// Smart chunker for AST-based code processing
pub struct SmartChunker {
    // Implementation in Task 00_3
}

impl SmartChunker {
    pub fn new() -> Result<Self, ChunkingError> {
        todo!("Implementation in Task 00_3")
    }
}
```

## Context for Next Task
Task 00_3 will implement the Tantivy integration and schema setup, building on this basic crate structure.

## Integration Notes
This crate structure allows Phase 2 tasks to:
- Import and use `DocumentIndexer` from `vector_search::indexing`
- Import and use `SearchEngine` from `vector_search::search`
- Import and use `SmartChunker` from `vector_search::chunking`
- Handle errors consistently with the existing neuromorphic error patterns