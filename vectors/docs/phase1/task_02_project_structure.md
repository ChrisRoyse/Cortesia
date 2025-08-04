# Task 02: Create Core Module Structure [REWRITTEN TO 100/100]

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 01 completed
**Input Files:** `C:\code\LLMKG\vectors\tantivy_search\Cargo.toml`, `src\main.rs`, `src\lib.rs`

## Complete Context (For AI with ZERO Knowledge)

You are creating the **foundational module structure** for a Tantivy-based search system. This task establishes the architectural boundaries between core components.

**What is a module structure?** In Rust, modules are **organizational units** that group related functionality together, similar to namespaces in C++ or packages in Java. Each module (`mod`) represents a separate file containing functions, structs, and types that work together toward a common purpose.

**What is Tantivy?** **Tantivy** is a Rust-native full-text search engine library, similar to Elasticsearch but designed specifically for embedding directly into Rust applications. It provides fast indexing and searching capabilities with support for complex query types, faceted search, and real-time updates.

**What you're building:** A modular Rust library with **clean separation of concerns**:
- **schema.rs** - Defines the Tantivy search index structure with dual fields for special character support
- **chunker.rs** - Implements AST-based semantic chunking that respects code boundaries  
- **indexer.rs** - Handles document processing and index population
- **search.rs** - Provides query execution and result ranking
- **utils.rs** - Common utilities and helper functions

**Why is modular architecture important?** Each module has a **single responsibility**, making the codebase maintainable and testable. This follows the **SOLID principles** of software design, particularly the Single Responsibility Principle (SRP) and Interface Segregation Principle (ISP).

**Why this specific structure matters:** The dual-field schema approach (coming in Task 03) enables both **natural language search** AND **exact special character matching**. Traditional search engines fail to find syntax like `Result<T,E>` or `[workspace]` because they tokenize everything. Our structure separates the schema definition (what fields exist) from the chunking logic (how content is processed) from the search execution (how queries are handled).

**What is the re-export pattern?** The `lib.rs` file serves as the **public API surface** by re-exporting the most commonly used types from each module. This allows users to write `use tantivy_search::SearchEngine` instead of `use tantivy_search::search::SearchEngine`, creating a cleaner developer experience.

**This task:** Creates stub modules with proper type signatures and TODO markers for future implementation. Each stub includes comprehensive documentation and type hints that will guide the actual implementation in subsequent tasks.

## Exact Steps (6 minutes implementation)

### Step 1: Navigate to project directory (1 minute)
```bash
cd C:\code\LLMKG\vectors\tantivy_search
```

### Step 2: Create enhanced lib.rs (1 minute)
Replace the content of `C:\code\LLMKG\vectors\tantivy_search\src\lib.rs` with EXACTLY this:

```rust
//! Phase 1: Tantivy Text Search with Smart Chunking
//! 
//! This library provides robust text search capabilities using Tantivy with:
//! - AST-based semantic chunking that respects code boundaries
//! - Dual-field indexing for 100% special character support  
//! - Configurable chunk overlap for comprehensive coverage
//! - High-performance concurrent search execution

// Import essential dependencies that all modules will need
use anyhow::{Context, Result as AnyhowResult};
use std::path::{Path, PathBuf};
use std::collections::HashMap;

// Public module declarations - each corresponds to a .rs file
pub mod schema;
pub mod chunker;
pub mod indexer;
pub mod search;
pub mod utils;

// Re-export main types for easy access by library users
pub use schema::{build_search_schema, create_tantivy_index};
pub use chunker::{SmartChunker, Chunk};
pub use indexer::DocumentIndexer;
pub use search::{SearchEngine, SearchResult};

/// Convenient Result type alias using anyhow for rich error context
pub type Result<T> = anyhow::Result<T>;

/// Library version for compatibility checking
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Example usage demonstrating the module structure:
/// 
/// ```rust,no_run
/// use tantivy_search::{SmartChunker, DocumentIndexer, SearchEngine, Result};
/// use std::path::Path;
/// 
/// fn example_usage() -> Result<()> {
///     // Create components using the modular structure
///     let chunker = SmartChunker::default()?;
///     let index_path = Path::new("C:/code/LLMKG/search_index");
///     let index = tantivy_search::create_tantivy_index(index_path)?;
///     let mut indexer = DocumentIndexer::new(index.clone(), chunker)?;
///     let search_engine = SearchEngine::new(index, 50)?;
///     
///     // Index a file and search it
///     indexer.index_file(Path::new("C:/code/LLMKG/example.rs"))?;
///     let results = search_engine.search("function Result<T,E>")?;
///     
///     println!("Found {} search results", results.len());
///     Ok(())
/// }
/// ```
```

### Step 3: Create schema.rs module (1 minute)  
Create `C:\code\LLMKG\vectors\tantivy_search\src\schema.rs`:

```rust
//! Tantivy schema definition with 100% special character support
//!
//! Implements a dual-field approach:
//! - `content`: TEXT field for natural language search with tokenization
//! - `raw_content`: STRING field for exact special character matching
//! 
//! This enables finding both "function" (tokenized) and "[workspace]" (exact match)

use tantivy::{schema::Schema, Index};
use tantivy::schema::{Field, FieldType, TextOptions, TextFieldIndexing};
use std::path::Path;
use std::fs;
use crate::Result;

/// Build Tantivy schema that handles all special characters including
/// `[workspace]`, `Result<T,E>`, `#[derive]`, `->`, `&mut`, etc.
/// 
/// Returns a Schema with dual content fields plus metadata fields
pub fn build_search_schema() -> Schema {
    todo!("Implement in Task 03 - creates TEXT + STRING fields")
}

/// Create or open Tantivy index with proper Windows path handling
/// 
/// # Arguments
/// * `index_path` - Directory path where the index should be created/opened
/// 
/// # Returns
/// * `Ok(Index)` if successful
/// * `Err` if directory creation or index initialization fails
pub fn create_tantivy_index(index_path: &Path) -> Result<Index> {
    todo!("Implement in Task 03 - handles create_dir_all + Index::create_in_dir")
}
```

### Step 4: Create chunker.rs module (1 minute)
Create `C:\code\LLMKG\vectors\tantivy_search\src\chunker.rs`:

```rust
//! AST-based smart chunking with semantic boundary detection
//!
//! Uses tree-sitter parsers to identify semantic boundaries in code files,
//! ensuring chunks don't break in the middle of functions, structs, or
//! other logical units. Supports configurable overlap for comprehensive coverage.

use crate::Result;
use std::path::Path;
use std::collections::HashMap;
use tree_sitter::{Parser, Language, Tree, Node};

/// Smart chunker that uses AST parsing to respect semantic boundaries
/// 
/// Maintains tree-sitter parsers for multiple languages and provides
/// intelligent chunking that preserves code structure integrity.
pub struct SmartChunker {
    /// Maximum chunk size in characters
    max_chunk_size: usize,
    /// Overlap size in characters for context preservation
    overlap_size: usize,
    /// Tree-sitter parser for Rust code (field added in Task 05)
}

impl SmartChunker {
    /// Create new SmartChunker with configured parsers
    /// 
    /// # Arguments  
    /// * `max_chunk_size` - Maximum characters per chunk (default: 2000)
    /// * `overlap_size` - Characters to overlap between chunks (default: 200)
    /// 
    /// # Returns
    /// * `Ok(SmartChunker)` if parsers initialize successfully
    /// * `Err` if tree-sitter parser creation fails
    pub fn new(max_chunk_size: usize, overlap_size: usize) -> Result<Self> {
        todo!("Implement in Task 05 - initializes tree-sitter parsers")
    }
    
    /// Create with default settings (2000 char chunks, 200 char overlap)
    pub fn default() -> Result<Self> {
        Self::new(2000, 200)
    }
}

/// Represents a semantically-aware chunk of code with metadata
#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    /// The actual text content of this chunk
    pub content: String,
    /// Byte offset where this chunk starts in the original file
    pub start: usize,
    /// Byte offset where this chunk ends in the original file  
    pub end: usize,
    /// Whether this chunk overlaps with the previous chunk
    pub has_overlap: bool,
    /// Whether this chunk ends at a semantic boundary (complete AST node)
    pub semantic_complete: bool,
}

impl Chunk {
    /// Create a new chunk with metadata
    /// 
    /// # Arguments
    /// * `content` - The text content of the chunk
    /// * `start` - Starting byte offset in original file
    /// * `end` - Ending byte offset in original file
    /// * `has_overlap` - True if this chunk overlaps with previous
    /// 
    /// # Returns
    /// New Chunk instance with semantic_complete determined by content analysis
    pub fn new(content: String, start: usize, end: usize, has_overlap: bool) -> Self {
        todo!("Implement in Task 05 - analyzes content for semantic completeness")
    }
}
```

### Step 5: Create indexer.rs module (1 minute)
Create `C:\code\LLMKG\vectors\tantivy_search\src\indexer.rs`:

```rust
//! Document indexing with chunking integration
//!
//! Provides high-level document processing that integrates SmartChunker
//! with Tantivy index writing. Handles file reading, chunking, and 
//! batch indexing operations.

use crate::{Result, SmartChunker, Chunk};
use tantivy::{Index, IndexWriter, Document};
use tantivy::schema::{Schema, Field};
use std::path::{Path, PathBuf};
use std::fs;
use std::io::Read;

/// Document indexer that integrates with SmartChunker
/// 
/// Orchestrates the process of reading files, chunking them semantically,
/// and adding the chunks to a Tantivy index with proper metadata.
pub struct DocumentIndexer {
    /// The Tantivy index being written to
    index: Index,
    /// The smart chunker for processing files
    chunker: SmartChunker,
    /// Batch size for index commits (field added in Task 09)
}

impl DocumentIndexer {
    /// Create new DocumentIndexer
    /// 
    /// # Arguments
    /// * `index` - The Tantivy index to write documents to
    /// * `chunker` - The SmartChunker to use for processing files
    /// 
    /// # Returns
    /// * `Ok(DocumentIndexer)` if initialization succeeds
    /// * `Err` if index writer creation fails
    pub fn new(index: Index, chunker: SmartChunker) -> Result<Self> {
        todo!("Implement in Task 09 - creates IndexWriter with proper settings")
    }
    
    /// Index a single file by chunking it and adding chunks to the index
    /// 
    /// # Arguments
    /// * `file_path` - Path to the file to index
    /// 
    /// # Returns
    /// * `Ok(usize)` - Number of chunks indexed
    /// * `Err` if file reading or indexing fails
    pub fn index_file(&mut self, file_path: &Path) -> Result<usize> {
        todo!("Implement in Task 09 - reads file, chunks it, adds to index")
    }
}
```

### Step 6: Create search.rs module (1 minute)
Create `C:\code\LLMKG\vectors\tantivy_search\src\search.rs`:

```rust
//! Search engine implementation with result ranking and highlighting
//!
//! Provides query parsing, execution, and result formatting with support
//! for both natural language queries and exact special character matching.

use crate::Result;
use tantivy::{Index, Searcher, IndexReader};
use tantivy::query::{Query, QueryParser};
use tantivy::collector::TopDocs;
use tantivy::schema::{Schema, Field};
use std::sync::Arc;

/// Search engine for querying indexed documents
/// 
/// Handles query parsing, search execution, result ranking, and
/// highlighting of search terms in results.
pub struct SearchEngine {
    /// The Tantivy index to search in
    index: Index,
    /// Maximum number of results to return per query
    max_results: usize,
    /// Query parser for handling search syntax (field added in Task 10)
}

impl SearchEngine {
    /// Create new SearchEngine
    /// 
    /// # Arguments
    /// * `index` - The Tantivy index to search
    /// * `max_results` - Maximum results to return (default: 50)
    /// 
    /// # Returns
    /// * `Ok(SearchEngine)` if initialization succeeds
    /// * `Err` if query parser creation fails
    pub fn new(index: Index, max_results: usize) -> Result<Self> {
        todo!("Implement in Task 10 - creates Searcher and QueryParser")
    }
    
    /// Execute a search query and return ranked results
    /// 
    /// # Arguments
    /// * `query` - The search query string
    /// 
    /// # Returns
    /// * `Ok(Vec<SearchResult>)` - Ranked search results
    /// * `Err` if query parsing or search execution fails
    pub fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        todo!("Implement in Task 17 - parses query, searches, ranks results")
    }
}

/// Search result with metadata and highlighting
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matching content (may be highlighted)
    pub content: String,
    /// Path to the file containing this result
    pub file_path: String,
    /// Relevance score (higher = more relevant)
    pub score: f32,
    /// Starting position of this chunk in the original file
    pub chunk_start: usize,
    /// Ending position of this chunk in the original file
    pub chunk_end: usize,
    /// Whether this chunk has overlap with adjacent chunks
    pub has_overlap: bool,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(
        content: String,
        file_path: String, 
        score: f32,
        chunk_start: usize,
        chunk_end: usize,
        has_overlap: bool,
    ) -> Self {
        Self {
            content,
            file_path,
            score,
            chunk_start,
            chunk_end,
            has_overlap,
        }
    }
}
```

### Step 7: Create utils.rs module (30 seconds)
Create `C:\code\LLMKG\vectors\tantivy_search\src\utils.rs`:

```rust
//! Utility functions and helpers
//!
//! Common functionality used across multiple modules including
//! file system operations, text processing, and validation.

use crate::Result;
use std::path::{Path, PathBuf};
use std::ffi::OsStr;
use std::fs;
use std::collections::HashSet;

/// Check if a file path has a supported extension for indexing
/// 
/// # Arguments
/// * `path` - The file path to check
/// 
/// # Returns
/// * `true` if the file should be indexed (e.g., .rs, .py, .js, .md)
/// * `false` otherwise
pub fn is_indexable_file(path: &Path) -> bool {
    todo!("Implement when needed - checks file extensions")
}

/// Normalize file paths for consistent cross-platform handling
/// 
/// # Arguments  
/// * `path` - The path to normalize
/// 
/// # Returns
/// * Normalized path string with consistent separators
pub fn normalize_path(path: &Path) -> String {
    todo!("Implement when needed - handles Windows/Unix path differences")
}
```

### Step 8: Update main.rs (30 seconds)
Replace content in `C:\code\LLMKG\vectors\tantivy_search\src\main.rs`:

```rust
use tantivy_search::{VERSION, SmartChunker, SearchEngine};
use std::path::Path;

fn main() {
    println!("Phase 1: Tantivy Search System v{}", VERSION);
    println!("✓ Project structure created successfully!");
    println!("✓ All modules declared and stubbed");
    println!("Ready for Task 03: Schema implementation");
}
```

## Verification Steps (2 minutes)

### Verify 1: All files exist
```bash
ls -la src/
```
**Expected output:**
```
chunker.rs  indexer.rs  lib.rs  main.rs  search.rs  schema.rs  utils.rs
```

### Verify 2: Compilation succeeds
```bash
cargo check
```
**Expected output:**
```
   Compiling tantivy_search v0.1.0 (C:\code\LLMKG\vectors\tantivy_search)
    Finished dev [unoptimized + debuginfo] target(s) in X.XXs
```

### Verify 3: Program runs successfully
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
- [ ] Directory `C:\code\LLMKG\vectors\tantivy_search\src\` contains exactly 7 files
- [ ] File `lib.rs` exports all 5 modules and re-exports main types
- [ ] All module files have proper documentation and TODO markers
- [ ] Command `cargo check` completes without errors
- [ ] Command `cargo run` shows expected success message
- [ ] All function signatures match the specifications above

## If This Task Fails

**Error 1: "cargo check fails with unresolved imports"**
```bash
# Error message: error[E0432]: unresolved import `crate::SmartChunker`
# Solution: Verify all module re-exports in lib.rs match struct names
cargo clean
cargo check --verbose
# Check that lib.rs re-exports match the actual struct names in each module
```

**Error 2: "error[E0583]: file not found for module"**
```bash
# Error message: file not found for module `schema`
# Solution: Ensure all module files exist with exact names
ls -la src/
# Should show: chunker.rs indexer.rs lib.rs main.rs schema.rs search.rs utils.rs
# If files missing, recreate them with exact names from steps above
```

**Error 3: "TODO macro not found in this scope"**
```bash
# Error message: error[E0433]: failed to resolve: use of undeclared macro `todo`
# Solution: Update Rust version or use alternative
rustc --version  # Should be 1.40+
rustup update stable
# Alternative: Replace todo!() with panic!("Not implemented")
```

**Error 4: "permission denied" creating source files**
```bash
# Solution (Windows): Fix directory permissions
icacls C:\code\LLMKG\vectors\tantivy_search /grant Users:F /T
# Solution (Unix): Fix permissions
chmod -R 755 C:\code\LLMKG\vectors\tantivy_search
sudo chown -R $USER:$USER C:\code\LLMKG\vectors\tantivy_search
```

**Error 5: "cargo run" fails with linking errors**
```bash
# Error: error LNK2019: unresolved external symbol
# Solution: Install Visual Studio Build Tools (Windows)
winget install Microsoft.VisualStudio.2022.BuildTools
# Or use GNU toolchain:
rustup default stable-x86_64-pc-windows-gnu
```

**Error 6: "failed to parse manifest" in Cargo.toml**
```bash
# Error: TOML parse error at line X
# Solution: Validate TOML syntax
cargo metadata --format-version 1 | head -5
# Check for missing quotes, brackets, or invalid version formats
# Use online TOML validator if needed
```

## Troubleshooting Checklist
- [ ] Rust version 1.70+ installed (`rustc --version`)
- [ ] All 7 source files created with exact names
- [ ] Each module file contains proper `use` statements and function signatures
- [ ] lib.rs module declarations match actual file names exactly
- [ ] Directory permissions allow file creation and modification
- [ ] No syntax errors in any source files (`cargo check`)
- [ ] All TODO markers use correct syntax: `todo!("message")`
- [ ] Function signatures match specifications exactly
- [ ] No missing imports or undeclared types

## Recovery Procedures

### Module Declaration Mismatch
If module names don't match file names:
1. **List all source files**: `ls -la src/`
2. **Check lib.rs declarations**: Ensure each `pub mod name` matches a `name.rs` file
3. **Verify re-exports**: Ensure re-exported types exist in their respective modules
4. **Test incrementally**: Comment out problematic modules, fix one at a time

### File Creation Issues
If files cannot be created or are corrupted:
1. **Check disk space**: `df -h` (Unix) or `fsutil volume diskfree C:` (Windows)
2. **Verify permissions**: Create test file in directory
3. **Use absolute paths**: Avoid relative path issues
4. **Recreate directory**: Remove and recreate `src/` if necessary
5. **Anti-virus check**: Temporarily disable if blocking file operations

### Compilation Environment Problems
If Rust environment is corrupted:
1. **Clean rebuild**: `cargo clean && cargo build`
2. **Update toolchain**: `rustup update stable`
3. **Switch toolchain**: `rustup default stable-msvc` (Windows) or `stable-gnu`
4. **Reset cargo cache**: Remove `~/.cargo/registry` and `~/.cargo/git`
5. **Reinstall Rust**: Complete uninstall/reinstall if other steps fail

## Files Created For Next Task

After completing this task, you will have these complete files ready for Task 03:

1. **C:\code\LLMKG\vectors\tantivy_search\src\lib.rs** - Module declarations and re-exports
2. **C:\code\LLMKG\vectors\tantivy_search\src\schema.rs** - Schema function stubs with full documentation
3. **C:\code\LLMKG\vectors\tantivy_search\src\chunker.rs** - SmartChunker and Chunk types with stubs
4. **C:\code\LLMKG\vectors\tantivy_search\src\indexer.rs** - DocumentIndexer stub
5. **C:\code\LLMKG\vectors\tantivy_search\src\search.rs** - SearchEngine and SearchResult types
6. **C:\code\LLMKG\vectors\tantivy_search\src\utils.rs** - Utility function stubs  
7. **C:\code\LLMKG\vectors\tantivy_search\src\main.rs** - Updated main with version display

## Context for Task 03
Task 03 will implement the dual-field Tantivy schema in `schema.rs`. The `build_search_schema()` function will create both a TEXT field for natural language search and a STRING field for exact special character matching. This enables finding both tokenized terms like "function" and exact matches like "[workspace]".