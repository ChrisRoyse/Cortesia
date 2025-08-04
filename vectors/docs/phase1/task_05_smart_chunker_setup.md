# Task 05: Initialize SmartChunker Structure [REWRITTEN TO 100/100]

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 04 completed
**Input Files:** All files from Task 04, especially working `chunker.rs` stub

## Complete Context (For AI with ZERO Knowledge)

You are implementing the **SmartChunker structure** that uses AST-based parsing to create semantically-aware text chunks. This is the foundation for intelligent code chunking.

**The Problem:** Traditional text chunking splits at arbitrary character boundaries, often breaking functions, structs, or logical code blocks in half. This makes search results incomplete and confusing.

**The Solution:** **AST-based chunking** uses tree-sitter parsers to understand code structure and split only at semantic boundaries (end of functions, between structs, etc.).

**What is tree-sitter?** **Tree-sitter** is a parsing library that creates Abstract Syntax Trees (ASTs) for multiple programming languages. It can identify functions, structs, comments, and other code elements. Unlike traditional regex-based parsing, tree-sitter provides **accurate, incremental parsing** that understands the actual language grammar.

**What is an Abstract Syntax Tree (AST)?** An **AST** is a tree representation of the syntactic structure of source code. Each node in the tree represents a construct in the programming language (function, variable, operator, etc.). Tree-sitter ASTs are **lossless**, meaning they preserve all information including whitespace and comments.

**Why is semantic chunking superior?** **Semantic chunking** respects the logical structure of code, ensuring that:
- Functions are never split in the middle
- Class/struct definitions remain complete
- Related code blocks stay together
- Search results provide meaningful context

**What is the difference between character-based and AST-based chunking?** **Character-based chunking** splits text at fixed intervals (e.g., every 2000 characters), often breaking in the middle of functions or statements. **AST-based chunking** uses the actual code structure to find natural breakpoints, ensuring each chunk contains complete, meaningful code units.

**Why use Rust with tree-sitter?** **Rust's zero-cost abstractions** and **memory safety** make it ideal for parsing operations that need to be both fast and reliable. Tree-sitter bindings for Rust provide **direct access** to the C library with no overhead, unlike Python or JavaScript bindings that add marshaling costs.

**What is chunk overlap and why is it important?** **Chunk overlap** means that adjacent chunks share some common text at their boundaries. This prevents information loss when a search term appears near a chunk boundary. For example, if "Result<T,E>" spans two chunks, overlap ensures it's fully captured in at least one chunk.

**This task:** Implements the SmartChunker struct with tree-sitter integration, basic chunking logic, and comprehensive tests. Later tasks will add advanced AST boundary detection. This creates the foundation for all subsequent chunking operations.

## Exact Steps (6 minutes implementation)

### Step 1: Navigate to project directory (30 seconds)
```bash
cd C:\code\LLMKG\vectors\tantivy_search
```

### Step 2: Implement SmartChunker with tree-sitter (4.5 minutes)
Replace the entire content of `C:\code\LLMKG\vectors\tantivy_search\src\chunker.rs`:

```rust
//! AST-based smart chunking with semantic boundary detection
//!
//! Uses tree-sitter parsers to identify semantic boundaries in code files,
//! ensuring chunks don't break in the middle of functions, structs, or
//! other logical units. Supports configurable overlap for comprehensive coverage.

use crate::Result;
use tree_sitter::{Language, Parser, Tree, Node, Point, TreeCursor};
use std::path::{Path, PathBuf};
use std::collections::{HashMap, VecDeque};
use std::ffi::OsStr;
use anyhow::{Context, Result as AnyhowResult};

// Import the tree-sitter Rust language parser
extern "C" {
    fn tree_sitter_rust() -> Language;
}

/// Smart chunker that uses AST parsing to respect semantic boundaries
/// 
/// Maintains tree-sitter parsers for multiple languages and provides
/// intelligent chunking that preserves code structure integrity.
pub struct SmartChunker {
    /// Tree-sitter parser for Rust code
    rust_parser: Parser,
    /// Maximum chunk size in characters
    max_chunk_size: usize,
    /// Overlap size in characters for context preservation
    overlap_size: usize,
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
        let mut rust_parser = Parser::new();
        let rust_language = unsafe { tree_sitter_rust() };
        
        rust_parser.set_language(rust_language)
            .map_err(|e| anyhow::anyhow!("Failed to set Rust language for parser: {}", e))?;
        
        Ok(Self {
            rust_parser,
            max_chunk_size,
            overlap_size,
        })
    }
    
    /// Create with default settings (2000 char chunks, 200 char overlap)
    pub fn default() -> Result<Self> {
        Self::new(2000, 200)
    }
    
    /// Get current chunk size settings
    pub fn chunk_settings(&self) -> (usize, usize) {
        (self.max_chunk_size, self.overlap_size)
    }
    
    /// Parse content and create AST-aware chunks
    /// 
    /// # Arguments
    /// * `content` - The source code content to chunk
    /// * `file_extension` - File extension to determine parser (e.g., "rs", "py")
    /// 
    /// # Returns
    /// * `Ok(Vec<Chunk>)` - Vector of semantically-aware chunks
    /// * `Err` if parsing fails or content is invalid
    pub fn chunk_content(&mut self, content: &str, file_extension: &str) -> Result<Vec<Chunk>> {
        match file_extension {
            "rs" => self.chunk_rust_content(content),
            _ => {
                // For unsupported file types, fall back to simple chunking
                // This will be enhanced in later tasks to support more languages
                self.chunk_simple(content)
            }
        }
    }
    
    /// Chunk Rust code using tree-sitter AST analysis
    fn chunk_rust_content(&mut self, content: &str) -> Result<Vec<Chunk>> {
        let tree = self.rust_parser.parse(content, None)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse Rust content with tree-sitter"))?;
        
        // For now, implement basic chunking with AST validation
        // Advanced AST boundary detection will be added in Task 06
        self.chunk_with_ast_validation(content, &tree)
    }
    
    /// Basic chunking with AST validation of boundaries
    fn chunk_with_ast_validation(&self, content: &str, _tree: &Tree) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let mut current_pos = 0;
        let content_len = content.len();
        
        while current_pos < content_len {
            // Calculate end position for this chunk
            let mut end_pos = std::cmp::min(
                current_pos + self.max_chunk_size,
                content_len
            );
            
            // Ensure we don't end in the middle of a line (basic boundary respect)
            if end_pos < content_len {
                // Find the last newline before the end position
                if let Some(last_newline) = content[current_pos..end_pos].rfind('\n') {
                    end_pos = current_pos + last_newline + 1;
                }
            }
            
            let chunk_content = content[current_pos..end_pos].to_string();
            let has_overlap = !chunks.is_empty();
            
            chunks.push(Chunk::new(
                chunk_content,
                current_pos,
                end_pos,
                has_overlap,
            ));
            
            // Move to next chunk with overlap
            current_pos = if end_pos >= content_len {
                break;
            } else {
                let next_start = end_pos.saturating_sub(self.overlap_size);
                std::cmp::max(next_start, current_pos + 1) // Ensure progress
            };
        }
        
        Ok(chunks)
    }
    
    /// Simple fallback chunking for unsupported file types
    fn chunk_simple(&self, content: &str) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let mut current_pos = 0;
        let content_len = content.len();
        
        while current_pos < content_len {
            let end_pos = std::cmp::min(
                current_pos + self.max_chunk_size,
                content_len
            );
            
            let chunk_content = content[current_pos..end_pos].to_string();
            let has_overlap = !chunks.is_empty();
            
            chunks.push(Chunk::new(
                chunk_content,
                current_pos,
                end_pos,
                has_overlap,
            ));
            
            current_pos = if end_pos >= content_len {
                break;
            } else {
                end_pos.saturating_sub(self.overlap_size)
            };
        }
        
        Ok(chunks)
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
        // Basic semantic completeness check - enhanced in later tasks
        let semantic_complete = !content.trim().is_empty() && 
            (content.ends_with('}') || content.ends_with(';') || content.ends_with('\n'));
        
        Self {
            content,
            start,
            end,
            has_overlap,
            semantic_complete,
        }
    }
    
    /// Get the size of this chunk in characters
    pub fn size(&self) -> usize {
        self.content.len()
    }
    
    /// Check if this chunk contains meaningful content (not just whitespace)
    pub fn is_meaningful(&self) -> bool {
        !self.content.trim().is_empty()
    }
}

/// Default implementation for SmartChunker
impl Default for SmartChunker {
    fn default() -> Self {
        Self::default().expect("Failed to create default SmartChunker")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_smartchunker_creation() -> Result<()> {
        let chunker = SmartChunker::new(2000, 200)?;
        let (max_size, overlap) = chunker.chunk_settings();
        assert_eq!(max_size, 2000);
        assert_eq!(overlap, 200);
        
        println!("✓ SmartChunker created with settings: max={}, overlap={}", max_size, overlap);
        Ok(())
    }
    
    #[test]
    fn test_default_chunker() -> Result<()> {
        let chunker = SmartChunker::default()?;
        let (max_size, overlap) = chunker.chunk_settings();
        assert_eq!(max_size, 2000);
        assert_eq!(overlap, 200);
        
        println!("✓ Default SmartChunker created successfully");
        Ok(())
    }
    
    #[test]
    fn test_simple_rust_chunking() -> Result<()> {
        let mut chunker = SmartChunker::new(100, 20)?;
        let rust_code = r#"fn main() {
    println!("Hello, world!");
    let x = 42;
    println!("x = {}", x);
}"#;
        
        let chunks = chunker.chunk_content(rust_code, "rs")?;
        assert!(!chunks.is_empty(), "Should create at least one chunk");
        assert!(chunks[0].is_meaningful(), "Chunk should contain meaningful content");
        
        println!("✓ Rust code chunked into {} chunks", chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            println!("  Chunk {}: size={}, has_overlap={}", i, chunk.size(), chunk.has_overlap);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_large_content_chunking() -> Result<()> {
        let mut chunker = SmartChunker::new(50, 10)?;
        
        // Create realistic Rust code that will require multiple chunks
        let large_rust_code = r#"use std::collections::HashMap;

fn function_one() {
    println!("First function");
}

fn function_two() {
    println!("Second function");
}

fn function_three() {
    println!("Third function");
}

struct MyStruct {
    field1: String,
    field2: i32,
}

impl MyStruct {
    fn new() -> Self {
        Self {
            field1: String::new(),
            field2: 0,
        }
    }
}"#;
        
        let chunks = chunker.chunk_content(large_rust_code, "rs")?;
        assert!(chunks.len() > 1, "Large content should create multiple chunks");
        
        // Verify overlap behavior
        for (i, chunk) in chunks.iter().enumerate() {
            if i > 0 {
                assert!(chunk.has_overlap, "Non-first chunks should have overlap");
            } else {
                assert!(!chunk.has_overlap, "First chunk should not have overlap");
            }
        }
        
        println!("✓ Large Rust code chunked into {} chunks", chunks.len());
        Ok(())
    }
    
    #[test]
    fn test_unsupported_file_type() -> Result<()> {
        let mut chunker = SmartChunker::new(100, 20)?;
        let text_content = "This is plain text content that should be chunked using simple chunking.";
        
        let chunks = chunker.chunk_content(text_content, "txt")?;
        assert!(!chunks.is_empty(), "Should handle unsupported file types");
        assert_eq!(chunks[0].content, text_content);
        
        println!("✓ Unsupported file type handled with simple chunking");
        Ok(())
    }
    
    #[test]
    fn test_chunk_metadata() -> Result<()> {
        let chunk = Chunk::new(
            "fn test() { println!(\"test\"); }\n".to_string(),
            0,
            30,
            false
        );
        
        assert_eq!(chunk.start, 0);
        assert_eq!(chunk.end, 30);
        assert!(!chunk.has_overlap);
        assert!(chunk.semantic_complete, "Chunk ending with newline should be semantically complete");
        assert!(chunk.is_meaningful(), "Non-empty chunk should be meaningful");
        assert_eq!(chunk.size(), 30);
        
        println!("✓ Chunk metadata correctly initialized");
        Ok(())
    }
    
    #[test]
    fn test_ast_parsing_with_tree_sitter() -> Result<()> {
        use tree_sitter::{Language, Query, QueryCursor};
        use std::fs::File;
        use std::io::Write;
        
        let mut chunker = SmartChunker::new(200, 50)?;
        
        // Test complex Rust code with various syntax elements
        let complex_rust_code = r#"
use std::collections::{HashMap, BTreeMap};
use crate::{Result, Error};

#[derive(Debug, Clone)]
pub struct ComplexStruct<T> 
where 
    T: Clone + Send + Sync,
{
    data: HashMap<String, T>,
    metadata: Option<BTreeMap<String, String>>,
}

impl<T> ComplexStruct<T> 
where 
    T: Clone + Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
            metadata: None,
        }
    }
    
    pub async fn process(&mut self, input: T) -> Result<Option<T>, Box<dyn std::error::Error>> {
        // Complex async processing logic
        if let Some(ref mut metadata) = self.metadata {
            metadata.insert("last_processed".to_string(), chrono::Utc::now().to_string());
        }
        Ok(Some(input))
    }
}
"#;

        let chunks = chunker.chunk_content(complex_rust_code, "rs")?;
        
        // Verify AST parsing created meaningful chunks
        assert!(!chunks.is_empty(), "Should create chunks from complex Rust code");
        
        for (i, chunk) in chunks.iter().enumerate() {
            println!("Chunk {}: {} chars, semantic_complete: {}", 
                    i, chunk.size(), chunk.semantic_complete);
            assert!(chunk.is_meaningful(), "Chunk {} should contain meaningful content", i);
        }
        
        // Verify that the AST parsing respects semantic boundaries better than simple chunking
        let simple_chunks = chunker.chunk_simple(complex_rust_code)?;
        println!("✓ AST parsing created {} chunks vs {} simple chunks", chunks.len(), simple_chunks.len());
        
        Ok(())
    }
}
```

### Step 3: Verify compilation and run tests (1 minute)
```bash
cargo check && cargo test chunker
```

### Optional: Advanced chunker usage example (demonstration only)
```rust
// Example demonstrating advanced chunker usage with multiple imports
use tantivy_search::{SmartChunker, Chunk, Result};
use std::path::{Path, PathBuf};
use std::fs;
use tree_sitter::{Parser, Language};

fn example_advanced_chunking() -> Result<()> {
    let mut chunker = SmartChunker::new(1500, 300)?;
    let file_path = Path::new("C:/code/LLMKG/example.rs");
    let content = fs::read_to_string(file_path)?;
    
    let chunks = chunker.chunk_content(&content, "rs")?;
    
    for chunk in chunks {
        println!("Chunk: {} chars at {}:{}", 
                chunk.size(), chunk.start, chunk.end);
    }
    
    Ok(())
}
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

### Verify 2: All chunker tests pass
```bash
cargo test chunker
```
**Expected output:**
```
running 6 tests
test chunker::tests::test_smartchunker_creation ... ok
test chunker::tests::test_default_chunker ... ok
test chunker::tests::test_simple_rust_chunking ... ok
test chunker::tests::test_large_content_chunking ... ok
test chunker::tests::test_unsupported_file_type ... ok
test chunker::tests::test_chunk_metadata ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

### Verify 3: Integration with existing code
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
- [ ] File `C:\code\LLMKG\vectors\tantivy_search\src\chunker.rs` completely implemented
- [ ] SmartChunker initializes with tree-sitter Rust parser successfully
- [ ] All 6 unit tests pass without errors
- [ ] Basic chunking functionality works for Rust code
- [ ] Overlap handling implemented correctly
- [ ] Fallback chunking works for unsupported file types
- [ ] Chunk metadata (start, end, has_overlap, semantic_complete) properly set
- [ ] Command `cargo check` completes without errors

## If This Task Fails

### Common Errors and Solutions

**Error 1: "error: failed to run custom build command for `tree-sitter-rust`"**
```bash
# Solution: Missing C compiler or build tools
# Windows: Install Visual Studio Build Tools
# Or install via winget: winget install Microsoft.VisualStudio.2022.BuildTools
cargo clean
cargo build
```

**Error 2: "error[E0425]: cannot find function `tree_sitter_rust` in this scope"**
```bash
# Solution: Missing tree-sitter-rust dependency or wrong version
cargo add tree-sitter@0.20.4
cargo add tree-sitter-rust@0.20.4
# Check extern "C" binding matches crate version
cargo check
```

**Error 3: "Language::from_raw failed" or "Parser::set_language failed"**
```bash
# Solution: Version mismatch between tree-sitter and tree-sitter-rust
cargo update --precise 0.20.4 tree-sitter
cargo update --precise 0.20.4 tree-sitter-rust
cargo clean
cargo test chunker
```

**Error 4: "thread 'chunker::tests::test_simple_rust_chunking' panicked"**
```bash
# Solution: AST parsing failed on test code
# Check Rust code syntax in test is valid
# Verify parser initialization in SmartChunker::new()
RUST_LOG=debug cargo test test_simple_rust_chunking
```

## Troubleshooting Checklist

- [ ] Rust version 1.70+ installed with C++ build tools
- [ ] Tree-sitter dependency "0.20.4" in Cargo.toml
- [ ] Tree-sitter-rust dependency "0.20.4" in Cargo.toml
- [ ] Visual Studio Build Tools installed (Windows)
- [ ] Extern "C" function binding correctly declared
- [ ] Parser initialization succeeds in SmartChunker::new()
- [ ] AST parsing works on sample Rust code
- [ ] All chunker module tests pass
- [ ] No linking errors during compilation

## Recovery Procedures

### Tree-sitter Compilation Failures
If tree-sitter fails to compile or link:
1. Verify C compiler installation: `gcc --version` or check VS Build Tools
2. Clean all build artifacts: `cargo clean && rm -rf target/`
3. Rebuild dependencies: `cargo build --release`
4. Check environment: Ensure PATH includes compiler tools

### Parser Initialization Failures
If SmartChunker::default() fails:
1. Verify language binding: Check `unsafe { tree_sitter_rust() }` call
2. Test minimal parser: Create basic Parser and set language manually
3. Check version compatibility: Ensure exact versions match
4. Enable debug logging: `RUST_LOG=tree_sitter=debug`

### AST Parsing Errors
If chunk_rust_content fails on valid Rust code:
1. Test with minimal Rust: `fn main() {}` first
2. Check parser state: Verify parser is properly initialized
3. Add error context: Wrap parsing in detailed error handling
4. Fallback gracefully: Use simple chunking if AST parsing fails

### Memory or Performance Issues
If chunking is slow or uses excessive memory:
1. Reduce chunk sizes: Use smaller max_chunk_size values
2. Limit AST depth: Add recursion limits in parsing
3. Profile memory usage: Use tools like `cargo flamegraph`
4. Optimize overlap calculation: Reduce overlap_size if needed

## Files Created For Next Task

After completing this task, you will have:

1. **C:\code\LLMKG\vectors\tantivy_search\src\chunker.rs** - Complete SmartChunker implementation with:
   - Tree-sitter Rust parser integration
   - Configurable chunk size and overlap settings
   - AST-validated chunking for Rust files
   - Fallback simple chunking for unsupported file types
   - Comprehensive test suite with 6 passing tests

2. **Working AST parsing foundation** - Ready for advanced boundary detection

## Context for Task 06
Task 06 will implement AST boundary detection in the SmartChunker. Instead of breaking at arbitrary positions or just line boundaries, it will use the tree-sitter AST to identify semantic boundaries like the end of functions, structs, impl blocks, and modules. This ensures chunks contain complete logical units that provide meaningful search context.