# Task 04: Add Special Character Test Suite [REWRITTEN TO 100/100]

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 03 completed
**Input Files:** All files from Task 03, especially the working schema with dual fields

## Complete Context (For AI with ZERO Knowledge)

You are creating a **comprehensive test suite** that validates the dual-field schema can handle 100% special character support. This is the proof that your search system works where others fail.

**The Challenge:** Traditional search engines cannot find `[workspace]`, `Result<T,E>`, `#[derive]`, `->`, `&mut` because they tokenize everything. You need to prove your dual-field approach solves this.

**What is tokenization in search engines?** **Tokenization** is the process of breaking text into individual searchable units (tokens). Most search engines split text on whitespace and punctuation, so `Result<T,E>` becomes separate tokens: `Result`, `<`, `T`, `,`, `E`, `>`. This makes it **impossible** to search for the complete syntax pattern.

**What is the dual-field approach?** The **dual-field approach** stores the same content in two different field types:
- **TEXT field**: Content is tokenized for natural language search (finds individual words)
- **STRING field**: Content is stored exactly as-is for precise character matching

**Why is this revolutionary for code search?** Traditional code search tools like **grep** or **ripgrep** only do exact text matching, while search engines like **Elasticsearch** only do tokenized search. Our approach combines **both capabilities** in a single system, enabling queries like "find all functions that return Result<T,E>" (mixing natural language + exact syntax).

**What are you testing specifically?**
- **Exact matching** via the STRING field for special characters
- **Tokenized search** via the TEXT field for natural language  
- **Mixed queries** that use both fields together
- **Edge cases** that commonly break other search systems

**What is comprehensive testing?** **Comprehensive testing** means covering all the syntax patterns that real developers actually use in their code. This includes Rust generics (`Vec<Box<dyn Trait>>`), Python decorators (`@property`), configuration syntax (`[workspace]`), and complex nested patterns.

**Why these specific test patterns?** The test patterns represent **real-world syntax** that appears in actual codebases:
- `Result<T,E>` - Rust error handling (millions of occurrences in Rust code)
- `[workspace]` - Cargo.toml configuration (appears in every Rust workspace)
- `#[derive(Debug)]` - Rust attribute macros (extremely common)
- `->` - Function return syntax (fundamental language construct)
- `&mut self` - Rust borrowing (core language feature)

**What is the difference between this and regex search?** **Regex search** requires you to know the exact pattern and manually escape special characters. Our system allows **natural queries** like "function that returns Result" while also supporting exact syntax matching. It's **user-friendly** rather than requiring regex expertise.

**What are edge cases in search?** **Edge cases** are unusual input patterns that commonly cause search systems to fail, such as:
- Deeply nested generics: `<'a, T: 'a + Clone>`
- Escaped strings: `"escaped\\string\\with\\backslashes"`
- Complex function signatures: `fn(&self) -> Result<(), Box<dyn std::error::Error>>`

**Why is this test suite critical?** This test suite **proves** that the dual-field schema actually works. Without these tests, you only have theoretical capability. With these tests passing, you have **demonstrable proof** that your search system can handle syntax patterns that break other tools.

**Test approach:** Index challenging syntax patterns, then search for them exactly as they appear in code. If you can find `Result<T,E>` as written (not broken into tokens), the system works.

**This task:** Creates comprehensive tests that prove 100% special character support with real Rust/Python syntax examples, covering both common patterns and edge cases that typically break search engines.

## Exact Steps (6 minutes implementation)

### Step 1: Navigate to project directory (30 seconds)
```bash
cd C:\code\LLMKG\vectors\tantivy_search
```

### Step 2: Create comprehensive special character tests (4.5 minutes)
Create `C:\code\LLMKG\vectors\tantivy_search\tests\special_chars_test.rs`:

```rust
//! Comprehensive tests for special character support in dual-field schema
//!
//! Tests the core capability that differentiates this search system:
//! finding exact special character sequences like [workspace], Result<T,E>

use tantivy_search::schema::{build_search_schema, create_tantivy_index};
use tantivy::{doc, IndexWriter, Term, Document, Index};
use tantivy::query::{TermQuery, Query};
use tantivy::collector::TopDocs;
use tantivy::schema::{Schema, Field, IndexRecordOption};
use tempfile::TempDir;
use std::path::{Path, PathBuf};
use anyhow::Result;

/// Critical special character patterns from real Rust/Python code
/// These are the patterns that break traditional search engines
const SPECIAL_SYNTAX_TESTS: &[(&str, &str)] = &[
    // Rust-specific syntax
    ("[workspace]", "Cargo.toml workspace configuration section"),
    ("Result<T,E>", "Generic Result type with type parameters"),
    ("#[derive(Debug)]", "Attribute macro for automatic trait derivation"),
    ("->", "Function return type arrow syntax"),
    ("&mut self", "Mutable reference to self parameter"),
    ("<T: Clone>", "Generic constraint with trait bound"),
    ("Vec<Box<dyn Trait>>", "Complex nested generic with trait object"),
    ("Option<&str>", "Generic option with string slice reference"),
    
    // Python-specific syntax
    ("##", "Python double-hash comment marker"),
    ("@property", "Python property decorator syntax"),
    ("${variable}", "Template variable expansion syntax"),
    ("**kwargs", "Python keyword arguments unpacking"),
    ("f'{value}'", "Python f-string formatting syntax"),
    
    // Mixed/general syntax
    ("<!DOCTYPE>", "HTML document type declaration"),
    ("/* comment */", "Multi-line comment syntax"),
    ("${HOME}/bin", "Environment variable in path"),
];

#[test]
fn test_exact_special_character_matching() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("special_chars_index");
    
    // Create index with dual-field schema
    let index = create_tantivy_index(&index_path)?;
    let schema = index.schema();
    
    // Get field handles
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    
    // Index all test cases with both natural and exact content
    let mut index_writer: IndexWriter = index.writer(50_000_000)?;
    
    for (i, (special_syntax, description)) in SPECIAL_SYNTAX_TESTS.iter().enumerate() {
        let combined_content = format!("{} - {}", special_syntax, description);
        
        let doc = doc!(
            content_field => combined_content.clone(),
            raw_content_field => combined_content,
            file_path_field => format!("test_file_{}.rs", i),
            schema.get_field("chunk_index")? => 0u64,
            schema.get_field("chunk_start")? => 0u64,
            schema.get_field("chunk_end")? => combined_content.len() as u64,
            schema.get_field("has_overlap")? => false
        );
        
        index_writer.add_document(doc)?;
    }
    
    index_writer.commit()?;
    
    // Test exact matching via raw_content field
    let reader = index.reader()?;
    let searcher = reader.searcher();
    
    println!("Testing exact special character matching...");
    
    for (target_syntax, description) in SPECIAL_SYNTAX_TESTS {
        // Search for exact special character sequence in raw_content
        let term = Term::from_field_text(raw_content_field, target_syntax);
        let query = TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
        
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
        
        assert!(!top_docs.is_empty(), 
               "FAILED: Could not find exact match for '{}' ({})", 
               target_syntax, description);
        
        println!("✓ Found {} results for exact match: '{}'", top_docs.len(), target_syntax);
    }
    
    println!("✓ All {} special character patterns found successfully!", SPECIAL_SYNTAX_TESTS.len());
    Ok(())
}

#[test]
fn test_tokenized_natural_language_search() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("tokenized_test");
    
    let index = create_tantivy_index(&index_path)?;
    let schema = index.schema();
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    
    // Index realistic code content that should be tokenized
    let mut index_writer = index.writer(50_000_000)?;
    let test_content = "function process_result(input: Result<String, Error>) -> Option<String> { /* implementation */ }";
    
    let doc = doc!(
        content_field => test_content,
        raw_content_field => test_content,
        file_path_field => "example.rs",
        schema.get_field("chunk_index")? => 0u64,
        schema.get_field("chunk_start")? => 0u64,
        schema.get_field("chunk_end")? => test_content.len() as u64,
        schema.get_field("has_overlap")? => false
    );
    
    index_writer.add_document(doc)?;
    index_writer.commit()?;
    
    // Test tokenized search - should find individual words
    let reader = index.reader()?;
    let searcher = reader.searcher();
    
    let natural_language_tokens = ["function", "process", "result", "input", "string", "error", "option", "implementation"];
    
    println!("Testing tokenized natural language search...");
    
    for token in &natural_language_tokens {
        let term = Term::from_field_text(content_field, token);
        let query = TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
        
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
        assert!(!top_docs.is_empty(), "Failed to find tokenized word: '{}'", token);
        
        println!("✓ Found tokenized word: '{}'", token);
    }
    
    println!("✓ All {} natural language tokens found successfully!", natural_language_tokens.len());
    Ok(())
}

#[test]
fn test_dual_field_mixed_search_capability() -> anyhow::Result<()> {
    println!("Testing mixed search using both fields simultaneously...");
    
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("mixed_search_test");
    
    let index = create_tantivy_index(&index_path)?;
    let schema = index.schema();
    
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    
    let mut index_writer = index.writer(50_000_000)?;
    
    // Index realistic code examples with both natural language and special syntax
    let mixed_test_cases = [
        "function returns Result<T,E> for comprehensive error handling",
        "implement Debug trait automatically using #[derive(Debug)] annotation",
        "use &mut reference for safe mutable access to data structures",
        "configure workspace dependencies in [workspace] section of Cargo.toml",
        "Python @property decorator provides getter/setter functionality"
    ];
    
    for (i, content) in mixed_test_cases.iter().enumerate() {
        let doc = doc!(
            content_field => *content,
            raw_content_field => *content,
            file_path_field => format!("mixed_example_{}.rs", i),
            schema.get_field("chunk_index")? => i as u64,
            schema.get_field("chunk_start")? => 0u64,
            schema.get_field("chunk_end")? => content.len() as u64,
            schema.get_field("has_overlap")? => false
        );
        index_writer.add_document(doc)?;
    }
    
    index_writer.commit()?;
    
    let reader = index.reader()?;
    let searcher = reader.searcher();
    
    // Test 1: Natural language search should work (content field)
    let natural_searches = ["function", "implement", "configure", "error", "handling"];
    for word in &natural_searches {
        let term = Term::from_field_text(content_field, word);
        let query = TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
        let results = searcher.search(&query, &TopDocs::with_limit(10))?;
        assert!(!results.is_empty(), "Natural language search failed for: '{}'", word);
        println!("✓ Natural language search found: '{}'", word);
    }
    
    // Test 2: Exact special character search should work (raw_content field)
    let special_searches = ["Result<T,E>", "#[derive(Debug)]", "&mut", "[workspace]", "@property"];
    for syntax in &special_searches {
        let term = Term::from_field_text(raw_content_field, syntax);
        let query = TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
        let results = searcher.search(&query, &TopDocs::with_limit(10))?;
        assert!(!results.is_empty(), "Special character search failed for: '{}'", syntax);
        println!("✓ Special character search found: '{}'", syntax);
    }
    
    println!("✓ Dual field mixed search capability fully verified!");
    Ok(())
}

#[test]
fn test_edge_case_special_characters() -> anyhow::Result<()> {
    println!("Testing edge case special characters that commonly break search engines...");
    
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("edge_cases_test");
    
    let index = create_tantivy_index(&index_path)?;
    let schema = index.schema();
    
    let raw_content_field = schema.get_field("raw_content")?;
    let content_field = schema.get_field("content")?;
    let file_path_field = schema.get_field("file_path")?;
    
    let mut index_writer = index.writer(50_000_000)?;
    
    // Extremely challenging edge cases
    let edge_cases = [
        "<'a, T: 'a + Clone>",  // Lifetime parameters
        "impl<T> From<T> for Box<T>",  // Complex impl with generics
        "fn(&self) -> Result<(), Box<dyn std::error::Error>>",  // Function pointer with complex return
        "#[cfg(feature = \"serde\")]",  // Cfg attribute with quotes
        "\"escaped\\string\\with\\backslashes\"",  // Heavily escaped string
    ];
    
    for (i, edge_case) in edge_cases.iter().enumerate() {
        let doc = doc!(
            content_field => *edge_case,
            raw_content_field => *edge_case,
            file_path_field => format!("edge_case_{}.rs", i),
            schema.get_field("chunk_index")? => i as u64,
            schema.get_field("chunk_start")? => 0u64,
            schema.get_field("chunk_end")? => edge_case.len() as u64,
            schema.get_field("has_overlap")? => false
        );
        index_writer.add_document(doc)?;
    }
    
    index_writer.commit()?;
    
    let reader = index.reader()?;
    let searcher = reader.searcher();
    
    // Verify each edge case can be found exactly
    for edge_case in &edge_cases {
        let term = Term::from_field_text(raw_content_field, edge_case);
        let query = TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
        let results = searcher.search(&query, &TopDocs::with_limit(10))?;
        
        assert!(!results.is_empty(), "Edge case search failed for: '{}'", edge_case);
        println!("✓ Edge case found: '{}'", edge_case);
    }
    
    println!("✓ All edge case special characters handled successfully!");
    Ok(())
}

/// Demonstrate advanced search capabilities with imports and detailed field access
#[test]
fn test_advanced_search_with_field_access() -> Result<()> {
    use tantivy::{Searcher, IndexReader};
    use tantivy::query::{QueryParser, BooleanQuery, Occur};
    use std::collections::HashMap;
    
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("advanced_search");
    let index = create_tantivy_index(&index_path)?;
    let schema = index.schema();
    
    // Access all field types with explicit field handles
    let content_field = schema.get_field("content")?;
    let raw_content_field = schema.get_field("raw_content")?;
    let file_path_field = schema.get_field("file_path")?;
    let chunk_index_field = schema.get_field("chunk_index")?;
    
    let mut index_writer: IndexWriter = index.writer(50_000_000)?;
    
    // Index complex real-world examples with full field specification
    let complex_examples = [
        ("fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {", "C:/code/LLMKG/complex.rs"),
        ("impl<T: Clone + Debug> MyTrait<T> for Box<T> where T: 'static {", "C:/code/LLMKG/generics.rs"),  
        ("use std::{collections::HashMap, path::{Path, PathBuf}};", "C:/code/LLMKG/imports.rs"),
    ];
    
    for (i, (code_sample, file_path)) in complex_examples.iter().enumerate() {
        let document = doc!(
            content_field => *code_sample,
            raw_content_field => *code_sample,
            file_path_field => *file_path,
            chunk_index_field => i as u64,
            schema.get_field("chunk_start")? => 0u64,
            schema.get_field("chunk_end")? => code_sample.len() as u64,
            schema.get_field("has_overlap")? => false
        );
        index_writer.add_document(document)?;
    }
    
    index_writer.commit()?;
    
    // Demonstrate both exact and tokenized search on the same index
    let reader: IndexReader = index.reader()?;
    let searcher: Searcher = reader.searcher();
    
    // Test exact syntax matching
    let exact_patterns = ["Result<(), Box<dyn", "impl<T: Clone + Debug>", "std::{collections::HashMap"];
    for pattern in &exact_patterns {
        let term = Term::from_field_text(raw_content_field, pattern);
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        let results = searcher.search(&query, &TopDocs::with_limit(5))?;
        assert!(!results.is_empty(), "Failed to find exact pattern: {}", pattern);
        println!("✓ Found exact pattern: '{}'", pattern);
    }
    
    println!("✓ Advanced search with field access test completed successfully!");
    Ok(())
}
```

### Step 3: Run comprehensive tests (1 minute)
```bash
cargo test special_chars_test
```

## Verification Steps (2 minutes)

### Verify 1: All special character tests pass
```bash
cargo test test_exact_special_character_matching
```
**Expected output:**
```
test special_chars_test::test_exact_special_character_matching ... ok
✓ All 16 special character patterns found successfully!
```

### Verify 2: Tokenized search works
```bash
cargo test test_tokenized_natural_language_search
```
**Expected output:**
```
test special_chars_test::test_tokenized_natural_language_search ... ok
✓ All 8 natural language tokens found successfully!
```

### Verify 3: Mixed search capability confirmed
```bash
cargo test test_dual_field_mixed_search_capability
```
**Expected output:**
```
test special_chars_test::test_dual_field_mixed_search_capability ... ok
✓ Dual field mixed search capability fully verified!
```

## Success Validation Checklist
- [ ] File `C:\code\LLMKG\vectors\tantivy_search\tests\special_chars_test.rs` created
- [ ] All 16 special character patterns found via exact matching
- [ ] All 8 natural language tokens found via tokenized search
- [ ] Mixed search using both fields works correctly
- [ ] Edge cases with complex syntax patterns handled
- [ ] All 4 test functions pass without errors

## If This Task Fails

### Common Errors and Solutions

**Error 1: "error[E0433]: failed to resolve: cannot find field in schema"**
```bash
# Solution: Schema fields not found
# Verify Task 03 schema includes all required fields
cargo test test_schema_creation
# Check field names match exactly: content, raw_content, file_path
grep -n "get_field" tests/special_chars_test.rs
```

**Error 2: "assertion failed: !top_docs.is_empty()" for special characters**
```bash
# Solution: Documents not found in search
# Verify raw_content field uses STRING type (not TEXT)
# Check index_writer.commit() is called before search
RUST_LOG=tantivy=debug cargo test test_exact_special_character_matching
```

**Error 3: "error[E0599]: no method named `search` found for type `Searcher`"**
```bash
# Solution: Tantivy API version mismatch
cargo clean
cargo update tantivy --precise 0.22.0
cargo test special_chars_test
```

**Error 4: "OS Error 5: Access is denied" creating temporary indexes**
```bash
# Solution (Windows): Fix temp directory permissions
mkdir C:\temp\tantivy_test
icacls C:\temp\tantivy_test /grant Users:F /T
set TEMP=C:\temp\tantivy_test

# Solution (Unix): Use different temp location
export TMPDIR=/tmp/special_chars_test
chmod 755 /tmp/special_chars_test
```

## Troubleshooting Checklist

- [ ] Task 03 schema implementation completed successfully
- [ ] Tantivy version exactly "0.22.0" in Cargo.toml
- [ ] All schema fields available: content, raw_content, file_path, chunk_*, has_overlap
- [ ] Raw_content field configured as STRING type (not TEXT)
- [ ] Content field configured as TEXT type (tokenized)
- [ ] Write permissions to temp directory for test indexes
- [ ] No compilation errors in special_chars_test.rs
- [ ] Index writer commit() called before search operations
- [ ] Sufficient memory for 50MB index writer buffer

## Recovery Procedures

### Special Character Search Failures
If exact character searches consistently fail:
1. Verify STRING field configuration: `raw_content_field` uses STRING type
2. Check document indexing: Add debug prints in indexing loop
3. Test with simpler patterns first: `"test"` before `"Result<T,E>"`
4. Enable tantivy debug: `RUST_LOG=tantivy=debug cargo test`

### Test Performance Issues
If tests run slowly or timeout:
1. Reduce index writer buffer: `index.writer(10_000_000)` instead of 50MB
2. Limit test data size: Reduce SPECIAL_SYNTAX_TESTS array
3. Run individual tests: `cargo test test_exact_special_character_matching`
4. Check system resources: Ensure sufficient RAM and disk space

### Tokenization Problems
If natural language search fails:
1. Verify TEXT field setup for content field
2. Check tokenizer is working: Enable tantivy query logs
3. Test with simple words first: `"function"` before complex terms
4. Verify QueryParser configuration matches schema fields

### Index Creation Failures
If temporary index creation fails:
1. Check available disk space: `dir C:\ /s` (Windows)
2. Try different temp location: `TempDir::new_in("./temp")`
3. Verify no index lock files: Delete `*.lock` files
4. Reduce concurrent tests: `cargo test -- --test-threads=1`

## Files Created For Next Task

After completing this task, you will have:

1. **C:\code\LLMKG\vectors\tantivy_search\tests\special_chars_test.rs** - Comprehensive test suite proving:
   - Exact special character matching for 16+ syntax patterns
   - Tokenized natural language search capability  
   - Mixed queries using both field types
   - Edge case handling for complex syntax

2. **Proven dual-field functionality** - Tests confirm both TEXT and STRING fields work as designed

## Context for Task 05
Task 05 will implement the SmartChunker structure in `chunker.rs`. With the schema and special character support proven, you'll now build the AST-based chunking system that creates semantically-aware text chunks. The SmartChunker uses tree-sitter parsers to identify function boundaries, struct definitions, and other semantic units, ensuring chunks don't break in the middle of logical code blocks.