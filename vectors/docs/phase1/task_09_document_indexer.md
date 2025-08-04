# Task 09: Implement Complete Document Indexer with Multi-Language Pipeline Integration

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 08 completed (multi-strategy language detection)
**Input Files:**
- `C:/code/LLMKG/vectors/tantivy_search/src/indexer.rs` (DocumentIndexer stub from Task 02)
- `C:/code/LLMKG/vectors/tantivy_search/src/chunker.rs` (SmartChunker with boundary detection)
- `C:/code/LLMKG/vectors/tantivy_search/src/schema.rs` (dual-field Tantivy schema)
- `C:/code/LLMKG/vectors/tantivy_search/src/utils.rs` (language detection)

## Complete Context (For AI with ZERO Knowledge)

**What is a Document Indexer?** A document indexer is the orchestration layer that coordinates the entire pipeline from raw files to searchable index. It combines file reading, language detection, semantic parsing, chunking, and index writing into a cohesive system. The DocumentIndexer is where all previous tasks come together into a functional search system.

**The Complete Pipeline:**
1. **File Reading**: Load content from disk with proper encoding handling
2. **Language Detection**: Use Task 08's detection to choose appropriate parser
3. **AST Parsing**: Use Task 06's boundary detection for supported languages
4. **Semantic Chunking**: Use Task 07's overlap calculation for intelligent chunks
5. **Index Writing**: Use Task 03's dual-field schema to store chunks in Tantivy

**Why This Architecture?** The indexer abstracts the complexity of the pipeline behind a simple `index_file()` interface. Callers don't need to understand AST parsing, boundary detection, or overlap calculation - they just provide a file path and content. This separation of concerns makes the system maintainable and testable.

**Dual Processing Strategy:**
- **Code Files** (Rust, Python): Use full AST-based semantic chunking for optimal results
- **Text Files** (Markdown, plain text): Use simple fixed-size chunking with overlap
- **Unknown Files**: Default to text processing with content-based language detection

**Index Integration:** The indexer uses the dual-field schema from Task 03:
- `content` field: Tokenized for natural language search
- `raw_content` field: Exact string for special character matching
- Metadata fields: File path, chunk positions, overlap flags for result reconstruction

**Error Handling Strategy:** The indexer must handle:
- File reading errors (permissions, encoding, missing files)
- Parse errors (malformed code, unsupported syntax)
- Index errors (disk space, permissions, corruption)
- Memory errors (large files, too many chunks)

This task implements the final integration layer that makes the search system functional.

## Exact Steps

1. **Replace src/indexer.rs** (5 minutes):
Replace entire content of `C:/code/LLMKG/vectors/tantivy_search/src/indexer.rs` with:

```rust
//! Document indexing with chunking integration

use crate::{SmartChunker, Chunk, create_tantivy_index, utils::detect_language};
use tantivy::{Index, IndexWriter, doc};
use std::path::Path;
use anyhow::Result;

/// Document indexer that integrates with SmartChunker
pub struct DocumentIndexer {
    index: Index,
    chunker: SmartChunker,
}

impl DocumentIndexer {
    /// Create new DocumentIndexer
    pub fn new(index_path: &Path) -> Result<Self> {
        let index = create_tantivy_index(index_path)?;
        let chunker = SmartChunker::new()?;
        
        Ok(Self { index, chunker })
    }
    
    /// Index a document file with chunking
    pub fn index_file(&mut self, file_path: &Path, content: &str) -> Result<()> {
        let language = detect_language(file_path, content);
        
        let chunks = if language == "rust" || language == "python" {
            // Use AST-based chunking for supported languages
            let boundaries = self.chunker.find_boundaries(content, &language)?;
            self.chunker.create_chunks(content, &boundaries)
        } else {
            // Simple chunking for other content
            self.simple_chunk(content)
        };
        
        self.index_chunks(file_path, &chunks)?;
        Ok(())
    }
    
    /// Simple chunking fallback for non-code content
    fn simple_chunk(&self, content: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let max_size = 2000;
        let overlap = 200;
        
        let mut start = 0;
        while start < content.len() {
            let end = (start + max_size).min(content.len());
            let chunk_content = content[start..end].to_string();
            chunks.push(Chunk::new(chunk_content, start, end, start > 0));
            
            if end >= content.len() {
                break;
            }
            start = end - overlap;
        }
        
        chunks
    }
    
    /// Index chunks into Tantivy
    fn index_chunks(&self, file_path: &Path, chunks: &[Chunk]) -> Result<()> {
        let mut writer = self.index.writer(50_000_000)?;
        let schema = self.index.schema();
        
        for (i, chunk) in chunks.iter().enumerate() {
            let doc = doc!(
                schema.get_field("content")? => &chunk.content,
                schema.get_field("raw_content")? => &chunk.content,
                schema.get_field("file_path")? => file_path.to_string_lossy().as_ref(),
                schema.get_field("chunk_index")? => i as u64,
                schema.get_field("chunk_start")? => chunk.start as u64,
                schema.get_field("chunk_end")? => chunk.end as u64,
                schema.get_field("has_overlap")? => chunk.has_overlap,
            );
            writer.add_document(doc)?;
        }
        
        writer.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::path::PathBuf;
    
    #[test]
    fn test_document_indexer_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let indexer = DocumentIndexer::new(temp_dir.path())?;
        
        // Should create successfully
        assert!(temp_dir.path().exists());
        
        Ok(())
    }
    
    #[test]
    fn test_file_indexing() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut indexer = DocumentIndexer::new(temp_dir.path())?;
        
        let rust_content = "pub fn hello() { println!(\"Hello\"); }";
        let file_path = PathBuf::from("test.rs");
        
        indexer.index_file(&file_path, rust_content)?;
        
        // Should complete without error
        Ok(())
    }
}
```

2. **Verify compilation** (1 minute):
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo check
```

3. **Test document indexer** (1 minute):
```bash
cargo test test_document_indexer
```

## Verification Steps (2 minutes)

### Verify 1: Document indexer compiles successfully
```bash
cargo check
```
**Expected output:**
```
   Compiling tantivy_search v0.1.0 (C:\code\LLMKG\vectors\tantivy_search)
    Finished dev [unoptimized + debuginfo] target(s) in X.XXs
```

### Verify 2: Document indexer tests pass
```bash
cargo test test_document_indexer -- --nocapture
```
**Expected output:**
```
test indexer::tests::test_document_indexer_creation ... ok
test indexer::tests::test_file_indexing ... ok
test result: ok. 2 passed; 0 failed; 0 ignored
```

### Verify 3: Full pipeline integration works
```bash
cargo test --lib | grep -E "indexer|chunker|schema"
```
**Expected output:** All indexer, chunker, and schema tests pass together

## If This Task Fails

**Error 1: "cannot find function `create_tantivy_index`"**
```bash
# Error: error[E0425]: cannot find function `create_tantivy_index`
# Solution: Import from schema module
use crate::schema::{create_tantivy_index, open_or_create_index};
# Or use the correct function name from Task 03
use crate::schema::{create_index, open_or_create_index};
```

**Error 2: "IndexWriter cannot be created with heap size"**
```bash
# Error: error in IndexWriter::new() with memory allocation
# Solution: Reduce heap size or handle memory constraints
let mut writer = self.index.writer(10_000_000)?; // Reduce from 50MB
# Or handle low-memory systems:
let heap_size = std::cmp::min(50_000_000, available_memory() / 4);
let mut writer = self.index.writer(heap_size)?;
```

**Error 3: "field not found in schema"**
```bash
# Error: thread 'main' panicked at 'Field not found: content'
# Solution: Verify schema field names match exactly
let schema = self.index.schema();
let content_field = schema.get_field("content")
    .map_err(|_| anyhow::anyhow!("Schema missing content field"))?;
# Debug available fields:
println!("Available fields: {:?}", schema.fields().map(|(field, _)| field.field_id()).collect::<Vec<_>>());
```

**Error 4: "chunk boundaries exceed content length"**
```bash
# Error: thread 'main' panicked at 'byte index 2000 is out of bounds of `hello`'
# Solution: Add bounds checking in chunking
let end = (start + max_size).min(content.len());
if start >= content.len() {
    break;
}
let chunk_content = content.get(start..end)
    .unwrap_or_default()
    .to_string();
```

**Error 5: "writer.commit() fails with disk space error"**
```bash
# Error: IO error: No space left on device
# Solution: Check disk space before committing
use std::fs;
let available_space = fs::metadata(&index_path)?.len();
if available_space < 100_000_000 { // 100MB minimum
    return Err(anyhow::anyhow!("Insufficient disk space for index"));
}
```

**Error 6: "language detection causes wrong parser selection"**
```bash
# Error: Parse error when trying to chunk Rust code
# Solution: Add fallback to simple chunking
let chunks = match self.chunker.find_boundaries(content, &language) {
    Ok(boundaries) => self.chunker.create_chunks(content, &boundaries),
    Err(_) => {
        eprintln!("AST parsing failed for {}, using simple chunking", language);
        self.simple_chunk(content)
    }
};
```

**Error 7: "file path contains invalid UTF-8"**
```bash
# Error: lossy conversion warning or panic
# Solution: Handle non-UTF8 paths properly
let file_path_str = file_path.to_string_lossy();
let file_path_bytes = file_path.as_os_str().as_encoded_bytes();
// Store as base64 if necessary:
let file_path_safe = base64::encode(file_path_bytes);
```

## Troubleshooting Checklist
- [ ] All imports resolve correctly (schema, chunker, utils modules)
- [ ] Schema fields match exactly with field names used in doc! macro
- [ ] Memory limits are appropriate for target system
- [ ] File path handling works with non-ASCII characters
- [ ] Language detection integrates properly with chunker
- [ ] Index writer commits successfully without errors
- [ ] Chunk boundaries are validated before string slicing
- [ ] Error handling covers all potential failure points

## Recovery Procedures

### Schema Mismatch Issues
If field names don't match between schema and indexer:
1. **List schema fields**: Debug print all available fields
2. **Check Task 03 schema**: Verify field names match exactly
3. **Test schema creation**: Ensure schema builds correctly
4. **Validate field types**: Check that field types match doc! macro usage

### Memory and Performance Issues
If indexing fails due to resource constraints:
1. **Reduce heap size**: Lower IndexWriter memory allocation
2. **Batch processing**: Process files in smaller batches
3. **Streaming chunks**: Avoid loading entire file content at once
4. **Monitor memory**: Track memory usage during large file processing

### File Processing Errors
If specific files fail to index:
1. **Validate encoding**: Ensure files are UTF-8 or handle other encodings
2. **Check file size**: Implement limits for extremely large files
3. **Handle binary files**: Skip or detect binary content appropriately
4. **Test edge cases**: Empty files, single-line files, files with only whitespace

### Index Corruption Recovery
If index becomes corrupted during development:
1. **Delete index directory**: Remove corrupted index completely
2. **Recreate from scratch**: Use create_index instead of open_or_create_index
3. **Validate schema version**: Ensure schema hasn't changed between runs
4. **Check disk space**: Verify sufficient space for index creation

## Success Validation Checklist
- [ ] File `src/indexer.rs` contains complete DocumentIndexer implementation
- [ ] DocumentIndexer can create new instances with index path
- [ ] Method `index_file` handles both code and text files
- [ ] Simple chunking fallback works for unsupported languages
- [ ] Chunks are indexed with all required metadata fields
- [ ] Index writer commits successfully after processing
- [ ] Test `test_document_indexer_creation` passes
- [ ] Test `test_file_indexing` passes with Rust content
- [ ] Memory usage is reasonable for typical file sizes
- [ ] Error handling covers file reading, parsing, and indexing failures

## Files Created For Next Task

Task 10 expects these EXACT files to exist:
1. **C:/code/LLMKG/vectors/tantivy_search/src/indexer.rs** - Complete DocumentIndexer with pipeline integration
2. **All previous files from Tasks 01-08** - Required for search engine implementation

## Context for Task 10

Task 10 will implement the SearchEngine that queries the index created by the DocumentIndexer. The search engine will use the dual-field schema to support both natural language queries and exact special character matching, providing ranked results with highlighting and metadata for result reconstruction.