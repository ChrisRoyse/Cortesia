# Task 00_4: Create Basic Document Indexing Functionality

**Estimated Time:** 9-10 minutes  
**Prerequisites:** Task 00_3 (Tantivy integration completed)  
**Dependencies:** Must be completed before Task 00_5

## Objective
Implement basic document indexing functionality that can process files from the filesystem, apply simple chunking strategies, and index them using the Tantivy schema from Task 00_3.

## Context
You are building the document processing pipeline that Phase 2 tasks will use. This includes file discovery, content processing, basic chunking (before AST-based smart chunking), and feeding documents into the Tantivy index. The system must handle common code file types and provide the foundation for more advanced chunking strategies.

## Task Details

### What You Need to Do
1. **Update src/chunking/mod.rs with basic chunking implementation:**
   - Simple line-based chunking strategy
   - Language detection from file extensions  
   - Content preprocessing for special characters
   - Chunk size management

2. **Create document processing pipeline:**
   - File discovery and filtering
   - Content reading and encoding handling
   - Chunk generation and indexing
   - Progress tracking and error recovery

3. **Implement integration between chunking and indexing:**
   - Connect SmartChunker to DocumentIndexer
   - Handle chunk metadata properly
   - Ensure consistent error handling

### Implementation Details

#### Update src/chunking/mod.rs
```rust
//! Smart chunking for AST-based document processing

use anyhow::{Context, Result as AnyhowResult};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, warn, info};

#[derive(Error, Debug)]
pub enum ChunkingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("AST parsing error: {0}")]
    AstParsing(String),
    
    #[error("Language detection error: {0}")]
    LanguageDetection(String),
    
    #[error("Content processing error: {0}")]
    ContentProcessing(String),
}

/// Code chunk with metadata
#[derive(Debug, Clone)]
pub struct CodeChunk {
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub chunk_type: String,
    pub language: String,
    pub file_path: String,
}

/// Chunking configuration
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    pub max_chunk_lines: usize,
    pub min_chunk_lines: usize,
    pub overlap_lines: usize,
    pub supported_extensions: Vec<String>,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            max_chunk_lines: 100,
            min_chunk_lines: 5,
            overlap_lines: 5,
            supported_extensions: vec![
                "rs".to_string(),
                "py".to_string(),
                "js".to_string(),
                "ts".to_string(),
                "md".to_string(),
                "toml".to_string(),
                "json".to_string(),
                "yaml".to_string(),
                "yml".to_string(),
            ],
        }
    }
}

/// Smart chunker for AST-based code processing
pub struct SmartChunker {
    config: ChunkingConfig,
}

impl SmartChunker {
    pub fn new() -> Result<Self, ChunkingError> {
        Ok(Self {
            config: ChunkingConfig::default(),
        })
    }
    
    pub fn with_config(config: ChunkingConfig) -> Result<Self, ChunkingError> {
        Ok(Self { config })
    }
    
    /// Detect language from file extension
    pub fn detect_language<P: AsRef<Path>>(file_path: P) -> Result<String, ChunkingError> {
        let path = file_path.as_ref();
        let extension = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("unknown");
            
        let language = match extension.to_lowercase().as_str() {
            "rs" => "rust",
            "py" => "python", 
            "js" => "javascript",
            "ts" => "typescript",
            "md" => "markdown",
            "toml" => "toml",
            "json" => "json",
            "yaml" | "yml" => "yaml",
            _ => "text",
        };
        
        debug!("Detected language '{}' for file: {:?}", language, path);
        Ok(language.to_string())
    }
    
    /// Check if file should be processed based on extension
    pub fn should_process_file<P: AsRef<Path>>(&self, file_path: P) -> bool {
        let path = file_path.as_ref();
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            self.config.supported_extensions.contains(&extension.to_lowercase())
        } else {
            false
        }
    }
    
    /// Process file into chunks
    pub fn chunk_file<P: AsRef<Path>>(&self, file_path: P) -> Result<Vec<CodeChunk>, ChunkingError> {
        let path = file_path.as_ref();
        let file_path_str = path.to_string_lossy().to_string();
        
        debug!("Chunking file: {:?}", path);
        
        if !self.should_process_file(path) {
            return Ok(vec![]);
        }
        
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read file: {:?}", path))?;
            
        let language = Self::detect_language(path)?;
        
        self.chunk_content(&content, &file_path_str, &language)
    }
    
    /// Chunk content string into CodeChunk objects
    pub fn chunk_content(
        &self,
        content: &str,
        file_path: &str,
        language: &str,
    ) -> Result<Vec<CodeChunk>, ChunkingError> {
        let lines: Vec<&str> = content.lines().collect();
        let mut chunks = Vec::new();
        
        if lines.is_empty() {
            return Ok(chunks);
        }
        
        let mut start_line = 0;
        
        while start_line < lines.len() {
            let end_line = std::cmp::min(
                start_line + self.config.max_chunk_lines,
                lines.len()
            );
            
            // Skip chunks that are too small (unless it's the last possible chunk)
            if end_line - start_line < self.config.min_chunk_lines && end_line < lines.len() {
                start_line += self.config.max_chunk_lines - self.config.overlap_lines;
                continue;
            }
            
            let chunk_lines = &lines[start_line..end_line];
            let chunk_content = chunk_lines.join("\n");
            
            let chunk = CodeChunk {
                content: chunk_content,
                start_line: start_line + 1, // 1-indexed for human readability
                end_line,
                chunk_type: "text_block".to_string(), // Will be enhanced with AST info later
                language: language.to_string(),
                file_path: file_path.to_string(),
            };
            
            chunks.push(chunk);
            
            // Calculate next start with overlap
            if end_line >= lines.len() {
                break;
            }
            
            start_line = end_line - self.config.overlap_lines;
        }
        
        debug!("Created {} chunks for file: {}", chunks.len(), file_path);
        Ok(chunks)
    }
    
    /// Process directory recursively
    pub fn chunk_directory<P: AsRef<Path>>(&self, dir_path: P) -> Result<Vec<CodeChunk>, ChunkingError> {
        let mut all_chunks = Vec::new();
        
        for entry in walkdir::WalkDir::new(dir_path.as_ref())
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.file_type().is_file() {
                match self.chunk_file(entry.path()) {
                    Ok(mut chunks) => {
                        all_chunks.append(&mut chunks);
                    }
                    Err(e) => {
                        warn!("Failed to chunk file {:?}: {}", entry.path(), e);
                        // Continue processing other files
                    }
                }
            }
        }
        
        info!("Processed directory with {} total chunks", all_chunks.len());
        Ok(all_chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    
    #[test]
    fn test_language_detection() {
        assert_eq!(SmartChunker::detect_language("test.rs").unwrap(), "rust");
        assert_eq!(SmartChunker::detect_language("test.py").unwrap(), "python");
        assert_eq!(SmartChunker::detect_language("test.md").unwrap(), "markdown");
        assert_eq!(SmartChunker::detect_language("test.unknown").unwrap(), "text");
    }
    
    #[test]
    fn test_should_process_file() {
        let chunker = SmartChunker::new().unwrap();
        assert!(chunker.should_process_file("test.rs"));
        assert!(chunker.should_process_file("test.py"));
        assert!(!chunker.should_process_file("test.exe"));
        assert!(!chunker.should_process_file("test"));
    }
    
    #[test]
    fn test_chunk_content() {
        let chunker = SmartChunker::new().unwrap();
        let content = "line1\nline2\nline3\nline4\nline5";
        
        let chunks = chunker.chunk_content(content, "test.txt", "text").unwrap();
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].language, "text");
        assert_eq!(chunks[0].file_path, "test.txt");
    }
    
    #[test]
    fn test_chunk_file() {
        let temp_dir = TempDir::new().unwrap();
        let test_file = temp_dir.path().join("test.rs");
        
        let content = r#"
fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_something() {
        assert_eq!(2 + 2, 4);
    }
}
        "#;
        
        fs::write(&test_file, content).unwrap();
        
        let chunker = SmartChunker::new().unwrap();
        let chunks = chunker.chunk_file(&test_file).unwrap();
        
        assert!(!chunks.is_empty());
        assert_eq!(chunks[0].language, "rust");
        assert!(chunks[0].content.contains("fn main()"));
    }
    
    #[test]
    fn test_special_characters() {
        let chunker = SmartChunker::new().unwrap();
        let content = r#"
[workspace]
members = ["crate1", "crate2"]

fn test() -> Result<T, E> {
    #[derive(Debug)]
    struct Test { field: Vec<Option<String>> }
}
        "#;
        
        let chunks = chunker.chunk_content(content, "Cargo.toml", "toml").unwrap();
        assert!(!chunks.is_empty());
        assert!(chunks[0].content.contains("[workspace]"));
        assert!(chunks[0].content.contains("Result<T, E>"));
    }
}
```

#### Create src/lib.rs integration helper
Add to the existing src/lib.rs:

```rust
/// Integration helper for indexing documents
pub async fn index_directory<P: AsRef<Path>>(
    directory_path: P,
    index_path: P,
) -> Result<usize> {
    use crate::chunking::SmartChunker;
    use crate::indexing::DocumentIndexer;
    
    let chunker = SmartChunker::new()?;
    let mut indexer = DocumentIndexer::new(index_path)?;
    
    let chunks = chunker.chunk_directory(directory_path)?;
    let total_chunks = chunks.len();
    
    for (chunk_index, chunk) in chunks.into_iter().enumerate() {
        indexer.add_document(
            &chunk.content,
            &chunk.file_path,
            chunk_index as u64,
            &chunk.language,
        )?;
    }
    
    indexer.commit()?;
    
    tracing::info!("Indexed {} chunks successfully", total_chunks);
    Ok(total_chunks)
}
```

### Expected Output Files
- **Modified:** `crates/vector-search/src/chunking/mod.rs` (complete implementation)
- **Modified:** `crates/vector-search/src/lib.rs` (add integration helper)
- **Required:** Add `walkdir` to workspace dependencies in root Cargo.toml

## Success Criteria
- [ ] SmartChunker processes files and creates chunks correctly
- [ ] Language detection works for supported file types
- [ ] Special characters in code are preserved in chunks
- [ ] Directory processing handles errors gracefully
- [ ] Integration with DocumentIndexer works correctly
- [ ] All tests pass with `cargo test -p vector-search`
- [ ] `index_directory` helper function compiles
- [ ] **Compilation Verification**: `cargo check -p vector-search` succeeds
- [ ] **Test Verification**: `cargo test -p vector-search test_chunk_content` passes
- [ ] **Integration Verification**: No dependency conflicts or missing imports

## Common Pitfalls to Avoid
- Don't lose special characters during content processing
- Handle empty files and directories gracefully
- Ensure chunk boundaries don't break in middle of critical code
- Test with realistic directory structures
- Don't panic on file read errors, log and continue

## Context for Next Task
Task 00_5 will implement basic search functionality that can query the indexed documents and return relevant results, completing the basic search pipeline.

## Final Verification Steps
After completing the implementation, run these commands to verify everything works:

```bash
# Verify compilation
cargo check -p vector-search

# Run specific tests
cargo test -p vector-search test_chunk_content
cargo test -p vector-search test_language_detection
cargo test -p vector-search test_special_characters

# Run all vector-search tests
cargo test -p vector-search

# Verify workspace still compiles
cargo check
```

## Integration Notes
This document processing pipeline provides:
- File discovery and filtering by extension
- Basic chunking with configurable parameters  
- Language detection for proper indexing
- Integration point for DocumentIndexer
- Foundation for Phase 2's AST-based smart chunking
- Error handling that continues processing despite individual file failures