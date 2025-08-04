# Task 012: CrossChunkBooleanHandler Structure

## Prerequisites
- Task 001-011 completed: Boolean search and validation functional
- BooleanSearchEngine with search_boolean() method working
- SearchResult struct available from boolean.rs (Task 005)
- DocumentLevelValidator available from validator.rs

## Required Imports
```rust
// Add to src/cross_chunk.rs for this task
use anyhow::{Result, Context};
use crate::boolean::{BooleanSearchEngine, SearchResult};
use std::collections::HashMap;
use tempfile::TempDir;
use std::path::Path;
use tantivy::schema::{Schema, TEXT, STORED};
use tantivy::{Index, doc};
```

## Context
You have boolean search working within individual chunks, but documents are often split into multiple chunks for indexing. The CrossChunkBooleanHandler ensures boolean logic works correctly across all chunks of a document.

For example:
- Document has "pub" in chunk 1 and "struct" in chunk 3
- Query "pub AND struct" should find this document
- Individual chunks might not satisfy the boolean logic, but the complete document does

**CRITICAL**: This task defines the CANONICAL DocumentResult struct for cross-chunk operations.

## Your Task (10 minutes max)
Implement the CrossChunkBooleanHandler structure with basic document grouping functionality.

## Success Criteria
1. Write failing test for CrossChunkBooleanHandler creation
2. Implement CrossChunkBooleanHandler struct
3. Write failing test for document grouping functionality
4. Implement basic search_across_chunks method structure
5. All tests pass after implementation

## Implementation Steps

### 1. RED: Write failing test for handler creation
```rust
// Create src/cross_chunk.rs
use anyhow::Result;
use crate::boolean::{BooleanSearchEngine, SearchResult};
use std::collections::HashMap;

pub struct CrossChunkBooleanHandler {
    boolean_engine: BooleanSearchEngine,
}

// Define DocumentResult for cross-chunk operations - CANONICAL DEFINITION
#[derive(Debug, Clone)]
pub struct DocumentResult {
    pub file_path: String,
    pub content: String,
    pub chunks: usize,
    pub score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::path::Path;
    
    #[test]
    fn test_cross_chunk_handler_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        
        create_chunked_test_index(&index_path)?;
        let boolean_engine = BooleanSearchEngine::new(&index_path)?;
        
        // This should fail initially - struct doesn't exist yet
        let handler = CrossChunkBooleanHandler {
            boolean_engine,
        };
        
        // Basic smoke test
        assert!(std::ptr::addr_of!(handler) as *const _ != std::ptr::null());
        
        Ok(())
    }
    
    fn create_chunked_test_index(index_path: &Path) -> Result<()> {
        use tantivy::schema::{Schema, TEXT, STORED};
        use tantivy::{Index, doc};
        
        let mut schema_builder = Schema::builder();
        let file_path_field = schema_builder.add_text_field("file_path", TEXT | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let raw_content_field = schema_builder.add_text_field("raw_content", TEXT | STORED);
        let chunk_index_field = schema_builder.add_u64_field("chunk_index", STORED);
        let schema = schema_builder.build();
        
        let index = Index::create_in_dir(index_path, schema.clone())?;
        let mut index_writer = index.writer(50_000_000)?;
        
        // Create chunks from the same document
        index_writer.add_document(doc!(
            file_path_field => "large_file.rs",
            content_field => "pub struct LargeStruct {",
            raw_content_field => "pub struct LargeStruct {",
            chunk_index_field => 0u64
        ))?;
        
        index_writer.add_document(doc!(
            file_path_field => "large_file.rs", 
            content_field => "    data: Vec<String>,",
            raw_content_field => "    data: Vec<String>,",
            chunk_index_field => 1u64
        ))?;
        
        index_writer.add_document(doc!(
            file_path_field => "large_file.rs",
            content_field => "} impl Display for LargeStruct {",
            raw_content_field => "} impl Display for LargeStruct {",
            chunk_index_field => 2u64
        ))?;
        
        index_writer.commit()?;
        Ok(())
    }
}
```

### 2. GREEN: Implement CrossChunkBooleanHandler structure
```rust
// Add to src/cross_chunk.rs
impl CrossChunkBooleanHandler {
    pub fn new(boolean_engine: BooleanSearchEngine) -> Self {
        Self { boolean_engine }
    }
}
```

### 3. RED: Write failing test for document grouping
```rust
// Add to tests in src/cross_chunk.rs
#[test]
fn test_document_grouping() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_chunked_test_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let handler = CrossChunkBooleanHandler::new(boolean_engine);
    
    // This method doesn't exist yet - should fail
    let grouped = handler.group_chunks_by_document(&get_test_chunk_results())?;
    
    // Should group chunks by file path
    assert_eq!(grouped.len(), 1, "Should group chunks from same document");
    assert!(grouped.contains_key("large_file.rs"));
    assert_eq!(grouped["large_file.rs"].len(), 3, "Should have 3 chunks for large_file.rs");
    
    Ok(())
}

fn get_test_chunk_results() -> Vec<SearchResult> {
    vec![
        SearchResult {
            file_path: "large_file.rs".to_string(),
            content: "pub struct LargeStruct {".to_string(),
            chunk_index: 0,
            score: 0.9,
        },
        SearchResult {
            file_path: "large_file.rs".to_string(),
            content: "    data: Vec<String>,".to_string(),
            chunk_index: 1,
            score: 0.7,
        },
        SearchResult {
            file_path: "large_file.rs".to_string(),
            content: "} impl Display for LargeStruct {".to_string(),
            chunk_index: 2,
            score: 0.8,
        },
        SearchResult {
            file_path: "other_file.rs".to_string(),
            content: "fn test() {}".to_string(),
            chunk_index: 0,
            score: 0.6,
        },
    ]
}
```

### 4. GREEN: Implement group_chunks_by_document method
```rust
// Add to CrossChunkBooleanHandler impl
impl CrossChunkBooleanHandler {
    fn group_chunks_by_document(&self, chunk_results: &[SearchResult]) -> Result<HashMap<String, Vec<SearchResult>>> {
        let mut document_groups: HashMap<String, Vec<SearchResult>> = HashMap::new();
        
        for result in chunk_results {
            document_groups.entry(result.file_path.clone())
                .or_insert_with(Vec::new)
                .push(result.clone());
        }
        
        Ok(document_groups)
    }
}
```

### 5. RED: Write failing test for search_across_chunks
```rust
#[test]
fn test_search_across_chunks() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("index");
    
    create_chunked_test_index(&index_path)?;
    let boolean_engine = BooleanSearchEngine::new(&index_path)?;
    let handler = CrossChunkBooleanHandler::new(boolean_engine);
    
    // This method doesn't exist yet - should fail
    let results = handler.search_across_chunks("pub AND Display")?;
    
    // Should find the document where "pub" is in chunk 0 and "Display" is in chunk 2
    assert_eq!(results.len(), 1, "Should find one document with terms across chunks");
    assert_eq!(results[0].file_path, "large_file.rs");
    assert_eq!(results[0].chunks, 3, "Should indicate 3 chunks were combined");
    assert!(results[0].content.contains("pub"));
    assert!(results[0].content.contains("Display"));
    
    Ok(())
}
```

### 6. GREEN: Implement basic search_across_chunks structure
```rust
// Add to CrossChunkBooleanHandler impl
impl CrossChunkBooleanHandler {
    pub fn search_across_chunks(&self, query: &str) -> Result<Vec<DocumentResult>> {
        // Get chunk-level results from the boolean engine
        let chunk_results = self.boolean_engine.search_boolean(query)?;
        
        // Group chunks by document
        let document_groups = self.group_chunks_by_document(&chunk_results)?;
        
        // Process each document group
        let mut document_results = Vec::new();
        for (file_path, chunks) in document_groups {
            if let Some(doc_result) = self.process_document_chunks(&file_path, &chunks, query)? {
                document_results.push(doc_result);
            }
        }
        
        // Sort by score
        document_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        Ok(document_results)
    }
    
    fn process_document_chunks(&self, file_path: &str, chunks: &[SearchResult], query: &str) -> Result<Option<DocumentResult>> {
        // Combine content from all chunks
        let combined_content = chunks.iter()
            .map(|c| c.content.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        
        // Get the highest score among chunks
        let max_score = chunks.iter()
            .map(|c| c.score)
            .fold(0.0f32, f32::max);
        
        // For now, assume all combined documents satisfy the query
        // Next task will add proper validation
        Ok(Some(DocumentResult {
            file_path: file_path.to_string(),
            content: combined_content,
            chunks: chunks.len(),
            score: max_score,
        }))
    }
}
```

## Validation Checklist
- [ ] CrossChunkBooleanHandler struct compiles and can be created
- [ ] Document grouping functionality works correctly
- [ ] Basic search_across_chunks method exists and compiles
- [ ] Chunk content is properly combined
- [ ] All tests pass

## Creates for Future Tasks
- CrossChunkBooleanHandler struct with document grouping
- **CANONICAL** DocumentResult struct for cross-chunk operations (other tasks import this)
- group_chunks_by_document() method
- Basic search_across_chunks() method structure

## Exports for Other Tasks
```rust
// From cross_chunk.rs
pub struct DocumentResult { file_path, content, chunks, score }
pub struct CrossChunkBooleanHandler { boolean_engine }

impl CrossChunkBooleanHandler {
    pub fn search_across_chunks(&self, query: &str) -> Result<Vec<DocumentResult>> { ... }
}
```

## Context for Next Task
Next task will implement proper boolean validation across chunks to ensure the combined document actually satisfies the boolean query requirements, using the DocumentResult struct and cross-chunk handler created here.