# Task 00_3: Set Up Tantivy Integration and Schema

**Estimated Time:** 8-10 minutes  
**Prerequisites:** Task 00_2 (crate structure created)  
**Dependencies:** Must be completed before Task 00_4

## Objective
Implement Tantivy integration with a proper schema for document indexing, establishing the foundation for full-text search that Phase 2 tasks require.

## Context
You are implementing the core Tantivy integration that will power the vector search system. This must handle special characters commonly found in code (like `[workspace]`, `Result<T,E>`, `#[derive]`, etc.) and provide the schema structure that document indexing and search operations will use.

## Task Details

### What You Need to Do
1. **Update src/indexing/mod.rs with complete Tantivy integration:**
   - Define schema with content, file_path, and chunk_index fields
   - Implement DocumentIndexer with real Tantivy backend
   - Handle index creation and management
   - Ensure Windows compatibility

2. **Create standardized schema configuration that handles special characters:**
   - file_path field: TEXT | STORED (consistent with test utilities)
   - content field: TEXT | STORED (processed content)  
   - raw_content field: TEXT | STORED (original unprocessed content)
   - chunk_index field: U64 | STORED (for ordering within files)
   
   **IMPORTANT**: This schema MUST match the StandardSchema in test_utils.rs to ensure consistency across all tests and implementations.

3. **Implement error handling for Tantivy operations:**
   - Index creation failures
   - Schema conflicts
   - I/O errors during index operations

### Implementation Details

#### Update src/indexing/mod.rs
```rust
//! Document indexing capabilities for vector search

use anyhow::{Context, Result as AnyhowResult};
use std::path::{Path, PathBuf};
use tantivy::{
    collector::TopDocs,
    doc,
    schema::{Field, Schema, STORED, TEXT, STRING, INDEXED},
    Index, IndexWriter, IndexReader,
    query::QueryParser,
};
use thiserror::Error;
use tracing::{info, warn, error, debug};

#[derive(Error, Debug)]
pub enum IndexingError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),
    
    #[error("Schema error: {0}")]
    Schema(String),
    
    #[error("Index path error: {0}")]
    IndexPath(String),
}

/// Schema fields for document indexing - MUST match StandardSchema in test_utils.rs
#[derive(Debug, Clone)]
pub struct DocumentSchema {
    pub schema: Schema,
    pub file_path_field: Field,
    pub content_field: Field,
    pub raw_content_field: Field,
    pub chunk_index_field: Field,
}

impl DocumentSchema {
    pub fn new() -> AnyhowResult<Self> {
        let mut schema_builder = Schema::builder();
        
        // Standardized field definitions - MUST match test_utils::StandardSchema
        let file_path_field = schema_builder.add_text_field("file_path", TEXT | STORED);
        let content_field = schema_builder.add_text_field("content", TEXT | STORED);
        let raw_content_field = schema_builder.add_text_field("raw_content", TEXT | STORED);
        let chunk_index_field = schema_builder.add_u64_field("chunk_index", STORED);
        
        let schema = schema_builder.build();
        
        Ok(DocumentSchema {
            schema,
            file_path_field,
            content_field,
            raw_content_field,
            chunk_index_field,
        })
    }
}

/// Document indexer for creating searchable indices
pub struct DocumentIndexer {
    index: Index,
    schema: DocumentSchema,
    writer: Option<IndexWriter>,
    index_dir: PathBuf,
}

impl DocumentIndexer {
    pub fn new<P: AsRef<Path>>(index_dir: P) -> Result<Self, IndexingError> {
        let index_dir = index_dir.as_ref().to_path_buf();
        
        info!("Creating document indexer at: {:?}", index_dir);
        
        // Create index directory if it doesn't exist
        std::fs::create_dir_all(&index_dir)
            .with_context(|| format!("Failed to create index directory: {:?}", index_dir))?;
        
        // Create schema
        let schema = DocumentSchema::new()
            .map_err(|e| IndexingError::Schema(e.to_string()))?;
        
        // Create or open index
        let index = if index_dir.join("meta.json").exists() {
            debug!("Opening existing index");
            Index::open_in_dir(&index_dir)?
        } else {
            debug!("Creating new index");
            Index::create_in_dir(&index_dir, schema.schema.clone())?
        };
        
        Ok(DocumentIndexer {
            index,
            schema,
            writer: None,
            index_dir,
        })
    }
    
    pub fn get_writer(&mut self) -> Result<&mut IndexWriter, IndexingError> {
        if self.writer.is_none() {
            info!("Creating index writer");
            let writer = self.index.writer(50_000_000)?; // 50MB heap
            self.writer = Some(writer);
        }
        Ok(self.writer.as_mut().unwrap())
    }
    
    pub fn add_document(
        &mut self,
        file_path: &str,
        content: &str,
        raw_content: &str,
        chunk_index: u64,
    ) -> Result<(), IndexingError> {
        let writer = self.get_writer()?;
        
        let doc = doc!(
            self.schema.file_path_field => file_path,
            self.schema.content_field => content,
            self.schema.raw_content_field => raw_content,
            self.schema.chunk_index_field => chunk_index
        );
        
        writer.add_document(doc)?;
        debug!("Added document: {} (chunk {})", file_path, chunk_index);
        
        Ok(())
    }
    
    pub fn commit(&mut self) -> Result<(), IndexingError> {
        if let Some(writer) = self.writer.as_mut() {
            info!("Committing index changes");
            writer.commit()?;
        }
        Ok(())
    }
    
    pub fn get_reader(&self) -> Result<IndexReader, IndexingError> {
        Ok(self.index.reader()?)
    }
    
    /// Get schema for search operations
    pub fn get_schema(&self) -> &DocumentSchema {
        &self.schema
    }
    
    /// Get document count in index
    pub fn document_count(&self) -> Result<usize, IndexingError> {
        let reader = self.get_reader()?;
        let searcher = reader.searcher();
        Ok(searcher.num_docs() as usize)
    }
}

impl Drop for DocumentIndexer {
    fn drop(&mut self) {
        if let Err(e) = self.commit() {
            error!("Failed to commit index on drop: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_schema_creation() {
        let schema = DocumentSchema::new().unwrap();
        assert_eq!(schema.schema.fields().count(), 4);
    }
    
    #[test]
    fn test_indexer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let indexer = DocumentIndexer::new(temp_dir.path()).unwrap();
        assert_eq!(indexer.document_count().unwrap(), 0);
    }
    
    #[test]
    fn test_document_indexing() {
        let temp_dir = TempDir::new().unwrap();
        let mut indexer = DocumentIndexer::new(temp_dir.path()).unwrap();
        
        indexer.add_document(
            "test.rs",
            "fn main() { println!(\"Hello [world]!\"); }",
            "fn main() {\n    println!(\"Hello [world]!\");\n}",
            0
        ).unwrap();
        
        indexer.commit().unwrap();
        assert_eq!(indexer.document_count().unwrap(), 1);
    }
    
    #[test]
    fn test_special_characters() {
        let temp_dir = TempDir::new().unwrap();
        let mut indexer = DocumentIndexer::new(temp_dir.path()).unwrap();
        
        // Test various special characters common in code
        let test_content = r#"[workspace] members = ["crate1", "crate2"] fn test() -> Result<T, E> { #[derive(Debug)] struct Test { field: Vec<Option<String>> } }"#;
        let raw_content = r#"
        [workspace]
        members = ["crate1", "crate2"]
        
        fn test() -> Result<T, E> {
            #[derive(Debug)]
            struct Test { field: Vec<Option<String>> }
        }
        "#;
        
        indexer.add_document("Cargo.toml", test_content, raw_content, 0).unwrap();
        indexer.commit().unwrap();
        assert_eq!(indexer.document_count().unwrap(), 1);
    }
}
```

### Expected Output Files
- **Modified:** `crates/vector-search/src/indexing/mod.rs` (complete implementation)
- **Validation:** Tests should pass with `cargo test -p vector-search`

## Success Criteria
- [ ] DocumentSchema properly defines all required fields matching test_utils::StandardSchema
- [ ] DocumentIndexer creates and manages Tantivy indices
- [ ] Special characters in code are handled correctly
- [ ] Index creation works on Windows file paths
- [ ] All tests pass including special character tests
- [ ] Error handling covers all Tantivy failure modes
- [ ] `cargo check -p vector-search` succeeds
- [ ] Schema consistency with test utilities verified
- [ ] Field ordering matches: file_path, content, raw_content, chunk_index

## Common Pitfalls to Avoid
- Don't use unsupported Tantivy field configurations
- Ensure proper error conversion from Tantivy errors
- Handle Windows path separators correctly
- Don't forget to implement Drop for IndexWriter cleanup
- Test with realistic code content including special characters

## Context for Next Task
Task 00_4 will implement basic document indexing functionality that uses this Tantivy schema to process and index files from the filesystem.

## Integration Notes
This Tantivy integration provides:
- Schema structure that Phase 2 tasks expect (matching test_utils::StandardSchema)
- Special character handling for code files
- Windows-compatible file path handling
- Error types that integrate with neuromorphic error patterns
- Foundation for both indexing and search operations
- Consistent field definitions across all components

**CRITICAL**: The DocumentSchema implementation MUST exactly match the StandardSchema from test_utils.rs to ensure all tests and integrations work correctly. Any deviation will cause compilation or test failures in dependent tasks.