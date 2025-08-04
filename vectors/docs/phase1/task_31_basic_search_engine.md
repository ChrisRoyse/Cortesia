# Task 31: Implement Basic SearchEngine Struct

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Tasks 1-30 completed (DocumentIndexer with dual-field indexing, SmartChunker with AST parsing)

## Complete Context
You have a working DocumentIndexer that creates Tantivy indexes with dual-field indexing (content + raw_content). Now you need the SearchEngine struct that provides the foundation for all search operations. This struct will manage the Tantivy index reader, schema access, and query parser setup for both processed and raw content fields.

The SearchEngine is the core interface that all subsequent search features will build upon. It handles index loading, query parsing configuration, and basic document retrieval patterns.

## Exact Steps

1. **Create the search engine file** (2 minutes):
Create file `C:/code/LLMKG/vectors/tantivy_search/src/search.rs` with EXACT content:
```rust
//! Basic search engine foundation with dual-field query support

use crate::schema::create_tantivy_index;
use tantivy::{Index, ReloadPolicy, query::QueryParser, collector::TopDocs, schema::Schema, Document as TantivyDoc};
use std::path::Path;
use anyhow::Result;

/// Core search engine for Tantivy-based content search
pub struct SearchEngine {
    index: Index,
    schema: Schema,
    query_parser: QueryParser,
}

impl SearchEngine {
    /// Create new SearchEngine from existing index directory
    pub fn new(index_path: &Path) -> Result<Self> {
        let index = create_tantivy_index(index_path)?;
        let schema = index.schema();
        
        // Configure query parser for both content fields
        let content_field = schema.get_field("content")?;
        let raw_content_field = schema.get_field("raw_content")?;
        
        let query_parser = QueryParser::for_index(
            &index,
            vec![content_field, raw_content_field],
        );
        
        Ok(Self {
            index,
            schema,
            query_parser,
        })
    }
    
    /// Get basic index statistics
    pub fn get_index_stats(&self) -> Result<IndexStats> {
        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        
        Ok(IndexStats {
            total_documents: searcher.num_docs() as usize,
            schema_fields: self.schema.fields().count(),
        })
    }
    
    /// Test if index is accessible and valid
    pub fn validate_index(&self) -> Result<bool> {
        let reader = self.index.reader()?;
        let _searcher = reader.searcher();
        Ok(true)
    }
}

/// Basic index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub total_documents: usize,
    pub schema_fields: usize,
}

#[cfg(test)]
mod basic_tests {
    use super::*;
    use crate::indexer::DocumentIndexer;
    use tempfile::TempDir;
    use std::fs;

    #[test]
    fn test_search_engine_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("test_index");
        
        // Create basic index first
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "test content")?;
        indexer.index_file(&test_file)?;
        indexer.commit()?;
        
        // Test SearchEngine creation
        let search_engine = SearchEngine::new(&index_path)?;
        
        let stats = search_engine.get_index_stats()?;
        assert!(stats.total_documents > 0);
        assert!(stats.schema_fields > 0);
        
        assert!(search_engine.validate_index()?);
        
        Ok(())
    }
    
    #[test]
    fn test_index_validation_missing_directory() {
        let missing_path = Path::new("/nonexistent/path");
        let result = SearchEngine::new(missing_path);
        assert!(result.is_err());
    }
}
```

2. **Update lib.rs** (1 minute):
Add to `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs`:
```rust
pub mod search;
```

3. **Verify compilation** (1 minute):
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo check
```

4. **Run basic tests** (1 minute):
```bash
cargo test search::basic_tests
```

## Success Validation
✓ SearchEngine struct compiles without errors
✓ Can create SearchEngine from existing index
✓ get_index_stats() returns valid statistics
✓ validate_index() works correctly
✓ Error handling for missing index directory
✓ Tests pass: `cargo test search::basic_tests`

## Next Task Input
Task 32 expects these EXACT files ready:
- `C:/code/LLMKG/vectors/tantivy_search/src/search.rs` (SearchEngine struct)
- SearchEngine::new() method functional
- Query parser configured for dual-field search
- Basic index validation working