# Task 00e: Create DocumentIndexer and IndexingConfig Structs

**Estimated Time: 10 minutes**
**Lines of Code: 30**
**Prerequisites: Task 00d completed**

## Context

Phase 3 tasks reference a `DocumentIndexer` struct for testing. This task creates the struct definitions needed.

## Your Task

Create `src/indexer.rs` with DocumentIndexer and IndexingConfig struct definitions.

## Required Implementation

Create `src/indexer.rs` with exactly this content:

```rust
//! Document indexing functionality for the vector search system

use std::path::Path;
use anyhow::Result;
use crate::boolean::BooleanSearchEngine;

/// Document indexer for managing search index operations
pub struct DocumentIndexer {
    /// The underlying search engine
    engine: BooleanSearchEngine,
    
    /// Indexing configuration
    config: IndexingConfig,
}

/// Configuration for document indexing operations
#[derive(Debug, Clone)]
pub struct IndexingConfig {
    /// Maximum size for document chunks (in characters)
    pub max_chunk_size: usize,
    
    /// Overlap size between chunks (in characters)
    pub chunk_overlap: usize,
    
    /// Whether to detect programming language
    pub detect_language: bool,
    
    /// Whether to extract semantic types
    pub extract_semantic_types: bool,
    
    /// Batch size for bulk indexing operations
    pub batch_size: usize,
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            max_chunk_size: 2000,
            chunk_overlap: 200,
            detect_language: true,
            extract_semantic_types: true,
            batch_size: 100,
        }
    }
}
```

## Success Criteria

- [ ] File `src/indexer.rs` created
- [ ] `DocumentIndexer` struct with `engine` and `config` fields
- [ ] `IndexingConfig` struct with all 5 configuration fields
- [ ] `Default` implementation for `IndexingConfig`
- [ ] Proper imports and documentation
- [ ] File compiles without errors

## Validation

Run `cargo check` - should compile without errors.

## Next Task

Task 00f will add the DocumentIndexer constructor and basic methods.