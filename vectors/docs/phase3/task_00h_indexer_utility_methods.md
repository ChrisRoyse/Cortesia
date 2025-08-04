# Task 00h: Add DocumentIndexer Utility Methods

**Estimated Time: 8 minutes**
**Lines of Code: 20**
**Prerequisites: Task 00g completed**

## Context

Phase 3 tasks may reference statistics and utility methods on DocumentIndexer. This task adds the final utility methods.

## Your Task

Add utility methods to complete the DocumentIndexer implementation in `src/indexer.rs`.

## Required Implementation

Add these methods to the existing `impl DocumentIndexer` block and add the IndexStats struct:

```rust
    /// Get index statistics
    pub fn stats(&self) -> Result<IndexStats> {
        Ok(IndexStats {
            total_documents: 0, // Minimal implementation
            index_size_bytes: 0,
        })
    }
}

/// Statistics about the search index
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// Total number of documents in the index
    pub total_documents: usize,
    
    /// Size of the index in bytes
    pub index_size_bytes: u64,
}
```

## Success Criteria

- [ ] `stats()` method added returning `Result<IndexStats>`
- [ ] `IndexStats` struct defined with proper fields
- [ ] `Debug` and `Clone` derives on `IndexStats`
- [ ] Documentation comments for struct and fields
- [ ] Minimal but functional implementation
- [ ] File compiles without errors

## Validation

Run `cargo check` - should compile without errors.

## Next Task

Task 00i will create the error types module.