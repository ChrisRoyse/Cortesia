# Task 00g: Add DocumentIndexer File Indexing Methods

**Estimated Time: 10 minutes**
**Lines of Code: 25**
**Prerequisites: Task 00f completed**

## Context

Phase 3 tasks need to index files using DocumentIndexer. This task adds the file indexing methods that tests expect.

## Your Task

Add file indexing methods to the `impl DocumentIndexer` block in `src/indexer.rs`.

## Required Implementation

Add these methods to the existing `impl DocumentIndexer` block:

```rust
    /// Index a single document file
    pub fn index_file(&mut self, file_path: &Path) -> Result<usize> {
        // Minimal implementation for Phase 3 compatibility
        let _content = std::fs::read_to_string(file_path)?;
        
        // For now, just return 1 to indicate one "chunk" was indexed
        // Real implementation will be added in later phases
        Ok(1)
    }
    
    /// Index multiple files in batch
    pub fn index_files<P: AsRef<Path>>(&mut self, file_paths: impl Iterator<Item = P>) -> Result<usize> {
        let mut total_indexed = 0;
        
        for file_path in file_paths {
            let path = file_path.as_ref();
            if let Ok(count) = self.index_file(path) {
                total_indexed += count;
            }
        }
        
        Ok(total_indexed)
    }
    
    /// Commit any pending changes to the index
    pub fn commit(&mut self) -> Result<()> {
        // Minimal implementation for Phase 3 compatibility
        Ok(())
    }
```

## Success Criteria

- [ ] `index_file()` method added accepting `&Path`, returning `Result<usize>`
- [ ] `index_files()` method added accepting iterator, returning `Result<usize>`
- [ ] `commit()` method added returning `Result<()>`
- [ ] Methods have minimal but functional implementations
- [ ] File reading logic in `index_file()` works correctly
- [ ] File compiles without errors

## Validation

Run `cargo check` - should compile without errors.

## Next Task

Task 00h will add utility methods to DocumentIndexer.