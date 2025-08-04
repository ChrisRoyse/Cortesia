# Task 04: Implement get_index_path() Helper Method

## Context
You are implementing Phase 4 of a vector indexing system. The `get_index_path()` method was added as a temporary placeholder in a previous task. Now you need to implement it properly to return the actual index path from the wrapped `DocumentIndexer`.

## Current State
- `src/parallel.rs` exists with `ParallelIndexer` struct
- `IndexingStats` is fully implemented
- `index_files_parallel()` method exists but uses a temporary `get_index_path()`

## Task Objective
Replace the temporary `get_index_path()` implementation with a proper one that retrieves the index path from the wrapped `DocumentIndexer`.

## Implementation Requirements

### 1. Understand DocumentIndexer structure
First, you need to check what methods are available on `DocumentIndexer`. Look at `src/indexer.rs` to understand the interface.

### 2. Improve get_index_path() method
Replace the temporary implementation in `src/parallel.rs`:
```rust
fn get_index_path(&self) -> Result<PathBuf> {
    // Access the indexer to get its index path
    let indexer = self.indexer.lock().unwrap();
    
    // Assuming DocumentIndexer has a get_index_path() or index_path field
    // Adapt based on the actual DocumentIndexer implementation
    Ok(indexer.get_index_path().clone())
    
    // Alternative if DocumentIndexer stores path as a field:
    // Ok(indexer.index_path.clone())
    
    // Alternative if you need to construct it:
    // Ok(PathBuf::from(&indexer.index_directory))
}
```

### 3. Add index path validation
Add a helper method to validate the index path:
```rust
fn ensure_index_path_exists(&self) -> Result<()> {
    let index_path = self.get_index_path()?;
    
    if !index_path.exists() {
        std::fs::create_dir_all(&index_path)
            .map_err(|e| anyhow::anyhow!("Failed to create index directory: {}", e))?;
    }
    
    Ok(())
}
```

### 4. Update index_files_parallel() to use validation
Modify the beginning of `index_files_parallel()` method:
```rust
pub fn index_files_parallel(&self, file_paths: Vec<PathBuf>) -> Result<IndexingStats> {
    // Ensure index path exists before starting
    self.ensure_index_path_exists()?;
    
    let stats = Arc::new(Mutex::new(IndexingStats::new()));
    
    // ... rest of the existing implementation ...
}
```

### 5. Add test for path handling
Add this test to the test module:
```rust
#[test]
fn test_index_path_handling() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("test_index");
    
    // Index path doesn't exist initially
    assert!(!index_path.exists());
    
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Get the index path
    let retrieved_path = parallel_indexer.get_index_path()?;
    assert_eq!(retrieved_path, index_path);
    
    // Ensure path validation works
    parallel_indexer.ensure_index_path_exists()?;
    assert!(index_path.exists());
    
    Ok(())
}

#[test]
fn test_index_path_with_existing_directory() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let index_path = temp_dir.path().join("existing_index");
    
    // Create the directory first
    std::fs::create_dir_all(&index_path)?;
    assert!(index_path.exists());
    
    let parallel_indexer = ParallelIndexer::new(&index_path)?;
    
    // Should work with existing directory
    parallel_indexer.ensure_index_path_exists()?;
    assert!(index_path.exists());
    
    Ok(())
}
```

## Success Criteria
- [ ] `get_index_path()` returns correct path from `DocumentIndexer`
- [ ] `ensure_index_path_exists()` creates directories as needed
- [ ] Index path validation is integrated into parallel indexing
- [ ] Both new tests pass
- [ ] Existing tests still pass
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- You may need to examine `src/indexer.rs` to understand the `DocumentIndexer` interface
- If `DocumentIndexer` doesn't have a path getter, you may need to store the path separately
- Handle both existing and non-existing index directories
- Ensure thread safety when accessing the wrapped indexer