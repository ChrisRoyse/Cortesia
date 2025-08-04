# Task 02: Implement index_files_parallel() Method

## Context
You are implementing Phase 4 of a vector indexing system. In the previous task, the basic `ParallelIndexer` struct was created. Now you need to implement the core parallel indexing functionality using Rayon.

## Current State
- `src/parallel.rs` exists with `ParallelIndexer` struct
- Constructor `new()` method is implemented
- Thread count detection is working

## Task Objective
Implement the `index_files_parallel()` method that processes multiple files in parallel using Rayon's `par_iter()`.

## Implementation Requirements

### 1. Add IndexingStats struct
Add this to `src/parallel.rs` before the `ParallelIndexer` implementation:
```rust
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct IndexingStats {
    pub files_processed: usize,
    pub total_size: usize,
    pub start_time: Instant,
}

impl IndexingStats {
    pub fn new() -> Self {
        Self {
            files_processed: 0,
            total_size: 0,
            start_time: Instant::now(),
        }
    }
    
    pub fn duration(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}
```

### 2. Add get_index_path() helper method
Add this method to the `ParallelIndexer` implementation:
```rust
impl ParallelIndexer {
    // ... existing methods ...
    
    fn get_index_path(&self) -> Result<PathBuf> {
        // This is a temporary implementation - will be improved later
        Ok(PathBuf::from("./index"))
    }
}
```

### 3. Implement index_files_parallel() method
Add this method to the `ParallelIndexer` implementation:
```rust
pub fn index_files_parallel(&self, file_paths: Vec<PathBuf>) -> Result<IndexingStats> {
    let stats = Arc::new(Mutex::new(IndexingStats::new()));
    
    // Process files in parallel using Rayon
    file_paths.par_iter().try_for_each(|file_path| -> Result<()> {
        let content = std::fs::read_to_string(file_path)?;
        
        // Each thread gets its own indexer instance
        let mut local_indexer = DocumentIndexer::new(&self.get_index_path()?)?;
        local_indexer.index_file(file_path)?;
        
        // Update stats atomically
        {
            let mut stats_guard = stats.lock().unwrap();
            stats_guard.files_processed += 1;
            stats_guard.total_size += content.len();
        }
        
        Ok(())
    })?;
    
    let final_stats = stats.lock().unwrap().clone();
    Ok(final_stats)
}
```

### 4. Update the test
Replace the existing test in `src/parallel.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    
    #[test]
    fn test_parallel_indexer_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        
        let parallel_indexer = ParallelIndexer::new(&index_path)?;
        assert!(parallel_indexer.get_thread_count() > 0);
        
        Ok(())
    }
    
    #[test]
    fn test_index_files_parallel_basic() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let parallel_indexer = ParallelIndexer::new(&index_path)?;
        
        // Create test files
        let mut test_files = Vec::new();
        for i in 0..3 {
            let file_path = temp_dir.path().join(format!("test_{}.rs", i));
            let content = format!("pub fn test_{}() {{ println!(\"Hello {}\"); }}", i, i);
            fs::write(&file_path, content)?;
            test_files.push(file_path);
        }
        
        // Index files in parallel
        let stats = parallel_indexer.index_files_parallel(test_files)?;
        
        assert_eq!(stats.files_processed, 3);
        assert!(stats.total_size > 0);
        assert!(stats.duration().as_millis() < 5000); // Should complete quickly
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] `IndexingStats` struct implemented correctly
- [ ] `get_index_path()` helper method added
- [ ] `index_files_parallel()` method implemented with Rayon
- [ ] Parallel processing works correctly
- [ ] Statistics tracking is accurate
- [ ] Both tests pass
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Focus on getting the basic parallel processing working
- Error handling should be robust
- The `get_index_path()` is temporary and will be improved in later tasks
- Rayon's `par_iter()` handles the parallel distribution automatically