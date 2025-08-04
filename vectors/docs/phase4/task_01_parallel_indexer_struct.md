# Task 01: Create Basic ParallelIndexer Struct

## Context
You are implementing Phase 4 of a vector indexing system that adds Rayon-based parallelism for enterprise-scale performance. This is the first task in implementing parallel indexing capabilities using Rust's Rayon crate.

## Project Structure
The project is a Rust-based document indexing system with the following key components:
- `src/indexer.rs` - Contains `DocumentIndexer` struct for sequential indexing
- `src/search.rs` - Contains search engine implementations
- `src/lib.rs` - Main library entry point

## Dependencies Already Available
```toml
[dependencies]
rayon = "1.7"
anyhow = "1.0"
walkdir = "2.3"
```

## Task Objective
Create the basic `ParallelIndexer` struct in a new file `src/parallel.rs` with:
1. Struct definition with proper fields
2. Constructor method `new()`
3. Basic error handling
4. Thread count detection

## Implementation Requirements

### 1. Create `src/parallel.rs` file
```rust
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::path::Path;
use anyhow::Result;
use crate::indexer::DocumentIndexer;

pub struct ParallelIndexer {
    indexer: Arc<Mutex<DocumentIndexer>>,
    num_threads: usize,
}

impl ParallelIndexer {
    pub fn new(index_path: &Path) -> Result<Self> {
        let indexer = Arc::new(Mutex::new(DocumentIndexer::new(index_path)?));
        let num_threads = std::thread::available_parallelism()?.get();
        
        Ok(Self {
            indexer,
            num_threads,
        })
    }
    
    pub fn get_thread_count(&self) -> usize {
        self.num_threads
    }
}
```

### 2. Update `src/lib.rs`
Add this line to expose the new module:
```rust
pub mod parallel;
```

### 3. Add Basic Test
Include a basic test at the bottom of `src/parallel.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_parallel_indexer_creation() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        
        let parallel_indexer = ParallelIndexer::new(&index_path)?;
        assert!(parallel_indexer.get_thread_count() > 0);
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] File `src/parallel.rs` created successfully
- [ ] `ParallelIndexer` struct compiles without errors
- [ ] Constructor properly handles the index path
- [ ] Thread count detection works
- [ ] Basic test passes
- [ ] No compilation warnings

## Time Limit
10 minutes

## Notes
- Focus only on the struct creation and constructor
- Don't implement indexing methods yet (those come in subsequent tasks)
- Ensure proper error handling with `anyhow::Result`
- The `DocumentIndexer` should already exist in the codebase