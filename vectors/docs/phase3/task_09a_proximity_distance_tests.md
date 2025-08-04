# Task 09a: Implement Proximity Distance Tests

**Estimated Time: 6 minutes**

## Context
All search methods are implemented. Create tests for proximity search functionality focusing on different term distances.

## Current System State
- `ProximitySearchEngine` has `search_proximity` method
- Need tests to verify proximity distance calculations work correctly
- Tests should use real file indexing, not mocks

## Your Task
Create proximity distance tests in `tests/proximity_tests.rs`.

## Required Implementation

```rust
#[cfg(test)]
mod proximity_distance_tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use crate::indexer::DocumentIndexer;
    use crate::proximity::ProximitySearchEngine;

    #[test]
    fn test_proximity_search_distances() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let proximity_engine = ProximitySearchEngine::new(&index_path)?;
        
        // Create test files with different term distances
        let test_files = vec![
            ("adjacent.rs", "pub fn new() -> Self"),
            ("one_between.rs", "pub static fn new() -> Self"), 
            ("two_between.rs", "pub const GLOBAL fn new() -> Self"),
            ("many_between.rs", "pub is a visibility modifier while fn creates functions"),
            ("far_apart.rs", "pub struct Data { } impl Display { fn fmt() {} }"),
        ];
        
        // Index all test files
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test distance 0 (adjacent terms)
        let results = proximity_engine.search_proximity("pub", "fn", 0)?;
        assert!(results.iter().any(|r| r.file_path.contains("adjacent.rs")));
        assert!(!results.iter().any(|r| r.file_path.contains("one_between.rs")));
        
        // Test distance 1 (one word between)
        let results = proximity_engine.search_proximity("pub", "fn", 1)?;
        assert!(results.iter().any(|r| r.file_path.contains("adjacent.rs")));
        assert!(results.iter().any(|r| r.file_path.contains("one_between.rs")));
        assert!(!results.iter().any(|r| r.file_path.contains("two_between.rs")));
        
        // Test distance 2 (two words between)
        let results = proximity_engine.search_proximity("pub", "fn", 2)?;
        assert!(results.len() >= 3); // Should find first three files
        
        // Test larger distance
        let results = proximity_engine.search_proximity("pub", "fn", 10)?;
        assert!(results.len() >= 4); // Should find most files
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Test for different proximity distances (0, 1, 2, 10)
- [ ] Test files with varying term distances
- [ ] Verification that distance limits work correctly
- [ ] Uses real file indexing with `DocumentIndexer`
- [ ] All tests pass without errors

## Test Data Design
- **adjacent.rs**: Terms right next to each other (distance 0)
- **one_between.rs**: One word between terms (distance 1)  
- **two_between.rs**: Two words between terms (distance 2)
- **many_between.rs**: Multiple words, tests larger distances
- **far_apart.rs**: Terms very far apart, tests distance limits

## Dependencies Needed
- `tempfile` for temporary test directories
- `anyhow` for error handling
- Access to `DocumentIndexer` for file indexing
- Access to `ProximitySearchEngine`