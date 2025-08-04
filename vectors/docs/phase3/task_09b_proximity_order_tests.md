# Task 09b: Implement Proximity Order Independence Tests

**Estimated Time: 4 minutes**

## Context
Test that proximity search works correctly regardless of term order (i.e., searching for "pub fn" vs "fn pub" should yield the same results).

## Current System State
- `ProximitySearchEngine` has `search_proximity` method
- Need to verify order independence functionality
- Tests should use real file indexing, not mocks

## Your Task
Add proximity order independence tests to `tests/proximity_tests.rs`.

## Required Implementation

```rust
#[cfg(test)]
mod proximity_order_tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use crate::indexer::DocumentIndexer;
    use crate::proximity::ProximitySearchEngine;

    #[test] 
    fn test_proximity_search_order_independence() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let proximity_engine = ProximitySearchEngine::new(&index_path)?;
        
        let file_path = temp_dir.path().join("test.rs");
        fs::write(&file_path, "fn pub new() -> Self")?;
        indexer.index_file(&file_path)?;
        
        // Test both term orders should find the same result
        let results1 = proximity_engine.search_proximity("pub", "fn", 1)?;
        let results2 = proximity_engine.search_proximity("fn", "pub", 1)?;
        
        assert_eq!(results1.len(), results2.len());
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Test for term order independence 
- [ ] Verification that "term1 term2" and "term2 term1" yield same results
- [ ] Uses real file indexing with `DocumentIndexer`
- [ ] All tests pass without errors

## Dependencies Needed
- `tempfile` for temporary test directories
- `anyhow` for error handling
- Access to `DocumentIndexer` for file indexing
- Access to `ProximitySearchEngine`

## Next Task
After proximity tests, you'll implement phrase search tests.