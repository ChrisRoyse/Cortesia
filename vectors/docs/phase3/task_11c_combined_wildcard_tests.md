# Task 11c: Implement Combined Wildcard Pattern Tests

**Estimated Time: 3 minutes**

## Context
Test wildcard search functionality with combined patterns using both star (*) and question mark (?) wildcards, plus edge cases.

## Current System State
- `WildcardSearchEngine` has `search_wildcard` method
- Need tests for combined patterns and edge cases
- Tests should use real file indexing, not mocks

## Your Task
Add combined wildcard pattern tests to `tests/wildcard_tests.rs`.

## Required Implementation

```rust
#[cfg(test)]
mod combined_wildcard_tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use crate::indexer::DocumentIndexer;
    use crate::wildcard::WildcardSearchEngine;

    #[test]
    fn test_combined_star_and_question_patterns() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let wildcard_engine = WildcardSearchEngine::new(&index_path)?;
        
        let test_files = vec![
            ("func1.rs", "fn func1_test() -> Result<()>"),
            ("func2.rs", "fn func2_demo() -> Result<()>"),
            ("method.rs", "fn method_call() -> Result<()>"),
        ];
        
        // Index all test files
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test combined pattern: "func?_*"
        let results = wildcard_engine.search_wildcard("func?_*")?;
        assert!(results.iter().any(|r| r.file_path.contains("func1.rs")));
        assert!(results.iter().any(|r| r.file_path.contains("func2.rs")));
        assert!(!results.iter().any(|r| r.file_path.contains("method.rs")));
        
        Ok(())
    }

    #[test]
    fn test_wildcard_edge_cases() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let wildcard_engine = WildcardSearchEngine::new(&index_path)?;
        
        let file_path = temp_dir.path().join("edge.rs");
        fs::write(&file_path, "a")?;
        indexer.index_file(&file_path)?;
        
        // Test single character with wildcards
        let results = wildcard_engine.search_wildcard("?")?;
        assert!(results.len() >= 1);
        
        let results = wildcard_engine.search_wildcard("*")?;
        assert!(results.len() >= 1);
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Test combined star and question mark patterns
- [ ] Test edge cases with minimal input
- [ ] Verify complex pattern matching
- [ ] Uses real file indexing with `DocumentIndexer`
- [ ] All tests pass without errors

## Test Patterns
- `func?_*` - "func" + one char + "_" + anything
- `?` - matches any single character
- `*` - matches anything

## Dependencies Needed
- `tempfile` for temporary test directories
- `anyhow` for error handling
- Access to `DocumentIndexer` for file indexing
- Access to `WildcardSearchEngine`

## Next Task
After wildcard tests, you'll implement regex and fuzzy search tests.