# Task 11a: Implement Star Wildcard Pattern Tests

**Estimated Time: 4 minutes**

## Context
Test wildcard search functionality focusing on star (*) patterns that match zero or more characters.

## Current System State
- `WildcardSearchEngine` has `search_wildcard` method
- Need tests to verify star wildcard patterns work correctly
- Tests should use real file indexing, not mocks

## Your Task
Create star wildcard pattern tests in `tests/wildcard_tests.rs`.

## Required Implementation

```rust
#[cfg(test)]
mod star_wildcard_tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use crate::indexer::DocumentIndexer;
    use crate::wildcard::WildcardSearchEngine;

    #[test]
    fn test_star_wildcard_patterns() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let wildcard_engine = WildcardSearchEngine::new(&index_path)?;
        
        // Create test files with various function patterns
        let test_files = vec![
            ("new_func.rs", "pub fn new() -> Self"),
            ("get_data.rs", "pub fn get_data() -> Data"),
            ("set_value.rs", "pub fn set_value(val: i32)"),
            ("init_system.rs", "pub fn init_system() -> Result<()>"),
        ];
        
        // Index all test files
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test star at end: "get*"
        let results = wildcard_engine.search_wildcard("get*")?;
        assert!(results.iter().any(|r| r.file_path.contains("get_data.rs")));
        
        // Test star at beginning: "*_data"
        let results = wildcard_engine.search_wildcard("*_data")?;
        assert!(results.iter().any(|r| r.file_path.contains("get_data.rs")));
        
        // Test star in middle: "pub*fn"
        let results = wildcard_engine.search_wildcard("pub*fn")?;
        assert!(results.len() >= 4); // Should match all files
        
        Ok(())
    }

    #[test]
    fn test_multiple_stars() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let wildcard_engine = WildcardSearchEngine::new(&index_path)?;
        
        let file_path = temp_dir.path().join("complex.rs");
        fs::write(&file_path, "impl Display for MyStruct")?;
        indexer.index_file(&file_path)?;
        
        // Test multiple stars: "*Display*MyStruct"
        let results = wildcard_engine.search_wildcard("*Display*MyStruct")?;
        assert!(results.len() >= 1);
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Test star wildcard at end of pattern
- [ ] Test star wildcard at beginning of pattern
- [ ] Test star wildcard in middle of pattern
- [ ] Test multiple star wildcards in one pattern
- [ ] Uses real file indexing with `DocumentIndexer`
- [ ] All tests pass without errors

## Dependencies Needed
- `tempfile` for temporary test directories
- `anyhow` for error handling
- Access to `DocumentIndexer` for file indexing
- Access to `WildcardSearchEngine`