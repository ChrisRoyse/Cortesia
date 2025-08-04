# Task 11b: Implement Question Mark Wildcard Tests

**Estimated Time: 3 minutes**

## Context
Test wildcard search functionality focusing on question mark (?) patterns that match exactly one character.

## Current System State
- `WildcardSearchEngine` has `search_wildcard` method
- Need tests to verify question mark wildcard patterns work correctly
- Tests should use real file indexing, not mocks

## Your Task
Add question mark wildcard tests to `tests/wildcard_tests.rs`.

## Required Implementation

```rust
#[cfg(test)]
mod question_wildcard_tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use crate::indexer::DocumentIndexer;
    use crate::wildcard::WildcardSearchEngine;

    #[test]
    fn test_question_mark_patterns() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let wildcard_engine = WildcardSearchEngine::new(&index_path)?;
        
        // Create test files with single character variations
        let test_files = vec![
            ("get1.rs", "fn get1() -> i32"),
            ("get2.rs", "fn get2() -> i32"),
            ("geta.rs", "fn geta() -> i32"),
            ("getxy.rs", "fn getxy() -> i32"),
        ];
        
        // Index all test files
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test single question mark: "get?"
        let results = wildcard_engine.search_wildcard("get?")?;
        assert!(results.iter().any(|r| r.file_path.contains("get1.rs")));
        assert!(results.iter().any(|r| r.file_path.contains("get2.rs")));
        assert!(results.iter().any(|r| r.file_path.contains("geta.rs")));
        assert!(!results.iter().any(|r| r.file_path.contains("getxy.rs"))); // Too long
        
        Ok(())
    }

    #[test]
    fn test_multiple_question_marks() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let wildcard_engine = WildcardSearchEngine::new(&index_path)?;
        
        let file_path = temp_dir.path().join("test.rs");
        fs::write(&file_path, "fn abc123def")?;
        indexer.index_file(&file_path)?;
        
        // Test multiple question marks: "abc???def"
        let results = wildcard_engine.search_wildcard("abc???def")?;
        assert!(results.len() >= 1);
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Test single question mark wildcard
- [ ] Test multiple question marks in sequence
- [ ] Verify exact character count matching
- [ ] Uses real file indexing with `DocumentIndexer`
- [ ] All tests pass without errors

## Test Patterns
- `get?` - matches "get" followed by exactly one character
- `abc???def` - matches "abc" + 3 characters + "def"

## Dependencies Needed
- `tempfile` for temporary test directories
- `anyhow` for error handling
- Access to `DocumentIndexer` for file indexing
- Access to `WildcardSearchEngine`