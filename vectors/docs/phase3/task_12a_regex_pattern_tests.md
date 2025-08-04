# Task 12a: Implement Regex Pattern Search Tests

**Estimated Time: 4 minutes**

## Context
Test regex search functionality focusing on common code patterns like function definitions and type declarations.

## Current System State
- `RegexSearchEngine` has `search_regex` method
- Need tests to verify regex patterns work correctly with code
- Tests should use real file indexing, not mocks

## Your Task
Create regex pattern search tests in `tests/regex_tests.rs`.

## Required Implementation

```rust
#[cfg(test)]
mod regex_pattern_tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use crate::indexer::DocumentIndexer;
    use crate::regex_search::RegexSearchEngine;

    #[test]
    fn test_function_pattern_regex() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let regex_engine = RegexSearchEngine::new(&index_path)?;
        
        // Create test files with different function patterns
        let test_files = vec![
            ("pub_fn.rs", "pub fn new() -> Self"),
            ("priv_fn.rs", "fn helper() -> i32"),
            ("async_fn.rs", "pub async fn fetch() -> Result<Data>"),
            ("generic_fn.rs", "fn process<T>(data: T) -> T"),
        ];
        
        // Index all test files
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test function definition pattern: "fn\s+\w+"
        let results = regex_engine.search_regex(r"fn\s+\w+")?;
        assert!(results.len() >= 4); // Should find all function definitions
        
        // Test pub function pattern: "pub\s+fn\s+\w+"
        let results = regex_engine.search_regex(r"pub\s+fn\s+\w+")?;
        assert!(results.iter().any(|r| r.file_path.contains("pub_fn.rs")));
        assert!(!results.iter().any(|r| r.file_path.contains("priv_fn.rs")));
        
        Ok(())
    }

    #[test]
    fn test_type_pattern_regex() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let regex_engine = RegexSearchEngine::new(&index_path)?;
        
        let file_path = temp_dir.path().join("struct.rs");
        fs::write(&file_path, "pub struct Data { field: String }")?;
        indexer.index_file(&file_path)?;
        
        // Test struct pattern: "struct\s+\w+"
        let results = regex_engine.search_regex(r"struct\s+\w+")?;
        assert!(results.iter().any(|r| r.file_path.contains("struct.rs")));
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Test function definition regex patterns
- [ ] Test type definition regex patterns
- [ ] Verify regex matching accuracy
- [ ] Uses real file indexing with `DocumentIndexer`
- [ ] All tests pass without errors

## Dependencies Needed
- `tempfile` for temporary test directories
- `anyhow` for error handling
- Access to `DocumentIndexer` for file indexing
- Access to `RegexSearchEngine`