# Task 12b: Implement Basic Fuzzy Search Tests

**Estimated Time: 3 minutes**

## Context
Test fuzzy search functionality for handling typos and minor spelling variations in search terms.

## Current System State
- `FuzzySearchEngine` has `search_fuzzy` method
- Need tests to verify fuzzy matching works with common typos
- Tests should use real file indexing, not mocks

## Your Task
Add basic fuzzy search tests to `tests/fuzzy_tests.rs`.

## Required Implementation

```rust
#[cfg(test)]
mod fuzzy_typo_tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use crate::indexer::DocumentIndexer;
    use crate::fuzzy::FuzzySearchEngine;

    #[test]
    fn test_basic_typo_correction() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let fuzzy_engine = FuzzySearchEngine::new(&index_path)?;
        
        // Create test files with common terms
        let test_files = vec![
            ("function.rs", "function implementation here"),
            ("variable.rs", "variable declaration here"),
            ("structure.rs", "structure definition here"),
        ];
        
        // Index all test files
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test single character typos
        let results = fuzzy_engine.search_fuzzy("functon", 1)?; // Missing 'i'
        assert!(results.iter().any(|r| r.file_path.contains("function.rs")));
        
        let results = fuzzy_engine.search_fuzzy("variabel", 1)?; // Swapped 'l' and 'e'
        assert!(results.iter().any(|r| r.file_path.contains("variable.rs")));
        
        let results = fuzzy_engine.search_fuzzy("structur", 1)?; // Missing final 'e'
        assert!(results.iter().any(|r| r.file_path.contains("structure.rs")));
        
        Ok(())
    }

    #[test]
    fn test_character_substitution() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let fuzzy_engine = FuzzySearchEngine::new(&index_path)?;
        
        let file_path = temp_dir.path().join("method.rs");
        fs::write(&file_path, "method call example")?;
        indexer.index_file(&file_path)?;
        
        // Test character substitution
        let results = fuzzy_engine.search_fuzzy("methad", 1)?; // 'o' -> 'a'
        assert!(results.len() >= 1);
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Test single character insertion, deletion, substitution
- [ ] Test common programming term typos
- [ ] Verify fuzzy matching within edit distance
- [ ] Uses real file indexing with `DocumentIndexer`
- [ ] All tests pass without errors

## Typo Examples Tested
- "functon" → "function" (missing character)
- "variabel" → "variable" (character swap)
- "structur" → "structure" (missing character)
- "methad" → "method" (character substitution)

## Dependencies Needed
- `tempfile` for temporary test directories
- `anyhow` for error handling
- Access to `DocumentIndexer` for file indexing
- Access to `FuzzySearchEngine`