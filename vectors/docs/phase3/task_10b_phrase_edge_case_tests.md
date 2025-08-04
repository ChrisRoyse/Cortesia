# Task 10b: Implement Phrase Search Edge Case Tests

**Estimated Time: 4 minutes**

## Context
Test phrase search functionality with edge cases like generic types, case sensitivity, and special characters.

## Current System State
- `PhraseSearchEngine` has `search_phrase` method
- Need tests for edge cases and boundary conditions
- Tests should use real file indexing, not mocks

## Your Task
Add phrase search edge case tests to `tests/phrase_tests.rs`.

## Required Implementation

```rust
#[cfg(test)]
mod phrase_edge_case_tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use crate::indexer::DocumentIndexer;
    use crate::phrase::PhraseSearchEngine;

    #[test]
    fn test_phrase_with_generic_types() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let phrase_engine = PhraseSearchEngine::new(&index_path)?;
        
        let file_path = temp_dir.path().join("generics.rs");
        fs::write(&file_path, "Vec<String> data = Vec::new();")?;
        indexer.index_file(&file_path)?;
        
        // Test phrase with generic syntax
        let results = phrase_engine.search_phrase("Vec String")?;
        assert!(results.len() >= 1);
        
        Ok(())
    }

    #[test]
    fn test_case_sensitivity() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let phrase_engine = PhraseSearchEngine::new(&index_path)?;
        
        let file_path = temp_dir.path().join("case.rs");
        fs::write(&file_path, "pub fn New() -> Self")?;
        indexer.index_file(&file_path)?;
        
        // Test case-insensitive matching
        let results1 = phrase_engine.search_phrase("pub fn")?;
        let results2 = phrase_engine.search_phrase("PUB FN")?;
        
        // Both should find the same results
        assert_eq!(results1.len(), results2.len());
        
        Ok(())
    }

    #[test]
    fn test_empty_and_single_word_phrases() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let phrase_engine = PhraseSearchEngine::new(&index_path)?;
        
        let file_path = temp_dir.path().join("simple.rs");
        fs::write(&file_path, "function")?;
        indexer.index_file(&file_path)?;
        
        // Test single word phrase
        let results = phrase_engine.search_phrase("function")?;
        assert!(results.len() >= 1);
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Test phrase search with generic types
- [ ] Test case sensitivity behavior
- [ ] Test single word phrases
- [ ] Handle edge cases gracefully
- [ ] Uses real file indexing with `DocumentIndexer`
- [ ] All tests pass without errors

## Dependencies Needed
- `tempfile` for temporary test directories
- `anyhow` for error handling
- Access to `DocumentIndexer` for file indexing
- Access to `PhraseSearchEngine`

## Next Task
After phrase tests, you'll implement wildcard search tests.