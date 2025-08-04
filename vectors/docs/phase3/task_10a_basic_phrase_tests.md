# Task 10a: Implement Basic Phrase Search Tests

**Estimated Time: 6 minutes**

## Context
Test exact phrase matching functionality where terms must appear in exact sequence.

## Current System State
- `PhraseSearchEngine` has `search_phrase` method
- Need tests to verify exact phrase matching works correctly
- Tests should use real file indexing, not mocks

## Your Task
Create basic phrase search tests in `tests/phrase_tests.rs`.

## Required Implementation

```rust
#[cfg(test)]
mod basic_phrase_tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use crate::indexer::DocumentIndexer;
    use crate::phrase::PhraseSearchEngine;

    #[test]
    fn test_exact_phrase_matching() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let phrase_engine = PhraseSearchEngine::new(&index_path)?;
        
        // Create test files with exact phrases
        let test_files = vec![
            ("exact_match.rs", "pub fn new() -> Self { }"),
            ("partial_match.rs", "pub static fn new() -> Self { }"),
            ("no_match.rs", "fn public new() -> Self { }"),
            ("reversed.rs", "fn pub new() -> Self { }"),
        ];
        
        // Index all test files
        for (filename, content) in test_files {
            let file_path = temp_dir.path().join(filename);
            fs::write(&file_path, content)?;
            indexer.index_file(&file_path)?;
        }
        
        // Test exact phrase "pub fn"
        let results = phrase_engine.search_phrase("pub fn")?;
        assert!(results.iter().any(|r| r.file_path.contains("exact_match.rs")));
        assert!(!results.iter().any(|r| r.file_path.contains("partial_match.rs")));
        assert!(!results.iter().any(|r| r.file_path.contains("no_match.rs")));
        assert!(!results.iter().any(|r| r.file_path.contains("reversed.rs")));
        
        // Test different phrase "fn new"
        let results = phrase_engine.search_phrase("fn new")?;
        assert!(results.len() >= 2); // Should find exact_match and partial_match
        
        Ok(())
    }

    #[test]
    fn test_phrase_with_punctuation() -> anyhow::Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("index");
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let phrase_engine = PhraseSearchEngine::new(&index_path)?;
        
        let file_path = temp_dir.path().join("punct.rs");
        fs::write(&file_path, "println!(\"Hello, world!\");")?;
        indexer.index_file(&file_path)?;
        
        // Test phrase with punctuation
        let results = phrase_engine.search_phrase("Hello world")?;
        assert!(results.len() >= 1);
        
        Ok(())
    }
}
```

## Success Criteria
- [ ] Test for exact phrase matching
- [ ] Verification that partial matches are excluded
- [ ] Test with punctuation in phrases
- [ ] Uses real file indexing with `DocumentIndexer`
- [ ] All tests pass without errors

## Test Data Design
- **exact_match.rs**: Contains exact phrase sequence
- **partial_match.rs**: Has extra words between phrase terms
- **no_match.rs**: Has similar but not exact terms
- **reversed.rs**: Has phrase terms in reverse order
- **punct.rs**: Tests phrase matching with punctuation

## Dependencies Needed
- `tempfile` for temporary test directories
- `anyhow` for error handling
- Access to `DocumentIndexer` for file indexing
- Access to `PhraseSearchEngine`