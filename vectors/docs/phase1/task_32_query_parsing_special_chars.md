# Task 32: Implement Query Parsing with Special Characters

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 31 completed (Basic SearchEngine struct)

## Complete Context
The SearchEngine can now be created, but it needs robust query parsing that handles special characters commonly found in code: `[workspace]`, `Result<T,E>`, `#[derive]`, `&mut`, etc. Standard Tantivy query parsing treats many of these as special syntax, causing failures.

This task implements intelligent query preprocessing and parsing that automatically detects when special characters should be treated literally vs. as query syntax, ensuring reliable search for code constructs.

## Exact Steps

1. **Add query parsing methods to SearchEngine** (4 minutes):
Add to `C:/code/LLMKG/vectors/tantivy_search/src/search.rs`:
```rust
use tantivy::query::{Query, TermQuery, BooleanQuery, Occur};
use tantivy::schema::Field;
use tantivy::Term;

impl SearchEngine {
    /// Parse query with intelligent special character handling
    pub fn parse_query(&self, query_str: &str) -> Result<Box<dyn Query>> {
        // First try standard parsing
        match self.query_parser.parse_query(query_str) {
            Ok(query) => Ok(query),
            Err(_) => {
                // If standard parsing fails, try literal search
                self.parse_as_literal_query(query_str)
            }
        }
    }
    
    /// Parse query as literal search across both content fields
    fn parse_as_literal_query(&self, query_str: &str) -> Result<Box<dyn Query>> {
        let content_field = self.schema.get_field("content")?;
        let raw_content_field = self.schema.get_field("raw_content")?;
        
        let mut boolean_query = BooleanQuery::new();
        
        // Search in processed content
        let content_term = Term::from_field_text(content_field, query_str);
        let content_query = TermQuery::new(content_term, tantivy::schema::IndexRecordOption::Basic);
        boolean_query.add_clause(Occur::Should, Box::new(content_query));
        
        // Search in raw content
        let raw_content_term = Term::from_field_text(raw_content_field, query_str);
        let raw_content_query = TermQuery::new(raw_content_term, tantivy::schema::IndexRecordOption::Basic);
        boolean_query.add_clause(Occur::Should, Box::new(raw_content_query));
        
        Ok(Box::new(boolean_query))
    }
    
    /// Determine if query contains special characters that need literal handling
    pub fn needs_literal_parsing(query_str: &str) -> bool {
        let special_chars = ['[', ']', '<', '>', '#', '&', '*', '?', '(', ')', '{', '}'];
        query_str.chars().any(|c| special_chars.contains(&c))
    }
    
    /// Preprocess query to handle common code patterns
    pub fn preprocess_query(query_str: &str) -> String {
        let mut processed = query_str.to_string();
        
        // Handle common Rust patterns
        if processed.contains("Result<") && processed.contains(">") {
            // For generic types, wrap in quotes for literal search
            processed = format!("\"{}\"", processed);
        } else if processed.starts_with("#[") && processed.ends_with("]") {
            // For attributes, wrap in quotes
            processed = format!("\"{}\"", processed);
        } else if processed.starts_with("[") && processed.ends_with("]") {
            // For TOML sections, wrap in quotes
            processed = format!("\"{}\"", processed);
        }
        
        processed
    }
}
```

2. **Add comprehensive query parsing tests** (3 minutes):
Add to the test module in `search.rs`:
```rust
#[cfg(test)]
mod query_parsing_tests {
    use super::*;
    use crate::indexer::DocumentIndexer;
    use tempfile::TempDir;
    use std::fs;

    fn setup_test_index_with_special_content() -> Result<(TempDir, SearchEngine)> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("special_char_index");
        
        let mut indexer = DocumentIndexer::new(&index_path)?;
        
        // Create test files with special characters
        let rust_file = temp_dir.path().join("test.rs");
        fs::write(&rust_file, r#"
            fn process<T, E>() -> Result<T, E> {
                #[derive(Debug)]
                struct Config {
                    value: &mut String,
                }
                Ok(todo!())
            }
        "#)?;
        
        let toml_file = temp_dir.path().join("config.toml");
        fs::write(&toml_file, r#"
            [workspace]
            members = ["backend", "frontend"]
            [dependencies]
            tokio = "1.0"
        "#)?;
        
        indexer.index_file(&rust_file)?;
        indexer.index_file(&toml_file)?;
        indexer.commit()?;
        
        let search_engine = SearchEngine::new(&index_path)?;
        Ok((temp_dir, search_engine))
    }
    
    #[test]
    fn test_special_character_detection() {
        assert!(SearchEngine::needs_literal_parsing("[workspace]"));
        assert!(SearchEngine::needs_literal_parsing("Result<T,E>"));
        assert!(SearchEngine::needs_literal_parsing("#[derive]"));
        assert!(SearchEngine::needs_literal_parsing("&mut"));
        assert!(!SearchEngine::needs_literal_parsing("simple query"));
    }
    
    #[test]
    fn test_query_preprocessing() {
        assert_eq!(
            SearchEngine::preprocess_query("Result<T,E>"),
            "\"Result<T,E>\""
        );
        assert_eq!(
            SearchEngine::preprocess_query("#[derive]"),
            "\"#[derive]\""
        );
        assert_eq!(
            SearchEngine::preprocess_query("[workspace]"),
            "\"[workspace]\""
        );
        assert_eq!(
            SearchEngine::preprocess_query("simple query"),
            "simple query"
        );
    }
    
    #[test]
    fn test_query_parsing_with_special_chars() -> Result<()> {
        let (_temp_dir, search_engine) = setup_test_index_with_special_content()?;
        
        let test_queries = vec![
            "[workspace]",
            "Result<T, E>",
            "#[derive]",
            "&mut",
            "simple",
        ];
        
        for query_str in test_queries {
            let query = search_engine.parse_query(query_str);
            assert!(query.is_ok(), "Should parse query: {}", query_str);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_literal_query_parsing() -> Result<()> {
        let (_temp_dir, search_engine) = setup_test_index_with_special_content()?;
        
        // Test that literal parsing works for complex queries
        let complex_query = "Result<T, E>";
        let query = search_engine.parse_as_literal_query(complex_query)?;
        
        // Query should be created without error
        assert!(!query.query_terms().is_empty());
        
        Ok(())
    }
}
```

3. **Verify compilation and tests** (2 minutes):
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo check
cargo test query_parsing_tests
```

## Success Validation
✓ SearchEngine::parse_query() handles special characters
✓ needs_literal_parsing() correctly identifies special cases
✓ preprocess_query() wraps special patterns in quotes
✓ parse_as_literal_query() creates valid Boolean queries
✓ All query parsing tests pass
✓ No compilation errors

## Next Task Input
Task 33 expects these EXACT methods ready:
- `SearchEngine::parse_query()` working with special characters
- Literal query parsing as fallback
- Query preprocessing for common code patterns
- Comprehensive test coverage for special character handling