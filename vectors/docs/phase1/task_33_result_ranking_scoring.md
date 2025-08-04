# Task 33: Implement Result Ranking and Scoring

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 32 completed (Query parsing with special characters)

## Complete Context
You have robust query parsing that handles special characters. Now you need the core search functionality with intelligent result ranking. This implements the actual search execution, document retrieval, and scoring logic that considers both content relevance and chunk metadata.

The ranking system must prioritize exact matches in raw content (for special characters) while also considering processed content matches, and provide meaningful scores for result ordering.

## Exact Steps

1. **Add search execution methods to SearchEngine** (4 minutes):
Add to `C:/code/LLMKG/vectors/tantivy_search/src/search.rs`:
```rust
/// Individual search result with comprehensive metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file_path: String,
    pub content: String,
    pub raw_content: String,
    pub chunk_index: u64,
    pub chunk_start: u64,
    pub chunk_end: u64,
    pub has_overlap: bool,
    pub score: f32,
    pub match_field: MatchField,
}

/// Which field contained the match
#[derive(Debug, Clone, PartialEq)]
pub enum MatchField {
    ProcessedContent,
    RawContent,
    Both,
}

/// Search configuration options
#[derive(Debug, Clone)]
pub struct SearchOptions {
    pub limit: usize,
    pub min_score: f32,
    pub boost_raw_matches: f32,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 50,
            min_score: 0.0,
            boost_raw_matches: 1.5, // Boost raw content matches for special chars
        }
    }
}

impl SearchEngine {
    /// Execute search with ranking and scoring
    pub fn search(&self, query_str: &str, options: SearchOptions) -> Result<Vec<SearchResult>> {
        let reader = self.index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommit)
            .try_into()?;
        let searcher = reader.searcher();
        
        let query = self.parse_query(query_str)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(options.limit))?;
        
        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            if score >= options.min_score {
                let doc = searcher.doc(doc_address)?;
                let mut result = self.create_search_result(doc, score, query_str)?;
                
                // Apply ranking boosts
                result.score = self.apply_ranking_boosts(result.score, &result, &options);
                
                results.push(result);
            }
        }
        
        // Sort by final score (highest first)
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        Ok(results)
    }
    
    /// Create SearchResult from Tantivy document
    fn create_search_result(&self, doc: TantivyDoc, base_score: f32, query_str: &str) -> Result<SearchResult> {
        let content = self.get_field_string(&doc, "content")?;
        let raw_content = self.get_field_string(&doc, "raw_content")?;
        
        // Determine which field(s) contain the match
        let match_field = self.determine_match_field(&content, &raw_content, query_str);
        
        Ok(SearchResult {
            file_path: self.get_field_string(&doc, "file_path")?,
            content,
            raw_content,
            chunk_index: self.get_field_u64(&doc, "chunk_index")?,
            chunk_start: self.get_field_u64(&doc, "chunk_start")?,
            chunk_end: self.get_field_u64(&doc, "chunk_end")?,
            has_overlap: self.get_field_bool(&doc, "has_overlap")?,
            score: base_score,
            match_field,
        })
    }
    
    /// Determine which field(s) contain the search match
    fn determine_match_field(&self, content: &str, raw_content: &str, query_str: &str) -> MatchField {
        let processed_query = SearchEngine::preprocess_query(query_str);
        let clean_query = processed_query.trim_matches('"');
        
        let in_processed = content.to_lowercase().contains(&clean_query.to_lowercase());
        let in_raw = raw_content.to_lowercase().contains(&clean_query.to_lowercase());
        
        match (in_processed, in_raw) {
            (true, true) => MatchField::Both,
            (true, false) => MatchField::ProcessedContent,
            (false, true) => MatchField::RawContent,
            (false, false) => MatchField::ProcessedContent, // Default fallback
        }
    }
    
    /// Apply ranking boosts based on match characteristics
    fn apply_ranking_boosts(&self, base_score: f32, result: &SearchResult, options: &SearchOptions) -> f32 {
        let mut boosted_score = base_score;
        
        // Boost raw content matches (important for special characters)
        if result.match_field == MatchField::RawContent || result.match_field == MatchField::Both {
            boosted_score *= options.boost_raw_matches;
        }
        
        // Boost results without overlap (cleaner chunks)
        if !result.has_overlap {
            boosted_score *= 1.1;
        }
        
        // Slight boost for earlier chunks (often more important)
        if result.chunk_index == 0 {
            boosted_score *= 1.05;
        }
        
        boosted_score
    }
    
    // Helper methods for field extraction
    fn get_field_string(&self, doc: &TantivyDoc, field_name: &str) -> Result<String> {
        let field = self.schema.get_field(field_name)?;
        doc.get_first(field)
            .and_then(|v| v.as_text())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("Field '{}' not found or not text", field_name))
    }
    
    fn get_field_u64(&self, doc: &TantivyDoc, field_name: &str) -> Result<u64> {
        let field = self.schema.get_field(field_name)?;
        doc.get_first(field)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow::anyhow!("Field '{}' not found or not u64", field_name))
    }
    
    fn get_field_bool(&self, doc: &TantivyDoc, field_name: &str) -> Result<bool> {
        let field = self.schema.get_field(field_name)?;
        doc.get_first(field)
            .and_then(|v| v.as_bool())
            .ok_or_else(|| anyhow::anyhow!("Field '{}' not found or not bool", field_name))
    }
}

impl SearchResult {
    /// Get a preview of the content with match highlighting
    pub fn get_preview(&self, max_length: usize) -> String {
        let content = if self.match_field == MatchField::RawContent {
            &self.raw_content
        } else {
            &self.content
        };
        
        if content.len() <= max_length {
            content.clone()
        } else {
            format!("{}...", &content[..max_length])
        }
    }
    
    /// Check if this is a high-confidence match
    pub fn is_high_confidence(&self) -> bool {
        self.score > 1.0 && (self.match_field == MatchField::Both || self.match_field == MatchField::RawContent)
    }
}
```

2. **Add comprehensive ranking tests** (3 minutes):
Add to the test module in `search.rs`:
```rust
#[cfg(test)]
mod ranking_tests {
    use super::*;
    use crate::indexer::DocumentIndexer;
    use tempfile::TempDir;
    use std::fs;

    fn setup_ranking_test_index() -> Result<(TempDir, SearchEngine)> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("ranking_index");
        
        let mut indexer = DocumentIndexer::new(&index_path)?;
        
        // Create files with varying relevance
        let high_relevance = temp_dir.path().join("exact_match.rs");
        fs::write(&high_relevance, r#"
            fn process<T, E>() -> Result<T, E> {
                // Exact Result<T, E> match here
                Ok(value)
            }
        "#)?;
        
        let medium_relevance = temp_dir.path().join("partial_match.rs");
        fs::write(&medium_relevance, r#"
            fn helper() -> Result<String, Error> {
                // Partial Result match
                todo!()
            }
        "#)?;
        
        let low_relevance = temp_dir.path().join("mentions.rs");
        fs::write(&low_relevance, r#"
            // This file mentions result handling
            fn process() { /* result processing */ }
        "#)?;
        
        indexer.index_file(&high_relevance)?;
        indexer.index_file(&medium_relevance)?;
        indexer.index_file(&low_relevance)?;
        indexer.commit()?;
        
        let search_engine = SearchEngine::new(&index_path)?;
        Ok((temp_dir, search_engine))
    }
    
    #[test]
    fn test_basic_search_execution() -> Result<()> {
        let (_temp_dir, search_engine) = setup_ranking_test_index()?;
        
        let results = search_engine.search("Result", SearchOptions::default())?;
        assert!(!results.is_empty(), "Should find results for 'Result'");
        
        // Verify result structure
        for result in &results {
            assert!(!result.file_path.is_empty());
            assert!(!result.content.is_empty());
            assert!(result.score > 0.0);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_special_character_search_ranking() -> Result<()> {
        let (_temp_dir, search_engine) = setup_ranking_test_index()?;
        
        let results = search_engine.search("Result<T, E>", SearchOptions::default())?;
        assert!(!results.is_empty(), "Should find results for 'Result<T, E>'");
        
        // The exact match should score higher
        let exact_match = results.iter().find(|r| r.file_path.contains("exact_match"));
        assert!(exact_match.is_some(), "Should find exact match file");
        
        let exact_result = exact_match.unwrap();
        assert!(exact_result.score > 1.0, "Exact match should have boosted score");
        
        Ok(())
    }
    
    #[test]
    fn test_match_field_determination() {
        let search_engine = SearchEngine::new(Path::new("/dummy")).unwrap_or_else(|_| panic!("Test setup"));
        
        // Test different match scenarios
        let content = "processed content with function";
        let raw_content = "raw content with Result<T,E>";
        
        let match_field = search_engine.determine_match_field(content, raw_content, "function");
        assert_eq!(match_field, MatchField::ProcessedContent);
        
        let match_field = search_engine.determine_match_field(content, raw_content, "Result<T,E>");
        assert_eq!(match_field, MatchField::RawContent);
        
        let both_content = "content with function and Result<T,E>";
        let match_field = search_engine.determine_match_field(both_content, both_content, "function");
        assert_eq!(match_field, MatchField::Both);
    }
    
    #[test]
    fn test_search_options_filtering() -> Result<()> {
        let (_temp_dir, search_engine) = setup_ranking_test_index()?;
        
        let options = SearchOptions {
            limit: 2,
            min_score: 0.1,
            boost_raw_matches: 2.0,
        };
        
        let results = search_engine.search("Result", options)?;
        assert!(results.len() <= 2, "Should respect limit");
        
        for result in &results {
            assert!(result.score >= 0.1, "Should respect min_score");
        }
        
        Ok(())
    }
    
    #[test]
    fn test_result_preview() {
        let result = SearchResult {
            file_path: "test.rs".to_string(),
            content: "This is processed content".to_string(),
            raw_content: "This is raw content with special chars".to_string(),
            chunk_index: 0,
            chunk_start: 0,
            chunk_end: 100,
            has_overlap: false,
            score: 1.0,
            match_field: MatchField::RawContent,
        };
        
        let preview = result.get_preview(20);
        assert!(preview.starts_with("This is raw"));
        assert!(preview.len() <= 23); // 20 chars + "..."
        
        assert!(result.is_high_confidence());
    }
}
```

3. **Verify compilation and tests** (2 minutes):
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo check
cargo test ranking_tests
```

## Success Validation
✓ SearchEngine::search() executes queries and returns ranked results
✓ SearchResult contains comprehensive metadata including match_field
✓ Ranking boosts work (raw content matches, non-overlapping chunks)
✓ Match field determination correctly identifies where matches occur
✓ Search options (limit, min_score, boost factors) are respected
✓ Result preview and confidence assessment methods work
✓ All ranking tests pass

## Next Task Input
Task 34 expects these EXACT components ready:
- `SearchEngine::search()` method with ranking
- `SearchResult` struct with match_field and scoring
- `apply_ranking_boosts()` for intelligent scoring
- Working search options and result filtering