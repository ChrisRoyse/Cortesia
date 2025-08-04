# Task 34: Implement Search Result Highlighting

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 33 completed (Result ranking and scoring)

## Complete Context
You have a working search engine that returns ranked results with comprehensive metadata. Now you need result highlighting that shows users exactly where their search terms appear in the content. This is crucial for code search where users need to see the exact context of matches, especially for special characters.

The highlighting system must handle both processed and raw content, work with special characters, and provide configurable snippet extraction around matches.

## Exact Steps

1. **Add highlighting functionality to SearchResult** (4 minutes):
Add to `C:/code/LLMKG/vectors/tantivy_search/src/search.rs`:
```rust
/// Highlighting configuration
#[derive(Debug, Clone)]
pub struct HighlightOptions {
    pub snippet_length: usize,
    pub max_snippets: usize,
    pub highlight_prefix: String,
    pub highlight_suffix: String,
    pub context_chars: usize,
}

impl Default for HighlightOptions {
    fn default() -> Self {
        Self {
            snippet_length: 200,
            max_snippets: 3,
            highlight_prefix: "**".to_string(),
            highlight_suffix: "**".to_string(),
            context_chars: 50,
        }
    }
}

/// A highlighted snippet from search results
#[derive(Debug, Clone)]
pub struct HighlightedSnippet {
    pub text: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub match_count: usize,
}

impl SearchResult {
    /// Generate highlighted snippets for this search result
    pub fn get_highlighted_snippets(&self, query_str: &str, options: &HighlightOptions) -> Vec<HighlightedSnippet> {
        let content_to_search = match self.match_field {
            MatchField::RawContent => &self.raw_content,
            MatchField::ProcessedContent => &self.content,
            MatchField::Both => &self.raw_content, // Prefer raw for special chars
        };
        
        self.extract_snippets_with_highlights(content_to_search, query_str, options)
    }
    
    /// Extract and highlight snippets from content
    fn extract_snippets_with_highlights(&self, content: &str, query_str: &str, options: &HighlightOptions) -> Vec<HighlightedSnippet> {
        let search_term = Self::clean_query_for_highlighting(query_str);
        let mut snippets = Vec::new();
        
        // Find all matches (case-insensitive)
        let content_lower = content.to_lowercase();
        let search_lower = search_term.to_lowercase();
        let mut match_positions = Vec::new();
        
        let mut start_pos = 0;
        while let Some(pos) = content_lower[start_pos..].find(&search_lower) {
            let absolute_pos = start_pos + pos;
            match_positions.push((absolute_pos, absolute_pos + search_term.len()));
            start_pos = absolute_pos + 1;
        }
        
        // Create snippets around matches
        let mut used_ranges = Vec::new();
        for (match_start, match_end) in match_positions.iter().take(options.max_snippets) {
            let snippet_start = match_start.saturating_sub(options.context_chars);
            let snippet_end = (match_end + options.context_chars).min(content.len());
            
            // Avoid overlapping snippets
            if used_ranges.iter().any(|(start, end)| {
                !(snippet_end <= *start || snippet_start >= *end)
            }) {
                continue;
            }
            
            let snippet_text = &content[snippet_start..snippet_end];
            let highlighted = self.apply_highlighting(snippet_text, &search_term, options);
            
            snippets.push(HighlightedSnippet {
                text: highlighted,
                start_offset: snippet_start,
                end_offset: snippet_end,
                match_count: 1, // Could be enhanced to count multiple matches in snippet
            });
            
            used_ranges.push((snippet_start, snippet_end));
        }
        
        // If no specific matches found, return beginning of content
        if snippets.is_empty() && !content.is_empty() {
            let fallback_length = options.snippet_length.min(content.len());
            snippets.push(HighlightedSnippet {
                text: content[..fallback_length].to_string(),
                start_offset: 0,
                end_offset: fallback_length,
                match_count: 0,
            });
        }
        
        snippets
    }
    
    /// Apply highlighting markup to a snippet
    fn apply_highlighting(&self, snippet: &str, search_term: &str, options: &HighlightOptions) -> String {
        let snippet_lower = snippet.to_lowercase();
        let search_lower = search_term.to_lowercase();
        let mut result = String::new();
        let mut last_end = 0;
        
        let mut start_pos = 0;
        while let Some(pos) = snippet_lower[start_pos..].find(&search_lower) {
            let absolute_pos = start_pos + pos;
            
            // Add text before match
            result.push_str(&snippet[last_end..absolute_pos]);
            
            // Add highlighted match
            result.push_str(&options.highlight_prefix);
            result.push_str(&snippet[absolute_pos..absolute_pos + search_term.len()]);
            result.push_str(&options.highlight_suffix);
            
            last_end = absolute_pos + search_term.len();
            start_pos = last_end;
        }
        
        // Add remaining text
        result.push_str(&snippet[last_end..]);
        result
    }
    
    /// Clean query string for highlighting (remove quotes, etc.)
    fn clean_query_for_highlighting(query_str: &str) -> String {
        query_str.trim_matches('"').trim().to_string()
    }
    
    /// Get a single best highlighted snippet
    pub fn get_best_snippet(&self, query_str: &str, options: &HighlightOptions) -> Option<HighlightedSnippet> {
        let snippets = self.get_highlighted_snippets(query_str, options);
        snippets.into_iter().max_by_key(|s| s.match_count)
    }
    
    /// Get formatted preview with highlighting
    pub fn get_highlighted_preview(&self, query_str: &str, max_length: usize) -> String {
        let options = HighlightOptions {
            snippet_length: max_length,
            max_snippets: 1,
            ..Default::default()
        };
        
        if let Some(snippet) = self.get_best_snippet(query_str, &options) {
            snippet.text
        } else {
            self.get_preview(max_length)
        }
    }
}

impl HighlightedSnippet {
    /// Check if this snippet contains highlighted matches
    pub fn has_highlights(&self) -> bool {
        self.match_count > 0
    }
    
    /// Get the snippet text without highlighting markup
    pub fn get_plain_text(&self, highlight_options: &HighlightOptions) -> String {
        self.text
            .replace(&highlight_options.highlight_prefix, "")
            .replace(&highlight_options.highlight_suffix, "")
    }
}
```

2. **Add comprehensive highlighting tests** (3 minutes):
Add to the test module in `search.rs`:
```rust
#[cfg(test)]
mod highlighting_tests {
    use super::*;
    use crate::indexer::DocumentIndexer;
    use tempfile::TempDir;
    use std::fs;

    fn create_sample_search_result() -> SearchResult {
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "This is a function that processes Result<T, E> types and handles errors properly.".to_string(),
            raw_content: "This is a function that processes Result<T, E> types and handles errors properly.".to_string(),
            chunk_index: 0,
            chunk_start: 0,
            chunk_end: 100,
            has_overlap: false,
            score: 1.0,
            match_field: MatchField::Both,
        }
    }
    
    #[test]
    fn test_basic_highlighting() {
        let result = create_sample_search_result();
        let options = HighlightOptions::default();
        
        let snippets = result.get_highlighted_snippets("function", &options);
        assert!(!snippets.is_empty(), "Should find snippets for 'function'");
        
        let snippet = &snippets[0];
        assert!(snippet.text.contains("**function**"), "Should highlight 'function'");
        assert!(snippet.has_highlights(), "Should indicate highlights present");
    }
    
    #[test]
    fn test_special_character_highlighting() {
        let result = create_sample_search_result();
        let options = HighlightOptions::default();
        
        let snippets = result.get_highlighted_snippets("Result<T, E>", &options);
        assert!(!snippets.is_empty(), "Should find snippets for 'Result<T, E>'");
        
        let snippet = &snippets[0];
        assert!(snippet.text.contains("**Result<T, E>**"), 
               "Should highlight special characters: {}", snippet.text);
    }
    
    #[test]
    fn test_custom_highlight_options() {
        let result = create_sample_search_result();
        let options = HighlightOptions {
            highlight_prefix: "<mark>".to_string(),
            highlight_suffix: "</mark>".to_string(),
            context_chars: 20,
            snippet_length: 100,
            max_snippets: 2,
        };
        
        let snippets = result.get_highlighted_snippets("function", &options);
        assert!(!snippets.is_empty());
        
        let snippet = &snippets[0];
        assert!(snippet.text.contains("<mark>function</mark>"));
    }
    
    #[test]
    fn test_multiple_matches_in_content() {
        let result = SearchResult {
            file_path: "test.rs".to_string(),
            content: "function one() {} function two() {} function three() {}".to_string(),
            raw_content: "function one() {} function two() {} function three() {}".to_string(),
            chunk_index: 0,
            chunk_start: 0,
            chunk_end: 100,
            has_overlap: false,
            score: 1.0,
            match_field: MatchField::Both,
        };
        
        let options = HighlightOptions {
            max_snippets: 3,
            context_chars: 10,
            ..Default::default()
        };
        
        let snippets = result.get_highlighted_snippets("function", &options);
        assert!(snippets.len() <= 3, "Should respect max_snippets limit");
        
        for snippet in &snippets {
            assert!(snippet.text.contains("**function**"));
        }
    }
    
    #[test]
    fn test_best_snippet_selection() {
        let result = create_sample_search_result();
        let options = HighlightOptions::default();
        
        let best_snippet = result.get_best_snippet("Result", &options);
        assert!(best_snippet.is_some(), "Should find best snippet");
        
        let snippet = best_snippet.unwrap();
        assert!(snippet.text.contains("**Result**"));
        assert!(snippet.match_count > 0);
    }
    
    #[test]
    fn test_highlighted_preview() {
        let result = create_sample_search_result();
        
        let preview = result.get_highlighted_preview("function", 100);
        assert!(preview.contains("**function**"), "Preview should contain highlights");
        assert!(preview.len() <= 120, "Should respect length limits (accounting for markup)");
    }
    
    #[test]
    fn test_plain_text_extraction() {
        let snippet = HighlightedSnippet {
            text: "This is a **highlighted** word in text".to_string(),
            start_offset: 0,
            end_offset: 39,
            match_count: 1,
        };
        
        let options = HighlightOptions::default();
        let plain = snippet.get_plain_text(&options);
        assert_eq!(plain, "This is a highlighted word in text");
        assert!(!plain.contains("**"));
    }
    
    #[test]
    fn test_case_insensitive_highlighting() {
        let result = SearchResult {
            file_path: "test.rs".to_string(),
            content: "FUNCTION and Function and function".to_string(),
            raw_content: "FUNCTION and Function and function".to_string(),
            chunk_index: 0,
            chunk_start: 0,
            chunk_end: 100,
            has_overlap: false,
            score: 1.0,
            match_field: MatchField::Both,
        };
        
        let snippets = result.get_highlighted_snippets("function", &HighlightOptions::default());
        assert!(!snippets.is_empty());
        
        let text = &snippets[0].text;
        assert!(text.contains("**FUNCTION**"));
        assert!(text.contains("**Function**"));
        assert!(text.contains("**function**"));
    }
    
    #[test]
    fn test_no_matches_fallback() {
        let result = create_sample_search_result();
        let options = HighlightOptions::default();
        
        let snippets = result.get_highlighted_snippets("nonexistent", &options);
        assert!(!snippets.is_empty(), "Should provide fallback snippet");
        
        let snippet = &snippets[0];
        assert_eq!(snippet.match_count, 0);
        assert!(!snippet.has_highlights());
    }
}
```

3. **Verify compilation and tests** (2 minutes):
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo check
cargo test highlighting_tests
```

## Success Validation
✓ SearchResult::get_highlighted_snippets() extracts and highlights matches
✓ Highlighting works with special characters like "Result<T, E>"
✓ Custom highlight options (markup, context length) are respected
✓ Multiple matches in content are handled correctly
✓ Best snippet selection works based on match count
✓ Case-insensitive highlighting functions properly
✓ Fallback snippets provided when no matches found
✓ Plain text extraction removes highlighting markup
✓ All highlighting tests pass

## Next Task Input
Task 35 expects these EXACT components ready:
- `SearchResult::get_highlighted_snippets()` method
- `HighlightedSnippet` struct with metadata
- `HighlightOptions` configuration struct
- Case-insensitive highlighting with special character support