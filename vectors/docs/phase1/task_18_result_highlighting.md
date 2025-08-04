# Task 18: Implement Result Highlighting and Snippet Generation

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 17 (Search execution)  
**Dependencies:** Tasks 01-17 must be completed

## Objective
Implement result highlighting and snippet generation to show users exactly where their search terms appear in matched content, with context and proper formatting for code and text.

## Context
Search results are much more useful when users can see exactly where and how their search terms matched. This requires intelligent snippet extraction that preserves code structure and highlights search terms while providing sufficient context.

## Task Details

### What You Need to Do

1. **Create highlighting and snippets in `src/highlighting.rs`:**

   ```rust
   use crate::indexing::utils::ChunkSearchResult;
   use tantivy::{query::Query, Searcher, schema::Field, DocAddress, Document};
   use std::collections::HashMap;
   use anyhow::Result;
   use regex::Regex;
   
   #[derive(Debug, Clone)]
   pub struct HighlightedResult {
       pub result: ChunkSearchResult,
       pub snippets: Vec<Snippet>,
       pub highlighted_content: String,
       pub match_count: usize,
   }
   
   #[derive(Debug, Clone)]
   pub struct Snippet {
       pub text: String,
       pub start_offset: usize,
       pub end_offset: usize,
       pub highlights: Vec<HighlightSpan>,
       pub context_before: String,
       pub context_after: String,
   }
   
   #[derive(Debug, Clone)]
   pub struct HighlightSpan {
       pub start: usize,
       pub end: usize,
       pub term: String,
       pub field: String,
   }
   
   #[derive(Debug, Clone)]
   pub struct HighlightConfig {
       pub snippet_length: usize,
       pub max_snippets: usize,
       pub context_size: usize,
       pub highlight_tag_open: String,
       pub highlight_tag_close: String,
       pub snippet_separator: String,
       pub preserve_code_structure: bool,
       pub merge_adjacent_highlights: bool,
   }
   
   impl Default for HighlightConfig {
       fn default() -> Self {
           Self {
               snippet_length: 200,
               max_snippets: 3,
               context_size: 50,
               highlight_tag_open: "<mark>".to_string(),
               highlight_tag_close: "</mark>".to_string(),
               snippet_separator: " ... ".to_string(),
               preserve_code_structure: true,
               merge_adjacent_highlights: true,
           }
       }
   }
   
   pub struct ResultHighlighter {
       config: HighlightConfig,
       term_extractors: HashMap<String, Regex>,
   }
   
   impl ResultHighlighter {
       /// Create a new result highlighter
       pub fn new(config: HighlightConfig) -> Self {
           let term_extractors = Self::create_term_extractors();
           
           Self {
               config,
               term_extractors,
           }
       }
       
       /// Highlight a search result with snippets
       pub fn highlight_result(
           &self,
           result: ChunkSearchResult,
           query: &dyn Query,
           searcher: &Searcher,
           doc_address: DocAddress,
       ) -> Result<HighlightedResult> {
           let doc = searcher.doc(doc_address)?;
           
           // Extract search terms from the query
           let search_terms = self.extract_search_terms(query);
           
           // Generate snippets
           let snippets = self.generate_snippets(&result.content, &search_terms)?;
           
           // Create highlighted content
           let highlighted_content = self.highlight_content(&result.content, &search_terms)?;
           
           // Count total matches
           let match_count = snippets.iter()
               .map(|s| s.highlights.len())
               .sum();
           
           Ok(HighlightedResult {
               result,
               snippets,
               highlighted_content,
               match_count,
           })
       }
       
       /// Generate contextual snippets around search terms
       fn generate_snippets(&self, content: &str, search_terms: &[String]) -> Result<Vec<Snippet>> {
           let mut snippets = Vec::new();
           let mut used_ranges = Vec::new();
           
           // Find all term matches
           let mut matches = Vec::new();
           for term in search_terms {
               matches.extend(self.find_term_matches(content, term));
           }
           
           // Sort matches by position
           matches.sort_by_key(|m| m.start);
           
           // Generate snippets around matches
           for match_info in matches {
               if snippets.len() >= self.config.max_snippets {
                   break;
               }
               
               // Check if this match overlaps with existing snippets
               let snippet_start = match_info.start.saturating_sub(self.config.context_size);
               let snippet_end = std::cmp::min(
                   match_info.end + self.config.context_size,
                   content.len()
               );
               
               // Skip if overlaps with existing snippet
               if used_ranges.iter().any(|(start, end)| {
                   snippet_start < *end && snippet_end > *start
               }) {
                   continue;
               }
               
               // Create snippet
               let snippet = self.create_snippet(
                   content,
                   snippet_start,
                   snippet_end,
                   search_terms,
               )?;
               
               snippets.push(snippet);
               used_ranges.push((snippet_start, snippet_end));
           }
           
           Ok(snippets)
       }
       
       /// Create a single snippet with highlights
       fn create_snippet(
           &self,
           content: &str,
           start: usize,
           end: usize,
           search_terms: &[String],
       ) -> Result<Snippet> {
           let snippet_text = &content[start..end];
           
           // Adjust snippet boundaries to preserve code structure if enabled
           let (adjusted_start, adjusted_end, adjusted_text) = if self.config.preserve_code_structure {
               self.adjust_snippet_boundaries(content, start, end)?
           } else {
               (start, end, snippet_text.to_string())
           };
           
           // Find highlights within this snippet
           let mut highlights = Vec::new();
           for term in search_terms {
               let term_matches = self.find_term_matches(&adjusted_text, term);
               for match_info in term_matches {
                   highlights.push(HighlightSpan {
                       start: match_info.start,
                       end: match_info.end,
                       term: term.clone(),
                       field: "content".to_string(), // Simplified
                   });
               }
           }
           
           // Sort highlights by position
           highlights.sort_by_key(|h| h.start);
           
           // Merge adjacent highlights if configured
           if self.config.merge_adjacent_highlights {
               highlights = self.merge_adjacent_highlights(highlights);
           }
           
           // Generate context
           let context_before = if adjusted_start > 0 {
               let context_start = adjusted_start.saturating_sub(self.config.context_size);
               content[context_start..adjusted_start].to_string()
           } else {
               String::new()
           };
           
           let context_after = if adjusted_end < content.len() {
               let context_end = std::cmp::min(adjusted_end + self.config.context_size, content.len());
               content[adjusted_end..context_end].to_string()
           } else {
               String::new()
           };
           
           Ok(Snippet {
               text: adjusted_text,
               start_offset: adjusted_start,
               end_offset: adjusted_end,
               highlights,
               context_before,
               context_after,
           })
       }
       
       /// Adjust snippet boundaries to preserve code structure
       fn adjust_snippet_boundaries(&self, content: &str, start: usize, end: usize) -> Result<(usize, usize, String)> {
           // Find line boundaries
           let line_start = content[..start].rfind('\n')
               .map(|pos| pos + 1)
               .unwrap_or(0);
           
           let line_end = content[end..].find('\n')
               .map(|pos| end + pos)
               .unwrap_or(content.len());
           
           // For code, try to include complete constructs
           let adjusted_start = self.find_code_boundary_start(content, line_start);
           let adjusted_end = self.find_code_boundary_end(content, line_end);
           
           let adjusted_text = content[adjusted_start..adjusted_end].to_string();
           
           Ok((adjusted_start, adjusted_end, adjusted_text))
       }
       
       /// Find appropriate start boundary for code
       fn find_code_boundary_start(&self, content: &str, mut start: usize) -> usize {
           // Look backwards for function/struct/class starts
           let code_start_patterns = [
               "fn ", "pub fn", "async fn",
               "struct ", "pub struct",
               "enum ", "pub enum",
               "impl ", "pub impl",
               "def ", "class ",
               "function ", "const ", "let ",
           ];
           
           let search_content = &content[start.saturating_sub(200)..start];
           
           for pattern in &code_start_patterns {
               if let Some(pos) = search_content.rfind(pattern) {
                   let actual_pos = start.saturating_sub(200) + pos;
                   if actual_pos < start {
                       // Find the line start for this pattern
                       if let Some(line_start) = content[..actual_pos].rfind('\n') {
                           return line_start + 1;
                       } else {
                           return actual_pos;
                       }
                   }
               }
           }
           
           start
       }
       
       /// Find appropriate end boundary for code
       fn find_code_boundary_end(&self, content: &str, mut end: usize) -> usize {
           // Look forward for complete constructs
           let remaining_content = &content[end..std::cmp::min(end + 200, content.len())];
           
           // Look for closing braces or function ends
           let mut brace_count = 0;
           let mut in_string = false;
           let mut escape_next = false;
           
           for (i, ch) in remaining_content.char_indices() {
               if escape_next {
                   escape_next = false;
                   continue;
               }
               
               match ch {
                   '"' if !in_string => in_string = true,
                   '"' if in_string => in_string = false,
                   '\\' if in_string => escape_next = true,
                   '{' if !in_string => brace_count += 1,
                   '}' if !in_string => {
                       brace_count -= 1;
                       if brace_count <= 0 {
                           // Find end of line
                           if let Some(line_end) = remaining_content[i..].find('\n') {
                               return end + i + line_end;
                           } else {
                               return end + i + 1;
                           }
                       }
                   },
                   _ => {}
               }
           }
           
           end
       }
       
       /// Find all matches for a term in content
       fn find_term_matches(&self, content: &str, term: &str) -> Vec<TermMatch> {
           let mut matches = Vec::new();
           let term_lower = term.to_lowercase();
           let content_lower = content.to_lowercase();
           
           let mut start = 0;
           while let Some(pos) = content_lower[start..].find(&term_lower) {
               let actual_pos = start + pos;
               matches.push(TermMatch {
                   start: actual_pos,
                   end: actual_pos + term.len(),
                   term: term.to_string(),
               });
               start = actual_pos + 1;
           }
           
           matches
       }
       
       /// Merge adjacent or overlapping highlights
       fn merge_adjacent_highlights(&self, mut highlights: Vec<HighlightSpan>) -> Vec<HighlightSpan> {
           if highlights.len() <= 1 {
               return highlights;
           }
           
           let mut merged = Vec::new();
           let mut current = highlights.remove(0);
           
           for next in highlights {
               // If highlights are adjacent or overlapping, merge them
               if next.start <= current.end + 5 { // 5 character tolerance
                   current.end = std::cmp::max(current.end, next.end);
                   current.term = format!("{} {}", current.term, next.term);
               } else {
                   merged.push(current);
                   current = next;
               }
           }
           
           merged.push(current);
           merged
       }
       
       /// Create highlighted content with HTML tags
       fn highlight_content(&self, content: &str, search_terms: &[String]) -> Result<String> {
           let mut highlighted = content.to_string();
           let mut offset = 0i32;
           
           // Find all matches and sort by position (reverse order for offset management)
           let mut all_matches = Vec::new();
           for term in search_terms {
               all_matches.extend(self.find_term_matches(content, term));
           }
           all_matches.sort_by_key(|m| std::cmp::Reverse(m.start));
           
           // Apply highlights from end to beginning to maintain offsets
           for match_info in all_matches {
               let start_pos = match_info.start;
               let end_pos = match_info.end;
               
               let highlighted_term = format!(
                   "{}{}{}",
                   self.config.highlight_tag_open,
                   &content[start_pos..end_pos],
                   self.config.highlight_tag_close
               );
               
               highlighted.replace_range(start_pos..end_pos, &highlighted_term);
           }
           
           Ok(highlighted)
       }
       
       /// Extract search terms from a query (simplified)
       fn extract_search_terms(&self, query: &dyn Query) -> Vec<String> {
           // This is a simplified implementation
           // In practice, you'd need to traverse the query tree
           let query_str = format!("{:?}", query);
           
           // Extract quoted terms and individual words
           let mut terms = Vec::new();
           
           // Look for quoted phrases first
           if let Some(regex) = self.term_extractors.get("quoted") {
               for captures in regex.captures_iter(&query_str) {
                   if let Some(quoted_term) = captures.get(1) {
                       terms.push(quoted_term.as_str().to_string());
                   }
               }
           }
           
           // Look for individual terms
           if let Some(regex) = self.term_extractors.get("terms") {
               for captures in regex.captures_iter(&query_str) {
                   if let Some(term) = captures.get(1) {
                       let term_str = term.as_str().to_string();
                       if !terms.contains(&term_str) && term_str.len() > 2 {
                           terms.push(term_str);
                       }
                   }
               }
           }
           
           terms
       }
       
       /// Create regex patterns for term extraction
       fn create_term_extractors() -> HashMap<String, Regex> {
           let mut extractors = HashMap::new();
           
           // Pattern for quoted phrases
           extractors.insert(
               "quoted".to_string(),
               Regex::new(r#""([^"]+)""#).unwrap()
           );
           
           // Pattern for individual terms
           extractors.insert(
               "terms".to_string(),
               Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b").unwrap()
           );
           
           extractors
       }
   }
   
   #[derive(Debug, Clone)]
   struct TermMatch {
       start: usize,
       end: usize,
       term: String,
   }
   
   /// Utility functions for working with highlighted results
   pub mod utils {
       use super::*;
       
       /// Generate plain text snippet without HTML tags
       pub fn plain_text_snippet(highlighted: &HighlightedResult) -> String {
           highlighted.snippets.iter()
               .map(|s| &s.text)
               .collect::<Vec<_>>()
               .join(" ... ")
       }
       
       /// Count unique highlighted terms
       pub fn count_unique_terms(highlighted: &HighlightedResult) -> usize {
           let mut unique_terms = std::collections::HashSet::new();
           for snippet in &highlighted.snippets {
               for highlight in &snippet.highlights {
                   unique_terms.insert(&highlight.term);
               }
           }
           unique_terms.len()
       }
       
       /// Extract all highlighted terms
       pub fn extract_highlighted_terms(highlighted: &HighlightedResult) -> Vec<String> {
           let mut terms = Vec::new();
           for snippet in &highlighted.snippets {
               for highlight in &snippet.highlights {
                   if !terms.contains(&highlight.term) {
                       terms.push(highlight.term.clone());
                   }
               }
           }
           terms
       }
   }
   ```

2. **Add module declaration to `src/lib.rs`:**

   ```rust
   pub mod highlighting;
   ```

3. **Add comprehensive tests for highlighting:**

   ```rust
   #[cfg(test)]
   mod highlighting_tests {
       use super::*;
       use crate::indexing::utils::ChunkSearchResult;
       use crate::indexing::utils::ChunkMetadata;
       
       fn create_test_result() -> ChunkSearchResult {
           ChunkSearchResult {
               metadata: ChunkMetadata {
                   id: "test_chunk".to_string(),
                   file_path: "test.rs".to_string(),
                   language: Some("rust".to_string()),
                   semantic_type: "function".to_string(),
                   chunk_index: 0,
                   total_chunks: 1,
                   start_byte: 0,
                   end_byte: 100,
               },
               content: "pub fn calculate_sum(a: i32, b: i32) -> i32 {\n    let result = a + b;\n    return result;\n}".to_string(),
               score: 1.0,
           }
       }
       
       #[test]
       fn test_highlighter_creation() {
           let config = HighlightConfig::default();
           let _highlighter = ResultHighlighter::new(config);
       }
       
       #[test]
       fn test_find_term_matches() {
           let highlighter = ResultHighlighter::new(HighlightConfig::default());
           let content = "This is a test. This test is important.";
           
           let matches = highlighter.find_term_matches(content, "test");
           
           assert_eq!(matches.len(), 2, "Should find two occurrences of 'test'");
           assert_eq!(matches[0].start, 10);
           assert_eq!(matches[0].end, 14);
           assert_eq!(matches[1].start, 21);
           assert_eq!(matches[1].end, 25);
       }
       
       #[test]
       fn test_case_insensitive_matching() {
           let highlighter = ResultHighlighter::new(HighlightConfig::default());
           let content = "Function and FUNCTION and function";
           
           let matches = highlighter.find_term_matches(content, "function");
           
           assert_eq!(matches.len(), 3, "Should find case-insensitive matches");
       }
       
       #[test]
       fn test_generate_snippets() -> Result<()> {
           let highlighter = ResultHighlighter::new(HighlightConfig::default());
           let content = "This is a long piece of text that contains multiple occurrences of the word calculate. We want to test snippet generation around the word calculate to ensure it works properly.";
           let search_terms = vec!["calculate".to_string()];
           
           let snippets = highlighter.generate_snippets(content, &search_terms)?;
           
           assert!(!snippets.is_empty(), "Should generate snippets");
           assert!(snippets[0].text.contains("calculate"), "Snippet should contain search term");
           assert!(!snippets[0].highlights.is_empty(), "Should have highlights");
           
           Ok(())
       }
       
       #[test]
       fn test_code_structure_preservation() -> Result<()> {
           let mut config = HighlightConfig::default();
           config.preserve_code_structure = true;
           let highlighter = ResultHighlighter::new(config);
           
           let content = r#"
           fn first_function() {
               println!("first");
           }
           
           fn calculate_sum(a: i32, b: i32) -> i32 {
               let result = a + b;
               return result;
           }
           
           fn third_function() {
               println!("third");
           }
           "#;
           
           let search_terms = vec!["calculate".to_string()];
           let snippets = highlighter.generate_snippets(content, &search_terms)?;
           
           assert!(!snippets.is_empty(), "Should generate snippets");
           
           let snippet = &snippets[0];
           assert!(snippet.text.contains("fn calculate_sum"), "Should include function definition");
           assert!(snippet.text.contains("{") && snippet.text.contains("}"), "Should preserve braces");
           
           Ok(())
       }
       
       #[test]
       fn test_highlight_content() -> Result<()> {
           let highlighter = ResultHighlighter::new(HighlightConfig::default());
           let content = "This function calculates the sum of two numbers.";
           let search_terms = vec!["function".to_string(), "calculates".to_string()];
           
           let highlighted = highlighter.highlight_content(content, &search_terms)?;
           
           assert!(highlighted.contains("<mark>function</mark>"), "Should highlight 'function'");
           assert!(highlighted.contains("<mark>calculates</mark>"), "Should highlight 'calculates'");
           
           Ok(())
       }
       
       #[test]
       fn test_merge_adjacent_highlights() {
           let highlighter = ResultHighlighter::new(HighlightConfig::default());
           
           let highlights = vec![
               HighlightSpan {
                   start: 0,
                   end: 5,
                   term: "hello".to_string(),
                   field: "content".to_string(),
               },
               HighlightSpan {
                   start: 6,
                   end: 11,
                   term: "world".to_string(),
                   field: "content".to_string(),
               },
           ];
           
           let merged = highlighter.merge_adjacent_highlights(highlights);
           
           assert_eq!(merged.len(), 1, "Should merge adjacent highlights");
           assert_eq!(merged[0].start, 0);
           assert_eq!(merged[0].end, 11);
           
       }
       
       #[test]
       fn test_snippet_context() -> Result<()> {
           let mut config = HighlightConfig::default();
           config.context_size = 10;
           let highlighter = ResultHighlighter::new(config);
           
           let content = "This is some text before the important word calculate that we want to find with some text after.";
           let search_terms = vec!["calculate".to_string()];
           
           let snippets = highlighter.generate_snippets(content, &search_terms)?;
           
           assert!(!snippets.is_empty(), "Should generate snippets");
           
           let snippet = &snippets[0];
           assert!(!snippet.context_before.is_empty(), "Should have context before");
           assert!(!snippet.context_after.is_empty(), "Should have context after");
           
           Ok(())
       }
       
       #[test]
       fn test_max_snippets_limit() -> Result<()> {
           let mut config = HighlightConfig::default();
           config.max_snippets = 2;
           let highlighter = ResultHighlighter::new(config);
           
           let content = "test one test two test three test four test five";
           let search_terms = vec!["test".to_string()];
           
           let snippets = highlighter.generate_snippets(content, &search_terms)?;
           
           assert!(snippets.len() <= 2, "Should respect max_snippets limit");
           
           Ok(())
       }
       
       #[test]
       fn test_plain_text_snippet_utility() {
           let mut result = create_test_result();
           
           let highlighted = HighlightedResult {
               result: result.clone(),
               snippets: vec![
                   Snippet {
                       text: "first snippet".to_string(),
                       start_offset: 0,
                       end_offset: 13,
                       highlights: Vec::new(),
                       context_before: String::new(),
                       context_after: String::new(),
                   },
                   Snippet {
                       text: "second snippet".to_string(),
                       start_offset: 20,
                       end_offset: 34,
                       highlights: Vec::new(),
                       context_before: String::new(),
                       context_after: String::new(),
                   },
               ],
               highlighted_content: "highlighted content".to_string(),
               match_count: 2,
           };
           
           let plain_text = utils::plain_text_snippet(&highlighted);
           assert_eq!(plain_text, "first snippet ... second snippet");
       }
       
       #[test]
       fn test_extract_highlighted_terms() {
           let result = create_test_result();
           
           let highlighted = HighlightedResult {
               result,
               snippets: vec![
                   Snippet {
                       text: "test snippet".to_string(),
                       start_offset: 0,
                       end_offset: 12,
                       highlights: vec![
                           HighlightSpan {
                               start: 0,
                               end: 4,
                               term: "test".to_string(),
                               field: "content".to_string(),
                           },
                           HighlightSpan {
                               start: 5,
                               end: 12,
                               term: "snippet".to_string(),
                               field: "content".to_string(),
                           },
                       ],
                       context_before: String::new(),
                       context_after: String::new(),
                   },
               ],
               highlighted_content: "highlighted".to_string(),
               match_count: 2,
           };
           
           let terms = utils::extract_highlighted_terms(&highlighted);
           assert_eq!(terms.len(), 2);
           assert!(terms.contains(&"test".to_string()));
           assert!(terms.contains(&"snippet".to_string()));
       }
       
       #[test]
       fn test_custom_highlight_tags() -> Result<()> {
           let mut config = HighlightConfig::default();
           config.highlight_tag_open = "**".to_string();
           config.highlight_tag_close = "**".to_string();
           let highlighter = ResultHighlighter::new(config);
           
           let content = "This is a test string";
           let search_terms = vec!["test".to_string()];
           
           let highlighted = highlighter.highlight_content(content, &search_terms)?;
           
           assert!(highlighted.contains("**test**"), "Should use custom tags");
           assert!(!highlighted.contains("<mark>"), "Should not use default tags");
           
           Ok(())
       }
   }
   ```

## Success Criteria
- [ ] Result highlighting compiles without errors
- [ ] All highlighting tests pass with `cargo test highlighting_tests`
- [ ] Term matching works with case-insensitive search
- [ ] Snippet generation creates appropriate context around matches
- [ ] Code structure preservation maintains function boundaries
- [ ] Content highlighting applies HTML tags correctly
- [ ] Adjacent highlights are merged when configured
- [ ] Context generation provides before and after text
- [ ] Snippet limits are respected
- [ ] Utility functions work correctly for plain text and term extraction
- [ ] Custom highlight tags can be configured
- [ ] Multiple search terms are highlighted correctly

## Common Pitfalls to Avoid
- Don't break HTML/XML when highlighting code that contains tags
- Handle Unicode characters properly in highlighting positions
- Be careful with overlapping highlights (merge or separate appropriately)
- Don't create snippets that cut words in half
- Handle very long content efficiently (don't process entire documents)
- Ensure highlight positions are accurate after content modifications
- Don't assume all search terms will be found in content
- Handle regex pattern compilation errors gracefully

## Context for Next Task
Task 19 will implement search result ranking and relevance scoring to ensure the most relevant results appear first, incorporating factors like chunk quality, recency, and term frequency.