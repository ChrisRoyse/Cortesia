# Micro-Task 039a: Create Search Filter Structure

## Objective
Create SearchFilter struct to support filtering search results by various criteria.

## Prerequisites
- Task 038e completed (search optimization tests passing and committed)

## Time Estimate
8 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add search filter structure to `src/searcher.rs`:
   ```rust
   /// Search filtering criteria
   #[derive(Debug, Clone, Default)]
   pub struct SearchFilter {
       pub file_extensions: Vec<String>,
       pub exclude_extensions: Vec<String>,
       pub path_contains: Option<String>,
       pub path_excludes: Option<String>,
       pub min_content_length: Option<usize>,
       pub max_content_length: Option<usize>,
   }
   
   impl SearchFilter {
       pub fn new() -> Self {
           Self::default()
       }
       
       pub fn with_extensions(mut self, extensions: Vec<String>) -> Self {
           self.file_extensions = extensions;
           self
       }
       
       pub fn excluding_extensions(mut self, extensions: Vec<String>) -> Self {
           self.exclude_extensions = extensions;
           self
       }
       
       pub fn with_path_containing(mut self, path_part: String) -> Self {
           self.path_contains = Some(path_part);
           self
       }
       
       pub fn excluding_path_containing(mut self, path_part: String) -> Self {
           self.path_excludes = Some(path_part);
           self
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] SearchFilter struct defined with common filtering criteria
- [ ] Builder pattern methods implemented for easy filter construction
- [ ] Code compiles successfully

## Next Task
task_039b_implement_filter_logic.md

---

# Micro-Task 039b: Implement Filter Logic

## Objective
Implement the logic to apply filters to search results.

## Prerequisites
- Task 039a completed (SearchFilter structure created)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add filter application logic to SearchFilter impl in `src/searcher.rs`:
   ```rust
   impl SearchFilter {
       /// Check if a search result passes all filter criteria
       pub fn matches(&self, result: &SearchResult) -> bool {
           // Extension filtering
           if !self.file_extensions.is_empty() {
               if let Some(ref ext) = result.extension {
                   if !self.file_extensions.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
                       return false;
                   }
               } else {
                   return false; // No extension but extensions required
               }
           }
           
           // Extension exclusion
           if !self.exclude_extensions.is_empty() {
               if let Some(ref ext) = result.extension {
                   if self.exclude_extensions.iter().any(|e| e.eq_ignore_ascii_case(ext)) {
                       return false;
                   }
               }
           }
           
           // Path filtering
           if let Some(ref path_contains) = self.path_contains {
               if let Some(ref path) = result.file_path {
                   if !path.to_lowercase().contains(&path_contains.to_lowercase()) {
                       return false;
                   }
               } else {
                   return false; // No path but path required
               }
           }
           
           // Path exclusion
           if let Some(ref path_excludes) = self.path_excludes {
               if let Some(ref path) = result.file_path {
                   if path.to_lowercase().contains(&path_excludes.to_lowercase()) {
                       return false;
                   }
               }
           }
           
           // Content length filtering
           let content_len = result.content.len();
           if let Some(min_len) = self.min_content_length {
               if content_len < min_len {
                   return false;
               }
           }
           if let Some(max_len) = self.max_content_length {
               if content_len > max_len {
                   return false;
               }
           }
           
           true
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Filter logic implemented for all criteria
- [ ] Case-insensitive matching for extensions and paths
- [ ] Content length filtering working

## Next Task
task_039c_add_filtered_search_methods.md

---

# Micro-Task 039c: Add Filtered Search Methods

## Objective
Add search methods that accept and apply SearchFilter.

## Prerequisites
- Task 039b completed (filter logic implemented)

## Time Estimate
9 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add filtered search methods to DocumentSearcher impl in `src/searcher.rs`:
   ```rust
   /// Search with filtering applied to results
   pub fn search_filtered(
       &self, 
       query_text: &str, 
       limit: usize, 
       filter: &SearchFilter
   ) -> TantivyResult<Vec<SearchResult>> {
       // Get raw results with higher limit to account for filtering
       let raw_limit = (limit * 2).max(50); // Get more results to filter from
       let raw_results = self.search(query_text, raw_limit)?;
       
       // Apply filter and take only requested number
       let filtered_results: Vec<SearchResult> = raw_results
           .into_iter()
           .filter(|result| filter.matches(result))
           .take(limit)
           .collect();
       
       Ok(filtered_results)
   }
   
   /// Multi-field search with filtering
   pub fn search_multi_field_filtered(
       &self, 
       query_text: &str, 
       limit: usize, 
       filter: &SearchFilter
   ) -> TantivyResult<Vec<SearchResult>> {
       let raw_limit = (limit * 2).max(50);
       let raw_results = self.search_multi_field(query_text, raw_limit)?;
       
       let filtered_results: Vec<SearchResult> = raw_results
           .into_iter()
           .filter(|result| filter.matches(result))
           .take(limit)
           .collect();
       
       Ok(filtered_results)
   }
   
   /// Filtered search with performance metrics
   pub fn search_filtered_with_metrics(
       &self, 
       query_text: &str, 
       limit: usize, 
       filter: &SearchFilter
   ) -> TantivyResult<(Vec<SearchResult>, SearchMetrics)> {
       let start = Instant::now();
       let results = self.search_filtered(query_text, limit, filter)?;
       
       let metrics = SearchMetrics::new(
           start.elapsed().as_millis() as u64,
           results.len(),
           false // Filtered searches don't use cache for simplicity
       );
       
       Ok((results, metrics))
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Filtered search methods implemented
- [ ] Over-fetching strategy to account for filtering
- [ ] Metrics support for filtered searches

## Next Task
task_039d_add_search_filter_tests.md

---

# Micro-Task 039d: Add Search Filter Tests

## Objective
Write comprehensive tests for search filtering functionality.

## Prerequisites
- Task 039c completed (filtered search methods added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add filter tests to `src/searcher.rs`:
   ```rust
   #[cfg(test)]
   mod filter_tests {
       use super::*;
       use crate::indexer::DocumentIndexer;
       use tempfile::TempDir;
       use std::path::PathBuf;
       
       #[test]
       fn test_extension_filtering() -> TantivyResult<()> {
           let temp_dir = TempDir::new().unwrap();
           let index_path = temp_dir.path().join("test_index");
           
           // Create index with different file types
           let mut indexer = DocumentIndexer::new(&index_path)?;
           
           indexer.add_document(
               "fn main() { println!(\"Hello from Rust!\"); }",
               "main",
               &PathBuf::from("main.rs"),
               None,
           )?;
           
           indexer.add_document(
               "console.log(\"Hello from JavaScript!\");",
               "main",
               &PathBuf::from("main.js"),
               None,
           )?;
           
           indexer.commit()?;
           
           let searcher = DocumentSearcher::new(&index_path)?;
           
           // Test filtering for Rust files only
           let filter = SearchFilter::new()
               .with_extensions(vec!["rs".to_string()]);
           
           let results = searcher.search_filtered("main", 10, &filter)?;
           
           assert_eq!(results.len(), 1);
           assert_eq!(results[0].extension.as_ref().unwrap(), "rs");
           
           Ok(())
       }
       
       #[test]
       fn test_path_filtering() -> TantivyResult<()> {
           let temp_dir = TempDir::new().unwrap();
           let index_path = temp_dir.path().join("test_index");
           
           let mut indexer = DocumentIndexer::new(&index_path)?;
           
           indexer.add_document(
               "test content",
               "src_file",
               &PathBuf::from("src/main.rs"),
               None,
           )?;
           
           indexer.add_document(
               "test content",
               "test_file",
               &PathBuf::from("tests/main.rs"),
               None,
           )?;
           
           indexer.commit()?;
           
           let searcher = DocumentSearcher::new(&index_path)?;
           
           // Filter for src directory only
           let filter = SearchFilter::new()
               .with_path_containing("src".to_string());
           
           let results = searcher.search_filtered("test", 10, &filter)?;
           
           assert_eq!(results.len(), 1);
           assert!(results[0].file_path.as_ref().unwrap().contains("src"));
           
           Ok(())
       }
       
       #[test]
       fn test_combined_filtering() -> TantivyResult<()> {
           let temp_dir = TempDir::new().unwrap();
           let index_path = temp_dir.path().join("test_index");
           
           let mut indexer = DocumentIndexer::new(&index_path)?;
           
           indexer.add_document(
               "test content for filtering",
               "test1",
               &PathBuf::from("src/test.rs"),
               None,
           )?;
           
           indexer.add_document(
               "test content for filtering",
               "test2",
               &PathBuf::from("src/test.js"),
               None,
           )?;
           
           indexer.commit()?;
           
           let searcher = DocumentSearcher::new(&index_path)?;
           
           // Filter for Rust files in src directory
           let filter = SearchFilter::new()
               .with_extensions(vec!["rs".to_string()])
               .with_path_containing("src".to_string());
           
           let results = searcher.search_filtered("test", 10, &filter)?;
           
           assert_eq!(results.len(), 1);
           assert_eq!(results[0].extension.as_ref().unwrap(), "rs");
           assert!(results[0].file_path.as_ref().unwrap().contains("src"));
           
           Ok(())
       }
   }
   ```
3. Test: `cargo test`
4. Return to root: `cd ..\..`
5. Commit: `git add crates\tantivy-core && git commit -m "Implement search filtering with extension, path, and content length criteria"`

## Success Criteria
- [ ] Extension filtering tests implemented and passing
- [ ] Path filtering tests implemented and passing
- [ ] Combined filtering tests implemented and passing
- [ ] Search filtering committed to Git

## Next Task
task_040_implement_search_result_ranking.md