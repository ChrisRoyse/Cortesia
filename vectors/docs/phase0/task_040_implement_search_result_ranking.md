# Micro-Task 040a: Create Result Ranking System

## Objective
Implement a customizable result ranking system to improve search result ordering.

## Prerequisites
- Task 039d completed (search filtering tests passing and committed)

## Time Estimate
9 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add ranking structures to `src/searcher.rs`:
   ```rust
   /// Result ranking configuration
   #[derive(Debug, Clone)]
   pub struct RankingConfig {
       pub title_boost: f32,
       pub extension_boost: HashMap<String, f32>,
       pub recency_weight: f32,
       pub length_preference: LengthPreference,
   }
   
   #[derive(Debug, Clone)]
   pub enum LengthPreference {
       Shorter,  // Prefer shorter content
       Longer,   // Prefer longer content  
       None,     // No length preference
   }
   
   impl Default for RankingConfig {
       fn default() -> Self {
           let mut extension_boost = HashMap::new();
           extension_boost.insert("rs".to_string(), 1.2);
           extension_boost.insert("py".to_string(), 1.1);
           extension_boost.insert("js".to_string(), 1.1);
           extension_boost.insert("md".to_string(), 0.9);
           extension_boost.insert("txt".to_string(), 0.8);
           
           Self {
               title_boost: 1.5,
               extension_boost,
               recency_weight: 0.1,
               length_preference: LengthPreference::None,
           }
       }
   }
   ```
3. Add HashMap import:
   ```rust
   use std::collections::HashMap;
   ```
4. Test: `cargo check`
5. Return to root: `cd ..\..`

## Success Criteria
- [ ] RankingConfig struct defined with boost factors
- [ ] LengthPreference enum implemented
- [ ] Default configuration with sensible values

## Next Task
task_040b_implement_result_scoring_logic.md

---

# Micro-Task 040b: Implement Result Scoring Logic

## Objective
Implement logic to calculate custom scores for search results based on ranking configuration.

## Prerequisites
- Task 040a completed (result ranking system created)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add scoring logic to `src/searcher.rs`:
   ```rust
   impl SearchResult {
       /// Calculate custom ranking score based on configuration
       pub fn calculate_custom_score(&self, config: &RankingConfig, query_text: &str) -> f32 {
           let mut score = self.score;
           
           // Title boost if query appears in title
           if self.title.to_lowercase().contains(&query_text.to_lowercase()) {
               score *= config.title_boost;
           }
           
           // Extension boost
           if let Some(ref ext) = self.extension {
               if let Some(boost) = config.extension_boost.get(ext) {
                   score *= boost;
               }
           }
           
           // Length preference adjustment
           let content_len = self.content.len() as f32;
           match config.length_preference {
               LengthPreference::Shorter => {
                   // Prefer shorter content (inverse relationship)
                   if content_len > 0.0 {
                       score *= (1000.0 / content_len).min(2.0); // Cap at 2x boost
                   }
               }
               LengthPreference::Longer => {
                   // Prefer longer content (logarithmic relationship)
                   score *= (content_len / 100.0).ln().max(1.0).min(2.0); // Cap at 2x boost
               }
               LengthPreference::None => {
                   // No adjustment
               }
           }
           
           score
       }
       
       /// Create a copy with updated custom score
       pub fn with_custom_score(&self, custom_score: f32) -> Self {
           let mut result = self.clone();
           result.score = custom_score;
           result
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Custom scoring logic implemented with all factors
- [ ] Title matching boost working
- [ ] Extension and length preferences applied
- [ ] Score capping prevents extreme values

## Next Task
task_040c_add_ranked_search_methods.md

---

# Micro-Task 040c: Add Ranked Search Methods

## Objective
Add search methods that apply custom ranking to results.

## Prerequisites
- Task 040b completed (result scoring logic implemented)

## Time Estimate
9 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add ranked search methods to DocumentSearcher impl in `src/searcher.rs`:
   ```rust
   /// Search with custom ranking applied
   pub fn search_ranked(
       &self, 
       query_text: &str, 
       limit: usize, 
       ranking_config: &RankingConfig
   ) -> TantivyResult<Vec<SearchResult>> {
       // Get more results to re-rank
       let raw_limit = (limit * 2).max(20);
       let mut results = self.search(query_text, raw_limit)?;
       
       // Apply custom scoring and re-sort
       for result in &mut results {
           let custom_score = result.calculate_custom_score(ranking_config, query_text);
           *result = result.with_custom_score(custom_score);
       }
       
       // Sort by custom score (descending) and take requested limit
       results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
       results.truncate(limit);
       
       Ok(results)
   }
   
   /// Multi-field search with custom ranking
   pub fn search_multi_field_ranked(
       &self, 
       query_text: &str, 
       limit: usize, 
       ranking_config: &RankingConfig
   ) -> TantivyResult<Vec<SearchResult>> {
       let raw_limit = (limit * 2).max(20);
       let mut results = self.search_multi_field(query_text, raw_limit)?;
       
       for result in &mut results {
           let custom_score = result.calculate_custom_score(ranking_config, query_text);
           *result = result.with_custom_score(custom_score);
       }
       
       results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
       results.truncate(limit);
       
       Ok(results)
   }
   
   /// Combined filtered and ranked search
   pub fn search_filtered_ranked(
       &self, 
       query_text: &str, 
       limit: usize, 
       filter: &SearchFilter,
       ranking_config: &RankingConfig
   ) -> TantivyResult<Vec<SearchResult>> {
       // Get more results to filter and rank
       let raw_limit = (limit * 3).max(30);
       let mut results = self.search(query_text, raw_limit)?;
       
       // Apply filtering first
       results = results.into_iter()
           .filter(|result| filter.matches(result))
           .collect();
       
       // Apply custom ranking
       for result in &mut results {
           let custom_score = result.calculate_custom_score(ranking_config, query_text);
           *result = result.with_custom_score(custom_score);
       }
       
       // Sort and truncate
       results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
       results.truncate(limit);
       
       Ok(results)
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Ranked search methods implemented
- [ ] Combined filtering and ranking working
- [ ] Proper sorting by custom scores

## Next Task
task_040d_add_ranking_tests_and_commit.md

---

# Micro-Task 040d: Add Ranking Tests and Commit

## Objective
Write tests for the ranking system and commit the complete searcher implementation.

## Prerequisites
- Task 040c completed (ranked search methods added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add ranking tests to `src/searcher.rs`:
   ```rust
   #[cfg(test)]
   mod ranking_tests {
       use super::*;
       use crate::indexer::DocumentIndexer;
       use tempfile::TempDir;
       use std::path::PathBuf;
       
       #[test]
       fn test_title_boost_ranking() -> TantivyResult<()> {
           let temp_dir = TempDir::new().unwrap();
           let index_path = temp_dir.path().join("test_index");
           
           let mut indexer = DocumentIndexer::new(&index_path)?;
           
           // Document with query in title
           indexer.add_document(
               "Some generic content here",
               "search query test", // Title contains "search"
               &PathBuf::from("doc1.rs"),
               None,
           )?;
           
           // Document with query in content only
           indexer.add_document(
               "This content contains the search term we want",
               "generic title",
               &PathBuf::from("doc2.rs"),
               None,
           )?;
           
           indexer.commit()?;
           
           let searcher = DocumentSearcher::new(&index_path)?;
           let ranking_config = RankingConfig::default();
           
           let results = searcher.search_ranked("search", 10, &ranking_config)?;
           
           // Document with title match should rank higher
           assert_eq!(results.len(), 2);
           assert!(results[0].title.contains("search"));
           
           Ok(())
       }
       
       #[test]
       fn test_extension_boost_ranking() -> TantivyResult<()> {
           let temp_dir = TempDir::new().unwrap();
           let index_path = temp_dir.path().join("test_index");
           
           let mut indexer = DocumentIndexer::new(&index_path)?;
           
           // Rust file (should get boost)
           indexer.add_document(
               "fn test() { }",
               "rust_file",
               &PathBuf::from("test.rs"),
               None,
           )?;
           
           // Text file (lower boost)
           indexer.add_document(
               "test function here",
               "text_file",
               &PathBuf::from("test.txt"),
               None,
           )?;
           
           indexer.commit()?;
           
           let searcher = DocumentSearcher::new(&index_path)?;
           let ranking_config = RankingConfig::default();
           
           let results = searcher.search_ranked("test", 10, &ranking_config)?;
           
           // Rust file should rank higher due to extension boost
           assert_eq!(results.len(), 2);
           assert_eq!(results[0].extension.as_ref().unwrap(), "rs");
           
           Ok(())
       }
       
       #[test]
       fn test_combined_filter_and_ranking() -> TantivyResult<()> {
           let temp_dir = TempDir::new().unwrap();
           let index_path = temp_dir.path().join("test_index");
           
           let mut indexer = DocumentIndexer::new(&index_path)?;
           
           indexer.add_document(
               "test content",
               "test file",
               &PathBuf::from("src/test.rs"),
               None,
           )?;
           
           indexer.add_document(
               "test content",
               "test file", 
               &PathBuf::from("other/test.py"),
               None,
           )?;
           
           indexer.commit()?;
           
           let searcher = DocumentSearcher::new(&index_path)?;
           let filter = SearchFilter::new().with_path_containing("src".to_string());
           let ranking_config = RankingConfig::default();
           
           let results = searcher.search_filtered_ranked("test", 10, &filter, &ranking_config)?;
           
           assert_eq!(results.len(), 1);
           assert!(results[0].file_path.as_ref().unwrap().contains("src"));
           
           Ok(())
       }
   }
   ```
3. Test: `cargo test`
4. Return to root: `cd ..\..`
5. Commit: `git add crates\tantivy-core && git commit -m "Complete searcher implementation with advanced ranking and filtering"`

## Success Criteria
- [ ] Title boost ranking tests implemented and passing
- [ ] Extension boost ranking tests implemented and passing
- [ ] Combined filter and ranking tests implemented and passing
- [ ] Complete searcher implementation committed to Git

## Next Task
task_041_create_vector_storage_mock.md