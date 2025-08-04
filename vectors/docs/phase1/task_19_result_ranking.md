# Task 19: Implement Advanced Result Ranking and Relevance Scoring

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 18 (Result highlighting)  
**Dependencies:** Tasks 01-18 must be completed

## Objective
Implement sophisticated result ranking that considers multiple factors beyond basic text matching, including code quality, documentation, semantic types, and user context to provide the most relevant results first.

## Context
Search relevance is critical for user experience. Basic term frequency isn't enough for code search - we need to consider code quality, documentation presence, semantic types, file importance, and user intent to rank results effectively.

## Task Details

### What You Need to Do

1. **Create advanced ranking in `src/ranking.rs`:**

   ```rust
   use crate::search::{SearchResult, ChunkSearchResult};
   use crate::highlighting::HighlightedResult;
   use tantivy::Score;
   use std::collections::HashMap;
   use anyhow::Result;
   
   #[derive(Debug, Clone)]
   pub struct RankingConfig {
       pub base_score_weight: f32,
       pub documentation_boost: f32,
       pub semantic_type_boosts: HashMap<String, f32>,
       pub language_preferences: HashMap<String, f32>,
       pub file_type_preferences: HashMap<String, f32>,
       pub recency_decay_days: f32,
       pub complexity_penalty_threshold: f32,
       pub complexity_penalty_factor: f32,
       pub exact_match_boost: f32,
       pub partial_match_penalty: f32,
   }
   
   impl Default for RankingConfig {
       fn default() -> Self {
           let mut semantic_type_boosts = HashMap::new();
           semantic_type_boosts.insert("function".to_string(), 1.2);
           semantic_type_boosts.insert("struct".to_string(), 1.1);
           semantic_type_boosts.insert("enum".to_string(), 1.05);
           semantic_type_boosts.insert("impl".to_string(), 1.15);
           semantic_type_boosts.insert("class".to_string(), 1.2);
           semantic_type_boosts.insert("text".to_string(), 0.8);
           
           let mut language_preferences = HashMap::new();
           language_preferences.insert("rust".to_string(), 1.0);
           language_preferences.insert("python".to_string(), 1.0);
           language_preferences.insert("javascript".to_string(), 0.95);
           language_preferences.insert("unknown".to_string(), 0.7);
           
           let mut file_type_preferences = HashMap::new();
           file_type_preferences.insert("rs".to_string(), 1.0);
           file_type_preferences.insert("py".to_string(), 1.0);
           file_type_preferences.insert("js".to_string(), 0.95);
           file_type_preferences.insert("md".to_string(), 0.8);
           file_type_preferences.insert("txt".to_string(), 0.7);
           
           Self {
               base_score_weight: 1.0,
               documentation_boost: 1.3,
               semantic_type_boosts,
               language_preferences,
               file_type_preferences,
               recency_decay_days: 365.0,
               complexity_penalty_threshold: 20.0,
               complexity_penalty_factor: 0.1,
               exact_match_boost: 1.5,
               partial_match_penalty: 0.8,
           }
       }
   }
   
   #[derive(Debug, Clone)]
   pub struct RankingFactors {
       pub base_score: f32,
       pub documentation_factor: f32,
       pub semantic_type_factor: f32,
       pub language_factor: f32,
       pub file_type_factor: f32,
       pub recency_factor: f32,
       pub complexity_factor: f32,
       pub match_quality_factor: f32,
       pub final_score: f32,
   }
   
   pub struct ResultRanker {
       config: RankingConfig,
   }
   
   impl ResultRanker {
       /// Create a new result ranker
       pub fn new(config: RankingConfig) -> Self {
           Self { config }
       }
       
       /// Rank and sort search results
       pub fn rank_results(&self, mut results: Vec<ChunkSearchResult>) -> Vec<(ChunkSearchResult, RankingFactors)> {
           let mut ranked_results = Vec::new();
           
           for result in results {
               let factors = self.calculate_ranking_factors(&result);
               ranked_results.push((result, factors));
           }
           
           // Sort by final score (highest first)
           ranked_results.sort_by(|a, b| 
               b.1.final_score.partial_cmp(&a.1.final_score).unwrap_or(std::cmp::Ordering::Equal)
           );
           
           ranked_results
       }
       
       /// Calculate all ranking factors for a result
       fn calculate_ranking_factors(&self, result: &ChunkSearchResult) -> RankingFactors {
           let base_score = result.score * self.config.base_score_weight;
           
           let documentation_factor = self.calculate_documentation_factor(&result.metadata);
           let semantic_type_factor = self.calculate_semantic_type_factor(&result.metadata.semantic_type);
           let language_factor = self.calculate_language_factor(&result.metadata.language);
           let file_type_factor = self.calculate_file_type_factor(&result.metadata.file_path);
           let recency_factor = self.calculate_recency_factor(); // Simplified - would need file timestamps
           let complexity_factor = 1.0; // Would need complexity metadata
           let match_quality_factor = self.calculate_match_quality_factor(&result.content);
           
           let final_score = base_score 
               * documentation_factor 
               * semantic_type_factor 
               * language_factor 
               * file_type_factor 
               * recency_factor 
               * complexity_factor 
               * match_quality_factor;
           
           RankingFactors {
               base_score,
               documentation_factor,
               semantic_type_factor,
               language_factor,
               file_type_factor,
               recency_factor,
               complexity_factor,
               match_quality_factor,
               final_score,
           }
       }
       
       /// Calculate documentation boost factor
       fn calculate_documentation_factor(&self, metadata: &crate::indexing::utils::ChunkMetadata) -> f32 {
           // Simple heuristic - look for documentation patterns in file path or content
           let has_docs = metadata.file_path.contains("doc") || 
                         metadata.file_path.contains("README") ||
                         metadata.semantic_type == "text";
           
           if has_docs {
               self.config.documentation_boost
           } else {
               1.0
           }
       }
       
       /// Calculate semantic type boost factor
       fn calculate_semantic_type_factor(&self, semantic_type: &str) -> f32 {
           self.config.semantic_type_boosts.get(semantic_type)
               .copied()
               .unwrap_or(1.0)
       }
       
       /// Calculate language preference factor
       fn calculate_language_factor(&self, language: &Option<String>) -> f32 {
           match language {
               Some(lang) => self.config.language_preferences.get(lang)
                   .copied()
                   .unwrap_or(1.0),
               None => self.config.language_preferences.get("unknown")
                   .copied()
                   .unwrap_or(0.7),
           }
       }
       
       /// Calculate file type preference factor
       fn calculate_file_type_factor(&self, file_path: &str) -> f32 {
           if let Some(extension) = std::path::Path::new(file_path)
               .extension()
               .and_then(|ext| ext.to_str()) {
               self.config.file_type_preferences.get(extension)
                   .copied()
                   .unwrap_or(1.0)
           } else {
               0.9 // Slight penalty for files without extensions
           }
       }
       
       /// Calculate recency factor (simplified)
       fn calculate_recency_factor(&self) -> f32 {
           // In a real implementation, you'd use file modification times
           // For now, return neutral factor
           1.0
       }
       
       /// Calculate match quality factor based on exact vs partial matches
       fn calculate_match_quality_factor(&self, content: &str) -> f32 {
           // This is a simplified implementation
           // In practice, you'd analyze the query terms and their matches
           
           // Boost if content looks like it has exact function/type matches
           let has_exact_patterns = content.contains("fn ") || 
                                   content.contains("struct ") ||
                                   content.contains("def ") ||
                                   content.contains("class ");
           
           if has_exact_patterns {
               self.config.exact_match_boost
           } else {
               1.0
           }
       }
       
       /// Boost results based on user context or preferences
       pub fn apply_context_boost(&self, results: &mut [(ChunkSearchResult, RankingFactors)], context: &UserContext) {
           for (result, factors) in results.iter_mut() {
               let mut context_boost = 1.0;
               
               // Boost based on recently accessed files
               if context.recent_files.contains(&result.metadata.file_path) {
                   context_boost *= 1.2;
               }
               
               // Boost based on preferred languages
               if let Some(ref language) = result.metadata.language {
                   if context.preferred_languages.contains(language) {
                       context_boost *= 1.15;
                   }
               }
               
               // Boost based on project context
               if let Some(ref project_path) = context.current_project_path {
                   if result.metadata.file_path.starts_with(project_path) {
                       context_boost *= 1.1;
                   }
               }
               
               factors.final_score *= context_boost;
           }
           
           // Re-sort after context boost
           results.sort_by(|a, b| 
               b.1.final_score.partial_cmp(&a.1.final_score).unwrap_or(std::cmp::Ordering::Equal)
           );
       }
       
       /// Group similar results to avoid redundancy
       pub fn deduplicate_similar_results(&self, results: Vec<(ChunkSearchResult, RankingFactors)>) -> Vec<(ChunkSearchResult, RankingFactors)> {
           let mut deduplicated = Vec::new();
           let mut seen_patterns = std::collections::HashSet::new();
           
           for (result, factors) in results {
               let pattern = self.extract_content_pattern(&result.content);
               
               if !seen_patterns.contains(&pattern) {
                   seen_patterns.insert(pattern);
                   deduplicated.push((result, factors));
               } else {
                   // If very similar content, only keep if significantly higher score
                   if let Some(last) = deduplicated.last_mut() {
                       if factors.final_score > last.1.final_score * 1.5 {
                           *last = (result, factors);
                       }
                   }
               }
           }
           
           deduplicated
       }
       
       /// Extract a pattern representing the content structure
       fn extract_content_pattern(&self, content: &str) -> String {
           // Simplified pattern extraction
           let mut pattern = String::new();
           
           if content.contains("fn ") { pattern.push_str("fn_"); }
           if content.contains("struct ") { pattern.push_str("struct_"); }
           if content.contains("impl ") { pattern.push_str("impl_"); }
           if content.contains("def ") { pattern.push_str("def_"); }
           if content.contains("class ") { pattern.push_str("class_"); }
           
           // Add content length category
           match content.len() {
               0..=100 => pattern.push_str("short"),
               101..=500 => pattern.push_str("medium"),
               _ => pattern.push_str("long"),
           }
           
           pattern
       }
   }
   
   #[derive(Debug, Clone)]
   pub struct UserContext {
       pub recent_files: Vec<String>,
       pub preferred_languages: Vec<String>,
       pub current_project_path: Option<String>,
       pub search_history: Vec<String>,
       pub frequently_accessed_types: HashMap<String, usize>,
   }
   
   impl Default for UserContext {
       fn default() -> Self {
           Self {
               recent_files: Vec::new(),
               preferred_languages: vec!["rust".to_string(), "python".to_string()],
               current_project_path: None,
               search_history: Vec::new(),
               frequently_accessed_types: HashMap::new(),
           }
       }
   }
   ```

2. **Add comprehensive tests for ranking:**

   ```rust
   #[cfg(test)]
   mod ranking_tests {
       use super::*;
       use crate::indexing::utils::{ChunkSearchResult, ChunkMetadata};
       
       fn create_test_result(file_path: &str, semantic_type: &str, language: Option<String>, score: f32, content: &str) -> ChunkSearchResult {
           ChunkSearchResult {
               metadata: ChunkMetadata {
                   id: "test".to_string(),
                   file_path: file_path.to_string(),
                   language,
                   semantic_type: semantic_type.to_string(),
                   chunk_index: 0,
                   total_chunks: 1,
                   start_byte: 0,
                   end_byte: content.len(),
               },
               content: content.to_string(),
               score,
           }
       }
       
       #[test]
       fn test_ranker_creation() {
           let config = RankingConfig::default();
           let _ranker = ResultRanker::new(config);
       }
       
       #[test]
       fn test_semantic_type_ranking() {
           let ranker = ResultRanker::new(RankingConfig::default());
           
           let function_result = create_test_result("test.rs", "function", Some("rust".to_string()), 1.0, "fn test() {}");
           let text_result = create_test_result("readme.txt", "text", None, 1.0, "This is text");
           
           let results = vec![text_result, function_result];
           let ranked = ranker.rank_results(results);
           
           // Function should rank higher than text
           assert_eq!(ranked[0].0.metadata.semantic_type, "function");
           assert_eq!(ranked[1].0.metadata.semantic_type, "text");
       }
       
       #[test]
       fn test_language_preference_ranking() {
           let ranker = ResultRanker::new(RankingConfig::default());
           
           let rust_result = create_test_result("test.rs", "function", Some("rust".to_string()), 1.0, "fn test() {}");
           let unknown_result = create_test_result("test.txt", "function", None, 1.0, "function test");
           
           let results = vec![unknown_result, rust_result];
           let ranked = ranker.rank_results(results);
           
           // Rust should rank higher than unknown language
           assert_eq!(ranked[0].0.metadata.language, Some("rust".to_string()));
           assert_eq!(ranked[1].0.metadata.language, None);
       }
       
       #[test]
       fn test_file_type_preference_ranking() {
           let ranker = ResultRanker::new(RankingConfig::default());
           
           let rust_file = create_test_result("test.rs", "function", Some("rust".to_string()), 1.0, "fn test() {}");
           let text_file = create_test_result("test.txt", "function", None, 1.0, "function test");
           
           let results = vec![text_file, rust_file];
           let ranked = ranker.rank_results(results);
           
           // .rs file should rank higher than .txt file
           assert!(ranked[0].0.metadata.file_path.ends_with(".rs"));
           assert!(ranked[1].0.metadata.file_path.ends_with(".txt"));
       }
       
       #[test]
       fn test_documentation_boost() {
           let ranker = ResultRanker::new(RankingConfig::default());
           
           let doc_result = create_test_result("README.md", "text", None, 1.0, "Documentation text");
           let code_result = create_test_result("test.rs", "function", Some("rust".to_string()), 1.0, "fn test() {}");
           
           let results = vec![code_result, doc_result];
           let ranked = ranker.rank_results(results);
           
           // Documentation should get boost
           let doc_factors = ranked.iter().find(|(r, _)| r.metadata.file_path.contains("README")).unwrap();
           assert!(doc_factors.1.documentation_factor > 1.0);
       }
       
       #[test]
       fn test_context_boost() {
           let ranker = ResultRanker::new(RankingConfig::default());
           
           let result1 = create_test_result("recent.rs", "function", Some("rust".to_string()), 1.0, "fn test() {}");
           let result2 = create_test_result("old.rs", "function", Some("rust".to_string()), 1.0, "fn test() {}");
           
           let mut results = vec![(result2, RankingFactors {
               base_score: 1.0,
               documentation_factor: 1.0,
               semantic_type_factor: 1.0,
               language_factor: 1.0,
               file_type_factor: 1.0,
               recency_factor: 1.0,
               complexity_factor: 1.0,
               match_quality_factor: 1.0,
               final_score: 1.0,
           }), (result1, RankingFactors {
               base_score: 1.0,
               documentation_factor: 1.0,
               semantic_type_factor: 1.0,
               language_factor: 1.0,
               file_type_factor: 1.0,
               recency_factor: 1.0,
               complexity_factor: 1.0,
               match_quality_factor: 1.0,
               final_score: 1.0,
           })];
           
           let context = UserContext {
               recent_files: vec!["recent.rs".to_string()],
               preferred_languages: vec!["rust".to_string()],
               current_project_path: None,
               search_history: Vec::new(),
               frequently_accessed_types: HashMap::new(),
           };
           
           ranker.apply_context_boost(&mut results, &context);
           
           // Recent file should rank higher after context boost
           assert_eq!(results[0].0.metadata.file_path, "recent.rs");
       }
       
       #[test]
       fn test_match_quality_factor() {
           let ranker = ResultRanker::new(RankingConfig::default());
           
           let exact_match = create_test_result("test.rs", "function", Some("rust".to_string()), 1.0, "fn exact_function_name() { }");
           let factors = ranker.calculate_ranking_factors(&exact_match);
           
           // Should get exact match boost
           assert!(factors.match_quality_factor >= 1.0);
       }
       
       #[test]
       fn test_deduplication() {
           let ranker = ResultRanker::new(RankingConfig::default());
           
           let result1 = create_test_result("test1.rs", "function", Some("rust".to_string()), 1.0, "fn test() { println!(\"hello\"); }");
           let result2 = create_test_result("test2.rs", "function", Some("rust".to_string()), 0.8, "fn test() { println!(\"world\"); }");
           let result3 = create_test_result("test3.rs", "struct", Some("rust".to_string()), 1.0, "struct Test { }");
           
           let factors1 = ranker.calculate_ranking_factors(&result1);
           let factors2 = ranker.calculate_ranking_factors(&result2);
           let factors3 = ranker.calculate_ranking_factors(&result3);
           
           let results = vec![(result1, factors1), (result2, factors2), (result3, factors3)];
           let deduplicated = ranker.deduplicate_similar_results(results);
           
           // Should keep different patterns but deduplicate similar ones
           assert!(deduplicated.len() <= 3);
           
           // Should have both function and struct patterns
           let has_function = deduplicated.iter().any(|(r, _)| r.metadata.semantic_type == "function");
           let has_struct = deduplicated.iter().any(|(r, _)| r.metadata.semantic_type == "struct");
           assert!(has_function && has_struct);
       }
       
       #[test]
       fn test_ranking_factors_calculation() {
           let ranker = ResultRanker::new(RankingConfig::default());
           
           let result = create_test_result("test.rs", "function", Some("rust".to_string()), 2.0, "fn test() {}");
           let factors = ranker.calculate_ranking_factors(&result);
           
           assert_eq!(factors.base_score, 2.0);
           assert!(factors.semantic_type_factor > 1.0); // Function boost
           assert_eq!(factors.language_factor, 1.0); // Rust preference
           assert_eq!(factors.file_type_factor, 1.0); // .rs preference
           assert!(factors.final_score > 0.0);
       }
       
       #[test]
       fn test_content_pattern_extraction() {
           let ranker = ResultRanker::new(RankingConfig::default());
           
           let pattern1 = ranker.extract_content_pattern("fn test() { println!(\"hello\"); }");
           let pattern2 = ranker.extract_content_pattern("struct Test { field: String }");
           let pattern3 = ranker.extract_content_pattern("short");
           
           assert!(pattern1.contains("fn_"));
           assert!(pattern2.contains("struct_"));
           assert!(pattern3.contains("short"));
           
           assert_ne!(pattern1, pattern2);
       }
   }
   ```

## Success Criteria
- [ ] Result ranking compiles without errors
- [ ] All ranking tests pass with `cargo test ranking_tests`
- [ ] Semantic type ranking prioritizes functions over text
- [ ] Language preference ranking works correctly
- [ ] File type preference ranking prioritizes code files
- [ ] Documentation boost increases scores for docs
- [ ] Context boost applies user preferences correctly
- [ ] Match quality factor rewards exact matches
- [ ] Deduplication removes similar results appropriately
- [ ] Ranking factors are calculated accurately
- [ ] Content pattern extraction identifies code structures

## Context for Next Task
Task 20 will implement search result caching and performance optimization to ensure fast search response times for repeated queries and large result sets.