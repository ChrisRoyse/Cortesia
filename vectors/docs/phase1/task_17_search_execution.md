# Task 17: Implement Search Execution Engine

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 16 (Query parsing)  
**Dependencies:** Tasks 01-16 must be completed

## Objective
Implement the search execution engine that takes parsed queries and executes them against the Tantivy index, with result ranking, filtering, and performance optimization.

## Context
The search engine must efficiently execute queries against the indexed chunks, apply filters, rank results by relevance, and return structured results with metadata. Performance is critical for good user experience.

## Task Details

### What You Need to Do

1. **Create search execution in `src/search.rs`:**

   ```rust
   use crate::query::{ParsedQuery, QueryFields, QueryFilters, SearchOptions};
   use crate::indexing::utils::{extract_chunk_metadata, create_search_result, ChunkSearchResult};
   use tantivy::{
       Index, IndexReader, Searcher, TopDocs, Score, DocAddress, Document, Term,
       query::{Query, BooleanQuery, TermQuery, RangeQuery},
       collector::{TopDocs as TopDocsCollector, Count},
       schema::Field,
   };
   use std::collections::HashMap;
   use anyhow::Result;
   use std::time::{Duration, Instant};
   
   #[derive(Debug, Clone)]
   pub struct SearchResult {
       pub results: Vec<ChunkSearchResult>,
       pub total_hits: usize,
       pub search_time: Duration,
       pub query_info: QueryInfo,
       pub facets: SearchFacets,
   }
   
   #[derive(Debug, Clone)]
   pub struct QueryInfo {
       pub original_query: String,
       pub processed_query: String,
       pub filters_applied: usize,
       pub fuzzy_used: bool,
       pub fields_searched: Vec<String>,
   }
   
   #[derive(Debug, Clone)]
   pub struct SearchFacets {
       pub languages: HashMap<String, usize>,
       pub semantic_types: HashMap<String, usize>,
       pub file_extensions: HashMap<String, usize>,
   }
   
   #[derive(Debug, Clone)]
   pub struct SearchConfig {
       pub max_results: usize,
       pub timeout_ms: u64,
       pub enable_facets: bool,
       pub enable_highlighting: bool,
       pub snippet_length: usize,
       pub min_score_threshold: f32,
   }
   
   impl Default for SearchConfig {
       fn default() -> Self {
           Self {
               max_results: 50,
               timeout_ms: 5000,
               enable_facets: true,
               enable_highlighting: true,
               snippet_length: 200,
               min_score_threshold: 0.1,
           }
       }
   }
   
   pub struct SearchEngine {
       index: Index,
       reader: IndexReader,
       fields: QueryFields,
       config: SearchConfig,
   }
   
   impl SearchEngine {
       /// Create a new search engine
       pub fn new(index: Index, config: SearchConfig) -> Result<Self> {
           let reader = index.reader()?;
           let schema = index.schema();
           
           let fields = QueryFields {
               content: schema.get_field("content").expect("Missing content field"),
               file_path: schema.get_field("file_path").expect("Missing file_path field"),
               language: schema.get_field("language").expect("Missing language field"),
               semantic_type: schema.get_field("semantic_type").expect("Missing semantic_type field"),
               id: schema.get_field("id").expect("Missing id field"),
               chunk_index: schema.get_field("chunk_index").expect("Missing chunk_index field"),
               total_chunks: schema.get_field("total_chunks").expect("Missing total_chunks field"),
               start_byte: schema.get_field("start_byte").expect("Missing start_byte field"),
               end_byte: schema.get_field("end_byte").expect("Missing end_byte field"),
               overlap_prev: schema.get_field("overlap_prev").expect("Missing overlap_prev field"),
               overlap_next: schema.get_field("overlap_next").expect("Missing overlap_next field"),
           };
           
           Ok(Self {
               index,
               reader,
               fields,
               config,
           })
       }
       
       /// Execute a search query
       pub fn search(&self, parsed_query: &ParsedQuery) -> Result<SearchResult> {
           let start_time = Instant::now();
           let searcher = self.reader.searcher();
           
           // Build the final query with filters
           let final_query = self.build_filtered_query(parsed_query)?;
           
           // Execute the search
           let top_docs = searcher.search(
               &final_query,
               &TopDocsCollector::with_limit(self.config.max_results)
           )?;
           
           // Get total count
           let total_hits = searcher.search(&final_query, &Count)?;
           
           // Convert results to structured format
           let mut results = Vec::new();
           for (score, doc_address) in top_docs {
               if score >= self.config.min_score_threshold {
                   let doc = searcher.doc(doc_address)?;
                   let result = create_search_result(&doc, score, &self.fields)?;
                   results.push(result);
               }
           }
           
           // Generate facets if enabled
           let facets = if self.config.enable_facets {
               self.generate_facets(&searcher, &final_query)?
           } else {
               SearchFacets {
                   languages: HashMap::new(),
                   semantic_types: HashMap::new(),
                   file_extensions: HashMap::new(),
               }
           };
           
           // Create query info
           let query_info = QueryInfo {
               original_query: parsed_query.original_text.clone(),
               processed_query: format!("{:?}", final_query), // Simplified
               filters_applied: self.count_filters(&parsed_query.filters),
               fuzzy_used: parsed_query.options.enable_fuzzy,
               fields_searched: parsed_query.fields.iter()
                   .map(|f| self.field_name(*f))
                   .collect(),
           };
           
           let search_time = start_time.elapsed();
           
           Ok(SearchResult {
               results,
               total_hits,
               search_time,
               query_info,
               facets,
           })
       }
       
       /// Build a query that combines the main query with filters
       fn build_filtered_query(&self, parsed_query: &ParsedQuery) -> Result<Box<dyn Query>> {
           let mut boolean_query = BooleanQuery::new();
           
           // Add the main query
           boolean_query.add_must(parsed_query.query.box_clone());
           
           // Add filters
           self.add_file_type_filters(&mut boolean_query, &parsed_query.filters)?;
           self.add_language_filters(&mut boolean_query, &parsed_query.filters)?;
           self.add_semantic_type_filters(&mut boolean_query, &parsed_query.filters)?;
           self.add_documentation_filters(&mut boolean_query, &parsed_query.filters)?;
           self.add_complexity_filters(&mut boolean_query, &parsed_query.filters)?;
           
           Ok(Box::new(boolean_query))
       }
       
       /// Add file type filters to the query
       fn add_file_type_filters(&self, boolean_query: &mut BooleanQuery, filters: &QueryFilters) -> Result<()> {
           if !filters.file_types.is_empty() {
               let mut file_type_query = BooleanQuery::new();
               
               for file_type in &filters.file_types {
                   // Create a query that matches files with the specified extension
                   let pattern = format!(".{}", file_type);
                   let term = Term::from_field_text(self.fields.file_path, &pattern);
                   let term_query = TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
                   file_type_query.add_should(Box::new(term_query));
               }
               
               boolean_query.add_must(Box::new(file_type_query));
           }
           
           Ok(())
       }
       
       /// Add language filters to the query
       fn add_language_filters(&self, boolean_query: &mut BooleanQuery, filters: &QueryFilters) -> Result<()> {
           if !filters.languages.is_empty() {
               let mut language_query = BooleanQuery::new();
               
               for language in &filters.languages {
                   let term = Term::from_field_text(self.fields.language, language);
                   let term_query = TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
                   language_query.add_should(Box::new(term_query));
               }
               
               boolean_query.add_must(Box::new(language_query));
           }
           
           Ok(())
       }
       
       /// Add semantic type filters to the query
       fn add_semantic_type_filters(&self, boolean_query: &mut BooleanQuery, filters: &QueryFilters) -> Result<()> {
           if !filters.semantic_types.is_empty() {
               let mut semantic_query = BooleanQuery::new();
               
               for semantic_type in &filters.semantic_types {
                   let term = Term::from_field_text(self.fields.semantic_type, semantic_type);
                   let term_query = TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
                   semantic_query.add_should(Box::new(term_query));
               }
               
               boolean_query.add_must(Box::new(semantic_query));
           }
           
           Ok(())
       }
       
       /// Add documentation filters to the query
       fn add_documentation_filters(&self, boolean_query: &mut BooleanQuery, filters: &QueryFilters) -> Result<()> {
           if let Some(has_documentation) = filters.has_documentation {
               if has_documentation {
                   // This is a simplified approach - in practice, you'd need a boolean field
                   // For now, we'll boost chunks that are likely to have documentation
                   let doc_indicators = ["/**", "///", "\"\"\"", "'''"];
                   let mut doc_query = BooleanQuery::new();
                   
                   for indicator in &doc_indicators {
                       let term = Term::from_field_text(self.fields.content, indicator);
                       let term_query = TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
                       doc_query.add_should(Box::new(term_query));
                   }
                   
                   boolean_query.add_should(Box::new(doc_query));
               }
           }
           
           Ok(())
       }
       
       /// Add complexity filters to the query
       fn add_complexity_filters(&self, boolean_query: &mut BooleanQuery, filters: &QueryFilters) -> Result<()> {
           if let Some((min_complexity, max_complexity)) = filters.complexity_range {
               // This would require a numeric field for complexity
               // For now, we'll skip this filter as it requires schema changes
               // In a real implementation, you'd add a complexity field to the schema
               println!("Complexity filter not implemented yet: {}-{}", min_complexity, max_complexity);
           }
           
           Ok(())
       }
       
       /// Generate faceted search results
       fn generate_facets(&self, searcher: &Searcher, query: &dyn Query) -> Result<SearchFacets> {
           // This is a simplified facet implementation
           // In practice, you'd use Tantivy's facet collectors
           
           let top_docs = searcher.search(query, &TopDocsCollector::with_limit(1000))?;
           
           let mut languages = HashMap::new();
           let mut semantic_types = HashMap::new();
           let mut file_extensions = HashMap::new();
           
           for (_score, doc_address) in top_docs {
               let doc = searcher.doc(doc_address)?;
               
               // Count languages
               if let Some(language_value) = doc.get_first(self.fields.language) {
                   if let Some(language) = language_value.as_text() {
                       *languages.entry(language.to_string()).or_insert(0) += 1;
                   }
               }
               
               // Count semantic types
               if let Some(semantic_value) = doc.get_first(self.fields.semantic_type) {
                   if let Some(semantic_type) = semantic_value.as_text() {
                       *semantic_types.entry(semantic_type.to_string()).or_insert(0) += 1;
                   }
               }
               
               // Count file extensions
               if let Some(file_path_value) = doc.get_first(self.fields.file_path) {
                   if let Some(file_path) = file_path_value.as_text() {
                       if let Some(extension) = std::path::Path::new(file_path)
                           .extension()
                           .and_then(|ext| ext.to_str()) {
                           *file_extensions.entry(extension.to_string()).or_insert(0) += 1;
                       }
                   }
               }
           }
           
           Ok(SearchFacets {
               languages,
               semantic_types,
               file_extensions,
           })
       }
       
       /// Count the number of filters applied
       fn count_filters(&self, filters: &QueryFilters) -> usize {
           let mut count = 0;
           
           if !filters.file_types.is_empty() { count += 1; }
           if !filters.languages.is_empty() { count += 1; }
           if !filters.semantic_types.is_empty() { count += 1; }
           if filters.has_documentation.is_some() { count += 1; }
           if filters.complexity_range.is_some() { count += 1; }
           if filters.date_range.is_some() { count += 1; }
           
           count
       }
       
       /// Get field name for a field
       fn field_name(&self, field: Field) -> String {
           if field == self.fields.content { "content".to_string() }
           else if field == self.fields.file_path { "file_path".to_string() }
           else if field == self.fields.language { "language".to_string() }
           else if field == self.fields.semantic_type { "semantic_type".to_string() }
           else if field == self.fields.id { "id".to_string() }
           else { "unknown".to_string() }
       }
       
       /// Suggest similar queries when search yields no results
       pub fn suggest_similar_queries(&self, parsed_query: &ParsedQuery) -> Result<Vec<String>> {
           let mut suggestions = Vec::new();
           
           // If no results, try some variations
           let original = &parsed_query.original_text;
           
           // Suggest without filters
           if !parsed_query.filters.file_types.is_empty() || 
              !parsed_query.filters.languages.is_empty() ||
              !parsed_query.filters.semantic_types.is_empty() {
               // Extract just the search terms
               let terms: Vec<&str> = original.split_whitespace()
                   .filter(|word| !word.contains(':'))
                   .collect();
               if !terms.is_empty() {
                   suggestions.push(terms.join(" "));
               }
           }
           
           // Suggest with fuzzy search
           if !parsed_query.options.enable_fuzzy {
               suggestions.push(format!("{} fuzzy:2", original));
           }
           
           // Suggest broader semantic types
           if parsed_query.filters.semantic_types.contains(&"function".to_string()) {
               suggestions.push(original.replace("type:function", "type:code"));
           }
           
           // Suggest different file types
           if parsed_query.filters.file_types.contains(&"rs".to_string()) {
               suggestions.push(original.replace("filetype:rs", "filetype:py"));
           }
           
           Ok(suggestions)
       }
       
       /// Reload the index reader (for incremental updates)
       pub fn reload(&mut self) -> Result<()> {
           self.reader.reload()?;
           Ok(())
       }
       
       /// Get search engine statistics
       pub fn get_stats(&self) -> Result<SearchEngineStats> {
           let searcher = self.reader.searcher();
           let num_docs = searcher.num_docs();
           let num_segments = searcher.segment_readers().len();
           
           Ok(SearchEngineStats {
               total_documents: num_docs as usize,
               segment_count: num_segments,
               index_size_estimate: 0, // Would need more complex calculation
           })
       }
   }
   
   #[derive(Debug, Clone)]
   pub struct SearchEngineStats {
       pub total_documents: usize,
       pub segment_count: usize,
       pub index_size_estimate: u64,
   }
   
   /// Trait extension for Query to support cloning
   trait QueryClone {
       fn box_clone(&self) -> Box<dyn Query>;
   }
   
   impl<T: Query + Clone + 'static> QueryClone for T {
       fn box_clone(&self) -> Box<dyn Query> {
           Box::new(self.clone())
       }
   }
   ```

2. **Add module declaration to `src/lib.rs`:**

   ```rust
   pub mod search;
   ```

3. **Add comprehensive tests for search execution:**

   ```rust
   #[cfg(test)]
   mod search_tests {
       use super::*;
       use crate::schema::create_schema;
       use crate::indexing::ChunkIndexer;
       use crate::chunker::TextChunk;
       use crate::query::QueryParser;
       use tempfile::TempDir;
       use tantivy::Index;
       
       fn create_test_search_setup() -> Result<(TempDir, SearchEngine, QueryParser)> {
           let temp_dir = TempDir::new()?;
           let schema = create_schema();
           let index = Index::create_in_dir(temp_dir.path(), schema)?;
           
           // Index some test data
           let mut indexer = ChunkIndexer::new(index.clone())?;
           
           let test_chunks = vec![
               TextChunk {
                   id: "test_1".to_string(),
                   content: "pub fn calculate_sum(a: i32, b: i32) -> i32 { a + b }".to_string(),
                   start_byte: 0,
                   end_byte: 50,
                   chunk_index: 0,
                   total_chunks: 1,
                   language: Some("rust".to_string()),
                   file_path: "math.rs".to_string(),
                   overlap_with_previous: 0,
                   overlap_with_next: 0,
                   semantic_type: "function".to_string(),
               },
               TextChunk {
                   id: "test_2".to_string(),
                   content: "struct Config { name: String, port: u16 }".to_string(),
                   start_byte: 0,
                   end_byte: 40,
                   chunk_index: 0,
                   total_chunks: 1,
                   language: Some("rust".to_string()),
                   file_path: "config.rs".to_string(),
                   overlap_with_previous: 0,
                   overlap_with_next: 0,
                   semantic_type: "struct".to_string(),
               },
               TextChunk {
                   id: "test_3".to_string(),
                   content: "def process_data(items): return [item.upper() for item in items]".to_string(),
                   start_byte: 0,
                   end_byte: 65,
                   chunk_index: 0,
                   total_chunks: 1,
                   language: Some("python".to_string()),
                   file_path: "processor.py".to_string(),
                   overlap_with_previous: 0,
                   overlap_with_next: 0,
                   semantic_type: "function".to_string(),
               },
           ];
           
           indexer.index_chunks(test_chunks)?;
           indexer.commit()?;
           
           let config = SearchConfig::default();
           let search_engine = SearchEngine::new(index.clone(), config)?;
           let query_parser = QueryParser::new(&index)?;
           
           Ok((temp_dir, search_engine, query_parser))
       }
       
       #[test]
       fn test_search_engine_creation() -> Result<()> {
           let (_temp_dir, _search_engine, _parser) = create_test_search_setup()?;
           Ok(())
       }
       
       #[test]
       fn test_simple_text_search() -> Result<()> {
           let (_temp_dir, search_engine, parser) = create_test_search_setup()?;
           
           let parsed_query = parser.parse("calculate")?;
           let results = search_engine.search(&parsed_query)?;
           
           assert!(results.total_hits > 0, "Should find results for 'calculate'");
           assert!(!results.results.is_empty(), "Should return search results");
           assert!(results.search_time > Duration::new(0, 0), "Should track search time");
           
           Ok(())
       }
       
       #[test]
       fn test_language_filter_search() -> Result<()> {
           let (_temp_dir, search_engine, parser) = create_test_search_setup()?;
           
           let parsed_query = parser.parse("function lang:rust")?;
           let results = search_engine.search(&parsed_query)?;
           
           assert!(results.total_hits > 0, "Should find Rust functions");
           
           // Verify all results are Rust files
           for result in &results.results {
               assert_eq!(result.metadata.language, Some("rust".to_string()));
           }
           
           Ok(())
       }
       
       #[test]
       fn test_semantic_type_filter() -> Result<()> {
           let (_temp_dir, search_engine, parser) = create_test_search_setup()?;
           
           let parsed_query = parser.parse("data type:function")?;
           let results = search_engine.search(&parsed_query)?;
           
           assert!(results.total_hits > 0, "Should find functions");
           
           // Verify all results are functions
           for result in &results.results {
               assert_eq!(result.metadata.semantic_type, "function");
           }
           
           Ok(())
       }
       
       #[test]
       fn test_file_type_filter() -> Result<()> {
           let (_temp_dir, search_engine, parser) = create_test_search_setup()?;
           
           let parsed_query = parser.parse("struct filetype:rs")?;
           let results = search_engine.search(&parsed_query)?;
           
           assert!(results.total_hits > 0, "Should find struct in Rust files");
           
           // Verify all results are from .rs files
           for result in &results.results {
               assert!(result.metadata.file_path.ends_with(".rs"));
           }
           
           Ok(())
       }
       
       #[test]
       fn test_complex_boolean_search() -> Result<()> {
           let (_temp_dir, search_engine, parser) = create_test_search_setup()?;
           
           let parsed_query = parser.parse("function AND (calculate OR process)")?;
           let results = search_engine.search(&parsed_query)?;
           
           assert!(results.total_hits > 0, "Should find results for boolean query");
           
           Ok(())
       }
       
       #[test]
       fn test_faceted_search() -> Result<()> {
           let (_temp_dir, search_engine, parser) = create_test_search_setup()?;
           
           let parsed_query = parser.parse("function")?;
           let results = search_engine.search(&parsed_query)?;
           
           // Check that facets are generated
           assert!(!results.facets.languages.is_empty(), "Should generate language facets");
           assert!(results.facets.languages.contains_key("rust"));
           assert!(results.facets.languages.contains_key("python"));
           
           Ok(())
       }
       
       #[test]
       fn test_no_results_query() -> Result<()> {
           let (_temp_dir, search_engine, parser) = create_test_search_setup()?;
           
           let parsed_query = parser.parse("nonexistent_function_name_12345")?;
           let results = search_engine.search(&parsed_query)?;
           
           assert_eq!(results.total_hits, 0, "Should find no results");
           assert!(results.results.is_empty(), "Should return empty results");
           
           Ok(())
       }
       
       #[test]
       fn test_search_suggestions() -> Result<()> {
           let (_temp_dir, search_engine, parser) = create_test_search_setup()?;
           
           let parsed_query = parser.parse("nonexistent filetype:rs lang:rust")?;
           let suggestions = search_engine.suggest_similar_queries(&parsed_query)?;
           
           assert!(!suggestions.is_empty(), "Should provide suggestions");
           
           // Should suggest query without filters
           assert!(suggestions.iter().any(|s| s == "nonexistent"), 
                  "Should suggest removing filters");
           
           Ok(())
       }
       
       #[test]
       fn test_min_score_threshold() -> Result<()> {
           let (_temp_dir, mut search_engine, parser) = create_test_search_setup()?;
           
           // Set high score threshold
           search_engine.config.min_score_threshold = 10.0; // Very high threshold
           
           let parsed_query = parser.parse("calculate")?;
           let results = search_engine.search(&parsed_query)?;
           
           // With high threshold, might get fewer results
           // This test verifies the threshold is applied
           for result in &results.results {
               assert!(result.score >= search_engine.config.min_score_threshold);
           }
           
           Ok(())
       }
       
       #[test]
       fn test_search_result_limit() -> Result<()> {
           let (_temp_dir, mut search_engine, parser) = create_test_search_setup()?;
           
           // Set low result limit
           search_engine.config.max_results = 1;
           
           let parsed_query = parser.parse("function")?; // Should match multiple chunks
           let results = search_engine.search(&parsed_query)?;
           
           assert!(results.results.len() <= 1, "Should respect max_results limit");
           
           Ok(())
       }
       
       #[test]
       fn test_search_engine_stats() -> Result<()> {
           let (_temp_dir, search_engine, _parser) = create_test_search_setup()?;
           
           let stats = search_engine.get_stats()?;
           
           assert!(stats.total_documents > 0, "Should have indexed documents");
           assert!(stats.segment_count > 0, "Should have at least one segment");
           
           Ok(())
       }
       
       #[test]
       fn test_query_info_tracking() -> Result<()> {
           let (_temp_dir, search_engine, parser) = create_test_search_setup()?;
           
           let parsed_query = parser.parse("calculate filetype:rs fuzzy:2")?;
           let results = search_engine.search(&parsed_query)?;
           
           assert_eq!(results.query_info.original_query, "calculate filetype:rs fuzzy:2");
           assert!(results.query_info.filters_applied > 0);
           assert!(results.query_info.fuzzy_used);
           assert!(!results.query_info.fields_searched.is_empty());
           
           Ok(())
       }
       
       #[test]
       fn test_empty_query_search() -> Result<()> {
           let (_temp_dir, search_engine, parser) = create_test_search_setup()?;
           
           let parsed_query = parser.parse("")?;
           let results = search_engine.search(&parsed_query)?;
           
           // Empty query should return all documents (match-all)
           assert!(results.total_hits > 0, "Empty query should match documents");
           
           Ok(())
       }
   }
   ```

## Success Criteria
- [ ] Search execution compiles without errors
- [ ] All search tests pass with `cargo test search_tests`
- [ ] Simple text searches return relevant results
- [ ] Language filters correctly filter results by programming language
- [ ] Semantic type filters work for functions, structs, etc.
- [ ] File type filters match appropriate extensions
- [ ] Boolean queries (AND, OR, NOT) execute correctly
- [ ] Faceted search generates language and semantic type counts
- [ ] No-results queries are handled gracefully
- [ ] Search suggestions are provided for failed queries
- [ ] Score thresholds filter low-relevance results
- [ ] Result limits are respected
- [ ] Search statistics are tracked accurately
- [ ] Query information is properly recorded

## Common Pitfalls to Avoid
- Don't assume queries will always return results
- Handle Tantivy errors gracefully (index corruption, etc.)
- Be careful with memory usage when processing large result sets
- Don't let search timeouts crash the application
- Handle Unicode properly in search results
- Ensure facet generation doesn't significantly slow down searches
- Be careful with score calculations and thresholds
- Don't forget to handle edge cases (empty index, malformed queries)

## Context for Next Task
Task 18 will implement result highlighting and snippet generation to show users exactly where their search terms appear in the matched content.