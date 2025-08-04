# Task 10: Implement SearchEngine with Multi-Field Querying

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 09 (DocumentIndexer)  
**Dependencies:** Tasks 01-09 must be completed

## Objective
Create the SearchEngine that provides fast, accurate search across indexed content with full special character support and intelligent result ranking.

## Context
You now have files indexed with chunks containing both processed content and raw content. The SearchEngine must query both fields to ensure special characters like `[workspace]`, `Result<T,E>`, `#[derive]` are found correctly. It must also provide meaningful result ranking and metadata.

## Task Details

### What You Need to Do

1. **Implement `src/search.rs` with the SearchEngine:**

   ```rust
   //! Search engine with multi-field querying and special character support
   
   use crate::schema::create_tantivy_index;
   use tantivy::{Index, ReloadPolicy, query::QueryParser, collector::TopDocs, schema::Schema, Document as TantivyDoc};
   use std::path::Path;
   use anyhow::Result;
   
   /// Search engine for querying indexed code content
   pub struct SearchEngine {
       index: Index,
       schema: Schema,
       query_parser: QueryParser,
   }
   
   impl SearchEngine {
       /// Create new SearchEngine from existing index
       pub fn new(index_path: &Path) -> Result<Self> {
           let index = create_tantivy_index(index_path)?;
           let schema = index.schema();
           
           // Create query parser for both content fields
           let query_parser = QueryParser::for_index(
               &index,
               vec![
                   schema.get_field("content")?,
                   schema.get_field("raw_content")?,
               ],
           );
           
           Ok(Self {
               index,
               schema,
               query_parser,
           })
       }
       
       /// Search for content with automatic query processing
       pub fn search(&self, query_str: &str, limit: usize) -> Result<Vec<SearchResult>> {
           let reader = self.index
               .reader_builder()
               .reload_policy(ReloadPolicy::OnCommit)
               .try_into()?;
           let searcher = reader.searcher();
           
           let query = self.query_parser.parse_query(query_str)
               .map_err(|e| anyhow::anyhow!("Failed to parse query '{}': {}", query_str, e))?;
           
           let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;
           
           let mut results = Vec::new();
           for (score, doc_address) in top_docs {
               let doc = searcher.doc(doc_address)?;
               let result = self.create_search_result(doc, score)?;
               results.push(result);
           }
           
           Ok(results)
       }
       
       /// Search with advanced options
       pub fn search_with_options(&self, query_str: &str, options: SearchOptions) -> Result<SearchResponse> {
           let results = self.search(query_str, options.limit)?;
           
           // Apply additional filtering if needed
           let filtered_results = if let Some(file_filter) = &options.file_filter {
               results.into_iter()
                   .filter(|r| r.file_path.contains(file_filter))
                   .collect()
           } else {
               results
           };
           
           Ok(SearchResponse {
               results: filtered_results,
               query: query_str.to_string(),
               total_found: results.len(),
               options,
           })
       }
       
       /// Create SearchResult from Tantivy document
       fn create_search_result(&self, doc: TantivyDoc, score: f32) -> Result<SearchResult> {
           Ok(SearchResult {
               file_path: self.get_field_string(&doc, "file_path")?,
               content: self.get_field_string(&doc, "content")?,
               chunk_index: self.get_field_u64(&doc, "chunk_index")?,
               chunk_start: self.get_field_u64(&doc, "chunk_start")?,
               chunk_end: self.get_field_u64(&doc, "chunk_end")?,
               has_overlap: self.get_field_bool(&doc, "has_overlap")?,
               score,
           })
       }
       
       /// Extract string field from document
       fn get_field_string(&self, doc: &TantivyDoc, field_name: &str) -> Result<String> {
           let field = self.schema.get_field(field_name)?;
           doc.get_first(field)
               .and_then(|v| v.as_text())
               .map(|s| s.to_string())
               .ok_or_else(|| anyhow::anyhow!("Field '{}' not found or not text", field_name))
       }
       
       /// Extract u64 field from document
       fn get_field_u64(&self, doc: &TantivyDoc, field_name: &str) -> Result<u64> {
           let field = self.schema.get_field(field_name)?;
           doc.get_first(field)
               .and_then(|v| v.as_u64())
               .ok_or_else(|| anyhow::anyhow!("Field '{}' not found or not u64", field_name))
       }
       
       /// Extract bool field from document
       fn get_field_bool(&self, doc: &TantivyDoc, field_name: &str) -> Result<bool> {
           let field = self.schema.get_field(field_name)?;
           doc.get_first(field)
               .and_then(|v| v.as_bool())
               .ok_or_else(|| anyhow::anyhow!("Field '{}' not found or not bool", field_name))
       }
       
       /// Get search statistics
       pub fn get_stats(&self) -> Result<SearchStats> {
           let reader = self.index.reader()?;
           let searcher = reader.searcher();
           
           Ok(SearchStats {
               total_documents: searcher.num_docs() as usize,
               index_size_mb: 1, // Placeholder - would calculate actual size
           })
       }
   }
   
   /// Individual search result
   #[derive(Debug, Clone)]
   pub struct SearchResult {
       pub file_path: String,
       pub content: String,
       pub chunk_index: u64,
       pub chunk_start: u64,
       pub chunk_end: u64,
       pub has_overlap: bool,
       pub score: f32,
   }
   
   impl SearchResult {
       /// Get a preview of the content around matches
       pub fn get_preview(&self, max_length: usize) -> String {
           if self.content.len() <= max_length {
               self.content.clone()
           } else {
               format!("{}...", &self.content[..max_length])
           }
       }
   }
   
   /// Search configuration options
   #[derive(Debug, Clone)]
   pub struct SearchOptions {
       pub limit: usize,
       pub file_filter: Option<String>,
   }
   
   impl Default for SearchOptions {
       fn default() -> Self {
           Self {
               limit: 50,
               file_filter: None,
           }
       }
   }
   
   /// Complete search response with metadata
   #[derive(Debug, Clone)]
   pub struct SearchResponse {
       pub results: Vec<SearchResult>,
       pub query: String,
       pub total_found: usize,
       pub options: SearchOptions,
   }
   
   /// Search engine statistics
   #[derive(Debug, Clone)]
   pub struct SearchStats {
       pub total_documents: usize,
       pub index_size_mb: usize,
   }
   ```

2. **Add comprehensive search tests:**

   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       use crate::indexer::DocumentIndexer;
       use tempfile::TempDir;
       use std::fs;
       
       fn setup_test_index() -> Result<(TempDir, DocumentIndexer, SearchEngine)> {
           let temp_dir = TempDir::new()?;
           let index_path = temp_dir.path().join("search_test_index");
           
           // Create and populate index
           let mut indexer = DocumentIndexer::new(&index_path)?;
           
           let test_files = vec![
               ("config.toml", r#"
                   [workspace]
                   members = ["backend", "frontend"]
                   [dependencies]
                   tokio = "1.0"
               "#),
               ("main.rs", r#"
                   pub fn process<T, E>() -> Result<T, E> {
                       // ## Processing function
                       #[derive(Debug)]
                       struct Config {
                           value: &mut String,
                       }
                       Ok(todo!())
                   }
               "#),
               ("utils.py", r#"
                   def helper():
                       """Helper function"""
                       return "Hello"
                   
                   class DataProcessor:
                       def __init__(self):
                           self.data = []
               "#),
           ];
           
           for (filename, content) in test_files {
               let file_path = temp_dir.path().join(filename);
               fs::write(&file_path, content)?;
               indexer.index_file(&file_path)?;
           }
           
           indexer.commit()?;
           
           let search_engine = SearchEngine::new(&index_path)?;
           Ok((temp_dir, indexer, search_engine))
       }
       
       #[test]
       fn test_basic_search() -> Result<()> {
           let (_temp_dir, _indexer, search_engine) = setup_test_index()?;
           
           let results = search_engine.search("function", 10)?;
           assert!(!results.is_empty(), "Should find function-related content");
           
           // Verify result structure
           for result in &results {
               assert!(!result.file_path.is_empty());
               assert!(!result.content.is_empty());
               assert!(result.score > 0.0);
           }
           
           Ok(())
       }
       
       #[test]
       fn test_special_character_search() -> Result<()> {
           let (_temp_dir, _indexer, search_engine) = setup_test_index()?;
           
           let special_queries = vec![
               "[workspace]",
               "Result<T, E>",
               "#[derive",
               "&mut",
               "##",
           ];
           
           for query in special_queries {
               let results = search_engine.search(query, 10)?;
               assert!(!results.is_empty(), "Should find results for query: {}", query);
               
               // Verify the results actually contain the special characters
               let found_match = results.iter().any(|r| r.content.contains(query));
               assert!(found_match, "Results should contain the searched special characters: {}", query);
           }
           
           Ok(())
       }
       
       #[test]
       fn test_search_with_options() -> Result<()> {
           let (_temp_dir, _indexer, search_engine) = setup_test_index()?;
           
           let options = SearchOptions {
               limit: 5,
               file_filter: Some("rs".to_string()),
           };
           
           let response = search_engine.search_with_options("function", options)?;
           assert_eq!(response.query, "function");
           assert!(response.results.len() <= 5);
           
           // Check file filtering
           for result in &response.results {
               assert!(result.file_path.contains("rs"), 
                      "Filtered results should only include .rs files");
           }
           
           Ok(())
       }
       
       #[test]
       fn test_empty_query_handling() -> Result<()> {
           let (_temp_dir, _indexer, search_engine) = setup_test_index()?;
           
           let results = search_engine.search("nonexistentqueryterm", 10)?;
           assert!(results.is_empty(), "Should return empty results for non-matching query");
           
           Ok(())
       }
       
       #[test]
       fn test_search_stats() -> Result<()> {
           let (_temp_dir, _indexer, search_engine) = setup_test_index()?;
           
           let stats = search_engine.get_stats()?;
           assert!(stats.total_documents > 0, "Should have indexed documents");
           
           Ok(())
       }
       
       #[test]
       fn test_result_preview() {
           let result = SearchResult {
               file_path: "test.rs".to_string(),
               content: "This is a very long content string that should be truncated when preview is requested".to_string(),
               chunk_index: 0,
               chunk_start: 0,
               chunk_end: 100,
               has_overlap: false,
               score: 1.0,
           };
           
           let preview = result.get_preview(20);
           assert!(preview.len() <= 23); // 20 chars + "..."
           assert!(preview.ends_with("..."));
       }
   }
   ```

## Success Criteria
- [ ] SearchEngine compiles without errors
- [ ] All search tests pass with `cargo test` on search module
- [ ] Basic text search works correctly
- [ ] Special character queries return accurate results
- [ ] Search options (limit, filtering) work properly
- [ ] Empty queries handled gracefully
- [ ] Result metadata is complete and accurate
- [ ] Search statistics are available

## If This Task Fails

### Common Errors and Solutions

**Error 1: "error[E0599]: no method named `parse_query` found for type `QueryParser`"**
```bash
# Solution: Tantivy API mismatch or incorrect usage
cargo clean
cargo update tantivy --precise 0.22.0
# Check QueryParser::for_index() call and field setup
cargo check
```

**Error 2: "failed to search: Query parsing failed"**
```bash
# Solution: Query parser not configured with correct fields
# Verify both content and raw_content fields are included
# Check field names match schema exactly
RUST_LOG=tantivy=debug cargo test test_basic_search
```

**Error 3: "no documents returned for valid query"**
```bash
# Solution: Index not properly populated or committed
# Verify indexer.commit() was called
# Check document creation and field mapping
cargo test test_search_integration -- --nocapture
```

**Error 4: "thread panicked at 'SearchEngine creation failed'"**
```bash
# Solution: Schema or index incompatibility
# Verify schema matches between indexer and search engine
# Check index directory exists and is readable
ls -la /tmp/test_index/
cargo test setup_test_index
```

## Troubleshooting Checklist

- [ ] Task 03 schema implementation completed successfully
- [ ] Task 05 chunker implementation working
- [ ] All previous indexing tests pass
- [ ] Tantivy version exactly "0.22.0" in Cargo.toml
- [ ] QueryParser includes both content and raw_content fields
- [ ] Index directory exists and has proper permissions
- [ ] Documents properly committed to index before searching
- [ ] Field names match exactly between schema and search code
- [ ] Search options struct properly initialized
- [ ] No compilation errors in search engine module

## Recovery Procedures

### Query Parser Configuration Issues
If query parsing consistently fails:
1. Verify field setup: Check content and raw_content fields exist in schema
2. Test field access: `schema.get_field("content")` should succeed
3. Debug query construction: Add println! statements in parse_query calls
4. Test with simple queries first: Single words before complex patterns

### Search Result Problems
If searches return no results despite indexed content:
1. Verify index commitment: Check that IndexWriter.commit() was called
2. Test index contents: Use tantivy command-line tools to inspect index
3. Check field mapping: Ensure document fields match search fields
4. Verify reader/searcher setup: Confirm proper index reader creation

### Performance Issues
If searches are slow or use excessive memory:
1. Limit result count: Use smaller limits in TopDocs
2. Optimize query complexity: Avoid overly broad wildcard searches
3. Check index size: Verify index isn't corrupted or oversized
4. Monitor memory usage: Use cargo flamegraph for profiling

### File Filtering Problems
If file type filtering doesn't work:
1. Check file_path field content: Verify paths are stored correctly
2. Test filter logic: Debug file extension extraction
3. Verify filter application: Ensure filtering happens before limit
4. Test with known file types: Start with simple .rs, .py extensions

## Common Pitfalls to Avoid
- Don't forget to include both content and raw_content fields in query parser
- Handle query parsing errors gracefully (don't crash on invalid syntax)
- Ensure field extraction handles missing fields properly
- Don't ignore score information (important for ranking)
- Test both positive and negative search cases

## Context for Next Task
Task 11 will create comprehensive integration tests that verify the complete indexing → searching workflow with special character handling.