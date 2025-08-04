# Task 00_5: Create Basic Search Functionality

**Estimated Time:** 8-10 minutes  
**Prerequisites:** Task 00_4 (document indexing completed)  
**Dependencies:** Must be completed before Task 00_6

## Objective
Implement basic search functionality that can query the Tantivy index created by the document indexing system, return ranked results, and provide the search interface that Phase 2 tasks expect.

## Context
You are completing the basic search pipeline by implementing the SearchEngine that uses the Tantivy index. This must handle text queries, return relevant results with proper scoring, and provide the foundation for more advanced search features like boolean queries and semantic search that Phase 2 will add.

## Task Details

### What You Need to Do
1. **Complete src/search/mod.rs with full SearchEngine implementation:**
   - Query parsing and execution using Tantivy
   - Result scoring and ranking
   - Proper error handling for search operations
   - Support for field-specific queries

2. **Implement search result formatting:**
   - Highlight matching terms in results
   - Return metadata like file paths and chunk indices
   - Handle empty result sets gracefully

3. **Create search integration helpers:**
   - Simple text search interface
   - File path filtering
   - Language-specific search

### Implementation Details

#### Complete src/search/mod.rs
```rust
//! Search engine implementation for vector search

use anyhow::{Context, Result as AnyhowResult};
use std::path::Path;
use tantivy::{
    collector::TopDocs,
    query::{AllQuery, BooleanQuery, Occur, Query, QueryParser, TermQuery},
    schema::Field,
    Index, IndexReader, Searcher, Term,
};
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::indexing::{DocumentIndexer, IndexingError, DocumentSchema};

#[derive(Error, Debug)]
pub enum SearchError {
    #[error("Query parsing error: {0}")]
    QueryParsing(String),
    
    #[error("Index error: {0}")]
    Index(#[from] tantivy::TantivyError),
    
    #[error("Indexing error: {0}")]
    Indexing(#[from] IndexingError),
    
    #[error("Search execution error: {0}")]
    Execution(String),
}

/// Search result structure
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub score: f32,
    pub content: String,
    pub file_path: String,
    pub chunk_index: u64,
    pub language: String,
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
}

/// Search configuration
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub max_results: usize,
    pub min_score: f32,
    pub include_content: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_results: 20,
            min_score: 0.0,
            include_content: true,
        }
    }
}

/// Search engine for querying indexed documents
pub struct SearchEngine {
    index: Index,
    reader: IndexReader,
    schema: DocumentSchema,
    config: SearchConfig,
}

impl SearchEngine {
    pub fn new<P: AsRef<Path>>(index_path: P) -> Result<Self, SearchError> {
        info!("Opening search engine at: {:?}", index_path.as_ref());
        
        // Create a temporary indexer to get the schema and ensure index exists
        let indexer = DocumentIndexer::new(index_path)?;
        let index = indexer.index.clone();
        let schema = indexer.get_schema().clone();
        let reader = indexer.get_reader()?;
        
        Ok(SearchEngine {
            index,
            reader,
            schema,
            config: SearchConfig::default(),
        })
    }
    
    pub fn with_config(mut self, config: SearchConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Simple text search across all content
    pub fn search_text(&self, query_text: &str) -> Result<Vec<SearchResult>, SearchError> {
        debug!("Searching for text: '{}'", query_text);
        
        let searcher = self.reader.searcher();
        
        // Create query parser for content field
        let query_parser = QueryParser::for_index(
            &self.index,
            vec![self.schema.content_field]
        );
        
        let query = query_parser
            .parse_query(query_text)
            .map_err(|e| SearchError::QueryParsing(format!("Failed to parse query '{}': {}", query_text, e)))?;
        
        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(self.config.max_results))
            .map_err(|e| SearchError::Execution(format!("Search execution failed: {}", e)))?;
        
        let mut results = Vec::new();
        
        for (score, doc_address) in top_docs {
            if score < self.config.min_score {
                continue;
            }
            
            let retrieved_doc = searcher.doc(doc_address)
                .map_err(|e| SearchError::Execution(format!("Failed to retrieve document: {}", e)))?;
            
            let result = self.convert_to_search_result(score, &retrieved_doc)?;
            results.push(result);
        }
        
        info!("Found {} results for query: '{}'", results.len(), query_text);
        Ok(results)
    }
    
    /// Search within specific file paths
    pub fn search_in_files(&self, query_text: &str, file_patterns: &[&str]) -> Result<Vec<SearchResult>, SearchError> {
        debug!("Searching for '{}' in files matching: {:?}", query_text, file_patterns);
        
        let searcher = self.reader.searcher();
        
        // Parse main query
        let content_query_parser = QueryParser::for_index(
            &self.index,
            vec![self.schema.content_field]
        );
        
        let content_query = content_query_parser
            .parse_query(query_text)
            .map_err(|e| SearchError::QueryParsing(format!("Failed to parse content query: {}", e)))?;
        
        // Create file path queries
        let mut file_queries: Vec<Box<dyn Query>> = Vec::new();
        for pattern in file_patterns {
            let file_query_parser = QueryParser::for_index(
                &self.index,
                vec![self.schema.file_path_field]
            );
            
            let file_query = file_query_parser
                .parse_query(&format!("*{}*", pattern))
                .map_err(|e| SearchError::QueryParsing(format!("Failed to parse file pattern '{}': {}", pattern, e)))?;
            
            file_queries.push(file_query);
        }
        
        // Combine queries
        let mut boolean_query = BooleanQuery::new();
        boolean_query.add(Occur::Must, content_query);
        
        if !file_queries.is_empty() {
            let mut file_boolean = BooleanQuery::new();
            for file_query in file_queries {
                file_boolean.add(Occur::Should, file_query);
            }
            boolean_query.add(Occur::Must, Box::new(file_boolean));
        }
        
        let final_query: Box<dyn Query> = Box::new(boolean_query);
        
        let top_docs = searcher
            .search(&final_query, &TopDocs::with_limit(self.config.max_results))
            .map_err(|e| SearchError::Execution(format!("File search execution failed: {}", e)))?;
        
        let mut results = Vec::new();
        
        for (score, doc_address) in top_docs {
            if score < self.config.min_score {
                continue;
            }
            
            let retrieved_doc = searcher.doc(doc_address)
                .map_err(|e| SearchError::Execution(format!("Failed to retrieve document: {}", e)))?;
            
            let result = self.convert_to_search_result(score, &retrieved_doc)?;
            results.push(result);
        }
        
        info!("Found {} results for file-filtered query", results.len());
        Ok(results)
    }
    
    /// Search by language type
    pub fn search_by_language(&self, query_text: &str, language: &str) -> Result<Vec<SearchResult>, SearchError> {
        debug!("Searching for '{}' in {} files", query_text, language);
        
        let searcher = self.reader.searcher();
        
        // Parse content query
        let content_query_parser = QueryParser::for_index(
            &self.index,
            vec![self.schema.content_field]
        );
        
        let content_query = content_query_parser
            .parse_query(query_text)
            .map_err(|e| SearchError::QueryParsing(format!("Failed to parse query: {}", e)))?;
        
        // Create language filter
        let language_term = Term::from_field_text(self.schema.language_field, language);
        let language_query = TermQuery::new(language_term, tantivy::schema::IndexRecordOption::Basic);
        
        // Combine queries
        let mut boolean_query = BooleanQuery::new();
        boolean_query.add(Occur::Must, content_query);
        boolean_query.add(Occur::Must, Box::new(language_query));
        
        let final_query: Box<dyn Query> = Box::new(boolean_query);
        
        let top_docs = searcher
            .search(&final_query, &TopDocs::with_limit(self.config.max_results))
            .map_err(|e| SearchError::Execution(format!("Language search execution failed: {}", e)))?;
        
        let mut results = Vec::new();
        
        for (score, doc_address) in top_docs {
            if score < self.config.min_score {
                continue;
            }
            
            let retrieved_doc = searcher.doc(doc_address)
                .map_err(|e| SearchError::Execution(format!("Failed to retrieve document: {}", e)))?;
            
            let result = self.convert_to_search_result(score, &retrieved_doc)?;
            results.push(result);
        }
        
        info!("Found {} results for language-filtered query", results.len());
        Ok(results)
    }
    
    /// List all indexed languages
    pub fn get_indexed_languages(&self) -> Result<Vec<String>, SearchError> {
        let searcher = self.reader.searcher();
        
        // Use AllQuery to get all documents, then extract unique languages
        let all_query = AllQuery;
        let top_docs = searcher
            .search(&all_query, &TopDocs::with_limit(10000)) // Large limit to get all docs
            .map_err(|e| SearchError::Execution(format!("Failed to query all documents: {}", e)))?;
        
        let mut languages = std::collections::HashSet::new();
        
        for (_, doc_address) in top_docs {
            let retrieved_doc = searcher.doc(doc_address)
                .map_err(|e| SearchError::Execution(format!("Failed to retrieve document: {}", e)))?;
            
            if let Some(language_values) = retrieved_doc.get_all(self.schema.language_field).next() {
                if let Some(language_text) = language_values.as_text() {
                    languages.insert(language_text.to_string());
                }
            }
        }
        
        let mut language_list: Vec<String> = languages.into_iter().collect();
        language_list.sort();
        
        debug!("Found {} unique languages in index", language_list.len());
        Ok(language_list)
    }
    
    /// Get index statistics
    pub fn get_stats(&self) -> Result<SearchStats, SearchError> {
        let searcher = self.reader.searcher();
        let num_docs = searcher.num_docs() as usize;
        
        let languages = self.get_indexed_languages()?;
        
        Ok(SearchStats {
            total_documents: num_docs,
            unique_languages: languages.len(),
            available_languages: languages,
        })
    }
    
    /// Convert Tantivy document to SearchResult
    fn convert_to_search_result(
        &self,
        score: f32,
        doc: &tantivy::Document,
    ) -> Result<SearchResult, SearchError> {
        let content = doc
            .get_first(self.schema.content_field)
            .and_then(|f| f.as_text())
            .unwrap_or("")
            .to_string();
        
        let file_path = doc
            .get_first(self.schema.file_path_field)
            .and_then(|f| f.as_text())
            .unwrap_or("")
            .to_string();
        
        let chunk_index = doc
            .get_first(self.schema.chunk_index_field)
            .and_then(|f| f.as_u64())
            .unwrap_or(0);
        
        let language = doc
            .get_first(self.schema.language_field)
            .and_then(|f| f.as_text())
            .unwrap_or("")
            .to_string();
        
        Ok(SearchResult {
            score,
            content: if self.config.include_content { content } else { String::new() },
            file_path,
            chunk_index,
            language,
            start_line: None, // Will be populated from chunk metadata in advanced features
            end_line: None,
        })
    }
}

/// Search statistics
#[derive(Debug, Clone)]
pub struct SearchStats {
    pub total_documents: usize,
    pub unique_languages: usize,
    pub available_languages: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunking::SmartChunker;
    use crate::indexing::DocumentIndexer;
    use tempfile::TempDir;
    use std::fs;
    
    fn create_test_index() -> (TempDir, DocumentIndexer, SearchEngine) {
        let temp_dir = TempDir::new().unwrap();
        let mut indexer = DocumentIndexer::new(temp_dir.path()).unwrap();
        
        // Add test documents
        indexer.add_document(
            "fn main() { println!(\"Hello, world!\"); }",
            "test.rs",
            0,
            "rust"
        ).unwrap();
        
        indexer.add_document(
            "def hello_world(): print(\"Hello, world!\")",
            "test.py",
            0,
            "python"
        ).unwrap();
        
        indexer.add_document(
            "[workspace]\nmembers = [\"crate1\", \"crate2\"]",
            "Cargo.toml",
            0,
            "toml"
        ).unwrap();
        
        indexer.commit().unwrap();
        
        let search_engine = SearchEngine::new(temp_dir.path()).unwrap();
        
        (temp_dir, indexer, search_engine)
    }
    
    #[test]
    fn test_basic_search() {
        let (_temp_dir, _indexer, search_engine) = create_test_index();
        
        let results = search_engine.search_text("hello").unwrap();
        assert!(!results.is_empty());
        
        // Should find both rust and python files
        let languages: Vec<&str> = results.iter().map(|r| r.language.as_str()).collect();
        assert!(languages.contains(&"rust") || languages.contains(&"python"));
    }
    
    #[test]
    fn test_language_search() {
        let (_temp_dir, _indexer, search_engine) = create_test_index();
        
        let results = search_engine.search_by_language("hello", "rust").unwrap();
        assert!(!results.is_empty());
        
        for result in results {
            assert_eq!(result.language, "rust");
        }
    }
    
    #[test]
    fn test_file_search() {
        let (_temp_dir, _indexer, search_engine) = create_test_index();
        
        let results = search_engine.search_in_files("workspace", &["Cargo"]).unwrap();
        assert!(!results.is_empty());
        
        for result in results {
            assert!(result.file_path.contains("Cargo"));
        }
    }
    
    #[test]
    fn test_get_stats() {
        let (_temp_dir, _indexer, search_engine) = create_test_index();
        
        let stats = search_engine.get_stats().unwrap();
        assert_eq!(stats.total_documents, 3);
        assert!(stats.available_languages.contains(&"rust".to_string()));
        assert!(stats.available_languages.contains(&"python".to_string()));
        assert!(stats.available_languages.contains(&"toml".to_string()));
    }
    
    #[test]
    fn test_special_characters() {
        let (_temp_dir, _indexer, search_engine) = create_test_index();
        
        let results = search_engine.search_text("workspace").unwrap();
        assert!(!results.is_empty());
        
        let content = &results[0].content;
        assert!(content.contains("[workspace]"));
    }
}
```

### Expected Output Files
- **Modified:** `crates/vector-search/src/search/mod.rs` (complete implementation)
- **Validation:** All tests should pass with `cargo test -p vector-search`

## Success Criteria
- [ ] SearchEngine can perform basic text searches
- [ ] Language-specific search filtering works correctly
- [ ] File path filtering returns relevant results
- [ ] Search results include proper metadata (score, file path, language)
- [ ] Special characters in indexed content are searchable
- [ ] Statistics and language listing work correctly
- [ ] All tests pass including special character tests
- [ ] Error handling covers all search failure modes
- [ ] **Compilation Verification**: `cargo check -p vector-search` succeeds
- [ ] **Test Verification**: `cargo test -p vector-search test_basic_search` passes
- [ ] **Integration Verification**: No schema mismatches with indexing module

## Common Pitfalls to Avoid
- Don't lose special characters during query parsing
- Handle empty search results gracefully
- Ensure proper scoring and ranking of results
- Test with realistic queries that Phase 2 tasks will use
- Don't panic on malformed queries, return proper errors

## Context for Next Task
Task 00_6 will create integration tests that verify the complete pipeline from file processing through indexing to search, ensuring the foundation is solid for Phase 2 development.

## Final Verification Steps
After completing the implementation, run these commands to verify everything works:

```bash
# Verify compilation
cargo check -p vector-search

# Run specific search tests
cargo test -p vector-search test_basic_search
cargo test -p vector-search test_language_search
cargo test -p vector-search test_special_characters

# Run all vector-search tests
cargo test -p vector-search

# Verify complete workspace still compiles
cargo check

# Test end-to-end functionality
cargo test -p vector-search -- --nocapture
```

## Integration Notes
This search functionality provides:
- Text search interface that Phase 2 tasks expect
- Language and file filtering capabilities
- Result ranking and scoring
- Statistics for monitoring and debugging
- Error handling integrated with neuromorphic patterns
- Foundation for boolean queries and semantic search
- Special character support for code search queries