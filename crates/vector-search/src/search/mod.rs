//! Search engine implementation for vector search

use thiserror::Error;
use tantivy::{Index, schema::{Schema, Field, TEXT, Value}, TantivyError, query::{QueryParser, FuzzyTermQuery, RegexQuery}, collector::TopDocs, TantivyDocument, Term};

#[derive(Error, Debug)]
pub enum SearchError {
    #[error("Query parsing error: {0}")]
    QueryParsing(String),
    
    #[error("Index error: {0}")]
    Index(String),
    
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
    
    #[error("Tantivy error: {0}")]
    Tantivy(#[from] TantivyError),
}

/// Search result structure
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub score: f32,
    pub content: String,
    pub file_path: String,
}

/// Search engine for querying indexed documents
pub struct SearchEngine {
    index: Index,
    content_field: Field,
}

impl SearchEngine {
    pub fn new() -> Result<Self, SearchError> {
        let mut schema_builder = Schema::builder();
        let content_field = schema_builder.add_text_field("content", TEXT);
        let schema = schema_builder.build();
        
        let index = Index::create_in_ram(schema);
        
        Ok(SearchEngine {
            index,
            content_field,
        })
    }
    
    /// Helper method to convert Tantivy search results to SearchResult structs
    fn convert_docs_to_results(&self, top_docs: Vec<(f32, tantivy::DocAddress)>, searcher: &tantivy::Searcher) -> Result<Vec<SearchResult>, SearchError> {
        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
            let content = retrieved_doc
                .get_first(self.content_field)
                .and_then(|field_value| field_value.as_str())
                .unwrap_or("")
                .to_string();
            
            results.push(SearchResult {
                score,
                content,
                file_path: "".to_string(), // Will be expanded in later tasks to include actual file paths
            });
        }
        Ok(results)
    }
    
    /// Get the schema for this search engine
    pub fn get_schema(&self) -> Schema {
        self.index.schema()
    }
    
    /// Parse a query string into a Tantivy query
    pub fn parse_query(&self, query_str: &str) -> Result<Box<dyn tantivy::query::Query>, SearchError> {
        let query_parser = QueryParser::for_index(&self.index, vec![self.content_field]);
        query_parser.parse_query(query_str)
            .map_err(|e| SearchError::QueryParsing(e.to_string()))
    }
    
    /// Perform a basic search on an empty index (returns empty results)
    pub fn search(&self, query_str: &str) -> Result<Vec<SearchResult>, SearchError> {
        let query = self.parse_query(query_str)?;
        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
        
        self.convert_docs_to_results(top_docs, &searcher)
    }
    
    /// Search for an exact phrase using Tantivy's phrase query syntax
    pub fn search_phrase(&self, phrase: &str) -> Result<Vec<SearchResult>, SearchError> {
        let phrase_query = format!("\"{}\"", phrase);
        self.search(&phrase_query)
    }
    
    /// Search for terms within specified distance using Tantivy's phrase-with-slop syntax
    pub fn search_proximity(&self, term1: &str, term2: &str, distance: u32) -> Result<Vec<SearchResult>, SearchError> {
        // Input validation
        if term1.trim().is_empty() || term2.trim().is_empty() {
            return Err(SearchError::QueryParsing("Proximity search terms cannot be empty".to_string()));
        }
        
        // Sanitize terms for query construction
        let clean_term1 = term1.trim().replace('"', "");
        let clean_term2 = term2.trim().replace('"', "");
        
        // CORRECT syntax: "term1 term2"~N (not "term1"~N "term2")
        let proximity_query = format!("\"{} {}\"~{}", clean_term1, clean_term2, distance);
        self.search(&proximity_query)
    }
    
    /// Search for fuzzy matches with edit distance
    pub fn search_fuzzy(&self, term: &str, distance: u8) -> Result<Vec<SearchResult>, SearchError> {
        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        
        let term = Term::from_field_text(self.content_field, term);
        let fuzzy_query = FuzzyTermQuery::new(term, distance, true);
        
        let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(10))?;
        
        self.convert_docs_to_results(top_docs, &searcher)
    }
    
    /// Add a document to the index for testing
    pub fn add_document(&mut self, _file_path: &str, content: &str) -> Result<(), SearchError> {
        let mut writer: tantivy::IndexWriter<TantivyDocument> = self.index.writer(50_000_000)?;
        
        let mut doc = TantivyDocument::default();
        doc.add_text(self.content_field, content);
        
        writer.add_document(doc)?;
        writer.commit()?;
        
        Ok(())
    }
    
    /// Search using regex patterns with Tantivy's RegexQuery API
    pub fn search_regex(&self, pattern: &str) -> Result<Vec<SearchResult>, SearchError> {
        if pattern.trim().is_empty() {
            return Err(SearchError::QueryParsing("Regex pattern cannot be empty".to_string()));
        }
        
        // Use REAL Tantivy RegexQuery API
        let regex_query = RegexQuery::from_pattern(pattern, self.content_field)
            .map_err(|e| SearchError::QueryParsing(format!("Invalid regex pattern: {}", e)))?;
        
        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        let top_docs = searcher.search(&regex_query, &TopDocs::with_limit(10))?;
        
        self.convert_docs_to_results(top_docs, &searcher)
    }
    
    /// Commit any pending changes to the index
    pub fn commit(&mut self) -> Result<(), SearchError> {
        let mut writer: tantivy::IndexWriter<TantivyDocument> = self.index.writer(50_000_000)?;
        writer.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_engine_new() {
        let engine = SearchEngine::new().unwrap();
        // Basic construction should work
        assert_eq!(engine.index.schema().num_fields(), 1);
    }
    
    #[test] 
    fn test_search_engine_has_content_field() {
        let engine = SearchEngine::new().unwrap();
        let schema = engine.get_schema();
        assert!(schema.get_field("content").is_ok());
    }
    
    #[test]
    fn test_basic_query_parsing() {
        let engine = SearchEngine::new().unwrap();
        let result = engine.parse_query("hello world");
        assert!(result.is_ok());
    }
    
    #[test] 
    fn test_basic_search_empty_index() {
        let engine = SearchEngine::new().unwrap();
        let results = engine.search("test").unwrap();
        assert_eq!(results.len(), 0); // Empty index returns no results
    }
    
    #[test]
    fn test_phrase_search_syntax() {
        let engine = SearchEngine::new().unwrap();
        // Test REAL Tantivy phrase syntax
        let result = engine.search_phrase("hello world");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_proximity_search() {
        let engine = SearchEngine::new().unwrap();
        let result = engine.search_proximity("hello", "world", 2);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_proximity_search_validation() {
        let engine = SearchEngine::new().unwrap();
        // Test empty terms should error
        let result = engine.search_proximity("", "world", 1);
        assert!(result.is_err());
        
        let result = engine.search_proximity("hello", "", 1);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_fuzzy_search() {
        let engine = SearchEngine::new().unwrap();
        let result = engine.search_fuzzy("hello", 1);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_regex_search() {
        let engine = SearchEngine::new().unwrap();
        let result = engine.search_regex(r"hel+o");
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_real_data_integration() {
        let mut engine = SearchEngine::new().unwrap();
        
        // Add real documents
        engine.add_document("test1.rs", "pub fn hello_world() { println!(\"Hello, world!\"); }").unwrap();
        engine.add_document("test2.rs", "fn greet() { println!(\"Hello there!\"); }").unwrap(); 
        engine.add_document("test3.rs", "struct Data { pub value: i32 }").unwrap();
        
        // Test phrase search with real data
        let results = engine.search_phrase("Hello, world").unwrap();
        assert!(!results.is_empty(), "Should find phrase in real data");
        
        // Test proximity search with real data
        let results = engine.search_proximity("pub", "fn", 0).unwrap();
        assert!(!results.is_empty(), "Should find adjacent terms in real data");
        
        // Test fuzzy search with real data
        let results = engine.search_fuzzy("hello", 1).unwrap();
        assert!(!results.is_empty(), "Should find fuzzy matches in real data");
        
        // Test regex search with real data
        let results = engine.search_regex(r"fn|struct").unwrap();
        assert!(!results.is_empty(), "Should find regex matches in real data");
    }
}