# Task 16: Implement Query Parsing and Preparation

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 15 (Pipeline integration)  
**Dependencies:** Tasks 01-15 must be completed

## Objective
Implement a robust query parser that handles natural language queries, special syntax for code search, field-specific searches, and query optimization for the Tantivy search engine.

## Context
Users need to search for code and text using various query types: simple text search, function names, specific file types, complex boolean queries, and semantic searches. The query parser must handle these different patterns and convert them into optimized Tantivy queries.

## Task Details

### What You Need to Do

1. **Create query parsing in `src/query.rs`:**

   ```rust
   use tantivy::{
       Index, Term, Query, Searcher,
       query::{BooleanQuery, TermQuery, FuzzyTermQuery, PhraseQuery, RangeQuery, RegexQuery, AllQuery},
       schema::{Field, Schema},
   };
   use std::collections::HashMap;
   use anyhow::Result;
   use regex::Regex;
   
   #[derive(Debug, Clone)]
   pub struct ParsedQuery {
       pub query: Box<dyn Query>,
       pub fields: Vec<Field>,
       pub filters: QueryFilters,
       pub options: SearchOptions,
       pub original_text: String,
   }
   
   #[derive(Debug, Clone)]
   pub struct QueryFilters {
       pub file_types: Vec<String>,
       pub languages: Vec<String>,
       pub semantic_types: Vec<String>,
       pub has_documentation: Option<bool>,
       pub complexity_range: Option<(f32, f32)>,
       pub date_range: Option<(std::time::SystemTime, std::time::SystemTime)>,
   }
   
   #[derive(Debug, Clone)]
   pub struct SearchOptions {
       pub fuzzy_distance: u8,
       pub enable_fuzzy: bool,
       pub phrase_slop: u32,
       pub boost_documented: bool,
       pub boost_recent: bool,
       pub max_results: usize,
       pub highlight_snippets: bool,
   }
   
   impl Default for QueryFilters {
       fn default() -> Self {
           Self {
               file_types: Vec::new(),
               languages: Vec::new(),
               semantic_types: Vec::new(),
               has_documentation: None,
               complexity_range: None,
               date_range: None,
           }
       }
   }
   
   impl Default for SearchOptions {
       fn default() -> Self {
           Self {
               fuzzy_distance: 2,
               enable_fuzzy: true,
               phrase_slop: 5,
               boost_documented: true,
               boost_recent: false,
               max_results: 50,
               highlight_snippets: true,
           }
       }
   }
   
   pub struct QueryParser {
       schema: Schema,
       fields: QueryFields,
       special_syntax_patterns: HashMap<String, Regex>,
   }
   
   #[derive(Debug, Clone)]
   pub struct QueryFields {
       pub content: Field,
       pub file_path: Field,
       pub language: Field,
       pub semantic_type: Field,
       pub id: Field,
       pub chunk_index: Field,
       pub total_chunks: Field,
       pub start_byte: Field,
       pub end_byte: Field,
       pub overlap_prev: Field,
       pub overlap_next: Field,
   }
   
   impl QueryParser {
       /// Create a new query parser
       pub fn new(index: &Index) -> Result<Self> {
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
           
           let special_syntax_patterns = Self::create_special_syntax_patterns();
           
           Ok(Self {
               schema,
               fields,
               special_syntax_patterns,
           })
       }
       
       /// Parse a query string into a structured query
       pub fn parse(&self, query_text: &str) -> Result<ParsedQuery> {
           let mut filters = QueryFilters::default();
           let mut options = SearchOptions::default();
           
           // Extract filters and options from query text
           let cleaned_query = self.extract_filters_and_options(query_text, &mut filters, &mut options);
           
           // Parse the main query
           let query = self.parse_main_query(&cleaned_query, &filters, &options)?;
           
           // Determine relevant fields for search
           let fields = self.determine_search_fields(&filters);
           
           Ok(ParsedQuery {
               query,
               fields,
               filters,
               options,
               original_text: query_text.to_string(),
           })
       }
       
       /// Extract filters and options from query text
       fn extract_filters_and_options(&self, query_text: &str, filters: &mut QueryFilters, options: &mut SearchOptions) -> String {
           let mut cleaned_query = query_text.to_string();
           
           // Extract file type filters: filetype:rs, ext:py
           if let Some(captures) = self.special_syntax_patterns.get("filetype").unwrap().captures(&cleaned_query) {
               if let Some(file_type) = captures.get(1) {
                   filters.file_types.push(file_type.as_str().to_string());
                   cleaned_query = cleaned_query.replace(&captures[0], "").trim().to_string();
               }
           }
           
           // Extract language filters: lang:rust, language:python
           if let Some(captures) = self.special_syntax_patterns.get("language").unwrap().captures(&cleaned_query) {
               if let Some(language) = captures.get(1) {
                   filters.languages.push(language.as_str().to_string());
                   cleaned_query = cleaned_query.replace(&captures[0], "").trim().to_string();
               }
           }
           
           // Extract semantic type filters: type:function, semantic:struct
           if let Some(captures) = self.special_syntax_patterns.get("semantic_type").unwrap().captures(&cleaned_query) {
               if let Some(sem_type) = captures.get(1) {
                   filters.semantic_types.push(sem_type.as_str().to_string());
                   cleaned_query = cleaned_query.replace(&captures[0], "").trim().to_string();
               }
           }
           
           // Extract documentation filter: has:docs, documented:true
           if let Some(captures) = self.special_syntax_patterns.get("documentation").unwrap().captures(&cleaned_query) {
               filters.has_documentation = Some(true);
               cleaned_query = cleaned_query.replace(&captures[0], "").trim().to_string();
           }
           
           // Extract fuzzy search option: fuzzy:2, ~2
           if let Some(captures) = self.special_syntax_patterns.get("fuzzy").unwrap().captures(&cleaned_query) {
               if let Some(distance_str) = captures.get(1) {
                   if let Ok(distance) = distance_str.as_str().parse::<u8>() {
                       options.fuzzy_distance = distance;
                       options.enable_fuzzy = true;
                   }
               }
               cleaned_query = cleaned_query.replace(&captures[0], "").trim().to_string();
           }
           
           // Extract phrase slop: "phrase query"~5
           if let Some(captures) = self.special_syntax_patterns.get("phrase_slop").unwrap().captures(&cleaned_query) {
               if let Some(slop_str) = captures.get(2) {
                   if let Ok(slop) = slop_str.as_str().parse::<u32>() {
                       options.phrase_slop = slop;
                   }
               }
           }
           
           cleaned_query
       }
       
       /// Parse the main query after filters are extracted
       fn parse_main_query(&self, query_text: &str, filters: &QueryFilters, options: &SearchOptions) -> Result<Box<dyn Query>> {
           if query_text.trim().is_empty() {
               // If no search terms, create a match-all query with filters
               return Ok(Box::new(AllQuery));
           }
           
           // Check for different query types
           if query_text.starts_with('"') && query_text.ends_with('"') {
               // Phrase query
               self.create_phrase_query(query_text, options)
           } else if query_text.contains("AND") || query_text.contains("OR") || query_text.contains("NOT") {
               // Boolean query
               self.create_boolean_query(query_text, options)
           } else if query_text.starts_with('/') && query_text.ends_with('/') {
               // Regex query
               self.create_regex_query(query_text)
           } else if self.looks_like_function_signature(query_text) {
               // Function signature search
               self.create_function_signature_query(query_text, options)
           } else if self.looks_like_code_pattern(query_text) {
               // Code pattern search
               self.create_code_pattern_query(query_text, options)
           } else {
               // Default text search
               self.create_text_search_query(query_text, options)
           }
       }
       
       /// Create a phrase query for exact phrase matching
       fn create_phrase_query(&self, query_text: &str, options: &SearchOptions) -> Result<Box<dyn Query>> {
           let phrase = query_text.trim_matches('"');
           let terms: Vec<Term> = phrase.split_whitespace()
               .map(|word| Term::from_field_text(self.fields.content, word))
               .collect();
           
           if terms.is_empty() {
               return Ok(Box::new(AllQuery));
           }
           
           let phrase_query = PhraseQuery::new_with_offset_and_slop(terms, options.phrase_slop);
           Ok(Box::new(phrase_query))
       }
       
       /// Create a boolean query for complex search expressions
       fn create_boolean_query(&self, query_text: &str, options: &SearchOptions) -> Result<Box<dyn Query>> {
           // Simple boolean query parsing (could be enhanced with proper parser)
           let mut boolean_query = BooleanQuery::new();
           
           // Split on boolean operators and create subqueries
           let parts: Vec<&str> = query_text.split_whitespace().collect();
           let mut current_terms = Vec::new();
           let mut current_operator = "AND"; // Default
           
           for part in parts {
               match part {
                   "AND" | "OR" | "NOT" => {
                       if !current_terms.is_empty() {
                           let subquery = self.create_term_query(&current_terms.join(" "), options)?;
                           match current_operator {
                               "AND" => boolean_query.add_must(subquery),
                               "OR" => boolean_query.add_should(subquery),
                               "NOT" => boolean_query.add_must_not(subquery),
                               _ => boolean_query.add_must(subquery),
                           }
                           current_terms.clear();
                       }
                       current_operator = part;
                   }
                   _ => {
                       current_terms.push(part);
                   }
               }
           }
           
           // Handle remaining terms
           if !current_terms.is_empty() {
               let subquery = self.create_term_query(&current_terms.join(" "), options)?;
               match current_operator {
                   "AND" => boolean_query.add_must(subquery),
                   "OR" => boolean_query.add_should(subquery),
                   "NOT" => boolean_query.add_must_not(subquery),
                   _ => boolean_query.add_must(subquery),
               }
           }
           
           Ok(Box::new(boolean_query))
       }
       
       /// Create a regex query for pattern matching
       fn create_regex_query(&self, query_text: &str) -> Result<Box<dyn Query>> {
           let pattern = query_text.trim_matches('/');
           let regex_query = RegexQuery::from_pattern(pattern, self.fields.content)?;
           Ok(Box::new(regex_query))
       }
       
       /// Create a function signature query
       fn create_function_signature_query(&self, query_text: &str, options: &SearchOptions) -> Result<Box<dyn Query>> {
           // Parse function signature pattern like "fn name(params) -> return"
           let mut boolean_query = BooleanQuery::new();
           
           // Look for function-specific terms
           if query_text.contains("fn ") || query_text.contains("def ") {
               let content_query = self.create_term_query(query_text, options)?;
               boolean_query.add_must(content_query);
           }
           
           // Boost function semantic type
           let function_term = Term::from_field_text(self.fields.semantic_type, "function");
           let function_query = TermQuery::new(function_term, tantivy::schema::IndexRecordOption::Basic);
           boolean_query.add_should(Box::new(function_query));
           
           Ok(Box::new(boolean_query))
       }
       
       /// Create a code pattern query
       fn create_code_pattern_query(&self, query_text: &str, options: &SearchOptions) -> Result<Box<dyn Query>> {
           let mut boolean_query = BooleanQuery::new();
           
           // Create content query
           let content_query = self.create_term_query(query_text, options)?;
           boolean_query.add_must(content_query);
           
           // Boost code semantic types
           let code_types = ["function", "struct", "enum", "impl", "class"];
           for code_type in &code_types {
               let type_term = Term::from_field_text(self.fields.semantic_type, code_type);
               let type_query = TermQuery::new(type_term, tantivy::schema::IndexRecordOption::Basic);
               boolean_query.add_should(Box::new(type_query));
           }
           
           Ok(Box::new(boolean_query))
       }
       
       /// Create a text search query for general text search
       fn create_text_search_query(&self, query_text: &str, options: &SearchOptions) -> Result<Box<dyn Query>> {
           self.create_term_query(query_text, options)
       }
       
       /// Create a term query with fuzzy support
       fn create_term_query(&self, query_text: &str, options: &SearchOptions) -> Result<Box<dyn Query>> {
           let terms: Vec<&str> = query_text.split_whitespace().collect();
           
           if terms.is_empty() {
               return Ok(Box::new(AllQuery));
           }
           
           if terms.len() == 1 {
               // Single term - use fuzzy if enabled
               let term = Term::from_field_text(self.fields.content, terms[0]);
               
               if options.enable_fuzzy {
                   let fuzzy_query = FuzzyTermQuery::new(term, options.fuzzy_distance, true);
                   Ok(Box::new(fuzzy_query))
               } else {
                   let term_query = TermQuery::new(term, tantivy::schema::IndexRecordOption::WithFreqsAndPositions);
                   Ok(Box::new(term_query))
               }
           } else {
               // Multiple terms - create boolean query
               let mut boolean_query = BooleanQuery::new();
               
               for term in terms {
                   let term_obj = Term::from_field_text(self.fields.content, term);
                   
                   let subquery: Box<dyn Query> = if options.enable_fuzzy {
                       Box::new(FuzzyTermQuery::new(term_obj, options.fuzzy_distance, true))
                   } else {
                       Box::new(TermQuery::new(term_obj, tantivy::schema::IndexRecordOption::WithFreqsAndPositions))
                   };
                   
                   boolean_query.add_must(subquery);
               }
               
               Ok(Box::new(boolean_query))
           }
       }
       
       /// Determine which fields to search based on filters
       fn determine_search_fields(&self, filters: &QueryFilters) -> Vec<Field> {
           let mut fields = vec![self.fields.content]; // Always search content
           
           if !filters.file_types.is_empty() || !filters.languages.is_empty() {
               fields.push(self.fields.file_path);
               fields.push(self.fields.language);
           }
           
           if !filters.semantic_types.is_empty() {
               fields.push(self.fields.semantic_type);
           }
           
           fields
       }
       
       /// Check if query looks like a function signature
       fn looks_like_function_signature(&self, query_text: &str) -> bool {
           query_text.contains("(") && query_text.contains(")") &&
           (query_text.contains("fn ") || query_text.contains("def ") || 
            query_text.contains("->") || query_text.contains(":"))
       }
       
       /// Check if query looks like a code pattern
       fn looks_like_code_pattern(&self, query_text: &str) -> bool {
           let code_indicators = ["struct", "impl", "class", "enum", "trait", "::", "=>", "{", "}", ";"];
           code_indicators.iter().any(|&indicator| query_text.contains(indicator))
       }
       
       /// Create special syntax patterns for query parsing
       fn create_special_syntax_patterns() -> HashMap<String, Regex> {
           let mut patterns = HashMap::new();
           
           patterns.insert("filetype".to_string(), 
               Regex::new(r"(?i)(?:filetype|ext):(\w+)").unwrap());
           
           patterns.insert("language".to_string(), 
               Regex::new(r"(?i)(?:lang|language):(\w+)").unwrap());
           
           patterns.insert("semantic_type".to_string(), 
               Regex::new(r"(?i)(?:type|semantic):(\w+)").unwrap());
           
           patterns.insert("documentation".to_string(), 
               Regex::new(r"(?i)(?:has:docs|documented:true|has:documentation)").unwrap());
           
           patterns.insert("fuzzy".to_string(), 
               Regex::new(r"(?i)(?:fuzzy:(\d+)|~(\d+))").unwrap());
           
           patterns.insert("phrase_slop".to_string(), 
               Regex::new(r#""([^"]+)"~(\d+)"#).unwrap());
           
           patterns
       }
   }
   ```

2. **Add module declaration to `src/lib.rs`:**

   ```rust
   pub mod query;
   ```

3. **Add comprehensive tests for query parsing:**

   ```rust
   #[cfg(test)]
   mod query_tests {
       use super::*;
       use crate::schema::create_schema;
       use tempfile::TempDir;
       use tantivy::Index;
       
       fn create_test_parser() -> Result<QueryParser> {
           let temp_dir = TempDir::new()?;
           let schema = create_schema();
           let index = Index::create_in_dir(temp_dir.path(), schema)?;
           QueryParser::new(&index)
       }
       
       #[test]
       fn test_query_parser_creation() -> Result<()> {
           let _parser = create_test_parser()?;
           Ok(())
       }
       
       #[test]
       fn test_simple_text_query() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("hello world")?;
           
           assert_eq!(parsed.original_text, "hello world");
           assert!(!parsed.fields.is_empty());
           assert!(parsed.options.enable_fuzzy);
           
           Ok(())
       }
       
       #[test]
       fn test_phrase_query() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse(r#""exact phrase""#)?;
           
           assert_eq!(parsed.original_text, r#""exact phrase""#);
           // Phrase query should be created
           
           Ok(())
       }
       
       #[test]
       fn test_boolean_query() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("rust AND function")?;
           
           assert_eq!(parsed.original_text, "rust AND function");
           // Boolean query should be created
           
           Ok(())
       }
       
       #[test]
       fn test_regex_query() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("/fn\\s+\\w+/")?;
           
           assert_eq!(parsed.original_text, "/fn\\s+\\w+/");
           // Regex query should be created
           
           Ok(())
       }
       
       #[test]
       fn test_filetype_filter() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("search terms filetype:rs")?;
           
           assert_eq!(parsed.filters.file_types, vec!["rs"]);
           assert!(parsed.original_text.contains("filetype:rs"));
           
           Ok(())
       }
       
       #[test]
       fn test_language_filter() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("function lang:rust")?;
           
           assert_eq!(parsed.filters.languages, vec!["rust"]);
           
           Ok(())
       }
       
       #[test]
       fn test_semantic_type_filter() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("implementation type:struct")?;
           
           assert_eq!(parsed.filters.semantic_types, vec!["struct"]);
           
           Ok(())
       }
       
       #[test]
       fn test_documentation_filter() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("well documented has:docs")?;
           
           assert_eq!(parsed.filters.has_documentation, Some(true));
           
           Ok(())
       }
       
       #[test]
       fn test_fuzzy_option() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("search fuzzy:3")?;
           
           assert_eq!(parsed.options.fuzzy_distance, 3);
           assert!(parsed.options.enable_fuzzy);
           
           Ok(())
       }
       
       #[test]
       fn test_function_signature_detection() -> Result<()> {
           let parser = create_test_parser()?;
           
           assert!(parser.looks_like_function_signature("fn main()"));
           assert!(parser.looks_like_function_signature("def process(data: str) -> bool"));
           assert!(parser.looks_like_function_signature("function calculate()"));
           assert!(!parser.looks_like_function_signature("simple text"));
           
           Ok(())
       }
       
       #[test]
       fn test_code_pattern_detection() -> Result<()> {
           let parser = create_test_parser()?;
           
           assert!(parser.looks_like_code_pattern("struct Config {}"));
           assert!(parser.looks_like_code_pattern("impl Something"));
           assert!(parser.looks_like_code_pattern("Result<T, E>"));
           assert!(!parser.looks_like_code_pattern("plain text search"));
           
           Ok(())
       }
       
       #[test]
       fn test_complex_query_parsing() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("calculate AND (sum OR total) filetype:rs has:docs")?;
           
           assert_eq!(parsed.filters.file_types, vec!["rs"]);
           assert_eq!(parsed.filters.has_documentation, Some(true));
           assert!(parsed.original_text.contains("calculate AND"));
           
           Ok(())
       }
       
       #[test]
       fn test_empty_query() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("")?;
           
           assert_eq!(parsed.original_text, "");
           // Should create match-all query
           
           Ok(())
       }
       
       #[test]
       fn test_whitespace_query() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("   \t\n   ")?;
           
           // Should handle whitespace-only query gracefully
           assert_eq!(parsed.original_text, "   \t\n   ");
           
           Ok(())
       }
       
       #[test]
       fn test_multiple_filters() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("search text filetype:rs lang:rust type:function has:docs")?;
           
           assert_eq!(parsed.filters.file_types, vec!["rs"]);
           assert_eq!(parsed.filters.languages, vec!["rust"]);
           assert_eq!(parsed.filters.semantic_types, vec!["function"]);
           assert_eq!(parsed.filters.has_documentation, Some(true));
           
           Ok(())
       }
       
       #[test]
       fn test_special_characters_in_query() -> Result<()> {
           let parser = create_test_parser()?;
           let parsed = parser.parse("Result<T, E> -> Option<Value>")?;
           
           assert!(parsed.original_text.contains("Result<T, E>"));
           // Should handle special characters in code queries
           
           Ok(())
       }
   }
   ```

## Success Criteria
- [ ] Query parsing compiles without errors
- [ ] All query tests pass with `cargo test query_tests`
- [ ] Simple text queries are parsed correctly
- [ ] Phrase queries with quotes are handled properly
- [ ] Boolean queries (AND, OR, NOT) work correctly
- [ ] Regex queries are supported with /pattern/ syntax
- [ ] File type filters (filetype:ext) are extracted
- [ ] Language filters (lang:language) are extracted
- [ ] Semantic type filters (type:semantic) are extracted
- [ ] Documentation filters (has:docs) are recognized
- [ ] Fuzzy search options are parsed
- [ ] Function signature detection works
- [ ] Code pattern detection identifies code-specific queries
- [ ] Complex multi-filter queries are handled
- [ ] Edge cases (empty, whitespace) are handled gracefully

## Common Pitfalls to Avoid
- Don't fail on malformed regex patterns (handle gracefully)
- Handle Unicode characters properly in queries
- Don't assume all queries will have valid syntax
- Be careful with regex compilation (cache compiled patterns)
- Handle case sensitivity appropriately for different query types
- Don't lose original query information during parsing
- Ensure filter extraction doesn't break the main query parsing

## Context for Next Task
Task 17 will implement the search execution engine that takes parsed queries and executes them against the Tantivy index, handling result ranking and filtering.