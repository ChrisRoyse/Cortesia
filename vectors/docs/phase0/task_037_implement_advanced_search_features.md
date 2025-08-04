# Micro-Task 037a: Add Multi-Field Search Support

## Objective
Extend searcher to support searching across multiple fields (title, content, file_path).

## Prerequisites
- Task 036e completed (basic searcher tests passing and committed)

## Time Estimate
8 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add to DocumentSearcher impl in `src/searcher.rs`:
   ```rust
   /// Search across multiple fields with field-specific weighting
   pub fn search_multi_field(&self, query_text: &str, limit: usize) -> TantivyResult<Vec<SearchResult>> {
       let searcher = self.reader.searcher();
       
       // Create query parser for multiple fields with weights
       let mut query_parser = QueryParser::for_index(
           searcher.index(),
           vec![
               self.schema_fields.content,    // Primary content search
               self.schema_fields.title,      // Title search
               self.schema_fields.file_path,  // Path search
           ]
       );
       
       // Set field boosts (title gets higher weight)
       query_parser.set_field_boost(self.schema_fields.title, 2.0);
       query_parser.set_field_boost(self.schema_fields.content, 1.0);
       query_parser.set_field_boost(self.schema_fields.file_path, 0.5);
       
       let query = query_parser.parse_query(query_text)?;
       let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;
       
       let mut results = Vec::new();
       for (_score, doc_address) in top_docs {
           let retrieved_doc = searcher.doc(doc_address)?;
           if let Some(search_result) = SearchResult::from_document(&retrieved_doc, &self.schema_fields) {
               results.push(search_result);
           }
       }
       
       Ok(results)
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Multi-field search implemented with field weighting
- [ ] Query parser configured for multiple fields
- [ ] Code compiles successfully

## Next Task
task_037b_add_field_specific_search_methods.md

---

# Micro-Task 037b: Add Field-Specific Search Methods

## Objective
Add methods for searching specific fields (title-only, path-only searches).

## Prerequisites
- Task 037a completed (multi-field search implemented)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add field-specific search methods to DocumentSearcher impl in `src/searcher.rs`:
   ```rust
   /// Search only in document titles
   pub fn search_titles(&self, query_text: &str, limit: usize) -> TantivyResult<Vec<SearchResult>> {
       let searcher = self.reader.searcher();
       let mut query_parser = QueryParser::for_index(
           searcher.index(),
           vec![self.schema_fields.title]
       );
       
       let query = query_parser.parse_query(query_text)?;
       let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;
       
       let mut results = Vec::new();
       for (_score, doc_address) in top_docs {
           let retrieved_doc = searcher.doc(doc_address)?;
           if let Some(search_result) = SearchResult::from_document(&retrieved_doc, &self.schema_fields) {
               results.push(search_result);
           }
       }
       
       Ok(results)
   }
   
   /// Search by file path or extension
   pub fn search_by_path(&self, path_query: &str, limit: usize) -> TantivyResult<Vec<SearchResult>> {
       let searcher = self.reader.searcher();
       let mut query_parser = QueryParser::for_index(
           searcher.index(),
           vec![self.schema_fields.file_path, self.schema_fields.extension]
       );
       
       let query = query_parser.parse_query(path_query)?;
       let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;
       
       let mut results = Vec::new();
       for (_score, doc_address) in top_docs {
           let retrieved_doc = searcher.doc(doc_address)?;
           if let Some(search_result) = SearchResult::from_document(&retrieved_doc, &self.schema_fields) {
               results.push(search_result);
           }
       }
       
       Ok(results)
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Title-specific search method implemented
- [ ] Path-specific search method implemented
- [ ] Both methods compile successfully

## Next Task
task_037c_add_fuzzy_search_support.md

---

# Micro-Task 037c: Add Fuzzy Search Support

## Objective
Implement fuzzy search functionality for handling typos and approximate matches.

## Prerequisites
- Task 037b completed (field-specific search methods added)

## Time Estimate
9 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add to DocumentSearcher impl in `src/searcher.rs`:
   ```rust
   use tantivy::query::FuzzyTermQuery;
   use tantivy::Term;
   ```
3. Add fuzzy search method:
   ```rust
   /// Fuzzy search with edit distance tolerance
   pub fn fuzzy_search(&self, query_text: &str, limit: usize, max_distance: u8) -> TantivyResult<Vec<SearchResult>> {
       let searcher = self.reader.searcher();
       
       // Create fuzzy term query for content field
       let term = Term::from_field_text(self.schema_fields.content, query_text);
       let fuzzy_query = FuzzyTermQuery::new(term, max_distance, true);
       
       let top_docs = searcher.search(&fuzzy_query, &TopDocs::with_limit(limit))?;
       
       let mut results = Vec::new();
       for (_score, doc_address) in top_docs {
           let retrieved_doc = searcher.doc(doc_address)?;
           if let Some(search_result) = SearchResult::from_document(&retrieved_doc, &self.schema_fields) {
               results.push(search_result);
           }
       }
       
       Ok(results)
   }
   
   /// Fuzzy search with default edit distance of 2
   pub fn fuzzy_search_default(&self, query_text: &str, limit: usize) -> TantivyResult<Vec<SearchResult>> {
       self.fuzzy_search(query_text, limit, 2)
   }
   ```
4. Test: `cargo check`
5. Return to root: `cd ..\..`

## Success Criteria
- [ ] Fuzzy search implemented with configurable edit distance
- [ ] Default fuzzy search convenience method added
- [ ] Code compiles successfully

## Next Task
task_037d_add_search_result_scoring.md

---

# Micro-Task 037d: Add Search Result Scoring

## Objective
Enhance SearchResult to include relevance scores and implement score-based sorting.

## Prerequisites
- Task 037c completed (fuzzy search support added)

## Time Estimate
8 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Update SearchResult struct in `src/searcher.rs`:
   ```rust
   /// Search result containing document information with scoring
   #[derive(Debug, Clone)]
   pub struct SearchResult {
       pub content: String,
       pub title: String,
       pub file_path: Option<String>,
       pub doc_id: String,
       pub extension: Option<String>,
       pub score: f32,
   }
   
   impl SearchResult {
       /// Create SearchResult from Tantivy document with score
       pub fn from_document_with_score(
           doc: &tantivy::Document, 
           fields: &SchemaFields, 
           score: f32
       ) -> Option<Self> {
           let content = doc.get_first(fields.content)?.as_text()?.to_string();
           let title = doc.get_first(fields.title)?.as_text()?.to_string();
           let doc_id = doc.get_first(fields.doc_id)?.as_text()?.to_string();
           
           let file_path = doc.get_first(fields.file_path)
               .and_then(|v| v.as_text())
               .map(|s| s.to_string());
               
           let extension = doc.get_first(fields.extension)
               .and_then(|v| v.as_text())
               .map(|s| s.to_string());
           
           Some(SearchResult {
               content,
               title,
               file_path,
               doc_id,
               extension,
               score,
           })
       }
       
       /// Backward compatibility - create with score 0.0
       pub fn from_document(doc: &tantivy::Document, fields: &SchemaFields) -> Option<Self> {
           Self::from_document_with_score(doc, fields, 0.0)
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] SearchResult enhanced with score field
- [ ] Backward compatibility maintained
- [ ] Score-aware constructor implemented

## Next Task
task_037e_update_search_methods_with_scoring.md

---

# Micro-Task 037e: Update Search Methods with Scoring

## Objective
Update all search methods to use the new scoring functionality.

## Prerequisites
- Task 037d completed (search result scoring added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Update the search method in DocumentSearcher impl in `src/searcher.rs`:
   ```rust
   /// Search for documents containing the query text (updated with scoring)
   pub fn search(&self, query_text: &str, limit: usize) -> TantivyResult<Vec<SearchResult>> {
       let searcher = self.reader.searcher();
       
       let mut query_parser = QueryParser::for_index(
           searcher.index(),
           vec![self.schema_fields.content]
       );
       
       let query = query_parser.parse_query(query_text)?;
       let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;
       
       let mut results = Vec::new();
       for (score, doc_address) in top_docs {
           let retrieved_doc = searcher.doc(doc_address)?;
           
           if let Some(search_result) = SearchResult::from_document_with_score(
               &retrieved_doc, 
               &self.schema_fields, 
               score
           ) {
               results.push(search_result);
           }
       }
       
       Ok(results)
   }
   ```
3. Update multi_field_search method similarly:
   ```rust
   // Replace the results collection loop in search_multi_field
   let mut results = Vec::new();
   for (score, doc_address) in top_docs {
       let retrieved_doc = searcher.doc(doc_address)?;
       if let Some(search_result) = SearchResult::from_document_with_score(
           &retrieved_doc, 
           &self.schema_fields, 
           score
       ) {
           results.push(search_result);
       }
   }
   ```
4. Test: `cargo check`
5. Return to root: `cd ..\..`

## Success Criteria
- [ ] Search methods updated to use scoring
- [ ] Scores properly captured from Tantivy results
- [ ] Code compiles successfully

## Next Task
task_037f_add_advanced_search_tests_and_commit.md