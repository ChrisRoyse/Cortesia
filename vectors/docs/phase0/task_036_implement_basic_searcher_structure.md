# Micro-Task 036a: Create Searcher File Structure

## Objective
Create the basic file structure for searcher.rs with imports and module setup.

## Prerequisites
- Task 035h completed (Indexer tests completed and committed)
- tantivy-core crate has indexer working

## Time Estimate
5 minutes

## Instructions
1. Navigate to tantivy-core crate: `cd crates\tantivy-core`
2. Create basic `src/searcher.rs` structure:
   ```rust
   //! Document searching functionality
   
   use crate::schema::{create_code_index, SchemaFields};
   use tantivy::{Index, IndexReader, Searcher, Result as TantivyResult};
   use tantivy::query::{Query, QueryParser};
   use tantivy::collector::TopDocs;
   use std::path::Path;
   ```
3. Add searcher module to `src/lib.rs`:
   ```rust
   pub mod searcher;
   ```
4. Test compilation: `cargo check`
5. Return to root: `cd ..\..`

## Success Criteria
- [ ] searcher.rs file created with basic imports
- [ ] Module added to lib.rs
- [ ] File compiles without errors

## Next Task
task_036b_create_document_searcher_struct.md

---

# Micro-Task 036b: Create DocumentSearcher Struct

## Objective
Define the DocumentSearcher struct with constructor method.

## Prerequisites
- Task 036a completed (searcher file structure created)

## Time Estimate
8 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add to `src/searcher.rs`:
   ```rust
   /// Document searcher for text queries
   pub struct DocumentSearcher {
       reader: IndexReader,
       schema_fields: SchemaFields,
   }
   
   impl DocumentSearcher {
       /// Create new searcher from index path
       pub fn new<P: AsRef<Path>>(index_path: P) -> TantivyResult<Self> {
           let index = Index::open_in_dir(index_path)?;
           let reader = index.reader()?;
           let schema = index.schema();
           let schema_fields = SchemaFields::from_schema(&schema)?;
           
           Ok(DocumentSearcher {
               reader,
               schema_fields,
           })
       }
       
       /// Create searcher from existing index
       pub fn from_index(index: &Index) -> TantivyResult<Self> {
           let reader = index.reader()?;
           let schema = index.schema();
           let schema_fields = SchemaFields::from_schema(&schema)?;
           
           Ok(DocumentSearcher {
               reader,
               schema_fields,
           })
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] DocumentSearcher struct defined with required fields
- [ ] Constructor methods implemented for both path and index
- [ ] Code compiles successfully

## Next Task
task_036c_implement_basic_search_method.md

---

# Micro-Task 036c: Implement Basic Search Method

## Objective
Implement the basic search method for content queries.

## Prerequisites
- Task 036b completed (DocumentSearcher struct created)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core` 
2. Add to DocumentSearcher impl in `src/searcher.rs`:
   ```rust
   /// Search for documents containing the query text
   pub fn search(&self, query_text: &str, limit: usize) -> TantivyResult<Vec<SearchResult>> {
       let searcher = self.reader.searcher();
       
       // Create query parser for content field
       let mut query_parser = QueryParser::for_index(
           searcher.index(),
           vec![self.schema_fields.content]
       );
       
       // Parse the query
       let query = query_parser.parse_query(query_text)?;
       
       // Search with specified limit
       let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;
       
       // Convert results to SearchResult structs
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
3. Test: `cargo check` (will show errors - expected for now)
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Basic search method implemented
- [ ] Query parsing and execution working
- [ ] Results processing outlined (SearchResult struct needed next)

## Next Task
task_036d_create_search_result_struct.md

---

# Micro-Task 036d: Create SearchResult Struct

## Objective
Create SearchResult struct to hold search results with document metadata.

## Prerequisites
- Task 036c completed (basic search method implemented)

## Time Estimate
9 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add to `src/searcher.rs`:
   ```rust
   /// Search result containing document information
   #[derive(Debug, Clone)]
   pub struct SearchResult {
       pub content: String,
       pub title: String,
       pub file_path: Option<String>,
       pub doc_id: String,
       pub extension: Option<String>,
   }
   
   impl SearchResult {
       /// Create SearchResult from Tantivy document
       pub fn from_document(doc: &tantivy::Document, fields: &SchemaFields) -> Option<Self> {
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
           })
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] SearchResult struct defined with all document fields
- [ ] from_document method implemented with proper field extraction
- [ ] Code compiles successfully

## Next Task
task_036e_add_search_tests_and_commit.md

---

# Micro-Task 036e: Add Search Tests and Commit

## Objective
Write tests for searcher functionality and commit the implementation.

## Prerequisites
- Task 036d completed (SearchResult struct created)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add tests to `src/searcher.rs`:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       use crate::indexer::DocumentIndexer;
       use tempfile::TempDir;
       use std::path::PathBuf;
       
       #[test]
       fn test_searcher_creation() -> TantivyResult<()> {
           let temp_dir = TempDir::new().unwrap();
           let index_path = temp_dir.path().join("test_index");
           
           // Create index first
           let mut indexer = DocumentIndexer::new(&index_path)?;
           let test_path = PathBuf::from("test.rs");
           indexer.add_document(
               "fn main() { println!(\"Hello, world!\"); }",
               "test",
               &test_path,
               None,
           )?;
           indexer.commit()?;
           
           // Test searcher creation
           let searcher = DocumentSearcher::new(&index_path)?;
           assert!(searcher.reader.searcher().num_docs() > 0);
           
           Ok(())
       }
       
       #[test]
       fn test_basic_search() -> TantivyResult<()> {
           let temp_dir = TempDir::new().unwrap();
           let index_path = temp_dir.path().join("test_index");
           
           // Create and populate index
           let mut indexer = DocumentIndexer::new(&index_path)?;
           let test_path = PathBuf::from("test.rs");
           indexer.add_document(
               "fn main() { println!(\"Hello, world!\"); }",
               "test",
               &test_path,
               None,
           )?;
           indexer.commit()?;
           
           // Test search
           let searcher = DocumentSearcher::new(&index_path)?;
           let results = searcher.search("main", 10)?;
           
           assert!(!results.is_empty());
           assert!(results[0].content.contains("main"));
           
           Ok(())
       }
   }
   ```
3. Test: `cargo test`
4. Return to root: `cd ..\..`
5. Commit: `git add crates\tantivy-core && git commit -m "Implement document searcher with basic text search and tests"`

## Success Criteria
- [ ] Comprehensive tests for searcher functionality
- [ ] All tests pass successfully
- [ ] Searcher implementation committed to Git

## Next Task
task_037_implement_advanced_search_features.md