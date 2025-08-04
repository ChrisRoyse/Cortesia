# Micro-Task 035a: Create Indexer File Structure

## Objective
Create the basic file structure for indexer.rs with imports and module setup.

## Prerequisites
- Task 034h completed (schema tests added and committed)

## Time Estimate  
5 minutes

## Instructions
1. Navigate to tantivy-core crate: `cd crates\tantivy-core`
2. Create basic `src/indexer.rs` structure:
   ```rust
   //! Document indexing functionality
   
   use crate::schema::{create_code_index, SchemaFields};
   use tantivy::{Index, IndexWriter, Document, Result as TantivyResult};
   use std::path::{Path, PathBuf};
   use uuid::Uuid;
   ```
3. Add uuid dependency to `Cargo.toml`:
   ```toml
   [dependencies]
   uuid = { workspace = true }
   ```
4. Test compilation: `cargo check`
5. Return to root: `cd ..\..`

## Success Criteria
- [ ] indexer.rs file created with basic imports
- [ ] UUID dependency added to Cargo.toml
- [ ] File compiles without errors

## Next Task
task_035b_create_document_indexer_struct.md

---

# Micro-Task 035b: Create DocumentIndexer Struct

## Objective
Define the DocumentIndexer struct with constructor method.

## Prerequisites
- Task 035a completed (indexer file structure created)

## Time Estimate
8 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add to `src/indexer.rs`:
   ```rust
   /// Document indexer with Windows path support
   pub struct DocumentIndexer {
       index: Index,
       writer: IndexWriter,
       schema_fields: SchemaFields,
   }
   
   impl DocumentIndexer {
       /// Create new indexer with specified index path
       pub fn new<P: AsRef<Path>>(index_path: P) -> TantivyResult<Self> {
           let index = create_code_index(index_path)?;
           let schema = index.schema();
           let schema_fields = SchemaFields::from_schema(&schema)?;
           
           // Create writer with 50MB heap (suitable for testing)
           let writer = index.writer(50_000_000)?;
           
           Ok(DocumentIndexer {
               index,
               writer,
               schema_fields,
           })
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] DocumentIndexer struct defined with required fields
- [ ] Constructor method implemented with index creation
- [ ] Code compiles successfully

## Next Task
task_035c_implement_add_document_method.md

---

# Micro-Task 035c: Implement Add Document Method

## Objective
Implement the add_document method for adding individual documents to the index.

## Prerequisites
- Task 035b completed (DocumentIndexer struct created)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`  
2. Add to DocumentIndexer impl in `src/indexer.rs`:
   ```rust
   /// Add a document to the index
   pub fn add_document(
       &mut self,
       content: &str,
       title: &str,
       file_path: &Path,
       doc_id: Option<Uuid>,
   ) -> TantivyResult<()> {
       let mut doc = Document::default();
       
       // Add content field
       doc.add_text(self.schema_fields.content, content);
       
       // Add title field
       doc.add_text(self.schema_fields.title, title);
       
       // Add file path (convert to string, handling Windows paths)
       let path_str = file_path.to_string_lossy();
       doc.add_text(self.schema_fields.file_path, &path_str);
       
       // Add document ID
       let id = doc_id.unwrap_or_else(Uuid::new_v4);
       doc.add_text(self.schema_fields.doc_id, &id.to_string());
       
       // Add file extension
       if let Some(ext) = file_path.extension() {
           if let Some(ext_str) = ext.to_str() {
               doc.add_text(self.schema_fields.extension, ext_str);
           }
       }
       
       self.writer.add_document(doc)?;
       Ok(())
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] add_document method implemented with all fields
- [ ] Windows path handling working correctly
- [ ] Method compiles without errors

## Next Task
task_035d_implement_batch_indexing_method.md

---

# Micro-Task 035d: Implement Batch Indexing Method

## Objective
Implement add_documents_from_paths method for batch indexing from file paths.

## Prerequisites
- Task 035c completed (add_document method implemented)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add to DocumentIndexer impl in `src/indexer.rs`:
   ```rust
   /// Add multiple documents from file paths
   pub fn add_documents_from_paths<P: AsRef<Path>>(
       &mut self,
       file_paths: &[P],
   ) -> TantivyResult<usize> {
       let mut added_count = 0;
       
       for path in file_paths {
           let path_ref = path.as_ref();
           
           // Skip if file doesn't exist
           if !path_ref.exists() {
               continue;
           }
           
           // Read file content (with basic error handling)
           let content = match std::fs::read_to_string(path_ref) {
               Ok(content) => content,
               Err(_) => continue, // Skip files we can't read
           };
           
           // Extract title from filename
           let title = path_ref
               .file_stem()
               .and_then(|s| s.to_str())
               .unwrap_or("untitled");
           
           self.add_document(&content, title, path_ref, None)?;
           added_count += 1;
       }
       
       Ok(added_count)
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Batch indexing method implemented
- [ ] File reading and error handling working
- [ ] Returns count of successfully added documents

## Next Task
task_035e_add_commit_and_access_methods.md

---

# Micro-Task 035e: Add Commit and Access Methods

## Objective
Add commit method and index accessor to complete the DocumentIndexer.

## Prerequisites
- Task 035d completed (batch indexing method implemented)

## Time Estimate
6 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add to DocumentIndexer impl in `src/indexer.rs`:
   ```rust
   /// Commit all pending changes to the index
   pub fn commit(mut self) -> TantivyResult<()> {
       self.writer.commit()?;
       Ok(())
   }
   
   /// Get the underlying index for search operations
   pub fn index(&self) -> &Index {
       &self.index
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Commit method implemented to finalize indexing
- [ ] Index accessor method added for search operations
- [ ] Methods compile successfully

## Next Task
task_035f_create_indexable_document_struct.md

---

# Micro-Task 035f: Create IndexableDocument Struct

## Objective
Create IndexableDocument helper struct for structured document creation.

## Prerequisites
- Task 035e completed (commit and access methods added)

## Time Estimate
9 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add to `src/indexer.rs`:
   ```rust
   /// Structured document for indexing
   #[derive(Debug, Clone)]
   pub struct IndexableDocument {
       pub content: String,
       pub title: String,
       pub file_path: Option<PathBuf>,
       pub doc_id: Uuid,
       pub extension: Option<String>,
   }
   
   impl IndexableDocument {
       pub fn new(content: String, title: String) -> Self {
           Self {
               content,
               title,
               file_path: None,
               doc_id: Uuid::new_v4(),
               extension: None,
           }
       }
       
       pub fn with_path(mut self, path: PathBuf) -> Self {
           self.extension = path.extension()
               .and_then(|ext| ext.to_str())
               .map(|s| s.to_string());
           self.file_path = Some(path);
           self
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] IndexableDocument struct defined with all fields
- [ ] Constructor and builder methods implemented
- [ ] Struct compiles successfully

## Next Task
task_035g_add_helper_functions.md

---

# Micro-Task 035g: Add Helper Functions

## Objective
Add create_text_document helper function for convenient document creation.

## Prerequisites
- Task 035f completed (IndexableDocument struct created)

## Time Estimate
6 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add to `src/indexer.rs`:
   ```rust
   /// Helper function to create a simple text document
   pub fn create_text_document(
       content: &str,
       title: &str,
       file_path: Option<&Path>,
   ) -> IndexableDocument {
       IndexableDocument {
           content: content.to_string(),
           title: title.to_string(),
           file_path: file_path.map(|p| p.to_path_buf()),
           doc_id: Uuid::new_v4(),
           extension: file_path
               .and_then(|p| p.extension())
               .and_then(|ext| ext.to_str())
               .map(|s| s.to_string()),
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Helper function implemented for document creation
- [ ] Function handles optional file paths correctly
- [ ] Code compiles successfully

## Next Task
task_035h_add_indexer_tests.md

---

# Micro-Task 035h: Add Indexer Tests

## Objective
Write comprehensive tests for the indexer functionality.

## Prerequisites
- Task 035g completed (helper functions added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add tests to `src/indexer.rs`:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       use tempfile::TempDir;
       
       #[test]
       fn test_indexer_creation() -> TantivyResult<()> {
           let temp_dir = TempDir::new().unwrap();
           let index_path = temp_dir.path().join("test_index");
           
           let indexer = DocumentIndexer::new(&index_path)?;
           assert!(index_path.exists());
           
           Ok(())
       }
       
       #[test]
       fn test_document_indexing() -> TantivyResult<()> {
           let temp_dir = TempDir::new().unwrap();
           let index_path = temp_dir.path().join("test_index");
           
           let mut indexer = DocumentIndexer::new(&index_path)?;
           
           let test_path = PathBuf::from("test.rs");
           indexer.add_document(
               "fn main() { println!(\"Hello, world!\"); }",
               "test",
               &test_path,
               None,
           )?;
           
           indexer.commit()?;
           Ok(())
       }
       
       #[test]
       fn test_indexable_document() {
           let doc = IndexableDocument::new(
               "test content".to_string(),
               "test title".to_string(),
           );
           
           assert_eq!(doc.content, "test content");
           assert_eq!(doc.title, "test title");
           assert!(doc.file_path.is_none());
       }
   }
   ```
3. Test: `cargo test`
4. Return to root: `cd ..\..`
5. Commit: `git add crates\tantivy-core && git commit -m "Implement document indexer with batch processing and tests"`

## Success Criteria
- [ ] Comprehensive tests for indexer functionality
- [ ] All tests pass successfully
- [ ] Indexer implementation committed to Git

## Next Task
task_036a_create_searcher_file_structure.md