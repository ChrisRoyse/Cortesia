# Micro-Task 034a: Create Schema File Structure

## Objective
Create the basic file structure for schema.rs with imports and module setup.

## Prerequisites
- Task 033f completed (dependency validation documented)
- tantivy-core crate exists with tantivy dependency

## Time Estimate
5 minutes

## Instructions
1. Navigate to tantivy-core crate: `cd crates\tantivy-core`
2. Create basic `src/schema.rs` structure:
   ```rust
   //! Tantivy schema definition with special character support
   
   use tantivy::schema::*;
   use tantivy::{Index, Result as TantivyResult};
   use std::path::Path;
   ```
3. Test compilation: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] schema.rs file created with basic imports
- [ ] File compiles without errors
- [ ] Module structure ready for implementation

## Next Task
task_034b_define_schema_builder_struct.md

---

# Micro-Task 034b: Define CodeAwareSchemaBuilder Struct

## Objective
Define the CodeAwareSchemaBuilder struct with constructor.

## Prerequisites
- Task 034a completed (schema file structure created)

## Time Estimate
8 minutes

## Instructions
1. Navigate to tantivy-core crate: `cd crates\tantivy-core`
2. Add to `src/schema.rs`:
   ```rust
   /// Schema builder for code-aware text indexing
   pub struct CodeAwareSchemaBuilder {
       schema_builder: SchemaBuilder,
   }
   
   impl CodeAwareSchemaBuilder {
       pub fn new() -> Self {
           Self {
               schema_builder: Schema::builder(),
           }
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] CodeAwareSchemaBuilder struct defined
- [ ] Constructor method implemented
- [ ] Code compiles successfully

## Next Task
task_034c_add_content_field_to_schema.md

---

# Micro-Task 034c: Add Content Field to Schema

## Objective
Implement the content field with code-aware tokenization in the schema builder.

## Prerequisites
- Task 034b completed (schema builder struct defined)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add build_code_schema method start to `src/schema.rs`:
   ```rust
   /// Build schema optimized for code content with special characters
   pub fn build_code_schema(mut self) -> Schema {
       // Content field with special character tokenization
       let text_options = TextOptions::default()
           .set_indexing_options(
               TextFieldIndexing::default()
                   .set_tokenizer("code_aware")
                   .set_index_option(IndexRecordOption::WithFreqsAndPositions)
           )
           .set_stored();
           
       let _content_field = self.schema_builder.add_text_field("content", text_options);
       
       self.schema_builder.build()
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Content field configuration implemented
- [ ] Code compiles without errors
- [ ] Schema can be built with content field

## Next Task
task_034d_add_title_and_path_fields.md

---

# Micro-Task 034d: Add Title and Path Fields

## Objective
Add title and file_path fields to the schema.

## Prerequisites
- Task 034c completed (content field added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Extend build_code_schema method in `src/schema.rs`:
   ```rust
   // Add these before the final self.schema_builder.build()
   
   // Title field for document titles
   let title_options = TextOptions::default()
       .set_indexing_options(
           TextFieldIndexing::default()
               .set_tokenizer("en_stem")
               .set_index_option(IndexRecordOption::WithFreqsAndPositions)
       )
       .set_stored();
       
   let _title_field = self.schema_builder.add_text_field("title", title_options);
   
   // File path field for exact matching
   let path_options = TextOptions::default()
       .set_indexing_options(
           TextFieldIndexing::default()
               .set_tokenizer("raw")
               .set_index_option(IndexRecordOption::Basic)
       )
       .set_stored();
       
   let _path_field = self.schema_builder.add_text_field("file_path", path_options);
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Title field with stemming tokenizer added
- [ ] File path field with raw tokenizer added
- [ ] Schema builds successfully with all fields

## Next Task
task_034e_add_id_and_extension_fields.md

---

# Micro-Task 034e: Add ID and Extension Fields

## Objective
Add document ID and file extension fields to complete the schema.

## Prerequisites
- Task 034d completed (title and path fields added)

## Time Estimate
8 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add final fields to build_code_schema method in `src/schema.rs`:
   ```rust
   // Add these before the final self.schema_builder.build()
   
   // Document ID for unique identification
   let _id_field = self.schema_builder.add_text_field("doc_id", STORED | FAST);
   
   // File extension for filtering
   let _ext_field = self.schema_builder.add_text_field("extension", STRING | STORED);
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Document ID field added with STORED and FAST flags
- [ ] Extension field added with STRING and STORED flags
- [ ] Complete schema builds successfully

## Next Task
task_034f_add_index_creation_function.md

---

# Micro-Task 034f: Add Index Creation Function

## Objective
Implement create_code_index function with Windows path handling.

## Prerequisites
- Task 034e completed (all schema fields added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add to `src/schema.rs`:
   ```rust
   /// Create index with Windows-compatible path handling
   pub fn create_code_index<P: AsRef<Path>>(index_path: P) -> TantivyResult<Index> {
       let schema = CodeAwareSchemaBuilder::new().build_code_schema();
       
       // Ensure directory exists (Windows-compatible)
       let path = index_path.as_ref();
       if let Some(parent) = path.parent() {
           std::fs::create_dir_all(parent)
               .map_err(|e| tantivy::TantivyError::IoError {
                   io_error: std::sync::Arc::new(e),
                   filepath: Some(parent.to_path_buf()),
               })?;
       }
       
       Index::create_in_dir(path, schema)
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] Index creation function implemented
- [ ] Windows path handling working
- [ ] Function compiles without errors

## Next Task
task_034g_add_schema_fields_struct.md

---

# Micro-Task 034g: Add SchemaFields Struct

## Objective
Create SchemaFields struct for convenient field access.

## Prerequisites
- Task 034f completed (index creation function added)

## Time Estimate
8 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add to `src/schema.rs`:
   ```rust
   /// Get field handles for a schema
   pub struct SchemaFields {
       pub content: Field,
       pub title: Field,
       pub file_path: Field,
       pub doc_id: Field,
       pub extension: Field,
   }
   
   impl SchemaFields {
       pub fn from_schema(schema: &Schema) -> TantivyResult<Self> {
           Ok(SchemaFields {
               content: schema.get_field("content")
                   .ok_or_else(|| tantivy::TantivyError::SchemaError("content field not found".to_string()))?,
               title: schema.get_field("title")
                   .ok_or_else(|| tantivy::TantivyError::SchemaError("title field not found".to_string()))?,
               file_path: schema.get_field("file_path")
                   .ok_or_else(|| tantivy::TantivyError::SchemaError("file_path field not found".to_string()))?,
               doc_id: schema.get_field("doc_id")
                   .ok_or_else(|| tantivy::TantivyError::SchemaError("doc_id field not found".to_string()))?,
               extension: schema.get_field("extension")
                   .ok_or_else(|| tantivy::TantivyError::SchemaError("extension field not found".to_string()))?,
           })
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`

## Success Criteria
- [ ] SchemaFields struct implemented
- [ ] from_schema method works correctly
- [ ] Proper error handling for missing fields

## Next Task
task_034h_add_basic_schema_tests.md

---

# Micro-Task 034h: Add Basic Schema Tests

## Objective
Write basic tests for schema creation and field extraction.

## Prerequisites
- Task 034g completed (SchemaFields struct added)

## Time Estimate
10 minutes

## Instructions
1. Navigate to tantivy-core: `cd crates\tantivy-core`
2. Add tempfile dependency to `Cargo.toml`:
   ```toml
   [dev-dependencies]
   tempfile = "3.8"
   ```
3. Add tests to `src/schema.rs`:
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;
       use tempfile::TempDir;
       
       #[test]
       fn test_schema_creation() {
           let schema = CodeAwareSchemaBuilder::new().build_code_schema();
           
           // Verify all expected fields exist
           assert!(schema.get_field("content").is_some());
           assert!(schema.get_field("title").is_some());
           assert!(schema.get_field("file_path").is_some());
           assert!(schema.get_field("doc_id").is_some());
           assert!(schema.get_field("extension").is_some());
       }
       
       #[test]
       fn test_schema_fields_extraction() {
           let schema = CodeAwareSchemaBuilder::new().build_code_schema();
           let fields = SchemaFields::from_schema(&schema);
           assert!(fields.is_ok());
       }
   }
   ```
4. Test: `cargo test`
5. Return to root: `cd ..\..`
6. Commit: `git add crates\tantivy-core && git commit -m "Implement Tantivy schema with field extraction and tests"`

## Success Criteria
- [ ] Basic schema tests implemented
- [ ] All tests pass successfully  
- [ ] Schema implementation committed to Git

## Next Task
task_035a_create_indexer_file_structure.md