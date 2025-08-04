# Task 028: Add create_document_schema Function

## Prerequisites Check
- [ ] Task 027 completed: cargo check passed for store
- [ ] TransactionalVectorStore compiles without errors
- [ ] Arrow schema imports are available
- [ ] Run: `cargo check` (should pass with store implementation)

## Context
Store implementation verified. Adding Arrow schema definition for document storage.

## Task Objective
Add create_document_schema() method to TransactionalVectorStore impl block

## Steps
1. Open src/vector_store.rs in editor
2. Add this method to existing TransactionalVectorStore impl block:
   ```rust
   /// Create the Arrow schema for document storage
   fn create_document_schema() -> Arc<Schema> {
       Arc::new(Schema::new(vec![
           // Unique document ID
           Field::new("id", DataType::Utf8, false),
           // Source file path
           Field::new("file_path", DataType::Utf8, false),
           // Text content of the chunk
           Field::new("content", DataType::Utf8, false),
           // Chunk index within the file
           Field::new("chunk_index", DataType::Int32, false),
           // Vector embedding (384 dimensions for sentence transformers)
           Field::new(
               "embedding", 
               DataType::FixedSizeList(
                   Arc::new(Field::new("item", DataType::Float32, true)),
                   384 // Standard sentence transformer dimension
               ), 
               false
           ),
       ]))
   }
   ```
3. Save file

## Success Criteria
- [ ] create_document_schema() method added to impl block
- [ ] Returns Arc<Schema> with 5 fields
- [ ] Uses correct DataType for each field
- [ ] Embedding field is FixedSizeList with 384 Float32 elements
- [ ] All fields marked as non-nullable (false)

## Time: 8 minutes

## Next Task
Task 029: Add schema getter methods