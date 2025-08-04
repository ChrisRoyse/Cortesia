# Task 057: Add Document Preparation Method

## Context
Adding method to prepare documents for insertion with embeddings.

## Task Objective
Add prepare_document_for_insertion method that validates and enriches documents.

## Steps
1. Open src/vector_store.rs
2. Add method to TransactionalVectorStore impl:
   ```rust
   fn prepare_document_for_insertion(&self, document: &VectorDocument) -> VectorStoreResult<()> {
       if document.content.is_empty() {
           return Err(VectorStoreError::ValidationError("Empty content".to_string()));
       }
       if document.file_path.is_empty() {
           return Err(VectorStoreError::ValidationError("Empty file path".to_string()));
       }
       Ok(())
   }
   ```
3. Save file

## Success Criteria
- [ ] Method validates document fields
- [ ] Returns appropriate errors
- [ ] File compiles

## Time: 4 minutes

## Next Task
Task 058: Add batch insertion preparation