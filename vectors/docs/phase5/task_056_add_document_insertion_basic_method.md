# Task 056: Add Document Insertion Basic Method

## Context
Starting document insertion functionality for vector store.

## Task Objective
Add insert_document_async method skeleton to TransactionalVectorStore.

## Steps
1. Open src/vector_store.rs
2. Add method to TransactionalVectorStore impl:
   ```rust
   pub async fn insert_document_async(&mut self, document: VectorDocument) -> VectorStoreResult<String> {
       let embedding = self.generate_embedding(&document.content).await?;
       // TODO: Implement actual insertion
       Ok(document.id)
   }
   ```
3. Save file

## Success Criteria
- [ ] Async method added
- [ ] Generates embedding for document
- [ ] Returns document ID
- [ ] File compiles

## Time: 4 minutes

## Next Task
Task 057: Add document preparation method