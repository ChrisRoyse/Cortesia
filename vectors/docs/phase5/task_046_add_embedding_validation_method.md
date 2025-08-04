# Task 046: Add Embedding Validation Method

## Context
Adding validation for generated embeddings.

## Task Objective
Add validate_embedding method to check dimensions and validity.

## Steps
1. Open src/vector_store.rs
2. Add method to TransactionalVectorStore impl:
   ```rust
   fn validate_embedding(&self, embedding: &[f32]) -> VectorStoreResult<()> {
       if embedding.len() != 384 {
           return Err(VectorStoreError::ValidationError("Wrong dimension".to_string()));
       }
       Ok(())
   }
   ```
3. Save file

## Success Criteria
- [ ] Method added with dimension check
- [ ] Returns VectorStoreResult
- [ ] File compiles

## Time: 4 minutes

## Next Task
Task 047: Add mock embedding generation