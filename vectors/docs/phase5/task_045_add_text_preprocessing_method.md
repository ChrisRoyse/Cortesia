# Task 045: Add Text Preprocessing Method

## Context
Creating text preprocessing for embedding generation.

## Task Objective
Add preprocess_text method to TransactionalVectorStore.

## Steps
1. Open src/vector_store.rs
2. In TransactionalVectorStore impl block, add:
   ```rust
   fn preprocess_text(&self, text: &str) -> String {
       text.trim().to_string()
   }
   ```
3. Save file

## Success Criteria
- [ ] Method added to impl block
- [ ] Basic trimming implemented
- [ ] File compiles

## Time: 4 minutes

## Next Task
Task 046: Add embedding validation method