# Task 049: Add Embedding Error Type

## Context
Adding specific error type for embedding generation failures.

## Task Objective
Add EmbeddingError variant to VectorStoreError enum.

## Steps
1. Open src/vector_store.rs
2. Find VectorStoreError enum
3. Add: EmbeddingError(String),
4. Save file

## Success Criteria
- [ ] Error variant added
- [ ] File compiles

## Time: 2 minutes

## Next Task
Task 050: Add transaction state enum