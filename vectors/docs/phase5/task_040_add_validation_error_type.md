# Task 040: Add Validation Error Type

## Prerequisites Check
- [ ] Task 039 completed: TableOperation error type exists
- [ ] src/vector_store.rs compiles without errors  
- [ ] VectorStoreError enum is accessible
- [ ] Run: `cargo check` (should pass)

## Context
Completing basic error type variants for vector store.

## Task Objective
Add ValidationError variant to VectorStoreError enum.

## Steps
1. Open src/vector_store.rs
2. Find VectorStoreError enum
3. Add: ValidationError(String),
4. Save file

## Success Criteria
- [ ] Error variant added
- [ ] File compiles

## Time: 2 minutes

## Next Task
Task 041: Add IO error conversion