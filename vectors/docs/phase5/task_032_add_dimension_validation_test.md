# Task 032: Add Dimension Validation Test

## Prerequisites Check
- [ ] Task 031 completed: schema creation test implemented
- [ ] Schema creation test passes successfully
- [ ] get_embedding_dimension function is available
- [ ] Run: `cargo test test_schema_creation` (should pass)

## Context
Schema creation test added. Adding dimension validation test.

## Task Objective
Add single test for embedding dimension validation

## Steps
1. Open src/vector_store.rs in editor
2. Add test function inside schema_tests module:
   ```rust
   #[test]
   fn test_embedding_dimension() {
       let dimension = TransactionalVectorStore::get_embedding_dimension();
       assert_eq!(dimension, 384);
   }
   ```
3. Save file

## Success Criteria
- [ ] Single test function with #[test]
- [ ] Tests get_embedding_dimension() method
- [ ] Verifies dimension equals 384
- [ ] Simple assertion with no complex logic

## Time: 3 minutes

## Next Task
Task 033: Add schema field validation test