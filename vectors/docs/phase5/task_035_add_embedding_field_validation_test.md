# Task 035: Add Embedding Field Validation Test

## Prerequisites Check
- [ ] Task 034 completed: field type validation test added
- [ ] All schema test functions compile correctly
- [ ] Schema creation and field validation working
- [ ] Run: `cargo test schema_tests` (basic tests should pass)

## Context
Basic field type tests added. Adding embedding field validation test.

## Task Objective
Add single test for embedding field type and dimension validation

## Steps
1. Open src/vector_store.rs in editor
2. Add test function inside schema_tests module:
   ```rust
   #[test]
   fn test_embedding_field_type() {
       let schema = TransactionalVectorStore::create_document_schema();
       let embedding_field = schema.field_with_name("embedding").unwrap();
       
       if let DataType::FixedSizeList(_, size) = embedding_field.data_type() {
           assert_eq!(*size, 384);
       } else {
           panic!("Embedding field should be FixedSizeList");
       }
   }
   ```
3. Save file

## Success Criteria
- [ ] Single test function with #[test]
- [ ] Tests embedding field specifically
- [ ] Validates FixedSizeList type with size 384
- [ ] Simple pattern match and assertion

## Time: 4 minutes

## Next Task
Task 036: Run schema tests