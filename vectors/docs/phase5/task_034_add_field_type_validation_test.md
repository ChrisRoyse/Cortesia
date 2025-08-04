# Task 034: Add Field Type Validation Test

## Context
Schema validation test added. Adding field type checking test.

## Task Objective
Add single test for specific field type validation

## Steps
1. Open src/vector_store.rs in editor
2. Add test function inside schema_tests module:
   ```rust
   #[test]
   fn test_field_types() {
       let schema = TransactionalVectorStore::create_document_schema();
       
       // Check ID field type
       let id_field = schema.field_with_name("id").unwrap();
       assert_eq!(id_field.data_type(), &DataType::Utf8);
       
       // Check chunk index field type
       let chunk_field = schema.field_with_name("chunk_index").unwrap();
       assert_eq!(chunk_field.data_type(), &DataType::Int32);
   }
   ```
3. Save file

## Success Criteria
- [ ] Single test function with #[test]
- [ ] Tests two field types: id and chunk_index
- [ ] Simple data type assertions
- [ ] No complex embedding field checking

## Time: 4 minutes

## Next Task
Task 035: Add embedding field validation test