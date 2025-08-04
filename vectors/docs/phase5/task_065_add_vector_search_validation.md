# Task 065: Add Vector Search Query Validation

## Prerequisites Check
- [ ] Task 064 completed: cosine similarity method added
- [ ] VectorStoreError types are defined
- [ ] ValidationError variant exists
- [ ] Run: `cargo check` (should pass)

## Context
Need to validate search query vectors before processing.

## Task Objective
Add validate_search_query method for input validation.

## Steps
1. Open src/vector_store.rs
2. Add validation method in TransactionalVectorStore impl:
   ```rust
   fn validate_search_query(&self, query_vector: &[f32]) -> VectorResult<()> {
       if query_vector.is_empty() {
           return Err(VectorStoreError::ValidationError("Query vector cannot be empty".into()));
       }
       if query_vector.len() != 768 {
           return Err(VectorStoreError::ValidationError("Query vector must have 768 dimensions".into()));
       }
       Ok(())
   }
   ```
3. Save file

## Success Criteria
- [ ] Validation method added
- [ ] Empty vector check implemented
- [ ] Dimension check implemented
- [ ] File compiles

## Time: 4 minutes

## Next Task
Task 066: Implement basic vector search logic