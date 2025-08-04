# Task 026: Update Exports for TransactionalVectorStore

## Prerequisites Check
- [ ] Task 025 completed: basic store test implemented
- [ ] Store creation test compiles correctly
- [ ] TransactionalVectorStore is fully implemented
- [ ] Run: `cargo check` (should pass with store test)

## Context
TransactionalVectorStore and test complete. Need to export it from module.

## Task Objective
Update lib.rs exports to include TransactionalVectorStore

## Steps
1. Open src/lib.rs (or src/main.rs)
2. Update the existing use statement to:
   ```rust
   pub mod vector_store;
   pub use vector_store::{VectorDocument, TransactionalVectorStore};
   ```
3. Save file

## Success Criteria
- [ ] pub use statement updated to include both types
- [ ] TransactionalVectorStore accessible outside module
- [ ] Both VectorDocument and TransactionalVectorStore exported
- [ ] File saves successfully

## Time: 2 minutes

## Next Task
Task 027: Run cargo check for store implementation