# Task 020: Define TransactionalVectorStore Struct

## Prerequisites Check
- [ ] Task 019 completed: VectorDocument cargo check passed
- [ ] src/vector_store.rs exists and compiles
- [ ] VectorDocument struct is implemented
- [ ] Run: `cargo check` (should pass without errors)

## Context
VectorDocument verified working. Now creating the main store struct for database operations.

## Task Objective
Add TransactionalVectorStore struct with connection and db_path fields

## Steps
1. Open src/vector_store.rs in editor
2. Add struct definition after VectorDocument impl blocks:
   ```rust
   pub struct TransactionalVectorStore {
       /// LanceDB connection handle
       connection: Connection,
       /// Database path for reference
       db_path: String,
   }
   ```
3. Save file

## Success Criteria
- [ ] TransactionalVectorStore struct defined
- [ ] Has connection field of type Connection
- [ ] Has db_path field of type String
- [ ] Both fields have documentation comments
- [ ] Struct is public

## Time: 3 minutes

## Next Task
Task 021: Add TransactionalVectorStore new() method