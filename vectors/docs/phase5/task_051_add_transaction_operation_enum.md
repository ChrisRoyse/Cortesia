# Task 051: Add Transaction Operation Enum

## Prerequisites Check
- [ ] Task 050 completed: TransactionState enum implemented
- [ ] Transaction infrastructure foundation is in place
- [ ] VectorDocument struct is available for use
- [ ] Run: `cargo check` (should pass with TransactionState)

## Context
Defining types of operations that can be batched in transactions.

## Task Objective
Add TransactionOperation enum with Insert, Update, Delete variants.

## Steps
1. Open src/vector_store.rs
2. Add enum:
   ```rust
   #[derive(Debug, Clone)]
   pub enum TransactionOperation {
       Insert { document: VectorDocument, embedding: Vec<f32> },
       Update { id: String, document: VectorDocument, embedding: Vec<f32> },
       Delete { id: String },
   }
   ```
3. Save file

## Success Criteria
- [ ] Enum with 3 variants added
- [ ] Each variant has proper fields
- [ ] Derives Debug, Clone
- [ ] File compiles

## Time: 4 minutes

## Next Task
Task 052: Add vector transaction struct