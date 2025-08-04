# Task 072: Add Error Recovery Strategies

## Prerequisites Check
- [ ] Task 071 completed: error context helpers added
- [ ] VectorStoreError impl block exists
- [ ] Helper methods compile correctly
- [ ] Run: `cargo check` (should pass)

## Context
Define error recovery and retry strategies for transient failures.

## Task Objective
Add error classification and recovery strategy methods.

## Steps
1. Open src/vector_store.rs
2. Add to VectorStoreError impl block:
   ```rust
   pub fn is_retryable(&self) -> bool {
       matches!(self, 
           VectorStoreError::ConnectionError(_) | 
           VectorStoreError::IoError(_)
       )
   }
   
   pub fn is_fatal(&self) -> bool {
       matches!(self,
           VectorStoreError::SchemaValidationError(_) |
           VectorStoreError::ValidationError(_)
       )
   }
   ```
3. Save file

## Success Criteria
- [ ] is_retryable method added
- [ ] is_fatal method added
- [ ] Error classification logic correct
- [ ] File compiles

## Time: 4 minutes

## Next Task
Task 073: Add error handling test module