# Task 086: Add Rollback Error Recovery Tests

## Prerequisites Check
- [ ] Task 085 completed: rollback timeout tests added
- [ ] VectorStoreError comprehensive types exist
- [ ] Error classification methods work
- [ ] Run: `cargo check` (should pass)

## Context
Test error scenarios that trigger rollback and recovery mechanisms.

## Task Objective
Add tests for error-induced rollback scenarios and error handling.

## Steps
1. Open src/vector_store.rs
2. Add test in rollback_scenario_tests module:
   ```rust
   #[test]
   fn test_rollback_error_recovery() {
       let mut tx = VectorTransaction {
           id: "error_rollback_tx".to_string(),
           state: TransactionState::Active,
           operations: vec![
               TransactionOperation::Insert,
               TransactionOperation::Update,
           ],
           created_at: std::time::SystemTime::now(),
           timeout: Duration::from_secs(30),
       };
       
       // Simulate error conditions
       let fatal_error = VectorStoreError::ValidationError("Data corruption".into());
       let retryable_error = VectorStoreError::ConnectionError("Network timeout".into());
       
       // Fatal errors should trigger immediate rollback
       if fatal_error.is_fatal() {
           tx.state = TransactionState::RolledBack;
       }
       
       // Verify rollback occurred
       assert!(!fatal_error.is_retryable());
       assert!(fatal_error.is_fatal());
       assert!(matches!(tx.state, TransactionState::RolledBack));
       
       // Retryable errors should not immediately rollback
       assert!(retryable_error.is_retryable());
       assert!(!retryable_error.is_fatal());
   }
   ```
3. Save file

## Success Criteria
- [ ] Error recovery test added
- [ ] Fatal error handling tested
- [ ] Retryable error distinction verified
- [ ] File compiles

## Time: 5 minutes

## Next Task
Task 087: Add cascade rollback tests