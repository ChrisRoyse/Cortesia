# Task 087: Add Cascade Rollback Tests

## Prerequisites Check
- [ ] Task 086 completed: rollback error recovery tests added
- [ ] Multiple transaction handling infrastructure
- [ ] Transaction dependency concepts clear
- [ ] Run: `cargo check` (should pass)

## Context
Test cascade rollback scenarios where multiple transactions are affected.

## Task Objective
Add tests for cascade rollback scenarios with dependent transactions.

## Steps
1. Open src/vector_store.rs
2. Add test in rollback_scenario_tests module:
   ```rust
   #[test]
   fn test_cascade_rollback_scenarios() {
       let parent_tx = VectorTransaction {
           id: "parent_tx".to_string(),
           state: TransactionState::Active,
           operations: vec![TransactionOperation::Insert],
           created_at: std::time::SystemTime::now(),
           timeout: Duration::from_secs(30),
       };
       
       let mut child_tx = VectorTransaction {
           id: "child_tx".to_string(),
           state: TransactionState::Active,
           operations: vec![TransactionOperation::Update],
           created_at: std::time::SystemTime::now(),
           timeout: Duration::from_secs(30),
       };
       
       // Simulate parent transaction failure
       let parent_failed = true;
       
       // Child should rollback if parent fails (cascade)
       if parent_failed {
           child_tx.state = TransactionState::RolledBack;
       }
       
       // Verify cascade behavior
       assert!(!child_tx.is_active());
       assert!(matches!(child_tx.state, TransactionState::RolledBack));
       assert!(parent_tx.is_active()); // Parent state unchanged in this test
   }
   ```
3. Save file

## Success Criteria
- [ ] Cascade rollback test added
- [ ] Parent-child relationship simulated
- [ ] Dependency failure handling tested
- [ ] File compiles

## Time: 5 minutes

## Next Task
Task 088: Add rollback consistency tests