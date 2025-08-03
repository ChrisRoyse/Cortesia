# Task 33h: Test Error Recovery

**Estimated Time**: 4 minutes  
**Dependencies**: 33g  
**Stage**: Data Integrity Testing  

## Objective
Test error state handling and data recovery mechanisms.

## Implementation Steps

1. Create `tests/integrity/error_recovery_test.rs`:
```rust
mod common;
use common::*;

#[tokio::test]
async fn test_corruption_detection_and_recovery() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create concept
    let concept_id = "corruption_test_concept";
    let req = create_test_concept_request(concept_id);
    brain_graph.allocate_memory(req).await.unwrap();
    
    // Simulate data corruption
    brain_graph.simulate_concept_corruption(concept_id).await.unwrap();
    
    // Run integrity check - should detect corruption
    let integrity_report = brain_graph
        .run_comprehensive_integrity_check()
        .await
        .unwrap();
    
    assert!(!integrity_report.is_valid, "Should detect corruption");
    assert!(integrity_report.corrupted_concepts.len() > 0);
    
    let corrupted = &integrity_report.corrupted_concepts[0];
    assert_eq!(corrupted.concept_id, concept_id);
    assert!(!corrupted.is_recoverable || corrupted.has_backup);
    
    // Attempt recovery
    let recovery_result = brain_graph
        .recover_corrupted_concept(concept_id)
        .await;
    
    if recovery_result.is_ok() {
        // Verify recovery success
        let post_recovery_report = brain_graph
            .run_comprehensive_integrity_check()
            .await
            .unwrap();
        
        assert!(post_recovery_report.is_valid, "Should be valid after recovery");
        
        // Verify concept is accessible
        let recovered_concept = brain_graph
            .get_concept(concept_id)
            .await
            .expect("Recovered concept should be accessible");
        
        assert!(!recovered_concept.content.is_empty());
    }
}

#[tokio::test]
async fn test_transaction_rollback_integrity() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Start transaction
    let transaction = brain_graph.begin_transaction().await.unwrap();
    
    // Perform operations within transaction
    let concept_ids = vec!["tx_concept_1", "tx_concept_2", "tx_concept_3"];
    
    for concept_id in &concept_ids {
        let req = create_test_concept_request(concept_id);
        transaction.allocate_memory(req).await.unwrap();
    }
    
    // Create relationships
    transaction.create_inheritance_relationship(
        "tx_concept_2", "tx_concept_1"
    ).await.unwrap();
    
    transaction.create_inheritance_relationship(
        "tx_concept_3", "tx_concept_2"
    ).await.unwrap();
    
    // Simulate error and rollback
    transaction.simulate_error().await;
    let rollback_result = transaction.rollback().await;
    
    assert!(rollback_result.is_ok(), "Rollback should succeed");
    
    // Verify all operations were rolled back
    for concept_id in &concept_ids {
        let concept_exists = brain_graph.concept_exists(concept_id).await.unwrap();
        assert!(!concept_exists, "Concept {} should not exist after rollback", concept_id);
    }
    
    // Verify database integrity after rollback
    let integrity_report = brain_graph
        .run_comprehensive_integrity_check()
        .await
        .unwrap();
    
    assert!(integrity_report.is_valid, "Database should be valid after rollback");
}

#[tokio::test]
async fn test_orphan_cleanup() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create concepts with relationships
    let parent_req = create_test_concept_request("orphan_parent");
    brain_graph.allocate_memory(parent_req).await.unwrap();
    
    let child_req = create_test_concept_request("orphan_child");
    brain_graph.allocate_memory(child_req).await.unwrap();
    
    brain_graph.create_inheritance_relationship(
        "orphan_child", "orphan_parent"
    ).await.unwrap();
    
    // Simulate orphan creation by deleting parent directly
    brain_graph.delete_concept_from_db_only("orphan_parent").await.unwrap();
    
    // Run orphan detection
    let orphan_report = brain_graph
        .detect_orphaned_concepts()
        .await
        .unwrap();
    
    assert!(orphan_report.orphans_found > 0, "Should detect orphaned concepts");
    assert!(orphan_report.orphaned_concepts.contains(&"orphan_child".to_string()));
    
    // Run orphan cleanup
    let cleanup_result = brain_graph
        .cleanup_orphaned_concepts()
        .await
        .unwrap();
    
    assert!(cleanup_result.concepts_cleaned > 0);
    assert_eq!(cleanup_result.cleanup_errors.len(), 0);
    
    // Verify orphan is cleaned up
    let post_cleanup_report = brain_graph
        .detect_orphaned_concepts()
        .await
        .unwrap();
    
    assert_eq!(post_cleanup_report.orphans_found, 0);
}
```

## Acceptance Criteria
- [ ] Error recovery test created
- [ ] Test validates corruption detection
- [ ] Test validates transaction rollback
- [ ] Test validates orphan cleanup

## Success Metrics
- Corruption is properly detected
- Recovery mechanisms work correctly
- Transaction integrity maintained
- Orphan cleanup prevents data inconsistency

## Next Task
33i_create_integrity_test_runner.md