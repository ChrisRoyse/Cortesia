# Task 33f: Test Temporal Versioning

**Estimated Time**: 5 minutes  
**Dependencies**: 33e  
**Stage**: Data Integrity Testing  

## Objective
Validate temporal versioning maintains accurate historical states.

## Implementation Steps

1. Create `tests/integrity/temporal_integrity_test.rs`:
```rust
mod common;
use common::*;
use chrono::{DateTime, Utc};

#[tokio::test]
async fn test_temporal_version_integrity() {
    let brain_graph = setup_integrity_test_graph().await;
    
    let concept_id = "temporal_test_concept";
    
    // Create initial version
    let initial_req = create_test_concept_request(concept_id);
    brain_graph.allocate_memory(initial_req).await.unwrap();
    
    let initial_timestamp = Utc::now();
    
    // Update concept to create new version
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    
    brain_graph.update_concept_content(
        concept_id,
        "Updated content - version 2"
    ).await.unwrap();
    
    let second_timestamp = Utc::now();
    
    // Update again
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    
    brain_graph.update_concept_content(
        concept_id,
        "Updated content - version 3"
    ).await.unwrap();
    
    // Validate temporal integrity
    let temporal_report = brain_graph
        .validate_temporal_integrity(concept_id)
        .await
        .expect("Failed to validate temporal integrity");
    
    assert!(temporal_report.is_valid, "Temporal versions should be valid");
    assert_eq!(temporal_report.version_count, 3);
    assert!(temporal_report.timestamps_are_ordered);
    assert_eq!(temporal_report.missing_versions.len(), 0);
    
    // Test version retrieval
    let current_version = brain_graph
        .get_concept_at_time(concept_id, None)
        .await
        .unwrap();
    
    assert!(current_version.content.contains("version 3"));
    
    let historical_version = brain_graph
        .get_concept_at_time(concept_id, Some(initial_timestamp))
        .await
        .unwrap();
    
    assert!(historical_version.content.contains("Test content"));
    assert!(!historical_version.content.contains("version"));
}

#[tokio::test]
async fn test_version_chain_consistency() {
    let brain_graph = setup_integrity_test_graph().await;
    
    let concept_id = "version_chain_test";
    
    // Create concept with multiple updates
    let req = create_test_concept_request(concept_id);
    brain_graph.allocate_memory(req).await.unwrap();
    
    let mut update_timestamps = Vec::new();
    
    for i in 1..=5 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        
        brain_graph.update_concept_content(
            concept_id,
            &format!("Version {} content", i)
        ).await.unwrap();
        
        update_timestamps.push(Utc::now());
    }
    
    // Validate version chain consistency
    let chain_report = brain_graph
        .validate_version_chain_consistency(concept_id)
        .await
        .unwrap();
    
    assert!(chain_report.is_consistent, "Version chain should be consistent");
    assert_eq!(chain_report.total_versions, 6); // Initial + 5 updates
    assert!(chain_report.chronological_order);
    
    // Test version navigation
    for (i, timestamp) in update_timestamps.iter().enumerate() {
        let version = brain_graph
            .get_concept_at_time(concept_id, Some(*timestamp))
            .await
            .unwrap();
        
        let expected_content = format!("Version {} content", i + 1);
        assert!(version.content.contains(&expected_content),
               "Version {} should contain correct content", i + 1);
    }
}

#[tokio::test]
async fn test_temporal_branch_integrity() {
    let brain_graph = setup_integrity_test_graph().await;
    
    let concept_id = "branch_test_concept";
    
    // Create concept
    let req = create_test_concept_request(concept_id);
    brain_graph.allocate_memory(req).await.unwrap();
    
    // Create branch
    let branch_id = brain_graph
        .create_temporal_branch(concept_id, "test_branch")
        .await
        .unwrap();
    
    // Make changes in branch
    brain_graph.update_concept_in_branch(
        concept_id,
        &branch_id,
        "Branch-specific content"
    ).await.unwrap();
    
    // Validate branch integrity
    let branch_report = brain_graph
        .validate_branch_integrity(&branch_id)
        .await
        .unwrap();
    
    assert!(branch_report.is_valid, "Branch should be valid");
    assert!(branch_report.divergence_point.is_some());
    assert_eq!(branch_report.conflicts.len(), 0);
    
    // Test branch isolation
    let main_content = brain_graph
        .get_concept(concept_id)
        .await
        .unwrap();
    
    let branch_content = brain_graph
        .get_concept_in_branch(concept_id, &branch_id)
        .await
        .unwrap();
    
    assert!(!main_content.content.contains("Branch-specific"));
    assert!(branch_content.content.contains("Branch-specific"));
}
```

## Acceptance Criteria
- [ ] Temporal versioning test created
- [ ] Test validates version integrity
- [ ] Test validates temporal branches

## Success Metrics
- All versions maintain correct timestamps
- Version chains are consistent
- Branch isolation works properly

## Next Task
33g_test_phase2_synchronization.md