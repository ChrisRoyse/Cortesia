# Task 33b: Test Referential Integrity

**Estimated Time**: 5 minutes  
**Dependencies**: 33a  
**Stage**: Data Integrity Testing  

## Objective
Test that all concept references point to valid, existing concepts.

## Implementation Steps

1. Create `tests/integrity/referential_integrity_test.rs`:
```rust
mod common;
use common::*;

#[tokio::test]
async fn test_concept_references_valid() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create parent concept
    let parent_req = create_test_concept_request("parent_concept");
    brain_graph.allocate_memory(parent_req).await.unwrap();
    
    // Create child concept with reference to parent
    let child_req = create_test_concept_request("child_concept");
    brain_graph.allocate_memory(child_req).await.unwrap();
    
    // Create inheritance relationship
    brain_graph.create_inheritance_relationship(
        "child_concept",
        "parent_concept"
    ).await.unwrap();
    
    // Validate referential integrity
    let integrity_report = brain_graph
        .validate_referential_integrity()
        .await
        .expect("Failed to validate referential integrity");
    
    assert!(integrity_report.is_valid, "Referential integrity should be valid");
    assert_eq!(integrity_report.broken_references.len(), 0);
    assert!(integrity_report.orphaned_concepts.is_empty());
    
    // Test concept exists check
    assert!(brain_graph.concept_exists("parent_concept").await.unwrap());
    assert!(brain_graph.concept_exists("child_concept").await.unwrap());
    assert!(!brain_graph.concept_exists("nonexistent_concept").await.unwrap());
}

#[tokio::test]
async fn test_broken_reference_detection() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create concept
    let concept_req = create_test_concept_request("test_concept");
    brain_graph.allocate_memory(concept_req).await.unwrap();
    
    // Simulate broken reference (direct database manipulation)
    brain_graph.create_reference_to_nonexistent(
        "test_concept",
        "nonexistent_target"
    ).await.unwrap();
    
    // Validate integrity - should detect broken reference
    let integrity_report = brain_graph
        .validate_referential_integrity()
        .await
        .expect("Failed to validate integrity");
    
    assert!(!integrity_report.is_valid, "Should detect broken reference");
    assert!(integrity_report.broken_references.len() > 0);
    
    let broken_ref = &integrity_report.broken_references[0];
    assert_eq!(broken_ref.source_concept, "test_concept");
    assert_eq!(broken_ref.target_concept, "nonexistent_target");
}
```

## Acceptance Criteria
- [ ] Referential integrity test created
- [ ] Test validates concept references
- [ ] Test detects broken references

## Success Metrics
- Valid references pass integrity check
- Broken references are detected
- No false positives or negatives

## Next Task
33c_test_inheritance_consistency.md