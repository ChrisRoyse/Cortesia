# Task 33c: Test Inheritance Consistency

**Estimated Time**: 4 minutes  
**Dependencies**: 33b  
**Stage**: Data Integrity Testing  

## Objective
Test that inheritance relationships maintain parent-child consistency.

## Implementation Steps

1. Add to `tests/integrity/referential_integrity_test.rs`:
```rust
#[tokio::test]
async fn test_inheritance_chain_consistency() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create inheritance hierarchy: grandparent -> parent -> child
    let concepts = ["grandparent", "parent", "child"];
    for concept_id in &concepts {
        let req = create_test_concept_request(concept_id);
        brain_graph.allocate_memory(req).await.unwrap();
    }
    
    // Create inheritance relationships
    brain_graph.create_inheritance_relationship("parent", "grandparent").await.unwrap();
    brain_graph.create_inheritance_relationship("child", "parent").await.unwrap();
    
    // Validate inheritance consistency
    let inheritance_report = brain_graph
        .validate_inheritance_consistency()
        .await
        .expect("Failed to validate inheritance");
    
    assert!(inheritance_report.is_consistent, "Inheritance should be consistent");
    assert_eq!(inheritance_report.circular_dependencies.len(), 0);
    assert_eq!(inheritance_report.orphaned_children.len(), 0);
    
    // Test inheritance chain resolution
    let chain = brain_graph
        .resolve_inheritance_chain("child")
        .await
        .expect("Failed to resolve chain");
    
    assert_eq!(chain.concepts.len(), 3);
    assert_eq!(chain.concepts[0], "child");
    assert_eq!(chain.concepts[1], "parent");
    assert_eq!(chain.concepts[2], "grandparent");
}

#[tokio::test]
async fn test_circular_inheritance_detection() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create concepts
    let concepts = ["concept_a", "concept_b", "concept_c"];
    for concept_id in &concepts {
        let req = create_test_concept_request(concept_id);
        brain_graph.allocate_memory(req).await.unwrap();
    }
    
    // Create circular inheritance: A -> B -> C -> A
    brain_graph.create_inheritance_relationship("concept_b", "concept_a").await.unwrap();
    brain_graph.create_inheritance_relationship("concept_c", "concept_b").await.unwrap();
    
    // This should fail or be detected
    let circular_result = brain_graph
        .create_inheritance_relationship("concept_a", "concept_c")
        .await;
    
    // Either the operation fails, or integrity validation detects it
    if circular_result.is_ok() {
        let inheritance_report = brain_graph
            .validate_inheritance_consistency()
            .await
            .unwrap();
        
        assert!(!inheritance_report.is_consistent, "Should detect circular inheritance");
        assert!(inheritance_report.circular_dependencies.len() > 0);
    }
}
```

## Acceptance Criteria
- [ ] Inheritance consistency test added
- [ ] Test validates inheritance chains
- [ ] Test detects circular dependencies

## Success Metrics
- Valid inheritance chains pass validation
- Circular dependencies are detected
- Chain resolution works correctly

## Next Task
33d_test_property_consistency.md