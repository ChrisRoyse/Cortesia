# Task 33d: Test Property Consistency

**Estimated Time**: 4 minutes  
**Dependencies**: 33c  
**Stage**: Data Integrity Testing  

## Objective
Test property inheritance chain consistency across updates.

## Implementation Steps

1. Create `tests/integrity/property_consistency_test.rs`:
```rust
mod common;
use common::*;

#[tokio::test]
async fn test_property_inheritance_consistency() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create parent with properties
    let parent_req = create_test_concept_request("parent_with_props");
    brain_graph.allocate_memory(parent_req).await.unwrap();
    
    brain_graph.set_concept_property(
        "parent_with_props",
        "color",
        "blue"
    ).await.unwrap();
    
    brain_graph.set_concept_property(
        "parent_with_props",
        "size",
        "large"
    ).await.unwrap();
    
    // Create child that inherits properties
    let child_req = create_test_concept_request("child_inheritor");
    brain_graph.allocate_memory(child_req).await.unwrap();
    
    brain_graph.create_inheritance_relationship(
        "child_inheritor",
        "parent_with_props"
    ).await.unwrap();
    
    // Validate property consistency
    let property_report = brain_graph
        .validate_property_consistency()
        .await
        .expect("Failed to validate properties");
    
    assert!(property_report.is_consistent, "Properties should be consistent");
    assert_eq!(property_report.inconsistent_properties.len(), 0);
    
    // Test inherited property resolution
    let resolved_props = brain_graph
        .resolve_all_properties("child_inheritor")
        .await
        .expect("Failed to resolve properties");
    
    assert_eq!(resolved_props.get("color").unwrap(), "blue");
    assert_eq!(resolved_props.get("size").unwrap(), "large");
}

#[tokio::test]
async fn test_property_override_consistency() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Setup inheritance with property override
    setup_property_hierarchy(&brain_graph).await;
    
    // Child overrides parent property
    brain_graph.set_concept_property(
        "child_override",
        "color",
        "red" // Override parent's "blue"
    ).await.unwrap();
    
    // Validate consistency after override
    let property_report = brain_graph
        .validate_property_consistency()
        .await
        .unwrap();
    
    assert!(property_report.is_consistent, "Override should maintain consistency");
    
    // Test property resolution with override
    let resolved_props = brain_graph
        .resolve_all_properties("child_override")
        .await
        .unwrap();
    
    assert_eq!(resolved_props.get("color").unwrap(), "red"); // Child's override
    assert_eq!(resolved_props.get("size").unwrap(), "large"); // Inherited from parent
}

async fn setup_property_hierarchy(graph: &BrainEnhancedGraphCore) {
    // Create parent
    let parent_req = create_test_concept_request("parent_override");
    graph.allocate_memory(parent_req).await.unwrap();
    
    graph.set_concept_property("parent_override", "color", "blue").await.unwrap();
    graph.set_concept_property("parent_override", "size", "large").await.unwrap();
    
    // Create child
    let child_req = create_test_concept_request("child_override");
    graph.allocate_memory(child_req).await.unwrap();
    
    graph.create_inheritance_relationship("child_override", "parent_override").await.unwrap();
}
```

## Acceptance Criteria
- [ ] Property consistency test created
- [ ] Test validates property inheritance
- [ ] Test validates property overrides

## Success Metrics
- Property inheritance works correctly
- Property overrides maintain consistency
- All properties resolve to correct values

## Next Task
33e_test_cache_consistency.md