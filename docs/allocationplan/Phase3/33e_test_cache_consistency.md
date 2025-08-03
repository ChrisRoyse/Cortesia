# Task 33e: Test Cache Consistency

**Estimated Time**: 4 minutes  
**Dependencies**: 33d  
**Stage**: Data Integrity Testing  

## Objective
Verify cache synchronization with underlying database state.

## Implementation Steps

1. Create `tests/integrity/cache_consistency_test.rs`:
```rust
mod common;
use common::*;

#[tokio::test]
async fn test_cache_database_synchronization() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create concept and populate cache
    let concept_id = "cache_sync_test";
    let req = create_test_concept_request(concept_id);
    brain_graph.allocate_memory(req).await.unwrap();
    
    // Access concept to populate cache
    let cached_concept = brain_graph
        .get_concept(concept_id)
        .await
        .expect("Failed to get concept");
    
    // Modify concept directly in database (bypass cache)
    brain_graph.update_concept_directly_in_db(
        concept_id,
        "Modified content bypassing cache"
    ).await.unwrap();
    
    // Check cache consistency
    let consistency_report = brain_graph
        .validate_cache_consistency()
        .await
        .expect("Failed to validate cache consistency");
    
    if !consistency_report.is_consistent {
        assert!(consistency_report.inconsistent_entries.len() > 0);
        let inconsistent = &consistency_report.inconsistent_entries[0];
        assert_eq!(inconsistent.concept_id, concept_id);
    }
    
    // Test cache invalidation fixes inconsistency
    brain_graph.invalidate_cache_entry(concept_id).await.unwrap();
    
    let post_invalidation_report = brain_graph
        .validate_cache_consistency()
        .await
        .unwrap();
    
    assert!(post_invalidation_report.is_consistent,
           "Cache should be consistent after invalidation");
}

#[tokio::test]
async fn test_inheritance_cache_consistency() {
    let brain_graph = setup_integrity_test_graph().await;
    
    // Create inheritance hierarchy
    setup_inheritance_hierarchy(&brain_graph).await;
    
    // Access inheritance chain to populate cache
    let chain = brain_graph
        .resolve_inheritance_chain("child_concept")
        .await
        .unwrap();
    
    assert!(chain.was_cached == false); // First access
    
    // Access again - should hit cache
    let cached_chain = brain_graph
        .resolve_inheritance_chain("child_concept")
        .await
        .unwrap();
    
    assert!(cached_chain.was_cached == true);
    
    // Modify inheritance relationship in database
    brain_graph.modify_inheritance_directly_in_db(
        "child_concept",
        "different_parent"
    ).await.unwrap();
    
    // Validate cache consistency
    let cache_report = brain_graph
        .validate_inheritance_cache_consistency()
        .await
        .unwrap();
    
    if !cache_report.is_consistent {
        // Cache should be automatically invalidated
        brain_graph.invalidate_inheritance_cache().await.unwrap();
    }
    
    // Verify cache is now consistent
    let final_report = brain_graph
        .validate_inheritance_cache_consistency()
        .await
        .unwrap();
    
    assert!(final_report.is_consistent);
}

async fn setup_inheritance_hierarchy(graph: &BrainEnhancedGraphCore) {
    // Create parent and child
    let parent_req = create_test_concept_request("parent_concept");
    graph.allocate_memory(parent_req).await.unwrap();
    
    let child_req = create_test_concept_request("child_concept");
    graph.allocate_memory(child_req).await.unwrap();
    
    graph.create_inheritance_relationship("child_concept", "parent_concept").await.unwrap();
}
```

## Acceptance Criteria
- [ ] Cache consistency test created
- [ ] Test validates cache-database sync
- [ ] Test validates inheritance cache

## Success Metrics
- Cache inconsistencies are detected
- Cache invalidation restores consistency
- Performance maintained with consistency checks

## Next Task
33f_test_temporal_versioning.md