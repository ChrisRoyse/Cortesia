# Task 34d: Test Cache Concurrency

**Estimated Time**: 4 minutes  
**Dependencies**: 34c  
**Stage**: Concurrency Testing  

## Objective
Test concurrent access to the inheritance resolution cache.

## Implementation Steps

1. Create `tests/concurrency/cache_concurrency_test.rs`:
```rust
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

mod common;
use common::*;

#[tokio::test]
async fn test_concurrent_cache_access() {
    let brain_graph = setup_concurrency_test_graph().await;
    
    // Create test concepts with inheritance
    setup_inheritance_hierarchy(&brain_graph).await;
    
    let cache_hit_counter = Arc::new(AtomicUsize::new(0));
    let cache_miss_counter = Arc::new(AtomicUsize::new(0));
    
    let tasks: Vec<_> = (0..20)
        .map(|i| {
            let graph = brain_graph.clone();
            let hit_counter = cache_hit_counter.clone();
            let miss_counter = cache_miss_counter.clone();
            
            tokio::spawn(async move {
                let concept_id = format!("child_concept_{}", i % 3); // Repeat some concepts
                
                let result = graph
                    .resolve_inheritance_chain(&concept_id)
                    .await;
                
                if let Ok(chain) = result {
                    if chain.was_cached {
                        hit_counter.fetch_add(1, Ordering::SeqCst);
                    } else {
                        miss_counter.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();
    
    futures::future::join_all(tasks).await;
    
    let hits = cache_hit_counter.load(Ordering::SeqCst);
    let misses = cache_miss_counter.load(Ordering::SeqCst);
    
    println!("Cache hits: {}, misses: {}", hits, misses);
    assert!(hits + misses > 15); // Most operations succeed
    assert!(hits > 0); // Some cache hits occur
}

async fn setup_inheritance_hierarchy(graph: &BrainEnhancedGraphCore) {
    // Create parent and child concepts for testing
    let parent_req = create_test_allocation_request("parent_concept");
    graph.allocate_memory(parent_req).await.unwrap();
    
    for i in 0..3 {
        let child_req = create_test_allocation_request(&format!("child_concept_{}", i));
        graph.allocate_memory(child_req).await.unwrap();
        
        graph.create_inheritance_relationship(
            &format!("child_concept_{}", i),
            "parent_concept"
        ).await.unwrap();
    }
}
```

## Acceptance Criteria
- [ ] Cache concurrency test created
- [ ] Test validates cache hit/miss behavior
- [ ] No cache corruption under concurrent access

## Success Metrics
- Cache operations complete successfully
- Cache hits occur for repeated requests
- No data inconsistencies

## Next Task
34e_test_deadlock_prevention.md