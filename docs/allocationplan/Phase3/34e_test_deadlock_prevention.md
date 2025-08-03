# Task 34e: Test Deadlock Prevention

**Estimated Time**: 5 minutes  
**Dependencies**: 34d  
**Stage**: Concurrency Testing  

## Objective
Test deadlock prevention mechanisms in complex locking scenarios.

## Implementation Steps

1. Create `tests/concurrency/deadlock_test.rs`:
```rust
use std::time::Duration;
use tokio::time::timeout;

mod common;
use common::*;

#[tokio::test]
async fn test_no_deadlocks_in_complex_operations() {
    let brain_graph = setup_concurrency_test_graph().await;
    
    // Create concepts that will be accessed in different orders
    let concepts = vec!["concept_a", "concept_b", "concept_c"];
    for concept_id in &concepts {
        let req = create_test_allocation_request(concept_id);
        brain_graph.allocate_memory(req).await.unwrap();
    }
    
    let tasks: Vec<_> = (0..10)
        .map(|i| {
            let graph = brain_graph.clone();
            let concepts = concepts.clone();
            
            tokio::spawn(async move {
                // Different threads access concepts in different orders
                let order: Vec<&str> = if i % 2 == 0 {
                    vec![concepts[0], concepts[1], concepts[2]]
                } else {
                    vec![concepts[2], concepts[1], concepts[0]]
                };
                
                for concept_id in order {
                    // Simulate complex operation requiring multiple locks
                    let _ = timeout(
                        Duration::from_millis(1000),
                        perform_complex_operation(&graph, concept_id)
                    ).await;
                }
            })
        })
        .collect();
    
    // All tasks should complete without deadlock
    let results = timeout(
        Duration::from_secs(10),
        futures::future::join_all(tasks)
    ).await;
    
    assert!(results.is_ok(), "Deadlock detected - tasks did not complete");
}

async fn perform_complex_operation(
    graph: &BrainEnhancedGraphCore,
    concept_id: &str
) -> Result<(), Box<dyn std::error::Error>> {
    // Read concept
    let concept = graph.get_concept(concept_id).await?;
    
    // Update concept
    let update_req = create_test_update_request(concept_id, "updated_content");
    graph.update_concept(update_req).await?;
    
    // Access inheritance chain (may require additional locks)
    let _ = graph.resolve_inheritance_chain(concept_id).await?;
    
    Ok(())
}
```

## Acceptance Criteria
- [ ] Deadlock prevention test created
- [ ] Test uses complex multi-resource scenarios
- [ ] All operations complete within timeout

## Success Metrics
- No deadlocks detected
- All tasks complete within 10 seconds
- System remains responsive

## Next Task
34f_test_resource_contention.md