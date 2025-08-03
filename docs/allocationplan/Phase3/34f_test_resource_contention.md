# Task 34f: Test Resource Contention

**Estimated Time**: 4 minutes  
**Dependencies**: 34e  
**Stage**: Concurrency Testing  

## Objective
Test graceful handling of resource contention under load.

## Implementation Steps

1. Add to `tests/concurrency/deadlock_test.rs`:
```rust
#[tokio::test]
async fn test_resource_contention_handling() {
    let brain_graph = setup_concurrency_test_graph().await;
    
    // Create single resource that will be heavily contended
    let contested_concept = "heavily_contested_concept";
    let req = create_test_allocation_request(contested_concept);
    brain_graph.allocate_memory(req).await.unwrap();
    
    let successful_operations = Arc::new(AtomicUsize::new(0));
    let timeout_operations = Arc::new(AtomicUsize::new(0));
    
    let tasks: Vec<_> = (0..30) // High contention
        .map(|_| {
            let graph = brain_graph.clone();
            let success_counter = successful_operations.clone();
            let timeout_counter = timeout_operations.clone();
            
            tokio::spawn(async move {
                let operation_result = timeout(
                    Duration::from_millis(2000),
                    update_contested_resource(&graph, contested_concept)
                ).await;
                
                match operation_result {
                    Ok(Ok(_)) => {
                        success_counter.fetch_add(1, Ordering::SeqCst);
                    }
                    Ok(Err(_)) | Err(_) => {
                        timeout_counter.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();
    
    futures::future::join_all(tasks).await;
    
    let successes = successful_operations.load(Ordering::SeqCst);
    let timeouts = timeout_operations.load(Ordering::SeqCst);
    
    println!("Resource contention results: {} successes, {} timeouts", successes, timeouts);
    
    // Should handle contention gracefully - some operations succeed
    assert!(successes > 0, "No operations succeeded under contention");
    assert!(successes + timeouts == 30, "All operations should complete or timeout");
}

async fn update_contested_resource(
    graph: &BrainEnhancedGraphCore,
    concept_id: &str
) -> Result<(), Box<dyn std::error::Error>> {
    // Simulate resource-intensive operation
    let concept = graph.get_concept(concept_id).await?;
    
    tokio::time::sleep(Duration::from_millis(10)).await; // Simulate work
    
    let update_req = create_test_update_request(concept_id, "contended_update");
    graph.update_concept(update_req).await?;
    
    Ok(())
}
```

## Acceptance Criteria
- [ ] Resource contention test added
- [ ] Test validates graceful handling
- [ ] System doesn't crash under high contention

## Success Metrics
- Some operations succeed despite contention
- System remains stable
- No resource leaks or corruption

## Next Task
34g_test_connection_pooling.md