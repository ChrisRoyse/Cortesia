# Task 34c: Test Read/Write Concurrency

**Estimated Time**: 4 minutes  
**Dependencies**: 34b  
**Stage**: Concurrency Testing  

## Objective
Test concurrent read and write operations on the same concepts.

## Implementation Steps

1. Add to `tests/concurrency/thread_safety_test.rs`:
```rust
#[tokio::test]
async fn test_concurrent_read_write() {
    let brain_graph = setup_concurrency_test_graph().await;
    
    // First create a concept
    let concept_id = "read_write_test_concept";
    let request = create_test_allocation_request(concept_id);
    brain_graph.allocate_memory(request).await.unwrap();
    
    let read_counter = Arc::new(AtomicUsize::new(0));
    let write_counter = Arc::new(AtomicUsize::new(0));
    
    // Spawn readers and writers
    let mut tasks = Vec::new();
    
    // 10 readers
    for _ in 0..10 {
        let graph = brain_graph.clone();
        let counter = read_counter.clone();
        tasks.push(tokio::spawn(async move {
            if let Ok(_) = graph.get_concept(concept_id).await {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        }));
    }
    
    // 5 writers (updating content)
    for i in 0..5 {
        let graph = brain_graph.clone();
        let counter = write_counter.clone();
        tasks.push(tokio::spawn(async move {
            let update_request = create_test_update_request(concept_id, &format!("updated_{}", i));
            if let Ok(_) = graph.update_concept(update_request).await {
                counter.fetch_add(1, Ordering::SeqCst);
            }
        }));
    }
    
    futures::future::join_all(tasks).await;
    
    assert!(read_counter.load(Ordering::SeqCst) >= 8); // Most reads succeed
    assert!(write_counter.load(Ordering::SeqCst) >= 3); // Most writes succeed
}
```

## Acceptance Criteria
- [ ] Read/write concurrency test added
- [ ] Test validates both operations succeed
- [ ] No deadlocks or data corruption

## Success Metrics
- Most read operations succeed
- Most write operations succeed
- No race conditions detected

## Next Task
34d_test_cache_concurrency.md