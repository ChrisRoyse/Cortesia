# Task 34b: Test Basic Thread Safety

**Estimated Time**: 5 minutes  
**Dependencies**: 34a  
**Stage**: Concurrency Testing  

## Objective
Create a simple test for concurrent memory allocation operations.

## Implementation Steps

1. Create `tests/concurrency/thread_safety_test.rs`:
```rust
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use tokio::task::JoinHandle;

mod common;
use common::*;

#[tokio::test]
async fn test_concurrent_memory_allocation() {
    let brain_graph = setup_concurrency_test_graph().await;
    let concurrent_operations = 50; // Start small
    let success_counter = Arc::new(AtomicUsize::new(0));
    
    let tasks: Vec<JoinHandle<()>> = (0..concurrent_operations)
        .map(|i| {
            let graph = brain_graph.clone();
            let counter = success_counter.clone();
            
            tokio::spawn(async move {
                let request = create_test_allocation_request(&format!("concurrent_{}", i));
                
                if let Ok(_) = graph.allocate_memory(request).await {
                    counter.fetch_add(1, Ordering::SeqCst);
                }
            })
        })
        .collect();
    
    futures::future::join_all(tasks).await;
    
    let successes = success_counter.load(Ordering::SeqCst);
    assert!(successes >= concurrent_operations * 80 / 100); // 80% success rate
}
```

## Acceptance Criteria
- [ ] Thread safety test created
- [ ] Test runs 50 concurrent operations
- [ ] Success rate validation included

## Success Metrics
- Test completes in under 10 seconds
- At least 80% success rate

## Next Task
34c_test_read_write_concurrency.md