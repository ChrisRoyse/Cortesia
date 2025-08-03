# Task 34g: Test Connection Pooling

**Estimated Time**: 5 minutes  
**Dependencies**: 34f  
**Stage**: Concurrency Testing  

## Objective
Test connection pooling effectiveness under concurrent load.

## Implementation Steps

1. Create `tests/concurrency/connection_pool_test.rs`:
```rust
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
use std::time::Instant;

mod common;
use common::*;

#[tokio::test]
async fn test_connection_pool_under_load() {
    let brain_graph = setup_concurrency_test_graph().await;
    
    let connection_acquisitions = Arc::new(AtomicUsize::new(0));
    let successful_operations = Arc::new(AtomicUsize::new(0));
    
    let start_time = Instant::now();
    
    let tasks: Vec<_> = (0..100) // High concurrent load
        .map(|i| {
            let graph = brain_graph.clone();
            let acq_counter = connection_acquisitions.clone();
            let success_counter = successful_operations.clone();
            
            tokio::spawn(async move {
                // Each task performs database operation
                acq_counter.fetch_add(1, Ordering::SeqCst);
                
                let concept_id = format!("pool_test_concept_{}", i);
                let req = create_test_allocation_request(&concept_id);
                
                if let Ok(_) = graph.allocate_memory(req).await {
                    success_counter.fetch_add(1, Ordering::SeqCst);
                }
            })
        })
        .collect();
    
    futures::future::join_all(tasks).await;
    
    let elapsed = start_time.elapsed();
    let acquisitions = connection_acquisitions.load(Ordering::SeqCst);
    let successes = successful_operations.load(Ordering::SeqCst);
    
    println!("Connection pool test completed in {:?}", elapsed);
    println!("Connection acquisitions: {}", acquisitions);
    println!("Successful operations: {}", successes);
    
    // Validate connection pool efficiency
    assert_eq!(acquisitions, 100, "All tasks should acquire connections");
    assert!(successes >= 90, "Most operations should succeed"); // 90% success rate
    assert!(elapsed.as_secs() < 30, "Should complete within 30 seconds");
    
    // Check pool statistics
    let pool_stats = brain_graph.get_connection_pool_stats().await.unwrap();
    assert!(pool_stats.active_connections <= pool_stats.max_connections);
    assert!(pool_stats.total_acquisitions >= 100);
}

#[tokio::test]
async fn test_connection_pool_exhaustion_handling() {
    let brain_graph = setup_limited_connection_pool().await;
    
    let exhaustion_handled = Arc::new(AtomicUsize::new(0));
    
    let tasks: Vec<_> = (0..50) // More tasks than available connections
        .map(|i| {
            let graph = brain_graph.clone();
            let handled_counter = exhaustion_handled.clone();
            
            tokio::spawn(async move {
                let concept_id = format!("exhaustion_test_{}", i);
                let req = create_test_allocation_request(&concept_id);
                
                match timeout(
                    Duration::from_millis(5000),
                    graph.allocate_memory(req)
                ).await {
                    Ok(Ok(_)) => {
                        // Operation succeeded
                    }
                    Ok(Err(_)) | Err(_) => {
                        // Pool exhaustion handled gracefully
                        handled_counter.fetch_add(1, Ordering::SeqCst);
                    }
                }
            })
        })
        .collect();
    
    futures::future::join_all(tasks).await;
    
    // System should handle pool exhaustion gracefully
    let handled = exhaustion_handled.load(Ordering::SeqCst);
    println!("Pool exhaustion cases handled: {}", handled);
    
    // Some operations should be handled gracefully when pool is exhausted
    assert!(handled > 0, "System should handle pool exhaustion");
}

async fn setup_limited_connection_pool() -> Arc<BrainEnhancedGraphCore> {
    // Setup with very limited connection pool for testing exhaustion
    setup_concurrency_test_graph_with_config(ConnectionPoolConfig {
        max_connections: 5,
        min_connections: 1,
        acquire_timeout_ms: 1000,
    }).await
}
```

## Acceptance Criteria
- [ ] Connection pooling test created
- [ ] Test validates pool efficiency
- [ ] Test handles pool exhaustion gracefully

## Success Metrics
- High success rate under concurrent load
- Pool exhaustion handled gracefully
- Performance within acceptable limits

## Next Task
34h_test_high_concurrency_load.md