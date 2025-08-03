# Task 35e: Create Metrics Collection Test

**Estimated Time**: 4 minutes  
**Dependencies**: 35d  
**Stage**: Production Testing  

## Objective
Create a test to verify metrics collection functionality.

## Implementation Steps

1. Add to `tests/production/health_check_test.rs`:
```rust
#[tokio::test]
async fn test_metrics_endpoint() {
    let brain_graph = setup_test_brain_graph().await;
    
    let metrics = brain_graph
        .get_basic_metrics()
        .await
        .expect("Failed to get metrics");
    
    assert!(metrics.total_concepts >= 0);
    assert!(metrics.memory_usage_mb > 0);
    assert!(metrics.uptime_seconds >= 0);
    
    // Test metrics are properly formatted
    assert!(metrics.memory_usage_mb < 10000); // Reasonable upper bound
    assert!(metrics.cache_hit_rate >= 0.0);
    assert!(metrics.cache_hit_rate <= 1.0);
}
```

## Acceptance Criteria
- [ ] Metrics collection test added
- [ ] Test validates metric ranges
- [ ] Test checks metric format

## Success Metrics
- Metrics collected in under 50ms
- All metrics within expected ranges

## Next Task
35f_create_error_handling_test.md