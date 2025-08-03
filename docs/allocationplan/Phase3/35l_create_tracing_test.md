# Task 35l: Create Tracing Test

**Estimated Time**: 4 minutes  
**Dependencies**: 35k  
**Stage**: Production Testing  

## Objective
Create a test for distributed tracing functionality.

## Implementation Steps

1. Create `tests/production/tracing_test.rs`:
```rust
#[tokio::test]
async fn test_operation_tracing() {
    let brain_graph = setup_test_brain_graph().await;
    
    let trace_id = "test_trace_001";
    brain_graph.start_trace(trace_id).await;
    
    // Perform traced operation
    let request = create_test_allocation_request("trace_test_concept");
    brain_graph.allocate_memory(request).await.unwrap();
    
    brain_graph.end_trace(trace_id).await;
    
    // Retrieve trace data
    let trace_data = brain_graph
        .get_trace_data(trace_id)
        .await
        .expect("Failed to get trace data");
    
    assert!(!trace_data.spans.is_empty());
    
    let allocation_span = trace_data.spans.iter()
        .find(|s| s.operation_name.contains("allocate"))
        .expect("Allocation span missing");
    
    assert!(allocation_span.duration_ms > 0);
    assert_eq!(allocation_span.trace_id, trace_id);
}
```

## Acceptance Criteria
- [ ] Tracing test created
- [ ] Test validates span generation
- [ ] Test checks trace metadata

## Success Metrics
- Trace data captured correctly
- Span timing information accurate

## Next Task
35m_create_runbook_validation.md