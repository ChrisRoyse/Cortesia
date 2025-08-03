# Task 35k: Create Alerting Test

**Estimated Time**: 4 minutes  
**Dependencies**: 35j  
**Stage**: Production Testing  

## Objective
Create a test for alert generation during system issues.

## Implementation Steps

1. Create `tests/production/alerting_test.rs`:
```rust
use std::time::Duration;

#[tokio::test]
async fn test_critical_failure_alert() {
    let brain_graph = setup_test_brain_graph().await;
    let alert_manager = setup_test_alert_manager().await;
    
    // Simulate critical failure
    brain_graph.simulate_critical_failure().await;
    
    // Wait for alert
    let alert_received = tokio::time::timeout(
        Duration::from_secs(5),
        alert_manager.wait_for_alert("critical_failure")
    ).await;
    
    assert!(alert_received.is_ok(),
           "Critical failure alert not received");
    
    let alert = alert_received.unwrap();
    assert_eq!(alert.severity, "critical");
    assert!(alert.message.contains("failure"));
    assert!(alert.timestamp.is_some());
}
```

## Acceptance Criteria
- [ ] Alerting test created
- [ ] Test simulates failure scenario
- [ ] Test validates alert generation

## Success Metrics
- Alert generated within 5 seconds
- Alert contains correct metadata

## Next Task
35l_create_tracing_test.md