# Task 35c: Create Input Validation Test

**Estimated Time**: 3 minutes  
**Dependencies**: 35b  
**Stage**: Production Testing  

## Objective
Create a test for input validation to prevent malformed requests.

## Implementation Steps

1. Add to `tests/production/security_test.rs`:
```rust
#[tokio::test]
async fn test_empty_concept_id_rejected() {
    let app = setup_test_api().await;
    let token = generate_test_token().await;
    
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/api/v1/memory/allocate")
        .header("authorization", &format!("Bearer {}", token))
        .header("content-type", "application/json")
        .body(json!({
            "concept_id": "",
            "content": "Test content"
        }).to_string())
        .unwrap();
    
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}
```

## Acceptance Criteria
- [ ] Input validation test added
- [ ] Test validates empty concept_id rejection
- [ ] Test compiles and runs

## Success Metrics
- Test completes in under 3 seconds
- Correctly identifies invalid input

## Next Task
35d_create_health_check_test.md