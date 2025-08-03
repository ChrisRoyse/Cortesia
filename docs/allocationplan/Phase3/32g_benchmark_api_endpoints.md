# Task 32g: Benchmark API Endpoints

**Estimated Time**: 5 minutes  
**Dependencies**: 32f  
**Stage**: Performance Benchmarking  

## Objective
Benchmark API endpoint response times under load.

## Implementation Steps

1. Create `tests/benchmarks/api_endpoint_bench.rs`:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tokio::runtime::Runtime;
use axum::http::StatusCode;
use tower::ServiceExt;
use serde_json::json;

mod common;
use common::*;
use llmkg::api::rest_server::{create_rest_router, ApiState};

fn benchmark_api_endpoints(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let app = rt.block_on(async {
        let brain_graph = setup_benchmark_brain_graph().await;
        let api_state = create_api_state(brain_graph).await;
        create_rest_router(api_state)
    });
    
    let mut group = c.benchmark_group("api_endpoints");
    
    // Benchmark memory allocation endpoint
    group.bench_function("post_memory_allocate", |b| {
        let mut counter = 0;
        b.iter(|| {
            let concept_id = format!("api_bench_concept_{}", counter);
            counter += 1;
            
            rt.block_on(async {
                let request = axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/v1/memory/allocate")
                    .header("content-type", "application/json")
                    .body(json!({
                        "concept_id": concept_id,
                        "concept_type": "Semantic",
                        "content": "Benchmark API content",
                        "priority": "Normal",
                        "user_id": "benchmark_user",
                        "request_id": format!("api_req_{}", concept_id)
                    }).to_string())
                    .unwrap();
                
                let response = app.clone().oneshot(request).await.unwrap();
                black_box(response);
            })
        })
    });
    
    // Benchmark search endpoint
    group.bench_function("post_search_semantic", |b| {
        b.iter(|| {
            rt.block_on(async {
                let request = axum::http::Request::builder()
                    .method("POST")
                    .uri("/api/v1/search/semantic")
                    .header("content-type", "application/json")
                    .body(json!({
                        "query_text": "benchmark search query",
                        "similarity_threshold": 0.8,
                        "limit": 10,
                        "use_ttfs_encoding": false
                    }).to_string())
                    .unwrap();
                
                let response = app.clone().oneshot(request).await.unwrap();
                black_box(response);
            })
        })
    });
    
    // Benchmark concept retrieval endpoint
    group.bench_function("get_concept", |b| {
        b.iter(|| {
            rt.block_on(async {
                let request = axum::http::Request::builder()
                    .method("GET")
                    .uri("/api/v1/concepts/benchmark_concept_1")
                    .body(String::new())
                    .unwrap();
                
                let response = app.clone().oneshot(request).await.unwrap();
                black_box(response);
            })
        })
    });
    
    // Benchmark health check endpoint
    group.bench_function("get_health", |b| {
        b.iter(|| {
            rt.block_on(async {
                let request = axum::http::Request::builder()
                    .method("GET")
                    .uri("/api/v1/health")
                    .body(String::new())
                    .unwrap();
                
                let response = app.clone().oneshot(request).await.unwrap();
                black_box(response);
            })
        })
    });
    
    group.finish();
}

fn benchmark_api_under_load(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let app = rt.block_on(async {
        let brain_graph = setup_benchmark_brain_graph().await;
        let api_state = create_api_state(brain_graph).await;
        create_rest_router(api_state)
    });
    
    let mut group = c.benchmark_group("api_load_testing");
    group.sample_size(50); // Fewer samples for load testing
    
    // Benchmark multiple concurrent requests
    group.bench_function("concurrent_health_checks", |b| {
        b.iter(|| {
            rt.block_on(async {
                let mut tasks = Vec::new();
                
                // Send 20 concurrent health check requests
                for _ in 0..20 {
                    let app_clone = app.clone();
                    let task = tokio::spawn(async move {
                        let request = axum::http::Request::builder()
                            .method("GET")
                            .uri("/api/v1/health")
                            .body(String::new())
                            .unwrap();
                        
                        app_clone.oneshot(request).await.unwrap()
                    });
                    tasks.push(task);
                }
                
                let responses: Vec<_> = futures::future::join_all(tasks).await;
                black_box(responses);
            })
        })
    });
    
    // Benchmark mixed endpoint usage
    group.bench_function("mixed_endpoint_load", |b| {
        let mut counter = 0;
        b.iter(|| {
            rt.block_on(async {
                let mut tasks = Vec::new();
                
                // Mix of different operations
                for i in 0..15 {
                    let app_clone = app.clone();
                    let current_counter = counter;
                    counter += 1;
                    
                    let task = tokio::spawn(async move {
                        if i % 3 == 0 {
                            // Memory allocation
                            let request = axum::http::Request::builder()
                                .method("POST")
                                .uri("/api/v1/memory/allocate")
                                .header("content-type", "application/json")
                                .body(json!({
                                    "concept_id": format!("load_test_{}", current_counter),
                                    "concept_type": "Semantic",
                                    "content": "Load test content",
                                    "priority": "Normal",
                                    "user_id": "load_test_user",
                                    "request_id": format!("load_req_{}", current_counter)
                                }).to_string())
                                .unwrap();
                            
                            app_clone.oneshot(request).await.unwrap()
                        } else if i % 3 == 1 {
                            // Search
                            let request = axum::http::Request::builder()
                                .method("POST")
                                .uri("/api/v1/search/semantic")
                                .header("content-type", "application/json")
                                .body(json!({
                                    "query_text": "load test query",
                                    "limit": 5
                                }).to_string())
                                .unwrap();
                            
                            app_clone.oneshot(request).await.unwrap()
                        } else {
                            // Health check
                            let request = axum::http::Request::builder()
                                .method("GET")
                                .uri("/api/v1/health")
                                .body(String::new())
                                .unwrap();
                            
                            app_clone.oneshot(request).await.unwrap()
                        }
                    });
                    tasks.push(task);
                }
                
                let responses: Vec<_> = futures::future::join_all(tasks).await;
                black_box(responses);
            })
        })
    });
    
    group.finish();
}

async fn create_api_state(brain_graph: Arc<BrainEnhancedGraphCore>) -> ApiState {
    ApiState {
        knowledge_graph_service: Arc::new(KnowledgeGraphService::new(brain_graph.clone())),
        allocation_service: Arc::new(MemoryAllocationService::new(brain_graph.clone())),
        retrieval_service: Arc::new(MemoryRetrievalService::new(brain_graph.clone())),
    }
}

criterion_group!(benches, benchmark_api_endpoints, benchmark_api_under_load);
criterion_main!(benches);
```

## Acceptance Criteria
- [ ] API endpoint benchmarks created
- [ ] Individual endpoint performance measured
- [ ] Load testing with concurrent requests

## Success Metrics
- Health check < 10ms response time
- Memory allocation < 200ms response time
- Search operations < 500ms response time
- System handles concurrent load gracefully

## Next Task
32h_create_benchmark_runner.md