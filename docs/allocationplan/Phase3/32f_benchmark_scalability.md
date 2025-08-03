# Task 32f: Benchmark Scalability

**Estimated Time**: 6 minutes  
**Dependencies**: 32e  
**Stage**: Performance Benchmarking  

## Objective
Test performance with increasing graph sizes and concurrent operations.

## Implementation Steps

1. Create `tests/benchmarks/scalability_bench.rs`:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tokio::runtime::Runtime;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

mod common;
use common::*;

fn benchmark_graph_size_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("graph_size_scaling");
    
    // Test different graph sizes
    let sizes = [1_000, 10_000, 50_000];
    
    for &size in &sizes {
        group.bench_with_input(
            BenchmarkId::new("search_performance", size),
            &size,
            |b, &size| {
                let brain_graph = rt.block_on(async {
                    let graph = setup_benchmark_brain_graph().await;
                    
                    // Pre-populate graph with specified size
                    for i in 0..size {
                        let req = create_benchmark_allocation_request(&format!("scale_concept_{}", i));
                        graph.allocate_memory(req).await.unwrap();
                    }
                    
                    graph
                });
                
                b.iter(|| {
                    rt.block_on(async {
                        let search_req = SearchRequest {
                            query_text: "scale_concept".to_string(),
                            search_type: SearchType::Semantic,
                            similarity_threshold: Some(0.8),
                            limit: Some(10),
                            user_context: UserContext::default(),
                            use_ttfs_encoding: false,
                            cortical_area_filter: None,
                        };
                        
                        black_box(
                            brain_graph.search_memory_with_semantic_similarity(search_req).await.unwrap()
                        );
                    })
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_concurrent_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        setup_benchmark_brain_graph().await
    });
    
    let mut group = c.benchmark_group("concurrent_operations");
    
    // Test different concurrency levels
    let concurrency_levels = [10, 50, 100];
    
    for &concurrency in &concurrency_levels {
        group.bench_with_input(
            BenchmarkId::new("concurrent_allocations", concurrency),
            &concurrency,
            |b, &concurrency| {
                b.iter(|| {
                    rt.block_on(async {
                        let counter = Arc::new(AtomicUsize::new(0));
                        let mut tasks = Vec::new();
                        
                        for _ in 0..concurrency {
                            let graph = brain_graph.clone();
                            let counter = counter.clone();
                            
                            let task = tokio::spawn(async move {
                                let id = counter.fetch_add(1, Ordering::SeqCst);
                                let concept_id = format!("concurrent_bench_{}_{}", concurrency, id);
                                let req = create_benchmark_allocation_request(&concept_id);
                                
                                graph.allocate_memory(req).await.unwrap()
                            });
                            
                            tasks.push(task);
                        }
                        
                        let results: Vec<_> = futures::future::join_all(tasks).await;
                        black_box(results);
                    })
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_memory_usage_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("memory_usage_scaling");
    group.sample_size(10); // Fewer samples for memory-intensive tests
    
    let concept_counts = [1_000, 5_000, 10_000];
    
    for &count in &concept_counts {
        group.bench_with_input(
            BenchmarkId::new("memory_footprint", count),
            &count,
            |b, &count| {
                b.iter(|| {
                    rt.block_on(async {
                        let graph = setup_benchmark_brain_graph().await;
                        
                        // Allocate specified number of concepts
                        for i in 0..count {
                            let req = create_benchmark_allocation_request(&format!("memory_concept_{}", i));
                            graph.allocate_memory(req).await.unwrap();
                        }
                        
                        // Measure memory usage
                        let memory_stats = graph.get_memory_usage_stats().await.unwrap();
                        black_box(memory_stats);
                    })
                })
            }
        );
    }
    
    group.finish();
}

fn benchmark_database_connection_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("database_connection_scaling");
    
    // Test with different numbers of concurrent database operations
    let concurrent_ops = [10, 50, 100];
    
    for &ops in &concurrent_ops {
        group.bench_with_input(
            BenchmarkId::new("concurrent_db_operations", ops),
            &ops,
            |b, &ops| {
                let brain_graph = rt.block_on(async {
                    setup_benchmark_brain_graph().await
                });
                
                b.iter(|| {
                    rt.block_on(async {
                        let mut tasks = Vec::new();
                        
                        for i in 0..ops {
                            let graph = brain_graph.clone();
                            
                            let task = tokio::spawn(async move {
                                // Perform database-intensive operation
                                let concept_id = format!("db_scale_concept_{}", i);
                                let req = create_benchmark_allocation_request(&concept_id);
                                
                                // Allocate and immediately retrieve (forces DB operations)
                                graph.allocate_memory(req).await.unwrap();
                                graph.get_concept(&concept_id).await.unwrap()
                            });
                            
                            tasks.push(task);
                        }
                        
                        let results: Vec<_> = futures::future::join_all(tasks).await;
                        black_box(results);
                    })
                })
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_graph_size_scaling,
    benchmark_concurrent_operations,
    benchmark_memory_usage_scaling,
    benchmark_database_connection_scaling
);
criterion_main!(benches);
```

## Acceptance Criteria
- [ ] Scalability benchmarks created
- [ ] Different graph sizes tested
- [ ] Concurrent operation performance measured
- [ ] Memory usage scaling analyzed

## Success Metrics
- Performance degrades gracefully with size
- Concurrent operations scale reasonably
- Memory usage growth is predictable

## Next Task
32g_benchmark_api_endpoints.md