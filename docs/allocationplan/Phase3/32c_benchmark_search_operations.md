# Task 32c: Benchmark Search Operations

**Estimated Time**: 5 minutes  
**Dependencies**: 32b  
**Stage**: Performance Benchmarking  

## Objective
Benchmark search performance for semantic, TTFS, and hierarchical searches.

## Implementation Steps

1. Create `tests/benchmarks/search_performance_bench.rs`:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tokio::runtime::Runtime;

mod common;
use common::*;

fn benchmark_search_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (brain_graph, _) = rt.block_on(async {
        let graph = setup_benchmark_brain_graph().await;
        
        // Pre-populate with searchable concepts
        for i in 0..1000 {
            let req = create_benchmark_allocation_request(&format!("search_concept_{}", i));
            graph.allocate_memory(req).await.unwrap();
        }
        
        (graph, ())
    });
    
    let mut group = c.benchmark_group("search_operations");
    
    // Benchmark semantic search
    group.bench_function("semantic_search", |b| {
        b.iter(|| {
            rt.block_on(async {
                let search_req = SearchRequest {
                    query_text: "benchmark search query".to_string(),
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
    });
    
    // Benchmark TTFS search
    group.bench_function("ttfs_search", |b| {
        b.iter(|| {
            rt.block_on(async {
                let search_req = SearchRequest {
                    query_text: "ttfs benchmark query".to_string(),
                    search_type: SearchType::TTFS,
                    similarity_threshold: Some(0.7),
                    limit: Some(10),
                    user_context: UserContext::default(),
                    use_ttfs_encoding: true,
                    cortical_area_filter: None,
                };
                
                black_box(
                    brain_graph.search_memory_with_ttfs_encoding(search_req).await.unwrap()
                );
            })
        })
    });
    
    // Benchmark different result set sizes
    group.bench_function("large_result_set", |b| {
        b.iter(|| {
            rt.block_on(async {
                let search_req = SearchRequest {
                    query_text: "concept".to_string(), // Matches many results
                    search_type: SearchType::Semantic,
                    similarity_threshold: Some(0.1), // Low threshold for more results
                    limit: Some(100), // Large result set
                    user_context: UserContext::default(),
                    use_ttfs_encoding: false,
                    cortical_area_filter: None,
                };
                
                black_box(
                    brain_graph.search_memory_with_semantic_similarity(search_req).await.unwrap()
                );
            })
        })
    });
    
    group.finish();
}

fn benchmark_spreading_activation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        let graph = setup_benchmark_brain_graph().await;
        
        // Create connected concepts for spreading activation
        setup_connected_concept_network(&graph).await;
        
        graph
    });
    
    let mut group = c.benchmark_group("spreading_activation");
    
    group.bench_function("spread_from_single_concept", |b| {
        b.iter(|| {
            rt.block_on(async {
                let spreading_req = SpreadingActivationRequest {
                    source_concepts: vec!["central_concept".to_string()],
                    activation_threshold: 0.5,
                    max_depth: 3,
                    decay_factor: 0.8,
                };
                
                black_box(
                    brain_graph.perform_spreading_activation(spreading_req).await.unwrap()
                );
            })
        })
    });
    
    group.finish();
}

async fn setup_connected_concept_network(graph: &BrainEnhancedGraphCore) {
    // Create central concept
    let central_req = create_benchmark_allocation_request("central_concept");
    graph.allocate_memory(central_req).await.unwrap();
    
    // Create connected concepts
    for i in 0..20 {
        let concept_id = format!("connected_concept_{}", i);
        let req = create_benchmark_allocation_request(&concept_id);
        graph.allocate_memory(req).await.unwrap();
        
        // Create semantic relationship to central concept
        graph.create_semantic_relationship(
            &concept_id,
            "central_concept",
            0.7 + (i as f32 * 0.01) // Varying relationship strengths
        ).await.unwrap();
    }
}

criterion_group!(benches, benchmark_search_operations, benchmark_spreading_activation);
criterion_main!(benches);
```

## Acceptance Criteria
- [ ] Search operation benchmarks created
- [ ] Semantic and TTFS search benchmarked
- [ ] Spreading activation performance measured

## Success Metrics
- Search performance < 100ms for 10 results
- TTFS search performance competitive with semantic
- Spreading activation completes within reasonable time

## Next Task
32d_benchmark_inheritance_resolution.md