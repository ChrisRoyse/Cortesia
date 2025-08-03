# Task 32e: Benchmark Cache Performance

**Estimated Time**: 4 minutes  
**Dependencies**: 32d  
**Stage**: Performance Benchmarking  

## Objective
Benchmark cache effectiveness and hit rates under various access patterns.

## Implementation Steps

1. Create `tests/benchmarks/cache_performance_bench.rs`:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tokio::runtime::Runtime;
use rand::{thread_rng, Rng, seq::SliceRandom};

mod common;
use common::*;

fn benchmark_cache_hit_rates(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        let graph = setup_benchmark_brain_graph().await;
        
        // Pre-populate with concepts
        for i in 0..1000 {
            let req = create_benchmark_allocation_request(&format!("cache_concept_{}", i));
            graph.allocate_memory(req).await.unwrap();
        }
        
        graph
    });
    
    let mut group = c.benchmark_group("cache_performance");
    
    // Benchmark cold cache access (first time access)
    group.bench_function("cold_cache_access", |b| {
        let mut counter = 0;
        b.iter(|| {
            let concept_id = format!("cache_concept_{}", counter % 1000);
            counter += 1;
            
            rt.block_on(async {
                // Clear cache before each access to simulate cold cache
                brain_graph.clear_concept_cache().await.unwrap();
                
                black_box(
                    brain_graph.get_concept(&concept_id).await.unwrap()
                );
            })
        })
    });
    
    // Benchmark warm cache access (repeated access)
    group.bench_function("warm_cache_access", |b| {
        let concept_ids: Vec<String> = (0..100)
            .map(|i| format!("cache_concept_{}", i))
            .collect();
        
        // Pre-warm the cache
        rt.block_on(async {
            for concept_id in &concept_ids {
                brain_graph.get_concept(concept_id).await.unwrap();
            }
        });
        
        let mut rng = thread_rng();
        b.iter(|| {
            let concept_id = concept_ids.choose(&mut rng).unwrap();
            
            rt.block_on(async {
                black_box(
                    brain_graph.get_concept(concept_id).await.unwrap()
                );
            })
        })
    });
    
    // Benchmark cache eviction performance
    group.bench_function("cache_eviction_patterns", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Access many concepts to trigger eviction
                for i in 0..200 {
                    let concept_id = format!("cache_concept_{}", i);
                    black_box(
                        brain_graph.get_concept(&concept_id).await.unwrap()
                    );
                }
            })
        })
    });
    
    group.finish();
}

fn benchmark_inheritance_cache(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        let graph = setup_benchmark_brain_graph().await;
        
        // Setup inheritance hierarchies for caching tests
        setup_inheritance_for_cache_testing(&graph).await;
        
        graph
    });
    
    let mut group = c.benchmark_group("inheritance_cache");
    
    // Benchmark inheritance resolution without cache
    group.bench_function("inheritance_no_cache", |b| {
        b.iter(|| {
            rt.block_on(async {
                brain_graph.clear_inheritance_cache().await.unwrap();
                
                black_box(
                    brain_graph.resolve_inheritance_chain("cache_test_child").await.unwrap()
                );
            })
        })
    });
    
    // Benchmark inheritance resolution with cache
    group.bench_function("inheritance_with_cache", |b| {
        // Pre-warm inheritance cache
        rt.block_on(async {
            brain_graph.resolve_inheritance_chain("cache_test_child").await.unwrap();
        });
        
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    brain_graph.resolve_inheritance_chain("cache_test_child").await.unwrap()
                );
            })
        })
    });
    
    group.finish();
}

fn benchmark_cache_memory_usage(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        setup_benchmark_brain_graph().await
    });
    
    let mut group = c.benchmark_group("cache_memory");
    group.sample_size(10); // Fewer samples for memory-intensive test
    
    // Benchmark memory usage with large cache
    group.bench_function("large_cache_memory_footprint", |b| {
        b.iter(|| {
            rt.block_on(async {
                // Fill cache with many concepts
                for i in 0..5000 {
                    let concept_id = format!("memory_test_concept_{}", i);
                    let req = create_benchmark_allocation_request(&concept_id);
                    brain_graph.allocate_memory(req).await.unwrap();
                    
                    // Access to populate cache
                    brain_graph.get_concept(&concept_id).await.unwrap();
                }
                
                // Measure cache memory usage
                let cache_stats = brain_graph.get_cache_memory_stats().await.unwrap();
                black_box(cache_stats);
            })
        })
    });
    
    group.finish();
}

async fn setup_inheritance_for_cache_testing(graph: &BrainEnhancedGraphCore) {
    // Create a moderate inheritance chain for cache testing
    let concepts = ["cache_test_grandparent", "cache_test_parent", "cache_test_child"];
    
    for concept_id in &concepts {
        let req = create_benchmark_allocation_request(concept_id);
        graph.allocate_memory(req).await.unwrap();
    }
    
    graph.create_inheritance_relationship("cache_test_parent", "cache_test_grandparent").await.unwrap();
    graph.create_inheritance_relationship("cache_test_child", "cache_test_parent").await.unwrap();
}

criterion_group!(benches, benchmark_cache_hit_rates, benchmark_inheritance_cache, benchmark_cache_memory_usage);
criterion_main!(benches);
```

## Acceptance Criteria
- [ ] Cache performance benchmarks created
- [ ] Hit rate vs miss rate performance measured
- [ ] Inheritance cache effectiveness benchmarked

## Success Metrics
- Warm cache access < 5ms
- Cache hit should be 10x faster than miss
- Memory usage growth acceptable

## Next Task
32f_benchmark_scalability.md