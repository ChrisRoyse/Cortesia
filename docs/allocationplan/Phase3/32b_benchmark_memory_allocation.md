# Task 32b: Benchmark Memory Allocation

**Estimated Time**: 5 minutes  
**Dependencies**: 32a  
**Stage**: Performance Benchmarking  

## Objective
Benchmark memory allocation performance across different concept types.

## Implementation Steps

1. Create `tests/benchmarks/memory_allocation_bench.rs`:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tokio::runtime::Runtime;

mod common;
use common::*;

fn benchmark_memory_allocation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        setup_benchmark_brain_graph().await
    });
    
    let config = BenchmarkConfig::default();
    let mut group = c.benchmark_group("memory_allocation");
    group.sample_size(config.sample_size);
    
    // Benchmark semantic concept allocation
    group.bench_function("semantic_concept", |b| {
        let mut counter = 0;
        b.iter(|| {
            let concept_id = format!("semantic_concept_{}", counter);
            counter += 1;
            
            rt.block_on(async {
                let req = create_benchmark_allocation_request(&concept_id);
                black_box(
                    brain_graph.allocate_memory(req).await.unwrap()
                );
            })
        })
    });
    
    // Benchmark episodic concept allocation
    group.bench_function("episodic_concept", |b| {
        let mut counter = 0;
        b.iter(|| {
            let concept_id = format!("episodic_concept_{}", counter);
            counter += 1;
            
            rt.block_on(async {
                let mut req = create_benchmark_allocation_request(&concept_id);
                req.concept_type = ConceptType::Episodic;
                black_box(
                    brain_graph.allocate_memory(req).await.unwrap()
                );
            })
        })
    });
    
    // Benchmark large content allocation
    group.bench_function("large_content", |b| {
        let mut counter = 0;
        let large_content = "x".repeat(10000); // 10KB content
        
        b.iter(|| {
            let concept_id = format!("large_concept_{}", counter);
            counter += 1;
            
            rt.block_on(async {
                let mut req = create_benchmark_allocation_request(&concept_id);
                req.content = large_content.clone();
                black_box(
                    brain_graph.allocate_memory(req).await.unwrap()
                );
            })
        })
    });
    
    group.finish();
}

fn benchmark_cortical_coordination(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        setup_benchmark_brain_graph().await
    });
    
    let mut group = c.benchmark_group("cortical_coordination");
    
    group.bench_function("with_cortical_coordination", |b| {
        let mut counter = 0;
        b.iter(|| {
            let concept_id = format!("cortical_concept_{}", counter);
            counter += 1;
            
            rt.block_on(async {
                let req = create_benchmark_allocation_request(&concept_id);
                black_box(
                    brain_graph.allocate_memory_with_cortical_coordination(req).await.unwrap()
                );
            })
        })
    });
    
    group.finish();
}

criterion_group!(benches, benchmark_memory_allocation, benchmark_cortical_coordination);
criterion_main!(benches);
```

## Acceptance Criteria
- [ ] Memory allocation benchmark created
- [ ] Different concept types benchmarked
- [ ] Cortical coordination performance measured

## Success Metrics
- Allocation performance < 50ms per operation
- Consistent performance across concept types
- Cortical coordination overhead < 20%

## Next Task
32c_benchmark_search_operations.md