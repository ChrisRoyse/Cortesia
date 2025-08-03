# Task 32d: Benchmark Inheritance Resolution

**Estimated Time**: 4 minutes  
**Dependencies**: 32c  
**Stage**: Performance Benchmarking  

## Objective
Benchmark inheritance resolution performance with varying chain depths.

## Implementation Steps

1. Create `tests/benchmarks/inheritance_bench.rs`:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tokio::runtime::Runtime;

mod common;
use common::*;

fn benchmark_inheritance_resolution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        let graph = setup_benchmark_brain_graph().await;
        
        // Create inheritance hierarchies of different depths
        setup_inheritance_hierarchies(&graph).await;
        
        graph
    });
    
    let mut group = c.benchmark_group("inheritance_resolution");
    
    // Benchmark shallow inheritance (depth 3)
    group.bench_function("shallow_inheritance_depth_3", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    brain_graph.resolve_inheritance_chain("child_depth_3").await.unwrap()
                );
            })
        })
    });
    
    // Benchmark medium inheritance (depth 7)
    group.bench_function("medium_inheritance_depth_7", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    brain_graph.resolve_inheritance_chain("child_depth_7").await.unwrap()
                );
            })
        })
    });
    
    // Benchmark deep inheritance (depth 15)
    group.bench_function("deep_inheritance_depth_15", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    brain_graph.resolve_inheritance_chain("child_depth_15").await.unwrap()
                );
            })
        })
    });
    
    group.finish();
}

fn benchmark_property_resolution(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let brain_graph = rt.block_on(async {
        let graph = setup_benchmark_brain_graph().await;
        
        // Setup property inheritance hierarchies
        setup_property_hierarchies(&graph).await;
        
        graph
    });
    
    let mut group = c.benchmark_group("property_resolution");
    
    // Benchmark property resolution with inheritance
    group.bench_function("resolve_inherited_properties", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    brain_graph.resolve_all_properties("property_child").await.unwrap()
                );
            })
        })
    });
    
    // Benchmark property resolution with caching
    group.bench_function("cached_property_resolution", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(
                    brain_graph.resolve_all_properties_cached("property_child").await.unwrap()
                );
            })
        })
    });
    
    group.finish();
}

async fn setup_inheritance_hierarchies(graph: &BrainEnhancedGraphCore) {
    // Create inheritance chains of different depths
    let depths = [3, 7, 15];
    
    for depth in depths {
        // Create root concept
        let root_id = format!("root_depth_{}", depth);
        let root_req = create_benchmark_allocation_request(&root_id);
        graph.allocate_memory(root_req).await.unwrap();
        
        let mut current_parent = root_id;
        
        // Create inheritance chain
        for level in 1..depth {
            let child_id = if level == depth - 1 {
                format!("child_depth_{}", depth) // The leaf we'll benchmark
            } else {
                format!("intermediate_{}_depth_{}", level, depth)
            };
            
            let child_req = create_benchmark_allocation_request(&child_id);
            graph.allocate_memory(child_req).await.unwrap();
            
            graph.create_inheritance_relationship(&child_id, &current_parent).await.unwrap();
            current_parent = child_id;
        }
    }
}

async fn setup_property_hierarchies(graph: &BrainEnhancedGraphCore) {
    // Create parent with multiple properties
    let parent_req = create_benchmark_allocation_request("property_parent");
    graph.allocate_memory(parent_req).await.unwrap();
    
    // Add multiple properties to parent
    for i in 0..20 {
        graph.set_concept_property(
            "property_parent",
            &format!("property_{}", i),
            &format!("value_{}", i)
        ).await.unwrap();
    }
    
    // Create child that inherits properties
    let child_req = create_benchmark_allocation_request("property_child");
    graph.allocate_memory(child_req).await.unwrap();
    
    graph.create_inheritance_relationship("property_child", "property_parent").await.unwrap();
    
    // Add some override properties to child
    for i in 0..5 {
        graph.set_concept_property(
            "property_child",
            &format!("property_{}", i),
            &format!("child_override_{}", i)
        ).await.unwrap();
    }
}

criterion_group!(benches, benchmark_inheritance_resolution, benchmark_property_resolution);
criterion_main!(benches);
```

## Acceptance Criteria
- [ ] Inheritance resolution benchmarks created
- [ ] Different depth hierarchies benchmarked
- [ ] Property resolution performance measured

## Success Metrics
- Shallow inheritance (depth 3) < 10ms
- Deep inheritance (depth 15) < 100ms
- Property resolution performance acceptable

## Next Task
32e_benchmark_cache_performance.md