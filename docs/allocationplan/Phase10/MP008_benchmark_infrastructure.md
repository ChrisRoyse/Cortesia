# MP008: Benchmark Infrastructure

## Task Description
Set up comprehensive benchmarking infrastructure using criterion.rs for accurate performance measurements of graph algorithms.

## Prerequisites
- MP001-MP007 completed
- Understanding of criterion.rs
- Knowledge of statistical benchmarking

## Detailed Steps

1. Add benchmark dependencies to `Cargo.toml`:
   ```toml
   [dev-dependencies]
   criterion = { version = "0.5", features = ["html_reports"] }
   
   [[bench]]
   name = "graph_benchmarks"
   harness = false
   ```

2. Create `benches/graph_benchmarks.rs`:
   ```rust
   use criterion::{black_box, criterion_group, criterion_main, Criterion};
   ```

3. Implement node operation benchmarks:
   - Node insertion timing
   - Node removal timing
   - Node lookup performance
   - Neighbor iteration speed

4. Implement edge operation benchmarks:
   - Edge insertion timing
   - Edge removal timing
   - Edge weight updates
   - Edge traversal speed

5. Add algorithm benchmarks:
   - Pathfinding performance (various sizes)
   - Centrality calculations
   - Clustering coefficient computation
   - Community detection timing

6. Create scaling benchmarks:
   ```rust
   fn bench_scaling(c: &mut Criterion) {
       let mut group = c.benchmark_group("graph_scaling");
       
       for size in [100, 1000, 10000, 100000] {
           group.bench_with_input(
               BenchmarkId::new("dijkstra", size),
               &size,
               |b, &size| {
                   let graph = generate_graph(size);
                   b.iter(|| {
                       dijkstra(&graph, 0, size - 1)
                   });
               },
           );
       }
   }
   ```

## Expected Output
```rust
// benches/graph_benchmarks.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use llmkg::neuromorphic::graph::*;

fn bench_node_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_operations");
    
    group.bench_function("insert_node", |b| {
        let mut graph = NeuromorphicGraph::new();
        let mut id = 0;
        b.iter(|| {
            graph.add_node(black_box(NeuromorphicNode::new(id)));
            id += 1;
        });
    });
    
    group.bench_function("lookup_node", |b| {
        let graph = generate_graph(10000);
        b.iter(|| {
            graph.get_node(black_box(5000))
        });
    });
}

fn bench_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("algorithms");
    
    let graph = generate_scale_free_graph(1000, 3);
    
    group.bench_function("betweenness_centrality", |b| {
        b.iter(|| {
            betweenness_centrality(&graph)
        });
    });
}

criterion_group!(benches, bench_node_operations, bench_algorithms);
criterion_main!(benches);
```

## Verification Steps
1. Run benchmarks with `cargo bench`
2. Check HTML reports in `target/criterion/`
3. Verify benchmark stability (low variance)
4. Compare results across different graph sizes

## Time Estimate
25 minutes

## Dependencies
- MP001-MP007: Complete graph system with test utilities