# AI Prompt: Micro Phase 4.7 - Optimization Performance Tests

You are tasked with creating performance benchmarks for the optimization system. Create `benches/task_4_4_optimization_performance.rs` using criterion.

## Your Task
Implement performance benchmarks that measure optimization speed, effectiveness, and overhead for the Task 4.4 optimization system.

## Expected Benchmark Functions
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_hierarchy_reorganization(c: &mut Criterion) {
    // Benchmark reorganizer performance
}

fn bench_tree_balancing(c: &mut Criterion) {
    // Benchmark balancer performance
}

fn bench_dead_branch_pruning(c: &mut Criterion) {
    // Benchmark pruner performance
}

fn bench_incremental_optimization(c: &mut Criterion) {
    // Benchmark incremental optimization overhead
}

criterion_group!(benches, bench_hierarchy_reorganization, bench_tree_balancing, bench_dead_branch_pruning, bench_incremental_optimization);
criterion_main!(benches);
```

## Performance Targets
- [ ] Reorganization completes in < 100ms for 1000-node hierarchy
- [ ] Tree balancing improves resolution speed by > 20%
- [ ] Dead branch pruning reduces memory usage by > 15%
- [ ] Incremental optimization overhead < 1% of normal operations

## File to Create: `benches/task_4_4_optimization_performance.rs`
## When Complete: Respond with "MICRO PHASE 4.7 COMPLETE"