# AI Prompt: Micro Phase 6.3 - Performance Benchmark Suite

You are tasked with creating a comprehensive performance benchmark suite for the complete Phase 4 system. Create `benches/phase_4_comprehensive_benchmarks.rs` using criterion.

## Your Task
Implement comprehensive performance benchmarks that measure the complete Phase 4 inheritance system performance across all components and scenarios.

## Expected Benchmark Functions
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_full_system_hierarchy_operations(c: &mut Criterion) {
    // Benchmark complete hierarchy operations
}

fn bench_exception_system_performance(c: &mut Criterion) {
    // Benchmark exception detection and handling
}

fn bench_compression_system_performance(c: &mut Criterion) {
    // Benchmark property compression
}

fn bench_optimization_system_performance(c: &mut Criterion) {
    // Benchmark hierarchy optimization
}

fn bench_metrics_system_performance(c: &mut Criterion) {
    // Benchmark metrics calculation
}

fn bench_real_world_scenarios(c: &mut Criterion) {
    // Benchmark realistic usage scenarios
}

criterion_group!(benches, bench_full_system_hierarchy_operations, bench_exception_system_performance, bench_compression_system_performance, bench_optimization_system_performance, bench_metrics_system_performance, bench_real_world_scenarios);
criterion_main!(benches);
```

## Performance Targets
- [ ] Complete system operations < 1ms for typical hierarchies
- [ ] Memory usage scales linearly with hierarchy size
- [ ] Compression achieves > 30% memory reduction
- [ ] Optimization improves performance by > 20%
- [ ] System handles 10,000+ node hierarchies efficiently

## File to Create: `benches/phase_4_comprehensive_benchmarks.rs`
## When Complete: Respond with "MICRO PHASE 6.3 COMPLETE"