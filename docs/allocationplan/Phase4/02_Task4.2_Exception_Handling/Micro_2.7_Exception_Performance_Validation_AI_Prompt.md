# AI Prompt: Micro Phase 2.7 - Exception Performance Validation

You are tasked with creating comprehensive performance benchmarks for the exception handling system. Your goal is to create `benches/task_4_2_exception_performance.rs` using criterion for precise performance measurement.

## Your Task
Implement detailed performance benchmarks that measure exception detection speed, storage efficiency, application performance, and system overhead for the complete Task 4.2 exception handling system.

## Specific Requirements
1. Create `benches/task_4_2_exception_performance.rs` using criterion
2. Benchmark exception detection performance across various scenarios
3. Measure exception storage and retrieval performance
4. Test exception application overhead during property resolution
5. Benchmark pattern learning and optimization performance
6. Validate memory efficiency and garbage collection

## Expected Benchmark Functions
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_exception_detection(c: &mut Criterion) {
    // Benchmark detection engine performance
}

fn bench_exception_storage(c: &mut Criterion) {
    // Benchmark storage operations (add, get, remove)
}

fn bench_exception_application(c: &mut Criterion) {
    // Benchmark exception application during property resolution
}

fn bench_pattern_learning(c: &mut Criterion) {
    // Benchmark learning system performance
}

fn bench_storage_optimization(c: &mut Criterion) {
    // Benchmark optimization and garbage collection
}

fn bench_concurrent_exception_access(c: &mut Criterion) {
    // Benchmark concurrent access performance
}

criterion_group!(
    benches,
    bench_exception_detection,
    bench_exception_storage,
    bench_exception_application,
    bench_pattern_learning,
    bench_storage_optimization,
    bench_concurrent_exception_access
);

criterion_main!(benches);
```

## Performance Targets
- [ ] Exception detection < 10μs per property
- [ ] Exception storage operations < 1μs
- [ ] Property resolution overhead < 5% with exceptions
- [ ] Pattern learning < 100μs per pattern
- [ ] Optimization reduces memory usage > 20%
- [ ] Concurrent access scales linearly

## Success Criteria
- [ ] All benchmarks meet performance requirements
- [ ] Memory usage is optimized and predictable
- [ ] Performance overhead is minimal for non-exception cases
- [ ] Concurrent performance scales appropriately

## File to Create
Create exactly this file: `benches/task_4_2_exception_performance.rs`

## Dependencies Required
Add to Cargo.toml:
```toml
[[bench]]
name = "task_4_2_exception_performance"
harness = false
```

## When Complete
Respond with "MICRO PHASE 2.7 COMPLETE" and a brief summary of performance benchmarks achieved.