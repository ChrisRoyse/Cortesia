# AI Prompt: Micro Phase 3.6 - Compression Performance Tests

You are tasked with creating comprehensive performance benchmarks for the property compression system. Your goal is to create `benches/task_4_3_compression_performance.rs` using criterion for precise measurement.

## Your Task
Implement detailed performance benchmarks that measure compression analysis speed, promotion performance, and overall compression effectiveness for the complete Task 4.3 system.

## Specific Requirements
1. Create `benches/task_4_3_compression_performance.rs` using criterion
2. Benchmark property analysis performance across various hierarchy sizes
3. Measure promotion engine performance and throughput
4. Test iterative compression convergence speed
5. Benchmark validation system performance
6. Measure memory usage before and after compression

## Expected Benchmark Functions
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_property_analysis(c: &mut Criterion) {
    // Benchmark analysis engine performance
}

fn bench_property_promotion(c: &mut Criterion) {
    // Benchmark promotion engine performance
}

fn bench_iterative_compression(c: &mut Criterion) {
    // Benchmark full iterative compression process
}

fn bench_compression_validation(c: &mut Criterion) {
    // Benchmark validation system performance
}

fn bench_compression_memory_usage(c: &mut Criterion) {
    // Benchmark memory efficiency of compression
}

criterion_group!(
    benches,
    bench_property_analysis,
    bench_property_promotion,
    bench_iterative_compression,
    bench_compression_validation,
    bench_compression_memory_usage
);

criterion_main!(benches);
```

## Performance Targets
- [ ] Property analysis < 10ms for 1000-node hierarchy
- [ ] Property promotion < 1ms per promotion
- [ ] Iterative compression converges within 5 iterations
- [ ] Validation completes in < 5ms
- [ ] Compression achieves > 30% memory reduction

## Success Criteria
- [ ] All benchmarks meet performance requirements
- [ ] Compression provides significant memory savings
- [ ] Performance scales appropriately with hierarchy size
- [ ] Validation overhead is minimal

## File to Create
Create exactly this file: `benches/task_4_3_compression_performance.rs`

## Dependencies Required
Add to Cargo.toml:
```toml
[[bench]]
name = "task_4_3_compression_performance"
harness = false
```

## When Complete
Respond with "MICRO PHASE 3.6 COMPLETE" and a brief summary of performance benchmarks achieved.