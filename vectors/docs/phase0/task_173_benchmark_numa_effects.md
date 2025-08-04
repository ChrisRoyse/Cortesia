# Micro-Task 173: Benchmark NUMA Effects

## Objective
Analyze NUMA (Non-Uniform Memory Access) effects on multi-threaded performance.

## Prerequisites
- Task 172 completed (Memory bandwidth profiled)

## Time Estimate
7 minutes

## Instructions
1. Create NUMA benchmark `bench_numa_effects.rs`
2. Test local vs remote memory access performance
3. Measure thread affinity impact
4. Run: `cargo run --release --bin bench_numa_effects`
5. Commit: `git add src/bin/bench_numa_effects.rs && git commit -m "Benchmark NUMA effects on performance"`

## Success Criteria
- [ ] NUMA benchmark created
- [ ] Memory locality tested
- [ ] Performance differences measured
- [ ] Results committed

## Next Task
task_174_measure_context_switching.md