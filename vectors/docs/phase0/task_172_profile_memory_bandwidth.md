# Micro-Task 172: Profile Memory Bandwidth

## Objective
Measure memory bandwidth utilization during large vector operations.

## Prerequisites
- Task 171 completed (Lock contention benchmarked)

## Time Estimate
8 minutes

## Instructions
1. Create memory bandwidth profiler `profile_memory_bandwidth.rs`
2. Test sequential vs random memory access patterns
3. Measure bandwidth saturation points
4. Run: `cargo run --release --bin profile_memory_bandwidth`
5. Commit: `git add src/bin/profile_memory_bandwidth.rs && git commit -m "Profile memory bandwidth utilization patterns"`

## Success Criteria
- [ ] Memory bandwidth profiler created
- [ ] Access patterns tested
- [ ] Bandwidth limits identified
- [ ] Results committed

## Next Task
task_173_benchmark_numa_effects.md