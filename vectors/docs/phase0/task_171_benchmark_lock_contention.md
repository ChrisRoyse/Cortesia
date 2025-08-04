# Micro-Task 171: Benchmark Lock Contention

## Objective
Measure lock contention impact on concurrent search operations performance.

## Prerequisites
- Task 170 completed (Branch prediction measured)

## Time Estimate
9 minutes

## Instructions
1. Create lock contention benchmark `bench_lock_contention.rs`
2. Test Mutex vs RwLock vs AtomicRef performance
3. Measure contention under different thread counts
4. Run: `cargo run --release --bin bench_lock_contention`
5. Commit: `git add src/bin/bench_lock_contention.rs && git commit -m "Benchmark lock contention in concurrent scenarios"`

## Success Criteria
- [ ] Lock contention benchmark created
- [ ] Different lock types compared
- [ ] Contention impact measured
- [ ] Results committed

## Next Task
task_172_profile_memory_bandwidth.md