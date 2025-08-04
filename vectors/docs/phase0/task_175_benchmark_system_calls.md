# Micro-Task 175: Benchmark System Calls

## Objective
Measure system call overhead for I/O operations and timing functions.

## Prerequisites
- Task 174 completed (Context switching measured)

## Time Estimate
7 minutes

## Instructions
1. Create system call benchmark `bench_system_calls.rs`
2. Test file I/O vs memory operations
3. Measure timing function overhead
4. Run: `cargo run --release --bin bench_system_calls`
5. Commit: `git add src/bin/bench_system_calls.rs && git commit -m "Benchmark system call overhead impact"`

## Success Criteria
- [ ] System call benchmark created
- [ ] I/O overhead measured
- [ ] Timing overhead quantified
- [ ] Results committed

## Next Task
task_176_generate_performance_profile.md