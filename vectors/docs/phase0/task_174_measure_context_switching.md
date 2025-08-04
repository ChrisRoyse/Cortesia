# Micro-Task 174: Measure Context Switching

## Objective
Measure context switching overhead impact on high-frequency operations.

## Prerequisites
- Task 173 completed (NUMA effects benchmarked)

## Time Estimate
8 minutes

## Instructions
1. Create context switching measurement `measure_context_switching.rs`
2. Test thread vs async performance
3. Measure switching overhead under load
4. Run: `cargo run --release --bin measure_context_switching`
5. Commit: `git add src/bin/measure_context_switching.rs && git commit -m "Measure context switching overhead impact"`

## Success Criteria
- [ ] Context switching measurement created
- [ ] Thread vs async compared
- [ ] Overhead quantified
- [ ] Results committed

## Next Task
task_175_benchmark_system_calls.md