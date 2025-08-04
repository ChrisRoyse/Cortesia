# Micro-Task 180: Benchmark Phrase Search

## Objective
Benchmark phrase search performance for multi-word queries and proximity matching.

## Prerequisites
- Task 179 completed (Fuzzy text search benchmarked)

## Time Estimate
8 minutes

## Instructions
1. Create phrase search benchmark `bench_phrase_search.rs`
2. Test exact phrase vs proximity search
3. Measure performance vs phrase length
4. Run: `cargo run --release --bin bench_phrase_search`
5. Commit: `git add src/bin/bench_phrase_search.rs && git commit -m "Benchmark phrase search and proximity matching"`

## Success Criteria
- [ ] Phrase search benchmark created
- [ ] Exact vs proximity compared
- [ ] Phrase length impact measured
- [ ] Results committed

## Next Task
task_181_benchmark_boolean_queries.md