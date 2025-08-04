# Micro-Task 196: Calculate Percentiles

## Objective
Calculate detailed percentile distributions for latency and throughput metrics.

## Prerequisites
- Task 195 completed (Performance distributions analyzed)

## Time Estimate
7 minutes

## Instructions
1. Create percentile calculator `calc_percentiles.rs`
2. Calculate P50, P90, P95, P99, P99.9 percentiles
3. Generate percentile summaries
4. Run: `cargo run --release --bin calc_percentiles`
5. Commit: `git add src/bin/calc_percentiles.rs && git commit -m "Calculate detailed percentile distributions"`

## Success Criteria
- [ ] Percentile calculator created
- [ ] Detailed percentiles calculated
- [ ] Summaries generated
- [ ] Results committed

## Next Task
task_197_regression_analysis.md