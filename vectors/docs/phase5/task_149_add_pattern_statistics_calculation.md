# Task 149: Add Pattern Statistics Calculation

## Prerequisites Check
- [ ] Task 148 completed: update_query_pattern method added
- [ ] Run: `cargo check` (should pass)

## Context
Add statistical calculations for query patterns including response time and success rate.

## Task Objective
Extend update_query_pattern method to calculate running averages for response time and success rate.

## Steps
1. Add statistics calculation to update_query_pattern method after frequency update:
   ```rust
   // Update statistics
   pattern.avg_response_time = (pattern.avg_response_time * (pattern.frequency - 1) as f64 + response_time) / pattern.frequency as f64;
   pattern.success_rate = (pattern.success_rate * (pattern.frequency - 1) as f64 + if success { 1.0 } else { 0.0 }) / pattern.frequency as f64;
   
   // Determine optimal mode if we have enough data
   if pattern.frequency >= self.config.min_queries_for_optimization {
       pattern.optimal_mode = self.determine_optimal_mode(query_hash).await;
   }
   ```

## Success Criteria
- [ ] Running average calculations implemented for response time and success rate
- [ ] Optimal mode determination trigger added
- [ ] Compiles without errors

## Time: 4 minutes