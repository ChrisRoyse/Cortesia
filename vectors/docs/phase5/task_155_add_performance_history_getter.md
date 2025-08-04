# Task 155: Add Performance History Getter

## Prerequisites Check
- [ ] Task 154 completed: query patterns getter added
- [ ] Run: `cargo check` (should pass)

## Context
Add getter method to retrieve performance history with optional limiting for analysis.

## Task Objective
Implement get_performance_history method to provide access to historical performance data.

## Steps
1. Add get_performance_history method to QueryOptimizer impl block:
   ```rust
   /// Get performance history
   pub async fn get_performance_history(&self, limit: Option<usize>) -> Vec<QueryPerformanceRecord> {
       let history = self.performance_history.read().await;
       let limit = limit.unwrap_or(100);
       history.iter()
           .rev()
           .take(limit)
           .cloned()
           .collect()
   }
   ```

## Success Criteria
- [ ] get_performance_history method added with optional limit parameter
- [ ] Returns most recent records first with proper limiting
- [ ] Compiles without errors

## Time: 3 minutes