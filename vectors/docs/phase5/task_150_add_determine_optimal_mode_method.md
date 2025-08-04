# Task 150: Add Determine Optimal Mode Method

## Prerequisites Check
- [ ] Task 149 completed: pattern statistics calculation added
- [ ] Run: `cargo check` (should pass)

## Context
Add method to determine the optimal search mode for a query pattern based on historical performance.

## Task Objective
Implement determine_optimal_mode method to analyze performance across different search modes.

## Steps
1. Add determine_optimal_mode method to QueryOptimizer impl block:
   ```rust
   /// Determine optimal search mode for a query pattern
   async fn determine_optimal_mode(&self, query_hash: &str) -> OptimalSearchMode {
       let history = self.performance_history.read().await;
       
       let mut text_performance = Vec::new();
       let mut vector_performance = Vec::new();
       let mut hybrid_performance = Vec::new();
       
       for record in history.iter() {
           if record.query_hash == query_hash && record.success {
               match record.search_mode.as_str() {
                   "text" => text_performance.push(record.response_time),
                   "vector" => vector_performance.push(record.response_time),
                   "hybrid" => hybrid_performance.push(record.response_time),
                   _ => {}
               }
           }
       }
       
       OptimalSearchMode::Unknown // Will implement comparison logic in next task
   }
   ```

## Success Criteria
- [ ] determine_optimal_mode method added with performance data collection
- [ ] Performance vectors for each search mode populated
- [ ] Compiles without errors

## Time: 6 minutes