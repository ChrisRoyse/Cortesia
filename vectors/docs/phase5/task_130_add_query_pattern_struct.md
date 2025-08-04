# Task 130: Add QueryPattern Struct

## Prerequisites Check
- [ ] Task 119 completed: performance optimization recommendations and alerts added
- [ ] All monitoring and alerting systems are functional
- [ ] Run: `cargo check` (should pass)

## Context
Add the QueryPattern struct for query performance analysis and pattern recognition.

## Task Objective
Define the QueryPattern struct to track query frequency, performance metrics, and optimization recommendations.

## Steps
1. Add QueryPattern struct to vector_store.rs:
   ```rust
   /// Query pattern analysis
   #[derive(Debug, Clone)]
   pub struct QueryPattern {
       /// Query text hash
       pub query_hash: String,
       /// Frequency of this pattern
       pub frequency: usize,
       /// Average response time
       pub avg_response_time: f64,
       /// Success rate
       pub success_rate: f64,
       /// Optimal search mode for this pattern
       pub optimal_mode: OptimalSearchMode,
       /// Last seen timestamp
       pub last_seen: Instant,
       /// Performance trend
       pub trend: TrendDirection,
   }
   ```

## Success Criteria
- [ ] QueryPattern struct added with all required fields
- [ ] Proper field types and visibility
- [ ] Compiles without errors

## Time: 3 minutes