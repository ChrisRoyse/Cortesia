# Task 132: Add QueryPerformanceRecord Struct

## Prerequisites Check
- [ ] Task 131 completed: OptimalSearchMode enum added
- [ ] Run: `cargo check` (should pass)

## Context
Add the QueryPerformanceRecord struct to track individual query execution metrics.

## Task Objective
Define the QueryPerformanceRecord struct to store detailed performance data for each query execution.

## Steps
1. Add QueryPerformanceRecord struct to vector_store.rs:
   ```rust
   /// Query performance record
   #[derive(Debug, Clone)]
   pub struct QueryPerformanceRecord {
       /// Query hash
       pub query_hash: String,
       /// Search mode used
       pub search_mode: String,
       /// Response time in milliseconds
       pub response_time: f64,
       /// Number of results returned
       pub result_count: usize,
       /// Relevance score (if available)
       pub relevance_score: Option<f64>,
       /// Timestamp
       pub timestamp: Instant,
       /// Success/failure
       pub success: bool,
   }
   ```

## Success Criteria
- [ ] QueryPerformanceRecord struct added with all required fields
- [ ] Proper field types including Option for relevance_score
- [ ] Compiles without errors

## Time: 3 minutes