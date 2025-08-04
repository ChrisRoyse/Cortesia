# Task 116: Add MetricType Enum and Basic Structures

## Prerequisites Check
- [ ] Task 115 completed: repair system components integrated
- [ ] AutoRepairSystem is functional
- [ ] Run: `cargo check` (should pass)

## Context
Create foundation monitoring module with metric type enumeration.

## Task Objective
Create `src/monitoring.rs` with imports and MetricType enum.

## Steps
1. Create `src/monitoring.rs` file with imports:
   ```rust
   use std::collections::HashMap;
   use std::sync::Arc;
   use std::time::{Duration, Instant};
   use tokio::sync::RwLock;
   use serde::{Serialize, Deserialize};
   use uuid::Uuid;
   ```
2. Add performance metric types:
   ```rust
   /// Types of performance metrics
   #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
   pub enum MetricType {
       /// Query response time
       QueryResponseTime,
       /// Search accuracy score
       SearchAccuracy,
       /// Cache hit ratio
       CacheHitRatio,
       /// System throughput (queries per second)
       Throughput,
       /// Memory usage
       MemoryUsage,
       /// Error rate
       ErrorRate,
       /// Index update time
       IndexUpdateTime,
       /// Consistency check time
       ConsistencyCheckTime,
   }
   ```
3. Verify compilation

## Success Criteria
- [ ] `src/monitoring.rs` file created with proper imports
- [ ] MetricType enum covers all important performance aspects
- [ ] Enum implements required traits (Debug, Clone, Hash, Serialize, Deserialize)
- [ ] Compiles without errors

## Time: 2 minutes

## Next Task
Task 117 will add MetricData structure.

## Notes  
Monitoring foundation starts with comprehensive metric type enumeration for all system components.