# Task 111: Create Repair Module Foundation

## Prerequisites Check
- [ ] Task 110 completed: consistency monitoring and reporting added
- [ ] ConsistencyManager is functional
- [ ] Run: `cargo check` (should pass)

## Context
Create foundation for automatic repair mechanisms that can detect and fix inconsistencies without manual intervention.

## Task Objective
Create the `src/repair.rs` module with basic structures for automatic repair operations.

## Steps
1. Create `src/repair.rs` file with imports:
   ```rust
   use std::collections::HashMap;
   use std::sync::Arc;
   use std::time::{Duration, Instant};
   use tokio::sync::RwLock;
   use serde::{Serialize, Deserialize};
   use uuid::Uuid;
   use crate::consistency::{ConsistencyState, SyncStrategy, RepairOperation, RepairResult};
   ```
2. Add repair trigger enum:
   ```rust
   /// Triggers that initiate automatic repair
   #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
   pub enum RepairTrigger {
       /// Scheduled periodic repair
       Scheduled,
       /// Consistency check detected issue
       ConsistencyCheck,
       /// Search operation failed
       SearchFailure,
       /// Cache miss with system inconsistency
       CacheMiss,
       /// Manual trigger
       Manual,
       /// System health below threshold
       HealthCheck,
   }
   ```
3. Add repair priority levels:
   ```rust
   /// Priority levels for repair operations
   #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
   pub enum RepairPriority {
       /// Critical - immediate repair needed
       Critical = 0,
       /// High - repair within minutes
       High = 1,
       /// Medium - repair within hours
       Medium = 2,
       /// Low - repair when convenient
       Low = 3,
   }
   ```
4. Add repair job structure:
   ```rust
   /// Automatic repair job
   #[derive(Debug, Clone)]
   pub struct RepairJob {
       /// Unique job ID
       pub id: String,
       /// Document ID to repair
       pub doc_id: String,
       /// Repair priority
       pub priority: RepairPriority,
       /// Trigger that initiated repair
       pub trigger: RepairTrigger,
       /// Synchronization strategy
       pub strategy: SyncStrategy,
       /// Job creation timestamp
       pub created_at: Instant,
       /// Job deadline (based on priority)
       pub deadline: Instant,
       /// Number of retry attempts
       pub retry_count: usize,
       /// Maximum retries allowed
       pub max_retries: usize,
   }
   ```
5. Add repair configuration:
   ```rust
   /// Configuration for automatic repair system
   #[derive(Debug, Clone)]
   pub struct AutoRepairConfig {
       /// Enable automatic repair
       pub enabled: bool,
       /// Maximum concurrent repair jobs
       pub max_concurrent_jobs: usize,
       /// Repair job timeout in seconds
       pub job_timeout_seconds: u64,
       /// Retry delay in seconds
       pub retry_delay_seconds: u64,
       /// Health check interval in seconds
       pub health_check_interval: u64,
       /// Critical priority deadline in seconds
       pub critical_deadline_seconds: u64,
       /// High priority deadline in seconds
       pub high_deadline_seconds: u64,
   }
   
   impl Default for AutoRepairConfig {
       fn default() -> Self {
           Self {
               enabled: true,
               max_concurrent_jobs: 5,
               job_timeout_seconds: 30,
               retry_delay_seconds: 5,
               health_check_interval: 60,
               critical_deadline_seconds: 30,
               high_deadline_seconds: 300, // 5 minutes
           }
       }
   }
   ```
6. Verify compilation

## Success Criteria
- [ ] `src/repair.rs` file created with proper imports
- [ ] RepairTrigger enum covers all automatic scenarios
- [ ] RepairPriority with ordered severity levels
- [ ] RepairJob structure with retry and deadline support
- [ ] AutoRepairConfig with reasonable defaults
- [ ] Job tracking with unique IDs and timestamps
- [ ] Compiles without errors

## Time: 4 minutes

## Next Task
Task 112 will implement the automatic repair scheduler.

## Notes
Repair foundation provides structured approach to automatic repair with prioritization and retry mechanisms.