# Task 125: Add AutoRepairStatus Struct

## Prerequisites Check
- [ ] Task 124 completed: Spawn cleanup task method added
- [ ] Run: `cargo check` (should pass)

## Context
Add AutoRepairStatus struct for system status reporting.

## Task Objective
Define AutoRepairStatus struct with all status fields.

## Steps
1. Add AutoRepairStatus struct:
   ```rust
   /// Auto repair system status
   #[derive(Debug, Clone)]
   pub struct AutoRepairStatus {
       /// Whether system is running
       pub running: bool,
       /// Number of queued jobs
       pub queued_jobs: usize,
       /// Number of running jobs
       pub running_jobs: usize,
       /// Last health score
       pub last_health_score: Option<f64>,
       /// System uptime
       pub system_uptime: Instant,
   }
   ```

## Success Criteria
- [ ] AutoRepairStatus struct added
- [ ] All status fields included
- [ ] Proper field types
- [ ] Compiles without errors

## Time: 2 minutes