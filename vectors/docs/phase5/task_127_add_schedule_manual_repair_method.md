# Task 127: Add Schedule Manual Repair Method

## Prerequisites Check
- [ ] Task 126 completed: Get status method added
- [ ] Run: `cargo check` (should pass)

## Context
Add manual repair scheduling method for AutoRepairSystem.

## Task Objective
Implement schedule_manual_repair() method for user-initiated repairs.

## Steps
1. Add schedule manual repair method to AutoRepairSystem:
   ```rust
   impl AutoRepairSystem {
       /// Manually schedule repair
       pub async fn schedule_manual_repair(
           &self,
           doc_id: String,
           priority: RepairPriority,
           strategy: SyncStrategy,
       ) -> String {
           self.scheduler.write().await.schedule_repair_job(
               doc_id,
               priority,
               RepairTrigger::Manual,
               strategy,
           ).await
       }
   }
   ```

## Success Criteria
- [ ] Manual repair scheduling method added
- [ ] Proper parameter types
- [ ] Returns job ID
- [ ] Compiles without errors

## Time: 3 minutes