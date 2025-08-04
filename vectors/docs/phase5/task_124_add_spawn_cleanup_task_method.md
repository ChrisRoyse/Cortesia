# Task 124: Add Spawn Cleanup Task Method

## Prerequisites Check
- [ ] Task 123 completed: Spawn health monitor method added
- [ ] Run: `cargo check` (should pass)

## Context
Add cleanup task spawner for AutoRepairSystem maintenance.

## Task Objective
Implement spawn_cleanup_task() method for timed-out operation cleanup.

## Steps
1. Add spawn cleanup task method to AutoRepairSystem:
   ```rust
   impl AutoRepairSystem {
       /// Spawn cleanup task
       fn spawn_cleanup_task(&self) -> tokio::task::JoinHandle<()> {
           let execution_engine = Arc::clone(&self.execution_engine);
           let running = Arc::clone(&self.running);
           
           tokio::spawn(async move {
               while *running.read().await {
                   // Cleanup timed out operations
                   let timed_out_jobs = execution_engine.cleanup_timed_out_operations().await;
                   if !timed_out_jobs.is_empty() {
                       println!("Cleaned up {} timed out repair operations", timed_out_jobs.len());
                   }
                   
                   tokio::time::sleep(Duration::from_secs(30)).await;
               }
           })
       }
   }
   ```

## Success Criteria
- [ ] Cleanup task spawner implemented
- [ ] Timeout operation handling
- [ ] Proper logging included
- [ ] Compiles without errors

## Time: 4 minutes