# Task 123: Add Spawn Health Monitor Method

## Prerequisites Check
- [ ] Task 122 completed: Spawn job processor method added
- [ ] Run: `cargo check` (should pass)

## Context
Add health monitoring task spawner for AutoRepairSystem.

## Task Objective
Implement spawn_health_monitor() method for trigger detection and job scheduling.

## Steps
1. Add spawn health monitor method to AutoRepairSystem:
   ```rust
   impl AutoRepairSystem {
       /// Spawn health monitoring task
       fn spawn_health_monitor(&self) -> tokio::task::JoinHandle<()> {
           let health_monitor = Arc::clone(&self.health_monitor);
           let scheduler = Arc::clone(&self.scheduler);
           let running = Arc::clone(&self.running);
           
           tokio::spawn(async move {
               while *running.read().await {
                   if health_monitor.needs_trigger_detection().await {
                       // Detect triggers
                       health_monitor.detect_repair_triggers().await;
                       health_monitor.update_trigger_check_timestamp().await;
                       
                       // Schedule detected repairs
                       let triggers = health_monitor.get_detected_triggers().await;
                       for trigger in triggers {
                           scheduler.write().await.schedule_repair_job(
                               trigger.doc_id,
                               trigger.priority,
                               trigger.trigger,
                               trigger.strategy,
                           ).await;
                       }
                   }
                   
                   tokio::time::sleep(Duration::from_secs(10)).await;
               }
           })
       }
   }
   ```

## Success Criteria
- [ ] Health monitor spawner implemented
- [ ] Trigger detection cycle
- [ ] Job scheduling integration
- [ ] Compiles without errors

## Time: 5 minutes