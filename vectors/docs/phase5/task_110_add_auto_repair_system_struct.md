# Task 117: Add AutoRepairSystem Struct

## Prerequisites Check
- [ ] Task 114 completed: health monitoring and trigger detection implemented
- [ ] Run: `cargo check` (should pass)

## Context
Add the main AutoRepairSystem struct that will coordinate all repair components.

## Task Objective
Define the AutoRepairSystem struct with all required component fields.

## Steps
1. Add AutoRepairSystem struct:
   ```rust
   /// Unified automatic repair system
   pub struct AutoRepairSystem {
       /// Configuration
       config: AutoRepairConfig,
       /// Job scheduler
       scheduler: Arc<RwLock<AutoRepairScheduler>>,
       /// Execution engine
       execution_engine: Arc<RepairExecutionEngine>,
       /// Health monitor
       health_monitor: Arc<HealthMonitor>,
       /// System running state
       running: Arc<RwLock<bool>>,
       /// Background task handles
       task_handles: Arc<RwLock<Vec<tokio::task::JoinHandle<()>>>>,
   }
   ```

## Success Criteria
- [ ] AutoRepairSystem struct added with all fields
- [ ] Proper Arc and RwLock wrapping for shared state
- [ ] Task handle management included
- [ ] Compiles without errors

## Time: 3 minutes