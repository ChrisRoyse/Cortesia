# Task 100_02: Add CleanupTask Struct

## Prerequisites Check
- [ ] Task 100_01 completed: ShutdownHandler struct added
- [ ] Run: `cargo check` (should pass)

## Context
Add the CleanupTask struct for registering cleanup operations with priority ordering.

## Task Objective
Define the CleanupTask struct to hold cleanup functions with priority management.

## Steps
1. Add CleanupTask struct to shutdown.rs:
   ```rust
   #[derive(Clone)]
   pub struct CleanupTask {
       name: String,
       cleanup_fn: Arc<dyn Fn() -> Result<()> + Send + Sync>,
       priority: u8, // Lower number = higher priority
   }
   ```

## Success Criteria
- [ ] CleanupTask struct added with all fields
- [ ] Proper derives and visibility
- [ ] Compiles without errors

## Time: 2 minutes