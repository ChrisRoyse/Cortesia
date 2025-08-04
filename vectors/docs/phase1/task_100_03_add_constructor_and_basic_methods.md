# Task 100_03: Add Constructor and Basic Methods

## Prerequisites Check
- [ ] Task 100_02 completed: CleanupTask struct added
- [ ] Run: `cargo check` (should pass)

## Context
Add constructor and basic state checking methods for ShutdownHandler.

## Task Objective
Implement ShutdownHandler::new() and is_shutdown_requested() methods.

## Steps
1. Add constructor and basic methods to ShutdownHandler:
   ```rust
   impl ShutdownHandler {
       pub fn new() -> Self {
           Self {
               shutdown_requested: Arc::new(AtomicBool::new(false)),
               cleanup_tasks: Arc::new(RwLock::new(Vec::new())),
           }
       }
       
       pub fn is_shutdown_requested(&self) -> bool {
           self.shutdown_requested.load(Ordering::Relaxed)
       }
   }
   ```

## Success Criteria
- [ ] Constructor method implemented
- [ ] Shutdown state checking method implemented
- [ ] Compiles without errors

## Time: 3 minutes