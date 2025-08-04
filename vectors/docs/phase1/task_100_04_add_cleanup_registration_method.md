# Task 100_04: Add Cleanup Registration Method

## Prerequisites Check
- [ ] Task 100_03 completed: Constructor and basic methods added
- [ ] Run: `cargo check` (should pass)

## Context
Add method to register cleanup tasks with priority ordering.

## Task Objective
Implement register_cleanup method with automatic priority sorting.

## Steps
1. Add register_cleanup method to ShutdownHandler:
   ```rust
   impl ShutdownHandler {
       pub async fn register_cleanup<F>(&self, name: &str, priority: u8, cleanup_fn: F)
       where
           F: Fn() -> Result<()> + Send + Sync + 'static,
       {
           let task = CleanupTask {
               name: name.to_string(),
               cleanup_fn: Arc::new(cleanup_fn),
               priority,
           };
           
           let mut tasks = self.cleanup_tasks.write().await;
           tasks.push(task);
           tasks.sort_by_key(|t| t.priority);
           
           info!("Registered cleanup task: {}", name);
       }
   }
   ```

## Success Criteria
- [ ] Register cleanup method implemented
- [ ] Priority-based sorting included
- [ ] Logging added
- [ ] Compiles without errors

## Time: 4 minutes