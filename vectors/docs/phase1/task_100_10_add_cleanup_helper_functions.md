# Task 100_10: Add Cleanup Helper Functions

## Prerequisites Check
- [ ] Task 100_09 completed: Index flush cleanup added
- [ ] Run: `cargo check` (should pass)

## Context
Add helper functions for registering common cleanup tasks (metrics, temp files).

## Task Objective
Implement register_metrics_cleanup and register_temp_cleanup helper functions.

## Steps
1. Add cleanup helper functions:
   ```rust
   // Integration with metrics
   pub async fn register_metrics_cleanup(shutdown_handler: &ShutdownHandler) {
       shutdown_handler.register_cleanup(
           "metrics_export",
           2,
           || {
               info!("Exporting final metrics...");
               
               // Force metrics export
               if let Some(handle) = metrics::try_recorder() {
                   // Metrics are automatically flushed on drop
                   info!("Metrics exported");
               }
               
               Ok(())
           }
       ).await;
   }
   
   // Integration with temp files
   pub async fn register_temp_cleanup(shutdown_handler: &ShutdownHandler, temp_dirs: Vec<PathBuf>) {
       shutdown_handler.register_cleanup(
           "temp_file_cleanup",
           3,
           move || {
               info!("Cleaning up {} temporary directories", temp_dirs.len());
               
               for dir in &temp_dirs {
                   if dir.exists() {
                       match std::fs::remove_dir_all(dir) {
                           Ok(_) => info!("Removed temp dir: {:?}", dir),
                           Err(e) => warn!("Failed to remove temp dir {:?}: {}", dir, e),
                       }
                   }
               }
               
               Ok(())
           }
       ).await;
   }
   ```

## Success Criteria
- [ ] Metrics cleanup helper function added
- [ ] Temp file cleanup helper function added
- [ ] Proper priority assignments
- [ ] Compiles without errors

## Time: 5 minutes