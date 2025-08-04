# Task 100_09: Add Index Flush Cleanup

## Prerequisites Check
- [ ] Task 100_08 completed: ManagedIndexer methods added
- [ ] Run: `cargo check` (should pass)

## Context
Add Tantivy index flushing cleanup registration to ManagedIndexer constructor.

## Task Objective
Register high-priority cleanup task for Tantivy index flushing in ManagedIndexer::new().

## Steps
1. Update ManagedIndexer::new() to register index flush cleanup:
   ```rust
   impl ManagedIndexer {
       pub async fn new(index_path: &Path) -> Result<Self> {
           let indexer = DocumentIndexer::new(index_path)?;
           let shutdown_handler = Arc::new(ShutdownHandler::new());
           
           // Register cleanup for index writer
           let index_path = index_path.to_path_buf();
           shutdown_handler.register_cleanup(
               "tantivy_index_flush",
               1, // High priority
               move || {
                   info!("Flushing Tantivy index...");
                   
                   // Re-open index to ensure all writes are committed
                   match Index::open_in_dir(&index_path) {
                       Ok(index) => {
                           if let Ok(mut writer) = index.writer(50_000_000) {
                               match writer.commit() {
                                   Ok(opstamp) => {
                                       info!("Index flushed successfully, opstamp: {}", opstamp);
                                       Ok(())
                                   }
                                   Err(e) => {
                                       error!("Failed to commit index: {}", e);
                                       Err(e.into())
                                   }
                               }
                           } else {
                               warn!("Could not obtain index writer for flush");
                               Ok(())
                           }
                       }
                       Err(e) => {
                           error!("Failed to open index for flush: {}", e);
                           Err(e.into())
                       }
                   }
               }
           ).await;
           
           Ok(Self { indexer, shutdown_handler })
       }
   }
   ```

## Success Criteria
- [ ] Index flush cleanup registered
- [ ] High priority assigned
- [ ] Proper error handling in cleanup
- [ ] Compiles without errors

## Time: 6 minutes