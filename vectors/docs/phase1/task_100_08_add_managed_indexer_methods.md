# Task 100_08: Add ManagedIndexer Methods

## Prerequisites Check
- [ ] Task 100_07 completed: ManagedIndexer struct added
- [ ] Run: `cargo check` (should pass)

## Context
Implement constructor and indexing methods for ManagedIndexer with shutdown checking.

## Task Objective
Add ManagedIndexer::new() and index_file() methods with shutdown awareness.

## Steps
1. Add ManagedIndexer implementation:
   ```rust
   impl ManagedIndexer {
       pub async fn new(index_path: &Path) -> Result<Self> {
           let indexer = DocumentIndexer::new(index_path)?;
           let shutdown_handler = Arc::new(ShutdownHandler::new());
           
           Ok(Self { indexer, shutdown_handler })
       }
       
       pub async fn index_file(&mut self, file_path: &Path) -> Result<()> {
           // Check if shutdown was requested
           if self.shutdown_handler.is_shutdown_requested() {
               return Err(anyhow::anyhow!("Shutdown in progress, refusing new indexing"));
           }
           
           self.indexer.index_file(file_path)
       }
   }
   ```

## Success Criteria
- [ ] Constructor method implemented
- [ ] Shutdown-aware indexing method added
- [ ] Proper error handling
- [ ] Compiles without errors

## Time: 4 minutes