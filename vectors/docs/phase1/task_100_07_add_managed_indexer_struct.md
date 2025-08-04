# Task 100_07: Add ManagedIndexer Struct

## Prerequisites Check
- [ ] Task 100_06 completed: Cleanup execution method added
- [ ] Run: `cargo check` (should pass)

## Context
Add ManagedIndexer wrapper for DocumentIndexer with shutdown integration.

## Task Objective
Define ManagedIndexer struct that wraps DocumentIndexer with shutdown awareness.

## Steps
1. Add ManagedIndexer struct:
   ```rust
   // Integration with DocumentIndexer
   pub struct ManagedIndexer {
       indexer: DocumentIndexer,
       shutdown_handler: Arc<ShutdownHandler>,
   }
   ```

## Success Criteria
- [ ] ManagedIndexer struct added
- [ ] Proper field types
- [ ] Compiles without errors

## Time: 2 minutes