# Task 022: Add Connection Accessor Methods

## Prerequisites Check
- [ ] Task 021 completed: async new() constructor implemented
- [ ] Constructor compiles and handles LanceDB connection
- [ ] impl TransactionalVectorStore block exists
- [ ] Run: `cargo check` (should pass with constructor)

## Context
TransactionalVectorStore constructor complete. Adding getter methods for fields.

## Task Objective
Add connection() and db_path() accessor methods to existing impl block

## Steps
1. Open src/vector_store.rs in editor
2. Add these methods to existing TransactionalVectorStore impl block:
   ```rust
   /// Get reference to the database connection
   pub fn connection(&self) -> &Connection {
       &self.connection
   }
   
   /// Get the database path
   pub fn db_path(&self) -> &str {
       &self.db_path
   }
   ```
3. Save file

## Success Criteria
- [ ] connection() method returns &Connection
- [ ] db_path() method returns &str
- [ ] Both methods are public
- [ ] Methods added to existing impl block
- [ ] Documentation comments included

## Time: 4 minutes

## Next Task
Task 023: Add Display trait implementation