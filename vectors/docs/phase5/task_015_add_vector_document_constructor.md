# Task 015: Add VectorDocument new() Constructor

## Prerequisites Check
- [ ] Task 014 completed: VectorDocument struct defined
- [ ] Struct has all required fields (id, file_path, content, chunk_index)
- [ ] Struct compiles with Debug and Clone derives
- [ ] Run: `cargo check` (should pass with struct definition)

## Context
VectorDocument struct defined. Adding basic constructor method.

## Task Objective
Add simple new() constructor method to VectorDocument impl block

## Steps
1. Open src/vector_store.rs in editor
2. Add impl block after struct definition:
   ```rust
   impl VectorDocument {
       pub fn new(id: String, file_path: String, content: String, chunk_index: i32) -> Self {
           Self {
               id,
               file_path,
               content,
               chunk_index,
           }
       }
   }
   ```
3. Save file

## Success Criteria
- [ ] impl VectorDocument block added
- [ ] new() method takes 4 parameters with correct types
- [ ] Method returns Self using struct initialization
- [ ] All fields properly assigned
- [ ] File saves successfully

## Time: 4 minutes

## Next Task
Task 016: Add VectorDocument generate_id() method