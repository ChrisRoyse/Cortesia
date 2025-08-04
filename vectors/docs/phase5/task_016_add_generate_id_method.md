# Task 016: Add VectorDocument generate_id() Method

## Prerequisites Check
- [ ] Task 015 completed: VectorDocument constructor implemented
- [ ] new() method compiles and works correctly
- [ ] impl VectorDocument block exists and is accessible
- [ ] Run: `cargo check` (should pass with constructor)

## Context
Constructor added. Now adding utility method for generating consistent document IDs.

## Task Objective
Add generate_id() static method to create IDs from file path and chunk index

## Steps
1. Open src/vector_store.rs in editor
2. Add method to existing impl VectorDocument block:
   ```rust
   /// Create a unique ID from file path and chunk index
   pub fn generate_id(file_path: &str, chunk_index: i32) -> String {
       format!("{}_{}", file_path.replace(['\\', '/', '.'], "_"), chunk_index)
   }
   ```
3. Save file

## Success Criteria
- [ ] generate_id() method added to existing impl block
- [ ] Method is public and static (takes &str, not &self)
- [ ] Replaces path separators and dots with underscores
- [ ] Returns String with format "path_chunkindex"
- [ ] Documentation comment included

## Time: 4 minutes

## Next Task
Task 017: Add VectorDocument test helper method