# Task 017: Add VectorDocument Test Helper Method

## Prerequisites Check
- [ ] Task 016 completed: generate_id() method implemented
- [ ] All VectorDocument methods compile without errors
- [ ] impl VectorDocument block is complete and functional
- [ ] Run: `cargo check` (should pass with generate_id method)

## Context
Core VectorDocument methods complete. Adding test-only convenience method.

## Task Objective
Add test_document() helper method in conditional compilation block

## Steps
1. Open src/vector_store.rs in editor
2. Add conditional impl block after main impl:
   ```rust
   #[cfg(test)]
   impl VectorDocument {
       pub fn test_document(file_name: &str, content: &str, chunk_index: i32) -> Self {
           let id = Self::generate_id(file_name, chunk_index);
           Self::new(id, file_name.to_string(), content.to_string(), chunk_index)
       }
   }
   ```
3. Save file

## Success Criteria
- [ ] #[cfg(test)] conditional compilation attribute added
- [ ] test_document() method in separate impl block
- [ ] Method uses generate_id() internally
- [ ] Creates new instance with test data
- [ ] Only compiles during testing

## Time: 4 minutes

## Next Task
Task 018: Update module exports in lib.rs