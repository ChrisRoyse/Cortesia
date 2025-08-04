# Task 014: Define VectorDocument Struct Basic Fields

## Prerequisites Check
- [ ] Task 013 completed: basic imports added to vector_store.rs
- [ ] All import statements compile without errors  
- [ ] File contains LanceDB and Arrow imports
- [ ] Run: `cargo check` (should pass with imports)

## Context
Imports added to vector_store.rs. Now defining the core data structure for document chunks.

## Task Objective
Add VectorDocument struct definition with four basic fields

## Steps
1. Open src/vector_store.rs in editor
2. Add struct definition after imports:
   ```rust
   #[derive(Debug, Clone)]
   pub struct VectorDocument {
       /// Unique identifier for this document chunk
       pub id: String,
       /// Path to the source file
       pub file_path: String,
       /// Text content of this chunk
       pub content: String,
       /// Index of this chunk within the file (0-based)
       pub chunk_index: i32,
   }
   ```
3. Save file

## Success Criteria
- [ ] VectorDocument struct defined with exact field names and types
- [ ] Includes Debug and Clone derives
- [ ] All fields are public
- [ ] Documentation comments for each field included
- [ ] File saves successfully

## Time: 5 minutes

## Next Task
Task 015: Add VectorDocument new() constructor