# Task 043: Add Embedding Config Struct

## Prerequisites Check
- [ ] Task 042 completed: result type alias added
- [ ] All error handling infrastructure is in place
- [ ] VectorStoreError enum is complete and functional
- [ ] Run: `cargo check` (should pass with error handling)

## Context
Starting embedding generation system with configuration.

## Task Objective
Define EmbeddingConfig struct with basic fields.

## Steps
1. Open src/vector_store.rs
2. Add struct:
   ```rust
   #[derive(Debug, Clone)]
   pub struct EmbeddingConfig {
       pub dimension: usize,
       pub max_text_length: usize,
   }
   ```
3. Save file

## Success Criteria
- [ ] Struct added with 2 fields
- [ ] Derives Debug, Clone
- [ ] File compiles

## Time: 3 minutes

## Next Task
Task 044: Add embedding config default