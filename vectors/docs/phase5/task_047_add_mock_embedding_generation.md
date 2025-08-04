# Task 047: Add Mock Embedding Generation

## Context
Creating deterministic embedding generation for testing.

## Task Objective
Add generate_mock_embedding method that creates 384-dimensional vectors.

## Steps
1. Open src/vector_store.rs
2. Add method to TransactionalVectorStore impl:
   ```rust
   fn generate_mock_embedding(&self, text: &str) -> Vec<f32> {
       let mut embedding = vec![0.1f32; 384];
       // Simple deterministic generation based on text length
       for (i, value) in embedding.iter_mut().enumerate() {
           *value = ((text.len() + i) as f32 * 0.01).sin();
       }
       embedding
   }
   ```
3. Save file

## Success Criteria
- [ ] Method generates 384-dimensional vector
- [ ] Deterministic based on input text
- [ ] File compiles

## Time: 5 minutes

## Next Task
Task 048: Add single embedding generation method