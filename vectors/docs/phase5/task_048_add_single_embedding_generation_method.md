# Task 048: Add Single Embedding Generation Method

## Context
Creating public interface for single text embedding generation.

## Task Objective
Add generate_embedding async method that processes single text input.

## Steps
1. Open src/vector_store.rs
2. Add public method to TransactionalVectorStore impl:
   ```rust
   pub async fn generate_embedding(&self, text: &str) -> VectorStoreResult<Vec<f32>> {
       let preprocessed = self.preprocess_text(text);
       let embedding = self.generate_mock_embedding(&preprocessed);
       self.validate_embedding(&embedding)?;
       Ok(embedding)
   }
   ```
3. Save file

## Success Criteria
- [ ] Public async method added
- [ ] Uses preprocessing and validation
- [ ] Returns VectorStoreResult
- [ ] File compiles

## Time: 4 minutes

## Next Task
Task 049: Add embedding error type