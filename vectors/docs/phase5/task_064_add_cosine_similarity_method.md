# Task 064: Add Cosine Similarity Calculation Method

## Prerequisites Check
- [ ] Task 063 completed: SearchConfig default implemented
- [ ] All vector math dependencies available
- [ ] f32 vector types working correctly
- [ ] Run: `cargo check` (should pass)

## Context
Need cosine similarity calculation for vector comparison.

## Task Objective
Add cosine_similarity helper method for vector comparison.

## Steps
1. Open src/vector_store.rs
2. Add helper method after SearchConfig impl:
   ```rust
   fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
       let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
       let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
       let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
       dot_product / (norm_a * norm_b)
   }
   ```
3. Save file

## Success Criteria
- [ ] cosine_similarity function added
- [ ] Mathematical operations correct
- [ ] File compiles

## Time: 5 minutes

## Next Task
Task 065: Add vector search query validation