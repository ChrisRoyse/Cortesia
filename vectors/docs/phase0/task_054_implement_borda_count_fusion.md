# Micro-Task 054: Implement Borda Count Fusion

## Objective
Add Borda Count fusion algorithm for result combination.

## Context
Borda Count is a position-based fusion method that assigns points based on ranking position.

## Prerequisites
- Task 053 completed (RRF implementation added)

## Time Estimate
10 minutes

## Instructions
1. Add to `src/fusion/fusion_algorithms.rs`:
   ```rust
   pub struct BordaCount {
       pub max_points: usize,
   }
   
   impl FusionAlgorithm for BordaCount {
       fn fuse(&self, text_results: Vec<(String, f32)>, 
               vector_results: Vec<(String, f32)>) -> Vec<(String, f32)> {
           let mut scores: HashMap<String, f32> = HashMap::new();
           let n = self.max_points;
           
           for (pos, (id, _)) in text_results.iter().enumerate() {
               let points = (n - pos.min(n - 1)) as f32;
               *scores.entry(id.clone()).or_insert(0.0) += points;
           }
           
           for (pos, (id, _)) in vector_results.iter().enumerate() {
               let points = (n - pos.min(n - 1)) as f32;
               *scores.entry(id.clone()).or_insert(0.0) += points;
           }
           
           let mut results: Vec<_> = scores.into_iter().collect();
           results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
           results
       }
   }
   ```
2. Add test for Borda Count
3. Run: `cargo test borda`

## Expected Output
- Borda Count struct created
- Position-based scoring implemented
- Tests passing

## Success Criteria
- [ ] Borda Count algorithm implemented
- [ ] Position-based points calculation working
- [ ] Test passes successfully
- [ ] Results properly sorted

## Next Task
task_055_implement_combsum_fusion.md