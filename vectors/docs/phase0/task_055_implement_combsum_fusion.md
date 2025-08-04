# Micro-Task 055: Implement CombSUM Fusion

## Objective
Add CombSUM fusion algorithm with score normalization.

## Context
CombSUM normalizes and combines scores from different search systems.

## Prerequisites
- Task 054 completed (Borda Count added)

## Time Estimate
10 minutes

## Instructions
1. Add to `src/fusion/fusion_algorithms.rs`:
   ```rust
   pub struct CombSUM {
       pub text_weight: f32,
       pub vector_weight: f32,
   }
   
   impl CombSUM {
       fn normalize_scores(results: &[(String, f32)]) -> Vec<(String, f32)> {
           let max = results.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
           let min = results.iter().map(|(_, s)| *s).fold(f32::MAX, f32::min);
           let range = max - min;
           
           results.iter().map(|(id, score)| {
               let normalized = if range > 0.0 {
                   (score - min) / range
               } else { 0.5 };
               (id.clone(), normalized)
           }).collect()
       }
   }
   
   impl FusionAlgorithm for CombSUM {
       fn fuse(&self, text_results: Vec<(String, f32)>, 
               vector_results: Vec<(String, f32)>) -> Vec<(String, f32)> {
           let text_norm = Self::normalize_scores(&text_results);
           let vector_norm = Self::normalize_scores(&vector_results);
           
           let mut scores: HashMap<String, f32> = HashMap::new();
           
           for (id, score) in text_norm {
               *scores.entry(id).or_insert(0.0) += score * self.text_weight;
           }
           
           for (id, score) in vector_norm {
               *scores.entry(id).or_insert(0.0) += score * self.vector_weight;
           }
           
           let mut results: Vec<_> = scores.into_iter().collect();
           results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
           results
       }
   }
   ```
2. Add unit test
3. Commit: `git add -A && git commit -m "Complete fusion algorithms"`

## Expected Output
- CombSUM with normalization implemented
- Weighted combination working
- All fusion algorithms complete

## Success Criteria
- [ ] CombSUM struct created
- [ ] Score normalization working
- [ ] Weighted combination implemented
- [ ] Tests passing

## Next Task
task_056_create_query_builder_foundation.md