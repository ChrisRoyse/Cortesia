# Micro-Task 048: Create Search Quality Metrics

## Objective
Implement comprehensive search quality metrics including NDCG, MAP, and relevance scoring.

## Prerequisites
- Task 047 completed (cross-validation testing implemented)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add search quality metrics to `src/validation.rs`:
   ```rust
   /// Advanced search quality metrics
   #[derive(Debug, Clone)]
   pub struct SearchQualityMetrics {
       pub ndcg_at_k: f32,
       pub map_score: f32,
       pub mrr_score: f32,
       pub relevance_distribution: Vec<f32>,
   }
   
   impl SimilarityValidator {
       /// Calculate Normalized Discounted Cumulative Gain (NDCG@k)
       pub fn calculate_ndcg(&self, results: &[SimilarityResult], relevance_scores: &[f32], k: usize) -> f32 {
           if results.is_empty() || relevance_scores.is_empty() {
               return 0.0;
           }
           
           let k = k.min(results.len()).min(relevance_scores.len());
           
           // Calculate DCG
           let mut dcg = 0.0;
           for i in 0..k {
               let relevance = relevance_scores[i];
               let discount = (i as f32 + 2.0).log2();
               dcg += (2.0_f32.powf(relevance) - 1.0) / discount;
           }
           
           // Calculate IDCG (Ideal DCG)
           let mut sorted_relevance = relevance_scores[..k].to_vec();
           sorted_relevance.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
           
           let mut idcg = 0.0;
           for (i, &relevance) in sorted_relevance.iter().enumerate() {
               let discount = (i as f32 + 2.0).log2();
               idcg += (2.0_f32.powf(relevance) - 1.0) / discount;
           }
           
           if idcg == 0.0 { 0.0 } else { dcg / idcg }
       }
       
       /// Calculate Mean Average Precision (MAP)
       pub fn calculate_map(&self, all_results: &[Vec<SimilarityResult>], all_relevance: &[Vec<bool>]) -> f32 {
           if all_results.is_empty() || all_relevance.is_empty() {
               return 0.0;
           }
           
           let mut total_ap = 0.0;
           let mut valid_queries = 0;
           
           for (results, relevance) in all_results.iter().zip(all_relevance.iter()) {
               if let Some(ap) = self.calculate_average_precision(results, relevance) {
                   total_ap += ap;
                   valid_queries += 1;
               }
           }
           
           if valid_queries == 0 { 0.0 } else { total_ap / valid_queries as f32 }
       }
       
       /// Calculate Average Precision for a single query
       fn calculate_average_precision(&self, results: &[SimilarityResult], relevance: &[bool]) -> Option<f32> {
           if results.is_empty() || relevance.is_empty() {
               return None;
           }
           
           let mut relevant_found = 0;
           let mut total_relevant = relevance.iter().filter(|&&r| r).count();
           
           if total_relevant == 0 {
               return Some(0.0);
           }
           
           let mut precision_sum = 0.0;
           
           for (i, &is_relevant) in relevance.iter().enumerate().take(results.len()) {
               if is_relevant {
                   relevant_found += 1;
                   precision_sum += relevant_found as f32 / (i + 1) as f32;
               }
           }
           
           Some(precision_sum / total_relevant as f32)
       }
       
       /// Calculate Mean Reciprocal Rank (MRR)
       pub fn calculate_mrr(&self, all_results: &[Vec<SimilarityResult>], all_relevance: &[Vec<bool>]) -> f32 {
           if all_results.is_empty() || all_relevance.is_empty() {
               return 0.0;
           }
           
           let mut total_rr = 0.0;
           let mut valid_queries = 0;
           
           for (results, relevance) in all_results.iter().zip(all_relevance.iter()) {
               if let Some(rr) = self.calculate_reciprocal_rank(results, relevance) {
                   total_rr += rr;
                   valid_queries += 1;
               }
           }
           
           if valid_queries == 0 { 0.0 } else { total_rr / valid_queries as f32 }
       }
       
       /// Calculate Reciprocal Rank for a single query
       fn calculate_reciprocal_rank(&self, results: &[SimilarityResult], relevance: &[bool]) -> Option<f32> {
           for (i, &is_relevant) in relevance.iter().enumerate().take(results.len()) {
               if is_relevant {
                   return Some(1.0 / (i + 1) as f32);
               }
           }
           Some(0.0)
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`
5. Commit: `git add crates\lancedb-integration && git commit -m "Add advanced search quality metrics (NDCG, MAP, MRR)"`

## Success Criteria
- [ ] NDCG calculation with proper discounting
- [ ] MAP and MRR implementations
- [ ] Quality metrics integration with validation framework

## Next Task
task_049_implement_relevance_feedback_testing.md