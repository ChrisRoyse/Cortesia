# Micro-Task 049: Implement Relevance Feedback Testing

## Objective
Implement relevance feedback mechanisms to improve search results through user feedback simulation.

## Prerequisites
- Task 048 completed (search quality metrics implemented)

## Time Estimate
10 minutes

## Instructions
1. Navigate to lancedb-integration crate: `cd crates\lancedb-integration`
2. Add relevance feedback to `src/validation.rs`:
   ```rust
   /// Relevance feedback configuration
   #[derive(Debug, Clone)]
   pub struct RelevanceFeedback {
       pub relevant_docs: Vec<Uuid>,
       pub irrelevant_docs: Vec<Uuid>,
       pub feedback_weight: f32,
   }
   
   /// Feedback-enhanced search validator
   impl SimilarityValidator {
       /// Simulate user feedback and measure improvement
       pub fn test_relevance_feedback(&self, query: &str, initial_k: usize) -> Result<FeedbackTestResult, String> {
           let config = SearchConfig {
               limit: initial_k,
               threshold: 0.0,
               include_metadata: true,
           };
           
           // Initial search
           let initial_results = self.storage.search_by_content(query, &config)
               .map_err(|e| e.to_string())?;
           
           // Simulate user feedback (mark top results as relevant, bottom as irrelevant)
           let feedback = self.simulate_user_feedback(&initial_results);
           
           // Enhanced search with feedback
           let enhanced_results = self.search_with_feedback(query, &feedback, initial_k)?;
           
           // Calculate improvement metrics
           let initial_score = self.calculate_result_quality(&initial_results);
           let enhanced_score = self.calculate_result_quality(&enhanced_results);
           
           Ok(FeedbackTestResult {
               query: query.to_string(),
               initial_results: initial_results.len(),
               enhanced_results: enhanced_results.len(),
               initial_quality_score: initial_score,
               enhanced_quality_score: enhanced_score,
               improvement_ratio: enhanced_score / initial_score.max(0.001),
               feedback,
           })
       }
       
       /// Simulate user feedback based on result quality
       fn simulate_user_feedback(&self, results: &[SimilarityResult]) -> RelevanceFeedback {
           let mut relevant_docs = Vec::new();
           let mut irrelevant_docs = Vec::new();
           
           // Simple heuristic: top 30% are relevant, bottom 30% are irrelevant
           let relevant_count = (results.len() as f32 * 0.3).ceil() as usize;
           let irrelevant_start = (results.len() as f32 * 0.7).floor() as usize;
           
           for (i, result) in results.iter().enumerate() {
               if i < relevant_count {
                   relevant_docs.push(result.record.id);
               } else if i >= irrelevant_start {
                   irrelevant_docs.push(result.record.id);
               }
           }
           
           RelevanceFeedback {
               relevant_docs,
               irrelevant_docs,
               feedback_weight: 0.3,
           }
       }
       
       /// Search with relevance feedback
       fn search_with_feedback(&self, query: &str, feedback: &RelevanceFeedback, k: usize) -> Result<Vec<SimilarityResult>, String> {
           let query_embedding = crate::mock_storage::VectorRecord::generate_mock_embedding(query);
           
           // Get all records
           let all_records = self.storage.get_all().map_err(|e| e.to_string())?;
           
           let mut enhanced_results = Vec::new();
           
           for record in all_records {
               // Calculate base similarity
               let base_similarity = crate::mock_storage::VectorMath::cosine_similarity(&query_embedding, &record.embedding)
                   .map_err(|e| e.to_string())?;
               
               // Apply feedback boost/penalty
               let feedback_adjustment = if feedback.relevant_docs.contains(&record.id) {
                   feedback.feedback_weight
               } else if feedback.irrelevant_docs.contains(&record.id) {
                   -feedback.feedback_weight
               } else {
                   0.0
               };
               
               let enhanced_similarity = (base_similarity + feedback_adjustment).max(0.0).min(1.0);
               
               enhanced_results.push(crate::mock_storage::SimilarityResult::new(record, enhanced_similarity));
           }
           
           // Sort by enhanced similarity
           enhanced_results.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score)
               .unwrap_or(std::cmp::Ordering::Equal));
           
           enhanced_results.truncate(k);
           Ok(enhanced_results)
       }
       
       /// Calculate result quality score
       fn calculate_result_quality(&self, results: &[SimilarityResult]) -> f32 {
           if results.is_empty() {
               return 0.0;
           }
           
           // Quality based on similarity scores and diversity
           let avg_similarity: f32 = results.iter().map(|r| r.similarity_score).sum::<f32>() / results.len() as f32;
           
           // Diversity penalty for very similar results
           let mut diversity_score = 1.0;
           for i in 0..results.len() {
               for j in (i + 1)..results.len() {
                   let content_similarity = self.calculate_content_similarity(&results[i].record.content, &results[j].record.content);
                   if content_similarity > 0.9 {
                       diversity_score *= 0.95; // Slight penalty for very similar results
                   }
               }
           }
           
           avg_similarity * diversity_score
       }
       
       /// Calculate content similarity (simple character-based)
       fn calculate_content_similarity(&self, content1: &str, content2: &str) -> f32 {
           let chars1: std::collections::HashSet<char> = content1.chars().collect();
           let chars2: std::collections::HashSet<char> = content2.chars().collect();
           
           let intersection = chars1.intersection(&chars2).count();
           let union = chars1.union(&chars2).count();
           
           if union == 0 { 0.0 } else { intersection as f32 / union as f32 }
       }
   }
   
   /// Results of relevance feedback testing
   #[derive(Debug, Clone)]
   pub struct FeedbackTestResult {
       pub query: String,
       pub initial_results: usize,
       pub enhanced_results: usize,
       pub initial_quality_score: f32,
       pub enhanced_quality_score: f32,
       pub improvement_ratio: f32,
       pub feedback: RelevanceFeedback,
   }
   
   impl FeedbackTestResult {
       pub fn shows_improvement(&self) -> bool {
           self.improvement_ratio > 1.0
       }
       
       pub fn improvement_percentage(&self) -> f32 {
           (self.improvement_ratio - 1.0) * 100.0
       }
   }
   ```
3. Test: `cargo check`
4. Return to root: `cd ..\..`
5. Commit: `git add crates\lancedb-integration && git commit -m "Implement relevance feedback testing with user simulation"`

## Success Criteria
- [ ] Relevance feedback simulation implementation
- [ ] Search enhancement based on feedback
- [ ] Quality improvement measurement
- [ ] Feedback effectiveness testing

## Next Task
task_050_create_similarity_search_test_suite.md