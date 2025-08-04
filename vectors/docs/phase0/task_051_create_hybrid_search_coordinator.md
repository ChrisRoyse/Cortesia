# Micro-Task 051: Create Hybrid Search Coordinator

## Objective
Create a hybrid search coordinator that combines text search (Tantivy) and vector search (LanceDB mock) for optimal results.

## Prerequisites
- Task 050 completed (similarity search test suite implemented and committed)
- tantivy-core searcher implementation working
- lancedb-integration vector storage working

## Time Estimate
10 minutes

## Instructions
1. Navigate to vector-indexing crate: `cd crates\vector-indexing`
2. Create `src/hybrid_search.rs`:
   ```rust
   //! Hybrid search combining text and vector search capabilities
   
   use tantivy_core::searcher::{DocumentSearcher, SearchResult as TextSearchResult};
   use lancedb_integration::mock_storage::{MockVectorStorage, SimilarityResult, SearchConfig, VectorSearchFilter};
   use std::collections::HashMap;
   use uuid::Uuid;
   
   /// Hybrid search coordinator combining text and vector search
   pub struct HybridSearchCoordinator {
       text_searcher: DocumentSearcher,
       vector_storage: MockVectorStorage,
       config: HybridSearchConfig,
   }
   
   /// Configuration for hybrid search behavior
   #[derive(Debug, Clone)]
   pub struct HybridSearchConfig {
       pub text_weight: f32,
       pub vector_weight: f32,
       pub normalization_method: NormalizationMethod,
       pub fusion_method: FusionMethod,
       pub min_text_score: f32,
       pub min_vector_score: f32,
   }
   
   #[derive(Debug, Clone)]
   pub enum NormalizationMethod {
       MinMax,
       ZScore,
       Sigmoid,
   }
   
   #[derive(Debug, Clone)]
   pub enum FusionMethod {
       LinearCombination,
       ReciprocalRankFusion,
       BordaCount,
   }
   
   impl Default for HybridSearchConfig {
       fn default() -> Self {
           Self {
               text_weight: 0.6,
               vector_weight: 0.4,
               normalization_method: NormalizationMethod::MinMax,
               fusion_method: FusionMethod::LinearCombination,
               min_text_score: 0.0,
               min_vector_score: 0.0,
           }
       }
   }
   
   /// Hybrid search result combining both text and vector scores
   #[derive(Debug, Clone)]
   pub struct HybridSearchResult {
       pub doc_id: String,
       pub content: String,
       pub title: String,
       pub file_path: Option<String>,
       pub extension: Option<String>,
       pub text_score: f32,
       pub vector_score: f32,
       pub combined_score: f32,
       pub search_type: SearchResultType,
   }
   
   #[derive(Debug, Clone)]
   pub enum SearchResultType {
       TextOnly,
       VectorOnly,
       Hybrid,
   }
   
   impl HybridSearchCoordinator {
       /// Create new hybrid search coordinator
       pub fn new(
           text_searcher: DocumentSearcher,
           vector_storage: MockVectorStorage,
           config: HybridSearchConfig,
       ) -> Self {
           Self {
               text_searcher,
               vector_storage,
               config,
           }
       }
       
       /// Perform hybrid search combining text and vector results
       pub fn hybrid_search(
           &self,
           query: &str,
           limit: usize,
       ) -> Result<Vec<HybridSearchResult>, String> {
           // Perform text search
           let text_results = self.text_searcher.search(query, limit * 2)
               .map_err(|e| format!("Text search failed: {}", e))?;
           
           // Perform vector search
           let vector_config = SearchConfig {
               limit: limit * 2,
               threshold: self.config.min_vector_score,
               include_metadata: true,
           };
           
           let vector_results = self.vector_storage.search_by_content(query, &vector_config)
               .map_err(|e| format!("Vector search failed: {}", e))?;
           
           // Combine and rank results
           let combined_results = self.combine_results(&text_results, &vector_results)?;
           
           // Apply final ranking and limit
           let mut final_results = combined_results;
           final_results.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score)
               .unwrap_or(std::cmp::Ordering::Equal));
           final_results.truncate(limit);
           
           Ok(final_results)
       }
       
       /// Combine text and vector search results
       fn combine_results(
           &self,
           text_results: &[TextSearchResult],
           vector_results: &[SimilarityResult],
       ) -> Result<Vec<HybridSearchResult>, String> {
           let mut result_map: HashMap<String, HybridSearchResult> = HashMap::new();
           
           // Normalize scores
           let normalized_text_scores = self.normalize_scores(
               &text_results.iter().map(|r| r.score).collect::<Vec<_>>()
           );
           
           let normalized_vector_scores = self.normalize_scores(
               &vector_results.iter().map(|r| r.similarity_score).collect::<Vec<_>>()
           );
           
           // Process text results
           for (text_result, &norm_score) in text_results.iter().zip(normalized_text_scores.iter()) {
               if norm_score >= self.config.min_text_score {
                   let hybrid_result = HybridSearchResult {
                       doc_id: text_result.doc_id.clone(),
                       content: text_result.content.clone(),
                       title: text_result.title.clone(),
                       file_path: text_result.file_path.clone(),
                       extension: text_result.extension.clone(),
                       text_score: norm_score,
                       vector_score: 0.0,
                       combined_score: norm_score * self.config.text_weight,
                       search_type: SearchResultType::TextOnly,
                   };
                   
                   result_map.insert(text_result.doc_id.clone(), hybrid_result);
               }
           }
           
           // Process vector results and merge
           for (vector_result, &norm_score) in vector_results.iter().zip(normalized_vector_scores.iter()) {
               if norm_score >= self.config.min_vector_score {
                   let doc_id = vector_result.record.doc_id.to_string();
                   
                   match result_map.get_mut(&doc_id) {
                       Some(existing) => {
                           // Hybrid result - combine scores
                           existing.vector_score = norm_score;
                           existing.combined_score = self.calculate_combined_score(
                               existing.text_score,
                               norm_score,
                           );
                           existing.search_type = SearchResultType::Hybrid;
                       }
                       None => {
                           // Vector-only result
                           let hybrid_result = HybridSearchResult {
                               doc_id: doc_id.clone(),
                               content: vector_result.record.content.clone(),
                               title: vector_result.record.title.clone(),
                               file_path: vector_result.record.file_path.clone(),
                               extension: vector_result.record.extension.clone(),
                               text_score: 0.0,
                               vector_score: norm_score,
                               combined_score: norm_score * self.config.vector_weight,
                               search_type: SearchResultType::VectorOnly,
                           };
                           
                           result_map.insert(doc_id, hybrid_result);
                       }
                   }
               }
           }
           
           Ok(result_map.into_values().collect())
       }
       
       /// Normalize scores based on configuration
       fn normalize_scores(&self, scores: &[f32]) -> Vec<f32> {
           if scores.is_empty() {
               return Vec::new();
           }
           
           match self.config.normalization_method {
               NormalizationMethod::MinMax => {
                   let min_score = scores.iter().cloned().fold(f32::INFINITY, f32::min);
                   let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                   
                   if max_score == min_score {
                       vec![1.0; scores.len()]
                   } else {
                       scores.iter()
                           .map(|&score| (score - min_score) / (max_score - min_score))
                           .collect()
                   }
               }
               NormalizationMethod::ZScore => {
                   let mean = scores.iter().sum::<f32>() / scores.len() as f32;
                   let variance = scores.iter()
                       .map(|&score| (score - mean).powi(2))
                       .sum::<f32>() / scores.len() as f32;
                   let std_dev = variance.sqrt();
                   
                   if std_dev == 0.0 {
                       vec![0.0; scores.len()]
                   } else {
                       scores.iter()
                           .map(|&score| (score - mean) / std_dev)
                           .map(|z| (z + 3.0) / 6.0) // Normalize to 0-1 range
                           .collect()
                   }
               }
               NormalizationMethod::Sigmoid => {
                   scores.iter()
                       .map(|&score| 1.0 / (1.0 + (-score).exp()))
                       .collect()
               }
           }
       }
       
       /// Calculate combined score based on fusion method
       fn calculate_combined_score(&self, text_score: f32, vector_score: f32) -> f32 {
           match self.config.fusion_method {
               FusionMethod::LinearCombination => {
                   text_score * self.config.text_weight + vector_score * self.config.vector_weight
               }
               FusionMethod::ReciprocalRankFusion => {
                   // Simplified RRF - would need actual ranks in full implementation
                   let k = 60.0; // RRF constant
                   1.0 / (k + text_score.recip()) + 1.0 / (k + vector_score.recip())
               }
               FusionMethod::BordaCount => {
                   // Simplified Borda count - would need actual ranks
                   (text_score + vector_score) / 2.0
               }
           }
       }
   }
   ```
3. Add hybrid_search module to `src/lib.rs`:
   ```rust
   pub mod hybrid_search;
   pub use hybrid_search::{HybridSearchCoordinator, HybridSearchConfig, HybridSearchResult};
   ```
4. Test: `cargo check`
5. Return to root: `cd ..\..`

## Success Criteria
- [ ] HybridSearchCoordinator combining text and vector search
- [ ] Multiple normalization and fusion methods
- [ ] Configurable weighting and thresholds
- [ ] Proper result combination and ranking

## Next Task
task_052_implement_search_result_fusion.md