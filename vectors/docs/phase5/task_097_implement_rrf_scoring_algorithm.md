# Task 097: Implement RRF Scoring Algorithm

## Prerequisites Check
- [ ] Task 096 completed: RRF data structures added
- [ ] RrfConfig and RankedResult structs are defined
- [ ] Run: `cargo check` (should pass)

## Context
Implement the core Reciprocal Rank Fusion scoring algorithm.

## Task Objective
Add the RRF scoring method that combines rankings from multiple search sources.

## Steps
1. Add RRF scoring implementation:
   ```rust
   impl UnifiedSearchSystem {
       /// Calculate RRF score for a result
       fn calculate_rrf_score(
           &self,
           text_rank: Option<usize>,
           vector_rank: Option<usize>,
           config: &RrfConfig,
       ) -> f32 {
           let mut rrf_score = 0.0;
           
           // Add text search contribution
           if let Some(rank) = text_rank {
               rrf_score += config.text_weight / (config.k_parameter + rank as f32 + 1.0);
           }
           
           // Add vector search contribution
           if let Some(rank) = vector_rank {
               rrf_score += config.vector_weight / (config.k_parameter + rank as f32 + 1.0);
           }
           
           rrf_score
       }
   }
   ```
2. Add method to create ranked results:
   ```rust
   impl UnifiedSearchSystem {
       /// Create ranked result with RRF score
       fn create_ranked_result(
           &self,
           result: UnifiedResult,
           text_rank: Option<usize>,
           vector_rank: Option<usize>,
           config: &RrfConfig,
       ) -> RankedResult {
           let rrf_score = self.calculate_rrf_score(text_rank, vector_rank, config);
           
           RankedResult {
               result,
               text_rank,
               vector_rank,
               rrf_score,
           }
       }
   }
   ```
3. Verify compilation

## Success Criteria
- [ ] RRF scoring algorithm correctly implemented
- [ ] Formula follows standard 1/(k + rank) pattern
- [ ] Both text and vector rankings are considered
- [ ] Weighted contributions are properly calculated
- [ ] Helper method for creating ranked results
- [ ] Compiles without errors

## Time: 5 minutes

## Next Task
Task 098 will implement result fusion and deduplication logic.

## Notes
RRF algorithm uses reciprocal of rank plus constant for score normalization and combination.