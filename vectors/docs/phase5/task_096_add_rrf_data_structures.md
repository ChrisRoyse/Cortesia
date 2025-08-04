# Task 096: Add Reciprocal Rank Fusion Data Structures

## Prerequisites Check
- [ ] Task 095 completed: basic constructors implemented
- [ ] Unified search module foundation is complete
- [ ] Run: `cargo check` (should pass)

## Context
Begin implementing Reciprocal Rank Fusion (RRF) for combining text and vector search results.

## Task Objective
Add data structures needed for RRF algorithm implementation.

## Steps
1. Add RRF configuration struct:
   ```rust
   /// Configuration for reciprocal rank fusion
   #[derive(Debug, Clone)]
   pub struct RrfConfig {
       /// K parameter for RRF formula: 1/(k + rank)
       pub k_parameter: f32,
       /// Maximum results to consider from each source
       pub max_results_per_source: usize,
       /// Weight for text search results
       pub text_weight: f32,
       /// Weight for vector search results
       pub vector_weight: f32,
   }
   ```
2. Add ranked result structure:
   ```rust
   /// Intermediate result with ranking information
   #[derive(Debug, Clone)]
   struct RankedResult {
       /// Original result
       result: UnifiedResult,
       /// Rank from text search (None if not found)
       text_rank: Option<usize>,
       /// Rank from vector search (None if not found)
       vector_rank: Option<usize>,
       /// Combined RRF score
       rrf_score: f32,
   }
   ```
3. Add default implementation for RrfConfig:
   ```rust
   impl Default for RrfConfig {
       fn default() -> Self {
           Self {
               k_parameter: 60.0,
               max_results_per_source: 100,
               text_weight: 1.0,
               vector_weight: 1.0,
           }
       }
   }
   ```
4. Verify compilation

## Success Criteria
- [ ] RrfConfig struct with proper parameters
- [ ] RankedResult struct for intermediate processing
- [ ] Sensible default values for RRF parameters
- [ ] Compiles without errors

## Time: 4 minutes

## Next Task
Task 097 will implement the core RRF scoring algorithm.

## Notes
RRF parameters follow standard literature values, with k=60 being commonly used.