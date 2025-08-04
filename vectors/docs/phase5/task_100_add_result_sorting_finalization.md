# Task 100: Add Result Sorting and Finalization

## Prerequisites Check
- [ ] Task 099 completed: Deduplication logic implemented
- [ ] HashMap with merged results exists
- [ ] src/unified_search.rs compiles without errors

## Context
Final step of result fusion. Sort merged results by combined score.

## Task Objective
Sort deduplicated results by combined score and convert to final Vec.

## Steps
1. Add sorting and finalization:
   ```rust
   /// Sort results by combined score
   fn finalize_results(&self, results: HashMap<String, UnifiedResult>) -> Vec<UnifiedResult> {
       let mut final_results: Vec<UnifiedResult> = results.into_values().collect();
       
       // Sort by combined score (highest first)
       final_results.sort_by(|a, b| {
           b.combined_score.partial_cmp(&a.combined_score)
               .unwrap_or(std::cmp::Ordering::Equal)
       });
       
       final_results
   }
   ```

## Success Criteria
- [ ] Sorting method added
- [ ] Results converted from HashMap to Vec
- [ ] File compiles

## Time: 3 minutes

## Next Task
Task 101: Add cache checking for hybrid search