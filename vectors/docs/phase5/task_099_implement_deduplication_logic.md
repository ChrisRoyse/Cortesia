# Task 099: Implement Result Deduplication Logic

## Prerequisites Check
- [ ] Task 098 completed: Fusion HashMap setup
- [ ] HashMap and result structures available
- [ ] src/unified_search.rs compiles without errors

## Context
Building result fusion system. Need deduplication logic to merge duplicate results.

## Task Objective
Add method to deduplicate results by file path and merge scores.

## Steps
1. Add deduplication method:
   ```rust
   /// Deduplicate results and merge scores
   fn deduplicate_results(&self, results: &mut HashMap<String, UnifiedResult>) {
       // Results are already deduplicated by HashMap key (file_path)
       // Update combined scores based on available text/vector scores
       for result in results.values_mut() {
           result.combined_score = self.calculate_combined_score(
               result.text_score, 
               result.vector_score
           );
       }
   }
   ```

## Success Criteria
- [ ] Deduplication method added
- [ ] Score merging logic implemented
- [ ] File compiles

## Time: 4 minutes

## Next Task
Task 100: Add result sorting and finalization