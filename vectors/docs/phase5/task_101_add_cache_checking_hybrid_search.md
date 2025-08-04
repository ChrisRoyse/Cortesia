# Task 101: Add Cache Checking for Hybrid Search

## Prerequisites Check
- [ ] Task 100 completed: Result sorting finalized
- [ ] Cache system available from previous tasks
- [ ] src/unified_search.rs compiles without errors

## Context
Starting hybrid search implementation. First step is cache checking.

## Task Objective
Add cache key generation and cache checking logic for hybrid search.

## Steps
1. Add cache checking method:
   ```rust
   /// Generate cache key for hybrid search
   fn generate_cache_key(&self, query: &str, mode: &SearchMode) -> String {
       format!("hybrid:{}:{:?}", query, mode)
   }
   
   /// Check cache for existing results
   async fn check_cache(&self, cache_key: &str) -> Option<Vec<UnifiedResult>> {
       if let Some(cached) = self.cache.read().await.get(cache_key) {
           Some(cached.clone())
       } else {
           None
       }
   }
   ```

## Success Criteria
- [ ] Cache key generation method added
- [ ] Cache checking method added  
- [ ] File compiles

## Time: 4 minutes

## Next Task
Task 102: Implement parallel search execution