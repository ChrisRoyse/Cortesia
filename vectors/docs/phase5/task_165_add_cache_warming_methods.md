# Task 107: Add Cache Warming Methods

## Prerequisites Check
- [ ] Task 106 completed: serialization helpers implemented
- [ ] Cache integration helpers are functional
- [ ] Run: `cargo check` (should pass)

## Context
Add cache warming and maintenance methods for optimal performance.

## Task Objective
Implement cache warming and reset functionality for AsyncMemoryCache.

## Steps
1. Add cache warming and maintenance methods:
   ```rust
   impl AsyncMemoryCache {
       /// Warm cache with common queries
       pub async fn warm_cache(&self, queries: Vec<(String, String, u64)>) {
           let mut cache = self.inner.write().await;
           for (key, data, ttl) in queries {
               cache.insert(key, data, ttl);
           }
       }
       
       /// Clear cache and reset statistics
       pub async fn reset(&self) {
           let mut cache = self.inner.write().await;
           cache.clear();
           cache.stats.hits = 0;
           cache.stats.misses = 0;
           cache.stats.hit_ratio = 0.0;
       }
       
       /// Force cleanup of expired entries
       pub async fn force_cleanup(&self) {
           let mut cache = self.inner.write().await;
           cache.cleanup_expired();
       }
   }
   ```
2. Verify compilation

## Success Criteria
- [ ] Cache warming with preloaded queries
- [ ] Complete cache reset with statistics clearing
- [ ] Force cleanup for expired entries
- [ ] Proper async write lock handling
- [ ] Compiles without errors

## Time: 2 minutes

## Next Task
Task 108 will add integration helpers for unified search.

## Notes
Cache warming enables optimal performance by preloading frequently accessed data.