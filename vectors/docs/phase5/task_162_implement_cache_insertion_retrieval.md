# Task 103: Implement Cache Insertion and Retrieval Methods

## Prerequisites Check
- [ ] Task 102 completed: MemoryEfficientCache struct implemented
- [ ] Cache constructor and basic methods are working
- [ ] Run: `cargo check` (should pass)

## Context
Implement core cache operations for storing and retrieving data with TTL and LRU support.

## Task Objective
Add methods for cache insertion, retrieval, and basic cache management operations.

## Steps
1. Add cache retrieval method:
   ```rust
   impl MemoryEfficientCache {
       /// Get cached data if exists and not expired
       pub fn get(&mut self, key: &str) -> Option<&CacheEntry> {
           if let Some(entry) = self.entries.get_mut(key) {
               // Check if expired
               if entry.created_at.elapsed().as_secs() > entry.ttl {
                   self.entries.remove(key);
                   self.stats.misses += 1;
                   self.update_stats();
                   return None;
               }
               
               // Update access tracking
               entry.access_count += 1;
               entry.last_accessed = Instant::now();
               
               self.stats.hits += 1;
               self.update_stats();
               Some(entry)
           } else {
               self.stats.misses += 1;
               self.update_stats();
               None
           }
       }
   }
   ```
2. Add cache insertion method:
   ```rust
   impl MemoryEfficientCache {
       /// Insert data into cache with TTL
       pub fn insert(&mut self, key: String, data: String, ttl: u64) {
           let size_bytes = key.len() + data.len() + std::mem::size_of::<CacheEntry>();
           
           // Check if we need to evict entries
           self.maybe_evict_for_space(size_bytes);
           
           let entry = CacheEntry {
               data,
               created_at: Instant::now(),
               ttl,
               access_count: 1,
               last_accessed: Instant::now(),
               size_bytes,
           };
           
           self.entries.insert(key, entry);
           self.stats.memory_bytes += size_bytes;
           self.update_stats();
       }
       
       /// Check if key exists in cache (without affecting access stats)
       pub fn contains_key(&self, key: &str) -> bool {
           if let Some(entry) = self.entries.get(key) {
               entry.created_at.elapsed().as_secs() <= entry.ttl
           } else {
               false
           }
       }
   }
   ```
3. Add statistics update helper:
   ```rust
   impl MemoryEfficientCache {
       /// Update cache statistics
       fn update_stats(&mut self) {
           self.stats.entries = self.entries.len();
           self.stats.hit_ratio = if self.stats.hits + self.stats.misses > 0 {
               self.stats.hits as f64 / (self.stats.hits + self.stats.misses) as f64
           } else {
               0.0
           };
       }
   }
   ```
4. Verify compilation

## Success Criteria
- [ ] Cache retrieval with TTL expiration checking
- [ ] Cache insertion with size calculation
- [ ] Access tracking for LRU implementation
- [ ] Statistics are updated on each operation
- [ ] contains_key method for cache probing
- [ ] Memory usage tracking is accurate
- [ ] Compiles without errors

## Time: 6 minutes

## Next Task
Task 104 will implement LRU eviction and memory management.

## Notes
Core cache operations handle TTL expiration automatically while tracking access patterns for future LRU eviction.