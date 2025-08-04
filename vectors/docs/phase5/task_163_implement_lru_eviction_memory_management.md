# Task 104: Implement LRU Eviction and Memory Management

## Prerequisites Check
- [ ] Task 103 completed: cache insertion and retrieval methods implemented
- [ ] Basic cache operations are working
- [ ] Run: `cargo check` (should pass)

## Context
Implement LRU eviction and memory management to keep cache within configured limits.

## Task Objective
Add eviction strategies and memory management methods to prevent cache overflow.

## Steps
1. Add LRU eviction method:
   ```rust
   impl MemoryEfficientCache {
       /// Evict entries to make space for new data
       fn maybe_evict_for_space(&mut self, needed_bytes: usize) {
           // Check if we exceed memory limit
           while self.stats.memory_bytes + needed_bytes > self.config.max_memory_bytes 
               || self.entries.len() >= self.config.max_entries {
               
               if self.entries.is_empty() {
                   break;
               }
               
               // Find LRU entry (least recently accessed with lowest access count)
               let lru_key = self.find_lru_key();
               if let Some(key) = lru_key {
                   self.remove_entry(&key);
               } else {
                   break;
               }
           }
       }
       
       /// Find the least recently used key
       fn find_lru_key(&self) -> Option<String> {
           self.entries
               .iter()
               .min_by(|(_, a), (_, b)| {
                   // First compare access count, then last accessed time
                   a.access_count
                       .cmp(&b.access_count)
                       .then_with(|| a.last_accessed.cmp(&b.last_accessed))
               })
               .map(|(key, _)| key.clone())
       }
       
       /// Remove entry and update stats
       fn remove_entry(&mut self, key: &str) {
           if let Some(entry) = self.entries.remove(key) {
               self.stats.memory_bytes = self.stats.memory_bytes.saturating_sub(entry.size_bytes);
               self.update_stats();
           }
       }
   }
   ```
2. Add cleanup method for expired entries:
   ```rust
   impl MemoryEfficientCache {
       /// Clean up expired entries
       pub fn cleanup_expired(&mut self) {
           let now = Instant::now();
           let expired_keys: Vec<String> = self.entries
               .iter()
               .filter(|(_, entry)| now.duration_since(entry.created_at).as_secs() > entry.ttl)
               .map(|(key, _)| key.clone())
               .collect();
           
           for key in expired_keys {
               self.remove_entry(&key);
           }
           
           self.last_cleanup = now;
       }
       
       /// Check if cleanup is needed and perform if so
       pub fn maybe_cleanup(&mut self) {
           if self.last_cleanup.elapsed().as_secs() >= self.config.cleanup_interval {
               self.cleanup_expired();
           }
       }
   }
   ```
3. Add cache clearing method:
   ```rust
   impl MemoryEfficientCache {
       /// Clear all cache entries
       pub fn clear(&mut self) {
           self.entries.clear();
           self.stats.memory_bytes = 0;
           self.update_stats();
       }
       
       /// Get current cache size
       pub fn len(&self) -> usize {
           self.entries.len()
       }
       
       /// Check if cache is empty
       pub fn is_empty(&self) -> bool {
           self.entries.is_empty()
       }
   }
   ```
4. Verify compilation

## Success Criteria
- [ ] LRU eviction based on access count and time
- [ ] Memory limit enforcement prevents overflow
- [ ] Entry count limit enforcement
- [ ] Expired entry cleanup with configurable intervals
- [ ] Memory usage tracking is accurate after evictions
- [ ] Cache clearing and size methods
- [ ] Compiles without errors

## Time: 7 minutes

## Next Task
Task 105 will add cache integration helpers and async support.

## Notes
LRU eviction uses both access frequency and recency to make intelligent eviction decisions while respecting memory limits.