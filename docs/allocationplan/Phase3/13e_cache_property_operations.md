# Task 13e: Implement Property Cache Operations

**Time**: 7 minutes (1.5 min read, 4 min implement, 1.5 min verify)
**Dependencies**: 13d_cache_chain_operations.md
**Stage**: Inheritance System

## Objective
Add methods for caching property resolutions.

## Implementation
Add to `src/inheritance/cache/inheritance_cache_manager.rs`:

```rust
impl InheritanceCacheManager {
    pub async fn get_cached_properties(&self, cache_key: &str) -> Option<CachedPropertyResolution> {
        let start_time = std::time::Instant::now();
        
        let result = {
            let cache = self.property_cache.read().await;
            cache.get(cache_key).cloned()
        };
        
        // Update statistics
        let execution_time = start_time.elapsed().as_millis() as f64;
        let is_hit = result.is_some();
        
        {
            let mut stats = self.cache_stats.write().await;
            if is_hit {
                stats.hit_count += 1;
            } else {
                stats.miss_count += 1;
            }
        }
        
        // Update access count if found
        if result.is_some() {
            let mut cache = self.property_cache.write().await;
            if let Some(cached) = cache.get_mut(cache_key) {
                cached.access_count += 1;
                cached.last_accessed = chrono::Utc::now();
            }
        }
        
        result
    }

    pub async fn store_properties(&self, cache_key: String, properties: crate::inheritance::property_types::ResolvedProperties) {
        let now = chrono::Utc::now();
        
        // Estimate cache size (simplified)
        let estimated_size = std::mem::size_of_val(&properties) + 
            properties.direct_properties.len() * 100 +
            properties.inherited_properties.len() * 150;
        
        let cached_properties = CachedPropertyResolution {
            resolved_properties: properties,
            cached_at: now,
            access_count: 0,
            last_accessed: now,
            cache_size_bytes: estimated_size,
        };
        
        {
            let mut cache = self.property_cache.write().await;
            cache.insert(cache_key, cached_properties);
        }
        
        // Update statistics
        {
            let mut stats = self.cache_stats.write().await;
            stats.total_entries += 1;
            stats.memory_usage_bytes += estimated_size;
        }
        
        // Check if we need to evict
        self.check_property_eviction().await;
    }

    async fn check_property_eviction(&self) {
        let stats = self.cache_stats.read().await;
        let memory_usage_mb = stats.memory_usage_bytes / 1024 / 1024;
        
        if memory_usage_mb > self.config.max_memory_usage_mb {
            drop(stats);
            self.evict_properties_by_memory().await;
        }
    }

    async fn evict_properties_by_memory(&self) {
        let mut cache = self.property_cache.write().await;
        
        // Find largest entries to evict
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by_key(|(_, cached)| std::cmp::Reverse(cached.cache_size_bytes));
        
        let mut bytes_freed = 0;
        let target_bytes = self.config.max_memory_usage_mb * 1024 * 1024 / 4; // Free 25%
        
        for (key, cached) in entries.iter().take(10) { // Limit evictions
            if bytes_freed >= target_bytes {
                break;
            }
            bytes_freed += cached.cache_size_bytes;
            cache.remove(*key);
        }
        
        // Update statistics
        let mut stats = self.cache_stats.write().await;
        stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(bytes_freed);
        stats.eviction_count += 1;
    }
}
```

## Success Criteria
- Property caching operations work
- Memory-based eviction functions correctly

## Next Task
13f_cache_invalidation.md