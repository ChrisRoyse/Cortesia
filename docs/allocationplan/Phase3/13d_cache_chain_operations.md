# Task 13d: Implement Cache Chain Operations

**Time**: 8 minutes (1.5 min read, 5 min implement, 1.5 min verify)
**Dependencies**: 13c_cache_manager_struct.md
**Stage**: Inheritance System

## Objective
Add methods for caching and retrieving inheritance chains.

## Implementation
Add to `src/inheritance/cache/inheritance_cache_manager.rs`:

```rust
impl InheritanceCacheManager {
    pub async fn get_cached_chain(&self, concept_id: &str) -> Option<CachedInheritanceChain> {
        let start_time = std::time::Instant::now();
        
        let result = {
            let cache = self.chain_cache.read().await;
            cache.get(concept_id).map(|cached| {
                // Check if cache entry is still valid
                if chrono::Utc::now().signed_duration_since(cached.ttl_expires_at).num_seconds() > 0 {
                    None // Expired
                } else {
                    Some(cached.clone())
                }
            }).flatten()
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
            let mut cache = self.chain_cache.write().await;
            if let Some(cached) = cache.get_mut(concept_id) {
                cached.access_count += 1;
                cached.last_accessed = chrono::Utc::now();
            }
        }
        
        result
    }

    pub async fn store_chain(&self, concept_id: String, chain: crate::inheritance::hierarchy_types::InheritanceChain) {
        let now = chrono::Utc::now();
        let ttl_expires_at = now + chrono::Duration::minutes(self.config.ttl_minutes);
        
        let cached_chain = CachedInheritanceChain {
            chain,
            cached_at: now,
            access_count: 0,
            last_accessed: now,
            ttl_expires_at,
            dependency_concepts: Vec::new(), // TODO: Extract from chain
        };
        
        {
            let mut cache = self.chain_cache.write().await;
            cache.insert(concept_id.clone(), cached_chain);
        }
        
        // Update statistics
        {
            let mut stats = self.cache_stats.write().await;
            stats.total_entries += 1;
        }
        
        // Check if we need to evict
        self.check_eviction().await;
    }

    async fn check_eviction(&self) {
        let cache_size = self.chain_cache.read().await.len();
        if cache_size > self.config.max_chain_cache_size {
            self.evict_least_recently_used().await;
        }
    }

    async fn evict_least_recently_used(&self) {
        // Simple LRU eviction
        let mut cache = self.chain_cache.write().await;
        if let Some((oldest_key, _)) = cache.iter()
            .min_by_key(|(_, cached)| cached.last_accessed) {
            let oldest_key = oldest_key.clone();
            cache.remove(&oldest_key);
            
            // Update statistics
            let mut stats = self.cache_stats.write().await;
            stats.eviction_count += 1;
        }
    }
}
```

## Success Criteria
- Chain caching operations work correctly
- TTL and eviction logic functions

## Next Task
13e_cache_property_operations.md