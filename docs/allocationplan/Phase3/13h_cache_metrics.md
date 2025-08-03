# Task 13h: Implement Cache Metrics

**Time**: 5 minutes
**Dependencies**: 13g_cache_warming.md
**Stage**: Inheritance System

## Objective
Add comprehensive cache metrics and monitoring.

## Implementation
Add to `src/inheritance/cache/inheritance_cache_manager.rs`:

```rust
impl InheritanceCacheManager {
    pub async fn get_detailed_metrics(&self) -> DetailedCacheMetrics {
        let stats = self.cache_stats.read().await;
        let chain_cache = self.chain_cache.read().await;
        let property_cache = self.property_cache.read().await;
        
        let total_requests = stats.hit_count + stats.miss_count;
        let hit_rate = if total_requests > 0 {
            stats.hit_count as f64 / total_requests as f64
        } else {
            0.0
        };
        
        DetailedCacheMetrics {
            basic_stats: stats.clone(),
            hit_rate,
            chain_cache_size: chain_cache.len(),
            property_cache_size: property_cache.len(),
            memory_usage_mb: stats.memory_usage_bytes as f64 / 1024.0 / 1024.0,
            average_chain_depth: self.calculate_average_chain_depth(&chain_cache),
            most_accessed_concepts: self.get_most_accessed_concepts(&chain_cache, &property_cache),
        }
    }

    fn calculate_average_chain_depth(&self, chain_cache: &std::collections::HashMap<String, CachedInheritanceChain>) -> f64 {
        if chain_cache.is_empty() {
            return 0.0;
        }
        
        let total_depth: u32 = chain_cache.values()
            .map(|cached| cached.chain.total_depth)
            .sum();
        
        total_depth as f64 / chain_cache.len() as f64
    }

    fn get_most_accessed_concepts(
        &self,
        chain_cache: &std::collections::HashMap<String, CachedInheritanceChain>,
        property_cache: &std::collections::HashMap<String, CachedPropertyResolution>,
    ) -> Vec<(String, u32)> {
        let mut access_counts = std::collections::HashMap::new();
        
        // Collect access counts from chain cache
        for (concept_id, cached) in chain_cache.iter() {
            access_counts.insert(concept_id.clone(), cached.access_count);
        }
        
        // Collect access counts from property cache
        for (cache_key, cached) in property_cache.iter() {
            if let Some(concept_id) = cache_key.split(':').next() {
                let entry = access_counts.entry(concept_id.to_string()).or_insert(0);
                *entry += cached.access_count;
            }
        }
        
        let mut sorted: Vec<_> = access_counts.into_iter().collect();
        sorted.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        sorted.into_iter().take(10).collect()
    }

    pub async fn record_cache_operation(&self, operation: CacheOperation, concept_id: &str, execution_time_ms: f64, cache_hit: bool) {
        let metric = CacheMetrics {
            operation,
            concept_id: concept_id.to_string(),
            execution_time_ms,
            cache_hit,
            timestamp: chrono::Utc::now(),
        };
        
        // Update running averages
        {
            let mut stats = self.cache_stats.write().await;
            let total_operations = stats.hit_count + stats.miss_count + 1;
            stats.average_access_time_ms = (stats.average_access_time_ms * (total_operations - 1) as f64 + execution_time_ms) / total_operations as f64;
        }
    }
}

#[derive(Debug)]
pub struct DetailedCacheMetrics {
    pub basic_stats: InheritanceCacheStats,
    pub hit_rate: f64,
    pub chain_cache_size: usize,
    pub property_cache_size: usize,
    pub memory_usage_mb: f64,
    pub average_chain_depth: f64,
    pub most_accessed_concepts: Vec<(String, u32)>,
}
```

## Success Criteria
- Detailed metrics are properly calculated
- Hit rates and performance data are accurate

## Next Task
13i_cache_integration.md