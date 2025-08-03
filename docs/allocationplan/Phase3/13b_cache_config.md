# Task 13b: Create Cache Configuration

**Time**: 3 minutes
**Dependencies**: 13a_cache_types.md
**Stage**: Inheritance System

## Objective
Create configuration structure for cache settings.

## Implementation
Add to `src/inheritance/cache/cache_types.rs`:

```rust
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_chain_cache_size: usize,
    pub max_property_cache_size: usize,
    pub ttl_minutes: i64,
    pub max_memory_usage_mb: usize,
    pub eviction_threshold: f64,
    pub enable_predictive_caching: bool,
    pub cache_warming_enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_chain_cache_size: 5000,
            max_property_cache_size: 10000,
            ttl_minutes: 30,
            max_memory_usage_mb: 100,
            eviction_threshold: 0.8,
            enable_predictive_caching: true,
            cache_warming_enabled: false,
        }
    }
}

#[derive(Debug)]
pub enum CacheOperation {
    Get,
    Put,
    Invalidate,
    Evict,
}

#[derive(Debug)]
pub struct CacheMetrics {
    pub operation: CacheOperation,
    pub concept_id: String,
    pub execution_time_ms: f64,
    pub cache_hit: bool,
    pub timestamp: DateTime<Utc>,
}
```

## Success Criteria
- Configuration structure compiles
- Default values are reasonable

## Next Task
13c_cache_manager_struct.md