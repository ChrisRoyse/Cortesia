# Task 13c: Create Cache Manager Structure

**Time**: 6 minutes
**Dependencies**: 13b_cache_config.md
**Stage**: Inheritance System

## Objective
Create the main inheritance cache manager structure.

## Implementation
Create `src/inheritance/cache/inheritance_cache_manager.rs`:

```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, VecDeque};
use crate::inheritance::cache::cache_types::*;

pub struct InheritanceCacheManager {
    chain_cache: Arc<RwLock<HashMap<String, CachedInheritanceChain>>>,
    property_cache: Arc<RwLock<HashMap<String, CachedPropertyResolution>>>,
    dependency_graph: Arc<RwLock<HashMap<String, Vec<String>>>>,
    cache_stats: Arc<RwLock<InheritanceCacheStats>>,
    config: CacheConfig,
    eviction_queue: Arc<RwLock<VecDeque<CacheEvictionCandidate>>>,
}

impl InheritanceCacheManager {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            chain_cache: Arc::new(RwLock::new(HashMap::new())),
            property_cache: Arc::new(RwLock::new(HashMap::new())),
            dependency_graph: Arc::new(RwLock::new(HashMap::new())),
            cache_stats: Arc::new(RwLock::new(InheritanceCacheStats::default())),
            config,
            eviction_queue: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    pub async fn get_cache_statistics(&self) -> InheritanceCacheStats {
        self.cache_stats.read().await.clone()
    }

    pub fn get_config(&self) -> &CacheConfig {
        &self.config
    }
}
```

## Success Criteria
- Cache manager structure compiles
- Constructor and basic methods work

## Next Task
13d_cache_chain_operations.md