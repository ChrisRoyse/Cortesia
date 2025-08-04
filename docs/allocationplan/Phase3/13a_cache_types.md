# Task 13a: Create Cache Data Types

**Time**: 5 minutes (1 min read, 3 min implement, 1 min verify)
**Dependencies**: 12l_integration_with_inheritance.md
**Stage**: Inheritance System

## Objective
Create basic cache data structures for inheritance caching.

## Implementation
Create `src/inheritance/cache/cache_types.rs`:

```rust
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc};
use crate::inheritance::hierarchy_types::InheritanceChain;
use crate::inheritance::property_types::ResolvedProperties;

#[derive(Debug, Clone)]
pub struct CachedInheritanceChain {
    pub chain: InheritanceChain,
    pub cached_at: DateTime<Utc>,
    pub access_count: u32,
    pub last_accessed: DateTime<Utc>,
    pub ttl_expires_at: DateTime<Utc>,
    pub dependency_concepts: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CachedPropertyResolution {
    pub resolved_properties: ResolvedProperties,
    pub cached_at: DateTime<Utc>,
    pub access_count: u32,
    pub last_accessed: DateTime<Utc>,
    pub cache_size_bytes: usize,
}

#[derive(Debug, Default)]
pub struct InheritanceCacheStats {
    pub total_entries: usize,
    pub memory_usage_bytes: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub invalidation_count: u64,
    pub average_access_time_ms: f64,
}

#[derive(Debug)]
pub struct CacheEvictionCandidate {
    pub concept_id: String,
    pub last_accessed: DateTime<Utc>,
    pub access_frequency: f64,
    pub cache_size_bytes: usize,
    pub eviction_score: f64,
}
```

## Success Criteria
- All cache types compile without errors
- Structures are properly defined

## Next Task
13b_cache_config.md