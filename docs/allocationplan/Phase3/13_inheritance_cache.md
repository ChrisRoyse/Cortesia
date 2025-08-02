# Task 13: Inheritance Cache System
**Estimated Time**: 12-18 minutes
**Dependencies**: 12_property_inheritance.md
**Stage**: Inheritance System

## Objective
Build a high-performance caching system for inheritance chains and property resolutions to minimize database queries and optimize property inheritance operations.

## Specific Requirements

### 1. Multi-Level Cache Architecture
- Inheritance chain caching with TTL management
- Property resolution caching with dependency tracking
- Cache invalidation on hierarchy changes
- Memory-efficient cache storage with LRU eviction

### 2. Cache Consistency Management
- Automatic cache invalidation on structural changes
- Cascading invalidation for dependent chains
- Cache warming strategies for frequently accessed concepts
- Cache coherence across concurrent operations

### 3. Performance Optimization
- Batch cache operations for multiple concepts
- Predictive caching for inheritance patterns
- Cache hit ratio monitoring and optimization
- Memory usage tracking and bounds enforcement

## Implementation Steps

### 1. Create Cache Data Structures
```rust
// src/inheritance/cache/cache_types.rs
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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
    pub resolution_metadata: ResolutionMetadata,
    pub cached_at: DateTime<Utc>,
    pub access_count: u32,
    pub last_accessed: DateTime<Utc>,
    pub cache_size_bytes: usize,
}

#[derive(Debug)]
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

### 2. Implement Core Cache Manager
```rust
// src/inheritance/cache/inheritance_cache_manager.rs
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
    
    pub async fn get_inheritance_chain(
        &self,
        concept_id: &str,
    ) -> Option<InheritanceChain> {
        let access_start = Instant::now();
        
        let mut cache = self.chain_cache.write().await;
        let mut stats = self.cache_stats.write().await;
        
        if let Some(cached_entry) = cache.get_mut(concept_id) {
            // Check TTL expiration
            if Utc::now() < cached_entry.ttl_expires_at {
                // Update access statistics
                cached_entry.access_count += 1;
                cached_entry.last_accessed = Utc::now();
                
                stats.hit_count += 1;
                stats.average_access_time_ms = self.update_average_time(
                    stats.average_access_time_ms,
                    access_start.elapsed().as_millis() as f64,
                    stats.hit_count + stats.miss_count,
                );
                
                return Some(cached_entry.chain.clone());
            } else {
                // TTL expired, remove entry
                cache.remove(concept_id);
                stats.eviction_count += 1;
            }
        }
        
        stats.miss_count += 1;
        None
    }
    
    pub async fn store_inheritance_chain(
        &self,
        concept_id: String,
        chain: InheritanceChain,
    ) -> Result<(), CacheError> {
        let storage_start = Instant::now();
        
        // Check memory limits before storing
        if self.would_exceed_memory_limit(&chain).await {
            self.evict_least_recently_used().await?;
        }
        
        let now = Utc::now();
        let cached_entry = CachedInheritanceChain {
            chain: chain.clone(),
            cached_at: now,
            access_count: 0,
            last_accessed: now,
            ttl_expires_at: now + chrono::Duration::seconds(self.config.chain_ttl_seconds),
            dependency_concepts: self.extract_dependency_concepts(&chain),
        };
        
        // Update dependency graph
        self.update_dependency_graph(&concept_id, &cached_entry.dependency_concepts).await;
        
        // Store in cache
        self.chain_cache.write().await.insert(concept_id.clone(), cached_entry);
        
        // Update statistics
        let mut stats = self.cache_stats.write().await;
        stats.total_entries += 1;
        stats.memory_usage_bytes += self.estimate_chain_memory_size(&chain);
        
        Ok(())
    }
    
    pub async fn get_property_resolution(
        &self,
        concept_id: &str,
        include_inherited: bool,
    ) -> Option<ResolvedProperties> {
        let cache_key = format!("{}:{}", concept_id, include_inherited);
        let access_start = Instant::now();
        
        let mut cache = self.property_cache.write().await;
        let mut stats = self.cache_stats.write().await;
        
        if let Some(cached_resolution) = cache.get_mut(&cache_key) {
            // Update access statistics
            cached_resolution.access_count += 1;
            cached_resolution.last_accessed = Utc::now();
            
            stats.hit_count += 1;
            stats.average_access_time_ms = self.update_average_time(
                stats.average_access_time_ms,
                access_start.elapsed().as_millis() as f64,
                stats.hit_count + stats.miss_count,
            );
            
            return Some(cached_resolution.resolved_properties.clone());
        }
        
        stats.miss_count += 1;
        None
    }
    
    pub async fn store_property_resolution(
        &self,
        concept_id: String,
        include_inherited: bool,
        resolved_properties: ResolvedProperties,
        resolution_metadata: ResolutionMetadata,
    ) -> Result<(), CacheError> {
        let cache_key = format!("{}:{}", concept_id, include_inherited);
        
        // Check memory limits
        let cache_size = self.estimate_property_resolution_size(&resolved_properties);
        if self.would_exceed_memory_limit_for_properties(cache_size).await {
            self.evict_property_resolutions().await?;
        }
        
        let now = Utc::now();
        let cached_resolution = CachedPropertyResolution {
            resolved_properties,
            resolution_metadata,
            cached_at: now,
            access_count: 0,
            last_accessed: now,
            cache_size_bytes: cache_size,
        };
        
        self.property_cache.write().await.insert(cache_key, cached_resolution);
        
        // Update memory usage
        let mut stats = self.cache_stats.write().await;
        stats.memory_usage_bytes += cache_size;
        
        Ok(())
    }
    
    pub async fn invalidate_concept(&self, concept_id: &str) {
        let invalidation_start = Instant::now();
        
        // Remove direct cache entries
        self.chain_cache.write().await.remove(concept_id);
        
        // Remove property resolution entries
        let mut property_cache = self.property_cache.write().await;
        let keys_to_remove: Vec<String> = property_cache
            .keys()
            .filter(|key| key.starts_with(&format!("{}:", concept_id)))
            .cloned()
            .collect();
        
        for key in keys_to_remove {
            property_cache.remove(&key);
        }
        
        // Invalidate dependent concepts using dependency graph
        if let Some(dependent_concepts) = self.dependency_graph.read().await.get(concept_id) {
            for dependent_concept in dependent_concepts {
                self.invalidate_concept_recursive(dependent_concept).await;
            }
        }
        
        // Update statistics
        let mut stats = self.cache_stats.write().await;
        stats.invalidation_count += 1;
    }
    
    async fn evict_least_recently_used(&self) -> Result<(), CacheError> {
        let mut chain_cache = self.chain_cache.write().await;
        let mut eviction_candidates = Vec::new();
        
        // Collect eviction candidates with scoring
        for (concept_id, cached_chain) in chain_cache.iter() {
            let recency_score = self.calculate_recency_score(cached_chain.last_accessed);
            let frequency_score = self.calculate_frequency_score(cached_chain.access_count);
            let size_score = self.calculate_size_score(&cached_chain.chain);
            
            let eviction_score = (recency_score * 0.4) + (frequency_score * 0.3) + (size_score * 0.3);
            
            eviction_candidates.push(CacheEvictionCandidate {
                concept_id: concept_id.clone(),
                last_accessed: cached_chain.last_accessed,
                access_frequency: cached_chain.access_count as f64,
                cache_size_bytes: self.estimate_chain_memory_size(&cached_chain.chain),
                eviction_score,
            });
        }
        
        // Sort by eviction score (higher score = better candidate for eviction)
        eviction_candidates.sort_by(|a, b| b.eviction_score.partial_cmp(&a.eviction_score).unwrap());
        
        // Evict top candidates until memory is under limit
        let target_evictions = (eviction_candidates.len() as f64 * 0.1).ceil() as usize; // Evict 10%
        for candidate in eviction_candidates.iter().take(target_evictions) {
            chain_cache.remove(&candidate.concept_id);
            
            // Update statistics
            let mut stats = self.cache_stats.write().await;
            stats.eviction_count += 1;
            stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(candidate.cache_size_bytes);
        }
        
        Ok(())
    }
}
```

### 3. Implement Cache Warming Strategy
```rust
// src/inheritance/cache/cache_warmer.rs
pub struct CacheWarmer {
    cache_manager: Arc<InheritanceCacheManager>,
    hierarchy_manager: Arc<InheritanceHierarchyManager>,
    property_engine: Arc<PropertyInheritanceEngine>,
    warming_stats: Arc<RwLock<CacheWarmingStats>>,
}

impl CacheWarmer {
    pub async fn warm_frequently_accessed_concepts(&self) -> Result<CacheWarmingResult, WarmingError> {
        let warming_start = Instant::now();
        
        // Get access patterns from cache statistics
        let access_patterns = self.analyze_access_patterns().await?;
        
        // Identify concepts to warm
        let concepts_to_warm = self.identify_warming_candidates(&access_patterns).await?;
        
        let mut warmed_count = 0;
        let mut failed_count = 0;
        
        // Warm inheritance chains
        for concept_id in &concepts_to_warm {
            match self.warm_inheritance_chain(concept_id).await {
                Ok(_) => warmed_count += 1,
                Err(_) => failed_count += 1,
            }
        }
        
        // Warm property resolutions
        for concept_id in &concepts_to_warm {
            match self.warm_property_resolutions(concept_id).await {
                Ok(_) => {},
                Err(_) => failed_count += 1,
            }
        }
        
        let warming_time = warming_start.elapsed();
        
        // Update warming statistics
        let mut stats = self.warming_stats.write().await;
        stats.total_warming_operations += 1;
        stats.concepts_warmed += warmed_count;
        stats.warming_failures += failed_count;
        stats.total_warming_time += warming_time;
        
        Ok(CacheWarmingResult {
            concepts_warmed: warmed_count,
            warming_failures: failed_count,
            warming_time,
        })
    }
    
    async fn warm_inheritance_chain(&self, concept_id: &str) -> Result<(), WarmingError> {
        // Check if already cached
        if self.cache_manager.get_inheritance_chain(concept_id).await.is_some() {
            return Ok(());
        }
        
        // Load from database and cache
        let chain = self.hierarchy_manager.get_inheritance_chain(concept_id).await?;
        self.cache_manager.store_inheritance_chain(concept_id.to_string(), chain).await?;
        
        Ok(())
    }
    
    async fn warm_property_resolutions(&self, concept_id: &str) -> Result<(), WarmingError> {
        // Warm both inherited and non-inherited property resolutions
        for include_inherited in [true, false] {
            if self.cache_manager.get_property_resolution(concept_id, include_inherited).await.is_none() {
                let properties = self.property_engine.resolve_properties(concept_id, include_inherited).await?;
                let metadata = ResolutionMetadata::new();
                
                self.cache_manager.store_property_resolution(
                    concept_id.to_string(),
                    include_inherited,
                    properties,
                    metadata,
                ).await?;
            }
        }
        
        Ok(())
    }
    
    async fn identify_warming_candidates(&self, access_patterns: &AccessPatterns) -> Result<Vec<String>, AnalysisError> {
        let mut candidates = Vec::new();
        
        // Sort concepts by access frequency and recency
        let mut concept_scores: Vec<(String, f64)> = access_patterns
            .concept_access_frequency
            .iter()
            .map(|(concept_id, frequency)| {
                let recency_bonus = access_patterns
                    .recent_access_times
                    .get(concept_id)
                    .map(|time| {
                        let hours_since_access = Utc::now()
                            .signed_duration_since(*time)
                            .num_hours() as f64;
                        1.0 / (1.0 + hours_since_access * 0.1) // Exponential decay
                    })
                    .unwrap_or(0.0);
                
                let score = (*frequency as f64) * (1.0 + recency_bonus);
                (concept_id.clone(), score)
            })
            .collect();
        
        concept_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top candidates (up to 100 concepts)
        candidates.extend(
            concept_scores
                .iter()
                .take(100)
                .map(|(concept_id, _)| concept_id.clone())
        );
        
        Ok(candidates)
    }
}
```

## Acceptance Criteria

### Functional Requirements
- [ ] Inheritance chain caching with configurable TTL
- [ ] Property resolution caching with dependency tracking
- [ ] Automatic cache invalidation on hierarchy changes
- [ ] LRU-based cache eviction with intelligent scoring
- [ ] Cache warming for frequently accessed concepts

### Performance Requirements
- [ ] Cache hit ratio > 85% for inheritance chain queries
- [ ] Cache hit ratio > 80% for property resolution queries
- [ ] Cache lookup time < 1ms
- [ ] Memory usage stays within configured bounds
- [ ] Eviction operations complete within 5ms

### Testing Requirements
- [ ] Unit tests for cache data structures
- [ ] Integration tests for cache invalidation scenarios
- [ ] Performance tests for cache operations under load
- [ ] Memory usage tests for large cache scenarios

## Validation Steps

1. **Test cache storage and retrieval**:
   ```rust
   let cache_manager = InheritanceCacheManager::new(config);
   cache_manager.store_inheritance_chain(concept_id, chain).await?;
   let cached_chain = cache_manager.get_inheritance_chain(concept_id).await;
   assert!(cached_chain.is_some());
   ```

2. **Test cache invalidation**:
   ```rust
   cache_manager.invalidate_concept("modified_concept").await;
   // Verify dependent concepts are also invalidated
   ```

3. **Run cache performance tests**:
   ```bash
   cargo test inheritance_cache_tests --release
   ```

## Files to Create/Modify
- `src/inheritance/cache/cache_types.rs` - Cache data structures
- `src/inheritance/cache/inheritance_cache_manager.rs` - Core cache manager
- `src/inheritance/cache/cache_warmer.rs` - Cache warming strategies
- `src/inheritance/cache/mod.rs` - Module exports
- `tests/inheritance/cache_tests.rs` - Cache test suite

## Success Metrics
- Cache hit ratio > 85% for inheritance operations
- Memory usage efficiency > 90%
- Cache invalidation accuracy: 100%
- Average cache operation time < 1ms

## Next Task
Upon completion, proceed to **14_exception_handling.md** to implement exception and override system for inheritance.