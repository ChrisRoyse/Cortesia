# Micro Phase 1.5: Property Cache System

**Estimated Time**: 35 minutes
**Dependencies**: Micro 1.4 (Property Resolution Engine)
**Objective**: Implement a high-performance cache for property resolution results

## Task Description

Create a thread-safe, LRU-based caching system that dramatically improves property resolution performance by caching resolution results.

## Deliverables

Create `src/properties/cache.rs` with:

1. **PropertyCache struct**: Thread-safe LRU cache
2. **Cache invalidation**: Smart invalidation when properties change
3. **Cache statistics**: Hit/miss rates, eviction tracking
4. **Memory management**: Bounded cache size with configurable limits
5. **TTL support**: Time-based cache expiration

## Success Criteria

- [ ] Cache improves resolution performance > 10x for repeated lookups
- [ ] Thread-safe concurrent access without deadlocks
- [ ] LRU eviction policy works correctly
- [ ] Cache hit rate > 80% for typical workloads
- [ ] Memory usage stays within configured bounds
- [ ] Cache invalidation works correctly when hierarchy changes

## Implementation Requirements

```rust
pub struct PropertyCache {
    cache: Arc<DashMap<CacheKey, CacheEntry>>,
    capacity: usize,
    ttl: Duration,
    stats: CacheStatistics,
    eviction_policy: EvictionPolicy,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    node_id: NodeId,
    property_name: String,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    value: Option<PropertyValue>,
    source_node: Option<NodeId>,
    created_at: Instant,
    last_accessed: AtomicU64,
    access_count: AtomicU32,
}

pub struct CacheStatistics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub invalidations: AtomicU64,
}

impl PropertyCache {
    pub fn new(capacity: usize, ttl: Duration) -> Self;
    
    pub fn get(&self, node: NodeId, property: &str) -> Option<CacheEntry>;
    
    pub fn insert(&self, node: NodeId, property: &str, value: Option<PropertyValue>, source: Option<NodeId>);
    
    pub fn invalidate_node(&self, node: NodeId);
    
    pub fn invalidate_property(&self, property: &str);
    
    pub fn clear(&self);
    
    pub fn get_statistics(&self) -> CacheStatistics;
    
    pub fn hit_rate(&self) -> f64;
}
```

## Test Requirements

Must pass caching performance and correctness tests:
```rust
#[test]
fn test_cache_basic_operations() {
    let cache = PropertyCache::new(1000, Duration::from_secs(60));
    let node = NodeId(1);
    
    // Cache miss
    assert!(cache.get(node, "test_prop").is_none());
    
    // Insert and hit
    cache.insert(node, "test_prop", Some(PropertyValue::String("value".to_string())), Some(node));
    let entry = cache.get(node, "test_prop").unwrap();
    assert_eq!(entry.value, Some(PropertyValue::String("value".to_string())));
    
    let stats = cache.get_statistics();
    assert_eq!(stats.misses.load(Ordering::Relaxed), 1);
    assert_eq!(stats.hits.load(Ordering::Relaxed), 1);
}

#[test]
fn test_cache_performance_improvement() {
    let hierarchy = create_deep_hierarchy(10);
    let resolver = PropertyResolver::new(ResolutionStrategy::DepthFirst);
    let cache = PropertyCache::new(10000, Duration::from_secs(60));
    
    let leaf = NodeId(9);
    
    // First lookup (cache miss)
    let start = Instant::now();
    let result1 = resolver.resolve_property(&hierarchy, leaf, "root_property");
    let uncached_time = start.elapsed();
    
    // Cache the result
    cache.insert(leaf, "root_property", result1.value.clone(), result1.source_node);
    
    // Second lookup (cache hit)
    let start = Instant::now();
    let cached_result = cache.get(leaf, "root_property");
    let cached_time = start.elapsed();
    
    assert!(cached_result.is_some());
    assert!(cached_time < uncached_time / 10); // >10x improvement
}

#[test]
fn test_cache_invalidation() {
    let cache = PropertyCache::new(1000, Duration::from_secs(60));
    let node = NodeId(1);
    
    cache.insert(node, "test_prop", Some(PropertyValue::String("value".to_string())), Some(node));
    assert!(cache.get(node, "test_prop").is_some());
    
    // Invalidate node
    cache.invalidate_node(node);
    assert!(cache.get(node, "test_prop").is_none());
}

#[test]
fn test_lru_eviction() {
    let cache = PropertyCache::new(2, Duration::from_secs(60)); // Small capacity
    
    // Fill cache
    cache.insert(NodeId(1), "prop1", Some(PropertyValue::String("val1".to_string())), None);
    cache.insert(NodeId(2), "prop2", Some(PropertyValue::String("val2".to_string())), None);
    
    // Access first entry to make it recently used
    cache.get(NodeId(1), "prop1");
    
    // Insert third entry, should evict second entry
    cache.insert(NodeId(3), "prop3", Some(PropertyValue::String("val3".to_string())), None);
    
    assert!(cache.get(NodeId(1), "prop1").is_some()); // Still cached
    assert!(cache.get(NodeId(2), "prop2").is_none());  // Evicted
    assert!(cache.get(NodeId(3), "prop3").is_some());  // Newly cached
}

#[test]
fn test_concurrent_cache_access() {
    let cache = Arc::new(PropertyCache::new(10000, Duration::from_secs(60)));
    let num_threads = 8;
    let operations_per_thread = 1000;
    
    let handles: Vec<_> = (0..num_threads).map(|thread_id| {
        let cache = cache.clone();
        thread::spawn(move || {
            for i in 0..operations_per_thread {
                let node = NodeId(thread_id as u64 * 1000 + i as u64);
                let value = PropertyValue::String(format!("value_{}_{}", thread_id, i));
                
                cache.insert(node, "test_prop", Some(value.clone()), Some(node));
                let retrieved = cache.get(node, "test_prop");
                assert!(retrieved.is_some());
                assert_eq!(retrieved.unwrap().value, Some(value));
            }
        })
    }).collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let stats = cache.get_statistics();
    assert_eq!(stats.hits.load(Ordering::Relaxed), num_threads * operations_per_thread);
}
```

## File Location
`src/properties/cache.rs`

## Next Micro Phase
After completion, proceed to Micro 1.6: Multiple Inheritance DAG Support