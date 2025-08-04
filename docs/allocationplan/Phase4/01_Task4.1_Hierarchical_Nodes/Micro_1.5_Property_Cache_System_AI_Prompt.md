# AI Prompt: Micro Phase 1.5 - Property Cache System

You are tasked with implementing a high-performance cache for property resolution results. Your goal is to create `src/properties/cache.rs` with a thread-safe, LRU-based caching system that dramatically improves property resolution performance.

## Your Task
Implement the `PropertyCache` struct with LRU eviction, TTL support, smart invalidation, and comprehensive statistics tracking to accelerate repeated property lookups.

## Specific Requirements
1. Create `src/properties/cache.rs` with thread-safe PropertyCache using concurrent data structures
2. Implement LRU (Least Recently Used) eviction policy with configurable capacity
3. Add TTL (Time To Live) support for automatic cache expiration
4. Implement smart cache invalidation when properties or nodes change
5. Provide detailed cache statistics (hit/miss rates, evictions, etc.)
6. Ensure memory usage stays within configured bounds
7. Optimize for concurrent access patterns

## Expected Code Structure
You must implement these exact signatures:

```rust
use dashmap::DashMap;
use std::sync::{Arc, atomic::{AtomicU32, AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use crate::hierarchy::node::NodeId;
use crate::properties::value::PropertyValue;

#[derive(Debug, Clone)]
pub enum EvictionPolicy {
    LRU,
    LFU, // Least Frequently Used
    TTL, // Time To Live only
}

pub struct PropertyCache {
    cache: Arc<DashMap<CacheKey, CacheEntry>>,
    capacity: usize,
    ttl: Duration,
    stats: Arc<CacheStatistics>,
    eviction_policy: EvictionPolicy,
    access_order: Arc<DashMap<CacheKey, u64>>, // For LRU tracking
    next_access_id: AtomicU64,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    node_id: NodeId,
    property_name: String,
}

impl CacheKey {
    fn new(node_id: NodeId, property_name: &str) -> Self {
        Self {
            node_id,
            property_name: property_name.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
struct CacheEntry {
    value: Option<PropertyValue>,
    source_node: Option<NodeId>,
    created_at: Instant,
    last_accessed: AtomicU64, // Timestamp in nanos since epoch
    access_count: AtomicU32,
}

impl CacheEntry {
    fn new(value: Option<PropertyValue>, source_node: Option<NodeId>) -> Self {
        Self {
            value,
            source_node,
            created_at: Instant::now(),
            last_accessed: AtomicU64::new(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64
            ),
            access_count: AtomicU32::new(1),
        }
    }
    
    fn record_access(&self) {
        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_accessed.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Relaxed
        );
    }
    
    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }
}

#[derive(Debug, Default)]
pub struct CacheStatistics {
    pub hits: AtomicU64,
    pub misses: AtomicU64,
    pub evictions: AtomicU64,
    pub invalidations: AtomicU64,
    pub expired: AtomicU64,
}

impl CacheStatistics {
    fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_invalidation(&self) {
        self.invalidations.fetch_add(1, Ordering::Relaxed);
    }
    
    fn record_expiration(&self) {
        self.expired.fetch_add(1, Ordering::Relaxed);
    }
}

impl PropertyCache {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            capacity,
            ttl,
            stats: Arc::new(CacheStatistics::default()),
            eviction_policy: EvictionPolicy::LRU,
            access_order: Arc::new(DashMap::new()),
            next_access_id: AtomicU64::new(1),
        }
    }
    
    pub fn with_eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }
    
    pub fn get(&self, node: NodeId, property: &str) -> Option<CacheEntry> {
        let key = CacheKey::new(node, property);
        
        if let Some(entry) = self.cache.get(&key) {
            // Check if expired
            if entry.is_expired(self.ttl) {
                self.stats.record_expiration();
                self.cache.remove(&key);
                self.access_order.remove(&key);
                self.stats.record_miss();
                return None;
            }
            
            // Record access for LRU
            entry.record_access();
            let access_id = self.next_access_id.fetch_add(1, Ordering::Relaxed);
            self.access_order.insert(key, access_id);
            
            self.stats.record_hit();
            Some(entry.clone())
        } else {
            self.stats.record_miss();
            None
        }
    }
    
    pub fn insert(&self, node: NodeId, property: &str, value: Option<PropertyValue>, source: Option<NodeId>) {
        let key = CacheKey::new(node, property);
        let entry = CacheEntry::new(value, source);
        
        // Check if we need to evict
        if self.cache.len() >= self.capacity {
            self.evict_one();
        }
        
        let access_id = self.next_access_id.fetch_add(1, Ordering::Relaxed);
        self.cache.insert(key.clone(), entry);
        self.access_order.insert(key, access_id);
    }
    
    pub fn invalidate_node(&self, node: NodeId) {
        let keys_to_remove: Vec<CacheKey> = self.cache
            .iter()
            .filter(|kv| kv.key().node_id == node)
            .map(|kv| kv.key().clone())
            .collect();
        
        for key in keys_to_remove {
            self.cache.remove(&key);
            self.access_order.remove(&key);
            self.stats.record_invalidation();
        }
    }
    
    pub fn invalidate_property(&self, property: &str) {
        let keys_to_remove: Vec<CacheKey> = self.cache
            .iter()
            .filter(|kv| kv.key().property_name == property)
            .map(|kv| kv.key().clone())
            .collect();
        
        for key in keys_to_remove {
            self.cache.remove(&key);
            self.access_order.remove(&key);
            self.stats.record_invalidation();
        }
    }
    
    pub fn clear(&self) {
        let count = self.cache.len();
        self.cache.clear();
        self.access_order.clear();
        self.stats.invalidations.fetch_add(count as u64, Ordering::Relaxed);
    }
    
    pub fn get_statistics(&self) -> CacheStatistics {
        CacheStatistics {
            hits: AtomicU64::new(self.stats.hits.load(Ordering::Relaxed)),
            misses: AtomicU64::new(self.stats.misses.load(Ordering::Relaxed)),
            evictions: AtomicU64::new(self.stats.evictions.load(Ordering::Relaxed)),
            invalidations: AtomicU64::new(self.stats.invalidations.load(Ordering::Relaxed)),
            expired: AtomicU64::new(self.stats.expired.load(Ordering::Relaxed)),
        }
    }
    
    pub fn hit_rate(&self) -> f64 {
        let hits = self.stats.hits.load(Ordering::Relaxed) as f64;
        let total = hits + self.stats.misses.load(Ordering::Relaxed) as f64;
        if total == 0.0 { 0.0 } else { hits / total }
    }
    
    pub fn size(&self) -> usize {
        self.cache.len()
    }
    
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    fn evict_one(&self) {
        match self.eviction_policy {
            EvictionPolicy::LRU => self.evict_lru(),
            EvictionPolicy::LFU => self.evict_lfu(),
            EvictionPolicy::TTL => self.evict_expired(),
        }
    }
    
    fn evict_lru(&self) {
        // Find the key with the smallest access ID (oldest access)
        if let Some(oldest_key) = self.access_order
            .iter()
            .min_by_key(|kv| *kv.value())
            .map(|kv| kv.key().clone())
        {
            self.cache.remove(&oldest_key);
            self.access_order.remove(&oldest_key);
            self.stats.record_eviction();
        }
    }
    
    fn evict_lfu(&self) {
        // Find the key with the smallest access count
        if let Some(lfu_key) = self.cache
            .iter()
            .min_by_key(|kv| kv.value().access_count.load(Ordering::Relaxed))
            .map(|kv| kv.key().clone())
        {
            self.cache.remove(&lfu_key);
            self.access_order.remove(&lfu_key);
            self.stats.record_eviction();
        }
    }
    
    fn evict_expired(&self) {
        let now = Instant::now();
        let expired_keys: Vec<CacheKey> = self.cache
            .iter()
            .filter(|kv| kv.value().is_expired(self.ttl))
            .map(|kv| kv.key().clone())
            .collect();
        
        for key in expired_keys {
            self.cache.remove(&key);
            self.access_order.remove(&key);
            self.stats.record_expiration();
        }
    }
}

impl Default for PropertyCache {
    fn default() -> Self {
        Self::new(10000, Duration::from_secs(300)) // 10k entries, 5 minute TTL
    }
}
```

## Success Criteria (You must verify these)
- [ ] Cache improves resolution performance > 10x for repeated lookups
- [ ] Thread-safe concurrent access without deadlocks or data races
- [ ] LRU eviction policy works correctly under capacity pressure
- [ ] Cache hit rate > 80% for typical workloads with repeated access
- [ ] Memory usage stays within configured bounds
- [ ] Cache invalidation works correctly when hierarchy changes
- [ ] TTL expiration removes stale entries automatically
- [ ] Statistics accurately track all operations
- [ ] Code compiles without warnings
- [ ] All tests pass

## Test Requirements
You must implement and verify these tests pass:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::sync::Arc;

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
        assert!(cache.hit_rate() > 0.4); // 1 hit out of 2 total accesses
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
        
        // Test property invalidation
        cache.insert(NodeId(1), "shared_prop", Some(PropertyValue::Boolean(true)), None);
        cache.insert(NodeId(2), "shared_prop", Some(PropertyValue::Boolean(false)), None);
        
        cache.invalidate_property("shared_prop");
        assert!(cache.get(NodeId(1), "shared_prop").is_none());
        assert!(cache.get(NodeId(2), "shared_prop").is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let cache = PropertyCache::new(2, Duration::from_secs(60)); // Small capacity
        
        // Fill cache
        cache.insert(NodeId(1), "prop1", Some(PropertyValue::String("val1".to_string())), None);
        cache.insert(NodeId(2), "prop2", Some(PropertyValue::String("val2".to_string())), None);
        
        // Access first entry to make it recently used
        cache.get(NodeId(1), "prop1");
        
        // Insert third entry, should evict second entry (LRU)
        cache.insert(NodeId(3), "prop3", Some(PropertyValue::String("val3".to_string())), None);
        
        assert!(cache.get(NodeId(1), "prop1").is_some()); // Still cached (recently used)
        assert!(cache.get(NodeId(2), "prop2").is_none());  // Evicted (not recently used)
        assert!(cache.get(NodeId(3), "prop3").is_some());  // Newly cached
        
        let stats = cache.get_statistics();
        assert!(stats.evictions.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_ttl_expiration() {
        let cache = PropertyCache::new(1000, Duration::from_millis(10)); // Very short TTL
        let node = NodeId(1);
        
        cache.insert(node, "test_prop", Some(PropertyValue::String("value".to_string())), Some(node));
        assert!(cache.get(node, "test_prop").is_some());
        
        // Wait for expiration
        thread::sleep(Duration::from_millis(20));
        
        assert!(cache.get(node, "test_prop").is_none()); // Should be expired
        
        let stats = cache.get_statistics();
        assert!(stats.expired.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_concurrent_cache_access() {
        let cache = Arc::new(PropertyCache::new(10000, Duration::from_secs(60)));
        let num_threads = 8;
        let operations_per_thread = 100; // Reduced for test speed
        
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
        assert_eq!(stats.hits.load(Ordering::Relaxed), (num_threads * operations_per_thread) as u64);
        assert!(cache.hit_rate() > 0.99); // Should be nearly 100% hit rate
    }

    #[test]
    fn test_memory_bounded() {
        let capacity = 100;
        let cache = PropertyCache::new(capacity, Duration::from_secs(60));
        
        // Insert more than capacity
        for i in 0..capacity * 2 {
            cache.insert(
                NodeId(i as u64), 
                "test_prop", 
                Some(PropertyValue::Integer(i as i64)), 
                None
            );
        }
        
        // Cache size should not exceed capacity
        assert!(cache.size() <= capacity);
        
        let stats = cache.get_statistics();
        assert!(stats.evictions.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn test_statistics_accuracy() {
        let cache = PropertyCache::new(100, Duration::from_secs(60));
        
        // Perform known operations
        cache.insert(NodeId(1), "prop1", Some(PropertyValue::Boolean(true)), None);
        cache.get(NodeId(1), "prop1"); // Hit
        cache.get(NodeId(2), "prop2"); // Miss
        cache.invalidate_node(NodeId(1)); // Invalidation
        
        let stats = cache.get_statistics();
        assert_eq!(stats.hits.load(Ordering::Relaxed), 1);
        assert_eq!(stats.misses.load(Ordering::Relaxed), 1);
        assert_eq!(stats.invalidations.load(Ordering::Relaxed), 1);
        assert_eq!(cache.hit_rate(), 0.5); // 1 hit out of 2 accesses
    }
}
```

## File to Create
Create exactly this file: `src/properties/cache.rs`

## Dependencies Required
No additional dependencies beyond what's already in the project.

## When Complete
Respond with "MICRO PHASE 1.5 COMPLETE" and a brief summary of what you implemented, including:
- Eviction policy strategy used
- Thread safety mechanisms employed
- Performance optimizations implemented
- TTL implementation approach
- Confirmation that all tests pass