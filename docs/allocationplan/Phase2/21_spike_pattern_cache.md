# Task 21: Spike Pattern Cache

## Metadata
- **Micro-Phase**: 2.21
- **Duration**: 30-35 minutes
- **Dependencies**: Task 12 (ttfs_spike_pattern)
- **Output**: `src/ttfs_encoding/spike_pattern_cache.rs`

## Description
Implement a high-performance LRU (Least Recently Used) cache for TTFS spike patterns to achieve >90% hit rate. This cache optimizes repeated pattern access and encoding operations with intelligent prefetching and compression.

## Test Requirements
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttfs_encoding::{TTFSSpikePattern, ConceptId, SpikeEvent, NeuronId};
    use std::time::{Duration, Instant};

    #[test]
    fn test_basic_cache_operations() {
        let mut cache = SpikePatternCache::new(CacheConfig::default());
        
        let pattern = create_test_pattern("test_concept");
        let key = CacheKey::from_concept_id("test_concept");
        
        // Test insertion
        cache.insert(key.clone(), pattern.clone());
        assert_eq!(cache.size(), 1);
        
        // Test retrieval
        let retrieved = cache.get(&key);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().concept_id().as_str(), "test_concept");
        
        // Test cache hit
        let stats = cache.statistics();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.hit_rate, 1.0);
    }
    
    #[test]
    fn test_lru_eviction() {
        let mut cache = SpikePatternCache::new(CacheConfig::with_capacity(3));
        
        // Insert patterns beyond capacity
        let patterns = vec![
            ("pattern1", create_test_pattern("pattern1")),
            ("pattern2", create_test_pattern("pattern2")),
            ("pattern3", create_test_pattern("pattern3")),
            ("pattern4", create_test_pattern("pattern4")), // Should evict pattern1
        ];
        
        for (name, pattern) in &patterns {
            cache.insert(CacheKey::from_concept_id(name), pattern.clone());
        }
        
        assert_eq!(cache.size(), 3);
        
        // pattern1 should be evicted (least recently used)
        let pattern1_key = CacheKey::from_concept_id("pattern1");
        assert!(cache.get(&pattern1_key).is_none());
        
        // Others should still be present
        let pattern4_key = CacheKey::from_concept_id("pattern4");
        assert!(cache.get(&pattern4_key).is_some());
    }
    
    #[test]
    fn test_cache_hit_rate_target() {
        let mut cache = SpikePatternCache::new(CacheConfig::default());
        
        // Create diverse test patterns
        let mut test_patterns = Vec::new();
        for i in 0..50 {
            let pattern_name = format!("pattern_{}", i);
            test_patterns.push((pattern_name.clone(), create_test_pattern(&pattern_name)));
        }
        
        // Insert all patterns
        for (name, pattern) in &test_patterns {
            cache.insert(CacheKey::from_concept_id(name), pattern.clone());
        }
        
        // Simulate realistic access pattern (80/20 rule)
        let mut requests = 0;
        for _ in 0..1000 {
            let pattern_index = if requests % 10 < 8 {
                // 80% of requests go to first 20% of patterns
                requests % 10
            } else {
                // 20% of requests go to remaining patterns
                10 + (requests % 40)
            };
            
            if pattern_index < test_patterns.len() {
                let key = CacheKey::from_concept_id(&test_patterns[pattern_index].0);
                cache.get(&key);
                requests += 1;
            }
        }
        
        let stats = cache.statistics();
        assert!(stats.hit_rate >= 0.9, "Hit rate {:.2}% below target 90%", stats.hit_rate * 100.0);
    }
    
    #[test]
    fn test_cache_performance() {
        let mut cache = SpikePatternCache::new(CacheConfig::high_performance());
        
        // Pre-populate cache
        for i in 0..100 {
            let pattern_name = format!("pattern_{}", i);
            let pattern = create_test_pattern(&pattern_name);
            cache.insert(CacheKey::from_concept_id(&pattern_name), pattern);
        }
        
        // Measure cache access performance
        let start = Instant::now();
        for i in 0..1000 {
            let pattern_name = format!("pattern_{}", i % 100);
            let key = CacheKey::from_concept_id(&pattern_name);
            cache.get(&key);
        }
        let access_time = start.elapsed();
        
        // Should be very fast
        let avg_access_time = access_time / 1000;
        assert!(avg_access_time < Duration::from_nanos(1000), // <1μs per access
            "Average access time {:?} too slow", avg_access_time);
        
        let stats = cache.statistics();
        assert_eq!(stats.cache_hits, 1000);
        assert_eq!(stats.hit_rate, 1.0);
    }
    
    #[test]
    fn test_pattern_compression() {
        let mut config = CacheConfig::default();
        config.enable_compression = true;
        let mut cache = SpikePatternCache::new(config);
        
        let large_pattern = create_large_pattern("large_pattern", 1000);
        let key = CacheKey::from_concept_id("large_pattern");
        
        cache.insert(key.clone(), large_pattern.clone());
        
        // Verify compression occurred
        let stats = cache.statistics();
        assert!(stats.compression_ratio > 1.0);
        assert!(stats.total_compressed_size < stats.total_original_size);
        
        // Verify retrieval correctness
        let retrieved = cache.get(&key).unwrap();
        assert_eq!(retrieved.concept_id(), large_pattern.concept_id());
        assert_eq!(retrieved.spike_count(), large_pattern.spike_count());
    }
    
    #[test]
    fn test_ttl_expiration() {
        let mut config = CacheConfig::default();
        config.default_ttl = Some(Duration::from_millis(100));
        let mut cache = SpikePatternCache::new(config);
        
        let pattern = create_test_pattern("expiring_pattern");
        let key = CacheKey::from_concept_id("expiring_pattern");
        
        cache.insert(key.clone(), pattern);
        
        // Should be present initially
        assert!(cache.get(&key).is_some());
        
        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));
        
        // Should be expired and removed
        assert!(cache.get(&key).is_none());
        
        let stats = cache.statistics();
        assert!(stats.expired_entries > 0);
    }
    
    #[test]
    fn test_cache_warming() {
        let mut cache = SpikePatternCache::new(CacheConfig::default());
        
        // Create predictable patterns for warming
        let patterns_to_warm = vec![
            "frequently_used_1",
            "frequently_used_2", 
            "frequently_used_3",
        ];
        
        let warm_patterns: Vec<_> = patterns_to_warm.iter()
            .map(|name| (CacheKey::from_concept_id(name), create_test_pattern(name)))
            .collect();
        
        // Warm the cache
        cache.warm_cache(&warm_patterns);
        
        assert_eq!(cache.size(), 3);
        
        // Verify all warmed patterns are accessible
        for (key, original_pattern) in &warm_patterns {
            let cached_pattern = cache.get(key).unwrap();
            assert_eq!(cached_pattern.concept_id(), original_pattern.concept_id());
        }
        
        let stats = cache.statistics();
        assert_eq!(stats.cache_hits, 3);
    }
    
    #[test]
    fn test_cache_prefetching() {
        let mut config = CacheConfig::default();
        config.enable_prefetching = true;
        let mut cache = SpikePatternCache::new(config);
        
        // Insert base pattern
        let base_pattern = create_test_pattern("base_pattern");
        let base_key = CacheKey::from_concept_id("base_pattern");
        cache.insert(base_key.clone(), base_pattern);
        
        // Access base pattern to trigger prefetching
        cache.get(&base_key);
        
        // Define related patterns that should be prefetched
        let related_keys = vec![
            CacheKey::from_concept_id("base_pattern_related"),
            CacheKey::from_concept_id("base_pattern_variant"),
        ];
        
        // Simulate prefetch suggestions
        for related_key in &related_keys {
            let related_pattern = create_test_pattern(related_key.as_str());
            cache.suggest_prefetch(related_key.clone(), related_pattern);
        }
        
        // Allow prefetching to occur
        std::thread::sleep(Duration::from_millis(10));
        
        let stats = cache.statistics();
        assert!(stats.prefetched_entries > 0);
    }
    
    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;
        
        let cache = Arc::new(std::sync::Mutex::new(
            SpikePatternCache::new(CacheConfig::default())
        ));
        
        // Pre-populate cache
        {
            let mut cache_lock = cache.lock().unwrap();
            for i in 0..10 {
                let pattern_name = format!("concurrent_pattern_{}", i);
                let pattern = create_test_pattern(&pattern_name);
                cache_lock.insert(CacheKey::from_concept_id(&pattern_name), pattern);
            }
        }
        
        let mut handles = vec![];
        
        // Spawn multiple threads for concurrent access
        for thread_id in 0..4 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let pattern_name = format!("concurrent_pattern_{}", i % 10);
                    let key = CacheKey::from_concept_id(&pattern_name);
                    
                    let cache_lock = cache_clone.lock().unwrap();
                    let _result = cache_lock.get(&key);
                    // Simulate some processing time
                    drop(cache_lock);
                    std::thread::sleep(Duration::from_nanos(100));
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        let cache_lock = cache.lock().unwrap();
        let stats = cache_lock.statistics();
        assert_eq!(stats.cache_hits, 400); // 4 threads * 100 requests
        assert_eq!(stats.hit_rate, 1.0);
    }
    
    #[test]
    fn test_memory_pressure_handling() {
        let mut config = CacheConfig::default();
        config.max_memory_mb = 1; // Very limited memory
        let mut cache = SpikePatternCache::new(config);
        
        // Insert patterns until memory pressure
        let mut patterns_inserted = 0;
        for i in 0..100 {
            let pattern_name = format!("memory_pattern_{}", i);
            let large_pattern = create_large_pattern(&pattern_name, 100);
            let key = CacheKey::from_concept_id(&pattern_name);
            
            cache.insert(key, large_pattern);
            patterns_inserted += 1;
            
            // Check if memory pressure triggered evictions
            if cache.size() < patterns_inserted {
                break;
            }
        }
        
        let stats = cache.statistics();
        assert!(stats.evicted_entries > 0);
        assert!(stats.memory_pressure_events > 0);
        assert!(cache.current_memory_usage_mb() <= 1.0);
    }
    
    #[test]
    fn test_cache_persistence() {
        let temp_file = "/tmp/test_cache.bin";
        
        {
            let mut cache = SpikePatternCache::new(CacheConfig::default());
            
            // Populate cache
            for i in 0..5 {
                let pattern_name = format!("persistent_pattern_{}", i);
                let pattern = create_test_pattern(&pattern_name);
                cache.insert(CacheKey::from_concept_id(&pattern_name), pattern);
            }
            
            // Save cache to disk
            cache.save_to_disk(temp_file).unwrap();
        }
        
        // Load cache from disk
        let mut restored_cache = SpikePatternCache::new(CacheConfig::default());
        restored_cache.load_from_disk(temp_file).unwrap();
        
        assert_eq!(restored_cache.size(), 5);
        
        // Verify patterns are correctly restored
        for i in 0..5 {
            let pattern_name = format!("persistent_pattern_{}", i);
            let key = CacheKey::from_concept_id(&pattern_name);
            assert!(restored_cache.get(&key).is_some());
        }
        
        // Cleanup
        std::fs::remove_file(temp_file).ok();
    }
    
    // Helper functions
    fn create_test_pattern(name: &str) -> TTFSSpikePattern {
        let spikes = vec![
            SpikeEvent::new(NeuronId(0), Duration::from_micros(500), 0.9),
            SpikeEvent::new(NeuronId(1), Duration::from_millis(1), 0.8),
            SpikeEvent::new(NeuronId(2), Duration::from_millis(2), 0.7),
        ];
        
        TTFSSpikePattern::new(
            ConceptId::new(name),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(3),
        )
    }
    
    fn create_large_pattern(name: &str, spike_count: usize) -> TTFSSpikePattern {
        let spikes: Vec<_> = (0..spike_count)
            .map(|i| SpikeEvent::new(
                NeuronId(i % 100),
                Duration::from_micros(500 + i as u64 * 100),
                0.5 + (i as f32 * 0.001) % 0.5,
            ))
            .collect();
        
        TTFSSpikePattern::new(
            ConceptId::new(name),
            Duration::from_micros(500),
            spikes,
            Duration::from_millis(spike_count as u64 / 10),
        )
    }
}
```

## Implementation
```rust
use crate::ttfs_encoding::{TTFSSpikePattern, ConceptId};
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant, SystemTime};
use serde::{Serialize, Deserialize};

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache capacity (number of patterns)
    pub max_capacity: usize,
    
    /// Maximum memory usage in MB
    pub max_memory_mb: f32,
    
    /// Default TTL for cache entries
    pub default_ttl: Option<Duration>,
    
    /// Enable pattern compression
    pub enable_compression: bool,
    
    /// Enable predictive prefetching
    pub enable_prefetching: bool,
    
    /// Enable access statistics tracking
    pub enable_statistics: bool,
    
    /// Eviction strategy
    pub eviction_strategy: EvictionStrategy,
    
    /// Compression level (1-9)
    pub compression_level: u32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_capacity: 10000,
            max_memory_mb: 100.0,
            default_ttl: None,
            enable_compression: false,
            enable_prefetching: false,
            enable_statistics: true,
            eviction_strategy: EvictionStrategy::LRU,
            compression_level: 6,
        }
    }
}

impl CacheConfig {
    /// Create high-performance cache configuration
    pub fn high_performance() -> Self {
        Self {
            max_capacity: 50000,
            max_memory_mb: 500.0,
            enable_compression: true,
            enable_prefetching: true,
            compression_level: 3, // Fast compression
            ..Default::default()
        }
    }
    
    /// Create memory-constrained configuration
    pub fn memory_constrained() -> Self {
        Self {
            max_capacity: 1000,
            max_memory_mb: 10.0,
            enable_compression: true,
            compression_level: 9, // Maximum compression
            default_ttl: Some(Duration::from_secs(300)), // 5 minutes
            ..Default::default()
        }
    }
    
    /// Create configuration with specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            max_capacity: capacity,
            ..Default::default()
        }
    }
}

/// Cache eviction strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EvictionStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// Time-based expiration
    TTL,
    /// Size-based (largest patterns first)
    Size,
}

/// Cache key for spike patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Concept identifier
    concept_id: String,
    /// Optional context hash for variations
    context_hash: Option<u64>,
}

impl CacheKey {
    /// Create cache key from concept ID
    pub fn from_concept_id(concept_id: &str) -> Self {
        Self {
            concept_id: concept_id.to_string(),
            context_hash: None,
        }
    }
    
    /// Create cache key with context
    pub fn from_concept_with_context(concept_id: &str, context: &str) -> Self {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        context.hash(&mut hasher);
        
        Self {
            concept_id: concept_id.to_string(),
            context_hash: Some(hasher.finish()),
        }
    }
    
    /// Get concept ID
    pub fn concept_id(&self) -> &str {
        &self.concept_id
    }
    
    /// Convert to string representation
    pub fn as_str(&self) -> String {
        match self.context_hash {
            Some(hash) => format!("{}:{:x}", self.concept_id, hash),
            None => self.concept_id.clone(),
        }
    }
}

/// Cached pattern entry
#[derive(Debug, Clone)]
struct CacheEntry {
    /// The cached pattern
    pattern: TTFSSpikePattern,
    /// Compressed pattern data (if compression enabled)
    compressed_data: Option<Vec<u8>>,
    /// Last access time
    last_access: Instant,
    /// Creation time
    created_at: Instant,
    /// Expiration time
    expires_at: Option<Instant>,
    /// Access count
    access_count: u64,
    /// Memory size estimate
    memory_size: usize,
}

impl CacheEntry {
    fn new(pattern: TTFSSpikePattern, ttl: Option<Duration>, enable_compression: bool) -> Self {
        let now = Instant::now();
        let memory_size = Self::estimate_memory_size(&pattern);
        
        let compressed_data = if enable_compression {
            Self::compress_pattern(&pattern).ok()
        } else {
            None
        };
        
        Self {
            pattern,
            compressed_data,
            last_access: now,
            created_at: now,
            expires_at: ttl.map(|duration| now + duration),
            access_count: 0,
            memory_size,
        }
    }
    
    fn access(&mut self) -> &TTFSSpikePattern {
        self.last_access = Instant::now();
        self.access_count += 1;
        &self.pattern
    }
    
    fn is_expired(&self) -> bool {
        self.expires_at.map_or(false, |expires| Instant::now() > expires)
    }
    
    fn estimate_memory_size(pattern: &TTFSSpikePattern) -> usize {
        // Rough estimate: base size + spike count * spike size
        std::mem::size_of::<TTFSSpikePattern>() + 
        pattern.spike_count() * std::mem::size_of::<crate::ttfs_encoding::SpikeEvent>() +
        pattern.concept_id().as_str().len()
    }
    
    fn compress_pattern(pattern: &TTFSSpikePattern) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let serialized = bincode::serialize(pattern)?;
        let compressed = zstd::encode_all(&serialized[..], 6)?;
        Ok(compressed)
    }
    
    fn decompress_pattern(compressed_data: &[u8]) -> Result<TTFSSpikePattern, Box<dyn std::error::Error>> {
        let decompressed = zstd::decode_all(compressed_data)?;
        let pattern = bincode::deserialize(&decompressed)?;
        Ok(pattern)
    }
}

/// Cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    /// Total cache requests
    pub total_requests: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Hit rate (0.0-1.0)
    pub hit_rate: f32,
    /// Number of evicted entries
    pub evicted_entries: u64,
    /// Number of expired entries
    pub expired_entries: u64,
    /// Number of prefetched entries
    pub prefetched_entries: u64,
    /// Memory pressure events
    pub memory_pressure_events: u64,
    /// Total original size (bytes)
    pub total_original_size: usize,
    /// Total compressed size (bytes)
    pub total_compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Average access time
    pub average_access_time: Duration,
}

/// Main spike pattern cache
#[derive(Debug)]
pub struct SpikePatternCache {
    /// Configuration
    config: CacheConfig,
    
    /// Cache storage
    cache: HashMap<CacheKey, CacheEntry>,
    
    /// LRU ordering
    lru_order: VecDeque<CacheKey>,
    
    /// LFU frequency tracking
    frequency: HashMap<CacheKey, u64>,
    
    /// Statistics
    statistics: CacheStatistics,
    
    /// Current memory usage (bytes)
    current_memory_usage: usize,
    
    /// Prefetch queue
    prefetch_queue: VecDeque<(CacheKey, TTFSSpikePattern)>,
}

impl SpikePatternCache {
    /// Create new spike pattern cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            lru_order: VecDeque::new(),
            frequency: HashMap::new(),
            statistics: CacheStatistics::default(),
            current_memory_usage: 0,
            prefetch_queue: VecDeque::new(),
        }
    }
    
    /// Insert pattern into cache
    pub fn insert(&mut self, key: CacheKey, pattern: TTFSSpikePattern) {
        // Check if we need to evict entries
        self.ensure_capacity_for_insert(&key, &pattern);
        
        // Create cache entry
        let entry = CacheEntry::new(pattern, self.config.default_ttl, self.config.enable_compression);
        self.current_memory_usage += entry.memory_size;
        
        // Update statistics
        if self.config.enable_statistics {
            self.statistics.total_original_size += entry.memory_size;
            if let Some(ref compressed) = entry.compressed_data {
                self.statistics.total_compressed_size += compressed.len();
                self.update_compression_ratio();
            }
        }
        
        // Insert into cache
        if let Some(old_entry) = self.cache.insert(key.clone(), entry) {
            self.current_memory_usage = self.current_memory_usage.saturating_sub(old_entry.memory_size);
        } else {
            // New entry, update LRU
            self.lru_order.push_back(key.clone());
        }
        
        // Update LFU
        *self.frequency.entry(key).or_insert(0) += 1;
    }
    
    /// Get pattern from cache
    pub fn get(&mut self, key: &CacheKey) -> Option<TTFSSpikePattern> {
        let start_time = Instant::now();
        
        self.statistics.total_requests += 1;
        
        // Check for expired entries
        if let Some(entry) = self.cache.get(key) {
            if entry.is_expired() {
                self.remove_expired(key);
                self.statistics.cache_misses += 1;
                self.update_hit_rate();
                return None;
            }
        }
        
        if let Some(entry) = self.cache.get_mut(key) {
            // Cache hit
            self.statistics.cache_hits += 1;
            
            // Update LRU order
            self.update_lru_order(key);
            
            // Update LFU frequency
            *self.frequency.entry(key.clone()).or_insert(0) += 1;
            
            let access_time = start_time.elapsed();
            self.update_average_access_time(access_time);
            self.update_hit_rate();
            
            Some(entry.access().clone())
        } else {
            // Cache miss
            self.statistics.cache_misses += 1;
            self.update_hit_rate();
            None
        }
    }
    
    /// Remove pattern from cache
    pub fn remove(&mut self, key: &CacheKey) -> Option<TTFSSpikePattern> {
        if let Some(entry) = self.cache.remove(key) {
            self.current_memory_usage = self.current_memory_usage.saturating_sub(entry.memory_size);
            
            // Remove from LRU order
            self.lru_order.retain(|k| k != key);
            
            // Remove from frequency tracking
            self.frequency.remove(key);
            
            Some(entry.pattern)
        } else {
            None
        }
    }
    
    /// Clear entire cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.lru_order.clear();
        self.frequency.clear();
        self.current_memory_usage = 0;
        
        // Reset statistics
        if self.config.enable_statistics {
            self.statistics = CacheStatistics::default();
        }
    }
    
    /// Get cache size (number of entries)
    pub fn size(&self) -> usize {
        self.cache.len()
    }
    
    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
    
    /// Get current memory usage in MB
    pub fn current_memory_usage_mb(&self) -> f32 {
        self.current_memory_usage as f32 / (1024.0 * 1024.0)
    }
    
    /// Get cache statistics
    pub fn statistics(&self) -> &CacheStatistics {
        &self.statistics
    }
    
    /// Warm cache with frequently used patterns
    pub fn warm_cache(&mut self, patterns: &[(CacheKey, TTFSSpikePattern)]) {
        for (key, pattern) in patterns {
            self.insert(key.clone(), pattern.clone());
        }
    }
    
    /// Suggest pattern for prefetching
    pub fn suggest_prefetch(&mut self, key: CacheKey, pattern: TTFSSpikePattern) {
        if self.config.enable_prefetching && !self.cache.contains_key(&key) {
            self.prefetch_queue.push_back((key, pattern));
            
            // Process prefetch queue asynchronously (simplified synchronous version)
            if self.prefetch_queue.len() > 10 {
                self.process_prefetch_queue();
            }
        }
    }
    
    /// Save cache to disk
    pub fn save_to_disk(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let cache_data: Vec<(CacheKey, TTFSSpikePattern)> = self.cache.iter()
            .map(|(key, entry)| (key.clone(), entry.pattern.clone()))
            .collect();
        
        let serialized = bincode::serialize(&cache_data)?;
        std::fs::write(path, serialized)?;
        Ok(())
    }
    
    /// Load cache from disk
    pub fn load_from_disk(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let cache_data: Vec<(CacheKey, TTFSSpikePattern)> = bincode::deserialize(&data)?;
        
        for (key, pattern) in cache_data {
            self.insert(key, pattern);
        }
        
        Ok(())
    }
    
    // Internal helper methods
    
    fn ensure_capacity_for_insert(&mut self, key: &CacheKey, pattern: &TTFSSpikePattern) {
        let pattern_size = CacheEntry::estimate_memory_size(pattern);
        
        // Check memory pressure
        if self.current_memory_usage + pattern_size > (self.config.max_memory_mb * 1024.0 * 1024.0) as usize {
            self.statistics.memory_pressure_events += 1;
            self.handle_memory_pressure(pattern_size);
        }
        
        // Check capacity
        if !self.cache.contains_key(key) && self.cache.len() >= self.config.max_capacity {
            self.evict_entries(1);
        }
    }
    
    fn handle_memory_pressure(&mut self, needed_size: usize) {
        let target_usage = (self.config.max_memory_mb * 0.8 * 1024.0 * 1024.0) as usize;
        
        while self.current_memory_usage + needed_size > target_usage && !self.cache.is_empty() {
            match self.config.eviction_strategy {
                EvictionStrategy::LRU => self.evict_lru(),
                EvictionStrategy::LFU => self.evict_lfu(),
                EvictionStrategy::TTL => self.evict_expired(),
                EvictionStrategy::Size => self.evict_largest(),
            }
        }
    }
    
    fn evict_entries(&mut self, count: usize) {
        for _ in 0..count {
            if self.cache.is_empty() {
                break;
            }
            
            match self.config.eviction_strategy {
                EvictionStrategy::LRU => self.evict_lru(),
                EvictionStrategy::LFU => self.evict_lfu(),
                EvictionStrategy::TTL => self.evict_expired(),
                EvictionStrategy::Size => self.evict_largest(),
            }
        }
    }
    
    fn evict_lru(&mut self) {
        if let Some(oldest_key) = self.lru_order.pop_front() {
            self.remove(&oldest_key);
            self.statistics.evicted_entries += 1;
        }
    }
    
    fn evict_lfu(&mut self) {
        if let Some((least_frequent_key, _)) = self.frequency.iter()
            .filter(|(key, _)| self.cache.contains_key(key))
            .min_by_key(|(_, &freq)| freq)
            .map(|(key, freq)| (key.clone(), *freq))
        {
            self.remove(&least_frequent_key);
            self.statistics.evicted_entries += 1;
        }
    }
    
    fn evict_expired(&mut self) {
        let expired_keys: Vec<CacheKey> = self.cache.iter()
            .filter(|(_, entry)| entry.is_expired())
            .map(|(key, _)| key.clone())
            .collect();
        
        for key in expired_keys {
            self.remove_expired(&key);
        }
    }
    
    fn evict_largest(&mut self) {
        if let Some((largest_key, _)) = self.cache.iter()
            .max_by_key(|(_, entry)| entry.memory_size)
            .map(|(key, entry)| (key.clone(), entry.memory_size))
        {
            self.remove(&largest_key);
            self.statistics.evicted_entries += 1;
        }
    }
    
    fn remove_expired(&mut self, key: &CacheKey) {
        if self.remove(key).is_some() {
            self.statistics.expired_entries += 1;
        }
    }
    
    fn update_lru_order(&mut self, key: &CacheKey) {
        // Remove from current position
        self.lru_order.retain(|k| k != key);
        // Add to end (most recently used)
        self.lru_order.push_back(key.clone());
    }
    
    fn update_hit_rate(&mut self) {
        if self.statistics.total_requests > 0 {
            self.statistics.hit_rate = self.statistics.cache_hits as f32 / self.statistics.total_requests as f32;
        }
    }
    
    fn update_average_access_time(&mut self, access_time: Duration) {
        let total_requests = self.statistics.total_requests;
        if total_requests == 1 {
            self.statistics.average_access_time = access_time;
        } else {
            let current_total = self.statistics.average_access_time.as_nanos() as u64 * (total_requests - 1);
            let new_total = current_total + access_time.as_nanos() as u64;
            self.statistics.average_access_time = Duration::from_nanos(new_total / total_requests);
        }
    }
    
    fn update_compression_ratio(&mut self) {
        if self.statistics.total_compressed_size > 0 {
            self.statistics.compression_ratio = self.statistics.total_original_size as f32 / self.statistics.total_compressed_size as f32;
        }
    }
    
    fn process_prefetch_queue(&mut self) {
        while let Some((key, pattern)) = self.prefetch_queue.pop_front() {
            if !self.cache.contains_key(&key) && self.cache.len() < self.config.max_capacity {
                self.insert(key, pattern);
                self.statistics.prefetched_entries += 1;
            }
        }
    }
}

/// Cache warming strategies
pub enum WarmingStrategy {
    /// Load most frequently accessed patterns
    FrequencyBased(usize),
    /// Load patterns by access recency
    RecencyBased(Duration),
    /// Load specific pattern list
    Explicit(Vec<String>),
}

/// Prefetching hints for related patterns
#[derive(Debug, Clone)]
pub struct PrefetchHint {
    /// Base pattern that triggered the hint
    pub base_pattern: String,
    /// Related patterns to prefetch
    pub related_patterns: Vec<String>,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}
```

## Verification Steps
1. Implement LRU cache with configurable capacity and eviction
2. Add compression support for memory efficiency
3. Implement TTL-based expiration and cleanup
4. Add prefetching capabilities with queue management
5. Implement comprehensive statistics tracking
6. Add persistence support for cache warming

## Success Criteria
- [ ] Cache achieves >90% hit rate with realistic access patterns
- [ ] LRU eviction works correctly under capacity pressure
- [ ] Memory usage stays within configured limits
- [ ] Cache access times are <1μs per operation
- [ ] Compression reduces memory usage significantly
- [ ] All test cases pass with performance requirements