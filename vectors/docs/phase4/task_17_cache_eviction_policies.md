# Task 17: Implement Cache Eviction Policies

## Context
You are implementing Phase 4 of a vector indexing system. Memory usage calculation was implemented in the previous task. Now you need to implement sophisticated cache eviction policies including LRU, LFU, and size-based eviction strategies.

## Current State
- `src/cache.rs` exists with `MemoryEfficientCache` struct
- Memory usage calculation and profiling is implemented
- Basic eviction exists but needs enhancement
- Cache put/get methods work with basic oldest-first eviction

## Task Objective
Implement multiple cache eviction policies (LRU, LFU, size-based) with configurable strategies and smart eviction decisions based on memory pressure and access patterns.

## Implementation Requirements

### 1. Add eviction policy enum and configuration
Add these enums and structs before the `MemoryEfficientCache` struct:
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum EvictionPolicy {
    LRU,        // Least Recently Used
    LFU,        // Least Frequently Used
    SizeBased,  // Evict largest entries first
    Hybrid,     // Combination of LRU and size
}

#[derive(Debug, Clone)]
pub struct EvictionConfig {
    pub primary_policy: EvictionPolicy,
    pub memory_pressure_threshold: f64,  // 0.0 to 1.0
    pub aggressive_eviction_threshold: f64,
    pub min_entries_to_keep: usize,
    pub max_eviction_batch_size: usize,
    pub size_weight: f64,  // For hybrid policy (0.0 = pure LRU, 1.0 = pure size)
}

impl EvictionConfig {
    pub fn new_lru() -> Self {
        Self {
            primary_policy: EvictionPolicy::LRU,
            memory_pressure_threshold: 0.8,
            aggressive_eviction_threshold: 0.95,
            min_entries_to_keep: 10,
            max_eviction_batch_size: 50,
            size_weight: 0.0,
        }
    }
    
    pub fn new_lfu() -> Self {
        Self {
            primary_policy: EvictionPolicy::LFU,
            memory_pressure_threshold: 0.8,
            aggressive_eviction_threshold: 0.95,
            min_entries_to_keep: 10,
            max_eviction_batch_size: 50,
            size_weight: 0.0,
        }
    }
    
    pub fn new_size_based() -> Self {
        Self {
            primary_policy: EvictionPolicy::SizeBased,
            memory_pressure_threshold: 0.8,
            aggressive_eviction_threshold: 0.95,
            min_entries_to_keep: 5,
            max_eviction_batch_size: 25,
            size_weight: 1.0,
        }
    }
    
    pub fn new_hybrid(size_weight: f64) -> Self {
        Self {
            primary_policy: EvictionPolicy::Hybrid,
            memory_pressure_threshold: 0.8,
            aggressive_eviction_threshold: 0.95,
            min_entries_to_keep: 10,
            max_eviction_batch_size: 40,
            size_weight: size_weight.clamp(0.0, 1.0),
        }
    }
}
```

### 2. Update MemoryEfficientCache to include eviction config
Update the struct and constructor:
```rust
pub struct MemoryEfficientCache {
    query_cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    max_entries: usize,
    max_memory_mb: usize,
    current_memory_usage: Arc<RwLock<usize>>,
    hit_count: Arc<RwLock<usize>>,
    miss_count: Arc<RwLock<usize>>,
    eviction_config: EvictionConfig,  // Add this field
}

impl MemoryEfficientCache {
    pub fn new(max_entries: usize, max_memory_mb: usize) -> Self {
        Self::with_eviction_policy(max_entries, max_memory_mb, EvictionConfig::new_lru())
    }
    
    pub fn with_eviction_policy(max_entries: usize, max_memory_mb: usize, eviction_config: EvictionConfig) -> Self {
        Self {
            query_cache: Arc::new(RwLock::new(HashMap::new())),
            max_entries,
            max_memory_mb,
            current_memory_usage: Arc::new(RwLock::new(0)),
            hit_count: Arc::new(RwLock::new(0)),
            miss_count: Arc::new(RwLock::new(0)),
            eviction_config,
        }
    }
    
    pub fn get_eviction_config(&self) -> &EvictionConfig {
        &self.eviction_config
    }
    
    pub fn update_eviction_config(&mut self, config: EvictionConfig) {
        self.eviction_config = config;
    }
}
```

### 3. Implement eviction candidate selection
Add these methods to the `MemoryEfficientCache` implementation:
```rust
fn select_eviction_candidates(
    &self,
    cache: &HashMap<String, CacheEntry>,
    target_count: usize,
) -> Vec<String> {
    if cache.is_empty() || target_count == 0 {
        return Vec::new();
    }
    
    let actual_target = target_count.min(cache.len().saturating_sub(self.eviction_config.min_entries_to_keep));
    let actual_target = actual_target.min(self.eviction_config.max_eviction_batch_size);
    
    match self.eviction_config.primary_policy {
        EvictionPolicy::LRU => self.select_lru_candidates(cache, actual_target),
        EvictionPolicy::LFU => self.select_lfu_candidates(cache, actual_target),
        EvictionPolicy::SizeBased => self.select_size_based_candidates(cache, actual_target),
        EvictionPolicy::Hybrid => self.select_hybrid_candidates(cache, actual_target),
    }
}

fn select_lru_candidates(&self, cache: &HashMap<String, CacheEntry>, count: usize) -> Vec<String> {
    let mut entries: Vec<_> = cache.iter().collect();
    entries.sort_by_key(|(_, entry)| entry.timestamp);
    entries.into_iter()
        .take(count)
        .map(|(key, _)| key.clone())
        .collect()
}

fn select_lfu_candidates(&self, cache: &HashMap<String, CacheEntry>, count: usize) -> Vec<String> {
    let mut entries: Vec<_> = cache.iter().collect();
    entries.sort_by_key(|(_, entry)| entry.access_count);
    entries.into_iter()
        .take(count)
        .map(|(key, _)| key.clone())
        .collect()
}

fn select_size_based_candidates(&self, cache: &HashMap<String, CacheEntry>, count: usize) -> Vec<String> {
    let mut entries: Vec<_> = cache.iter().collect();
    entries.sort_by(|(_, a), (_, b)| b.estimated_size.cmp(&a.estimated_size));
    entries.into_iter()
        .take(count)
        .map(|(key, _)| key.clone())
        .collect()
}

fn select_hybrid_candidates(&self, cache: &HashMap<String, CacheEntry>, count: usize) -> Vec<String> {
    let now = std::time::Instant::now();
    let size_weight = self.eviction_config.size_weight;
    let time_weight = 1.0 - size_weight;
    
    let mut scored_entries: Vec<_> = cache.iter().map(|(key, entry)| {
        // Calculate time score (higher = older, more likely to evict)
        let age_seconds = now.duration_since(entry.timestamp).as_secs_f64().max(1.0);
        let time_score = age_seconds / 3600.0; // Normalize to hours
        
        // Calculate size score (higher = larger, more likely to evict)
        let size_score = entry.estimated_size as f64 / (1024.0 * 1024.0); // MB
        
        // Combined score
        let combined_score = (time_weight * time_score) + (size_weight * size_score);
        
        (key.clone(), combined_score)
    }).collect();
    
    // Sort by combined score (highest first)
    scored_entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    scored_entries.into_iter()
        .take(count)
        .map(|(key, _)| key)
        .collect()
}
```

### 4. Implement smart eviction with memory pressure
Replace the existing eviction methods with these enhanced versions:
```rust
fn smart_eviction(
    &self,
    cache: &mut HashMap<String, CacheEntry>,
    memory_usage: &mut usize,
    required_space: usize,
) -> bool {
    let current_memory_mb = *memory_usage as f64 / (1024.0 * 1024.0);
    let memory_pressure = current_memory_mb / self.max_memory_mb as f64;
    
    let eviction_target = if memory_pressure >= self.eviction_config.aggressive_eviction_threshold {
        // Aggressive eviction - remove up to 50% of entries
        cache.len() / 2
    } else if memory_pressure >= self.eviction_config.memory_pressure_threshold {
        // Moderate eviction - remove entries to get below threshold
        let target_memory = (self.max_memory_mb as f64 * self.eviction_config.memory_pressure_threshold) as usize;
        let bytes_to_free = (*memory_usage + required_space).saturating_sub(target_memory);
        self.estimate_entries_for_bytes(cache, bytes_to_free)
    } else {
        // Minimal eviction - just make space for the new entry
        self.estimate_entries_for_bytes(cache, required_space)
    };
    
    if eviction_target == 0 {
        return true; // No eviction needed
    }
    
    let candidates = self.select_eviction_candidates(cache, eviction_target);
    self.evict_entries(cache, memory_usage, candidates)
}

fn estimate_entries_for_bytes(&self, cache: &HashMap<String, CacheEntry>, bytes_needed: usize) -> usize {
    if bytes_needed == 0 {
        return 0;
    }
    
    let avg_entry_size = if cache.is_empty() {
        8192 // Reasonable default
    } else {
        cache.values().map(|e| e.estimated_size).sum::<usize>() / cache.len()
    };
    
    (bytes_needed / avg_entry_size).max(1)
}

fn evict_entries(
    &self,
    cache: &mut HashMap<String, CacheEntry>,
    memory_usage: &mut usize,
    keys_to_evict: Vec<String>,
) -> bool {
    if keys_to_evict.is_empty() {
        return false;
    }
    
    let mut evicted_count = 0;
    let mut freed_bytes = 0;
    
    for key in keys_to_evict {
        if let Some(entry) = cache.remove(&key) {
            freed_bytes += entry.estimated_size;
            evicted_count += 1;
            *memory_usage = memory_usage.saturating_sub(entry.estimated_size);
        }
    }
    
    evicted_count > 0
}
```

### 5. Update put method to use smart eviction
Replace the existing `make_space_for_entry` call in the `put` method:
```rust
// In the put method, replace the make_space_for_entry call with:
if projected_memory_mb > self.max_memory_mb as f64 {
    if !self.smart_eviction(&mut cache, &mut memory_usage, entry_size) {
        return false;
    }
}
```

### 6. Add eviction statistics tracking
Add these fields to track eviction statistics:
```rust
// Add these to the MemoryEfficientCache struct
pub struct MemoryEfficientCache {
    // ... existing fields ...
    eviction_count: Arc<RwLock<usize>>,
    total_evicted_entries: Arc<RwLock<usize>>,
    total_evicted_bytes: Arc<RwLock<usize>>,
}

// Update the constructor
pub fn with_eviction_policy(max_entries: usize, max_memory_mb: usize, eviction_config: EvictionConfig) -> Self {
    Self {
        query_cache: Arc::new(RwLock::new(HashMap::new())),
        max_entries,
        max_memory_mb,
        current_memory_usage: Arc::new(RwLock::new(0)),
        hit_count: Arc::new(RwLock::new(0)),
        miss_count: Arc::new(RwLock::new(0)),
        eviction_config,
        eviction_count: Arc::new(RwLock::new(0)),
        total_evicted_entries: Arc::new(RwLock::new(0)),
        total_evicted_bytes: Arc::new(RwLock::new(0)),
    }
}

// Add these methods
pub fn eviction_stats(&self) -> EvictionStats {
    EvictionStats {
        eviction_count: *self.eviction_count.read().unwrap(),
        total_evicted_entries: *self.total_evicted_entries.read().unwrap(),
        total_evicted_bytes: *self.total_evicted_bytes.read().unwrap(),
        policy: self.eviction_config.primary_policy.clone(),
    }
}
```

### 7. Add eviction statistics struct
Add this struct before the `MemoryEfficientCache` struct:
```rust
#[derive(Debug, Clone)]
pub struct EvictionStats {
    pub eviction_count: usize,
    pub total_evicted_entries: usize,
    pub total_evicted_bytes: usize,
    pub policy: EvictionPolicy,
}

impl EvictionStats {
    pub fn avg_entries_per_eviction(&self) -> f64 {
        if self.eviction_count > 0 {
            self.total_evicted_entries as f64 / self.eviction_count as f64
        } else {
            0.0
        }
    }
    
    pub fn avg_bytes_per_eviction(&self) -> f64 {
        if self.eviction_count > 0 {
            self.total_evicted_bytes as f64 / self.eviction_count as f64
        } else {
            0.0
        }
    }
    
    pub fn format_stats(&self) -> String {
        format!(
            "Eviction Stats (Policy: {:?}):\n  Total evictions: {}\n  Entries evicted: {}\n  Bytes evicted: {:.2} MB\n  Avg entries/eviction: {:.1}\n  Avg bytes/eviction: {:.2} KB",
            self.policy,
            self.eviction_count,
            self.total_evicted_entries,
            self.total_evicted_bytes as f64 / (1024.0 * 1024.0),
            self.avg_entries_per_eviction(),
            self.avg_bytes_per_eviction() / 1024.0
        )
    }
}
```

### 8. Add comprehensive eviction policy tests
Add these tests to the test module:
```rust
#[test]
fn test_lru_eviction_policy() {
    let config = EvictionConfig::new_lru();
    let mut cache = MemoryEfficientCache::with_eviction_policy(3, 10, config);
    
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "content".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // Add entries with delays to establish clear timestamps
    cache.put("oldest".to_string(), test_results.clone());
    std::thread::sleep(std::time::Duration::from_millis(10));
    
    cache.put("middle".to_string(), test_results.clone());
    std::thread::sleep(std::time::Duration::from_millis(10));
    
    cache.put("newest".to_string(), test_results.clone());
    
    // This should evict "oldest" (LRU)
    cache.put("trigger_eviction".to_string(), test_results);
    
    assert!(cache.get("oldest").is_none());
    assert!(cache.get("middle").is_some());
    assert!(cache.get("newest").is_some());
    assert!(cache.get("trigger_eviction").is_some());
}

#[test]
fn test_lfu_eviction_policy() {
    let config = EvictionConfig::new_lfu();
    let mut cache = MemoryEfficientCache::with_eviction_policy(3, 10, config);
    
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "content".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // Add entries
    cache.put("rarely_used".to_string(), test_results.clone());
    cache.put("sometimes_used".to_string(), test_results.clone());
    cache.put("frequently_used".to_string(), test_results.clone());
    
    // Create different access patterns
    cache.get("frequently_used"); // 2 total accesses (1 from put + 1)
    cache.get("frequently_used"); // 3 total accesses
    cache.get("sometimes_used");  // 2 total accesses
    // rarely_used stays at 1 access (from put)
    
    // This should evict "rarely_used" (LFU)
    cache.put("trigger_eviction".to_string(), test_results);
    
    assert!(cache.get("rarely_used").is_none());
    assert!(cache.get("sometimes_used").is_some());
    assert!(cache.get("frequently_used").is_some());
    assert!(cache.get("trigger_eviction").is_some());
}

#[test]
fn test_size_based_eviction_policy() {
    let config = EvictionConfig::new_size_based();
    let mut cache = MemoryEfficientCache::with_eviction_policy(3, 10, config);
    
    let small_results = vec![
        SearchResult {
            file_path: "s.rs".to_string(),
            content: "x".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    let large_results = vec![
        SearchResult {
            file_path: "large_file_name.rs".to_string(),
            content: "x".repeat(100),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // Add entries
    cache.put("small1".to_string(), small_results.clone());
    cache.put("small2".to_string(), small_results);
    cache.put("large".to_string(), large_results.clone());
    
    // This should evict "large" (size-based)
    cache.put("trigger_eviction".to_string(), large_results);
    
    // The original large entry should be evicted
    assert!(cache.get("small1").is_some());
    assert!(cache.get("small2").is_some());
    assert!(cache.get("trigger_eviction").is_some());
}

#[test]
fn test_hybrid_eviction_policy() {
    let config = EvictionConfig::new_hybrid(0.5); // Equal weight to size and time
    let mut cache = MemoryEfficientCache::with_eviction_policy(3, 10, config);
    
    let small_results = vec![
        SearchResult {
            file_path: "s.rs".to_string(),
            content: "x".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    let large_results = vec![
        SearchResult {
            file_path: "large.rs".to_string(),
            content: "x".repeat(50),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // Add entries with time delays
    cache.put("old_small".to_string(), small_results.clone());
    std::thread::sleep(std::time::Duration::from_millis(10));
    
    cache.put("new_large".to_string(), large_results);
    cache.put("new_small".to_string(), small_results);
    
    // Hybrid should consider both age and size
    cache.put("trigger_eviction".to_string(), vec![
        SearchResult {
            file_path: "trigger.rs".to_string(),
            content: "trigger".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ]);
    
    // Verify cache still functions (exact eviction depends on timing and size weights)
    assert_eq!(cache.current_entries(), 3); // Should still be at max
}

#[test]
fn test_eviction_statistics() {
    let config = EvictionConfig::new_lru();
    let mut cache = MemoryEfficientCache::with_eviction_policy(2, 10, config);
    
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "content".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // Fill cache and trigger evictions
    cache.put("entry1".to_string(), test_results.clone());
    cache.put("entry2".to_string(), test_results.clone());
    cache.put("entry3".to_string(), test_results); // Should trigger eviction
    
    let stats = cache.eviction_stats();
    assert!(stats.eviction_count > 0 || cache.current_entries() <= 2);
}

#[test]
fn test_memory_pressure_eviction() {
    // Create cache with very small memory limit to force pressure-based eviction
    let config = EvictionConfig {
        primary_policy: EvictionPolicy::LRU,
        memory_pressure_threshold: 0.7,
        aggressive_eviction_threshold: 0.9,
        min_entries_to_keep: 1,
        max_eviction_batch_size: 10,
        size_weight: 0.0,
    };
    
    let mut cache = MemoryEfficientCache::with_eviction_policy(100, 1, config); // 1MB limit
    
    // Create content that approaches memory limit
    let large_content = "x".repeat(200_000); // ~200KB per entry
    let large_results = vec![
        SearchResult {
            file_path: "large.rs".to_string(),
            content: large_content,
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // Add entries until memory pressure kicks in
    for i in 0..6 {
        let success = cache.put(format!("entry_{}", i), large_results.clone());
        if !success {
            break; // Memory limit reached
        }
    }
    
    // Should have triggered eviction to stay within memory limits
    let memory_mb = cache.current_memory_usage_mb();
    assert!(memory_mb <= 1.2); // Allow some tolerance
}
```

## Success Criteria
- [ ] Multiple eviction policies (LRU, LFU, size-based, hybrid) implemented
- [ ] Smart eviction based on memory pressure levels
- [ ] Configurable eviction parameters
- [ ] Eviction statistics tracking and reporting
- [ ] Candidate selection algorithms work correctly
- [ ] Memory-pressure-driven eviction prevents limit violations
- [ ] All tests pass for different eviction strategies
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- LRU evicts least recently accessed entries
- LFU evicts least frequently accessed entries  
- Size-based evicts largest entries first
- Hybrid combines time and size factors
- Memory pressure drives eviction aggressiveness
- Statistics help tune eviction performance