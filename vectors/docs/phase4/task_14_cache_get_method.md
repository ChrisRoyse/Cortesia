# Task 14: Implement Cache Get Method

## Context
You are implementing Phase 4 of a vector indexing system. The basic cache structure was created in the previous task. Now you need to implement the cache retrieval functionality with hit tracking and access pattern optimization.

## Current State
- `src/cache.rs` exists with `MemoryEfficientCache` struct
- `CacheEntry` struct handles memory estimation
- Basic configuration and statistics are in place

## Task Objective
Implement the `get()` method for cache retrieval with hit tracking, access counting, and performance optimization.

## Implementation Requirements

### 1. Add hit/miss tracking fields
Update the `MemoryEfficientCache` struct to include hit tracking:
```rust
pub struct MemoryEfficientCache {
    query_cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    max_entries: usize,
    max_memory_mb: usize,
    current_memory_usage: Arc<RwLock<usize>>,
    hit_count: Arc<RwLock<usize>>,    // Add this field
    miss_count: Arc<RwLock<usize>>,   // Add this field
}
```

### 2. Update constructor to initialize hit tracking
Modify the `new()` method:
```rust
pub fn new(max_entries: usize, max_memory_mb: usize) -> Self {
    Self {
        query_cache: Arc::new(RwLock::new(HashMap::new())),
        max_entries,
        max_memory_mb,
        current_memory_usage: Arc::new(RwLock::new(0)),
        hit_count: Arc::new(RwLock::new(0)),
        miss_count: Arc::new(RwLock::new(0)),
    }
}
```

### 3. Implement the get() method
Add this method to the `MemoryEfficientCache` implementation:
```rust
pub fn get(&self, query: &str) -> Option<Vec<SearchResult>> {
    let mut cache = self.query_cache.write().unwrap();
    
    if let Some(entry) = cache.get_mut(query) {
        // Cache hit - update access patterns
        entry.touch();
        
        // Update hit statistics
        {
            let mut hit_count = self.hit_count.write().unwrap();
            *hit_count += 1;
        }
        
        // Return a clone of the cached results
        Some(entry.results.clone())
    } else {
        // Cache miss - update statistics
        {
            let mut miss_count = self.miss_count.write().unwrap();
            *miss_count += 1;
        }
        
        None
    }
}
```

### 4. Add cache statistics accessors
Add these methods to get cache performance metrics:
```rust
pub fn hit_count(&self) -> usize {
    *self.hit_count.read().unwrap()
}

pub fn miss_count(&self) -> usize {
    *self.miss_count.read().unwrap()
}

pub fn total_requests(&self) -> usize {
    self.hit_count() + self.miss_count()
}

pub fn hit_rate(&self) -> f64 {
    let total = self.total_requests();
    if total > 0 {
        self.hit_count() as f64 / total as f64
    } else {
        0.0
    }
}
```

### 5. Update CacheStats to include hit/miss data
Modify the `get_stats()` method:
```rust
pub fn get_stats(&self) -> CacheStats {
    let cache_guard = self.query_cache.read().unwrap();
    let memory_usage = *self.current_memory_usage.read().unwrap();
    let hits = self.hit_count();
    let misses = self.miss_count();
    let total = hits + misses;
    
    CacheStats {
        entries: cache_guard.len(),
        memory_usage_bytes: memory_usage,
        memory_usage_mb: memory_usage as f64 / (1024.0 * 1024.0),
        hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        total_hits: hits,
        total_misses: misses,
    }
}
```

### 6. Add cache performance debugging
Add this method for debugging cache performance:
```rust
pub fn debug_info(&self) -> String {
    let cache = self.query_cache.read().unwrap();
    let stats = self.get_stats();
    
    let mut debug_info = format!(
        "Cache Debug Info:\n  Entries: {}/{}\n  Memory: {:.2}/{} MB ({:.1}% full)\n  Hit Rate: {:.1}% ({} hits, {} misses)\n",
        stats.entries, self.max_entries,
        stats.memory_usage_mb, self.max_memory_mb,
        stats.memory_utilization(self.max_memory_mb) * 100.0,
        stats.hit_rate * 100.0, stats.total_hits, stats.total_misses
    );
    
    // Add most accessed entries info
    if !cache.is_empty() {
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by(|a, b| b.1.access_count.cmp(&a.1.access_count));
        
        debug_info.push_str("  Top accessed queries:\n");
        for (i, (query, entry)) in entries.iter().take(5).enumerate() {
            debug_info.push_str(&format!(
                "    {}. '{}' ({} accesses, {} results)\n",
                i + 1, 
                if query.len() > 30 { &query[..30] } else { query },
                entry.access_count,
                entry.results.len()
            ));
        }
    }
    
    debug_info
}
```

### 7. Add comprehensive get() tests
Add these tests to the test module:
```rust
#[test]
fn test_cache_get_hit() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    // Initially empty
    assert!(cache.get("test query").is_none());
    assert_eq!(cache.hit_count(), 0);
    assert_eq!(cache.miss_count(), 1);
    
    // Add an entry directly for testing (put method will be implemented later)
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "pub fn test() {}".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    {
        let mut cache_map = cache.query_cache.write().unwrap();
        cache_map.insert("test query".to_string(), CacheEntry::new(test_results.clone()));
    }
    
    // Now should get a hit
    let cached_results = cache.get("test query");
    assert!(cached_results.is_some());
    assert_eq!(cached_results.unwrap().len(), 1);
    assert_eq!(cache.hit_count(), 1);
    assert_eq!(cache.miss_count(), 1);
}

#[test]
fn test_cache_get_miss() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    // Multiple misses
    assert!(cache.get("query1").is_none());
    assert!(cache.get("query2").is_none());
    assert!(cache.get("query3").is_none());
    
    assert_eq!(cache.hit_count(), 0);
    assert_eq!(cache.miss_count(), 3);
    assert_eq!(cache.total_requests(), 3);
    assert_eq!(cache.hit_rate(), 0.0);
}

#[test]
fn test_cache_access_counting() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    // Add an entry
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "content".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    {
        let mut cache_map = cache.query_cache.write().unwrap();
        cache_map.insert("popular query".to_string(), CacheEntry::new(test_results));
    }
    
    // Access multiple times
    for _ in 0..5 {
        let _ = cache.get("popular query");
    }
    
    // Check access count was updated
    {
        let cache_map = cache.query_cache.read().unwrap();
        let entry = cache_map.get("popular query").unwrap();
        assert_eq!(entry.access_count, 6); // 1 initial + 5 accesses
    }
    
    assert_eq!(cache.hit_count(), 5);
    assert_eq!(cache.miss_count(), 0);
    assert_eq!(cache.hit_rate(), 1.0);
}

#[test]
fn test_cache_hit_rate_calculation() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    // Add some test data
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "content".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    {
        let mut cache_map = cache.query_cache.write().unwrap();
        cache_map.insert("cached query".to_string(), CacheEntry::new(test_results));
    }
    
    // 3 hits, 2 misses = 60% hit rate
    let _ = cache.get("cached query");  // hit
    let _ = cache.get("cached query");  // hit
    let _ = cache.get("cached query");  // hit
    let _ = cache.get("missing1");      // miss
    let _ = cache.get("missing2");      // miss
    
    assert_eq!(cache.hit_count(), 3);
    assert_eq!(cache.miss_count(), 2);
    assert_eq!(cache.total_requests(), 5);
    assert!((cache.hit_rate() - 0.6).abs() < 0.001); // 60% hit rate
}

#[test]
fn test_cache_debug_info() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    let debug_info = cache.debug_info();
    assert!(debug_info.contains("Cache Debug Info"));
    assert!(debug_info.contains("Entries: 0/100"));
    assert!(debug_info.contains("Hit Rate: 0.0%"));
}
```

## Success Criteria
- [ ] `get()` method implemented with hit/miss tracking
- [ ] Hit rate calculations are accurate
- [ ] Access counting works correctly
- [ ] Cache statistics are updated properly
- [ ] Debug information provides useful insights
- [ ] All tests pass
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Use write lock for get() to update access patterns
- Clone results to avoid borrowing issues
- Hit rate calculation handles division by zero
- Debug info helps with cache tuning and optimization
- Access counting enables LRU eviction in later tasks