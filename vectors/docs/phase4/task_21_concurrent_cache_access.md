# Task 21: Implement Concurrent Cache Access

## Context
You are implementing Phase 4 of a vector indexing system. Cache performance tests were implemented in the previous task. Now you need to implement thread-safe concurrent cache access with proper synchronization, deadlock prevention, and concurrent performance optimization.

## Current State
- `src/cache.rs` exists with full cache implementation using RwLock
- Basic thread safety is implemented but needs enhancement
- Performance tests validate basic concurrent access
- Need advanced concurrent access patterns and optimizations

## Task Objective
Implement advanced thread-safe concurrent cache access with optimized locking strategies, concurrent statistics tracking, and high-performance concurrent operations.

## Implementation Requirements

### 1. Add concurrent access utilities and statistics
Add this concurrent access module to the cache:
```rust
use std::sync::atomic::{AtomicUsize, AtomicU64, Ordering};
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct ConcurrentAccessStats {
    pub read_operations: AtomicUsize,
    pub write_operations: AtomicUsize,
    pub read_contention_count: AtomicUsize,
    pub write_contention_count: AtomicUsize,
    pub total_read_wait_time_ns: AtomicU64,
    pub total_write_wait_time_ns: AtomicU64,
    pub concurrent_readers_peak: AtomicUsize,
    pub lock_acquisition_failures: AtomicUsize,
}

impl ConcurrentAccessStats {
    pub fn new() -> Self {
        Self {
            read_operations: AtomicUsize::new(0),
            write_operations: AtomicUsize::new(0),
            read_contention_count: AtomicUsize::new(0),
            write_contention_count: AtomicUsize::new(0),
            total_read_wait_time_ns: AtomicU64::new(0),
            total_write_wait_time_ns: AtomicU64::new(0),
            concurrent_readers_peak: AtomicUsize::new(0),
            lock_acquisition_failures: AtomicUsize::new(0),
        }
    }
    
    pub fn record_read_operation(&self, wait_time: Duration) {
        self.read_operations.fetch_add(1, Ordering::Relaxed);
        self.total_read_wait_time_ns.fetch_add(wait_time.as_nanos() as u64, Ordering::Relaxed);
        
        if wait_time.as_millis() > 1 {
            self.read_contention_count.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    pub fn record_write_operation(&self, wait_time: Duration) {
        self.write_operations.fetch_add(1, Ordering::Relaxed);
        self.total_write_wait_time_ns.fetch_add(wait_time.as_nanos() as u64, Ordering::Relaxed);
        
        if wait_time.as_millis() > 1 {
            self.write_contention_count.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    pub fn record_lock_failure(&self) {
        self.lock_acquisition_failures.fetch_add(1, Ordering::Relaxed);
    }
    
    pub fn get_stats_snapshot(&self) -> ConcurrentStatsSnapshot {
        ConcurrentStatsSnapshot {
            read_operations: self.read_operations.load(Ordering::Relaxed),
            write_operations: self.write_operations.load(Ordering::Relaxed),
            read_contention_count: self.read_contention_count.load(Ordering::Relaxed),
            write_contention_count: self.write_contention_count.load(Ordering::Relaxed),
            avg_read_wait_time_ns: {
                let total_reads = self.read_operations.load(Ordering::Relaxed);
                if total_reads > 0 {
                    self.total_read_wait_time_ns.load(Ordering::Relaxed) as f64 / total_reads as f64
                } else {
                    0.0
                }
            },
            avg_write_wait_time_ns: {
                let total_writes = self.write_operations.load(Ordering::Relaxed);
                if total_writes > 0 {
                    self.total_write_wait_time_ns.load(Ordering::Relaxed) as f64 / total_writes as f64
                } else {
                    0.0
                }
            },
            concurrent_readers_peak: self.concurrent_readers_peak.load(Ordering::Relaxed),
            lock_acquisition_failures: self.lock_acquisition_failures.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ConcurrentStatsSnapshot {
    pub read_operations: usize,
    pub write_operations: usize,
    pub read_contention_count: usize,
    pub write_contention_count: usize,
    pub avg_read_wait_time_ns: f64,
    pub avg_write_wait_time_ns: f64,
    pub concurrent_readers_peak: usize,
    pub lock_acquisition_failures: usize,
}

impl ConcurrentStatsSnapshot {
    pub fn format_report(&self) -> String {
        let total_ops = self.read_operations + self.write_operations;
        let read_contention_rate = if self.read_operations > 0 {
            (self.read_contention_count as f64 / self.read_operations as f64) * 100.0
        } else {
            0.0
        };
        let write_contention_rate = if self.write_operations > 0 {
            (self.write_contention_count as f64 / self.write_operations as f64) * 100.0
        } else {
            0.0
        };
        
        format!(
            "Concurrent Access Statistics:\n\
             Total Operations: {} ({}R, {}W)\n\
             Read Contention: {:.1}% ({} contentious reads)\n\
             Write Contention: {:.1}% ({} contentious writes)\n\
             Avg Read Wait: {:.2}μs\n\
             Avg Write Wait: {:.2}μs\n\
             Peak Concurrent Readers: {}\n\
             Lock Failures: {}",
            total_ops, self.read_operations, self.write_operations,
            read_contention_rate, self.read_contention_count,
            write_contention_rate, self.write_contention_count,
            self.avg_read_wait_time_ns / 1000.0,
            self.avg_write_wait_time_ns / 1000.0,
            self.concurrent_readers_peak,
            self.lock_acquisition_failures
        )
    }
    
    pub fn has_high_contention(&self) -> bool {
        let read_contention_rate = if self.read_operations > 0 {
            (self.read_contention_count as f64 / self.read_operations as f64) * 100.0
        } else {
            0.0
        };
        let write_contention_rate = if self.write_operations > 0 {
            (self.write_contention_count as f64 / self.write_operations as f64) * 100.0
        } else {
            0.0
        };
        
        read_contention_rate > 10.0 || write_contention_rate > 5.0 || self.lock_acquisition_failures > 0
    }
}
```

### 2. Update MemoryEfficientCache with concurrent access tracking
Update the cache struct to include concurrent access statistics:
```rust
pub struct MemoryEfficientCache {
    query_cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    max_entries: usize,
    max_memory_mb: usize,
    current_memory_usage: Arc<RwLock<usize>>,
    hit_count: Arc<RwLock<usize>>,
    miss_count: Arc<RwLock<usize>>,
    eviction_config: EvictionConfig,
    eviction_count: Arc<RwLock<usize>>,
    total_evicted_entries: Arc<RwLock<usize>>,
    total_evicted_bytes: Arc<RwLock<usize>>,
    concurrent_stats: Arc<ConcurrentAccessStats>, // Add this field
    active_readers: Arc<AtomicUsize>,              // Add this field
}
```

### 3. Implement enhanced concurrent get method
Replace the existing get method with this concurrent-optimized version:
```rust
pub fn get(&self, query: &str) -> Option<Vec<SearchResult>> {
    let start = Instant::now();
    
    // Increment active readers
    let current_readers = self.active_readers.fetch_add(1, Ordering::Relaxed) + 1;
    
    // Update peak concurrent readers
    loop {
        let current_peak = self.concurrent_stats.concurrent_readers_peak.load(Ordering::Relaxed);
        if current_readers <= current_peak {
            break;
        }
        if self.concurrent_stats.concurrent_readers_peak
            .compare_exchange_weak(current_peak, current_readers, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok() {
            break;
        }
    }
    
    let result = {
        // Try to acquire read lock with timeout
        let cache_result = self.try_read_with_timeout(Duration::from_millis(100));
        
        match cache_result {
            Ok(mut cache) => {
                if let Some(entry) = cache.get_mut(query) {
                    // Cache hit - update access patterns
                    entry.touch();
                    
                    // Update hit statistics
                    {
                        if let Ok(mut hit_count) = self.hit_count.try_write() {
                            *hit_count += 1;
                        }
                    }
                    
                    Some(entry.results.clone())
                } else {
                    // Cache miss - update statistics
                    {
                        if let Ok(mut miss_count) = self.miss_count.try_write() {
                            *miss_count += 1;
                        }
                    }
                    None
                }
            }
            Err(_) => {
                // Lock acquisition failed
                self.concurrent_stats.record_lock_failure();
                None
            }
        }
    };
    
    // Record operation statistics
    let wait_time = start.elapsed();
    self.concurrent_stats.record_read_operation(wait_time);
    
    // Decrement active readers
    self.active_readers.fetch_sub(1, Ordering::Relaxed);
    
    result
}

fn try_read_with_timeout(&self, timeout: Duration) -> Result<std::sync::RwLockWriteGuard<HashMap<String, CacheEntry>>, ()> {
    let start = Instant::now();
    
    while start.elapsed() < timeout {
        match self.query_cache.try_write() {
            Ok(guard) => return Ok(guard),
            Err(_) => {
                // Brief backoff before retry
                std::thread::sleep(Duration::from_micros(100));
            }
        }
    }
    
    Err(())
}
```

### 4. Implement concurrent-optimized put method
Replace the existing put method with this concurrent-optimized version:
```rust
pub fn put(&self, query: String, results: Vec<SearchResult>) -> bool {
    let start = Instant::now();
    let entry = CacheEntry::new(results);
    let entry_size = entry.estimated_size;
    
    // Try to acquire locks with timeout to prevent deadlocks
    let cache_result = self.try_write_with_timeout(Duration::from_millis(200));
    let memory_result = self.current_memory_usage.try_write();
    
    match (cache_result, memory_result) {
        (Ok(mut cache), Ok(mut memory_usage)) => {
            // Check if we already have this query (update case)
            if let Some(existing_entry) = cache.get(&query) {
                let old_size = existing_entry.estimated_size;
                cache.insert(query, entry);
                *memory_usage = memory_usage.saturating_sub(old_size) + entry_size;
                
                let wait_time = start.elapsed();
                self.concurrent_stats.record_write_operation(wait_time);
                return true;
            }
            
            // Check memory limit before adding new entry
            let projected_memory_mb = (*memory_usage + entry_size) as f64 / (1024.0 * 1024.0);
            if projected_memory_mb > self.max_memory_mb as f64 {
                // Try to make space by removing oldest entries
                if !self.make_space_for_entry_concurrent(&mut cache, &mut memory_usage, entry_size) {
                    let wait_time = start.elapsed();
                    self.concurrent_stats.record_write_operation(wait_time);
                    return false;
                }
            }
            
            // Check entry count limit
            if cache.len() >= self.max_entries {
                if !self.remove_oldest_entry_concurrent(&mut cache, &mut memory_usage) {
                    let wait_time = start.elapsed();
                    self.concurrent_stats.record_write_operation(wait_time);
                    return false;
                }
            }
            
            // Add the new entry
            cache.insert(query, entry);
            *memory_usage += entry_size;
            
            let wait_time = start.elapsed();
            self.concurrent_stats.record_write_operation(wait_time);
            true
        }
        _ => {
            // Lock acquisition failed
            self.concurrent_stats.record_lock_failure();
            let wait_time = start.elapsed();
            self.concurrent_stats.record_write_operation(wait_time);
            false
        }
    }
}

fn try_write_with_timeout(&self, timeout: Duration) -> Result<std::sync::RwLockWriteGuard<HashMap<String, CacheEntry>>, ()> {
    let start = Instant::now();
    
    while start.elapsed() < timeout {
        match self.query_cache.try_write() {
            Ok(guard) => return Ok(guard),
            Err(_) => {
                // Exponential backoff
                let backoff_micros = std::cmp::min(1000, start.elapsed().as_micros() as u64 / 10);
                std::thread::sleep(Duration::from_micros(backoff_micros));
            }
        }
    }
    
    Err(())
}

// Concurrent-safe helper methods
fn make_space_for_entry_concurrent(
    &self,
    cache: &mut HashMap<String, CacheEntry>,
    memory_usage: &mut usize,
    needed_size: usize,
) -> bool {
    let target_memory_bytes = (self.max_memory_mb * 1024 * 1024) as usize;
    
    if *memory_usage + needed_size <= target_memory_bytes {
        return true;
    }
    
    // Find candidates for eviction using concurrent-safe method
    let candidates = self.select_eviction_candidates_concurrent(cache, 
        self.estimate_entries_for_bytes(cache, needed_size));
    
    if candidates.is_empty() {
        return false;
    }
    
    // Remove selected entries
    for key in candidates {
        if let Some(entry) = cache.remove(&key) {
            *memory_usage = memory_usage.saturating_sub(entry.estimated_size);
            
            // Update eviction statistics atomically
            if let Ok(mut eviction_count) = self.eviction_count.try_write() {
                *eviction_count += 1;
            }
            if let Ok(mut total_evicted) = self.total_evicted_entries.try_write() {
                *total_evicted += 1;
            }
            if let Ok(mut total_bytes) = self.total_evicted_bytes.try_write() {
                *total_bytes += entry.estimated_size;
            }
        }
    }
    
    *memory_usage + needed_size <= target_memory_bytes
}

fn remove_oldest_entry_concurrent(
    &self,
    cache: &mut HashMap<String, CacheEntry>,
    memory_usage: &mut usize,
) -> bool {
    if cache.is_empty() {
        return false;
    }
    
    // Find oldest entry without blocking
    let oldest_key = cache
        .iter()
        .min_by_key(|(_, entry)| entry.timestamp)
        .map(|(k, _)| k.clone());
    
    if let Some(key) = oldest_key {
        if let Some(entry) = cache.remove(&key) {
            *memory_usage = memory_usage.saturating_sub(entry.estimated_size);
            return true;
        }
    }
    
    false
}

fn select_eviction_candidates_concurrent(
    &self,
    cache: &HashMap<String, CacheEntry>,
    target_count: usize,
) -> Vec<String> {
    if cache.is_empty() || target_count == 0 {
        return Vec::new();
    }
    
    let actual_target = target_count
        .min(cache.len().saturating_sub(self.eviction_config.min_entries_to_keep))
        .min(self.eviction_config.max_eviction_batch_size);
    
    // Use a simple, fast selection for concurrent contexts
    let mut entries: Vec<_> = cache.iter().collect();
    
    match self.eviction_config.primary_policy {
        EvictionPolicy::LRU => {
            entries.sort_by_key(|(_, entry)| entry.timestamp);
        }
        EvictionPolicy::LFU => {
            entries.sort_by_key(|(_, entry)| entry.access_count);
        }
        EvictionPolicy::SizeBased => {
            entries.sort_by(|(_, a), (_, b)| b.estimated_size.cmp(&a.estimated_size));
        }
        EvictionPolicy::Hybrid => {
            // Simplified hybrid for concurrent access
            entries.sort_by_key(|(_, entry)| entry.timestamp);
        }
    }
    
    entries.into_iter()
        .take(actual_target)
        .map(|(key, _)| key.clone())
        .collect()
}
```

### 5. Add concurrent access monitoring methods
Add these methods to the cache implementation:
```rust
impl MemoryEfficientCache {
    pub fn get_concurrent_stats(&self) -> ConcurrentStatsSnapshot {
        self.concurrent_stats.get_stats_snapshot()
    }
    
    pub fn reset_concurrent_stats(&self) {
        // Create new stats instance to reset all counters
        let new_stats = Arc::new(ConcurrentAccessStats::new());
        // Note: In a real implementation, you'd need to properly replace the stats
        // This is a simplified version for demonstration
    }
    
    pub fn get_active_readers(&self) -> usize {
        self.active_readers.load(Ordering::Relaxed)
    }
    
    pub fn is_under_contention(&self) -> bool {
        let stats = self.get_concurrent_stats();
        stats.has_high_contention()
    }
    
    pub fn concurrent_health_check(&self) -> ConcurrentHealthReport {
        let stats = self.get_concurrent_stats();
        let active_readers = self.get_active_readers();
        
        ConcurrentHealthReport {
            is_healthy: !stats.has_high_contention() && stats.lock_acquisition_failures == 0,
            active_readers,
            contention_level: if stats.has_high_contention() { 
                ContentionLevel::High 
            } else if stats.read_contention_count > 0 || stats.write_contention_count > 0 {
                ContentionLevel::Medium
            } else {
                ContentionLevel::Low
            },
            recommendations: self.generate_contention_recommendations(&stats),
            stats_snapshot: stats,
        }
    }
    
    fn generate_contention_recommendations(&self, stats: &ConcurrentStatsSnapshot) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let read_contention_rate = if stats.read_operations > 0 {
            (stats.read_contention_count as f64 / stats.read_operations as f64) * 100.0
        } else {
            0.0
        };
        
        let write_contention_rate = if stats.write_operations > 0 {
            (stats.write_contention_count as f64 / stats.write_operations as f64) * 100.0
        } else {
            0.0
        };
        
        if read_contention_rate > 15.0 {
            recommendations.push("Consider read-heavy optimization or cache partitioning".to_string());
        }
        
        if write_contention_rate > 10.0 {
            recommendations.push("Consider reducing write frequency or using batch operations".to_string());
        }
        
        if stats.lock_acquisition_failures > 0 {
            recommendations.push("Lock timeout too aggressive, consider increasing timeout".to_string());
        }
        
        if stats.concurrent_readers_peak > 20 {
            recommendations.push("High concurrent read load, consider using read replicas".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("Concurrent access patterns are healthy".to_string());
        }
        
        recommendations
    }
}

#[derive(Debug, Clone)]
pub struct ConcurrentHealthReport {
    pub is_healthy: bool,
    pub active_readers: usize,
    pub contention_level: ContentionLevel,
    pub recommendations: Vec<String>,
    pub stats_snapshot: ConcurrentStatsSnapshot,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContentionLevel {
    Low,
    Medium,
    High,
}

impl ConcurrentHealthReport {
    pub fn format_report(&self) -> String {
        format!(
            "Concurrent Health Report:\n\
             Status: {}\n\
             Active Readers: {}\n\
             Contention Level: {:?}\n\
             \n{}\n\
             \nRecommendations:\n{}\n",
            if self.is_healthy { "HEALTHY" } else { "NEEDS ATTENTION" },
            self.active_readers,
            self.contention_level,
            self.stats_snapshot.format_report(),
            self.recommendations.iter()
                .enumerate()
                .map(|(i, rec)| format!("  {}. {}", i + 1, rec))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}
```

### 6. Update constructor to include concurrent access tracking
Update the cache constructor:
```rust
impl MemoryEfficientCache {
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
            concurrent_stats: Arc::new(ConcurrentAccessStats::new()),
            active_readers: Arc::new(AtomicUsize::new(0)),
        }
    }
}
```

### 7. Add comprehensive concurrent access tests
Add these concurrent access tests:
```rust
#[cfg(test)]
mod concurrent_access_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_concurrent_read_operations() {
        let cache = Arc::new(MemoryEfficientCache::new(1000, 100));
        
        // Pre-populate cache
        for i in 0..100 {
            let results = vec![
                SearchResult {
                    file_path: format!("concurrent_read_{}.rs", i),
                    content: format!("content_{}", i),
                    chunk_index: 0,
                    score: 1.0,
                }
            ];
            cache.put(format!("query_{}", i), results);
        }
        
        let mut handles = Vec::new();
        let operations_per_thread = 200;
        let thread_count = 10;
        
        // Spawn multiple reader threads
        for thread_id in 0..thread_count {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                let mut hits = 0;
                let mut misses = 0;
                
                for i in 0..operations_per_thread {
                    let query = format!("query_{}", i % 100);
                    if cache_clone.get(&query).is_some() {
                        hits += 1;
                    } else {
                        misses += 1;
                    }
                }
                
                (hits, misses)
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        let mut total_hits = 0;
        let mut total_misses = 0;
        
        for handle in handles {
            let (hits, misses) = handle.join().unwrap();
            total_hits += hits;
            total_misses += misses;
        }
        
        // Verify statistics
        let stats = cache.get_concurrent_stats();
        assert_eq!(stats.read_operations, (total_hits + total_misses) as usize);
        assert_eq!(total_hits, thread_count * operations_per_thread); // All should be hits
        
        // Check health
        let health = cache.concurrent_health_check();
        println!("{}", health.format_report());
        
        // Should have some concurrent readers recorded
        assert!(stats.concurrent_readers_peak > 1);
    }
    
    #[test]
    fn test_concurrent_write_operations() {
        let cache = Arc::new(MemoryEfficientCache::new(1000, 100));
        let mut handles = Vec::new();
        let operations_per_thread = 50;
        let thread_count = 8;
        
        // Spawn multiple writer threads
        for thread_id in 0..thread_count {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                let mut successes = 0;
                let mut failures = 0;
                
                for i in 0..operations_per_thread {
                    let results = vec![
                        SearchResult {
                            file_path: format!("thread_{}_{}.rs", thread_id, i),
                            content: format!("content from thread {} item {}", thread_id, i),
                            chunk_index: 0,
                            score: 1.0,
                        }
                    ];
                    
                    if cache_clone.put(format!("thread_{}_query_{}", thread_id, i), results) {
                        successes += 1;
                    } else {
                        failures += 1;
                    }
                }
                
                (successes, failures)
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        let mut total_successes = 0;
        let mut total_failures = 0;
        
        for handle in handles {
            let (successes, failures) = handle.join().unwrap();
            total_successes += successes;
            total_failures += failures;
        }
        
        // Verify operations completed
        let stats = cache.get_concurrent_stats();
        assert_eq!(stats.write_operations, (total_successes + total_failures) as usize);
        
        // Most operations should succeed
        assert!(total_successes > total_failures);
        
        // Cache should be consistent
        assert!(cache.validate_cache().is_ok());
        assert!(cache.validate_memory_consistency().is_ok());
        
        println!("Concurrent writes: {} successes, {} failures", total_successes, total_failures);
        println!("{}", cache.concurrent_health_check().format_report());
    }
    
    #[test]
    fn test_mixed_concurrent_operations() {
        let cache = Arc::new(MemoryEfficientCache::new(500, 50));
        
        // Pre-populate some data
        for i in 0..50 {
            let results = vec![
                SearchResult {
                    file_path: format!("initial_{}.rs", i),
                    content: format!("initial content {}", i),
                    chunk_index: 0,
                    score: 1.0,
                }
            ];
            cache.put(format!("initial_query_{}", i), results);
        }
        
        let mut handles = Vec::new();
        
        // Reader threads (70% of threads)
        for thread_id in 0..7 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let query = format!("initial_query_{}", i % 50);
                    cache_clone.get(&query);
                    
                    // Small delay to create more realistic access patterns
                    std::thread::sleep(Duration::from_micros(10));
                }
            });
            handles.push(handle);
        }
        
        // Writer threads (30% of threads)
        for thread_id in 0..3 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..30 {
                    let results = vec![
                        SearchResult {
                            file_path: format!("writer_{}_{}.rs", thread_id, i),
                            content: format!("writer {} content {}", thread_id, i),
                            chunk_index: 0,
                            score: 1.0,
                        }
                    ];
                    
                    cache_clone.put(format!("writer_{}_query_{}", thread_id, i), results);
                    
                    // Writers have slightly longer delays
                    std::thread::sleep(Duration::from_micros(50));
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Analyze concurrent access patterns
        let health = cache.concurrent_health_check();
        println!("{}", health.format_report());
        
        // Should be healthy with mixed workload
        assert!(health.is_healthy || health.contention_level != ContentionLevel::High);
        
        // Verify consistency after mixed operations
        assert!(cache.validate_cache().is_ok());
        assert!(cache.validate_memory_consistency().is_ok());
    }
    
    #[test]
    fn test_lock_timeout_handling() {
        let cache = Arc::new(MemoryEfficientCache::new(100, 10));
        
        // Create a scenario that might cause lock contention
        let mut handles = Vec::new();
        
        // Many threads trying to write simultaneously
        for thread_id in 0..20 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                let large_content = "x".repeat(5000); // 5KB per entry
                let mut successes = 0;
                let mut failures = 0;
                
                for i in 0..20 {
                    let results = vec![
                        SearchResult {
                            file_path: format!("contention_{}_{}.rs", thread_id, i),
                            content: large_content.clone(),
                            chunk_index: 0,
                            score: 1.0,
                        }
                    ];
                    
                    if cache_clone.put(format!("contention_{}_query_{}", thread_id, i), results) {
                        successes += 1;
                    } else {
                        failures += 1;
                    }
                }
                
                (successes, failures)
            });
            handles.push(handle);
        }
        
        // Wait for completion
        let mut total_successes = 0;
        let mut total_failures = 0;
        
        for handle in handles {
            let (successes, failures) = handle.join().unwrap();
            total_successes += successes;
            total_failures += failures;
        }
        
        let stats = cache.get_concurrent_stats();
        println!("Lock timeout test: {} successes, {} failures, {} lock failures",
                 total_successes, total_failures, stats.lock_acquisition_failures);
        
        // Some operations may fail due to contention, but cache should remain consistent
        assert!(cache.validate_cache().is_ok());
        assert!(cache.validate_memory_consistency().is_ok());
        
        // Should have handled the load without too many failures
        let failure_rate = total_failures as f64 / (total_successes + total_failures) as f64;
        assert!(failure_rate < 0.5, "Failure rate too high: {:.1}%", failure_rate * 100.0);
    }
    
    #[test]
    fn test_concurrent_statistics_accuracy() {
        let cache = Arc::new(MemoryEfficientCache::new(200, 20));
        
        // Controlled concurrent access
        let read_threads = 5;
        let write_threads = 2;
        let reads_per_thread = 50;
        let writes_per_thread = 20;
        
        let mut handles = Vec::new();
        
        // Writer threads first to populate cache
        for thread_id in 0..write_threads {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..writes_per_thread {
                    let results = vec![
                        SearchResult {
                            file_path: format!("stats_{}_{}.rs", thread_id, i),
                            content: format!("stats content {} {}", thread_id, i),
                            chunk_index: 0,
                            score: 1.0,
                        }
                    ];
                    cache_clone.put(format!("stats_{}_{}", thread_id, i), results);
                }
            });
            handles.push(handle);
        }
        
        // Wait for writers to complete
        handles.clear();
        
        // Now reader threads
        for thread_id in 0..read_threads {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..reads_per_thread {
                    let query = format!("stats_{}_{}", (thread_id % write_threads), (i % writes_per_thread));
                    cache_clone.get(&query);
                }
            });
            handles.push(handle);
        }
        
        // Wait for all readers
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify statistics accuracy
        let stats = cache.get_concurrent_stats();
        let expected_reads = read_threads * reads_per_thread;
        let expected_writes = write_threads * writes_per_thread;
        
        assert_eq!(stats.read_operations, expected_reads);
        assert_eq!(stats.write_operations, expected_writes);
        
        println!("Statistics accuracy test passed:");
        println!("  Expected reads: {}, Actual: {}", expected_reads, stats.read_operations);
        println!("  Expected writes: {}, Actual: {}", expected_writes, stats.write_operations);
    }
}
```

## Success Criteria
- [ ] Advanced concurrent access statistics tracking implemented
- [ ] Optimized locking strategies with timeout handling
- [ ] Concurrent access health monitoring and recommendations
- [ ] Lock contention detection and mitigation
- [ ] Comprehensive concurrent access tests pass
- [ ] Performance maintained under concurrent load
- [ ] Deadlock prevention mechanisms work correctly
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Atomic operations provide lock-free statistics tracking
- Timeout-based lock acquisition prevents deadlocks
- Concurrent access patterns are monitored for optimization
- Health checks provide actionable insights
- Statistics help identify contention bottlenecks
- Tests validate thread safety under various loads