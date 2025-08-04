# Task 15: Implement Cache Put Method

## Context
You are implementing Phase 4 of a vector indexing system. The cache retrieval functionality was implemented in the previous task. Now you need to implement cache storage with size checks, eviction policies, and memory management.

## Current State
- `src/cache.rs` exists with `MemoryEfficientCache` struct
- `get()` method implemented with hit tracking in task 14
- `CacheEntry` struct handles memory estimation
- Basic configuration and statistics are in place

## Task Objective
Implement the `put()` method for cache storage with memory limit enforcement, entry count limits, and basic eviction when limits are exceeded.

## Implementation Requirements

### 1. Implement the put() method
Add this method to the `MemoryEfficientCache` implementation:
```rust
pub fn put(&self, query: String, results: Vec<SearchResult>) -> bool {
    let entry = CacheEntry::new(results);
    let entry_size = entry.estimated_size;
    
    let mut cache = self.query_cache.write().unwrap();
    let mut memory_usage = self.current_memory_usage.write().unwrap();
    
    // Check if we already have this query (update case)
    if let Some(existing_entry) = cache.get(&query) {
        let old_size = existing_entry.estimated_size;
        cache.insert(query, entry);
        *memory_usage = memory_usage.saturating_sub(old_size) + entry_size;
        return true;
    }
    
    // Check memory limit before adding new entry
    let projected_memory_mb = (*memory_usage + entry_size) as f64 / (1024.0 * 1024.0);
    if projected_memory_mb > self.max_memory_mb as f64 {
        // Try to make space by removing oldest entries
        if !self.make_space_for_entry(&mut cache, &mut memory_usage, entry_size) {
            return false; // Could not make enough space
        }
    }
    
    // Check entry count limit
    if cache.len() >= self.max_entries {
        // Try to make space by removing oldest entry
        if !self.remove_oldest_entry(&mut cache, &mut memory_usage) {
            return false; // Could not remove entry
        }
    }
    
    // Add the new entry
    cache.insert(query, entry);
    *memory_usage += entry_size;
    
    true
}
```

### 2. Implement space management helper methods
Add these private helper methods:
```rust
fn make_space_for_entry(
    &self,
    cache: &mut HashMap<String, CacheEntry>,
    memory_usage: &mut usize,
    needed_size: usize,
) -> bool {
    let target_memory_bytes = (self.max_memory_mb * 1024 * 1024) as usize;
    let required_space = needed_size;
    
    // Find entries to remove (oldest first)
    let mut entries_to_remove: Vec<_> = cache
        .iter()
        .map(|(k, v)| (k.clone(), v.timestamp, v.estimated_size))
        .collect();
    
    // Sort by timestamp (oldest first)
    entries_to_remove.sort_by_key(|(_, timestamp, _)| *timestamp);
    
    let mut freed_space = 0;
    let mut removed_keys = Vec::new();
    
    for (key, _, size) in entries_to_remove {
        if *memory_usage + required_space <= target_memory_bytes + freed_space {
            break;
        }
        
        freed_space += size;
        removed_keys.push(key);
        
        // Don't remove more than half the cache at once
        if removed_keys.len() >= cache.len() / 2 {
            break;
        }
    }
    
    // Remove the selected entries
    for key in removed_keys {
        if let Some(entry) = cache.remove(&key) {
            *memory_usage = memory_usage.saturating_sub(entry.estimated_size);
        }
    }
    
    // Check if we freed enough space
    *memory_usage + required_space <= target_memory_bytes
}

fn remove_oldest_entry(
    &self,
    cache: &mut HashMap<String, CacheEntry>,
    memory_usage: &mut usize,
) -> bool {
    if cache.is_empty() {
        return false;
    }
    
    // Find the oldest entry
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
```

### 3. Add cache validation method
Add this method to validate cache consistency:
```rust
pub fn validate_cache(&self) -> Result<(), String> {
    let cache = self.query_cache.read().unwrap();
    let reported_memory = *self.current_memory_usage.read().unwrap();
    
    // Calculate actual memory usage
    let actual_memory: usize = cache.values()
        .map(|entry| entry.estimated_size)
        .sum();
    
    if actual_memory != reported_memory {
        return Err(format!(
            "Memory usage mismatch: reported {}, actual {}",
            reported_memory, actual_memory
        ));
    }
    
    // Check memory limit
    let memory_mb = reported_memory as f64 / (1024.0 * 1024.0);
    if memory_mb > self.max_memory_mb as f64 * 1.1 { // Allow 10% tolerance
        return Err(format!(
            "Memory usage {:.2}MB exceeds limit {}MB",
            memory_mb, self.max_memory_mb
        ));
    }
    
    // Check entry count limit
    if cache.len() > self.max_entries {
        return Err(format!(
            "Entry count {} exceeds limit {}",
            cache.len(), self.max_entries
        ));
    }
    
    Ok(())
}
```

### 4. Add cache clearing methods
Add these utility methods:
```rust
pub fn clear(&self) {
    let mut cache = self.query_cache.write().unwrap();
    let mut memory_usage = self.current_memory_usage.write().unwrap();
    
    cache.clear();
    *memory_usage = 0;
}

pub fn remove(&self, query: &str) -> bool {
    let mut cache = self.query_cache.write().unwrap();
    let mut memory_usage = self.current_memory_usage.write().unwrap();
    
    if let Some(entry) = cache.remove(query) {
        *memory_usage = memory_usage.saturating_sub(entry.estimated_size);
        true
    } else {
        false
    }
}

pub fn contains_key(&self, query: &str) -> bool {
    let cache = self.query_cache.read().unwrap();
    cache.contains_key(query)
}
```

### 5. Add comprehensive put() tests
Add these tests to the test module:
```rust
#[test]
fn test_cache_put_basic() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "pub fn test() {}".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // Put should succeed
    assert!(cache.put("test query".to_string(), test_results.clone()));
    assert_eq!(cache.current_entries(), 1);
    assert!(cache.current_memory_usage() > 0);
    
    // Should be able to retrieve it
    let cached = cache.get("test query");
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().len(), 1);
}

#[test]
fn test_cache_put_update_existing() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    let initial_results = vec![
        SearchResult {
            file_path: "test1.rs".to_string(),
            content: "content1".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    let updated_results = vec![
        SearchResult {
            file_path: "test2.rs".to_string(),
            content: "content2".to_string(),
            chunk_index: 0,
            score: 0.9,
        },
        SearchResult {
            file_path: "test3.rs".to_string(),
            content: "content3".to_string(),
            chunk_index: 1,
            score: 0.8,
        }
    ];
    
    // Put initial
    assert!(cache.put("query".to_string(), initial_results));
    assert_eq!(cache.current_entries(), 1);
    let initial_memory = cache.current_memory_usage();
    
    // Update with more results
    assert!(cache.put("query".to_string(), updated_results));
    assert_eq!(cache.current_entries(), 1); // Still one entry
    
    // Memory usage should have changed
    let final_memory = cache.current_memory_usage();
    assert_ne!(initial_memory, final_memory);
    
    // Should get updated results
    let cached = cache.get("query").unwrap();
    assert_eq!(cached.len(), 2);
}

#[test]
fn test_cache_entry_limit_eviction() {
    let cache = MemoryEfficientCache::new(2, 100); // Only 2 entries allowed
    
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "small content".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // Add first two entries
    assert!(cache.put("query1".to_string(), test_results.clone()));
    assert!(cache.put("query2".to_string(), test_results.clone()));
    assert_eq!(cache.current_entries(), 2);
    
    // Adding third should evict oldest
    assert!(cache.put("query3".to_string(), test_results.clone()));
    assert_eq!(cache.current_entries(), 2);
    
    // query1 should be evicted (oldest)
    assert!(cache.get("query1").is_none());
    assert!(cache.get("query2").is_some());
    assert!(cache.get("query3").is_some());
}

#[test]
fn test_cache_memory_limit_eviction() {
    // Very small memory limit to force eviction
    let cache = MemoryEfficientCache::new(100, 1); // 1MB limit
    
    // Create large content that should exceed limit
    let large_content = "x".repeat(300_000); // ~300KB per result
    let large_results = vec![
        SearchResult {
            file_path: "large1.rs".to_string(),
            content: large_content.clone(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // First entry should fit
    assert!(cache.put("large1".to_string(), large_results.clone()));
    assert_eq!(cache.current_entries(), 1);
    
    // Second entry should fit
    assert!(cache.put("large2".to_string(), large_results.clone()));
    
    // Third entry should trigger eviction or fail
    let result = cache.put("large3".to_string(), large_results.clone());
    
    // Either it succeeds with eviction, or fails due to size
    if result {
        // If it succeeded, check that memory limit is respected
        let memory_mb = cache.current_memory_usage_mb();
        assert!(memory_mb <= 1.1); // Allow small tolerance
    }
}

#[test]
fn test_cache_clear() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "content".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // Add some entries
    cache.put("query1".to_string(), test_results.clone());
    cache.put("query2".to_string(), test_results.clone());
    
    assert_eq!(cache.current_entries(), 2);
    assert!(cache.current_memory_usage() > 0);
    
    // Clear cache
    cache.clear();
    
    assert_eq!(cache.current_entries(), 0);
    assert_eq!(cache.current_memory_usage(), 0);
    assert!(cache.get("query1").is_none());
    assert!(cache.get("query2").is_none());
}

#[test]
fn test_cache_remove() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "content".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // Add entry
    cache.put("query".to_string(), test_results);
    assert!(cache.contains_key("query"));
    assert_eq!(cache.current_entries(), 1);
    let initial_memory = cache.current_memory_usage();
    
    // Remove entry
    assert!(cache.remove("query"));
    assert!(!cache.contains_key("query"));
    assert_eq!(cache.current_entries(), 0);
    assert_eq!(cache.current_memory_usage(), 0);
    
    // Remove non-existent entry
    assert!(!cache.remove("nonexistent"));
}

#[test]
fn test_cache_validation() {
    let cache = MemoryEfficientCache::new(100, 10);
    
    // Empty cache should validate
    assert!(cache.validate_cache().is_ok());
    
    let test_results = vec![
        SearchResult {
            file_path: "test.rs".to_string(),
            content: "content".to_string(),
            chunk_index: 0,
            score: 1.0,
        }
    ];
    
    // Add entry and validate
    cache.put("query".to_string(), test_results);
    assert!(cache.validate_cache().is_ok());
}
```

## Success Criteria
- [ ] `put()` method implemented with memory and entry limits
- [ ] Cache eviction works correctly when limits are exceeded
- [ ] Memory usage tracking is accurate
- [ ] Cache validation catches inconsistencies
- [ ] Clear and remove operations work properly
- [ ] All tests pass
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Eviction removes oldest entries first (timestamp-based)
- Memory usage is tracked precisely with add/subtract operations
- Put method handles both new entries and updates
- Validation helps catch cache corruption issues
- Space management prevents memory limit violations