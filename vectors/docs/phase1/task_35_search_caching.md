# Task 35: Implement Caching for Repeated Queries

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 34 completed (Search result highlighting)

## Complete Context
You have a complete search engine with ranking and highlighting. Now you need caching to improve performance for repeated queries. This is especially important for code search where developers often search for the same patterns multiple times during development sessions.

The caching system must handle query variations (e.g., "function" vs "Function"), cache both results and parsed queries, and provide configurable cache eviction policies.

## Exact Steps

1. **Add caching infrastructure to SearchEngine** (4 minutes):
Add to `C:/code/LLMKG/vectors/tantivy_search/src/search.rs`:
```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    pub max_entries: usize,
    pub ttl_seconds: u64,
    pub enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl_seconds: 300, // 5 minutes
            enabled: true,
        }
    }
}

/// Cached search result entry
#[derive(Debug, Clone)]
struct CacheEntry {
    results: Vec<SearchResult>,
    timestamp: u64,
    hit_count: usize,
}

/// Search cache with LRU eviction and TTL
pub struct SearchCache {
    cache: Arc<Mutex<HashMap<String, CacheEntry>>>,
    config: CacheConfig,
}

impl SearchCache {
    /// Create new search cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }
    
    /// Get cached results for a query
    pub fn get(&self, cache_key: &str) -> Option<Vec<SearchResult>> {
        if !self.config.enabled {
            return None;
        }
        
        let mut cache = self.cache.lock().unwrap();
        
        if let Some(entry) = cache.get_mut(cache_key) {
            let current_time = Self::current_timestamp();
            
            // Check TTL
            if current_time - entry.timestamp > self.config.ttl_seconds {
                cache.remove(cache_key);
                return None;
            }
            
            // Update hit count for LRU
            entry.hit_count += 1;
            Some(entry.results.clone())
        } else {
            None
        }
    }
    
    /// Cache search results for a query
    pub fn put(&self, cache_key: String, results: Vec<SearchResult>) {
        if !self.config.enabled {
            return;
        }
        
        let mut cache = self.cache.lock().unwrap();
        
        // Evict entries if at capacity
        if cache.len() >= self.config.max_entries {
            self.evict_lru_entries(&mut cache);
        }
        
        let entry = CacheEntry {
            results,
            timestamp: Self::current_timestamp(),
            hit_count: 1,
        };
        
        cache.insert(cache_key, entry);
    }
    
    /// Evict least recently used entries
    fn evict_lru_entries(&self, cache: &mut HashMap<String, CacheEntry>) {
        let entries_to_remove = cache.len() / 4; // Remove 25% when full
        
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by_key(|(_, entry)| entry.hit_count);
        
        for (key, _) in entries.into_iter().take(entries_to_remove) {
            cache.remove(key);
        }
    }
    
    /// Clear all cached entries
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let cache = self.cache.lock().unwrap();
        
        let mut total_hits = 0;
        let mut expired_count = 0;
        let current_time = Self::current_timestamp();
        
        for entry in cache.values() {
            total_hits += entry.hit_count;
            if current_time - entry.timestamp > self.config.ttl_seconds {
                expired_count += 1;
            }
        }
        
        CacheStats {
            total_entries: cache.len(),
            total_hits,
            expired_entries: expired_count,
            max_capacity: self.config.max_entries,
        }
    }
    
    /// Clean up expired entries
    pub fn cleanup_expired(&self) {
        let mut cache = self.cache.lock().unwrap();
        let current_time = Self::current_timestamp();
        
        cache.retain(|_, entry| {
            current_time - entry.timestamp <= self.config.ttl_seconds
        });
    }
    
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_hits: usize,
    pub expired_entries: usize,
    pub max_capacity: usize,
}

impl SearchEngine {
    /// Create SearchEngine with caching enabled
    pub fn with_cache(index_path: &Path, cache_config: CacheConfig) -> Result<Self> {
        let mut engine = Self::new(index_path)?;
        engine.cache = Some(SearchCache::new(cache_config));
        Ok(engine)
    }
    
    /// Search with caching support
    pub fn search_cached(&self, query_str: &str, options: SearchOptions) -> Result<Vec<SearchResult>> {
        let cache_key = self.create_cache_key(query_str, &options);
        
        // Try cache first
        if let Some(cache) = &self.cache {
            if let Some(cached_results) = cache.get(&cache_key) {
                return Ok(cached_results);
            }
        }
        
        // Perform actual search
        let results = self.search(query_str, options)?;
        
        // Cache the results
        if let Some(cache) = &self.cache {
            cache.put(cache_key, results.clone());
        }
        
        Ok(results)
    }
    
    /// Create a cache key from query and options
    fn create_cache_key(&self, query_str: &str, options: &SearchOptions) -> String {
        // Normalize query for consistent caching
        let normalized_query = query_str.to_lowercase().trim().to_string();
        format!("{}|{}|{}", normalized_query, options.limit, options.min_score)
    }
    
    /// Get cache statistics if caching is enabled
    pub fn get_cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(|cache| cache.get_stats())
    }
    
    /// Clear search cache
    pub fn clear_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.clear();
        }
    }
    
    /// Perform cache maintenance (cleanup expired entries)
    pub fn maintain_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.cleanup_expired();
        }
    }
}

// Add cache field to SearchEngine struct
impl SearchEngine {
    // Update the existing struct to include cache
    // Note: This would typically be done by modifying the existing struct definition
    // For the task, we're showing the concept here
    fn add_cache_field(&mut self, cache: Option<SearchCache>) {
        // In real implementation, this would be a field in the struct:
        // pub struct SearchEngine {
        //     index: Index,
        //     schema: Schema,
        //     query_parser: QueryParser,
        //     cache: Option<SearchCache>,  // <-- Add this field
        // }
    }
}
```

2. **Update SearchEngine struct definition** (1 minute):
Replace the SearchEngine struct definition at the top of the file:
```rust
/// Core search engine for Tantivy-based content search
pub struct SearchEngine {
    index: Index,
    schema: Schema,
    query_parser: QueryParser,
    cache: Option<SearchCache>,
}
```

3. **Add comprehensive caching tests** (3 minutes):
Add to the test module in `search.rs`:
```rust
#[cfg(test)]
mod caching_tests {
    use super::*;
    use crate::indexer::DocumentIndexer;
    use tempfile::TempDir;
    use std::fs;
    use std::thread;
    use std::time::Duration;

    fn setup_cached_search_engine() -> Result<(TempDir, SearchEngine)> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("cached_index");
        
        let mut indexer = DocumentIndexer::new(&index_path)?;
        
        let test_file = temp_dir.path().join("test.rs");
        fs::write(&test_file, r#"
            fn process_function() -> Result<String, Error> {
                println!("Processing function");
                Ok("processed".to_string())
            }
        "#)?;
        
        indexer.index_file(&test_file)?;
        indexer.commit()?;
        
        let cache_config = CacheConfig {
            max_entries: 10,
            ttl_seconds: 2, // Short TTL for testing
            enabled: true,
        };
        
        let search_engine = SearchEngine::with_cache(&index_path, cache_config)?;
        Ok((temp_dir, search_engine))
    }
    
    #[test]
    fn test_basic_caching() -> Result<()> {
        let (_temp_dir, search_engine) = setup_cached_search_engine()?;
        let options = SearchOptions::default();
        
        // First search - should miss cache
        let results1 = search_engine.search_cached("function", options.clone())?;
        assert!(!results1.is_empty());
        
        // Second search - should hit cache
        let results2 = search_engine.search_cached("function", options)?;
        assert_eq!(results1.len(), results2.len());
        
        // Verify cache stats
        let stats = search_engine.get_cache_stats().unwrap();
        assert!(stats.total_entries > 0);
        assert!(stats.total_hits > 0);
        
        Ok(())
    }
    
    #[test]
    fn test_cache_key_normalization() -> Result<()> {
        let (_temp_dir, search_engine) = setup_cached_search_engine()?;
        let options = SearchOptions::default();
        
        // These should hit the same cache entry due to normalization
        let _results1 = search_engine.search_cached("Function", options.clone())?;
        let _results2 = search_engine.search_cached("function", options.clone())?;
        let _results3 = search_engine.search_cached("  FUNCTION  ", options)?;
        
        let stats = search_engine.get_cache_stats().unwrap();
        assert_eq!(stats.total_entries, 1, "Should normalize to same cache key");
        assert!(stats.total_hits >= 2, "Should have cache hits");
        
        Ok(())
    }
    
    #[test]
    fn test_cache_ttl_expiration() -> Result<()> {
        let (_temp_dir, search_engine) = setup_cached_search_engine()?;
        let options = SearchOptions::default();
        
        // First search
        let _results1 = search_engine.search_cached("function", options.clone())?;
        
        // Wait for TTL to expire
        thread::sleep(Duration::from_secs(3));
        
        // Should not find in cache due to TTL expiration
        let _results2 = search_engine.search_cached("function", options)?;
        
        // Cleanup expired entries
        search_engine.maintain_cache();
        
        let stats = search_engine.get_cache_stats().unwrap();
        // After cleanup, expired entries should be removed
        assert!(stats.expired_entries == 0);
        
        Ok(())
    }
    
    #[test]
    fn test_cache_capacity_eviction() -> Result<()> {
        let (_temp_dir, search_engine) = setup_cached_search_engine()?;
        let options = SearchOptions::default();
        
        // Fill cache beyond capacity
        for i in 0..15 {
            let query = format!("query{}", i);
            let _ = search_engine.search_cached(&query, options.clone())?;
        }
        
        let stats = search_engine.get_cache_stats().unwrap();
        assert!(stats.total_entries <= stats.max_capacity, 
               "Cache should not exceed max capacity");
        
        Ok(())
    }
    
    #[test]
    fn test_cache_disabled() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let index_path = temp_dir.path().join("no_cache_index");
        
        let mut indexer = DocumentIndexer::new(&index_path)?;
        let test_file = temp_dir.path().join("test.rs");
        fs::write(&test_file, "fn test() {}")?;
        indexer.index_file(&test_file)?;
        indexer.commit()?;
        
        let cache_config = CacheConfig {
            enabled: false,
            ..Default::default()
        };
        
        let search_engine = SearchEngine::with_cache(&index_path, cache_config)?;
        let options = SearchOptions::default();
        
        let _results = search_engine.search_cached("test", options)?;
        
        // Cache should be empty when disabled
        assert!(search_engine.get_cache_stats().is_none() || 
               search_engine.get_cache_stats().unwrap().total_entries == 0);
        
        Ok(())
    }
    
    #[test]
    fn test_cache_clear() -> Result<()> {
        let (_temp_dir, search_engine) = setup_cached_search_engine()?;
        let options = SearchOptions::default();
        
        // Add some entries to cache
        let _results1 = search_engine.search_cached("function", options.clone())?;
        let _results2 = search_engine.search_cached("process", options)?;
        
        let stats_before = search_engine.get_cache_stats().unwrap();
        assert!(stats_before.total_entries > 0);
        
        // Clear cache
        search_engine.clear_cache();
        
        let stats_after = search_engine.get_cache_stats().unwrap();
        assert_eq!(stats_after.total_entries, 0);
        
        Ok(())
    }
    
    #[test]
    fn test_different_search_options_different_cache() -> Result<()> {
        let (_temp_dir, search_engine) = setup_cached_search_engine()?;
        
        let options1 = SearchOptions { limit: 10, ..Default::default() };
        let options2 = SearchOptions { limit: 20, ..Default::default() };
        
        let _results1 = search_engine.search_cached("function", options1)?;
        let _results2 = search_engine.search_cached("function", options2)?;
        
        let stats = search_engine.get_cache_stats().unwrap();
        assert_eq!(stats.total_entries, 2, "Different options should create different cache entries");
        
        Ok(())
    }
}
```

4. **Verify compilation and tests** (2 minutes):
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo check
cargo test caching_tests
```

## Success Validation
✓ SearchEngine::with_cache() creates engine with caching enabled
✓ search_cached() method checks cache before performing search
✓ Cache key normalization works (case-insensitive, trimmed queries)
✓ TTL expiration removes old entries correctly
✓ LRU eviction prevents cache from growing beyond capacity
✓ Cache can be disabled via configuration
✓ Cache statistics provide insight into usage
✓ Different search options create separate cache entries
✓ All caching tests pass

## Next Task Input
Task 36 expects these EXACT components ready:
- `SearchEngine` with `cache: Option<SearchCache>` field
- `SearchCache` with LRU eviction and TTL support
- `search_cached()` method for cached searching
- Cache statistics and maintenance methods