# Task 20: Implement Search Result Caching and Performance Optimization

**Estimated Time:** 10 minutes  
**Prerequisites:** Task 19 (Result ranking)  
**Dependencies:** Tasks 01-19 must be completed

## Objective
Implement intelligent search result caching and performance optimizations to ensure fast search response times for repeated queries and large result sets.

## Context
Search performance is critical for user experience. Caching frequently accessed results, optimizing query execution, and managing memory usage efficiently can dramatically improve search responsiveness, especially for repeated or similar queries.

## Task Details

### What You Need to Do

1. **Create search caching in `src/caching.rs`:**

   ```rust
   use crate::search::{SearchResult, SearchConfig};
   use crate::query::ParsedQuery;
   use std::collections::HashMap;
   use std::sync::{Arc, RwLock};
   use std::time::{Duration, Instant, SystemTime};
   use std::hash::{Hash, Hasher};
   use std::collections::hash_map::DefaultHasher;
   use anyhow::Result;
   
   #[derive(Debug, Clone)]
   pub struct CacheConfig {
       pub max_entries: usize,
       pub ttl_seconds: u64,
       pub max_memory_mb: usize,
       pub enable_result_caching: bool,
       pub enable_query_caching: bool,
       pub cache_hit_boost: f32,
   }
   
   impl Default for CacheConfig {
       fn default() -> Self {
           Self {
               max_entries: 1000,
               ttl_seconds: 3600, // 1 hour
               max_memory_mb: 100,
               enable_result_caching: true,
               enable_query_caching: true,
               cache_hit_boost: 1.0,
           }
       }
   }
   
   #[derive(Debug, Clone)]
   struct CacheEntry {
       result: SearchResult,
       created_at: Instant,
       access_count: usize,
       last_accessed: Instant,
       estimated_size: usize,
   }
   
   #[derive(Debug, Clone)]
   pub struct CacheStats {
       pub total_requests: usize,
       pub cache_hits: usize,
       pub cache_misses: usize,
       pub hit_rate: f32,
       pub total_entries: usize,
       pub memory_usage_mb: f32,
       pub avg_response_time_ms: f32,
   }
   
   pub struct SearchCache {
       cache: Arc<RwLock<HashMap<u64, CacheEntry>>>,
       config: CacheConfig,
       stats: Arc<RwLock<CacheStats>>,
   }
   
   impl SearchCache {
       /// Create a new search cache
       pub fn new(config: CacheConfig) -> Self {
           Self {
               cache: Arc::new(RwLock::new(HashMap::new())),
               config,
               stats: Arc::new(RwLock::new(CacheStats {
                   total_requests: 0,
                   cache_hits: 0,
                   cache_misses: 0,
                   hit_rate: 0.0,
                   total_entries: 0,
                   memory_usage_mb: 0.0,
                   avg_response_time_ms: 0.0,
               })),
           }
       }
       
       /// Get cached result if available
       pub fn get(&self, query: &ParsedQuery) -> Option<SearchResult> {
           if !self.config.enable_result_caching {
               return None;
           }
           
           let query_hash = self.hash_query(query);
           let mut cache = self.cache.write().unwrap();
           
           if let Some(entry) = cache.get_mut(&query_hash) {
               // Check if entry is still valid
               if entry.created_at.elapsed().as_secs() <= self.config.ttl_seconds {
                   entry.access_count += 1;
                   entry.last_accessed = Instant::now();
                   
                   // Update stats
                   self.update_stats(true);
                   
                   return Some(entry.result.clone());
               } else {
                   // Entry expired, remove it
                   cache.remove(&query_hash);
               }
           }
           
           self.update_stats(false);
           None
       }
       
       /// Store result in cache
       pub fn put(&self, query: &ParsedQuery, result: SearchResult) -> Result<()> {
           if !self.config.enable_result_caching {
               return Ok(());
           }
           
           let query_hash = self.hash_query(query);
           let estimated_size = self.estimate_result_size(&result);
           
           let entry = CacheEntry {
               result,
               created_at: Instant::now(),
               access_count: 1,
               last_accessed: Instant::now(),
               estimated_size,
           };
           
           let mut cache = self.cache.write().unwrap();
           
           // Check if we need to evict entries
           self.evict_if_needed(&mut cache, estimated_size);
           
           cache.insert(query_hash, entry);
           
           Ok(())
       }
       
       /// Hash a query for caching
       fn hash_query(&self, query: &ParsedQuery) -> u64 {
           let mut hasher = DefaultHasher::new();
           
           // Hash the query text
           query.original_text.hash(&mut hasher);
           
           // Hash filters
           query.filters.file_types.hash(&mut hasher);
           query.filters.languages.hash(&mut hasher);
           query.filters.semantic_types.hash(&mut hasher);
           query.filters.has_documentation.hash(&mut hasher);
           
           // Hash options
           query.options.fuzzy_distance.hash(&mut hasher);
           query.options.enable_fuzzy.hash(&mut hasher);
           query.options.max_results.hash(&mut hasher);
           
           hasher.finish()
       }
       
       /// Estimate memory size of a search result
       fn estimate_result_size(&self, result: &SearchResult) -> usize {
           let mut size = std::mem::size_of::<SearchResult>();
           
           // Estimate result content size
           for chunk_result in &result.results {
               size += chunk_result.content.len();
               size += chunk_result.metadata.file_path.len();
               size += std::mem::size_of_val(chunk_result);
           }
           
           // Estimate facets size
           size += result.facets.languages.len() * 50; // Rough estimate
           size += result.facets.semantic_types.len() * 50;
           size += result.facets.file_extensions.len() * 20;
           
           size
       }
       
       /// Evict entries if cache is full
       fn evict_if_needed(&self, cache: &mut HashMap<u64, CacheEntry>, new_entry_size: usize) {
           let current_size = self.calculate_current_memory_usage(cache);
           let max_size = self.config.max_memory_mb * 1024 * 1024;
           
           if cache.len() >= self.config.max_entries || 
              current_size + new_entry_size > max_size {
               
               // Use LRU eviction strategy
               let mut entries_by_access: Vec<_> = cache.iter().collect();
               entries_by_access.sort_by_key(|(_, entry)| entry.last_accessed);
               
               // Remove oldest entries until we have space
               let entries_to_remove = std::cmp::max(
                   1,
                   (cache.len() + 1).saturating_sub(self.config.max_entries)
               );
               
               for i in 0..entries_to_remove {
                   if i < entries_by_access.len() {
                       let key = *entries_by_access[i].0;
                       cache.remove(&key);
                   }
               }
           }
       }
       
       /// Calculate current memory usage
       fn calculate_current_memory_usage(&self, cache: &HashMap<u64, CacheEntry>) -> usize {
           cache.values()
               .map(|entry| entry.estimated_size)
               .sum()
       }
       
       /// Update cache statistics
       fn update_stats(&self, hit: bool) {
           let mut stats = self.stats.write().unwrap();
           stats.total_requests += 1;
           
           if hit {
               stats.cache_hits += 1;
           } else {
               stats.cache_misses += 1;
           }
           
           stats.hit_rate = stats.cache_hits as f32 / stats.total_requests as f32;
           
           let cache = self.cache.read().unwrap();
           stats.total_entries = cache.len();
           stats.memory_usage_mb = self.calculate_current_memory_usage(&cache) as f32 / (1024.0 * 1024.0);
       }
       
       /// Get cache statistics
       pub fn get_stats(&self) -> CacheStats {
           self.stats.read().unwrap().clone()
       }
       
       /// Clear the cache
       pub fn clear(&self) {
           let mut cache = self.cache.write().unwrap();
           cache.clear();
           
           let mut stats = self.stats.write().unwrap();
           *stats = CacheStats {
               total_requests: 0,
               cache_hits: 0,
               cache_misses: 0,
               hit_rate: 0.0,
               total_entries: 0,
               memory_usage_mb: 0.0,
               avg_response_time_ms: 0.0,
           };
       }
       
       /// Prune expired entries
       pub fn prune_expired(&self) {
           let mut cache = self.cache.write().unwrap();
           let ttl = Duration::from_secs(self.config.ttl_seconds);
           
           cache.retain(|_, entry| entry.created_at.elapsed() <= ttl);
           
           // Update stats
           let mut stats = self.stats.write().unwrap();
           stats.total_entries = cache.len();
           stats.memory_usage_mb = self.calculate_current_memory_usage(&cache) as f32 / (1024.0 * 1024.0);
       }
       
       /// Get cache efficiency metrics
       pub fn get_efficiency_metrics(&self) -> CacheEfficiencyMetrics {
           let cache = self.cache.read().unwrap();
           let stats = self.stats.read().unwrap();
           
           let mut access_counts: Vec<usize> = cache.values()
               .map(|entry| entry.access_count)
               .collect();
           access_counts.sort_unstable();
           
           let median_access_count = if access_counts.is_empty() {
               0
           } else {
               access_counts[access_counts.len() / 2]
           };
           
           let avg_access_count = if cache.is_empty() {
               0.0
           } else {
               cache.values()
                   .map(|entry| entry.access_count)
                   .sum::<usize>() as f32 / cache.len() as f32
           };
           
           CacheEfficiencyMetrics {
               hit_rate: stats.hit_rate,
               memory_efficiency: if self.config.max_memory_mb > 0 {
                   1.0 - (stats.memory_usage_mb / self.config.max_memory_mb as f32)
               } else {
                   1.0
               },
               avg_access_count,
               median_access_count,
               cache_utilization: cache.len() as f32 / self.config.max_entries as f32,
           }
       }
   }
   
   #[derive(Debug, Clone)]
   pub struct CacheEfficiencyMetrics {
       pub hit_rate: f32,
       pub memory_efficiency: f32,
       pub avg_access_count: f32,
       pub median_access_count: usize,
       pub cache_utilization: f32,
   }
   
   /// Performance optimization utilities
   pub mod optimization {
       use super::*;
       
       /// Optimize query for better cache hits
       pub fn normalize_query_for_caching(query: &mut ParsedQuery) {
           // Sort filters for consistent hashing
           query.filters.file_types.sort();
           query.filters.languages.sort();
           query.filters.semantic_types.sort();
           
           // Normalize query text (trim, lowercase certain parts)
           query.original_text = query.original_text.trim().to_string();
       }
       
       /// Preload frequently used queries
       pub fn preload_common_queries(cache: &SearchCache, common_queries: &[String]) -> Result<()> {
           // This would be implemented with actual search execution
           // For now, just a placeholder
           println!("Preloading {} common queries", common_queries.len());
           Ok(())
       }
       
       /// Analyze cache performance and suggest optimizations
       pub fn analyze_cache_performance(cache: &SearchCache) -> CachePerformanceAnalysis {
           let stats = cache.get_stats();
           let efficiency = cache.get_efficiency_metrics();
           
           let mut recommendations = Vec::new();
           
           if efficiency.hit_rate < 0.3 {
               recommendations.push("Consider increasing cache size or TTL".to_string());
           }
           
           if efficiency.memory_efficiency < 0.2 {
               recommendations.push("Cache is using too much memory, consider reducing max_entries".to_string());
           }
           
           if efficiency.cache_utilization > 0.9 {
               recommendations.push("Cache is nearly full, consider increasing max_entries".to_string());
           }
           
           CachePerformanceAnalysis {
               overall_score: (efficiency.hit_rate + efficiency.memory_efficiency + efficiency.cache_utilization) / 3.0,
               bottlenecks: recommendations,
               efficiency_metrics: efficiency,
           }
       }
   }
   
   #[derive(Debug, Clone)]
   pub struct CachePerformanceAnalysis {
       pub overall_score: f32,
       pub bottlenecks: Vec<String>,
       pub efficiency_metrics: CacheEfficiencyMetrics,
   }
   ```

2. **Add tests:**

   ```rust
   #[cfg(test)]
   mod caching_tests {
       use super::*;
       use crate::query::{ParsedQuery, QueryFilters, SearchOptions};
       use crate::search::{SearchResult, QueryInfo, SearchFacets};
       use std::collections::HashMap;
       
       fn create_test_query(text: &str) -> ParsedQuery {
           ParsedQuery {
               query: Box::new(tantivy::query::AllQuery),
               fields: Vec::new(),
               filters: QueryFilters::default(),
               options: SearchOptions::default(),
               original_text: text.to_string(),
           }
       }
       
       fn create_test_result() -> SearchResult {
           SearchResult {
               results: Vec::new(),
               total_hits: 1,
               search_time: Duration::from_millis(10),
               query_info: QueryInfo {
                   original_query: "test".to_string(),
                   processed_query: "test".to_string(),
                   filters_applied: 0,
                   fuzzy_used: false,
                   fields_searched: Vec::new(),
               },
               facets: SearchFacets {
                   languages: HashMap::new(),
                   semantic_types: HashMap::new(),
                   file_extensions: HashMap::new(),
               },
           }
       }
       
       #[test]
       fn test_cache_creation() {
           let config = CacheConfig::default();
           let _cache = SearchCache::new(config);
       }
       
       #[test]
       fn test_cache_put_and_get() -> Result<()> {
           let cache = SearchCache::new(CacheConfig::default());
           let query = create_test_query("test query");
           let result = create_test_result();
           
           // Should miss initially
           assert!(cache.get(&query).is_none());
           
           // Store result
           cache.put(&query, result.clone())?;
           
           // Should hit now
           let cached_result = cache.get(&query);
           assert!(cached_result.is_some());
           assert_eq!(cached_result.unwrap().total_hits, result.total_hits);
           
           Ok(())
       }
       
       #[test]
       fn test_cache_stats() -> Result<()> {
           let cache = SearchCache::new(CacheConfig::default());
           let query = create_test_query("test query");
           let result = create_test_result();
           
           // Initial stats
           let initial_stats = cache.get_stats();
           assert_eq!(initial_stats.total_requests, 0);
           assert_eq!(initial_stats.cache_hits, 0);
           
           // Miss
           cache.get(&query);
           let stats_after_miss = cache.get_stats();
           assert_eq!(stats_after_miss.total_requests, 1);
           assert_eq!(stats_after_miss.cache_misses, 1);
           
           // Store and hit
           cache.put(&query, result)?;
           cache.get(&query);
           let stats_after_hit = cache.get_stats();
           assert_eq!(stats_after_hit.cache_hits, 1);
           assert!(stats_after_hit.hit_rate > 0.0);
           
           Ok(())
       }
       
       #[test]
       fn test_cache_expiration() -> Result<()> {
           let mut config = CacheConfig::default();
           config.ttl_seconds = 1; // 1 second TTL
           let cache = SearchCache::new(config);
           
           let query = create_test_query("test query");
           let result = create_test_result();
           
           cache.put(&query, result)?;
           
           // Should hit immediately
           assert!(cache.get(&query).is_some());
           
           // Wait for expiration
           std::thread::sleep(Duration::from_secs(2));
           
           // Should miss after expiration
           assert!(cache.get(&query).is_none());
           
           Ok(())
       }
       
       #[test]
       fn test_cache_eviction() -> Result<()> {
           let mut config = CacheConfig::default();
           config.max_entries = 2; // Very small cache
           let cache = SearchCache::new(config);
           
           let query1 = create_test_query("query 1");
           let query2 = create_test_query("query 2");
           let query3 = create_test_query("query 3");
           let result = create_test_result();
           
           // Fill cache
           cache.put(&query1, result.clone())?;
           cache.put(&query2, result.clone())?;
           
           // Both should be cached
           assert!(cache.get(&query1).is_some());
           assert!(cache.get(&query2).is_some());
           
           // Add third entry (should evict oldest)
           cache.put(&query3, result)?;
           
           // First entry should be evicted
           assert!(cache.get(&query1).is_none());
           assert!(cache.get(&query3).is_some());
           
           Ok(())
       }
       
       #[test]
       fn test_query_normalization() {
           let mut query = create_test_query("  Test Query  ");
           query.filters.file_types = vec!["rs".to_string(), "py".to_string()];
           query.filters.languages = vec!["rust".to_string(), "python".to_string()];
           
           optimization::normalize_query_for_caching(&mut query);
           
           assert_eq!(query.original_text, "Test Query");
           // Filters should be sorted
           assert_eq!(query.filters.file_types, vec!["py".to_string(), "rs".to_string()]);
       }
       
       #[test]
       fn test_efficiency_metrics() -> Result<()> {
           let cache = SearchCache::new(CacheConfig::default());
           let query = create_test_query("test query");
           let result = create_test_result();
           
           // Add some cache activity
           cache.put(&query, result)?;
           cache.get(&query); // Hit
           cache.get(&query); // Another hit
           
           let metrics = cache.get_efficiency_metrics();
           
           assert!(metrics.hit_rate > 0.0);
           assert!(metrics.cache_utilization >= 0.0);
           assert!(metrics.avg_access_count > 0.0);
           
           Ok(())
       }
       
       #[test]
       fn test_cache_clear() -> Result<()> {
           let cache = SearchCache::new(CacheConfig::default());
           let query = create_test_query("test query");
           let result = create_test_result();
           
           cache.put(&query, result)?;
           assert!(cache.get(&query).is_some());
           
           cache.clear();
           assert!(cache.get(&query).is_none());
           
           let stats = cache.get_stats();
           assert_eq!(stats.total_entries, 0);
           
           Ok(())
       }
   }
   ```

## Success Criteria
- [ ] Search caching compiles without errors
- [ ] All caching tests pass with `cargo test caching_tests`
- [ ] Cache stores and retrieves results correctly
- [ ] Cache statistics track hits, misses, and performance
- [ ] Cache expiration removes old entries
- [ ] Cache eviction works with size limits
- [ ] Query normalization improves cache efficiency
- [ ] Efficiency metrics provide useful insights
- [ ] Cache clearing works properly

## Context for Next Task
Task 21 will move to integration testing, creating end-to-end test workflows that validate the complete system from file indexing through search and result presentation.