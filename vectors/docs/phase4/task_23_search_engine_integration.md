# Task 23: Implement Search Engine Integration

## Context
You are implementing Phase 4 of a vector indexing system. Cache statistics were implemented in the previous task. Now you need to integrate the cache with existing search engines, providing transparent caching for search operations with cache-aware search optimization.

## Current State
- `src/cache.rs` exists with complete cache implementation
- Comprehensive statistics and performance monitoring are available
- Search engines exist in `src/search.rs` (from previous phases)
- Need integration layer for transparent caching

## Task Objective
Implement search engine integration with transparent caching, cache-aware search optimization, and intelligent cache warming strategies.

## Implementation Requirements

### 1. Add cache-aware search engine wrapper
Add this integration module to create a new file `src/cached_search.rs`:
```rust
use crate::cache::{MemoryEfficientCache, CacheConfiguration};
use crate::search::{SearchEngine, SearchResult, BooleanQuery};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub struct CachedSearchEngine {
    cache: Arc<MemoryEfficientCache>,
    search_engine: Arc<SearchEngine>,
    config: CacheIntegrationConfig,
    query_transformer: QueryTransformer,
    cache_warmer: Arc<CacheWarmer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheIntegrationConfig {
    pub enable_caching: bool,
    pub cache_ttl_seconds: u64,
    pub enable_query_normalization: bool,
    pub enable_result_deduplication: bool,
    pub enable_cache_warming: bool,
    pub max_cached_result_size: usize,
    pub cache_key_strategy: CacheKeyStrategy,
    pub result_transformation: ResultTransformationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheKeyStrategy {
    Exact,           // Exact query match
    Normalized,      // Normalized query (case, whitespace)
    Semantic,        // Semantically equivalent queries
    Fingerprint,     // Hash-based fingerprint
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultTransformationConfig {
    pub enable_result_compression: bool,
    pub enable_result_filtering: bool,
    pub max_results_per_query: usize,
    pub min_score_threshold: f64,
}

impl CacheIntegrationConfig {
    pub fn default() -> Self {
        Self {
            enable_caching: true,
            cache_ttl_seconds: 3600, // 1 hour
            enable_query_normalization: true,
            enable_result_deduplication: true,
            enable_cache_warming: true,
            max_cached_result_size: 1024 * 1024, // 1MB
            cache_key_strategy: CacheKeyStrategy::Normalized,
            result_transformation: ResultTransformationConfig {
                enable_result_compression: false,
                enable_result_filtering: true,
                max_results_per_query: 100,
                min_score_threshold: 0.1,
            },
        }
    }
    
    pub fn performance_optimized() -> Self {
        Self {
            enable_caching: true,
            cache_ttl_seconds: 7200, // 2 hours
            enable_query_normalization: true,
            enable_result_deduplication: true,
            enable_cache_warming: true,
            max_cached_result_size: 5 * 1024 * 1024, // 5MB
            cache_key_strategy: CacheKeyStrategy::Semantic,
            result_transformation: ResultTransformationConfig {
                enable_result_compression: true,
                enable_result_filtering: true,
                max_results_per_query: 200,
                min_score_threshold: 0.05,
            },
        }
    }
}

impl CachedSearchEngine {
    pub fn new(
        search_engine: SearchEngine,
        cache_config: CacheConfiguration,
        integration_config: CacheIntegrationConfig,
    ) -> Self {
        let cache = Arc::new(MemoryEfficientCache::from_config(cache_config).unwrap());
        let search_engine = Arc::new(search_engine);
        let cache_warmer = Arc::new(CacheWarmer::new(
            Arc::clone(&cache),
            Arc::clone(&search_engine),
            integration_config.clone(),
        ));
        
        Self {
            cache,
            search_engine,
            config: integration_config.clone(),
            query_transformer: QueryTransformer::new(integration_config.cache_key_strategy.clone()),
            cache_warmer,
        }
    }
    
    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>, SearchError> {
        if !self.config.enable_caching {
            return self.search_engine.search(query).await;
        }
        
        let start_time = Instant::now();
        
        // Transform and normalize query for caching
        let cache_key = self.query_transformer.generate_cache_key(query);
        
        // Try cache first
        if let Some(cached_results) = self.cache.get(&cache_key) {
            // Apply any necessary transformations to cached results
            let results = self.transform_cached_results(cached_results);
            return Ok(results);
        }
        
        // Cache miss - perform actual search
        let search_results = self.search_engine.search(query).await?;
        
        // Transform results for caching if needed
        let cacheable_results = self.prepare_results_for_caching(&search_results);
        
        // Cache the results if they meet criteria
        if self.should_cache_results(&cacheable_results) {
            let cache_success = self.cache.put(cache_key.clone(), cacheable_results.clone());
            
            if !cache_success {
                // Log cache failure but don't fail the search
                eprintln!("Warning: Failed to cache search results for query: {}", query);
            }
        }
        
        // Record search performance
        let search_duration = start_time.elapsed();
        self.record_search_metrics(query, &search_results, search_duration, false);
        
        Ok(search_results)
    }
    
    pub async fn search_boolean(&self, query: &BooleanQuery) -> Result<Vec<SearchResult>, SearchError> {
        if !self.config.enable_caching {
            return self.search_engine.search_boolean(query).await;
        }
        
        let start_time = Instant::now();
        
        // Generate cache key for boolean query
        let cache_key = self.query_transformer.generate_boolean_cache_key(query);
        
        // Try cache first
        if let Some(cached_results) = self.cache.get(&cache_key) {
            let results = self.transform_cached_results(cached_results);
            self.record_search_metrics(&format!("{:?}", query), &results, start_time.elapsed(), true);
            return Ok(results);
        }
        
        // Cache miss - perform actual search
        let search_results = self.search_engine.search_boolean(query).await?;
        
        // Cache results
        let cacheable_results = self.prepare_results_for_caching(&search_results);
        if self.should_cache_results(&cacheable_results) {
            self.cache.put(cache_key, cacheable_results);
        }
        
        let search_duration = start_time.elapsed();
        self.record_search_metrics(&format!("{:?}", query), &search_results, search_duration, false);
        
        Ok(search_results)
    }
    
    fn transform_cached_results(&self, cached_results: Vec<SearchResult>) -> Vec<SearchResult> {
        if !self.config.result_transformation.enable_result_filtering {
            return cached_results;
        }
        
        cached_results
            .into_iter()
            .filter(|result| result.score >= self.config.result_transformation.min_score_threshold)
            .take(self.config.result_transformation.max_results_per_query)
            .collect()
    }
    
    fn prepare_results_for_caching(&self, results: &[SearchResult]) -> Vec<SearchResult> {
        let mut cacheable_results = results.to_vec();
        
        // Apply result transformations
        if self.config.result_transformation.enable_result_filtering {
            cacheable_results = cacheable_results
                .into_iter()
                .filter(|result| result.score >= self.config.result_transformation.min_score_threshold)
                .take(self.config.result_transformation.max_results_per_query)
                .collect();
        }
        
        // Apply deduplication if enabled
        if self.config.enable_result_deduplication {
            cacheable_results = self.deduplicate_results(cacheable_results);
        }
        
        cacheable_results
    }
    
    fn deduplicate_results(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        let mut seen_paths = std::collections::HashSet::new();
        results
            .into_iter()
            .filter(|result| seen_paths.insert(result.file_path.clone()))
            .collect()
    }
    
    fn should_cache_results(&self, results: &[SearchResult]) -> bool {
        if results.is_empty() {
            return false;
        }
        
        // Check result size limit
        let estimated_size: usize = results
            .iter()
            .map(|r| r.file_path.len() + r.content.len() + std::mem::size_of::<SearchResult>())
            .sum();
        
        estimated_size <= self.config.max_cached_result_size
    }
    
    fn record_search_metrics(&self, query: &str, results: &[SearchResult], duration: Duration, was_cached: bool) {
        // This would integrate with the statistics collector
        // For now, we'll just log basic metrics
        println!(
            "Search: query='{}', results={}, duration={:.2}ms, cached={}",
            if query.len() > 50 { &query[..50] } else { query },
            results.len(),
            duration.as_secs_f64() * 1000.0,
            was_cached
        );
    }
    
    pub fn warm_cache(&self, queries: Vec<String>) -> CacheWarmingResult {
        self.cache_warmer.warm_cache(queries)
    }
    
    pub fn get_cache_stats(&self) -> CacheIntegrationStats {
        let cache_stats = self.cache.get_stats();
        let concurrent_stats = self.cache.get_concurrent_stats();
        
        CacheIntegrationStats {
            cache_hit_rate: cache_stats.hit_rate,
            total_searches: cache_stats.total_hits + cache_stats.total_misses,
            cache_hits: cache_stats.total_hits,
            cache_misses: cache_stats.total_misses,
            current_cache_size_mb: cache_stats.memory_usage_mb,
            cache_entries: cache_stats.entries,
            avg_search_duration_cached_ms: 0.0,   // Would need separate tracking
            avg_search_duration_uncached_ms: 0.0, // Would need separate tracking
            cache_efficiency_score: cache_stats.hit_rate * (cache_stats.entries as f64 / 1000.0),
            concurrent_access_stats: concurrent_stats,
        }
    }
    
    pub fn clear_cache(&self) {
        self.cache.clear();
    }
    
    pub fn invalidate_query(&self, query: &str) {
        let cache_key = self.query_transformer.generate_cache_key(query);
        self.cache.remove(&cache_key);
    }
}

#[derive(Debug, Clone)]
pub struct QueryTransformer {
    strategy: CacheKeyStrategy,
}

impl QueryTransformer {
    pub fn new(strategy: CacheKeyStrategy) -> Self {
        Self { strategy }
    }
    
    pub fn generate_cache_key(&self, query: &str) -> String {
        match self.strategy {
            CacheKeyStrategy::Exact => query.to_string(),
            CacheKeyStrategy::Normalized => self.normalize_query(query),
            CacheKeyStrategy::Semantic => self.generate_semantic_key(query),
            CacheKeyStrategy::Fingerprint => self.generate_fingerprint(query),
        }
    }
    
    pub fn generate_boolean_cache_key(&self, query: &BooleanQuery) -> String {
        // Convert boolean query to string representation for caching
        let query_str = format!("{:?}", query);
        self.generate_cache_key(&query_str)
    }
    
    fn normalize_query(&self, query: &str) -> String {
        query
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    fn generate_semantic_key(&self, query: &str) -> String {
        // Simplified semantic key generation
        // In a real implementation, this might use NLP techniques
        let normalized = self.normalize_query(query);
        
        // Extract and sort key terms
        let mut terms: Vec<&str> = normalized
            .split_whitespace()
            .filter(|term| term.len() > 2) // Remove short words
            .collect();
        terms.sort();
        
        format!("semantic:{}", terms.join("+"))
    }
    
    fn generate_fingerprint(&self, query: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let normalized = self.normalize_query(query);
        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        
        format!("fp:{:x}", hasher.finish())
    }
}

#[derive(Debug)]
pub struct CacheWarmer {
    cache: Arc<MemoryEfficientCache>,
    search_engine: Arc<SearchEngine>,
    config: CacheIntegrationConfig,
}

impl CacheWarmer {
    pub fn new(
        cache: Arc<MemoryEfficientCache>,
        search_engine: Arc<SearchEngine>,
        config: CacheIntegrationConfig,
    ) -> Self {
        Self {
            cache,
            search_engine,
            config,
        }
    }
    
    pub fn warm_cache(&self, queries: Vec<String>) -> CacheWarmingResult {
        let start_time = Instant::now();
        let mut successful_warmups = 0;
        let mut failed_warmups = 0;
        let mut total_results_cached = 0;
        
        let query_transformer = QueryTransformer::new(self.config.cache_key_strategy.clone());
        
        for query in &queries {
            match self.warm_single_query(query, &query_transformer) {
                Ok(result_count) => {
                    successful_warmups += 1;
                    total_results_cached += result_count;
                }
                Err(_) => {
                    failed_warmups += 1;
                }
            }
        }
        
        CacheWarmingResult {
            total_queries: queries.len(),
            successful_warmups,
            failed_warmups,
            total_results_cached,
            duration: start_time.elapsed(),
        }
    }
    
    fn warm_single_query(&self, query: &str, transformer: &QueryTransformer) -> Result<usize, SearchError> {
        let cache_key = transformer.generate_cache_key(query);
        
        // Skip if already cached
        if self.cache.get(&cache_key).is_some() {
            return Ok(0);
        }
        
        // Perform search (this would need to be made async in real implementation)
        let results = self.search_engine.search(query)?;
        
        // Cache results
        let cacheable_results = self.prepare_results_for_caching(&results);
        let result_count = cacheable_results.len();
        
        if self.should_cache_results(&cacheable_results) {
            self.cache.put(cache_key, cacheable_results);
            Ok(result_count)
        } else {
            Err(SearchError::CacheError("Results too large to cache".to_string()))
        }
    }
    
    fn prepare_results_for_caching(&self, results: &[SearchResult]) -> Vec<SearchResult> {
        let mut cacheable_results = results.to_vec();
        
        if self.config.result_transformation.enable_result_filtering {
            cacheable_results = cacheable_results
                .into_iter()
                .filter(|result| result.score >= self.config.result_transformation.min_score_threshold)
                .take(self.config.result_transformation.max_results_per_query)
                .collect();
        }
        
        cacheable_results
    }
    
    fn should_cache_results(&self, results: &[SearchResult]) -> bool {
        if results.is_empty() {
            return false;
        }
        
        let estimated_size: usize = results
            .iter()
            .map(|r| r.file_path.len() + r.content.len() + std::mem::size_of::<SearchResult>())
            .sum();
        
        estimated_size <= self.config.max_cached_result_size
    }
}

#[derive(Debug, Clone)]
pub struct CacheIntegrationStats {
    pub cache_hit_rate: f64,
    pub total_searches: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub current_cache_size_mb: f64,
    pub cache_entries: usize,
    pub avg_search_duration_cached_ms: f64,
    pub avg_search_duration_uncached_ms: f64,
    pub cache_efficiency_score: f64,
    pub concurrent_access_stats: crate::cache::ConcurrentStatsSnapshot,
}

impl CacheIntegrationStats {
    pub fn format_integration_report(&self) -> String {
        format!(
            "Cache Integration Performance Report:\n\
             \nSearch Performance:\n\
             Total Searches: {} ({} hits, {} misses)\n\
             Cache Hit Rate: {:.1}%\n\
             Avg Duration (Cached): {:.2}ms\n\
             Avg Duration (Uncached): {:.2}ms\n\
             Performance Improvement: {:.1}x\n\
             \nCache Status:\n\
             Current Size: {:.2}MB ({} entries)\n\
             Cache Efficiency Score: {:.3}\n\
             \nConcurrent Access:\n\
             {}",
            self.total_searches, self.cache_hits, self.cache_misses,
            self.cache_hit_rate * 100.0,
            self.avg_search_duration_cached_ms,
            self.avg_search_duration_uncached_ms,
            if self.avg_search_duration_cached_ms > 0.0 {
                self.avg_search_duration_uncached_ms / self.avg_search_duration_cached_ms
            } else {
                1.0
            },
            self.current_cache_size_mb, self.cache_entries,
            self.cache_efficiency_score,
            self.concurrent_access_stats.format_report()
        )
    }
}

#[derive(Debug, Clone)]
pub struct CacheWarmingResult {
    pub total_queries: usize,
    pub successful_warmups: usize,
    pub failed_warmups: usize,
    pub total_results_cached: usize,
    pub duration: Duration,
}

impl CacheWarmingResult {
    pub fn format_warming_report(&self) -> String {
        let success_rate = if self.total_queries > 0 {
            (self.successful_warmups as f64 / self.total_queries as f64) * 100.0
        } else {
            0.0
        };
        
        format!(
            "Cache Warming Results:\n\
             Total Queries: {}\n\
             Successful: {} ({:.1}%)\n\
             Failed: {}\n\
             Results Cached: {}\n\
             Duration: {:.2}s\n\
             Avg Time per Query: {:.2}ms",
            self.total_queries,
            self.successful_warmups, success_rate,
            self.failed_warmups,
            self.total_results_cached,
            self.duration.as_secs_f64(),
            if self.total_queries > 0 {
                (self.duration.as_secs_f64() * 1000.0) / self.total_queries as f64
            } else {
                0.0
            }
        )
    }
}

#[derive(Debug)]
pub enum SearchError {
    SearchEngineError(String),
    CacheError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::SearchEngineError(msg) => write!(f, "Search engine error: {}", msg),
            SearchError::CacheError(msg) => write!(f, "Cache error: {}", msg),
            SearchError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for SearchError {}

// For the purpose of this task, we'll define a minimal SearchEngine trait
// In reality, this would import from the existing search module
pub trait SearchEngine {
    fn search(&self, query: &str) -> Result<Vec<SearchResult>, SearchError>;
    fn search_boolean(&self, query: &BooleanQuery) -> Result<Vec<SearchResult>, SearchError>;
}

// Temporary implementations for compilation
impl SearchEngine for crate::search::SearchEngine {
    fn search(&self, query: &str) -> Result<Vec<SearchResult>, SearchError> {
        // This would integrate with the actual search engine implementation
        Ok(vec![])
    }
    
    fn search_boolean(&self, query: &BooleanQuery) -> Result<Vec<SearchResult>, SearchError> {
        // This would integrate with the actual boolean search implementation
        Ok(vec![])
    }
}
```

### 2. Add cache integration tests
Add these integration tests to `src/cached_search.rs`:
```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::cache::CacheConfiguration;
    use std::time::Duration;
    
    struct MockSearchEngine {
        results: Vec<SearchResult>,
        call_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }
    
    impl MockSearchEngine {
        fn new(results: Vec<SearchResult>) -> Self {
            Self {
                results,
                call_count: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            }
        }
        
        fn get_call_count(&self) -> usize {
            self.call_count.load(std::sync::atomic::Ordering::Relaxed)
        }
    }
    
    impl SearchEngine for MockSearchEngine {
        fn search(&self, _query: &str) -> Result<Vec<SearchResult>, SearchError> {
            self.call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(10)); // Simulate search time
            Ok(self.results.clone())
        }
        
        fn search_boolean(&self, _query: &BooleanQuery) -> Result<Vec<SearchResult>, SearchError> {
            self.call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(self.results.clone())
        }
    }
    
    fn create_test_results() -> Vec<SearchResult> {
        vec![
            SearchResult {
                file_path: "test1.rs".to_string(),
                content: "pub fn test1() {}".to_string(),
                chunk_index: 0,
                score: 1.0,
            },
            SearchResult {
                file_path: "test2.rs".to_string(),
                content: "pub fn test2() {}".to_string(),
                chunk_index: 0,
                score: 0.8,
            },
        ]
    }
    
    #[tokio::test]
    async fn test_cache_hit_functionality() {
        let test_results = create_test_results();
        let mock_engine = MockSearchEngine::new(test_results.clone());
        
        let cache_config = CacheConfiguration::minimal();
        let integration_config = CacheIntegrationConfig::default();
        
        let cached_engine = CachedSearchEngine::new(
            mock_engine,
            cache_config,
            integration_config,
        );
        
        // First search should miss cache and call search engine
        let results1 = cached_engine.search("test query").await.unwrap();
        assert_eq!(results1.len(), 2);
        assert_eq!(cached_engine.search_engine.get_call_count(), 1);
        
        // Second search should hit cache and not call search engine
        let results2 = cached_engine.search("test query").await.unwrap();
        assert_eq!(results2.len(), 2);
        assert_eq!(cached_engine.search_engine.get_call_count(), 1); // Still 1
        
        // Verify cache stats
        let stats = cached_engine.get_cache_stats();
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.cache_hit_rate, 0.5);
    }
    
    #[tokio::test]
    async fn test_query_normalization() {
        let test_results = create_test_results();
        let mock_engine = MockSearchEngine::new(test_results);
        
        let cache_config = CacheConfiguration::minimal();
        let mut integration_config = CacheIntegrationConfig::default();
        integration_config.cache_key_strategy = CacheKeyStrategy::Normalized;
        
        let cached_engine = CachedSearchEngine::new(
            mock_engine,
            cache_config,
            integration_config,
        );
        
        // These queries should be considered equivalent after normalization
        let _results1 = cached_engine.search("Test Query").await.unwrap();
        let _results2 = cached_engine.search("test query").await.unwrap();
        let _results3 = cached_engine.search("  TEST   QUERY  ").await.unwrap();
        
        // Should only call search engine once due to normalization
        assert_eq!(cached_engine.search_engine.get_call_count(), 1);
        
        let stats = cached_engine.get_cache_stats();
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.cache_misses, 1);
    }
    
    #[tokio::test]
    async fn test_result_filtering() {
        let test_results = vec![
            SearchResult {
                file_path: "high_score.rs".to_string(),
                content: "high score content".to_string(),
                chunk_index: 0,
                score: 0.9,
            },
            SearchResult {
                file_path: "low_score.rs".to_string(),
                content: "low score content".to_string(),
                chunk_index: 0,
                score: 0.05, // Below threshold
            },
        ];
        
        let mock_engine = MockSearchEngine::new(test_results);
        
        let cache_config = CacheConfiguration::minimal();
        let mut integration_config = CacheIntegrationConfig::default();
        integration_config.result_transformation.min_score_threshold = 0.1;
        
        let cached_engine = CachedSearchEngine::new(
            mock_engine,
            cache_config,
            integration_config,
        );
        
        let results = cached_engine.search("test query").await.unwrap();
        
        // Should filter out low score result
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, "high_score.rs");
        assert!(results[0].score >= 0.1);
    }
    
    #[tokio::test]
    async fn test_cache_warming() {
        let test_results = create_test_results();
        let mock_engine = MockSearchEngine::new(test_results);
        
        let cache_config = CacheConfiguration::minimal();
        let integration_config = CacheIntegrationConfig::default();
        
        let cached_engine = CachedSearchEngine::new(
            mock_engine,
            cache_config,
            integration_config,
        );
        
        // Warm cache with multiple queries
        let warming_queries = vec![
            "query1".to_string(),
            "query2".to_string(),
            "query3".to_string(),
        ];
        
        let warming_result = cached_engine.warm_cache(warming_queries);
        
        assert_eq!(warming_result.total_queries, 3);
        assert_eq!(warming_result.successful_warmups, 3);
        assert_eq!(warming_result.failed_warmups, 0);
        assert!(warming_result.total_results_cached > 0);
        
        // Verify search engine was called for warming
        assert_eq!(cached_engine.search_engine.get_call_count(), 3);
        
        // Now searches should hit cache
        let _results = cached_engine.search("query1").await.unwrap();
        assert_eq!(cached_engine.search_engine.get_call_count(), 3); // Still 3
        
        println!("{}", warming_result.format_warming_report());
    }
    
    #[tokio::test]
    async fn test_cache_size_limits() {
        let large_content = "x".repeat(100_000); // 100KB content
        let large_results = vec![
            SearchResult {
                file_path: "large1.rs".to_string(),
                content: large_content.clone(),
                chunk_index: 0,
                score: 1.0,
            },
            SearchResult {
                file_path: "large2.rs".to_string(),
                content: large_content,
                chunk_index: 0,
                score: 0.9,
            },
        ];
        
        let mock_engine = MockSearchEngine::new(large_results);
        
        let mut cache_config = CacheConfiguration::minimal();
        cache_config.max_memory_mb = 1; // Very small cache
        
        let mut integration_config = CacheIntegrationConfig::default();
        integration_config.max_cached_result_size = 50_000; // 50KB limit
        
        let cached_engine = CachedSearchEngine::new(
            mock_engine,
            cache_config,
            integration_config,
        );
        
        // First search - results too large to cache
        let _results1 = cached_engine.search("large query").await.unwrap();
        
        // Second search - should call search engine again (not cached)
        let _results2 = cached_engine.search("large query").await.unwrap();
        
        assert_eq!(cached_engine.search_engine.get_call_count(), 2);
        
        let stats = cached_engine.get_cache_stats();
        assert_eq!(stats.cache_hits, 0); // No hits due to size limits
    }
    
    #[tokio::test]
    async fn test_cache_invalidation() {
        let test_results = create_test_results();
        let mock_engine = MockSearchEngine::new(test_results);
        
        let cache_config = CacheConfiguration::minimal();
        let integration_config = CacheIntegrationConfig::default();
        
        let cached_engine = CachedSearchEngine::new(
            mock_engine,
            cache_config,
            integration_config,
        );
        
        // Cache a query
        let _results1 = cached_engine.search("test query").await.unwrap();
        assert_eq!(cached_engine.search_engine.get_call_count(), 1);
        
        // Verify cache hit
        let _results2 = cached_engine.search("test query").await.unwrap();
        assert_eq!(cached_engine.search_engine.get_call_count(), 1);
        
        // Invalidate the query
        cached_engine.invalidate_query("test query");
        
        // Should call search engine again after invalidation
        let _results3 = cached_engine.search("test query").await.unwrap();
        assert_eq!(cached_engine.search_engine.get_call_count(), 2);
    }
    
    #[tokio::test]
    async fn test_different_cache_key_strategies() {
        let test_results = create_test_results();
        
        let strategies = vec![
            CacheKeyStrategy::Exact,
            CacheKeyStrategy::Normalized,
            CacheKeyStrategy::Semantic,
            CacheKeyStrategy::Fingerprint,
        ];
        
        for strategy in strategies {
            let mock_engine = MockSearchEngine::new(test_results.clone());
            let cache_config = CacheConfiguration::minimal();
            let mut integration_config = CacheIntegrationConfig::default();
            integration_config.cache_key_strategy = strategy.clone();
            
            let cached_engine = CachedSearchEngine::new(
                mock_engine,
                cache_config,
                integration_config,
            );
            
            // Test with similar queries
            let _results1 = cached_engine.search("test query").await.unwrap();
            let _results2 = cached_engine.search("TEST QUERY").await.unwrap();
            
            let call_count = cached_engine.search_engine.get_call_count();
            
            match strategy {
                CacheKeyStrategy::Exact => {
                    assert_eq!(call_count, 2); // Different cases = different keys
                }
                CacheKeyStrategy::Normalized => {
                    assert_eq!(call_count, 1); // Normalized to same key
                }
                CacheKeyStrategy::Semantic | CacheKeyStrategy::Fingerprint => {
                    // These might be the same or different depending on implementation
                    assert!(call_count <= 2);
                }
            }
            
            println!("Strategy {:?}: {} calls", strategy, call_count);
        }
    }
    
    #[tokio::test]
    async fn test_integration_statistics() {
        let test_results = create_test_results();
        let mock_engine = MockSearchEngine::new(test_results);
        
        let cache_config = CacheConfiguration::balanced();
        let integration_config = CacheIntegrationConfig::default();
        
        let cached_engine = CachedSearchEngine::new(
            mock_engine,
            cache_config,
            integration_config,
        );
        
        // Perform various searches
        let _results1 = cached_engine.search("query1").await.unwrap(); // Miss
        let _results2 = cached_engine.search("query1").await.unwrap(); // Hit
        let _results3 = cached_engine.search("query2").await.unwrap(); // Miss
        let _results4 = cached_engine.search("query1").await.unwrap(); // Hit
        let _results5 = cached_engine.search("query2").await.unwrap(); // Hit
        
        let stats = cached_engine.get_cache_stats();
        
        assert_eq!(stats.total_searches, 5);
        assert_eq!(stats.cache_hits, 3);
        assert_eq!(stats.cache_misses, 2);
        assert_eq!(stats.cache_hit_rate, 0.6);
        
        let report = stats.format_integration_report();
        println!("{}", report);
        
        assert!(report.contains("Cache Hit Rate: 60.0%"));
        assert!(report.contains("Total Searches: 5"));
    }
}
```

### 3. Update lib.rs to include cached search module
Add this line to `src/lib.rs`:
```rust
pub mod cached_search;
```

## Success Criteria
- [ ] Cache-aware search engine wrapper implemented
- [ ] Transparent caching for search operations works correctly
- [ ] Multiple cache key strategies (exact, normalized, semantic, fingerprint) implemented
- [ ] Result filtering and transformation capabilities work
- [ ] Cache warming functionality enables proactive caching
- [ ] Cache invalidation allows selective cache clearing
- [ ] Integration statistics provide performance insights
- [ ] All integration tests pass validation
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Integration layer provides transparent caching for existing search engines
- Multiple cache key strategies optimize for different use cases
- Result transformation enables efficient caching of large result sets
- Cache warming improves performance for predictable query patterns
- Statistics integration enables monitoring of cache effectiveness
- Asynchronous operations maintain non-blocking search performance