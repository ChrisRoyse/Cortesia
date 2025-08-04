# Task 45: Search Engine Performance Monitoring Integration

## Context
You are implementing Phase 4 of a vector indexing system. Parallel indexer integration with performance monitoring is now complete. This task integrates performance monitoring with the search engine to provide comprehensive query performance tracking, search optimization, and user experience monitoring.

## Current State
- `src/monitor.rs` exists with complete performance monitoring functionality
- Parallel indexer integration provides comprehensive indexing performance tracking
- Need integration with search engine for query performance monitoring

## Task Objective
Integrate the performance monitoring system with the search engine to provide detailed query performance tracking, search result quality monitoring, caching effectiveness analysis, and search optimization recommendations.

## Implementation Requirements

### 1. Create monitored search engine wrapper
Create a new file `src/search_monitor.rs`:
```rust
use crate::monitor::{SharedPerformanceMonitor, PerformanceMonitor, RealTimeMonitor};
use crate::search::{SearchEngine, SearchQuery, SearchResult, SearchError}; // Assuming search module exists
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};
use std::collections::HashMap;

pub struct MonitoredSearchEngine {
    pub engine: SearchEngine,
    pub monitor: SharedPerformanceMonitor,
    pub query_monitor: Arc<Mutex<QueryPerformanceMonitor>>,
    pub config: SearchMonitoringConfig,
}

#[derive(Debug, Clone)]
pub struct SearchMonitoringConfig {
    pub enable_query_analysis: bool,
    pub enable_result_quality_tracking: bool,
    pub enable_cache_monitoring: bool,
    pub enable_latency_percentiles: bool,
    pub track_query_patterns: bool,
    pub slow_query_threshold: Duration,
    pub result_relevance_tracking: bool,
    pub user_satisfaction_monitoring: bool,
}

impl Default for SearchMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_query_analysis: true,
            enable_result_quality_tracking: true,
            enable_cache_monitoring: true,
            enable_latency_percentiles: true,
            track_query_patterns: true,
            slow_query_threshold: Duration::from_millis(100),
            result_relevance_tracking: true,
            user_satisfaction_monitoring: true,
        }
    }
}

pub struct QueryPerformanceMonitor {
    query_types: HashMap<QueryType, PerformanceMonitor>,
    cache_performance: CachePerformanceTracker,
    query_patterns: QueryPatternAnalyzer,
    result_quality: ResultQualityTracker,
    slow_queries: Vec<SlowQueryRecord>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum QueryType {
    ExactMatch,
    FuzzySearch,
    RangeQuery,
    VectorSimilarity,
    CompoundQuery,
    FilteredSearch,
}

#[derive(Debug, Clone)]
pub struct SearchPerformanceResult {
    pub query_result: SearchResult,
    pub execution_time: Duration,
    pub cache_hit: bool,
    pub result_count: usize,
    pub quality_score: f64,
    pub query_complexity: QueryComplexity,
}

#[derive(Debug, Clone)]
pub struct QueryComplexity {
    pub vector_dimensions: usize,
    pub filter_count: usize,
    pub has_fuzzy_matching: bool,
    pub has_range_conditions: bool,
    pub estimated_index_scans: usize,
}

impl MonitoredSearchEngine {
    pub fn new(engine: SearchEngine, config: SearchMonitoringConfig) -> Self {
        Self {
            engine,
            monitor: SharedPerformanceMonitor::new(),
            query_monitor: Arc::new(Mutex::new(QueryPerformanceMonitor::new())),
            config,
        }
    }
    
    /// Execute search query with comprehensive monitoring
    pub fn search_monitored(&self, query: SearchQuery) -> Result<SearchPerformanceResult, SearchError> {
        let start_time = Instant::now();
        let query_type = self.classify_query(&query);
        let complexity = self.analyze_query_complexity(&query);
        
        // Check cache first if enabled
        let (cache_hit, cache_result) = if self.config.enable_cache_monitoring {
            self.check_cache(&query)
        } else {
            (false, None)
        };
        
        let result = if let Some(cached_result) = cache_result {
            cached_result
        } else {
            // Execute actual search
            let search_result = self.monitor.time_query(|| {
                self.engine.search(query.clone())
            })?;
            
            // Store in cache if enabled
            if self.config.enable_cache_monitoring {
                self.update_cache(&query, &search_result);
            }
            
            search_result
        };
        
        let execution_time = start_time.elapsed();
        let result_count = result.matches.len();
        
        // Calculate result quality score
        let quality_score = if self.config.enable_result_quality_tracking {
            self.calculate_result_quality(&query, &result)
        } else {
            0.0
        };
        
        // Record performance metrics
        self.record_query_performance(query_type, execution_time, cache_hit, &complexity);
        
        // Check for slow queries
        if execution_time > self.config.slow_query_threshold {
            self.record_slow_query(&query, execution_time, &complexity);
        }
        
        // Analyze query patterns if enabled
        if self.config.track_query_patterns {
            self.analyze_query_pattern(&query, execution_time);
        }
        
        Ok(SearchPerformanceResult {
            query_result: result,
            execution_time,
            cache_hit,
            result_count,
            quality_score,
            query_complexity: complexity,
        })
    }
    
    fn classify_query(&self, query: &SearchQuery) -> QueryType {
        // Analyze query structure to determine type
        if query.has_exact_match() {
            QueryType::ExactMatch
        } else if query.has_fuzzy_search() {
            QueryType::FuzzySearch
        } else if query.has_range_conditions() {
            QueryType::RangeQuery
        } else if query.has_vector_similarity() {
            QueryType::VectorSimilarity
        } else if query.has_multiple_conditions() {
            QueryType::CompoundQuery
        } else if query.has_filters() {
            QueryType::FilteredSearch
        } else {
            QueryType::ExactMatch // Default
        }
    }
    
    fn analyze_query_complexity(&self, query: &SearchQuery) -> QueryComplexity {
        QueryComplexity {
            vector_dimensions: query.get_vector_dimensions(),
            filter_count: query.get_filter_count(),
            has_fuzzy_matching: query.has_fuzzy_search(),
            has_range_conditions: query.has_range_conditions(),
            estimated_index_scans: self.estimate_index_scans(query),
        }
    }
    
    fn estimate_index_scans(&self, query: &SearchQuery) -> usize {
        // Estimate how many index scans will be required
        let mut scan_count = 1; // Base scan
        
        if query.has_filters() {
            scan_count += query.get_filter_count();
        }
        
        if query.has_range_conditions() {
            scan_count += 1; // Range queries typically require additional scans
        }
        
        if query.has_fuzzy_search() {
            scan_count *= 2; // Fuzzy searches are more expensive
        }
        
        scan_count
    }
    
    fn check_cache(&self, query: &SearchQuery) -> (bool, Option<SearchResult>) {
        // Simplified cache implementation - in practice would use actual cache
        let query_hash = self.calculate_query_hash(query);
        
        if let Ok(mut query_monitor) = self.query_monitor.lock() {
            if let Some(cached_result) = query_monitor.cache_performance.get_cached_result(&query_hash) {
                query_monitor.cache_performance.record_cache_hit();
                return (true, Some(cached_result));
            }
        }
        
        if let Ok(mut query_monitor) = self.query_monitor.lock() {
            query_monitor.cache_performance.record_cache_miss();
        }
        
        (false, None)
    }
    
    fn update_cache(&self, query: &SearchQuery, result: &SearchResult) {
        let query_hash = self.calculate_query_hash(query);
        
        if let Ok(mut query_monitor) = self.query_monitor.lock() {
            query_monitor.cache_performance.store_result(query_hash, result.clone());
        }
    }
    
    fn calculate_query_hash(&self, query: &SearchQuery) -> u64 {
        // Simplified hash calculation - in practice would use proper hashing
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query.hash(&mut hasher);
        hasher.finish()
    }
    
    fn calculate_result_quality(&self, query: &SearchQuery, result: &SearchResult) -> f64 {
        let mut quality_score = 100.0;
        
        // Penalize if no results found
        if result.matches.is_empty() {
            return 0.0;
        }
        
        // Check result relevance (simplified scoring)
        let avg_relevance = result.matches.iter()
            .map(|m| m.relevance_score)
            .sum::<f64>() / result.matches.len() as f64;
        
        quality_score *= avg_relevance;
        
        // Bonus for good result count (not too few, not too many)
        let result_count_factor = match result.matches.len() {
            0 => 0.0,
            1..=10 => 1.0,
            11..=50 => 0.9,
            51..=100 => 0.8,
            _ => 0.7, // Too many results might indicate poor query specificity
        };
        
        quality_score *= result_count_factor;
        
        // Consider query complexity vs result quality trade-off
        if query.has_fuzzy_search() && avg_relevance > 0.8 {
            quality_score *= 1.1; // Bonus for good fuzzy search results
        }
        
        quality_score.min(100.0)
    }
    
    fn record_query_performance(&self, query_type: QueryType, execution_time: Duration, cache_hit: bool, complexity: &QueryComplexity) {
        if let Ok(mut query_monitor) = self.query_monitor.lock() {
            // Record in type-specific monitor
            query_monitor.query_types
                .entry(query_type)
                .or_insert_with(PerformanceMonitor::new)
                .record_query_time(execution_time);
            
            // Update cache performance
            if cache_hit {
                query_monitor.cache_performance.record_cache_hit();
            } else {
                query_monitor.cache_performance.record_cache_miss();
            }
        }
        
        // Record in global monitor
        self.monitor.record_query_time(execution_time);
    }
    
    fn record_slow_query(&self, query: &SearchQuery, execution_time: Duration, complexity: &QueryComplexity) {
        let slow_query = SlowQueryRecord {
            timestamp: SystemTime::now(),
            query_hash: self.calculate_query_hash(query),
            execution_time,
            complexity: complexity.clone(),
            query_text: query.to_debug_string(),
        };
        
        if let Ok(mut query_monitor) = self.query_monitor.lock() {
            query_monitor.slow_queries.push(slow_query);
            
            // Keep only recent slow queries (last 1000)
            if query_monitor.slow_queries.len() > 1000 {
                query_monitor.slow_queries.remove(0);
            }
        }
    }
    
    fn analyze_query_pattern(&self, query: &SearchQuery, execution_time: Duration) {
        if let Ok(mut query_monitor) = self.query_monitor.lock() {
            query_monitor.query_patterns.analyze_pattern(query, execution_time);
        }
    }
    
    /// Get comprehensive search performance statistics
    pub fn get_search_statistics(&self) -> SearchPerformanceStatistics {
        let global_stats = self.monitor.get_stats();
        
        let type_specific_stats = if let Ok(query_monitor) = self.query_monitor.lock() {
            query_monitor.query_types.iter()
                .map(|(query_type, monitor)| (query_type.clone(), monitor.get_stats()))
                .collect()
        } else {
            HashMap::new()
        };
        
        let cache_stats = if let Ok(query_monitor) = self.query_monitor.lock() {
            query_monitor.cache_performance.get_statistics()
        } else {
            CacheStatistics::default()
        };
        
        let slow_query_count = if let Ok(query_monitor) = self.query_monitor.lock() {
            query_monitor.slow_queries.len()
        } else {
            0
        };
        
        SearchPerformanceStatistics {
            global_stats,
            type_specific_stats,
            cache_stats,
            slow_query_count,
            monitoring_enabled: true,
        }
    }
    
    /// Generate search performance report
    pub fn generate_search_report(&self) -> SearchPerformanceReport {
        let stats = self.get_search_statistics();
        let recommendations = self.generate_search_recommendations(&stats);
        
        SearchPerformanceReport {
            timestamp: SystemTime::now(),
            statistics: stats,
            recommendations,
            slow_queries: self.get_recent_slow_queries(10),
        }
    }
    
    fn generate_search_recommendations(&self, stats: &SearchPerformanceStatistics) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Cache performance recommendations
        if stats.cache_stats.hit_rate < 0.3 {
            recommendations.push("Low cache hit rate. Consider adjusting cache size or TTL settings.".to_string());
        }
        
        // Query performance recommendations
        if let Some(global_stats) = &stats.global_stats {
            if global_stats.p99_query_time > Duration::from_millis(500) {
                recommendations.push("High P99 query latency detected. Consider query optimization or index tuning.".to_string());
            }
            
            if global_stats.avg_query_time > Duration::from_millis(100) {
                recommendations.push("Average query time is high. Consider optimizing frequent query patterns.".to_string());
            }
        }
        
        // Type-specific recommendations
        for (query_type, type_stats) in &stats.type_specific_stats {
            if type_stats.avg_query_time > Duration::from_millis(200) {
                recommendations.push(format!("{:?} queries are slow. Consider specific optimizations for this query type.", query_type));
            }
        }
        
        // Slow query recommendations
        if stats.slow_query_count > 10 {
            recommendations.push(format!("{} slow queries detected. Review and optimize problematic query patterns.", stats.slow_query_count));
        }
        
        if recommendations.is_empty() {
            recommendations.push("Search performance looks good. No specific optimizations needed.".to_string());
        }
        
        recommendations
    }
    
    fn get_recent_slow_queries(&self, limit: usize) -> Vec<SlowQueryRecord> {
        if let Ok(query_monitor) = self.query_monitor.lock() {
            query_monitor.slow_queries
                .iter()
                .rev()
                .take(limit)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
}

// Supporting structures and implementations
impl QueryPerformanceMonitor {
    fn new() -> Self {
        Self {
            query_types: HashMap::new(),
            cache_performance: CachePerformanceTracker::new(),
            query_patterns: QueryPatternAnalyzer::new(),
            result_quality: ResultQualityTracker::new(),
            slow_queries: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CachePerformanceTracker {
    cache_hits: usize,
    cache_misses: usize,
    cached_results: HashMap<u64, (SearchResult, SystemTime)>,
    max_cache_size: usize,
}

impl CachePerformanceTracker {
    fn new() -> Self {
        Self {
            cache_hits: 0,
            cache_misses: 0,
            cached_results: HashMap::new(),
            max_cache_size: 1000,
        }
    }
    
    fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }
    
    fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }
    
    fn get_cached_result(&self, query_hash: &u64) -> Option<SearchResult> {
        self.cached_results.get(query_hash).map(|(result, _)| result.clone())
    }
    
    fn store_result(&mut self, query_hash: u64, result: SearchResult) {
        if self.cached_results.len() >= self.max_cache_size {
            // Remove oldest entry (simplified LRU)
            if let Some(oldest_key) = self.cached_results.keys().next().copied() {
                self.cached_results.remove(&oldest_key);
            }
        }
        
        self.cached_results.insert(query_hash, (result, SystemTime::now()));
    }
    
    fn get_statistics(&self) -> CacheStatistics {
        let total_requests = self.cache_hits + self.cache_misses;
        let hit_rate = if total_requests > 0 {
            self.cache_hits as f64 / total_requests as f64
        } else {
            0.0
        };
        
        CacheStatistics {
            hit_rate,
            total_hits: self.cache_hits,
            total_misses: self.cache_misses,
            cache_size: self.cached_results.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QueryPatternAnalyzer {
    patterns: HashMap<String, QueryPatternStats>,
}

impl QueryPatternAnalyzer {
    fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }
    
    fn analyze_pattern(&mut self, query: &SearchQuery, execution_time: Duration) {
        let pattern = self.extract_pattern(query);
        
        let stats = self.patterns.entry(pattern).or_insert_with(QueryPatternStats::new);
        stats.record_execution(execution_time);
    }
    
    fn extract_pattern(&self, query: &SearchQuery) -> String {
        // Extract query pattern (simplified)
        let mut pattern = String::new();
        
        if query.has_exact_match() {
            pattern.push_str("exact:");
        }
        if query.has_fuzzy_search() {
            pattern.push_str("fuzzy:");
        }
        if query.has_range_conditions() {
            pattern.push_str("range:");
        }
        if query.has_vector_similarity() {
            pattern.push_str("vector:");
        }
        if query.has_filters() {
            pattern.push_str(&format!("filters({}):", query.get_filter_count()));
        }
        
        if pattern.is_empty() {
            pattern = "simple".to_string();
        }
        
        pattern
    }
}

#[derive(Debug, Clone)]
pub struct QueryPatternStats {
    execution_count: usize,
    total_time: Duration,
    min_time: Duration,
    max_time: Duration,
}

impl QueryPatternStats {
    fn new() -> Self {
        Self {
            execution_count: 0,
            total_time: Duration::from_millis(0),
            min_time: Duration::from_secs(u64::MAX),
            max_time: Duration::from_millis(0),
        }
    }
    
    fn record_execution(&mut self, execution_time: Duration) {
        self.execution_count += 1;
        self.total_time += execution_time;
        self.min_time = self.min_time.min(execution_time);
        self.max_time = self.max_time.max(execution_time);
    }
}

#[derive(Debug, Clone)]
pub struct ResultQualityTracker {
    quality_scores: Vec<f64>,
    relevance_scores: Vec<f64>,
}

impl ResultQualityTracker {
    fn new() -> Self {
        Self {
            quality_scores: Vec::new(),
            relevance_scores: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SlowQueryRecord {
    pub timestamp: SystemTime,
    pub query_hash: u64,
    pub execution_time: Duration,
    pub complexity: QueryComplexity,
    pub query_text: String,
}

#[derive(Debug, Clone)]
pub struct SearchPerformanceStatistics {
    pub global_stats: Option<crate::monitor::PerformanceStats>,
    pub type_specific_stats: HashMap<QueryType, crate::monitor::PerformanceStats>,
    pub cache_stats: CacheStatistics,
    pub slow_query_count: usize,
    pub monitoring_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub hit_rate: f64,
    pub total_hits: usize,
    pub total_misses: usize,
    pub cache_size: usize,
}

impl Default for CacheStatistics {
    fn default() -> Self {
        Self {
            hit_rate: 0.0,
            total_hits: 0,
            total_misses: 0,
            cache_size: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchPerformanceReport {
    pub timestamp: SystemTime,
    pub statistics: SearchPerformanceStatistics,
    pub recommendations: Vec<String>,
    pub slow_queries: Vec<SlowQueryRecord>,
}
```

### 2. Add search engine mock for testing
Create `src/search.rs` (mock implementation for testing):
```rust
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct SearchEngine {
    index: HashMap<String, Vec<SearchMatch>>,
}

#[derive(Debug, Clone, Hash)]
pub struct SearchQuery {
    pub text: Option<String>,
    pub vector: Option<Vec<f32>>,
    pub filters: HashMap<String, String>,
    pub fuzzy: bool,
    pub range_conditions: Vec<RangeCondition>,
    pub max_results: usize,
}

#[derive(Debug, Clone, Hash)]
pub struct RangeCondition {
    pub field: String,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub matches: Vec<SearchMatch>,
    pub total_matches: usize,
    pub execution_time_ms: u64,
}

#[derive(Debug, Clone)]
pub struct SearchMatch {
    pub id: String,
    pub content: String,
    pub relevance_score: f64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug)]
pub enum SearchError {
    InvalidQuery,
    IndexNotFound,
    ExecutionTimeout,
    InternalError(String),
}

impl SearchEngine {
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }
    
    pub fn search(&self, query: SearchQuery) -> Result<SearchResult, SearchError> {
        // Simulate search processing time
        std::thread::sleep(std::time::Duration::from_millis(
            10 + (query.get_estimated_complexity() * 5) as u64
        ));
        
        let mut matches = Vec::new();
        
        // Simulate search results based on query
        if let Some(text) = &query.text {
            matches.extend(self.search_by_text(text, query.fuzzy));
        }
        
        if let Some(vector) = &query.vector {
            matches.extend(self.search_by_vector(vector));
        }
        
        // Apply filters
        if !query.filters.is_empty() {
            matches = self.apply_filters(matches, &query.filters);
        }
        
        // Apply range conditions
        if !query.range_conditions.is_empty() {
            matches = self.apply_range_conditions(matches, &query.range_conditions);
        }
        
        // Limit results
        matches.truncate(query.max_results);
        
        Ok(SearchResult {
            total_matches: matches.len(),
            matches,
            execution_time_ms: 0, // Will be set by monitoring
        })
    }
    
    fn search_by_text(&self, text: &str, fuzzy: bool) -> Vec<SearchMatch> {
        // Simulate text search results
        let result_count = if fuzzy { 
            (text.len() / 2).max(1).min(20) 
        } else { 
            text.len().max(1).min(10) 
        };
        
        (0..result_count).map(|i| {
            SearchMatch {
                id: format!("text_match_{}", i),
                content: format!("Content matching '{}' ({})", text, i),
                relevance_score: 1.0 - (i as f64 * 0.1),
                metadata: HashMap::new(),
            }
        }).collect()
    }
    
    fn search_by_vector(&self, vector: &[f32]) -> Vec<SearchMatch> {
        // Simulate vector similarity search
        let result_count = (vector.len() / 10).max(1).min(15);
        
        (0..result_count).map(|i| {
            SearchMatch {
                id: format!("vector_match_{}", i),
                content: format!("Vector similarity match {}", i),
                relevance_score: 0.9 - (i as f64 * 0.05),
                metadata: HashMap::new(),
            }
        }).collect()
    }
    
    fn apply_filters(&self, matches: Vec<SearchMatch>, filters: &HashMap<String, String>) -> Vec<SearchMatch> {
        // Simulate filter application - remove some results
        let filter_selectivity = 0.7; // 70% of results pass filters
        let keep_count = (matches.len() as f64 * filter_selectivity) as usize;
        matches.into_iter().take(keep_count).collect()
    }
    
    fn apply_range_conditions(&self, matches: Vec<SearchMatch>, _conditions: &[RangeCondition]) -> Vec<SearchMatch> {
        // Simulate range condition application
        let range_selectivity = 0.8; // 80% of results pass range conditions
        let keep_count = (matches.len() as f64 * range_selectivity) as usize;
        matches.into_iter().take(keep_count).collect()
    }
}

impl SearchQuery {
    pub fn new() -> Self {
        Self {
            text: None,
            vector: None,
            filters: HashMap::new(),
            fuzzy: false,
            range_conditions: Vec::new(),
            max_results: 50,
        }
    }
    
    pub fn with_text(mut self, text: String) -> Self {
        self.text = Some(text);
        self
    }
    
    pub fn with_vector(mut self, vector: Vec<f32>) -> Self {
        self.vector = Some(vector);
        self
    }
    
    pub fn with_fuzzy(mut self, fuzzy: bool) -> Self {
        self.fuzzy = fuzzy;
        self
    }
    
    pub fn has_exact_match(&self) -> bool {
        self.text.is_some() && !self.fuzzy
    }
    
    pub fn has_fuzzy_search(&self) -> bool {
        self.fuzzy && self.text.is_some()
    }
    
    pub fn has_range_conditions(&self) -> bool {
        !self.range_conditions.is_empty()
    }
    
    pub fn has_vector_similarity(&self) -> bool {
        self.vector.is_some()
    }
    
    pub fn has_multiple_conditions(&self) -> bool {
        let condition_count = 
            if self.text.is_some() { 1 } else { 0 } +
            if self.vector.is_some() { 1 } else { 0 } +
            if !self.filters.is_empty() { 1 } else { 0 } +
            if !self.range_conditions.is_empty() { 1 } else { 0 };
        condition_count > 1
    }
    
    pub fn has_filters(&self) -> bool {
        !self.filters.is_empty()
    }
    
    pub fn get_vector_dimensions(&self) -> usize {
        self.vector.as_ref().map(|v| v.len()).unwrap_or(0)
    }
    
    pub fn get_filter_count(&self) -> usize {
        self.filters.len()
    }
    
    pub fn get_estimated_complexity(&self) -> usize {
        let mut complexity = 1;
        
        if self.has_fuzzy_search() {
            complexity += 2;
        }
        if self.has_vector_similarity() {
            complexity += 3;
        }
        if self.has_filters() {
            complexity += self.get_filter_count();
        }
        if self.has_range_conditions() {
            complexity += self.range_conditions.len();
        }
        
        complexity
    }
    
    pub fn to_debug_string(&self) -> String {
        format!("SearchQuery {{ text: {:?}, vector_dims: {}, filters: {}, fuzzy: {}, ranges: {} }}", 
            self.text, 
            self.get_vector_dimensions(),
            self.get_filter_count(),
            self.fuzzy,
            self.range_conditions.len()
        )
    }
}

impl std::fmt::Display for SearchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchError::InvalidQuery => write!(f, "Invalid search query"),
            SearchError::IndexNotFound => write!(f, "Search index not found"),
            SearchError::ExecutionTimeout => write!(f, "Search execution timeout"),
            SearchError::InternalError(msg) => write!(f, "Internal search error: {}", msg),
        }
    }
}

impl std::error::Error for SearchError {}
```

### 3. Add comprehensive search monitoring tests
Add to `src/search_monitor.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{SearchEngine, SearchQuery};
    
    #[test]
    fn test_monitored_search_basic_functionality() {
        let engine = SearchEngine::new();
        let config = SearchMonitoringConfig::default();
        let monitored_engine = MonitoredSearchEngine::new(engine, config);
        
        let query = SearchQuery::new()
            .with_text("test query".to_string());
        
        let result = monitored_engine.search_monitored(query).unwrap();
        
        assert!(!result.query_result.matches.is_empty());
        assert!(result.execution_time > Duration::from_millis(0));
        assert_eq!(result.cache_hit, false); // First query should not be cached
        assert!(result.quality_score >= 0.0);
    }
    
    #[test]
    fn test_query_type_classification() {
        let engine = SearchEngine::new();
        let config = SearchMonitoringConfig::default();
        let monitored_engine = MonitoredSearchEngine::new(engine, config);
        
        // Test exact match
        let exact_query = SearchQuery::new().with_text("exact".to_string());
        assert_eq!(monitored_engine.classify_query(&exact_query), QueryType::ExactMatch);
        
        // Test fuzzy search
        let fuzzy_query = SearchQuery::new()
            .with_text("fuzzy".to_string())
            .with_fuzzy(true);
        assert_eq!(monitored_engine.classify_query(&fuzzy_query), QueryType::FuzzySearch);
        
        // Test vector similarity
        let vector_query = SearchQuery::new()
            .with_vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(monitored_engine.classify_query(&vector_query), QueryType::VectorSimilarity);
    }
    
    #[test]
    fn test_query_complexity_analysis() {
        let engine = SearchEngine::new();
        let config = SearchMonitoringConfig::default();
        let monitored_engine = MonitoredSearchEngine::new(engine, config);
        
        let mut query = SearchQuery::new()
            .with_text("complex query".to_string())
            .with_vector(vec![1.0; 128])
            .with_fuzzy(true);
        
        query.filters.insert("category".to_string(), "test".to_string());
        query.range_conditions.push(crate::search::RangeCondition {
            field: "score".to_string(),
            min: 0.5,
            max: 1.0,
        });
        
        let complexity = monitored_engine.analyze_query_complexity(&query);
        
        assert_eq!(complexity.vector_dimensions, 128);
        assert_eq!(complexity.filter_count, 1);
        assert!(complexity.has_fuzzy_matching);
        assert!(complexity.has_range_conditions);
        assert!(complexity.estimated_index_scans > 1);
    }
    
    #[test]
    fn test_cache_monitoring() {
        let engine = SearchEngine::new();
        let config = SearchMonitoringConfig {
            enable_cache_monitoring: true,
            ..Default::default()
        };
        let monitored_engine = MonitoredSearchEngine::new(engine, config);
        
        let query = SearchQuery::new()
            .with_text("cacheable query".to_string());
        
        // First execution - should not be cached
        let result1 = monitored_engine.search_monitored(query.clone()).unwrap();
        assert_eq!(result1.cache_hit, false);
        
        // Second execution - should be cached
        let result2 = monitored_engine.search_monitored(query).unwrap();
        assert_eq!(result2.cache_hit, true);
        
        let stats = monitored_engine.get_search_statistics();
        assert!(stats.cache_stats.total_hits > 0);
        assert!(stats.cache_stats.total_misses > 0);
        assert!(stats.cache_stats.hit_rate > 0.0);
    }
    
    #[test]
    fn test_slow_query_detection() {
        let engine = SearchEngine::new();
        let config = SearchMonitoringConfig {
            slow_query_threshold: Duration::from_millis(1), // Very low threshold for testing
            ..Default::default()
        };
        let monitored_engine = MonitoredSearchEngine::new(engine, config);
        
        // Create a complex query that should be slow
        let mut slow_query = SearchQuery::new()
            .with_text("slow complex query".to_string())
            .with_vector(vec![1.0; 256])
            .with_fuzzy(true);
        
        slow_query.filters.insert("category".to_string(), "test".to_string());
        
        let _result = monitored_engine.search_monitored(slow_query).unwrap();
        
        let slow_queries = monitored_engine.get_recent_slow_queries(10);
        assert!(!slow_queries.is_empty());
        
        let stats = monitored_engine.get_search_statistics();
        assert!(stats.slow_query_count > 0);
    }
    
    #[test]
    fn test_performance_statistics() {
        let engine = SearchEngine::new();
        let config = SearchMonitoringConfig::default();
        let monitored_engine = MonitoredSearchEngine::new(engine, config);
        
        // Execute various types of queries
        let queries = vec![
            SearchQuery::new().with_text("query1".to_string()),
            SearchQuery::new().with_text("query2".to_string()).with_fuzzy(true),
            SearchQuery::new().with_vector(vec![1.0; 64]),
        ];
        
        for query in queries {
            let _result = monitored_engine.search_monitored(query).unwrap();
        }
        
        let stats = monitored_engine.get_search_statistics();
        
        assert!(stats.monitoring_enabled);
        assert!(stats.global_stats.is_some());
        assert!(!stats.type_specific_stats.is_empty());
        
        if let Some(global_stats) = stats.global_stats {
            assert_eq!(global_stats.total_queries, 3);
            assert!(global_stats.avg_query_time > Duration::from_millis(0));
        }
        
        // Should have stats for different query types
        assert!(stats.type_specific_stats.len() >= 2);
    }
    
    #[test]
    fn test_search_report_generation() {
        let engine = SearchEngine::new();
        let config = SearchMonitoringConfig::default();
        let monitored_engine = MonitoredSearchEngine::new(engine, config);
        
        // Execute some queries to generate data
        for i in 0..10 {
            let query = SearchQuery::new()
                .with_text(format!("test query {}", i));
            let _result = monitored_engine.search_monitored(query).unwrap();
        }
        
        let report = monitored_engine.generate_search_report();
        
        assert!(!report.recommendations.is_empty());
        assert!(report.statistics.monitoring_enabled);
        
        if let Some(global_stats) = report.statistics.global_stats {
            assert_eq!(global_stats.total_queries, 10);
        }
    }
    
    #[test]
    fn test_result_quality_calculation() {
        let engine = SearchEngine::new();
        let config = SearchMonitoringConfig {
            enable_result_quality_tracking: true,
            ..Default::default()
        };
        let monitored_engine = MonitoredSearchEngine::new(engine, config);
        
        let query = SearchQuery::new()
            .with_text("quality test".to_string());
        
        let result = monitored_engine.search_monitored(query).unwrap();
        
        // Quality score should be calculated
        assert!(result.quality_score >= 0.0);
        assert!(result.quality_score <= 100.0);
        
        // With actual results, quality should be > 0
        if !result.query_result.matches.is_empty() {
            assert!(result.quality_score > 0.0);
        }
    }
    
    #[test]
    fn test_query_pattern_analysis() {
        let engine = SearchEngine::new();
        let config = SearchMonitoringConfig {
            track_query_patterns: true,
            ..Default::default()
        };
        let monitored_engine = MonitoredSearchEngine::new(engine, config);
        
        // Execute similar queries to establish patterns
        for _ in 0..5 {
            let query = SearchQuery::new()
                .with_text("pattern test".to_string())
                .with_fuzzy(true);
            let _result = monitored_engine.search_monitored(query).unwrap();
        }
        
        // Pattern analysis should be working (tested indirectly through successful execution)
        let stats = monitored_engine.get_search_statistics();
        assert!(stats.monitoring_enabled);
    }
}
```

## Success Criteria
- [ ] Search engine integrates seamlessly with performance monitoring
- [ ] Query type classification works correctly for different search patterns
- [ ] Cache monitoring tracks hit rates and performance accurately
- [ ] Slow query detection identifies performance bottlenecks
- [ ] Query complexity analysis provides meaningful insights
- [ ] Result quality tracking evaluates search effectiveness
- [ ] Performance statistics include comprehensive search metrics
- [ ] Search reports provide actionable optimization recommendations
- [ ] Query pattern analysis identifies common usage patterns
- [ ] All search monitoring tests pass consistently

## Time Limit
10 minutes

## Notes
- Provides comprehensive search performance monitoring integration
- Supports multiple query types with specific performance tracking
- Includes cache performance monitoring for optimization insights
- Enables slow query detection and analysis for performance tuning
- Supports result quality tracking for search effectiveness measurement
- Provides detailed performance reports with optimization recommendations
- Includes query pattern analysis for usage optimization
- Maintains monitoring accuracy across different search scenarios