# Task 19: Create Cache Tests Setup

## Context
You are implementing Phase 4 of a vector indexing system. Cache configuration was implemented in the previous task. Now you need to create a comprehensive test suite for all cache functionality with proper test utilities, fixtures, and integration tests.

## Current State
- `src/cache.rs` exists with full cache implementation
- Configuration, eviction policies, and memory management are complete
- Individual unit tests exist but need comprehensive test suite
- Need integration tests and performance testing framework

## Task Objective
Create a comprehensive cache testing framework with test utilities, fixtures, integration tests, and performance benchmarks.

## Implementation Requirements

### 1. Create test utilities module
Add this test utilities section to the cache module:
```rust
#[cfg(test)]
pub mod test_utils {
    use super::*;
    use std::collections::HashMap;
    
    pub struct CacheTestFixture {
        pub cache: MemoryEfficientCache,
        pub test_results: Vec<Vec<SearchResult>>,
        pub test_queries: Vec<String>,
    }
    
    impl CacheTestFixture {
        pub fn new_with_config(config: CacheConfiguration) -> Self {
            let cache = MemoryEfficientCache::from_config(config).unwrap();
            Self {
                cache,
                test_results: Self::generate_test_results(),
                test_queries: Self::generate_test_queries(),
            }
        }
        
        pub fn new_minimal() -> Self {
            Self::new_with_config(CacheConfiguration::minimal())
        }
        
        pub fn new_balanced() -> Self {
            Self::new_with_config(CacheConfiguration::balanced())
        }
        
        pub fn new_performance() -> Self {
            Self::new_with_config(CacheConfiguration::performance_optimized())
        }
        
        fn generate_test_results() -> Vec<Vec<SearchResult>> {
            vec![
                // Small result set
                vec![
                    SearchResult {
                        file_path: "small.rs".to_string(),
                        content: "fn small() {}".to_string(),
                        chunk_index: 0,
                        score: 1.0,
                    }
                ],
                // Medium result set
                vec![
                    SearchResult {
                        file_path: "medium1.rs".to_string(),
                        content: "pub fn medium_function() { println!(\"Medium\"); }".to_string(),
                        chunk_index: 0,
                        score: 0.9,
                    },
                    SearchResult {
                        file_path: "medium2.rs".to_string(),
                        content: "impl MediumStruct { fn method(&self) {} }".to_string(),
                        chunk_index: 1,
                        score: 0.8,
                    }
                ],
                // Large result set
                vec![
                    SearchResult {
                        file_path: "large1.rs".to_string(),
                        content: "x".repeat(1000),
                        chunk_index: 0,
                        score: 1.0,
                    },
                    SearchResult {
                        file_path: "large2.rs".to_string(),
                        content: "y".repeat(800),
                        chunk_index: 1,
                        score: 0.95,
                    },
                    SearchResult {
                        file_path: "large3.rs".to_string(),
                        content: "z".repeat(1200),
                        chunk_index: 2,
                        score: 0.85,
                    }
                ],
                // Variable size results
                Self::generate_variable_size_results(10),
            ]
        }
        
        fn generate_variable_size_results(count: usize) -> Vec<SearchResult> {
            (0..count).map(|i| {
                let size = (i + 1) * 100; // Varying sizes
                SearchResult {
                    file_path: format!("var_{}.rs", i),
                    content: "x".repeat(size),
                    chunk_index: i,
                    score: 1.0 - (i as f64 * 0.05),
                }
            }).collect()
        }
        
        fn generate_test_queries() -> Vec<String> {
            vec![
                "simple query".to_string(),
                "function definition".to_string(),
                "struct implementation".to_string(),
                "error handling".to_string(),
                "async fn".to_string(),
                "test case".to_string(),
                "database query".to_string(),
                "api endpoint".to_string(),
                "configuration setup".to_string(),
                "performance optimization".to_string(),
            ]
        }
        
        pub fn populate_cache(&mut self, entry_count: usize) {
            let entry_count = entry_count.min(self.test_queries.len());
            
            for i in 0..entry_count {
                let query = &self.test_queries[i];
                let results = &self.test_results[i % self.test_results.len()];
                self.cache.put(query.clone(), results.clone());
            }
        }
        
        pub fn create_memory_pressure(&mut self) {
            // Fill cache to near memory limit
            let large_content = "x".repeat(50000); // 50KB per result
            let mut counter = 0;
            
            loop {
                let large_results = vec![
                    SearchResult {
                        file_path: format!("pressure_{}.rs", counter),
                        content: large_content.clone(),
                        chunk_index: 0,
                        score: 1.0,
                    }
                ];
                
                if !self.cache.put(format!("pressure_query_{}", counter), large_results) {
                    break; // Memory limit reached
                }
                
                counter += 1;
                if counter > 1000 { // Safety limit
                    break;
                }
            }
        }
        
        pub fn assert_cache_consistency(&self) {
            assert!(self.cache.validate_cache().is_ok());
            assert!(self.cache.validate_memory_consistency().is_ok());
        }
        
        pub fn get_cache_stats(&self) -> CacheStats {
            self.cache.get_stats()
        }
    }
    
    pub struct PerformanceTestHarness {
        pub cache: MemoryEfficientCache,
        pub queries: Vec<String>,
        pub results: Vec<Vec<SearchResult>>,
    }
    
    impl PerformanceTestHarness {
        pub fn new(cache_config: CacheConfiguration, test_data_size: usize) -> Self {
            let cache = MemoryEfficientCache::from_config(cache_config).unwrap();
            let (queries, results) = Self::generate_performance_test_data(test_data_size);
            
            Self {
                cache,
                queries,
                results,
            }
        }
        
        fn generate_performance_test_data(size: usize) -> (Vec<String>, Vec<Vec<SearchResult>>) {
            let mut queries = Vec::new();
            let mut results = Vec::new();
            
            for i in 0..size {
                queries.push(format!("performance_query_{}", i));
                
                let result_count = (i % 10) + 1; // 1-10 results per query
                let mut query_results = Vec::new();
                
                for j in 0..result_count {
                    let content_size = 100 + (i * 10) + (j * 50); // Variable content sizes
                    query_results.push(SearchResult {
                        file_path: format!("perf_file_{}_{}.rs", i, j),
                        content: "x".repeat(content_size),
                        chunk_index: j,
                        score: 1.0 - (j as f64 * 0.1),
                    });
                }
                
                results.push(query_results);
            }
            
            (queries, results)
        }
        
        pub fn benchmark_put_operations(&mut self, iterations: usize) -> PutBenchmarkResult {
            let start = std::time::Instant::now();
            let mut successful_puts = 0;
            let mut failed_puts = 0;
            
            for i in 0..iterations {
                let query_idx = i % self.queries.len();
                let result_idx = i % self.results.len();
                
                if self.cache.put(
                    format!("{}_{}", self.queries[query_idx], i),
                    self.results[result_idx].clone()
                ) {
                    successful_puts += 1;
                } else {
                    failed_puts += 1;
                }
            }
            
            let duration = start.elapsed();
            
            PutBenchmarkResult {
                iterations,
                successful_puts,
                failed_puts,
                total_duration: duration,
                avg_duration_per_put: duration / iterations as u32,
                puts_per_second: successful_puts as f64 / duration.as_secs_f64(),
            }
        }
        
        pub fn benchmark_get_operations(&self, iterations: usize) -> GetBenchmarkResult {
            let start = std::time::Instant::now();
            let mut hits = 0;
            let mut misses = 0;
            
            for i in 0..iterations {
                let query_idx = i % self.queries.len();
                let query = &self.queries[query_idx];
                
                if self.cache.get(query).is_some() {
                    hits += 1;
                } else {
                    misses += 1;
                }
            }
            
            let duration = start.elapsed();
            
            GetBenchmarkResult {
                iterations,
                hits,
                misses,
                total_duration: duration,
                avg_duration_per_get: duration / iterations as u32,
                gets_per_second: iterations as f64 / duration.as_secs_f64(),
                hit_rate: hits as f64 / iterations as f64,
            }
        }
        
        pub fn benchmark_mixed_operations(&mut self, iterations: usize, put_ratio: f64) -> MixedBenchmarkResult {
            let start = std::time::Instant::now();
            let mut puts = 0;
            let mut gets = 0;
            let mut put_successes = 0;
            let mut get_hits = 0;
            
            for i in 0..iterations {
                let should_put = (i as f64 / iterations as f64) < put_ratio;
                
                if should_put {
                    puts += 1;
                    let query_idx = i % self.queries.len();
                    let result_idx = i % self.results.len();
                    
                    if self.cache.put(
                        format!("mixed_{}_{}", self.queries[query_idx], i),
                        self.results[result_idx].clone()
                    ) {
                        put_successes += 1;
                    }
                } else {
                    gets += 1;
                    let query_idx = i % self.queries.len();
                    
                    if self.cache.get(&self.queries[query_idx]).is_some() {
                        get_hits += 1;
                    }
                }
            }
            
            let duration = start.elapsed();
            
            MixedBenchmarkResult {
                iterations,
                puts,
                gets,
                put_successes,
                get_hits,
                total_duration: duration,
                operations_per_second: iterations as f64 / duration.as_secs_f64(),
            }
        }
    }
    
    #[derive(Debug)]
    pub struct PutBenchmarkResult {
        pub iterations: usize,
        pub successful_puts: usize,
        pub failed_puts: usize,
        pub total_duration: std::time::Duration,
        pub avg_duration_per_put: std::time::Duration,
        pub puts_per_second: f64,
    }
    
    #[derive(Debug)]
    pub struct GetBenchmarkResult {
        pub iterations: usize,
        pub hits: usize,
        pub misses: usize,
        pub total_duration: std::time::Duration,
        pub avg_duration_per_get: std::time::Duration,
        pub gets_per_second: f64,
        pub hit_rate: f64,
    }
    
    #[derive(Debug)]
    pub struct MixedBenchmarkResult {
        pub iterations: usize,
        pub puts: usize,
        pub gets: usize,
        pub put_successes: usize,
        pub get_hits: usize,
        pub total_duration: std::time::Duration,
        pub operations_per_second: f64,
    }
    
    impl PutBenchmarkResult {
        pub fn format_report(&self) -> String {
            format!(
                "Put Benchmark Results:\n  Iterations: {}\n  Successful: {} ({:.1}%)\n  Failed: {} ({:.1}%)\n  Duration: {:.2}s\n  Avg per put: {:.2}ms\n  Puts/sec: {:.0}",
                self.iterations,
                self.successful_puts,
                (self.successful_puts as f64 / self.iterations as f64) * 100.0,
                self.failed_puts,
                (self.failed_puts as f64 / self.iterations as f64) * 100.0,
                self.total_duration.as_secs_f64(),
                self.avg_duration_per_put.as_secs_f64() * 1000.0,
                self.puts_per_second
            )
        }
    }
    
    impl GetBenchmarkResult {
        pub fn format_report(&self) -> String {
            format!(
                "Get Benchmark Results:\n  Iterations: {}\n  Hits: {} ({:.1}%)\n  Misses: {} ({:.1}%)\n  Duration: {:.2}s\n  Avg per get: {:.2}ms\n  Gets/sec: {:.0}",
                self.iterations,
                self.hits,
                self.hit_rate * 100.0,
                self.misses,
                (1.0 - self.hit_rate) * 100.0,
                self.total_duration.as_secs_f64(),
                self.avg_duration_per_get.as_secs_f64() * 1000.0,
                self.gets_per_second
            )
        }
    }
    
    impl MixedBenchmarkResult {
        pub fn format_report(&self) -> String {
            format!(
                "Mixed Operations Benchmark Results:\n  Total operations: {}\n  Puts: {} (success rate: {:.1}%)\n  Gets: {} (hit rate: {:.1}%)\n  Duration: {:.2}s\n  Operations/sec: {:.0}",
                self.iterations,
                self.puts,
                if self.puts > 0 { (self.put_successes as f64 / self.puts as f64) * 100.0 } else { 0.0 },
                self.gets,
                if self.gets > 0 { (self.get_hits as f64 / self.gets as f64) * 100.0 } else { 0.0 },
                self.total_duration.as_secs_f64(),
                self.operations_per_second
            )
        }
    }
}
```

### 2. Add comprehensive integration tests
Add these integration tests to the test module:
```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use super::test_utils::*;
    
    #[test]
    fn test_cache_full_lifecycle() {
        let mut fixture = CacheTestFixture::new_balanced();
        
        // Start with empty cache
        assert_eq!(fixture.cache.current_entries(), 0);
        assert_eq!(fixture.cache.current_memory_usage(), 0);
        
        // Populate cache
        fixture.populate_cache(5);
        assert_eq!(fixture.cache.current_entries(), 5);
        assert!(fixture.cache.current_memory_usage() > 0);
        
        // Test cache hits
        for query in fixture.test_queries.iter().take(5) {
            assert!(fixture.cache.get(query).is_some());
        }
        
        // Test cache misses
        assert!(fixture.cache.get("nonexistent query").is_none());
        
        // Check statistics
        let stats = fixture.get_cache_stats();
        assert_eq!(stats.total_hits, 5);
        assert_eq!(stats.total_misses, 1);
        assert_eq!(stats.hit_rate, 5.0 / 6.0);
        
        // Verify consistency
        fixture.assert_cache_consistency();
    }
    
    #[test]
    fn test_cache_eviction_under_pressure() {
        let mut fixture = CacheTestFixture::new_with_config(
            CacheConfiguration::memory_constrained()
        );
        
        // Create memory pressure
        fixture.create_memory_pressure();
        
        // Cache should still be consistent
        fixture.assert_cache_consistency();
        
        // Should be within memory limits
        let memory_mb = fixture.cache.current_memory_usage_mb();
        assert!(memory_mb <= fixture.cache.max_memory_mb() as f64 * 1.1); // 10% tolerance
        
        // Should still accept new entries
        let new_results = vec![
            SearchResult {
                file_path: "new.rs".to_string(),
                content: "new content".to_string(),
                chunk_index: 0,
                score: 1.0,
            }
        ];
        
        assert!(fixture.cache.put("new_query".to_string(), new_results));
        assert!(fixture.cache.get("new_query").is_some());
    }
    
    #[test]
    fn test_cache_configuration_changes() {
        let initial_config = CacheConfiguration::minimal();
        let mut cache = MemoryEfficientCache::from_config(initial_config).unwrap();
        
        // Add some data
        let test_results = vec![
            SearchResult {
                file_path: "test.rs".to_string(),
                content: "test content".to_string(),
                chunk_index: 0,
                score: 1.0,
            }
        ];
        
        for i in 0..10 {
            cache.put(format!("query_{}", i), test_results.clone());
        }
        
        let initial_entries = cache.current_entries();
        assert!(initial_entries > 0);
        
        // Reconfigure to smaller limits
        let mut new_config = CacheConfiguration::minimal();
        new_config.max_entries = 5;
        
        cache.reconfigure(new_config).unwrap();
        
        // Should have evicted entries to meet new limit
        assert!(cache.current_entries() <= 5);
        assert!(cache.validate_cache().is_ok());
    }
    
    #[test]
    fn test_concurrent_cache_access() {
        use std::sync::Arc;
        use std::thread;
        
        let cache = Arc::new(MemoryEfficientCache::new(1000, 100));
        let mut handles = Vec::new();
        
        // Spawn multiple threads doing puts
        for thread_id in 0..4 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..50 {
                    let results = vec![
                        SearchResult {
                            file_path: format!("thread_{}_{}.rs", thread_id, i),
                            content: format!("content from thread {} item {}", thread_id, i),
                            chunk_index: 0,
                            score: 1.0,
                        }
                    ];
                    
                    cache_clone.put(format!("thread_{}_query_{}", thread_id, i), results);
                }
            });
            handles.push(handle);
        }
        
        // Spawn threads doing gets
        for thread_id in 0..2 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let query = format!("thread_{}_query_{}", thread_id % 4, i % 50);
                    let _ = cache_clone.get(&query);
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify cache consistency
        assert!(cache.validate_cache().is_ok());
        assert!(cache.validate_memory_consistency().is_ok());
        
        // Should have some entries
        assert!(cache.current_entries() > 0);
        assert!(cache.total_requests() > 0);
    }
    
    #[test]
    fn test_different_eviction_policies() {
        let policies = vec![
            ("LRU", CacheConfiguration {
                eviction: EvictionSettings {
                    policy: EvictionPolicyConfig::LRU,
                    ..CacheConfiguration::minimal().eviction
                },
                ..CacheConfiguration::minimal()
            }),
            ("LFU", CacheConfiguration {
                eviction: EvictionSettings {
                    policy: EvictionPolicyConfig::LFU,
                    ..CacheConfiguration::minimal().eviction
                },
                ..CacheConfiguration::minimal()
            }),
            ("SizeBased", CacheConfiguration {
                eviction: EvictionSettings {
                    policy: EvictionPolicyConfig::SizeBased,
                    ..CacheConfiguration::minimal().eviction
                },
                ..CacheConfiguration::minimal()
            }),
        ];
        
        for (policy_name, mut config) in policies {
            config.max_entries = 3; // Force eviction
            let mut fixture = CacheTestFixture::new_with_config(config);
            
            // Fill cache beyond capacity
            fixture.populate_cache(5);
            
            // Should have evicted entries
            assert!(fixture.cache.current_entries() <= 3);
            fixture.assert_cache_consistency();
            
            println!("Policy {} passed basic eviction test", policy_name);
        }
    }
    
    #[test]
    fn test_cache_statistics_accuracy() {
        let mut fixture = CacheTestFixture::new_balanced();
        
        // Perform known operations
        let test_results = vec![
            SearchResult {
                file_path: "stats_test.rs".to_string(),
                content: "stats content".to_string(),
                chunk_index: 0,
                score: 1.0,
            }
        ];
        
        // 3 puts
        fixture.cache.put("query1".to_string(), test_results.clone());
        fixture.cache.put("query2".to_string(), test_results.clone());
        fixture.cache.put("query3".to_string(), test_results);
        
        // 2 hits, 2 misses
        assert!(fixture.cache.get("query1").is_some());
        assert!(fixture.cache.get("query2").is_some());
        assert!(fixture.cache.get("nonexistent1").is_none());
        assert!(fixture.cache.get("nonexistent2").is_none());
        
        // Verify statistics
        assert_eq!(fixture.cache.hit_count(), 2);
        assert_eq!(fixture.cache.miss_count(), 2);
        assert_eq!(fixture.cache.total_requests(), 4);
        assert_eq!(fixture.cache.hit_rate(), 0.5);
        
        let stats = fixture.get_cache_stats();
        assert_eq!(stats.entries, 3);
        assert_eq!(stats.total_hits, 2);
        assert_eq!(stats.total_misses, 2);
        assert_eq!(stats.hit_rate, 0.5);
    }
}
```

### 3. Add performance benchmark tests
Add these performance tests:
```rust
#[cfg(test)]
mod performance_tests {
    use super::*;
    use super::test_utils::*;
    
    #[test]
    fn test_put_performance_benchmark() {
        let mut harness = PerformanceTestHarness::new(
            CacheConfiguration::performance_optimized(),
            1000
        );
        
        let result = harness.benchmark_put_operations(1000);
        
        println!("{}", result.format_report());
        
        // Performance assertions
        assert!(result.puts_per_second > 100.0); // Should handle at least 100 puts/sec
        assert!(result.avg_duration_per_put.as_millis() < 50); // Less than 50ms per put
        assert!(result.successful_puts > result.failed_puts); // More successes than failures
    }
    
    #[test]
    fn test_get_performance_benchmark() {
        let mut harness = PerformanceTestHarness::new(
            CacheConfiguration::performance_optimized(),
            500
        );
        
        // Populate cache first
        for i in 0..200 {
            harness.cache.put(
                harness.queries[i].clone(),
                harness.results[i % harness.results.len()].clone()
            );
        }
        
        let result = harness.benchmark_get_operations(1000);
        
        println!("{}", result.format_report());
        
        // Performance assertions
        assert!(result.gets_per_second > 1000.0); // Should handle at least 1000 gets/sec
        assert!(result.avg_duration_per_get.as_millis() < 5); // Less than 5ms per get
        assert!(result.hit_rate > 0.1); // Should have some hits
    }
    
    #[test]
    fn test_mixed_operations_benchmark() {
        let mut harness = PerformanceTestHarness::new(
            CacheConfiguration::balanced(),
            500
        );
        
        let result = harness.benchmark_mixed_operations(1000, 0.3); // 30% puts, 70% gets
        
        println!("{}", result.format_report());
        
        // Performance assertions
        assert!(result.operations_per_second > 200.0); // Should handle at least 200 ops/sec
        assert!(result.puts > 0);
        assert!(result.gets > 0);
        assert!(result.puts + result.gets == result.iterations);
    }
    
    #[test]
    fn test_memory_efficiency() {
        let mut fixture = CacheTestFixture::new_with_config(
            CacheConfiguration::memory_constrained()
        );
        
        // Add entries with known sizes
        let small_results = vec![
            SearchResult {
                file_path: "small.rs".to_string(),
                content: "small".to_string(),
                chunk_index: 0,
                score: 1.0,
            }
        ];
        
        let large_results = vec![
            SearchResult {
                file_path: "large.rs".to_string(),
                content: "x".repeat(10000),
                chunk_index: 0,
                score: 1.0,
            }
        ];
        
        fixture.cache.put("small".to_string(), small_results);
        fixture.cache.put("large".to_string(), large_results);
        
        let profile = fixture.cache.get_memory_profile();
        
        // Memory efficiency checks
        assert!(profile.total_calculated_usage > 0);
        assert!(profile.content_data_bytes > 0);
        assert!(profile.largest_entry_size > profile.avg_entry_size);
        
        println!("Memory efficiency test passed:\n{}", 
                 fixture.cache.memory_efficiency_report());
    }
    
    #[test]
    fn test_eviction_performance() {
        let mut fixture = CacheTestFixture::new_with_config(
            CacheConfiguration {
                max_entries: 100,
                max_memory_mb: 10,
                ..CacheConfiguration::balanced()
            }
        );
        
        let start = std::time::Instant::now();
        
        // Add many entries to trigger evictions
        for i in 0..500 {
            let results = vec![
                SearchResult {
                    file_path: format!("eviction_test_{}.rs", i),
                    content: "x".repeat(1000),
                    chunk_index: 0,
                    score: 1.0,
                }
            ];
            
            fixture.cache.put(format!("eviction_query_{}", i), results);
        }
        
        let duration = start.elapsed();
        
        // Should complete in reasonable time despite evictions
        assert!(duration.as_secs() < 5); // Less than 5 seconds
        
        // Should be within limits
        assert!(fixture.cache.current_entries() <= 100);
        assert!(fixture.cache.current_memory_usage_mb() <= 10.0 * 1.1); // 10% tolerance
        
        fixture.assert_cache_consistency();
        
        println!("Eviction performance test completed in {:.2}s", duration.as_secs_f64());
    }
    
    #[test]
    fn test_configuration_impact_on_performance() {
        let configs = vec![
            ("minimal", CacheConfiguration::minimal()),
            ("balanced", CacheConfiguration::balanced()),
            ("performance", CacheConfiguration::performance_optimized()),
        ];
        
        for (name, config) in configs {
            let mut harness = PerformanceTestHarness::new(config, 200);
            
            let start = std::time::Instant::now();
            
            // Perform mixed operations
            let result = harness.benchmark_mixed_operations(500, 0.5);
            
            let duration = start.elapsed();
            
            println!("Config '{}': {:.0} ops/sec in {:.2}s", 
                     name, result.operations_per_second, duration.as_secs_f64());
            
            // All configurations should handle basic performance
            assert!(result.operations_per_second > 50.0);
        }
    }
}
```

### 4. Add stress testing module
Add these stress tests:
```rust
#[cfg(test)]
mod stress_tests {
    use super::*;
    use super::test_utils::*;
    
    #[test]
    #[ignore] // Run only when explicitly requested
    fn test_large_cache_stress() {
        let mut fixture = CacheTestFixture::new_with_config(
            CacheConfiguration {
                max_entries: 10000,
                max_memory_mb: 1000,
                ..CacheConfiguration::performance_optimized()
            }
        );
        
        let start = std::time::Instant::now();
        
        // Add large number of entries
        for i in 0..5000 {
            let results = vec![
                SearchResult {
                    file_path: format!("stress_test_{}.rs", i),
                    content: format!("Content for stress test entry {}", i),
                    chunk_index: 0,
                    score: 1.0 - (i as f64 * 0.0001),
                }
            ];
            
            fixture.cache.put(format!("stress_query_{}", i), results);
            
            if i % 1000 == 0 {
                fixture.assert_cache_consistency();
                println!("Added {} entries, memory: {:.2} MB", 
                         i + 1, fixture.cache.current_memory_usage_mb());
            }
        }
        
        let duration = start.elapsed();
        println!("Stress test completed in {:.2}s", duration.as_secs_f64());
        
        // Final consistency check
        fixture.assert_cache_consistency();
        
        // Performance check
        assert!(duration.as_secs() < 60); // Should complete within 60 seconds
    }
    
    #[test]
    #[ignore] // Run only when explicitly requested
    fn test_memory_limit_stress() {
        let mut fixture = CacheTestFixture::new_with_config(
            CacheConfiguration {
                max_entries: 100000,
                max_memory_mb: 50, // Small memory limit
                ..CacheConfiguration::memory_constrained()
            }
        );
        
        // Try to add way more data than memory allows
        let large_content = "x".repeat(10000); // 10KB per entry
        
        for i in 0..1000 {
            let results = vec![
                SearchResult {
                    file_path: format!("memory_stress_{}.rs", i),
                    content: large_content.clone(),
                    chunk_index: 0,
                    score: 1.0,
                }
            ];
            
            fixture.cache.put(format!("memory_stress_query_{}", i), results);
            
            if i % 100 == 0 {
                fixture.assert_cache_consistency();
                let memory_mb = fixture.cache.current_memory_usage_mb();
                assert!(memory_mb <= 55.0); // Should stay within limit (5MB tolerance)
            }
        }
        
        fixture.assert_cache_consistency();
    }
}
```

## Success Criteria
- [ ] Comprehensive test utilities and fixtures created
- [ ] Integration tests cover full cache lifecycle
- [ ] Performance benchmark framework implemented
- [ ] Concurrent access testing validates thread safety
- [ ] Stress tests validate cache under extreme conditions
- [ ] All test configurations validate successfully
- [ ] Performance benchmarks provide actionable metrics
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Test fixtures provide consistent test data
- Performance harness enables systematic benchmarking
- Integration tests validate real-world scenarios
- Stress tests ensure cache handles extreme conditions
- Concurrent tests validate thread safety
- Benchmarks help optimize cache performance