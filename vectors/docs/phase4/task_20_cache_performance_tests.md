# Task 20: Implement Cache Performance Tests

## Context
You are implementing Phase 4 of a vector indexing system. Cache tests setup was implemented in the previous task with comprehensive test utilities. Now you need to implement detailed performance tests with benchmarking, profiling, and optimization validation.

## Current State
- `src/cache.rs` exists with full cache implementation
- Comprehensive test utilities and fixtures are available
- Basic performance benchmarks exist but need detailed analysis
- Need systematic performance validation and regression testing

## Task Objective
Implement detailed cache performance tests with benchmarking, profiling, regression detection, and performance optimization validation.

## Implementation Requirements

### 1. Add detailed performance analysis utilities
Add this performance analysis module to the cache test section:
```rust
#[cfg(test)]
pub mod performance_analysis {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::{Duration, Instant};
    
    #[derive(Debug, Clone)]
    pub struct PerformanceProfile {
        pub operation_type: String,
        pub sample_count: usize,
        pub min_duration: Duration,
        pub max_duration: Duration,
        pub avg_duration: Duration,
        pub median_duration: Duration,
        pub p95_duration: Duration,
        pub p99_duration: Duration,
        pub total_duration: Duration,
        pub operations_per_second: f64,
        pub memory_usage_samples: Vec<usize>,
        pub cache_hit_rates: Vec<f64>,
    }
    
    impl PerformanceProfile {
        pub fn new(operation_type: String) -> Self {
            Self {
                operation_type,
                sample_count: 0,
                min_duration: Duration::MAX,
                max_duration: Duration::ZERO,
                avg_duration: Duration::ZERO,
                median_duration: Duration::ZERO,
                p95_duration: Duration::ZERO,
                p99_duration: Duration::ZERO,
                total_duration: Duration::ZERO,
                operations_per_second: 0.0,
                memory_usage_samples: Vec::new(),
                cache_hit_rates: Vec::new(),
            }
        }
        
        pub fn from_durations(operation_type: String, mut durations: Vec<Duration>) -> Self {
            if durations.is_empty() {
                return Self::new(operation_type);
            }
            
            durations.sort();
            let sample_count = durations.len();
            
            let min_duration = durations[0];
            let max_duration = durations[sample_count - 1];
            let total_duration: Duration = durations.iter().sum();
            let avg_duration = total_duration / sample_count as u32;
            
            let median_duration = durations[sample_count / 2];
            let p95_duration = durations[(sample_count as f64 * 0.95) as usize];
            let p99_duration = durations[(sample_count as f64 * 0.99) as usize];
            
            let operations_per_second = sample_count as f64 / total_duration.as_secs_f64();
            
            Self {
                operation_type,
                sample_count,
                min_duration,
                max_duration,
                avg_duration,
                median_duration,
                p95_duration,
                p99_duration,
                total_duration,
                operations_per_second,
                memory_usage_samples: Vec::new(),
                cache_hit_rates: Vec::new(),
            }
        }
        
        pub fn format_detailed_report(&self) -> String {
            format!(
                "Performance Profile: {}\n\
                 Samples: {}\n\
                 Duration Stats:\n\
                   Min: {:.2}ms\n\
                   Max: {:.2}ms\n\
                   Avg: {:.2}ms\n\
                   Median: {:.2}ms\n\
                   P95: {:.2}ms\n\
                   P99: {:.2}ms\n\
                 Throughput: {:.0} ops/sec\n\
                 Total Time: {:.2}s",
                self.operation_type,
                self.sample_count,
                self.min_duration.as_secs_f64() * 1000.0,
                self.max_duration.as_secs_f64() * 1000.0,
                self.avg_duration.as_secs_f64() * 1000.0,
                self.median_duration.as_secs_f64() * 1000.0,
                self.p95_duration.as_secs_f64() * 1000.0,
                self.p99_duration.as_secs_f64() * 1000.0,
                self.operations_per_second,
                self.total_duration.as_secs_f64()
            )
        }
        
        pub fn is_performance_regression(&self, baseline: &PerformanceProfile, threshold: f64) -> bool {
            if baseline.sample_count == 0 {
                return false;
            }
            
            let avg_regression = (self.avg_duration.as_secs_f64() - baseline.avg_duration.as_secs_f64()) 
                                / baseline.avg_duration.as_secs_f64();
            let p95_regression = (self.p95_duration.as_secs_f64() - baseline.p95_duration.as_secs_f64())
                                / baseline.p95_duration.as_secs_f64();
                                
            avg_regression > threshold || p95_regression > threshold
        }
    }
    
    pub struct PerformanceBenchmarker {
        cache: MemoryEfficientCache,
        test_data: Vec<(String, Vec<SearchResult>)>,
        warmup_iterations: usize,
        measurement_iterations: usize,
    }
    
    impl PerformanceBenchmarker {
        pub fn new(cache_config: CacheConfiguration, data_size: usize) -> Self {
            let cache = MemoryEfficientCache::from_config(cache_config).unwrap();
            let test_data = Self::generate_test_data(data_size);
            
            Self {
                cache,
                test_data,
                warmup_iterations: 100,
                measurement_iterations: 1000,
            }
        }
        
        fn generate_test_data(size: usize) -> Vec<(String, Vec<SearchResult>)> {
            (0..size).map(|i| {
                let query = format!("benchmark_query_{}", i);
                let result_count = (i % 5) + 1; // 1-5 results per query
                
                let results = (0..result_count).map(|j| {
                    let content_size = 100 + (i * 5) + (j * 20);
                    SearchResult {
                        file_path: format!("bench_file_{}_{}.rs", i, j),
                        content: "x".repeat(content_size),
                        chunk_index: j,
                        score: 1.0 - (j as f64 * 0.1),
                    }
                }).collect();
                
                (query, results)
            }).collect()
        }
        
        pub fn benchmark_put_operations(&mut self) -> PerformanceProfile {
            // Warmup
            for i in 0..self.warmup_iterations {
                let (query, results) = &self.test_data[i % self.test_data.len()];
                self.cache.put(format!("warmup_{}", query), results.clone());
            }
            
            // Clear cache for clean measurement
            self.cache.clear();
            
            // Measurement
            let mut durations = Vec::new();
            let mut memory_samples = Vec::new();
            
            for i in 0..self.measurement_iterations {
                let (query, results) = &self.test_data[i % self.test_data.len()];
                
                let start = Instant::now();
                self.cache.put(format!("bench_{}", query), results.clone());
                let duration = start.elapsed();
                
                durations.push(duration);
                memory_samples.push(self.cache.current_memory_usage());
            }
            
            let mut profile = PerformanceProfile::from_durations("PUT".to_string(), durations);
            profile.memory_usage_samples = memory_samples;
            profile
        }
        
        pub fn benchmark_get_operations(&mut self, hit_ratio: f64) -> PerformanceProfile {
            // Populate cache for gets
            let populate_count = (self.test_data.len() as f64 * hit_ratio) as usize;
            for i in 0..populate_count {
                let (query, results) = &self.test_data[i];
                self.cache.put(query.clone(), results.clone());
            }
            
            // Warmup
            for i in 0..self.warmup_iterations {
                let query_idx = i % self.test_data.len();
                let (query, _) = &self.test_data[query_idx];
                self.cache.get(query);
            }
            
            // Measurement
            let mut durations = Vec::new();
            let mut hit_rates = Vec::new();
            let initial_hits = self.cache.hit_count();
            let initial_total = self.cache.total_requests();
            
            for i in 0..self.measurement_iterations {
                let query_idx = i % self.test_data.len();
                let (query, _) = &self.test_data[query_idx];
                
                let start = Instant::now();
                self.cache.get(query);
                let duration = start.elapsed();
                
                durations.push(duration);
                
                // Sample hit rate periodically
                if i % 100 == 0 {
                    let current_hits = self.cache.hit_count() - initial_hits;
                    let current_total = self.cache.total_requests() - initial_total;
                    if current_total > 0 {
                        hit_rates.push(current_hits as f64 / current_total as f64);
                    }
                }
            }
            
            let mut profile = PerformanceProfile::from_durations("GET".to_string(), durations);
            profile.cache_hit_rates = hit_rates;
            profile
        }
        
        pub fn benchmark_mixed_workload(&mut self, put_ratio: f64) -> PerformanceProfile {
            // Warmup
            for i in 0..self.warmup_iterations {
                if (i as f64 / self.warmup_iterations as f64) < put_ratio {
                    let (query, results) = &self.test_data[i % self.test_data.len()];
                    self.cache.put(format!("warmup_{}", query), results.clone());
                } else {
                    let (query, _) = &self.test_data[i % self.test_data.len()];
                    self.cache.get(query);
                }
            }
            
            // Measurement
            let mut durations = Vec::new();
            let mut memory_samples = Vec::new();
            let mut hit_rates = Vec::new();
            let initial_hits = self.cache.hit_count();
            let initial_total = self.cache.total_requests();
            
            for i in 0..self.measurement_iterations {
                let is_put = (i as f64 / self.measurement_iterations as f64) < put_ratio;
                let (query, results) = &self.test_data[i % self.test_data.len()];
                
                let start = Instant::now();
                if is_put {
                    self.cache.put(format!("mixed_{}", query), results.clone());
                } else {
                    self.cache.get(query);
                }
                let duration = start.elapsed();
                
                durations.push(duration);
                
                if i % 50 == 0 {
                    memory_samples.push(self.cache.current_memory_usage());
                    let current_hits = self.cache.hit_count() - initial_hits;
                    let current_total = self.cache.total_requests() - initial_total;
                    if current_total > 0 {
                        hit_rates.push(current_hits as f64 / current_total as f64);
                    }
                }
            }
            
            let mut profile = PerformanceProfile::from_durations("MIXED".to_string(), durations);
            profile.memory_usage_samples = memory_samples;
            profile.cache_hit_rates = hit_rates;
            profile
        }
        
        pub fn benchmark_eviction_performance(&mut self) -> PerformanceProfile {
            // Configure for aggressive eviction
            let small_config = CacheConfiguration {
                max_entries: 100,
                max_memory_mb: 5,
                ..CacheConfiguration::balanced()
            };
            
            self.cache.reconfigure(small_config).unwrap();
            
            // Generate data that will cause evictions
            let large_content = "x".repeat(10000); // 10KB per entry
            let mut durations = Vec::new();
            let mut memory_samples = Vec::new();
            
            for i in 0..self.measurement_iterations {
                let results = vec![
                    SearchResult {
                        file_path: format!("eviction_test_{}.rs", i),
                        content: large_content.clone(),
                        chunk_index: 0,
                        score: 1.0,
                    }
                ];
                
                let start = Instant::now();
                self.cache.put(format!("eviction_query_{}", i), results);
                let duration = start.elapsed();
                
                durations.push(duration);
                
                if i % 20 == 0 {
                    memory_samples.push(self.cache.current_memory_usage());
                }
            }
            
            let mut profile = PerformanceProfile::from_durations("EVICTION".to_string(), durations);
            profile.memory_usage_samples = memory_samples;
            profile
        }
    }
    
    pub struct PerformanceRegression {
        pub operation_type: String,
        pub avg_regression_percent: f64,
        pub p95_regression_percent: f64,
        pub throughput_regression_percent: f64,
        pub is_significant: bool,
    }
    
    impl PerformanceRegression {
        pub fn analyze(current: &PerformanceProfile, baseline: &PerformanceProfile, threshold: f64) -> Self {
            let avg_regression = ((current.avg_duration.as_secs_f64() - baseline.avg_duration.as_secs_f64()) 
                                / baseline.avg_duration.as_secs_f64()) * 100.0;
            
            let p95_regression = ((current.p95_duration.as_secs_f64() - baseline.p95_duration.as_secs_f64())
                                / baseline.p95_duration.as_secs_f64()) * 100.0;
                                
            let throughput_regression = ((baseline.operations_per_second - current.operations_per_second)
                                       / baseline.operations_per_second) * 100.0;
            
            let is_significant = avg_regression > threshold || p95_regression > threshold || throughput_regression > threshold;
            
            Self {
                operation_type: current.operation_type.clone(),
                avg_regression_percent: avg_regression,
                p95_regression_percent: p95_regression,
                throughput_regression_percent: throughput_regression,
                is_significant,
            }
        }
        
        pub fn format_report(&self) -> String {
            format!(
                "Regression Analysis: {}\n\
                 Average Duration: {:+.1}%\n\
                 P95 Duration: {:+.1}%\n\
                 Throughput: {:+.1}%\n\
                 Significant: {}",
                self.operation_type,
                self.avg_regression_percent,
                self.p95_regression_percent,
                self.throughput_regression_percent,
                self.is_significant
            )
        }
    }
}
```

### 2. Add comprehensive performance test suite
Add these detailed performance tests:
```rust
#[cfg(test)]
mod detailed_performance_tests {
    use super::*;
    use super::performance_analysis::*;
    
    #[test]
    fn test_put_operation_performance_characteristics() {
        let configs = vec![
            ("minimal", CacheConfiguration::minimal()),
            ("balanced", CacheConfiguration::balanced()),
            ("performance", CacheConfiguration::performance_optimized()),
        ];
        
        for (config_name, config) in configs {
            let mut benchmarker = PerformanceBenchmarker::new(config, 1000);
            let profile = benchmarker.benchmark_put_operations();
            
            println!("{} - {}", config_name, profile.format_detailed_report());
            
            // Performance requirements
            assert!(profile.avg_duration.as_millis() < 10, 
                   "PUT average duration too high for {}: {}ms", 
                   config_name, profile.avg_duration.as_millis());
            
            assert!(profile.p95_duration.as_millis() < 50,
                   "PUT P95 duration too high for {}: {}ms",
                   config_name, profile.p95_duration.as_millis());
            
            assert!(profile.operations_per_second > 100.0,
                   "PUT throughput too low for {}: {:.0} ops/sec",
                   config_name, profile.operations_per_second);
        }
    }
    
    #[test]
    fn test_get_operation_performance_characteristics() {
        let hit_ratios = vec![0.0, 0.5, 0.9]; // Test different hit ratios
        
        for hit_ratio in hit_ratios {
            let mut benchmarker = PerformanceBenchmarker::new(
                CacheConfiguration::balanced(), 1000
            );
            
            let profile = benchmarker.benchmark_get_operations(hit_ratio);
            
            println!("GET ({}% hit ratio) - {}", 
                     (hit_ratio * 100.0) as u32, 
                     profile.format_detailed_report());
            
            // Performance requirements (should be very fast)
            assert!(profile.avg_duration.as_millis() < 2,
                   "GET average duration too high with {}% hits: {}ms",
                   (hit_ratio * 100.0) as u32, profile.avg_duration.as_millis());
            
            assert!(profile.operations_per_second > 1000.0,
                   "GET throughput too low with {}% hits: {:.0} ops/sec",
                   (hit_ratio * 100.0) as u32, profile.operations_per_second);
            
            // Verify hit rate is approximately correct
            if !profile.cache_hit_rates.is_empty() {
                let actual_hit_rate = profile.cache_hit_rates.last().unwrap();
                let hit_rate_diff = (actual_hit_rate - hit_ratio).abs();
                assert!(hit_rate_diff < 0.1, 
                       "Hit rate deviation too high: expected {:.1}%, got {:.1}%",
                       hit_ratio * 100.0, actual_hit_rate * 100.0);
            }
        }
    }
    
    #[test]
    fn test_mixed_workload_performance() {
        let workload_ratios = vec![
            (0.1, "read-heavy"),   // 10% puts, 90% gets
            (0.5, "balanced"),     // 50% puts, 50% gets  
            (0.8, "write-heavy"),  // 80% puts, 20% gets
        ];
        
        for (put_ratio, workload_name) in workload_ratios {
            let mut benchmarker = PerformanceBenchmarker::new(
                CacheConfiguration::balanced(), 500
            );
            
            let profile = benchmarker.benchmark_mixed_workload(put_ratio);
            
            println!("{} workload - {}", workload_name, profile.format_detailed_report());
            
            // Performance requirements vary by workload
            let max_avg_ms = if put_ratio > 0.7 { 8 } else { 5 }; // Write-heavy can be slower
            let min_throughput = if put_ratio > 0.7 { 200.0 } else { 500.0 };
            
            assert!(profile.avg_duration.as_millis() < max_avg_ms,
                   "{} workload average duration too high: {}ms",
                   workload_name, profile.avg_duration.as_millis());
            
            assert!(profile.operations_per_second > min_throughput,
                   "{} workload throughput too low: {:.0} ops/sec",
                   workload_name, profile.operations_per_second);
            
            // Memory usage should be stable
            if profile.memory_usage_samples.len() > 1 {
                let memory_growth = profile.memory_usage_samples.last().unwrap() 
                                  - profile.memory_usage_samples.first().unwrap();
                assert!(memory_growth < 10 * 1024 * 1024, // Less than 10MB growth
                       "{} workload memory growth too high: {} bytes",
                       workload_name, memory_growth);
            }
        }
    }
    
    #[test]
    fn test_eviction_performance_impact() {
        let mut benchmarker = PerformanceBenchmarker::new(
            CacheConfiguration::balanced(), 200
        );
        
        let profile = benchmarker.benchmark_eviction_performance();
        
        println!("Eviction Performance - {}", profile.format_detailed_report());
        
        // Eviction adds overhead but should still be reasonable
        assert!(profile.avg_duration.as_millis() < 20,
               "Eviction average duration too high: {}ms",
               profile.avg_duration.as_millis());
        
        assert!(profile.p95_duration.as_millis() < 100,
               "Eviction P95 duration too high: {}ms", 
               profile.p95_duration.as_millis());
        
        assert!(profile.operations_per_second > 50.0,
               "Eviction throughput too low: {:.0} ops/sec",
               profile.operations_per_second);
        
        // Memory usage should be bounded
        if !profile.memory_usage_samples.is_empty() {
            let max_memory = profile.memory_usage_samples.iter().max().unwrap();
            assert!(*max_memory < 6 * 1024 * 1024, // Should stay under 6MB
                   "Memory usage exceeded limits during eviction: {} bytes", max_memory);
        }
    }
    
    #[test]
    fn test_cache_size_impact_on_performance() {
        let cache_sizes = vec![
            (10, 1),     // Very small
            (100, 10),   // Small
            (1000, 100), // Medium
            (5000, 500), // Large
        ];
        
        for (max_entries, max_memory_mb) in cache_sizes {
            let config = CacheConfiguration {
                max_entries,
                max_memory_mb,
                ..CacheConfiguration::balanced()
            };
            
            let mut benchmarker = PerformanceBenchmarker::new(config, 200);
            let profile = benchmarker.benchmark_mixed_workload(0.3);
            
            println!("Cache size {}E/{}MB - Throughput: {:.0} ops/sec, Avg: {:.1}ms",
                     max_entries, max_memory_mb, 
                     profile.operations_per_second,
                     profile.avg_duration.as_secs_f64() * 1000.0);
            
            // Larger caches shouldn't be significantly slower
            assert!(profile.avg_duration.as_millis() < 15,
                   "Cache size {}/{} average duration too high: {}ms",
                   max_entries, max_memory_mb, profile.avg_duration.as_millis());
            
            // Should maintain minimum throughput
            assert!(profile.operations_per_second > 50.0,
                   "Cache size {}/{} throughput too low: {:.0} ops/sec",
                   max_entries, max_memory_mb, profile.operations_per_second);
        }
    }
    
    #[test]
    fn test_eviction_policy_performance_comparison() {
        let policies = vec![
            ("LRU", EvictionPolicyConfig::LRU),
            ("LFU", EvictionPolicyConfig::LFU),
            ("SizeBased", EvictionPolicyConfig::SizeBased),
            ("Hybrid", EvictionPolicyConfig::Hybrid { size_weight: 0.5 }),
        ];
        
        let mut results = Vec::new();
        
        for (policy_name, policy_config) in policies {
            let config = CacheConfiguration {
                max_entries: 200,
                max_memory_mb: 20,
                eviction: EvictionSettings {
                    policy: policy_config,
                    ..CacheConfiguration::balanced().eviction
                },
                ..CacheConfiguration::balanced()
            };
            
            let mut benchmarker = PerformanceBenchmarker::new(config, 300);
            let profile = benchmarker.benchmark_eviction_performance();
            
            results.push((policy_name, profile));
        }
        
        // Compare results
        for (policy_name, profile) in &results {
            println!("{} Policy - Throughput: {:.0} ops/sec, P95: {:.1}ms",
                     policy_name, profile.operations_per_second,
                     profile.p95_duration.as_secs_f64() * 1000.0);
            
            // All policies should meet minimum performance
            assert!(profile.operations_per_second > 30.0,
                   "{} policy throughput too low: {:.0} ops/sec",
                   policy_name, profile.operations_per_second);
        }
        
        // Find best performing policy
        let best_policy = results.iter()
            .max_by(|a, b| a.1.operations_per_second.partial_cmp(&b.1.operations_per_second).unwrap())
            .unwrap();
        
        println!("Best performing eviction policy: {} ({:.0} ops/sec)",
                 best_policy.0, best_policy.1.operations_per_second);
    }
    
    #[test]
    fn test_memory_pressure_performance_degradation() {
        let pressure_levels = vec![
            (0.5, "low"),      // 50% memory usage
            (0.8, "medium"),   // 80% memory usage  
            (0.95, "high"),    // 95% memory usage
        ];
        
        for (pressure_level, pressure_name) in pressure_levels {
            let config = CacheConfiguration {
                max_entries: 1000,
                max_memory_mb: 50,
                eviction: EvictionSettings {
                    memory_pressure_threshold: pressure_level,
                    aggressive_eviction_threshold: pressure_level + 0.05,
                    ..CacheConfiguration::balanced().eviction
                },
                ..CacheConfiguration::balanced()
            };
            
            let mut benchmarker = PerformanceBenchmarker::new(config, 500);
            
            // Fill cache to pressure level
            let target_memory = (50.0 * pressure_level) as usize * 1024 * 1024;
            let large_content = "x".repeat(5000);
            let mut counter = 0;
            
            while benchmarker.cache.current_memory_usage() < target_memory && counter < 200 {
                let results = vec![
                    SearchResult {
                        file_path: format!("pressure_fill_{}.rs", counter),
                        content: large_content.clone(),
                        chunk_index: 0,
                        score: 1.0,
                    }
                ];
                benchmarker.cache.put(format!("pressure_fill_{}", counter), results);
                counter += 1;
            }
            
            // Now benchmark performance under pressure
            let profile = benchmarker.benchmark_mixed_workload(0.2);
            
            println!("{} pressure - Throughput: {:.0} ops/sec, P95: {:.1}ms, Memory: {:.1}MB",
                     pressure_name, profile.operations_per_second,
                     profile.p95_duration.as_secs_f64() * 1000.0,
                     benchmarker.cache.current_memory_usage_mb());
            
            // Performance should degrade gracefully under pressure
            let min_throughput = match pressure_name {
                "low" => 200.0,
                "medium" => 100.0,
                "high" => 50.0,
                _ => 30.0,
            };
            
            assert!(profile.operations_per_second > min_throughput,
                   "{} pressure throughput too low: {:.0} ops/sec",
                   pressure_name, profile.operations_per_second);
        }
    }
    
    #[test]
    fn test_performance_regression_detection() {
        // Create a baseline profile
        let mut baseline_benchmarker = PerformanceBenchmarker::new(
            CacheConfiguration::balanced(), 500
        );
        let baseline_profile = baseline_benchmarker.benchmark_mixed_workload(0.3);
        
        // Simulate a performance regression by using a slower configuration
        let regressed_config = CacheConfiguration {
            max_entries: 50,   // Much smaller, should cause more evictions
            max_memory_mb: 5,  // Much smaller memory
            eviction: EvictionSettings {
                max_eviction_batch_size: 1, // Inefficient eviction
                ..CacheConfiguration::balanced().eviction
            },
            ..CacheConfiguration::balanced()
        };
        
        let mut regressed_benchmarker = PerformanceBenchmarker::new(regressed_config, 500);
        let regressed_profile = regressed_benchmarker.benchmark_mixed_workload(0.3);
        
        // Analyze regression
        let regression = PerformanceRegression::analyze(&regressed_profile, &baseline_profile, 20.0);
        
        println!("Baseline: {:.0} ops/sec, {:.1}ms avg",
                 baseline_profile.operations_per_second,
                 baseline_profile.avg_duration.as_secs_f64() * 1000.0);
        
        println!("Regressed: {:.0} ops/sec, {:.1}ms avg",
                 regressed_profile.operations_per_second,
                 regressed_profile.avg_duration.as_secs_f64() * 1000.0);
        
        println!("{}", regression.format_report());
        
        // Should detect significant regression
        assert!(regression.is_significant,
               "Failed to detect performance regression");
        
        assert!(regression.avg_regression_percent > 20.0 || regression.throughput_regression_percent > 20.0,
               "Regression not significant enough: avg={:.1}%, throughput={:.1}%",
               regression.avg_regression_percent, regression.throughput_regression_percent);
    }
}
```

### 3. Add scalability performance tests
Add these scalability-focused tests:
```rust
#[cfg(test)]
mod scalability_performance_tests {
    use super::*;
    use super::performance_analysis::*;
    
    #[test]
    fn test_cache_performance_scaling() {
        let data_sizes = vec![100, 500, 1000, 2000, 5000];
        let mut results = Vec::new();
        
        for data_size in data_sizes {
            let mut benchmarker = PerformanceBenchmarker::new(
                CacheConfiguration::performance_optimized(), data_size
            );
            
            let profile = benchmarker.benchmark_mixed_workload(0.3);
            results.push((data_size, profile));
            
            println!("Data size {}: {:.0} ops/sec, {:.1}ms avg",
                     data_size, profile.operations_per_second,
                     profile.avg_duration.as_secs_f64() * 1000.0);
        }
        
        // Verify performance doesn't degrade significantly with data size
        let baseline_throughput = results[0].1.operations_per_second;
        
        for (data_size, profile) in &results[1..] {
            let throughput_ratio = profile.operations_per_second / baseline_throughput;
            
            assert!(throughput_ratio > 0.5, // Should maintain at least 50% of baseline throughput
                   "Throughput degraded too much with data size {}: {:.1}% of baseline",
                   data_size, throughput_ratio * 100.0);
        }
    }
    
    #[test]
    fn test_cache_memory_scaling_efficiency() {
        let memory_limits = vec![10, 50, 100, 200, 500]; // MB
        
        for memory_limit in memory_limits {
            let config = CacheConfiguration {
                max_entries: memory_limit * 100, // Scale entries with memory
                max_memory_mb: memory_limit,
                ..CacheConfiguration::performance_optimized()
            };
            
            let mut benchmarker = PerformanceBenchmarker::new(config, 1000);
            
            // Fill cache to near capacity
            let fill_iterations = memory_limit * 20; // Adjust based on memory
            for i in 0..fill_iterations {
                let results = vec![
                    SearchResult {
                        file_path: format!("scale_test_{}.rs", i),
                        content: "x".repeat(1000),
                        chunk_index: 0,
                        score: 1.0,
                    }
                ];
                
                if !benchmarker.cache.put(format!("scale_query_{}", i), results) {
                    break; // Memory limit reached
                }
            }
            
            let profile = benchmarker.benchmark_get_operations(0.8);
            let memory_mb = benchmarker.cache.current_memory_usage_mb();
            let memory_efficiency = profile.operations_per_second / memory_mb;
            
            println!("Memory {}MB: {:.0} ops/sec, {:.1}MB used, {:.1} ops/sec/MB",
                     memory_limit, profile.operations_per_second, memory_mb, memory_efficiency);
            
            // Performance should scale reasonably with memory
            assert!(profile.operations_per_second > 500.0,
                   "Performance too low with {}MB: {:.0} ops/sec",
                   memory_limit, profile.operations_per_second);
            
            // Should use memory efficiently
            assert!(memory_mb <= memory_limit as f64 * 1.1,
                   "Memory usage exceeded limit: {:.1}MB > {}MB",
                   memory_mb, memory_limit);
        }
    }
    
    #[test]
    fn test_concurrent_performance_scaling() {
        use std::sync::Arc;
        use std::thread;
        
        let thread_counts = vec![1, 2, 4, 8];
        let operations_per_thread = 500;
        
        for thread_count in thread_counts {
            let cache = Arc::new(MemoryEfficientCache::new(5000, 500));
            
            // Pre-populate cache
            for i in 0..1000 {
                let results = vec![
                    SearchResult {
                        file_path: format!("concurrent_{}.rs", i),
                        content: format!("content_{}", i),
                        chunk_index: 0,
                        score: 1.0,
                    }
                ];
                cache.put(format!("concurrent_query_{}", i), results);
            }
            
            let start = std::time::Instant::now();
            let mut handles = Vec::new();
            
            // Spawn threads performing mixed operations
            for thread_id in 0..thread_count {
                let cache_clone = Arc::clone(&cache);
                let handle = thread::spawn(move || {
                    for i in 0..operations_per_thread {
                        if i % 4 == 0 { // 25% puts
                            let results = vec![
                                SearchResult {
                                    file_path: format!("thread_{}_{}.rs", thread_id, i),
                                    content: format!("thread {} content {}", thread_id, i),
                                    chunk_index: 0,
                                    score: 1.0,
                                }
                            ];
                            cache_clone.put(format!("thread_{}_query_{}", thread_id, i), results);
                        } else { // 75% gets
                            let query_id = i % 1000;
                            cache_clone.get(&format!("concurrent_query_{}", query_id));
                        }
                    }
                });
                handles.push(handle);
            }
            
            // Wait for completion
            for handle in handles {
                handle.join().unwrap();
            }
            
            let duration = start.elapsed();
            let total_operations = thread_count * operations_per_thread;
            let throughput = total_operations as f64 / duration.as_secs_f64();
            
            println!("{} threads: {:.0} ops/sec ({:.2}s for {} ops)",
                     thread_count, throughput, duration.as_secs_f64(), total_operations);
            
            // Should maintain reasonable performance with more threads
            let min_throughput = match thread_count {
                1 => 500.0,
                2 => 800.0,
                4 => 1200.0,
                8 => 1500.0,
                _ => 100.0,
            };
            
            assert!(throughput > min_throughput,
                   "Concurrent throughput too low with {} threads: {:.0} ops/sec",
                   thread_count, throughput);
            
            // Verify cache consistency after concurrent access
            assert!(cache.validate_cache().is_ok());
            assert!(cache.validate_memory_consistency().is_ok());
        }
    }
}
```

## Success Criteria
- [ ] Detailed performance analysis utilities implemented
- [ ] Comprehensive performance test suite covers all operations
- [ ] Performance regression detection works correctly
- [ ] Scalability tests validate performance under load
- [ ] Concurrent performance tests validate thread safety
- [ ] Performance requirements are verified for all configurations
- [ ] Benchmarking provides actionable performance metrics
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Performance profiles provide detailed latency analysis
- Regression detection prevents performance regressions
- Scalability tests ensure cache handles growth
- Concurrent tests validate multi-threaded performance
- Benchmarking framework enables continuous performance monitoring
- Different workload patterns test realistic scenarios