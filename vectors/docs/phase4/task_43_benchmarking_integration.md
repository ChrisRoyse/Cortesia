# Task 43: Benchmarking Integration with Criterion

## Context
You are implementing Phase 4 of a vector indexing system. Comprehensive monitoring test suite is now available. This task integrates the performance monitoring system with Criterion benchmarking framework for automated performance testing and regression detection.

## Current State
- `src/monitor.rs` exists with complete performance monitoring functionality
- Comprehensive test suite validates all monitoring components
- Need integration with benchmarking tools for automated performance validation

## Task Objective
Integrate the performance monitoring system with Criterion benchmarking framework to provide automated performance testing, baseline establishment, regression detection, and continuous performance validation.

## Implementation Requirements

### 1. Add Criterion dependency
Add to `Cargo.toml`:
```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "monitor_benchmarks"
harness = false
```

### 2. Create benchmark utilities
Create a new file `src/monitor/benchmark_utils.rs`:
```rust
use super::*;
use criterion::{Criterion, BenchmarkId, BatchSize, black_box};
use std::time::{Duration, Instant};

pub struct BenchmarkMonitor {
    monitor: PerformanceMonitor,
    criterion_monitor: PerformanceMonitor,
}

impl BenchmarkMonitor {
    pub fn new() -> Self {
        Self {
            monitor: PerformanceMonitor::new(),
            criterion_monitor: PerformanceMonitor::new(),
        }
    }
    
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            monitor: PerformanceMonitor::with_capacity(capacity),
            criterion_monitor: PerformanceMonitor::with_capacity(capacity),
        }
    }
    
    /// Benchmark a query operation with monitoring
    pub fn bench_query<F, R>(&mut self, mut operation: F) -> R
    where
        F: FnMut() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();
        
        self.monitor.record_query_time(duration);
        result
    }
    
    /// Benchmark an indexing operation with monitoring
    pub fn bench_index<F, R>(&mut self, mut operation: F) -> R
    where
        F: FnMut() -> R,
    {
        let start = Instant::now();
        let result = operation();
        let duration = start.elapsed();
        
        self.monitor.record_index_time(duration);
        result
    }
    
    /// Record Criterion benchmark results for comparison
    pub fn record_criterion_result(&mut self, benchmark_name: &str, duration: Duration, is_query: bool) {
        if is_query {
            self.criterion_monitor.record_query_time(duration);
        } else {
            self.criterion_monitor.record_index_time(duration);
        }
    }
    
    /// Compare monitoring results with Criterion results
    pub fn compare_with_criterion(&self) -> BenchmarkComparison {
        let monitor_stats = self.monitor.get_stats();
        let criterion_stats = self.criterion_monitor.get_stats();
        
        BenchmarkComparison {
            monitor_query_avg: monitor_stats.avg_query_time,
            criterion_query_avg: criterion_stats.avg_query_time,
            monitor_index_avg: monitor_stats.avg_index_time,
            criterion_index_avg: criterion_stats.avg_index_time,
            monitor_samples: monitor_stats.total_queries + monitor_stats.total_indexes,
            criterion_samples: criterion_stats.total_queries + criterion_stats.total_indexes,
        }
    }
    
    /// Generate performance baseline from current measurements
    pub fn generate_baseline(&self) -> PerformanceBaseline {
        let stats = self.monitor.get_stats();
        let advanced_stats = self.monitor.get_advanced_stats(0.95);
        
        PerformanceBaseline {
            timestamp: std::time::SystemTime::now(),
            query_baseline: QueryBaseline {
                avg_time: stats.avg_query_time,
                p95_time: stats.p95_query_time,
                p99_time: stats.p99_query_time,
                std_dev: advanced_stats.query_std_dev,
                sample_count: stats.total_queries,
            },
            index_baseline: IndexBaseline {
                avg_time: stats.avg_index_time,
                p95_time: stats.p95_index_time,
                p99_time: stats.p99_index_time,
                std_dev: advanced_stats.index_std_dev,
                sample_count: stats.total_indexes,
            },
        }
    }
    
    /// Detect performance regression against baseline
    pub fn detect_regression(&self, baseline: &PerformanceBaseline, threshold: f64) -> RegressionReport {
        let current_stats = self.monitor.get_stats();
        let current_advanced = self.monitor.get_advanced_stats(0.95);
        
        let query_regression = self.calculate_regression_factor(
            baseline.query_baseline.avg_time,
            current_stats.avg_query_time,
        );
        
        let index_regression = self.calculate_regression_factor(
            baseline.index_baseline.avg_time,
            current_stats.avg_index_time,
        );
        
        let p95_query_regression = self.calculate_regression_factor(
            baseline.query_baseline.p95_time,
            current_stats.p95_query_time,
        );
        
        let p99_query_regression = self.calculate_regression_factor(
            baseline.query_baseline.p99_time,
            current_stats.p99_query_time,
        );
        
        RegressionReport {
            has_regression: query_regression > threshold || index_regression > threshold ||
                           p95_query_regression > threshold || p99_query_regression > threshold,
            query_avg_regression: query_regression,
            index_avg_regression: index_regression,
            query_p95_regression: p95_query_regression,
            query_p99_regression: p99_query_regression,
            threshold,
            baseline_timestamp: baseline.timestamp,
            current_timestamp: std::time::SystemTime::now(),
        }
    }
    
    fn calculate_regression_factor(&self, baseline: Duration, current: Duration) -> f64 {
        if baseline.as_secs_f64() == 0.0 {
            return 0.0;
        }
        
        let baseline_ms = baseline.as_secs_f64() * 1000.0;
        let current_ms = current.as_secs_f64() * 1000.0;
        
        (current_ms - baseline_ms) / baseline_ms
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkComparison {
    pub monitor_query_avg: Duration,
    pub criterion_query_avg: Duration,
    pub monitor_index_avg: Duration,
    pub criterion_index_avg: Duration,
    pub monitor_samples: usize,
    pub criterion_samples: usize,
}

#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub timestamp: std::time::SystemTime,
    pub query_baseline: QueryBaseline,
    pub index_baseline: IndexBaseline,
}

#[derive(Debug, Clone)]
pub struct QueryBaseline {
    pub avg_time: Duration,
    pub p95_time: Duration,
    pub p99_time: Duration,
    pub std_dev: Duration,
    pub sample_count: usize,
}

#[derive(Debug, Clone)]
pub struct IndexBaseline {
    pub avg_time: Duration,
    pub p95_time: Duration,
    pub p99_time: Duration,
    pub std_dev: Duration,
    pub sample_count: usize,
}

#[derive(Debug, Clone)]
pub struct RegressionReport {
    pub has_regression: bool,
    pub query_avg_regression: f64,
    pub index_avg_regression: f64,
    pub query_p95_regression: f64,
    pub query_p99_regression: f64,
    pub threshold: f64,
    pub baseline_timestamp: std::time::SystemTime,
    pub current_timestamp: std::time::SystemTime,
}

impl RegressionReport {
    pub fn summary(&self) -> String {
        let mut summary = String::new();
        summary.push_str("=== REGRESSION ANALYSIS REPORT ===\n");
        summary.push_str(&format!("Baseline: {:?}\n", self.baseline_timestamp));
        summary.push_str(&format!("Current: {:?}\n", self.current_timestamp));
        summary.push_str(&format!("Threshold: {:.1}%\n", self.threshold * 100.0));
        summary.push_str(&format!("Has Regression: {}\n\n", self.has_regression));
        
        summary.push_str("REGRESSION FACTORS:\n");
        summary.push_str(&format!("  Query Average: {:+.1}%\n", self.query_avg_regression * 100.0));
        summary.push_str(&format!("  Index Average: {:+.1}%\n", self.index_avg_regression * 100.0));
        summary.push_str(&format!("  Query P95: {:+.1}%\n", self.query_p95_regression * 100.0));
        summary.push_str(&format!("  Query P99: {:+.1}%\n", self.query_p99_regression * 100.0));
        
        if self.has_regression {
            summary.push_str("\nWARNING: Performance regression detected!\n");
            
            if self.query_avg_regression > self.threshold {
                summary.push_str("  - Query average time has regressed significantly\n");
            }
            if self.index_avg_regression > self.threshold {
                summary.push_str("  - Index average time has regressed significantly\n");
            }
            if self.query_p95_regression > self.threshold {
                summary.push_str("  - Query P95 time has regressed significantly\n");
            }
            if self.query_p99_regression > self.threshold {
                summary.push_str("  - Query P99 time has regressed significantly\n");
            }
        } else {
            summary.push_str("\nGOOD: No significant performance regression detected.\n");
        }
        
        summary
    }
}

/// Create benchmark scenarios for testing
pub fn create_benchmark_scenarios() -> Vec<BenchmarkScenario> {
    vec![
        BenchmarkScenario {
            name: "small_queries".to_string(),
            description: "Small, fast query operations".to_string(),
            operation_type: OperationType::Query,
            sizes: vec![10, 50, 100],
            expected_max_time: Duration::from_millis(50),
        },
        BenchmarkScenario {
            name: "medium_queries".to_string(),
            description: "Medium complexity query operations".to_string(),
            operation_type: OperationType::Query,
            sizes: vec![500, 1000, 2000],
            expected_max_time: Duration::from_millis(200),
        },
        BenchmarkScenario {
            name: "large_queries".to_string(),
            description: "Large, complex query operations".to_string(),
            operation_type: OperationType::Query,
            sizes: vec![5000, 10000, 20000],
            expected_max_time: Duration::from_millis(1000),
        },
        BenchmarkScenario {
            name: "small_indexes".to_string(),
            description: "Small indexing operations".to_string(),
            operation_type: OperationType::Index,
            sizes: vec![10, 50, 100],
            expected_max_time: Duration::from_millis(200),
        },
        BenchmarkScenario {
            name: "large_indexes".to_string(),
            description: "Large indexing operations".to_string(),
            operation_type: OperationType::Index,
            sizes: vec![1000, 5000, 10000],
            expected_max_time: Duration::from_millis(2000),
        },
    ]
}

#[derive(Debug, Clone)]
pub struct BenchmarkScenario {
    pub name: String,
    pub description: String,
    pub operation_type: OperationType,
    pub sizes: Vec<usize>,
    pub expected_max_time: Duration,
}

#[derive(Debug, Clone)]
pub enum OperationType {
    Query,
    Index,
}

/// Simulate vector operations for benchmarking
pub fn simulate_query_operation(size: usize) -> Vec<f32> {
    // Simulate some computational work
    let mut result = Vec::with_capacity(size);
    for i in 0..size {
        let value = (i as f32 * 0.1).sin() + (i as f32 * 0.05).cos();
        result.push(value);
    }
    
    // Add some additional work to make it more realistic
    result.sort_by(|a, b| a.partial_cmp(b).unwrap());
    result
}

pub fn simulate_index_operation(size: usize) -> std::collections::HashMap<usize, Vec<f32>> {
    let mut index = std::collections::HashMap::new();
    
    for i in 0..size {
        let vector: Vec<f32> = (0..128)
            .map(|j| ((i + j) as f32 * 0.01).sin())
            .collect();
        index.insert(i, vector);
    }
    
    // Simulate some index processing
    for (_, vector) in index.iter_mut() {
        vector.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    
    index
}
```

### 3. Create benchmark suite
Create `benches/monitor_benchmarks.rs`:
```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, BatchSize};
use llmkg::monitor::{PerformanceMonitor, SharedPerformanceMonitor};
use llmkg::monitor::benchmark_utils::*;
use std::time::Duration;

fn bench_monitor_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("monitor_operations");
    
    // Benchmark basic recording operations
    group.bench_function("record_query_time", |b| {
        let mut monitor = PerformanceMonitor::new();
        let duration = Duration::from_millis(25);
        
        b.iter(|| {
            monitor.record_query_time(duration);
        });
    });
    
    group.bench_function("record_index_time", |b| {
        let mut monitor = PerformanceMonitor::new();
        let duration = Duration::from_millis(150);
        
        b.iter(|| {
            monitor.record_index_time(duration);
        });
    });
    
    // Benchmark statistics calculations
    for &sample_count in &[100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("get_stats", sample_count),
            &sample_count,
            |b, &count| {
                let mut monitor = PerformanceMonitor::new();
                
                // Pre-fill with data
                for i in 0..count {
                    monitor.record_query_time(Duration::from_millis(20 + (i % 50) as u64));
                    if i % 2 == 0 {
                        monitor.record_index_time(Duration::from_millis(100 + (i % 100) as u64));
                    }
                }
                
                b.iter(|| {
                    let _stats = monitor.get_stats();
                });
            },
        );
    }
    
    // Benchmark advanced statistics
    for &sample_count in &[100, 1000, 5000] {
        group.bench_with_input(
            BenchmarkId::new("get_advanced_stats", sample_count),
            &sample_count,
            |b, &count| {
                let mut monitor = PerformanceMonitor::new();
                
                // Pre-fill with varied data
                for i in 0..count {
                    let base_time = 30 + (i / 10) as u64; // Trending data
                    let noise = (i % 20) as u64; // Some variation
                    monitor.record_query_time(Duration::from_millis(base_time + noise));
                }
                
                b.iter(|| {
                    let _advanced_stats = monitor.get_advanced_stats(0.95);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_operations");
    
    for &thread_count in &[1, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("shared_monitor_recording", thread_count),
            &thread_count,
            |b, &threads| {
                b.iter_batched(
                    || SharedPerformanceMonitor::new(),
                    |shared_monitor| {
                        let handles: Vec<_> = (0..threads).map(|_| {
                            let monitor_clone = shared_monitor.clone();
                            std::thread::spawn(move || {
                                for i in 0..100 {
                                    monitor_clone.record_query_time(Duration::from_millis(20 + (i % 10)));
                                }
                            })
                        }).collect();
                        
                        for handle in handles {
                            handle.join().unwrap();
                        }
                        
                        shared_monitor.get_stats()
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

fn bench_real_workload_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("workload_simulation");
    
    let scenarios = create_benchmark_scenarios();
    
    for scenario in scenarios {
        for &size in &scenario.sizes {
            let benchmark_name = format!("{}_{}", scenario.name, size);
            
            group.bench_with_input(
                BenchmarkId::new(&benchmark_name, size),
                &size,
                |b, &size| {
                    let mut bench_monitor = BenchmarkMonitor::new();
                    
                    b.iter(|| {
                        match scenario.operation_type {
                            OperationType::Query => {
                                bench_monitor.bench_query(|| {
                                    simulate_query_operation(size)
                                })
                            }
                            OperationType::Index => {
                                bench_monitor.bench_index(|| {
                                    simulate_index_operation(size)
                                })
                            }
                        }
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_report_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("report_generation");
    
    for &sample_count in &[100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("generate_report", sample_count),
            &sample_count,
            |b, &count| {
                let mut monitor = PerformanceMonitor::new();
                
                // Pre-fill with realistic data
                for i in 0..count {
                    monitor.record_query_time(Duration::from_millis(25 + (i % 100) as u64));
                    if i % 3 == 0 {
                        monitor.record_index_time(Duration::from_millis(150 + (i % 200) as u64));
                    }
                }
                
                let config = llmkg::monitor::ReportConfig::default();
                
                b.iter(|| {
                    let _report = monitor.generate_report(&config);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    // Test different capacity limits
    for &capacity in &[100, 1000, 10000] {
        group.bench_with_input(
            BenchmarkId::new("bounded_recording", capacity),
            &capacity,
            |b, &cap| {
                b.iter_batched(
                    || PerformanceMonitor::with_capacity(cap),
                    |mut monitor| {
                        // Record more operations than capacity to test memory bounds
                        for i in 0..(cap * 3) {
                            monitor.record_query_time(Duration::from_millis(20 + (i % 50) as u64));
                        }
                        monitor.get_stats()
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

fn bench_regression_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_detection");
    
    group.bench_function("baseline_generation", |b| {
        let mut bench_monitor = BenchmarkMonitor::new();
        
        // Fill with data
        for i in 0..1000 {
            bench_monitor.bench_query(|| simulate_query_operation(100 + i % 50));
        }
        
        b.iter(|| {
            let _baseline = bench_monitor.generate_baseline();
        });
    });
    
    group.bench_function("regression_detection", |b| {
        let mut bench_monitor = BenchmarkMonitor::new();
        
        // Generate baseline
        for i in 0..1000 {
            bench_monitor.bench_query(|| simulate_query_operation(100));
        }
        let baseline = bench_monitor.generate_baseline();
        
        // Generate new data (slightly slower)
        let mut test_monitor = BenchmarkMonitor::new();
        for i in 0..1000 {
            test_monitor.bench_query(|| simulate_query_operation(120)); // 20% larger
        }
        
        b.iter(|| {
            let _regression = test_monitor.detect_regression(&baseline, 0.1);
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_monitor_operations,
    bench_concurrent_operations,
    bench_real_workload_simulation,
    bench_report_generation,
    bench_memory_efficiency,
    bench_regression_detection
);
criterion_main!(benches);
```

### 4. Add benchmarking integration tests
Add to `src/monitor.rs` in the tests module:
```rust
#[cfg(test)]
mod benchmark_integration_tests {
    use super::*;
    use super::benchmark_utils::*;
    
    #[test]
    fn test_benchmark_monitor_basic_functionality() {
        let mut bench_monitor = BenchmarkMonitor::new();
        
        // Test query benchmarking
        let result = bench_monitor.bench_query(|| {
            std::thread::sleep(Duration::from_millis(10));
            "query_result"
        });
        
        assert_eq!(result, "query_result");
        
        // Test index benchmarking
        let result = bench_monitor.bench_index(|| {
            std::thread::sleep(Duration::from_millis(20));
            42
        });
        
        assert_eq!(result, 42);
        
        // Verify recordings
        let comparison = bench_monitor.compare_with_criterion();
        assert!(comparison.monitor_query_avg >= Duration::from_millis(10));
        assert!(comparison.monitor_index_avg >= Duration::from_millis(20));
        assert_eq!(comparison.monitor_samples, 2);
    }
    
    #[test]
    fn test_baseline_generation() {
        let mut bench_monitor = BenchmarkMonitor::new();
        
        // Record some known data
        for i in 1..=10 {
            bench_monitor.bench_query(|| simulate_query_operation(50));
            bench_monitor.bench_index(|| simulate_index_operation(25));
        }
        
        let baseline = bench_monitor.generate_baseline();
        
        assert_eq!(baseline.query_baseline.sample_count, 10);
        assert_eq!(baseline.index_baseline.sample_count, 10);
        assert!(baseline.query_baseline.avg_time > Duration::from_millis(0));
        assert!(baseline.index_baseline.avg_time > Duration::from_millis(0));
    }
    
    #[test]
    fn test_regression_detection() {
        // Create baseline with fast operations
        let mut baseline_monitor = BenchmarkMonitor::new();
        for _ in 0..50 {
            baseline_monitor.bench_query(|| simulate_query_operation(50)); // Small operations
        }
        let baseline = baseline_monitor.generate_baseline();
        
        // Create test data with slower operations
        let mut test_monitor = BenchmarkMonitor::new();
        for _ in 0..50 {
            test_monitor.bench_query(|| simulate_query_operation(100)); // Larger operations
        }
        
        // Should detect regression with 10% threshold
        let regression = test_monitor.detect_regression(&baseline, 0.1);
        assert!(regression.has_regression);
        assert!(regression.query_avg_regression > 0.1);
        
        // Should not detect regression with very high threshold
        let no_regression = test_monitor.detect_regression(&baseline, 5.0);
        assert!(!no_regression.has_regression);
    }
    
    #[test]
    fn test_benchmark_scenarios() {
        let scenarios = create_benchmark_scenarios();
        assert!(!scenarios.is_empty());
        
        for scenario in scenarios {
            let mut bench_monitor = BenchmarkMonitor::new();
            
            // Test each size in the scenario
            for &size in &scenario.sizes {
                let start = std::time::Instant::now();
                
                match scenario.operation_type {
                    OperationType::Query => {
                        bench_monitor.bench_query(|| simulate_query_operation(size));
                    }
                    OperationType::Index => {
                        bench_monitor.bench_index(|| simulate_index_operation(size));
                    }
                }
                
                let elapsed = start.elapsed();
                
                // Should complete within expected time bounds
                assert!(elapsed <= scenario.expected_max_time * 2, 
                    "Scenario '{}' with size {} took too long: {:?} (expected max: {:?})", 
                    scenario.name, size, elapsed, scenario.expected_max_time);
            }
        }
    }
    
    #[test]
    fn test_regression_report_formatting() {
        let mut bench_monitor = BenchmarkMonitor::new();
        
        // Create some data
        for _ in 0..20 {
            bench_monitor.bench_query(|| simulate_query_operation(100));
        }
        
        let baseline = bench_monitor.generate_baseline();
        
        // Create slightly different data
        let mut test_monitor = BenchmarkMonitor::new();
        for _ in 0..20 {
            test_monitor.bench_query(|| simulate_query_operation(110));
        }
        
        let regression = test_monitor.detect_regression(&baseline, 0.05);
        let summary = regression.summary();
        
        assert!(summary.contains("REGRESSION ANALYSIS REPORT"));
        assert!(summary.contains("Threshold:"));
        assert!(summary.contains("Query Average:"));
        assert!(summary.contains("REGRESSION FACTORS"));
        
        if regression.has_regression {
            assert!(summary.contains("WARNING: Performance regression detected!"));
        } else {
            assert!(summary.contains("GOOD: No significant performance regression detected."));
        }
    }
    
    #[test]
    fn test_criterion_result_recording() {
        let mut bench_monitor = BenchmarkMonitor::new();
        
        // Simulate recording Criterion results
        bench_monitor.record_criterion_result("test_query", Duration::from_millis(25), true);
        bench_monitor.record_criterion_result("test_index", Duration::from_millis(150), false);
        
        let comparison = bench_monitor.compare_with_criterion();
        assert_eq!(comparison.criterion_query_avg, Duration::from_millis(25));
        assert_eq!(comparison.criterion_index_avg, Duration::from_millis(150));
        assert_eq!(comparison.criterion_samples, 2);
    }
    
    #[test]
    fn test_performance_consistency() {
        let mut bench_monitor = BenchmarkMonitor::new();
        let iterations = 100;
        
        // Run same operation multiple times
        for _ in 0..iterations {
            bench_monitor.bench_query(|| simulate_query_operation(50));
        }
        
        let baseline = bench_monitor.generate_baseline();
        
        // Coefficient of variation should be reasonable (< 50%)
        let cv = baseline.query_baseline.std_dev.as_secs_f64() / baseline.query_baseline.avg_time.as_secs_f64();
        assert!(cv < 0.5, "Coefficient of variation too high: {:.2}", cv);
        
        // P95 should not be too much higher than average
        let p95_ratio = baseline.query_baseline.p95_time.as_secs_f64() / baseline.query_baseline.avg_time.as_secs_f64();
        assert!(p95_ratio < 3.0, "P95/Average ratio too high: {:.2}", p95_ratio);
    }
}
```

### 5. Add integration with existing monitor
Add to the main `PerformanceMonitor` implementation:
```rust
impl PerformanceMonitor {
    /// Create a benchmark-enabled monitor
    pub fn for_benchmarking() -> Self {
        Self::with_capacity(50000) // Larger capacity for benchmark data
    }
    
    /// Export performance data for external analysis
    pub fn export_benchmark_data(&self) -> BenchmarkData {
        let stats = self.get_stats();
        let advanced_stats = self.get_advanced_stats(0.95);
        
        BenchmarkData {
            timestamp: std::time::SystemTime::now(),
            query_times: self.query_times.iter().cloned().collect(),
            index_times: self.index_times.iter().cloned().collect(),
            basic_stats: stats,
            advanced_stats,
        }
    }
    
    /// Compare current performance with historical data
    pub fn compare_with_historical(&self, historical: &BenchmarkData, threshold: f64) -> ComparisonResult {
        let current_stats = self.get_stats();
        
        let query_regression = if historical.basic_stats.avg_query_time.as_secs_f64() > 0.0 {
            (current_stats.avg_query_time.as_secs_f64() - historical.basic_stats.avg_query_time.as_secs_f64()) 
                / historical.basic_stats.avg_query_time.as_secs_f64()
        } else {
            0.0
        };
        
        let index_regression = if historical.basic_stats.avg_index_time.as_secs_f64() > 0.0 {
            (current_stats.avg_index_time.as_secs_f64() - historical.basic_stats.avg_index_time.as_secs_f64()) 
                / historical.basic_stats.avg_index_time.as_secs_f64()
        } else {
            0.0
        };
        
        ComparisonResult {
            has_regression: query_regression > threshold || index_regression > threshold,
            query_regression,
            index_regression,
            threshold,
            sample_counts_changed: current_stats.total_queries != historical.basic_stats.total_queries,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkData {
    pub timestamp: std::time::SystemTime,
    pub query_times: Vec<Duration>,
    pub index_times: Vec<Duration>,
    pub basic_stats: PerformanceStats,
    pub advanced_stats: AdvancedPerformanceStats,
}

#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub has_regression: bool,
    pub query_regression: f64,
    pub index_regression: f64,
    pub threshold: f64,
    pub sample_counts_changed: bool,
}
```

## Success Criteria
- [ ] Criterion benchmarking integration works correctly
- [ ] Benchmark utilities provide comprehensive performance testing
- [ ] Regression detection accurately identifies performance changes
- [ ] Baseline generation captures performance characteristics
- [ ] Benchmark scenarios cover realistic workloads
- [ ] Integration tests validate benchmarking functionality
- [ ] Performance comparisons are accurate and meaningful
- [ ] HTML reports are generated with benchmark results
- [ ] All benchmark tests pass consistently
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Integrates with Criterion for industry-standard benchmarking
- Provides regression detection with configurable thresholds
- Supports baseline establishment for continuous monitoring
- Includes realistic workload simulation for testing
- Generates comprehensive benchmark reports
- Enables automated performance validation in CI/CD pipelines
- Supports concurrent benchmarking for scalability testing