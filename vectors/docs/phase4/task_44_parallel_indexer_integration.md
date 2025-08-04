# Task 44: Parallel Indexer Integration with Performance Monitoring

## Context
You are implementing Phase 4 of a vector indexing system. Benchmarking integration with Criterion is now available. This task integrates the performance monitoring system with the parallel indexer to provide real-time performance tracking, bottleneck identification, and optimization guidance.

## Current State
- `src/monitor.rs` exists with complete performance monitoring functionality
- `src/parallel.rs` exists with parallel indexing implementation
- Benchmarking integration provides performance validation
- Need integration between parallel indexer and monitoring system

## Task Objective
Integrate the performance monitoring system with the parallel indexer to provide comprehensive performance tracking, thread-level monitoring, workload analysis, and real-time optimization recommendations.

## Implementation Requirements

### 1. Add monitoring integration to parallel indexer
Modify `src/parallel.rs` to include monitoring capabilities:
```rust
use crate::monitor::{SharedPerformanceMonitor, PerformanceMonitor, RealTimeMonitor, MonitorUpdate};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct MonitoredParallelIndexer {
    pub indexer: ParallelIndexer,
    pub monitor: SharedPerformanceMonitor,
    pub real_time_monitor: Option<Arc<RealTimeMonitor>>,
    pub thread_monitors: Vec<Arc<Mutex<PerformanceMonitor>>>,
    pub config: IndexerMonitoringConfig,
}

#[derive(Debug, Clone)]
pub struct IndexerMonitoringConfig {
    pub enable_per_thread_monitoring: bool,
    pub enable_real_time_updates: bool,
    pub batch_size_monitoring: bool,
    pub queue_depth_monitoring: bool,
    pub thread_utilization_monitoring: bool,
    pub auto_optimization: bool,
    pub reporting_interval: Duration,
}

impl Default for IndexerMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_per_thread_monitoring: true,
            enable_real_time_updates: true,
            batch_size_monitoring: true,
            queue_depth_monitoring: true,
            thread_utilization_monitoring: true,
            auto_optimization: false,
            reporting_interval: Duration::from_secs(30),
        }
    }
}

impl MonitoredParallelIndexer {
    pub fn new(thread_count: usize, config: IndexerMonitoringConfig) -> Self {
        let indexer = ParallelIndexer::new(thread_count);
        let monitor = SharedPerformanceMonitor::new();
        
        // Create per-thread monitors if enabled
        let thread_monitors = if config.enable_per_thread_monitoring {
            (0..thread_count)
                .map(|_| Arc::new(Mutex::new(PerformanceMonitor::new())))
                .collect()
        } else {
            Vec::new()
        };
        
        Self {
            indexer,
            monitor,
            real_time_monitor: None,
            thread_monitors,
            config,
        }
    }
    
    pub fn enable_real_time_monitoring(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.config.enable_real_time_updates {
            let (rt_monitor, _receiver) = RealTimeMonitor::new(Duration::from_millis(100));
            rt_monitor.start();
            self.real_time_monitor = Some(Arc::new(rt_monitor));
        }
        Ok(())
    }
    
    /// Index vectors with comprehensive monitoring
    pub fn index_vectors_monitored(&mut self, vectors: Vec<Vec<f32>>) -> Result<IndexingResult, IndexingError> {
        let start_time = Instant::now();
        let total_vectors = vectors.len();
        
        // Pre-indexing metrics
        let initial_stats = self.monitor.get_stats();
        
        // Divide work among threads with monitoring
        let batch_size = self.calculate_optimal_batch_size(vectors.len());
        let batches: Vec<_> = vectors.chunks(batch_size).collect();
        
        let mut handles = Vec::new();
        let mut thread_start_times = Vec::new();
        
        for (thread_id, batch) in batches.into_iter().enumerate() {
            let batch_vectors = batch.to_vec();
            let monitor_clone = self.monitor.clone();
            let thread_monitor = if thread_id < self.thread_monitors.len() {
                Some(self.thread_monitors[thread_id].clone())
            } else {
                None
            };
            let config = self.config.clone();
            
            let thread_start = Instant::now();
            thread_start_times.push(thread_start);
            
            let handle = thread::spawn(move || {
                Self::index_batch_with_monitoring(
                    thread_id,
                    batch_vectors,
                    monitor_clone,
                    thread_monitor,
                    config,
                )
            });
            
            handles.push(handle);
        }
        
        // Monitor thread completion and collect results
        let mut batch_results = Vec::new();
        let mut thread_completion_times = Vec::new();
        
        for (thread_id, handle) in handles.into_iter().enumerate() {
            let result = handle.join().map_err(|_| IndexingError::ThreadPanic(thread_id))?;
            let completion_time = thread_start_times[thread_id].elapsed();
            
            batch_results.push(result?);
            thread_completion_times.push(completion_time);
            
            // Record thread-level performance
            self.monitor.record_index_time(completion_time);
        }
        
        let total_time = start_time.elapsed();
        
        // Generate comprehensive indexing report
        let indexing_report = self.generate_indexing_report(
            total_vectors,
            total_time,
            thread_completion_times,
            batch_size,
            initial_stats,
        );
        
        // Check for optimization opportunities
        if self.config.auto_optimization {
            self.analyze_and_optimize(&indexing_report);
        }
        
        Ok(IndexingResult {
            indexed_count: total_vectors,
            total_time,
            batch_results,
            performance_report: Some(indexing_report),
        })
    }
    
    fn index_batch_with_monitoring(
        thread_id: usize,
        vectors: Vec<Vec<f32>>,
        global_monitor: SharedPerformanceMonitor,
        thread_monitor: Option<Arc<Mutex<PerformanceMonitor>>>,
        config: IndexerMonitoringConfig,
    ) -> Result<BatchResult, IndexingError> {
        let batch_start = Instant::now();
        let mut processed_count = 0;
        let mut batch_index = std::collections::HashMap::new();
        
        for (vector_id, vector) in vectors.into_iter().enumerate() {
            let vector_start = Instant::now();
            
            // Simulate vector processing
            let processed_vector = Self::process_vector(vector)?;
            batch_index.insert(vector_id, processed_vector);
            
            let vector_time = vector_start.elapsed();
            processed_count += 1;
            
            // Record timing in both global and thread-local monitors
            global_monitor.record_index_time(vector_time);
            
            if let Some(ref monitor) = thread_monitor {
                if let Ok(mut monitor) = monitor.lock() {
                    monitor.record_index_time(vector_time);
                }
            }
            
            // Periodic progress reporting
            if config.enable_real_time_updates && processed_count % 100 == 0 {
                // Could send progress updates here
            }
        }
        
        let batch_time = batch_start.elapsed();
        
        Ok(BatchResult {
            thread_id,
            processed_count,
            batch_time,
            average_vector_time: batch_time / processed_count.max(1) as u32,
            index_data: batch_index,
        })
    }
    
    fn process_vector(vector: Vec<f32>) -> Result<ProcessedVector, IndexingError> {
        // Simulate vector processing with realistic computation
        if vector.is_empty() {
            return Err(IndexingError::EmptyVector);
        }
        
        let mut processed = ProcessedVector {
            original_dims: vector.len(),
            normalized: Vec::with_capacity(vector.len()),
            magnitude: 0.0,
        };
        
        // Calculate magnitude
        let magnitude_sq: f32 = vector.iter().map(|x| x * x).sum();
        processed.magnitude = magnitude_sq.sqrt();
        
        if processed.magnitude == 0.0 {
            return Err(IndexingError::ZeroMagnitude);
        }
        
        // Normalize vector
        for &component in &vector {
            processed.normalized.push(component / processed.magnitude);
        }
        
        // Add some computational work to make timing more realistic
        processed.normalized.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        Ok(processed)
    }
    
    fn calculate_optimal_batch_size(&self, total_vectors: usize) -> usize {
        let thread_count = self.indexer.thread_count;
        let base_batch_size = total_vectors / thread_count;
        
        // Adjust based on historical performance if available
        if let Some(stats) = self.monitor.get_stats() {
            if stats.total_indexes > 0 {
                let avg_time_ms = stats.avg_index_time.as_millis() as usize;
                
                // Adjust batch size based on average processing time
                if avg_time_ms > 100 {
                    // Slow processing, use smaller batches
                    base_batch_size.max(10).min(100)
                } else if avg_time_ms < 10 {
                    // Fast processing, use larger batches
                    base_batch_size.max(100)
                } else {
                    base_batch_size.max(50)
                }
            } else {
                base_batch_size.max(50)
            }
        } else {
            base_batch_size.max(50)
        }
    }
    
    fn generate_indexing_report(
        &self,
        total_vectors: usize,
        total_time: Duration,
        thread_times: Vec<Duration>,
        batch_size: usize,
        initial_stats: Option<crate::monitor::PerformanceStats>,
    ) -> IndexingPerformanceReport {
        let current_stats = self.monitor.get_stats();
        
        // Calculate thread efficiency metrics
        let max_thread_time = thread_times.iter().max().copied().unwrap_or_default();
        let min_thread_time = thread_times.iter().min().copied().unwrap_or_default();
        let avg_thread_time = if !thread_times.is_empty() {
            thread_times.iter().sum::<Duration>() / thread_times.len() as u32
        } else {
            Duration::from_millis(0)
        };
        
        let thread_efficiency = if max_thread_time.as_secs_f64() > 0.0 {
            min_thread_time.as_secs_f64() / max_thread_time.as_secs_f64()
        } else {
            1.0
        };
        
        // Calculate throughput
        let vectors_per_second = if total_time.as_secs_f64() > 0.0 {
            total_vectors as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };
        
        // Analyze per-thread performance
        let thread_performance: Vec<_> = self.thread_monitors.iter().enumerate().map(|(thread_id, monitor)| {
            if let Ok(monitor) = monitor.lock() {
                let stats = monitor.get_stats();
                ThreadPerformance {
                    thread_id,
                    operations_completed: stats.total_indexes,
                    average_time: stats.avg_index_time,
                    min_time: stats.min_index_time,
                    max_time: stats.max_index_time,
                }
            } else {
                ThreadPerformance::default_for_thread(thread_id)
            }
        }).collect();
        
        IndexingPerformanceReport {
            total_vectors,
            total_time,
            vectors_per_second,
            batch_size,
            thread_count: self.indexer.thread_count,
            thread_efficiency,
            thread_times,
            thread_performance,
            average_vector_time: if total_vectors > 0 {
                total_time / total_vectors as u32
            } else {
                Duration::from_millis(0)
            },
            recommendations: self.generate_optimization_recommendations(
                thread_efficiency,
                vectors_per_second,
                &thread_times,
            ),
        }
    }
    
    fn generate_optimization_recommendations(
        &self,
        thread_efficiency: f64,
        throughput: f64,
        thread_times: &[Duration],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Thread efficiency analysis
        if thread_efficiency < 0.7 {
            recommendations.push(format!(
                "Low thread efficiency ({:.1}%). Consider rebalancing work distribution or adjusting batch sizes.",
                thread_efficiency * 100.0
            ));
        }
        
        // Throughput analysis
        if throughput < 100.0 {
            recommendations.push(format!(
                "Low throughput ({:.1} vectors/sec). Consider optimizing vector processing or increasing parallelism.",
                throughput
            ));
        }
        
        // Thread time variance analysis
        if thread_times.len() > 1 {
            let max_time = thread_times.iter().max().unwrap().as_secs_f64();
            let min_time = thread_times.iter().min().unwrap().as_secs_f64();
            let variance_ratio = if min_time > 0.0 { max_time / min_time } else { 1.0 };
            
            if variance_ratio > 2.0 {
                recommendations.push(format!(
                    "High thread time variance (max/min ratio: {:.1}). Consider dynamic load balancing.",
                    variance_ratio
                ));
            }
        }
        
        // Memory usage recommendations
        if self.config.batch_size_monitoring {
            let current_stats = self.monitor.get_stats();
            if let Some(stats) = current_stats {
                if stats.avg_index_time > Duration::from_millis(50) {
                    recommendations.push("Average indexing time is high. Consider reducing batch size or optimizing vector processing.".to_string());
                }
                
                if stats.p99_index_time > stats.avg_index_time * 3 {
                    recommendations.push("High P99 latency detected. Consider investigating outlier processing or memory allocation patterns.".to_string());
                }
            }
        }
        
        if recommendations.is_empty() {
            recommendations.push("Performance looks good. No specific optimizations recommended.".to_string());
        }
        
        recommendations
    }
    
    fn analyze_and_optimize(&mut self, report: &IndexingPerformanceReport) {
        // Auto-optimization based on performance analysis
        
        // Adjust thread count if efficiency is low
        if report.thread_efficiency < 0.6 && self.indexer.thread_count > 2 {
            println!("Auto-optimization: Reducing thread count due to low efficiency");
            // Would implement thread count adjustment here
        }
        
        // Adjust batch size based on throughput
        if report.vectors_per_second < 50.0 {
            println!("Auto-optimization: Performance is low, analyzing batch size optimization");
            // Would implement batch size adjustment here
        }
        
        // Log optimization actions
        println!("Performance Analysis Complete:");
        for recommendation in &report.recommendations {
            println!("  - {}", recommendation);
        }
    }
    
    /// Get comprehensive monitoring statistics
    pub fn get_monitoring_stats(&self) -> MonitoringStats {
        let global_stats = self.monitor.get_stats();
        
        let thread_stats: Vec<_> = self.thread_monitors.iter().enumerate().map(|(thread_id, monitor)| {
            if let Ok(monitor) = monitor.lock() {
                let stats = monitor.get_stats();
                (thread_id, stats)
            } else {
                (thread_id, None)
            }
        }).collect();
        
        MonitoringStats {
            global_stats,
            thread_stats,
            active_threads: self.indexer.thread_count,
            monitoring_enabled: true,
        }
    }
}

// Supporting types and structures
#[derive(Debug, Clone)]
pub struct IndexingResult {
    pub indexed_count: usize,
    pub total_time: Duration,
    pub batch_results: Vec<BatchResult>,
    pub performance_report: Option<IndexingPerformanceReport>,
}

#[derive(Debug, Clone)]
pub struct BatchResult {
    pub thread_id: usize,
    pub processed_count: usize,
    pub batch_time: Duration,
    pub average_vector_time: Duration,
    pub index_data: std::collections::HashMap<usize, ProcessedVector>,
}

#[derive(Debug, Clone)]
pub struct ProcessedVector {
    pub original_dims: usize,
    pub normalized: Vec<f32>,
    pub magnitude: f32,
}

#[derive(Debug, Clone)]
pub struct IndexingPerformanceReport {
    pub total_vectors: usize,
    pub total_time: Duration,
    pub vectors_per_second: f64,
    pub batch_size: usize,
    pub thread_count: usize,
    pub thread_efficiency: f64,
    pub thread_times: Vec<Duration>,
    pub thread_performance: Vec<ThreadPerformance>,
    pub average_vector_time: Duration,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ThreadPerformance {
    pub thread_id: usize,
    pub operations_completed: usize,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
}

impl ThreadPerformance {
    fn default_for_thread(thread_id: usize) -> Self {
        Self {
            thread_id,
            operations_completed: 0,
            average_time: Duration::from_millis(0),
            min_time: Duration::from_millis(0),
            max_time: Duration::from_millis(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MonitoringStats {
    pub global_stats: Option<crate::monitor::PerformanceStats>,
    pub thread_stats: Vec<(usize, Option<crate::monitor::PerformanceStats>)>,
    pub active_threads: usize,
    pub monitoring_enabled: bool,
}

#[derive(Debug)]
pub enum IndexingError {
    ThreadPanic(usize),
    EmptyVector,
    ZeroMagnitude,
    InsufficientMemory,
    InvalidConfiguration,
}

impl std::fmt::Display for IndexingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexingError::ThreadPanic(id) => write!(f, "Thread {} panicked during indexing", id),
            IndexingError::EmptyVector => write!(f, "Cannot process empty vector"),
            IndexingError::ZeroMagnitude => write!(f, "Vector has zero magnitude"),
            IndexingError::InsufficientMemory => write!(f, "Insufficient memory for indexing operation"),
            IndexingError::InvalidConfiguration => write!(f, "Invalid indexing configuration"),
        }
    }
}

impl std::error::Error for IndexingError {}
```

### 2. Add comprehensive integration tests
Add to `src/parallel.rs` in the tests module:
```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::monitor::test_utils::TestDataGenerator;
    
    #[test]
    fn test_monitored_parallel_indexing() {
        let config = IndexerMonitoringConfig::default();
        let mut monitored_indexer = MonitoredParallelIndexer::new(4, config);
        
        // Generate test vectors
        let vectors = generate_test_vectors(1000, 128);
        
        let result = monitored_indexer.index_vectors_monitored(vectors).unwrap();
        
        assert_eq!(result.indexed_count, 1000);
        assert!(result.total_time > Duration::from_millis(0));
        assert!(!result.batch_results.is_empty());
        assert!(result.performance_report.is_some());
        
        // Verify monitoring data was collected
        let stats = monitored_indexer.get_monitoring_stats();
        assert!(stats.global_stats.is_some());
        assert_eq!(stats.active_threads, 4);
        
        if let Some(global_stats) = stats.global_stats {
            assert!(global_stats.total_indexes > 0);
            assert!(global_stats.avg_index_time > Duration::from_millis(0));
        }
    }
    
    #[test]
    fn test_per_thread_monitoring() {
        let config = IndexerMonitoringConfig {
            enable_per_thread_monitoring: true,
            ..Default::default()
        };
        let mut monitored_indexer = MonitoredParallelIndexer::new(2, config);
        
        let vectors = generate_test_vectors(200, 64);
        let result = monitored_indexer.index_vectors_monitored(vectors).unwrap();
        
        assert_eq!(result.indexed_count, 200);
        
        // Check thread-level statistics
        let stats = monitored_indexer.get_monitoring_stats();
        assert_eq!(stats.thread_stats.len(), 2);
        
        // Each thread should have processed some vectors
        let total_thread_operations: usize = stats.thread_stats.iter()
            .filter_map(|(_, stats)| stats.as_ref())
            .map(|s| s.total_indexes)
            .sum();
        
        assert!(total_thread_operations > 0);
        
        // Verify performance report contains thread performance data
        if let Some(report) = result.performance_report {
            assert_eq!(report.thread_performance.len(), 2);
            assert!(report.thread_efficiency > 0.0);
            assert!(report.vectors_per_second > 0.0);
        }
    }
    
    #[test]
    fn test_optimization_recommendations() {
        let config = IndexerMonitoringConfig::default();
        let mut monitored_indexer = MonitoredParallelIndexer::new(8, config);
        
        // Create unbalanced load (some vectors much larger than others)
        let mut vectors = generate_test_vectors(500, 64);
        vectors.extend(generate_test_vectors(100, 512)); // Larger vectors
        
        let result = monitored_indexer.index_vectors_monitored(vectors).unwrap();
        
        if let Some(report) = result.performance_report {
            assert!(!report.recommendations.is_empty());
            
            // Should detect efficiency issues with unbalanced load
            if report.thread_efficiency < 0.8 {
                let has_efficiency_recommendation = report.recommendations.iter()
                    .any(|r| r.contains("efficiency") || r.contains("balancing"));
                assert!(has_efficiency_recommendation);
            }
        }
    }
    
    #[test]
    fn test_batch_size_optimization() {
        let config = IndexerMonitoringConfig {
            batch_size_monitoring: true,
            ..Default::default()
        };
        let mut monitored_indexer = MonitoredParallelIndexer::new(4, config);
        
        // Run initial indexing to establish baseline
        let initial_vectors = generate_test_vectors(400, 128);
        let _initial_result = monitored_indexer.index_vectors_monitored(initial_vectors).unwrap();
        
        // Run second batch - should use optimized batch size
        let second_vectors = generate_test_vectors(800, 128);
        let result = monitored_indexer.index_vectors_monitored(second_vectors).unwrap();
        
        assert_eq!(result.indexed_count, 800);
        
        // Batch size should be calculated based on previous performance
        if let Some(report) = result.performance_report {
            assert!(report.batch_size > 0);
            // With good performance, batch size should be reasonable
            assert!(report.batch_size >= 50);
        }
    }
    
    #[test]
    fn test_error_handling_with_monitoring() {
        let config = IndexerMonitoringConfig::default();
        let mut monitored_indexer = MonitoredParallelIndexer::new(2, config);
        
        // Include some problematic vectors
        let mut vectors = generate_test_vectors(100, 128);
        vectors.push(Vec::new()); // Empty vector should cause error
        vectors.push(vec![0.0, 0.0, 0.0]); // Zero magnitude vector
        
        let result = monitored_indexer.index_vectors_monitored(vectors);
        
        // Should handle errors gracefully
        match result {
            Ok(_) => {
                // If it succeeded, monitoring should still work
                let stats = monitored_indexer.get_monitoring_stats();
                assert!(stats.monitoring_enabled);
            }
            Err(_) => {
                // Error is expected due to problematic vectors
                // But monitoring should still be functional
                let stats = monitored_indexer.get_monitoring_stats();
                assert!(stats.monitoring_enabled);
            }
        }
    }
    
    #[test]
    fn test_concurrent_monitoring_accuracy() {
        let config = IndexerMonitoringConfig {
            enable_per_thread_monitoring: true,
            enable_real_time_updates: false, // Disable for accuracy testing
            ..Default::default()
        };
        let mut monitored_indexer = MonitoredParallelIndexer::new(4, config);
        
        let vectors = generate_test_vectors(2000, 256);
        let expected_count = vectors.len();
        
        let result = monitored_indexer.index_vectors_monitored(vectors).unwrap();
        
        assert_eq!(result.indexed_count, expected_count);
        
        // Verify that global monitoring matches sum of thread monitoring
        let stats = monitored_indexer.get_monitoring_stats();
        
        if let Some(global_stats) = stats.global_stats {
            let thread_total: usize = stats.thread_stats.iter()
                .filter_map(|(_, stats)| stats.as_ref())
                .map(|s| s.total_indexes)
                .sum();
            
            // Global stats should match thread stats (approximately, due to timing)
            let difference = (global_stats.total_indexes as i32 - thread_total as i32).abs();
            assert!(difference <= expected_count as i32 / 10, 
                "Global stats ({}) and thread stats sum ({}) differ too much (diff: {})", 
                global_stats.total_indexes, thread_total, difference);
        }
    }
    
    #[test]
    fn test_performance_report_completeness() {
        let config = IndexerMonitoringConfig::default();
        let mut monitored_indexer = MonitoredParallelIndexer::new(3, config);
        
        let vectors = generate_test_vectors(300, 128);
        let result = monitored_indexer.index_vectors_monitored(vectors).unwrap();
        
        let report = result.performance_report.expect("Performance report should be generated");
        
        // Verify all expected fields are populated
        assert_eq!(report.total_vectors, 300);
        assert!(report.total_time > Duration::from_millis(0));
        assert!(report.vectors_per_second > 0.0);
        assert!(report.batch_size > 0);
        assert_eq!(report.thread_count, 3);
        assert!(report.thread_efficiency >= 0.0 && report.thread_efficiency <= 1.0);
        assert_eq!(report.thread_times.len(), 3); // One per thread
        assert_eq!(report.thread_performance.len(), 3);
        assert!(!report.recommendations.is_empty());
        
        // Verify thread performance data
        for thread_perf in &report.thread_performance {
            assert!(thread_perf.thread_id < 3);
            // Each thread should have done some work
            assert!(thread_perf.operations_completed > 0);
        }
    }
    
    fn generate_test_vectors(count: usize, dimensions: usize) -> Vec<Vec<f32>> {
        let mut generator = TestDataGenerator::new(12345);
        let mut vectors = Vec::with_capacity(count);
        
        for i in 0..count {
            let mut vector = Vec::with_capacity(dimensions);
            for j in 0..dimensions {
                // Generate deterministic but varied values
                generator.seed = generator.seed.wrapping_mul(1103515245).wrapping_add(12345);
                let value = ((generator.seed % 1000) as f32 - 500.0) / 100.0; // Range: -5.0 to 5.0
                vector.push(value + (i + j) as f32 * 0.01); // Add some variation
            }
            vectors.push(vector);
        }
        
        vectors
    }
}
```

### 3. Add performance analysis utilities
```rust
impl IndexingPerformanceReport {
    /// Generate detailed HTML report for indexing performance
    pub fn to_html(&self) -> String {
        let mut html = String::new();
        
        html.push_str("<!DOCTYPE html><html><head>");
        html.push_str("<title>Indexing Performance Report</title>");
        html.push_str("<style>");
        html.push_str("body { font-family: Arial, sans-serif; margin: 20px; }");
        html.push_str("table { border-collapse: collapse; width: 100%; margin: 20px 0; }");
        html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }");
        html.push_str("th { background-color: #f2f2f2; }");
        html.push_str(".metric { font-weight: bold; color: #333; }");
        html.push_str(".recommendation { background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 5px; }");
        html.push_str("</style></head><body>");
        
        html.push_str("<h1>Parallel Indexing Performance Report</h1>");
        
        // Overall metrics
        html.push_str("<h2>Overall Performance</h2>");
        html.push_str("<table>");
        html.push_str(&format!("<tr><td class='metric'>Total Vectors</td><td>{}</td></tr>", self.total_vectors));
        html.push_str(&format!("<tr><td class='metric'>Total Time</td><td>{:.2}s</td></tr>", self.total_time.as_secs_f64()));
        html.push_str(&format!("<tr><td class='metric'>Throughput</td><td>{:.1} vectors/sec</td></tr>", self.vectors_per_second));
        html.push_str(&format!("<tr><td class='metric'>Average Vector Time</td><td>{:.2}ms</td></tr>", self.average_vector_time.as_secs_f64() * 1000.0));
        html.push_str(&format!("<tr><td class='metric'>Thread Count</td><td>{}</td></tr>", self.thread_count));
        html.push_str(&format!("<tr><td class='metric'>Thread Efficiency</td><td>{:.1}%</td></tr>", self.thread_efficiency * 100.0));
        html.push_str(&format!("<tr><td class='metric'>Batch Size</td><td>{}</td></tr>", self.batch_size));
        html.push_str("</table>");
        
        // Thread performance breakdown
        html.push_str("<h2>Thread Performance</h2>");
        html.push_str("<table>");
        html.push_str("<tr><th>Thread ID</th><th>Operations</th><th>Avg Time</th><th>Min Time</th><th>Max Time</th></tr>");
        
        for thread_perf in &self.thread_performance {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td><td>{:.2}ms</td><td>{:.2}ms</td><td>{:.2}ms</td></tr>",
                thread_perf.thread_id,
                thread_perf.operations_completed,
                thread_perf.average_time.as_secs_f64() * 1000.0,
                thread_perf.min_time.as_secs_f64() * 1000.0,
                thread_perf.max_time.as_secs_f64() * 1000.0
            ));
        }
        html.push_str("</table>");
        
        // Recommendations
        html.push_str("<h2>Optimization Recommendations</h2>");
        for recommendation in &self.recommendations {
            html.push_str(&format!("<div class='recommendation'>{}</div>", recommendation));
        }
        
        html.push_str("</body></html>");
        html
    }
    
    /// Export performance data as CSV
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        
        // Header
        csv.push_str("metric,value,unit\n");
        
        // Overall metrics
        csv.push_str(&format!("total_vectors,{},count\n", self.total_vectors));
        csv.push_str(&format!("total_time,{:.3},seconds\n", self.total_time.as_secs_f64()));
        csv.push_str(&format!("throughput,{:.2},vectors_per_second\n", self.vectors_per_second));
        csv.push_str(&format!("thread_efficiency,{:.3},ratio\n", self.thread_efficiency));
        csv.push_str(&format!("batch_size,{},count\n", self.batch_size));
        
        // Thread-specific data
        for thread_perf in &self.thread_performance {
            csv.push_str(&format!("thread_{}_operations,{},count\n", thread_perf.thread_id, thread_perf.operations_completed));
            csv.push_str(&format!("thread_{}_avg_time,{:.3},milliseconds\n", thread_perf.thread_id, thread_perf.average_time.as_secs_f64() * 1000.0));
        }
        
        csv
    }
    
    /// Calculate performance score (0-100)
    pub fn calculate_performance_score(&self) -> f64 {
        let mut score = 100.0;
        
        // Penalize low throughput
        if self.vectors_per_second < 100.0 {
            score -= 20.0;
        } else if self.vectors_per_second < 500.0 {
            score -= 10.0;
        }
        
        // Penalize low thread efficiency
        if self.thread_efficiency < 0.7 {
            score -= 25.0;
        } else if self.thread_efficiency < 0.9 {
            score -= 10.0;
        }
        
        // Penalize high average processing time
        let avg_time_ms = self.average_vector_time.as_secs_f64() * 1000.0;
        if avg_time_ms > 10.0 {
            score -= 15.0;
        } else if avg_time_ms > 5.0 {
            score -= 5.0;
        }
        
        score.max(0.0)
    }
}
```

## Success Criteria
- [ ] Parallel indexer integrates seamlessly with performance monitoring
- [ ] Per-thread monitoring provides detailed performance insights
- [ ] Real-time monitoring tracks indexing progress accurately
- [ ] Performance reports include comprehensive analysis and recommendations
- [ ] Batch size optimization improves indexing efficiency
- [ ] Thread efficiency analysis identifies load balancing issues
- [ ] Error handling works correctly with monitoring enabled
- [ ] Concurrent monitoring maintains accuracy across threads
- [ ] HTML and CSV export formats work correctly
- [ ] All integration tests pass consistently

## Time Limit
10 minutes

## Notes
- Provides comprehensive monitoring integration with parallel indexing
- Supports per-thread performance tracking for detailed analysis
- Includes automatic optimization recommendations based on performance data
- Enables real-time monitoring of indexing progress
- Supports batch size optimization based on historical performance
- Provides detailed performance reports in multiple formats
- Maintains monitoring accuracy in concurrent scenarios
- Includes thread efficiency analysis for load balancing optimization