# Task 42: Comprehensive Monitoring Test Suite Setup

## Context
You are implementing Phase 4 of a vector indexing system. Real-time monitoring with live dashboards is now available. This task creates a comprehensive test suite for all monitoring functionality including unit tests, integration tests, load tests, and test utilities.

## Current State
- `src/monitor.rs` exists with complete performance monitoring functionality
- Advanced statistical calculations, reporting, and real-time monitoring are implemented
- Need comprehensive test coverage for all monitoring components

## Task Objective
Create a comprehensive test suite that thoroughly validates all monitoring functionality including performance tracking, statistical calculations, reporting, real-time monitoring, and edge cases.

## Implementation Requirements

### 1. Create test utilities module
Create a new file `src/monitor/test_utils.rs`:
```rust
use super::*;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime};

pub struct TestDataGenerator {
    seed: u64,
}

impl TestDataGenerator {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
    
    /// Generate deterministic test durations
    pub fn generate_durations(&mut self, count: usize, base_ms: u64, variance_ms: u64) -> Vec<Duration> {
        let mut durations = Vec::with_capacity(count);
        
        for i in 0..count {
            // Simple linear congruential generator for deterministic randomness
            self.seed = (self.seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
            let random_offset = (self.seed % (variance_ms * 2)) as i64 - variance_ms as i64;
            let duration_ms = (base_ms as i64 + random_offset).max(1) as u64;
            durations.push(Duration::from_millis(duration_ms));
        }
        
        durations
    }
    
    /// Generate trending data (increasing or decreasing)
    pub fn generate_trending_durations(&mut self, count: usize, start_ms: u64, end_ms: u64) -> Vec<Duration> {
        let mut durations = Vec::with_capacity(count);
        let slope = (end_ms as i64 - start_ms as i64) as f64 / (count - 1) as f64;
        
        for i in 0..count {
            // Add some random variation
            self.seed = (self.seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
            let noise = ((self.seed % 20) as i64 - 10) as f64; // ±10ms noise
            
            let base_value = start_ms as f64 + slope * i as f64 + noise;
            let duration_ms = base_value.max(1.0) as u64;
            durations.push(Duration::from_millis(duration_ms));
        }
        
        durations
    }
    
    /// Generate outlier data with occasional spikes
    pub fn generate_outlier_durations(&mut self, count: usize, normal_ms: u64, outlier_ms: u64, outlier_rate: f64) -> Vec<Duration> {
        let mut durations = Vec::with_capacity(count);
        
        for _ in 0..count {
            self.seed = (self.seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
            let is_outlier = (self.seed as f64 / u32::MAX as f64) < outlier_rate;
            
            let duration_ms = if is_outlier { outlier_ms } else { normal_ms };
            durations.push(Duration::from_millis(duration_ms));
        }
        
        durations
    }
    
    /// Generate performance-degrading data
    pub fn generate_degradation_pattern(&mut self, count: usize, good_duration: Duration, bad_duration: Duration, degradation_point: usize) -> Vec<Duration> {
        let mut durations = Vec::with_capacity(count);
        
        for i in 0..count {
            let base_duration = if i < degradation_point { good_duration } else { bad_duration };
            
            // Add small random variation
            self.seed = (self.seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
            let variance = (self.seed % 10) as u64; // 0-9ms variance
            let final_duration = base_duration + Duration::from_millis(variance);
            
            durations.push(final_duration);
        }
        
        durations
    }
}

pub struct MockTimeProvider {
    current_time: Arc<Mutex<SystemTime>>,
}

impl MockTimeProvider {
    pub fn new(start_time: SystemTime) -> Self {
        Self {
            current_time: Arc::new(Mutex::new(start_time)),
        }
    }
    
    pub fn advance_time(&self, duration: Duration) {
        if let Ok(mut time) = self.current_time.lock() {
            *time += duration;
        }
    }
    
    pub fn get_current_time(&self) -> SystemTime {
        self.current_time.lock().unwrap().clone()
    }
}

pub struct TestPerformanceScenario {
    pub name: String,
    pub query_durations: Vec<Duration>,
    pub index_durations: Vec<Duration>,
    pub expected_alerts: Vec<AlertSeverity>,
    pub expected_recommendations: Vec<String>,
}

impl TestPerformanceScenario {
    pub fn good_performance() -> Self {
        let mut generator = TestDataGenerator::new(12345);
        
        Self {
            name: "Good Performance".to_string(),
            query_durations: generator.generate_durations(100, 25, 5), // 25±5ms
            index_durations: generator.generate_durations(50, 100, 20), // 100±20ms
            expected_alerts: Vec::new(),
            expected_recommendations: vec!["GOOD PERFORMANCE".to_string()],
        }
    }
    
    pub fn high_variability() -> Self {
        let mut generator = TestDataGenerator::new(54321);
        
        Self {
            name: "High Variability".to_string(),
            query_durations: generator.generate_durations(100, 50, 40), // 50±40ms (high variance)
            index_durations: generator.generate_durations(50, 200, 150), // 200±150ms
            expected_alerts: Vec::new(),
            expected_recommendations: vec!["HIGH VARIABILITY".to_string()],
        }
    }
    
    pub fn performance_degradation() -> Self {
        let mut generator = TestDataGenerator::new(98765);
        
        Self {
            name: "Performance Degradation".to_string(),
            query_durations: generator.generate_trending_durations(100, 20, 80), // 20ms -> 80ms
            index_durations: generator.generate_trending_durations(50, 100, 300), // 100ms -> 300ms
            expected_alerts: Vec::new(),
            expected_recommendations: vec!["PERFORMANCE DEGRADATION".to_string(), "INDEXING DEGRADATION".to_string()],
        }
    }
    
    pub fn with_outliers() -> Self {
        let mut generator = TestDataGenerator::new(11111);
        
        Self {
            name: "With Outliers".to_string(),
            query_durations: generator.generate_outlier_durations(100, 30, 500, 0.05), // 5% outliers
            index_durations: generator.generate_outlier_durations(50, 150, 2000, 0.03), // 3% outliers
            expected_alerts: Vec::new(),
            expected_recommendations: vec!["OUTLIERS DETECTED".to_string()],
        }
    }
    
    pub fn slow_performance_with_alerts() -> Self {
        let mut generator = TestDataGenerator::new(22222);
        
        Self {
            name: "Slow Performance".to_string(),
            query_durations: generator.generate_durations(100, 1200, 200), // 1200±200ms (above threshold)
            index_durations: generator.generate_durations(50, 6000, 1000), // 6000±1000ms (above threshold)
            expected_alerts: vec![AlertSeverity::Warning, AlertSeverity::Warning], // Query and index alerts
            expected_recommendations: vec!["LOW THROUGHPUT".to_string()],
        }
    }
}

pub fn run_scenario_on_monitor(scenario: &TestPerformanceScenario, monitor: &mut PerformanceMonitor) {
    // Record query times
    for &duration in &scenario.query_durations {
        monitor.record_query_time(duration);
    }
    
    // Record index times
    for &duration in &scenario.index_durations {
        monitor.record_index_time(duration);
    }
}

pub fn assert_performance_expectations(
    stats: &PerformanceStats,
    advanced_stats: &AdvancedPerformanceStats,
    scenario: &TestPerformanceScenario,
) {
    // Verify basic counts
    assert_eq!(stats.total_queries, scenario.query_durations.len());
    assert_eq!(stats.total_indexes, scenario.index_durations.len());
    
    // Verify calculations are reasonable
    if !scenario.query_durations.is_empty() {
        assert!(stats.avg_query_time > Duration::from_millis(0));
        assert!(stats.min_query_time <= stats.avg_query_time);
        assert!(stats.avg_query_time <= stats.max_query_time);
        assert!(stats.p95_query_time <= stats.max_query_time);
        assert!(stats.p99_query_time <= stats.max_query_time);
    }
    
    if !scenario.index_durations.is_empty() {
        assert!(stats.avg_index_time > Duration::from_millis(0));
        assert!(stats.min_index_time <= stats.avg_index_time);
        assert!(stats.avg_index_time <= stats.max_index_time);
    }
}

pub fn create_load_test_data(operations: usize, threads: usize) -> Vec<Vec<Duration>> {
    let mut generator = TestDataGenerator::new(42);
    let mut thread_data = Vec::with_capacity(threads);
    
    let ops_per_thread = operations / threads;
    
    for thread_id in 0..threads {
        let base_latency = 20 + (thread_id * 5) as u64; // Slightly different latency per thread
        let durations = generator.generate_durations(ops_per_thread, base_latency, 10);
        thread_data.push(durations);
    }
    
    thread_data
}
```

### 2. Create comprehensive unit tests
Add to `src/monitor.rs` in the tests module:
```rust
#[cfg(test)]
mod comprehensive_tests {
    use super::*;
    use super::test_utils::*;
    
    #[test]
    fn test_performance_scenarios() {
        let scenarios = vec![
            TestPerformanceScenario::good_performance(),
            TestPerformanceScenario::high_variability(),
            TestPerformanceScenario::performance_degradation(),
            TestPerformanceScenario::with_outliers(),
        ];
        
        for scenario in scenarios {
            let mut monitor = PerformanceMonitor::new();
            run_scenario_on_monitor(&scenario, &mut monitor);
            
            let stats = monitor.get_stats();
            let advanced_stats = monitor.get_advanced_stats(0.95);
            
            // Basic validation
            assert_performance_expectations(&stats, &advanced_stats, &scenario);
            
            // Generate and validate report
            let config = ReportConfig::default();
            let report = monitor.generate_report(&config);
            
            assert!(!report.summary.is_empty());
            assert!(!report.recommendations.is_empty());
            
            println!("Scenario '{}' completed successfully", scenario.name);
        }
    }
    
    #[test]
    fn test_statistical_accuracy() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add known data: 10, 20, 30, 40, 50 ms
        let known_durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];
        
        for duration in known_durations {
            monitor.record_query_time(duration);
        }
        
        let stats = monitor.get_stats();
        let advanced_stats = monitor.get_advanced_stats(0.95);
        
        // Verify exact calculations
        assert_eq!(stats.avg_query_time, Duration::from_millis(30)); // (10+20+30+40+50)/5 = 30
        assert_eq!(stats.min_query_time, Duration::from_millis(10));
        assert_eq!(stats.max_query_time, Duration::from_millis(50));
        
        // Verify percentiles
        let percentiles = monitor.calculate_percentiles(&monitor.query_times, &[25.0, 50.0, 75.0]);
        assert_eq!(percentiles[0], Duration::from_millis(20)); // 25th percentile
        assert_eq!(percentiles[1], Duration::from_millis(30)); // 50th percentile (median)
        assert_eq!(percentiles[2], Duration::from_millis(40)); // 75th percentile
        
        // Verify IQR
        assert_eq!(advanced_stats.query_iqr, Duration::from_millis(20)); // Q3 - Q1 = 40 - 20
        
        // Verify standard deviation (for this data set, std dev ≈ 15.81ms)
        let std_dev_ms = advanced_stats.query_std_dev.as_secs_f64() * 1000.0;
        assert!((std_dev_ms - 15.81).abs() < 0.1);
    }
    
    #[test]
    fn test_trend_analysis_accuracy() {
        let mut monitor = PerformanceMonitor::new();
        
        // Perfect linear trend: 10, 20, 30, 40, 50 ms
        for i in 1..=5 {
            monitor.record_query_time(Duration::from_millis(i * 10));
        }
        
        let advanced_stats = monitor.get_advanced_stats(0.95);
        
        // Should detect strong positive trend
        assert!(advanced_stats.query_trend_slope > 9.0); // Should be ~10ms per operation
        assert!(advanced_stats.query_trend_slope < 11.0);
        
        // Should have high correlation (close to 1.0)
        assert!(advanced_stats.query_correlation > 0.99);
        
        // Prediction should be around 60ms for next value
        let prediction_ms = advanced_stats.query_prediction.as_secs_f64() * 1000.0;
        assert!(prediction_ms > 55.0 && prediction_ms < 65.0);
    }
    
    #[test]
    fn test_outlier_detection_accuracy() {
        let mut monitor = PerformanceMonitor::new();
        
        // Normal data: 20, 25, 30, 35, 40 ms
        for i in 4..=8 {
            monitor.record_query_time(Duration::from_millis(i * 5));
        }
        
        // Add outliers: 200ms (way above normal range)
        monitor.record_query_time(Duration::from_millis(200));
        
        let advanced_stats = monitor.get_advanced_stats(0.95);
        
        // Should detect the outlier
        assert!(!advanced_stats.query_outliers.is_empty());
        assert!(advanced_stats.query_outliers.contains(&Duration::from_millis(200)));
    }
    
    #[test]
    fn test_confidence_intervals() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add consistent data (low variance)
        for _ in 0..20 {
            monitor.record_query_time(Duration::from_millis(100));
        }
        
        let advanced_stats = monitor.get_advanced_stats(0.95);
        let (lower, upper) = advanced_stats.query_confidence_interval;
        
        // With consistent data, confidence interval should be very narrow
        let interval_width = upper.saturating_sub(lower);
        assert!(interval_width < Duration::from_millis(5));
        
        // Mean should be within the interval
        let mean = monitor.average(&monitor.query_times);
        assert!(mean >= lower && mean <= upper);
        assert_eq!(mean, Duration::from_millis(100));
    }
    
    #[test]
    fn test_monitor_capacity_limits() {
        let mut monitor = PerformanceMonitor::with_capacity(5);
        
        // Add more samples than capacity
        for i in 1..=10 {
            monitor.record_query_time(Duration::from_millis(i * 10));
        }
        
        let (query_samples, _) = monitor.current_sample_count();
        assert_eq!(query_samples, 5); // Should be limited to capacity
        
        let stats = monitor.get_stats();
        assert_eq!(stats.total_queries, 10); // Total count should be accurate
        
        // Should only have the last 5 samples
        let recent_times = monitor.get_recent_query_times(10);
        assert_eq!(recent_times.len(), 5);
        
        // Most recent should be 100ms, oldest in buffer should be 60ms
        assert_eq!(recent_times[0], Duration::from_millis(100));
        assert_eq!(recent_times[4], Duration::from_millis(60));
    }
    
    #[test]
    fn test_performance_degradation_detection() {
        let mut monitor = PerformanceMonitor::new();
        
        // Generate performance degradation pattern
        let mut generator = TestDataGenerator::new(777);
        let durations = generator.generate_degradation_pattern(100, Duration::from_millis(20), Duration::from_millis(80), 50);
        
        for duration in durations {
            monitor.record_query_time(duration);
        }
        
        // Should detect degradation
        assert!(monitor.is_performance_degrading(25, 0.3)); // 30% threshold
        
        // Should not detect with very high threshold
        assert!(!monitor.is_performance_degrading(25, 5.0)); // 500% threshold
    }
    
    #[test]
    fn test_empty_monitor_edge_cases() {
        let monitor = PerformanceMonitor::new();
        
        let stats = monitor.get_stats();
        let advanced_stats = monitor.get_advanced_stats(0.95);
        
        // Should handle empty data gracefully
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.total_indexes, 0);
        assert_eq!(stats.avg_query_time, Duration::from_millis(0));
        assert_eq!(advanced_stats.query_std_dev, Duration::from_millis(0));
        assert_eq!(advanced_stats.query_cv, 0.0);
        assert!(advanced_stats.query_outliers.is_empty());
        
        // Should not detect degradation with no data
        assert!(!monitor.is_performance_degrading(10, 0.1));
        
        // Report generation should work
        let config = ReportConfig::default();
        let report = monitor.generate_report(&config);
        assert!(!report.summary.is_empty());
        assert!(!report.recommendations.is_empty());
    }
    
    #[test]
    fn test_single_sample_edge_cases() {
        let mut monitor = PerformanceMonitor::new();
        monitor.record_query_time(Duration::from_millis(50));
        
        let stats = monitor.get_stats();
        let advanced_stats = monitor.get_advanced_stats(0.95);
        
        // Should handle single sample correctly
        assert_eq!(stats.total_queries, 1);
        assert_eq!(stats.avg_query_time, Duration::from_millis(50));
        assert_eq!(stats.min_query_time, Duration::from_millis(50));
        assert_eq!(stats.max_query_time, Duration::from_millis(50));
        assert_eq!(stats.p95_query_time, Duration::from_millis(50));
        assert_eq!(stats.p99_query_time, Duration::from_millis(50));
        
        // Standard deviation should be 0 for single sample
        assert_eq!(advanced_stats.query_std_dev, Duration::from_millis(0));
        assert_eq!(advanced_stats.query_cv, 0.0);
    }
}
```

### 3. Create integration tests
Add to `src/monitor.rs`:
```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use super::test_utils::*;
    use std::sync::{Arc, Barrier};
    
    #[test]
    fn test_concurrent_monitoring() {
        let shared_monitor = SharedPerformanceMonitor::new();
        let num_threads = 4;
        let operations_per_thread = 100;
        let barrier = Arc::new(Barrier::new(num_threads));
        
        let handles: Vec<_> = (0..num_threads).map(|thread_id| {
            let monitor_clone = shared_monitor.clone();
            let barrier_clone = barrier.clone();
            
            thread::spawn(move || {
                barrier_clone.wait(); // Synchronize start
                
                for i in 0..operations_per_thread {
                    let query_time = Duration::from_millis(20 + (thread_id * 5) as u64 + (i % 10) as u64);
                    let index_time = Duration::from_millis(100 + (thread_id * 10) as u64 + (i % 20) as u64);
                    
                    monitor_clone.record_query_time(query_time);
                    monitor_clone.record_index_time(index_time);
                    
                    // Occasionally check stats
                    if i % 25 == 0 {
                        let _stats = monitor_clone.get_stats();
                    }
                }
            })
        }).collect();
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Verify final results
        if let Some(stats) = shared_monitor.get_stats() {
            assert_eq!(stats.total_queries, num_threads * operations_per_thread);
            assert_eq!(stats.total_indexes, num_threads * operations_per_thread);
            assert!(stats.avg_query_time > Duration::from_millis(0));
            assert!(stats.avg_index_time > Duration::from_millis(0));
        } else {
            panic!("Failed to get stats from shared monitor");
        }
    }
    
    #[test]
    fn test_real_time_monitoring_integration() {
        let update_interval = Duration::from_millis(50);
        let (rt_monitor, receiver) = RealTimeMonitor::new(update_interval);
        
        rt_monitor.start();
        
        // Record some operations
        for i in 1..=10 {
            rt_monitor.record_query_time(Duration::from_millis(i * 10));
            thread::sleep(Duration::from_millis(10));
        }
        
        // Wait for updates to be processed
        thread::sleep(Duration::from_millis(200));
        
        rt_monitor.stop();
        
        // Collect updates
        let updates: Vec<_> = receiver.try_iter().collect();
        
        // Should have received metric updates
        assert!(!updates.is_empty());
        
        let metric_updates = updates.iter().filter(|u| matches!(u.update_type, UpdateType::MetricUpdate)).count();
        assert!(metric_updates > 0);
    }
    
    #[test]
    fn test_alert_system_integration() {
        let thresholds = AlertThresholds {
            max_query_time: Duration::from_millis(50),
            max_index_time: Duration::from_millis(200),
            min_throughput_qps: 5.0,
            ..Default::default()
        };
        
        let (rt_monitor, receiver) = RealTimeMonitor::with_thresholds(Duration::from_millis(25), thresholds);
        rt_monitor.start();
        
        // Record operations that should trigger alerts
        rt_monitor.record_query_time(Duration::from_millis(100)); // Above threshold
        rt_monitor.record_index_time(Duration::from_millis(500)); // Above threshold
        
        // Wait for processing
        thread::sleep(Duration::from_millis(100));
        
        rt_monitor.stop();
        
        // Check for alerts
        let updates: Vec<_> = receiver.try_iter().collect();
        let alerts: Vec<_> = updates.iter()
            .filter(|u| matches!(u.update_type, UpdateType::Alert))
            .collect();
        
        assert!(!alerts.is_empty(), "Expected alerts for threshold violations");
    }
    
    #[test]
    fn test_dashboard_integration() {
        let mut dashboard = LiveDashboard::new("Integration Test Dashboard".to_string());
        
        // Add charts
        let query_chart = LiveChart {
            chart_id: "query_time".to_string(),
            title: "Query Time".to_string(),
            chart_type: ChartType::TimeSeries,
            data_window: 50,
            real_time_data: Vec::new(),
            update_frequency: Duration::from_millis(100),
        };
        
        let throughput_chart = LiveChart {
            chart_id: "throughput".to_string(),
            title: "Throughput".to_string(),
            chart_type: ChartType::TimeSeries,
            data_window: 50,
            real_time_data: Vec::new(),
            update_frequency: Duration::from_millis(100),
        };
        
        dashboard.add_chart(query_chart);
        dashboard.add_chart(throughput_chart);
        
        // Simulate performance updates
        for i in 1..=20 {
            let stats = PerformanceStats {
                total_queries: i,
                queries_per_second: i as f64 * 0.5,
                avg_query_time: Duration::from_millis(25 + i),
                p99_query_time: Duration::from_millis(50 + i * 2),
                ..Default::default()
            };
            
            let advanced_stats = AdvancedPerformanceStats::default();
            dashboard.update_from_monitor(stats, advanced_stats);
            
            // Add some updates
            let update = MonitorUpdate {
                timestamp: SystemTime::now(),
                update_type: UpdateType::MetricUpdate,
                data: UpdateData::QueryMetric {
                    time: Duration::from_millis(25 + i),
                    total_count: i,
                },
            };
            
            dashboard.add_update(update);
        }
        
        // Verify dashboard state
        assert_eq!(dashboard.current_stats.total_queries, 20);
        assert_eq!(dashboard.recent_updates.len(), 20);
        assert_eq!(dashboard.charts.len(), 2);
        
        // Verify charts have data
        let query_chart = dashboard.charts.iter().find(|c| c.chart_id == "query_time").unwrap();
        assert_eq!(query_chart.real_time_data.len(), 20);
        
        // Generate HTML and verify
        let html = dashboard.generate_html_dashboard();
        assert!(html.contains("Integration Test Dashboard"));
        assert!(html.contains("20")); // Should show total queries
        assert!(html.contains("refresh")); // Should have auto-refresh
    }
    
    #[test]
    fn test_end_to_end_monitoring_workflow() {
        // Create monitor
        let mut monitor = PerformanceMonitor::new();
        
        // Generate realistic load
        let mut generator = TestDataGenerator::new(12345);
        let query_durations = generator.generate_durations(200, 30, 10);
        let index_durations = generator.generate_durations(100, 150, 50);
        
        // Record operations
        for duration in query_durations {
            monitor.record_query_time(duration);
        }
        for duration in index_durations {
            monitor.record_index_time(duration);
        }
        
        // Get comprehensive statistics
        let stats = monitor.get_stats();
        let advanced_stats = monitor.get_advanced_stats(0.95);
        
        // Generate reports in all formats
        let config = ReportConfig::default();
        let report = monitor.generate_report(&config);
        
        let text_report = report.export_text();
        let csv_report = report.export_csv();
        let html_report = report.export_html();
        
        // Verify all reports contain expected data
        assert!(text_report.contains("200")); // Total queries
        assert!(text_report.contains("100")); // Total indexes
        assert!(csv_report.contains("total_queries,200"));
        assert!(html_report.contains("Performance Monitoring Report"));
        
        // Verify recommendations
        assert!(!report.recommendations.is_empty());
        
        // Verify charts
        assert!(!report.charts.is_empty());
        let time_series_charts = report.charts.iter()
            .filter(|c| matches!(c.chart_type, ChartType::TimeSeries))
            .count();
        assert!(time_series_charts >= 2); // Query and index time series
        
        println!("End-to-end workflow completed successfully");
        println!("Generated text report: {} characters", text_report.len());
        println!("Generated CSV report: {} characters", csv_report.len());
        println!("Generated HTML report: {} characters", html_report.len());
    }
}
```

### 4. Create load testing module
```rust
#[cfg(test)]
mod load_tests {
    use super::*;
    use super::test_utils::*;
    
    #[test]
    fn test_high_volume_monitoring() {
        let mut monitor = PerformanceMonitor::with_capacity(10000);
        let operations = 50000;
        
        let start_time = Instant::now();
        
        // Record large number of operations
        let mut generator = TestDataGenerator::new(9999);
        let durations = generator.generate_durations(operations, 25, 15);
        
        for duration in durations {
            monitor.record_query_time(duration);
        }
        
        let recording_time = start_time.elapsed();
        
        // Measure statistics calculation time
        let stats_start = Instant::now();
        let stats = monitor.get_stats();
        let advanced_stats = monitor.get_advanced_stats(0.95);
        let stats_time = stats_start.elapsed();
        
        // Measure report generation time
        let report_start = Instant::now();
        let config = ReportConfig::default();
        let report = monitor.generate_report(&config);
        let report_time = report_start.elapsed();
        
        // Verify performance is reasonable
        assert!(recording_time < Duration::from_secs(1), "Recording {} operations took too long: {:?}", operations, recording_time);
        assert!(stats_time < Duration::from_millis(100), "Statistics calculation took too long: {:?}", stats_time);
        assert!(report_time < Duration::from_millis(500), "Report generation took too long: {:?}", report_time);
        
        // Verify correctness
        assert_eq!(stats.total_queries, operations);
        assert!(stats.avg_query_time > Duration::from_millis(0));
        assert!(!report.summary.is_empty());
        
        println!("Load test completed:");
        println!("  Operations: {}", operations);
        println!("  Recording time: {:?}", recording_time);
        println!("  Statistics time: {:?}", stats_time);
        println!("  Report time: {:?}", report_time);
        println!("  Operations/sec during recording: {:.0}", operations as f64 / recording_time.as_secs_f64());
    }
    
    #[test]
    fn test_concurrent_load() {
        let shared_monitor = SharedPerformanceMonitor::with_capacity(50000);
        let num_threads = 8;
        let operations_per_thread = 5000;
        let total_operations = num_threads * operations_per_thread;
        
        let load_data = create_load_test_data(total_operations, num_threads);
        let start_time = Instant::now();
        
        let handles: Vec<_> = load_data.into_iter().enumerate().map(|(thread_id, durations)| {
            let monitor_clone = shared_monitor.clone();
            
            thread::spawn(move || {
                for (i, duration) in durations.into_iter().enumerate() {
                    monitor_clone.record_query_time(duration);
                    
                    // Mix in some index operations
                    if i % 3 == 0 {
                        monitor_clone.record_index_time(duration * 3);
                    }
                }
            })
        }).collect();
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        let total_time = start_time.elapsed();
        
        // Verify results
        if let Some(stats) = shared_monitor.get_stats() {
            assert_eq!(stats.total_queries, total_operations);
            assert!(stats.total_indexes > 0);
            assert!(stats.avg_query_time > Duration::from_millis(0));
            
            let throughput = total_operations as f64 / total_time.as_secs_f64();
            
            println!("Concurrent load test completed:");
            println!("  Threads: {}", num_threads);
            println!("  Total operations: {}", total_operations);
            println!("  Total time: {:?}", total_time);
            println!("  Throughput: {:.0} ops/sec", throughput);
            
            // Should handle high concurrent load efficiently
            assert!(throughput > 10000.0, "Throughput too low: {:.0} ops/sec", throughput);
        } else {
            panic!("Failed to get stats after concurrent load test");
        }
    }
    
    #[test]
    fn test_memory_usage_under_load() {
        let capacity = 1000;
        let mut monitor = PerformanceMonitor::with_capacity(capacity);
        let operations = 100000; // Much more than capacity
        
        // Record many operations
        let mut generator = TestDataGenerator::new(4444);
        let durations = generator.generate_durations(operations, 30, 10);
        
        for duration in durations {
            monitor.record_query_time(duration);
        }
        
        // Verify memory usage is bounded
        let (query_samples, _) = monitor.current_sample_count();
        assert_eq!(query_samples, capacity);
        
        // Verify total count is still accurate
        let stats = monitor.get_stats();
        assert_eq!(stats.total_queries, operations);
        
        // Verify recent data is what we expect (should be the last `capacity` samples)
        let recent_times = monitor.get_recent_query_times(capacity);
        assert_eq!(recent_times.len(), capacity);
        
        println!("Memory usage test passed:");
        println!("  Capacity: {}", capacity);
        println!("  Operations recorded: {}", operations);
        println!("  Samples retained: {}", query_samples);
    }
}
```

## Success Criteria
- [ ] Test utilities generate deterministic and varied test data
- [ ] Comprehensive unit tests cover all monitoring functionality
- [ ] Integration tests verify component interactions work correctly
- [ ] Load tests validate performance under high-volume scenarios
- [ ] Concurrent tests ensure thread-safety works properly
- [ ] Edge case tests handle empty data and single samples correctly
- [ ] Statistical accuracy tests verify mathematical correctness
- [ ] Memory usage tests confirm bounded resource consumption
- [ ] All tests pass consistently and reliably
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Test utilities provide deterministic data generation for reliable testing
- Comprehensive scenarios test realistic performance patterns
- Load testing validates scalability and performance characteristics
- Integration tests ensure all components work together correctly
- Edge case testing ensures robust handling of unusual conditions
- Statistical accuracy testing validates mathematical implementations
- Concurrent testing ensures thread-safety under load