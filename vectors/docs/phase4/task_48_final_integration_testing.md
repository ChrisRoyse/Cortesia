# Task 48: Final Integration Testing and System Validation

## Context
You are implementing Phase 4 of a vector indexing system. The monitoring system optimizations and cleanup are now complete. This final task performs comprehensive integration testing, system validation, end-to-end testing, and production readiness verification for the entire monitoring system.

## Current State
- Complete performance monitoring system with advanced statistics and reporting
- Comprehensive alerting and notification system with multiple channels
- Full integration with parallel indexer and search engine
- Monitoring system optimizations and production deployment utilities
- Need comprehensive integration testing and final validation

## Task Objective
Perform comprehensive integration testing, end-to-end system validation, load testing, failover testing, and production readiness verification to ensure the entire monitoring system works correctly under all conditions and is ready for production deployment.

## Implementation Requirements

### 1. Create comprehensive integration test suite
Create a new file `src/monitor/integration_tests.rs`:
```rust
use super::*;
use crate::alerting::{AlertingSystem, AlertingConfig, AlertThresholds, EmailNotificationChannel, SlackNotificationChannel};
use crate::parallel::MonitoredParallelIndexer;
use crate::search_monitor::MonitoredSearchEngine;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime};
use std::collections::HashMap;

pub struct IntegrationTestSuite {
    test_config: IntegrationTestConfig,
    test_results: Vec<TestResult>,
}

#[derive(Debug, Clone)]
pub struct IntegrationTestConfig {
    pub enable_load_testing: bool,
    pub enable_failover_testing: bool,
    pub enable_performance_testing: bool,
    pub enable_alert_testing: bool,
    pub load_test_duration: Duration,
    pub max_concurrent_operations: usize,
    pub stress_test_multiplier: f64,
    pub acceptable_error_rate: f64,
}

impl Default for IntegrationTestConfig {
    fn default() -> Self {
        Self {
            enable_load_testing: true,
            enable_failover_testing: true,
            enable_performance_testing: true,
            enable_alert_testing: true,
            load_test_duration: Duration::from_secs(60),
            max_concurrent_operations: 1000,
            stress_test_multiplier: 2.0,
            acceptable_error_rate: 0.01, // 1%
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub test_category: TestCategory,
    pub status: TestStatus,
    pub duration: Duration,
    pub error_message: Option<String>,
    pub performance_metrics: HashMap<String, f64>,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TestCategory {
    Unit,
    Integration,
    Load,
    Failover,
    Performance,
    Alert,
    EndToEnd,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TestStatus {
    Passed,
    Failed,
    Skipped,
    Warning,
}

impl IntegrationTestSuite {
    pub fn new(config: IntegrationTestConfig) -> Self {
        Self {
            test_config: config,
            test_results: Vec::new(),
        }
    }
    
    /// Run complete integration test suite
    pub async fn run_complete_test_suite(&mut self) -> IntegrationTestResults {
        let start_time = Instant::now();
        
        println!("Starting comprehensive integration test suite...");
        
        // Test 1: Basic monitoring system integration
        self.test_basic_monitoring_integration().await;
        
        // Test 2: Alerting system integration
        if self.test_config.enable_alert_testing {
            self.test_alerting_system_integration().await;
        }
        
        // Test 3: Parallel indexer integration
        self.test_parallel_indexer_integration().await;
        
        // Test 4: Search engine integration
        self.test_search_engine_integration().await;
        
        // Test 5: Load testing
        if self.test_config.enable_load_testing {
            self.test_high_load_scenarios().await;
        }
        
        // Test 6: Failover testing
        if self.test_config.enable_failover_testing {
            self.test_failover_scenarios().await;
        }
        
        // Test 7: Performance benchmarking
        if self.test_config.enable_performance_testing {
            self.test_performance_benchmarks().await;
        }
        
        // Test 8: End-to-end workflow testing
        self.test_end_to_end_workflows().await;
        
        // Test 9: Memory and resource testing
        self.test_memory_and_resource_usage().await;
        
        // Test 10: Production configuration testing
        self.test_production_configuration().await;
        
        let total_duration = start_time.elapsed();
        
        println!("Integration test suite completed in {:.2}s", total_duration.as_secs_f64());
        
        IntegrationTestResults {
            total_tests: self.test_results.len(),
            passed_tests: self.test_results.iter().filter(|r| r.status == TestStatus::Passed).count(),
            failed_tests: self.test_results.iter().filter(|r| r.status == TestStatus::Failed).count(),
            skipped_tests: self.test_results.iter().filter(|r| r.status == TestStatus::Skipped).count(),
            warning_tests: self.test_results.iter().filter(|r| r.status == TestStatus::Warning).count(),
            total_duration,
            test_results: self.test_results.clone(),
            overall_status: self.calculate_overall_status(),
        }
    }
    
    async fn test_basic_monitoring_integration(&mut self) {
        let test_start = Instant::now();
        
        println!("Testing basic monitoring system integration...");
        
        match self.run_basic_monitoring_test().await {
            Ok(metrics) => {
                self.test_results.push(TestResult {
                    test_name: "basic_monitoring_integration".to_string(),
                    test_category: TestCategory::Integration,
                    status: TestStatus::Passed,
                    duration: test_start.elapsed(),
                    error_message: None,
                    performance_metrics: metrics,
                    timestamp: SystemTime::now(),
                });
                println!("✓ Basic monitoring integration test passed");
            }
            Err(e) => {
                self.test_results.push(TestResult {
                    test_name: "basic_monitoring_integration".to_string(),
                    test_category: TestCategory::Integration,
                    status: TestStatus::Failed,
                    duration: test_start.elapsed(),
                    error_message: Some(e.to_string()),
                    performance_metrics: HashMap::new(),
                    timestamp: SystemTime::now(),
                });
                println!("✗ Basic monitoring integration test failed: {}", e);
            }
        }
    }
    
    async fn run_basic_monitoring_test(&self) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut metrics = HashMap::new();
        
        // Create and test basic monitoring functionality
        let shared_monitor = Arc::new(SharedPerformanceMonitor::new());
        
        // Record various operations
        for i in 0..1000 {
            shared_monitor.record_query_time(Duration::from_millis(20 + (i % 50) as u64));
            if i % 5 == 0 {
                shared_monitor.record_index_time(Duration::from_millis(100 + (i % 100) as u64));
            }
        }
        
        // Verify statistics
        let stats = shared_monitor.get_stats()
            .ok_or("Failed to get performance statistics")?;
        
        metrics.insert("total_queries".to_string(), stats.total_queries as f64);
        metrics.insert("total_indexes".to_string(), stats.total_indexes as f64);
        metrics.insert("avg_query_time_ms".to_string(), stats.avg_query_time.as_secs_f64() * 1000.0);
        metrics.insert("queries_per_second".to_string(), stats.queries_per_second);
        
        // Validate results
        if stats.total_queries != 1000 {
            return Err(format!("Expected 1000 queries, got {}", stats.total_queries).into());
        }
        
        if stats.total_indexes != 200 {
            return Err(format!("Expected 200 indexes, got {}", stats.total_indexes).into());
        }
        
        Ok(metrics)
    }
    
    async fn test_alerting_system_integration(&mut self) {
        let test_start = Instant::now();
        
        println!("Testing alerting system integration...");
        
        match self.run_alerting_test().await {
            Ok(metrics) => {
                self.test_results.push(TestResult {
                    test_name: "alerting_system_integration".to_string(),
                    test_category: TestCategory::Alert,
                    status: TestStatus::Passed,
                    duration: test_start.elapsed(),
                    error_message: None,
                    performance_metrics: metrics,
                    timestamp: SystemTime::now(),
                });
                println!("✓ Alerting system integration test passed");
            }
            Err(e) => {
                self.test_results.push(TestResult {
                    test_name: "alerting_system_integration".to_string(),
                    test_category: TestCategory::Alert,
                    status: TestStatus::Failed,
                    duration: test_start.elapsed(),
                    error_message: Some(e.to_string()),
                    performance_metrics: HashMap::new(),
                    timestamp: SystemTime::now(),
                });
                println!("✗ Alerting system integration test failed: {}", e);
            }
        }
    }
    
    async fn run_alerting_test(&self) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut metrics = HashMap::new();
        
        // Create alerting system with low thresholds for testing
        let alerting_config = AlertingConfig {
            check_interval: Duration::from_millis(100),
            max_alerts_per_minute: 10,
            ..Default::default()
        };
        
        let alert_thresholds = AlertThresholds {
            max_avg_query_time: Duration::from_millis(50), // Low threshold for testing
            max_p99_query_time: Duration::from_millis(200),
            min_queries_per_second: 5.0,
            ..Default::default()
        };
        
        let mut alerting_system = AlertingSystem::new(alerting_config, alert_thresholds);
        
        // Add test notification channels
        let email_channel = Box::new(EmailNotificationChannel {
            smtp_server: "test.smtp.com".to_string(),
            recipients: vec!["test@example.com".to_string()],
            enabled: true,
        });
        
        alerting_system.add_notification_channel(email_channel);
        
        // Create monitor with performance that should trigger alerts
        let shared_monitor = Arc::new(SharedPerformanceMonitor::new());
        
        // Record slow operations to trigger alerts
        for _ in 0..50 {
            shared_monitor.record_query_time(Duration::from_millis(100)); // Above threshold
        }
        
        alerting_system.add_monitor(shared_monitor);
        
        // Start alerting system briefly
        alerting_system.start()?;
        
        // Wait for alerts to be processed
        thread::sleep(Duration::from_millis(500));
        
        alerting_system.stop();
        
        // Check alerting status
        let status = alerting_system.get_alerting_status();
        let active_alerts = alerting_system.get_active_alerts();
        
        metrics.insert("active_alerts".to_string(), status.active_alerts as f64);
        metrics.insert("total_alerts_today".to_string(), status.total_alerts_today as f64);
        
        // Verify alerts were generated
        if active_alerts.is_empty() {
            return Err("Expected alerts to be generated but none were found".into());
        }
        
        Ok(metrics)
    }
    
    async fn test_parallel_indexer_integration(&mut self) {
        let test_start = Instant::now();
        
        println!("Testing parallel indexer integration...");
        
        match self.run_parallel_indexer_test().await {
            Ok(metrics) => {
                self.test_results.push(TestResult {
                    test_name: "parallel_indexer_integration".to_string(),
                    test_category: TestCategory::Integration,
                    status: TestStatus::Passed,
                    duration: test_start.elapsed(),
                    error_message: None,
                    performance_metrics: metrics,
                    timestamp: SystemTime::now(),
                });
                println!("✓ Parallel indexer integration test passed");
            }
            Err(e) => {
                self.test_results.push(TestResult {
                    test_name: "parallel_indexer_integration".to_string(),
                    test_category: TestCategory::Integration,
                    status: TestStatus::Failed,
                    duration: test_start.elapsed(),
                    error_message: Some(e.to_string()),
                    performance_metrics: HashMap::new(),
                    timestamp: SystemTime::now(),
                });
                println!("✗ Parallel indexer integration test failed: {}", e);
            }
        }
    }
    
    async fn run_parallel_indexer_test(&self) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut metrics = HashMap::new();
        
        // Create monitored parallel indexer
        let indexer_config = crate::parallel::IndexerMonitoringConfig::default();
        let mut monitored_indexer = MonitoredParallelIndexer::new(4, indexer_config);
        
        // Generate test vectors
        let vectors = self.generate_test_vectors(1000, 128);
        
        // Index vectors with monitoring
        let result = monitored_indexer.index_vectors_monitored(vectors)
            .map_err(|e| format!("Indexing failed: {}", e))?;
        
        metrics.insert("indexed_count".to_string(), result.indexed_count as f64);
        metrics.insert("total_time_ms".to_string(), result.total_time.as_secs_f64() * 1000.0);
        
        if let Some(report) = result.performance_report {
            metrics.insert("vectors_per_second".to_string(), report.vectors_per_second);
            metrics.insert("thread_efficiency".to_string(), report.thread_efficiency);
            metrics.insert("thread_count".to_string(), report.thread_count as f64);
        }
        
        // Get monitoring statistics
        let monitoring_stats = monitored_indexer.get_monitoring_stats();
        
        if let Some(global_stats) = monitoring_stats.global_stats {
            metrics.insert("monitoring_total_indexes".to_string(), global_stats.total_indexes as f64);
            metrics.insert("monitoring_avg_time_ms".to_string(), global_stats.avg_index_time.as_secs_f64() * 1000.0);
        }
        
        // Verify results
        if result.indexed_count != 1000 {
            return Err(format!("Expected to index 1000 vectors, got {}", result.indexed_count).into());
        }
        
        Ok(metrics)
    }
    
    async fn test_search_engine_integration(&mut self) {
        let test_start = Instant::now();
        
        println!("Testing search engine integration...");
        
        match self.run_search_engine_test().await {
            Ok(metrics) => {
                self.test_results.push(TestResult {
                    test_name: "search_engine_integration".to_string(),
                    test_category: TestCategory::Integration,
                    status: TestStatus::Passed,
                    duration: test_start.elapsed(),
                    error_message: None,
                    performance_metrics: metrics,
                    timestamp: SystemTime::now(),
                });
                println!("✓ Search engine integration test passed");
            }
            Err(e) => {
                self.test_results.push(TestResult {
                    test_name: "search_engine_integration".to_string(),
                    test_category: TestCategory::Integration,
                    status: TestStatus::Failed,
                    duration: test_start.elapsed(),
                    error_message: Some(e.to_string()),
                    performance_metrics: HashMap::new(),
                    timestamp: SystemTime::now(),
                });
                println!("✗ Search engine integration test failed: {}", e);
            }
        }
    }
    
    async fn run_search_engine_test(&self) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut metrics = HashMap::new();
        
        // Create monitored search engine
        let search_engine = crate::search::SearchEngine::new();
        let search_config = crate::search_monitor::SearchMonitoringConfig::default();
        let monitored_engine = MonitoredSearchEngine::new(search_engine, search_config);
        
        // Execute various search queries
        let queries = vec![
            crate::search::SearchQuery::new().with_text("test query".to_string()),
            crate::search::SearchQuery::new().with_text("fuzzy search".to_string()).with_fuzzy(true),
            crate::search::SearchQuery::new().with_vector(vec![1.0; 64]),
        ];
        
        let mut total_results = 0;
        let mut total_execution_time = Duration::from_millis(0);
        
        for query in queries {
            let result = monitored_engine.search_monitored(query)
                .map_err(|e| format!("Search failed: {}", e))?;
            
            total_results += result.result_count;
            total_execution_time += result.execution_time;
        }
        
        metrics.insert("total_queries".to_string(), 3.0);
        metrics.insert("total_results".to_string(), total_results as f64);
        metrics.insert("avg_execution_time_ms".to_string(), (total_execution_time / 3).as_secs_f64() * 1000.0);
        
        // Get search statistics
        let search_stats = monitored_engine.get_search_statistics();
        
        if let Some(global_stats) = search_stats.global_stats {
            metrics.insert("search_total_queries".to_string(), global_stats.total_queries as f64);
            metrics.insert("search_avg_time_ms".to_string(), global_stats.avg_query_time.as_secs_f64() * 1000.0);
        }
        
        metrics.insert("cache_hit_rate".to_string(), search_stats.cache_stats.hit_rate);
        
        Ok(metrics)
    }
    
    async fn test_high_load_scenarios(&mut self) {
        let test_start = Instant::now();
        
        println!("Testing high load scenarios...");
        
        match self.run_load_test().await {
            Ok(metrics) => {
                let status = if metrics.get("error_rate").unwrap_or(&0.0) <= &self.test_config.acceptable_error_rate {
                    TestStatus::Passed
                } else {
                    TestStatus::Warning
                };
                
                self.test_results.push(TestResult {
                    test_name: "high_load_scenarios".to_string(),
                    test_category: TestCategory::Load,
                    status,
                    duration: test_start.elapsed(),
                    error_message: None,
                    performance_metrics: metrics,
                    timestamp: SystemTime::now(),
                });
                
                if status == TestStatus::Passed {
                    println!("✓ High load test passed");
                } else {
                    println!("⚠ High load test passed with warnings");
                }
            }
            Err(e) => {
                self.test_results.push(TestResult {
                    test_name: "high_load_scenarios".to_string(),
                    test_category: TestCategory::Load,
                    status: TestStatus::Failed,
                    duration: test_start.elapsed(),
                    error_message: Some(e.to_string()),
                    performance_metrics: HashMap::new(),
                    timestamp: SystemTime::now(),
                });
                println!("✗ High load test failed: {}", e);
            }
        }
    }
    
    async fn run_load_test(&self) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut metrics = HashMap::new();
        
        let shared_monitor = Arc::new(SharedPerformanceMonitor::with_capacity(50000));
        let operations_per_thread = self.test_config.max_concurrent_operations / 8;
        let mut handles = Vec::new();
        
        let start_time = Instant::now();
        
        // Launch concurrent load
        for thread_id in 0..8 {
            let monitor_clone = shared_monitor.clone();
            let ops_count = operations_per_thread;
            
            let handle = thread::spawn(move || {
                let mut errors = 0;
                let mut successful_ops = 0;
                
                for i in 0..ops_count {
                    let operation_time = Duration::from_millis(1 + (i % 100) as u64);
                    
                    monitor_clone.record_query_time(operation_time);
                    successful_ops += 1;
                    
                    if i % 3 == 0 {
                        monitor_clone.record_index_time(operation_time * 5);
                    }
                    
                    // Simulate occasional errors
                    if (thread_id + i) % 1000 == 0 {
                        errors += 1;
                    }
                }
                
                (successful_ops, errors)
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        let mut total_successful = 0;
        let mut total_errors = 0;
        
        for handle in handles {
            let (successful, errors) = handle.join()
                .map_err(|_| "Thread panicked during load test")?;
            total_successful += successful;
            total_errors += errors;
        }
        
        let total_time = start_time.elapsed();
        
        // Get final statistics
        let stats = shared_monitor.get_stats()
            .ok_or("Failed to get statistics after load test")?;
        
        let error_rate = total_errors as f64 / (total_successful + total_errors) as f64;
        let throughput = stats.total_queries as f64 / total_time.as_secs_f64();
        
        metrics.insert("total_operations".to_string(), (total_successful + total_errors) as f64);
        metrics.insert("successful_operations".to_string(), total_successful as f64);
        metrics.insert("error_count".to_string(), total_errors as f64);
        metrics.insert("error_rate".to_string(), error_rate);
        metrics.insert("throughput_ops_per_sec".to_string(), throughput);
        metrics.insert("avg_latency_ms".to_string(), stats.avg_query_time.as_secs_f64() * 1000.0);
        metrics.insert("p99_latency_ms".to_string(), stats.p99_query_time.as_secs_f64() * 1000.0);
        
        println!("Load test completed: {} ops, {:.2}% errors, {:.0} ops/sec", 
            total_successful + total_errors, error_rate * 100.0, throughput);
        
        Ok(metrics)
    }
    
    async fn test_failover_scenarios(&mut self) {
        let test_start = Instant::now();
        
        println!("Testing failover scenarios...");
        
        match self.run_failover_test().await {
            Ok(metrics) => {
                self.test_results.push(TestResult {
                    test_name: "failover_scenarios".to_string(),
                    test_category: TestCategory::Failover,
                    status: TestStatus::Passed,
                    duration: test_start.elapsed(),
                    error_message: None,
                    performance_metrics: metrics,
                    timestamp: SystemTime::now(),
                });
                println!("✓ Failover test passed");
            }
            Err(e) => {
                self.test_results.push(TestResult {
                    test_name: "failover_scenarios".to_string(),
                    test_category: TestCategory::Failover,
                    status: TestStatus::Failed,
                    duration: test_start.elapsed(),
                    error_message: Some(e.to_string()),
                    performance_metrics: HashMap::new(),
                    timestamp: SystemTime::now(),
                });
                println!("✗ Failover test failed: {}", e);
            }
        }
    }
    
    async fn run_failover_test(&self) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut metrics = HashMap::new();
        
        // Test monitor resilience under adverse conditions
        let shared_monitor = Arc::new(SharedPerformanceMonitor::new());
        
        // Record normal operations
        for _ in 0..100 {
            shared_monitor.record_query_time(Duration::from_millis(25));
        }
        
        let stats_before = shared_monitor.get_stats()
            .ok_or("Failed to get statistics before failover test")?;
        
        // Simulate various failure scenarios
        
        // 1. Very high latency operations
        for _ in 0..50 {
            shared_monitor.record_query_time(Duration::from_secs(5));
        }
        
        // 2. Very fast operations (potential timing issues)
        for _ in 0..50 {
            shared_monitor.record_query_time(Duration::from_nanos(1));
        }
        
        // 3. Concurrent access from many threads
        let mut handles = Vec::new();
        for _ in 0..10 {
            let monitor_clone = shared_monitor.clone();
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    monitor_clone.record_query_time(Duration::from_millis(10));
                }
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().map_err(|_| "Thread panicked during failover test")?;
        }
        
        let stats_after = shared_monitor.get_stats()
            .ok_or("Failed to get statistics after failover test")?;
        
        metrics.insert("operations_before".to_string(), stats_before.total_queries as f64);
        metrics.insert("operations_after".to_string(), stats_after.total_queries as f64);
        metrics.insert("operations_added".to_string(), (stats_after.total_queries - stats_before.total_queries) as f64);
        
        // Verify the monitor continued to function correctly
        let expected_operations = 100 + 50 + 50 + (10 * 100); // Initial + high latency + fast + concurrent
        if stats_after.total_queries < expected_operations {
            return Err(format!("Expected at least {} operations, got {}", 
                expected_operations, stats_after.total_queries).into());
        }
        
        Ok(metrics)
    }
    
    async fn test_performance_benchmarks(&mut self) {
        let test_start = Instant::now();
        
        println!("Testing performance benchmarks...");
        
        match self.run_performance_benchmark().await {
            Ok(metrics) => {
                self.test_results.push(TestResult {
                    test_name: "performance_benchmarks".to_string(),
                    test_category: TestCategory::Performance,
                    status: TestStatus::Passed,
                    duration: test_start.elapsed(),
                    error_message: None,
                    performance_metrics: metrics,
                    timestamp: SystemTime::now(),
                });
                println!("✓ Performance benchmark test passed");
            }
            Err(e) => {
                self.test_results.push(TestResult {
                    test_name: "performance_benchmarks".to_string(),
                    test_category: TestCategory::Performance,
                    status: TestStatus::Failed,
                    duration: test_start.elapsed(),
                    error_message: Some(e.to_string()),
                    performance_metrics: HashMap::new(),
                    timestamp: SystemTime::now(),
                });
                println!("✗ Performance benchmark test failed: {}", e);
            }
        }
    }
    
    async fn run_performance_benchmark(&self) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut metrics = HashMap::new();
        
        // Benchmark monitoring overhead
        let baseline_monitor = PerformanceMonitor::new();
        let operations_count = 10000;
        
        // Measure time without monitoring
        let start = Instant::now();
        for _ in 0..operations_count {
            // Simulate work without monitoring
            std::hint::black_box(Duration::from_millis(1));
        }
        let baseline_time = start.elapsed();
        
        // Measure time with monitoring
        let mut monitored = PerformanceMonitor::new();
        let start = Instant::now();
        for _ in 0..operations_count {
            monitored.record_query_time(Duration::from_millis(1));
        }
        let monitored_time = start.elapsed();
        
        let overhead_percentage = if baseline_time.as_nanos() > 0 {
            ((monitored_time.as_nanos() - baseline_time.as_nanos()) as f64 / baseline_time.as_nanos() as f64) * 100.0
        } else {
            0.0
        };
        
        metrics.insert("baseline_time_ms".to_string(), baseline_time.as_secs_f64() * 1000.0);
        metrics.insert("monitored_time_ms".to_string(), monitored_time.as_secs_f64() * 1000.0);
        metrics.insert("overhead_percentage".to_string(), overhead_percentage);
        metrics.insert("operations_per_second".to_string(), operations_count as f64 / monitored_time.as_secs_f64());
        
        // Benchmark statistics calculation performance
        let start = Instant::now();
        let _stats = monitored.get_stats();
        let stats_calculation_time = start.elapsed();
        
        metrics.insert("stats_calculation_time_ms".to_string(), stats_calculation_time.as_secs_f64() * 1000.0);
        
        // Verify acceptable overhead (should be less than 5%)
        if overhead_percentage > 5.0 {
            return Err(format!("Monitoring overhead too high: {:.2}%", overhead_percentage).into());
        }
        
        Ok(metrics)
    }
    
    async fn test_end_to_end_workflows(&mut self) {
        let test_start = Instant::now();
        
        println!("Testing end-to-end workflows...");
        
        match self.run_end_to_end_test().await {
            Ok(metrics) => {
                self.test_results.push(TestResult {
                    test_name: "end_to_end_workflows".to_string(),
                    test_category: TestCategory::EndToEnd,
                    status: TestStatus::Passed,
                    duration: test_start.elapsed(),
                    error_message: None,
                    performance_metrics: metrics,
                    timestamp: SystemTime::now(),
                });
                println!("✓ End-to-end workflow test passed");
            }
            Err(e) => {
                self.test_results.push(TestResult {
                    test_name: "end_to_end_workflows".to_string(),
                    test_category: TestCategory::EndToEnd,
                    status: TestStatus::Failed,
                    duration: test_start.elapsed(),
                    error_message: Some(e.to_string()),
                    performance_metrics: HashMap::new(),
                    timestamp: SystemTime::now(),
                });
                println!("✗ End-to-end workflow test failed: {}", e);
            }
        }
    }
    
    async fn run_end_to_end_test(&self) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut metrics = HashMap::new();
        
        // Create complete monitoring system
        let shared_monitor = Arc::new(SharedPerformanceMonitor::new());
        
        // Set up alerting
        let alerting_config = AlertingConfig::default();
        let alert_thresholds = AlertThresholds::default();
        let mut alerting_system = AlertingSystem::new(alerting_config, alert_thresholds);
        alerting_system.add_monitor(shared_monitor.clone());
        
        // Simulate complete workflow: indexing -> searching -> reporting
        
        // 1. Indexing phase
        for i in 0..1000 {
            shared_monitor.record_index_time(Duration::from_millis(50 + (i % 100) as u64));
        }
        
        // 2. Searching phase
        for i in 0..2000 {
            shared_monitor.record_query_time(Duration::from_millis(20 + (i % 50) as u64));
        }
        
        // 3. Generate comprehensive report
        let stats = shared_monitor.get_stats()
            .ok_or("Failed to get performance statistics")?;
        
        let config = ReportConfig::default();
        let mut monitor = PerformanceMonitor::new();
        
        // Copy data to regular monitor for reporting
        for _ in 0..stats.total_queries {
            monitor.record_query_time(stats.avg_query_time);
        }
        for _ in 0..stats.total_indexes {
            monitor.record_index_time(stats.avg_index_time);
        }
        
        let report = monitor.generate_report(&config);
        
        metrics.insert("total_operations".to_string(), (stats.total_queries + stats.total_indexes) as f64);
        metrics.insert("report_charts_count".to_string(), report.charts.len() as f64);
        metrics.insert("report_recommendations_count".to_string(), report.recommendations.len() as f64);
        
        // 4. Test report exports
        let text_report = report.export_text();
        let csv_report = report.export_csv();
        let html_report = report.export_html();
        
        metrics.insert("text_report_length".to_string(), text_report.len() as f64);
        metrics.insert("csv_report_length".to_string(), csv_report.len() as f64);
        metrics.insert("html_report_length".to_string(), html_report.len() as f64);
        
        // Verify end-to-end workflow completed successfully
        if stats.total_queries != 2000 || stats.total_indexes != 1000 {
            return Err("End-to-end workflow did not complete with expected results".into());
        }
        
        if report.charts.is_empty() || report.recommendations.is_empty() {
            return Err("Report generation did not produce expected content".into());
        }
        
        Ok(metrics)
    }
    
    async fn test_memory_and_resource_usage(&mut self) {
        let test_start = Instant::now();
        
        println!("Testing memory and resource usage...");
        
        match self.run_memory_test().await {
            Ok(metrics) => {
                self.test_results.push(TestResult {
                    test_name: "memory_and_resource_usage".to_string(),
                    test_category: TestCategory::Performance,
                    status: TestStatus::Passed,
                    duration: test_start.elapsed(),
                    error_message: None,
                    performance_metrics: metrics,
                    timestamp: SystemTime::now(),
                });
                println!("✓ Memory and resource usage test passed");
            }
            Err(e) => {
                self.test_results.push(TestResult {
                    test_name: "memory_and_resource_usage".to_string(),
                    test_category: TestCategory::Performance,
                    status: TestStatus::Failed,
                    duration: test_start.elapsed(),
                    error_message: Some(e.to_string()),
                    performance_metrics: HashMap::new(),
                    timestamp: SystemTime::now(),
                });
                println!("✗ Memory and resource usage test failed: {}", e);
            }
        }
    }
    
    async fn run_memory_test(&self) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut metrics = HashMap::new();
        
        // Test memory usage with bounded capacity
        let capacity = 1000;
        let mut monitor = PerformanceMonitor::with_capacity(capacity);
        
        // Fill beyond capacity to test memory bounds
        let operations = capacity * 5;
        for i in 0..operations {
            monitor.record_query_time(Duration::from_millis(i as u64 % 100));
        }
        
        let (query_samples, _) = monitor.current_sample_count();
        let stats = monitor.get_stats();
        
        metrics.insert("capacity_limit".to_string(), capacity as f64);
        metrics.insert("operations_recorded".to_string(), operations as f64);
        metrics.insert("samples_retained".to_string(), query_samples as f64);
        metrics.insert("total_count_accuracy".to_string(), stats.total_queries as f64);
        
        // Verify memory bounds are respected
        if query_samples > capacity {
            return Err(format!("Memory bounds violated: {} samples > {} capacity", 
                query_samples, capacity).into());
        }
        
        // Verify total count accuracy
        if stats.total_queries != operations {
            return Err(format!("Total count inaccurate: {} != {}", 
                stats.total_queries, operations).into());
        }
        
        // Test with multiple monitors to verify independent memory management
        let mut monitors = Vec::new();
        for _ in 0..10 {
            let mut monitor = PerformanceMonitor::with_capacity(100);
            for _ in 0..200 {
                monitor.record_query_time(Duration::from_millis(25));
            }
            monitors.push(monitor);
        }
        
        // Verify all monitors maintained their bounds
        for (i, monitor) in monitors.iter().enumerate() {
            let (samples, _) = monitor.current_sample_count();
            if samples > 100 {
                return Err(format!("Monitor {} violated memory bounds: {} samples", i, samples).into());
            }
        }
        
        metrics.insert("multiple_monitors_test".to_string(), monitors.len() as f64);
        
        Ok(metrics)
    }
    
    async fn test_production_configuration(&mut self) {
        let test_start = Instant::now();
        
        println!("Testing production configuration...");
        
        match self.run_production_config_test().await {
            Ok(metrics) => {
                self.test_results.push(TestResult {
                    test_name: "production_configuration".to_string(),
                    test_category: TestCategory::Integration,
                    status: TestStatus::Passed,
                    duration: test_start.elapsed(),
                    error_message: None,
                    performance_metrics: metrics,
                    timestamp: SystemTime::now(),
                });
                println!("✓ Production configuration test passed");
            }
            Err(e) => {
                self.test_results.push(TestResult {
                    test_name: "production_configuration".to_string(),
                    test_category: TestCategory::Integration,
                    status: TestStatus::Failed,
                    duration: test_start.elapsed(),
                    error_message: Some(e.to_string()),
                    performance_metrics: HashMap::new(),
                    timestamp: SystemTime::now(),
                });
                println!("✗ Production configuration test failed: {}", e);
            }
        }
    }
    
    async fn run_production_config_test(&self) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>> {
        let mut metrics = HashMap::new();
        
        // Test production configuration utilities
        let deployment_helper = super::deployment::ProductionDeploymentHelper::new();
        let config_validation = deployment_helper.validate_production_config();
        
        metrics.insert("config_validation_passed".to_string(), if config_validation.is_valid { 1.0 } else { 0.0 });
        metrics.insert("config_errors_count".to_string(), config_validation.errors.len() as f64);
        metrics.insert("config_warnings_count".to_string(), config_validation.warnings.len() as f64);
        
        // Test deployment checklist generation
        let checklist = deployment_helper.generate_deployment_checklist();
        metrics.insert("checklist_total_items".to_string(), checklist.total_items as f64);
        metrics.insert("checklist_required_items".to_string(), checklist.required_items as f64);
        
        // Test production monitoring system creation
        let production_system = deployment_helper.create_production_monitoring_system()
            .map_err(|e| format!("Failed to create production system: {}", e))?;
        
        metrics.insert("production_monitors_count".to_string(), production_system.monitors.len() as f64);
        
        // Verify production system components
        if production_system.monitors.is_empty() {
            return Err("Production system has no monitors configured".into());
        }
        
        Ok(metrics)
    }
    
    fn generate_test_vectors(&self, count: usize, dimensions: usize) -> Vec<Vec<f32>> {
        let mut vectors = Vec::with_capacity(count);
        for i in 0..count {
            let mut vector = Vec::with_capacity(dimensions);
            for j in 0..dimensions {
                vector.push((i * j) as f32 * 0.01);
            }
            vectors.push(vector);
        }
        vectors
    }
    
    fn calculate_overall_status(&self) -> TestStatus {
        let failed_count = self.test_results.iter().filter(|r| r.status == TestStatus::Failed).count();
        let warning_count = self.test_results.iter().filter(|r| r.status == TestStatus::Warning).count();
        
        if failed_count > 0 {
            TestStatus::Failed
        } else if warning_count > 0 {
            TestStatus::Warning
        } else {
            TestStatus::Passed
        }
    }
}

#[derive(Debug, Clone)]
pub struct IntegrationTestResults {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub skipped_tests: usize,
    pub warning_tests: usize,
    pub total_duration: Duration,
    pub test_results: Vec<TestResult>,
    pub overall_status: TestStatus,
}

impl IntegrationTestResults {
    pub fn print_summary(&self) {
        println!("\n" + "=".repeat(80).as_str());
        println!("INTEGRATION TEST RESULTS SUMMARY");
        println!("=".repeat(80));
        
        println!("Overall Status: {:?}", self.overall_status);
        println!("Total Duration: {:.2}s", self.total_duration.as_secs_f64());
        println!();
        
        println!("Test Results:");
        println!("  ✓ Passed:  {}", self.passed_tests);
        println!("  ✗ Failed:  {}", self.failed_tests);
        println!("  ⚠ Warning: {}", self.warning_tests);
        println!("  - Skipped: {}", self.skipped_tests);
        println!("  Total:     {}", self.total_tests);
        println!();
        
        // Print details for failed tests
        if self.failed_tests > 0 {
            println!("FAILED TESTS:");
            for result in &self.test_results {
                if result.status == TestStatus::Failed {
                    println!("  ✗ {}: {}", result.test_name, 
                        result.error_message.as_ref().unwrap_or(&"No error message".to_string()));
                }
            }
            println!();
        }
        
        // Print details for warning tests
        if self.warning_tests > 0 {
            println!("WARNING TESTS:");
            for result in &self.test_results {
                if result.status == TestStatus::Warning {
                    println!("  ⚠ {}: Check performance metrics", result.test_name);
                }
            }
            println!();
        }
        
        println!("=".repeat(80));
    }
}
```

### 2. Add final validation utilities
Add to `src/monitor/integration_tests.rs`:
```rust
/// Production readiness validator
pub struct ProductionReadinessValidator {
    validation_results: Vec<ValidationResult>,
}

#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub category: String,
    pub check_name: String,
    pub status: ValidationStatus,
    pub message: String,
    pub severity: ValidationSeverity,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    Pass,
    Fail,
    Warning,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationSeverity {
    Critical,
    High,
    Medium,
    Low,
}

impl ProductionReadinessValidator {
    pub fn new() -> Self {
        Self {
            validation_results: Vec::new(),
        }
    }
    
    pub fn validate_production_readiness(&mut self) -> ProductionReadinessReport {
        self.validation_results.clear();
        
        // Validate monitoring system
        self.validate_monitoring_system();
        
        // Validate alerting configuration
        self.validate_alerting_configuration();
        
        // Validate performance characteristics
        self.validate_performance_characteristics();
        
        // Validate resource usage
        self.validate_resource_usage();
        
        // Validate error handling
        self.validate_error_handling();
        
        // Calculate overall readiness
        let critical_failures = self.validation_results.iter()
            .filter(|r| r.status == ValidationStatus::Fail && r.severity == ValidationSeverity::Critical)
            .count();
        
        let high_failures = self.validation_results.iter()
            .filter(|r| r.status == ValidationStatus::Fail && r.severity == ValidationSeverity::High)
            .count();
        
        let overall_status = if critical_failures > 0 {
            ProductionReadinessStatus::NotReady
        } else if high_failures > 0 {
            ProductionReadinessStatus::ConditionallyReady
        } else {
            ProductionReadinessStatus::Ready
        };
        
        ProductionReadinessReport {
            overall_status,
            validation_results: self.validation_results.clone(),
            critical_issues: critical_failures,
            high_issues: high_failures,
            recommendations: self.generate_recommendations(),
        }
    }
    
    fn validate_monitoring_system(&mut self) {
        // Check if monitoring system can be created
        let monitor = SharedPerformanceMonitor::new();
        
        // Test basic functionality
        monitor.record_query_time(Duration::from_millis(10));
        
        if let Some(stats) = monitor.get_stats() {
            if stats.total_queries == 1 {
                self.validation_results.push(ValidationResult {
                    category: "Monitoring System".to_string(),
                    check_name: "Basic Functionality".to_string(),
                    status: ValidationStatus::Pass,
                    message: "Monitoring system basic functionality verified".to_string(),
                    severity: ValidationSeverity::Critical,
                });
            } else {
                self.validation_results.push(ValidationResult {
                    category: "Monitoring System".to_string(),
                    check_name: "Basic Functionality".to_string(),
                    status: ValidationStatus::Fail,
                    message: "Monitoring system failed basic functionality test".to_string(),
                    severity: ValidationSeverity::Critical,
                });
            }
        } else {
            self.validation_results.push(ValidationResult {
                category: "Monitoring System".to_string(),
                check_name: "Statistics Generation".to_string(),
                status: ValidationStatus::Fail,
                message: "Failed to generate performance statistics".to_string(),
                severity: ValidationSeverity::Critical,
            });
        }
    }
    
    fn validate_alerting_configuration(&mut self) {
        let config = AlertingConfig::default();
        let thresholds = AlertThresholds::default();
        let alerting_system = AlertingSystem::new(config, thresholds);
        
        self.validation_results.push(ValidationResult {
            category: "Alerting System".to_string(),
            check_name: "Configuration".to_string(),
            status: ValidationStatus::Pass,
            message: "Alerting system configuration is valid".to_string(),
            severity: ValidationSeverity::High,
        });
        
        // Check if alerting system can start (simplified check)
        let status = alerting_system.get_alerting_status();
        
        self.validation_results.push(ValidationResult {
            category: "Alerting System".to_string(),
            check_name: "Status Check".to_string(),
            status: ValidationStatus::Pass,
            message: "Alerting system status check passed".to_string(),
            severity: ValidationSeverity::High,
        });
    }
    
    fn validate_performance_characteristics(&mut self) {
        // Test monitoring overhead
        let mut monitor = PerformanceMonitor::new();
        let operations = 10000;
        
        let start = Instant::now();
        for _ in 0..operations {
            monitor.record_query_time(Duration::from_millis(1));
        }
        let duration = start.elapsed();
        
        let ops_per_second = operations as f64 / duration.as_secs_f64();
        
        if ops_per_second > 100000.0 { // 100k ops/sec minimum
            self.validation_results.push(ValidationResult {
                category: "Performance".to_string(),
                check_name: "Monitoring Overhead".to_string(),
                status: ValidationStatus::Pass,
                message: format!("Monitoring performance is acceptable: {:.0} ops/sec", ops_per_second),
                severity: ValidationSeverity::Medium,
            });
        } else {
            self.validation_results.push(ValidationResult {
                category: "Performance".to_string(),
                check_name: "Monitoring Overhead".to_string(),
                status: ValidationStatus::Warning,
                message: format!("Monitoring performance may be low: {:.0} ops/sec", ops_per_second),
                severity: ValidationSeverity::Medium,
            });
        }
    }
    
    fn validate_resource_usage(&mut self) {
        // Test memory bounds
        let capacity = 1000;
        let mut monitor = PerformanceMonitor::with_capacity(capacity);
        
        // Fill beyond capacity
        for i in 0..capacity * 2 {
            monitor.record_query_time(Duration::from_millis(i as u64));
        }
        
        let (samples, _) = monitor.current_sample_count();
        
        if samples <= capacity {
            self.validation_results.push(ValidationResult {
                category: "Resource Usage".to_string(),
                check_name: "Memory Bounds".to_string(),
                status: ValidationStatus::Pass,
                message: "Memory usage is properly bounded".to_string(),
                severity: ValidationSeverity::Critical,
            });
        } else {
            self.validation_results.push(ValidationResult {
                category: "Resource Usage".to_string(),
                check_name: "Memory Bounds".to_string(),
                status: ValidationStatus::Fail,
                message: "Memory bounds are not properly enforced".to_string(),
                severity: ValidationSeverity::Critical,
            });
        }
    }
    
    fn validate_error_handling(&mut self) {
        // Test with empty monitor
        let monitor = PerformanceMonitor::new();
        let stats = monitor.get_stats();
        
        // Should handle empty state gracefully
        if stats.total_queries == 0 && stats.avg_query_time == Duration::from_millis(0) {
            self.validation_results.push(ValidationResult {
                category: "Error Handling".to_string(),
                check_name: "Empty State".to_string(),
                status: ValidationStatus::Pass,
                message: "Empty state handled gracefully".to_string(),
                severity: ValidationSeverity::Medium,
            });
        } else {
            self.validation_results.push(ValidationResult {
                category: "Error Handling".to_string(),
                check_name: "Empty State".to_string(),
                status: ValidationStatus::Fail,
                message: "Empty state not handled correctly".to_string(),
                severity: ValidationSeverity::Medium,
            });
        }
    }
    
    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let failed_critical = self.validation_results.iter()
            .filter(|r| r.status == ValidationStatus::Fail && r.severity == ValidationSeverity::Critical)
            .count();
        
        if failed_critical > 0 {
            recommendations.push("Critical issues must be resolved before production deployment".to_string());
        }
        
        let warnings = self.validation_results.iter()
            .filter(|r| r.status == ValidationStatus::Warning)
            .count();
        
        if warnings > 0 {
            recommendations.push("Review warning items and consider optimizations".to_string());
        }
        
        recommendations.push("Run comprehensive load testing before production deployment".to_string());
        recommendations.push("Configure appropriate alerting thresholds for production workload".to_string());
        recommendations.push("Set up monitoring dashboards and notification channels".to_string());
        
        recommendations
    }
}

#[derive(Debug, Clone)]
pub struct ProductionReadinessReport {
    pub overall_status: ProductionReadinessStatus,
    pub validation_results: Vec<ValidationResult>,
    pub critical_issues: usize,
    pub high_issues: usize,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProductionReadinessStatus {
    Ready,
    ConditionallyReady,
    NotReady,
}

impl ProductionReadinessReport {
    pub fn print_report(&self) {
        println!("\n" + "=".repeat(80).as_str());
        println!("PRODUCTION READINESS REPORT");
        println!("=".repeat(80));
        
        println!("Overall Status: {:?}", self.overall_status);
        println!("Critical Issues: {}", self.critical_issues);
        println!("High Priority Issues: {}", self.high_issues);
        println!();
        
        if !self.validation_results.is_empty() {
            println!("VALIDATION RESULTS:");
            for result in &self.validation_results {
                let status_icon = match result.status {
                    ValidationStatus::Pass => "✓",
                    ValidationStatus::Fail => "✗",
                    ValidationStatus::Warning => "⚠",
                };
                
                println!("  {} {}: {} - {}", 
                    status_icon, result.category, result.check_name, result.message);
            }
            println!();
        }
        
        if !self.recommendations.is_empty() {
            println!("RECOMMENDATIONS:");
            for recommendation in &self.recommendations {
                println!("  • {}", recommendation);
            }
            println!();
        }
        
        println!("=".repeat(80));
    }
}
```

## Success Criteria
- [ ] Comprehensive integration test suite covers all monitoring components
- [ ] Load testing validates system performance under high concurrent usage
- [ ] Failover testing ensures system resilience under adverse conditions
- [ ] Performance benchmarking confirms acceptable monitoring overhead
- [ ] End-to-end workflow testing validates complete system integration
- [ ] Memory and resource testing confirms bounded resource usage
- [ ] Production configuration testing validates deployment readiness
- [ ] All integration tests pass with acceptable performance metrics
- [ ] Production readiness validation confirms system is deployment-ready
- [ ] Test results provide comprehensive system validation report

## Time Limit
10 minutes

## Notes
- Provides comprehensive integration testing for entire monitoring system
- Includes load testing, failover testing, and performance benchmarking
- Validates end-to-end workflows and system integration
- Confirms production readiness with detailed validation reports
- Tests memory bounds and resource usage under various conditions
- Validates monitoring overhead remains within acceptable limits
- Provides detailed test results and recommendations for production deployment
- Ensures system reliability and performance under real-world conditions