//! Performance Metrics Collection System
//! 
//! Provides comprehensive performance measurement and analysis capabilities.

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// Comprehensive performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Latency statistics
    pub latency_stats: LatencyStats,
    /// Memory usage statistics
    pub memory_stats: MemoryStats,
    /// CPU usage statistics
    pub cpu_stats: CpuStats,
    /// I/O statistics
    pub io_stats: IoStats,
    /// Network statistics
    pub network_stats: NetworkStats,
    /// Custom application metrics
    pub custom_metrics: HashMap<String, f64>,
    /// Timestamp when metrics were collected
    pub timestamp: SystemTime,
    /// Duration over which metrics were collected
    pub collection_duration: Duration,
}

/// Latency measurement statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Minimum latency observed
    pub min: Duration,
    /// Maximum latency observed
    pub max: Duration,
    /// Mean (average) latency
    pub mean: Duration,
    /// Median latency
    pub median: Duration,
    /// 95th percentile latency
    pub p95: Duration,
    /// 99th percentile latency
    pub p99: Duration,
    /// 99.9th percentile latency
    pub p999: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Total number of measurements
    pub count: u64,
    /// Sum of all measurements
    pub sum: Duration,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Peak resident set size in bytes
    pub peak_rss_bytes: u64,
    /// Average resident set size in bytes
    pub average_rss_bytes: u64,
    /// Current heap allocations
    pub heap_allocations: u64,
    /// Current heap deallocations
    pub heap_deallocations: u64,
    /// Peak heap size in bytes
    pub peak_heap_bytes: u64,
    /// Current heap size in bytes
    pub current_heap_bytes: u64,
    /// Stack size in bytes
    pub stack_size_bytes: u64,
    /// Virtual memory size in bytes
    pub virtual_memory_bytes: u64,
    /// Page faults (major)
    pub major_page_faults: u64,
    /// Page faults (minor)
    pub minor_page_faults: u64,
}

/// CPU usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuStats {
    /// Total CPU time used
    pub total_cpu_time: Duration,
    /// User mode CPU time
    pub user_cpu_time: Duration,
    /// System mode CPU time
    pub system_cpu_time: Duration,
    /// Average CPU utilization percentage
    pub average_cpu_percent: f64,
    /// Peak CPU utilization percentage
    pub peak_cpu_percent: f64,
    /// Number of context switches
    pub context_switches: u64,
    /// Number of CPU cores utilized
    pub cores_used: u32,
    /// CPU cache misses
    pub cache_misses: u64,
    /// CPU cache hits
    pub cache_hits: u64,
}

/// I/O operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoStats {
    /// Total bytes read from disk
    pub disk_read_bytes: u64,
    /// Total bytes written to disk
    pub disk_write_bytes: u64,
    /// Number of read operations
    pub read_operations: u64,
    /// Number of write operations
    pub write_operations: u64,
    /// Average read latency
    pub avg_read_latency: Duration,
    /// Average write latency
    pub avg_write_latency: Duration,
    /// I/O wait time
    pub io_wait_time: Duration,
    /// Sequential read percentage
    pub sequential_read_percent: f64,
    /// Random read percentage
    pub random_read_percent: f64,
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Number of packets sent
    pub packets_sent: u64,
    /// Number of packets received
    pub packets_received: u64,
    /// Network errors
    pub network_errors: u64,
    /// Connection count
    pub connections: u32,
    /// Average network latency
    pub avg_latency: Duration,
    /// Bandwidth utilization (bytes/sec)
    pub bandwidth_utilization: f64,
}

/// Real-time metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    /// Collection interval
    pub interval: Duration,
    /// Whether collection is active
    pub active: bool,
    /// Collected samples
    pub samples: Arc<Mutex<VecDeque<PerformanceMetrics>>>,
    /// Maximum number of samples to keep
    pub max_samples: usize,
    /// Latency measurements
    pub latency_measurements: Arc<Mutex<Vec<Duration>>>,
    /// Custom metrics storage
    pub custom_metrics: Arc<Mutex<HashMap<String, f64>>>,
    /// Collection start time
    pub start_time: Option<Instant>,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new(interval: Duration, max_samples: usize) -> Self {
        Self {
            interval,
            active: false,
            samples: Arc::new(Mutex::new(VecDeque::with_capacity(max_samples))),
            max_samples,
            latency_measurements: Arc::new(Mutex::new(Vec::new())),
            custom_metrics: Arc::new(Mutex::new(HashMap::new())),
            start_time: None,
        }
    }

    /// Start collecting metrics
    pub fn start(&mut self) -> Result<()> {
        if self.active {
            return Err(anyhow!("Metrics collection already active"));
        }

        self.active = true;
        self.start_time = Some(Instant::now());
        
        // Clear previous data
        if let Ok(mut samples) = self.samples.lock() {
            samples.clear();
        }
        if let Ok(mut measurements) = self.latency_measurements.lock() {
            measurements.clear();
        }
        if let Ok(mut customs) = self.custom_metrics.lock() {
            customs.clear();
        }

        Ok(())
    }

    /// Stop collecting metrics
    pub fn stop(&mut self) -> Result<Vec<PerformanceMetrics>> {
        if !self.active {
            return Err(anyhow!("Metrics collection not active"));
        }

        self.active = false;
        self.start_time = None;

        let samples = self.samples.lock()
            .map_err(|_| anyhow!("Failed to acquire samples lock"))?;
        
        Ok(samples.iter().cloned().collect())
    }

    /// Record a latency measurement
    pub fn record_latency(&self, latency: Duration) -> Result<()> {
        if !self.active {
            return Ok(()); // Silently ignore if not collecting
        }

        let mut measurements = self.latency_measurements.lock()
            .map_err(|_| anyhow!("Failed to acquire latency lock"))?;
        
        measurements.push(latency);
        
        // Limit the number of measurements to prevent memory growth
        if measurements.len() > 10000 {
            measurements.drain(0..5000); // Remove oldest half
        }

        Ok(())
    }

    /// Record a custom metric
    pub fn record_custom_metric(&self, name: String, value: f64) -> Result<()> {
        let mut customs = self.custom_metrics.lock()
            .map_err(|_| anyhow!("Failed to acquire custom metrics lock"))?;
        
        customs.insert(name, value);
        Ok(())
    }

    /// Collect current system metrics
    pub fn collect_system_metrics(&self) -> Result<PerformanceMetrics> {
        let timestamp = SystemTime::now();
        let collection_duration = self.start_time
            .map(|start| start.elapsed())
            .unwrap_or_default();

        // Collect latency statistics
        let latency_stats = self.compute_latency_stats()?;
        
        // Collect system metrics
        let memory_stats = self.collect_memory_stats()?;
        let cpu_stats = self.collect_cpu_stats()?;
        let io_stats = self.collect_io_stats()?;
        let network_stats = self.collect_network_stats()?;
        
        // Get custom metrics
        let custom_metrics = self.custom_metrics.lock()
            .map_err(|_| anyhow!("Failed to acquire custom metrics lock"))?
            .clone();

        Ok(PerformanceMetrics {
            latency_stats,
            memory_stats,
            cpu_stats,
            io_stats,
            network_stats,
            custom_metrics,
            timestamp,
            collection_duration,
        })
    }

    /// Sample current metrics and store them
    pub fn sample(&self) -> Result<()> {
        if !self.active {
            return Ok(());
        }

        let metrics = self.collect_system_metrics()?;
        
        let mut samples = self.samples.lock()
            .map_err(|_| anyhow!("Failed to acquire samples lock"))?;
        
        samples.push_back(metrics);
        
        // Maintain max samples limit
        while samples.len() > self.max_samples {
            samples.pop_front();
        }

        Ok(())
    }

    /// Compute latency statistics from measurements
    fn compute_latency_stats(&self) -> Result<LatencyStats> {
        let measurements = self.latency_measurements.lock()
            .map_err(|_| anyhow!("Failed to acquire latency lock"))?;

        if measurements.is_empty() {
            return Ok(LatencyStats {
                min: Duration::from_nanos(0),
                max: Duration::from_nanos(0),
                mean: Duration::from_nanos(0),
                median: Duration::from_nanos(0),
                p95: Duration::from_nanos(0),
                p99: Duration::from_nanos(0),
                p999: Duration::from_nanos(0),
                std_dev: Duration::from_nanos(0),
                count: 0,
                sum: Duration::from_nanos(0),
            });
        }

        let mut sorted_measurements = measurements.clone();
        sorted_measurements.sort();

        let count = sorted_measurements.len() as u64;
        let sum: Duration = sorted_measurements.iter().sum();
        let mean = sum / count as u32;

        let min = sorted_measurements[0];
        let max = sorted_measurements[sorted_measurements.len() - 1];
        
        let median = sorted_measurements[sorted_measurements.len() / 2];
        let p95 = sorted_measurements[(sorted_measurements.len() as f64 * 0.95) as usize];
        let p99 = sorted_measurements[(sorted_measurements.len() as f64 * 0.99) as usize];
        let p999 = sorted_measurements[(sorted_measurements.len() as f64 * 0.999) as usize];

        // Calculate standard deviation
        let variance: f64 = sorted_measurements.iter()
            .map(|&x| {
                let diff = x.as_nanos() as f64 - mean.as_nanos() as f64;
                diff * diff
            })
            .sum::<f64>() / count as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        Ok(LatencyStats {
            min,
            max,
            mean,
            median,
            p95,
            p99,
            p999,
            std_dev,
            count,
            sum,
        })
    }

    /// Collect memory statistics
    fn collect_memory_stats(&self) -> Result<MemoryStats> {
        // In a real implementation, this would use system APIs
        // For now, return mock realistic data
        Ok(MemoryStats {
            peak_rss_bytes: 50 * 1024 * 1024, // 50MB
            average_rss_bytes: 45 * 1024 * 1024, // 45MB
            heap_allocations: 1000,
            heap_deallocations: 950,
            peak_heap_bytes: 30 * 1024 * 1024, // 30MB
            current_heap_bytes: 25 * 1024 * 1024, // 25MB
            stack_size_bytes: 8 * 1024 * 1024, // 8MB
            virtual_memory_bytes: 100 * 1024 * 1024, // 100MB
            major_page_faults: 5,
            minor_page_faults: 1000,
        })
    }

    /// Collect CPU statistics
    fn collect_cpu_stats(&self) -> Result<CpuStats> {
        // Mock data for now
        Ok(CpuStats {
            total_cpu_time: Duration::from_millis(500),
            user_cpu_time: Duration::from_millis(400),
            system_cpu_time: Duration::from_millis(100),
            average_cpu_percent: 25.0,
            peak_cpu_percent: 80.0,
            context_switches: 1000,
            cores_used: 2,
            cache_misses: 5000,
            cache_hits: 95000,
        })
    }

    /// Collect I/O statistics
    fn collect_io_stats(&self) -> Result<IoStats> {
        // Mock data for now
        Ok(IoStats {
            disk_read_bytes: 1024 * 1024, // 1MB
            disk_write_bytes: 512 * 1024, // 512KB
            read_operations: 100,
            write_operations: 50,
            avg_read_latency: Duration::from_micros(100),
            avg_write_latency: Duration::from_micros(200),
            io_wait_time: Duration::from_millis(10),
            sequential_read_percent: 80.0,
            random_read_percent: 20.0,
        })
    }

    /// Collect network statistics
    fn collect_network_stats(&self) -> Result<NetworkStats> {
        // Mock data for now
        Ok(NetworkStats {
            bytes_sent: 10 * 1024, // 10KB
            bytes_received: 5 * 1024, // 5KB
            packets_sent: 100,
            packets_received: 80,
            network_errors: 0,
            connections: 2,
            avg_latency: Duration::from_millis(50),
            bandwidth_utilization: 1024.0, // 1KB/s
        })
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> Result<PerformanceSummary> {
        let samples = self.samples.lock()
            .map_err(|_| anyhow!("Failed to acquire samples lock"))?;

        if samples.is_empty() {
            return Ok(PerformanceSummary::default());
        }

        let sample_count = samples.len();
        
        // Aggregate statistics across samples
        let total_latency_count: u64 = samples.iter().map(|s| s.latency_stats.count).sum();
        let avg_memory_bytes: u64 = samples.iter()
            .map(|s| s.memory_stats.average_rss_bytes)
            .sum::<u64>() / sample_count as u64;
        let avg_cpu_percent: f64 = samples.iter()
            .map(|s| s.cpu_stats.average_cpu_percent)
            .sum::<f64>() / sample_count as f64;
        let total_io_read: u64 = samples.iter().map(|s| s.io_stats.disk_read_bytes).sum();
        let total_io_write: u64 = samples.iter().map(|s| s.io_stats.disk_write_bytes).sum();

        let collection_duration = samples.last()
            .map(|s| s.collection_duration)
            .unwrap_or_default();

        Ok(PerformanceSummary {
            sample_count,
            collection_duration,
            total_latency_measurements: total_latency_count,
            average_memory_bytes: avg_memory_bytes,
            average_cpu_percent: avg_cpu_percent,
            total_disk_read_bytes: total_io_read,
            total_disk_write_bytes: total_io_write,
            custom_metrics_count: samples.last()
                .map(|s| s.custom_metrics.len())
                .unwrap_or(0),
        })
    }

    /// Export metrics to JSON
    pub fn export_to_json(&self) -> Result<String> {
        let samples = self.samples.lock()
            .map_err(|_| anyhow!("Failed to acquire samples lock"))?;
        
        serde_json::to_string_pretty(&*samples)
            .map_err(|e| anyhow!("Failed to serialize metrics: {}", e))
    }

    /// Clear all collected metrics
    pub fn clear(&self) -> Result<()> {
        let mut samples = self.samples.lock()
            .map_err(|_| anyhow!("Failed to acquire samples lock"))?;
        samples.clear();

        let mut measurements = self.latency_measurements.lock()
            .map_err(|_| anyhow!("Failed to acquire latency lock"))?;
        measurements.clear();

        let mut customs = self.custom_metrics.lock()
            .map_err(|_| anyhow!("Failed to acquire custom metrics lock"))?;
        customs.clear();

        Ok(())
    }
}

/// Performance summary for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Number of samples collected
    pub sample_count: usize,
    /// Total collection duration
    pub collection_duration: Duration,
    /// Total latency measurements
    pub total_latency_measurements: u64,
    /// Average memory usage
    pub average_memory_bytes: u64,
    /// Average CPU usage
    pub average_cpu_percent: f64,
    /// Total disk reads
    pub total_disk_read_bytes: u64,
    /// Total disk writes
    pub total_disk_write_bytes: u64,
    /// Number of custom metrics
    pub custom_metrics_count: usize,
}

impl Default for PerformanceSummary {
    fn default() -> Self {
        Self {
            sample_count: 0,
            collection_duration: Duration::from_secs(0),
            total_latency_measurements: 0,
            average_memory_bytes: 0,
            average_cpu_percent: 0.0,
            total_disk_read_bytes: 0,
            total_disk_write_bytes: 0,
            custom_metrics_count: 0,
        }
    }
}

/// Metrics comparison for performance regression detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsComparison {
    /// Baseline metrics
    pub baseline: PerformanceMetrics,
    /// Current metrics
    pub current: PerformanceMetrics,
    /// Comparison results
    pub comparison: ComparisonResults,
}

/// Comparison results for different metric categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResults {
    /// Latency comparison
    pub latency: MetricComparison,
    /// Memory comparison
    pub memory: MetricComparison,
    /// CPU comparison
    pub cpu: MetricComparison,
    /// I/O comparison
    pub io: MetricComparison,
    /// Overall regression detected
    pub regression_detected: bool,
}

/// Individual metric comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    /// Percentage change (positive = increase, negative = decrease)
    pub percent_change: f64,
    /// Whether change is significant
    pub significant: bool,
    /// Whether change indicates regression
    pub regression: bool,
    /// Threshold that was applied
    pub threshold: f64,
    /// Human-readable description
    pub description: String,
}

impl MetricsComparison {
    /// Compare two performance metrics
    pub fn compare(baseline: PerformanceMetrics, current: PerformanceMetrics, 
                  thresholds: &ComparisonThresholds) -> Self {
        
        let latency_change = Self::calculate_percentage_change(
            baseline.latency_stats.mean.as_nanos() as f64,
            current.latency_stats.mean.as_nanos() as f64,
        );
        
        let memory_change = Self::calculate_percentage_change(
            baseline.memory_stats.average_rss_bytes as f64,
            current.memory_stats.average_rss_bytes as f64,
        );
        
        let cpu_change = Self::calculate_percentage_change(
            baseline.cpu_stats.average_cpu_percent,
            current.cpu_stats.average_cpu_percent,
        );
        
        let io_change = Self::calculate_percentage_change(
            (baseline.io_stats.disk_read_bytes + baseline.io_stats.disk_write_bytes) as f64,
            (current.io_stats.disk_read_bytes + current.io_stats.disk_write_bytes) as f64,
        );

        let latency = MetricComparison {
            percent_change: latency_change,
            significant: latency_change.abs() > thresholds.latency_threshold,
            regression: latency_change > thresholds.latency_threshold,
            threshold: thresholds.latency_threshold,
            description: format!("Latency changed by {:.2}%", latency_change),
        };

        let memory = MetricComparison {
            percent_change: memory_change,
            significant: memory_change.abs() > thresholds.memory_threshold,
            regression: memory_change > thresholds.memory_threshold,
            threshold: thresholds.memory_threshold,
            description: format!("Memory usage changed by {:.2}%", memory_change),
        };

        let cpu = MetricComparison {
            percent_change: cpu_change,
            significant: cpu_change.abs() > thresholds.cpu_threshold,
            regression: cpu_change > thresholds.cpu_threshold,
            threshold: thresholds.cpu_threshold,
            description: format!("CPU usage changed by {:.2}%", cpu_change),
        };

        let io = MetricComparison {
            percent_change: io_change,
            significant: io_change.abs() > thresholds.io_threshold,
            regression: io_change > thresholds.io_threshold,
            threshold: thresholds.io_threshold,
            description: format!("I/O usage changed by {:.2}%", io_change),
        };

        let regression_detected = latency.regression || memory.regression || 
                                 cpu.regression || io.regression;

        let comparison = ComparisonResults {
            latency,
            memory,
            cpu,
            io,
            regression_detected,
        };

        Self {
            baseline,
            current,
            comparison,
        }
    }

    /// Calculate percentage change between two values
    fn calculate_percentage_change(baseline: f64, current: f64) -> f64 {
        if baseline == 0.0 {
            if current == 0.0 {
                0.0
            } else {
                100.0 // Arbitrary large change for zero baseline
            }
        } else {
            ((current - baseline) / baseline) * 100.0
        }
    }
}

/// Thresholds for performance comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonThresholds {
    /// Latency change threshold (percentage)
    pub latency_threshold: f64,
    /// Memory change threshold (percentage)
    pub memory_threshold: f64,
    /// CPU change threshold (percentage)
    pub cpu_threshold: f64,
    /// I/O change threshold (percentage)
    pub io_threshold: f64,
}

impl Default for ComparisonThresholds {
    fn default() -> Self {
        Self {
            latency_threshold: 5.0,   // 5% increase is significant
            memory_threshold: 10.0,   // 10% increase is significant
            cpu_threshold: 15.0,      // 15% increase is significant
            io_threshold: 20.0,       // 20% increase is significant
        }
    }
}

/// Utility functions for metrics analysis
pub mod utils {
    use super::*;

    /// Create a metrics collector with default settings
    pub fn default_collector() -> MetricsCollector {
        MetricsCollector::new(Duration::from_millis(100), 1000)
    }

    /// Create a high-frequency metrics collector
    pub fn high_frequency_collector() -> MetricsCollector {
        MetricsCollector::new(Duration::from_millis(10), 10000)
    }

    /// Measure the performance of a closure
    pub fn measure_performance<T, F>(f: F) -> (T, Duration)
    where
        F: FnOnce() -> T,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Format duration for human-readable display
    pub fn format_duration(duration: Duration) -> String {
        let nanos = duration.as_nanos();
        
        if nanos < 1_000 {
            format!("{}ns", nanos)
        } else if nanos < 1_000_000 {
            format!("{:.2}μs", nanos as f64 / 1_000.0)
        } else if nanos < 1_000_000_000 {
            format!("{:.2}ms", nanos as f64 / 1_000_000.0)
        } else {
            format!("{:.2}s", nanos as f64 / 1_000_000_000.0)
        }
    }

    /// Format bytes for human-readable display
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_metrics_collector_creation() {
        let collector = MetricsCollector::new(Duration::from_millis(100), 1000);
        assert_eq!(collector.interval, Duration::from_millis(100));
        assert_eq!(collector.max_samples, 1000);
        assert!(!collector.active);
    }

    #[test]
    fn test_metrics_collection_lifecycle() {
        let mut collector = MetricsCollector::new(Duration::from_millis(10), 100);
        
        // Start collection
        assert!(collector.start().is_ok());
        assert!(collector.active);
        
        // Record some latency measurements
        for i in 0..10 {
            collector.record_latency(Duration::from_millis(i)).unwrap();
        }
        
        // Record custom metrics
        collector.record_custom_metric("test_metric".to_string(), 42.0).unwrap();
        
        // Sample metrics
        collector.sample().unwrap();
        
        // Stop collection
        let samples = collector.stop().unwrap();
        assert!(!collector.active);
        assert!(!samples.is_empty());
    }

    #[test]
    fn test_latency_statistics() {
        let mut collector = MetricsCollector::new(Duration::from_millis(10), 100);
        collector.start().unwrap();
        
        // Record known latency values
        let latencies = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]; // milliseconds
        for &lat in &latencies {
            collector.record_latency(Duration::from_millis(lat)).unwrap();
        }
        
        let stats = collector.compute_latency_stats().unwrap();
        
        assert_eq!(stats.count, latencies.len() as u64);
        assert_eq!(stats.min, Duration::from_millis(1));
        assert_eq!(stats.max, Duration::from_millis(10));
        assert_eq!(stats.median, Duration::from_millis(5));
    }

    #[test]
    fn test_custom_metrics() {
        let collector = MetricsCollector::new(Duration::from_millis(10), 100);
        
        collector.record_custom_metric("metric1".to_string(), 10.0).unwrap();
        collector.record_custom_metric("metric2".to_string(), 20.0).unwrap();
        
        let customs = collector.custom_metrics.lock().unwrap();
        assert_eq!(customs.get("metric1"), Some(&10.0));
        assert_eq!(customs.get("metric2"), Some(&20.0));
    }

    #[test]
    fn test_performance_summary() {
        let mut collector = MetricsCollector::new(Duration::from_millis(10), 100);
        collector.start().unwrap();
        
        // Record some data
        collector.record_latency(Duration::from_millis(5)).unwrap();
        collector.sample().unwrap();
        
        thread::sleep(Duration::from_millis(20));
        collector.sample().unwrap();
        
        let summary = collector.get_performance_summary().unwrap();
        assert_eq!(summary.sample_count, 2);
        assert!(summary.collection_duration > Duration::from_millis(10));
    }

    #[test]
    fn test_metrics_comparison() {
        let baseline = PerformanceMetrics {
            latency_stats: LatencyStats {
                mean: Duration::from_millis(10),
                min: Duration::from_millis(1),
                max: Duration::from_millis(20),
                median: Duration::from_millis(10),
                p95: Duration::from_millis(18),
                p99: Duration::from_millis(19),
                p999: Duration::from_millis(20),
                std_dev: Duration::from_millis(5),
                count: 100,
                sum: Duration::from_secs(1),
            },
            memory_stats: MemoryStats {
                average_rss_bytes: 1024 * 1024, // 1MB
                peak_rss_bytes: 2 * 1024 * 1024,
                heap_allocations: 100,
                heap_deallocations: 90,
                peak_heap_bytes: 1024 * 1024,
                current_heap_bytes: 512 * 1024,
                stack_size_bytes: 8 * 1024,
                virtual_memory_bytes: 10 * 1024 * 1024,
                major_page_faults: 1,
                minor_page_faults: 100,
            },
            cpu_stats: CpuStats {
                average_cpu_percent: 25.0,
                peak_cpu_percent: 50.0,
                total_cpu_time: Duration::from_millis(100),
                user_cpu_time: Duration::from_millis(80),
                system_cpu_time: Duration::from_millis(20),
                context_switches: 1000,
                cores_used: 2,
                cache_misses: 100,
                cache_hits: 9900,
            },
            io_stats: IoStats {
                disk_read_bytes: 1024,
                disk_write_bytes: 512,
                read_operations: 10,
                write_operations: 5,
                avg_read_latency: Duration::from_micros(100),
                avg_write_latency: Duration::from_micros(200),
                io_wait_time: Duration::from_millis(1),
                sequential_read_percent: 80.0,
                random_read_percent: 20.0,
            },
            network_stats: NetworkStats {
                bytes_sent: 1024,
                bytes_received: 512,
                packets_sent: 10,
                packets_received: 8,
                network_errors: 0,
                connections: 1,
                avg_latency: Duration::from_millis(10),
                bandwidth_utilization: 128.0,
            },
            custom_metrics: HashMap::new(),
            timestamp: SystemTime::now(),
            collection_duration: Duration::from_secs(1),
        };
        
        let mut current = baseline.clone();
        current.latency_stats.mean = Duration::from_millis(12); // 20% increase
        current.memory_stats.average_rss_bytes = 1200 * 1024; // ~17% increase
        
        let thresholds = ComparisonThresholds::default();
        let comparison = MetricsComparison::compare(baseline, current, &thresholds);
        
        assert!(comparison.comparison.latency.regression);
        assert!(comparison.comparison.memory.regression);
        assert!(comparison.comparison.regression_detected);
    }

    #[test]
    fn test_utils_functions() {
        // Test performance measurement
        let (result, duration) = utils::measure_performance(|| {
            thread::sleep(Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert!(duration >= Duration::from_millis(8)); // Allow some tolerance
        
        // Test formatting functions
        assert_eq!(utils::format_duration(Duration::from_nanos(500)), "500ns");
        assert_eq!(utils::format_duration(Duration::from_micros(1500)), "1.50ms");
        assert_eq!(utils::format_bytes(1536), "1.50 KB");
        assert_eq!(utils::format_bytes(1024 * 1024), "1.00 MB");
    }

    #[test]
    fn test_json_export() {
        let mut collector = MetricsCollector::new(Duration::from_millis(10), 100);
        collector.start().unwrap();
        
        collector.record_latency(Duration::from_millis(5)).unwrap();
        collector.sample().unwrap();
        
        let json = collector.export_to_json().unwrap();
        assert!(json.contains("latency_stats"));
        assert!(json.contains("memory_stats"));
    }

    #[test]
    fn test_collector_clear() {
        let mut collector = MetricsCollector::new(Duration::from_millis(10), 100);
        collector.start().unwrap();
        
        collector.record_latency(Duration::from_millis(5)).unwrap();
        collector.sample().unwrap();
        
        collector.clear().unwrap();
        
        let summary = collector.get_performance_summary().unwrap();
        assert_eq!(summary.sample_count, 0);
    }
}