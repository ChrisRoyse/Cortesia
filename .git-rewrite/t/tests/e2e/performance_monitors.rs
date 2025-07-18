//! Performance Monitoring for E2E Simulations
//! 
//! Comprehensive performance monitoring systems for end-to-end simulation validation.

use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use anyhow::Result;

/// E2E Performance Monitor for tracking workflow performance
pub struct E2EPerformanceMonitor {
    workflow_metrics: HashMap<String, WorkflowPerformanceMetrics>,
    resource_usage: ResourceUsageMetrics,
    start_time: Instant,
    monitoring_interval: Duration,
}

/// Performance metrics for individual workflows
#[derive(Debug, Clone)]
pub struct WorkflowPerformanceMetrics {
    pub workflow_name: String,
    pub execution_count: u32,
    pub total_execution_time: Duration,
    pub min_execution_time: Duration,
    pub max_execution_time: Duration,
    pub success_count: u32,
    pub error_count: u32,
    pub last_execution: Option<Instant>,
    pub throughput_samples: VecDeque<ThroughputSample>,
    pub latency_percentiles: LatencyPercentiles,
}

/// Resource usage tracking
#[derive(Debug, Clone)]
pub struct ResourceUsageMetrics {
    pub memory_samples: VecDeque<MemorySample>,
    pub cpu_samples: VecDeque<CpuSample>,
    pub disk_io_samples: VecDeque<DiskIoSample>,
    pub network_io_samples: VecDeque<NetworkIoSample>,
    pub peak_memory_usage: u64,
    pub peak_cpu_usage: f64,
}

/// Throughput measurement sample
#[derive(Debug, Clone)]
pub struct ThroughputSample {
    pub timestamp: Instant,
    pub operations_per_second: f64,
    pub concurrent_operations: u32,
}

/// Memory usage sample
#[derive(Debug, Clone)]
pub struct MemorySample {
    pub timestamp: Instant,
    pub total_memory_mb: u64,
    pub used_memory_mb: u64,
    pub heap_memory_mb: u64,
    pub off_heap_memory_mb: u64,
}

/// CPU usage sample
#[derive(Debug, Clone)]
pub struct CpuSample {
    pub timestamp: Instant,
    pub cpu_percentage: f64,
    pub system_load: f64,
    pub gc_time_ms: u64,
}

/// Disk I/O sample
#[derive(Debug, Clone)]
pub struct DiskIoSample {
    pub timestamp: Instant,
    pub read_bytes_per_sec: u64,
    pub write_bytes_per_sec: u64,
    pub read_ops_per_sec: u64,
    pub write_ops_per_sec: u64,
}

/// Network I/O sample
#[derive(Debug, Clone)]
pub struct NetworkIoSample {
    pub timestamp: Instant,
    pub bytes_received_per_sec: u64,
    pub bytes_sent_per_sec: u64,
    pub connections_active: u32,
    pub requests_per_sec: f64,
}

/// Latency percentile calculations
#[derive(Debug, Clone)]
pub struct LatencyPercentiles {
    pub p50: Duration,
    pub p90: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub p999: Duration,
    pub latency_samples: VecDeque<Duration>,
}

impl E2EPerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(monitoring_interval: Duration) -> Self {
        Self {
            workflow_metrics: HashMap::new(),
            resource_usage: ResourceUsageMetrics::new(),
            start_time: Instant::now(),
            monitoring_interval,
        }
    }

    /// Record workflow execution
    pub fn record_workflow_execution(
        &mut self,
        workflow_name: &str,
        execution_time: Duration,
        success: bool
    ) {
        let metrics = self.workflow_metrics
            .entry(workflow_name.to_string())
            .or_insert_with(|| WorkflowPerformanceMetrics::new(workflow_name));

        metrics.record_execution(execution_time, success);
    }

    /// Record throughput measurement
    pub fn record_throughput(
        &mut self,
        workflow_name: &str,
        operations_per_second: f64,
        concurrent_operations: u32
    ) {
        if let Some(metrics) = self.workflow_metrics.get_mut(workflow_name) {
            metrics.record_throughput(operations_per_second, concurrent_operations);
        }
    }

    /// Record resource usage
    pub fn record_resource_usage(&mut self, resource_sample: ResourceSample) {
        match resource_sample {
            ResourceSample::Memory(sample) => {
                self.resource_usage.record_memory_sample(sample);
            },
            ResourceSample::Cpu(sample) => {
                self.resource_usage.record_cpu_sample(sample);
            },
            ResourceSample::DiskIo(sample) => {
                self.resource_usage.record_disk_io_sample(sample);
            },
            ResourceSample::NetworkIo(sample) => {
                self.resource_usage.record_network_io_sample(sample);
            },
        }
    }

    /// Get performance summary for all workflows
    pub fn get_performance_summary(&self) -> PerformanceSummary {
        let total_workflows = self.workflow_metrics.len();
        let total_executions: u32 = self.workflow_metrics.values()
            .map(|m| m.execution_count)
            .sum();
        
        let total_successes: u32 = self.workflow_metrics.values()
            .map(|m| m.success_count)
            .sum();

        let overall_success_rate = if total_executions > 0 {
            total_successes as f64 / total_executions as f64
        } else {
            1.0
        };

        let avg_execution_time = if total_executions > 0 {
            let total_time: Duration = self.workflow_metrics.values()
                .map(|m| m.total_execution_time)
                .sum();
            total_time / total_executions
        } else {
            Duration::from_secs(0)
        };

        let current_throughput = self.calculate_current_throughput();
        let resource_efficiency = self.calculate_resource_efficiency();

        PerformanceSummary {
            total_workflows,
            total_executions,
            overall_success_rate,
            avg_execution_time,
            current_throughput,
            resource_efficiency,
            monitoring_duration: self.start_time.elapsed(),
            peak_memory_usage_mb: self.resource_usage.peak_memory_usage / 1024 / 1024,
            peak_cpu_usage_percentage: self.resource_usage.peak_cpu_usage,
        }
    }

    /// Get detailed metrics for a specific workflow
    pub fn get_workflow_metrics(&self, workflow_name: &str) -> Option<&WorkflowPerformanceMetrics> {
        self.workflow_metrics.get(workflow_name)
    }

    /// Get current resource usage
    pub fn get_current_resource_usage(&self) -> CurrentResourceUsage {
        CurrentResourceUsage {
            memory_usage_mb: self.resource_usage.get_current_memory_usage(),
            cpu_usage_percentage: self.resource_usage.get_current_cpu_usage(),
            disk_io_rate_mbps: self.resource_usage.get_current_disk_io_rate(),
            network_io_rate_mbps: self.resource_usage.get_current_network_io_rate(),
        }
    }

    /// Calculate performance trends
    pub fn get_performance_trends(&self) -> PerformanceTrends {
        PerformanceTrends {
            throughput_trend: self.calculate_throughput_trend(),
            latency_trend: self.calculate_latency_trend(),
            memory_usage_trend: self.calculate_memory_usage_trend(),
            cpu_usage_trend: self.calculate_cpu_usage_trend(),
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.workflow_metrics.clear();
        self.resource_usage = ResourceUsageMetrics::new();
        self.start_time = Instant::now();
    }

    // Private helper methods

    fn calculate_current_throughput(&self) -> f64 {
        let recent_samples: Vec<f64> = self.workflow_metrics.values()
            .flat_map(|m| m.throughput_samples.iter())
            .filter(|sample| sample.timestamp.elapsed() < Duration::from_minutes(1))
            .map(|sample| sample.operations_per_second)
            .collect();

        if recent_samples.is_empty() {
            0.0
        } else {
            recent_samples.iter().sum::<f64>() / recent_samples.len() as f64
        }
    }

    fn calculate_resource_efficiency(&self) -> f64 {
        let memory_efficiency = self.resource_usage.calculate_memory_efficiency();
        let cpu_efficiency = self.resource_usage.calculate_cpu_efficiency();
        
        (memory_efficiency + cpu_efficiency) / 2.0
    }

    fn calculate_throughput_trend(&self) -> TrendDirection {
        // Simplified trend calculation
        let recent_throughput = self.calculate_current_throughput();
        let historical_throughput = self.calculate_historical_throughput();
        
        if recent_throughput > historical_throughput * 1.1 {
            TrendDirection::Increasing
        } else if recent_throughput < historical_throughput * 0.9 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    fn calculate_latency_trend(&self) -> TrendDirection {
        // Compare recent vs historical latency
        let recent_latencies: Vec<Duration> = self.workflow_metrics.values()
            .flat_map(|m| m.latency_percentiles.latency_samples.iter())
            .filter(|&&latency| latency > Duration::from_secs(0))
            .take(100)
            .cloned()
            .collect();

        if recent_latencies.len() < 10 {
            return TrendDirection::Stable;
        }

        let recent_avg = recent_latencies.iter().sum::<Duration>() / recent_latencies.len() as u32;
        let historical_avg = self.calculate_historical_latency();

        if recent_avg > historical_avg + Duration::from_millis(10) {
            TrendDirection::Increasing
        } else if recent_avg < historical_avg - Duration::from_millis(10) {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    fn calculate_memory_usage_trend(&self) -> TrendDirection {
        if self.resource_usage.memory_samples.len() < 10 {
            return TrendDirection::Stable;
        }

        let recent_memory: u64 = self.resource_usage.memory_samples
            .iter()
            .rev()
            .take(5)
            .map(|sample| sample.used_memory_mb)
            .sum::<u64>() / 5;

        let historical_memory: u64 = self.resource_usage.memory_samples
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .map(|sample| sample.used_memory_mb)
            .sum::<u64>() / 5;

        if recent_memory > historical_memory + 50 { // 50MB increase
            TrendDirection::Increasing
        } else if recent_memory < historical_memory - 50 { // 50MB decrease
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    fn calculate_cpu_usage_trend(&self) -> TrendDirection {
        if self.resource_usage.cpu_samples.len() < 10 {
            return TrendDirection::Stable;
        }

        let recent_cpu: f64 = self.resource_usage.cpu_samples
            .iter()
            .rev()
            .take(5)
            .map(|sample| sample.cpu_percentage)
            .sum::<f64>() / 5.0;

        let historical_cpu: f64 = self.resource_usage.cpu_samples
            .iter()
            .rev()
            .skip(5)
            .take(5)
            .map(|sample| sample.cpu_percentage)
            .sum::<f64>() / 5.0;

        if recent_cpu > historical_cpu + 5.0 { // 5% increase
            TrendDirection::Increasing
        } else if recent_cpu < historical_cpu - 5.0 { // 5% decrease
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    fn calculate_historical_throughput(&self) -> f64 {
        let historical_samples: Vec<f64> = self.workflow_metrics.values()
            .flat_map(|m| m.throughput_samples.iter())
            .filter(|sample| sample.timestamp.elapsed() > Duration::from_minutes(1))
            .map(|sample| sample.operations_per_second)
            .collect();

        if historical_samples.is_empty() {
            0.0
        } else {
            historical_samples.iter().sum::<f64>() / historical_samples.len() as f64
        }
    }

    fn calculate_historical_latency(&self) -> Duration {
        let historical_latencies: Vec<Duration> = self.workflow_metrics.values()
            .flat_map(|m| m.latency_percentiles.latency_samples.iter())
            .skip(100) // Skip recent samples
            .cloned()
            .collect();

        if historical_latencies.is_empty() {
            Duration::from_millis(20) // Default baseline
        } else {
            historical_latencies.iter().sum::<Duration>() / historical_latencies.len() as u32
        }
    }
}

impl WorkflowPerformanceMetrics {
    fn new(workflow_name: &str) -> Self {
        Self {
            workflow_name: workflow_name.to_string(),
            execution_count: 0,
            total_execution_time: Duration::from_secs(0),
            min_execution_time: Duration::from_secs(u64::MAX),
            max_execution_time: Duration::from_secs(0),
            success_count: 0,
            error_count: 0,
            last_execution: None,
            throughput_samples: VecDeque::new(),
            latency_percentiles: LatencyPercentiles::new(),
        }
    }

    fn record_execution(&mut self, execution_time: Duration, success: bool) {
        self.execution_count += 1;
        self.total_execution_time += execution_time;
        self.last_execution = Some(Instant::now());

        if execution_time < self.min_execution_time {
            self.min_execution_time = execution_time;
        }
        if execution_time > self.max_execution_time {
            self.max_execution_time = execution_time;
        }

        if success {
            self.success_count += 1;
        } else {
            self.error_count += 1;
        }

        self.latency_percentiles.record_latency(execution_time);
    }

    fn record_throughput(&mut self, operations_per_second: f64, concurrent_operations: u32) {
        let sample = ThroughputSample {
            timestamp: Instant::now(),
            operations_per_second,
            concurrent_operations,
        };

        self.throughput_samples.push_back(sample);

        // Keep only recent samples (last 1000)
        while self.throughput_samples.len() > 1000 {
            self.throughput_samples.pop_front();
        }
    }

    /// Get average execution time
    pub fn avg_execution_time(&self) -> Duration {
        if self.execution_count > 0 {
            self.total_execution_time / self.execution_count
        } else {
            Duration::from_secs(0)
        }
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.execution_count > 0 {
            self.success_count as f64 / self.execution_count as f64
        } else {
            1.0
        }
    }

    /// Get current throughput
    pub fn current_throughput(&self) -> f64 {
        if let Some(latest_sample) = self.throughput_samples.back() {
            if latest_sample.timestamp.elapsed() < Duration::from_minutes(1) {
                return latest_sample.operations_per_second;
            }
        }
        0.0
    }
}

impl ResourceUsageMetrics {
    fn new() -> Self {
        Self {
            memory_samples: VecDeque::new(),
            cpu_samples: VecDeque::new(),
            disk_io_samples: VecDeque::new(),
            network_io_samples: VecDeque::new(),
            peak_memory_usage: 0,
            peak_cpu_usage: 0.0,
        }
    }

    fn record_memory_sample(&mut self, sample: MemorySample) {
        if sample.used_memory_mb * 1024 * 1024 > self.peak_memory_usage {
            self.peak_memory_usage = sample.used_memory_mb * 1024 * 1024;
        }

        self.memory_samples.push_back(sample);
        
        // Keep only recent samples (last 1000)
        while self.memory_samples.len() > 1000 {
            self.memory_samples.pop_front();
        }
    }

    fn record_cpu_sample(&mut self, sample: CpuSample) {
        if sample.cpu_percentage > self.peak_cpu_usage {
            self.peak_cpu_usage = sample.cpu_percentage;
        }

        self.cpu_samples.push_back(sample);
        
        // Keep only recent samples (last 1000)
        while self.cpu_samples.len() > 1000 {
            self.cpu_samples.pop_front();
        }
    }

    fn record_disk_io_sample(&mut self, sample: DiskIoSample) {
        self.disk_io_samples.push_back(sample);
        
        // Keep only recent samples (last 1000)
        while self.disk_io_samples.len() > 1000 {
            self.disk_io_samples.pop_front();
        }
    }

    fn record_network_io_sample(&mut self, sample: NetworkIoSample) {
        self.network_io_samples.push_back(sample);
        
        // Keep only recent samples (last 1000)
        while self.network_io_samples.len() > 1000 {
            self.network_io_samples.pop_front();
        }
    }

    fn get_current_memory_usage(&self) -> u64 {
        self.memory_samples.back()
            .map(|sample| sample.used_memory_mb)
            .unwrap_or(0)
    }

    fn get_current_cpu_usage(&self) -> f64 {
        self.cpu_samples.back()
            .map(|sample| sample.cpu_percentage)
            .unwrap_or(0.0)
    }

    fn get_current_disk_io_rate(&self) -> f64 {
        self.disk_io_samples.back()
            .map(|sample| (sample.read_bytes_per_sec + sample.write_bytes_per_sec) as f64 / 1024.0 / 1024.0)
            .unwrap_or(0.0)
    }

    fn get_current_network_io_rate(&self) -> f64 {
        self.network_io_samples.back()
            .map(|sample| (sample.bytes_received_per_sec + sample.bytes_sent_per_sec) as f64 / 1024.0 / 1024.0)
            .unwrap_or(0.0)
    }

    fn calculate_memory_efficiency(&self) -> f64 {
        if self.memory_samples.is_empty() {
            return 1.0;
        }

        let avg_memory_usage = self.memory_samples.iter()
            .map(|sample| sample.used_memory_mb)
            .sum::<u64>() as f64 / self.memory_samples.len() as f64;

        let peak_memory_mb = self.peak_memory_usage as f64 / 1024.0 / 1024.0;

        if peak_memory_mb > 0.0 {
            1.0 - (avg_memory_usage / peak_memory_mb).min(1.0)
        } else {
            1.0
        }
    }

    fn calculate_cpu_efficiency(&self) -> f64 {
        if self.cpu_samples.is_empty() {
            return 1.0;
        }

        let avg_cpu_usage = self.cpu_samples.iter()
            .map(|sample| sample.cpu_percentage)
            .sum::<f64>() / self.cpu_samples.len() as f64;

        if self.peak_cpu_usage > 0.0 {
            1.0 - (avg_cpu_usage / self.peak_cpu_usage).min(1.0)
        } else {
            1.0
        }
    }
}

impl LatencyPercentiles {
    fn new() -> Self {
        Self {
            p50: Duration::from_secs(0),
            p90: Duration::from_secs(0),
            p95: Duration::from_secs(0),
            p99: Duration::from_secs(0),
            p999: Duration::from_secs(0),
            latency_samples: VecDeque::new(),
        }
    }

    fn record_latency(&mut self, latency: Duration) {
        self.latency_samples.push_back(latency);

        // Keep only recent samples (last 10000)
        while self.latency_samples.len() > 10000 {
            self.latency_samples.pop_front();
        }

        // Recalculate percentiles
        self.calculate_percentiles();
    }

    fn calculate_percentiles(&mut self) {
        if self.latency_samples.is_empty() {
            return;
        }

        let mut sorted_samples: Vec<Duration> = self.latency_samples.iter().cloned().collect();
        sorted_samples.sort();

        let len = sorted_samples.len();
        
        self.p50 = sorted_samples[len * 50 / 100];
        self.p90 = sorted_samples[len * 90 / 100];
        self.p95 = sorted_samples[len * 95 / 100];
        self.p99 = sorted_samples[len * 99 / 100];
        self.p999 = sorted_samples[len * 999 / 1000];
    }
}

// Resource sample types
pub enum ResourceSample {
    Memory(MemorySample),
    Cpu(CpuSample),
    DiskIo(DiskIoSample),
    NetworkIo(NetworkIoSample),
}

// Summary and trend types
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub total_workflows: usize,
    pub total_executions: u32,
    pub overall_success_rate: f64,
    pub avg_execution_time: Duration,
    pub current_throughput: f64,
    pub resource_efficiency: f64,
    pub monitoring_duration: Duration,
    pub peak_memory_usage_mb: u64,
    pub peak_cpu_usage_percentage: f64,
}

#[derive(Debug, Clone)]
pub struct CurrentResourceUsage {
    pub memory_usage_mb: u64,
    pub cpu_usage_percentage: f64,
    pub disk_io_rate_mbps: f64,
    pub network_io_rate_mbps: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    pub throughput_trend: TrendDirection,
    pub latency_trend: TrendDirection,
    pub memory_usage_trend: TrendDirection,
    pub cpu_usage_trend: TrendDirection,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor_creation() {
        let monitor = E2EPerformanceMonitor::new(Duration::from_secs(1));
        let summary = monitor.get_performance_summary();
        
        assert_eq!(summary.total_workflows, 0);
        assert_eq!(summary.total_executions, 0);
        assert_eq!(summary.overall_success_rate, 1.0);
    }

    #[test]
    fn test_workflow_performance_recording() {
        let mut monitor = E2EPerformanceMonitor::new(Duration::from_secs(1));
        
        monitor.record_workflow_execution("test_workflow", Duration::from_millis(100), true);
        monitor.record_workflow_execution("test_workflow", Duration::from_millis(150), true);
        monitor.record_workflow_execution("test_workflow", Duration::from_millis(200), false);
        
        let metrics = monitor.get_workflow_metrics("test_workflow").unwrap();
        assert_eq!(metrics.execution_count, 3);
        assert_eq!(metrics.success_count, 2);
        assert_eq!(metrics.error_count, 1);
        assert_eq!(metrics.success_rate(), 2.0 / 3.0);
        assert_eq!(metrics.min_execution_time, Duration::from_millis(100));
        assert_eq!(metrics.max_execution_time, Duration::from_millis(200));
    }

    #[test]
    fn test_throughput_recording() {
        let mut monitor = E2EPerformanceMonitor::new(Duration::from_secs(1));
        
        monitor.record_workflow_execution("test_workflow", Duration::from_millis(100), true);
        monitor.record_throughput("test_workflow", 50.0, 10);
        
        let metrics = monitor.get_workflow_metrics("test_workflow").unwrap();
        assert_eq!(metrics.current_throughput(), 50.0);
    }

    #[test]
    fn test_resource_usage_recording() {
        let mut monitor = E2EPerformanceMonitor::new(Duration::from_secs(1));
        
        let memory_sample = MemorySample {
            timestamp: Instant::now(),
            total_memory_mb: 1024,
            used_memory_mb: 512,
            heap_memory_mb: 256,
            off_heap_memory_mb: 256,
        };
        
        monitor.record_resource_usage(ResourceSample::Memory(memory_sample));
        
        let usage = monitor.get_current_resource_usage();
        assert_eq!(usage.memory_usage_mb, 512);
    }

    #[test]
    fn test_latency_percentiles() {
        let mut percentiles = LatencyPercentiles::new();
        
        // Add some latency samples
        for i in 1..=100 {
            percentiles.record_latency(Duration::from_millis(i));
        }
        
        assert!(percentiles.p50 > Duration::from_secs(0));
        assert!(percentiles.p90 > percentiles.p50);
        assert!(percentiles.p95 > percentiles.p90);
        assert!(percentiles.p99 > percentiles.p95);
        assert!(percentiles.p999 > percentiles.p99);
    }

    #[test]
    fn test_performance_trends() {
        let mut monitor = E2EPerformanceMonitor::new(Duration::from_secs(1));
        
        // Add some historical data
        for i in 0..20 {
            monitor.record_workflow_execution("test", Duration::from_millis(50 + i), true);
            monitor.record_throughput("test", 100.0 - i as f64, 10);
        }
        
        let trends = monitor.get_performance_trends();
        
        // Should detect trends based on the data pattern
        assert!(matches!(trends.throughput_trend, TrendDirection::Decreasing | TrendDirection::Stable));
        assert!(matches!(trends.latency_trend, TrendDirection::Increasing | TrendDirection::Stable));
    }

    #[test]
    fn test_resource_efficiency_calculation() {
        let mut resource_usage = ResourceUsageMetrics::new();
        
        // Add some samples
        for i in 1..=10 {
            let memory_sample = MemorySample {
                timestamp: Instant::now(),
                total_memory_mb: 1024,
                used_memory_mb: 100 * i,
                heap_memory_mb: 50 * i,
                off_heap_memory_mb: 50 * i,
            };
            resource_usage.record_memory_sample(memory_sample);
            
            let cpu_sample = CpuSample {
                timestamp: Instant::now(),
                cpu_percentage: 10.0 * i as f64,
                system_load: 1.0,
                gc_time_ms: 10,
            };
            resource_usage.record_cpu_sample(cpu_sample);
        }
        
        let memory_efficiency = resource_usage.calculate_memory_efficiency();
        let cpu_efficiency = resource_usage.calculate_cpu_efficiency();
        
        assert!(memory_efficiency >= 0.0 && memory_efficiency <= 1.0);
        assert!(cpu_efficiency >= 0.0 && cpu_efficiency <= 1.0);
    }

    #[test]
    fn test_monitor_reset() {
        let mut monitor = E2EPerformanceMonitor::new(Duration::from_secs(1));
        
        monitor.record_workflow_execution("test", Duration::from_millis(100), true);
        assert_eq!(monitor.get_performance_summary().total_executions, 1);
        
        monitor.reset();
        assert_eq!(monitor.get_performance_summary().total_executions, 0);
        assert!(monitor.workflow_metrics.is_empty());
    }
}