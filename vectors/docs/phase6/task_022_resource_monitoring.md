# Task 022: System Resource Monitoring

## Context
You are implementing a comprehensive validation system for a Rust-based vector indexing system. This builds on Tasks 008, 017, 018, 019, and 021 (PerformanceBenchmark, EnhancedPerformanceMetrics, ConcurrentBenchmark, AdvancedPercentileCalculations, and ConcurrentResultsAnalyzer). The system resource monitoring provides real-time tracking of CPU, memory, disk, and network resources with performance counter integration and resource leak detection.

## Project Structure
```
src/
  validation/
    performance.rs  <- Extend this file
  lib.rs
```

## Task Description
Implement comprehensive system resource monitoring with process-level resource tracking, performance counter integration for Windows, resource leak detection and alerting, historical resource usage analysis, and export capabilities for monitoring systems.

## Requirements
1. Add to existing `src/validation/performance.rs`
2. System resource monitoring (CPU, memory, disk, network)
3. Process-level resource tracking with thread granularity
4. Performance counter integration for Windows compatibility
5. Resource leak detection with automatic alerting
6. Historical resource usage analysis and trending
7. Export capabilities for external monitoring systems

## Expected Code Structure to Add
```rust
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tokio::time::interval;

#[cfg(target_os = "windows")]
use windows::Win32::System::Performance::*;
#[cfg(target_os = "windows")]
use windows::Win32::Foundation::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResourceMonitor {
    monitoring_config: ResourceMonitoringConfig,
    resource_collectors: Vec<Arc<dyn ResourceCollector + Send + Sync>>,
    historical_data: Arc<RwLock<HistoricalResourceData>>,
    leak_detector: ResourceLeakDetector,
    alerting_system: ResourceAlertingSystem,
    export_manager: ResourceExportManager,
    monitoring_active: Arc<tokio::sync::RwLock<bool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoringConfig {
    pub sampling_interval_ms: u64,
    pub history_retention_minutes: u64,
    pub leak_detection_threshold_mb: f64,
    pub cpu_alert_threshold_percent: f64,
    pub memory_alert_threshold_percent: f64,
    pub disk_io_alert_threshold_mbps: f64,
    pub network_alert_threshold_mbps: f64,
    pub enable_performance_counters: bool,
    pub export_to_prometheus: bool,
    pub export_to_csv: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub timestamp: SystemTime,
    pub cpu_metrics: CpuMetrics,
    pub memory_metrics: MemoryMetrics,
    pub disk_metrics: DiskMetrics,
    pub network_metrics: NetworkMetrics,
    pub process_metrics: ProcessMetrics,
    pub thread_metrics: Vec<ThreadResourceMetrics>,
    pub performance_counters: Option<WindowsPerformanceCounters>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMetrics {
    pub overall_usage_percent: f64,
    pub per_core_usage: Vec<f64>,
    pub process_cpu_percent: f64,
    pub thread_cpu_times: HashMap<u32, f64>,
    pub context_switches_per_sec: f64,
    pub cpu_frequency_mhz: f64,
    pub cpu_temperature_celsius: Option<f64>,
    pub cpu_efficiency_score: f64, // Work done per CPU cycle
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub total_system_memory_mb: f64,
    pub available_memory_mb: f64,
    pub used_memory_mb: f64,
    pub memory_usage_percent: f64,
    pub process_memory_mb: f64,
    pub process_virtual_memory_mb: f64,
    pub working_set_mb: f64,
    pub peak_working_set_mb: f64,
    pub page_file_usage_mb: f64,
    pub memory_pressure_score: f64, // 0-100, higher = more pressure
    pub garbage_collection_info: GarbageCollectionMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarbageCollectionMetrics {
    pub gen0_collections: u64,
    pub gen1_collections: u64,
    pub gen2_collections: u64,
    pub total_gc_time_ms: f64,
    pub gc_pressure_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskMetrics {
    pub disk_usage_by_drive: HashMap<String, DiskDriveMetrics>,
    pub process_io_read_bytes_per_sec: f64,
    pub process_io_write_bytes_per_sec: f64,
    pub disk_queue_length: f64,
    pub average_disk_response_time_ms: f64,
    pub disk_io_efficiency_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskDriveMetrics {
    pub total_space_gb: f64,
    pub free_space_gb: f64,
    pub used_space_gb: f64,
    pub usage_percent: f64,
    pub read_bytes_per_sec: f64,
    pub write_bytes_per_sec: f64,
    pub io_operations_per_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub network_interfaces: HashMap<String, NetworkInterfaceMetrics>,
    pub process_network_bytes_sent: u64,
    pub process_network_bytes_received: u64,
    pub total_connections: usize,
    pub active_connections: usize,
    pub network_latency_ms: f64,
    pub packet_loss_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterfaceMetrics {
    pub bytes_sent_per_sec: f64,
    pub bytes_received_per_sec: f64,
    pub packets_sent_per_sec: f64,
    pub packets_received_per_sec: f64,
    pub errors_per_sec: f64,
    pub utilization_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessMetrics {
    pub process_id: u32,
    pub thread_count: usize,
    pub handle_count: usize,
    pub uptime_seconds: f64,
    pub total_processor_time_ms: f64,
    pub user_processor_time_ms: f64,
    pub privileged_processor_time_ms: f64,
    pub io_read_operations: u64,
    pub io_write_operations: u64,
    pub io_other_operations: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadResourceMetrics {
    pub thread_id: u32,
    pub cpu_time_ms: f64,
    pub context_switches: u64,
    pub priority: i32,
    pub state: ThreadState,
    pub memory_allocated_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreadState {
    Running,
    Waiting,
    Suspended,
    Terminated,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowsPerformanceCounters {
    pub processor_queue_length: f64,
    pub disk_queue_length: f64,
    pub memory_pages_per_sec: f64,
    pub network_utilization: f64,
    pub cache_hit_ratio: f64,
    pub system_calls_per_sec: f64,
}

impl SystemResourceMonitor {
    pub async fn new(config: ResourceMonitoringConfig) -> Result<Self> {
        let mut resource_collectors: Vec<Arc<dyn ResourceCollector + Send + Sync>> = Vec::new();
        
        // Add standard collectors
        resource_collectors.push(Arc::new(CpuCollector::new().await?));
        resource_collectors.push(Arc::new(MemoryCollector::new().await?));
        resource_collectors.push(Arc::new(DiskCollector::new().await?));
        resource_collectors.push(Arc::new(NetworkCollector::new().await?));
        resource_collectors.push(Arc::new(ProcessCollector::new().await?));
        
        // Add Windows-specific collectors if enabled
        #[cfg(target_os = "windows")]
        if config.enable_performance_counters {
            resource_collectors.push(Arc::new(WindowsPerformanceCounterCollector::new().await?));
        }
        
        Ok(Self {
            monitoring_config: config.clone(),
            resource_collectors,
            historical_data: Arc::new(RwLock::new(HistoricalResourceData::new(config.history_retention_minutes))),
            leak_detector: ResourceLeakDetector::new(config.leak_detection_threshold_mb),
            alerting_system: ResourceAlertingSystem::new(config.clone()),
            export_manager: ResourceExportManager::new(config.clone()),
            monitoring_active: Arc::new(tokio::sync::RwLock::new(false)),
        })
    }
    
    pub async fn start_monitoring(&self) -> Result<()> {
        {
            let mut active = self.monitoring_active.write().await;
            if *active {
                return Err(anyhow::anyhow!("Resource monitoring is already active"));
            }
            *active = true;
        }
        
        println!("Starting system resource monitoring...");
        
        let monitoring_active = Arc::clone(&self.monitoring_active);
        let historical_data = Arc::clone(&self.historical_data);
        let collectors = self.resource_collectors.clone();
        let leak_detector = self.leak_detector.clone();
        let alerting_system = self.alerting_system.clone();
        let export_manager = self.export_manager.clone();
        let config = self.monitoring_config.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(config.sampling_interval_ms));
            
            while *monitoring_active.read().await {
                interval.tick().await;
                
                match Self::collect_resource_snapshot(&collectors).await {
                    Ok(snapshot) => {
                        // Store in historical data
                        {
                            let mut history = historical_data.write().await;
                            history.add_snapshot(snapshot.clone());
                        }
                        
                        // Check for resource leaks
                        if let Some(leak_alert) = leak_detector.check_for_leaks(&snapshot).await {
                            alerting_system.send_alert(ResourceAlert::Leak(leak_alert)).await;
                        }
                        
                        // Check resource thresholds
                        if let Some(threshold_alert) = Self::check_resource_thresholds(&snapshot, &config) {
                            alerting_system.send_alert(threshold_alert).await;
                        }
                        
                        // Export metrics if configured
                        if config.export_to_prometheus || config.export_to_csv {
                            export_manager.export_snapshot(&snapshot).await;
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to collect resource snapshot: {}", e);
                    }
                }
            }
            
            println!("Resource monitoring stopped.");
        });
        
        Ok(())
    }
    
    pub async fn stop_monitoring(&self) {
        let mut active = self.monitoring_active.write().await;
        *active = false;
        println!("Stopping resource monitoring...");
    }
    
    pub async fn get_current_snapshot(&self) -> Result<ResourceSnapshot> {
        Self::collect_resource_snapshot(&self.resource_collectors).await
    }
    
    pub async fn get_historical_analysis(&self, duration_minutes: u64) -> Result<ResourceAnalysisReport> {
        let history = self.historical_data.read().await;
        let snapshots = history.get_snapshots_since(
            SystemTime::now() - Duration::from_secs(duration_minutes * 60)
        );
        
        if snapshots.is_empty() {
            return Err(anyhow::anyhow!("No historical data available for the specified duration"));
        }
        
        Ok(self.analyze_resource_history(&snapshots))
    }
    
    pub async fn detect_resource_anomalies(&self) -> Result<Vec<ResourceAnomaly>> {
        let history = self.historical_data.read().await;
        let recent_snapshots = history.get_recent_snapshots(100); // Last 100 samples
        
        if recent_snapshots.len() < 10 {
            return Ok(Vec::new());
        }
        
        let mut anomalies = Vec::new();
        
        // CPU anomaly detection
        let cpu_usage: Vec<f64> = recent_snapshots.iter().map(|s| s.cpu_metrics.overall_usage_percent).collect();
        if let Some(anomaly) = self.detect_statistical_anomaly(&cpu_usage, "CPU Usage") {
            anomalies.push(anomaly);
        }
        
        // Memory anomaly detection
        let memory_usage: Vec<f64> = recent_snapshots.iter().map(|s| s.memory_metrics.memory_usage_percent).collect();
        if let Some(anomaly) = self.detect_statistical_anomaly(&memory_usage, "Memory Usage") {
            anomalies.push(anomaly);
        }
        
        // Process memory growth detection
        let process_memory: Vec<f64> = recent_snapshots.iter().map(|s| s.memory_metrics.process_memory_mb).collect();
        if let Some(leak_anomaly) = self.detect_memory_leak(&process_memory) {
            anomalies.push(leak_anomaly);
        }
        
        Ok(anomalies)
    }
    
    async fn collect_resource_snapshot(collectors: &[Arc<dyn ResourceCollector + Send + Sync>]) -> Result<ResourceSnapshot> {
        let timestamp = SystemTime::now();
        let mut cpu_metrics = CpuMetrics::default();
        let mut memory_metrics = MemoryMetrics::default();
        let mut disk_metrics = DiskMetrics::default();
        let mut network_metrics = NetworkMetrics::default();
        let mut process_metrics = ProcessMetrics::default();
        let mut thread_metrics = Vec::new();
        let mut performance_counters = None;
        
        // Collect from all resource collectors
        for collector in collectors {
            match collector.collect().await {
                Ok(resource_data) => {
                    match resource_data {
                        ResourceData::Cpu(cpu) => cpu_metrics = cpu,
                        ResourceData::Memory(mem) => memory_metrics = mem,
                        ResourceData::Disk(disk) => disk_metrics = disk,
                        ResourceData::Network(net) => network_metrics = net,
                        ResourceData::Process(proc) => process_metrics = proc,
                        ResourceData::Threads(threads) => thread_metrics = threads,
                        ResourceData::WindowsCounters(counters) => performance_counters = Some(counters),
                    }
                }
                Err(e) => {
                    eprintln!("Resource collector failed: {}", e);
                }
            }
        }
        
        Ok(ResourceSnapshot {
            timestamp,
            cpu_metrics,
            memory_metrics,
            disk_metrics,
            network_metrics,
            process_metrics,
            thread_metrics,
            performance_counters,
        })
    }
    
    fn check_resource_thresholds(snapshot: &ResourceSnapshot, config: &ResourceMonitoringConfig) -> Option<ResourceAlert> {
        // CPU threshold check
        if snapshot.cpu_metrics.overall_usage_percent > config.cpu_alert_threshold_percent {
            return Some(ResourceAlert::Threshold(ThresholdAlert {
                resource_type: "CPU".to_string(),
                current_value: snapshot.cpu_metrics.overall_usage_percent,
                threshold_value: config.cpu_alert_threshold_percent,
                severity: AlertSeverity::High,
                message: format!(
                    "CPU usage ({:.1}%) exceeded threshold ({:.1}%)",
                    snapshot.cpu_metrics.overall_usage_percent,
                    config.cpu_alert_threshold_percent
                ),
            }));
        }
        
        // Memory threshold check
        if snapshot.memory_metrics.memory_usage_percent > config.memory_alert_threshold_percent {
            return Some(ResourceAlert::Threshold(ThresholdAlert {
                resource_type: "Memory".to_string(),
                current_value: snapshot.memory_metrics.memory_usage_percent,
                threshold_value: config.memory_alert_threshold_percent,
                severity: AlertSeverity::High,
                message: format!(
                    "Memory usage ({:.1}%) exceeded threshold ({:.1}%)",
                    snapshot.memory_metrics.memory_usage_percent,
                    config.memory_alert_threshold_percent
                ),
            }));
        }
        
        None
    }
    
    fn analyze_resource_history(&self, snapshots: &[ResourceSnapshot]) -> ResourceAnalysisReport {
        if snapshots.is_empty() {
            return ResourceAnalysisReport::default();
        }
        
        // CPU analysis
        let cpu_usage: Vec<f64> = snapshots.iter().map(|s| s.cpu_metrics.overall_usage_percent).collect();
        let cpu_analysis = self.analyze_metric_series(&cpu_usage, "CPU Usage");
        
        // Memory analysis
        let memory_usage: Vec<f64> = snapshots.iter().map(|s| s.memory_metrics.memory_usage_percent).collect();
        let memory_analysis = self.analyze_metric_series(&memory_usage, "Memory Usage");
        
        // Process memory growth analysis
        let process_memory: Vec<f64> = snapshots.iter().map(|s| s.memory_metrics.process_memory_mb).collect();
        let memory_trend = self.calculate_trend(&process_memory);
        
        // Disk I/O analysis
        let disk_read: Vec<f64> = snapshots.iter().map(|s| s.disk_metrics.process_io_read_bytes_per_sec).collect();
        let disk_write: Vec<f64> = snapshots.iter().map(|s| s.disk_metrics.process_io_write_bytes_per_sec).collect();
        
        ResourceAnalysisReport {
            analysis_period: AnalysisPeriod {
                start_time: snapshots.first().unwrap().timestamp,
                end_time: snapshots.last().unwrap().timestamp,
                sample_count: snapshots.len(),
            },
            cpu_analysis,
            memory_analysis,
            memory_trend,
            disk_io_summary: DiskIOSummary {
                average_read_mbps: disk_read.iter().sum::<f64>() / disk_read.len() as f64 / 1_048_576.0,
                average_write_mbps: disk_write.iter().sum::<f64>() / disk_write.len() as f64 / 1_048_576.0,
                peak_read_mbps: disk_read.iter().cloned().fold(0.0, f64::max) / 1_048_576.0,
                peak_write_mbps: disk_write.iter().cloned().fold(0.0, f64::max) / 1_048_576.0,
            },
            performance_score: self.calculate_overall_performance_score(snapshots),
            recommendations: self.generate_performance_recommendations(snapshots),
        }
    }
    
    fn analyze_metric_series(&self, values: &[f64], metric_name: &str) -> MetricAnalysis {
        if values.is_empty() {
            return MetricAnalysis::default();
        }
        
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        
        let trend = self.calculate_trend(values);
        
        MetricAnalysis {
            metric_name: metric_name.to_string(),
            min_value: min,
            max_value: max,
            mean_value: mean,
            median_value: self.calculate_median(values),
            std_deviation: std_dev,
            coefficient_of_variation: if mean > 0.0 { std_dev / mean } else { 0.0 },
            trend,
            stability_score: self.calculate_stability_score(values),
        }
    }
    
    fn calculate_trend(&self, values: &[f64]) -> ResourceTrend {
        if values.len() < 3 {
            return ResourceTrend::Stable;
        }
        
        // Simple linear regression to detect trend
        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x_sq_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();
        
        let slope = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum.powi(2));
        
        if slope > 0.1 {
            ResourceTrend::Increasing
        } else if slope < -0.1 {
            ResourceTrend::Decreasing
        } else {
            ResourceTrend::Stable
        }
    }
    
    fn calculate_median(&self, values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let len = sorted.len();
        if len % 2 == 0 {
            (sorted[len / 2 - 1] + sorted[len / 2]) / 2.0
        } else {
            sorted[len / 2]
        }
    }
    
    fn calculate_stability_score(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 100.0;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        let cv = if mean > 0.0 { variance.sqrt() / mean } else { 0.0 };
        
        // Convert coefficient of variation to stability score (0-100)
        (100.0 / (1.0 + cv * 10.0)).min(100.0)
    }
    
    fn detect_statistical_anomaly(&self, values: &[f64], metric_name: &str) -> Option<ResourceAnomaly> {
        if values.len() < 10 {
            return None;
        }
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        
        // Check for values beyond 3 standard deviations (outliers)
        let recent_value = values.last().unwrap();
        if (recent_value - mean).abs() > 3.0 * std_dev {
            return Some(ResourceAnomaly {
                anomaly_type: AnomalyType::StatisticalOutlier,
                metric_name: metric_name.to_string(),
                severity: if (recent_value - mean).abs() > 4.0 * std_dev {
                    AnomalySeverity::High
                } else {
                    AnomalySeverity::Medium
                },
                description: format!(
                    "{} value {:.2} is {:.2} standard deviations from mean {:.2}",
                    metric_name,
                    recent_value,
                    (recent_value - mean).abs() / std_dev,
                    mean
                ),
                detected_at: SystemTime::now(),
                confidence_score: 0.95,
            });
        }
        
        None
    }
    
    fn detect_memory_leak(&self, memory_values: &[f64]) -> Option<ResourceAnomaly> {
        if memory_values.len() < 20 {
            return None;
        }
        
        let trend = self.calculate_trend(memory_values);
        
        match trend {
            ResourceTrend::Increasing => {
                // Check if memory is consistently increasing
                let recent_half = &memory_values[memory_values.len() / 2..];
                let early_half = &memory_values[..memory_values.len() / 2];
                
                let recent_avg = recent_half.iter().sum::<f64>() / recent_half.len() as f64;
                let early_avg = early_half.iter().sum::<f64>() / early_half.len() as f64;
                
                let growth_rate = (recent_avg - early_avg) / early_avg;
                
                if growth_rate > 0.2 { // 20% growth
                    return Some(ResourceAnomaly {
                        anomaly_type: AnomalyType::MemoryLeak,
                        metric_name: "Process Memory".to_string(),
                        severity: if growth_rate > 0.5 {
                            AnomalySeverity::High
                        } else {
                            AnomalySeverity::Medium
                        },
                        description: format!(
                            "Potential memory leak detected: {:.1}% memory growth over monitoring period",
                            growth_rate * 100.0
                        ),
                        detected_at: SystemTime::now(),
                        confidence_score: 0.8,
                    });
                }
            }
            _ => {}
        }
        
        None
    }
    
    fn calculate_overall_performance_score(&self, snapshots: &[ResourceSnapshot]) -> f64 {
        if snapshots.is_empty() {
            return 0.0;
        }
        
        // Calculate average resource utilization
        let avg_cpu = snapshots.iter().map(|s| s.cpu_metrics.overall_usage_percent).sum::<f64>() / snapshots.len() as f64;
        let avg_memory = snapshots.iter().map(|s| s.memory_metrics.memory_usage_percent).sum::<f64>() / snapshots.len() as f64;
        
        // Performance score: lower resource usage = higher score, with efficiency bonus
        let cpu_score = if avg_cpu < 50.0 { 100.0 - avg_cpu } else { 50.0 - (avg_cpu - 50.0) };
        let memory_score = if avg_memory < 70.0 { 100.0 - avg_memory } else { 30.0 - (avg_memory - 70.0) * 2.0 };
        
        ((cpu_score + memory_score) / 2.0).max(0.0).min(100.0)
    }
    
    fn generate_performance_recommendations(&self, snapshots: &[ResourceSnapshot]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let avg_cpu = snapshots.iter().map(|s| s.cpu_metrics.overall_usage_percent).sum::<f64>() / snapshots.len() as f64;
        let avg_memory = snapshots.iter().map(|s| s.memory_metrics.memory_usage_percent).sum::<f64>() / snapshots.len() as f64;
        
        if avg_cpu > 80.0 {
            recommendations.push("High CPU usage detected. Consider optimizing algorithms or increasing CPU resources.".to_string());
        }
        
        if avg_memory > 85.0 {
            recommendations.push("High memory usage detected. Consider memory optimization or increasing available RAM.".to_string());
        }
        
        // Check for memory growth
        let memory_values: Vec<f64> = snapshots.iter().map(|s| s.memory_metrics.process_memory_mb).collect();
        if let ResourceTrend::Increasing = self.calculate_trend(&memory_values) {
            recommendations.push("Memory usage is trending upward. Monitor for potential memory leaks.".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("Resource usage appears optimal. Continue monitoring for sustained performance.".to_string());
        }
        
        recommendations
    }
}

// Resource Collector Trait
#[async_trait::async_trait]
pub trait ResourceCollector {
    async fn collect(&self) -> Result<ResourceData>;
}

#[derive(Debug, Clone)]
pub enum ResourceData {
    Cpu(CpuMetrics),
    Memory(MemoryMetrics),
    Disk(DiskMetrics),
    Network(NetworkMetrics),
    Process(ProcessMetrics),
    Threads(Vec<ThreadResourceMetrics>),
    WindowsCounters(WindowsPerformanceCounters),
}

// Default implementations
impl Default for CpuMetrics {
    fn default() -> Self {
        Self {
            overall_usage_percent: 0.0,
            per_core_usage: Vec::new(),
            process_cpu_percent: 0.0,
            thread_cpu_times: HashMap::new(),
            context_switches_per_sec: 0.0,
            cpu_frequency_mhz: 0.0,
            cpu_temperature_celsius: None,
            cpu_efficiency_score: 0.0,
        }
    }
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            total_system_memory_mb: 0.0,
            available_memory_mb: 0.0,
            used_memory_mb: 0.0,
            memory_usage_percent: 0.0,
            process_memory_mb: 0.0,
            process_virtual_memory_mb: 0.0,
            working_set_mb: 0.0,
            peak_working_set_mb: 0.0,
            page_file_usage_mb: 0.0,
            memory_pressure_score: 0.0,
            garbage_collection_info: GarbageCollectionMetrics::default(),
        }
    }
}

impl Default for GarbageCollectionMetrics {
    fn default() -> Self {
        Self {
            gen0_collections: 0,
            gen1_collections: 0,
            gen2_collections: 0,
            total_gc_time_ms: 0.0,
            gc_pressure_score: 0.0,
        }
    }
}

impl Default for DiskMetrics {
    fn default() -> Self {
        Self {
            disk_usage_by_drive: HashMap::new(),
            process_io_read_bytes_per_sec: 0.0,
            process_io_write_bytes_per_sec: 0.0,
            disk_queue_length: 0.0,
            average_disk_response_time_ms: 0.0,
            disk_io_efficiency_score: 0.0,
        }
    }
}

// Additional supporting structures would be implemented here...
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAnalysisReport {
    pub analysis_period: AnalysisPeriod,
    pub cpu_analysis: MetricAnalysis,
    pub memory_analysis: MetricAnalysis,
    pub memory_trend: ResourceTrend,
    pub disk_io_summary: DiskIOSummary,
    pub performance_score: f64,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisPeriod {
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricAnalysis {
    pub metric_name: String,
    pub min_value: f64,
    pub max_value: f64,
    pub mean_value: f64,
    pub median_value: f64,
    pub std_deviation: f64,
    pub coefficient_of_variation: f64,
    pub trend: ResourceTrend,
    pub stability_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceTrend {
    Increasing,
    Decreasing,
    Stable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOSummary {
    pub average_read_mbps: f64,
    pub average_write_mbps: f64,
    pub peak_read_mbps: f64,
    pub peak_write_mbps: f64,
}

impl Default for ResourceAnalysisReport {
    fn default() -> Self {
        Self {
            analysis_period: AnalysisPeriod {
                start_time: SystemTime::UNIX_EPOCH,
                end_time: SystemTime::UNIX_EPOCH,
                sample_count: 0,
            },
            cpu_analysis: MetricAnalysis::default(),
            memory_analysis: MetricAnalysis::default(),
            memory_trend: ResourceTrend::Stable,
            disk_io_summary: DiskIOSummary::default(),
            performance_score: 0.0,
            recommendations: Vec::new(),
        }
    }
}

impl Default for MetricAnalysis {
    fn default() -> Self {
        Self {
            metric_name: String::new(),
            min_value: 0.0,
            max_value: 0.0,
            mean_value: 0.0,
            median_value: 0.0,
            std_deviation: 0.0,
            coefficient_of_variation: 0.0,
            trend: ResourceTrend::Stable,
            stability_score: 100.0,
        }
    }
}

impl Default for DiskIOSummary {
    fn default() -> Self {
        Self {
            average_read_mbps: 0.0,
            average_write_mbps: 0.0,
            peak_read_mbps: 0.0,
            peak_write_mbps: 0.0,
        }
    }
}

// Additional resource-related structures...
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAnomaly {
    pub anomaly_type: AnomalyType,
    pub metric_name: String,
    pub severity: AnomalySeverity,
    pub description: String,
    pub detected_at: SystemTime,
    pub confidence_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    StatisticalOutlier,
    MemoryLeak,
    PerformanceDegradation,
    ResourceExhaustion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

// Placeholder structures for compilation
#[derive(Debug, Clone)]
pub struct HistoricalResourceData {
    retention_minutes: u64,
    snapshots: VecDeque<ResourceSnapshot>,
}

impl HistoricalResourceData {
    fn new(retention_minutes: u64) -> Self {
        Self {
            retention_minutes,
            snapshots: VecDeque::new(),
        }
    }
    
    fn add_snapshot(&mut self, snapshot: ResourceSnapshot) {
        self.snapshots.push_back(snapshot);
        
        // Remove old snapshots beyond retention period
        let cutoff_time = SystemTime::now() - Duration::from_secs(self.retention_minutes * 60);
        while let Some(oldest) = self.snapshots.front() {
            if oldest.timestamp < cutoff_time {
                self.snapshots.pop_front();
            } else {
                break;
            }
        }
    }
    
    fn get_snapshots_since(&self, since: SystemTime) -> Vec<ResourceSnapshot> {
        self.snapshots.iter()
            .filter(|s| s.timestamp >= since)
            .cloned()
            .collect()
    }
    
    fn get_recent_snapshots(&self, count: usize) -> Vec<ResourceSnapshot> {
        self.snapshots.iter()
            .rev()
            .take(count)
            .cloned()
            .collect()
    }
}

// Additional placeholder structures
#[derive(Debug, Clone)]
pub struct ResourceLeakDetector {
    threshold_mb: f64,
}

impl ResourceLeakDetector {
    fn new(threshold_mb: f64) -> Self {
        Self { threshold_mb }
    }
    
    async fn check_for_leaks(&self, _snapshot: &ResourceSnapshot) -> Option<LeakAlert> {
        // Implementation would check for resource leaks
        None
    }
}

#[derive(Debug, Clone)]
pub struct ResourceAlertingSystem {
    config: ResourceMonitoringConfig,
}

impl ResourceAlertingSystem {
    fn new(config: ResourceMonitoringConfig) -> Self {
        Self { config }
    }
    
    async fn send_alert(&self, alert: ResourceAlert) {
        println!("RESOURCE ALERT: {:?}", alert);
    }
}

#[derive(Debug, Clone)]
pub struct ResourceExportManager {
    config: ResourceMonitoringConfig,
}

impl ResourceExportManager {
    fn new(config: ResourceMonitoringConfig) -> Self {
        Self { config }
    }
    
    async fn export_snapshot(&self, _snapshot: &ResourceSnapshot) {
        // Implementation would export to configured systems
    }
}

#[derive(Debug, Clone)]
pub enum ResourceAlert {
    Threshold(ThresholdAlert),
    Leak(LeakAlert),
}

#[derive(Debug, Clone)]
pub struct ThresholdAlert {
    pub resource_type: String,
    pub current_value: f64,
    pub threshold_value: f64,
    pub severity: AlertSeverity,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct LeakAlert {
    pub resource_type: String,
    pub leak_rate: f64,
    pub severity: AlertSeverity,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

// Concrete collector implementations would be added here...
struct CpuCollector;
struct MemoryCollector;
struct DiskCollector;
struct NetworkCollector;
struct ProcessCollector;

#[cfg(target_os = "windows")]
struct WindowsPerformanceCounterCollector;

impl CpuCollector {
    async fn new() -> Result<Self> { Ok(Self) }
}

#[async_trait::async_trait]
impl ResourceCollector for CpuCollector {
    async fn collect(&self) -> Result<ResourceData> {
        // Implementation would collect actual CPU metrics
        Ok(ResourceData::Cpu(CpuMetrics::default()))
    }
}

// Similar implementations for other collectors...
impl MemoryCollector {
    async fn new() -> Result<Self> { Ok(Self) }
}

#[async_trait::async_trait]
impl ResourceCollector for MemoryCollector {
    async fn collect(&self) -> Result<ResourceData> {
        Ok(ResourceData::Memory(MemoryMetrics::default()))
    }
}

impl DiskCollector {
    async fn new() -> Result<Self> { Ok(Self) }
}

#[async_trait::async_trait]
impl ResourceCollector for DiskCollector {
    async fn collect(&self) -> Result<ResourceData> {
        Ok(ResourceData::Disk(DiskMetrics::default()))
    }
}

impl NetworkCollector {
    async fn new() -> Result<Self> { Ok(Self) }
}

#[async_trait::async_trait]
impl ResourceCollector for NetworkCollector {
    async fn collect(&self) -> Result<ResourceData> {
        Ok(ResourceData::Network(NetworkMetrics::default()))
    }
}

impl ProcessCollector {
    async fn new() -> Result<Self> { Ok(Self) }
}

#[async_trait::async_trait]
impl ResourceCollector for ProcessCollector {
    async fn collect(&self) -> Result<ResourceData> {
        Ok(ResourceData::Process(ProcessMetrics::default()))
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            network_interfaces: HashMap::new(),
            process_network_bytes_sent: 0,
            process_network_bytes_received: 0,
            total_connections: 0,
            active_connections: 0,
            network_latency_ms: 0.0,
            packet_loss_percent: 0.0,
        }
    }
}

impl Default for ProcessMetrics {
    fn default() -> Self {
        Self {
            process_id: 0,
            thread_count: 0,
            handle_count: 0,
            uptime_seconds: 0.0,
            total_processor_time_ms: 0.0,
            user_processor_time_ms: 0.0,
            privileged_processor_time_ms: 0.0,
            io_read_operations: 0,
            io_write_operations: 0,
            io_other_operations: 0,
        }
    }
}
```

## Success Criteria
- System resource monitoring compiles without errors and starts successfully
- Real-time resource tracking captures CPU, memory, disk, and network metrics accurately
- Process-level tracking provides detailed per-thread resource information
- Windows performance counter integration works on Windows systems
- Resource leak detection identifies memory and handle leaks correctly
- Historical analysis generates meaningful trends and insights
- Anomaly detection flags unusual resource usage patterns
- Export capabilities work with Prometheus and CSV formats

## Time Limit
10 minutes maximum