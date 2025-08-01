//! Production Performance Monitoring System
//! 
//! Comprehensive monitoring solution with real-time metrics collection,
//! alerting, and performance analysis for production deployment.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

use super::config::{MonitoringConfig, AlertThresholds};

/// Main performance monitoring system
pub struct PerformanceMonitor {
    config: MonitoringConfig,
    metrics_collector: Arc<MetricsCollector>,
    alert_manager: Arc<AlertManager>,
    dashboard_updater: Arc<DashboardUpdater>,
    timer_registry: Arc<RwLock<HashMap<String, Vec<Instant>>>>,
    active_timers: Arc<RwLock<HashMap<String, Instant>>>,
}

/// Metrics collection and storage system
pub struct MetricsCollector {
    metrics_store: Arc<RwLock<MetricsStore>>,
    collection_interval: Duration,
    retention_period: Duration,
}

/// Alert management system
pub struct AlertManager {
    thresholds: AlertThresholds,
    active_alerts: Arc<RwLock<HashMap<String, Alert>>>,
    alert_history: Arc<RwLock<Vec<Alert>>>,
}

/// Dashboard update system
pub struct DashboardUpdater {
    enabled: bool,
    port: u16,
    refresh_interval: Duration,
    current_metrics: Arc<RwLock<DashboardMetrics>>,
}

/// Comprehensive metrics storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsStore {
    pub system_metrics: SystemMetrics,
    pub operation_metrics: HashMap<String, OperationMetrics>,
    pub performance_history: Vec<PerformanceSnapshot>,
    pub error_metrics: ErrorMetrics,
    pub resource_usage: ResourceUsage,
}

/// System-wide metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub uptime: Duration,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_response_time: Duration,
    pub requests_per_second: f64,
    pub concurrent_connections: usize,
    pub last_updated: DateTime<Utc>,
}

/// Operation-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMetrics {
    pub operation_name: String,
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub average_duration: Duration,
    pub min_duration: Duration,
    pub max_duration: Duration,
    pub p50_duration: Duration,
    pub p95_duration: Duration,
    pub p99_duration: Duration,
    pub throughput: f64,
    pub error_rate: f32,
    pub last_execution: Option<DateTime<Utc>>,
}

/// Performance snapshot for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: f32,
    pub memory_usage: u64,
    pub memory_percentage: f32,
    pub disk_usage: u64,
    pub network_in: u64,
    pub network_out: u64,
    pub active_connections: usize,
    pub queue_size: usize,
    pub response_time: Duration,
    pub throughput: f64,
    pub error_rate: f32,
}

/// Error tracking metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    pub total_errors: u64,
    pub error_by_type: HashMap<String, u64>,
    pub error_by_component: HashMap<String, u64>,
    pub recent_errors: Vec<ErrorRecord>,
    pub error_rate_trend: Vec<f32>,
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_cores: usize,
    pub cpu_usage_per_core: Vec<f32>,
    pub total_memory: u64,
    pub used_memory: u64,
    pub available_memory: u64,
    pub memory_usage_percentage: f32,
    pub disk_total: u64,
    pub disk_used: u64,
    pub disk_available: u64,
    pub network_interfaces: Vec<NetworkInterface>,
    pub thread_count: usize,
    pub file_descriptors: usize,
}

/// Network interface statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    pub name: String,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub errors: u64,
}

/// Alert definition and tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub triggered_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, String>,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighCpuUsage,
    HighMemoryUsage,
    HighDiskUsage,
    SlowResponseTime,
    HighErrorRate,
    ServiceUnavailable,
    QueueOverflow,
    DatabaseConnectionFailure,
    Custom(String),
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

/// Error record for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRecord {
    pub timestamp: DateTime<Utc>,
    pub error_type: String,
    pub component: String,
    pub message: String,
    pub stack_trace: Option<String>,
    pub context: HashMap<String, String>,
}

/// Dashboard metrics display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    pub current_rps: f64,
    pub average_response_time: Duration,
    pub success_rate: f32,
    pub active_connections: usize,
    pub resource_usage: ResourceUsage,
    pub recent_alerts: Vec<Alert>,
    pub performance_trends: PerformanceTrends,
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub response_time_trend: Vec<(DateTime<Utc>, Duration)>,
    pub throughput_trend: Vec<(DateTime<Utc>, f64)>,
    pub error_rate_trend: Vec<(DateTime<Utc>, f32)>,
    pub resource_usage_trend: Vec<(DateTime<Utc>, f32)>,
}

/// Performance timer for operation measurement
pub struct PerformanceTimer {
    operation: String,
    start_time: Instant,
    monitor: Arc<PerformanceMonitor>,
}

impl PerformanceMonitor {
    /// Create new performance monitoring system
    pub fn new(config: MonitoringConfig) -> Result<Self, MonitoringError> {
        let metrics_collector = Arc::new(MetricsCollector::new(
            config.collection_interval,
            config.retention_period,
        )?);
        
        let alert_manager = Arc::new(AlertManager::new(config.alert_thresholds.clone())?);
        
        let dashboard_updater = Arc::new(DashboardUpdater::new(
            config.dashboard_config.clone(),
        )?);
        
        Ok(Self {
            config,
            metrics_collector,
            alert_manager,
            dashboard_updater,
            timer_registry: Arc::new(RwLock::new(HashMap::new())),
            active_timers: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Start a performance timer for an operation
    pub fn start_timer(&self, operation: &str) -> PerformanceTimer {
        let timer_id = format!("{}_{}", operation, Utc::now().timestamp_nanos_opt().unwrap_or(0));
        let start_time = Instant::now();
        
        // Store active timer
        let mut active_timers = futures::executor::block_on(self.active_timers.write());
        active_timers.insert(timer_id.clone(), start_time);
        
        PerformanceTimer {
            operation: operation.to_string(),
            start_time,
            monitor: Arc::new(self.clone()),
        }
    }
    
    /// Record document processing metrics
    pub async fn record_document_processing(&self, processing_duration: Duration, peak_memory: u64) {
        let operation = "document_processing";
        
        // Record basic metrics
        self.record_operation_success(operation, processing_duration).await;
        
        // Record detailed metrics
        self.metrics_collector.record_processing_metrics(processing_duration, peak_memory).await;
        
        // Check for performance alerts
        let processing_metrics = ProcessingMetrics {
            total_duration: processing_duration,
            memory_peak_usage: peak_memory,
        };
        self.check_performance_alerts(operation, &processing_metrics).await;
    }
    
    /// Record query processing metrics
    pub async fn record_query_processing(&self, processing_duration: Duration, peak_memory: u64) {
        let operation = "query_processing";
        
        // Record basic metrics
        self.record_operation_success(operation, processing_duration).await;
        
        // Record query-specific metrics
        self.metrics_collector.record_query_metrics(processing_duration, peak_memory).await;
        
        // Check for performance alerts
        let processing_metrics = ProcessingMetrics {
            total_duration: processing_duration,
            memory_peak_usage: peak_memory,
        };
        self.check_performance_alerts(operation, &processing_metrics).await;
    }
    
    /// Record operation failure
    pub async fn record_operation_failure(&self, operation: &str, error: &str, duration: Duration) {
        // Update operation metrics
        self.metrics_collector.record_failure(operation, duration).await;
        
        // Record error
        let error_record = ErrorRecord {
            timestamp: Utc::now(),
            error_type: "OperationFailure".to_string(),
            component: operation.to_string(),
            message: error.to_string(),
            stack_trace: None,
            context: HashMap::new(),
        };
        
        self.metrics_collector.record_error(error_record).await;
        
        // Check for error rate alerts
        self.check_error_rate_alerts(operation).await;
    }
    
    /// Get current system metrics
    pub async fn get_current_metrics(&self) -> CurrentPerformanceMetrics {
        let metrics_store = self.metrics_collector.metrics_store.read().await;
        
        CurrentPerformanceMetrics {
            requests_per_second: metrics_store.system_metrics.requests_per_second,
            average_response_time: metrics_store.system_metrics.average_response_time,
            error_rate: self.calculate_current_error_rate(&metrics_store).await,
            throughput: self.calculate_current_throughput(&metrics_store).await,
            memory_usage: metrics_store.resource_usage.used_memory,
            cpu_usage: self.calculate_average_cpu_usage(&metrics_store),
            active_connections: metrics_store.system_metrics.concurrent_connections,
        }
    }
    
    /// Get performance report for time range
    pub async fn generate_performance_report(&self, time_range: TimeRange) -> PerformanceReport {
        let metrics_store = self.metrics_collector.metrics_store.read().await;
        
        let snapshots = self.filter_snapshots_by_time_range(
            &metrics_store.performance_history,
            time_range,
        );
        
        PerformanceReport {
            time_range,
            summary: self.calculate_summary_statistics(&snapshots),
            detailed_metrics: self.extract_detailed_metrics(&snapshots),
            alerts_triggered: self.get_alerts_in_range(time_range).await,
            performance_trends: self.calculate_performance_trends(&snapshots),
            recommendations: self.generate_performance_recommendations(&snapshots).await,
        }
    }
    
    /// Get peak memory usage
    pub fn get_peak_memory_usage(&self) -> u64 {
        // Implementation would track peak memory usage
        // Simplified version returns current usage
        futures::executor::block_on(async {
            let metrics_store = self.metrics_collector.metrics_store.read().await;
            metrics_store.resource_usage.used_memory
        })
    }
    
    /// Get memory usage information
    pub async fn get_memory_usage(&self) -> MemoryUsage {
        let metrics_store = self.metrics_collector.metrics_store.read().await;
        let resource_usage = &metrics_store.resource_usage;
        
        MemoryUsage {
            current_usage: resource_usage.used_memory,
            peak_usage: resource_usage.used_memory, // Simplified
            available: resource_usage.available_memory,
            percentage_used: resource_usage.memory_usage_percentage,
        }
    }
    
    // Private helper methods
    
    async fn record_operation_success(&self, operation: &str, duration: Duration) {
        self.metrics_collector.record_success(operation, duration).await;
    }
    
    async fn check_performance_alerts(&self, operation: &str, metrics: &ProcessingMetrics) {
        // Check response time threshold
        if metrics.total_duration > self.config.alert_thresholds.response_time {
            let alert = Alert {
                id: format!("slow_response_{}_{}", operation, Utc::now().timestamp()),
                alert_type: AlertType::SlowResponseTime,
                severity: AlertSeverity::Warning,
                message: format!(
                    "Operation {} took {}ms, exceeding threshold of {}ms",
                    operation,
                    metrics.total_duration.as_millis(),
                    self.config.alert_thresholds.response_time.as_millis()
                ),
                triggered_at: Utc::now(),
                resolved_at: None,
                metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("operation".to_string(), operation.to_string());
                    metadata.insert("duration".to_string(), metrics.total_duration.as_millis().to_string());
                    metadata
                },
            };
            
            self.alert_manager.trigger_alert(alert).await;
        }
        
        // Check memory usage
        if metrics.memory_peak_usage > (self.config.alert_thresholds.memory_usage * 1_000_000_000.0) as u64 {
            let alert = Alert {
                id: format!("high_memory_{}_{}", operation, Utc::now().timestamp()),
                alert_type: AlertType::HighMemoryUsage,
                severity: AlertSeverity::Critical,
                message: format!(
                    "Operation {} used {}MB memory, exceeding threshold",
                    operation,
                    metrics.memory_peak_usage / 1_000_000
                ),
                triggered_at: Utc::now(),
                resolved_at: None,
                metadata: HashMap::new(),
            };
            
            self.alert_manager.trigger_alert(alert).await;
        }
    }
    
    async fn check_error_rate_alerts(&self, operation: &str) {
        let current_error_rate = self.calculate_operation_error_rate(operation).await;
        
        if current_error_rate > self.config.alert_thresholds.error_rate {
            let alert = Alert {
                id: format!("high_error_rate_{}_{}", operation, Utc::now().timestamp()),
                alert_type: AlertType::HighErrorRate,
                severity: AlertSeverity::Critical,
                message: format!(
                    "Operation {} has error rate of {:.2}%, exceeding threshold of {:.2}%",
                    operation,
                    current_error_rate,
                    self.config.alert_thresholds.error_rate
                ),
                triggered_at: Utc::now(),
                resolved_at: None,
                metadata: HashMap::new(),
            };
            
            self.alert_manager.trigger_alert(alert).await;
        }
    }
    
    async fn calculate_current_error_rate(&self, metrics_store: &MetricsStore) -> f32 {
        let total_requests = metrics_store.system_metrics.total_requests;
        let failed_requests = metrics_store.system_metrics.failed_requests;
        
        if total_requests == 0 {
            0.0
        } else {
            (failed_requests as f32 / total_requests as f32) * 100.0
        }
    }
    
    async fn calculate_current_throughput(&self, metrics_store: &MetricsStore) -> f32 {
        metrics_store.system_metrics.requests_per_second as f32
    }
    
    fn calculate_average_cpu_usage(&self, metrics_store: &MetricsStore) -> f32 {
        let cpu_usage = &metrics_store.resource_usage.cpu_usage_per_core;
        if cpu_usage.is_empty() {
            0.0
        } else {
            cpu_usage.iter().sum::<f32>() / cpu_usage.len() as f32
        }
    }
    
    async fn calculate_operation_error_rate(&self, operation: &str) -> f32 {
        let metrics_store = self.metrics_collector.metrics_store.read().await;
        
        if let Some(op_metrics) = metrics_store.operation_metrics.get(operation) {
            if op_metrics.total_executions == 0 {
                0.0
            } else {
                (op_metrics.failed_executions as f32 / op_metrics.total_executions as f32) * 100.0
            }
        } else {
            0.0
        }
    }
    
    fn filter_snapshots_by_time_range(
        &self,
        snapshots: &[PerformanceSnapshot],
        time_range: TimeRange,
    ) -> Vec<PerformanceSnapshot> {
        snapshots
            .iter()
            .filter(|snapshot| {
                snapshot.timestamp >= time_range.start && snapshot.timestamp <= time_range.end
            })
            .cloned()
            .collect()
    }
    
    fn calculate_summary_statistics(&self, snapshots: &[PerformanceSnapshot]) -> SummaryStatistics {
        if snapshots.is_empty() {
            return SummaryStatistics::default();
        }
        
        let response_times: Vec<Duration> = snapshots.iter().map(|s| s.response_time).collect();
        let throughputs: Vec<f64> = snapshots.iter().map(|s| s.throughput).collect();
        let error_rates: Vec<f32> = snapshots.iter().map(|s| s.error_rate).collect();
        
        SummaryStatistics {
            avg_response_time: Duration::from_nanos(
                (response_times.iter().map(|d| d.as_nanos()).sum::<u128>() / response_times.len() as u128) as u64
            ),
            max_response_time: response_times.iter().max().cloned().unwrap_or_default(),
            min_response_time: response_times.iter().min().cloned().unwrap_or_default(),
            avg_throughput: throughputs.iter().sum::<f64>() / throughputs.len() as f64,
            max_throughput: throughputs.iter().fold(0.0, |a, &b| a.max(b)),
            min_throughput: throughputs.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            avg_error_rate: error_rates.iter().sum::<f32>() / error_rates.len() as f32,
            max_error_rate: error_rates.iter().fold(0.0, |a, &b| a.max(b)),
        }
    }
    
    fn extract_detailed_metrics(&self, snapshots: &[PerformanceSnapshot]) -> DetailedMetrics {
        DetailedMetrics {
            cpu_usage_history: snapshots.iter().map(|s| (s.timestamp, s.cpu_usage)).collect(),
            memory_usage_history: snapshots.iter().map(|s| (s.timestamp, s.memory_usage)).collect(),
            response_time_history: snapshots.iter().map(|s| (s.timestamp, s.response_time)).collect(),
            throughput_history: snapshots.iter().map(|s| (s.timestamp, s.throughput)).collect(),
            error_rate_history: snapshots.iter().map(|s| (s.timestamp, s.error_rate)).collect(),
        }
    }
    
    async fn get_alerts_in_range(&self, time_range: TimeRange) -> Vec<Alert> {
        let alert_history = self.alert_manager.alert_history.read().await;
        
        alert_history
            .iter()
            .filter(|alert| {
                alert.triggered_at >= time_range.start && alert.triggered_at <= time_range.end
            })
            .cloned()
            .collect()
    }
    
    fn calculate_performance_trends(&self, snapshots: &[PerformanceSnapshot]) -> PerformanceTrends {
        PerformanceTrends {
            response_time_trend: snapshots.iter().map(|s| (s.timestamp, s.response_time)).collect(),
            throughput_trend: snapshots.iter().map(|s| (s.timestamp, s.throughput)).collect(),
            error_rate_trend: snapshots.iter().map(|s| (s.timestamp, s.error_rate)).collect(),
            resource_usage_trend: snapshots.iter().map(|s| (s.timestamp, s.cpu_usage)).collect(),
        }
    }
    
    async fn generate_performance_recommendations(&self, snapshots: &[PerformanceSnapshot]) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if let Some(latest) = snapshots.last() {
            if latest.cpu_usage > 80.0 {
                recommendations.push("Consider scaling up CPU resources or optimizing CPU-intensive operations".to_string());
            }
            
            if latest.memory_percentage > 85.0 {
                recommendations.push("Memory usage is high - consider increasing memory or optimizing memory usage".to_string());
            }
            
            if latest.response_time > Duration::from_millis(1000) {
                recommendations.push("Response times are slow - investigate bottlenecks and optimize critical paths".to_string());
            }
            
            if latest.error_rate > 5.0 {
                recommendations.push("Error rate is elevated - investigate and fix underlying issues".to_string());
            }
        }
        
        recommendations
    }
}

impl Clone for PerformanceMonitor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics_collector: Arc::clone(&self.metrics_collector),
            alert_manager: Arc::clone(&self.alert_manager),
            dashboard_updater: Arc::clone(&self.dashboard_updater),
            timer_registry: Arc::clone(&self.timer_registry),
            active_timers: Arc::clone(&self.active_timers),
        }
    }
}

impl Drop for PerformanceTimer {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        let operation = self.operation.clone();
        let monitor = Arc::clone(&self.monitor);
        
        // Record the timing asynchronously
        tokio::spawn(async move {
            monitor.record_operation_success(&operation, duration).await;
        });
    }
}

impl MetricsCollector {
    fn new(collection_interval: Duration, retention_period: Duration) -> Result<Self, MonitoringError> {
        Ok(Self {
            metrics_store: Arc::new(RwLock::new(MetricsStore::default())),
            collection_interval,
            retention_period,
        })
    }
    
    async fn record_success(&self, operation: &str, duration: Duration) {
        let mut store = self.metrics_store.write().await;
        
        // Update system metrics
        store.system_metrics.total_requests += 1;
        store.system_metrics.successful_requests += 1;
        
        // Update operation metrics
        let op_metrics = store.operation_metrics
            .entry(operation.to_string())
            .or_insert_with(|| OperationMetrics::new(operation));
        
        op_metrics.record_success(duration);
    }
    
    async fn record_failure(&self, operation: &str, duration: Duration) {
        let mut store = self.metrics_store.write().await;
        
        // Update system metrics
        store.system_metrics.total_requests += 1;
        store.system_metrics.failed_requests += 1;
        
        // Update operation metrics
        let op_metrics = store.operation_metrics
            .entry(operation.to_string())
            .or_insert_with(|| OperationMetrics::new(operation));
        
        op_metrics.record_failure(duration);
    }
    
    async fn record_processing_metrics(&self, duration: Duration, peak_memory: u64) {
        // Implementation would record detailed processing metrics
    }
    
    async fn record_query_metrics(&self, duration: Duration, peak_memory: u64) {
        // Implementation would record detailed query metrics
    }
    
    async fn record_error(&self, error_record: ErrorRecord) {
        let mut store = self.metrics_store.write().await;
        store.error_metrics.record_error(error_record);
    }
}

impl AlertManager {
    fn new(thresholds: AlertThresholds) -> Result<Self, MonitoringError> {
        Ok(Self {
            thresholds,
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    async fn trigger_alert(&self, alert: Alert) {
        // Add to active alerts
        let mut active_alerts = self.active_alerts.write().await;
        active_alerts.insert(alert.id.clone(), alert.clone());
        
        // Add to history
        let mut alert_history = self.alert_history.write().await;
        alert_history.push(alert.clone());
        
        // Trigger alert notification (implementation would send notifications)
        self.send_alert_notification(&alert).await;
    }
    
    async fn send_alert_notification(&self, _alert: &Alert) {
        // Implementation would send notifications via email, Slack, etc.
    }
}

impl DashboardUpdater {
    fn new(config: super::config::DashboardConfig) -> Result<Self, MonitoringError> {
        Ok(Self {
            enabled: config.enabled,
            port: config.port,
            refresh_interval: config.refresh_interval,
            current_metrics: Arc::new(RwLock::new(DashboardMetrics::default())),
        })
    }
    
    async fn update_processing_metrics(&self, _duration: Duration, _peak_memory: u64) {
        // Implementation would update dashboard with processing metrics
    }
    
    async fn update_query_metrics(&self, _duration: Duration, _peak_memory: u64) {
        // Implementation would update dashboard with query metrics
    }
}

impl OperationMetrics {
    fn new(operation_name: &str) -> Self {
        Self {
            operation_name: operation_name.to_string(),
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            average_duration: Duration::from_millis(0),
            min_duration: Duration::from_millis(0),
            max_duration: Duration::from_millis(0),
            p50_duration: Duration::from_millis(0),
            p95_duration: Duration::from_millis(0),
            p99_duration: Duration::from_millis(0),
            throughput: 0.0,
            error_rate: 0.0,
            last_execution: None,
        }
    }
    
    fn record_success(&mut self, duration: Duration) {
        self.total_executions += 1;
        self.successful_executions += 1;
        self.update_duration_stats(duration);
        self.last_execution = Some(Utc::now());
    }
    
    fn record_failure(&mut self, duration: Duration) {
        self.total_executions += 1;
        self.failed_executions += 1;
        self.update_duration_stats(duration);
        self.update_error_rate();
        self.last_execution = Some(Utc::now());
    }
    
    fn update_duration_stats(&mut self, duration: Duration) {
        if self.total_executions == 1 {
            self.min_duration = duration;
            self.max_duration = duration;
        } else {
            self.min_duration = self.min_duration.min(duration);
            self.max_duration = self.max_duration.max(duration);
        }
        
        // Update average (simplified calculation)
        let total_nanos = self.average_duration.as_nanos() as u64 * (self.total_executions - 1) + duration.as_nanos() as u64;
        self.average_duration = Duration::from_nanos(total_nanos / self.total_executions);
    }
    
    fn update_error_rate(&mut self) {
        self.error_rate = (self.failed_executions as f32 / self.total_executions as f32) * 100.0;
    }
}

impl ErrorMetrics {
    fn record_error(&mut self, error_record: ErrorRecord) {
        self.total_errors += 1;
        
        *self.error_by_type.entry(error_record.error_type.clone()).or_insert(0) += 1;
        *self.error_by_component.entry(error_record.component.clone()).or_insert(0) += 1;
        
        self.recent_errors.push(error_record);
        
        // Keep only recent errors (last 1000)
        if self.recent_errors.len() > 1000 {
            self.recent_errors.remove(0);
        }
    }
}

impl Default for MetricsStore {
    fn default() -> Self {
        Self {
            system_metrics: SystemMetrics {
                uptime: Duration::from_secs(0),
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                average_response_time: Duration::from_millis(0),
                requests_per_second: 0.0,
                concurrent_connections: 0,
                last_updated: Utc::now(),
            },
            operation_metrics: HashMap::new(),
            performance_history: Vec::new(),
            error_metrics: ErrorMetrics {
                total_errors: 0,
                error_by_type: HashMap::new(),
                error_by_component: HashMap::new(),
                recent_errors: Vec::new(),
                error_rate_trend: Vec::new(),
            },
            resource_usage: ResourceUsage {
                cpu_cores: 1,
                cpu_usage_per_core: vec![0.0],
                total_memory: 0,
                used_memory: 0,
                available_memory: 0,
                memory_usage_percentage: 0.0,
                disk_total: 0,
                disk_used: 0,
                disk_available: 0,
                network_interfaces: Vec::new(),
                thread_count: 0,
                file_descriptors: 0,
            },
        }
    }
}

impl Default for DashboardMetrics {
    fn default() -> Self {
        Self {
            current_rps: 0.0,
            average_response_time: Duration::from_millis(0),
            success_rate: 100.0,
            active_connections: 0,
            resource_usage: ResourceUsage {
                cpu_cores: 1,
                cpu_usage_per_core: vec![0.0],
                total_memory: 0,
                used_memory: 0,
                available_memory: 0,
                memory_usage_percentage: 0.0,
                disk_total: 0,
                disk_used: 0,
                disk_available: 0,
                network_interfaces: Vec::new(),
                thread_count: 0,
                file_descriptors: 0,
            },
            recent_alerts: Vec::new(),
            performance_trends: PerformanceTrends {
                response_time_trend: Vec::new(),
                throughput_trend: Vec::new(),
                error_rate_trend: Vec::new(),
                resource_usage_trend: Vec::new(),
            },
        }
    }
}

/// Current performance metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentPerformanceMetrics {
    pub requests_per_second: f64,
    pub average_response_time: Duration,
    pub error_rate: f32,
    pub throughput: f32,
    pub memory_usage: u64,
    pub cpu_usage: f32,
    pub active_connections: usize,
}

/// Time range for analysis
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Performance analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub time_range: TimeRange,
    pub summary: SummaryStatistics,
    pub detailed_metrics: DetailedMetrics,
    pub alerts_triggered: Vec<Alert>,
    pub performance_trends: PerformanceTrends,
    pub recommendations: Vec<String>,
}

/// Summary statistics for performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryStatistics {
    pub avg_response_time: Duration,
    pub max_response_time: Duration,
    pub min_response_time: Duration,
    pub avg_throughput: f64,
    pub max_throughput: f64,
    pub min_throughput: f64,
    pub avg_error_rate: f32,
    pub max_error_rate: f32,
}

impl Default for SummaryStatistics {
    fn default() -> Self {
        Self {
            avg_response_time: Duration::from_millis(0),
            max_response_time: Duration::from_millis(0),
            min_response_time: Duration::from_millis(0),
            avg_throughput: 0.0,
            max_throughput: 0.0,
            min_throughput: 0.0,
            avg_error_rate: 0.0,
            max_error_rate: 0.0,
        }
    }
}

/// Detailed metrics for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedMetrics {
    pub cpu_usage_history: Vec<(DateTime<Utc>, f32)>,
    pub memory_usage_history: Vec<(DateTime<Utc>, u64)>,
    pub response_time_history: Vec<(DateTime<Utc>, Duration)>,
    pub throughput_history: Vec<(DateTime<Utc>, f64)>,
    pub error_rate_history: Vec<(DateTime<Utc>, f32)>,
}

/// Monitoring error types
#[derive(Debug, thiserror::Error)]
pub enum MonitoringError {
    #[error("Initialization error: {0}")]
    InitializationError(String),
    #[error("Collection error: {0}")]
    CollectionError(String),
    #[error("Alert error: {0}")]
    AlertError(String),
    #[error("Dashboard error: {0}")]
    DashboardError(String),
}

// Supporting types for processing metrics
#[derive(Debug, Clone)]
pub struct ProcessingMetrics {
    pub total_duration: Duration,
    pub memory_peak_usage: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub current_usage: u64,
    pub peak_usage: u64,
    pub available: u64,
    pub percentage_used: f32,
}