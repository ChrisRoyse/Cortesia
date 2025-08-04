# Micro Task 39: Performance Monitoring

**Priority**: CRITICAL  
**Estimated Time**: 30 minutes  
**Dependencies**: Task 38 (Caching System) completed  
**Skills Required**: Performance monitoring, metrics collection, alerting

## Objective

Implement comprehensive real-time performance monitoring system with metrics collection, alerting, and visualization capabilities to ensure production system health and optimize performance bottlenecks.

## Context

The production system requires continuous monitoring of query performance, resource utilization, cache effectiveness, and system health. The monitoring system must provide actionable insights and early warning of performance degradation.

## Specifications

### Core Monitoring Requirements

1. **Real-Time Metrics Collection**
   - Query performance metrics
   - Resource utilization tracking
   - Cache effectiveness monitoring
   - System health indicators

2. **Performance Alerting**
   - Threshold-based alerts
   - Anomaly detection
   - Performance regression detection
   - Resource exhaustion warnings

3. **Monitoring Targets**
   - 99.9% uptime monitoring
   - < 100ms metric collection overhead
   - Real-time dashboard updates
   - Historical trend analysis

## Implementation Guide

### Step 1: Performance Metrics Collector
```rust
// File: src/monitoring/performance_monitor.rs

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicF64, Ordering};
use serde::{Serialize, Deserialize};

pub struct PerformanceMonitor {
    // Metric collectors
    query_metrics: Arc<QueryMetricsCollector>,
    system_metrics: Arc<SystemMetricsCollector>,
    cache_metrics: Arc<CacheMetricsCollector>,
    
    // Alerting system
    alert_manager: Arc<AlertManager>,
    
    // Data storage
    metrics_store: Arc<RwLock<MetricsStore>>,
    
    // Configuration
    config: PerformanceMonitorConfig,
    
    // Real-time monitoring
    monitoring_tasks: Vec<tokio::task::JoinHandle<()>>,
}

#[derive(Debug, Clone)]
pub struct PerformanceMonitorConfig {
    pub collection_interval: Duration,
    pub metrics_retention: Duration,
    pub enable_alerting: bool,
    pub enable_anomaly_detection: bool,
    pub dashboard_update_interval: Duration,
    pub export_prometheus: bool,
}

impl Default for PerformanceMonitorConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_secs(1),
            metrics_retention: Duration::from_hours(24),
            enable_alerting: true,
            enable_anomaly_detection: true,
            dashboard_update_interval: Duration::from_secs(5),
            export_prometheus: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: Instant,
    pub query_metrics: QueryPerformanceMetrics,
    pub system_metrics: SystemPerformanceMetrics,
    pub cache_metrics: CachePerformanceMetrics,
    pub health_score: f32,
}

impl PerformanceMonitor {
    pub async fn new(config: PerformanceMonitorConfig) -> Result<Self> {
        Ok(Self {
            query_metrics: Arc::new(QueryMetricsCollector::new()),
            system_metrics: Arc::new(SystemMetricsCollector::new()),
            cache_metrics: Arc::new(CacheMetricsCollector::new()),
            alert_manager: Arc::new(AlertManager::new()),
            metrics_store: Arc::new(RwLock::new(MetricsStore::new(config.metrics_retention))),
            config,
            monitoring_tasks: Vec::new(),
        })
    }
    
    pub async fn start_monitoring(&mut self) -> Result<()> {
        // Start real-time metrics collection
        let metrics_task = self.start_metrics_collection().await;
        self.monitoring_tasks.push(metrics_task);
        
        // Start alerting system
        if self.config.enable_alerting {
            let alert_task = self.start_alerting().await;
            self.monitoring_tasks.push(alert_task);
        }
        
        // Start anomaly detection
        if self.config.enable_anomaly_detection {
            let anomaly_task = self.start_anomaly_detection().await;
            self.monitoring_tasks.push(anomaly_task);
        }
        
        // Start dashboard updates
        let dashboard_task = self.start_dashboard_updates().await;
        self.monitoring_tasks.push(dashboard_task);
        
        Ok(())
    }
    
    async fn start_metrics_collection(&self) -> tokio::task::JoinHandle<()> {
        let query_collector = self.query_metrics.clone();
        let system_collector = self.system_metrics.clone();
        let cache_collector = self.cache_metrics.clone();
        let store = self.metrics_store.clone();
        let interval = self.config.collection_interval;
        
        tokio::spawn(async move {
            let mut collection_interval = tokio::time::interval(interval);
            
            loop {
                collection_interval.tick().await;
                
                // Collect metrics from all sources
                let query_metrics = query_collector.collect().await;
                let system_metrics = system_collector.collect().await;
                let cache_metrics = cache_collector.collect().await;
                
                // Calculate health score
                let health_score = Self::calculate_health_score(
                    &query_metrics,
                    &system_metrics,
                    &cache_metrics,
                );
                
                // Create performance snapshot
                let snapshot = PerformanceSnapshot {
                    timestamp: Instant::now(),
                    query_metrics,
                    system_metrics,
                    cache_metrics,
                    health_score,
                };
                
                // Store snapshot
                store.write().await.add_snapshot(snapshot);
            }
        })
    }
    
    fn calculate_health_score(
        query_metrics: &QueryPerformanceMetrics,
        system_metrics: &SystemPerformanceMetrics,
        cache_metrics: &CachePerformanceMetrics,
    ) -> f32 {
        let mut score = 100.0;
        
        // Query performance impact
        if query_metrics.avg_response_time > Duration::from_millis(50) {
            score -= 20.0;
        }
        if query_metrics.error_rate > 0.05 {
            score -= 30.0;
        }
        
        // System resource impact
        if system_metrics.cpu_usage > 0.8 {
            score -= 15.0;
        }
        if system_metrics.memory_usage > 0.9 {
            score -= 25.0;
        }
        
        // Cache effectiveness impact
        if cache_metrics.hit_rate < 0.7 {
            score -= 10.0;
        }
        
        score.max(0.0)
    }
    
    pub async fn record_query_execution(&self, duration: Duration, success: bool) {
        self.query_metrics.record_execution(duration, success).await;
    }
    
    pub async fn get_current_performance(&self) -> PerformanceSnapshot {
        let query_metrics = self.query_metrics.collect().await;
        let system_metrics = self.system_metrics.collect().await;
        let cache_metrics = self.cache_metrics.collect().await;
        
        PerformanceSnapshot {
            timestamp: Instant::now(),
            health_score: Self::calculate_health_score(
                &query_metrics,
                &system_metrics,
                &cache_metrics,
            ),
            query_metrics,
            system_metrics,
            cache_metrics,
        }
    }
}
```

### Step 2: Query Performance Metrics
```rust
// File: src/monitoring/query_metrics.rs

pub struct QueryMetricsCollector {
    // Counters
    total_queries: AtomicU64,
    successful_queries: AtomicU64,
    failed_queries: AtomicU64,
    
    // Timing metrics
    response_times: Arc<RwLock<VecDeque<Duration>>>,
    processing_times: Arc<RwLock<VecDeque<Duration>>>,
    
    // Throughput tracking
    queries_per_second: Arc<RwLock<VecDeque<f64>>>,
    last_measurement: Arc<RwLock<Instant>>,
    
    // Query type breakdown
    query_type_metrics: Arc<RwLock<HashMap<QueryIntentType, QueryTypeMetrics>>>,
    
    // Real-time stats
    current_concurrent_queries: AtomicU64,
    peak_concurrent_queries: AtomicU64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceMetrics {
    pub total_queries: u64,
    pub success_rate: f32,
    pub error_rate: f32,
    pub avg_response_time: Duration,
    pub p95_response_time: Duration,
    pub p99_response_time: Duration,
    pub queries_per_second: f64,
    pub current_concurrent: u64,
    pub peak_concurrent: u64,
    pub query_type_breakdown: HashMap<QueryIntentType, QueryTypeMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryTypeMetrics {
    pub count: u64,
    pub avg_response_time: Duration,
    pub success_rate: f32,
    pub cache_hit_rate: f32,
}

impl QueryMetricsCollector {
    pub fn new() -> Self {
        Self {
            total_queries: AtomicU64::new(0),
            successful_queries: AtomicU64::new(0),
            failed_queries: AtomicU64::new(0),
            response_times: Arc::new(RwLock::new(VecDeque::new())),
            processing_times: Arc::new(RwLock::new(VecDeque::new())),
            queries_per_second: Arc::new(RwLock::new(VecDeque::new())),
            last_measurement: Arc::new(RwLock::new(Instant::now())),
            query_type_metrics: Arc::new(RwLock::new(HashMap::new())),
            current_concurrent_queries: AtomicU64::new(0),
            peak_concurrent_queries: AtomicU64::new(0),
        }
    }
    
    pub async fn record_execution(&self, duration: Duration, success: bool) {
        // Update counters
        self.total_queries.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successful_queries.fetch_add(1, Ordering::Relaxed);
        } else {
            self.failed_queries.fetch_add(1, Ordering::Relaxed);
        }
        
        // Record timing
        let mut response_times = self.response_times.write().await;
        response_times.push_back(duration);
        
        // Keep only recent measurements (last 1000)
        if response_times.len() > 1000 {
            response_times.pop_front();
        }
        
        // Update throughput calculation
        self.update_throughput().await;
    }
    
    pub async fn record_query_start(&self) -> QueryExecutionGuard {
        let current = self.current_concurrent_queries.fetch_add(1, Ordering::Relaxed) + 1;
        
        // Update peak if necessary
        loop {
            let peak = self.peak_concurrent_queries.load(Ordering::Relaxed);
            if current <= peak {
                break;
            }
            
            if self.peak_concurrent_queries
                .compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok() {
                break;
            }
        }
        
        QueryExecutionGuard {
            collector: self,
            start_time: Instant::now(),
        }
    }
    
    async fn update_throughput(&self) {
        let now = Instant::now();
        let mut last_measurement = self.last_measurement.write().await;
        let time_diff = now.duration_since(*last_measurement);
        
        if time_diff >= Duration::from_secs(1) {
            let queries_in_period = self.total_queries.load(Ordering::Relaxed);
            let qps = queries_in_period as f64 / time_diff.as_secs_f64();
            
            let mut qps_history = self.queries_per_second.write().await;
            qps_history.push_back(qps);
            
            // Keep only recent measurements
            if qps_history.len() > 300 { // 5 minutes at 1-second intervals
                qps_history.pop_front();
            }
            
            *last_measurement = now;
        }
    }
    
    pub async fn collect(&self) -> QueryPerformanceMetrics {
        let total = self.total_queries.load(Ordering::Relaxed);
        let successful = self.successful_queries.load(Ordering::Relaxed);
        let failed = self.failed_queries.load(Ordering::Relaxed);
        
        let success_rate = if total > 0 {
            successful as f32 / total as f32
        } else {
            0.0
        };
        
        let error_rate = if total > 0 {
            failed as f32 / total as f32
        } else {
            0.0
        };
        
        // Calculate response time percentiles
        let response_times = self.response_times.read().await;
        let (avg_response_time, p95_response_time, p99_response_time) = 
            self.calculate_percentiles(&response_times);
        
        // Calculate average QPS
        let qps_history = self.queries_per_second.read().await;
        let queries_per_second = if !qps_history.is_empty() {
            qps_history.iter().sum::<f64>() / qps_history.len() as f64
        } else {
            0.0
        };
        
        QueryPerformanceMetrics {
            total_queries: total,
            success_rate,
            error_rate,
            avg_response_time,
            p95_response_time,
            p99_response_time,
            queries_per_second,
            current_concurrent: self.current_concurrent_queries.load(Ordering::Relaxed),
            peak_concurrent: self.peak_concurrent_queries.load(Ordering::Relaxed),
            query_type_breakdown: self.query_type_metrics.read().await.clone(),
        }
    }
    
    fn calculate_percentiles(&self, times: &VecDeque<Duration>) -> (Duration, Duration, Duration) {
        if times.is_empty() {
            return (Duration::ZERO, Duration::ZERO, Duration::ZERO);
        }
        
        let mut sorted_times: Vec<Duration> = times.iter().copied().collect();
        sorted_times.sort();
        
        let len = sorted_times.len();
        let avg = Duration::from_nanos(
            sorted_times.iter().map(|d| d.as_nanos()).sum::<u128>() / len as u128
        );
        
        let p95_idx = (len as f64 * 0.95) as usize;
        let p99_idx = (len as f64 * 0.99) as usize;
        
        let p95 = sorted_times.get(p95_idx).copied().unwrap_or(Duration::ZERO);
        let p99 = sorted_times.get(p99_idx).copied().unwrap_or(Duration::ZERO);
        
        (avg, p95, p99)
    }
}

pub struct QueryExecutionGuard<'a> {
    collector: &'a QueryMetricsCollector,
    start_time: Instant,
}

impl<'a> Drop for QueryExecutionGuard<'a> {
    fn drop(&mut self) {
        self.collector.current_concurrent_queries.fetch_sub(1, Ordering::Relaxed);
    }
}
```

### Step 3: System Resource Monitoring
```rust
// File: src/monitoring/system_metrics.rs

use sysinfo::{System, SystemExt, CpuExt, ProcessExt};

pub struct SystemMetricsCollector {
    system_info: Arc<RwLock<System>>,
    process_metrics: Arc<ProcessMetrics>,
    memory_tracker: Arc<MemoryTracker>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPerformanceMetrics {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    pub disk_usage: f32,
    pub network_rx_mb: f64,
    pub network_tx_mb: f64,
    pub thread_count: usize,
    pub handle_count: usize,
    pub uptime: Duration,
}

impl SystemMetricsCollector {
    pub fn new() -> Self {
        Self {
            system_info: Arc::new(RwLock::new(System::new_all())),
            process_metrics: Arc::new(ProcessMetrics::new()),
            memory_tracker: Arc::new(MemoryTracker::new()),
        }
    }
    
    pub async fn collect(&self) -> SystemPerformanceMetrics {
        // Update system information
        self.system_info.write().await.refresh_all();
        let system = self.system_info.read().await;
        
        // CPU usage
        let cpu_usage = system.global_cpu_info().cpu_usage();
        
        // Memory information
        let memory_total_mb = system.total_memory() / 1024 / 1024;
        let memory_used_mb = system.used_memory() / 1024 / 1024;
        let memory_usage = memory_used_mb as f32 / memory_total_mb as f32;
        
        // Process-specific metrics
        let process_metrics = self.process_metrics.collect().await;
        
        SystemPerformanceMetrics {
            cpu_usage,
            memory_usage,
            memory_total_mb,
            memory_used_mb,
            disk_usage: 0.0, // Would implement disk monitoring
            network_rx_mb: 0.0, // Would implement network monitoring
            network_tx_mb: 0.0,
            thread_count: process_metrics.thread_count,
            handle_count: process_metrics.handle_count,
            uptime: process_metrics.uptime,
        }
    }
}

struct ProcessMetrics {
    start_time: Instant,
}

impl ProcessMetrics {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }
    
    async fn collect(&self) -> ProcessSpecificMetrics {
        ProcessSpecificMetrics {
            thread_count: std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1),
            handle_count: 0, // Platform-specific implementation needed
            uptime: self.start_time.elapsed(),
        }
    }
}

#[derive(Debug, Clone)]
struct ProcessSpecificMetrics {
    thread_count: usize,
    handle_count: usize,
    uptime: Duration,
}
```

### Step 4: Alerting and Anomaly Detection
```rust
// File: src/monitoring/alerting.rs

pub struct AlertManager {
    alert_rules: Vec<AlertRule>,
    alert_channels: Vec<Box<dyn AlertChannel>>,
    active_alerts: Arc<RwLock<HashMap<AlertId, ActiveAlert>>>,
    anomaly_detector: Arc<AnomalyDetector>,
}

#[derive(Debug, Clone)]
pub struct AlertRule {
    pub id: AlertId,
    pub name: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub duration: Duration,
    pub severity: AlertSeverity,
}

#[derive(Debug, Clone)]
pub enum AlertCondition {
    ResponseTimeHigh,
    ErrorRateHigh,
    CpuUsageHigh,
    MemoryUsageHigh,
    CacheHitRateLow,
    ThroughputLow,
    ConcurrentQueriesHigh,
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

impl AlertManager {
    pub fn new() -> Self {
        let mut alert_rules = Vec::new();
        
        // Default alert rules
        alert_rules.push(AlertRule {
            id: AlertId::generate(),
            name: "High Response Time".to_string(),
            condition: AlertCondition::ResponseTimeHigh,
            threshold: 100.0, // 100ms
            duration: Duration::from_secs(30),
            severity: AlertSeverity::Warning,
        });
        
        alert_rules.push(AlertRule {
            id: AlertId::generate(),
            name: "High Error Rate".to_string(),
            condition: AlertCondition::ErrorRateHigh,
            threshold: 0.05, // 5%
            duration: Duration::from_secs(60),
            severity: AlertSeverity::Critical,
        });
        
        alert_rules.push(AlertRule {
            id: AlertId::generate(),
            name: "High CPU Usage".to_string(),
            condition: AlertCondition::CpuUsageHigh,
            threshold: 80.0, // 80%
            duration: Duration::from_secs(120),
            severity: AlertSeverity::Warning,
        });
        
        Self {
            alert_rules,
            alert_channels: Vec::new(),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            anomaly_detector: Arc::new(AnomalyDetector::new()),
        }
    }
    
    pub async fn check_alerts(&self, snapshot: &PerformanceSnapshot) -> Vec<AlertTrigger> {
        let mut triggered_alerts = Vec::new();
        
        for rule in &self.alert_rules {
            if self.evaluate_alert_rule(rule, snapshot).await {
                triggered_alerts.push(AlertTrigger {
                    rule_id: rule.id,
                    timestamp: Instant::now(),
                    value: self.get_metric_value(rule, snapshot),
                    threshold: rule.threshold,
                });
            }
        }
        
        // Check for anomalies
        if let Some(anomaly) = self.anomaly_detector.detect_anomaly(snapshot).await {
            triggered_alerts.push(AlertTrigger {
                rule_id: AlertId::anomaly(),
                timestamp: Instant::now(),
                value: anomaly.score,
                threshold: anomaly.threshold,
            });
        }
        
        triggered_alerts
    }
    
    async fn evaluate_alert_rule(&self, rule: &AlertRule, snapshot: &PerformanceSnapshot) -> bool {
        let current_value = self.get_metric_value(rule, snapshot);
        
        match rule.condition {
            AlertCondition::ResponseTimeHigh => current_value > rule.threshold,
            AlertCondition::ErrorRateHigh => current_value > rule.threshold,
            AlertCondition::CpuUsageHigh => current_value > rule.threshold,
            AlertCondition::MemoryUsageHigh => current_value > rule.threshold,
            AlertCondition::CacheHitRateLow => current_value < rule.threshold,
            AlertCondition::ThroughputLow => current_value < rule.threshold,
            AlertCondition::ConcurrentQueriesHigh => current_value > rule.threshold,
        }
    }
    
    fn get_metric_value(&self, rule: &AlertRule, snapshot: &PerformanceSnapshot) -> f64 {
        match rule.condition {
            AlertCondition::ResponseTimeHigh => snapshot.query_metrics.avg_response_time.as_millis() as f64,
            AlertCondition::ErrorRateHigh => snapshot.query_metrics.error_rate as f64,
            AlertCondition::CpuUsageHigh => snapshot.system_metrics.cpu_usage as f64,
            AlertCondition::MemoryUsageHigh => snapshot.system_metrics.memory_usage as f64 * 100.0,
            AlertCondition::CacheHitRateLow => snapshot.cache_metrics.hit_rate as f64,
            AlertCondition::ThroughputLow => snapshot.query_metrics.queries_per_second,
            AlertCondition::ConcurrentQueriesHigh => snapshot.query_metrics.current_concurrent as f64,
        }
    }
}

pub struct AnomalyDetector {
    baseline_metrics: Arc<RwLock<VecDeque<PerformanceSnapshot>>>,
    anomaly_threshold: f32,
}

impl AnomalyDetector {
    pub fn new() -> Self {
        Self {
            baseline_metrics: Arc::new(RwLock::new(VecDeque::new())),
            anomaly_threshold: 2.0, // 2 standard deviations
        }
    }
    
    pub async fn detect_anomaly(&self, snapshot: &PerformanceSnapshot) -> Option<AnomalyAlert> {
        let baseline = self.baseline_metrics.read().await;
        
        if baseline.len() < 30 { // Need baseline data
            return None;
        }
        
        // Check for response time anomalies
        let response_times: Vec<f64> = baseline
            .iter()
            .map(|s| s.query_metrics.avg_response_time.as_millis() as f64)
            .collect();
        
        let (mean, std_dev) = self.calculate_stats(&response_times);
        let current_response_time = snapshot.query_metrics.avg_response_time.as_millis() as f64;
        
        let z_score = (current_response_time - mean) / std_dev;
        
        if z_score.abs() > self.anomaly_threshold as f64 {
            return Some(AnomalyAlert {
                metric: "response_time".to_string(),
                score: z_score as f32,
                threshold: self.anomaly_threshold,
                current_value: current_response_time,
                expected_range: (mean - std_dev * 2.0, mean + std_dev * 2.0),
            });
        }
        
        None
    }
    
    fn calculate_stats(&self, values: &[f64]) -> (f64, f64) {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        
        (mean, std_dev)
    }
}
```

### Step 5: Dashboard and Visualization
```rust
// File: src/monitoring/dashboard.rs

pub struct PerformanceDashboard {
    metrics_store: Arc<RwLock<MetricsStore>>,
    live_metrics: Arc<RwLock<PerformanceSnapshot>>,
    dashboard_config: DashboardConfig,
}

#[derive(Debug, Clone)]
pub struct DashboardConfig {
    pub update_interval: Duration,
    pub chart_time_window: Duration,
    pub enable_real_time: bool,
    pub export_format: ExportFormat,
}

#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Prometheus,
    InfluxDB,
}

impl PerformanceDashboard {
    pub async fn get_dashboard_data(&self) -> DashboardData {
        let current_snapshot = self.live_metrics.read().await.clone();
        let historical_data = self.get_historical_data().await;
        
        DashboardData {
            current_metrics: current_snapshot,
            historical_charts: historical_data,
            alerts: self.get_active_alerts().await,
            health_trends: self.calculate_health_trends().await,
        }
    }
    
    async fn get_historical_data(&self) -> Vec<ChartData> {
        let store = self.metrics_store.read().await;
        let snapshots = store.get_recent_snapshots(self.dashboard_config.chart_time_window);
        
        vec![
            self.create_response_time_chart(&snapshots),
            self.create_throughput_chart(&snapshots),
            self.create_resource_usage_chart(&snapshots),
            self.create_cache_performance_chart(&snapshots),
        ]
    }
    
    fn create_response_time_chart(&self, snapshots: &[PerformanceSnapshot]) -> ChartData {
        ChartData {
            title: "Response Time".to_string(),
            chart_type: ChartType::Line,
            datasets: vec![
                Dataset {
                    label: "Average".to_string(),
                    data: snapshots
                        .iter()
                        .map(|s| DataPoint {
                            timestamp: s.timestamp,
                            value: s.query_metrics.avg_response_time.as_millis() as f64,
                        })
                        .collect(),
                },
                Dataset {
                    label: "P95".to_string(),
                    data: snapshots
                        .iter()
                        .map(|s| DataPoint {
                            timestamp: s.timestamp,
                            value: s.query_metrics.p95_response_time.as_millis() as f64,
                        })
                        .collect(),
                },
            ],
        }
    }
    
    pub async fn export_metrics(&self, format: ExportFormat) -> Result<String> {
        let snapshot = self.live_metrics.read().await;
        
        match format {
            ExportFormat::Json => Ok(serde_json::to_string_pretty(&*snapshot)?),
            ExportFormat::Prometheus => self.export_prometheus(&snapshot).await,
            ExportFormat::InfluxDB => self.export_influxdb(&snapshot).await,
        }
    }
    
    async fn export_prometheus(&self, snapshot: &PerformanceSnapshot) -> Result<String> {
        let mut output = String::new();
        
        // Query metrics
        output.push_str(&format!(
            "llmkg_queries_total {}\n",
            snapshot.query_metrics.total_queries
        ));
        output.push_str(&format!(
            "llmkg_query_success_rate {:.2}\n",
            snapshot.query_metrics.success_rate
        ));
        output.push_str(&format!(
            "llmkg_query_response_time_seconds {:.3}\n",
            snapshot.query_metrics.avg_response_time.as_secs_f64()
        ));
        output.push_str(&format!(
            "llmkg_queries_per_second {:.2}\n",
            snapshot.query_metrics.queries_per_second
        ));
        
        // System metrics
        output.push_str(&format!(
            "llmkg_cpu_usage_percent {:.2}\n",
            snapshot.system_metrics.cpu_usage
        ));
        output.push_str(&format!(
            "llmkg_memory_usage_percent {:.2}\n",
            snapshot.system_metrics.memory_usage * 100.0
        ));
        
        // Health score
        output.push_str(&format!(
            "llmkg_health_score {:.2}\n",
            snapshot.health_score
        ));
        
        Ok(output)
    }
}
```

## File Locations

- `src/monitoring/performance_monitor.rs` - Main monitoring system
- `src/monitoring/query_metrics.rs` - Query performance tracking
- `src/monitoring/system_metrics.rs` - System resource monitoring
- `src/monitoring/alerting.rs` - Alerting and anomaly detection
- `src/monitoring/dashboard.rs` - Dashboard and visualization
- `tests/monitoring/monitoring_tests.rs` - Test implementation

## Success Criteria

- [ ] Real-time metrics collection with < 100ms overhead
- [ ] Accurate performance trend tracking
- [ ] Alerting triggers correctly for threshold breaches
- [ ] Anomaly detection identifies performance regressions
- [ ] Dashboard provides actionable insights
- [ ] Metrics export supports multiple formats
- [ ] All tests pass including stress scenarios

## Test Requirements

```rust
#[tokio::test]
async fn test_performance_metrics_collection() {
    let mut monitor = PerformanceMonitor::new(PerformanceMonitorConfig::default()).await.unwrap();
    monitor.start_monitoring().await.unwrap();
    
    // Simulate query executions
    for i in 0..10 {
        let duration = Duration::from_millis(20 + (i * 5));
        monitor.record_query_execution(duration, true).await;
    }
    
    let snapshot = monitor.get_current_performance().await;
    
    assert_eq!(snapshot.query_metrics.total_queries, 10);
    assert!(snapshot.query_metrics.success_rate > 0.9);
    assert!(snapshot.query_metrics.avg_response_time > Duration::ZERO);
}

#[tokio::test]
async fn test_alert_triggering() {
    let alert_manager = AlertManager::new();
    
    // Create snapshot with high response time
    let snapshot = PerformanceSnapshot {
        query_metrics: QueryPerformanceMetrics {
            avg_response_time: Duration::from_millis(150), // Above 100ms threshold
            error_rate: 0.02,
            ..Default::default()
        },
        ..Default::default()
    };
    
    let alerts = alert_manager.check_alerts(&snapshot).await;
    assert!(!alerts.is_empty());
    
    // Should trigger high response time alert
    assert!(alerts.iter().any(|a| matches!(
        alert_manager.get_rule(a.rule_id).condition,
        AlertCondition::ResponseTimeHigh
    )));
}

#[tokio::test]
async fn test_anomaly_detection() {
    let anomaly_detector = AnomalyDetector::new();
    
    // Build baseline with normal response times (20-30ms)
    for i in 0..50 {
        let baseline_snapshot = create_test_snapshot(Duration::from_millis(20 + (i % 10)));
        anomaly_detector.add_baseline_metric(baseline_snapshot).await;
    }
    
    // Test with anomalous response time (200ms)
    let anomalous_snapshot = create_test_snapshot(Duration::from_millis(200));
    let anomaly = anomaly_detector.detect_anomaly(&anomalous_snapshot).await;
    
    assert!(anomaly.is_some());
    assert!(anomaly.unwrap().score > 2.0); // Should be > 2 standard deviations
}

#[tokio::test]
async fn test_dashboard_data_generation() {
    let dashboard = PerformanceDashboard::new().await;
    
    // Add some historical data
    for i in 0..60 {
        let snapshot = create_test_snapshot(Duration::from_millis(25 + (i % 20)));
        dashboard.add_snapshot(snapshot).await;
    }
    
    let dashboard_data = dashboard.get_dashboard_data().await;
    
    assert!(!dashboard_data.historical_charts.is_empty());
    assert!(dashboard_data.current_metrics.health_score > 0.0);
}

#[tokio::test]
async fn test_prometheus_export() {
    let dashboard = PerformanceDashboard::new().await;
    let snapshot = create_test_snapshot(Duration::from_millis(25));
    
    let prometheus_data = dashboard.export_metrics(ExportFormat::Prometheus).await.unwrap();
    
    assert!(prometheus_data.contains("llmkg_queries_total"));
    assert!(prometheus_data.contains("llmkg_query_response_time_seconds"));
    assert!(prometheus_data.contains("llmkg_health_score"));
}

#[tokio::test]
async fn test_concurrent_metrics_collection() {
    let monitor = Arc::new(PerformanceMonitor::new(PerformanceMonitorConfig::default()).await.unwrap());
    
    // Simulate concurrent query recording
    let handles: Vec<_> = (0..100).map(|i| {
        let m = monitor.clone();
        tokio::spawn(async move {
            let duration = Duration::from_millis(10 + (i % 50));
            m.record_query_execution(duration, true).await;
        })
    }).collect();
    
    futures::try_join_all(handles).await.unwrap();
    
    let snapshot = monitor.get_current_performance().await;
    assert_eq!(snapshot.query_metrics.total_queries, 100);
}

fn create_test_snapshot(response_time: Duration) -> PerformanceSnapshot {
    PerformanceSnapshot {
        timestamp: Instant::now(),
        query_metrics: QueryPerformanceMetrics {
            avg_response_time: response_time,
            success_rate: 0.95,
            error_rate: 0.05,
            queries_per_second: 50.0,
            total_queries: 1000,
            ..Default::default()
        },
        system_metrics: SystemPerformanceMetrics {
            cpu_usage: 45.0,
            memory_usage: 0.6,
            ..Default::default()
        },
        cache_metrics: CachePerformanceMetrics {
            hit_rate: 0.85,
            ..Default::default()
        },
        health_score: 92.5,
    }
}
```

## Quality Gates

- [ ] Metrics collection adds < 100ms overhead to query processing
- [ ] Memory usage for metrics storage remains bounded
- [ ] Alert false positive rate < 5%
- [ ] Dashboard updates in real-time without lag
- [ ] Anomaly detection accuracy > 90%
- [ ] System remains stable under monitoring load
- [ ] Export formats are valid and complete

## Next Task

Upon completion, proceed to **40_integration_tests.md** (already exists)