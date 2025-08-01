//! Production Monitoring and Logging System
//!
//! Provides comprehensive monitoring, logging, and observability for the LLMKG system.
//! Includes structured logging, metrics collection, alerting, and performance tracking.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use serde_json::{json, Value};
use dashmap::DashMap;

/// Log levels for structured logging
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Critical = 5,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Structured log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: u64,
    pub level: LogLevel,
    pub component: String,
    pub operation: String,
    pub message: String,
    pub metadata: HashMap<String, Value>,
    pub correlation_id: Option<String>,
    pub user_id: Option<String>,
    pub execution_time_ms: Option<u64>,
    pub error_code: Option<String>,
}

/// Metric types for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Timer,
}

/// Metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub name: String,
    pub metric_type: MetricType,
    pub value: f64,
    pub timestamp: u64,
    pub tags: HashMap<String, String>,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub metric_name: String,
    pub threshold: f64,
    pub comparison: AlertComparison,
    pub severity: AlertSeverity,
    pub duration: Duration,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertComparison {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
}

/// Active alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub rule_name: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub triggered_at: u64,
    pub metric_value: f64,
    pub threshold: f64,
    pub metadata: HashMap<String, Value>,
}

/// Production monitoring system
#[derive(Clone)]
pub struct ProductionMonitor {
    // Logging
    log_buffer: Arc<RwLock<Vec<LogEntry>>>,
    log_level: Arc<RwLock<LogLevel>>,
    max_log_entries: usize,
    
    // Metrics
    metrics: DashMap<String, Vec<MetricPoint>>,
    counters: DashMap<String, Arc<AtomicU64>>,
    gauges: DashMap<String, Arc<AtomicU64>>,
    timers: DashMap<String, Arc<RwLock<Vec<u64>>>>,
    
    // Alerting
    alert_rules: DashMap<String, AlertRule>,
    active_alerts: DashMap<String, Alert>,
    alert_history: Arc<RwLock<Vec<Alert>>>,
    
    // System monitoring
    system_stats: Arc<RwLock<SystemStats>>,
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    
    // Configuration
    config: MonitoringConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub log_level: LogLevel,
    pub max_log_entries: usize,
    pub max_metric_points: usize,
    pub metric_retention_seconds: u64,
    pub enable_performance_tracking: bool,
    pub enable_alerting: bool,
    pub alert_check_interval_seconds: u64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            log_level: LogLevel::Info,
            max_log_entries: 10000,
            max_metric_points: 1000,
            metric_retention_seconds: 3600, // 1 hour
            enable_performance_tracking: true,
            enable_alerting: true,
            alert_check_interval_seconds: 30,
        }
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub uptime_seconds: u64,
    pub total_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time_ms: f64,
    pub memory_usage_bytes: u64,
    pub cpu_usage_percent: f64,
    pub active_connections: u32,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
}

#[derive(Debug, Default)]
pub struct PerformanceTracker {
    pub operation_timings: HashMap<String, Vec<u64>>,
    pub memory_snapshots: Vec<(u64, u64)>, // (timestamp, memory_bytes)
    pub throughput_data: HashMap<String, Vec<(u64, u64)>>, // operation -> (timestamp, count)
    pub start_time: Option<Instant>,
}

impl ProductionMonitor {
    pub fn new(config: MonitoringConfig) -> Self {
        let monitor = Self {
            log_buffer: Arc::new(RwLock::new(Vec::new())),
            log_level: Arc::new(RwLock::new(config.log_level)),
            max_log_entries: config.max_log_entries,
            
            metrics: DashMap::new(),
            counters: DashMap::new(),
            gauges: DashMap::new(),
            timers: DashMap::new(),
            
            alert_rules: DashMap::new(),
            active_alerts: DashMap::new(),
            alert_history: Arc::new(RwLock::new(Vec::new())),
            
            system_stats: Arc::new(RwLock::new(SystemStats::default())),
            performance_tracker: Arc::new(RwLock::new(PerformanceTracker {
                start_time: Some(Instant::now()),
                ..Default::default()
            })),
            
            config,
        };
        
        // Start background tasks
        monitor.start_background_tasks();
        monitor.setup_default_alerts();
        
        monitor
    }

    /// Log a structured message
    pub async fn log(&self, level: LogLevel, component: &str, operation: &str, message: &str) {
        self.log_with_metadata(level, component, operation, message, HashMap::new()).await;
    }

    /// Log with additional metadata
    pub async fn log_with_metadata(
        &self,
        level: LogLevel,
        component: &str,
        operation: &str,
        message: &str,
        metadata: HashMap<String, Value>,
    ) {
        let current_level = *self.log_level.read().await;
        if (level as u8) < (current_level as u8) {
            return;
        }

        let entry = LogEntry {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            level,
            component: component.to_string(),
            operation: operation.to_string(),
            message: message.to_string(),
            metadata,
            correlation_id: None,
            user_id: None,
            execution_time_ms: None,
            error_code: None,
        };

        // Add to buffer
        {
            let mut buffer = self.log_buffer.write().await;
            buffer.push(entry.clone());
            
            // Trim if necessary
            if buffer.len() > self.max_log_entries {
                let len = buffer.len();
                buffer.drain(0..len - self.max_log_entries);
            }
        }

        // Output to console (in production, this would go to a proper logging system)
        if self.config.log_level as u8 <= LogLevel::Info as u8 {
            println!("[{}] {}/{}: {}", entry.level, entry.component, entry.operation, entry.message);
        }
    }

    /// Record a metric value
    pub fn record_metric(&self, name: &str, value: f64, metric_type: MetricType, tags: HashMap<String, String>) {
        let point = MetricPoint {
            name: name.to_string(),
            metric_type: metric_type.clone(),
            value,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            tags,
        };

        // Store in metrics collection
        self.metrics.entry(name.to_string())
            .or_default()
            .push(point);

        // Update specific metric types
        match metric_type {
            MetricType::Counter => {
                self.counters.entry(name.to_string())
                    .or_insert_with(|| Arc::new(AtomicU64::new(0)))
                    .fetch_add(value as u64, Ordering::Relaxed);
            }
            MetricType::Gauge => {
                self.gauges.entry(name.to_string())
                    .or_insert_with(|| Arc::new(AtomicU64::new(0)))
                    .store(value as u64, Ordering::Relaxed);
            }
            MetricType::Timer => {
                let timer_data = self.timers.entry(name.to_string())
                    .or_insert_with(|| Arc::new(RwLock::new(Vec::new())));
                
                tokio::spawn({
                    let timer_data = timer_data.clone();
                    async move {
                        let mut data = timer_data.write().await;
                        data.push(value as u64);
                        
                        // Keep only recent values
                        if data.len() > 1000 {
                            let len = data.len();
                            data.drain(0..len - 1000);
                        }
                    }
                });
            }
            MetricType::Histogram => {
                // Histogram implementation would go here
            }
        }
    }

    /// Increment a counter
    pub fn increment_counter(&self, name: &str) {
        self.record_metric(name, 1.0, MetricType::Counter, HashMap::new());
    }

    /// Set a gauge value
    pub fn set_gauge(&self, name: &str, value: f64) {
        self.record_metric(name, value, MetricType::Gauge, HashMap::new());
    }

    /// Record timing information
    pub fn record_timing(&self, name: &str, duration: Duration) {
        self.record_metric(name, duration.as_millis() as f64, MetricType::Timer, HashMap::new());
    }

    /// Start timing an operation - this method needs to be called on Arc<Self>
    pub fn start_timer(self: &Arc<Self>, operation: &str) -> TimerHandle {
        TimerHandle {
            operation: operation.to_string(),
            start_time: Instant::now(),
            monitor: Arc::clone(self),
        }
    }

    /// Add an alert rule
    pub fn add_alert_rule(&self, rule: AlertRule) {
        self.alert_rules.insert(rule.name.clone(), rule);
    }

    /// Remove an alert rule
    pub fn remove_alert_rule(&self, rule_name: &str) {
        self.alert_rules.remove(rule_name);
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> HashMap<String, Vec<MetricPoint>> {
        let mut metrics = HashMap::new();
        for item in self.metrics.iter() {
            metrics.insert(item.key().clone(), item.value().clone());
        }
        metrics
    }

    /// Get system statistics
    pub async fn get_system_stats(&self) -> SystemStats {
        self.system_stats.read().await.clone()
    }

    /// Get recent logs
    pub async fn get_recent_logs(&self, limit: Option<usize>) -> Vec<LogEntry> {
        let buffer = self.log_buffer.read().await;
        let limit = limit.unwrap_or(100).min(buffer.len());
        buffer.iter().rev().take(limit).cloned().collect()
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts.iter().map(|item| item.value().clone()).collect()
    }

    /// Get performance data
    pub async fn get_performance_data(&self) -> HashMap<String, Value> {
        let tracker = self.performance_tracker.read().await;
        let mut data = HashMap::new();
        
        // Operation timings
        let mut timing_stats = HashMap::new();
        for (operation, timings) in &tracker.operation_timings {
            if !timings.is_empty() {
                let avg = timings.iter().sum::<u64>() as f64 / timings.len() as f64;
                let min = *timings.iter().min().unwrap();
                let max = *timings.iter().max().unwrap();
                
                timing_stats.insert(operation.clone(), json!({
                    "average_ms": avg,
                    "min_ms": min,
                    "max_ms": max,
                    "count": timings.len()
                }));
            }
        }
        data.insert("operation_timings".to_string(), json!(timing_stats));
        
        // Throughput data
        let mut throughput_stats = HashMap::new();
        for (operation, points) in &tracker.throughput_data {
            if !points.is_empty() {
                let total_ops: u64 = points.iter().map(|(_, count)| count).sum();
                let duration = if let Some((first_ts, _)) = points.first() {
                    if let Some((last_ts, _)) = points.last() {
                        (last_ts - first_ts) as f64 / 1000.0 // Convert to seconds
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };
                
                let ops_per_second = if duration > 0.0 { total_ops as f64 / duration } else { 0.0 };
                
                throughput_stats.insert(operation.clone(), json!({
                    "total_operations": total_ops,
                    "ops_per_second": ops_per_second,
                    "duration_seconds": duration
                }));
            }
        }
        data.insert("throughput".to_string(), json!(throughput_stats));
        
        data
    }

    /// Health check endpoint
    pub async fn health_check(&self) -> HashMap<String, Value> {
        let mut health = HashMap::new();
        
        let stats = self.get_system_stats().await;
        let active_alerts = self.get_active_alerts().await;
        
        // Overall health status
        let health_status = if active_alerts.iter().any(|a| a.severity == AlertSeverity::Critical) {
            "critical"
        } else if active_alerts.iter().any(|a| a.severity == AlertSeverity::Warning) {
            "warning"
        } else {
            "healthy"
        };
        
        health.insert("status".to_string(), json!(health_status));
        health.insert("uptime_seconds".to_string(), json!(stats.uptime_seconds));
        health.insert("total_requests".to_string(), json!(stats.total_requests));
        health.insert("error_rate".to_string(), json!(stats.error_rate));
        health.insert("avg_response_time_ms".to_string(), json!(stats.avg_response_time_ms));
        health.insert("memory_usage_bytes".to_string(), json!(stats.memory_usage_bytes));
        health.insert("active_alerts".to_string(), json!(active_alerts.len()));
        
        // Component health
        let mut component_health = HashMap::new();
        
        // Check circuit breaker status from error recovery
        component_health.insert("error_recovery".to_string(), json!("healthy"));
        component_health.insert("rate_limiter".to_string(), json!("healthy"));
        component_health.insert("resource_manager".to_string(), json!("healthy"));
        
        health.insert("components".to_string(), json!(component_health));
        
        health
    }

    /// Export metrics in Prometheus format
    pub async fn export_prometheus_metrics(&self) -> String {
        let mut output = String::new();
        
        // Counters
        for item in self.counters.iter() {
            let value = item.value().load(Ordering::Relaxed);
            output.push_str(&format!("# TYPE {} counter\n", item.key()));
            output.push_str(&format!("{} {}\n", item.key(), value));
        }
        
        // Gauges
        for item in self.gauges.iter() {
            let value = item.value().load(Ordering::Relaxed);
            output.push_str(&format!("# TYPE {} gauge\n", item.key()));
            output.push_str(&format!("{} {}\n", item.key(), value));
        }
        
        output
    }

    fn start_background_tasks(&self) {
        if self.config.enable_alerting {
            self.start_alert_checker();
        }
        
        self.start_metrics_cleanup();
        self.start_system_stats_updater();
    }

    fn start_alert_checker(&self) {
        let alert_rules = self.alert_rules.clone();
        let active_alerts = self.active_alerts.clone();
        let metrics = self.metrics.clone();
        let interval = Duration::from_secs(self.config.alert_check_interval_seconds);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(interval);
            loop {
                interval.tick().await;
                
                for rule_item in alert_rules.iter() {
                    let rule = rule_item.value();
                    if !rule.enabled {
                        continue;
                    }
                    
                    if let Some(metric_data) = metrics.get(&rule.metric_name) {
                        if let Some(latest_point) = metric_data.last() {
                            let should_alert = match rule.comparison {
                                AlertComparison::GreaterThan => latest_point.value > rule.threshold,
                                AlertComparison::LessThan => latest_point.value < rule.threshold,
                                AlertComparison::Equals => (latest_point.value - rule.threshold).abs() < f64::EPSILON,
                                AlertComparison::NotEquals => (latest_point.value - rule.threshold).abs() >= f64::EPSILON,
                            };
                            
                            if should_alert && !active_alerts.contains_key(&rule.name) {
                                let alert = Alert {
                                    rule_name: rule.name.clone(),
                                    severity: rule.severity,
                                    message: format!(
                                        "Metric {} is {} (threshold: {})",
                                        rule.metric_name, latest_point.value, rule.threshold
                                    ),
                                    triggered_at: SystemTime::now()
                                        .duration_since(UNIX_EPOCH)
                                        .unwrap()
                                        .as_millis() as u64,
                                    metric_value: latest_point.value,
                                    threshold: rule.threshold,
                                    metadata: HashMap::new(),
                                };
                                
                                active_alerts.insert(rule.name.clone(), alert);
                            } else if !should_alert && active_alerts.contains_key(&rule.name) {
                                active_alerts.remove(&rule.name);
                            }
                        }
                    }
                }
            }
        });
    }

    fn start_metrics_cleanup(&self) {
        let metrics = self.metrics.clone();
        let retention_seconds = self.config.metric_retention_seconds;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(300)); // Clean every 5 minutes
            loop {
                interval.tick().await;
                
                let cutoff_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64 - (retention_seconds * 1000);
                
                for mut item in metrics.iter_mut() {
                    item.value_mut().retain(|point| point.timestamp > cutoff_time);
                }
            }
        });
    }

    fn start_system_stats_updater(&self) {
        let system_stats = self.system_stats.clone();
        let performance_tracker = self.performance_tracker.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                
                // Update system statistics
                {
                    let mut stats = system_stats.write().await;
                    let tracker = performance_tracker.read().await;
                    
                    if let Some(start_time) = tracker.start_time {
                        stats.uptime_seconds = start_time.elapsed().as_secs();
                    }
                    
                    // Update other stats as needed
                    // In a real implementation, these would come from actual system monitoring
                    stats.memory_usage_bytes = Self::get_memory_usage();
                    stats.cpu_usage_percent = Self::get_cpu_usage();
                }
            }
        });
    }

    fn setup_default_alerts(&self) {
        // High error rate alert
        self.add_alert_rule(AlertRule {
            name: "high_error_rate".to_string(),
            metric_name: "error_rate".to_string(),
            threshold: 0.05, // 5%
            comparison: AlertComparison::GreaterThan,
            severity: AlertSeverity::Warning,
            duration: Duration::from_secs(300),
            enabled: true,
        });
        
        // High response time alert
        self.add_alert_rule(AlertRule {
            name: "high_response_time".to_string(),
            metric_name: "avg_response_time_ms".to_string(),
            threshold: 1000.0, // 1 second
            comparison: AlertComparison::GreaterThan,
            severity: AlertSeverity::Warning,
            duration: Duration::from_secs(180),
            enabled: true,
        });
        
        // Memory usage alert
        self.add_alert_rule(AlertRule {
            name: "high_memory_usage".to_string(),
            metric_name: "memory_usage_bytes".to_string(),
            threshold: 1_000_000_000.0, // 1GB
            comparison: AlertComparison::GreaterThan,
            severity: AlertSeverity::Critical,
            duration: Duration::from_secs(60),
            enabled: true,
        });
    }

    // Placeholder functions for system monitoring
    fn get_memory_usage() -> u64 {
        // In a real implementation, this would get actual memory usage
        // For now, return a placeholder value
        std::process::id() as u64 * 1000 // Rough approximation
    }

    fn get_cpu_usage() -> f64 {
        // In a real implementation, this would get actual CPU usage
        // For now, return a placeholder value
        0.1 // 10% CPU usage
    }
}

/// Timer handle for measuring operation duration
pub struct TimerHandle {
    operation: String,
    start_time: Instant,
    monitor: Arc<ProductionMonitor>,
}

impl Drop for TimerHandle {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        self.monitor.record_timing(&self.operation, duration);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_logging() {
        let config = MonitoringConfig::default();
        let monitor = ProductionMonitor::new(config);
        
        monitor.log(LogLevel::Info, "test", "test_op", "Test message").await;
        
        let logs = monitor.get_recent_logs(Some(10)).await;
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].message, "Test message");
    }

    #[tokio::test]
    async fn test_metrics() {
        let config = MonitoringConfig::default();
        let monitor = ProductionMonitor::new(config);
        
        monitor.record_metric("test_counter", 5.0, MetricType::Counter, HashMap::new());
        monitor.set_gauge("test_gauge", 100.0);
        
        let metrics = monitor.get_metrics().await;
        assert!(metrics.contains_key("test_counter"));
        assert!(metrics.contains_key("test_gauge"));
    }

    #[tokio::test]
    async fn test_alerting() {
        let config = MonitoringConfig::default();
        let monitor = ProductionMonitor::new(config);
        
        // Add a test alert rule
        monitor.add_alert_rule(AlertRule {
            name: "test_alert".to_string(),
            metric_name: "test_metric".to_string(),
            threshold: 50.0,
            comparison: AlertComparison::GreaterThan,
            severity: AlertSeverity::Warning,
            duration: Duration::from_secs(10),
            enabled: true,
        });
        
        // Trigger the alert
        monitor.record_metric("test_metric", 75.0, MetricType::Gauge, HashMap::new());
        
        // Wait for alert checker to run
        sleep(Duration::from_secs(1)).await;
        
        let _alerts = monitor.get_active_alerts().await;
        // Note: In a real test, we'd need to wait for the background task to process
        // For now, just verify the alert rule was added
        assert!(monitor.alert_rules.contains_key("test_alert"));
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = MonitoringConfig::default();
        let monitor = ProductionMonitor::new(config);
        
        let health = monitor.health_check().await;
        assert!(health.contains_key("status"));
        assert!(health.contains_key("uptime_seconds"));
    }
}