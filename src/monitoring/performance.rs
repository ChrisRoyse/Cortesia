use crate::error::{GraphError, Result};
use crate::monitoring::observability::{MetricsCollector, TraceExporter};
use crate::monitoring::alerts::AlertManager;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use uuid;

/// Performance monitoring system
pub struct PerformanceMonitor {
    metrics_collector: Arc<MetricsCollector>,
    trace_exporter: Arc<TraceExporter>,
    alert_manager: Arc<AlertManager>,
    active_operations: Arc<RwLock<HashMap<String, ActiveOperation>>>,
    performance_history: Arc<RwLock<Vec<PerformanceSnapshot>>>,
    config: MonitoringConfig,
}

impl std::fmt::Debug for PerformanceMonitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PerformanceMonitor")
            .field("metrics_collector", &"MetricsCollector")
            .field("trace_exporter", &"TraceExporter")
            .field("alert_manager", &"AlertManager")
            .field("active_operations", &"RwLock<HashMap>")
            .field("performance_history", &"RwLock<Vec>")
            .field("config", &self.config)
            .finish()
    }
}

impl PerformanceMonitor {
    pub fn new(
        metrics_collector: Arc<MetricsCollector>,
        trace_exporter: Arc<TraceExporter>,
        alert_manager: Arc<AlertManager>,
        config: MonitoringConfig,
    ) -> Self {
        Self {
            metrics_collector,
            trace_exporter,
            alert_manager,
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            performance_history: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }

    /// Create a new performance monitor with default configuration
    pub async fn new_with_defaults() -> Result<Self> {
        // Create default components
        let metrics_collector = Arc::new(MetricsCollector::new_async().await?);
        let trace_exporter = Arc::new(TraceExporter::new_async().await?);
        let alert_manager = Arc::new(AlertManager::new_async().await?);
        let config = MonitoringConfig::default();

        Ok(Self {
            metrics_collector,
            trace_exporter,
            alert_manager,
            active_operations: Arc::new(RwLock::new(HashMap::new())),
            performance_history: Arc::new(RwLock::new(Vec::new())),
            config,
        })
    }

    pub async fn track_operation<T, F>(&self, operation: Operation, func: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>>,
    {
        let operation_id = self.start_operation_tracking(operation.clone()).await?;
        let start_time = Instant::now();
        
        let result = func.await;
        
        let duration = start_time.elapsed();
        self.end_operation_tracking(&operation_id, &operation, duration, result.is_ok()).await?;
        
        result
    }

    pub async fn start_operation_tracking(&self, operation: Operation) -> Result<String> {
        let operation_id = format!("op_{}", uuid::Uuid::new_v4());
        
        let active_op = ActiveOperation {
            id: operation_id.clone(),
            operation: operation.clone(),
            start_time: Instant::now(),
            start_timestamp: Utc::now(),
        };
        
        {
            let mut active_ops = self.active_operations.write().await;
            active_ops.insert(operation_id.clone(), active_op);
        }
        
        // Record operation start
        self.metrics_collector.record_operation_start(&operation).await?;
        
        // Create trace span
        self.trace_exporter.start_span(&operation_id, &operation.name).await?;
        
        Ok(operation_id)
    }

    pub async fn end_operation_tracking(
        &self,
        operation_id: &str,
        operation: &Operation,
        duration: Duration,
        success: bool,
    ) -> Result<()> {
        // Remove from active operations
        let active_op = {
            let mut active_ops = self.active_operations.write().await;
            active_ops.remove(operation_id)
        };
        
        if let Some(active_op) = active_op {
            // Create operation metrics
            let metrics = OperationMetrics {
                operation_id: operation_id.to_string(),
                operation_name: operation.name.clone(),
                operation_type: operation.operation_type.clone(),
                duration,
                success,
                start_time: active_op.start_timestamp,
                end_time: Utc::now(),
                resource_usage: self.collect_resource_usage().await?,
                custom_metrics: operation.custom_metrics.clone(),
            };
            
            // Record metrics
            self.metrics_collector.record_operation_metrics(&metrics).await?;
            
            // End trace span
            self.trace_exporter.end_span(operation_id, success).await?;
            
            // Check for alerts
            self.check_performance_alerts(&metrics).await?;
            
            // Update performance history
            self.update_performance_history(&metrics).await?;
        }
        
        Ok(())
    }

    pub async fn get_current_metrics(&self) -> Result<PerformanceMetrics> {
        let active_ops = self.active_operations.read().await;
        let active_operation_count = active_ops.len();
        
        let resource_metrics = self.collect_resource_usage().await?;
        let latency_metrics = self.calculate_latency_metrics().await?;
        let throughput_metrics = self.calculate_throughput_metrics().await?;
        
        Ok(PerformanceMetrics {
            timestamp: Utc::now(),
            active_operations: active_operation_count,
            resource_usage: resource_metrics,
            latency: latency_metrics,
            throughput: throughput_metrics,
            error_rate: self.calculate_error_rate().await?,
            system_health: self.assess_system_health().await?,
        })
    }

    pub async fn get_operation_history(&self, operation_type: Option<String>) -> Result<Vec<OperationMetrics>> {
        let history = self.performance_history.read().await;
        
        let filtered_history: Vec<OperationMetrics> = history.iter()
            .filter_map(|snapshot| {
                snapshot.operations.iter().find(|op| {
                    operation_type.as_ref().is_none_or(|t| &op.operation_type == t)
                }).cloned()
            })
            .collect();
        
        Ok(filtered_history)
    }

    /// Get current metrics as a HashMap
    pub async fn get_metrics(&self) -> Result<HashMap<String, f64>> {
        let mut metrics = HashMap::new();
        
        // Get basic counts
        let history = self.performance_history.read().await;
        let total_operations = history.len();
        let successful_operations = history.iter()
            .flat_map(|s| &s.operations)
            .filter(|op| op.success)
            .count();
        
        metrics.insert("total_queries".to_string(), total_operations as f64);
        metrics.insert("success_rate".to_string(), 
            if total_operations > 0 { 
                successful_operations as f64 / total_operations as f64 
            } else { 
                0.0 
            });
        
        // Get average response time
        let avg_response_time = if total_operations > 0 {
            history.iter()
                .flat_map(|s| &s.operations)
                .map(|op| op.duration.as_millis() as f64)
                .sum::<f64>() / total_operations as f64
        } else {
            0.0
        };
        metrics.insert("avg_response_time".to_string(), avg_response_time);
        
        // Resource metrics
        let resource_metrics = self.collect_resource_usage().await?;
        metrics.insert("memory_usage_mb".to_string(), resource_metrics.memory_usage_mb);
        metrics.insert("cpu_usage_percent".to_string(), resource_metrics.cpu_usage_percent);
        
        // Default values for other metrics
        metrics.insert("cache_hit_rate".to_string(), 0.0);
        metrics.insert("active_entities".to_string(), 0.0);
        
        Ok(metrics)
    }
    
    pub async fn get_performance_summary(&self, duration: Duration) -> Result<PerformanceSummary> {
        let cutoff_time = Utc::now() - chrono::Duration::from_std(duration).unwrap();
        let history = self.performance_history.read().await;
        
        let recent_snapshots: Vec<_> = history.iter()
            .filter(|snapshot| snapshot.timestamp > cutoff_time)
            .collect();
        
        if recent_snapshots.is_empty() {
            return Ok(PerformanceSummary::default());
        }
        
        let total_operations: usize = recent_snapshots.iter()
            .map(|s| s.operations.len())
            .sum();
        
        let successful_operations: usize = recent_snapshots.iter()
            .flat_map(|s| &s.operations)
            .filter(|op| op.success)
            .count();
        
        let avg_latency = recent_snapshots.iter()
            .flat_map(|s| &s.operations)
            .map(|op| op.duration.as_millis() as f64)
            .sum::<f64>() / total_operations as f64;
        
        let max_latency = recent_snapshots.iter()
            .flat_map(|s| &s.operations)
            .map(|op| op.duration)
            .max()
            .unwrap_or(Duration::from_millis(0));
        
        let avg_memory_usage = recent_snapshots.iter()
            .map(|s| s.resource_metrics.memory_usage_mb)
            .sum::<f64>() / recent_snapshots.len() as f64;
        
        let avg_cpu_usage = recent_snapshots.iter()
            .map(|s| s.resource_metrics.cpu_usage_percent)
            .sum::<f64>() / recent_snapshots.len() as f64;
        
        Ok(PerformanceSummary {
            time_period: duration,
            total_operations,
            successful_operations,
            success_rate: successful_operations as f64 / total_operations as f64,
            avg_latency_ms: avg_latency,
            max_latency_ms: max_latency.as_millis() as f64,
            avg_memory_usage_mb: avg_memory_usage,
            avg_cpu_usage_percent: avg_cpu_usage,
            alerts_triggered: self.alert_manager.get_recent_alerts_count(duration).await?,
        })
    }

    async fn collect_resource_usage(&self) -> Result<ResourceMetrics> {
        use sysinfo::{System, Pid};
        
        let mut sys = System::new_all();
        sys.refresh_all();
        
        // Get current process info
        let pid = Pid::from(std::process::id() as usize);
        
        let process = sys.process(pid).ok_or_else(|| {
            GraphError::InvalidInput("Failed to find current process".into())
        })?;
        
        Ok(ResourceMetrics {
            memory_usage_mb: process.memory() as f64 / 1024.0 / 1024.0,
            cpu_usage_percent: process.cpu_usage() as f64,
            disk_usage_mb: process.disk_usage().total_written_bytes as f64 / 1024.0 / 1024.0,
            network_io_bytes: 0, // sysinfo doesn't provide network I/O per process easily
            open_file_handles: 42, // Placeholder - sysinfo doesn't expose this easily
            thread_count: 1, // Placeholder - sysinfo doesn't expose thread count easily
        })
    }

    async fn calculate_latency_metrics(&self) -> Result<LatencyMetrics> {
        let history = self.performance_history.read().await;
        let recent_operations: Vec<_> = history.iter()
            .flat_map(|s| &s.operations)
            .take(1000) // Last 1000 operations
            .collect();
        
        if recent_operations.is_empty() {
            return Ok(LatencyMetrics::default());
        }
        
        let mut latencies: Vec<u64> = recent_operations.iter()
            .map(|op| op.duration.as_millis() as u64)
            .collect();
        
        latencies.sort_unstable();
        
        let len = latencies.len();
        let p50 = latencies[len / 2];
        let p90 = latencies[(len as f64 * 0.9) as usize];
        let p95 = latencies[(len as f64 * 0.95) as usize];
        let p99 = latencies[(len as f64 * 0.99) as usize];
        
        let avg = latencies.iter().sum::<u64>() as f64 / len as f64;
        let max = latencies.iter().max().copied().unwrap_or(0);
        let min = latencies.iter().min().copied().unwrap_or(0);
        
        Ok(LatencyMetrics {
            avg_ms: avg,
            p50_ms: p50,
            p90_ms: p90,
            p95_ms: p95,
            p99_ms: p99,
            max_ms: max,
            min_ms: min,
        })
    }

    async fn calculate_throughput_metrics(&self) -> Result<ThroughputMetrics> {
        let history = self.performance_history.read().await;
        let recent_snapshots: Vec<_> = history.iter()
            .filter(|s| s.timestamp > Utc::now() - chrono::Duration::minutes(5))
            .collect();
        
        if recent_snapshots.is_empty() {
            return Ok(ThroughputMetrics::default());
        }
        
        let total_operations: usize = recent_snapshots.iter()
            .map(|s| s.operations.len())
            .sum();
        
        let time_span_minutes = 5.0; // 5 minutes
        let ops_per_minute = total_operations as f64 / time_span_minutes;
        let ops_per_second = ops_per_minute / 60.0;
        
        Ok(ThroughputMetrics {
            operations_per_second: ops_per_second,
            operations_per_minute: ops_per_minute,
            requests_per_second: ops_per_second, // Assuming 1:1 for now
            bytes_per_second: ops_per_second * 1024.0, // Mock calculation
        })
    }

    async fn calculate_error_rate(&self) -> Result<f64> {
        let history = self.performance_history.read().await;
        let recent_operations: Vec<_> = history.iter()
            .flat_map(|s| &s.operations)
            .take(1000)
            .collect();
        
        if recent_operations.is_empty() {
            return Ok(0.0);
        }
        
        let failed_operations = recent_operations.iter()
            .filter(|op| !op.success)
            .count();
        
        Ok(failed_operations as f64 / recent_operations.len() as f64)
    }

    async fn assess_system_health(&self) -> Result<SystemHealth> {
        let metrics = self.collect_resource_usage().await?;
        let error_rate = self.calculate_error_rate().await?;
        
        let health_score = if error_rate > 0.1 {
            SystemHealth::Critical
        } else if metrics.memory_usage_mb > 1024.0 || metrics.cpu_usage_percent > 80.0 {
            SystemHealth::Warning
        } else if error_rate > 0.05 {
            SystemHealth::Degraded
        } else {
            SystemHealth::Healthy
        };
        
        Ok(health_score)
    }

    async fn check_performance_alerts(&self, metrics: &OperationMetrics) -> Result<()> {
        // Check latency alerts
        if metrics.duration > self.config.max_operation_duration {
            self.alert_manager.trigger_alert(
                "High Latency".to_string(),
                format!("Operation {} took {:.2}ms", metrics.operation_name, metrics.duration.as_millis()),
                crate::monitoring::alerts::AlertSeverity::Warning,
            ).await?;
        }
        
        // Check resource usage alerts
        if metrics.resource_usage.memory_usage_mb > self.config.max_memory_usage_mb {
            self.alert_manager.trigger_alert(
                "High Memory Usage".to_string(),
                format!("Memory usage: {:.2}MB", metrics.resource_usage.memory_usage_mb),
                crate::monitoring::alerts::AlertSeverity::Critical,
            ).await?;
        }
        
        Ok(())
    }

    async fn update_performance_history(&self, metrics: &OperationMetrics) -> Result<()> {
        let snapshot = PerformanceSnapshot {
            timestamp: Utc::now(),
            operations: vec![metrics.clone()],
            resource_metrics: metrics.resource_usage.clone(),
        };
        
        {
            let mut history = self.performance_history.write().await;
            history.push(snapshot);
            
            // Keep only recent history
            let cutoff_time = Utc::now() - chrono::Duration::hours(24);
            history.retain(|s| s.timestamp > cutoff_time);
        }
        
        Ok(())
    }
}

/// Configuration for performance monitoring
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub max_operation_duration: Duration,
    pub max_memory_usage_mb: f64,
    pub max_cpu_usage_percent: f64,
    pub alert_thresholds: HashMap<String, f64>,
    pub history_retention_hours: i64,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            max_operation_duration: Duration::from_secs(30),
            max_memory_usage_mb: 1024.0,
            max_cpu_usage_percent: 80.0,
            alert_thresholds: HashMap::new(),
            history_retention_hours: 24,
        }
    }
}

/// Operation being tracked
#[derive(Debug, Clone)]
pub struct Operation {
    pub name: String,
    pub operation_type: String,
    pub custom_metrics: HashMap<String, f64>,
}

/// Active operation tracking
#[derive(Debug, Clone)]
struct ActiveOperation {
    pub id: String,
    pub operation: Operation,
    pub start_time: Instant,
    pub start_timestamp: DateTime<Utc>,
}

/// Performance metrics snapshot
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub timestamp: DateTime<Utc>,
    pub active_operations: usize,
    pub resource_usage: ResourceMetrics,
    pub latency: LatencyMetrics,
    pub throughput: ThroughputMetrics,
    pub error_rate: f64,
    pub system_health: SystemHealth,
}

/// Individual operation metrics
#[derive(Debug, Clone)]
pub struct OperationMetrics {
    pub operation_id: String,
    pub operation_name: String,
    pub operation_type: String,
    pub duration: Duration,
    pub success: bool,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub resource_usage: ResourceMetrics,
    pub custom_metrics: HashMap<String, f64>,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub disk_usage_mb: f64,
    pub network_io_bytes: u64,
    pub open_file_handles: usize,
    pub thread_count: usize,
}

/// Latency metrics
#[derive(Debug, Clone, Default)]
pub struct LatencyMetrics {
    pub avg_ms: f64,
    pub p50_ms: u64,
    pub p90_ms: u64,
    pub p95_ms: u64,
    pub p99_ms: u64,
    pub max_ms: u64,
    pub min_ms: u64,
}

/// Throughput metrics
#[derive(Debug, Clone, Default)]
pub struct ThroughputMetrics {
    pub operations_per_second: f64,
    pub operations_per_minute: f64,
    pub requests_per_second: f64,
    pub bytes_per_second: f64,
}

/// System health status
#[derive(Debug, Clone)]
pub enum SystemHealth {
    Healthy,
    Degraded,
    Warning,
    Critical,
}

/// Performance history snapshot
#[derive(Debug, Clone)]
struct PerformanceSnapshot {
    pub timestamp: DateTime<Utc>,
    pub operations: Vec<OperationMetrics>,
    pub resource_metrics: ResourceMetrics,
}

/// Performance summary over a time period
#[derive(Debug, Clone, Default)]
pub struct PerformanceSummary {
    pub time_period: Duration,
    pub total_operations: usize,
    pub successful_operations: usize,
    pub success_rate: f64,
    pub avg_latency_ms: f64,
    pub max_latency_ms: f64,
    pub avg_memory_usage_mb: f64,
    pub avg_cpu_usage_percent: f64,
    pub alerts_triggered: usize,
}


