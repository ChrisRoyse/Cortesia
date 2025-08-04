# Task 45: Logging and Monitoring System

**Time:** 10 minutes (2 min read, 6 min implement, 2 min verify)
**Prerequisites:** Task 44 completed
**Input Files:**
- C:/code/LLMKG/vectors/tantivy_search/src/degradation.rs
- C:/code/LLMKG/vectors/tantivy_search/src/error.rs
- C:/code/LLMKG/vectors/tantivy_search/Cargo.toml

## Complete Context (For AI with ZERO Knowledge)

You are implementing a **logging and monitoring system** for the Tantivy-based search system. This provides structured logging, metrics collection, performance monitoring, and alerting capabilities essential for production observability.

**What is Logging and Monitoring?** A system that captures, processes, and analyzes operational data to provide visibility into system behavior, performance metrics, error patterns, and health status for debugging and optimization.

**System Context:** After task 44, we have graceful degradation with fallback strategies. This task adds comprehensive observability to track system behavior, detect issues early, and provide actionable insights.

**This Task:** Creates a MonitoringSystem with structured logging, metrics collection, performance tracking, and alert generation for production operations.

## Exact Steps (6 minutes implementation)

### Step 1: Add Monitoring Dependencies (1 minute)
Edit `C:/code/LLMKG/vectors/tantivy_search/Cargo.toml`, add to `[dependencies]` section:
```toml
# Logging and monitoring
tracing = "0.1.40"
tracing-subscriber = { version = "0.3.18", features = ["env-filter", "json"] }
tracing-appender = "0.2.3"
metrics = "0.23.0"
metrics-exporter-prometheus = "0.15.3"
sysinfo = "0.30.13"
```

### Step 2: Create Monitoring System Module (2.5 minutes)
Create `C:/code/LLMKG/vectors/tantivy_search/src/monitoring.rs`:
```rust
use crate::error::{SearchError, ErrorSeverity};
use metrics::{counter, gauge, histogram, register_counter, register_gauge, register_histogram};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{System, SystemExt, CpuExt, DiskExt, NetworkExt};
use tokio::sync::RwLock;
use tracing::{error, info, warn, debug, Level};
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f32,
    pub memory_usage: u64,
    pub memory_total: u64,
    pub disk_usage: u64,
    pub disk_total: u64,
    pub network_received: u64,
    pub network_transmitted: u64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchMetrics {
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub average_response_time: Duration,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub index_size: u64,
    pub active_connections: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    pub cpu_threshold: f32,
    pub memory_threshold: f32,
    pub response_time_threshold: Duration,
    pub error_rate_threshold: f32,
    pub disk_usage_threshold: f32,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            cpu_threshold: 80.0,          // 80% CPU usage
            memory_threshold: 85.0,       // 85% memory usage
            response_time_threshold: Duration::from_millis(1000), // 1 second
            error_rate_threshold: 0.05,   // 5% error rate
            disk_usage_threshold: 90.0,   // 90% disk usage
        }
    }
}

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub metrics_interval: Duration,
    pub log_level: Level,
    pub enable_json_logging: bool,
    pub enable_file_logging: bool,
    pub log_directory: String,
    pub enable_prometheus_metrics: bool,
    pub prometheus_port: u16,
    pub alert_config: AlertConfig,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            metrics_interval: Duration::from_secs(30),
            log_level: Level::INFO,
            enable_json_logging: false,
            enable_file_logging: true,
            log_directory: "logs".to_string(),
            enable_prometheus_metrics: true,
            prometheus_port: 9090,
            alert_config: AlertConfig::default(),
        }
    }
}

pub struct MonitoringSystem {
    config: MonitoringConfig,
    system_info: Arc<RwLock<System>>,
    search_metrics: Arc<RwLock<SearchMetrics>>,
    is_running: Arc<RwLock<bool>>,
    alert_history: Arc<RwLock<Vec<Alert>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: SystemTime,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    HighCpuUsage,
    HighMemoryUsage,
    SlowResponseTime,
    HighErrorRate,
    DiskSpaceLow,
    ComponentFailure,
    SystemError,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}
```

### Step 3: Implement Monitoring Logic (2 minutes)
Continue in `src/monitoring.rs`:
```rust
impl MonitoringSystem {
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            system_info: Arc::new(RwLock::new(System::new_all())),
            search_metrics: Arc::new(RwLock::new(SearchMetrics {
                total_queries: 0,
                successful_queries: 0,
                failed_queries: 0,
                average_response_time: Duration::ZERO,
                cache_hits: 0,
                cache_misses: 0,
                index_size: 0,
                active_connections: 0,
            })),
            is_running: Arc::new(RwLock::new(false)),
            alert_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn initialize(&self) -> Result<(), SearchError> {
        // Initialize tracing subscriber
        let filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(self.config.log_level.to_string()));

        let subscriber = FmtSubscriber::builder()
            .with_env_filter(filter)
            .with_target(false);

        if self.config.enable_json_logging {
            subscriber.json().init();
        } else {
            subscriber.init();
        }

        // Register Prometheus metrics
        if self.config.enable_prometheus_metrics {
            register_counter!("search_queries_total", "Total number of search queries");
            register_counter!("search_errors_total", "Total number of search errors");
            register_histogram!("search_duration_seconds", "Search query duration in seconds");
            register_gauge!("system_cpu_usage_percent", "CPU usage percentage");
            register_gauge!("system_memory_usage_bytes", "Memory usage in bytes");
            register_gauge!("active_connections", "Number of active connections");
        }

        info!("Monitoring system initialized successfully");
        Ok(())
    }

    pub async fn start(&self) -> Result<(), SearchError> {
        if *self.is_running.read().await {
            return Err(SearchError::InternalError {
                message: "Monitoring system already running".to_string(),
                component: "monitoring".to_string(),
                debug_info: None,
            });
        }

        *self.is_running.write().await = true;
        
        // Start metrics collection loop
        let system_info = Arc::clone(&self.system_info);
        let config = self.config.clone();
        let is_running = Arc::clone(&self.is_running);
        let alert_history = Arc::clone(&self.alert_history);

        tokio::spawn(async move {
            while *is_running.read().await {
                let start_time = Instant::now();
                
                // Update system metrics
                {
                    let mut system = system_info.write().await;
                    system.refresh_all();
                    
                    // Collect and record metrics
                    let cpu_usage = system.global_cpu_info().cpu_usage();
                    let memory_used = system.used_memory();
                    let memory_total = system.total_memory();
                    
                    gauge!("system_cpu_usage_percent", cpu_usage as f64);
                    gauge!("system_memory_usage_bytes", memory_used as f64);
                    
                    // Check for alerts
                    if cpu_usage > config.alert_config.cpu_threshold {
                        let alert = Alert {
                            id: format!("cpu_high_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()),
                            alert_type: AlertType::HighCpuUsage,
                            message: format!("CPU usage is {}%, threshold is {}%", cpu_usage, config.alert_config.cpu_threshold),
                            severity: AlertSeverity::Warning,
                            timestamp: SystemTime::now(),
                            resolved: false,
                        };
                        
                        alert_history.write().await.push(alert.clone());
                        warn!("Alert generated: {:?}", alert);
                    }
                    
                    let memory_usage_percent = (memory_used as f32 / memory_total as f32) * 100.0;
                    if memory_usage_percent > config.alert_config.memory_threshold {
                        let alert = Alert {
                            id: format!("memory_high_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()),
                            alert_type: AlertType::HighMemoryUsage,
                            message: format!("Memory usage is {:.1}%, threshold is {}%", memory_usage_percent, config.alert_config.memory_threshold),
                            severity: AlertSeverity::Critical,
                            timestamp: SystemTime::now(),
                            resolved: false,
                        };
                        
                        alert_history.write().await.push(alert.clone());
                        error!("Alert generated: {:?}", alert);
                    }
                }
                
                let elapsed = start_time.elapsed();
                if elapsed < config.metrics_interval {
                    tokio::time::sleep(config.metrics_interval - elapsed).await;
                }
            }
        });

        info!("Monitoring system started");
        Ok(())
    }

    pub async fn stop(&self) -> Result<(), SearchError> {
        *self.is_running.write().await = false;
        info!("Monitoring system stopped");
        Ok(())
    }

    pub async fn record_query(&self, duration: Duration, success: bool) {
        let mut metrics = self.search_metrics.write().await;
        metrics.total_queries += 1;
        
        if success {
            metrics.successful_queries += 1;
        } else {
            metrics.failed_queries += 1;
        }
        
        // Update average response time (simple moving average)
        let total_time = metrics.average_response_time * (metrics.total_queries - 1) as u32 + duration;
        metrics.average_response_time = total_time / metrics.total_queries as u32;

        // Update Prometheus metrics
        counter!("search_queries_total", 1);
        if !success {
            counter!("search_errors_total", 1);
        }
        histogram!("search_duration_seconds", duration.as_secs_f64());

        // Check for slow response time alert
        if duration > self.config.alert_config.response_time_threshold {
            let alert = Alert {
                id: format!("slow_response_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()),
                alert_type: AlertType::SlowResponseTime,
                message: format!("Query took {:?}, threshold is {:?}", duration, self.config.alert_config.response_time_threshold),
                severity: AlertSeverity::Warning,
                timestamp: SystemTime::now(),
                resolved: false,
            };
            
            self.alert_history.write().await.push(alert.clone());
            warn!("Alert generated: {:?}", alert);
        }
    }

    pub async fn record_error(&self, error: &SearchError) {
        counter!("search_errors_total", 1);
        
        match error.severity() {
            ErrorSeverity::Critical => {
                error!("Critical error occurred: {:?}", error);
                let alert = Alert {
                    id: format!("critical_error_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()),
                    alert_type: AlertType::SystemError,
                    message: format!("Critical error: {}", error),
                    severity: AlertSeverity::Critical,
                    timestamp: SystemTime::now(),
                    resolved: false,
                };
                self.alert_history.write().await.push(alert);
            }
            ErrorSeverity::High => {
                error!("High severity error: {:?}", error);
            }
            ErrorSeverity::Medium => {
                warn!("Medium severity error: {:?}", error);
            }
            ErrorSeverity::Low => {
                debug!("Low severity error: {:?}", error);
            }
        }
    }

    pub async fn get_system_metrics(&self) -> SystemMetrics {
        let system = self.system_info.read().await;
        SystemMetrics {
            cpu_usage: system.global_cpu_info().cpu_usage(),
            memory_usage: system.used_memory(),
            memory_total: system.total_memory(),
            disk_usage: system.disks().iter().map(|disk| disk.available_space()).sum(),
            disk_total: system.disks().iter().map(|disk| disk.total_space()).sum(),
            network_received: system.networks().iter().map(|(_, network)| network.received()).sum(),
            network_transmitted: system.networks().iter().map(|(_, network)| network.transmitted()).sum(),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }

    pub async fn get_search_metrics(&self) -> SearchMetrics {
        self.search_metrics.read().await.clone()
    }

    pub async fn get_alerts(&self, severity_filter: Option<AlertSeverity>) -> Vec<Alert> {
        let alerts = self.alert_history.read().await;
        match severity_filter {
            Some(min_severity) => alerts.iter()
                .filter(|alert| alert.severity >= min_severity)
                .cloned()
                .collect(),
            None => alerts.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitoring_system_lifecycle() {
        let config = MonitoringConfig::default();
        let monitor = MonitoringSystem::new(config);
        
        assert!(monitor.initialize().await.is_ok());
        assert!(monitor.start().await.is_ok());
        
        // Record some metrics
        monitor.record_query(Duration::from_millis(100), true).await;
        monitor.record_query(Duration::from_millis(200), false).await;
        
        let metrics = monitor.get_search_metrics().await;
        assert_eq!(metrics.total_queries, 2);
        assert_eq!(metrics.successful_queries, 1);
        assert_eq!(metrics.failed_queries, 1);
        
        assert!(monitor.stop().await.is_ok());
    }

    #[tokio::test]
    async fn test_alert_generation() {
        let config = MonitoringConfig::default();
        let monitor = MonitoringSystem::new(config);
        
        // Record a slow query that should generate an alert
        monitor.record_query(Duration::from_secs(2), true).await;
        
        let alerts = monitor.get_alerts(None).await;
        assert!(!alerts.is_empty());
        
        let slow_alerts: Vec<_> = alerts.iter()
            .filter(|alert| matches!(alert.alert_type, AlertType::SlowResponseTime))
            .collect();
        assert!(!slow_alerts.is_empty());
    }
}
```

### Step 4: Integration Points (0.5 minutes)
Add to `C:/code/LLMKG/vectors/tantivy_search/src/lib.rs`:
```rust
pub mod monitoring;
pub use monitoring::{MonitoringSystem, MonitoringConfig, SystemMetrics, SearchMetrics};
```

## Verification Steps (2 minutes)
```bash
cd C:/code/LLMKG/vectors/tantivy_search
cargo check
cargo test monitoring
cargo test test_monitoring_system_lifecycle
cargo test test_alert_generation
```

## Success Validation Checklist
- [ ] File exists: `src/monitoring.rs` with MonitoringSystem and metrics collection
- [ ] Dependencies added: `tracing`, `metrics`, `metrics-exporter-prometheus`, `sysinfo`
- [ ] Structured logging with configurable levels and JSON output support
- [ ] System metrics collection (CPU, memory, disk, network)
- [ ] Search metrics tracking (queries, response times, errors)
- [ ] Alert system with configurable thresholds and severity levels
- [ ] Prometheus metrics integration for external monitoring
- [ ] Command `cargo check` completes without errors
- [ ] All monitoring tests pass successfully

## Files Created For Next Task
1. **C:/code/LLMKG/vectors/tantivy_search/src/monitoring.rs** - Comprehensive monitoring system with logging and alerts
2. **Enhanced observability** - System now provides detailed metrics and monitoring capabilities

**Next Task (Task 46)** will implement recovery mechanisms for automatic system healing and error recovery.