# Task 41: Real-Time Monitoring with Live Dashboards

## Context
You are implementing Phase 4 of a vector indexing system. Performance reporting is now available. This task implements real-time monitoring capabilities with live dashboards, streaming updates, and interactive performance visualization.

## Current State
- `src/monitor.rs` exists with `PerformanceMonitor` struct
- Advanced statistical calculations and reporting are implemented
- Need real-time monitoring with live updates and dashboards

## Task Objective
Implement real-time monitoring capabilities including live dashboard generation, streaming performance data, real-time alerts, and interactive visualization for continuous performance monitoring.

## Implementation Requirements

### 1. Add real-time monitoring structures
Add these real-time monitoring structures to `src/monitor.rs`:
```rust
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone)]
pub struct RealTimeMonitor {
    monitor: SharedPerformanceMonitor,
    update_sender: Sender<MonitorUpdate>,
    is_running: Arc<AtomicBool>,
    update_interval: Duration,
    alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone)]
pub struct MonitorUpdate {
    pub timestamp: std::time::SystemTime,
    pub update_type: UpdateType,
    pub data: UpdateData,
}

#[derive(Debug, Clone)]
pub enum UpdateType {
    MetricUpdate,
    Alert,
    StatisticalChange,
    PerformanceDegradation,
}

#[derive(Debug, Clone)]
pub enum UpdateData {
    QueryMetric { time: Duration, total_count: usize },
    IndexMetric { time: Duration, total_count: usize },
    Alert { message: String, severity: AlertSeverity },
    Stats { basic: PerformanceStats, advanced: AdvancedPerformanceStats },
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub max_query_time: Duration,
    pub max_index_time: Duration,
    pub min_throughput_qps: f64,
    pub max_cv_threshold: f64,
    pub max_p99_ratio: f64, // P99/Average ratio
    pub degradation_window: usize,
    pub degradation_threshold: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_query_time: Duration::from_millis(1000), // 1 second
            max_index_time: Duration::from_millis(5000),  // 5 seconds
            min_throughput_qps: 1.0,
            max_cv_threshold: 0.5, // 50% coefficient of variation
            max_p99_ratio: 3.0,    // P99 should not be > 3x average
            degradation_window: 100,
            degradation_threshold: 0.2, // 20% performance degradation
        }
    }
}

#[derive(Debug, Clone)]
pub struct LiveDashboard {
    pub title: String,
    pub last_updated: std::time::SystemTime,
    pub current_stats: PerformanceStats,
    pub advanced_stats: AdvancedPerformanceStats,
    pub recent_updates: Vec<MonitorUpdate>,
    pub alerts: Vec<MonitorUpdate>,
    pub charts: Vec<LiveChart>,
}

#[derive(Debug, Clone)]
pub struct LiveChart {
    pub chart_id: String,
    pub title: String,
    pub chart_type: ChartType,
    pub data_window: usize,
    pub real_time_data: Vec<(f64, f64)>,
    pub update_frequency: Duration,
}
```

### 2. Implement real-time monitor
```rust
impl RealTimeMonitor {
    pub fn new(update_interval: Duration) -> (Self, Receiver<MonitorUpdate>) {
        let (sender, receiver) = channel();
        
        let monitor = Self {
            monitor: SharedPerformanceMonitor::new(),
            update_sender: sender,
            is_running: Arc::new(AtomicBool::new(false)),
            update_interval,
            alert_thresholds: AlertThresholds::default(),
        };
        
        (monitor, receiver)
    }
    
    pub fn with_thresholds(update_interval: Duration, thresholds: AlertThresholds) -> (Self, Receiver<MonitorUpdate>) {
        let (mut monitor, receiver) = Self::new(update_interval);
        monitor.alert_thresholds = thresholds;
        (monitor, receiver)
    }
    
    pub fn start(&self) {
        self.is_running.store(true, Ordering::Relaxed);
        let monitor_clone = self.monitor.clone();
        let sender_clone = self.update_sender.clone();
        let is_running_clone = self.is_running.clone();
        let interval = self.update_interval;
        let thresholds = self.alert_thresholds.clone();
        
        thread::spawn(move || {
            let mut last_stats = monitor_clone.get_stats().unwrap_or_default();
            
            while is_running_clone.load(Ordering::Relaxed) {
                thread::sleep(interval);
                
                if let Some(current_stats) = monitor_clone.get_stats() {
                    // Send metric updates
                    if current_stats.total_queries > last_stats.total_queries {
                        let _ = sender_clone.send(MonitorUpdate {
                            timestamp: std::time::SystemTime::now(),
                            update_type: UpdateType::MetricUpdate,
                            data: UpdateData::QueryMetric {
                                time: current_stats.avg_query_time,
                                total_count: current_stats.total_queries,
                            },
                        });
                    }
                    
                    if current_stats.total_indexes > last_stats.total_indexes {
                        let _ = sender_clone.send(MonitorUpdate {
                            timestamp: std::time::SystemTime::now(),
                            update_type: UpdateType::MetricUpdate,
                            data: UpdateData::IndexMetric {
                                time: current_stats.avg_index_time,
                                total_count: current_stats.total_indexes,
                            },
                        });
                    }
                    
                    // Check for alerts
                    Self::check_alerts(&current_stats, &thresholds, &sender_clone);
                    
                    // Send periodic statistical updates
                    let _ = sender_clone.send(MonitorUpdate {
                        timestamp: std::time::SystemTime::now(),
                        update_type: UpdateType::StatisticalChange,
                        data: UpdateData::Stats {
                            basic: current_stats.clone(),
                            advanced: AdvancedPerformanceStats {
                                basic_stats: current_stats.clone(),
                                // Note: Would need access to full monitor for advanced stats
                                ..Default::default()
                            },
                        },
                    });
                    
                    last_stats = current_stats;
                }
            }
        });
    }
    
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }
    
    pub fn record_query_time(&self, duration: Duration) {
        self.monitor.record_query_time(duration);
    }
    
    pub fn record_index_time(&self, duration: Duration) {
        self.monitor.record_index_time(duration);
    }
    
    pub fn get_current_stats(&self) -> Option<PerformanceStats> {
        self.monitor.get_stats()
    }
    
    fn check_alerts(stats: &PerformanceStats, thresholds: &AlertThresholds, sender: &Sender<MonitorUpdate>) {
        // Check query time threshold
        if stats.avg_query_time > thresholds.max_query_time {
            let _ = sender.send(MonitorUpdate {
                timestamp: std::time::SystemTime::now(),
                update_type: UpdateType::Alert,
                data: UpdateData::Alert {
                    message: format!("Average query time ({:.2}ms) exceeds threshold ({:.2}ms)", 
                        stats.avg_query_time.as_secs_f64() * 1000.0,
                        thresholds.max_query_time.as_secs_f64() * 1000.0),
                    severity: AlertSeverity::Warning,
                },
            });
        }
        
        // Check P99 query time
        if stats.p99_query_time > thresholds.max_query_time * 2 {
            let _ = sender.send(MonitorUpdate {
                timestamp: std::time::SystemTime::now(),
                update_type: UpdateType::Alert,
                data: UpdateData::Alert {
                    message: format!("P99 query time ({:.2}ms) is critically high", 
                        stats.p99_query_time.as_secs_f64() * 1000.0),
                    severity: AlertSeverity::Critical,
                },
            });
        }
        
        // Check index time threshold
        if stats.avg_index_time > thresholds.max_index_time {
            let _ = sender.send(MonitorUpdate {
                timestamp: std::time::SystemTime::now(),
                update_type: UpdateType::Alert,
                data: UpdateData::Alert {
                    message: format!("Average index time ({:.2}ms) exceeds threshold ({:.2}ms)", 
                        stats.avg_index_time.as_secs_f64() * 1000.0,
                        thresholds.max_index_time.as_secs_f64() * 1000.0),
                    severity: AlertSeverity::Warning,
                },
            });
        }
        
        // Check throughput
        if stats.queries_per_second < thresholds.min_throughput_qps && stats.total_queries > 10 {
            let _ = sender.send(MonitorUpdate {
                timestamp: std::time::SystemTime::now(),
                update_type: UpdateType::Alert,
                data: UpdateData::Alert {
                    message: format!("Query throughput ({:.2} qps) is below minimum threshold ({:.2} qps)", 
                        stats.queries_per_second, thresholds.min_throughput_qps),
                    severity: AlertSeverity::Warning,
                },
            });
        }
        
        // Check P99/Average ratio
        if stats.avg_query_time.as_secs_f64() > 0.0 {
            let p99_ratio = stats.p99_query_time.as_secs_f64() / stats.avg_query_time.as_secs_f64();
            if p99_ratio > thresholds.max_p99_ratio {
                let _ = sender.send(MonitorUpdate {
                    timestamp: std::time::SystemTime::now(),
                    update_type: UpdateType::Alert,
                    data: UpdateData::Alert {
                        message: format!("P99/Average ratio ({:.2}) indicates high tail latency", p99_ratio),
                        severity: AlertSeverity::Warning,
                    },
                });
            }
        }
    }
}
```

### 3. Implement live dashboard
```rust
impl LiveDashboard {
    pub fn new(title: String) -> Self {
        Self {
            title,
            last_updated: std::time::SystemTime::now(),
            current_stats: PerformanceStats::default(),
            advanced_stats: AdvancedPerformanceStats::default(),
            recent_updates: Vec::new(),
            alerts: Vec::new(),
            charts: Vec::new(),
        }
    }
    
    pub fn update_from_monitor(&mut self, stats: PerformanceStats, advanced_stats: AdvancedPerformanceStats) {
        self.current_stats = stats;
        self.advanced_stats = advanced_stats;
        self.last_updated = std::time::SystemTime::now();
        
        // Update charts with new data
        self.update_charts();
    }
    
    pub fn add_update(&mut self, update: MonitorUpdate) {
        // Keep only recent updates (last 100)
        if self.recent_updates.len() >= 100 {
            self.recent_updates.remove(0);
        }
        
        // Separate alerts from regular updates
        match &update.data {
            UpdateData::Alert { severity, .. } => {
                if self.alerts.len() >= 50 {
                    self.alerts.remove(0);
                }
                self.alerts.push(update);
            }
            _ => {
                self.recent_updates.push(update);
            }
        }
    }
    
    fn update_charts(&mut self) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        
        // Update query time chart
        if let Some(query_chart) = self.charts.iter_mut().find(|c| c.chart_id == "query_time") {
            query_chart.real_time_data.push((current_time, self.current_stats.avg_query_time.as_secs_f64() * 1000.0));
            if query_chart.real_time_data.len() > query_chart.data_window {
                query_chart.real_time_data.remove(0);
            }
        }
        
        // Update throughput chart
        if let Some(throughput_chart) = self.charts.iter_mut().find(|c| c.chart_id == "throughput") {
            throughput_chart.real_time_data.push((current_time, self.current_stats.queries_per_second));
            if throughput_chart.real_time_data.len() > throughput_chart.data_window {
                throughput_chart.real_time_data.remove(0);
            }
        }
    }
    
    pub fn add_chart(&mut self, chart: LiveChart) {
        self.charts.push(chart);
    }
    
    pub fn generate_html_dashboard(&self) -> String {
        let mut html = String::new();
        
        writeln!(&mut html, "<!DOCTYPE html>").unwrap();
        writeln!(&mut html, "<html><head>").unwrap();
        writeln!(&mut html, "<title>{}</title>", self.title).unwrap();
        writeln!(&mut html, "<meta http-equiv='refresh' content='5'>").unwrap(); // Auto-refresh every 5 seconds
        writeln!(&mut html, "<style>").unwrap();
        writeln!(&mut html, "body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}").unwrap();
        writeln!(&mut html, ".dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}").unwrap();
        writeln!(&mut html, ".metric-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}").unwrap();
        writeln!(&mut html, ".metric-value {{ font-size: 2em; font-weight: bold; color: #333; }}").unwrap();
        writeln!(&mut html, ".metric-label {{ color: #666; margin-top: 5px; }}").unwrap();
        writeln!(&mut html, ".alert {{ padding: 10px; margin: 5px 0; border-radius: 5px; }}").unwrap();
        writeln!(&mut html, ".alert-critical {{ background-color: #f8d7da; border-left: 4px solid #dc3545; }}").unwrap();
        writeln!(&mut html, ".alert-warning {{ background-color: #fff3cd; border-left: 4px solid #ffc107; }}").unwrap();
        writeln!(&mut html, ".alert-info {{ background-color: #d1ecf1; border-left: 4px solid #17a2b8; }}").unwrap();
        writeln!(&mut html, ".timestamp {{ color: #888; font-size: 0.9em; }}").unwrap();
        writeln!(&mut html, "h1 {{ text-align: center; color: #333; }}").unwrap();
        writeln!(&mut html, "h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}").unwrap();
        writeln!(&mut html, "</style>").unwrap();
        writeln!(&mut html, "</head><body>").unwrap();
        
        writeln!(&mut html, "<h1>{}</h1>", self.title).unwrap();
        writeln!(&mut html, "<div class='timestamp'>Last Updated: {:?}</div>", self.last_updated).unwrap();
        
        // Current metrics dashboard
        writeln!(&mut html, "<div class='dashboard'>").unwrap();
        
        // Query metrics
        writeln!(&mut html, "<div class='metric-card'>").unwrap();
        writeln!(&mut html, "<div class='metric-value'>{}</div>", self.current_stats.total_queries).unwrap();
        writeln!(&mut html, "<div class='metric-label'>Total Queries</div>").unwrap();
        writeln!(&mut html, "</div>").unwrap();
        
        writeln!(&mut html, "<div class='metric-card'>").unwrap();
        writeln!(&mut html, "<div class='metric-value'>{:.2}</div>", self.current_stats.queries_per_second).unwrap();
        writeln!(&mut html, "<div class='metric-label'>Queries/Second</div>").unwrap();
        writeln!(&mut html, "</div>").unwrap();
        
        writeln!(&mut html, "<div class='metric-card'>").unwrap();
        writeln!(&mut html, "<div class='metric-value'>{:.2}ms</div>", self.current_stats.avg_query_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut html, "<div class='metric-label'>Avg Query Time</div>").unwrap();
        writeln!(&mut html, "</div>").unwrap();
        
        writeln!(&mut html, "<div class='metric-card'>").unwrap();
        writeln!(&mut html, "<div class='metric-value'>{:.2}ms</div>", self.current_stats.p99_query_time.as_secs_f64() * 1000.0).unwrap();
        writeln!(&mut html, "<div class='metric-label'>P99 Query Time</div>").unwrap();
        writeln!(&mut html, "</div>").unwrap();
        
        writeln!(&mut html, "</div>").unwrap(); // End dashboard
        
        // Alerts section
        if !self.alerts.is_empty() {
            writeln!(&mut html, "<h2>Active Alerts</h2>").unwrap();
            for alert in &self.alerts {
                if let UpdateData::Alert { message, severity } = &alert.data {
                    let alert_class = match severity {
                        AlertSeverity::Critical => "alert-critical",
                        AlertSeverity::Warning => "alert-warning",
                        AlertSeverity::Info => "alert-info",
                    };
                    writeln!(&mut html, "<div class='alert {}'>", alert_class).unwrap();
                    writeln!(&mut html, "<strong>{:?}:</strong> {}", severity, message).unwrap();
                    writeln!(&mut html, "<div class='timestamp'>{:?}</div>", alert.timestamp).unwrap();
                    writeln!(&mut html, "</div>").unwrap();
                }
            }
        }
        
        // Recent updates
        writeln!(&mut html, "<h2>Recent Activity</h2>").unwrap();
        for update in self.recent_updates.iter().rev().take(10) {
            writeln!(&mut html, "<div class='alert alert-info'>").unwrap();
            match &update.data {
                UpdateData::QueryMetric { time, total_count } => {
                    writeln!(&mut html, "Query recorded: {:.2}ms (Total: {})", time.as_secs_f64() * 1000.0, total_count).unwrap();
                }
                UpdateData::IndexMetric { time, total_count } => {
                    writeln!(&mut html, "Index recorded: {:.2}ms (Total: {})", time.as_secs_f64() * 1000.0, total_count).unwrap();
                }
                UpdateData::Stats { basic, .. } => {
                    writeln!(&mut html, "Stats updated: {} queries, {:.2} qps", basic.total_queries, basic.queries_per_second).unwrap();
                }
                UpdateData::Alert { message, .. } => {
                    writeln!(&mut html, "Alert: {}", message).unwrap();
                }
            }
            writeln!(&mut html, "<div class='timestamp'>{:?}</div>", update.timestamp).unwrap();
            writeln!(&mut html, "</div>").unwrap();
        }
        
        writeln!(&mut html, "</body></html>").unwrap();
        
        html
    }
    
    pub fn get_current_alerts(&self) -> Vec<&MonitorUpdate> {
        self.alerts.iter()
            .filter(|update| {
                // Only show alerts from the last 5 minutes
                let five_minutes_ago = std::time::SystemTime::now() - Duration::from_secs(300);
                update.timestamp > five_minutes_ago
            })
            .collect()
    }
    
    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }
}
```

### 4. Add real-time monitoring tests
```rust
#[cfg(test)]
mod realtime_tests {
    use super::*;
    use std::time::{Duration, SystemTime};
    
    #[test]
    fn test_real_time_monitor_creation() {
        let (monitor, receiver) = RealTimeMonitor::new(Duration::from_millis(100));
        
        // Should be able to record metrics
        monitor.record_query_time(Duration::from_millis(50));
        
        let stats = monitor.get_current_stats();
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().total_queries, 1);
    }
    
    #[test]
    fn test_alert_thresholds() {
        let thresholds = AlertThresholds {
            max_query_time: Duration::from_millis(100),
            max_index_time: Duration::from_millis(500),
            min_throughput_qps: 5.0,
            ..Default::default()
        };
        
        let (monitor, receiver) = RealTimeMonitor::with_thresholds(Duration::from_millis(50), thresholds);
        
        // Record a slow query that should trigger an alert
        monitor.record_query_time(Duration::from_millis(150));
        
        monitor.start();
        
        // Wait for potential updates
        thread::sleep(Duration::from_millis(100));
        
        // Check if we received any updates (alerts should be sent)
        let updates: Vec<_> = receiver.try_iter().collect();
        assert!(!updates.is_empty());
        
        monitor.stop();
    }
    
    #[test]
    fn test_live_dashboard_creation() {
        let mut dashboard = LiveDashboard::new("Test Dashboard".to_string());
        
        assert_eq!(dashboard.title, "Test Dashboard");
        assert!(dashboard.recent_updates.is_empty());
        assert!(dashboard.alerts.is_empty());
    }
    
    #[test]
    fn test_dashboard_updates() {
        let mut dashboard = LiveDashboard::new("Test Dashboard".to_string());
        
        // Add a regular update
        let update = MonitorUpdate {
            timestamp: SystemTime::now(),
            update_type: UpdateType::MetricUpdate,
            data: UpdateData::QueryMetric {
                time: Duration::from_millis(50),
                total_count: 1,
            },
        };
        
        dashboard.add_update(update);
        assert_eq!(dashboard.recent_updates.len(), 1);
        assert_eq!(dashboard.alerts.len(), 0);
        
        // Add an alert
        let alert = MonitorUpdate {
            timestamp: SystemTime::now(),
            update_type: UpdateType::Alert,
            data: UpdateData::Alert {
                message: "Test alert".to_string(),
                severity: AlertSeverity::Warning,
            },
        };
        
        dashboard.add_update(alert);
        assert_eq!(dashboard.recent_updates.len(), 1);
        assert_eq!(dashboard.alerts.len(), 1);
    }
    
    #[test]
    fn test_html_dashboard_generation() {
        let mut dashboard = LiveDashboard::new("Test Dashboard".to_string());
        
        // Set some test data
        dashboard.current_stats = PerformanceStats {
            total_queries: 100,
            queries_per_second: 10.5,
            avg_query_time: Duration::from_millis(25),
            p99_query_time: Duration::from_millis(75),
            ..Default::default()
        };
        
        let html = dashboard.generate_html_dashboard();
        
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test Dashboard"));
        assert!(html.contains("100")); // total queries
        assert!(html.contains("10.50")); // queries per second
        assert!(html.contains("25.00ms")); // avg query time
        assert!(html.contains("75.00ms")); // p99 query time
        assert!(html.contains("refresh")); // auto-refresh meta tag
    }
    
    #[test]
    fn test_chart_updates() {
        let mut dashboard = LiveDashboard::new("Test Dashboard".to_string());
        
        // Add a chart
        let chart = LiveChart {
            chart_id: "query_time".to_string(),
            title: "Query Time".to_string(),
            chart_type: ChartType::TimeSeries,
            data_window: 100,
            real_time_data: Vec::new(),
            update_frequency: Duration::from_millis(100),
        };
        
        dashboard.add_chart(chart);
        
        // Update dashboard with stats
        let stats = PerformanceStats {
            avg_query_time: Duration::from_millis(50),
            ..Default::default()
        };
        let advanced_stats = AdvancedPerformanceStats::default();
        
        dashboard.update_from_monitor(stats, advanced_stats);
        
        // Check that chart was updated
        let query_chart = dashboard.charts.iter().find(|c| c.chart_id == "query_time").unwrap();
        assert_eq!(query_chart.real_time_data.len(), 1);
        assert_eq!(query_chart.real_time_data[0].1, 50.0); // 50ms
    }
    
    #[test]
    fn test_alert_filtering() {
        let mut dashboard = LiveDashboard::new("Test Dashboard".to_string());
        
        // Add an old alert (6 minutes ago)
        let old_alert = MonitorUpdate {
            timestamp: SystemTime::now() - Duration::from_secs(360),
            update_type: UpdateType::Alert,
            data: UpdateData::Alert {
                message: "Old alert".to_string(),
                severity: AlertSeverity::Warning,
            },
        };
        
        // Add a recent alert
        let recent_alert = MonitorUpdate {
            timestamp: SystemTime::now(),
            update_type: UpdateType::Alert,
            data: UpdateData::Alert {
                message: "Recent alert".to_string(),
                severity: AlertSeverity::Critical,
            },
        };
        
        dashboard.add_update(old_alert);
        dashboard.add_update(recent_alert);
        
        let current_alerts = dashboard.get_current_alerts();
        assert_eq!(current_alerts.len(), 1); // Only recent alert should be returned
        
        if let UpdateData::Alert { message, .. } = &current_alerts[0].data {
            assert_eq!(message, "Recent alert");
        }
    }
}
```

### 5. Add default implementations for missing traits
```rust
impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            avg_query_time: Duration::from_millis(0),
            p95_query_time: Duration::from_millis(0),
            p99_query_time: Duration::from_millis(0),
            min_query_time: Duration::from_millis(0),
            max_query_time: Duration::from_millis(0),
            avg_index_time: Duration::from_millis(0),
            p95_index_time: Duration::from_millis(0),
            p99_index_time: Duration::from_millis(0),
            min_index_time: Duration::from_millis(0),
            max_index_time: Duration::from_millis(0),
            total_queries: 0,
            total_indexes: 0,
            uptime: Duration::from_millis(0),
            queries_per_second: 0.0,
            indexes_per_second: 0.0,
        }
    }
}

impl Default for AdvancedPerformanceStats {
    fn default() -> Self {
        Self {
            basic_stats: PerformanceStats::default(),
            query_std_dev: Duration::from_millis(0),
            query_cv: 0.0,
            query_iqr: Duration::from_millis(0),
            query_outliers: Vec::new(),
            query_trend_slope: 0.0,
            query_correlation: 0.0,
            query_confidence_interval: (Duration::from_millis(0), Duration::from_millis(0)),
            query_prediction: Duration::from_millis(0),
            index_std_dev: Duration::from_millis(0),
            index_cv: 0.0,
            index_iqr: Duration::from_millis(0),
            index_outliers: Vec::new(),
            index_trend_slope: 0.0,
            index_correlation: 0.0,
            index_confidence_interval: (Duration::from_millis(0), Duration::from_millis(0)),
            index_prediction: Duration::from_millis(0),
        }
    }
}
```

## Success Criteria
- [ ] Real-time monitor starts and stops correctly
- [ ] Live updates are sent through channels successfully
- [ ] Alert thresholds trigger appropriate notifications
- [ ] Live dashboard generates proper HTML with auto-refresh
- [ ] Charts update with real-time data correctly
- [ ] Alert filtering shows only recent alerts
- [ ] Thread-safe operation works correctly
- [ ] HTML dashboard includes all performance metrics
- [ ] All tests pass with correct functionality
- [ ] No compilation errors or warnings

## Time Limit
10 minutes

## Notes
- Uses channels for real-time communication between monitor and dashboard
- Auto-refreshing HTML dashboard for live monitoring
- Configurable alert thresholds with multiple severity levels
- Thread-safe real-time monitoring with atomic operations
- Automatic chart updates with rolling data windows
- Alert filtering to show only recent notifications
- Live dashboard includes metric cards and activity feed