//! Real-time Performance Monitor
//! 
//! Provides real-time performance monitoring with configurable sampling and alerting.

use crate::infrastructure::{PerformanceMetrics, MetricsCollector, TestConfig};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{broadcast, mpsc};
use tokio::time::interval;

/// Real-time performance monitor
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Monitor configuration
    config: MonitorConfig,
    /// Currently active test monitors
    active_monitors: Arc<RwLock<HashMap<String, TestMonitor>>>,
    /// Global metrics collector
    global_collector: Arc<Mutex<MetricsCollector>>,
    /// Event broadcaster for real-time updates
    event_broadcaster: broadcast::Sender<MonitorEvent>,
    /// Monitor state
    state: Arc<Mutex<MonitorState>>,
    /// Alert manager
    alert_manager: Arc<Mutex<AlertManager>>,
}

/// Monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Global sampling interval
    pub global_sampling_interval: Duration,
    /// Per-test sampling interval
    pub test_sampling_interval: Duration,
    /// Maximum number of samples to keep per test
    pub max_samples_per_test: usize,
    /// Maximum number of global samples
    pub max_global_samples: usize,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Enable performance trending
    pub enable_trending: bool,
    /// Trend analysis window
    pub trend_window: Duration,
}

/// Alert thresholds for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Memory usage threshold (percentage)
    pub memory_threshold_percent: f64,
    /// CPU usage threshold (percentage)
    pub cpu_threshold_percent: f64,
    /// Latency threshold (milliseconds)
    pub latency_threshold_ms: f64,
    /// I/O rate threshold (bytes/sec)
    pub io_rate_threshold_bps: u64,
    /// Network rate threshold (bytes/sec)
    pub network_rate_threshold_bps: u64,
}

/// Monitor state
#[derive(Debug, Clone)]
struct MonitorState {
    /// Whether global monitoring is active
    pub global_active: bool,
    /// Global monitoring start time
    pub global_start_time: Option<Instant>,
    /// Total tests monitored
    pub total_tests_monitored: u64,
    /// Active test count
    pub active_test_count: u32,
}

/// Test-specific monitor
#[derive(Debug)]
struct TestMonitor {
    /// Test name
    test_name: String,
    /// Test-specific metrics collector
    collector: MetricsCollector,
    /// Monitoring start time
    start_time: Instant,
    /// Whether monitoring is active
    active: bool,
    /// Alert state for this test
    alert_state: AlertState,
    /// Performance trends
    trends: PerformanceTrends,
}

/// Alert state for a test
#[derive(Debug, Clone)]
struct AlertState {
    /// Current active alerts
    active_alerts: Vec<Alert>,
    /// Alert history
    alert_history: VecDeque<Alert>,
    /// Last alert check time
    last_check: Instant,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Alert ID
    pub id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert level
    pub level: AlertLevel,
    /// Test name (if test-specific)
    pub test_name: Option<String>,
    /// Alert message
    pub message: String,
    /// Timestamp when alert was triggered
    pub timestamp: SystemTime,
    /// Metric values that triggered the alert
    pub trigger_values: HashMap<String, f64>,
    /// Whether alert is still active
    pub active: bool,
}

/// Types of performance alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    /// High memory usage
    HighMemoryUsage,
    /// High CPU usage
    HighCpuUsage,
    /// High latency
    HighLatency,
    /// High I/O rate
    HighIoRate,
    /// High network usage
    HighNetworkUsage,
    /// Performance regression
    PerformanceRegression,
    /// Resource exhaustion
    ResourceExhaustion,
    /// Anomaly detection
    Anomaly,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Performance trends for a test
#[derive(Debug, Clone)]
struct PerformanceTrends {
    /// Latency trend (samples over time)
    latency_trend: VecDeque<(Instant, Duration)>,
    /// Memory trend
    memory_trend: VecDeque<(Instant, u64)>,
    /// CPU trend
    cpu_trend: VecDeque<(Instant, f64)>,
    /// Trend analysis window
    window_duration: Duration,
}

/// Alert manager
#[derive(Debug)]
struct AlertManager {
    /// Active alerts by ID
    active_alerts: HashMap<String, Alert>,
    /// Alert callbacks
    alert_callbacks: Vec<Box<dyn Fn(&Alert) + Send + Sync>>,
    /// Alert suppression rules
    suppression_rules: Vec<SuppressionRule>,
}

/// Alert suppression rule
#[derive(Debug, Clone)]
struct SuppressionRule {
    /// Alert type to suppress
    alert_type: AlertType,
    /// Test name pattern (regex)
    test_pattern: Option<String>,
    /// Suppression duration
    duration: Duration,
    /// Last triggered time
    last_triggered: Option<Instant>,
}

/// Monitor events for real-time updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitorEvent {
    /// Test monitoring started
    TestStarted {
        test_name: String,
        timestamp: SystemTime,
    },
    /// Test monitoring stopped
    TestStopped {
        test_name: String,
        timestamp: SystemTime,
        summary: TestMonitorSummary,
    },
    /// Performance metrics update
    MetricsUpdate {
        test_name: Option<String>,
        metrics: PerformanceMetrics,
    },
    /// Alert triggered
    AlertTriggered {
        alert: Alert,
    },
    /// Alert resolved
    AlertResolved {
        alert_id: String,
        timestamp: SystemTime,
    },
    /// Global monitoring state change
    GlobalStateChange {
        active: bool,
        timestamp: SystemTime,
    },
}

/// Test monitoring summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMonitorSummary {
    /// Test name
    pub test_name: String,
    /// Monitoring duration
    pub duration: Duration,
    /// Total samples collected
    pub sample_count: usize,
    /// Average metrics
    pub average_metrics: PerformanceMetrics,
    /// Peak metrics
    pub peak_metrics: PerformanceMetrics,
    /// Alerts triggered
    pub alerts_triggered: u32,
    /// Performance grade
    pub performance_grade: PerformanceGrade,
}

/// Performance grade for a test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceGrade {
    Excellent,
    Good,
    Fair,
    Poor,
    Failed,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(config: &TestConfig) -> Result<Self> {
        let monitor_config = MonitorConfig {
            global_sampling_interval: Duration::from_millis(100),
            test_sampling_interval: Duration::from_millis(50),
            max_samples_per_test: 1000,
            max_global_samples: 10000,
            enable_alerts: true,
            alert_thresholds: AlertThresholds {
                memory_threshold_percent: 90.0,
                cpu_threshold_percent: 95.0,
                latency_threshold_ms: 1000.0,
                io_rate_threshold_bps: 100 * 1024 * 1024, // 100 MB/s
                network_rate_threshold_bps: 10 * 1024 * 1024, // 10 MB/s
            },
            enable_trending: true,
            trend_window: Duration::from_minutes(5),
        };

        let global_collector = Arc::new(Mutex::new(MetricsCollector::new(
            monitor_config.global_sampling_interval,
            monitor_config.max_global_samples,
        )));

        let (event_broadcaster, _) = broadcast::channel(1000);

        let state = Arc::new(Mutex::new(MonitorState {
            global_active: false,
            global_start_time: None,
            total_tests_monitored: 0,
            active_test_count: 0,
        }));

        let alert_manager = Arc::new(Mutex::new(AlertManager {
            active_alerts: HashMap::new(),
            alert_callbacks: Vec::new(),
            suppression_rules: Vec::new(),
        }));

        Ok(Self {
            config: monitor_config,
            active_monitors: Arc::new(RwLock::new(HashMap::new())),
            global_collector,
            event_broadcaster,
            state,
            alert_manager,
        })
    }

    /// Start global performance monitoring
    pub async fn start_global_monitoring(&self) -> Result<()> {
        let mut state = self.state.lock()
            .map_err(|_| anyhow!("Failed to acquire state lock"))?;

        if state.global_active {
            return Err(anyhow!("Global monitoring already active"));
        }

        state.global_active = true;
        state.global_start_time = Some(Instant::now());

        let mut collector = self.global_collector.lock()
            .map_err(|_| anyhow!("Failed to acquire collector lock"))?;
        collector.start()?;

        // Start background monitoring task
        self.start_global_monitoring_task().await?;

        // Broadcast event
        let _ = self.event_broadcaster.send(MonitorEvent::GlobalStateChange {
            active: true,
            timestamp: SystemTime::now(),
        });

        Ok(())
    }

    /// Stop global performance monitoring
    pub async fn stop_global_monitoring(&self) -> Result<Vec<PerformanceMetrics>> {
        let mut state = self.state.lock()
            .map_err(|_| anyhow!("Failed to acquire state lock"))?;

        if !state.global_active {
            return Err(anyhow!("Global monitoring not active"));
        }

        state.global_active = false;
        state.global_start_time = None;

        let mut collector = self.global_collector.lock()
            .map_err(|_| anyhow!("Failed to acquire collector lock"))?;
        let samples = collector.stop()?;

        // Broadcast event
        let _ = self.event_broadcaster.send(MonitorEvent::GlobalStateChange {
            active: false,
            timestamp: SystemTime::now(),
        });

        Ok(samples)
    }

    /// Start monitoring a specific test
    pub async fn start_test_monitoring(&self, test_name: &str) -> Result<()> {
        let mut monitors = self.active_monitors.write()
            .map_err(|_| anyhow!("Failed to acquire monitors lock"))?;

        if monitors.contains_key(test_name) {
            return Err(anyhow!("Test {} already being monitored", test_name));
        }

        let mut collector = MetricsCollector::new(
            self.config.test_sampling_interval,
            self.config.max_samples_per_test,
        );
        collector.start()?;

        let test_monitor = TestMonitor {
            test_name: test_name.to_string(),
            collector,
            start_time: Instant::now(),
            active: true,
            alert_state: AlertState {
                active_alerts: Vec::new(),
                alert_history: VecDeque::new(),
                last_check: Instant::now(),
            },
            trends: PerformanceTrends {
                latency_trend: VecDeque::new(),
                memory_trend: VecDeque::new(),
                cpu_trend: VecDeque::new(),
                window_duration: self.config.trend_window,
            },
        };

        monitors.insert(test_name.to_string(), test_monitor);

        // Update state
        let mut state = self.state.lock()
            .map_err(|_| anyhow!("Failed to acquire state lock"))?;
        state.active_test_count += 1;
        state.total_tests_monitored += 1;

        // Start test monitoring task
        self.start_test_monitoring_task(test_name).await?;

        // Broadcast event
        let _ = self.event_broadcaster.send(MonitorEvent::TestStarted {
            test_name: test_name.to_string(),
            timestamp: SystemTime::now(),
        });

        Ok(())
    }

    /// Stop monitoring a specific test
    pub async fn stop_test_monitoring(&self, test_name: &str) -> Result<HashMap<String, f64>> {
        let mut monitors = self.active_monitors.write()
            .map_err(|_| anyhow!("Failed to acquire monitors lock"))?;

        let test_monitor = monitors.remove(test_name)
            .ok_or_else(|| anyhow!("Test {} not being monitored", test_name))?;

        let samples = test_monitor.collector.stop()?;
        let duration = test_monitor.start_time.elapsed();

        // Update state
        let mut state = self.state.lock()
            .map_err(|_| anyhow!("Failed to acquire state lock"))?;
        state.active_test_count -= 1;

        // Create summary
        let summary = self.create_test_summary(test_name, &samples, duration).await?;

        // Extract performance metrics as HashMap
        let mut metrics_map = HashMap::new();
        if let Some(last_sample) = samples.last() {
            metrics_map.insert("avg_latency_ms".to_string(), 
                             last_sample.latency_stats.mean.as_millis() as f64);
            metrics_map.insert("memory_bytes".to_string(), 
                             last_sample.memory_stats.average_rss_bytes as f64);
            metrics_map.insert("cpu_percent".to_string(), 
                             last_sample.cpu_stats.average_cpu_percent);
        }

        // Broadcast event
        let _ = self.event_broadcaster.send(MonitorEvent::TestStopped {
            test_name: test_name.to_string(),
            timestamp: SystemTime::now(),
            summary,
        });

        Ok(metrics_map)
    }

    /// Record a latency measurement for a test
    pub async fn record_latency(&self, test_name: &str, latency: Duration) -> Result<()> {
        let monitors = self.active_monitors.read()
            .map_err(|_| anyhow!("Failed to acquire monitors lock"))?;

        if let Some(monitor) = monitors.get(test_name) {
            monitor.collector.record_latency(latency)?;
        }

        Ok(())
    }

    /// Record a custom metric for a test
    pub async fn record_custom_metric(&self, test_name: &str, metric_name: String, value: f64) -> Result<()> {
        let monitors = self.active_monitors.read()
            .map_err(|_| anyhow!("Failed to acquire monitors lock"))?;

        if let Some(monitor) = monitors.get(test_name) {
            monitor.collector.record_custom_metric(metric_name, value)?;
        }

        Ok(())
    }

    /// Get current performance snapshot
    pub async fn get_performance_snapshot(&self) -> Result<PerformanceSnapshot> {
        let monitors = self.active_monitors.read()
            .map_err(|_| anyhow!("Failed to acquire monitors lock"))?;

        let state = self.state.lock()
            .map_err(|_| anyhow!("Failed to acquire state lock"))?;

        let mut test_snapshots = HashMap::new();
        for (test_name, monitor) in monitors.iter() {
            if let Ok(metrics) = monitor.collector.collect_system_metrics() {
                test_snapshots.insert(test_name.clone(), metrics);
            }
        }

        let global_metrics = if state.global_active {
            let collector = self.global_collector.lock()
                .map_err(|_| anyhow!("Failed to acquire collector lock"))?;
            Some(collector.collect_system_metrics()?)
        } else {
            None
        };

        Ok(PerformanceSnapshot {
            timestamp: SystemTime::now(),
            global_metrics,
            test_metrics: test_snapshots,
            active_test_count: state.active_test_count,
            global_monitoring_active: state.global_active,
        })
    }

    /// Subscribe to real-time monitor events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<MonitorEvent> {
        self.event_broadcaster.subscribe()
    }

    /// Add an alert callback
    pub fn add_alert_callback<F>(&self, callback: F) -> Result<()>
    where
        F: Fn(&Alert) + Send + Sync + 'static,
    {
        let mut alert_manager = self.alert_manager.lock()
            .map_err(|_| anyhow!("Failed to acquire alert manager lock"))?;
        
        alert_manager.alert_callbacks.push(Box::new(callback));
        Ok(())
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Result<Vec<Alert>> {
        let alert_manager = self.alert_manager.lock()
            .map_err(|_| anyhow!("Failed to acquire alert manager lock"))?;
        
        Ok(alert_manager.active_alerts.values().cloned().collect())
    }

    /// Start background global monitoring task
    async fn start_global_monitoring_task(&self) -> Result<()> {
        let collector = Arc::clone(&self.global_collector);
        let broadcaster = self.event_broadcaster.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.global_sampling_interval);
            
            loop {
                interval.tick().await;
                
                if let Ok(collector) = collector.lock() {
                    if !collector.active {
                        break;
                    }
                    
                    if let Ok(()) = collector.sample() {
                        if let Ok(metrics) = collector.collect_system_metrics() {
                            let _ = broadcaster.send(MonitorEvent::MetricsUpdate {
                                test_name: None,
                                metrics,
                            });
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Start background test monitoring task
    async fn start_test_monitoring_task(&self, test_name: &str) -> Result<()> {
        let test_name = test_name.to_string();
        let monitors = Arc::clone(&self.active_monitors);
        let broadcaster = self.event_broadcaster.clone();
        let config = self.config.clone();
        let alert_manager = Arc::clone(&self.alert_manager);

        tokio::spawn(async move {
            let mut interval = interval(config.test_sampling_interval);
            
            loop {
                interval.tick().await;
                
                let should_continue = {
                    if let Ok(monitors) = monitors.read() {
                        if let Some(monitor) = monitors.get(&test_name) {
                            if !monitor.active {
                                false
                            } else {
                                // Sample metrics
                                if let Ok(()) = monitor.collector.sample() {
                                    if let Ok(metrics) = monitor.collector.collect_system_metrics() {
                                        let _ = broadcaster.send(MonitorEvent::MetricsUpdate {
                                            test_name: Some(test_name.clone()),
                                            metrics: metrics.clone(),
                                        });
                                        
                                        // Check for alerts
                                        if config.enable_alerts {
                                            if let Ok(mut alert_mgr) = alert_manager.lock() {
                                                Self::check_alerts(&test_name, &metrics, &config.alert_thresholds, &mut alert_mgr, &broadcaster);
                                            }
                                        }
                                    }
                                }
                                true
                            }
                        } else {
                            false
                        }
                    } else {
                        false
                    }
                };
                
                if !should_continue {
                    break;
                }
            }
        });

        Ok(())
    }

    /// Check for performance alerts
    fn check_alerts(
        test_name: &str,
        metrics: &PerformanceMetrics,
        thresholds: &AlertThresholds,
        alert_manager: &mut AlertManager,
        broadcaster: &broadcast::Sender<MonitorEvent>,
    ) {
        let mut new_alerts = Vec::new();

        // Check memory usage
        let memory_percent = (metrics.memory_stats.average_rss_bytes as f64 / 
                            metrics.memory_stats.peak_rss_bytes as f64) * 100.0;
        if memory_percent > thresholds.memory_threshold_percent {
            let alert = Alert {
                id: format!("mem_{}_{}", test_name, SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()),
                alert_type: AlertType::HighMemoryUsage,
                level: AlertLevel::Warning,
                test_name: Some(test_name.to_string()),
                message: format!("High memory usage: {:.2}%", memory_percent),
                timestamp: SystemTime::now(),
                trigger_values: [("memory_percent".to_string(), memory_percent)].iter().cloned().collect(),
                active: true,
            };
            new_alerts.push(alert);
        }

        // Check CPU usage
        if metrics.cpu_stats.average_cpu_percent > thresholds.cpu_threshold_percent {
            let alert = Alert {
                id: format!("cpu_{}_{}", test_name, SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()),
                alert_type: AlertType::HighCpuUsage,
                level: AlertLevel::Warning,
                test_name: Some(test_name.to_string()),
                message: format!("High CPU usage: {:.2}%", metrics.cpu_stats.average_cpu_percent),
                timestamp: SystemTime::now(),
                trigger_values: [("cpu_percent".to_string(), metrics.cpu_stats.average_cpu_percent)].iter().cloned().collect(),
                active: true,
            };
            new_alerts.push(alert);
        }

        // Check latency
        let latency_ms = metrics.latency_stats.mean.as_millis() as f64;
        if latency_ms > thresholds.latency_threshold_ms {
            let alert = Alert {
                id: format!("lat_{}_{}", test_name, SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()),
                alert_type: AlertType::HighLatency,
                level: AlertLevel::Warning,
                test_name: Some(test_name.to_string()),
                message: format!("High latency: {:.2}ms", latency_ms),
                timestamp: SystemTime::now(),
                trigger_values: [("latency_ms".to_string(), latency_ms)].iter().cloned().collect(),
                active: true,
            };
            new_alerts.push(alert);
        }

        // Process new alerts
        for alert in new_alerts {
            alert_manager.active_alerts.insert(alert.id.clone(), alert.clone());
            
            // Call alert callbacks
            for callback in &alert_manager.alert_callbacks {
                callback(&alert);
            }
            
            // Broadcast alert event
            let _ = broadcaster.send(MonitorEvent::AlertTriggered { alert });
        }
    }

    /// Create test monitoring summary
    async fn create_test_summary(
        &self,
        test_name: &str,
        samples: &[PerformanceMetrics],
        duration: Duration,
    ) -> Result<TestMonitorSummary> {
        if samples.is_empty() {
            return Ok(TestMonitorSummary {
                test_name: test_name.to_string(),
                duration,
                sample_count: 0,
                average_metrics: self.create_empty_metrics(),
                peak_metrics: self.create_empty_metrics(),
                alerts_triggered: 0,
                performance_grade: PerformanceGrade::Failed,
            });
        }

        // Calculate averages and peaks
        let sample_count = samples.len();
        
        let avg_latency_ns: u64 = samples.iter()
            .map(|s| s.latency_stats.mean.as_nanos() as u64)
            .sum::<u64>() / sample_count as u64;
        
        let avg_memory: u64 = samples.iter()
            .map(|s| s.memory_stats.average_rss_bytes)
            .sum::<u64>() / sample_count as u64;
        
        let avg_cpu: f64 = samples.iter()
            .map(|s| s.cpu_stats.average_cpu_percent)
            .sum::<f64>() / sample_count as f64;

        let peak_memory = samples.iter()
            .map(|s| s.memory_stats.peak_rss_bytes)
            .max()
            .unwrap_or(0);
        
        let peak_cpu = samples.iter()
            .map(|s| s.cpu_stats.peak_cpu_percent)
            .fold(0.0f64, |a, b| a.max(*b));

        // Create average and peak metrics
        let average_metrics = self.create_metrics_with_values(
            Duration::from_nanos(avg_latency_ns),
            avg_memory,
            avg_cpu,
        );
        
        let peak_metrics = self.create_metrics_with_values(
            Duration::from_nanos(avg_latency_ns), // Use average for latency
            peak_memory,
            peak_cpu,
        );

        // Determine performance grade
        let performance_grade = self.calculate_performance_grade(&average_metrics);

        Ok(TestMonitorSummary {
            test_name: test_name.to_string(),
            duration,
            sample_count,
            average_metrics,
            peak_metrics,
            alerts_triggered: 0, // Would track this from alert history
            performance_grade,
        })
    }

    /// Create empty metrics structure
    fn create_empty_metrics(&self) -> PerformanceMetrics {
        self.create_metrics_with_values(Duration::from_nanos(0), 0, 0.0)
    }

    /// Create metrics with specific values
    fn create_metrics_with_values(&self, latency: Duration, memory: u64, cpu: f64) -> PerformanceMetrics {
        PerformanceMetrics {
            latency_stats: crate::infrastructure::LatencyStats {
                min: latency,
                max: latency,
                mean: latency,
                median: latency,
                p95: latency,
                p99: latency,
                p999: latency,
                std_dev: Duration::from_nanos(0),
                count: 1,
                sum: latency,
            },
            memory_stats: crate::infrastructure::MemoryStats {
                peak_rss_bytes: memory,
                average_rss_bytes: memory,
                heap_allocations: 0,
                heap_deallocations: 0,
                peak_heap_bytes: memory,
                current_heap_bytes: memory,
                stack_size_bytes: 0,
                virtual_memory_bytes: memory,
                major_page_faults: 0,
                minor_page_faults: 0,
            },
            cpu_stats: crate::infrastructure::CpuStats {
                total_cpu_time: Duration::from_secs(0),
                user_cpu_time: Duration::from_secs(0),
                system_cpu_time: Duration::from_secs(0),
                average_cpu_percent: cpu,
                peak_cpu_percent: cpu,
                context_switches: 0,
                cores_used: 1,
                cache_misses: 0,
                cache_hits: 0,
            },
            io_stats: crate::infrastructure::IoStats {
                disk_read_bytes: 0,
                disk_write_bytes: 0,
                read_operations: 0,
                write_operations: 0,
                avg_read_latency: Duration::from_nanos(0),
                avg_write_latency: Duration::from_nanos(0),
                io_wait_time: Duration::from_nanos(0),
                sequential_read_percent: 0.0,
                random_read_percent: 0.0,
            },
            network_stats: crate::infrastructure::NetworkStats {
                bytes_sent: 0,
                bytes_received: 0,
                packets_sent: 0,
                packets_received: 0,
                network_errors: 0,
                connections: 0,
                avg_latency: Duration::from_nanos(0),
                bandwidth_utilization: 0.0,
            },
            custom_metrics: HashMap::new(),
            timestamp: SystemTime::now(),
            collection_duration: Duration::from_secs(0),
        }
    }

    /// Calculate performance grade based on metrics
    fn calculate_performance_grade(&self, metrics: &PerformanceMetrics) -> PerformanceGrade {
        let latency_ms = metrics.latency_stats.mean.as_millis() as f64;
        let memory_mb = metrics.memory_stats.average_rss_bytes as f64 / (1024.0 * 1024.0);
        let cpu_percent = metrics.cpu_stats.average_cpu_percent;

        // Simple grading logic based on LLMKG performance targets
        if latency_ms <= 1.0 && memory_mb <= 70.0 && cpu_percent <= 25.0 {
            PerformanceGrade::Excellent
        } else if latency_ms <= 5.0 && memory_mb <= 100.0 && cpu_percent <= 50.0 {
            PerformanceGrade::Good
        } else if latency_ms <= 10.0 && memory_mb <= 200.0 && cpu_percent <= 75.0 {
            PerformanceGrade::Fair
        } else if latency_ms <= 100.0 && memory_mb <= 500.0 && cpu_percent <= 95.0 {
            PerformanceGrade::Poor
        } else {
            PerformanceGrade::Failed
        }
    }
}

/// Performance snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Timestamp of snapshot
    pub timestamp: SystemTime,
    /// Global performance metrics (if global monitoring is active)
    pub global_metrics: Option<PerformanceMetrics>,
    /// Per-test performance metrics
    pub test_metrics: HashMap<String, PerformanceMetrics>,
    /// Number of active tests being monitored
    pub active_test_count: u32,
    /// Whether global monitoring is active
    pub global_monitoring_active: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            global_sampling_interval: Duration::from_millis(100),
            test_sampling_interval: Duration::from_millis(50),
            max_samples_per_test: 1000,
            max_global_samples: 10000,
            enable_alerts: true,
            alert_thresholds: AlertThresholds::default(),
            enable_trending: true,
            trend_window: Duration::from_minutes(5),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            memory_threshold_percent: 90.0,
            cpu_threshold_percent: 95.0,
            latency_threshold_ms: 1000.0,
            io_rate_threshold_bps: 100 * 1024 * 1024,
            network_rate_threshold_bps: 10 * 1024 * 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::TestConfig;

    #[tokio::test]
    async fn test_monitor_creation() {
        let config = TestConfig::default();
        let monitor = PerformanceMonitor::new(&config).unwrap();
        
        // Should be created successfully
        assert!(!monitor.state.lock().unwrap().global_active);
    }

    #[tokio::test]
    async fn test_global_monitoring_lifecycle() {
        let config = TestConfig::default();
        let monitor = PerformanceMonitor::new(&config).unwrap();
        
        // Start global monitoring
        monitor.start_global_monitoring().await.unwrap();
        assert!(monitor.state.lock().unwrap().global_active);
        
        // Stop global monitoring
        let samples = monitor.stop_global_monitoring().await.unwrap();
        assert!(!monitor.state.lock().unwrap().global_active);
        assert!(samples.is_empty()); // No samples yet since it was very brief
    }

    #[tokio::test]
    async fn test_test_monitoring_lifecycle() {
        let config = TestConfig::default();
        let monitor = PerformanceMonitor::new(&config).unwrap();
        
        let test_name = "test_example";
        
        // Start test monitoring
        monitor.start_test_monitoring(test_name).await.unwrap();
        assert_eq!(monitor.state.lock().unwrap().active_test_count, 1);
        
        // Record some metrics
        monitor.record_latency(test_name, Duration::from_millis(5)).await.unwrap();
        monitor.record_custom_metric(test_name, "custom".to_string(), 42.0).await.unwrap();
        
        // Stop test monitoring
        let metrics = monitor.stop_test_monitoring(test_name).await.unwrap();
        assert_eq!(monitor.state.lock().unwrap().active_test_count, 0);
        assert!(!metrics.is_empty());
    }

    #[tokio::test]
    async fn test_performance_snapshot() {
        let config = TestConfig::default();
        let monitor = PerformanceMonitor::new(&config).unwrap();
        
        monitor.start_test_monitoring("test1").await.unwrap();
        monitor.start_test_monitoring("test2").await.unwrap();
        
        let snapshot = monitor.get_performance_snapshot().await.unwrap();
        assert_eq!(snapshot.active_test_count, 2);
        assert_eq!(snapshot.test_metrics.len(), 2);
        assert!(snapshot.test_metrics.contains_key("test1"));
        assert!(snapshot.test_metrics.contains_key("test2"));
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let config = TestConfig::default();
        let monitor = PerformanceMonitor::new(&config).unwrap();
        
        let mut receiver = monitor.subscribe_to_events();
        
        // Start monitoring should trigger event
        monitor.start_test_monitoring("test").await.unwrap();
        
        // Should receive test started event
        let event = receiver.recv().await.unwrap();
        match event {
            MonitorEvent::TestStarted { test_name, .. } => {
                assert_eq!(test_name, "test");
            }
            _ => panic!("Expected TestStarted event"),
        }
    }

    #[test]
    fn test_performance_grade_calculation() {
        let config = TestConfig::default();
        let monitor = PerformanceMonitor::new(&config).unwrap();
        
        // Test excellent performance
        let excellent_metrics = monitor.create_metrics_with_values(
            Duration::from_millis(1),
            50 * 1024 * 1024, // 50MB
            20.0,
        );
        assert!(matches!(monitor.calculate_performance_grade(&excellent_metrics), PerformanceGrade::Excellent));
        
        // Test poor performance
        let poor_metrics = monitor.create_metrics_with_values(
            Duration::from_millis(50),
            300 * 1024 * 1024, // 300MB
            80.0,
        );
        assert!(matches!(monitor.calculate_performance_grade(&poor_metrics), PerformanceGrade::Poor));
    }

    #[test]
    fn test_alert_creation() {
        let alert = Alert {
            id: "test_alert".to_string(),
            alert_type: AlertType::HighMemoryUsage,
            level: AlertLevel::Warning,
            test_name: Some("test".to_string()),
            message: "Test alert message".to_string(),
            timestamp: SystemTime::now(),
            trigger_values: HashMap::new(),
            active: true,
        };
        
        assert_eq!(alert.id, "test_alert");
        assert!(matches!(alert.alert_type, AlertType::HighMemoryUsage));
        assert!(matches!(alert.level, AlertLevel::Warning));
        assert!(alert.active);
    }
}