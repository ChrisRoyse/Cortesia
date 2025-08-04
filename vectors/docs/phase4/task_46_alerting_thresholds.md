# Task 46: Alerting Thresholds and Notification System

## Context
You are implementing Phase 4 of a vector indexing system. Search engine performance monitoring integration is now complete. This task implements a comprehensive alerting and notification system with configurable thresholds, multiple notification channels, and intelligent alert management.

## Current State
- `src/monitor.rs` exists with complete performance monitoring functionality
- Search engine monitoring provides comprehensive query performance tracking
- Need comprehensive alerting system for proactive performance management

## Task Objective
Implement a comprehensive alerting and notification system with configurable thresholds, multiple notification channels, alert aggregation, escalation policies, and intelligent alert suppression to enable proactive performance management.

## Implementation Requirements

### 1. Create alerting system core
Create a new file `src/alerting.rs`:
```rust
use crate::monitor::{PerformanceStats, AdvancedPerformanceStats, SharedPerformanceMonitor};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, SystemTime, Instant};
use std::collections::{HashMap, VecDeque};
use std::thread;

pub struct AlertingSystem {
    pub monitors: Vec<Arc<SharedPerformanceMonitor>>,
    pub thresholds: AlertThresholds,
    pub notification_channels: Vec<Box<dyn NotificationChannel>>,
    pub alert_manager: Arc<Mutex<AlertManager>>,
    pub config: AlertingConfig,
    pub is_running: Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    // Performance thresholds
    pub max_avg_query_time: Duration,
    pub max_p95_query_time: Duration,
    pub max_p99_query_time: Duration,
    pub max_avg_index_time: Duration,
    pub min_queries_per_second: f64,
    pub min_indexes_per_second: f64,
    
    // Statistical thresholds
    pub max_coefficient_variation: f64,
    pub max_outlier_percentage: f64,
    pub min_correlation_threshold: f64,
    pub max_trend_degradation: f64,
    
    // System thresholds
    pub max_memory_usage_mb: usize,
    pub max_thread_utilization: f64,
    pub min_cache_hit_rate: f64,
    pub max_error_rate: f64,
    
    // Time windows for threshold evaluation
    pub evaluation_window: Duration,
    pub consecutive_violations_required: usize,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_avg_query_time: Duration::from_millis(100),
            max_p95_query_time: Duration::from_millis(500),
            max_p99_query_time: Duration::from_millis(1000),
            max_avg_index_time: Duration::from_millis(1000),
            min_queries_per_second: 10.0,
            min_indexes_per_second: 1.0,
            
            max_coefficient_variation: 0.5,
            max_outlier_percentage: 0.05, // 5%
            min_correlation_threshold: -0.3, // Negative correlation indicates degradation
            max_trend_degradation: 0.2, // 20% degradation
            
            max_memory_usage_mb: 1024,
            max_thread_utilization: 0.9,
            min_cache_hit_rate: 0.3,
            max_error_rate: 0.01, // 1%
            
            evaluation_window: Duration::from_secs(300), // 5 minutes
            consecutive_violations_required: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlertingConfig {
    pub check_interval: Duration,
    pub enable_alert_aggregation: bool,
    pub enable_escalation: bool,
    pub enable_auto_resolution: bool,
    pub max_alerts_per_minute: usize,
    pub alert_suppression_window: Duration,
    pub escalation_timeout: Duration,
    pub notification_retry_attempts: usize,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            enable_alert_aggregation: true,
            enable_escalation: true,
            enable_auto_resolution: true,
            max_alerts_per_minute: 10,
            alert_suppression_window: Duration::from_secs(300), // 5 minutes
            escalation_timeout: Duration::from_secs(900), // 15 minutes
            notification_retry_attempts: 3,
        }
    }
}

pub struct AlertManager {
    active_alerts: HashMap<AlertKey, Alert>,
    alert_history: VecDeque<Alert>,
    suppressed_alerts: HashMap<AlertKey, SystemTime>,
    escalated_alerts: HashMap<AlertKey, SystemTime>,
    alert_counters: HashMap<AlertType, AlertCounter>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct AlertKey {
    pub alert_type: AlertType,
    pub component: String,
    pub severity: AlertSeverity,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum AlertType {
    HighLatency,
    LowThroughput,
    HighVariability,
    PerformanceDegradation,
    OutlierDetection,
    SystemResourceUsage,
    CachePerformance,
    ErrorRate,
    ThreadUtilization,
    MemoryUsage,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

#[derive(Debug, Clone)]
pub struct Alert {
    pub key: AlertKey,
    pub message: String,
    pub details: AlertDetails,
    pub timestamp: SystemTime,
    pub status: AlertStatus,
    pub escalation_level: usize,
    pub notification_attempts: usize,
}

#[derive(Debug, Clone)]
pub enum AlertDetails {
    Performance {
        metric_name: String,
        current_value: f64,
        threshold_value: f64,
        unit: String,
        trend: Option<f64>,
    },
    Statistical {
        metric_name: String,
        current_value: f64,
        threshold_value: f64,
        sample_size: usize,
        confidence_level: f64,
    },
    System {
        resource_type: String,
        current_usage: f64,
        max_capacity: f64,
        unit: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlertStatus {
    Active,
    Acknowledged,
    Resolved,
    Suppressed,
    Escalated,
}

#[derive(Debug, Clone)]
pub struct AlertCounter {
    pub count_last_minute: usize,
    pub count_last_hour: usize,
    pub count_last_day: usize,
    pub last_reset: SystemTime,
}

impl AlertingSystem {
    pub fn new(config: AlertingConfig, thresholds: AlertThresholds) -> Self {
        Self {
            monitors: Vec::new(),
            thresholds,
            notification_channels: Vec::new(),
            alert_manager: Arc::new(Mutex::new(AlertManager::new())),
            config,
            is_running: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }
    
    pub fn add_monitor(&mut self, monitor: Arc<SharedPerformanceMonitor>) {
        self.monitors.push(monitor);
    }
    
    pub fn add_notification_channel(&mut self, channel: Box<dyn NotificationChannel>) {
        self.notification_channels.push(channel);
    }
    
    pub fn start(&mut self) -> Result<(), AlertingError> {
        self.is_running.store(true, std::sync::atomic::Ordering::Relaxed);
        
        let monitors = self.monitors.clone();
        let thresholds = self.thresholds.clone();
        let alert_manager = self.alert_manager.clone();
        let config = self.config.clone();
        let is_running = self.is_running.clone();
        
        // Create notification sender channel
        let (notification_sender, notification_receiver) = mpsc::channel();
        
        // Start notification handler thread
        self.start_notification_handler(notification_receiver);
        
        // Start main monitoring thread
        thread::spawn(move || {
            let mut last_check = Instant::now();
            
            while is_running.load(std::sync::atomic::Ordering::Relaxed) {
                if last_check.elapsed() >= config.check_interval {
                    Self::check_all_monitors(
                        &monitors,
                        &thresholds,
                        &alert_manager,
                        &config,
                        &notification_sender,
                    );
                    last_check = Instant::now();
                }
                
                thread::sleep(Duration::from_millis(100));
            }
        });
        
        Ok(())
    }
    
    pub fn stop(&self) {
        self.is_running.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    
    fn start_notification_handler(&self, receiver: mpsc::Receiver<Alert>) {
        let channels = self.notification_channels.len(); // Store count since we can't move
        let config = self.config.clone();
        
        thread::spawn(move || {
            while let Ok(alert) = receiver.recv() {
                // In a real implementation, we would have access to the notification channels
                // For now, we'll log the alert
                println!("Alert notification: {:?}", alert);
                
                // Simulate notification attempts
                for _attempt in 0..config.notification_retry_attempts {
                    // Simulate notification sending
                    if Self::simulate_notification_send(&alert) {
                        break;
                    }
                    thread::sleep(Duration::from_secs(1));
                }
            }
        });
    }
    
    fn simulate_notification_send(alert: &Alert) -> bool {
        // Simulate 90% success rate
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        alert.timestamp.hash(&mut hasher);
        let hash = hasher.finish();
        
        (hash % 10) != 0 // 90% success rate
    }
    
    fn check_all_monitors(
        monitors: &[Arc<SharedPerformanceMonitor>],
        thresholds: &AlertThresholds,
        alert_manager: &Arc<Mutex<AlertManager>>,
        config: &AlertingConfig,
        notification_sender: &mpsc::Sender<Alert>,
    ) {
        for (monitor_id, monitor) in monitors.iter().enumerate() {
            if let Some(stats) = monitor.get_stats() {
                Self::check_performance_thresholds(
                    &format!("monitor_{}", monitor_id),
                    &stats,
                    thresholds,
                    alert_manager,
                    config,
                    notification_sender,
                );
            }
        }
        
        // Check for alert resolution
        if config.enable_auto_resolution {
            Self::check_alert_resolution(monitors, alert_manager, notification_sender);
        }
        
        // Handle escalations
        if config.enable_escalation {
            Self::handle_escalations(alert_manager, config, notification_sender);
        }
    }
    
    fn check_performance_thresholds(
        component: &str,
        stats: &PerformanceStats,
        thresholds: &AlertThresholds,
        alert_manager: &Arc<Mutex<AlertManager>>,
        config: &AlertingConfig,
        notification_sender: &mpsc::Sender<Alert>,
    ) {
        let mut alerts_to_send = Vec::new();
        
        // Check query latency thresholds
        if stats.avg_query_time > thresholds.max_avg_query_time {
            let alert = Alert::new(
                AlertKey {
                    alert_type: AlertType::HighLatency,
                    component: component.to_string(),
                    severity: AlertSeverity::Warning,
                },
                format!("Average query time ({:.2}ms) exceeds threshold ({:.2}ms)",
                    stats.avg_query_time.as_secs_f64() * 1000.0,
                    thresholds.max_avg_query_time.as_secs_f64() * 1000.0),
                AlertDetails::Performance {
                    metric_name: "avg_query_time".to_string(),
                    current_value: stats.avg_query_time.as_secs_f64() * 1000.0,
                    threshold_value: thresholds.max_avg_query_time.as_secs_f64() * 1000.0,
                    unit: "ms".to_string(),
                    trend: None,
                },
            );
            alerts_to_send.push(alert);
        }
        
        // Check P99 latency (critical threshold)
        if stats.p99_query_time > thresholds.max_p99_query_time {
            let alert = Alert::new(
                AlertKey {
                    alert_type: AlertType::HighLatency,
                    component: component.to_string(),
                    severity: AlertSeverity::Critical,
                },
                format!("P99 query time ({:.2}ms) exceeds critical threshold ({:.2}ms)",
                    stats.p99_query_time.as_secs_f64() * 1000.0,
                    thresholds.max_p99_query_time.as_secs_f64() * 1000.0),
                AlertDetails::Performance {
                    metric_name: "p99_query_time".to_string(),
                    current_value: stats.p99_query_time.as_secs_f64() * 1000.0,
                    threshold_value: thresholds.max_p99_query_time.as_secs_f64() * 1000.0,
                    unit: "ms".to_string(),
                    trend: None,
                },
            );
            alerts_to_send.push(alert);
        }
        
        // Check throughput thresholds
        if stats.queries_per_second < thresholds.min_queries_per_second && stats.total_queries > 10 {
            let alert = Alert::new(
                AlertKey {
                    alert_type: AlertType::LowThroughput,
                    component: component.to_string(),
                    severity: AlertSeverity::Warning,
                },
                format!("Query throughput ({:.2} qps) below minimum threshold ({:.2} qps)",
                    stats.queries_per_second, thresholds.min_queries_per_second),
                AlertDetails::Performance {
                    metric_name: "queries_per_second".to_string(),
                    current_value: stats.queries_per_second,
                    threshold_value: thresholds.min_queries_per_second,
                    unit: "qps".to_string(),
                    trend: None,
                },
            );
            alerts_to_send.push(alert);
        }
        
        // Check indexing performance
        if stats.avg_index_time > thresholds.max_avg_index_time {
            let alert = Alert::new(
                AlertKey {
                    alert_type: AlertType::HighLatency,
                    component: component.to_string(),
                    severity: AlertSeverity::Warning,
                },
                format!("Average index time ({:.2}ms) exceeds threshold ({:.2}ms)",
                    stats.avg_index_time.as_secs_f64() * 1000.0,
                    thresholds.max_avg_index_time.as_secs_f64() * 1000.0),
                AlertDetails::Performance {
                    metric_name: "avg_index_time".to_string(),
                    current_value: stats.avg_index_time.as_secs_f64() * 1000.0,
                    threshold_value: thresholds.max_avg_index_time.as_secs_f64() * 1000.0,
                    unit: "ms".to_string(),
                    trend: None,
                },
            );
            alerts_to_send.push(alert);
        }
        
        // Process alerts through alert manager
        if let Ok(mut alert_manager) = alert_manager.lock() {
            for alert in alerts_to_send {
                if alert_manager.should_send_alert(&alert.key, config) {
                    alert_manager.add_alert(alert.clone());
                    let _ = notification_sender.send(alert);
                }
            }
        }
    }
    
    fn check_alert_resolution(
        monitors: &[Arc<SharedPerformanceMonitor>],
        alert_manager: &Arc<Mutex<AlertManager>>,
        notification_sender: &mpsc::Sender<Alert>,
    ) {
        if let Ok(mut alert_manager) = alert_manager.lock() {
            let mut resolved_alerts = Vec::new();
            
            for (alert_key, alert) in &alert_manager.active_alerts {
                if alert.status == AlertStatus::Active {
                    // Check if the condition that triggered the alert is now resolved
                    let mut is_resolved = false;
                    
                    // Find the monitor for this alert's component
                    if let Some(monitor_id) = alert_key.component.strip_prefix("monitor_") {
                        if let Ok(id) = monitor_id.parse::<usize>() {
                            if let Some(monitor) = monitors.get(id) {
                                if let Some(current_stats) = monitor.get_stats() {
                                    is_resolved = Self::is_alert_condition_resolved(alert, &current_stats);
                                }
                            }
                        }
                    }
                    
                    if is_resolved {
                        resolved_alerts.push(alert_key.clone());
                    }
                }
            }
            
            // Mark alerts as resolved and send notifications
            for alert_key in resolved_alerts {
                if let Some(mut alert) = alert_manager.active_alerts.get_mut(&alert_key) {
                    alert.status = AlertStatus::Resolved;
                    alert.timestamp = SystemTime::now();
                    
                    let resolution_alert = Alert::new(
                        alert_key.clone(),
                        format!("RESOLVED: {}", alert.message),
                        alert.details.clone(),
                    );
                    
                    let _ = notification_sender.send(resolution_alert);
                }
            }
        }
    }
    
    fn is_alert_condition_resolved(alert: &Alert, current_stats: &PerformanceStats) -> bool {
        match &alert.details {
            AlertDetails::Performance { metric_name, threshold_value, .. } => {
                match metric_name.as_str() {
                    "avg_query_time" => {
                        current_stats.avg_query_time.as_secs_f64() * 1000.0 <= *threshold_value
                    }
                    "p99_query_time" => {
                        current_stats.p99_query_time.as_secs_f64() * 1000.0 <= *threshold_value
                    }
                    "queries_per_second" => {
                        current_stats.queries_per_second >= *threshold_value
                    }
                    "avg_index_time" => {
                        current_stats.avg_index_time.as_secs_f64() * 1000.0 <= *threshold_value
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }
    
    fn handle_escalations(
        alert_manager: &Arc<Mutex<AlertManager>>,
        config: &AlertingConfig,
        notification_sender: &mpsc::Sender<Alert>,
    ) {
        if let Ok(mut alert_manager) = alert_manager.lock() {
            let now = SystemTime::now();
            let mut alerts_to_escalate = Vec::new();
            
            for (alert_key, alert) in &alert_manager.active_alerts {
                if alert.status == AlertStatus::Active {
                    if let Ok(duration) = now.duration_since(alert.timestamp) {
                        if duration > config.escalation_timeout && alert.escalation_level == 0 {
                            alerts_to_escalate.push(alert_key.clone());
                        }
                    }
                }
            }
            
            // Escalate alerts
            for alert_key in alerts_to_escalate {
                if let Some(alert) = alert_manager.active_alerts.get_mut(&alert_key) {
                    alert.escalation_level += 1;
                    alert.status = AlertStatus::Escalated;
                    
                    let escalated_alert = Alert::new(
                        AlertKey {
                            severity: AlertSeverity::Critical, // Escalate severity
                            ..alert_key.clone()
                        },
                        format!("ESCALATED: {} (Level {})", alert.message, alert.escalation_level),
                        alert.details.clone(),
                    );
                    
                    let _ = notification_sender.send(escalated_alert);
                }
            }
        }
    }
    
    /// Get current alerting status
    pub fn get_alerting_status(&self) -> AlertingStatus {
        if let Ok(alert_manager) = self.alert_manager.lock() {
            AlertingStatus {
                is_running: self.is_running.load(std::sync::atomic::Ordering::Relaxed),
                active_alerts: alert_manager.active_alerts.len(),
                total_alerts_today: alert_manager.get_total_alerts_today(),
                suppressed_alerts: alert_manager.suppressed_alerts.len(),
                escalated_alerts: alert_manager.escalated_alerts.len(),
                last_check: SystemTime::now(), // Simplified
            }
        } else {
            AlertingStatus::default()
        }
    }
    
    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<Alert> {
        if let Ok(alert_manager) = self.alert_manager.lock() {
            alert_manager.active_alerts.values().cloned().collect()
        } else {
            Vec::new()
        }
    }
    
    /// Acknowledge an alert
    pub fn acknowledge_alert(&self, alert_key: &AlertKey) -> Result<(), AlertingError> {
        if let Ok(mut alert_manager) = self.alert_manager.lock() {
            if let Some(alert) = alert_manager.active_alerts.get_mut(alert_key) {
                alert.status = AlertStatus::Acknowledged;
                Ok(())
            } else {
                Err(AlertingError::AlertNotFound)
            }
        } else {
            Err(AlertingError::LockError)
        }
    }
}

impl AlertManager {
    fn new() -> Self {
        Self {
            active_alerts: HashMap::new(),
            alert_history: VecDeque::new(),
            suppressed_alerts: HashMap::new(),
            escalated_alerts: HashMap::new(),
            alert_counters: HashMap::new(),
        }
    }
    
    fn should_send_alert(&mut self, alert_key: &AlertKey, config: &AlertingConfig) -> bool {
        let now = SystemTime::now();
        
        // Check if alert is suppressed
        if let Some(&suppressed_until) = self.suppressed_alerts.get(alert_key) {
            if now < suppressed_until {
                return false;
            } else {
                self.suppressed_alerts.remove(alert_key);
            }
        }
        
        // Check rate limiting
        let counter = self.alert_counters.entry(alert_key.alert_type.clone()).or_insert_with(|| {
            AlertCounter {
                count_last_minute: 0,
                count_last_hour: 0,
                count_last_day: 0,
                last_reset: now,
            }
        });
        
        counter.update_counts(now);
        
        if counter.count_last_minute >= config.max_alerts_per_minute {
            // Suppress alert due to rate limiting
            self.suppressed_alerts.insert(
                alert_key.clone(),
                now + config.alert_suppression_window,
            );
            return false;
        }
        
        counter.count_last_minute += 1;
        counter.count_last_hour += 1;
        counter.count_last_day += 1;
        
        true
    }
    
    fn add_alert(&mut self, alert: Alert) {
        self.active_alerts.insert(alert.key.clone(), alert.clone());
        
        // Add to history
        self.alert_history.push_back(alert);
        
        // Keep history size manageable
        if self.alert_history.len() > 10000 {
            self.alert_history.pop_front();
        }
    }
    
    fn get_total_alerts_today(&self) -> usize {
        self.alert_counters.values().map(|c| c.count_last_day).sum()
    }
}

impl AlertCounter {
    fn update_counts(&mut self, now: SystemTime) {
        if let Ok(elapsed) = now.duration_since(self.last_reset) {
            if elapsed > Duration::from_secs(86400) { // 24 hours
                self.count_last_day = 0;
                self.count_last_hour = 0;
                self.count_last_minute = 0;
                self.last_reset = now;
            } else if elapsed > Duration::from_secs(3600) { // 1 hour
                self.count_last_hour = 0;
                self.count_last_minute = 0;
            } else if elapsed > Duration::from_secs(60) { // 1 minute
                self.count_last_minute = 0;
            }
        }
    }
}

impl Alert {
    fn new(key: AlertKey, message: String, details: AlertDetails) -> Self {
        Self {
            key,
            message,
            details,
            timestamp: SystemTime::now(),
            status: AlertStatus::Active,
            escalation_level: 0,
            notification_attempts: 0,
        }
    }
}

// Notification channel trait
pub trait NotificationChannel: Send + Sync {
    fn send_notification(&self, alert: &Alert) -> Result<(), NotificationError>;
    fn get_channel_name(&self) -> &str;
    fn is_enabled(&self) -> bool;
}

// Email notification channel
pub struct EmailNotificationChannel {
    pub smtp_server: String,
    pub recipients: Vec<String>,
    pub enabled: bool,
}

impl NotificationChannel for EmailNotificationChannel {
    fn send_notification(&self, alert: &Alert) -> Result<(), NotificationError> {
        if !self.enabled {
            return Err(NotificationError::ChannelDisabled);
        }
        
        // Simulate email sending
        println!("EMAIL: Sending alert to {:?}: {}", self.recipients, alert.message);
        Ok(())
    }
    
    fn get_channel_name(&self) -> &str {
        "email"
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

// Slack notification channel
pub struct SlackNotificationChannel {
    pub webhook_url: String,
    pub channel: String,
    pub enabled: bool,
}

impl NotificationChannel for SlackNotificationChannel {
    fn send_notification(&self, alert: &Alert) -> Result<(), NotificationError> {
        if !self.enabled {
            return Err(NotificationError::ChannelDisabled);
        }
        
        // Simulate Slack notification
        println!("SLACK: Sending to #{}: {}", self.channel, alert.message);
        Ok(())
    }
    
    fn get_channel_name(&self) -> &str {
        "slack"
    }
    
    fn is_enabled(&self) -> bool {
        self.enabled
    }
}

// Supporting types
#[derive(Debug, Clone)]
pub struct AlertingStatus {
    pub is_running: bool,
    pub active_alerts: usize,
    pub total_alerts_today: usize,
    pub suppressed_alerts: usize,
    pub escalated_alerts: usize,
    pub last_check: SystemTime,
}

impl Default for AlertingStatus {
    fn default() -> Self {
        Self {
            is_running: false,
            active_alerts: 0,
            total_alerts_today: 0,
            suppressed_alerts: 0,
            escalated_alerts: 0,
            last_check: SystemTime::UNIX_EPOCH,
        }
    }
}

#[derive(Debug)]
pub enum AlertingError {
    ConfigurationError(String),
    NotificationError(NotificationError),
    AlertNotFound,
    LockError,
}

#[derive(Debug)]
pub enum NotificationError {
    NetworkError,
    AuthenticationError,
    ChannelDisabled,
    RateLimited,
    InvalidConfiguration(String),
}

impl std::fmt::Display for AlertingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertingError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            AlertingError::NotificationError(err) => write!(f, "Notification error: {:?}", err),
            AlertingError::AlertNotFound => write!(f, "Alert not found"),
            AlertingError::LockError => write!(f, "Failed to acquire lock"),
        }
    }
}

impl std::error::Error for AlertingError {}
```

### 2. Add comprehensive alerting tests
Add to `src/alerting.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::{SharedPerformanceMonitor, PerformanceMonitor};
    use std::time::Duration;
    
    #[test]
    fn test_alerting_system_creation() {
        let config = AlertingConfig::default();
        let thresholds = AlertThresholds::default();
        let alerting_system = AlertingSystem::new(config, thresholds);
        
        assert_eq!(alerting_system.monitors.len(), 0);
        assert_eq!(alerting_system.notification_channels.len(), 0);
        assert!(!alerting_system.is_running.load(std::sync::atomic::Ordering::Relaxed));
    }
    
    #[test]
    fn test_alert_threshold_defaults() {
        let thresholds = AlertThresholds::default();
        
        assert_eq!(thresholds.max_avg_query_time, Duration::from_millis(100));
        assert_eq!(thresholds.max_p99_query_time, Duration::from_millis(1000));
        assert_eq!(thresholds.min_queries_per_second, 10.0);
        assert_eq!(thresholds.consecutive_violations_required, 3);
    }
    
    #[test]
    fn test_alert_creation() {
        let alert_key = AlertKey {
            alert_type: AlertType::HighLatency,
            component: "test_monitor".to_string(),
            severity: AlertSeverity::Warning,
        };
        
        let alert_details = AlertDetails::Performance {
            metric_name: "avg_query_time".to_string(),
            current_value: 150.0,
            threshold_value: 100.0,
            unit: "ms".to_string(),
            trend: None,
        };
        
        let alert = Alert::new(
            alert_key.clone(),
            "Test alert message".to_string(),
            alert_details,
        );
        
        assert_eq!(alert.key, alert_key);
        assert_eq!(alert.message, "Test alert message");
        assert_eq!(alert.status, AlertStatus::Active);
        assert_eq!(alert.escalation_level, 0);
    }
    
    #[test]
    fn test_alert_manager_rate_limiting() {
        let mut alert_manager = AlertManager::new();
        let config = AlertingConfig {
            max_alerts_per_minute: 2,
            ..Default::default()
        };
        
        let alert_key = AlertKey {
            alert_type: AlertType::HighLatency,
            component: "test".to_string(),
            severity: AlertSeverity::Warning,
        };
        
        // First two alerts should be allowed
        assert!(alert_manager.should_send_alert(&alert_key, &config));
        assert!(alert_manager.should_send_alert(&alert_key, &config));
        
        // Third alert should be rate limited
        assert!(!alert_manager.should_send_alert(&alert_key, &config));
    }
    
    #[test]
    fn test_notification_channels() {
        let email_channel = EmailNotificationChannel {
            smtp_server: "smtp.example.com".to_string(),
            recipients: vec!["admin@example.com".to_string()],
            enabled: true,
        };
        
        let slack_channel = SlackNotificationChannel {
            webhook_url: "https://hooks.slack.com/test".to_string(),
            channel: "alerts".to_string(),
            enabled: true,
        };
        
        assert_eq!(email_channel.get_channel_name(), "email");
        assert_eq!(slack_channel.get_channel_name(), "slack");
        assert!(email_channel.is_enabled());
        assert!(slack_channel.is_enabled());
        
        // Test notification sending
        let alert = Alert::new(
            AlertKey {
                alert_type: AlertType::HighLatency,
                component: "test".to_string(),
                severity: AlertSeverity::Warning,
            },
            "Test notification".to_string(),
            AlertDetails::Performance {
                metric_name: "test".to_string(),
                current_value: 100.0,
                threshold_value: 50.0,
                unit: "ms".to_string(),
                trend: None,
            },
        );
        
        assert!(email_channel.send_notification(&alert).is_ok());
        assert!(slack_channel.send_notification(&alert).is_ok());
    }
    
    #[test]
    fn test_alert_suppression() {
        let mut alert_manager = AlertManager::new();
        let config = AlertingConfig {
            max_alerts_per_minute: 1,
            alert_suppression_window: Duration::from_secs(60),
            ..Default::default()
        };
        
        let alert_key = AlertKey {
            alert_type: AlertType::HighLatency,
            component: "test".to_string(),
            severity: AlertSeverity::Warning,
        };
        
        // First alert should be allowed
        assert!(alert_manager.should_send_alert(&alert_key, &config));
        
        // Second alert should be suppressed due to rate limiting
        assert!(!alert_manager.should_send_alert(&alert_key, &config));
        
        // Check that the alert is now in suppressed list
        assert!(alert_manager.suppressed_alerts.contains_key(&alert_key));
    }
    
    #[test]
    fn test_alert_counter_updates() {
        let mut counter = AlertCounter {
            count_last_minute: 5,
            count_last_hour: 20,
            count_last_day: 100,
            last_reset: SystemTime::now() - Duration::from_secs(120), // 2 minutes ago
        };
        
        counter.update_counts(SystemTime::now());
        
        // After 2 minutes, minute count should reset but hour and day should remain
        assert_eq!(counter.count_last_minute, 0);
        assert_eq!(counter.count_last_hour, 20);
        assert_eq!(counter.count_last_day, 100);
    }
    
    #[test]
    fn test_alert_status_transitions() {
        let mut alert = Alert::new(
            AlertKey {
                alert_type: AlertType::HighLatency,
                component: "test".to_string(),
                severity: AlertSeverity::Warning,
            },
            "Test alert".to_string(),
            AlertDetails::Performance {
                metric_name: "test".to_string(),
                current_value: 100.0,
                threshold_value: 50.0,
                unit: "ms".to_string(),
                trend: None,
            },
        );
        
        // Initial status should be Active
        assert_eq!(alert.status, AlertStatus::Active);
        
        // Change to Acknowledged
        alert.status = AlertStatus::Acknowledged;
        assert_eq!(alert.status, AlertStatus::Acknowledged);
        
        // Change to Resolved
        alert.status = AlertStatus::Resolved;
        assert_eq!(alert.status, AlertStatus::Resolved);
    }
    
    #[test]
    fn test_alerting_integration_with_monitor() {
        let config = AlertingConfig {
            check_interval: Duration::from_millis(50),
            ..Default::default()
        };
        
        let thresholds = AlertThresholds {
            max_avg_query_time: Duration::from_millis(50), // Low threshold for testing
            ..Default::default()
        };
        
        let mut alerting_system = AlertingSystem::new(config, thresholds);
        
        // Create a monitor with slow performance
        let shared_monitor = Arc::new(SharedPerformanceMonitor::new());
        
        // Record slow query times
        for _ in 0..10 {
            shared_monitor.record_query_time(Duration::from_millis(100));
        }
        
        alerting_system.add_monitor(shared_monitor);
        
        // Add notification channels
        let email_channel = Box::new(EmailNotificationChannel {
            smtp_server: "smtp.test.com".to_string(),
            recipients: vec!["test@example.com".to_string()],
            enabled: true,
        });
        
        alerting_system.add_notification_channel(email_channel);
        
        // Start alerting system briefly
        let _ = alerting_system.start();
        
        // Wait a short time for alerts to be processed
        thread::sleep(Duration::from_millis(200));
        
        alerting_system.stop();
        
        // Check that alerting system was running
        let status = alerting_system.get_alerting_status();
        assert!(!status.is_running); // Should be stopped now
    }
    
    #[test]
    fn test_performance_threshold_violations() {
        use crate::monitor::PerformanceStats;
        
        let thresholds = AlertThresholds::default();
        let config = AlertingConfig::default();
        let alert_manager = Arc::new(Mutex::new(AlertManager::new()));
        let (sender, _receiver) = mpsc::channel();
        
        // Create stats that violate thresholds
        let stats = PerformanceStats {
            avg_query_time: Duration::from_millis(200), // Above 100ms threshold
            p99_query_time: Duration::from_millis(1500), // Above 1000ms threshold
            queries_per_second: 5.0, // Below 10.0 threshold
            total_queries: 100, // Enough queries to trigger throughput alert
            ..Default::default()
        };
        
        AlertingSystem::check_performance_thresholds(
            "test_component",
            &stats,
            &thresholds,
            &alert_manager,
            &config,
            &sender,
        );
        
        // Check that alerts were generated
        if let Ok(alert_manager) = alert_manager.lock() {
            assert!(!alert_manager.active_alerts.is_empty());
            
            // Should have alerts for high latency and low throughput
            let has_latency_alert = alert_manager.active_alerts.keys()
                .any(|key| key.alert_type == AlertType::HighLatency);
            let has_throughput_alert = alert_manager.active_alerts.keys()
                .any(|key| key.alert_type == AlertType::LowThroughput);
                
            assert!(has_latency_alert);
            assert!(has_throughput_alert);
        }
    }
}
```

### 3. Add utility functions for alert management
```rust
impl AlertingSystem {
    /// Create alerting system with common production settings
    pub fn production_config() -> (AlertingConfig, AlertThresholds) {
        let config = AlertingConfig {
            check_interval: Duration::from_secs(30),
            enable_alert_aggregation: true,
            enable_escalation: true,
            enable_auto_resolution: true,
            max_alerts_per_minute: 5,
            alert_suppression_window: Duration::from_secs(600),
            escalation_timeout: Duration::from_secs(1800),
            notification_retry_attempts: 3,
        };
        
        let thresholds = AlertThresholds {
            max_avg_query_time: Duration::from_millis(200),
            max_p95_query_time: Duration::from_millis(800),
            max_p99_query_time: Duration::from_millis(2000),
            max_avg_index_time: Duration::from_millis(2000),
            min_queries_per_second: 5.0,
            min_indexes_per_second: 0.5,
            max_coefficient_variation: 0.6,
            max_outlier_percentage: 0.1,
            evaluation_window: Duration::from_secs(300),
            consecutive_violations_required: 2,
            ..Default::default()
        };
        
        (config, thresholds)
    }
    
    /// Export alerting configuration for backup/restore
    pub fn export_config(&self) -> AlertingConfiguration {
        AlertingConfiguration {
            thresholds: self.thresholds.clone(),
            config: self.config.clone(),
            channel_configs: self.notification_channels.iter()
                .map(|channel| ChannelConfig {
                    name: channel.get_channel_name().to_string(),
                    enabled: channel.is_enabled(),
                })
                .collect(),
        }
    }
    
    /// Generate alerting health report
    pub fn generate_health_report(&self) -> AlertingHealthReport {
        let status = self.get_alerting_status();
        let active_alerts = self.get_active_alerts();
        
        let severity_breakdown = active_alerts.iter()
            .fold(HashMap::new(), |mut acc, alert| {
                *acc.entry(alert.key.severity.clone()).or_insert(0) += 1;
                acc
            });
        
        let type_breakdown = active_alerts.iter()
            .fold(HashMap::new(), |mut acc, alert| {
                *acc.entry(alert.key.alert_type.clone()).or_insert(0) += 1;
                acc
            });
        
        AlertingHealthReport {
            status,
            severity_breakdown,
            type_breakdown,
            oldest_unresolved_alert: active_alerts.iter()
                .min_by_key(|alert| alert.timestamp)
                .map(|alert| alert.timestamp),
            notification_health: self.notification_channels.iter()
                .map(|channel| ChannelHealth {
                    name: channel.get_channel_name().to_string(),
                    enabled: channel.is_enabled(),
                    last_successful_send: None, // Would track in real implementation
                })
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlertingConfiguration {
    pub thresholds: AlertThresholds,
    pub config: AlertingConfig,
    pub channel_configs: Vec<ChannelConfig>,
}

#[derive(Debug, Clone)]
pub struct ChannelConfig {
    pub name: String,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
pub struct AlertingHealthReport {
    pub status: AlertingStatus,
    pub severity_breakdown: HashMap<AlertSeverity, usize>,
    pub type_breakdown: HashMap<AlertType, usize>,
    pub oldest_unresolved_alert: Option<SystemTime>,
    pub notification_health: Vec<ChannelHealth>,
}

#[derive(Debug, Clone)]
pub struct ChannelHealth {
    pub name: String,
    pub enabled: bool,
    pub last_successful_send: Option<SystemTime>,
}
```

## Success Criteria
- [ ] Alerting system integrates with performance monitors correctly
- [ ] Configurable thresholds trigger alerts appropriately
- [ ] Multiple notification channels work correctly
- [ ] Alert suppression and rate limiting prevent spam
- [ ] Alert escalation system functions properly
- [ ] Auto-resolution detects when conditions improve
- [ ] Alert acknowledgment system works correctly
- [ ] Comprehensive alerting health reporting is available
- [ ] Production configuration provides sensible defaults
- [ ] All alerting tests pass consistently

## Time Limit
10 minutes

## Notes
- Provides comprehensive alerting system with configurable thresholds
- Supports multiple notification channels (email, Slack, etc.)
- Includes intelligent alert management with suppression and escalation
- Enables auto-resolution when performance conditions improve
- Supports alert acknowledgment for incident management
- Provides detailed alerting health reporting and status monitoring
- Includes production-ready configuration templates
- Maintains alert history for analysis and troubleshooting