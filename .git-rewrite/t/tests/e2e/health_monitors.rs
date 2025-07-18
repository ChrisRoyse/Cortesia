//! Health Monitoring for E2E Simulations
//! 
//! Comprehensive health monitoring systems for tracking system health and component status
//! during end-to-end simulation testing.

use std::time::{Duration, Instant};
use std::collections::{HashMap, VecDeque};
use anyhow::Result;

/// E2E Health Monitor for tracking system health
pub struct E2EHealthMonitor {
    component_health: HashMap<String, ComponentHealthStatus>,
    system_health_history: VecDeque<SystemHealthSnapshot>,
    alert_thresholds: HealthThresholds,
    monitoring_start_time: Instant,
    health_check_interval: Duration,
}

/// Health status for individual components
#[derive(Debug, Clone)]
pub struct ComponentHealthStatus {
    pub component_name: String,
    pub is_healthy: bool,
    pub last_health_check: Instant,
    pub consecutive_failures: u32,
    pub health_score: f64, // 0.0 to 1.0
    pub response_time: Duration,
    pub resource_usage: ComponentResourceUsage,
    pub error_rate: f64,
    pub uptime_percentage: f64,
}

/// Resource usage for a component
#[derive(Debug, Clone)]
pub struct ComponentResourceUsage {
    pub cpu_percentage: f64,
    pub memory_mb: u64,
    pub disk_io_mbps: f64,
    pub network_io_mbps: f64,
    pub thread_count: u32,
    pub connection_count: u32,
}

/// System-wide health snapshot
#[derive(Debug, Clone)]
pub struct SystemHealthSnapshot {
    pub timestamp: Instant,
    pub overall_health_score: f64,
    pub healthy_components: u32,
    pub total_components: u32,
    pub system_load: f64,
    pub memory_pressure: f64,
    pub disk_pressure: f64,
    pub network_latency_ms: f64,
    pub active_alerts: u32,
    pub performance_degradation: f64,
}

/// Health monitoring metrics
#[derive(Debug, Clone)]
pub struct SystemHealthMetrics {
    pub avg_health_score: f64,
    pub min_health_score: f64,
    pub max_health_score: f64,
    pub health_score_variance: f64,
    pub total_health_checks: u32,
    pub failed_health_checks: u32,
    pub avg_response_time: Duration,
    pub uptime_percentage: f64,
    pub component_failure_rate: f64,
}

/// Health alert thresholds
#[derive(Debug, Clone)]
pub struct HealthThresholds {
    pub min_health_score: f64,
    pub max_response_time: Duration,
    pub max_error_rate: f64,
    pub max_consecutive_failures: u32,
    pub min_uptime_percentage: f64,
    pub max_memory_usage_mb: u64,
    pub max_cpu_percentage: f64,
}

/// Health alert
#[derive(Debug, Clone)]
pub struct HealthAlert {
    pub alert_id: String,
    pub component_name: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: Instant,
    pub threshold_violated: String,
    pub current_value: String,
}

/// Types of health alerts
#[derive(Debug, Clone)]
pub enum AlertType {
    ComponentDown,
    HighResponseTime,
    HighErrorRate,
    ResourceExhaustion,
    ConsecutiveFailures,
    LowUptime,
    PerformanceDegradation,
}

/// Alert severity levels
#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

impl E2EHealthMonitor {
    /// Create a new health monitor
    pub fn new(health_check_interval: Duration) -> Self {
        Self {
            component_health: HashMap::new(),
            system_health_history: VecDeque::new(),
            alert_thresholds: HealthThresholds::default(),
            monitoring_start_time: Instant::now(),
            health_check_interval,
        }
    }

    /// Register a component for health monitoring
    pub fn register_component(&mut self, component_name: &str) {
        let health_status = ComponentHealthStatus {
            component_name: component_name.to_string(),
            is_healthy: true,
            last_health_check: Instant::now(),
            consecutive_failures: 0,
            health_score: 1.0,
            response_time: Duration::from_millis(0),
            resource_usage: ComponentResourceUsage::default(),
            error_rate: 0.0,
            uptime_percentage: 100.0,
        };

        self.component_health.insert(component_name.to_string(), health_status);
    }

    /// Update component health status
    pub fn update_component_health(
        &mut self,
        component_name: &str,
        is_healthy: bool,
        response_time: Duration,
        resource_usage: ComponentResourceUsage,
        error_rate: f64
    ) -> Result<()> {
        if let Some(status) = self.component_health.get_mut(component_name) {
            let was_healthy = status.is_healthy;
            status.is_healthy = is_healthy;
            status.last_health_check = Instant::now();
            status.response_time = response_time;
            status.resource_usage = resource_usage;
            status.error_rate = error_rate;

            if is_healthy {
                status.consecutive_failures = 0;
                status.health_score = self.calculate_health_score(status);
            } else {
                status.consecutive_failures += 1;
                status.health_score = 0.0;
            }

            // Update uptime percentage
            let total_time = self.monitoring_start_time.elapsed();
            let downtime = if !is_healthy && was_healthy {
                self.health_check_interval
            } else if !is_healthy {
                status.consecutive_failures as u64 * self.health_check_interval.as_secs()
            } else {
                0
            };

            status.uptime_percentage = ((total_time.as_secs() - downtime) as f64 / total_time.as_secs() as f64) * 100.0;
        }

        Ok(())
    }

    /// Check all component health and generate alerts
    pub fn check_system_health(&mut self) -> Vec<HealthAlert> {
        let mut alerts = Vec::new();

        for (component_name, status) in &self.component_health {
            // Check various health thresholds
            if !status.is_healthy {
                alerts.push(HealthAlert {
                    alert_id: format!("component_down_{}", component_name),
                    component_name: component_name.clone(),
                    alert_type: AlertType::ComponentDown,
                    severity: AlertSeverity::Critical,
                    message: format!("Component {} is down", component_name),
                    timestamp: Instant::now(),
                    threshold_violated: "is_healthy".to_string(),
                    current_value: "false".to_string(),
                });
            }

            if status.response_time > self.alert_thresholds.max_response_time {
                alerts.push(HealthAlert {
                    alert_id: format!("high_response_time_{}", component_name),
                    component_name: component_name.clone(),
                    alert_type: AlertType::HighResponseTime,
                    severity: AlertSeverity::Warning,
                    message: format!("Component {} has high response time: {:?}", component_name, status.response_time),
                    timestamp: Instant::now(),
                    threshold_violated: format!("max_response_time ({:?})", self.alert_thresholds.max_response_time),
                    current_value: format!("{:?}", status.response_time),
                });
            }

            if status.error_rate > self.alert_thresholds.max_error_rate {
                alerts.push(HealthAlert {
                    alert_id: format!("high_error_rate_{}", component_name),
                    component_name: component_name.clone(),
                    alert_type: AlertType::HighErrorRate,
                    severity: AlertSeverity::Warning,
                    message: format!("Component {} has high error rate: {:.2}%", component_name, status.error_rate * 100.0),
                    timestamp: Instant::now(),
                    threshold_violated: format!("max_error_rate ({:.2}%)", self.alert_thresholds.max_error_rate * 100.0),
                    current_value: format!("{:.2}%", status.error_rate * 100.0),
                });
            }

            if status.consecutive_failures > self.alert_thresholds.max_consecutive_failures {
                alerts.push(HealthAlert {
                    alert_id: format!("consecutive_failures_{}", component_name),
                    component_name: component_name.clone(),
                    alert_type: AlertType::ConsecutiveFailures,
                    severity: AlertSeverity::Critical,
                    message: format!("Component {} has {} consecutive failures", component_name, status.consecutive_failures),
                    timestamp: Instant::now(),
                    threshold_violated: format!("max_consecutive_failures ({})", self.alert_thresholds.max_consecutive_failures),
                    current_value: status.consecutive_failures.to_string(),
                });
            }

            if status.uptime_percentage < self.alert_thresholds.min_uptime_percentage {
                alerts.push(HealthAlert {
                    alert_id: format!("low_uptime_{}", component_name),
                    component_name: component_name.clone(),
                    alert_type: AlertType::LowUptime,
                    severity: AlertSeverity::Critical,
                    message: format!("Component {} has low uptime: {:.2}%", component_name, status.uptime_percentage),
                    timestamp: Instant::now(),
                    threshold_violated: format!("min_uptime_percentage ({:.2}%)", self.alert_thresholds.min_uptime_percentage),
                    current_value: format!("{:.2}%", status.uptime_percentage),
                });
            }

            // Check resource usage
            if status.resource_usage.memory_mb > self.alert_thresholds.max_memory_usage_mb {
                alerts.push(HealthAlert {
                    alert_id: format!("high_memory_{}", component_name),
                    component_name: component_name.clone(),
                    alert_type: AlertType::ResourceExhaustion,
                    severity: AlertSeverity::Warning,
                    message: format!("Component {} has high memory usage: {} MB", component_name, status.resource_usage.memory_mb),
                    timestamp: Instant::now(),
                    threshold_violated: format!("max_memory_usage_mb ({})", self.alert_thresholds.max_memory_usage_mb),
                    current_value: format!("{} MB", status.resource_usage.memory_mb),
                });
            }

            if status.resource_usage.cpu_percentage > self.alert_thresholds.max_cpu_percentage {
                alerts.push(HealthAlert {
                    alert_id: format!("high_cpu_{}", component_name),
                    component_name: component_name.clone(),
                    alert_type: AlertType::ResourceExhaustion,
                    severity: AlertSeverity::Warning,
                    message: format!("Component {} has high CPU usage: {:.1}%", component_name, status.resource_usage.cpu_percentage),
                    timestamp: Instant::now(),
                    threshold_violated: format!("max_cpu_percentage ({:.1}%)", self.alert_thresholds.max_cpu_percentage),
                    current_value: format!("{:.1}%", status.resource_usage.cpu_percentage),
                });
            }
        }

        // Create system health snapshot
        let snapshot = self.create_system_health_snapshot();
        self.system_health_history.push_back(snapshot);

        // Keep only recent history (last 1000 snapshots)
        while self.system_health_history.len() > 1000 {
            self.system_health_history.pop_front();
        }

        alerts
    }

    /// Get overall system health metrics
    pub fn get_system_health_metrics(&self) -> SystemHealthMetrics {
        let health_scores: Vec<f64> = self.component_health.values()
            .map(|status| status.health_score)
            .collect();

        let avg_health_score = if health_scores.is_empty() {
            1.0
        } else {
            health_scores.iter().sum::<f64>() / health_scores.len() as f64
        };

        let min_health_score = health_scores.iter().cloned().fold(1.0, f64::min);
        let max_health_score = health_scores.iter().cloned().fold(0.0, f64::max);

        let health_score_variance = if health_scores.len() > 1 {
            let mean = avg_health_score;
            let variance = health_scores.iter()
                .map(|score| (score - mean).powi(2))
                .sum::<f64>() / health_scores.len() as f64;
            variance
        } else {
            0.0
        };

        let total_health_checks = self.component_health.len() as u32;
        let failed_health_checks = self.component_health.values()
            .filter(|status| !status.is_healthy)
            .count() as u32;

        let avg_response_time = if self.component_health.is_empty() {
            Duration::from_secs(0)
        } else {
            let total_response_time: Duration = self.component_health.values()
                .map(|status| status.response_time)
                .sum();
            total_response_time / self.component_health.len() as u32
        };

        let avg_uptime = if self.component_health.is_empty() {
            100.0
        } else {
            self.component_health.values()
                .map(|status| status.uptime_percentage)
                .sum::<f64>() / self.component_health.len() as f64
        };

        let component_failure_rate = if total_health_checks > 0 {
            failed_health_checks as f64 / total_health_checks as f64
        } else {
            0.0
        };

        SystemHealthMetrics {
            avg_health_score,
            min_health_score,
            max_health_score,
            health_score_variance,
            total_health_checks,
            failed_health_checks,
            avg_response_time,
            uptime_percentage: avg_uptime,
            component_failure_rate,
        }
    }

    /// Get component health status
    pub fn get_component_health(&self, component_name: &str) -> Option<&ComponentHealthStatus> {
        self.component_health.get(component_name)
    }

    /// Get all component health statuses
    pub fn get_all_component_health(&self) -> &HashMap<String, ComponentHealthStatus> {
        &self.component_health
    }

    /// Get system health history
    pub fn get_health_history(&self) -> &VecDeque<SystemHealthSnapshot> {
        &self.system_health_history
    }

    /// Set health alert thresholds
    pub fn set_alert_thresholds(&mut self, thresholds: HealthThresholds) {
        self.alert_thresholds = thresholds;
    }

    /// Reset all health monitoring data
    pub fn reset(&mut self) {
        self.component_health.clear();
        self.system_health_history.clear();
        self.monitoring_start_time = Instant::now();
    }

    /// Generate health report
    pub fn generate_health_report(&self) -> HealthReport {
        let metrics = self.get_system_health_metrics();
        let total_monitoring_time = self.monitoring_start_time.elapsed();
        
        HealthReport {
            monitoring_duration: total_monitoring_time,
            system_metrics: metrics,
            component_count: self.component_health.len(),
            healthy_components: self.component_health.values().filter(|s| s.is_healthy).count(),
            critical_alerts: 0, // Would be calculated from alert history
            warning_alerts: 0,  // Would be calculated from alert history
            info_alerts: 0,     // Would be calculated from alert history
        }
    }

    // Private helper methods

    fn calculate_health_score(&self, status: &ComponentHealthStatus) -> f64 {
        let mut score = 1.0;

        // Penalize high response times
        if status.response_time > Duration::from_millis(100) {
            score *= 0.9;
        }
        if status.response_time > Duration::from_millis(500) {
            score *= 0.8;
        }

        // Penalize high error rates
        if status.error_rate > 0.01 { // 1%
            score *= 0.9;
        }
        if status.error_rate > 0.05 { // 5%
            score *= 0.7;
        }

        // Penalize high resource usage
        if status.resource_usage.cpu_percentage > 80.0 {
            score *= 0.95;
        }
        if status.resource_usage.memory_mb > 1024 {
            score *= 0.95;
        }

        // Penalize low uptime
        if status.uptime_percentage < 99.0 {
            score *= 0.8;
        }
        if status.uptime_percentage < 95.0 {
            score *= 0.6;
        }

        score.max(0.0).min(1.0)
    }

    fn create_system_health_snapshot(&self) -> SystemHealthSnapshot {
        let healthy_components = self.component_health.values().filter(|s| s.is_healthy).count() as u32;
        let total_components = self.component_health.len() as u32;
        
        let overall_health_score = if total_components > 0 {
            self.component_health.values()
                .map(|status| status.health_score)
                .sum::<f64>() / total_components as f64
        } else {
            1.0
        };

        let avg_cpu = if total_components > 0 {
            self.component_health.values()
                .map(|status| status.resource_usage.cpu_percentage)
                .sum::<f64>() / total_components as f64
        } else {
            0.0
        };

        let avg_memory = if total_components > 0 {
            self.component_health.values()
                .map(|status| status.resource_usage.memory_mb)
                .sum::<u64>() / total_components as u64
        } else {
            0
        };

        let avg_response_time = if total_components > 0 {
            let total_response_time: Duration = self.component_health.values()
                .map(|status| status.response_time)
                .sum();
            total_response_time.as_millis() as f64 / total_components as f64
        } else {
            0.0
        };

        SystemHealthSnapshot {
            timestamp: Instant::now(),
            overall_health_score,
            healthy_components,
            total_components,
            system_load: avg_cpu / 100.0,
            memory_pressure: avg_memory as f64 / 1024.0, // Convert to GB
            disk_pressure: 0.3, // Simulated
            network_latency_ms: avg_response_time,
            active_alerts: (total_components - healthy_components),
            performance_degradation: 1.0 - overall_health_score,
        }
    }
}

impl ComponentResourceUsage {
    pub fn default() -> Self {
        Self {
            cpu_percentage: 0.0,
            memory_mb: 0,
            disk_io_mbps: 0.0,
            network_io_mbps: 0.0,
            thread_count: 0,
            connection_count: 0,
        }
    }
}

impl HealthThresholds {
    pub fn default() -> Self {
        Self {
            min_health_score: 0.8,
            max_response_time: Duration::from_millis(500),
            max_error_rate: 0.05, // 5%
            max_consecutive_failures: 3,
            min_uptime_percentage: 99.0,
            max_memory_usage_mb: 2048, // 2GB
            max_cpu_percentage: 85.0,
        }
    }

    pub fn strict() -> Self {
        Self {
            min_health_score: 0.95,
            max_response_time: Duration::from_millis(100),
            max_error_rate: 0.01, // 1%
            max_consecutive_failures: 1,
            min_uptime_percentage: 99.9,
            max_memory_usage_mb: 1024, // 1GB
            max_cpu_percentage: 70.0,
        }
    }

    pub fn lenient() -> Self {
        Self {
            min_health_score: 0.6,
            max_response_time: Duration::from_secs(2),
            max_error_rate: 0.1, // 10%
            max_consecutive_failures: 5,
            min_uptime_percentage: 95.0,
            max_memory_usage_mb: 4096, // 4GB
            max_cpu_percentage: 95.0,
        }
    }
}

/// Summary health report
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub monitoring_duration: Duration,
    pub system_metrics: SystemHealthMetrics,
    pub component_count: usize,
    pub healthy_components: usize,
    pub critical_alerts: u32,
    pub warning_alerts: u32,
    pub info_alerts: u32,
}

impl HealthReport {
    /// Check if the system is healthy overall
    pub fn is_system_healthy(&self) -> bool {
        self.system_metrics.avg_health_score >= 0.8 &&
        self.system_metrics.component_failure_rate <= 0.1 &&
        self.critical_alerts == 0
    }

    /// Generate summary text
    pub fn generate_summary(&self) -> String {
        format!(
            "Health Report Summary:\n\
            Monitoring Duration: {:?}\n\
            Overall Health Score: {:.2}\n\
            Healthy Components: {}/{}\n\
            Average Uptime: {:.2}%\n\
            Component Failure Rate: {:.2}%\n\
            Critical Alerts: {}\n\
            Warning Alerts: {}\n\
            System Status: {}",
            self.monitoring_duration,
            self.system_metrics.avg_health_score,
            self.healthy_components,
            self.component_count,
            self.system_metrics.uptime_percentage,
            self.system_metrics.component_failure_rate * 100.0,
            self.critical_alerts,
            self.warning_alerts,
            if self.is_system_healthy() { "HEALTHY" } else { "UNHEALTHY" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_monitor_creation() {
        let monitor = E2EHealthMonitor::new(Duration::from_secs(30));
        let metrics = monitor.get_system_health_metrics();
        
        assert_eq!(metrics.total_health_checks, 0);
        assert_eq!(metrics.failed_health_checks, 0);
        assert_eq!(metrics.avg_health_score, 1.0);
    }

    #[test]
    fn test_component_registration_and_health_update() {
        let mut monitor = E2EHealthMonitor::new(Duration::from_secs(30));
        
        monitor.register_component("test_component");
        
        let resource_usage = ComponentResourceUsage {
            cpu_percentage: 50.0,
            memory_mb: 512,
            disk_io_mbps: 10.0,
            network_io_mbps: 5.0,
            thread_count: 10,
            connection_count: 5,
        };
        
        monitor.update_component_health(
            "test_component",
            true,
            Duration::from_millis(100),
            resource_usage,
            0.01
        ).unwrap();
        
        let health_status = monitor.get_component_health("test_component").unwrap();
        assert!(health_status.is_healthy);
        assert_eq!(health_status.response_time, Duration::from_millis(100));
        assert_eq!(health_status.error_rate, 0.01);
    }

    #[test]
    fn test_health_alerts_generation() {
        let mut monitor = E2EHealthMonitor::new(Duration::from_secs(30));
        monitor.set_alert_thresholds(HealthThresholds::strict());
        
        monitor.register_component("failing_component");
        
        // Update with unhealthy status
        let resource_usage = ComponentResourceUsage {
            cpu_percentage: 95.0, // Above threshold
            memory_mb: 2048,      // Above threshold
            disk_io_mbps: 10.0,
            network_io_mbps: 5.0,
            thread_count: 50,
            connection_count: 20,
        };
        
        monitor.update_component_health(
            "failing_component",
            false, // Component is down
            Duration::from_millis(2000), // High response time
            resource_usage,
            0.15 // High error rate
        ).unwrap();
        
        let alerts = monitor.check_system_health();
        
        // Should generate multiple alerts
        assert!(!alerts.is_empty());
        
        // Check for specific alert types
        let has_component_down_alert = alerts.iter().any(|alert| {
            matches!(alert.alert_type, AlertType::ComponentDown)
        });
        assert!(has_component_down_alert);
        
        let has_high_response_time_alert = alerts.iter().any(|alert| {
            matches!(alert.alert_type, AlertType::HighResponseTime)
        });
        assert!(has_high_response_time_alert);
        
        let has_high_error_rate_alert = alerts.iter().any(|alert| {
            matches!(alert.alert_type, AlertType::HighErrorRate)
        });
        assert!(has_high_error_rate_alert);
    }

    #[test]
    fn test_health_score_calculation() {
        let mut monitor = E2EHealthMonitor::new(Duration::from_secs(30));
        monitor.register_component("test_component");
        
        // Test with good health metrics
        let good_resource_usage = ComponentResourceUsage {
            cpu_percentage: 30.0,
            memory_mb: 256,
            disk_io_mbps: 5.0,
            network_io_mbps: 2.0,
            thread_count: 5,
            connection_count: 3,
        };
        
        monitor.update_component_health(
            "test_component",
            true,
            Duration::from_millis(50),
            good_resource_usage,
            0.001
        ).unwrap();
        
        let health_status = monitor.get_component_health("test_component").unwrap();
        assert!(health_status.health_score > 0.9);
        
        // Test with poor health metrics
        let poor_resource_usage = ComponentResourceUsage {
            cpu_percentage: 90.0, // High CPU
            memory_mb: 2048,      // High memory
            disk_io_mbps: 50.0,
            network_io_mbps: 25.0,
            thread_count: 100,
            connection_count: 50,
        };
        
        monitor.update_component_health(
            "test_component",
            true,
            Duration::from_millis(800), // High response time
            poor_resource_usage,
            0.08 // High error rate
        ).unwrap();
        
        let health_status = monitor.get_component_health("test_component").unwrap();
        assert!(health_status.health_score < 0.8);
    }

    #[test]
    fn test_system_health_metrics() {
        let mut monitor = E2EHealthMonitor::new(Duration::from_secs(30));
        
        // Register multiple components with different health states
        monitor.register_component("healthy_component");
        monitor.register_component("unhealthy_component");
        monitor.register_component("marginal_component");
        
        let good_usage = ComponentResourceUsage::default();
        let bad_usage = ComponentResourceUsage {
            cpu_percentage: 95.0,
            memory_mb: 4096,
            disk_io_mbps: 100.0,
            network_io_mbps: 50.0,
            thread_count: 200,
            connection_count: 100,
        };
        
        monitor.update_component_health("healthy_component", true, Duration::from_millis(50), good_usage.clone(), 0.001).unwrap();
        monitor.update_component_health("unhealthy_component", false, Duration::from_millis(2000), bad_usage, 0.2).unwrap();
        monitor.update_component_health("marginal_component", true, Duration::from_millis(300), good_usage, 0.05).unwrap();
        
        let metrics = monitor.get_system_health_metrics();
        
        assert_eq!(metrics.total_health_checks, 3);
        assert_eq!(metrics.failed_health_checks, 1);
        assert!(metrics.avg_health_score < 1.0);
        assert!(metrics.component_failure_rate > 0.0);
    }

    #[test]
    fn test_health_thresholds() {
        let default_thresholds = HealthThresholds::default();
        let strict_thresholds = HealthThresholds::strict();
        let lenient_thresholds = HealthThresholds::lenient();
        
        assert!(strict_thresholds.min_health_score > default_thresholds.min_health_score);
        assert!(strict_thresholds.max_response_time < default_thresholds.max_response_time);
        assert!(strict_thresholds.max_error_rate < default_thresholds.max_error_rate);
        
        assert!(lenient_thresholds.min_health_score < default_thresholds.min_health_score);
        assert!(lenient_thresholds.max_response_time > default_thresholds.max_response_time);
        assert!(lenient_thresholds.max_error_rate > default_thresholds.max_error_rate);
    }

    #[test]
    fn test_health_report_generation() {
        let mut monitor = E2EHealthMonitor::new(Duration::from_secs(30));
        monitor.register_component("test_component");
        
        let resource_usage = ComponentResourceUsage::default();
        monitor.update_component_health("test_component", true, Duration::from_millis(50), resource_usage, 0.001).unwrap();
        
        let report = monitor.generate_health_report();
        
        assert_eq!(report.component_count, 1);
        assert_eq!(report.healthy_components, 1);
        assert!(report.is_system_healthy());
        
        let summary = report.generate_summary();
        assert!(summary.contains("HEALTHY"));
        assert!(summary.contains("Health Report Summary"));
    }

    #[test]
    fn test_health_history_tracking() {
        let mut monitor = E2EHealthMonitor::new(Duration::from_secs(30));
        monitor.register_component("test_component");
        
        let resource_usage = ComponentResourceUsage::default();
        
        // Generate several health checks
        for i in 0..5 {
            monitor.update_component_health(
                "test_component", 
                i % 2 == 0, // Alternate between healthy/unhealthy
                Duration::from_millis(50), 
                resource_usage.clone(), 
                0.001
            ).unwrap();
            monitor.check_system_health();
        }
        
        let history = monitor.get_health_history();
        assert_eq!(history.len(), 5);
        
        // Check that snapshots are ordered by time
        for i in 1..history.len() {
            assert!(history[i].timestamp >= history[i-1].timestamp);
        }
    }

    #[test]
    fn test_monitor_reset() {
        let mut monitor = E2EHealthMonitor::new(Duration::from_secs(30));
        monitor.register_component("test_component");
        
        let resource_usage = ComponentResourceUsage::default();
        monitor.update_component_health("test_component", true, Duration::from_millis(50), resource_usage, 0.001).unwrap();
        monitor.check_system_health();
        
        assert_eq!(monitor.component_health.len(), 1);
        assert_eq!(monitor.system_health_history.len(), 1);
        
        monitor.reset();
        
        assert_eq!(monitor.component_health.len(), 0);
        assert_eq!(monitor.system_health_history.len(), 0);
    }
}