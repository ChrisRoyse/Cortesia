# Task 19d: Implement Alert System

**Estimated Time**: 4 minutes  
**Dependencies**: 19c  
**Stage**: Performance Monitoring  

## Objective
Implement basic alert system for performance thresholds.

## Implementation Steps

1. Create `src/monitoring/alerts.rs`:
```rust
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use tokio::sync::mpsc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub metric_value: f64,
    pub threshold: f64,
    pub timestamp: DateTime<Utc>,
    pub resolved: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertType {
    HighMemoryUsage,
    HighCpuUsage,
    SlowResponseTime,
    HighErrorRate,
    DatabaseConnectivity,
    CacheFailure,
    LowCacheHitRate,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

pub struct AlertManager {
    thresholds: HashMap<AlertType, AlertThreshold>,
    active_alerts: HashMap<String, Alert>,
    alert_sender: mpsc::UnboundedSender<Alert>,
}

#[derive(Debug, Clone)]
struct AlertThreshold {
    warning: f64,
    critical: f64,
}

impl AlertManager {
    pub fn new() -> (Self, mpsc::UnboundedReceiver<Alert>) {
        let (alert_sender, alert_receiver) = mpsc::unbounded_channel();
        
        let mut thresholds = HashMap::new();
        
        // Configure default thresholds
        thresholds.insert(AlertType::HighMemoryUsage, AlertThreshold {
            warning: 70.0,
            critical: 85.0,
        });
        
        thresholds.insert(AlertType::HighCpuUsage, AlertThreshold {
            warning: 60.0,
            critical: 80.0,
        });
        
        thresholds.insert(AlertType::SlowResponseTime, AlertThreshold {
            warning: 500.0, // ms
            critical: 1000.0,
        });
        
        thresholds.insert(AlertType::HighErrorRate, AlertThreshold {
            warning: 2.0, // percent
            critical: 5.0,
        });
        
        thresholds.insert(AlertType::LowCacheHitRate, AlertThreshold {
            warning: 70.0, // percent
            critical: 50.0,
        });
        
        let manager = Self {
            thresholds,
            active_alerts: HashMap::new(),
            alert_sender,
        };
        
        (manager, alert_receiver)
    }
    
    pub fn check_metric(&mut self, alert_type: AlertType, value: f64, context: &str) {
        if let Some(threshold) = self.thresholds.get(&alert_type) {
            let severity = if value >= threshold.critical {
                Some(AlertSeverity::Critical)
            } else if value >= threshold.warning {
                Some(AlertSeverity::Warning)
            } else {
                None
            };
            
            if let Some(severity) = severity {
                let alert_id = format!("{:?}_{}", alert_type, Utc::now().timestamp());
                
                // Check if similar alert is already active
                let existing_active = self.active_alerts.values()
                    .any(|a| a.alert_type == alert_type && !a.resolved);
                
                if !existing_active {
                    let alert = Alert {
                        id: alert_id.clone(),
                        alert_type: alert_type.clone(),
                        severity,
                        message: self.format_alert_message(&alert_type, value, context),
                        metric_value: value,
                        threshold: match severity {
                            AlertSeverity::Critical => threshold.critical,
                            _ => threshold.warning,
                        },
                        timestamp: Utc::now(),
                        resolved: false,
                    };
                    
                    self.active_alerts.insert(alert_id, alert.clone());
                    
                    if let Err(_) = self.alert_sender.send(alert) {
                        eprintln!("Failed to send alert notification");
                    }
                }
            } else {
                // Resolve any active alerts of this type
                self.resolve_alerts_of_type(&alert_type);
            }
        }
    }
    
    fn format_alert_message(&self, alert_type: &AlertType, value: f64, context: &str) -> String {
        match alert_type {
            AlertType::HighMemoryUsage => {
                format!("High memory usage: {:.1}% in {}", value, context)
            }
            AlertType::HighCpuUsage => {
                format!("High CPU usage: {:.1}% in {}", value, context)
            }
            AlertType::SlowResponseTime => {
                format!("Slow response time: {:.1}ms in {}", value, context)
            }
            AlertType::HighErrorRate => {
                format!("High error rate: {:.1}% in {}", value, context)
            }
            AlertType::DatabaseConnectivity => {
                format!("Database connectivity issue in {}", context)
            }
            AlertType::CacheFailure => {
                format!("Cache failure detected in {}", context)
            }
            AlertType::LowCacheHitRate => {
                format!("Low cache hit rate: {:.1}% in {}", value, context)
            }
        }
    }
    
    fn resolve_alerts_of_type(&mut self, alert_type: &AlertType) {
        for alert in self.active_alerts.values_mut() {
            if alert.alert_type == *alert_type && !alert.resolved {
                alert.resolved = true;
                
                let resolved_alert = Alert {
                    id: format!("{}_resolved", alert.id),
                    alert_type: alert_type.clone(),
                    severity: AlertSeverity::Info,
                    message: format!("Resolved: {}", alert.message),
                    metric_value: 0.0,
                    threshold: 0.0,
                    timestamp: Utc::now(),
                    resolved: true,
                };
                
                let _ = self.alert_sender.send(resolved_alert);
            }
        }
    }
    
    pub fn get_active_alerts(&self) -> Vec<&Alert> {
        self.active_alerts.values().filter(|a| !a.resolved).collect()
    }
    
    pub fn get_alert_history(&self) -> Vec<&Alert> {
        self.active_alerts.values().collect()
    }
}

// Simple alert handler for demonstration
pub async fn handle_alerts(mut alert_receiver: mpsc::UnboundedReceiver<Alert>) {
    while let Some(alert) = alert_receiver.recv().await {
        match alert.severity {
            AlertSeverity::Critical => {
                eprintln!("🚨 CRITICAL ALERT: {}", alert.message);
                // In production, send to monitoring systems, PagerDuty, etc.
            }
            AlertSeverity::Warning => {
                println!("⚠️  WARNING: {}", alert.message);
            }
            AlertSeverity::Info => {
                println!("ℹ️  INFO: {}", alert.message);
            }
        }
    }
}
```

## Acceptance Criteria
- [ ] Alert system implemented
- [ ] Configurable thresholds
- [ ] Alert deduplication
- [ ] Alert resolution tracking

## Success Metrics
- Alerts trigger at appropriate thresholds
- No alert spam
- Proper alert resolution

## Next Task
19e_implement_system_monitoring.md