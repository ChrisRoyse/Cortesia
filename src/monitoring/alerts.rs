use crate::error::Result;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};
use uuid;

/// Alert manager for handling system alerts
pub struct AlertManager {
    alert_rules: Arc<RwLock<HashMap<String, AlertRule>>>,
    active_alerts: Arc<RwLock<HashMap<String, ActiveAlert>>>,
    alert_history: Arc<RwLock<Vec<AlertRecord>>>,
    notification_channels: Arc<RwLock<Vec<Box<dyn NotificationChannel>>>>,
}

impl AlertManager {
    pub fn new() -> Self {
        Self {
            alert_rules: Arc::new(RwLock::new(HashMap::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(Vec::new())),
            notification_channels: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn new_async() -> Result<Self> {
        Ok(Self {
            alert_rules: Arc::new(RwLock::new(HashMap::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(Vec::new())),
            notification_channels: Arc::new(RwLock::new(Vec::new())),
        })
    }

    pub async fn add_alert_rule(&self, rule: AlertRule) -> Result<()> {
        let mut rules = self.alert_rules.write().await;
        rules.insert(rule.name.clone(), rule);
        Ok(())
    }

    pub async fn remove_alert_rule(&self, rule_name: &str) -> Result<bool> {
        let mut rules = self.alert_rules.write().await;
        Ok(rules.remove(rule_name).is_some())
    }

    pub async fn trigger_alert(&self, name: String, message: String, severity: AlertSeverity) -> Result<()> {
        let alert_id = format!("alert_{}", uuid::Uuid::new_v4());
        
        let alert = ActiveAlert {
            id: alert_id.clone(),
            name: name.clone(),
            message: message.clone(),
            severity,
            triggered_at: Utc::now(),
            acknowledged: false,
            resolved: false,
        };

        // Add to active alerts
        {
            let mut active_alerts = self.active_alerts.write().await;
            active_alerts.insert(alert_id.clone(), alert.clone());
        }

        // Add to history
        {
            let mut history = self.alert_history.write().await;
            history.push(AlertRecord {
                id: alert_id.clone(),
                name: name.clone(),
                message: message.clone(),
                severity,
                triggered_at: alert.triggered_at,
                acknowledged_at: None,
                resolved_at: None,
            });
        }

        // Send notifications
        self.send_notifications(&alert).await?;

        Ok(())
    }

    pub async fn acknowledge_alert(&self, alert_id: &str) -> Result<bool> {
        let mut active_alerts = self.active_alerts.write().await;
        
        if let Some(alert) = active_alerts.get_mut(alert_id) {
            alert.acknowledged = true;
            
            // Update history
            {
                let mut history = self.alert_history.write().await;
                if let Some(record) = history.iter_mut().find(|r| r.id == alert_id) {
                    record.acknowledged_at = Some(Utc::now());
                }
            }
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn resolve_alert(&self, alert_id: &str) -> Result<bool> {
        let mut active_alerts = self.active_alerts.write().await;
        
        if let Some(_alert) = active_alerts.remove(alert_id) {
            // Update history
            {
                let mut history = self.alert_history.write().await;
                if let Some(record) = history.iter_mut().find(|r| r.id == alert_id) {
                    record.resolved_at = Some(Utc::now());
                }
            }
            
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub async fn get_active_alerts(&self) -> Result<Vec<ActiveAlert>> {
        let active_alerts = self.active_alerts.read().await;
        Ok(active_alerts.values().cloned().collect())
    }

    pub async fn get_alert_history(&self, limit: Option<usize>) -> Result<Vec<AlertRecord>> {
        let history = self.alert_history.read().await;
        
        if let Some(limit) = limit {
            Ok(history.iter().rev().take(limit).cloned().collect())
        } else {
            Ok(history.clone())
        }
    }

    pub async fn get_recent_alerts_count(&self, duration: Duration) -> Result<usize> {
        let cutoff_time = Utc::now() - chrono::Duration::from_std(duration).unwrap();
        let history = self.alert_history.read().await;
        
        let count = history.iter()
            .filter(|record| record.triggered_at > cutoff_time)
            .count();
        
        Ok(count)
    }

    pub async fn add_notification_channel(&self, channel: Box<dyn NotificationChannel>) -> Result<()> {
        let mut channels = self.notification_channels.write().await;
        channels.push(channel);
        Ok(())
    }

    pub async fn evaluate_conditions(&self, metrics: &HashMap<String, f64>) -> Result<()> {
        let rules = self.alert_rules.read().await;
        
        for rule in rules.values() {
            let should_trigger = self.evaluate_rule_conditions(rule, metrics).await?;
            
            if should_trigger {
                self.trigger_alert(
                    rule.name.clone(),
                    rule.description.clone(),
                    rule.severity,
                ).await?;
            }
        }
        
        Ok(())
    }

    async fn evaluate_rule_conditions(&self, rule: &AlertRule, metrics: &HashMap<String, f64>) -> Result<bool> {
        for condition in &rule.conditions {
            let metric_value = metrics.get(&condition.metric_name).copied().unwrap_or(0.0);
            
            let condition_met = match condition.condition_type {
                AlertConditionType::GreaterThan => metric_value > condition.threshold,
                AlertConditionType::LessThan => metric_value < condition.threshold,
                AlertConditionType::Equal => (metric_value - condition.threshold).abs() < f64::EPSILON,
                AlertConditionType::NotEqual => (metric_value - condition.threshold).abs() >= f64::EPSILON,
            };
            
            if condition_met {
                return Ok(true);
            }
        }
        
        Ok(false)
    }

    async fn send_notifications(&self, alert: &ActiveAlert) -> Result<()> {
        let channels = self.notification_channels.read().await;
        
        let notification = AlertNotification {
            alert_id: alert.id.clone(),
            title: alert.name.clone(),
            message: alert.message.clone(),
            severity: alert.severity,
            timestamp: alert.triggered_at,
        };
        
        for channel in channels.iter() {
            if let Err(e) = channel.send_notification(&notification).await {
                eprintln!("Failed to send notification: {}", e);
            }
        }
        
        Ok(())
    }
}

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub name: String,
    pub description: String,
    pub severity: AlertSeverity,
    pub conditions: Vec<AlertCondition>,
    pub cooldown_duration: Duration,
}

/// Alert condition
#[derive(Debug, Clone)]
pub struct AlertCondition {
    pub metric_name: String,
    pub condition_type: AlertConditionType,
    pub threshold: f64,
}

/// Alert condition types
#[derive(Debug, Clone)]
pub enum AlertConditionType {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Active alert
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    pub id: String,
    pub name: String,
    pub message: String,
    pub severity: AlertSeverity,
    pub triggered_at: DateTime<Utc>,
    pub acknowledged: bool,
    pub resolved: bool,
}

/// Alert record for history
#[derive(Debug, Clone)]
pub struct AlertRecord {
    pub id: String,
    pub name: String,
    pub message: String,
    pub severity: AlertSeverity,
    pub triggered_at: DateTime<Utc>,
    pub acknowledged_at: Option<DateTime<Utc>>,
    pub resolved_at: Option<DateTime<Utc>>,
}

/// Alert notification
#[derive(Debug, Clone)]
pub struct AlertNotification {
    pub alert_id: String,
    pub title: String,
    pub message: String,
    pub severity: AlertSeverity,
    pub timestamp: DateTime<Utc>,
}

/// Notification channel trait
#[async_trait::async_trait]
pub trait NotificationChannel: Send + Sync {
    async fn send_notification(&self, notification: &AlertNotification) -> Result<()>;
}

/// Email notification channel
pub struct EmailNotificationChannel {
    smtp_server: String,
    recipients: Vec<String>,
}

impl EmailNotificationChannel {
    pub fn new(smtp_server: String, recipients: Vec<String>) -> Self {
        Self {
            smtp_server,
            recipients,
        }
    }
}

#[async_trait::async_trait]
impl NotificationChannel for EmailNotificationChannel {
    async fn send_notification(&self, notification: &AlertNotification) -> Result<()> {
        // Mock email sending
        println!("Sending email notification: {} - {}", notification.title, notification.message);
        Ok(())
    }
}

/// Slack notification channel
pub struct SlackNotificationChannel {
    webhook_url: String,
    channel: String,
}

impl SlackNotificationChannel {
    pub fn new(webhook_url: String, channel: String) -> Self {
        Self {
            webhook_url,
            channel,
        }
    }
}

#[async_trait::async_trait]
impl NotificationChannel for SlackNotificationChannel {
    async fn send_notification(&self, notification: &AlertNotification) -> Result<()> {
        // Mock Slack sending
        println!("Sending Slack notification to {}: {} - {}", 
                 self.channel, notification.title, notification.message);
        Ok(())
    }
}

/// Console notification channel for testing
pub struct ConsoleNotificationChannel;

#[async_trait::async_trait]
impl NotificationChannel for ConsoleNotificationChannel {
    async fn send_notification(&self, notification: &AlertNotification) -> Result<()> {
        println!("[ALERT] {:?} - {}: {}", 
                 notification.severity, notification.title, notification.message);
        Ok(())
    }
}
