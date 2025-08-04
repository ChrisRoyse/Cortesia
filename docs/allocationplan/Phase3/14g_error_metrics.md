# Task 14g: Implement Error Metrics

**Time**: 6 minutes (1 min read, 4 min implement, 1 min verify)
**Dependencies**: 14f_error_recovery_strategies.md
**Stage**: Inheritance System

## Objective
Add error metrics and monitoring for the inheritance system.

## Implementation
Create `src/inheritance/error_metrics.rs`:

```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    pub total_errors: u64,
    pub errors_by_type: HashMap<String, u64>,
    pub recovery_attempts: u64,
    pub successful_recoveries: u64,
    pub retry_attempts: u64,
    pub average_resolution_time_ms: f64,
    pub error_rate_per_hour: f64,
    pub last_updated: DateTime<Utc>,
}

impl Default for ErrorMetrics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            errors_by_type: HashMap::new(),
            recovery_attempts: 0,
            successful_recoveries: 0,
            retry_attempts: 0,
            average_resolution_time_ms: 0.0,
            error_rate_per_hour: 0.0,
            last_updated: Utc::now(),
        }
    }
}

pub struct ErrorMetricsCollector {
    metrics: Arc<RwLock<ErrorMetrics>>,
    error_history: Arc<RwLock<Vec<ErrorEvent>>>,
    collection_start: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct ErrorEvent {
    pub error_type: String,
    pub timestamp: DateTime<Utc>,
    pub resolution_time_ms: Option<f64>,
    pub recovery_attempted: bool,
    pub recovery_successful: bool,
    pub retry_count: u32,
}

impl ErrorMetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(ErrorMetrics::default())),
            error_history: Arc::new(RwLock::new(Vec::new())),
            collection_start: Utc::now(),
        }
    }

    pub async fn record_error(&self, error_type: &str) {
        let event = ErrorEvent {
            error_type: error_type.to_string(),
            timestamp: Utc::now(),
            resolution_time_ms: None,
            recovery_attempted: false,
            recovery_successful: false,
            retry_count: 0,
        };

        {
            let mut metrics = self.metrics.write().await;
            metrics.total_errors += 1;
            *metrics.errors_by_type.entry(error_type.to_string()).or_insert(0) += 1;
            metrics.last_updated = Utc::now();
        }

        {
            let mut history = self.error_history.write().await;
            history.push(event);
            
            // Keep only last 1000 events
            if history.len() > 1000 {
                history.drain(0..100);
            }
        }

        self.update_error_rate().await;
    }

    pub async fn record_recovery_attempt(&self, error_type: &str, successful: bool) {
        {
            let mut metrics = self.metrics.write().await;
            metrics.recovery_attempts += 1;
            if successful {
                metrics.successful_recoveries += 1;
            }
        }

        // Update the most recent error event of this type
        {
            let mut history = self.error_history.write().await;
            if let Some(event) = history.iter_mut()
                .rev()
                .find(|e| e.error_type == error_type && !e.recovery_attempted) {
                event.recovery_attempted = true;
                event.recovery_successful = successful;
            }
        }
    }

    pub async fn record_retry(&self, error_type: &str) {
        {
            let mut metrics = self.metrics.write().await;
            metrics.retry_attempts += 1;
        }

        {
            let mut history = self.error_history.write().await;
            if let Some(event) = history.iter_mut()
                .rev()
                .find(|e| e.error_type == error_type) {
                event.retry_count += 1;
            }
        }
    }

    pub async fn record_resolution(&self, error_type: &str, resolution_time_ms: f64) {
        {
            let mut history = self.error_history.write().await;
            if let Some(event) = history.iter_mut()
                .rev()
                .find(|e| e.error_type == error_type && e.resolution_time_ms.is_none()) {
                event.resolution_time_ms = Some(resolution_time_ms);
            }
        }

        self.update_average_resolution_time().await;
    }

    async fn update_error_rate(&self) {
        let hours_elapsed = Utc::now()
            .signed_duration_since(self.collection_start)
            .num_hours() as f64;
        
        if hours_elapsed > 0.0 {
            let metrics = self.metrics.read().await;
            let error_rate = metrics.total_errors as f64 / hours_elapsed;
            drop(metrics);
            
            self.metrics.write().await.error_rate_per_hour = error_rate;
        }
    }

    async fn update_average_resolution_time(&self) {
        let history = self.error_history.read().await;
        let resolved_events: Vec<_> = history.iter()
            .filter_map(|e| e.resolution_time_ms)
            .collect();
        
        if !resolved_events.is_empty() {
            let average = resolved_events.iter().sum::<f64>() / resolved_events.len() as f64;
            drop(history);
            
            self.metrics.write().await.average_resolution_time_ms = average;
        }
    }

    pub async fn get_metrics(&self) -> ErrorMetrics {
        self.metrics.read().await.clone()
    }

    pub async fn get_error_summary(&self) -> ErrorSummary {
        let metrics = self.metrics.read().await;
        let history = self.error_history.read().await;
        
        let recent_errors = history.iter()
            .filter(|e| Utc::now().signed_duration_since(e.timestamp).num_hours() < 24)
            .count();
        
        let recovery_rate = if metrics.recovery_attempts > 0 {
            metrics.successful_recoveries as f64 / metrics.recovery_attempts as f64
        } else {
            0.0
        };

        ErrorSummary {
            total_errors: metrics.total_errors,
            recent_errors_24h: recent_errors as u64,
            recovery_rate,
            error_rate_per_hour: metrics.error_rate_per_hour,
            most_common_error: self.find_most_common_error(&metrics.errors_by_type),
        }
    }

    fn find_most_common_error(&self, errors_by_type: &HashMap<String, u64>) -> Option<String> {
        errors_by_type.iter()
            .max_by_key(|(_, count)| *count)
            .map(|(error_type, _)| error_type.clone())
    }
}

#[derive(Debug, Serialize)]
pub struct ErrorSummary {
    pub total_errors: u64,
    pub recent_errors_24h: u64,
    pub recovery_rate: f64,
    pub error_rate_per_hour: f64,
    pub most_common_error: Option<String>,
}
```

## Success Criteria
- Error metrics are accurately tracked
- Recovery rates are calculated correctly
- Performance data is available

## Next Task
14h_circuit_breaker.md