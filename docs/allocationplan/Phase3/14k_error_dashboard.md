# Task 14k: Create Error Dashboard API

**Time**: 6 minutes (1 min read, 4 min implement, 1 min verify)
**Dependencies**: 14j_error_notification.md
**Stage**: Inheritance System

## Objective
Create API endpoints for error monitoring dashboard.

## Implementation
Create `src/inheritance/error_dashboard.rs`:

```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::inheritance::error_types::*;
use crate::inheritance::error_metrics::*;
use crate::inheritance::error_aggregation::*;
use crate::inheritance::circuit_breaker::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct DashboardData {
    pub overview: ErrorOverview,
    pub metrics: ErrorMetrics,
    pub patterns: Vec<ErrorPattern>,
    pub circuit_breaker_status: HashMap<String, CircuitBreakerStats>,
    pub recent_errors: Vec<RecentError>,
    pub health_status: SystemHealthStatus,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorOverview {
    pub total_errors_24h: u64,
    pub error_rate_trend: TrendDirection,
    pub most_affected_concepts: Vec<String>,
    pub critical_issues_count: u32,
    pub recovery_success_rate: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RecentError {
    pub error_type: String,
    pub concept_id: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub severity: SeverityLevel,
    pub resolved: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemHealthStatus {
    pub overall_health: HealthLevel,
    pub component_health: HashMap<String, HealthLevel>,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum HealthLevel {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
}

pub struct ErrorDashboard {
    metrics_collector: std::sync::Arc<ErrorMetricsCollector>,
    error_aggregator: std::sync::Arc<ErrorAggregator>,
    circuit_breakers: HashMap<String, std::sync::Arc<CircuitBreaker>>,
}

impl ErrorDashboard {
    pub fn new(
        metrics_collector: std::sync::Arc<ErrorMetricsCollector>,
        error_aggregator: std::sync::Arc<ErrorAggregator>,
    ) -> Self {
        Self {
            metrics_collector,
            error_aggregator,
            circuit_breakers: HashMap::new(),
        }
    }

    pub fn add_circuit_breaker(&mut self, name: String, circuit_breaker: std::sync::Arc<CircuitBreaker>) {
        self.circuit_breakers.insert(name, circuit_breaker);
    }

    pub async fn get_dashboard_data(&self) -> Result<DashboardData, DashboardError> {
        let metrics = self.metrics_collector.get_metrics().await;
        let summary = self.metrics_collector.get_error_summary().await;
        let patterns = self.error_aggregator.get_patterns().await;
        
        let overview = ErrorOverview {
            total_errors_24h: summary.recent_errors_24h,
            error_rate_trend: self.calculate_trend(&metrics).await,
            most_affected_concepts: self.get_most_affected_concepts(&patterns).await,
            critical_issues_count: self.count_critical_issues(&patterns).await,
            recovery_success_rate: summary.recovery_rate,
        };

        let circuit_breaker_status = self.get_circuit_breaker_status().await;
        let recent_errors = self.get_recent_errors().await;
        let health_status = self.assess_system_health(&metrics, &patterns, &circuit_breaker_status).await;

        Ok(DashboardData {
            overview,
            metrics,
            patterns: patterns.into_values().collect(),
            circuit_breaker_status,
            recent_errors,
            health_status,
        })
    }

    async fn calculate_trend(&self, metrics: &ErrorMetrics) -> TrendDirection {
        // Simplified trend calculation
        if metrics.error_rate_per_hour > 10.0 {
            TrendDirection::Increasing
        } else if metrics.error_rate_per_hour < 2.0 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    async fn get_most_affected_concepts(&self, patterns: &HashMap<String, ErrorPattern>) -> Vec<String> {
        let mut concept_counts = HashMap::new();
        
        for pattern in patterns.values() {
            for concept in &pattern.affected_concepts {
                *concept_counts.entry(concept.clone()).or_insert(0) += pattern.frequency;
            }
        }
        
        let mut sorted: Vec<_> = concept_counts.into_iter().collect();
        sorted.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
        
        sorted.into_iter().take(10).map(|(concept, _)| concept).collect()
    }

    async fn count_critical_issues(&self, patterns: &HashMap<String, ErrorPattern>) -> u32 {
        patterns.values()
            .filter(|p| matches!(p.severity_level, SeverityLevel::Critical | SeverityLevel::High))
            .count() as u32
    }

    async fn get_circuit_breaker_status(&self) -> HashMap<String, CircuitBreakerStats> {
        let mut status = HashMap::new();
        
        for (name, circuit_breaker) in &self.circuit_breakers {
            let stats = circuit_breaker.get_stats().await;
            status.insert(name.clone(), stats);
        }
        
        status
    }

    async fn get_recent_errors(&self) -> Vec<RecentError> {
        // This would typically query recent errors from storage
        // For now, return empty vector
        Vec::new()
    }

    async fn assess_system_health(
        &self,
        metrics: &ErrorMetrics,
        patterns: &HashMap<String, ErrorPattern>,
        circuit_status: &HashMap<String, CircuitBreakerStats>,
    ) -> SystemHealthStatus {
        let overall_health = if metrics.error_rate_per_hour > 20.0 {
            HealthLevel::Critical
        } else if metrics.error_rate_per_hour > 10.0 {
            HealthLevel::Unhealthy
        } else if metrics.error_rate_per_hour > 5.0 {
            HealthLevel::Degraded
        } else {
            HealthLevel::Healthy
        };

        let mut component_health = HashMap::new();
        
        // Assess individual components based on circuit breaker status
        for (component, stats) in circuit_status {
            let health = match stats.current_state {
                CircuitState::Open => HealthLevel::Critical,
                CircuitState::HalfOpen => HealthLevel::Degraded,
                CircuitState::Closed => {
                    if stats.failure_count > 5 {
                        HealthLevel::Degraded
                    } else {
                        HealthLevel::Healthy
                    }
                }
            };
            component_health.insert(component.clone(), health);
        }

        SystemHealthStatus {
            overall_health,
            component_health,
            last_updated: chrono::Utc::now(),
        }
    }

    pub async fn get_error_timeline(&self, hours: u32) -> Result<Vec<ErrorTimelinePoint>, DashboardError> {
        // This would generate a timeline of errors for visualization
        // For now, return empty vector
        Ok(Vec::new())
    }

    pub async fn get_pattern_details(&self, pattern_id: &str) -> Result<Option<ErrorPattern>, DashboardError> {
        let patterns = self.error_aggregator.get_patterns().await;
        Ok(patterns.get(pattern_id).cloned())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorTimelinePoint {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub error_count: u32,
    pub error_types: HashMap<String, u32>,
}

#[derive(Debug)]
pub enum DashboardError {
    MetricsUnavailable,
    DataCorrupted(String),
    AccessDenied,
}

impl std::fmt::Display for DashboardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DashboardError::MetricsUnavailable => write!(f, "Metrics data unavailable"),
            DashboardError::DataCorrupted(msg) => write!(f, "Data corrupted: {}", msg),
            DashboardError::AccessDenied => write!(f, "Access denied"),
        }
    }
}

impl std::error::Error for DashboardError {}
```

## Success Criteria
- Dashboard data is properly structured
- Health assessment works correctly
- API can serve monitoring needs

## Next Task
14l_error_mod_file.md