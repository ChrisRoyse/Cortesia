//! Real-time Test Dashboard
//! 
//! Provides a web-based dashboard for monitoring test execution in real-time.

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;

/// Dashboard configuration
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    pub host: String,
    pub port: u16,
    pub auto_refresh_interval_ms: u64,
    pub max_recent_tests: usize,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            auto_refresh_interval_ms: 1000,
            max_recent_tests: 100,
        }
    }
}

/// Real-time metrics for the dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    pub total_tests_run: usize,
    pub tests_passed: usize,
    pub tests_failed: usize,
    pub current_pass_rate: f64,
    pub average_test_duration_ms: f64,
    pub peak_memory_usage_mb: f64,
    pub current_cpu_usage: f64,
    pub tests_per_second: f64,
    pub estimated_completion_time: Option<String>,
}

/// Individual test result for dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTestResult {
    pub name: String,
    pub status: String,
    pub duration_ms: u64,
    pub memory_mb: f64,
    pub timestamp: String,
    pub error_message: Option<String>,
}

/// Dashboard state
#[derive(Debug, Clone)]
pub struct DashboardState {
    pub metrics: DashboardMetrics,
    pub recent_tests: Vec<DashboardTestResult>,
    pub test_categories: HashMap<String, usize>,
    pub performance_history: Vec<(String, f64)>, // (timestamp, duration)
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            metrics: DashboardMetrics {
                total_tests_run: 0,
                tests_passed: 0,
                tests_failed: 0,
                current_pass_rate: 0.0,
                average_test_duration_ms: 0.0,
                peak_memory_usage_mb: 0.0,
                current_cpu_usage: 0.0,
                tests_per_second: 0.0,
                estimated_completion_time: None,
            },
            recent_tests: Vec::new(),
            test_categories: HashMap::new(),
            performance_history: Vec::new(),
        }
    }
}

/// Test dashboard server
pub struct TestDashboard {
    config: DashboardConfig,
    state: Arc<RwLock<DashboardState>>,
}

impl TestDashboard {
    /// Create a new test dashboard
    pub fn new(config: &DashboardConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            state: Arc::new(RwLock::new(DashboardState::default())),
        })
    }

    /// Start the dashboard server
    pub async fn start(&self) -> Result<()> {
        let app = Router::new()
            .route("/", get(dashboard_index))
            .route("/api/metrics", get(get_metrics))
            .route("/api/tests", get(get_recent_tests))
            .route("/api/update", post(update_test_result))
            .layer(ServiceBuilder::new().layer(CorsLayer::permissive()))
            .with_state(self.state.clone());

        let addr = format!("{}:{}", self.config.host, self.config.port);
        println!("🚀 Test Dashboard starting at http://{}", addr);
        
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;
        
        Ok(())
    }

    /// Update dashboard with new test result
    pub async fn update_test_result(&self, test_result: DashboardTestResult) -> Result<()> {
        let mut state = self.state.write().await;
        
        // Update metrics
        state.metrics.total_tests_run += 1;
        if test_result.status == "PASSED" {
            state.metrics.tests_passed += 1;
        } else {
            state.metrics.tests_failed += 1;
        }
        
        state.metrics.current_pass_rate = if state.metrics.total_tests_run > 0 {
            (state.metrics.tests_passed as f64 / state.metrics.total_tests_run as f64) * 100.0
        } else {
            0.0
        };

        // Update average duration
        let total_duration: u64 = state.recent_tests.iter().map(|t| t.duration_ms).sum::<u64>() + test_result.duration_ms;
        state.metrics.average_test_duration_ms = total_duration as f64 / state.metrics.total_tests_run as f64;

        // Update peak memory
        if test_result.memory_mb > state.metrics.peak_memory_usage_mb {
            state.metrics.peak_memory_usage_mb = test_result.memory_mb;
        }

        // Add to recent tests
        state.recent_tests.push(test_result.clone());
        if state.recent_tests.len() > self.config.max_recent_tests {
            state.recent_tests.remove(0);
        }

        // Update test categories
        let category = self.extract_test_category(&test_result.name);
        *state.test_categories.entry(category).or_insert(0) += 1;

        // Add to performance history
        state.performance_history.push((test_result.timestamp, test_result.duration_ms as f64));
        if state.performance_history.len() > 50 {
            state.performance_history.remove(0);
        }

        Ok(())
    }

    fn extract_test_category(&self, test_name: &str) -> String {
        if test_name.contains("entity") {
            "Entity Tests".to_string()
        } else if test_name.contains("graph") {
            "Graph Tests".to_string()
        } else if test_name.contains("storage") {
            "Storage Tests".to_string()
        } else if test_name.contains("embedding") {
            "Embedding Tests".to_string()
        } else if test_name.contains("query") {
            "Query Tests".to_string()
        } else if test_name.contains("federation") {
            "Federation Tests".to_string()
        } else {
            "Other Tests".to_string()
        }
    }
}

// HTTP handlers

async fn dashboard_index() -> Html<&'static str> {
    Html(include_str!("dashboard_template.html"))
}

async fn get_metrics(State(state): State<Arc<RwLock<DashboardState>>>) -> Json<DashboardMetrics> {
    let state = state.read().await;
    Json(state.metrics.clone())
}

async fn get_recent_tests(State(state): State<Arc<RwLock<DashboardState>>>) -> Json<Vec<DashboardTestResult>> {
    let state = state.read().await;
    Json(state.recent_tests.clone())
}

async fn update_test_result(
    State(state): State<Arc<RwLock<DashboardState>>>,
    Json(test_result): Json<DashboardTestResult>,
) -> StatusCode {
    // Update state logic would be here (simplified for now)
    StatusCode::OK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dashboard_creation() {
        let config = DashboardConfig::default();
        let dashboard = TestDashboard::new(&config);
        assert!(dashboard.is_ok());
    }

    #[tokio::test]
    async fn test_dashboard_state_update() {
        let config = DashboardConfig::default();
        let dashboard = TestDashboard::new(&config).unwrap();

        let test_result = DashboardTestResult {
            name: "test_entity_creation".to_string(),
            status: "PASSED".to_string(),
            duration_ms: 150,
            memory_mb: 2.5,
            timestamp: chrono::Utc::now().to_rfc3339(),
            error_message: None,
        };

        dashboard.update_test_result(test_result).await.unwrap();

        let state = dashboard.state.read().await;
        assert_eq!(state.metrics.total_tests_run, 1);
        assert_eq!(state.metrics.tests_passed, 1);
        assert_eq!(state.metrics.current_pass_rate, 100.0);
        assert_eq!(state.recent_tests.len(), 1);
    }

    #[test]
    fn test_category_extraction() {
        let config = DashboardConfig::default();
        let dashboard = TestDashboard::new(&config).unwrap();

        assert_eq!(dashboard.extract_test_category("test_entity_creation"), "Entity Tests");
        assert_eq!(dashboard.extract_test_category("test_graph_operations"), "Graph Tests");
        assert_eq!(dashboard.extract_test_category("test_storage_csr"), "Storage Tests");
        assert_eq!(dashboard.extract_test_category("test_embedding_simd"), "Embedding Tests");
        assert_eq!(dashboard.extract_test_category("test_unknown"), "Other Tests");
    }

    #[tokio::test]
    async fn test_metrics_calculation() {
        let config = DashboardConfig::default();
        let dashboard = TestDashboard::new(&config).unwrap();

        // Add passing test
        let passing_test = DashboardTestResult {
            name: "test_pass".to_string(),
            status: "PASSED".to_string(),
            duration_ms: 100,
            memory_mb: 1.0,
            timestamp: chrono::Utc::now().to_rfc3339(),
            error_message: None,
        };

        // Add failing test
        let failing_test = DashboardTestResult {
            name: "test_fail".to_string(),
            status: "FAILED".to_string(),
            duration_ms: 200,
            memory_mb: 3.0,
            timestamp: chrono::Utc::now().to_rfc3339(),
            error_message: Some("Test failed".to_string()),
        };

        dashboard.update_test_result(passing_test).await.unwrap();
        dashboard.update_test_result(failing_test).await.unwrap();

        let state = dashboard.state.read().await;
        assert_eq!(state.metrics.total_tests_run, 2);
        assert_eq!(state.metrics.tests_passed, 1);
        assert_eq!(state.metrics.tests_failed, 1);
        assert_eq!(state.metrics.current_pass_rate, 50.0);
        assert_eq!(state.metrics.average_test_duration_ms, 150.0);
        assert_eq!(state.metrics.peak_memory_usage_mb, 3.0);
    }
}