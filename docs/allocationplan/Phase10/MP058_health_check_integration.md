# MP058: Health Check Integration

## Task Description
Implement comprehensive health check system for monitoring component status, dependency health, and system readiness across neuromorphic graph algorithm infrastructure.

## Prerequisites
- MP001-MP050 completed
- Understanding of health check patterns
- Knowledge of service monitoring principles
- Familiarity with dependency management

## Detailed Steps

1. Create `src/neuromorphic/health/checks.rs`

2. Implement health check framework with component monitoring:
   ```rust
   use async_trait::async_trait;
   use serde::{Serialize, Deserialize};
   use std::collections::HashMap;
   use std::sync::Arc;
   use std::time::{Duration, Instant};
   use chrono::{DateTime, Utc};
   use uuid::Uuid;
   use tokio::sync::RwLock;
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub enum HealthStatus {
       Healthy,
       Degraded,
       Unhealthy,
       Unknown,
   }
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct HealthCheckResult {
       pub component_name: String,
       pub status: HealthStatus,
       pub message: Option<String>,
       pub details: HashMap<String, serde_json::Value>,
       pub check_duration: Duration,
       pub timestamp: DateTime<Utc>,
       pub error: Option<String>,
   }
   
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct SystemHealth {
       pub overall_status: HealthStatus,
       pub components: HashMap<String, HealthCheckResult>,
       pub dependencies: HashMap<String, DependencyHealth>,
       pub system_metrics: SystemMetrics,
       pub uptime: Duration,
       pub last_check: DateTime<Utc>,
   }
   
   #[async_trait]
   pub trait HealthCheck: Send + Sync {
       async fn check_health(&self) -> HealthCheckResult;
       fn component_name(&self) -> &str;
       fn check_interval(&self) -> Duration;
       fn timeout(&self) -> Duration;
       fn is_critical(&self) -> bool;
   }
   
   pub struct HealthMonitor {
       checks: Arc<RwLock<HashMap<String, Arc<dyn HealthCheck>>>>,
       results: Arc<RwLock<HashMap<String, HealthCheckResult>>>,
       config: HealthMonitorConfig,
       dependency_tracker: DependencyTracker,
       alert_manager: Arc<dyn AlertManager>,
   }
   
   #[derive(Debug, Clone)]
   pub struct HealthMonitorConfig {
       pub global_timeout: Duration,
       pub check_interval: Duration,
       pub failure_threshold: u32,
       pub recovery_threshold: u32,
       pub enable_auto_recovery: bool,
       pub critical_component_timeout: Duration,
   }
   
   impl HealthMonitor {
       pub fn new(
           config: HealthMonitorConfig,
           alert_manager: Arc<dyn AlertManager>,
       ) -> Self {
           Self {
               checks: Arc::new(RwLock::new(HashMap::new())),
               results: Arc::new(RwLock::new(HashMap::new())),
               config,
               dependency_tracker: DependencyTracker::new(),
               alert_manager,
           }
       }
       
       pub async fn register_health_check(&self, check: Arc<dyn HealthCheck>) {
           let component_name = check.component_name().to_string();
           let mut checks = self.checks.write().await;
           checks.insert(component_name, check);
       }
       
       pub async fn start_monitoring(&self) -> Result<(), HealthError> {
           tracing::info!("Starting health monitoring system");
           
           let checks = self.checks.read().await.clone();
           
           for (component_name, health_check) in checks {
               let health_check = health_check.clone();
               let results = self.results.clone();
               let alert_manager = self.alert_manager.clone();
               let config = self.config.clone();
               
               tokio::spawn(async move {
                   Self::monitor_component(
                       component_name,
                       health_check,
                       results,
                       alert_manager,
                       config,
                   ).await;
               });
           }
           
           // Start system health aggregator
           let system_monitor = self.clone();
           tokio::spawn(async move {
               system_monitor.aggregate_system_health().await;
           });
           
           Ok(())
       }
       
       async fn monitor_component(
           component_name: String,
           health_check: Arc<dyn HealthCheck>,
           results: Arc<RwLock<HashMap<String, HealthCheckResult>>>,
           alert_manager: Arc<dyn AlertManager>,
           config: HealthMonitorConfig,
       ) {
           let mut interval = tokio::time::interval(health_check.check_interval());
           let mut consecutive_failures = 0u32;
           let mut consecutive_successes = 0u32;
           let mut last_status = HealthStatus::Unknown;
           
           loop {
               interval.tick().await;
               
               let check_start = Instant::now();
               let timeout = if health_check.is_critical() {
                   config.critical_component_timeout
               } else {
                   health_check.timeout()
               };
               
               let result = match tokio::time::timeout(timeout, health_check.check_health()).await {
                   Ok(mut result) => {
                       result.check_duration = check_start.elapsed();
                       result.timestamp = Utc::now();
                       result
                   }
                   Err(_) => HealthCheckResult {
                       component_name: component_name.clone(),
                       status: HealthStatus::Unhealthy,
                       message: Some("Health check timeout".to_string()),
                       details: HashMap::new(),
                       check_duration: timeout,
                       timestamp: Utc::now(),
                       error: Some(format!("Timeout after {:?}", timeout)),
                   }
               };
               
               // Update failure/success counters
               match result.status {
                   HealthStatus::Healthy => {
                       consecutive_failures = 0;
                       consecutive_successes += 1;
                   }
                   HealthStatus::Degraded => {
                       consecutive_failures += 1;
                       consecutive_successes = 0;
                   }
                   HealthStatus::Unhealthy => {
                       consecutive_failures += 1;
                       consecutive_successes = 0;
                   }
                   HealthStatus::Unknown => {
                       // Don't change counters for unknown status
                   }
               }
               
               // Check for status transitions
               let status_changed = last_status != result.status;
               
               if status_changed {
                   // Trigger alerts for status changes
                   if health_check.is_critical() || result.status == HealthStatus::Unhealthy {
                       let alert = HealthAlert {
                           component_name: component_name.clone(),
                           old_status: last_status.clone(),
                           new_status: result.status.clone(),
                           message: result.message.clone().unwrap_or_default(),
                           is_critical: health_check.is_critical(),
                           timestamp: result.timestamp,
                       };
                       
                       alert_manager.send_health_alert(alert).await;
                   }
                   
                   last_status = result.status.clone();
               }
               
               // Check thresholds
               if consecutive_failures >= config.failure_threshold {
                   tracing::error!(
                       "Component {} has failed {} consecutive health checks",
                       component_name,
                       consecutive_failures
                   );
               }
               
               // Store result
               {
                   let mut results_map = results.write().await;
                   results_map.insert(component_name.clone(), result);
               }
           }
       }
   }
   ```

3. Implement specific health checks for graph algorithm components:
   ```rust
   pub struct GraphAlgorithmHealthCheck {
       algorithm_registry: Arc<AlgorithmRegistry>,
       test_graph: Arc<TestGraph>,
       performance_thresholds: PerformanceThresholds,
   }
   
   #[derive(Debug, Clone)]
   pub struct PerformanceThresholds {
       pub max_execution_time_ms: u64,
       pub max_memory_usage_mb: f64,
       pub min_success_rate: f64,
   }
   
   #[async_trait]
   impl HealthCheck for GraphAlgorithmHealthCheck {
       async fn check_health(&self) -> HealthCheckResult {
           let start_time = Instant::now();
           let mut details = HashMap::new();
           let mut status = HealthStatus::Healthy;
           let mut messages = Vec::new();
           
           // Check algorithm registry availability
           match self.algorithm_registry.list_algorithms().await {
               Ok(algorithms) => {
                   details.insert("available_algorithms".to_string(), 
                                 serde_json::Value::Number(algorithms.len().into()));
                   
                   if algorithms.is_empty() {
                       status = HealthStatus::Degraded;
                       messages.push("No algorithms registered".to_string());
                   }
               }
               Err(e) => {
                   status = HealthStatus::Unhealthy;
                   details.insert("registry_error".to_string(), 
                                 serde_json::Value::String(e.to_string()));
                   return HealthCheckResult {
                       component_name: self.component_name().to_string(),
                       status,
                       message: Some("Algorithm registry unavailable".to_string()),
                       details,
                       check_duration: start_time.elapsed(),
                       timestamp: Utc::now(),
                       error: Some(e.to_string()),
                   };
               }
           }
           
           // Run test algorithm execution
           match self.run_test_algorithm().await {
               Ok(test_result) => {
                   details.insert("test_execution_time_ms".to_string(),
                                 serde_json::Value::Number(test_result.execution_time_ms.into()));
                   details.insert("test_memory_usage_mb".to_string(),
                                 serde_json::Value::Number(
                                     serde_json::Number::from_f64(test_result.memory_usage_mb).unwrap()
                                 ));
                   
                   // Check performance thresholds
                   if test_result.execution_time_ms > self.performance_thresholds.max_execution_time_ms {
                       status = HealthStatus::Degraded;
                       messages.push(format!(
                           "Test execution time {}ms exceeds threshold {}ms",
                           test_result.execution_time_ms,
                           self.performance_thresholds.max_execution_time_ms
                       ));
                   }
                   
                   if test_result.memory_usage_mb > self.performance_thresholds.max_memory_usage_mb {
                       status = HealthStatus::Degraded;
                       messages.push(format!(
                           "Memory usage {:.2}MB exceeds threshold {:.2}MB",
                           test_result.memory_usage_mb,
                           self.performance_thresholds.max_memory_usage_mb
                       ));
                   }
               }
               Err(e) => {
                   status = HealthStatus::Unhealthy;
                   messages.push(format!("Test algorithm execution failed: {}", e));
               }
           }
           
           HealthCheckResult {
               component_name: self.component_name().to_string(),
               status,
               message: if messages.is_empty() { None } else { Some(messages.join("; ")) },
               details,
               check_duration: start_time.elapsed(),
               timestamp: Utc::now(),
               error: None,
           }
       }
       
       fn component_name(&self) -> &str {
           "graph_algorithms"
       }
       
       fn check_interval(&self) -> Duration {
           Duration::from_secs(30)
       }
       
       fn timeout(&self) -> Duration {
           Duration::from_secs(10)
       }
       
       fn is_critical(&self) -> bool {
           true
       }
   }
   
   impl GraphAlgorithmHealthCheck {
       async fn run_test_algorithm(&self) -> Result<TestExecutionResult, HealthError> {
           let start_time = Instant::now();
           let initial_memory = self.get_memory_usage().await;
           
           // Run a simple Dijkstra test
           let dijkstra = DijkstraAlgorithm::new();
           let result = dijkstra.execute(&self.test_graph, 0, 5).await
               .map_err(|e| HealthError::TestExecutionFailed(e.to_string()))?;
           
           let execution_time = start_time.elapsed();
           let peak_memory = self.get_memory_usage().await;
           
           Ok(TestExecutionResult {
               execution_time_ms: execution_time.as_millis() as u64,
               memory_usage_mb: (peak_memory - initial_memory) as f64 / 1024.0 / 1024.0,
               result_valid: result.is_some(),
           })
       }
   }
   ```

4. Implement dependency health tracking:
   ```rust
   #[derive(Debug, Clone, Serialize, Deserialize)]
   pub struct DependencyHealth {
       pub name: String,
       pub status: HealthStatus,
       pub response_time_ms: Option<u64>,
       pub last_check: DateTime<Utc>,
       pub error_rate: f64,
       pub availability: f64,
   }
   
   pub struct DependencyTracker {
       dependencies: Arc<RwLock<HashMap<String, DependencyMonitor>>>,
       config: DependencyConfig,
   }
   
   #[derive(Debug, Clone)]
   pub struct DependencyConfig {
       pub check_interval: Duration,
       pub timeout: Duration,
       pub error_rate_threshold: f64,
       pub availability_threshold: f64,
   }
   
   pub struct DependencyMonitor {
       name: String,
       endpoint: String,
       check_type: DependencyCheckType,
       last_checks: Vec<DependencyCheckResult>,
       max_history: usize,
   }
   
   #[derive(Debug, Clone)]
   pub enum DependencyCheckType {
       Database { connection_string: String },
       Redis { host: String, port: u16 },
       Http { url: String, expected_status: u16 },
       MessageQueue { broker_url: String },
       FileSystem { path: String },
   }
   
   impl DependencyTracker {
       pub async fn check_database_dependency(&self, name: &str, connection_string: &str) -> DependencyHealth {
           let start_time = Instant::now();
           
           match sqlx::PgPool::connect(connection_string).await {
               Ok(pool) => {
                   match sqlx::query("SELECT 1").fetch_one(&pool).await {
                       Ok(_) => DependencyHealth {
                           name: name.to_string(),
                           status: HealthStatus::Healthy,
                           response_time_ms: Some(start_time.elapsed().as_millis() as u64),
                           last_check: Utc::now(),
                           error_rate: 0.0,
                           availability: 100.0,
                       },
                       Err(e) => DependencyHealth {
                           name: name.to_string(),
                           status: HealthStatus::Unhealthy,
                           response_time_ms: Some(start_time.elapsed().as_millis() as u64),
                           last_check: Utc::now(),
                           error_rate: 100.0,
                           availability: 0.0,
                       }
                   }
               }
               Err(_) => DependencyHealth {
                   name: name.to_string(),
                   status: HealthStatus::Unhealthy,
                   response_time_ms: None,
                   last_check: Utc::now(),
                   error_rate: 100.0,
                   availability: 0.0,
               }
           }
       }
       
       pub async fn check_redis_dependency(&self, name: &str, host: &str, port: u16) -> DependencyHealth {
           let start_time = Instant::now();
           let redis_url = format!("redis://{}:{}", host, port);
           
           match redis::Client::open(redis_url) {
               Ok(client) => {
                   match client.get_connection() {
                       Ok(mut conn) => {
                           match redis::cmd("PING").query::<String>(&mut conn) {
                               Ok(response) if response == "PONG" => DependencyHealth {
                                   name: name.to_string(),
                                   status: HealthStatus::Healthy,
                                   response_time_ms: Some(start_time.elapsed().as_millis() as u64),
                                   last_check: Utc::now(),
                                   error_rate: 0.0,
                                   availability: 100.0,
                               },
                               _ => DependencyHealth {
                                   name: name.to_string(),
                                   status: HealthStatus::Unhealthy,
                                   response_time_ms: Some(start_time.elapsed().as_millis() as u64),
                                   last_check: Utc::now(),
                                   error_rate: 100.0,
                                   availability: 0.0,
                               }
                           }
                       }
                       Err(_) => DependencyHealth {
                           name: name.to_string(),
                           status: HealthStatus::Unhealthy,
                           response_time_ms: None,
                           last_check: Utc::now(),
                           error_rate: 100.0,
                           availability: 0.0,
                       }
                   }
               }
               Err(_) => DependencyHealth {
                   name: name.to_string(),
                   status: HealthStatus::Unhealthy,
                   response_time_ms: None,
                   last_check: Utc::now(),
                   error_rate: 100.0,
                   availability: 0.0,
               }
           }
       }
   }
   ```

5. Implement readiness and liveness probes:
   ```rust
   use axum::{
       extract::State,
       http::StatusCode,
       response::Json,
       routing::get,
       Router,
   };
   
   pub struct HealthEndpoints {
       health_monitor: Arc<HealthMonitor>,
   }
   
   impl HealthEndpoints {
       pub fn new(health_monitor: Arc<HealthMonitor>) -> Self {
           Self { health_monitor }
       }
       
       pub fn routes(&self) -> Router {
           Router::new()
               .route("/health", get(Self::health_check))
               .route("/health/live", get(Self::liveness_check))
               .route("/health/ready", get(Self::readiness_check))
               .route("/health/components", get(Self::component_health))
               .with_state(Arc::new(self.clone()))
       }
       
       async fn health_check(
           State(endpoints): State<Arc<HealthEndpoints>>,
       ) -> Result<Json<SystemHealth>, StatusCode> {
           match endpoints.health_monitor.get_system_health().await {
               Ok(health) => {
                   let status_code = match health.overall_status {
                       HealthStatus::Healthy => StatusCode::OK,
                       HealthStatus::Degraded => StatusCode::OK, // Still serving traffic
                       HealthStatus::Unhealthy => StatusCode::SERVICE_UNAVAILABLE,
                       HealthStatus::Unknown => StatusCode::SERVICE_UNAVAILABLE,
                   };
                   Ok(Json(health))
               }
               Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
           }
       }
       
       async fn liveness_check(
           State(endpoints): State<Arc<HealthEndpoints>>,
       ) -> StatusCode {
           // Liveness check - basic process health
           // This should only fail if the process is completely broken
           
           let basic_checks = vec![
               endpoints.check_memory_usage().await,
               endpoints.check_thread_health().await,
               endpoints.check_basic_functionality().await,
           ];
           
           let all_passed = basic_checks.iter().all(|&check| check);
           
           if all_passed {
               StatusCode::OK
           } else {
               StatusCode::SERVICE_UNAVAILABLE
           }
       }
       
       async fn readiness_check(
           State(endpoints): State<Arc<HealthEndpoints>>,
       ) -> StatusCode {
           // Readiness check - ability to serve traffic
           // This should fail if dependencies are down or system is overloaded
           
           match endpoints.health_monitor.get_system_health().await {
               Ok(health) => {
                   let critical_components_healthy = health.components
                       .values()
                       .filter(|result| {
                           // Assume critical components have specific names or patterns
                           result.component_name.contains("database") ||
                           result.component_name.contains("graph_algorithms") ||
                           result.component_name.contains("auth")
                       })
                       .all(|result| matches!(result.status, HealthStatus::Healthy | HealthStatus::Degraded));
                   
                   if critical_components_healthy {
                       StatusCode::OK
                   } else {
                       StatusCode::SERVICE_UNAVAILABLE
                   }
               }
               Err(_) => StatusCode::SERVICE_UNAVAILABLE,
           }
       }
       
       async fn component_health(
           State(endpoints): State<Arc<HealthEndpoints>>,
       ) -> Result<Json<HashMap<String, HealthCheckResult>>, StatusCode> {
           match endpoints.health_monitor.get_component_health().await {
               Ok(components) => Ok(Json(components)),
               Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
           }
       }
   }
   
   impl HealthEndpoints {
       async fn check_memory_usage(&self) -> bool {
           // Check if memory usage is within acceptable limits
           if let Ok(memory_info) = sys_info::mem_info() {
               let used_percentage = (memory_info.total - memory_info.free) * 100 / memory_info.total;
               used_percentage < 95 // Allow up to 95% memory usage
           } else {
               false
           }
       }
       
       async fn check_thread_health(&self) -> bool {
           // Basic thread health check
           tokio::task::yield_now().await;
           true // If we got here, basic async functionality works
       }
       
       async fn check_basic_functionality(&self) -> bool {
           // Test basic system functionality
           let test_data = vec![1, 2, 3, 4, 5];
           let sum: i32 = test_data.iter().sum();
           sum == 15 // Basic arithmetic works
       }
   }
   ```

## Expected Output
```rust
pub trait HealthMonitoring {
    async fn register_health_check(&self, check: Arc<dyn HealthCheck>);
    async fn get_system_health(&self) -> Result<SystemHealth, HealthError>;
    async fn get_component_health(&self, component: &str) -> Result<HealthCheckResult, HealthError>;
    async fn check_dependencies(&self) -> Result<HashMap<String, DependencyHealth>, HealthError>;
}

#[derive(Debug)]
pub enum HealthError {
    ComponentNotFound(String),
    CheckExecutionFailed(String),
    DependencyUnavailable(String),
    TimeoutError,
    ConfigurationError(String),
}

pub struct HealthMetrics {
    pub total_checks: u64,
    pub failed_checks: u64,
    pub average_check_duration: Duration,
    pub component_availability: HashMap<String, f64>,
}
```

## Verification Steps
1. Test health check execution under various failure scenarios
2. Verify dependency health tracking accuracy
3. Test readiness/liveness probe responses
4. Validate health status aggregation logic
5. Test alert triggering for component failures
6. Benchmark health check overhead on system performance

## Time Estimate
25 minutes

## Dependencies
- MP001-MP050: Previous implementations
- axum: Web framework for health endpoints
- tokio: Async runtime
- serde: Serialization support
- sqlx: Database health checks
- redis: Redis health checks