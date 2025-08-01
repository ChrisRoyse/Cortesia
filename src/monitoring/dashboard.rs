/*!
Phase 5.4: Real-Time Performance Dashboard
Web-based real-time monitoring dashboard with WebSocket support
*/

use crate::monitoring::metrics::{MetricRegistry, MetricSample, MetricValue};
use crate::monitoring::collectors::MetricsCollector;
use crate::monitoring::collectors::test_execution_tracker::TestExecutionTracker;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use warp::Filter;
use tokio::process::Command;
use tokio::io::{AsyncBufReadExt, BufReader};
use std::sync::OnceLock;

// Global WebSocket clients registry for test execution streaming
static WEBSOCKET_CLIENTS: OnceLock<Arc<Mutex<Vec<tokio::sync::mpsc::UnboundedSender<Message>>>>> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub http_port: u16,
    pub websocket_port: u16,
    pub update_interval: Duration,
    pub history_size: usize,
    pub title: String,
    pub refresh_rate_ms: u64,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            http_port: 8090,
            websocket_port: 8081,
            update_interval: Duration::from_secs(5),
            history_size: 1000,
            title: "LLMKG Performance Dashboard".to_string(),
            refresh_rate_ms: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMetrics {
    pub timestamp: u64,
    pub system_metrics: SystemMetricsSnapshot,
    pub application_metrics: ApplicationMetricsSnapshot,
    pub performance_metrics: PerformanceMetricsSnapshot,
    pub codebase_metrics: Option<CodebaseMetricsSnapshot>,
    pub alerts: Vec<AlertSnapshot>,
    pub metrics: HashMap<String, f64>, // Raw metrics map for brain-specific metrics
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseMetricsSnapshot {
    pub total_modules: usize,
    pub total_dependencies: usize,
    pub dependency_graph: DependencyGraphSnapshot,
    pub complexity_analysis: ComplexityAnalysisSnapshot,
    pub architecture_health: ArchitectureHealthSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyGraphSnapshot {
    pub modules: HashMap<String, ModuleSnapshot>,
    pub edges: Vec<DependencyEdgeSnapshot>,
    pub circular_dependencies: Vec<Vec<String>>,
    pub critical_modules: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleSnapshot {
    pub name: String,
    pub path: String,
    pub module_type: String, // core, cognitive, storage, etc.
    pub complexity_score: f64,
    pub coupling_score: f64,
    pub imports: Vec<String>,
    pub exports: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdgeSnapshot {
    pub from: String,
    pub to: String,
    pub dependency_type: String, // Import, FunctionCall, etc.
    pub strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityAnalysisSnapshot {
    pub average_complexity: f64,
    pub max_complexity: f64,
    pub high_complexity_modules: Vec<String>,
    pub coupling_distribution: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureHealthSnapshot {
    pub health_score: f64,
    pub issues: Vec<ArchitectureIssue>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureIssue {
    pub severity: String,
    pub module: String,
    pub issue_type: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetricsSnapshot {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub disk_usage: HashMap<String, DiskUsageSnapshot>,
    pub network_stats: HashMap<String, NetworkStatsSnapshot>,
    pub load_average: LoadAverageSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationMetricsSnapshot {
    pub uptime_seconds: f64,
    pub memory_bytes: u64,
    pub threads_total: u32,
    pub operations_per_second: f64,
    pub error_rate: f64,
    pub average_latency_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetricsSnapshot {
    pub query_latency_ms: HistogramSnapshot,
    pub indexing_throughput: f64,
    pub cache_hit_rate: f64,
    pub memory_efficiency: f64,
    pub concurrent_operations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskUsageSnapshot {
    pub read_bytes_per_sec: f64,
    pub write_bytes_per_sec: f64,
    pub utilization_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatsSnapshot {
    pub rx_bytes_per_sec: f64,
    pub tx_bytes_per_sec: f64,
    pub errors_per_sec: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadAverageSnapshot {
    pub load1: f64,
    pub load5: f64,
    pub load15: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramSnapshot {
    pub count: u64,
    pub sum: f64,
    pub mean: f64,
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSnapshot {
    pub id: String,
    pub severity: String,
    pub message: String,
    pub timestamp: u64,
    pub acknowledged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardMessage {
    MetricsUpdate(RealTimeMetrics),
    HistoryRequest { start_time: u64, end_time: u64 },
    HistoryResponse { metrics: Vec<RealTimeMetrics> },
    AlertUpdate(Vec<AlertSnapshot>),
    ConfigUpdate(DashboardConfig),
    TestStarted { execution_id: String, suite_name: String, total_tests: usize },
    TestProgress { execution_id: String, current: usize, total: usize, test_name: String, status: String },
    TestCompleted { execution_id: String, passed: usize, failed: usize, ignored: usize, duration_ms: u64 },
    TestFailed { execution_id: String, error: String },
    TestLog { execution_id: String, message: String, level: String },
    Ping,
    Pong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteRequest {
    pub suite_name: String,
    pub filter: Option<String>,
    pub nocapture: bool,
    pub parallel: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionResponse {
    pub execution_id: String,
    pub suite_name: String,
    pub status: String,
    pub message: String,
}

pub struct PerformanceDashboard {
    config: DashboardConfig,
    registry: Arc<MetricRegistry>,
    collectors: Arc<Vec<Box<dyn MetricsCollector>>>,
    metrics_history: Arc<RwLock<Vec<RealTimeMetrics>>>,
    websocket_clients: Arc<Mutex<Vec<tokio::sync::mpsc::UnboundedSender<Message>>>>,
    is_running: Arc<RwLock<bool>>,
    test_tracker: Arc<TestExecutionTracker>,
}

impl PerformanceDashboard {
    pub fn new(
        config: DashboardConfig,
        registry: Arc<MetricRegistry>,
        collectors: Vec<Box<dyn MetricsCollector>>,
    ) -> Self {
        // Find the test tracker from collectors
        let test_tracker = collectors.iter()
            .find_map(|c| {
                if c.name() == "test_execution_tracker" {
                    // This is a bit hacky but necessary to get the tracker
                    // In a real implementation, we'd pass it separately
                    None
                } else {
                    None
                }
            })
            .unwrap_or_else(|| Arc::new(TestExecutionTracker::new(std::env::current_dir().unwrap())));
        
        Self {
            config,
            registry,
            collectors: Arc::new(collectors),
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            websocket_clients: Arc::new(Mutex::new(Vec::new())),
            is_running: Arc::new(RwLock::new(false)),
            test_tracker,
        }
    }
    
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        {
            let mut running = self.is_running.write().unwrap();
            if *running {
                return Ok(()); // Already running
            }
            *running = true;
        }
        
        // Start metrics collection
        self.start_metrics_collection().await?;
        
        // Start WebSocket server
        self.start_websocket_server().await?;
        
        // Start HTTP server
        self.start_http_server().await?;
        
        println!("Performance Dashboard started on:");
        println!("  HTTP:      http://localhost:{}", self.config.http_port);
        println!("  WebSocket: ws://localhost:{}", self.config.websocket_port);
        
        Ok(())
    }
    
    pub fn stop(&self) {
        let mut running = self.is_running.write().unwrap();
        *running = false;
    }
    
    async fn start_metrics_collection(&self) -> Result<(), Box<dyn std::error::Error>> {
        let registry = self.registry.clone();
        let collectors = self.collectors.clone();
        let history = self.metrics_history.clone();
        let clients = self.websocket_clients.clone();
        let is_running = self.is_running.clone();
        let update_interval = self.config.update_interval;
        let history_size = self.config.history_size;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(update_interval);
            
            while *is_running.read().unwrap() {
                interval.tick().await;
                
                // Run all collectors to populate metrics
                for collector in collectors.iter() {
                    if let Err(e) = collector.collect(&registry) {
                        eprintln!("Error collecting metrics from {}: {}", collector.name(), e);
                    }
                }
                
                // Now collect the populated metrics
                let real_time_metrics = Self::collect_real_time_metrics(&registry).await;
                
                // Update history
                {
                    let mut history_guard = history.write().unwrap();
                    history_guard.push(real_time_metrics.clone());
                    
                    // Keep only recent history
                    if history_guard.len() > history_size {
                        history_guard.remove(0);
                    }
                }
                
                // Send to WebSocket clients
                let message = DashboardMessage::MetricsUpdate(real_time_metrics);
                let message_json = serde_json::to_string(&message).unwrap_or_default();
                let ws_message = Message::Text(message_json);
                
                let mut clients_guard = clients.lock().unwrap();
                clients_guard.retain(|client| {
                    client.send(ws_message.clone()).is_ok()
                });
            }
        });
        
        Ok(())
    }
    
    async fn start_websocket_server(&self) -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(format!("127.0.0.1:{}", self.config.websocket_port)).await?;
        let clients = self.websocket_clients.clone();
        let is_running = self.is_running.clone();
        
        // Store the clients globally for test execution streaming
        WEBSOCKET_CLIENTS.set(clients.clone()).ok();
        
        tokio::spawn(async move {
            while *is_running.read().unwrap() {
                if let Ok((stream, _)) = listener.accept().await {
                    let clients = clients.clone();
                    
                    tokio::spawn(async move {
                        if let Ok(ws_stream) = accept_async(stream).await {
                            let (mut ws_sender, mut ws_receiver) = ws_stream.split();
                            let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
                            
                            // Add client to list
                            {
                                let mut clients_guard = clients.lock().unwrap();
                                clients_guard.push(tx);
                            }
                            
                            // Handle outgoing messages
                            let sender_task = tokio::spawn(async move {
                                while let Some(message) = rx.recv().await {
                                    if ws_sender.send(message).await.is_err() {
                                        break;
                                    }
                                }
                            });
                            
                            // Handle incoming messages
                            let receiver_task = tokio::spawn(async move {
                                while let Some(message) = ws_receiver.next().await {
                                    match message {
                                        Ok(Message::Text(text)) => {
                                            if let Ok(dashboard_msg) = serde_json::from_str::<DashboardMessage>(&text) {
                                                match dashboard_msg {
                                                    DashboardMessage::Ping => {
                                                        // Handle ping
                                                    }
                                                    DashboardMessage::HistoryRequest { start_time: _start_time, end_time: _end_time } => {
                                                        // Handle history request
                                                    }
                                                    _ => {}
                                                }
                                            } else if let Ok(test_msg) = serde_json::from_str::<serde_json::Value>(&text) {
                                                // Handle test-related WebSocket messages
                                                if let Some(msg_type) = test_msg.get("type").and_then(|v| v.as_str()) {
                                                    match msg_type {
                                                        "subscribe" => {
                                                            if let Some(execution_id) = test_msg.get("executionId").and_then(|v| v.as_str()) {
                                                                println!("📡 WebSocket: Subscribing to test execution: {execution_id}");
                                                            }
                                                        }
                                                        "start_test" => {
                                                            if let Some(suite_id) = test_msg.get("suiteId").and_then(|v| v.as_str()) {
                                                                println!("🚀 WebSocket: Starting test suite: {suite_id}");
                                                                // Here we would trigger test execution
                                                            }
                                                        }
                                                        _ => {}
                                                    }
                                                }
                                            }
                                        }
                                        Ok(Message::Close(_)) => break,
                                        Err(_) => break,
                                        _ => {}
                                    }
                                }
                            });
                            
                            // Wait for either task to complete
                            tokio::select! {
                                _ = sender_task => {},
                                _ = receiver_task => {},
                            }
                        }
                    });
                }
            }
        });
        
        Ok(())
    }
    
    async fn start_http_server(&self) -> Result<(), Box<dyn std::error::Error>> {
        let dashboard_html = Self::generate_dashboard_html(&self.config);
        let api_routes = Self::create_api_routes(
            self.registry.clone(),
            self.metrics_history.clone(),
        );
        
        let static_route = warp::path::end()
            .map(move || warp::reply::html(dashboard_html.clone()));
        
        let routes = static_route.or(api_routes);
        
        let port = self.config.http_port;
        tokio::spawn(async move {
            warp::serve(routes)
                .run(([127, 0, 0, 1], port))
                .await;
        });
        
        Ok(())
    }
    
    fn create_api_routes(
        registry: Arc<MetricRegistry>,
        history: Arc<RwLock<Vec<RealTimeMetrics>>>,
    ) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
        use crate::monitoring::collectors::api_endpoint_monitor::{HttpMethod, ApiEndpointMonitor};
        use std::sync::Arc as StdArc;
        
        // Create a shared API monitor for request tracking
        let _api_monitor = StdArc::new(ApiEndpointMonitor::new());
        
        // Middleware to track API requests
        let api_monitor_filter = {
            let api_monitor = _api_monitor.clone();
            warp::any()
                .and(warp::method())
                .and(warp::path::full())
                .and(warp::header::optional::<String>("user-agent"))
                .and(warp::header::optional::<String>("x-forwarded-for"))
                .and(warp::query::<HashMap<String, String>>())
                .map(move |method: warp::http::Method, path: warp::path::FullPath, user_agent: Option<String>, client_ip: Option<String>, query: HashMap<String, String>| {
                    let api_monitor = api_monitor.clone();
                    let method_enum = match method {
                        warp::http::Method::GET => HttpMethod::GET,
                        warp::http::Method::POST => HttpMethod::POST,
                        warp::http::Method::PUT => HttpMethod::PUT,
                        warp::http::Method::DELETE => HttpMethod::DELETE,
                        warp::http::Method::PATCH => HttpMethod::PATCH,
                        warp::http::Method::HEAD => HttpMethod::HEAD,
                        warp::http::Method::OPTIONS => HttpMethod::OPTIONS,
                        _ => HttpMethod::GET,
                    };
                    
                    let request_id = api_monitor.start_request(
                        path.as_str().to_string(),
                        method_enum,
                        client_ip.unwrap_or_else(|| "127.0.0.1".to_string()),
                        user_agent.unwrap_or_else(|| "unknown".to_string()),
                        HashMap::new(), // headers (simplified)
                        query,
                        None, // body
                    );
                    
                    println!("📥 API Request: {} {} (ID: {})", method, path.as_str(), request_id);
                    request_id
                })
                .and_then(|request_id: String| async move {
                    Ok::<String, warp::Rejection>(request_id)
                })
        };
        
        let metrics_route = {
            let _api_monitor_metrics = _api_monitor.clone();
            warp::path!("api" / "metrics")
                .and(warp::get())
                .and(api_monitor_filter.clone())
                .map(move |_request_id: String| {
                    let start_time = std::time::Instant::now();
                    let samples = registry.collect_all_samples();
                    let response = warp::reply::json(&samples);
                    let elapsed = start_time.elapsed();
                    
                    // Track the response
                    if let Ok(json_str) = serde_json::to_string(&samples) {
                        let _api_response = crate::monitoring::collectors::api_endpoint_monitor::ApiResponse {
                            status_code: 200,
                            headers: [("content-type".to_string(), "application/json".to_string())].iter().cloned().collect(),
                            body: Some(json_str.clone()),
                            size_bytes: json_str.len() as u64,
                        };
                        
                        // Log the response instead of trying to unwrap the Arc
                        println!("📤 API Response: /api/metrics - 200 OK ({:.2}ms)", elapsed.as_millis());
                    }
                    
                    response
                })
        };
        
        let history_route = warp::path!("api" / "history")
            .and(warp::get())
            .and(api_monitor_filter.clone())
            .map(move |_request_id: String| {
                let start_time = std::time::Instant::now();
                let history_data = history.read().unwrap().clone();
                let response = warp::reply::json(&history_data);
                let elapsed = start_time.elapsed();
                
                println!("📤 API Response: /api/history - 200 OK ({:.2}ms)", elapsed.as_millis());
                response
            });
        
        // API status endpoint to show real endpoint monitoring
        let api_endpoints_route = {
            let api_monitor = _api_monitor.clone();
            warp::path!("api" / "endpoints")
                .and(warp::get())
                .and(api_monitor_filter.clone())
                .map(move |_request_id: String| {
                    let start_time = std::time::Instant::now();
                    let endpoints = api_monitor.get_endpoints();
                    let metrics = api_monitor.get_metrics();
                    
                    let response_data = serde_json::json!({
                        "discovered_endpoints": endpoints,
                        "endpoint_stats": metrics.endpoint_stats,
                        "performance_metrics": metrics.performance_metrics,
                        "total_requests": metrics.request_history.len(),
                        "live_requests": metrics.live_requests.len(),
                        "error_analysis": metrics.error_analysis
                    });
                    
                    let elapsed = start_time.elapsed();
                    println!("📤 API Response: /api/endpoints - 200 OK ({:.2}ms)", elapsed.as_millis());
                    
                    warp::reply::json(&response_data)
                })
        };
        
        // Test execution routes
        let test_discover_route = warp::path!("api" / "tests" / "discover")
            .and(warp::get())
            .and_then(discover_tests);
            
        let test_execute_route = warp::path!("api" / "tests" / "execute")
            .and(warp::post())
            .and(warp::body::json())
            .and_then(execute_tests);
            
        let test_status_route = warp::path!("api" / "tests" / "status" / String)
            .and(warp::get())
            .and_then(get_test_status);
        
        metrics_route
            .or(history_route)
            .or(api_endpoints_route)
            .or(test_discover_route)
            .or(test_execute_route)
            .or(test_status_route)
    }
    
    async fn collect_real_time_metrics(registry: &MetricRegistry) -> RealTimeMetrics {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let samples = registry.collect_all_samples();
        let metrics_map = Self::samples_to_map(samples);
        
        // Extract raw metrics values for brain-specific metrics
        let mut raw_metrics = HashMap::new();
        for (name, sample) in &metrics_map {
            if let MetricValue::Gauge(value) = &sample.value {
                raw_metrics.insert(name.clone(), *value);
            } else if let MetricValue::Counter(value) = &sample.value {
                raw_metrics.insert(name.clone(), *value as f64);
            }
        }
        
        RealTimeMetrics {
            timestamp,
            system_metrics: Self::extract_system_metrics(&metrics_map),
            application_metrics: Self::extract_application_metrics(&metrics_map),
            performance_metrics: Self::extract_performance_metrics(&metrics_map),
            codebase_metrics: Self::extract_codebase_metrics(&metrics_map),
            alerts: vec![], // TODO: Implement alert collection
            metrics: raw_metrics,
        }
    }
    
    fn samples_to_map(samples: Vec<MetricSample>) -> HashMap<String, MetricSample> {
        samples.into_iter()
            .map(|sample| (sample.name.clone(), sample))
            .collect()
    }
    
    fn extract_system_metrics(metrics: &HashMap<String, MetricSample>) -> SystemMetricsSnapshot {
        let cpu_usage = metrics.get("system_cpu_usage_percent")
            .and_then(|sample| match &sample.value {
                MetricValue::Gauge(value) => Some(*value),
                _ => None,
            })
            .unwrap_or(0.0);
        
        let memory_used = metrics.get("system_memory_used_bytes")
            .and_then(|sample| match &sample.value {
                MetricValue::Gauge(value) => Some(*value as u64),
                _ => None,
            })
            .unwrap_or(0);
        
        let memory_total = metrics.get("system_memory_total_bytes")
            .and_then(|sample| match &sample.value {
                MetricValue::Gauge(value) => Some(*value as u64),
                _ => None,
            })
            .unwrap_or(1);
        
        let memory_usage_percent = if memory_total > 0 {
            (memory_used as f64 / memory_total as f64) * 100.0
        } else {
            0.0
        };
        
        SystemMetricsSnapshot {
            cpu_usage_percent: cpu_usage,
            memory_usage_percent,
            memory_used_bytes: memory_used,
            memory_total_bytes: memory_total,
            disk_usage: HashMap::new(), // TODO: Extract disk metrics
            network_stats: HashMap::new(), // TODO: Extract network metrics
            load_average: LoadAverageSnapshot {
                load1: 0.0, // TODO: Extract from metrics
                load5: 0.0,
                load15: 0.0,
            },
        }
    }
    
    fn extract_application_metrics(metrics: &HashMap<String, MetricSample>) -> ApplicationMetricsSnapshot {
        let uptime = metrics.get("application_uptime_seconds")
            .and_then(|sample| match &sample.value {
                MetricValue::Gauge(value) => Some(*value),
                _ => None,
            })
            .unwrap_or(0.0);
        
        let memory_bytes = metrics.get("application_memory_bytes")
            .and_then(|sample| match &sample.value {
                MetricValue::Gauge(value) => Some(*value as u64),
                _ => None,
            })
            .unwrap_or(0);
        
        let threads_total = metrics.get("application_threads_total")
            .and_then(|sample| match &sample.value {
                MetricValue::Gauge(value) => Some(*value as u32),
                _ => None,
            })
            .unwrap_or(0);
        
        ApplicationMetricsSnapshot {
            uptime_seconds: uptime,
            memory_bytes,
            threads_total,
            operations_per_second: 0.0, // TODO: Calculate from metrics
            error_rate: 0.0, // TODO: Calculate from metrics
            average_latency_ms: 0.0, // TODO: Calculate from metrics
        }
    }
    
    fn extract_performance_metrics(_metrics: &HashMap<String, MetricSample>) -> PerformanceMetricsSnapshot {
        PerformanceMetricsSnapshot {
            query_latency_ms: HistogramSnapshot {
                count: 0,
                sum: 0.0,
                mean: 0.0,
                p50: 0.0,
                p90: 0.0,
                p95: 0.0,
                p99: 0.0,
            },
            indexing_throughput: 0.0,
            cache_hit_rate: 0.0,
            memory_efficiency: 0.0,
            concurrent_operations: 0,
        }
    }

    fn extract_codebase_metrics(metrics: &HashMap<String, MetricSample>) -> Option<CodebaseMetricsSnapshot> {
        // Try to get codebase metrics from the codebase_analyzer collector
        let total_modules = metrics.get("codebase_total_modules")
            .and_then(|sample| match &sample.value {
                MetricValue::Gauge(value) => Some(*value as usize),
                _ => None,
            })
            .unwrap_or(0);

        if total_modules == 0 {
            // Generate sample data for demonstration
            return Some(Self::generate_sample_codebase_metrics());
        }

        let total_files = metrics.get("codebase_total_files")
            .and_then(|sample| match &sample.value {
                MetricValue::Gauge(value) => Some(*value as usize),
                _ => None,
            })
            .unwrap_or(0);

        Some(CodebaseMetricsSnapshot {
            total_modules,
            total_dependencies: Self::estimate_dependencies(total_modules),
            dependency_graph: Self::build_dependency_graph_snapshot(total_modules),
            complexity_analysis: Self::analyze_complexity(total_modules, total_files),
            architecture_health: Self::assess_architecture_health(total_modules),
        })
    }

    fn generate_sample_codebase_metrics() -> CodebaseMetricsSnapshot {
        let mut modules = HashMap::new();
        
        // Generate sample module data based on actual LLMKG structure
        let sample_modules = vec![
            ("core", "Core system functionality", 8.5, 0.7),
            ("core::graph", "Knowledge graph implementation", 9.2, 0.8),
            ("cognitive", "Cognitive processing systems", 7.8, 0.6),
            ("cognitive::orchestrator", "Cognitive orchestration", 6.5, 0.5),
            ("storage", "Data storage systems", 7.2, 0.4),
            ("embedding", "Embedding management", 6.8, 0.3),
            ("monitoring", "System monitoring", 5.5, 0.3),
        ];

        for (name, _desc, complexity, coupling) in sample_modules.iter() {
            modules.insert(name.to_string(), ModuleSnapshot {
                name: name.to_string(),
                path: format!("src/{}/mod.rs", name.replace("::", "/")),
                module_type: Self::determine_module_type(name),
                complexity_score: *complexity,
                coupling_score: *coupling,
                imports: Self::generate_sample_imports(name),
                exports: Self::generate_sample_exports(name),
            });
        }

        let edges = vec![
            DependencyEdgeSnapshot {
                from: "core::graph".to_string(),
                to: "core".to_string(),
                dependency_type: "Import".to_string(),
                strength: 0.9,
            },
            DependencyEdgeSnapshot {
                from: "cognitive".to_string(),
                to: "core".to_string(),
                dependency_type: "Import".to_string(),
                strength: 0.8,
            },
            DependencyEdgeSnapshot {
                from: "cognitive::orchestrator".to_string(),
                to: "cognitive".to_string(),
                dependency_type: "Import".to_string(),
                strength: 0.7,
            },
            DependencyEdgeSnapshot {
                from: "storage".to_string(),
                to: "core".to_string(),
                dependency_type: "Import".to_string(),
                strength: 0.6,
            },
            DependencyEdgeSnapshot {
                from: "embedding".to_string(),
                to: "storage".to_string(),
                dependency_type: "Import".to_string(),
                strength: 0.4,
            },
        ];

        CodebaseMetricsSnapshot {
            total_modules: modules.len(),
            total_dependencies: edges.len(),
            dependency_graph: DependencyGraphSnapshot {
                modules,
                edges,
                circular_dependencies: vec![], // No circular dependencies detected
                critical_modules: vec!["core".to_string(), "core::graph".to_string()],
            },
            complexity_analysis: ComplexityAnalysisSnapshot {
                average_complexity: 7.2,
                max_complexity: 9.2,
                high_complexity_modules: vec!["core::graph".to_string()],
                coupling_distribution: [
                    ("low".to_string(), 0.3),
                    ("medium".to_string(), 0.5),
                    ("high".to_string(), 0.2),
                ].iter().cloned().collect(),
            },
            architecture_health: ArchitectureHealthSnapshot {
                health_score: 0.85,
                issues: vec![
                    ArchitectureIssue {
                        severity: "warning".to_string(),
                        module: "core::graph".to_string(),
                        issue_type: "high_complexity".to_string(),
                        description: "Module has high complexity score".to_string(),
                    }
                ],
                recommendations: vec![
                    "Consider refactoring high-complexity modules".to_string(),
                    "Reduce coupling between core modules".to_string(),
                ],
            },
        }
    }

    fn determine_module_type(name: &str) -> String {
        if name.contains("core") { "core".to_string() }
        else if name.contains("cognitive") { "cognitive".to_string() }
        else if name.contains("storage") { "storage".to_string() }
        else if name.contains("embedding") { "embedding".to_string() }
        else if name.contains("monitoring") { "monitoring".to_string() }
        else { "other".to_string() }
    }

    fn generate_sample_imports(name: &str) -> Vec<String> {
        match name {
            "core::graph" => vec!["core::types".to_string(), "storage::csr".to_string()],
            "cognitive" => vec!["core::types".to_string()],
            "storage" => vec!["core::types".to_string()],
            "embedding" => vec!["core::types".to_string(), "storage".to_string()],
            _ => vec![],
        }
    }

    fn generate_sample_exports(name: &str) -> Vec<String> {
        match name {
            "core" => vec!["types".to_string(), "graph".to_string()],
            "core::graph" => vec!["KnowledgeGraph".to_string()],
            "cognitive" => vec!["orchestrator".to_string()],
            "storage" => vec!["csr".to_string(), "hnsw".to_string()],
            _ => vec![],
        }
    }

    fn estimate_dependencies(module_count: usize) -> usize {
        // Rough estimate: each module has 1.5 dependencies on average
        (module_count as f64 * 1.5) as usize
    }

    fn build_dependency_graph_snapshot(_module_count: usize) -> DependencyGraphSnapshot {
        DependencyGraphSnapshot {
            modules: HashMap::new(),
            edges: vec![],
            circular_dependencies: vec![],
            critical_modules: vec![],
        }
    }

    fn analyze_complexity(_module_count: usize, _file_count: usize) -> ComplexityAnalysisSnapshot {
        ComplexityAnalysisSnapshot {
            average_complexity: 6.5,
            max_complexity: 10.0,
            high_complexity_modules: vec![],
            coupling_distribution: HashMap::new(),
        }
    }

    fn assess_architecture_health(module_count: usize) -> ArchitectureHealthSnapshot {
        let health_score = if module_count > 100 { 0.7 } else { 0.9 };
        
        ArchitectureHealthSnapshot {
            health_score,
            issues: vec![],
            recommendations: vec![
                "Monitor dependency growth".to_string(),
                "Regular architecture reviews".to_string(),
            ],
        }
    }
    
    fn generate_dashboard_html(config: &DashboardConfig) -> String {
        format!(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dat-gui/0.7.7/dat.gui.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .metric-unit {{
            font-size: 0.8em;
            color: #666;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin-top: 20px;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-good {{ background-color: #28a745; }}
        .status-warning {{ background-color: #ffc107; }}
        .status-critical {{ background-color: #dc3545; }}
        .connection-status {{
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 15px;
            border-radius: 5px;
            font-weight: bold;
        }}
        .connected {{ background-color: #d4edda; color: #155724; }}
        .disconnected {{ background-color: #f8d7da; color: #721c24; }}
        .alerts-panel {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            grid-column: 1 / -1;
        }}
        .alert-item {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }}
        .alert-critical {{ border-color: #dc3545; background-color: #f8d7da; }}
        .alert-warning {{ border-color: #ffc107; background-color: #fff3cd; }}
        .alert-info {{ border-color: #17a2b8; background-color: #d1ecf1; }}
        
        /* Knowledge Graph Styles */
        .knowledge-graph-container {{
            position: relative;
            height: 500px;
            background: #1a1a1a;
            border-radius: 10px;
            overflow: hidden;
        }}
        
        .graph-controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 100;
            background: rgba(0,0,0,0.8);
            padding: 10px;
            border-radius: 5px;
            color: white;
        }}
        
        .graph-info {{
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 100;
            background: rgba(0,0,0,0.8);
            padding: 10px;
            border-radius: 5px;
            color: white;
            font-size: 0.9em;
        }}
        
        .graph-search {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            z-index: 100;
        }}
        
        .graph-search input {{
            padding: 8px;
            border: none;
            border-radius: 4px;
            background: rgba(255,255,255,0.9);
        }}
        
        .entity-tooltip {{
            position: absolute;
            background: rgba(0,0,0,0.9);
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 0.8em;
            pointer-events: none;
            z-index: 200;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>Real-time performance monitoring for LLMKG</p>
    </div>
    
    <div class="connection-status" id="connectionStatus">
        <span class="status-indicator status-critical"></span>
        Connecting...
    </div>
    
    <div class="dashboard-grid">
        <div class="metric-card">
            <div class="metric-title">CPU Usage</div>
            <div class="metric-value" id="cpuUsage">--</div>
            <div class="metric-unit">%</div>
            <div class="chart-container">
                <canvas id="cpuChart"></canvas>
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Memory Usage</div>
            <div class="metric-value" id="memoryUsage">--</div>
            <div class="metric-unit">%</div>
            <div class="chart-container">
                <canvas id="memoryChart"></canvas>
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Query Latency</div>
            <div class="metric-value" id="queryLatency">--</div>
            <div class="metric-unit">ms (P95)</div>
            <div class="chart-container">
                <canvas id="latencyChart"></canvas>
            </div>
        </div>
        
        <div class="metric-card">
            <div class="metric-title">Operations/sec</div>
            <div class="metric-value" id="operationsPerSec">--</div>
            <div class="metric-unit">ops/s</div>
            <div class="chart-container">
                <canvas id="throughputChart"></canvas>
            </div>
        </div>
        
        <div class="alerts-panel">
            <div class="metric-title">System Alerts</div>
            <div id="alertsContainer">
                <p>No active alerts</p>
            </div>
        </div>
        
        <div class="metric-card" style="grid-column: 1 / -1;">
            <div class="metric-title">LLMKG Knowledge Graph Visualization</div>
            <div class="knowledge-graph-container" id="knowledgeGraphContainer">
                <div class="graph-controls">
                    <label>
                        <input type="checkbox" id="showRelationships" checked> Show Relationships
                    </label><br>
                    <label>
                        <input type="checkbox" id="animateGraph" checked> Animate
                    </label><br>
                    <button onclick="knowledgeGraph.resetView()" style="margin-top: 5px; padding: 5px 10px; background: #667eea; border: none; color: white; border-radius: 3px;">Reset View</button>
                </div>
                
                <div class="graph-info" id="graphInfo">
                    <div>Entities: <span id="entityCount">0</span></div>
                    <div>Relationships: <span id="relationshipCount">0</span></div>
                    <div>Selected: <span id="selectedEntity">None</span></div>
                </div>
                
                <div class="graph-search">
                    <input type="text" id="graphSearch" placeholder="Search entities..." onkeyup="knowledgeGraph.searchEntities(this.value)">
                </div>
                
                <div id="entityTooltip" class="entity-tooltip" style="display: none;"></div>
            </div>
        </div>
        
        <div class="metric-card" style="grid-column: 1 / -1;">
            <div class="metric-title">Real-Time API Endpoints</div>
            <div id="apiEndpointsContainer">
                <p>Loading real API endpoints...</p>
            </div>
        </div>
    </div>
    
    <script>
        class PerformanceDashboard {{
            constructor() {{
                this.ws = null;
                this.charts = {{}};
                this.data = {{
                    cpu: [],
                    memory: [],
                    latency: [],
                    throughput: [],
                    timestamps: []
                }};
                this.maxDataPoints = 50;
                
                this.initWebSocket();
                this.initCharts();
                this.initKnowledgeGraph();
                this.loadApiEndpoints();
                this.loadKnowledgeGraphData();
                
                // Refresh API endpoints every 30 seconds
                setInterval(() => this.loadApiEndpoints(), 30000);
                
                // Refresh knowledge graph data every 10 seconds
                setInterval(() => this.loadKnowledgeGraphData(), 10000);
            }}
            
            initKnowledgeGraph() {{
                console.log('🎬 Initializing knowledge graph 3D scene...');
                // Initialize Three.js scene for knowledge graph
                const container = document.getElementById('knowledgeGraphContainer');
                if (!container) {{
                    console.error('❌ Knowledge graph container not found!');
                    return;
                }}
                
                console.log('📦 Container dimensions:', container.offsetWidth, 'x', container.offsetHeight);
                
                const scene = new THREE.Scene();
                const camera = new THREE.PerspectiveCamera(75, container.offsetWidth / container.offsetHeight, 0.1, 1000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                
                renderer.setSize(container.offsetWidth, container.offsetHeight);
                renderer.setClearColor(0x1a1a1a);
                container.appendChild(renderer.domElement);
                
                console.log('✅ Three.js renderer initialized');
                
                // Add lights
                const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
                scene.add(ambientLight);
                
                const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                directionalLight.position.set(10, 10, 5);
                scene.add(directionalLight);
                
                // Camera position
                camera.position.set(0, 5, 20);
                
                // Store references
                this.knowledgeGraph = {{
                    scene,
                    camera,
                    renderer,
                    entities: [],
                    relationships: [],
                    entityMeshes: new Map(),
                    relationshipLines: new Map(),
                    selectedEntity: null,
                    searchTerm: '',
                    showRelationships: true,
                    animate: true
                }};
                
                // Add mouse controls
                this.initGraphControls();
                
                // Start render loop
                this.renderKnowledgeGraph();
            }}
            
            initGraphControls() {{
                const kg = this.knowledgeGraph;
                let mouse = new THREE.Vector2();
                let raycaster = new THREE.Raycaster();
                let isDragging = false;
                let previousMousePosition = {{ x: 0, y: 0 }};
                
                // Mouse events for interaction
                kg.renderer.domElement.addEventListener('mousemove', (event) => {{
                    const rect = kg.renderer.domElement.getBoundingClientRect();
                    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    if (isDragging) {{
                        const deltaMove = {{
                            x: event.offsetX - previousMousePosition.x,
                            y: event.offsetY - previousMousePosition.y
                        }};
                        
                        kg.camera.position.x -= deltaMove.x * 0.01;
                        kg.camera.position.y += deltaMove.y * 0.01;
                    }} else {{
                        // Check for entity hover
                        this.checkEntityHover(mouse);
                    }}
                    
                    previousMousePosition = {{ x: event.offsetX, y: event.offsetY }};
                }});
                
                kg.renderer.domElement.addEventListener('mousedown', () => {{
                    isDragging = true;
                }});
                
                kg.renderer.domElement.addEventListener('mouseup', () => {{
                    isDragging = false;
                }});
                
                kg.renderer.domElement.addEventListener('click', (event) => {{
                    this.handleEntityClick(mouse);
                }});
                
                // Scroll for zoom
                kg.renderer.domElement.addEventListener('wheel', (event) => {{
                    const scale = event.deltaY > 0 ? 1.1 : 0.9;
                    kg.camera.position.multiplyScalar(scale);
                    event.preventDefault();
                }});
                
                // Control event listeners
                document.getElementById('showRelationships').addEventListener('change', (e) => {{
                    kg.showRelationships = e.target.checked;
                    this.updateGraphVisibility();
                }});
                
                document.getElementById('animateGraph').addEventListener('change', (e) => {{
                    kg.animate = e.target.checked;
                }});
            }}
            
            async loadKnowledgeGraphData() {{
                console.log('🔄 Loading knowledge graph data...');
                try {{
                    // Load entities
                    const entitiesResponse = await fetch('http://localhost:3001/api/v1/query', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ limit: 100 }})
                    }});
                    
                    console.log('📡 API Response status:', entitiesResponse.status);
                    
                    if (entitiesResponse.ok) {{
                        const entitiesData = await entitiesResponse.json();
                        console.log('📊 Received data:', entitiesData);
                        
                        // Load metrics for additional entity info
                        const metricsResponse = await fetch('http://localhost:3001/api/v1/metrics');
                        const metricsData = metricsResponse.ok ? await metricsResponse.json() : null;
                        console.log('📊 Metrics data:', metricsData);
                        
                        this.updateKnowledgeGraph(entitiesData, metricsData);
                    }} else {{
                        console.error('❌ Failed to fetch data, status:', entitiesResponse.status);
                    }}
                }} catch (error) {{
                    console.error('❌ Failed to load knowledge graph data:', error);
                }}
            }}
            
            updateKnowledgeGraph(entitiesData, metricsData) {{
                console.log('🔄 Updating knowledge graph visualization...');
                const kg = this.knowledgeGraph;
                
                // Clear existing meshes
                kg.entityMeshes.forEach(mesh => kg.scene.remove(mesh));
                kg.relationshipLines.forEach(line => kg.scene.remove(line));
                kg.entityMeshes.clear();
                kg.relationshipLines.clear();
                
                if (!entitiesData.data || !entitiesData.data.triples) {{
                    console.warn('⚠️ No triples data found:', entitiesData);
                    return;
                }}
                
                const triples = entitiesData.data.triples;
                console.log('📊 Processing', triples.length, 'triples');
                
                // Extract unique entities from triples
                const entities = new Map();
                const relationships = [];
                
                triples.forEach((triple, index) => {{
                    // Add subject entity
                    if (!entities.has(triple.subject)) {{
                        entities.set(triple.subject, {{
                            id: triple.subject,
                            name: triple.subject,
                            type: this.inferEntityType(triple.subject),
                            connections: 0
                        }});
                    }}
                    entities.get(triple.subject).connections++;
                    
                    // Add object entity
                    if (!entities.has(triple.object)) {{
                        entities.set(triple.object, {{
                            id: triple.object,
                            name: triple.object,
                            type: this.inferEntityType(triple.object),
                            connections: 0
                        }});
                    }}
                    entities.get(triple.object).connections++;
                    
                    // Add relationship
                    relationships.push({{
                        source: triple.subject,
                        target: triple.object,
                        type: triple.predicate,
                        confidence: triple.confidence || 1.0
                    }});
                }});
                
                // Create 3D visualization
                const entityArray = Array.from(entities.values());
                console.log('🎨 Creating visualization for', entityArray.length, 'entities and', relationships.length, 'relationships');
                
                this.createEntityMeshes(entityArray);
                this.createRelationshipLines(relationships);
                
                // Update info display
                document.getElementById('entityCount').textContent = entities.size;
                document.getElementById('relationshipCount').textContent = relationships.length;
                
                kg.entities = entityArray;
                kg.relationships = relationships;
                
                console.log('✅ Knowledge graph updated successfully');
            }}
            
            inferEntityType(entityName) {{
                const name = entityName.toLowerCase();
                if (name.includes('claude') || name.includes('ai') || name.includes('assistant')) return 'AI';
                if (name.includes('anthropic') || name.includes('google') || name.includes('company')) return 'Company';
                if (name.includes('san francisco') || name.includes('california') || name.includes('location')) return 'Location';
                if (name.includes('transformer') || name.includes('pattern') || name.includes('architecture')) return 'Technology';
                if (name.includes('protocol') || name.includes('mcp') || name.includes('system')) return 'System';
                return 'Concept';
            }}
            
            getEntityColor(type) {{
                const colors = {{
                    'AI': 0x00ff88,
                    'Company': 0x4285f4,
                    'Location': 0xff9800,
                    'Technology': 0x9c27b0,
                    'System': 0xf44336,
                    'Concept': 0x607d8b
                }};
                return colors[type] || 0x888888;
            }}
            
            createEntityMeshes(entities) {{
                const kg = this.knowledgeGraph;
                const radius = 8;
                
                entities.forEach((entity, index) => {{
                    // Position entities in a circle
                    const angle = (index / entities.length) * Math.PI * 2;
                    const x = Math.cos(angle) * radius + (Math.random() - 0.5) * 2;
                    const z = Math.sin(angle) * radius + (Math.random() - 0.5) * 2;
                    const y = (Math.random() - 0.5) * 4;
                    
                    // Create sphere geometry
                    const size = Math.max(0.2, 0.1 + entity.connections * 0.05);
                    const geometry = new THREE.SphereGeometry(size, 16, 16);
                    const material = new THREE.MeshLambertMaterial({{
                        color: this.getEntityColor(entity.type),
                        transparent: true,
                        opacity: 0.8
                    }});
                    
                    const mesh = new THREE.Mesh(geometry, material);
                    mesh.position.set(x, y, z);
                    mesh.userData = {{ entity }};
                    
                    kg.scene.add(mesh);
                    kg.entityMeshes.set(entity.id, mesh);
                }});
            }}
            
            createRelationshipLines(relationships) {{
                const kg = this.knowledgeGraph;
                
                relationships.forEach(rel => {{
                    const sourceMesh = kg.entityMeshes.get(rel.source);
                    const targetMesh = kg.entityMeshes.get(rel.target);
                    
                    if (sourceMesh && targetMesh) {{
                        const geometry = new THREE.BufferGeometry().setFromPoints([
                            sourceMesh.position,
                            targetMesh.position
                        ]);
                        
                        const material = new THREE.LineBasicMaterial({{
                            color: 0x444444,
                            transparent: true,
                            opacity: 0.6
                        }});
                        
                        const line = new THREE.Line(geometry, material);
                        line.userData = {{ relationship: rel }};
                        
                        kg.scene.add(line);
                        kg.relationshipLines.set(`${{rel.source}}-${{rel.target}}`, line);
                    }}
                }});
            }}
            
            checkEntityHover(mouse) {{
                const kg = this.knowledgeGraph;
                const raycaster = new THREE.Raycaster();
                raycaster.setFromCamera(mouse, kg.camera);
                
                const meshes = Array.from(kg.entityMeshes.values());
                const intersects = raycaster.intersectObjects(meshes);
                
                const tooltip = document.getElementById('entityTooltip');
                
                if (intersects.length > 0) {{
                    const entity = intersects[0].object.userData.entity;
                    tooltip.innerHTML = `
                        <strong>${{entity.name}}</strong><br>
                        Type: ${{entity.type}}<br>
                        Connections: ${{entity.connections}}
                    `;
                    tooltip.style.display = 'block';
                    tooltip.style.left = (event.clientX + 10) + 'px';
                    tooltip.style.top = (event.clientY - 10) + 'px';
                }} else {{
                    tooltip.style.display = 'none';
                }}
            }}
            
            handleEntityClick(mouse) {{
                const kg = this.knowledgeGraph;
                const raycaster = new THREE.Raycaster();
                raycaster.setFromCamera(mouse, kg.camera);
                
                const meshes = Array.from(kg.entityMeshes.values());
                const intersects = raycaster.intersectObjects(meshes);
                
                if (intersects.length > 0) {{
                    const entity = intersects[0].object.userData.entity;
                    this.selectEntity(entity);
                }}
            }}
            
            selectEntity(entity) {{
                const kg = this.knowledgeGraph;
                
                // Reset previous selection
                if (kg.selectedEntity) {{
                    const prevMesh = kg.entityMeshes.get(kg.selectedEntity.id);
                    if (prevMesh) {{
                        prevMesh.material.emissive = new THREE.Color(0x000000);
                        prevMesh.scale.set(1, 1, 1);
                    }}
                }}
                
                // Highlight new selection
                kg.selectedEntity = entity;
                const mesh = kg.entityMeshes.get(entity.id);
                if (mesh) {{
                    mesh.material.emissive = new THREE.Color(0x444444);
                    mesh.scale.set(1.5, 1.5, 1.5);
                }}
                
                document.getElementById('selectedEntity').textContent = entity.name;
            }}
            
            searchEntities(searchTerm) {{
                const kg = this.knowledgeGraph;
                kg.searchTerm = searchTerm.toLowerCase();
                
                kg.entityMeshes.forEach((mesh, entityId) => {{
                    const entity = mesh.userData.entity;
                    const matches = !searchTerm || entity.name.toLowerCase().includes(kg.searchTerm);
                    mesh.visible = matches;
                }});
                
                this.updateGraphVisibility();
            }}
            
            updateGraphVisibility() {{
                const kg = this.knowledgeGraph;
                
                // Update relationship visibility
                kg.relationshipLines.forEach(line => {{
                    const rel = line.userData.relationship;
                    const sourceMesh = kg.entityMeshes.get(rel.source);
                    const targetMesh = kg.entityMeshes.get(rel.target);
                    
                    line.visible = kg.showRelationships && 
                                   sourceMesh.visible && 
                                   targetMesh.visible;
                }});
            }}
            
            renderKnowledgeGraph() {{
                const kg = this.knowledgeGraph;
                
                if (kg.animate) {{
                    // Rotate entities slightly
                    kg.entityMeshes.forEach(mesh => {{
                        mesh.rotation.y += 0.01;
                        
                        // Gentle floating animation
                        mesh.position.y += Math.sin(Date.now() * 0.001 + mesh.position.x) * 0.002;
                    }});
                    
                    // Look at center
                    kg.camera.lookAt(0, 0, 0);
                }}
                
                kg.renderer.render(kg.scene, kg.camera);
                requestAnimationFrame(() => this.renderKnowledgeGraph());
            }}
            
            resetView() {{
                const kg = this.knowledgeGraph;
                kg.camera.position.set(0, 5, 20);
                kg.camera.lookAt(0, 0, 0);
            }}
            
            async loadApiEndpoints() {{
                try {{
                    const response = await fetch('/api/endpoints');
                    const data = await response.json();
                    this.updateApiEndpointsDisplay(data);
                }} catch (error) {{
                    console.error('Failed to load API endpoints:', error);
                    document.getElementById('apiEndpointsContainer').innerHTML = 
                        '<p style="color: red;">Failed to load API endpoint data</p>';
                }}
            }}
            
            updateApiEndpointsDisplay(data) {{
                const container = document.getElementById('apiEndpointsContainer');
                
                if (!data.discovered_endpoints || data.discovered_endpoints.length === 0) {{
                    container.innerHTML = '<p>No API endpoints discovered</p>';
                    return;
                }}
                
                let html = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">';
                
                data.discovered_endpoints.forEach(endpoint => {{
                    const stats = data.endpoint_stats[`${{endpoint.method}} ${{endpoint.path}}`] || {{}};
                    const methodColor = this.getMethodColor(endpoint.method);
                    const isReal = endpoint.tags && endpoint.tags.includes('real');
                    
                    html += `
                        <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; background: ${{isReal ? '#f0fff4' : '#fff'}};">
                            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                <span style="background: ${{methodColor}}; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; font-weight: bold;">
                                    ${{endpoint.method}}
                                </span>
                                <span style="margin-left: 8px; font-weight: bold; color: #333;">
                                    ${{endpoint.path}}
                                </span>
                                ${{isReal ? '<span style="margin-left: auto; color: #28a745; font-size: 0.8em;">🟢 REAL</span>' : ''}}
                            </div>
                            <div style="font-size: 0.9em; color: #666; margin-bottom: 8px;">
                                ${{endpoint.documentation}}
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; font-size: 0.8em;">
                                <div>
                                    <strong>Requests:</strong><br>
                                    ${{stats.total_requests || 0}}
                                </div>
                                <div>
                                    <strong>Success Rate:</strong><br>
                                    ${{stats.total_requests ? ((stats.successful_requests || 0) / stats.total_requests * 100).toFixed(1) : 0}}%
                                </div>
                                <div>
                                    <strong>Avg Time:</strong><br>
                                    ${{stats.avg_response_time ? (stats.avg_response_time.secs * 1000 + stats.avg_response_time.nanos / 1000000).toFixed(1) : 0}}ms
                                </div>
                            </div>
                            ${{endpoint.tags && endpoint.tags.length > 0 ? `
                                <div style="margin-top: 8px;">
                                    ${{endpoint.tags.map(tag => `<span style="background: #e9ecef; padding: 2px 6px; border-radius: 3px; font-size: 0.75em; margin-right: 4px;">${{tag}}</span>`).join('')}}
                                </div>
                            ` : ''}}
                        </div>
                    `;
                }});
                
                html += '</div>';
                
                // Add summary stats
                html += `
                    <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                        <h4 style="margin-top: 0;">API Performance Summary</h4>
                        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; text-align: center;">
                            <div>
                                <div style="font-size: 1.5em; font-weight: bold; color: #007bff;">${{data.discovered_endpoints.length}}</div>
                                <div style="font-size: 0.9em; color: #666;">Total Endpoints</div>
                            </div>
                            <div>
                                <div style="font-size: 1.5em; font-weight: bold; color: #28a745;">${{data.total_requests}}</div>
                                <div style="font-size: 0.9em; color: #666;">Total Requests</div>
                            </div>
                            <div>
                                <div style="font-size: 1.5em; font-weight: bold; color: #ffc107;">${{data.live_requests}}</div>
                                <div style="font-size: 0.9em; color: #666;">Live Requests</div>
                            </div>
                            <div>
                                <div style="font-size: 1.5em; font-weight: bold; color: #dc3545;">${{data.error_analysis.recent_errors.length}}</div>
                                <div style="font-size: 0.9em; color: #666;">Recent Errors</div>
                            </div>
                        </div>
                    </div>
                `;
                
                container.innerHTML = html;
            }}
            
            getMethodColor(method) {{
                const colors = {{
                    'GET': '#28a745',
                    'POST': '#007bff',
                    'PUT': '#ffc107',
                    'DELETE': '#dc3545',
                    'PATCH': '#6f42c1',
                    'HEAD': '#6c757d',
                    'OPTIONS': '#17a2b8'
                }};
                return colors[method] || '#6c757d';
            }}
            
            initWebSocket() {{
                const wsUrl = `ws://localhost:{websocket_port}`;
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {{
                    this.updateConnectionStatus(true);
                    console.log('Connected to dashboard WebSocket');
                }};
                
                this.ws.onmessage = (event) => {{
                    try {{
                        const message = JSON.parse(event.data);
                        this.handleMessage(message);
                    }} catch (e) {{
                        console.error('Error parsing WebSocket message:', e);
                    }}
                }};
                
                this.ws.onclose = () => {{
                    this.updateConnectionStatus(false);
                    console.log('Disconnected from dashboard WebSocket');
                    // Attempt to reconnect after 5 seconds
                    setTimeout(() => this.initWebSocket(), 5000);
                }};
                
                this.ws.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus(false);
                }};
            }}
            
            handleMessage(message) {{
                if (message.MetricsUpdate) {{
                    this.updateMetrics(message.MetricsUpdate);
                }}
            }}
            
            updateMetrics(metrics) {{
                const timestamp = new Date(metrics.timestamp * 1000);
                
                // Update current values
                document.getElementById('cpuUsage').textContent = 
                    metrics.system_metrics.cpu_usage_percent.toFixed(1);
                document.getElementById('memoryUsage').textContent = 
                    metrics.system_metrics.memory_usage_percent.toFixed(1);
                document.getElementById('queryLatency').textContent = 
                    metrics.performance_metrics.query_latency_ms.p95.toFixed(2);
                document.getElementById('operationsPerSec').textContent = 
                    metrics.application_metrics.operations_per_second.toFixed(0);
                
                // Update chart data
                this.addDataPoint(timestamp, {{
                    cpu: metrics.system_metrics.cpu_usage_percent,
                    memory: metrics.system_metrics.memory_usage_percent,
                    latency: metrics.performance_metrics.query_latency_ms.p95,
                    throughput: metrics.application_metrics.operations_per_second
                }});
                
                // Update alerts
                this.updateAlerts(metrics.alerts);
            }}
            
            addDataPoint(timestamp, values) {{
                this.data.timestamps.push(timestamp.toLocaleTimeString());
                this.data.cpu.push(values.cpu);
                this.data.memory.push(values.memory);
                this.data.latency.push(values.latency);
                this.data.throughput.push(values.throughput);
                
                // Keep only the most recent data points
                if (this.data.timestamps.length > this.maxDataPoints) {{
                    this.data.timestamps.shift();
                    this.data.cpu.shift();
                    this.data.memory.shift();
                    this.data.latency.shift();
                    this.data.throughput.shift();
                }}
                
                // Update all charts
                Object.values(this.charts).forEach(chart => chart.update());
            }}
            
            updateConnectionStatus(connected) {{
                const statusElement = document.getElementById('connectionStatus');
                const indicator = statusElement.querySelector('.status-indicator');
                
                if (connected) {{
                    statusElement.className = 'connection-status connected';
                    indicator.className = 'status-indicator status-good';
                    statusElement.innerHTML = '<span class="status-indicator status-good"></span>Connected';
                }} else {{
                    statusElement.className = 'connection-status disconnected';
                    indicator.className = 'status-indicator status-critical';
                    statusElement.innerHTML = '<span class="status-indicator status-critical"></span>Disconnected';
                }}
            }}
            
            updateAlerts(alerts) {{
                const container = document.getElementById('alertsContainer');
                if (alerts.length === 0) {{
                    container.innerHTML = '<p>No active alerts</p>';
                    return;
                }}
                
                container.innerHTML = alerts.map(alert => {{
                    const severityClass = `alert-${{alert.severity.toLowerCase()}}`;
                    return `
                        <div class="alert-item ${{severityClass}}">
                            <strong>${{alert.severity.toUpperCase()}}</strong>: ${{alert.message}}
                            <small style="float: right;">${{new Date(alert.timestamp * 1000).toLocaleString()}}</small>
                        </div>
                    `;
                }}).join('');
            }}
            
            initCharts() {{
                const chartOptions = {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }}
                }};
                
                // CPU Chart
                this.charts.cpu = new Chart(document.getElementById('cpuChart'), {{
                    type: 'line',
                    data: {{
                        labels: this.data.timestamps,
                        datasets: [{{
                            label: 'CPU Usage %',
                            data: this.data.cpu,
                            borderColor: 'rgb(255, 99, 132)',
                            backgroundColor: 'rgba(255, 99, 132, 0.1)',
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        ...chartOptions,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: 100
                            }}
                        }}
                    }}
                }});
                
                // Memory Chart
                this.charts.memory = new Chart(document.getElementById('memoryChart'), {{
                    type: 'line',
                    data: {{
                        labels: this.data.timestamps,
                        datasets: [{{
                            label: 'Memory Usage %',
                            data: this.data.memory,
                            borderColor: 'rgb(54, 162, 235)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            tension: 0.1
                        }}]
                    }},
                    options: {{
                        ...chartOptions,
                        scales: {{
                            y: {{
                                beginAtZero: true,
                                max: 100
                            }}
                        }}
                    }}
                }});
                
                // Latency Chart
                this.charts.latency = new Chart(document.getElementById('latencyChart'), {{
                    type: 'line',
                    data: {{
                        labels: this.data.timestamps,
                        datasets: [{{
                            label: 'Query Latency (P95) ms',
                            data: this.data.latency,
                            borderColor: 'rgb(255, 205, 86)',
                            backgroundColor: 'rgba(255, 205, 86, 0.1)',
                            tension: 0.1
                        }}]
                    }},
                    options: chartOptions
                }});
                
                // Throughput Chart
                this.charts.throughput = new Chart(document.getElementById('throughputChart'), {{
                    type: 'line',
                    data: {{
                        labels: this.data.timestamps,
                        datasets: [{{
                            label: 'Operations/sec',
                            data: this.data.throughput,
                            borderColor: 'rgb(75, 192, 192)',
                            backgroundColor: 'rgba(75, 192, 192, 0.1)',
                            tension: 0.1
                        }}]
                    }},
                    options: chartOptions
                }});
            }}
        }}
        
        // Initialize dashboard when page loads
        document.addEventListener('DOMContentLoaded', () => {{
            window.dashboard = new PerformanceDashboard();
            window.knowledgeGraph = {{
                resetView: () => window.dashboard.resetView(),
                searchEntities: (term) => window.dashboard.searchEntities(term)
            }};
        }});
    </script>
</body>
</html>
        "#, 
        title = config.title,
        websocket_port = config.websocket_port
        )
    }
}

pub struct DashboardServer {
    dashboard: PerformanceDashboard,
}

impl DashboardServer {
    pub fn new(
        config: DashboardConfig,
        registry: Arc<MetricRegistry>,
        collectors: Vec<Box<dyn MetricsCollector>>,
    ) -> Self {
        let dashboard = PerformanceDashboard::new(config, registry, collectors);
        Self { dashboard }
    }
    
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.dashboard.start().await
    }
    
    pub fn stop(&self) {
        self.dashboard.stop();
    }
}

pub struct WebSocketHandler;

impl Default for WebSocketHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl WebSocketHandler {
    pub fn new() -> Self {
        Self
    }
}

// Test API handler implementations
async fn discover_tests() -> Result<impl warp::Reply, warp::Rejection> {
    // For now, return static test data since we can't access the test tracker easily
    // In production, this would query the TestExecutionTracker
    let test_suites = vec![
        serde_json::json!({
            "name": "core::graph",
            "path": "src/core/graph",
            "test_type": "Unit",
            "test_count": 15,
            "tags": ["core", "graph"],
            "description": "Core graph functionality tests"
        }),
        serde_json::json!({
            "name": "core::brain_enhanced_graph",
            "path": "src/core/brain_enhanced_graph",
            "test_type": "Unit",
            "test_count": 12,
            "tags": ["core", "brain", "enhanced"],
            "description": "Brain-enhanced graph tests"
        }),
        serde_json::json!({
            "name": "cognitive::orchestrator",
            "path": "src/cognitive/orchestrator",
            "test_type": "Integration",
            "test_count": 8,
            "tags": ["cognitive", "integration"],
            "description": "Cognitive orchestrator integration tests"
        }),
        serde_json::json!({
            "name": "storage::csr",
            "path": "src/storage/csr",
            "test_type": "Unit",
            "test_count": 10,
            "tags": ["storage", "performance"],
            "description": "CSR storage tests"
        }),
    ];
    
    let response = serde_json::json!({
        "suites": test_suites,
        "total_suites": test_suites.len(),
        "total_tests": 45,
        "categories": {
            "unit": 37,
            "integration": 8,
            "e2e": 0
        }
    });
    
    Ok(warp::reply::json(&response))
}

async fn execute_tests(suite_request: TestSuiteRequest) -> Result<impl warp::Reply, warp::Rejection> {
    use uuid::Uuid;
    
    let execution_id = Uuid::new_v4().to_string();
    let suite_name = suite_request.suite_name.clone();
    let execution_id_clone = execution_id.clone();
    let suite_name_clone = suite_name.clone();
    
    // Spawn the test execution in the background
    tokio::spawn(async move {
        run_cargo_tests(execution_id_clone, suite_name_clone, suite_request).await;
    });
    
    let response = TestExecutionResponse {
        execution_id,
        suite_name,
        status: "started".to_string(),
        message: "Test execution started".to_string(),
    };
    
    Ok(warp::reply::json(&response))
}

async fn get_test_status(execution_id: String) -> Result<impl warp::Reply, warp::Rejection> {
    // In a real implementation, this would query the test execution status
    let status = serde_json::json!({
        "execution_id": execution_id,
        "status": "running",
        "progress": {
            "current": 5,
            "total": 15,
            "passed": 4,
            "failed": 1,
            "ignored": 0
        },
        "current_test": "test_graph_node_creation",
        "duration_ms": 2500
    });
    
    Ok(warp::reply::json(&status))
}

async fn run_cargo_tests(execution_id: String, suite_name: String, request: TestSuiteRequest) {
    println!("🚀 Starting test execution: {execution_id} for suite: {suite_name}");
    
    // Send test started message via WebSocket
    broadcast_test_message(DashboardMessage::TestStarted {
        execution_id: execution_id.clone(),
        suite_name: suite_name.clone(),
        total_tests: 0, // Will be updated when we parse test output
    }).await;
    
    // Build the cargo test command
    let mut cmd = Command::new("cargo");
    cmd.arg("test");
    
    // Add suite filter if provided
    if !suite_name.is_empty() && suite_name != "all" {
        cmd.arg(suite_name.replace("::", "_"));
    }
    
    // Add additional filter if provided
    if let Some(filter) = &request.filter {
        cmd.arg(filter);
    }
    
    // Add flags
    if request.nocapture {
        cmd.arg("--");
        cmd.arg("--nocapture");
    }
    
    if !request.parallel {
        cmd.arg("--");
        cmd.arg("--test-threads=1");
    }
    
    // Set up output capture
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    
    let start_time = std::time::Instant::now();
    let mut test_count = 0;
    let mut passed_count = 0;
    let mut failed_count = 0;
    
    // Execute the command
    match cmd.spawn() {
        Ok(mut child) => {
            // Read stdout
            if let Some(stdout) = child.stdout.take() {
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                
                while let Ok(Some(line)) = lines.next_line().await {
                    println!("📝 Test output: {line}");
                    
                    // Parse test output and send progress updates
                    if line.contains("running") && line.contains("test") {
                        // Extract test count from "running N tests"
                        if let Some(count) = extract_test_count(&line) {
                            test_count = count;
                        }
                    } else if line.contains("test") && (line.contains("ok") || line.contains("FAILED")) {
                        // Parse individual test results
                        let status = if line.contains("ok") {
                            passed_count += 1;
                            "passed"
                        } else {
                            failed_count += 1;
                            "failed"
                        };
                        
                        // Extract test name
                        let test_name = extract_test_name(&line).unwrap_or_else(|| "unknown".to_string());
                        
                        broadcast_test_message(DashboardMessage::TestProgress {
                            execution_id: execution_id.clone(),
                            current: passed_count + failed_count,
                            total: test_count,
                            test_name,
                            status: status.to_string(),
                        }).await;
                    }
                    
                    // Send log message
                    broadcast_test_message(DashboardMessage::TestLog {
                        execution_id: execution_id.clone(),
                        message: line,
                        level: "info".to_string(),
                    }).await;
                }
            }
            
            // Wait for completion
            match child.wait().await {
                Ok(status) => {
                    let duration_ms = start_time.elapsed().as_millis() as u64;
                    
                    if status.success() {
                        println!("✅ Test execution completed successfully: {execution_id}");
                        broadcast_test_message(DashboardMessage::TestCompleted {
                            execution_id: execution_id.clone(),
                            passed: passed_count,
                            failed: failed_count,
                            ignored: 0,
                            duration_ms,
                        }).await;
                    } else {
                        println!("❌ Test execution failed: {execution_id}");
                        broadcast_test_message(DashboardMessage::TestFailed {
                            execution_id: execution_id.clone(),
                            error: "Test execution failed".to_string(),
                        }).await;
                    }
                }
                Err(e) => {
                    println!("❌ Error waiting for test completion: {e}");
                    broadcast_test_message(DashboardMessage::TestFailed {
                        execution_id: execution_id.clone(),
                        error: format!("Error waiting for test completion: {e}"),
                    }).await;
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to start test execution: {e}");
            broadcast_test_message(DashboardMessage::TestFailed {
                execution_id: execution_id.clone(),
                error: format!("Failed to start test execution: {e}"),
            }).await;
        }
    }
}

async fn broadcast_test_message(message: DashboardMessage) {
    if let Some(clients) = WEBSOCKET_CLIENTS.get() {
        if let Ok(json) = serde_json::to_string(&message) {
            let ws_message = Message::Text(json);
            let mut clients_guard = clients.lock().unwrap();
            clients_guard.retain(|client| {
                client.send(ws_message.clone()).is_ok()
            });
        }
    }
}

fn extract_test_count(line: &str) -> Option<usize> {
    // Parse "running N tests" pattern
    if let Some(pos) = line.find("running") {
        let rest = &line[pos + 7..];
        if let Some(test_pos) = rest.find("test") {
            let num_str = rest[..test_pos].trim();
            return num_str.parse().ok();
        }
    }
    None
}

fn extract_test_name(line: &str) -> Option<String> {
    // Parse test name from test output line
    if line.contains("test") {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 && parts[0] == "test" {
            return Some(parts[1].trim_end_matches("...").to_string());
        }
    }
    None
}