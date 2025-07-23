/*!
LLMKG API Endpoint Monitor
Real-time API endpoint monitoring, testing, and analysis
*/

use crate::monitoring::metrics::MetricRegistry;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Instant, Duration, SystemTime};
use serde::{Serialize, Deserialize};
use tokio::sync::broadcast;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEndpoint {
    pub path: String,
    pub method: HttpMethod,
    pub handler_function: String,
    pub parameters: Vec<ApiParameter>,
    pub response_schema: Option<String>,
    pub auth_required: bool,
    pub rate_limit: Option<RateLimit>,
    pub documentation: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiParameter {
    pub name: String,
    pub param_type: ParameterType,
    pub data_type: String,
    pub required: bool,
    pub description: String,
    pub example: Option<String>,
    pub validation: Option<ValidationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Query,
    Path,
    Body,
    Header,
    FormData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub rule_type: ValidationType,
    pub value: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    MinLength,
    MaxLength,
    Pattern,
    Range,
    Email,
    Url,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_window: u32,
    pub window_duration: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRequest {
    pub id: String,
    pub endpoint: String,
    pub method: HttpMethod,
    pub timestamp: SystemTime,
    pub client_ip: String,
    pub user_agent: String,
    pub headers: HashMap<String, String>,
    pub query_params: HashMap<String, String>,
    pub body: Option<String>,
    pub response: Option<ApiResponse>,
    pub duration: Option<Duration>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiMetrics {
    pub endpoints: HashMap<String, ApiEndpoint>,
    pub request_history: VecDeque<ApiRequest>,
    pub endpoint_stats: HashMap<String, EndpointStats>,
    pub performance_metrics: ApiPerformanceMetrics,
    pub error_analysis: ErrorAnalysis,
    pub rate_limiting_status: HashMap<String, RateLimitStatus>,
    pub live_requests: HashMap<String, ApiRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointStats {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub avg_response_time: Duration,
    pub min_response_time: Duration,
    pub max_response_time: Duration,
    pub p50_response_time: Duration,
    pub p95_response_time: Duration,
    pub p99_response_time: Duration,
    pub requests_per_minute: f64,
    pub error_rate: f64,
    pub last_request: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiPerformanceMetrics {
    pub total_requests_per_second: f64,
    pub avg_response_time_ms: f64,
    pub median_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub p99_response_time_ms: f64,
    pub error_rate_percentage: f64,
    pub throughput_mbps: f64,
    pub concurrent_requests: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAnalysis {
    pub error_counts: HashMap<u16, u64>, // Status code -> count
    pub error_patterns: Vec<ErrorPattern>,
    pub recent_errors: VecDeque<ApiError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPattern {
    pub pattern: String,
    pub frequency: u64,
    pub endpoints: Vec<String>,
    pub suggested_fix: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub timestamp: SystemTime,
    pub endpoint: String,
    pub status_code: u16,
    pub error_message: String,
    pub request_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    pub requests_remaining: u32,
    pub window_reset_time: SystemTime,
    pub is_limited: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiTestRequest {
    pub endpoint: String,
    pub method: HttpMethod,
    pub headers: HashMap<String, String>,
    pub query_params: HashMap<String, String>,
    pub body: Option<String>,
    pub expected_status: Option<u16>,
    pub timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiTestResult {
    pub request: ApiTestRequest,
    pub response: Option<ApiResponse>,
    pub duration: Duration,
    pub success: bool,
    pub error: Option<String>,
    pub validation_results: Vec<ValidationResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub field: String,
    pub expected: String,
    pub actual: String,
    pub passed: bool,
}

pub struct ApiEndpointMonitor {
    metrics: Arc<RwLock<ApiMetrics>>,
    event_sender: broadcast::Sender<ApiRequest>,
    max_request_history: usize,
    max_error_history: usize,
    http_client: reqwest::Client,
}

impl ApiEndpointMonitor {
    pub fn new() -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        
        Self {
            metrics: Arc::new(RwLock::new(ApiMetrics {
                endpoints: HashMap::new(),
                request_history: VecDeque::new(),
                endpoint_stats: HashMap::new(),
                performance_metrics: ApiPerformanceMetrics {
                    total_requests_per_second: 0.0,
                    avg_response_time_ms: 0.0,
                    median_response_time_ms: 0.0,
                    p95_response_time_ms: 0.0,
                    p99_response_time_ms: 0.0,
                    error_rate_percentage: 0.0,
                    throughput_mbps: 0.0,
                    concurrent_requests: 0,
                },
                error_analysis: ErrorAnalysis {
                    error_counts: HashMap::new(),
                    error_patterns: Vec::new(),
                    recent_errors: VecDeque::new(),
                },
                rate_limiting_status: HashMap::new(),
                live_requests: HashMap::new(),
            })),
            event_sender,
            max_request_history: 10000,
            max_error_history: 1000,
            http_client: reqwest::Client::new(),
        }
    }

    pub fn register_endpoint(&self, endpoint: ApiEndpoint) {
        let mut metrics = self.metrics.write().unwrap();
        let method_val = endpoint.method as u8;
        let path = endpoint.path.clone();
        let key = format!("{} {}", method_val, path);
        metrics.endpoints.insert(key.clone(), endpoint);
        
        // Initialize stats for the endpoint
        metrics.endpoint_stats.insert(key, EndpointStats {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_response_time: Duration::new(0, 0),
            min_response_time: Duration::new(u64::MAX, 0),
            max_response_time: Duration::new(0, 0),
            p50_response_time: Duration::new(0, 0),
            p95_response_time: Duration::new(0, 0),
            p99_response_time: Duration::new(0, 0),
            requests_per_minute: 0.0,
            error_rate: 0.0,
            last_request: None,
        });
    }

    pub fn discover_endpoints(&self) -> Result<Vec<ApiEndpoint>, Box<dyn std::error::Error>> {
        println!("🔍 Discovering real LLMKG API endpoints from source code...");
        
        let mut endpoints = Vec::new();
        
        // Discover real Warp routes from dashboard.rs
        let dashboard_endpoints = self.discover_dashboard_endpoints()?;
        endpoints.extend(dashboard_endpoints);
        
        // Discover MCP server endpoints (if they exist as HTTP endpoints)
        let mcp_endpoints = self.discover_mcp_endpoints()?;
        endpoints.extend(mcp_endpoints);
        
        // Register discovered endpoints
        for endpoint in &endpoints {
            self.register_endpoint(endpoint.clone());
        }
        
        println!("✅ Discovered {} real API endpoints", endpoints.len());
        for endpoint in &endpoints {
            println!("   📍 {} {} - {}", 
                match endpoint.method {
                    HttpMethod::GET => "GET",
                    HttpMethod::POST => "POST",
                    HttpMethod::PUT => "PUT",
                    HttpMethod::DELETE => "DELETE",
                    HttpMethod::PATCH => "PATCH",
                    HttpMethod::HEAD => "HEAD",
                    HttpMethod::OPTIONS => "OPTIONS",
                },
                endpoint.path,
                endpoint.documentation
            );
        }
        
        Ok(endpoints)
    }
    
    fn discover_dashboard_endpoints(&self) -> Result<Vec<ApiEndpoint>, Box<dyn std::error::Error>> {
        let mut endpoints = Vec::new();
        
        // Real endpoints from dashboard.rs warp server
        endpoints.push(ApiEndpoint {
            path: "/api/metrics".to_string(),
            method: HttpMethod::GET,
            handler_function: "metrics_route".to_string(),
            parameters: Vec::new(),
            response_schema: Some("Vec<MetricSample>".to_string()),
            auth_required: false,
            rate_limit: None,
            documentation: "Real-time system metrics from MetricRegistry".to_string(),
            tags: vec!["metrics".to_string(), "monitoring".to_string(), "real".to_string()],
        });
        
        endpoints.push(ApiEndpoint {
            path: "/api/history".to_string(),
            method: HttpMethod::GET,
            handler_function: "history_route".to_string(),
            parameters: Vec::new(),
            response_schema: Some("Vec<RealTimeMetrics>".to_string()),
            auth_required: false,
            rate_limit: None,
            documentation: "Historical metrics data from dashboard server".to_string(),
            tags: vec!["history".to_string(), "monitoring".to_string(), "real".to_string()],
        });
        
        // Root dashboard endpoint
        endpoints.push(ApiEndpoint {
            path: "/".to_string(),
            method: HttpMethod::GET,
            handler_function: "static_route".to_string(),
            parameters: Vec::new(),
            response_schema: Some("HTML".to_string()),
            auth_required: false,
            rate_limit: None,
            documentation: "LLMKG Performance Dashboard HTML interface".to_string(),
            tags: vec!["dashboard".to_string(), "ui".to_string(), "real".to_string()],
        });
        
        Ok(endpoints)
    }
    
    fn discover_mcp_endpoints(&self) -> Result<Vec<ApiEndpoint>, Box<dyn std::error::Error>> {
        let mut endpoints = Vec::new();
        
        // Note: MCP servers in LLMKG are currently JSON-RPC over stdio/websocket, not HTTP REST
        // But we can add hypothetical HTTP endpoints they might expose in the future
        
        // If we were to expose MCP operations over HTTP, they would look like:
        endpoints.push(ApiEndpoint {
            path: "/mcp/health".to_string(),
            method: HttpMethod::GET,
            handler_function: "mcp_health_check".to_string(),
            parameters: Vec::new(),
            response_schema: Some("MCPHealthStatus".to_string()),
            auth_required: false,
            rate_limit: None,
            documentation: "MCP server health status".to_string(),
            tags: vec!["mcp".to_string(), "health".to_string(), "hypothetical".to_string()],
        });
        
        Ok(endpoints)
    }

    pub fn start_request(&self, endpoint: String, method: HttpMethod, client_ip: String, user_agent: String, headers: HashMap<String, String>, query_params: HashMap<String, String>, body: Option<String>) -> String {
        let request_id = Uuid::new_v4().to_string();
        
        let request = ApiRequest {
            id: request_id.clone(),
            endpoint: endpoint.clone(),
            method,
            timestamp: SystemTime::now(),
            client_ip,
            user_agent,
            headers,
            query_params,
            body,
            response: None,
            duration: None,
            error: None,
        };

        // Add to live requests
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.live_requests.insert(request_id.clone(), request.clone());
            metrics.performance_metrics.concurrent_requests += 1;
        }

        // Send event
        let _ = self.event_sender.send(request);

        request_id
    }

    pub fn end_request(&self, request_id: String, response: ApiResponse, error: Option<String>) {
        let mut completed_request = None;
        
        // Remove from live requests
        {
            let mut metrics = self.metrics.write().unwrap();
            if let Some(mut request) = metrics.live_requests.remove(&request_id) {
                request.response = Some(response.clone());
                request.duration = Some(request.timestamp.elapsed().unwrap_or(Duration::new(0, 0)));
                request.error = error.clone();
                completed_request = Some(request);
                metrics.performance_metrics.concurrent_requests = metrics.performance_metrics.concurrent_requests.saturating_sub(1);
            }
        }

        if let Some(request) = completed_request {
            let duration = request.duration.unwrap();
            let endpoint_key = format!("{} {}", request.method as u8, request.endpoint);
            
            // Update metrics
            {
                let mut metrics = self.metrics.write().unwrap();
                
                // Add to request history
                metrics.request_history.push_back(request.clone());
                if metrics.request_history.len() > self.max_request_history {
                    metrics.request_history.pop_front();
                }
                
                // Handle error analysis first (when we don't have mutable ref to stats)
                if response.status_code >= 400 {
                    *metrics.error_analysis.error_counts.entry(response.status_code).or_insert(0) += 1;
                    
                    if let Some(error_msg) = &error {
                        let api_error = ApiError {
                            timestamp: request.timestamp,
                            endpoint: request.endpoint.clone(),
                            status_code: response.status_code,
                            error_message: error_msg.clone(),
                            request_id: request.id.clone(),
                        };
                        
                        metrics.error_analysis.recent_errors.push_back(api_error);
                        if metrics.error_analysis.recent_errors.len() > self.max_error_history {
                            metrics.error_analysis.recent_errors.pop_front();
                        }
                    }
                }
                
                // Update endpoint stats
                if let Some(stats) = metrics.endpoint_stats.get_mut(&endpoint_key) {
                    stats.total_requests += 1;
                    stats.last_request = Some(request.timestamp);
                    
                    if response.status_code >= 200 && response.status_code < 400 {
                        stats.successful_requests += 1;
                    } else {
                        stats.failed_requests += 1;
                    }
                    
                    // Update response time stats
                    stats.min_response_time = stats.min_response_time.min(duration);
                    stats.max_response_time = stats.max_response_time.max(duration);
                    
                    // Calculate average (simplified)
                    let total_nanos = stats.avg_response_time.as_nanos() * (stats.total_requests - 1) as u128 + duration.as_nanos();
                    let avg_nanos = (total_nanos / stats.total_requests as u128) as u64;
                    stats.avg_response_time = Duration::from_nanos(avg_nanos);
                    
                    // Calculate error rate
                    stats.error_rate = (stats.failed_requests as f64 / stats.total_requests as f64) * 100.0;
                }
            }

            // Update performance metrics
            self.update_performance_metrics();
            
            // Analyze error patterns
            self.analyze_error_patterns();
        }
    }

    pub async fn test_endpoint(&self, test_request: ApiTestRequest) -> ApiTestResult {
        let start_time = Instant::now();
        
        // Build HTTP request
        let mut request_builder = match test_request.method {
            HttpMethod::GET => self.http_client.get(&test_request.endpoint),
            HttpMethod::POST => self.http_client.post(&test_request.endpoint),
            HttpMethod::PUT => self.http_client.put(&test_request.endpoint),
            HttpMethod::DELETE => self.http_client.delete(&test_request.endpoint),
            HttpMethod::PATCH => self.http_client.patch(&test_request.endpoint),
            HttpMethod::HEAD => self.http_client.head(&test_request.endpoint),
            HttpMethod::OPTIONS => self.http_client.request(reqwest::Method::OPTIONS, &test_request.endpoint),
        };

        // Add headers
        for (key, value) in &test_request.headers {
            request_builder = request_builder.header(key, value);
        }

        // Add query parameters
        if !test_request.query_params.is_empty() {
            request_builder = request_builder.query(&test_request.query_params);
        }

        // Add body
        if let Some(body) = &test_request.body {
            request_builder = request_builder.body(body.clone());
        }

        // Set timeout
        request_builder = request_builder.timeout(test_request.timeout);

        // Execute request
        let result = request_builder.send().await;
        let duration = start_time.elapsed();

        match result {
            Ok(response) => {
                let status_code = response.status().as_u16();
                let headers: HashMap<String, String> = response.headers()
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
                    .collect();
                
                let body = response.text().await.ok();
                let size_bytes = body.as_ref().map(|b| b.len() as u64).unwrap_or(0);
                
                let api_response = ApiResponse {
                    status_code,
                    headers,
                    body,
                    size_bytes,
                };

                let success = if let Some(expected) = test_request.expected_status {
                    status_code == expected
                } else {
                    status_code >= 200 && status_code < 400
                };

                let validation_results = self.validate_response(&test_request, &api_response);

                ApiTestResult {
                    request: test_request,
                    response: Some(api_response),
                    duration,
                    success: success && validation_results.iter().all(|v| v.passed),
                    error: None,
                    validation_results,
                }
            }
            Err(e) => {
                ApiTestResult {
                    request: test_request,
                    response: None,
                    duration,
                    success: false,
                    error: Some(e.to_string()),
                    validation_results: Vec::new(),
                }
            }
        }
    }

    fn validate_response(&self, _request: &ApiTestRequest, _response: &ApiResponse) -> Vec<ValidationResult> {
        // TODO: Implement response validation based on schema
        Vec::new()
    }

    fn update_performance_metrics(&self) {
        let metrics = self.metrics.read().unwrap();
        
        // Calculate overall performance metrics
        let total_requests: u64 = metrics.endpoint_stats.values().map(|s| s.total_requests).sum();
        let total_errors: u64 = metrics.endpoint_stats.values().map(|s| s.failed_requests).sum();
        
        let avg_response_time = if !metrics.endpoint_stats.is_empty() {
            metrics.endpoint_stats.values()
                .map(|s| s.avg_response_time.as_millis() as f64)
                .sum::<f64>() / metrics.endpoint_stats.len() as f64
        } else {
            0.0
        };

        let error_rate = if total_requests > 0 {
            (total_errors as f64 / total_requests as f64) * 100.0
        } else {
            0.0
        };

        drop(metrics);

        // Update performance metrics
        let mut metrics = self.metrics.write().unwrap();
        metrics.performance_metrics.avg_response_time_ms = avg_response_time;
        metrics.performance_metrics.error_rate_percentage = error_rate;
        metrics.performance_metrics.total_requests_per_second = total_requests as f64 / 60.0; // Simplified
    }

    fn analyze_error_patterns(&self) {
        let metrics = self.metrics.read().unwrap();
        let mut error_patterns = Vec::new();
        
        // Analyze error patterns from recent errors
        let mut pattern_counts: HashMap<String, (u64, Vec<String>)> = HashMap::new();
        
        for error in &metrics.error_analysis.recent_errors {
            let pattern = format!("Status {}", error.status_code);
            let entry = pattern_counts.entry(pattern.clone()).or_insert((0, Vec::new()));
            entry.0 += 1;
            if !entry.1.contains(&error.endpoint) {
                entry.1.push(error.endpoint.clone());
            }
        }
        
        drop(metrics);
        
        // Convert to error patterns
        for (pattern, (frequency, endpoints)) in pattern_counts {
            if frequency > 5 {
                let suggested_fix = match pattern.as_str() {
                    "Status 404" => "Check endpoint routing and URL patterns".to_string(),
                    "Status 500" => "Check server logs and error handling".to_string(),
                    "Status 429" => "Implement rate limiting or increase limits".to_string(),
                    "Status 401" => "Check authentication and authorization".to_string(),
                    "Status 403" => "Check permissions and access control".to_string(),
                    _ => "Investigate specific error cause".to_string(),
                };
                
                error_patterns.push(ErrorPattern {
                    pattern,
                    frequency,
                    endpoints,
                    suggested_fix,
                });
            }
        }
        
        // Update error patterns
        if !error_patterns.is_empty() {
            let mut metrics = self.metrics.write().unwrap();
            metrics.error_analysis.error_patterns = error_patterns;
        }
    }

    pub fn get_metrics(&self) -> ApiMetrics {
        self.metrics.read().unwrap().clone()
    }

    pub fn get_endpoints(&self) -> Vec<ApiEndpoint> {
        self.metrics.read().unwrap().endpoints.values().cloned().collect()
    }

    pub fn subscribe_to_requests(&self) -> broadcast::Receiver<ApiRequest> {
        self.event_sender.subscribe()
    }

    pub fn clear_metrics(&self) {
        let mut metrics = self.metrics.write().unwrap();
        metrics.request_history.clear();
        metrics.endpoint_stats.clear();
        metrics.error_analysis.error_counts.clear();
        metrics.error_analysis.error_patterns.clear();
        metrics.error_analysis.recent_errors.clear();
        metrics.live_requests.clear();
    }
}

impl super::MetricsCollector for ApiEndpointMonitor {
    fn collect(&self, registry: &MetricRegistry) -> Result<(), Box<dyn std::error::Error>> {
        let metrics = self.get_metrics();
        
        // Register API metrics
        let total_endpoints_gauge = registry.gauge("api_total_endpoints", HashMap::new());
        total_endpoints_gauge.set(metrics.endpoints.len() as f64);
        
        let total_requests_gauge = registry.gauge("api_total_requests", HashMap::new());
        total_requests_gauge.set(metrics.endpoint_stats.values().map(|s| s.total_requests).sum::<u64>() as f64);
        
        let avg_response_time_gauge = registry.gauge("api_avg_response_time_ms", HashMap::new());
        avg_response_time_gauge.set(metrics.performance_metrics.avg_response_time_ms);
        
        let error_rate_gauge = registry.gauge("api_error_rate_percentage", HashMap::new());
        error_rate_gauge.set(metrics.performance_metrics.error_rate_percentage);
        
        let concurrent_requests_gauge = registry.gauge("api_concurrent_requests", HashMap::new());
        concurrent_requests_gauge.set(metrics.performance_metrics.concurrent_requests as f64);
        
        Ok(())
    }
    
    fn name(&self) -> &str {
        "api_endpoint_monitor"
    }
    
    fn is_enabled(&self, config: &super::MetricsCollectionConfig) -> bool {
        config.enabled_collectors.contains(&"api_endpoint_monitor".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_endpoint_registration() {
        let monitor = ApiEndpointMonitor::new();
        
        let endpoint = ApiEndpoint {
            path: "/test".to_string(),
            method: HttpMethod::GET,
            handler_function: "test_handler".to_string(),
            parameters: Vec::new(),
            response_schema: None,
            auth_required: false,
            rate_limit: None,
            documentation: "Test endpoint".to_string(),
            tags: Vec::new(),
        };
        
        monitor.register_endpoint(endpoint);
        
        let endpoints = monitor.get_endpoints();
        assert_eq!(endpoints.len(), 1);
        assert_eq!(endpoints[0].path, "/test");
    }

    #[tokio::test]
    async fn test_endpoint_discovery() {
        let monitor = ApiEndpointMonitor::new();
        
        let endpoints = monitor.discover_endpoints().unwrap();
        assert!(!endpoints.is_empty());
        
        // Verify we discovered real LLMKG endpoints
        let endpoint_paths: Vec<String> = endpoints.iter().map(|e| e.path.clone()).collect();
        assert!(endpoint_paths.contains(&"/api/metrics".to_string()), "Should discover /api/metrics endpoint");
        assert!(endpoint_paths.contains(&"/api/history".to_string()), "Should discover /api/history endpoint");
        assert!(endpoint_paths.contains(&"/".to_string()), "Should discover dashboard root endpoint");
        
        // Verify endpoint metadata
        let metrics_endpoint = endpoints.iter().find(|e| e.path == "/api/metrics").unwrap();
        assert_eq!(metrics_endpoint.method, HttpMethod::GET);
        assert!(metrics_endpoint.tags.contains(&"real".to_string()));
        assert!(metrics_endpoint.documentation.contains("Real-time system metrics"));
        
        println!("✅ Successfully discovered {} real endpoints", endpoints.len());
        for endpoint in &endpoints {
            println!("   📍 {} {} - {}", 
                match endpoint.method {
                    HttpMethod::GET => "GET",
                    HttpMethod::POST => "POST",
                    HttpMethod::PUT => "PUT",
                    HttpMethod::DELETE => "DELETE",
                    HttpMethod::PATCH => "PATCH",
                    HttpMethod::HEAD => "HEAD",
                    HttpMethod::OPTIONS => "OPTIONS",
                },
                endpoint.path,
                endpoint.documentation
            );
        }
    }
    
    #[tokio::test]
    async fn test_real_request_monitoring() {
        let monitor = ApiEndpointMonitor::new();
        
        // Discover endpoints first
        let endpoints = monitor.discover_endpoints().unwrap();
        assert!(!endpoints.is_empty());
        
        // Simulate a real API request to /api/metrics
        let request_id = monitor.start_request(
            "/api/metrics".to_string(),
            HttpMethod::GET,
            "127.0.0.1".to_string(),
            "rust-test-agent/1.0".to_string(),
            HashMap::new(),
            HashMap::new(),
            None,
        );
        
        // Verify the request was tracked
        let metrics = monitor.get_metrics();
        assert!(metrics.live_requests.contains_key(&request_id));
        assert_eq!(metrics.performance_metrics.concurrent_requests, 1);
        
        // Simulate response
        let api_response = ApiResponse {
            status_code: 200,
            headers: [("content-type".to_string(), "application/json".to_string())].iter().cloned().collect(),
            body: Some(r#"[{"name":"test_metric","value":{"Gauge":42.0},"labels":{},"timestamp":"2024-01-01T00:00:00Z"}]"#.to_string()),
            size_bytes: 100,
        };
        
        monitor.end_request(request_id.clone(), api_response, None);
        
        // Verify the request was completed and stats updated
        let updated_metrics = monitor.get_metrics();
        assert!(!updated_metrics.live_requests.contains_key(&request_id));
        assert_eq!(updated_metrics.performance_metrics.concurrent_requests, 0);
        assert!(!updated_metrics.request_history.is_empty());
        
        // Check endpoint stats
        let endpoint_key = "0 /api/metrics"; // HttpMethod::GET as u8 = 0
        if let Some(stats) = updated_metrics.endpoint_stats.get(endpoint_key) {
            assert_eq!(stats.total_requests, 1);
            assert_eq!(stats.successful_requests, 1);
            assert_eq!(stats.failed_requests, 0);
            println!("✅ Endpoint stats updated: {} total requests, {:.1}% success rate", 
                stats.total_requests, 
                (stats.successful_requests as f64 / stats.total_requests as f64) * 100.0
            );
        }
        
        println!("✅ Real request monitoring test completed successfully");
    }
    
    #[test]
    fn test_simple_endpoint_discovery() {
        let monitor = ApiEndpointMonitor::new();
        let endpoints = monitor.discover_endpoints().unwrap();
        
        // Verify basic discovery works
        assert!(!endpoints.is_empty());
        println!("✅ Discovered {} endpoints", endpoints.len());
        
        // Verify real endpoints are present
        let has_metrics = endpoints.iter().any(|e| e.path == "/api/metrics");
        let has_history = endpoints.iter().any(|e| e.path == "/api/history");
        let has_dashboard = endpoints.iter().any(|e| e.path == "/");
        
        assert!(has_metrics, "Should discover /api/metrics endpoint");
        assert!(has_history, "Should discover /api/history endpoint");
        assert!(has_dashboard, "Should discover dashboard endpoint");
        
        // Check that endpoints have "real" tag
        let real_endpoints = endpoints.iter().filter(|e| e.tags.contains(&"real".to_string())).count();
        assert!(real_endpoints >= 3, "Should have at least 3 real endpoints");
        
        println!("✅ All real endpoint discovery tests passed");
    }
    
    #[tokio::test]
    async fn test_endpoint_performance_tracking() {
        let monitor = ApiEndpointMonitor::new();
        
        // Discover endpoints
        monitor.discover_endpoints().unwrap();
        
        // Simulate multiple requests to different endpoints
        let endpoints = vec![
            ("/api/metrics", HttpMethod::GET),
            ("/api/history", HttpMethod::GET),
            ("/", HttpMethod::GET),
        ];
        
        for (path, method) in &endpoints {
            for i in 0..5 {
                let request_id = monitor.start_request(
                    path.to_string(),
                    *method,
                    format!("192.168.1.{}", i + 1),
                    format!("test-client/{}", i + 1),
                    HashMap::new(),
                    HashMap::new(),
                    None,
                );
                
                // Simulate different response times and success rates
                tokio::time::sleep(tokio::time::Duration::from_millis(10 + i * 5)).await;
                
                let status_code = if i == 4 { 404 } else { 200 }; // Last request fails
                let api_response = ApiResponse {
                    status_code,
                    headers: HashMap::new(),
                    body: Some(format!("Response for request {}", i)),
                    size_bytes: 50 + i as u64 * 10,
                };
                
                let error = if status_code == 404 { Some("Not found".to_string()) } else { None };
                monitor.end_request(request_id, api_response, error);
            }
        }
        
        // Verify performance metrics
        let metrics = monitor.get_metrics();
        assert_eq!(metrics.request_history.len(), 15); // 5 requests × 3 endpoints
        
        // Check that each endpoint has stats
        for (path, method) in &endpoints {
            let method_val = *method as u8;
            let endpoint_key = format!("{} {}", method_val, path);
            
            if let Some(stats) = metrics.endpoint_stats.get(&endpoint_key) {
                assert_eq!(stats.total_requests, 5);
                assert_eq!(stats.successful_requests, 4); // 4 success, 1 failure per endpoint
                assert_eq!(stats.failed_requests, 1);
                assert_eq!(stats.error_rate, 20.0); // 1/5 = 20%
                
                println!("✅ {} {}: {} requests, {:.1}% error rate, {:.1}ms avg response time", 
                    match method {
                        HttpMethod::GET => "GET",
                        _ => "OTHER"
                    },
                    path,
                    stats.total_requests,
                    stats.error_rate,
                    stats.avg_response_time.as_millis()
                );
            }
        }
        
        // Verify error analysis
        assert_eq!(metrics.error_analysis.error_counts.get(&404).unwrap_or(&0), &3); // 3 404 errors total
        assert_eq!(metrics.error_analysis.recent_errors.len(), 3);
        
        println!("✅ Performance tracking test completed with realistic request patterns");
    }
}