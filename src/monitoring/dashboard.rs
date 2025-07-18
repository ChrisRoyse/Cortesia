/*!
Phase 5.4: Real-Time Performance Dashboard
Web-based real-time monitoring dashboard with WebSocket support
*/

use crate::monitoring::metrics::{MetricRegistry, MetricSample, MetricValue};
use crate::monitoring::collectors::MetricsCollector;
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use warp::Filter;

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
            http_port: 8080,
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
    pub alerts: Vec<AlertSnapshot>,
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
    Ping,
    Pong,
}

pub struct PerformanceDashboard {
    config: DashboardConfig,
    registry: Arc<MetricRegistry>,
    collectors: Vec<Box<dyn MetricsCollector>>,
    metrics_history: Arc<RwLock<Vec<RealTimeMetrics>>>,
    websocket_clients: Arc<Mutex<Vec<tokio::sync::mpsc::UnboundedSender<Message>>>>,
    is_running: Arc<RwLock<bool>>,
}

impl PerformanceDashboard {
    pub fn new(
        config: DashboardConfig,
        registry: Arc<MetricRegistry>,
        collectors: Vec<Box<dyn MetricsCollector>>,
    ) -> Self {
        Self {
            config,
            registry,
            collectors,
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            websocket_clients: Arc::new(Mutex::new(Vec::new())),
            is_running: Arc::new(RwLock::new(false)),
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
        let _collectors = self.collectors.iter().map(|c| c.name().to_string()).collect::<Vec<_>>();
        let history = self.metrics_history.clone();
        let clients = self.websocket_clients.clone();
        let is_running = self.is_running.clone();
        let update_interval = self.config.update_interval;
        let history_size = self.config.history_size;
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(update_interval);
            
            while *is_running.read().unwrap() {
                interval.tick().await;
                
                // Collect metrics (this would need to be implemented)
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
        let metrics_route = warp::path!("api" / "metrics")
            .and(warp::get())
            .map(move || {
                let samples = registry.collect_all_samples();
                warp::reply::json(&samples)
            });
        
        let history_route = warp::path!("api" / "history")
            .and(warp::get())
            .map(move || {
                let history_data = history.read().unwrap().clone();
                warp::reply::json(&history_data)
            });
        
        metrics_route.or(history_route)
    }
    
    async fn collect_real_time_metrics(registry: &MetricRegistry) -> RealTimeMetrics {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let samples = registry.collect_all_samples();
        let metrics_map = Self::samples_to_map(samples);
        
        RealTimeMetrics {
            timestamp,
            system_metrics: Self::extract_system_metrics(&metrics_map),
            application_metrics: Self::extract_application_metrics(&metrics_map),
            performance_metrics: Self::extract_performance_metrics(&metrics_map),
            alerts: vec![], // TODO: Implement alert collection
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
    
    fn extract_performance_metrics(metrics: &HashMap<String, MetricSample>) -> PerformanceMetricsSnapshot {
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
    
    fn generate_dashboard_html(config: &DashboardConfig) -> String {
        format!(r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
            new PerformanceDashboard();
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

impl WebSocketHandler {
    pub fn new() -> Self {
        Self
    }
}