use llmkg::monitoring::dashboard::{DashboardConfig, DashboardServer};
use llmkg::monitoring::metrics::MetricRegistry;
use llmkg::monitoring::collectors::{SystemMetricsCollector, ApplicationMetricsCollector, SystemMetricsConfig, ApplicationMetricsConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the metric registry
    let registry = Arc::new(MetricRegistry::new());
    
    // Create collector configs
    let system_config = SystemMetricsConfig {
        collect_cpu: true,
        collect_memory: true,
        collect_disk: true,
        collect_network: true,
        collect_load: true,
    };
    
    let app_config = ApplicationMetricsConfig {
        collect_performance: true,
        collect_operations: true,
        collect_errors: true,
        collect_resources: true,
    };
    
    // Create collectors
    let collectors: Vec<Box<dyn llmkg::monitoring::collectors::MetricsCollector>> = vec![
        Box::new(SystemMetricsCollector::new(system_config)),
        Box::new(ApplicationMetricsCollector::new(app_config)),
    ];
    
    // Configure the dashboard
    let config = DashboardConfig {
        http_port: 8080,
        websocket_port: 8081,
        ..Default::default()
    };
    
    println!("Starting LLMKG Dashboard Server...");
    println!("HTTP Dashboard: http://localhost:{}", config.http_port);
    println!("WebSocket: ws://localhost:{}", config.websocket_port);
    
    // Create and start the server
    let server = DashboardServer::new(config, registry, collectors);
    server.start().await?;
    
    Ok(())
}