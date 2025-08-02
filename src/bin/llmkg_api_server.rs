use llmkg::api::server::{LLMKGApiServer, ApiServerConfig};
use llmkg::enhanced_knowledge_storage::logging;
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    if let Err(e) = logging::init_logging() {
        eprintln!("Failed to initialize logging: {e}");
        std::process::exit(1);
    }
    
    // Configure the API server
    let config = ApiServerConfig {
        api_port: 3001,
        dashboard_http_port: 8090,
        dashboard_websocket_port: 8081,
        embedding_dim: 384,
        max_nodes: 1000000,
    };
    
    info!("🚀 Starting LLMKG API Server...");
    info!("📡 API Port: {}", config.api_port);
    info!("📊 Dashboard Port: {}", config.dashboard_http_port);
    info!("🔌 WebSocket Port: {}", config.dashboard_websocket_port);
    
    // Create and run the server
    let server = LLMKGApiServer::new(config).map_err(|e| {
        error!("Failed to create API server: {}", e);
        e
    })?;
    
    info!("LLMKG API Server starting up...");
    server.run().await.map_err(|e| {
        error!("API server failed: {}", e);
        e
    })?;
    
    Ok(())
}