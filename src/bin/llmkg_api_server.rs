use llmkg::api::server::{LLMKGApiServer, ApiServerConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure the API server
    let config = ApiServerConfig {
        api_port: 3001,
        dashboard_http_port: 8090,
        dashboard_websocket_port: 8081,
        embedding_dim: 384,
        max_nodes: 1000000,
    };
    
    println!("🚀 Starting LLMKG API Server...");
    println!("📡 API Port: {}", config.api_port);
    println!("📊 Dashboard Port: {}", config.dashboard_http_port);
    println!("🔌 WebSocket Port: {}", config.dashboard_websocket_port);
    
    // Create and run the server
    let server = LLMKGApiServer::new(config)?;
    server.run().await?;
    
    Ok(())
}