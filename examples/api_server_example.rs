//! Production API Server Example
//! 
//! This example demonstrates how to start the Enhanced Knowledge Storage API server
//! with a production configuration.

use std::env;
use tracing::{info, Level};
use tracing_subscriber;

use llmkg::enhanced_knowledge_storage::production::{
    ApiServer, ProductionConfig, Environment
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting Enhanced Knowledge Storage API Server");

    // Load configuration from environment or defaults
    let config = match ProductionConfig::load() {
        Ok(config) => config,
        Err(_) => {
            info!("Using default development configuration");
            ProductionConfig::for_environment(Environment::Development)?
        }
    };

    // Override with environment variables for demo
    let mut config = config;
    if let Ok(port) = env::var("API_PORT") {
        config.api_config.port = port.parse().unwrap_or(8080);
    }
    if let Ok(host) = env::var("API_HOST") {
        config.api_config.host = host;
    }

    info!(
        "API Configuration: {}:{} (TLS: {})",
        config.api_config.host,
        config.api_config.port,
        config.api_config.tls_enabled
    );

    // Create and start the API server
    let api_server = ApiServer::new(config).await?;
    
    info!("API Server initialized successfully");
    info!("Starting server...");
    info!("Swagger UI available at: http://{}:{}/swagger-ui", 
          api_server.config.host, api_server.config.port);
    
    // Start the server (this will block)
    api_server.start().await?;

    Ok(())
}