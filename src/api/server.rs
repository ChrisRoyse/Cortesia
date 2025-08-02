use std::sync::Arc;
use parking_lot::RwLock;
use crate::core::knowledge_engine::KnowledgeEngine;
use crate::monitoring::dashboard::{DashboardConfig, DashboardServer};
use crate::monitoring::metrics::MetricRegistry;
use crate::monitoring::collectors::{MetricsCollector, SystemMetricsCollector, ApplicationMetricsCollector, SystemMetricsConfig, ApplicationMetricsConfig, KnowledgeEngineMetricsCollector};
use warp::Filter;
use super::routes;

/// Unified API server that includes both LLMKG operations and monitoring
pub struct LLMKGApiServer {
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
    metric_registry: Arc<MetricRegistry>,
    config: ApiServerConfig,
}

#[derive(Debug, Clone)]
pub struct ApiServerConfig {
    pub api_port: u16,
    pub dashboard_http_port: u16,
    pub dashboard_websocket_port: u16,
    pub embedding_dim: usize,
    pub max_nodes: usize,
}

impl Default for ApiServerConfig {
    fn default() -> Self {
        Self {
            api_port: 3001,
            dashboard_http_port: 8090,
            dashboard_websocket_port: 8081,
            embedding_dim: 384,
            max_nodes: 1000000,
        }
    }
}

impl LLMKGApiServer {
    pub fn new(config: ApiServerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let knowledge_engine = Arc::new(RwLock::new(
            KnowledgeEngine::new(config.embedding_dim, config.max_nodes)?
        ));
        
        let metric_registry = Arc::new(MetricRegistry::new());
        
        Ok(Self {
            knowledge_engine,
            metric_registry,
            config,
        })
    }
    
    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        // Start the monitoring dashboard in the background
        let dashboard_config = DashboardConfig {
            http_port: self.config.dashboard_http_port,
            websocket_port: self.config.dashboard_websocket_port,
            ..Default::default()
        };
        
        // Create collectors including brain metrics
        let collectors: Vec<Box<dyn MetricsCollector>> = vec![
            Box::new(SystemMetricsCollector::new(SystemMetricsConfig::default())),
            Box::new(ApplicationMetricsCollector::new(ApplicationMetricsConfig::default())),
            Box::new(KnowledgeEngineMetricsCollector::new(self.knowledge_engine.clone())),
        ];
        
        let dashboard = DashboardServer::new(
            dashboard_config,
            self.metric_registry.clone(),
            collectors,
        );
        
        // Start dashboard in background
        tokio::spawn(async move {
            if let Err(e) = dashboard.start().await {
                eprintln!("Dashboard error: {e}");
            }
        });
        
        // Create API routes
        let api = routes::api_routes(self.knowledge_engine.clone());
        
        // Add monitoring endpoints to API
        let metric_registry_clone = self.metric_registry.clone();
        let monitoring_api = warp::path("api")
            .and(warp::path("v1"))
            .and(warp::path("monitoring"))
            .and(warp::path("metrics"))
            .and(warp::get())
            .map(move || {
                let samples = metric_registry_clone.collect_all_samples();
                warp::reply::json(&samples)
            });
        
        let routes = api.or(monitoring_api);
        
        println!("🚀 LLMKG API Server starting...");
        println!("📡 API endpoints: http://localhost:{}/api/v1", self.config.api_port);
        println!("📊 Dashboard: http://localhost:{}", self.config.dashboard_http_port);
        println!("🔌 WebSocket: ws://localhost:{}", self.config.dashboard_websocket_port);
        println!("📖 API Discovery: http://localhost:{}/api/v1/discovery", self.config.api_port);
        
        warp::serve(routes)
            .run(([127, 0, 0, 1], self.config.api_port))
            .await;
        
        Ok(())
    }
}