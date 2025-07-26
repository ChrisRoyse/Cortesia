//! Production-Ready MCP Server
//!
//! This module provides a production-ready wrapper around the LLM-friendly MCP server
//! that integrates all production features including error recovery, monitoring,
//! rate limiting, health checks, and graceful shutdown.

use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::Value;
use std::collections::HashMap;

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::mcp::llm_friendly_server::LLMFriendlyMCPServer;
use crate::mcp::shared_types::{LLMMCPRequest, LLMMCPResponse, LLMMCPTool, PerformanceInfo};
use crate::production::{
    ProductionSystem, ProductionConfig, create_production_system_with_config
};
use crate::error::Result;
use crate::cognitive::orchestrator::CognitiveOrchestratorConfig;
use crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use crate::federation::registry::DatabaseRegistry;

/// Production-ready MCP server with comprehensive production features
pub struct ProductionMCPServer {
    // Core MCP server
    inner_server: LLMFriendlyMCPServer,
    
    // Production system
    production_system: Arc<ProductionSystem>,
    
    // Configuration
    config: ProductionConfig,
}

impl ProductionMCPServer {
    /// Create a new production MCP server
    pub fn new(
        knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
        config: Option<ProductionConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        
        // Create the inner MCP server with required dependencies
        // TODO: These should be passed in or created properly
        use crate::cognitive::orchestrator::CognitiveOrchestrator;
        use crate::neural::neural_server::NeuralProcessingServer;
        use crate::federation::coordinator::FederationCoordinator;
        
        // Create BrainEnhancedKnowledgeGraph
        let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new(768)?); // 768 is standard BERT embedding size
        
        // Create CognitiveOrchestrator
        let cognitive_config = CognitiveOrchestratorConfig::default();
        let cognitive_orchestrator = Arc::new(
            futures::executor::block_on(CognitiveOrchestrator::new(brain_graph, cognitive_config))?
        );
        
        // Create NeuralProcessingServer
        let neural_server = Arc::new(futures::executor::block_on(
            NeuralProcessingServer::new("127.0.0.1:50051".to_string())
        )?);
        
        // Create DatabaseRegistry and FederationCoordinator
        let registry = Arc::new(DatabaseRegistry::new()?);
        let federation_coordinator = Arc::new(
            futures::executor::block_on(FederationCoordinator::new(registry))?
        );
        
        let inner_server = LLMFriendlyMCPServer::new(
            knowledge_engine.clone(),
            cognitive_orchestrator,
            neural_server,
            federation_coordinator,
        )?;
        
        // Create the production system
        let production_system = Arc::new(create_production_system_with_config(
            knowledge_engine,
            config.clone(),
        ));
        
        Ok(Self {
            inner_server,
            production_system,
            config,
        })
    }

    /// Get all available tools (same as inner server)
    pub fn get_available_tools(&self) -> Vec<LLMMCPTool> {
        let mut tools = self.inner_server.get_available_tools();
        
        // Add production-specific tools
        tools.extend(self.get_production_tools());
        
        tools
    }

    /// Handle an incoming MCP request with full production protection
    pub async fn handle_request(&self, request: LLMMCPRequest) -> Result<LLMMCPResponse> {
        let method = request.method.clone();
        let start_time = std::time::Instant::now();
        
        // Handle production-specific methods
        if let Some(response) = self.handle_production_request(&request).await? {
            return Ok(response);
        }
        
        // For regular operations, use production system protection
        let result = self.production_system.execute_protected_operation(
            &method,
            None, // Could extract user_id from request metadata if available
            || async {
                // Execute the request through the inner server
                self.inner_server.handle_request(request.clone()).await
            }
        ).await;

        match result {
            Ok(response) => Ok(response),
            Err(error) => {
                // Create error response in MCP format
                let elapsed = start_time.elapsed().as_millis() as f64;
                Ok(LLMMCPResponse {
                    success: false,
                    data: serde_json::Value::Null,
                    message: error.to_string(),
                    helpful_info: Some("Check system health and try again".to_string()),
                    suggestions: vec![
                        "Try reducing request complexity".to_string(),
                        "Check system resources".to_string(),
                        "Wait and retry if rate limited".to_string(),
                    ],
                    performance: PerformanceInfo {
                        execution_time_ms: elapsed,
                        memory_used_bytes: 0,
                        cache_hit: false,
                        complexity_score: 1.0,
                    },
                })
            }
        }
    }

    /// Get comprehensive system status including production metrics
    pub async fn get_system_status(&self) -> HashMap<String, Value> {
        let mut status = HashMap::new();
        
        // Get base server health
        let base_health = self.inner_server.get_health().await;
        status.insert("mcp_server".to_string(), serde_json::json!(base_health));
        
        // Get production system status
        let production_status = self.production_system.get_system_status().await;
        status.extend(production_status);
        
        // Add overall assessment
        let overall_health = self.assess_overall_health(&status).await;
        status.insert("overall_health".to_string(), serde_json::json!(overall_health));
        
        status
    }

    /// Get health report in a standard format
    pub async fn get_health_report(&self) -> serde_json::Value {
        let production_report = self.production_system.get_health_report().await;
        let base_health = self.inner_server.get_health().await;
        
        serde_json::json!({
            "timestamp": chrono::Utc::now().timestamp(),
            "overall_status": production_report.overall_status,
            "uptime_seconds": production_report.uptime_seconds,
            "mcp_server": base_health,
            "production_system": production_report,
            "checks_performed": production_report.component_results.len(),
        })
    }

    /// Get metrics in Prometheus format
    pub async fn get_prometheus_metrics(&self) -> String {
        let mut metrics = self.production_system.get_prometheus_metrics().await;
        
        // Add MCP-specific metrics
        let base_health = self.inner_server.get_health().await;
        if let Some(total_ops) = base_health.get("total_operations") {
            metrics.push_str(&format!("# TYPE mcp_total_operations counter\n"));
            metrics.push_str(&format!("mcp_total_operations {}\n", total_ops));
        }
        
        if let Some(avg_response) = base_health.get("avg_response_time_ms") {
            metrics.push_str(&format!("# TYPE mcp_avg_response_time_ms gauge\n"));
            metrics.push_str(&format!("mcp_avg_response_time_ms {}\n", avg_response));
        }
        
        metrics
    }

    /// Initiate graceful shutdown
    pub async fn shutdown(&self) -> Result<crate::production::graceful_shutdown::ShutdownReport> {
        self.production_system.shutdown().await
    }

    /// Access the inner MCP server for advanced operations
    pub fn inner_server(&self) -> &LLMFriendlyMCPServer {
        &self.inner_server
    }

    /// Access the production system for advanced configuration
    pub fn production_system(&self) -> &Arc<ProductionSystem> {
        &self.production_system
    }

    /// Check if the server is ready to handle requests
    pub async fn is_ready(&self) -> bool {
        !self.production_system.shutdown_manager().is_shutting_down()
    }

    /// Check if the server is healthy
    pub async fn is_healthy(&self) -> bool {
        let health_report = self.production_system.get_health_report().await;
        matches!(health_report.overall_status, crate::production::HealthStatus::Healthy)
    }

    // Private helper methods

    async fn handle_production_request(&self, request: &LLMMCPRequest) -> Result<Option<LLMMCPResponse>> {
        let start_time = std::time::Instant::now();
        
        let response = match request.method.as_str() {
            "get_system_status" => {
                let status = self.get_system_status().await;
                Some(LLMMCPResponse {
                    success: true,
                    data: serde_json::json!(status),
                    message: "System status retrieved successfully".to_string(),
                    helpful_info: Some("Monitor these metrics for system health".to_string()),
                    suggestions: vec![
                        "Check 'overall_health' for quick assessment".to_string(),
                        "Review 'health.components' for detailed status".to_string(),
                        "Monitor 'resources' for capacity planning".to_string(),
                    ],
                    performance: PerformanceInfo {
                        execution_time_ms: start_time.elapsed().as_millis() as f64,
                        memory_used_bytes: 0,
                        cache_hit: true,
                        complexity_score: 0.1,
                    },
                })
            }
            
            "get_health_report" => {
                let report = self.get_health_report().await;
                Some(LLMMCPResponse {
                    success: true,
                    data: report,
                    message: "Health report generated successfully".to_string(),
                    helpful_info: Some("Detailed system health assessment".to_string()),
                    suggestions: vec![
                        "Review component health regularly".to_string(),
                        "Address any 'warning' or 'critical' statuses".to_string(),
                    ],
                    performance: PerformanceInfo {
                        execution_time_ms: start_time.elapsed().as_millis() as f64,
                        memory_used_bytes: 0,
                        cache_hit: false,
                        complexity_score: 0.2,
                    },
                })
            }
            
            "get_prometheus_metrics" => {
                let metrics = self.get_prometheus_metrics().await;
                Some(LLMMCPResponse {
                    success: true,
                    data: serde_json::json!({"metrics": metrics}),
                    message: "Prometheus metrics exported successfully".to_string(),
                    helpful_info: Some("Use these metrics with monitoring systems".to_string()),
                    suggestions: vec![
                        "Configure Prometheus to scrape these metrics".to_string(),
                        "Set up alerts based on key metrics".to_string(),
                    ],
                    performance: PerformanceInfo {
                        execution_time_ms: start_time.elapsed().as_millis() as f64,
                        memory_used_bytes: 0,
                        cache_hit: true,
                        complexity_score: 0.1,
                    },
                })
            }
            
            "readiness_check" => {
                let ready = self.is_ready().await;
                Some(LLMMCPResponse {
                    success: ready,
                    data: serde_json::json!({"ready": ready}),
                    message: if ready { 
                        "Server is ready to handle requests".to_string() 
                    } else { 
                        "Server is not ready (shutting down)".to_string() 
                    },
                    helpful_info: Some("Use this endpoint for load balancer health checks".to_string()),
                    suggestions: if ready {
                        vec!["Server is ready for requests".to_string()]
                    } else {
                        vec!["Wait for server restart".to_string()]
                    },
                    performance: PerformanceInfo {
                        execution_time_ms: start_time.elapsed().as_millis() as f64,
                        memory_used_bytes: 0,
                        cache_hit: true,
                        complexity_score: 0.01,
                    },
                })
            }
            
            "liveness_check" => {
                let healthy = self.is_healthy().await;
                Some(LLMMCPResponse {
                    success: healthy,
                    data: serde_json::json!({"healthy": healthy}),
                    message: if healthy { 
                        "Server is healthy".to_string() 
                    } else { 
                        "Server has health issues".to_string() 
                    },
                    helpful_info: Some("Use this endpoint for kubernetes liveness probes".to_string()),
                    suggestions: if healthy {
                        vec!["All systems operational".to_string()]
                    } else {
                        vec![
                            "Check get_health_report for details".to_string(),
                            "Review system logs for errors".to_string(),
                        ]
                    },
                    performance: PerformanceInfo {
                        execution_time_ms: start_time.elapsed().as_millis() as f64,
                        memory_used_bytes: 0,
                        cache_hit: false,
                        complexity_score: 0.1,
                    },
                })
            }
            
            _ => None,
        };
        
        Ok(response)
    }

    fn get_production_tools(&self) -> Vec<LLMMCPTool> {
        vec![
            LLMMCPTool {
                name: "get_system_status".to_string(),
                description: "Get comprehensive system status including health, resources, and performance metrics".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
                examples: vec![],
                tips: vec!["Use this to monitor overall system health".to_string()],
            },
            LLMMCPTool {
                name: "get_health_report".to_string(),
                description: "Get detailed health report with component status and diagnostics".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
                examples: vec![],
                tips: vec!["Review component health regularly for proactive monitoring".to_string()],
            },
            LLMMCPTool {
                name: "get_prometheus_metrics".to_string(),
                description: "Export system metrics in Prometheus format for monitoring integration".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
                examples: vec![],
                tips: vec!["Integrate with Prometheus for automated monitoring".to_string()],
            },
            LLMMCPTool {
                name: "readiness_check".to_string(),
                description: "Check if server is ready to handle requests (for load balancers)".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
                examples: vec![],
                tips: vec!["Use for load balancer health checks".to_string()],
            },
            LLMMCPTool {
                name: "liveness_check".to_string(),
                description: "Check if server is healthy and functioning properly (for kubernetes)".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
                examples: vec![],
                tips: vec!["Use for Kubernetes liveness probes".to_string()],
            },
        ]
    }

    async fn assess_overall_health(&self, status: &HashMap<String, Value>) -> String {
        // Simple health assessment based on key indicators
        let mut score = 100.0f64;
        
        // Check health status
        if let Some(health) = status.get("health") {
            if let Some(overall_status) = health.get("overall_status") {
                match overall_status.as_str() {
                    Some("critical") => score -= 50.0,
                    Some("warning") => score -= 20.0,
                    Some("unknown") => score -= 10.0,
                    _ => {}
                }
            }
        }
        
        // Check resource usage
        if let Some(resources) = status.get("resources") {
            if let Some(memory_usage) = resources.get("memory_bytes") {
                if let Some(memory_limit) = resources.get("memory_limit_bytes") {
                    if let (Some(usage), Some(limit)) = (memory_usage.as_u64(), memory_limit.as_u64()) {
                        let usage_percent = (usage as f64 / limit as f64) * 100.0;
                        if usage_percent > 90.0 {
                            score -= 15.0;
                        } else if usage_percent > 75.0 {
                            score -= 5.0;
                        }
                    }
                }
            }
        }
        
        // Check error rates
        if let Some(error_recovery) = status.get("error_recovery") {
            // Check for high error rates across operations
            let mut total_attempts = 0u64;
            let mut total_failures = 0u64;
            
            if let Some(stats) = error_recovery.as_object() {
                for (_op_name, op_stats) in stats {
                    if let Some(op_stats) = op_stats.as_object() {
                        if let (Some(attempts), Some(failures)) = (
                            op_stats.get("total_attempts").and_then(|v| v.as_u64()),
                            op_stats.get("failed_attempts").and_then(|v| v.as_u64())
                        ) {
                            total_attempts += attempts;
                            total_failures += failures;
                        }
                    }
                }
            }
            
            if total_attempts > 0 {
                let error_rate = (total_failures as f64 / total_attempts as f64) * 100.0;
                if error_rate > 10.0 {
                    score -= 20.0;
                } else if error_rate > 5.0 {
                    score -= 10.0;
                }
            }
        }
        
        // Determine health category
        if score >= 95.0 {
            "excellent".to_string()
        } else if score >= 85.0 {
            "good".to_string()
        } else if score >= 70.0 {
            "fair".to_string()
        } else if score >= 50.0 {
            "poor".to_string()
        } else {
            "critical".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::knowledge_engine::KnowledgeEngine;

    #[tokio::test]
    async fn test_production_mcp_server_creation() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
        let server = ProductionMCPServer::new(engine, None).unwrap();
        
        // Should be ready initially
        assert!(server.is_ready().await);
        
        // Should have production tools
        let tools = server.get_available_tools();
        assert!(tools.iter().any(|t| t.name == "get_system_status"));
        assert!(tools.iter().any(|t| t.name == "readiness_check"));
        assert!(tools.iter().any(|t| t.name == "liveness_check"));
    }

    #[tokio::test]
    async fn test_production_request_handling() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
        let server = ProductionMCPServer::new(engine, None).unwrap();
        
        // Test system status request
        let request = LLMMCPRequest {
            method: "get_system_status".to_string(),
            params: serde_json::Value::Null,
        };
        
        let response = server.handle_request(request).await.unwrap();
        assert!(response.success);
        assert!(response.data.is_object());
    }

    #[tokio::test]
    async fn test_health_checks() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
        let server = ProductionMCPServer::new(engine, None).unwrap();
        
        // Test readiness check
        let readiness_request = LLMMCPRequest {
            method: "readiness_check".to_string(),
            params: serde_json::Value::Null,
        };
        
        let response = server.handle_request(readiness_request).await.unwrap();
        assert!(response.success);
        
        // Test liveness check
        let liveness_request = LLMMCPRequest {
            method: "liveness_check".to_string(),
            params: serde_json::Value::Null,
        };
        
        let response = server.handle_request(liveness_request).await.unwrap();
        assert!(response.success);
    }

    #[tokio::test]
    async fn test_regular_operation_protection() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
        let server = ProductionMCPServer::new(engine, None).unwrap();
        
        // Test a regular MCP operation (should be protected by production system)
        let request = LLMMCPRequest {
            method: "store_fact".to_string(),
            params: serde_json::json!({
                "subject": "test_subject",
                "predicate": "is",
                "object": "test_object",
                "confidence": 0.9
            }),
        };
        
        let response = server.handle_request(request).await.unwrap();
        assert!(response.success);
        
        // Check that metrics were recorded
        let status = server.get_system_status().await;
        assert!(status.contains_key("error_recovery"));
    }
}