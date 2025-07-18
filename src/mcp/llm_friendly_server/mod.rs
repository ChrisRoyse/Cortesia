//! LLM-Friendly MCP Server
//!
//! This module provides a Model Context Protocol (MCP) server designed specifically
//! for LLM interaction with the knowledge graph. It offers simplified, high-level
//! operations that are easy for LLMs to understand and use.

pub mod types;
pub mod tools;
pub mod validation;
pub mod query_generation;
pub mod search_fusion;
pub mod utils;
pub mod handlers;

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::mcp::shared_types::{LLMMCPRequest, LLMMCPResponse, LLMMCPTool};
use crate::error::Result;
use types::UsageStats;
use tools::get_tools;
use utils::update_usage_stats;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};
use std::collections::HashMap;

/// LLM-Friendly MCP Server
/// 
/// Provides high-level, intuitive operations for knowledge graph interaction
/// that are specifically designed for LLM consumption and generation.
pub struct LLMFriendlyMCPServer {
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
    usage_stats: Arc<RwLock<UsageStats>>,
}

impl LLMFriendlyMCPServer {
    /// Create a new LLM-friendly MCP server
    pub fn new(knowledge_engine: Arc<RwLock<KnowledgeEngine>>) -> Self {
        Self {
            knowledge_engine,
            usage_stats: Arc::new(RwLock::new(UsageStats::default())),
        }
    }

    /// Get all available tools
    pub fn get_available_tools(&self) -> Vec<LLMMCPTool> {
        get_tools()
    }

    /// Handle an incoming MCP request
    pub async fn handle_request(&self, request: LLMMCPRequest) -> Result<LLMMCPResponse> {
        // Update request stats
        let start_time = std::time::Instant::now();
        
        let result = match request.method.as_str() {
            // Storage operations
            "store_fact" => {
                handlers::storage::handle_store_fact(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    request.params.unwrap_or(serde_json::Value::Null),
                ).await
            }
            "store_knowledge" => {
                handlers::storage::handle_store_knowledge(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    request.params.unwrap_or(serde_json::Value::Null),
                ).await
            }

            // Query operations
            "find_facts" => {
                handlers::query::handle_find_facts(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    request.params.unwrap_or(serde_json::Value::Null),
                ).await
            }
            "ask_question" => {
                handlers::query::handle_ask_question(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    request.params.unwrap_or(serde_json::Value::Null),
                ).await
            }

            // Exploration operations
            "explore_connections" => {
                handlers::exploration::handle_explore_connections(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    request.params.unwrap_or(serde_json::Value::Null),
                ).await
            }
            "get_suggestions" => {
                handlers::exploration::handle_get_suggestions(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    request.params.unwrap_or(serde_json::Value::Null),
                ).await
            }

            // Advanced operations
            "generate_graph_query" => {
                handlers::advanced::handle_generate_graph_query(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    request.params.unwrap_or(serde_json::Value::Null),
                ).await
            }
            "hybrid_search" => {
                handlers::advanced::handle_hybrid_search(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    request.params.unwrap_or(serde_json::Value::Null),
                ).await
            }
            "validate_knowledge" => {
                handlers::advanced::handle_validate_knowledge(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    request.params.unwrap_or(serde_json::Value::Null),
                ).await
            }

            // Statistics operations
            "get_stats" => {
                handlers::stats::handle_get_stats(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    request.params.unwrap_or(serde_json::Value::Null),
                ).await
            }

            // Unknown method
            _ => Err(format!("Unknown method: {}", request.method))
        };

        // Record response time
        let elapsed = start_time.elapsed().as_millis() as f64;
        {
            let mut stats = self.usage_stats.write().await;
            stats.total_operations += 1;
            stats.avg_response_time_ms = 
                (stats.avg_response_time_ms * (stats.total_operations - 1) as f64 + elapsed) 
                / stats.total_operations as f64;
        }

        // Format response
        match result {
            Ok((data, message, suggestions)) => Ok(LLMMCPResponse {
                success: true,
                data,
                message,
                helpful_info: None,
                suggestions,
                performance: PerformanceInfo {
                    execution_time_ms: elapsed,
                    memory_used_bytes: 0,
                    cache_hit: false,
                    complexity_score: 0.5,
                },
            }),
            Err(error_msg) => Ok(LLMMCPResponse {
                success: false,
                data: serde_json::Value::Null,
                message: error_msg,
                helpful_info: None,
                suggestions: vec!["Check the input parameters".to_string()],
                performance: PerformanceInfo {
                    execution_time_ms: elapsed,
                    memory_used_bytes: 0,
                    cache_hit: false,
                    complexity_score: 0.5,
                },
            }),
        }
    }

    /// Get current usage statistics
    pub async fn get_usage_stats(&self) -> UsageStats {
        self.usage_stats.read().await.clone()
    }

    /// Reset usage statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.usage_stats.write().await;
        *stats = UsageStats::default();
    }

    /// Get server health information
    pub async fn get_health(&self) -> HashMap<String, Value> {
        let stats = self.usage_stats.read().await;
        let engine_available = self.knowledge_engine.try_read().is_ok();
        
        let mut health = HashMap::new();
        health.insert("status".to_string(), json!(if engine_available { "healthy" } else { "degraded" }));
        health.insert("total_operations".to_string(), json!(stats.total_operations));
        health.insert("avg_response_time_ms".to_string(), json!(stats.avg_response_time_ms));
        health.insert("uptime_seconds".to_string(), json!(stats.uptime.elapsed().as_secs()));
        
        health
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::brain_types::EntityKey;
    
    #[tokio::test]
    async fn test_server_creation() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new()));
        let server = LLMFriendlyMCPServer::new(engine);
        
        let tools = server.get_available_tools();
        assert!(!tools.is_empty());
        assert!(tools.iter().any(|t| t.name == "store_fact"));
        assert!(tools.iter().any(|t| t.name == "ask_question"));
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new()));
        let server = LLMFriendlyMCPServer::new(engine);
        
        let health = server.get_health().await;
        assert_eq!(health.get("status").unwrap(), &json!("healthy"));
        assert_eq!(health.get("total_operations").unwrap(), &json!(0));
    }
    
    #[tokio::test]
    async fn test_unknown_method() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new()));
        let server = LLMFriendlyMCPServer::new(engine);
        
        let request = LLMMCPRequest {
            id: "test".to_string(),
            method: "unknown_method".to_string(),
            params: None,
        };
        
        let response = server.handle_request(request).await.unwrap();
        assert!(response.error.is_some());
    }
}