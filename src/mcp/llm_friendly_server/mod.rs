//! LLM-Friendly MCP Server
//!
//! This module provides a Model Context Protocol (MCP) server designed specifically
//! for LLM interaction with the knowledge graph. It offers simplified, high-level
//! operations that are easy for LLMs to understand and use.

pub mod types;
pub mod tools;
pub mod validation;
pub mod query_generation;
pub mod query_generation_enhanced;
pub mod query_generation_native;
pub mod divergent_graph_traversal;
pub mod search_fusion;
pub mod utils;
pub mod handlers;
pub mod migration;
pub mod temporal_tracking;
pub mod database_branching;
pub mod reasoning_engine;

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::mcp::shared_types::{LLMMCPRequest, LLMMCPResponse, LLMMCPTool, PerformanceInfo};
use crate::mcp::MODEL_MANAGER;
// TODO: Temporarily disabled enhanced storage imports
// use crate::enhanced_knowledge_storage::{
//     hierarchical_storage::{HierarchicalStorageEngine, HierarchicalStorageConfig},
//     retrieval_system::{RetrievalEngine, RetrievalConfig},
// };
use crate::error::Result;
use crate::versioning::MultiDatabaseVersionManager;
use types::UsageStats;
use tools::get_tools;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};
use std::collections::HashMap;

/// Enhanced storage configuration for LLM-friendly MCP server
#[derive(Debug, Clone)]
pub struct EnhancedStorageConfig {
    pub enable_intelligent_processing: bool,
    pub enable_multi_hop_reasoning: bool,
    pub model_memory_limit: u64,
    pub max_processing_time_seconds: u64,
    pub fallback_on_failure: bool,
    pub cache_enhanced_results: bool,
}

impl Default for EnhancedStorageConfig {
    fn default() -> Self {
        Self {
            enable_intelligent_processing: true,
            enable_multi_hop_reasoning: true,
            model_memory_limit: 2_000_000_000, // 2GB
            max_processing_time_seconds: 30,
            fallback_on_failure: true,
            cache_enhanced_results: true,
        }
    }
}

/// LLM-Friendly MCP Server with Enhanced Storage Capabilities
/// 
/// Provides high-level, intuitive operations for knowledge graph interaction
/// that are specifically designed for LLM consumption and generation.
/// Now includes advanced AI-powered processing and retrieval capabilities.
pub struct LLMFriendlyMCPServer {
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
    usage_stats: Arc<RwLock<UsageStats>>,
    version_manager: Arc<MultiDatabaseVersionManager>,
    enhanced_config: EnhancedStorageConfig,
    // TODO: Temporarily disabled enhanced storage fields
    // hierarchical_storage: Option<Arc<HierarchicalStorageEngine>>,
    // retrieval_engine: Option<Arc<RetrievalEngine>>,
}

impl LLMFriendlyMCPServer {
    /// Create a new LLM-friendly MCP server with default enhanced storage
    pub fn new(knowledge_engine: Arc<RwLock<KnowledgeEngine>>) -> Result<Self> {
        Self::new_with_enhanced_config(knowledge_engine, EnhancedStorageConfig::default())
    }
    
    /// Create a new LLM-friendly MCP server with custom enhanced storage configuration
    pub fn new_with_enhanced_config(
        knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
        enhanced_config: EnhancedStorageConfig,
    ) -> Result<Self> {
        let version_manager = Arc::new(MultiDatabaseVersionManager::new()?);
        
        // Initialize the branch manager
        let branch_manager_clone = version_manager.clone();
        tokio::spawn(async move {
            database_branching::initialize_branch_manager(branch_manager_clone).await;
        });
        
        // TODO: Temporarily disabled enhanced storage initialization
        
        Ok(Self {
            knowledge_engine,
            usage_stats: Arc::new(RwLock::new(UsageStats::default())),
            version_manager,
            enhanced_config,
            // TODO: Temporarily disabled enhanced storage fields
            // hierarchical_storage,
            // retrieval_engine,
        })
    }
    
    // TODO: Enhanced storage initialization function temporarily removed
    
    /// Check if enhanced storage is available
    pub fn has_enhanced_storage(&self) -> bool {
        // TODO: Temporarily disabled
        false
    }
    
    /// Get enhanced storage configuration
    pub fn get_enhanced_config(&self) -> &EnhancedStorageConfig {
        &self.enhanced_config
    }

    /// Get all available tools
    pub fn get_available_tools(&self) -> Vec<LLMMCPTool> {
        get_tools()
    }

    /// Handle an incoming MCP request
    pub async fn handle_request(&self, request: LLMMCPRequest) -> Result<LLMMCPResponse> {
        // Update request stats
        let start_time = std::time::Instant::now();
        
        // Determine actual method and params (handle migration)
        let (method, params) = if let Some((new_method, new_params)) = migration::migrate_tool_call(&request.method, request.params.clone()) {
            log::warn!("{}", migration::deprecation_warning(&request.method));
            (new_method, new_params)
        } else {
            (request.method.clone(), request.params.clone())
        };
        
        // Wrap in timeout to prevent hanging
        let timeout_duration = tokio::time::Duration::from_secs(5);
        let timeout_result = tokio::time::timeout(timeout_duration, async {
            match method.as_str() {
            // Storage operations
            "store_fact" => {
                handlers::storage::handle_store_fact(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            "store_knowledge" => {
                handlers::storage::handle_store_knowledge(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }

            // Query operations
            "find_facts" => {
                handlers::query::handle_find_facts(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            "ask_question" => {
                handlers::query::handle_ask_question(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }

            // Exploration operations
            // "explore_connections" => migration (now part of analyze_graph)

            // Advanced operations
            "hybrid_search" => {
                handlers::advanced::handle_hybrid_search(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            "validate_knowledge" => {
                handlers::advanced::handle_validate_knowledge(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            "analyze_graph" => {
                handlers::graph_analysis::handle_analyze_graph(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }

            // Tier 1 Advanced Cognitive Tools
            "divergent_thinking_engine" => {
                handlers::cognitive::handle_divergent_thinking_engine(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            "time_travel_query" => {
                handlers::temporal::handle_time_travel_query(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            // Deprecated - handled by migration
            // "simd_ultra_fast_search" => migration
            // "analyze_graph_centrality" => migration

            // Tier 2 Advanced Tools
            // Deprecated - handled by migration
            // "hierarchical_clustering" => migration
            // "predict_graph_structure" => migration
            "cognitive_reasoning_chains" => {
                handlers::advanced::handle_cognitive_reasoning_chains(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            // Deprecated - handled by migration
            // "approximate_similarity_search" => migration
            // "knowledge_quality_metrics" => migration

            // Statistics operations
            "get_stats" => {
                handlers::stats::handle_get_stats(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            
            // Branching operations
            "create_branch" => {
                handlers::temporal::handle_create_branch(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                    self.version_manager.clone(),
                ).await
            }
            "list_branches" => {
                handlers::temporal::handle_list_branches(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            "compare_branches" => {
                handlers::temporal::handle_compare_branches(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            "merge_branches" => {
                handlers::temporal::handle_merge_branches(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }

            // Unknown method
            _ => {
                Err(format!("Unknown method: {method}"))
            }
        }
        }).await;

        // Handle timeout
        let result = match timeout_result {
            Ok(result) => result,
            Err(_) => Err(format!("Request timed out after {} seconds", timeout_duration.as_secs()))
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

    /// Get server health information including enhanced storage status
    pub async fn get_health(&self) -> HashMap<String, Value> {
        let stats = self.usage_stats.read().await;
        let engine_available = self.knowledge_engine.try_read().is_ok();
        let enhanced_available = self.has_enhanced_storage();
        
        let status = if engine_available && (!self.enhanced_config.enable_intelligent_processing || enhanced_available) {
            "healthy"
        } else {
            "degraded"
        };
        
        let mut health = HashMap::new();
        health.insert("status".to_string(), json!(status));
        health.insert("total_operations".to_string(), json!(stats.total_operations));
        health.insert("avg_response_time_ms".to_string(), json!(stats.avg_response_time_ms));
        health.insert("uptime_seconds".to_string(), json!(stats.uptime.elapsed().as_secs()));
        
        // Enhanced storage status
        health.insert("enhanced_storage".to_string(), json!({
            "enabled": self.enhanced_config.enable_intelligent_processing,
            "available": enhanced_available,
            "multi_hop_reasoning": self.enhanced_config.enable_multi_hop_reasoning,
            "memory_limit_gb": self.enhanced_config.model_memory_limit / 1_000_000_000,
            "fallback_on_failure": self.enhanced_config.fallback_on_failure
        }));
        
        // Model manager statistics
        let model_stats = MODEL_MANAGER.get_stats().await;
        health.insert("model_manager".to_string(), json!({
            "active_models": model_stats.active_models,
            "memory_usage_mb": model_stats.total_memory_usage / 1_000_000,
            "available_memory_mb": model_stats.available_memory / 1_000_000,
            "cache_utilization": model_stats.cache_utilization,
            "success_rate": model_stats.loader_success_rate,
            "tasks_processed": model_stats.total_tasks_processed
        }));
        
        health
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    
    #[tokio::test]
    async fn test_server_creation() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
        let server = LLMFriendlyMCPServer::new(engine).unwrap();
        
        let tools = server.get_available_tools();
        assert!(!tools.is_empty());
        assert!(tools.iter().any(|t| t.name == "store_fact"));
        assert!(tools.iter().any(|t| t.name == "ask_question"));
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
        let server = LLMFriendlyMCPServer::new(engine).unwrap();
        
        let health = server.get_health().await;
        assert_eq!(health.get("status").unwrap(), &json!("healthy"));
        assert_eq!(health.get("total_operations").unwrap(), &json!(0));
    }
    
    #[tokio::test]
    async fn test_unknown_method() {
        let engine = Arc::new(RwLock::new(KnowledgeEngine::new(768, 1_000_000).unwrap()));
        let server = LLMFriendlyMCPServer::new(engine).unwrap();
        
        let request = LLMMCPRequest {
            method: "unknown_method".to_string(),
            params: serde_json::Value::Null,
        };
        
        let response = server.handle_request(request).await.unwrap();
        assert!(!response.success);
        assert!(response.message.contains("Unknown method"));
    }
}