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
pub mod error_handling;

use crate::core::knowledge_engine::KnowledgeEngine;
use crate::cognitive::orchestrator::CognitiveOrchestrator;
use crate::neural::neural_server::NeuralProcessingServer;
use crate::federation::coordinator::FederationCoordinator;
use crate::mcp::shared_types::{LLMMCPRequest, LLMMCPResponse, LLMMCPTool, PerformanceInfo};
use crate::error::Result;
use crate::versioning::MultiDatabaseVersionManager;
use types::UsageStats;
use tools::get_tools;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::{json, Value};
use std::collections::HashMap;

/// LLM-Friendly MCP Server
/// 
/// Provides high-level, intuitive operations for knowledge graph interaction
/// that are specifically designed for LLM consumption and generation.
/// 
/// **Enhanced with Cognitive Intelligence**: All 28 tools now include cognitive metadata,
/// neural confidence scoring, and reasoning pattern analysis.
pub struct LLMFriendlyMCPServer {
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
    cognitive_orchestrator: Arc<CognitiveOrchestrator>,
    neural_server: Arc<NeuralProcessingServer>,
    federation_coordinator: Arc<FederationCoordinator>,
    usage_stats: Arc<RwLock<UsageStats>>,
    version_manager: Arc<MultiDatabaseVersionManager>,
}

impl LLMFriendlyMCPServer {
    /// Create a new LLM-friendly MCP server with cognitive enhancements
    pub fn new(
        knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
        cognitive_orchestrator: Arc<CognitiveOrchestrator>,
        neural_server: Arc<NeuralProcessingServer>,
        federation_coordinator: Arc<FederationCoordinator>,
    ) -> Result<Self> {
        let version_manager = Arc::new(MultiDatabaseVersionManager::new()?);
        
        // Initialize the branch manager
        let branch_manager_clone = version_manager.clone();
        tokio::spawn(async move {
            database_branching::initialize_branch_manager(branch_manager_clone).await;
        });
        
        Ok(Self {
            knowledge_engine,
            cognitive_orchestrator,
            neural_server,
            federation_coordinator,
            usage_stats: Arc::new(RwLock::new(UsageStats::default())),
            version_manager,
        })
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
            // Storage operations (COGNITIVE ENHANCED)
            "store_fact" => {
                handlers::storage::handle_store_fact_enhanced(
                    &self.knowledge_engine,
                    &self.cognitive_orchestrator,
                    &self.neural_server,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            "store_knowledge" => {
                handlers::storage::handle_store_knowledge_enhanced(
                    &self.knowledge_engine,
                    &self.cognitive_orchestrator,
                    &self.neural_server,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }

            // Query operations (using regular handlers - no enhanced versions exist)
            "find_facts" => {
                handlers::query::handle_find_facts(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            "ask_question" => {
                // Use cognitive-enhanced question answering
                handlers::cognitive_query::handle_ask_question_cognitive_enhanced(
                    &self.knowledge_engine,
                    &self.cognitive_orchestrator,
                    &self.neural_server,
                    &self.federation_coordinator,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }

            // Exploration operations (using regular handler - no enhanced version exists)
            // "explore_connections" => migration (now part of analyze_graph)
            "get_suggestions" => {
                handlers::exploration::handle_get_suggestions(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }

            // Advanced operations (mixed - some have enhanced versions)
            "generate_graph_query" => {
                handlers::advanced::handle_generate_graph_query(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            "hybrid_search" => {
                // This one has an enhanced version
                handlers::enhanced_search::handle_hybrid_search_enhanced(
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

            // Tier 1 Advanced Cognitive Tools (using regular handlers - no enhanced versions exist)
            "neural_importance_scoring" => {
                handlers::cognitive::handle_neural_importance_scoring(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            "divergent_thinking_engine" => {
                handlers::cognitive::handle_divergent_thinking_engine(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            "time_travel_query" => {
                // Using temporal version instead of cognitive
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

            // Statistics operations (using regular handler - no enhanced version exists)
            "get_stats" => {
                handlers::stats::handle_get_stats(
                    &self.knowledge_engine,
                    &self.usage_stats,
                    params.clone(),
                ).await
            }
            
            // Branching operations (using regular handlers - no enhanced versions exist)
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

            // NEW COGNITIVE-SPECIFIC TOOLS
            "cognitive_reasoning" => {
                handlers::storage::handle_cognitive_reasoning(
                    &self.cognitive_orchestrator,
                    &self.usage_stats,
                    params.clone(),
                ).await.map_err(|e| e.to_string())
            }
            "neural_train_model" => {
                handlers::storage::handle_neural_train_model(
                    &self.neural_server,
                    &self.usage_stats,
                    params.clone(),
                ).await.map_err(|e| e.to_string())
            }
            "neural_predict" => {
                handlers::storage::handle_neural_predict(
                    &self.neural_server,
                    &self.usage_stats,
                    params.clone(),
                ).await.map_err(|e| e.to_string())
            }

            // Unknown method
            _ => {
                Err(format!("Unknown method: {}", method))
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