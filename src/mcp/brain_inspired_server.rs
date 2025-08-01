use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::json;
use ahash::AHashMap;

use crate::core::brain_types::{
    BrainInspiredEntity, EntityDirection
};
use crate::core::types::{EntityKey, AttributeValue};
use crate::versioning::temporal_graph::TemporalKnowledgeGraph;
use crate::cognitive::{CognitiveOrchestrator, CognitiveOrchestratorConfig, ReasoningStrategy, CognitivePatternType};
use crate::error::{Result, GraphError};

use crate::mcp::shared_types::{MCPTool, MCPRequest, MCPResponse, MCPContent};

/// Brain-inspired MCP server with cognitive capabilities
pub struct BrainInspiredMCPServer {
    pub knowledge_graph: Arc<RwLock<TemporalKnowledgeGraph>>,
    pub cognitive_orchestrator: Option<Arc<CognitiveOrchestrator>>,
}

impl BrainInspiredMCPServer {
    pub async fn new(
        knowledge_graph: Arc<RwLock<TemporalKnowledgeGraph>>,
    ) -> Result<Self> {
        Ok(Self {
            knowledge_graph,
            cognitive_orchestrator: None,
        })
    }

    /// Initialize with cognitive orchestrator for cognitive capabilities
    pub async fn new_with_cognitive_capabilities(
        knowledge_graph: Arc<RwLock<TemporalKnowledgeGraph>>,
        brain_graph: Arc<crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph>,
    ) -> Result<Self> {
        // Initialize cognitive orchestrator
        let cognitive_config = CognitiveOrchestratorConfig::default();
        let cognitive_orchestrator = CognitiveOrchestrator::new(
            brain_graph,
            cognitive_config,
        ).await?;
        
        Ok(Self {
            knowledge_graph,
            cognitive_orchestrator: Some(Arc::new(cognitive_orchestrator)),
        })
    }

    /// Get all available tools
    pub fn get_tools(&self) -> Vec<MCPTool> {
        let mut tools = vec![
            self.create_store_knowledge_tool(),
            self.create_query_tool(),
        ];

        // Add cognitive tools if orchestrator is available
        if self.cognitive_orchestrator.is_some() {
            tools.push(self.create_cognitive_reasoning_tool());
        }

        tools
    }

    /// Handle tool execution
    pub async fn handle_tool_call(&self, request: MCPRequest) -> Result<MCPResponse> {
        match request.tool.as_str() {
            "store_knowledge" => self.handle_store_knowledge(request.arguments).await,
            "query" => self.handle_query(request.arguments).await,
            "cognitive_reasoning" => self.handle_cognitive_reasoning_tool_call(request.arguments).await,
            _ => Err(GraphError::InvalidInput(format!("Unknown tool: {}", request.tool))),
        }
    }

    /// Store knowledge using graph-based processing
    pub async fn handle_store_knowledge(&self, args: serde_json::Value) -> Result<MCPResponse> {
        let text = args["text"].as_str()
            .ok_or_else(|| GraphError::InvalidInput("Missing 'text' parameter".to_string()))?;
        let context = args["context"].as_str().map(|s| s.to_string());

        self.handle_store_fact_simple(text, context).await
    }

    /// Simple graph construction implementation
    pub async fn handle_store_fact_simple(
        &self,
        text: &str,
        context: Option<String>,
    ) -> Result<MCPResponse> {
        // Simple storage using graph-based processing
        let graph = self.knowledge_graph.write().await;
        let mut metadata = AHashMap::new();
        metadata.insert("text".to_string(), AttributeValue::String(text.to_string()));
        if let Some(ctx) = context {
            metadata.insert("context".to_string(), AttributeValue::String(ctx));
        }
        
        // Store as a simple temporal entity
        let brain_entity = BrainInspiredEntity {
            id: EntityKey::default(),
            concept_id: format!("fact_{}", uuid::Uuid::new_v4()),
            direction: EntityDirection::Input,
            properties: metadata.into_iter().map(|(k, v)| match v {
                AttributeValue::String(s) => (k, crate::core::types::AttributeValue::String(s)),
                AttributeValue::Number(n) => (k, crate::core::types::AttributeValue::Number(n)),
                _ => (k, crate::core::types::AttributeValue::String(format!("{:?}", v))),
            }).collect(),
            embedding: vec![0.0; 384], // Default embedding
            activation_state: 0.0,
            last_activation: std::time::SystemTime::now(),
            last_update: std::time::SystemTime::now(),
        };
        
        let time_range = crate::versioning::temporal_graph::TimeRange::new(chrono::Utc::now());
        graph.insert_temporal_entity(brain_entity, time_range).await?;
        
        Ok(MCPResponse {
            content: vec![MCPContent {
                type_: "text".to_string(),
                text: format!("Stored fact: {}", text),
            }],
            is_error: false,
        })
    }

    /// Simple query using graph-based processing
    pub async fn handle_query(&self, args: serde_json::Value) -> Result<MCPResponse> {
        let query = args["query"].as_str()
            .ok_or_else(|| GraphError::InvalidInput("Missing 'query' parameter".to_string()))?;
        let query_type = args["query_type"].as_str().unwrap_or("exact");
        let top_k = args["top_k"].as_u64().unwrap_or(10) as usize;
        
        // Search the knowledge graph
        let graph = self.knowledge_graph.read().await;
        let mut results = Vec::new();
        
        match query_type {
            "exact" => {
                // Perform exact match search
                let entities = graph.search_entities(query).await;
                for entity in entities.into_iter().take(top_k) {
                    results.push(format!(
                        "Entity: {}\nActivation: {:.3}",
                        entity.concept_id, entity.activation_state
                    ));
                }
            },
            "pattern" => {
                // Use cognitive pattern matching
                if let Some(orchestrator) = &self.cognitive_orchestrator {
                    let reasoning_result = orchestrator.reason(
                        query,
                        None,
                        ReasoningStrategy::Specific(CognitivePatternType::Divergent),
                    ).await?;
                    
                    // Extract results from the reasoning result
                    results.push(format!(
                        "Query: {}\nAnswer: {}\nConfidence: {:.3}\nStrategy: {:?}",
                        reasoning_result.query,
                        reasoning_result.final_answer,
                        reasoning_result.quality_metrics.overall_confidence,
                        reasoning_result.strategy_used
                    ));
                } else {
                    return Err(GraphError::ProcessingError("Cognitive orchestrator not available for pattern queries".to_string()));
                }
            },
            _ => {
                return Err(GraphError::InvalidInput(format!("Unknown query type: {}", query_type)));
            }
        }
        
        let response_text = if results.is_empty() {
            format!("No results found for query: '{}'", query)
        } else {
            format!(
                "Query Results for '{}' (type: {}):\n\n{}",
                query,
                query_type,
                results.join("\n\n")
            )
        };
        
        Ok(MCPResponse {
            content: vec![MCPContent {
                type_: "text".to_string(),
                text: response_text,
            }],
            is_error: false,
        })
    }

    /// Handle cognitive reasoning tool call
    async fn handle_cognitive_reasoning_tool_call(&self, args: serde_json::Value) -> Result<MCPResponse> {
        let query = args["query"].as_str()
            .ok_or_else(|| GraphError::InvalidInput("Missing 'query' parameter".to_string()))?;
        let context = args["context"].as_str();
        let pattern = args["pattern"].as_str().and_then(|p| match p {
            "convergent" => Some(CognitivePatternType::Convergent),
            "divergent" => Some(CognitivePatternType::Divergent),
            "lateral" => Some(CognitivePatternType::Lateral),
            "systems" => Some(CognitivePatternType::Systems),
            "critical" => Some(CognitivePatternType::Critical),
            "abstract" => Some(CognitivePatternType::Abstract),
            "adaptive" => Some(CognitivePatternType::Adaptive),
            _ => None,
        });

        if let Some(orchestrator) = &self.cognitive_orchestrator {
            let strategy = if let Some(pattern_type) = pattern {
                ReasoningStrategy::Specific(pattern_type)
            } else {
                ReasoningStrategy::Automatic
            };
            
            let result = orchestrator.reason(query, context, strategy).await?;
            
            Ok(MCPResponse {
                content: vec![MCPContent {
                    type_: "text".to_string(),
                    text: format!(
                        "Cognitive Reasoning Result:\n\nQuery: {}\nStrategy: {:?}\nAnswer: {}\nConfidence: {:.2}\nExecution Time: {}ms\nPatterns Used: {:?}",
                        result.query,
                        result.strategy_used,
                        result.final_answer,
                        result.quality_metrics.overall_confidence,
                        result.execution_metadata.total_time_ms,
                        result.execution_metadata.patterns_executed
                    ),
                }],
                is_error: false,
            })
        } else {
            Err(GraphError::ProcessingError("Cognitive orchestrator not initialized".to_string()))
        }
    }

    /// Create tool definitions
    fn create_store_knowledge_tool(&self) -> MCPTool {
        MCPTool {
            name: "store_knowledge".to_string(),
            description: "Store knowledge in the graph".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The knowledge to store"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context for the knowledge"
                    }
                },
                "required": ["text"]
            }),
        }
    }

    fn create_query_tool(&self) -> MCPTool {
        MCPTool {
            name: "query".to_string(),
            description: "Query knowledge graph".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to execute"
                    },
                    "query_type": {
                        "type": "string",
                        "description": "Type of query: exact or pattern",
                        "enum": ["exact", "pattern"],
                        "default": "exact"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }),
        }
    }

    /// Create cognitive reasoning tool
    fn create_cognitive_reasoning_tool(&self) -> MCPTool {
        MCPTool {
            name: "cognitive_reasoning".to_string(),
            description: "Execute cognitive reasoning using patterns".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to reason about"
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional context for reasoning"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Specific cognitive pattern to use",
                        "enum": ["convergent", "divergent", "lateral", "systems", "critical", "abstract", "adaptive"]
                    }
                },
                "required": ["query"]
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::KnowledgeGraph;

    #[tokio::test]
    async fn test_mcp_server_creation() {
        let graph = KnowledgeGraph::new(384).unwrap();
        let temporal_graph = Arc::new(RwLock::new(TemporalKnowledgeGraph::new(graph)));
        
        let mcp_server = BrainInspiredMCPServer::new(temporal_graph).await.unwrap();
        let tools = mcp_server.get_tools();
        
        assert!(!tools.is_empty());
        assert!(tools.iter().any(|t| t.name == "store_knowledge"));
        assert!(tools.iter().any(|t| t.name == "query"));
    }
}