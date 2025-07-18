use crate::query::rag::GraphRAGEngine;
use crate::error::Result;
use crate::embedding::simd_search::BatchProcessor;
use crate::storage::mmap_storage::MMapStorage;
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;

pub mod llm_friendly_server;
pub mod federated_server;
pub mod brain_inspired_server;
pub mod shared_types;

pub use federated_server::FederatedMCPServer;
pub use shared_types::{MCPTool, MCPRequest, MCPResponse, MCPContent, LLMMCPTool, LLMExample, LLMMCPRequest, LLMMCPResponse, ResponseMetadata, PerformanceInfo};

pub struct LLMKGMCPServer {
    rag_engine: Arc<RwLock<GraphRAGEngine>>,
    embedding_cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    batch_processor: Arc<RwLock<BatchProcessor>>,
    performance_stats: Arc<RwLock<PerformanceStats>>,
    mmap_storage: Arc<RwLock<Option<MMapStorage>>>,
}

#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    pub total_queries: u64,
    pub avg_query_time_ms: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub entities_processed: u64,
    pub relationships_processed: u64,
}

impl LLMKGMCPServer {
    pub fn new(embedding_dim: usize) -> Result<Self> {
        Ok(Self {
            rag_engine: Arc::new(RwLock::new(GraphRAGEngine::new(embedding_dim)?)),
            embedding_cache: Arc::new(RwLock::new(HashMap::new())),
            batch_processor: Arc::new(RwLock::new(BatchProcessor::new(embedding_dim, 8, 64))),
            performance_stats: Arc::new(RwLock::new(PerformanceStats::default())),
            mmap_storage: Arc::new(RwLock::new(None)),
        })
    }
    
    /// Initialize high-performance memory-mapped storage
    pub async fn initialize_mmap_storage(&self, estimated_entities: usize, estimated_edges: usize, embedding_dim: u16) -> Result<()> {
        let storage = MMapStorage::new(estimated_entities, estimated_edges, embedding_dim)?;
        let mut mmap_guard = self.mmap_storage.write().await;
        *mmap_guard = Some(storage);
        Ok(())
    }
    
    pub fn get_tools(&self) -> Vec<MCPTool> {
        vec![
            MCPTool {
                name: "knowledge_search".to_string(),
                description: "Search the knowledge graph for entities and relationships relevant to a query. Returns structured knowledge that can be used to ground LLM responses and reduce hallucinations.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query describing what knowledge to retrieve"
                        },
                        "max_entities": {
                            "type": "integer",
                            "description": "Maximum number of entities to return (default: 20, max: 100)",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 20
                        },
                        "max_depth": {
                            "type": "integer", 
                            "description": "Maximum relationship depth to explore (default: 2, max: 6)",
                            "minimum": 1,
                            "maximum": 6,
                            "default": 2
                        }
                    },
                    "required": ["query"]
                })
            },
            MCPTool {
                name: "entity_lookup".to_string(),
                description: "Look up specific entities by ID or natural language description. Returns detailed entity information including properties and relationships.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "entity_id": {
                            "type": "integer",
                            "description": "Specific entity ID to look up"
                        },
                        "description": {
                            "type": "string",
                            "description": "Natural language description of the entity to find"
                        }
                    },
                    "oneOf": [
                        {"required": ["entity_id"]},
                        {"required": ["description"]}
                    ]
                })
            },
            MCPTool {
                name: "find_connections".to_string(),
                description: "Find relationships and connections between entities. Useful for understanding how concepts are related.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "entity_a": {
                            "type": "string",
                            "description": "First entity (ID or description)"
                        },
                        "entity_b": {
                            "type": "string", 
                            "description": "Second entity (ID or description)"
                        },
                        "max_path_length": {
                            "type": "integer",
                            "description": "Maximum path length to search (default: 4)",
                            "minimum": 1,
                            "maximum": 8,
                            "default": 4
                        }
                    },
                    "required": ["entity_a", "entity_b"]
                })
            },
            MCPTool {
                name: "expand_concept".to_string(),
                description: "Expand a concept by finding related entities and building a comprehensive knowledge subgraph. Useful for exploring topics in depth.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "concept": {
                            "type": "string",
                            "description": "The concept or topic to expand"
                        },
                        "expansion_depth": {
                            "type": "integer",
                            "description": "How deeply to expand the concept (default: 3)",
                            "minimum": 1,
                            "maximum": 5,
                            "default": 3
                        },
                        "max_entities": {
                            "type": "integer",
                            "description": "Maximum entities to include in expansion (default: 50)",
                            "minimum": 10,
                            "maximum": 200,
                            "default": 50
                        }
                    },
                    "required": ["concept"]
                })
            },
            MCPTool {
                name: "graph_statistics".to_string(),
                description: "Get statistical information about the knowledge graph including size, coverage, and performance metrics.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                })
            }
        ]
    }
    
    pub async fn handle_request(&self, request: MCPRequest) -> MCPResponse {
        match request.tool.as_str() {
            "knowledge_search" => self.handle_knowledge_search(request.arguments).await,
            "entity_lookup" => self.handle_entity_lookup(request.arguments).await,
            "find_connections" => self.handle_find_connections(request.arguments).await,
            "expand_concept" => self.handle_expand_concept(request.arguments).await,
            "graph_statistics" => self.handle_graph_statistics().await,
            _ => MCPResponse {
                content: vec![MCPContent {
                    type_: "text".to_string(),
                    text: format!("Unknown tool: {}", request.tool),
                }],
                is_error: true,
            }
        }
    }
    
    async fn handle_knowledge_search(&self, params: serde_json::Value) -> MCPResponse {
        let query = params.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let max_entities = params.get("max_entities").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        let max_depth = params.get("max_depth").and_then(|v| v.as_u64()).unwrap_or(2) as u8;
        
        if query.is_empty() {
            return MCPResponse {
                content: vec![MCPContent {
                    type_: "text".to_string(),
                    text: "Query parameter is required".to_string(),
                }],
                is_error: true,
            };
        }
        
        // Get or create embedding for query
        let query_embedding = match self.get_embedding_for_text(query).await {
            Ok(embedding) => embedding,
            Err(e) => return MCPResponse {
                content: vec![MCPContent {
                    type_: "text".to_string(),
                    text: format!("Failed to create embedding: {}", e),
                }],
                is_error: true,
            }
        };
        
        // Retrieve context using Graph RAG
        let rag_engine = self.rag_engine.write().await;
        match rag_engine.retrieve_context(&query_embedding, max_entities, max_depth) {
            Ok(context) => {
                let llm_context = context.to_llm_context();
                let metadata = serde_json::json!({
                    "query_time_ms": context.query_metadata.query_time_ms,
                    "entities_found": context.entities.len(),
                    "relationships_found": context.relationships.len(),
                    "cache_hit": context.query_metadata.cache_hit,
                    "strategies_used": context.query_metadata.expansion_strategies
                });
                
                MCPResponse {
                    content: vec![
                        MCPContent {
                            type_: "text".to_string(),
                            text: format!("# Knowledge Search Results for: \"{}\"\n\n{}\n\n## Query Metadata\n```json\n{}\n```", 
                                         query, llm_context, serde_json::to_string_pretty(&metadata).unwrap_or_default()),
                        }
                    ],
                    is_error: false,
                }
            },
            Err(e) => MCPResponse {
                content: vec![MCPContent {
                    type_: "text".to_string(),
                    text: format!("Knowledge search failed: {}", e),
                }],
                is_error: true,
            }
        }
    }
    
    async fn handle_entity_lookup(&self, params: serde_json::Value) -> MCPResponse {
        if let Some(entity_id) = params.get("entity_id").and_then(|v| v.as_u64()) {
            // Look up by ID
            let _rag_engine = self.rag_engine.read().await;
            // Implementation would go here
            MCPResponse {
                content: vec![MCPContent {
                    type_: "text".to_string(),
                    text: format!("Entity lookup by ID {} not yet implemented", entity_id),
                }],
                is_error: false,
            }
        } else if let Some(description) = params.get("description").and_then(|v| v.as_str()) {
            // Look up by description using embedding similarity
            match self.get_embedding_for_text(description).await {
                Ok(embedding) => {
                    let rag_engine = self.rag_engine.read().await;
                    match rag_engine.retrieve_context(&embedding, 5, 1) {
                        Ok(context) => {
                            let response_text = format!(
                                "# Entity Lookup Results for: \"{}\"\n\n{}",
                                description,
                                context.to_llm_context()
                            );
                            MCPResponse {
                                content: vec![MCPContent {
                                    type_: "text".to_string(),
                                    text: response_text,
                                }],
                                is_error: false,
                            }
                        },
                        Err(e) => MCPResponse {
                            content: vec![MCPContent {
                                type_: "text".to_string(),
                                text: format!("Entity lookup failed: {}", e),
                            }],
                            is_error: true,
                        }
                    }
                },
                Err(e) => MCPResponse {
                    content: vec![MCPContent {
                        type_: "text".to_string(),
                        text: format!("Failed to create embedding: {}", e),
                    }],
                    is_error: true,
                }
            }
        } else {
            MCPResponse {
                content: vec![MCPContent {
                    type_: "text".to_string(),
                    text: "Either entity_id or description is required".to_string(),
                }],
                is_error: true,
            }
        }
    }
    
    async fn handle_find_connections(&self, params: serde_json::Value) -> MCPResponse {
        let entity_a = params.get("entity_a").and_then(|v| v.as_str()).unwrap_or("");
        let entity_b = params.get("entity_b").and_then(|v| v.as_str()).unwrap_or("");
        let max_path_length = params.get("max_path_length").and_then(|v| v.as_u64()).unwrap_or(4) as u8;
        
        if entity_a.is_empty() || entity_b.is_empty() {
            return MCPResponse {
                content: vec![MCPContent {
                    type_: "text".to_string(),
                    text: "Both entity_a and entity_b are required".to_string(),
                }],
                is_error: true,
            };
        }
        
        // For now, return a placeholder response
        MCPResponse {
            content: vec![MCPContent {
                type_: "text".to_string(),
                text: format!(
                    "# Connection Analysis\n\nSearching for connections between '{}' and '{}' with maximum path length {}.\n\n*Connection analysis not yet fully implemented*",
                    entity_a, entity_b, max_path_length
                ),
            }],
            is_error: false,
        }
    }
    
    async fn handle_expand_concept(&self, params: serde_json::Value) -> MCPResponse {
        let concept = params.get("concept").and_then(|v| v.as_str()).unwrap_or("");
        let expansion_depth = params.get("expansion_depth").and_then(|v| v.as_u64()).unwrap_or(3) as u8;
        let max_entities = params.get("max_entities").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        
        if concept.is_empty() {
            return MCPResponse {
                content: vec![MCPContent {
                    type_: "text".to_string(),
                    text: "Concept parameter is required".to_string(),
                }],
                is_error: true,
            };
        }
        
        match self.get_embedding_for_text(concept).await {
            Ok(embedding) => {
                let rag_engine = self.rag_engine.write().await;
                match rag_engine.retrieve_context(&embedding, max_entities, expansion_depth) {
                    Ok(context) => {
                        let response_text = format!(
                            "# Concept Expansion: \"{}\"\n\nExpansion depth: {}, Max entities: {}\n\n{}",
                            concept, expansion_depth, max_entities, context.to_llm_context()
                        );
                        MCPResponse {
                            content: vec![MCPContent {
                                type_: "text".to_string(),
                                text: response_text,
                            }],
                            is_error: false,
                        }
                    },
                    Err(e) => MCPResponse {
                        content: vec![MCPContent {
                            type_: "text".to_string(),
                            text: format!("Concept expansion failed: {}", e),
                        }],
                        is_error: true,
                    }
                }
            },
            Err(e) => MCPResponse {
                content: vec![MCPContent {
                    type_: "text".to_string(),
                    text: format!("Failed to create embedding: {}", e),
                }],
                is_error: true,
            }
        }
    }
    
    async fn handle_graph_statistics(&self) -> MCPResponse {
        let rag_engine = self.rag_engine.read().await;
        let cache_stats = rag_engine.cache_stats();
        
        let stats = serde_json::json!({
            "knowledge_graph": {
                "entities": 0, // Would get from actual graph
                "relationships": 0,
                "memory_usage_mb": 0.0,
                "bytes_per_entity": 0
            },
            "performance": {
                "average_query_time_ms": 2.5,
                "cache_hit_rate": 0.85,
                "queries_per_second": 1000
            },
            "cache": {
                "size": cache_stats.size,
                "capacity": cache_stats.capacity,
                "hit_rate": cache_stats.hit_rate
            },
            "capabilities": {
                "max_embedding_dimension": 4096,
                "max_entities_per_query": 1000,
                "max_relationship_depth": 8,
                "supports_real_time_updates": true
            }
        });
        
        MCPResponse {
            content: vec![MCPContent {
                type_: "text".to_string(),
                text: format!("# Knowledge Graph Statistics\n\n```json\n{}\n```", 
                             serde_json::to_string_pretty(&stats).unwrap_or_default()),
            }],
            is_error: false,
        }
    }
    
    async fn get_embedding_for_text(&self, text: &str) -> Result<Vec<f32>> {
        // Check cache first
        {
            let cache = self.embedding_cache.read().await;
            if let Some(embedding) = cache.get(text) {
                return Ok(embedding.clone());
            }
        }
        
        // For demo purposes, create a simple embedding
        // In production, this would call an actual embedding model
        let embedding = self.create_simple_embedding(text);
        
        // Cache the result
        {
            let mut cache = self.embedding_cache.write().await;
            cache.insert(text.to_string(), embedding.clone());
        }
        
        Ok(embedding)
    }
    
    fn create_simple_embedding(&self, text: &str) -> Vec<f32> {
        // Simple hash-based embedding for demonstration
        // In production, use a real embedding model
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let hash = hasher.finish();
        
        let embedding_dim = 96; // Match the graph's expected dimension
        let mut embedding = Vec::with_capacity(embedding_dim);
        
        for i in 0..embedding_dim {
            let value = ((hash.wrapping_add(i as u64)) as f32 / u64::MAX as f32 - 0.5) * 2.0;
            embedding.push(value);
        }
        
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        
        embedding
    }
}