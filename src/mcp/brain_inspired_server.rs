use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json::json;
use ahash::AHashMap;

use crate::core::brain_types::{
    BrainInspiredEntity, EntityDirection, LogicGate,
    BrainInspiredRelationship, GraphOperation
};
use crate::core::types::{EntityKey, AttributeValue};
use crate::versioning::temporal_graph::{TemporalKnowledgeGraph, TimeRange};
use crate::neural::neural_server::NeuralProcessingServer;
use crate::neural::structure_predictor::GraphStructurePredictor;
use crate::neural::canonicalization::NeuralCanonicalizer;
use crate::cognitive::{CognitiveOrchestrator, CognitiveOrchestratorConfig, ReasoningStrategy, CognitivePatternType};
use crate::error::{Result, GraphError};

use crate::mcp::shared_types::{MCPTool, MCPRequest, MCPResponse, MCPContent};

/// Brain-inspired MCP server with Phase 2 cognitive capabilities
pub struct BrainInspiredMCPServer {
    pub knowledge_graph: Arc<RwLock<TemporalKnowledgeGraph>>,
    pub neural_server: Arc<NeuralProcessingServer>,
    pub structure_predictor: Arc<GraphStructurePredictor>,
    pub canonicalizer: Arc<NeuralCanonicalizer>,
    pub cognitive_orchestrator: Option<Arc<CognitiveOrchestrator>>,
}

impl BrainInspiredMCPServer {
    pub fn new(
        knowledge_graph: Arc<RwLock<TemporalKnowledgeGraph>>,
        neural_server: Arc<NeuralProcessingServer>,
    ) -> Self {
        let structure_predictor = Arc::new(GraphStructurePredictor::new(
            "structure_model".to_string(),
            neural_server.clone(),
        ));
        
        let canonicalizer = Arc::new(NeuralCanonicalizer::new_with_neural_server(
            neural_server.clone(),
        ));
        
        Self {
            knowledge_graph,
            neural_server,
            structure_predictor,
            canonicalizer,
            cognitive_orchestrator: None,
        }
    }

    /// Initialize with cognitive orchestrator for Phase 2 capabilities
    pub async fn new_with_cognitive_capabilities(
        knowledge_graph: Arc<RwLock<TemporalKnowledgeGraph>>,
        neural_server: Arc<NeuralProcessingServer>,
        brain_graph: Arc<crate::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph>,
    ) -> Result<Self> {
        let structure_predictor = Arc::new(GraphStructurePredictor::new(
            "structure_model".to_string(),
            neural_server.clone(),
        ));
        
        let canonicalizer = Arc::new(NeuralCanonicalizer::new_with_neural_server(
            neural_server.clone(),
        ));

        // Initialize cognitive orchestrator
        let cognitive_config = CognitiveOrchestratorConfig::default();
        let cognitive_orchestrator = CognitiveOrchestrator::new(
            brain_graph,
            cognitive_config,
        ).await?;
        
        Ok(Self {
            knowledge_graph,
            neural_server,
            structure_predictor,
            canonicalizer,
            cognitive_orchestrator: Some(Arc::new(cognitive_orchestrator)),
        })
    }

    /// Get all available tools
    pub fn get_tools(&self) -> Vec<MCPTool> {
        let mut tools = vec![
            self.create_store_knowledge_tool(),
            self.create_neural_query_tool(),
        ];

        // Add Phase 2 cognitive tools if orchestrator is available
        if self.cognitive_orchestrator.is_some() {
            tools.push(self.create_cognitive_reasoning_tool());
        }

        tools
    }

    /// Handle tool execution
    pub async fn handle_tool_call(&self, request: MCPRequest) -> Result<MCPResponse> {
        match request.tool.as_str() {
            "store_knowledge" => self.handle_store_knowledge(request.arguments).await,
            "neural_query" => self.handle_neural_query(request.arguments).await,
            "cognitive_reasoning" => self.handle_cognitive_reasoning_tool_call(request.arguments).await,
            _ => Err(GraphError::InvalidInput(format!("Unknown tool: {}", request.tool))),
        }
    }

    /// Store knowledge with neural structure prediction
    pub async fn handle_store_knowledge(&self, args: serde_json::Value) -> Result<MCPResponse> {
        let text = args["text"].as_str()
            .ok_or_else(|| GraphError::InvalidInput("Missing 'text' parameter".to_string()))?;
        let context = args["context"].as_str().map(|s| s.to_string());
        let use_neural = args["use_neural_construction"].as_bool().unwrap_or(true);

        self.handle_store_fact_neural(text, context, use_neural).await
    }

    /// Neural-powered graph construction implementation
    pub async fn handle_store_fact_neural(
        &self,
        text: &str,
        context: Option<String>,
        use_neural_construction: bool,
    ) -> Result<MCPResponse> {
        if use_neural_construction {
            // 1. Neural canonicalization of entities
            let canonical_entities = self.canonicalize_entities_neural(text).await?;
            
            // 2. Neural structure prediction
            let graph_operations = self.structure_predictor
                .predict_structure(text)
                .await?;
            
            // 3. Execute operations to create brain-inspired structure
            let created_entities = self.execute_graph_operations(
                graph_operations,
                canonical_entities,
            ).await?;
            
            // 4. Set up temporal metadata
            let temporal_metadata = self.create_temporal_metadata(
                text,
                context,
                created_entities.clone(),
            ).await?;
            
            // 5. Store with bi-temporal tracking
            let mut graph = self.knowledge_graph.write().await;
            let current_time = std::time::SystemTime::now();
            
            for entity in &created_entities {
                // Convert brain-inspired entity to temporal entity
                let temporal_entity = self.convert_to_temporal_entity(entity, &temporal_metadata, current_time).await?;
                
                // Store in temporal knowledge graph
                let time_range = crate::versioning::temporal_graph::TimeRange::new(
                    chrono::DateTime::from(current_time)
                );
                graph.insert_temporal_entity(temporal_entity.entity, time_range).await?;
            }
            
            Ok(MCPResponse {
                content: vec![MCPContent {
                    type_: "text".to_string(),
                    text: format!(
                        "Neural graph construction completed. Created {} entities with brain-inspired structure.",
                        created_entities.len()
                    ),
                }],
                is_error: false,
            })
        } else {
            // Fallback to traditional storage for compatibility
            self.handle_store_fact_traditional(text, context).await
        }
    }

    /// Neural query with cognitive pattern
    pub async fn handle_neural_query(&self, args: serde_json::Value) -> Result<MCPResponse> {
        let query = args["query"].as_str()
            .ok_or_else(|| GraphError::InvalidInput("Missing 'query' parameter".to_string()))?;
        let query_type = args["query_type"].as_str().unwrap_or("semantic");
        let top_k = args["top_k"].as_u64().unwrap_or(10) as usize;
        
        // Generate query embedding
        let query_embedding = self.neural_server
            .get_embedding(query)
            .await
            .map_err(|e| GraphError::ProcessingError(format!("Failed to generate query embedding: {}", e)))?;
        
        // Search the knowledge graph using embeddings
        let graph = self.knowledge_graph.read().await;
        let mut results = Vec::new();
        
        match query_type {
            "semantic" => {
                // Perform semantic similarity search
                // TODO: This is a simplified implementation - would use HNSW or similar index in production
                let entities = graph.get_all_entities().await;
                
                let mut scored_entities: Vec<(String, f32, AHashMap<String, String>)> = Vec::new();
                
                for entity in entities {
                    if !entity.embedding.is_empty() {
                        // Calculate cosine similarity
                        let similarity = self.calculate_cosine_similarity(&query_embedding, &entity.embedding);
                        if similarity > 0.5 { // Threshold for relevance
                            // Convert properties to string map
                            let attributes: AHashMap<String, String> = entity.properties.iter()
                                .map(|(k, v)| (k.clone(), match v {
                                    crate::core::types::AttributeValue::String(s) => s.clone(),
                                    crate::core::types::AttributeValue::Number(n) => n.to_string(),
                                    _ => format!("{:?}", v),
                                }))
                                .collect();
                            scored_entities.push((entity.concept_id.clone(), similarity, attributes));
                        }
                    }
                }
                
                // Sort by similarity score
                scored_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                
                // Take top K results
                for (entity_id, score, attributes) in scored_entities.into_iter().take(top_k) {
                    results.push(format!(
                        "Entity: {} (similarity: {:.3})\nAttributes: {:?}",
                        entity_id, score, attributes
                    ));
                }
            },
            "exact" => {
                // Perform exact match search
                let entities = graph.search_entities(query).await;
                for entity in entities.into_iter().take(top_k) {
                    results.push(format!(
                        "Entity: {}\nConcept: {}\nActivation: {:.3}",
                        entity.concept_id, entity.concept_id, entity.activation_state
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
                "Neural Query Results for '{}' (type: {}):\n\n{}",
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
    
    /// Calculate cosine similarity between two embeddings
    fn calculate_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
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
            description: "Store knowledge with neural graph structure prediction".to_string(),
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
                    },
                    "use_neural_construction": {
                        "type": "boolean",
                        "description": "Use neural-powered graph construction (default: true)"
                    }
                },
                "required": ["text"]
            }),
        }
    }

    fn create_neural_query_tool(&self) -> MCPTool {
        MCPTool {
            name: "neural_query".to_string(),
            description: "Query knowledge graph using neural activation patterns".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to execute"
                    },
                    "query_type": {
                        "type": "string",
                        "description": "Type of query: semantic, exact, or pattern",
                        "enum": ["semantic", "exact", "pattern"],
                        "default": "semantic"
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
            description: "Execute cognitive reasoning using Phase 2 patterns".to_string(),
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

    /// Canonicalize entities using neural processing
    async fn canonicalize_entities_neural(&self, text: &str) -> Result<AHashMap<String, String>> {
        // Extract entities from text and canonicalize them
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut canonical_map = AHashMap::new();
        
        for word in words {
            let canonical = self.canonicalizer.canonicalize_entity(word).await?;
            canonical_map.insert(word.to_string(), canonical);
        }
        
        Ok(canonical_map)
    }

    /// Execute graph operations to create brain-inspired structure
    async fn execute_graph_operations(
        &self,
        operations: Vec<GraphOperation>,
        canonical_entities: AHashMap<String, String>,
    ) -> Result<Vec<BrainInspiredEntity>> {
        let mut created_entities = Vec::new();
        let mut logic_gates = Vec::new();
        
        for operation in operations {
            match operation {
                GraphOperation::CreateNode { concept, node_type } => {
                    let canonical_id = canonical_entities.get(&concept)
                        .unwrap_or(&concept)
                        .clone();
                    
                    let entity = BrainInspiredEntity {
                        id: EntityKey::default(),
                        concept_id: canonical_id.clone(),
                        direction: node_type,
                        properties: std::collections::HashMap::new(),
                        embedding: self.generate_concept_embedding(&canonical_id).await?,
                        activation_state: 0.0,
                        last_activation: std::time::SystemTime::now(),
                        last_update: std::time::SystemTime::now(),
                    };
                    
                    created_entities.push(entity);
                },
                GraphOperation::CreateLogicGate { inputs, outputs, gate_type } => {
                    let gate = LogicGate {
                        gate_id: EntityKey::default(),
                        gate_type,
                        input_nodes: self.resolve_entity_keys(&inputs, &created_entities)?,
                        output_nodes: self.resolve_entity_keys(&outputs, &created_entities)?,
                        threshold: 0.5, // Default threshold
                        weight_matrix: vec![1.0; inputs.len()], // Equal weights initially
                    };
                    
                    logic_gates.push(gate);
                },
                GraphOperation::CreateRelationship { source, target, relation_type, weight } => {
                    // Create brain-inspired relationship with temporal metadata
                    let source_key = self.find_entity_key(&source, &created_entities)?;
                    let target_key = self.find_entity_key(&target, &created_entities)?;
                    
                    let now = std::time::SystemTime::now();
                    let relationship = BrainInspiredRelationship {
                        source: source_key,
                        target: target_key,
                        source_key,
                        target_key,
                        relation_type,
                        weight,
                        strength: weight,
                        is_inhibitory: weight < 0.0,
                        temporal_decay: 0.9,
                        last_strengthened: now,
                        last_update: now,
                        activation_count: 0,
                        usage_count: 0,
                        creation_time: now,
                        ingestion_time: now,
                        metadata: std::collections::HashMap::new(),
                    };
                    
                    // Store relationship in graph
                    self.store_relationship(relationship).await?;
                },
            }
        }
        
        // Store logic gates in the graph
        self.store_logic_gates(logic_gates).await?;
        
        Ok(created_entities)
    }

    /// Generate embedding for a concept
    async fn generate_concept_embedding(&self, _concept: &str) -> Result<Vec<f32>> {
        // Generate a mock embedding for now
        // TODO: Implement actual neural embedding generation
        Ok(vec![0.1; 384])
    }

    /// Resolve entity keys from concept names
    fn resolve_entity_keys(
        &self,
        concepts: &[String],
        entities: &[BrainInspiredEntity],
    ) -> Result<Vec<EntityKey>> {
        let mut keys = Vec::new();
        for concept in concepts {
            let key = self.find_entity_key(concept, entities)?;
            keys.push(key);
        }
        Ok(keys)
    }

    /// Find entity key by concept name
    fn find_entity_key(
        &self,
        concept: &str,
        entities: &[BrainInspiredEntity],
    ) -> Result<EntityKey> {
        entities.iter()
            .find(|e| e.concept_id == concept)
            .map(|e| e.id)
            .ok_or_else(|| GraphError::BrainEntityNotFound(concept.to_string()))
    }

    /// Store logic gates (placeholder - would integrate with graph storage)
    async fn store_logic_gates(&self, _gates: Vec<LogicGate>) -> Result<()> {
        // TODO: Implement actual storage logic
        Ok(())
    }

    /// Store relationship (placeholder - would integrate with graph storage)
    async fn store_relationship(&self, _relationship: BrainInspiredRelationship) -> Result<()> {
        // TODO: Implement actual storage logic
        Ok(())
    }

    /// Create temporal metadata for entities
    async fn create_temporal_metadata(
        &self,
        text: &str,
        context: Option<String>,
        entities: Vec<BrainInspiredEntity>,
    ) -> Result<AHashMap<String, AttributeValue>> {
        let mut metadata = AHashMap::new();
        metadata.insert("source_text".to_string(), AttributeValue::String(text.to_string()));
        if let Some(ctx) = context {
            metadata.insert("context".to_string(), AttributeValue::String(ctx));
        }
        metadata.insert("entity_count".to_string(), AttributeValue::Number(entities.len() as f64));
        metadata.insert("creation_time".to_string(), AttributeValue::String(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string()
        ));
        Ok(metadata)
    }

    /// Convert brain-inspired entity to temporal entity
    async fn convert_to_temporal_entity(
        &self,
        brain_entity: &BrainInspiredEntity,
        temporal_metadata: &AHashMap<String, AttributeValue>,
        _valid_time: std::time::SystemTime,
    ) -> Result<crate::versioning::temporal_graph::TemporalEntity> {
        use crate::versioning::temporal_graph::TemporalEntity;
        
        // Create an updated brain entity with metadata
        let mut updated_brain_entity = brain_entity.clone();
        
        // Add temporal metadata to properties
        for (key, value) in temporal_metadata {
            match value {
                AttributeValue::String(s) => {
                    updated_brain_entity.properties.insert(key.clone(), crate::core::types::AttributeValue::String(s.clone()));
                }
                AttributeValue::Number(n) => {
                    updated_brain_entity.properties.insert(key.clone(), crate::core::types::AttributeValue::Number(*n));
                }
                _ => {
                    updated_brain_entity.properties.insert(key.clone(), crate::core::types::AttributeValue::String(format!("{:?}", value)));
                }
            }
        }
        
        let temporal_entity = TemporalEntity {
            entity: updated_brain_entity,
            valid_time: TimeRange::new(chrono::Utc::now()),
            transaction_time: TimeRange::new(chrono::Utc::now()),
            version_id: 1,
            supersedes: None,
        };
        
        Ok(temporal_entity)
    }

    /// Traditional fact storage for compatibility
    async fn handle_store_fact_traditional(
        &self,
        text: &str,
        context: Option<String>,
    ) -> Result<MCPResponse> {
        // Simple traditional storage without neural processing
        let mut graph = self.knowledge_graph.write().await;
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
        
        let temporal_entity = crate::versioning::temporal_graph::TemporalEntity {
            entity: brain_entity,
            valid_time: TimeRange::new(chrono::Utc::now()),
            transaction_time: TimeRange::new(chrono::Utc::now()),
            version_id: 1,
            supersedes: None,
        };
        
        let time_range = crate::versioning::temporal_graph::TimeRange::new(chrono::Utc::now());
        graph.insert_temporal_entity(temporal_entity.entity, time_range).await?;
        
        Ok(MCPResponse {
            content: vec![MCPContent {
                type_: "text".to_string(),
                text: format!("Stored fact: {}", text),
            }],
            is_error: false,
        })
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
        let neural_server = Arc::new(
            NeuralProcessingServer::new("localhost:9000".to_string()).await.unwrap()
        );
        
        let mcp_server = BrainInspiredMCPServer::new(temporal_graph, neural_server);
        let tools = mcp_server.get_tools();
        
        assert!(!tools.is_empty());
        assert!(tools.iter().any(|t| t.name == "store_knowledge"));
        assert!(tools.iter().any(|t| t.name == "neural_query"));
    }
}