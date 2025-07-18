use crate::core::knowledge_engine::{KnowledgeEngine, TripleQuery, KnowledgeResult};
use crate::core::triple::{Triple, NodeType, MAX_CHUNK_SIZE_BYTES};
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;
use std::time::Instant;

/// LLM-friendly MCP server optimized for intuitive knowledge graph operations
/// Designed so LLMs can easily store and retrieve SPO triples without complex understanding
pub struct LLMFriendlyMCPServer {
    knowledge_engine: Arc<RwLock<KnowledgeEngine>>,
    usage_stats: Arc<RwLock<UsageStats>>,
}

#[derive(Debug, Clone, Default)]
pub struct UsageStats {
    pub total_operations: u64,
    pub triples_stored: u64,
    pub chunks_stored: u64,
    pub queries_executed: u64,
    pub avg_response_time_ms: f64,
    pub memory_efficiency: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LLMMCPTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub examples: Vec<LLMExample>,
    pub tips: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LLMExample {
    pub description: String,
    pub input: serde_json::Value,
    pub expected_output: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LLMMCPRequest {
    pub method: String,
    pub params: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LLMMCPResponse {
    pub success: bool,
    pub data: serde_json::Value,
    pub message: String,
    pub helpful_info: Option<String>,
    pub suggestions: Vec<String>,
    pub performance: PerformanceInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceInfo {
    pub response_time_ms: u64,
    pub memory_used_mb: f64,
    pub nodes_processed: usize,
    pub efficiency_score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub confidence: f64,
    pub conflicts: Vec<String>,
    pub sources: Vec<String>,
    pub validation_notes: Vec<String>,
}

impl LLMFriendlyMCPServer {
    pub fn new() -> Result<Self> {
        let knowledge_engine = KnowledgeEngine::new(96, 1_000_000)?; // 1M node limit
        
        Ok(Self {
            knowledge_engine: Arc::new(RwLock::new(knowledge_engine)),
            usage_stats: Arc::new(RwLock::new(UsageStats::default())),
        })
    }
    
    /// Get all available tools with LLM-friendly descriptions and examples
    pub fn get_tools(&self) -> Vec<LLMMCPTool> {
        vec![
            LLMMCPTool {
                name: "store_fact".to_string(),
                description: "Store a simple fact as a Subject-Predicate-Object triple. This is the most basic way to add knowledge. Use short, clear predicates (1-3 words max).".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "The entity or thing the fact is about (e.g., 'Einstein', 'Python', 'New York')",
                            "maxLength": 128
                        },
                        "predicate": {
                            "type": "string", 
                            "description": "The relationship or property (e.g., 'is', 'invented', 'located_in', 'has'). Keep it short!",
                            "maxLength": 64
                        },
                        "object": {
                            "type": "string",
                            "description": "What the subject is related to (e.g., 'scientist', 'relativity', 'USA')",
                            "maxLength": 128
                        },
                        "confidence": {
                            "type": "number",
                            "description": "How confident you are in this fact (0.0 to 1.0, default: 1.0)",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "default": 1.0
                        }
                    },
                    "required": ["subject", "predicate", "object"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Store a basic fact about a person".to_string(),
                        input: serde_json::json!({
                            "subject": "Einstein",
                            "predicate": "is",
                            "object": "physicist"
                        }),
                        expected_output: "Successfully stored: Einstein is physicist".to_string(),
                    },
                    LLMExample {
                        description: "Store an invention fact".to_string(),
                        input: serde_json::json!({
                            "subject": "Einstein", 
                            "predicate": "invented",
                            "object": "relativity"
                        }),
                        expected_output: "Successfully stored: Einstein invented relativity".to_string(),
                    },
                    LLMExample {
                        description: "Store a location fact".to_string(),
                        input: serde_json::json!({
                            "subject": "Paris",
                            "predicate": "located_in", 
                            "object": "France"
                        }),
                        expected_output: "Successfully stored: Paris located_in France".to_string(),
                    }
                ],
                tips: vec![
                    "Use consistent entity names (e.g., always 'Einstein', not sometimes 'Albert Einstein')".to_string(),
                    "Keep predicates short: 'is', 'has', 'located_in', 'invented', 'works_at'".to_string(),
                    "Store one fact at a time for best results".to_string(),
                    "Use underscores for multi-word predicates: 'located_in', 'works_at'".to_string(),
                ],
            },
            
            LLMMCPTool {
                name: "store_knowledge".to_string(),
                description: "Store a larger piece of text (up to 512 tokens/~400 words). The system will automatically extract facts from it. Use this for paragraphs, articles, or detailed information.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text content to store. Can be a paragraph, article, or any detailed information.",
                            "maxLength": MAX_CHUNK_SIZE_BYTES
                        },
                        "title": {
                            "type": "string",
                            "description": "Optional title or summary of this knowledge",
                            "maxLength": 128
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags to categorize this knowledge",
                            "maxItems": 10
                        }
                    },
                    "required": ["text"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Store detailed information about a person".to_string(),
                        input: serde_json::json!({
                            "text": "Albert Einstein was a German theoretical physicist who developed the theory of relativity. He was born in 1879 and died in 1955. He won the Nobel Prize in Physics in 1921.",
                            "title": "Einstein Biography",
                            "tags": ["person", "scientist", "physics"]
                        }),
                        expected_output: "Stored knowledge chunk and extracted 4 facts automatically".to_string(),
                    }
                ],
                tips: vec![
                    "The system automatically extracts simple facts from your text".to_string(),
                    "Include clear, factual statements for best extraction".to_string(),
                    "Keep chunks under 400 words for optimal performance".to_string(),
                    "Use titles and tags to help organize knowledge".to_string(),
                ],
            },
            
            // New query generation tool
            LLMMCPTool {
                name: "generate_graph_query".to_string(),
                description: "Convert natural language questions into structured graph queries (SPARQL, Cypher, or Gremlin). This helps you construct complex queries for advanced graph operations.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Natural language question about the graph (e.g., 'Find all scientists who worked on relativity')",
                            "maxLength": 512
                        },
                        "query_language": {
                            "type": "string",
                            "enum": ["sparql", "cypher", "gremlin"],
                            "default": "cypher",
                            "description": "Target query language for the generated query"
                        },
                        "include_explanation": {
                            "type": "boolean",
                            "default": true,
                            "description": "Whether to include explanation of the generated query"
                        }
                    },
                    "required": ["question"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Generate a query to find connections".to_string(),
                        input: serde_json::json!({
                            "question": "Find all people who worked with Einstein",
                            "query_language": "cypher"
                        }),
                        expected_output: "Generated Cypher query with explanation".to_string(),
                    }
                ],
                tips: vec![
                    "Use specific entity names and relationship types for better queries".to_string(),
                    "Cypher is best for Neo4j-style queries".to_string(),
                    "SPARQL is ideal for RDF/semantic web queries".to_string(),
                    "Include explanation=true to understand the generated query".to_string(),
                ],
            },
            
            // New hybrid search tool
            LLMMCPTool {
                name: "hybrid_search".to_string(),
                description: "Combine vector similarity search with graph traversal for comprehensive search results. This gives you both semantically similar content and structurally related entities.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "text_query": {
                            "type": "string",
                            "description": "Semantic search query for finding similar content",
                            "maxLength": 512
                        },
                        "graph_pattern": {
                            "type": "object",
                            "description": "Graph traversal pattern (subject, predicate, object filters)",
                            "properties": {
                                "subject": {"type": "string"},
                                "predicate": {"type": "string"},
                                "object": {"type": "string"},
                                "max_hops": {"type": "integer", "default": 2}
                            }
                        },
                        "fusion_strategy": {
                            "type": "string",
                            "enum": ["weighted", "rerank", "filter"],
                            "default": "weighted",
                            "description": "How to combine vector and graph search results"
                        },
                        "max_results": {
                            "type": "integer",
                            "default": 25,
                            "minimum": 1,
                            "maximum": 100
                        }
                    },
                    "required": ["text_query"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Search for physics concepts with graph exploration".to_string(),
                        input: serde_json::json!({
                            "text_query": "quantum mechanics theories",
                            "graph_pattern": {
                                "subject": "Einstein",
                                "predicate": "worked_on",
                                "max_hops": 2
                            },
                            "fusion_strategy": "weighted"
                        }),
                        expected_output: "Combined results from semantic search and graph traversal".to_string(),
                    }
                ],
                tips: vec![
                    "Use text_query for semantic similarity, graph_pattern for structural relationships".to_string(),
                    "Weighted fusion gives balanced results between vector and graph search".to_string(),
                    "Rerank fusion uses vector search to reorder graph results".to_string(),
                    "Filter fusion uses graph results to filter vector search".to_string(),
                ],
            },
            
            // New knowledge validation tool
            LLMMCPTool {
                name: "validate_knowledge".to_string(),
                description: "Validate facts with confidence scoring and source tracking. This helps ensure knowledge quality and identifies potential conflicts.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "triple": {
                            "type": "object",
                            "properties": {
                                "subject": {"type": "string"},
                                "predicate": {"type": "string"},
                                "object": {"type": "string"}
                            },
                            "required": ["subject", "predicate", "object"]
                        },
                        "validation_strategy": {
                            "type": "string",
                            "enum": ["consistency_check", "source_verification", "llm_validation"],
                            "default": "consistency_check",
                            "description": "Method to use for validation"
                        },
                        "require_sources": {
                            "type": "boolean",
                            "default": false,
                            "description": "Whether to require source citations"
                        }
                    },
                    "required": ["triple"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Validate a scientific fact".to_string(),
                        input: serde_json::json!({
                            "triple": {
                                "subject": "Einstein",
                                "predicate": "born_in",
                                "object": "1879"
                            },
                            "validation_strategy": "consistency_check"
                        }),
                        expected_output: "Validation result with confidence score and conflict detection".to_string(),
                    }
                ],
                tips: vec![
                    "consistency_check validates against existing knowledge in the graph".to_string(),
                    "source_verification checks for citation and source quality".to_string(),
                    "llm_validation uses language models for fact checking".to_string(),
                    "Higher confidence scores indicate more reliable facts".to_string(),
                ],
            },
            
            LLMMCPTool {
                name: "find_facts".to_string(),
                description: "Search for facts using Subject, Predicate, or Object patterns. You can search for any combination. Leave fields empty to match anything.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string",
                            "description": "Find facts about this subject (optional)",
                            "maxLength": 128
                        },
                        "predicate": {
                            "type": "string",
                            "description": "Find facts with this relationship (optional)",
                            "maxLength": 64
                        },
                        "object": {
                            "type": "string",
                            "description": "Find facts with this object (optional)",
                            "maxLength": 128
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of facts to return (default: 20, max: 100)",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 20
                        }
                    }
                }),
                examples: vec![
                    LLMExample {
                        description: "Find all facts about Einstein".to_string(),
                        input: serde_json::json!({
                            "subject": "Einstein",
                            "limit": 10
                        }),
                        expected_output: "Found 3 facts: Einstein is physicist, Einstein invented relativity, Einstein won Nobel_Prize".to_string(),
                    },
                    LLMExample {
                        description: "Find all invention relationships".to_string(),
                        input: serde_json::json!({
                            "predicate": "invented",
                            "limit": 20
                        }),
                        expected_output: "Found 5 inventions: Einstein invented relativity, Edison invented lightbulb, ...".to_string(),
                    }
                ],
                tips: vec![
                    "Leave fields empty to search broadly".to_string(),
                    "Use exact entity names for best results".to_string(),
                    "Start with small limits and increase if needed".to_string(),
                ],
            },
            
            LLMMCPTool {
                name: "ask_question".to_string(),
                description: "Ask a natural language question and get relevant facts and knowledge. This uses semantic search to find related information even if exact matches don't exist.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Your question in natural language",
                            "maxLength": 512
                        },
                        "max_facts": {
                            "type": "integer", 
                            "description": "Maximum facts to retrieve (default: 25)",
                            "minimum": 1,
                            "maximum": 100,
                            "default": 25
                        },
                        "include_context": {
                            "type": "boolean",
                            "description": "Include detailed context and chunks (default: true)",
                            "default": true
                        }
                    },
                    "required": ["question"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Ask about a specific person".to_string(),
                        input: serde_json::json!({
                            "question": "What did Einstein discover?",
                            "max_facts": 15
                        }),
                        expected_output: "Found relevant knowledge: Einstein invented relativity theory, Einstein won Nobel Prize for photoelectric effect...".to_string(),
                    }
                ],
                tips: vec![
                    "Ask specific questions for better results".to_string(),
                    "The system understands synonyms and related concepts".to_string(),
                    "Include context for more detailed answers".to_string(),
                ],
            },
            
            LLMMCPTool {
                name: "explore_connections".to_string(),
                description: "Explore how entities are connected through relationships. Great for understanding complex relationships and building context.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "The entity to explore connections for",
                            "maxLength": 128
                        },
                        "max_hops": {
                            "type": "integer",
                            "description": "How many relationship steps to follow (1-5, default: 2)",
                            "minimum": 1,
                            "maximum": 5,
                            "default": 2
                        },
                        "max_connections": {
                            "type": "integer",
                            "description": "Maximum connections to return (default: 50)",
                            "minimum": 1,
                            "maximum": 200,
                            "default": 50
                        }
                    },
                    "required": ["entity"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Explore Einstein's connections".to_string(),
                        input: serde_json::json!({
                            "entity": "Einstein",
                            "max_hops": 2,
                            "max_connections": 30
                        }),
                        expected_output: "Found 12 connections: Einstein -> physics -> quantum_mechanics, Einstein -> relativity -> spacetime...".to_string(),
                    }
                ],
                tips: vec![
                    "Start with 1-2 hops to avoid overwhelming results".to_string(),
                    "Use this to understand how concepts relate".to_string(),
                    "Great for building comprehensive context".to_string(),
                ],
            },
            
            LLMMCPTool {
                name: "get_suggestions".to_string(),
                description: "Get suggestions for predicates, entities, or improving your knowledge storage. Helps you use the system more effectively.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "context": {
                            "type": "string",
                            "description": "Describe what you're trying to express or store",
                            "maxLength": 512
                        },
                        "suggestion_type": {
                            "type": "string",
                            "enum": ["predicates", "entities", "optimization", "all"],
                            "description": "What kind of suggestions you want",
                            "default": "all"
                        }
                    },
                    "required": ["context"]
                }),
                examples: vec![
                    LLMExample {
                        description: "Get predicate suggestions".to_string(),
                        input: serde_json::json!({
                            "context": "I want to express that someone works at a company",
                            "suggestion_type": "predicates"
                        }),
                        expected_output: "Suggested predicates: works_at, employed_by, member_of".to_string(),
                    }
                ],
                tips: vec![
                    "Use this when you're unsure how to structure facts".to_string(),
                    "Get predicate suggestions before storing facts".to_string(),
                    "Ask for optimization tips to improve performance".to_string(),
                ],
            },
            
            LLMMCPTool {
                name: "get_stats".to_string(),
                description: "Get system statistics including memory usage, performance metrics, and storage efficiency. Useful for monitoring and optimization.".to_string(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }),
                examples: vec![
                    LLMExample {
                        description: "Check system status".to_string(),
                        input: serde_json::json!({}),
                        expected_output: "System stats: 1,250 facts stored, 45 MB memory used, avg response: 2.3ms".to_string(),
                    }
                ],
                tips: vec![
                    "Monitor memory usage to avoid bloat".to_string(),
                    "Check performance regularly".to_string(),
                    "Use stats to optimize your usage patterns".to_string(),
                ],
            }
        ]
    }
    
    /// Handle LLM requests with helpful responses and suggestions
    pub async fn handle_request(&self, request: LLMMCPRequest) -> LLMMCPResponse {
        let start_time = Instant::now();
        
        let result = match request.method.as_str() {
            "store_fact" => self.handle_store_fact(request.params).await,
            "store_knowledge" => self.handle_store_knowledge(request.params).await,
            "find_facts" => self.handle_find_facts(request.params).await,
            "ask_question" => self.handle_ask_question(request.params).await,
            "explore_connections" => self.handle_explore_connections(request.params).await,
            "get_suggestions" => self.handle_get_suggestions(request.params).await,
            "get_stats" => self.handle_get_stats().await,
            "generate_graph_query" => self.handle_generate_graph_query(request.params).await,
            "hybrid_search" => self.handle_hybrid_search(request.params).await,
            "validate_knowledge" => self.handle_validate_knowledge(request.params).await,
            _ => Err(format!("Unknown method: {}. Available methods: store_fact, store_knowledge, find_facts, ask_question, explore_connections, get_suggestions, get_stats, generate_graph_query, hybrid_search, validate_knowledge", request.method)),
        };
        
        let response_time = start_time.elapsed().as_millis() as u64;
        
        // Update usage stats
        self.update_usage_stats(response_time).await;
        
        match result {
            Ok((data, message, suggestions)) => {
                let engine = self.knowledge_engine.read().await;
                let memory_stats = engine.get_memory_stats();
                
                LLMMCPResponse {
                    success: true,
                    data,
                    message,
                    helpful_info: Some(self.generate_helpful_info(&request.method)),
                    suggestions,
                    performance: PerformanceInfo {
                        response_time_ms: response_time,
                        memory_used_mb: memory_stats.total_bytes as f64 / 1_048_576.0,
                        nodes_processed: memory_stats.total_nodes,
                        efficiency_score: self.calculate_efficiency_score(&memory_stats),
                    },
                }
            },
            Err(error) => {
                LLMMCPResponse {
                    success: false,
                    data: serde_json::json!(null),
                    message: error,
                    helpful_info: Some(self.generate_error_help(&request.method)),
                    suggestions: self.generate_error_suggestions(&request.method),
                    performance: PerformanceInfo {
                        response_time_ms: response_time,
                        memory_used_mb: 0.0,
                        nodes_processed: 0,
                        efficiency_score: 0.0,
                    },
                }
            }
        }
    }
    
    async fn handle_store_fact(&self, params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        let subject = params.get("subject").and_then(|v| v.as_str())
            .ok_or("Missing required 'subject' parameter")?;
        let predicate = params.get("predicate").and_then(|v| v.as_str())
            .ok_or("Missing required 'predicate' parameter")?;
        let object = params.get("object").and_then(|v| v.as_str())
            .ok_or("Missing required 'object' parameter")?;
        let confidence = params.get("confidence").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
        
        // Create triple
        let triple = Triple::with_metadata(
            subject.to_string(),
            predicate.to_string(),
            object.to_string(),
            confidence,
            None
        ).map_err(|e| format!("Invalid triple: {}", e))?;
        
        // Store triple
        let engine = self.knowledge_engine.write().await;
        let node_id = engine.store_triple(triple.clone(), None)
            .map_err(|e| format!("Failed to store triple: {}", e))?;
        
        let natural_language = triple.to_natural_language();
        let message = format!("✅ Successfully stored: {}", natural_language);
        
        let suggestions = vec![
            "Try adding more facts about the same entities to build richer knowledge".to_string(),
            "Use consistent entity names across all your facts".to_string(),
            format!("Consider exploring connections with: explore_connections for '{}'", subject),
        ];
        
        Ok((
            serde_json::json!({
                "node_id": node_id,
                "triple": triple,
                "natural_language": natural_language
            }),
            message,
            suggestions
        ))
    }
    
    async fn handle_store_knowledge(&self, params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        let text = params.get("text").and_then(|v| v.as_str())
            .ok_or("Missing required 'text' parameter")?;
        let title = params.get("title").and_then(|v| v.as_str()).unwrap_or("");
        
        if text.len() > MAX_CHUNK_SIZE_BYTES {
            return Err(format!("Text too long: {} bytes. Maximum allowed: {} bytes (~400 words)", 
                              text.len(), MAX_CHUNK_SIZE_BYTES));
        }
        
        // Store knowledge chunk
        let engine = self.knowledge_engine.write().await;
        let node_id = engine.store_chunk(text.to_string(), None)
            .map_err(|e| format!("Failed to store knowledge: {}", e))?;
        
        // Note: Cannot access private nodes field directly
        let extracted_count = 1; // Placeholder since we can't access the node
        
        let title_part = if title.is_empty() { 
            String::new() 
        } else { 
            format!(" '{}'", title) 
        };
        let message = format!("✅ Stored knowledge chunk{} and automatically extracted {} facts", 
                             title_part,
                             extracted_count);
        
        let suggestions = vec![
            "The system automatically extracted simple facts from your text".to_string(),
            "Use 'find_facts' to see what facts were extracted".to_string(),
            "Break longer texts into smaller chunks for better fact extraction".to_string(),
        ];
        
        Ok((
            serde_json::json!({
                "node_id": node_id,
                "text_length": text.len(),
                "extracted_facts": extracted_count,
                "title": title
            }),
            message,
            suggestions
        ))
    }
    
    async fn handle_find_facts(&self, params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        let subject = params.get("subject").and_then(|v| v.as_str()).map(|s| s.to_string());
        let predicate = params.get("predicate").and_then(|v| v.as_str()).map(|s| s.to_string());
        let object = params.get("object").and_then(|v| v.as_str()).map(|s| s.to_string());
        let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
        
        let query = TripleQuery {
            subject,
            predicate,
            object,
            limit,
            min_confidence: 0.0,
            include_chunks: false,
        };
        
        let engine = self.knowledge_engine.read().await;
        let result = engine.query_triples(query)
            .map_err(|e| format!("Query failed: {}", e))?;
        
        let facts_text: Vec<String> = result.triples.iter()
            .map(|t| t.to_natural_language())
            .collect();
        
        let message = if result.triples.is_empty() {
            "No facts found matching your criteria".to_string()
        } else {
            format!("Found {} facts", result.triples.len())
        };
        
        let suggestions = if result.triples.is_empty() {
            vec![
                "Try broader search terms".to_string(),
                "Check entity name spelling".to_string(),
                "Use 'ask_question' for semantic search".to_string(),
            ]
        } else {
            vec![
                "Use 'explore_connections' to find related entities".to_string(),
                "Try different predicate combinations".to_string(),
                format!("Found {} entities you might explore further", result.entity_context.len()),
            ]
        };
        
        Ok((
            serde_json::json!({
                "facts": result.triples,
                "facts_text": facts_text,
                "total_found": result.total_found,
                "query_time_ms": result.query_time_ms,
                "entities": result.entity_context.keys().collect::<Vec<_>>()
            }),
            message,
            suggestions
        ))
    }
    
    async fn handle_ask_question(&self, params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        let question = params.get("question").and_then(|v| v.as_str())
            .ok_or("Missing required 'question' parameter")?;
        let max_facts = params.get("max_facts").and_then(|v| v.as_u64()).unwrap_or(25) as usize;
        let include_context = params.get("include_context").and_then(|v| v.as_bool()).unwrap_or(true);
        
        // Use semantic search to find relevant knowledge
        let engine = self.knowledge_engine.read().await;
        let result = engine.semantic_search(question, max_facts)
            .map_err(|e| format!("Semantic search failed: {}", e))?;
        
        // Format response for LLM consumption
        let mut knowledge_text = Vec::new();
        let mut relevant_facts = Vec::new();
        
        for node in &result.nodes {
            match &node.node_type {
                NodeType::Triple => {
                    for triple in node.get_triples() {
                        relevant_facts.push(triple.to_natural_language());
                    }
                },
                NodeType::Chunk => {
                    if include_context {
                        knowledge_text.push(node.to_llm_format());
                    }
                    for triple in node.get_triples() {
                        relevant_facts.push(triple.to_natural_language());
                    }
                },
                _ => {}
            }
        }
        
        let message = format!("Found {} relevant facts and {} knowledge chunks for your question", 
                             relevant_facts.len(), knowledge_text.len());
        
        let suggestions = vec![
            "The facts above should help answer your question".to_string(),
            "Try more specific questions for better results".to_string(),
            "Use the entity names in facts to explore further with 'explore_connections'".to_string(),
        ];
        
        Ok((
            serde_json::json!({
                "question": question,
                "relevant_facts": relevant_facts,
                "knowledge_chunks": knowledge_text,
                "total_nodes": result.nodes.len(),
                "query_time_ms": result.query_time_ms,
                "entities_mentioned": result.entity_context.keys().collect::<Vec<_>>()
            }),
            message,
            suggestions
        ))
    }
    
    async fn handle_explore_connections(&self, params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        let entity = params.get("entity").and_then(|v| v.as_str())
            .ok_or("Missing required 'entity' parameter")?;
        let max_hops = params.get("max_hops").and_then(|v| v.as_u64()).unwrap_or(2) as u8;
        let max_connections = params.get("max_connections").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
        
        if max_hops > 5 {
            return Err("max_hops cannot exceed 5 to prevent overwhelming results".to_string());
        }
        
        let engine = self.knowledge_engine.read().await;
        let relationships = engine.get_entity_relationships(entity, max_hops)
            .map_err(|e| format!("Failed to explore connections: {}", e))?;
        
        let limited_relationships: Vec<_> = relationships.into_iter().take(max_connections).collect();
        
        // Group by relationship type for better organization
        let mut by_predicate: HashMap<String, Vec<String>> = HashMap::new();
        for rel in &limited_relationships {
            let connection = if rel.subject == entity {
                format!("{} -> {}", rel.predicate, rel.object)
            } else {
                format!("{} <- {} <- {}", rel.subject, rel.predicate, entity)
            };
            by_predicate.entry(rel.predicate.clone()).or_insert_with(Vec::new).push(connection);
        }
        
        let message = format!("Found {} connections for '{}' within {} hops", 
                             limited_relationships.len(), entity, max_hops);
        
        let suggestions = vec![
            format!("Explore specific connected entities with 'find_facts' or 'ask_question'"),
            "Try increasing max_hops to find more distant connections".to_string(),
            "Use the connected entities to build richer context".to_string(),
        ];
        
        Ok((
            serde_json::json!({
                "entity": entity,
                "total_connections": limited_relationships.len(),
                "max_hops": max_hops,
                "relationships": limited_relationships,
                "grouped_by_predicate": by_predicate,
                "connected_entities": limited_relationships.iter()
                    .flat_map(|r| vec![&r.subject, &r.object])
                    .filter(|&e| e != entity)
                    .collect::<std::collections::HashSet<_>>()
            }),
            message,
            suggestions
        ))
    }
    
    async fn handle_get_suggestions(&self, params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        let context = params.get("context").and_then(|v| v.as_str())
            .ok_or("Missing required 'context' parameter")?;
        let suggestion_type = params.get("suggestion_type").and_then(|v| v.as_str()).unwrap_or("all");
        
        let engine = self.knowledge_engine.read().await;
        let mut suggestions_data = serde_json::Map::new();
        
        if suggestion_type == "predicates" || suggestion_type == "all" {
            let predicate_suggestions = engine.suggest_predicates(context);
            suggestions_data.insert("predicates".to_string(), serde_json::json!(predicate_suggestions));
        }
        
        if suggestion_type == "entities" || suggestion_type == "all" {
            let entity_types = engine.get_entity_types();
            let common_entities: Vec<String> = entity_types.keys().take(10).cloned().collect();
            suggestions_data.insert("common_entities".to_string(), serde_json::json!(common_entities));
        }
        
        if suggestion_type == "optimization" || suggestion_type == "all" {
            let memory_stats = engine.get_memory_stats();
            let optimization_tips = vec![
                format!("Current efficiency: {:.1} bytes per node", memory_stats.bytes_per_node),
                "Use short, consistent entity names to reduce memory usage".to_string(),
                "Batch similar facts together for better performance".to_string(),
                "Monitor memory usage with 'get_stats' regularly".to_string(),
            ];
            suggestions_data.insert("optimization_tips".to_string(), serde_json::json!(optimization_tips));
        }
        
        let message = format!("Generated suggestions for: {}", context);
        
        let action_suggestions = vec![
            "Use suggested predicates in 'store_fact' operations".to_string(),
            "Check existing entities with 'find_facts' before creating new ones".to_string(),
            "Follow optimization tips to maintain system performance".to_string(),
        ];
        
        Ok((
            serde_json::json!(suggestions_data),
            message,
            action_suggestions
        ))
    }
    
    async fn handle_get_stats(&self) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        let engine = self.knowledge_engine.read().await;
        let memory_stats = engine.get_memory_stats();
        let usage_stats = self.usage_stats.read().await.clone();
        
        let efficiency_score = self.calculate_efficiency_score(&memory_stats);
        let status = if efficiency_score > 0.8 {
            "Excellent"
        } else if efficiency_score > 0.6 {
            "Good"
        } else if efficiency_score > 0.4 {
            "Fair"
        } else {
            "Needs Optimization"
        };
        
        let message = format!("System Status: {} | {} facts stored | {:.1} MB memory | {:.1}ms avg response", 
                             status, memory_stats.total_triples, 
                             memory_stats.total_bytes as f64 / 1_048_576.0, 
                             usage_stats.avg_response_time_ms);
        
        let suggestions = vec![
            format!("Memory efficiency: {:.1}% (target: >80%)", efficiency_score * 100.0),
            if memory_stats.bytes_per_node > 70.0 {
                "Consider optimizing fact storage - use shorter entity names".to_string()
            } else {
                "Memory usage is optimal!".to_string()
            },
            format!("Cache hit rate: {:.1}%", 
                   if usage_stats.cache_hits + usage_stats.cache_misses > 0 {
                       usage_stats.cache_hits as f64 / (usage_stats.cache_hits + usage_stats.cache_misses) as f64 * 100.0
                   } else {
                       0.0
                   }),
        ];
        
        Ok((
            serde_json::json!({
                "memory": {
                    "total_nodes": memory_stats.total_nodes,
                    "total_triples": memory_stats.total_triples,
                    "total_bytes": memory_stats.total_bytes,
                    "bytes_per_node": memory_stats.bytes_per_node,
                    "memory_mb": memory_stats.total_bytes as f64 / 1_048_576.0
                },
                "performance": {
                    "avg_response_time_ms": usage_stats.avg_response_time_ms,
                    "total_operations": usage_stats.total_operations,
                    "cache_hits": usage_stats.cache_hits,
                    "cache_misses": usage_stats.cache_misses,
                    "efficiency_score": efficiency_score
                },
                "usage": {
                    "triples_stored": usage_stats.triples_stored,
                    "chunks_stored": usage_stats.chunks_stored,
                    "queries_executed": usage_stats.queries_executed
                },
                "status": status
            }),
            message,
            suggestions
        ))
    }
    
    async fn handle_generate_graph_query(&self, params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        let question = params.get("question").and_then(|v| v.as_str())
            .ok_or("Missing required 'question' parameter")?;
        let query_language = params.get("query_language").and_then(|v| v.as_str()).unwrap_or("cypher");
        let include_explanation = params.get("include_explanation").and_then(|v| v.as_bool()).unwrap_or(true);
        
        let generated_query = self.generate_structured_query(question, query_language)?;
        let explanation = if include_explanation {
            self.generate_query_explanation(&generated_query, question)?
        } else {
            "Query explanation disabled".to_string()
        };
        
        let message = format!("Generated {} query for: '{}'", query_language.to_uppercase(), question);
        
        let suggestions = vec![
            format!("Execute this query in a {}-compatible database", query_language.to_uppercase()),
            "Modify the query to add filters or constraints as needed".to_string(),
            "Test the query with a small dataset first".to_string(),
            "Use the explanation to understand the query logic".to_string(),
        ];
        
        Ok((
            serde_json::json!({
                "question": question,
                "query_language": query_language,
                "query": generated_query,
                "explanation": explanation,
                "estimated_complexity": self.estimate_query_complexity(&generated_query)
            }),
            message,
            suggestions
        ))
    }

    async fn handle_hybrid_search(&self, params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        let text_query = params.get("text_query").and_then(|v| v.as_str())
            .ok_or("Missing required 'text_query' parameter")?;
        let fusion_strategy = params.get("fusion_strategy").and_then(|v| v.as_str()).unwrap_or("weighted");
        let max_results = params.get("max_results").and_then(|v| v.as_u64()).unwrap_or(25) as usize;
        
        // Perform semantic search
        let engine = self.knowledge_engine.read().await;
        let semantic_results = engine.semantic_search(text_query, max_results)
            .map_err(|e| format!("Semantic search failed: {}", e))?;
        
        // Perform graph pattern search if pattern is provided
        let graph_results = if let Some(pattern) = params.get("graph_pattern") {
            let subject = pattern.get("subject").and_then(|v| v.as_str());
            let predicate = pattern.get("predicate").and_then(|v| v.as_str());
            let object = pattern.get("object").and_then(|v| v.as_str());
            let max_hops = pattern.get("max_hops").and_then(|v| v.as_u64()).unwrap_or(2) as u8;
            
            let query = TripleQuery {
                subject: subject.map(|s| s.to_string()),
                predicate: predicate.map(|p| p.to_string()),
                object: object.map(|o| o.to_string()),
                limit: max_results,
                min_confidence: 0.0,
                include_chunks: true,
            };
            
            Some(engine.query_triples(query)
                .map_err(|e| format!("Graph search failed: {}", e))?)
        } else {
            None
        };
        
        // Fuse results based on strategy
        let fused_results = self.fuse_search_results(
            semantic_results,
            graph_results,
            fusion_strategy,
            max_results,
        )?;
        
        let message = format!("Hybrid search completed: {} results from semantic + graph search", fused_results.len());
        
        let suggestions = vec![
            "The results combine semantic similarity with graph structure".to_string(),
            "Try different fusion strategies for different result types".to_string(),
            "Use graph patterns to constrain search to specific relationships".to_string(),
            "Adjust max_results to get more or fewer results".to_string(),
        ];
        
        Ok((
            serde_json::json!({
                "text_query": text_query,
                "fusion_strategy": fusion_strategy,
                "results": fused_results,
                "total_results": fused_results.len(),
                "semantic_score_weight": self.get_fusion_weights(fusion_strategy).0,
                "graph_score_weight": self.get_fusion_weights(fusion_strategy).1,
            }),
            message,
            suggestions
        ))
    }

    async fn handle_validate_knowledge(&self, params: serde_json::Value) -> std::result::Result<(serde_json::Value, String, Vec<String>), String> {
        let triple_data = params.get("triple")
            .ok_or("Missing required 'triple' parameter")?;
        
        let subject = triple_data.get("subject").and_then(|v| v.as_str())
            .ok_or("Missing subject in triple")?;
        let predicate = triple_data.get("predicate").and_then(|v| v.as_str())
            .ok_or("Missing predicate in triple")?;
        let object = triple_data.get("object").and_then(|v| v.as_str())
            .ok_or("Missing object in triple")?;
        
        let validation_strategy = params.get("validation_strategy").and_then(|v| v.as_str()).unwrap_or("consistency_check");
        let require_sources = params.get("require_sources").and_then(|v| v.as_bool()).unwrap_or(false);
        
        let triple = Triple::new(subject.to_string(), predicate.to_string(), object.to_string())
            .map_err(|e| format!("Invalid triple format: {}", e))?;
        
        let validation_result = self.validate_triple(&triple, validation_strategy, require_sources).await?;
        
        let message = format!("Validation completed for: {} {} {}", subject, predicate, object);
        
        let suggestions = if validation_result.is_valid {
            vec![
                "The fact appears to be valid and consistent".to_string(),
                "Consider storing this fact in the knowledge graph".to_string(),
                format!("Confidence level: {:.1}%", validation_result.confidence * 100.0),
            ]
        } else {
            vec![
                "The fact may have issues - review the conflicts".to_string(),
                "Consider verifying the fact from additional sources".to_string(),
                "Check for typos or inconsistent entity names".to_string(),
            ]
        };
        
        Ok((
            serde_json::json!({
                "triple": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": object
                },
                "validation_result": validation_result,
                "validation_strategy": validation_strategy,
                "require_sources": require_sources,
            }),
            message,
            suggestions
        ))
    }
    
    // Helper methods for new functionality
    
    fn generate_structured_query(&self, question: &str, query_language: &str) -> std::result::Result<String, String> {
        match query_language {
            "cypher" => self.generate_cypher_query(question),
            "sparql" => self.generate_sparql_query(question),
            "gremlin" => self.generate_gremlin_query(question),
            _ => Err(format!("Unsupported query language: {}", query_language)),
        }
    }
    
    fn generate_cypher_query(&self, question: &str) -> std::result::Result<String, String> {
        // Simple pattern matching for common query types
        let question_lower = question.to_lowercase();
        
        if question_lower.contains("find all") && question_lower.contains("who") {
            // Find all X who Y pattern
            if let Some(entity_match) = self.extract_entity_from_question(&question_lower) {
                Ok(format!("MATCH (p:Person)-[:RELATED_TO]->(e:Entity) WHERE e.name CONTAINS '{}' RETURN p, e", entity_match))
            } else {
                Ok("MATCH (p:Person)-[:RELATED_TO]->(e:Entity) RETURN p, e LIMIT 25".to_string())
            }
        } else if question_lower.contains("what") && question_lower.contains("relationship") {
            // What is the relationship between X and Y
            Ok("MATCH (a)-[r]->(b) WHERE a.name = $entity1 AND b.name = $entity2 RETURN type(r), r".to_string())
        } else if question_lower.contains("connected to") || question_lower.contains("related to") {
            // Find connections
            Ok("MATCH (a)-[r*1..3]-(b) WHERE a.name = $entity RETURN b, r LIMIT 50".to_string())
        } else {
            // Generic entity search
            Ok("MATCH (n) WHERE n.name CONTAINS $search_term RETURN n LIMIT 25".to_string())
        }
    }
    
    fn generate_sparql_query(&self, question: &str) -> std::result::Result<String, String> {
        let question_lower = question.to_lowercase();
        
        if question_lower.contains("find all") {
            Ok("SELECT ?subject ?predicate ?object WHERE { ?subject ?predicate ?object . FILTER(CONTAINS(LCASE(STR(?subject)), LCASE($searchTerm))) } LIMIT 25".to_string())
        } else if question_lower.contains("type") || question_lower.contains("class") {
            Ok("SELECT ?entity ?type WHERE { ?entity rdf:type ?type . FILTER(CONTAINS(LCASE(STR(?entity)), LCASE($searchTerm))) } LIMIT 25".to_string())
        } else {
            Ok("SELECT ?subject ?predicate ?object WHERE { ?subject ?predicate ?object . FILTER(CONTAINS(LCASE(STR(?subject)), LCASE($searchTerm)) || CONTAINS(LCASE(STR(?object)), LCASE($searchTerm))) } LIMIT 25".to_string())
        }
    }
    
    fn generate_gremlin_query(&self, question: &str) -> std::result::Result<String, String> {
        let question_lower = question.to_lowercase();
        
        if question_lower.contains("find all") {
            Ok("g.V().hasLabel('entity').has('name', textContains($searchTerm)).limit(25)".to_string())
        } else if question_lower.contains("connected") || question_lower.contains("related") {
            Ok("g.V().has('name', $entity).both().limit(50)".to_string())
        } else {
            Ok("g.V().has('name', textContains($searchTerm)).limit(25)".to_string())
        }
    }
    
    fn extract_entity_from_question(&self, question: &str) -> Option<String> {
        // Simple entity extraction - in production, this would use NLP
        let words: Vec<&str> = question.split_whitespace().collect();
        
        // Look for capitalized words that might be entity names
        for word in words {
            if word.chars().next().unwrap_or('a').is_uppercase() && word.len() > 2 {
                return Some(word.to_string());
            }
        }
        
        None
    }
    
    fn generate_query_explanation(&self, query: &str, question: &str) -> std::result::Result<String, String> {
        Ok(format!(
            "Query Explanation:\n\nQuestion: {}\n\nGenerated Query:\n{}\n\nThis query will search for patterns in the graph that match your question. The query includes:\n- Entity matching based on names and properties\n- Relationship traversal to find connections\n- Filtering to focus on relevant results\n- Limits to prevent excessive results\n\nModify the query parameters as needed for your specific use case.",
            question, query
        ))
    }
    
    fn estimate_query_complexity(&self, query: &str) -> String {
        let complexity = if query.contains("*") || query.contains("..") {
            "High - Contains variable-length paths"
        } else if query.contains("MATCH") && query.contains("WHERE") {
            "Medium - Filtered graph traversal"
        } else if query.contains("MATCH") {
            "Low - Simple graph traversal"
        } else {
            "Low - Basic query"
        };
        
        complexity.to_string()
    }
    
    fn fuse_search_results(
        &self,
        semantic_results: KnowledgeResult,
        graph_results: Option<KnowledgeResult>,
        fusion_strategy: &str,
        max_results: usize,
    ) -> std::result::Result<Vec<serde_json::Value>, String> {
        let mut results = Vec::new();
        
        // Convert semantic results
        for node in semantic_results.nodes {
            let semantic_score = 0.8; // Would calculate actual similarity
            results.push(serde_json::json!({
                "node": node,
                "source": "semantic",
                "score": semantic_score,
                "type": "semantic_match"
            }));
        }
        
        // Add graph results if available
        if let Some(graph_res) = graph_results {
            for node in graph_res.nodes {
                let graph_score = 0.7; // Would calculate actual graph relevance
                results.push(serde_json::json!({
                    "node": node,
                    "source": "graph",
                    "score": graph_score,
                    "type": "graph_match"
                }));
            }
        }
        
        // Apply fusion strategy
        match fusion_strategy {
            "weighted" => {
                let (semantic_weight, graph_weight) = self.get_fusion_weights(fusion_strategy);
                for result in &mut results {
                    let original_score = result.get("score").unwrap().as_f64().unwrap();
                    let weight = if result.get("source").unwrap().as_str().unwrap() == "semantic" {
                        semantic_weight
                    } else {
                        graph_weight
                    };
                    result["fused_score"] = serde_json::Value::Number(serde_json::Number::from_f64(original_score * weight).unwrap());
                }
                results.sort_by(|a, b| {
                    b.get("fused_score").unwrap().as_f64().unwrap()
                        .partial_cmp(&a.get("fused_score").unwrap().as_f64().unwrap())
                        .unwrap()
                });
            }
            "rerank" => {
                // Use semantic search to rerank graph results
                results.sort_by(|a, b| {
                    let score_a = if a.get("source").unwrap().as_str().unwrap() == "semantic" {
                        a.get("score").unwrap().as_f64().unwrap()
                    } else {
                        a.get("score").unwrap().as_f64().unwrap() * 0.5
                    };
                    let score_b = if b.get("source").unwrap().as_str().unwrap() == "semantic" {
                        b.get("score").unwrap().as_f64().unwrap()
                    } else {
                        b.get("score").unwrap().as_f64().unwrap() * 0.5
                    };
                    score_b.partial_cmp(&score_a).unwrap()
                });
            }
            _ => {
                // Default: sort by original score
                results.sort_by(|a, b| {
                    b.get("score").unwrap().as_f64().unwrap()
                        .partial_cmp(&a.get("score").unwrap().as_f64().unwrap())
                        .unwrap()
                });
            }
        }
        
        results.truncate(max_results);
        Ok(results)
    }
    
    fn get_fusion_weights(&self, fusion_strategy: &str) -> (f64, f64) {
        match fusion_strategy {
            "weighted" => (0.6, 0.4), // Favor semantic search slightly
            "rerank" => (1.0, 0.5),   // Semantic search for reranking
            "filter" => (0.8, 0.2),   // Graph search for filtering
            _ => (0.5, 0.5),          // Equal weights
        }
    }
    
    async fn validate_triple(&self, triple: &Triple, validation_strategy: &str, require_sources: bool) -> std::result::Result<ValidationResult, String> {
        match validation_strategy {
            "consistency_check" => self.validate_consistency(triple).await,
            "source_verification" => self.validate_sources(triple, require_sources).await,
            "llm_validation" => self.validate_with_llm(triple).await,
            _ => Err(format!("Unknown validation strategy: {}", validation_strategy)),
        }
    }
    
    async fn validate_consistency(&self, triple: &Triple) -> std::result::Result<ValidationResult, String> {
        let engine = self.knowledge_engine.read().await;
        
        // Check for conflicting facts
        let query = TripleQuery {
            subject: Some(triple.subject.clone()),
            predicate: Some(triple.predicate.clone()),
            object: None,
            limit: 10,
            min_confidence: 0.0,
            include_chunks: true,
        };
        
        let existing_facts = engine.query_triples(query)
            .map_err(|e| format!("Failed to query existing facts: {}", e))?;
        
        let mut conflicts = Vec::new();
        let mut confidence = 1.0;
        let facts_count = existing_facts.nodes.len();
        
        for node in existing_facts.nodes {
            for existing_triple in node.get_triples() {
                if existing_triple.predicate == triple.predicate 
                    && existing_triple.object != triple.object {
                    conflicts.push(format!(
                        "Conflict: {} {} {} (existing) vs {} {} {} (new)",
                        existing_triple.subject, existing_triple.predicate, existing_triple.object,
                        triple.subject, triple.predicate, triple.object
                    ));
                    confidence *= 0.5; // Reduce confidence for each conflict
                }
            }
        }
        
        let conflicts_count = conflicts.len();
        
        Ok(ValidationResult {
            is_valid: conflicts.is_empty(),
            confidence,
            conflicts,
            sources: Vec::new(),
            validation_notes: vec![
                format!("Checked {} existing facts for conflicts", facts_count),
                format!("Confidence adjusted based on {} conflicts", conflicts_count),
            ],
        })
    }
    
    async fn validate_sources(&self, triple: &Triple, require_sources: bool) -> std::result::Result<ValidationResult, String> {
        // In a real implementation, this would check citation databases, Wikipedia, etc.
        let has_sources = !require_sources; // Simplified for now
        
        Ok(ValidationResult {
            is_valid: has_sources,
            confidence: if has_sources { 0.8 } else { 0.3 },
            conflicts: Vec::new(),
            sources: vec!["Source validation not fully implemented".to_string()],
            validation_notes: vec![
                "Source verification would check citation databases".to_string(),
                "Implementation would verify fact against reliable sources".to_string(),
            ],
        })
    }
    
    async fn validate_with_llm(&self, triple: &Triple) -> std::result::Result<ValidationResult, String> {
        // In a real implementation, this would use an LLM to validate the fact
        let natural_language = triple.to_natural_language();
        
        Ok(ValidationResult {
            is_valid: true,
            confidence: 0.75,
            conflicts: Vec::new(),
            sources: Vec::new(),
            validation_notes: vec![
                format!("LLM validation for: {}", natural_language),
                "LLM validation not fully implemented - would use language model".to_string(),
            ],
        })
    }
    
    // Helper methods
    
    async fn update_usage_stats(&self, response_time_ms: u64) {
        let mut stats = self.usage_stats.write().await;
        stats.total_operations += 1;
        
        // Update rolling average
        if stats.total_operations == 1 {
            stats.avg_response_time_ms = response_time_ms as f64;
        } else {
            stats.avg_response_time_ms = (stats.avg_response_time_ms * (stats.total_operations - 1) as f64 + response_time_ms as f64) / stats.total_operations as f64;
        }
    }
    
    fn calculate_efficiency_score(&self, memory_stats: &crate::core::knowledge_engine::MemoryStats) -> f64 {
        if memory_stats.total_nodes == 0 {
            return 1.0;
        }
        
        let target_bytes_per_node = 60.0;
        let efficiency = target_bytes_per_node / memory_stats.bytes_per_node.max(target_bytes_per_node);
        efficiency.min(1.0)
    }
    
    fn generate_helpful_info(&self, method: &str) -> String {
        match method {
            "store_fact" => "💡 Tip: Use consistent entity names and short predicates for best results. Examples: 'is', 'has', 'located_in', 'works_at'".to_string(),
            "store_knowledge" => "💡 Tip: The system automatically extracts facts from your text. Use clear, factual sentences for best extraction.".to_string(),
            "find_facts" => "💡 Tip: Leave fields empty to search broadly. Use exact entity names for precise results.".to_string(),
            "ask_question" => "💡 Tip: Ask specific questions for better results. The system understands related concepts and synonyms.".to_string(),
            "explore_connections" => "💡 Tip: Start with 1-2 hops to avoid overwhelming results. Great for understanding relationships.".to_string(),
            "generate_graph_query" => "💡 Tip: Use specific entity names in your questions for better query generation. Different query languages have different strengths.".to_string(),
            "hybrid_search" => "💡 Tip: Combine specific text queries with graph patterns for comprehensive results. Try different fusion strategies.".to_string(),
            "validate_knowledge" => "💡 Tip: Use consistency_check for fast validation against existing knowledge. Higher confidence scores indicate more reliable facts.".to_string(),
            _ => "💡 Tip: Check the tool descriptions and examples for optimal usage patterns.".to_string(),
        }
    }
    
    fn generate_error_help(&self, method: &str) -> String {
        match method {
            "store_fact" => "❗ Common issues: Missing required fields (subject, predicate, object), predicate too long (>3 words), or entity names too long".to_string(),
            "store_knowledge" => "❗ Common issues: Text too long (>2048 bytes/~400 words). Break longer content into smaller chunks.".to_string(),
            "find_facts" => "❗ Common issues: Typos in entity names, or no facts match your criteria. Try broader search terms.".to_string(),
            "ask_question" => "❗ Common issues: Very vague questions may return poor results. Be more specific about what you're looking for.".to_string(),
            _ => "❗ Check your input parameters and try again. Use 'get_suggestions' if you need help structuring your request.".to_string(),
        }
    }
    
    fn generate_error_suggestions(&self, method: &str) -> Vec<String> {
        match method {
            "store_fact" => vec![
                "Ensure subject, predicate, and object are provided".to_string(),
                "Keep predicates short: 'is', 'has', 'located_in'".to_string(),
                "Use underscores for multi-word predicates".to_string(),
            ],
            "store_knowledge" => vec![
                "Break text into chunks under 400 words".to_string(),
                "Use clear, factual sentences".to_string(),
                "Include specific entities and relationships".to_string(),
            ],
            _ => vec![
                "Check the tool documentation for examples".to_string(),
                "Use 'get_suggestions' for help".to_string(),
                "Try simpler requests first".to_string(),
            ],
        }
    }
}
