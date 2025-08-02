use wasm_bindgen::prelude::*;
use crate::core::graph::KnowledgeGraph;
use crate::core::types::{EntityData, Relationship};
use crate::error::GraphError;
use js_sys::{Array, Object, Reflect};
use web_sys::console;

pub mod fast_interface;

// Global panic hook for better error reporting
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct KnowledgeGraphWasm {
    inner: KnowledgeGraph,
}

#[wasm_bindgen]
impl KnowledgeGraphWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(embedding_dimension: usize) -> Result<KnowledgeGraphWasm, JsValue> {
        let graph = KnowledgeGraph::new(embedding_dimension)
            .map_err(|e| JsValue::from_str(&format!("Failed to create knowledge graph: {}", e)))?;
        
        Ok(Self { inner: graph })
    }
    
    /// Inserts an entity into the knowledge graph
    /// 
    /// # Parameters
    /// - id: Unique identifier for the entity
    /// - type_id: Type classification for the entity
    /// - properties: JSON string containing entity properties
    /// - embedding: Float32Array containing the entity's embedding vector
    /// 
    /// # Returns
    /// Success message or error
    #[wasm_bindgen]
    pub fn insert_entity(&self, id: u32, type_id: u16, properties: &str, embedding: &[f32]) -> Result<String, JsValue> {
        let entity_data = EntityData {
            type_id,
            properties: properties.to_string(),
            embedding: embedding.to_vec(),
        };
        
        self.inner.insert_entity(id, entity_data)
            .map(|_| "Entity inserted successfully".to_string())
            .map_err(|e| JsValue::from_str(&format!("Failed to insert entity: {}", e)))
    }
    
    /// Adds a relationship between two entities
    /// 
    /// # Parameters
    /// - from_id: Source entity ID
    /// - to_id: Target entity ID
    /// - relationship_type: Type of relationship (0-255)
    /// - weight: Strength of the relationship (0.0-1.0)
    #[wasm_bindgen]
    pub fn insert_relationship(&self, from_id: u32, to_id: u32, relationship_type: u8, weight: f32) -> Result<String, JsValue> {
        let relationship = Relationship {
            from: from_id,
            to: to_id,
            rel_type: relationship_type,
            weight,
        };
        
        self.inner.insert_relationship(relationship)
            .map(|_| "Relationship inserted successfully".to_string())
            .map_err(|e| JsValue::from_str(&format!("Failed to insert relationship: {}", e)))
    }
    
    /// Performs semantic search using an embedding vector
    /// 
    /// # Parameters
    /// - query_embedding: Float32Array containing the query embedding
    /// - max_results: Maximum number of results to return
    /// 
    /// # Returns
    /// JSON string containing search results with entity IDs and similarity scores
    #[wasm_bindgen]
    pub fn semantic_search(&self, query_embedding: &[f32], max_results: usize) -> Result<String, JsValue> {
        let results = self.inner.similarity_search(query_embedding, max_results)
            .map_err(|e| JsValue::from_str(&format!("Search failed: {}", e)))?;
        
        serde_json::to_string(&results)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize results: {}", e)))
    }
    
    /// Gets neighboring entities for a given entity
    /// 
    /// # Parameters
    /// - entity_id: ID of the entity to get neighbors for
    /// 
    /// # Returns
    /// Array of neighboring entity IDs
    #[wasm_bindgen]
    pub fn get_neighbors(&self, entity_id: u32) -> Result<Vec<u32>, JsValue> {
        self.inner.get_neighbors(entity_id)
            .map_err(|e| JsValue::from_str(&format!("Failed to get neighbors: {}", e)))
    }
    
    /// Finds a path between two entities
    /// 
    /// # Parameters
    /// - from_id: Starting entity ID
    /// - to_id: Target entity ID
    /// - max_depth: Maximum path length to search
    /// 
    /// # Returns
    /// Array of entity IDs representing the path, or null if no path exists
    #[wasm_bindgen]
    pub fn find_path(&self, from_id: u32, to_id: u32, max_depth: u8) -> Result<Option<Vec<u32>>, JsValue> {
        self.inner.find_path(from_id, to_id, max_depth)
            .map_err(|e| JsValue::from_str(&format!("Failed to find path: {}", e)))
    }
    
    /// Performs a complete knowledge graph query suitable for LLM context
    /// 
    /// # Parameters
    /// - query_embedding: Float32Array containing the query embedding
    /// - max_entities: Maximum number of entities to include in context
    /// - max_depth: Maximum relationship depth to explore
    /// 
    /// # Returns
    /// JSON string containing structured knowledge graph context
    #[wasm_bindgen]
    pub fn get_context(&self, query_embedding: &[f32], max_entities: usize, max_depth: u8) -> Result<String, JsValue> {
        let result = self.inner.query(query_embedding, max_entities, max_depth)
            .map_err(|e| JsValue::from_str(&format!("Query failed: {}", e)))?;
        
        serde_json::to_string(&result)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize context: {}", e)))
    }
    
    /// Returns system statistics and performance metrics
    /// 
    /// # Returns
    /// JSON string containing system statistics
    #[wasm_bindgen]
    pub fn get_stats(&self) -> String {
        let memory_usage = self.inner.memory_usage();
        let entity_count = self.inner.entity_count();
        let relationship_count = self.inner.relationship_count();
        
        let stats = serde_json::json!({
            "entity_count": entity_count,
            "relationship_count": relationship_count,
            "memory_usage": {
                "total_bytes": memory_usage.total_bytes(),
                "bytes_per_entity": memory_usage.bytes_per_entity(entity_count),
                "arena_bytes": memory_usage.arena_bytes,
                "graph_bytes": memory_usage.graph_bytes,
                "embedding_bytes": memory_usage.embedding_bank_bytes
            }
        });
        
        serde_json::to_string(&stats).unwrap_or_else(|_| "{}".to_string())
    }
    
    /// Returns API capabilities and documentation for LLM discovery
    /// 
    /// # Returns
    /// JSON string containing comprehensive API documentation
    #[wasm_bindgen]
    pub fn get_api_capabilities() -> String {
        let capabilities = serde_json::json!({
            "name": "LLMKG - LLM Knowledge Graph",
            "version": "0.1.0",
            "description": "Ultra-fast knowledge graph optimized for LLM integration",
            "capabilities": {
                "max_entities": 100_000_000,
                "max_embedding_dimension": 4096,
                "supported_formats": ["json", "binary"],
                "real_time_updates": true,
                "distributed_queries": false
            },
            "functions": [
                {
                    "name": "semantic_search",
                    "description": "Find entities similar to a query embedding using vector similarity",
                    "parameters": {
                        "query_embedding": {
                            "type": "Float32Array",
                            "description": "Embedding vector for the search query",
                            "required": true
                        },
                        "max_results": {
                            "type": "number",
                            "description": "Maximum number of results to return (default: 20, max: 1000)",
                            "required": false,
                            "default": 20
                        }
                    },
                    "returns": "JSON array of {id: number, similarity: number} objects",
                    "examples": [
                        {
                            "description": "Find entities similar to a concept",
                            "code": "kg.semantic_search(concept_embedding, 10)"
                        }
                    ]
                },
                {
                    "name": "get_context",
                    "description": "Retrieve comprehensive knowledge graph context for LLM consumption",
                    "parameters": {
                        "query_embedding": {
                            "type": "Float32Array",
                            "description": "Embedding vector for the context query",
                            "required": true
                        },
                        "max_entities": {
                            "type": "number", 
                            "description": "Maximum entities to include in context",
                            "required": false,
                            "default": 20
                        },
                        "max_depth": {
                            "type": "number",
                            "description": "Maximum relationship depth to explore",
                            "required": false,
                            "default": 2
                        }
                    },
                    "returns": "JSON object with entities, relationships, and metadata",
                    "examples": [
                        {
                            "description": "Get knowledge context for answering a question",
                            "code": "kg.get_context(question_embedding, 25, 3)"
                        }
                    ]
                },
                {
                    "name": "find_path",
                    "description": "Find the shortest path between two entities",
                    "parameters": {
                        "from_id": {"type": "number", "required": true},
                        "to_id": {"type": "number", "required": true},
                        "max_depth": {"type": "number", "required": false, "default": 6}
                    },
                    "returns": "Array of entity IDs representing the path, or null",
                    "examples": [
                        {
                            "description": "Find connection between two concepts",
                            "code": "kg.find_path(entity1_id, entity2_id, 4)"
                        }
                    ]
                }
            ],
            "performance_targets": {
                "query_latency_ms": "<1",
                "similarity_search_ms": "<5", 
                "context_retrieval_ms": "<10",
                "bytes_per_entity": "<70"
            },
            "usage_patterns": {
                "graph_rag": "Use get_context() to retrieve relevant knowledge before LLM generation",
                "entity_discovery": "Use semantic_search() to find entities matching a description",
                "relationship_exploration": "Use get_neighbors() and find_path() to understand connections",
                "knowledge_expansion": "Combine multiple queries to build comprehensive understanding"
            }
        });
        
        serde_json::to_string(&capabilities).unwrap_or_else(|_| "{}".to_string())
    }
    
    /// Explains the relationship between two entities
    /// 
    /// # Parameters
    /// - entity_a_id: First entity ID
    /// - entity_b_id: Second entity ID
    /// 
    /// # Returns
    /// JSON string with relationship explanation and supporting evidence
    #[wasm_bindgen]
    pub fn explain_relationship(&self, entity_a_id: u32, entity_b_id: u32) -> Result<String, JsValue> {
        // Find path and neighboring context
        let path = self.inner.find_path(entity_a_id, entity_b_id, 4)
            .map_err(|e| JsValue::from_str(&format!("Failed to find path: {}", e)))?;
        
        let neighbors_a = self.inner.get_neighbors(entity_a_id)
            .map_err(|e| JsValue::from_str(&format!("Failed to get neighbors: {}", e)))?;
        let neighbors_b = self.inner.get_neighbors(entity_b_id)
            .map_err(|e| JsValue::from_str(&format!("Failed to get neighbors: {}", e)))?;
        
        let explanation = serde_json::json!({
            "entities": [entity_a_id, entity_b_id],
            "direct_connection": path.as_ref().map(|p| p.len() == 2).unwrap_or(false),
            "path": path,
            "path_length": path.as_ref().map(|p| p.len().saturating_sub(1)).unwrap_or(0),
            "common_neighbors": neighbors_a.iter()
                .filter(|id| neighbors_b.contains(id))
                .collect::<Vec<_>>(),
            "relationship_strength": path.as_ref()
                .map(|p| 1.0 / (p.len() as f32).max(1.0))
                .unwrap_or(0.0)
        });
        
        serde_json::to_string(&explanation)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize explanation: {}", e)))
    }
}

// Helper functions for JavaScript interop
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[wasm_bindgen]
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}

// Performance measurement utilities
#[wasm_bindgen]
pub struct PerformanceTimer {
    start: f64,
}

#[wasm_bindgen]
impl PerformanceTimer {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let start = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        Self { start }
    }
    
    #[wasm_bindgen]
    pub fn elapsed_ms(&self) -> f64 {
        let now = web_sys::window()
            .unwrap()
            .performance()
            .unwrap()
            .now();
        now - self.start
    }
}