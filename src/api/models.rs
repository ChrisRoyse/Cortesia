use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Request models
#[derive(Debug, Deserialize)]
pub struct StoreTripleRequest {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: Option<f32>,
    pub metadata: Option<HashMap<String, String>>,
}

#[derive(Debug, Deserialize)]
pub struct StoreChunkRequest {
    pub text: String,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
pub struct StoreEntityRequest {
    pub name: String,
    pub entity_type: String,
    pub description: String,
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
pub struct QueryTriplesRequest {
    pub subject: Option<String>,
    pub predicate: Option<String>,
    pub object: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct SemanticSearchRequest {
    pub query: String,
    pub limit: usize,
}

#[derive(Debug, Deserialize)]
pub struct EntityRelationshipsRequest {
    pub entity_name: String,
    pub max_hops: Option<u8>,
}

// Response models
#[derive(Debug, Serialize)]
pub struct ApiResponse<T> {
    pub status: String,
    pub data: Option<T>,
    pub error: Option<String>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            status: "success".to_string(),
            data: Some(data),
            error: None,
        }
    }

    pub fn error(message: String) -> Self {
        Self {
            status: "error".to_string(),
            data: None,
            error: Some(message),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct StoreTripleResponse {
    pub node_id: String,
}

#[derive(Debug, Serialize)]
pub struct MetricsResponse {
    pub entity_count: usize,
    pub memory_stats: MemoryStatsJson,
    pub entity_types: HashMap<String, String>,
}

#[derive(Debug, Serialize)]
pub struct MemoryStatsJson {
    pub total_nodes: usize,
    pub total_triples: usize,
    pub total_bytes: usize,
    pub bytes_per_node: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

impl From<crate::core::knowledge_types::MemoryStats> for MemoryStatsJson {
    fn from(stats: crate::core::knowledge_types::MemoryStats) -> Self {
        Self {
            total_nodes: stats.total_nodes,
            total_triples: stats.total_triples,
            total_bytes: stats.total_bytes,
            bytes_per_node: stats.bytes_per_node,
            cache_hits: stats.cache_hits,
            cache_misses: stats.cache_misses,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TripleJson {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub confidence: f32,
    pub source: Option<String>,
}

impl From<crate::core::triple::Triple> for TripleJson {
    fn from(triple: crate::core::triple::Triple) -> Self {
        Self {
            subject: triple.subject,
            predicate: triple.predicate,
            object: triple.object,
            confidence: triple.confidence,
            source: triple.source,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct QueryResponse {
    pub triples: Vec<TripleJson>,
    pub chunks: Vec<ChunkJson>,
    pub query_time_ms: u128,
}

#[derive(Debug, Serialize)]
pub struct ChunkJson {
    pub id: String,
    pub text: String,
    pub score: f32,
}

// API discovery models
#[derive(Debug, Serialize)]
pub struct ApiEndpoint {
    pub path: String,
    pub method: String,
    pub description: String,
    pub request_schema: Option<serde_json::Value>,
    pub response_schema: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ApiDiscoveryResponse {
    pub version: String,
    pub endpoints: Vec<ApiEndpoint>,
}