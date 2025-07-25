use warp::{Rejection, Reply, reply};
use std::sync::Arc;
use parking_lot::RwLock;
use crate::core::knowledge_engine::KnowledgeEngine;
use crate::core::triple::{Triple};
use crate::core::knowledge_types::TripleQuery as CoreTripleQuery;
use super::models::*;
use super::routes::SuggestPredicatesQuery;

type SharedEngine = Arc<RwLock<KnowledgeEngine>>;

pub async fn get_api_discovery() -> Result<impl Reply, Rejection> {
    let endpoints = vec![
        ApiEndpoint {
            path: "/api/v1/triple".to_string(),
            method: "POST".to_string(),
            description: "Store a knowledge triple (subject-predicate-object)".to_string(),
            request_schema: Some(serde_json::json!({
                "subject": "string",
                "predicate": "string",
                "object": "string",
                "confidence": "number (optional)",
                "metadata": "object (optional)"
            })),
            response_schema: Some(serde_json::json!({
                "status": "success|error",
                "data": { "node_id": "string" }
            })),
        },
        ApiEndpoint {
            path: "/api/v1/chunk".to_string(),
            method: "POST".to_string(),
            description: "Store a text chunk with optional embedding".to_string(),
            request_schema: Some(serde_json::json!({
                "text": "string",
                "embedding": "array of numbers (optional)"
            })),
            response_schema: Some(serde_json::json!({
                "status": "success|error",
                "data": { "node_id": "string" }
            })),
        },
        ApiEndpoint {
            path: "/api/v1/entity".to_string(),
            method: "POST".to_string(),
            description: "Store an entity with properties".to_string(),
            request_schema: Some(serde_json::json!({
                "name": "string",
                "entity_type": "string",
                "description": "string",
                "properties": "object"
            })),
            response_schema: Some(serde_json::json!({
                "status": "success|error",
                "data": { "node_id": "string" }
            })),
        },
        ApiEndpoint {
            path: "/api/v1/query".to_string(),
            method: "POST".to_string(),
            description: "Query for triples matching criteria".to_string(),
            request_schema: Some(serde_json::json!({
                "subject": "string (optional)",
                "predicate": "string (optional)",
                "object": "string (optional)",
                "limit": "number (optional)"
            })),
            response_schema: Some(serde_json::json!({
                "status": "success|error",
                "data": {
                    "triples": "array",
                    "chunks": "array",
                    "query_time_ms": "number"
                }
            })),
        },
        ApiEndpoint {
            path: "/api/v1/search".to_string(),
            method: "POST".to_string(),
            description: "Semantic search for similar content".to_string(),
            request_schema: Some(serde_json::json!({
                "query": "string",
                "limit": "number"
            })),
            response_schema: Some(serde_json::json!({
                "status": "success|error",
                "data": {
                    "results": "array",
                    "query_time_ms": "number"
                }
            })),
        },
        ApiEndpoint {
            path: "/api/v1/metrics".to_string(),
            method: "GET".to_string(),
            description: "Get system metrics and statistics".to_string(),
            request_schema: None,
            response_schema: Some(serde_json::json!({
                "status": "success|error",
                "data": {
                    "entity_count": "number",
                    "memory_stats": "object",
                    "entity_types": "object"
                }
            })),
        },
    ];
    
    let response = ApiDiscoveryResponse {
        version: "1.0.0".to_string(),
        endpoints,
    };
    
    Ok(reply::json(&response))
}

pub async fn store_triple(
    req: StoreTripleRequest,
    engine: SharedEngine,
) -> Result<impl Reply, Rejection> {
    let triple = Triple {
        subject: req.subject,
        predicate: req.predicate,
        object: req.object,
        confidence: req.confidence.unwrap_or(1.0),
        source: None,
        enhanced_metadata: None,
    };
    
    match engine.write().store_triple(triple, None) {
        Ok(node_id) => {
            let response = ApiResponse::success(StoreTripleResponse { node_id });
            Ok(reply::json(&response))
        }
        Err(e) => {
            let response: ApiResponse<StoreTripleResponse> = ApiResponse::error(e.to_string());
            Ok(reply::json(&response))
        }
    }
}

pub async fn store_chunk(
    req: StoreChunkRequest,
    engine: SharedEngine,
) -> Result<impl Reply, Rejection> {
    match engine.write().store_chunk(req.text, req.embedding) {
        Ok(node_id) => {
            let response = ApiResponse::success(StoreTripleResponse { node_id });
            Ok(reply::json(&response))
        }
        Err(e) => {
            let response: ApiResponse<StoreTripleResponse> = ApiResponse::error(e.to_string());
            Ok(reply::json(&response))
        }
    }
}

pub async fn store_entity(
    req: StoreEntityRequest,
    engine: SharedEngine,
) -> Result<impl Reply, Rejection> {
    match engine.write().store_entity(req.name, req.entity_type, req.description, req.properties) {
        Ok(node_id) => {
            let response = ApiResponse::success(StoreTripleResponse { node_id });
            Ok(reply::json(&response))
        }
        Err(e) => {
            let response: ApiResponse<StoreTripleResponse> = ApiResponse::error(e.to_string());
            Ok(reply::json(&response))
        }
    }
}

pub async fn query_triples(
    req: QueryTriplesRequest,
    engine: SharedEngine,
) -> Result<impl Reply, Rejection> {
    let query = CoreTripleQuery {
        subject: req.subject,
        predicate: req.predicate,
        object: req.object,
        limit: req.limit.unwrap_or(100),
        min_confidence: 0.0,
        include_chunks: false,
    };
    
    match engine.read().query_triples(query) {
        Ok(result) => {
            let triples: Vec<TripleJson> = result.triples.into_iter().map(Into::into).collect();
            let chunks: Vec<ChunkJson> = vec![]; // No chunks in current result structure
            
            let response = ApiResponse::success(QueryResponse {
                triples,
                chunks,
                query_time_ms: result.query_time_ms as u128,
            });
            Ok(reply::json(&response))
        }
        Err(e) => {
            let response: ApiResponse<QueryResponse> = ApiResponse::error(e.to_string());
            Ok(reply::json(&response))
        }
    }
}

#[allow(unused_variables)]
pub async fn semantic_search(
    req: SemanticSearchRequest,
    engine: SharedEngine,
) -> Result<impl Reply, Rejection> {
    match engine.read().semantic_search(&req.query, req.limit) {
        Ok(result) => {
            let chunks: Vec<ChunkJson> = vec![]; // No chunks in current result structure
            
            let response = ApiResponse::success(serde_json::json!({
                "results": result.triples,
                "query_time_ms": result.query_time_ms,
            }));
            Ok(reply::json(&response))
        }
        Err(e) => {
            let response: ApiResponse<serde_json::Value> = ApiResponse::error(e.to_string());
            Ok(reply::json(&response))
        }
    }
}

pub async fn get_entity_relationships(
    req: EntityRelationshipsRequest,
    engine: SharedEngine,
) -> Result<impl Reply, Rejection> {
    let max_hops = req.max_hops.unwrap_or(2);
    
    match engine.read().get_entity_relationships(&req.entity_name, max_hops) {
        Ok(triples) => {
            let triples: Vec<TripleJson> = triples.into_iter().map(Into::into).collect();
            let response = ApiResponse::success(serde_json::json!({
                "entity": req.entity_name,
                "relationships": triples,
            }));
            Ok(reply::json(&response))
        }
        Err(e) => {
            let response: ApiResponse<serde_json::Value> = ApiResponse::error(e.to_string());
            Ok(reply::json(&response))
        }
    }
}

pub async fn get_metrics(engine: SharedEngine) -> Result<impl Reply, Rejection> {
    let engine = engine.read();
    let memory_stats = engine.get_memory_stats();
    let entity_types = engine.get_entity_types();
    let entity_count = engine.get_entity_count();
    
    let response = ApiResponse::success(MetricsResponse {
        entity_count,
        memory_stats: memory_stats.into(),
        entity_types,
    });
    
    Ok(reply::json(&response))
}

pub async fn get_entity_types(engine: SharedEngine) -> Result<impl Reply, Rejection> {
    let entity_types = engine.read().get_entity_types();
    let response = ApiResponse::success(entity_types);
    Ok(reply::json(&response))
}

pub async fn suggest_predicates(
    query: SuggestPredicatesQuery,
    engine: SharedEngine,
) -> Result<impl Reply, Rejection> {
    let suggestions = engine.read().suggest_predicates(&query.context);
    let response = ApiResponse::success(suggestions);
    Ok(reply::json(&response))
}