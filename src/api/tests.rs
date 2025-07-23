use super::*;
use crate::core::knowledge_engine::KnowledgeEngine;
use warp::test::request;
use serde_json::json;

#[tokio::test]
async fn test_store_triple_endpoint() {
    // Create a real knowledge engine
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(384, 10000).unwrap()));
    
    // Create the API filter
    let api = routes::api_routes(engine.clone());
    
    // Test storing a triple
    let triple_data = json!({
        "subject": "Einstein",
        "predicate": "invented",
        "object": "Theory of Relativity"
    });
    
    let resp = request()
        .method("POST")
        .path("/api/v1/triple")
        .json(&triple_data)
        .reply(&api)
        .await;
    
    assert_eq!(resp.status(), 200);
    
    // Verify the response contains a node_id
    let body: serde_json::Value = serde_json::from_slice(resp.body()).unwrap();
    assert!(body["node_id"].is_string());
    assert_eq!(body["status"], "success");
}

#[tokio::test]
async fn test_query_triples_endpoint() {
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(384, 10000).unwrap()));
    
    // First store a triple
    engine.write().store_triple(
        crate::core::triple::Triple {
            subject: "Einstein".to_string(),
            predicate: "invented".to_string(),
            object: "Theory of Relativity".to_string(),
            confidence: 1.0,
            source: None,
        },
        None
    ).unwrap();
    
    let api = routes::api_routes(engine.clone());
    
    // Query for the triple
    let query = json!({
        "subject": "Einstein"
    });
    
    let resp = request()
        .method("POST")
        .path("/api/v1/query")
        .json(&query)
        .reply(&api)
        .await;
    
    assert_eq!(resp.status(), 200);
    
    let body: serde_json::Value = serde_json::from_slice(resp.body()).unwrap();
    assert!(body["triples"].is_array());
    assert_eq!(body["triples"][0]["subject"], "Einstein");
}

#[tokio::test]
async fn test_get_metrics_endpoint() {
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(384, 10000).unwrap()));
    let api = routes::api_routes(engine);
    
    let resp = request()
        .method("GET")
        .path("/api/v1/metrics")
        .reply(&api)
        .await;
    
    assert_eq!(resp.status(), 200);
    
    let body: serde_json::Value = serde_json::from_slice(resp.body()).unwrap();
    assert!(body["entity_count"].is_number());
    assert!(body["memory_stats"].is_object());
}

#[tokio::test]
async fn test_semantic_search_endpoint() {
    let engine = Arc::new(RwLock::new(KnowledgeEngine::new(384, 10000).unwrap()));
    
    // Store some data first
    engine.write().store_chunk(
        "Einstein developed the theory of relativity".to_string(),
        None
    ).unwrap();
    
    let api = routes::api_routes(engine);
    
    let search_query = json!({
        "query": "relativity theory",
        "limit": 10
    });
    
    let resp = request()
        .method("POST")
        .path("/api/v1/search")
        .json(&search_query)
        .reply(&api)
        .await;
    
    assert_eq!(resp.status(), 200);
    
    let body: serde_json::Value = serde_json::from_slice(resp.body()).unwrap();
    assert!(body["results"].is_array());
}

use std::sync::Arc;
use parking_lot::RwLock;