use warp::{Filter, Rejection, Reply};
use std::sync::Arc;
use parking_lot::RwLock;
use crate::core::knowledge_engine::KnowledgeEngine;
use super::handlers;

type SharedEngine = Arc<RwLock<KnowledgeEngine>>;

pub fn api_routes(
    engine: SharedEngine
) -> impl Filter<Extract = impl Reply, Error = Rejection> + Clone {
    let api_v1 = warp::path("api").and(warp::path("v1"));
    
    // API discovery endpoint
    let discovery = api_v1
        .and(warp::path("discovery"))
        .and(warp::get())
        .and_then(handlers::get_api_discovery);
    
    // Store triple endpoint
    let store_triple = api_v1
        .and(warp::path("triple"))
        .and(warp::post())
        .and(warp::body::json())
        .and(with_engine(engine.clone()))
        .and_then(handlers::store_triple);
    
    // Store chunk endpoint
    let store_chunk = api_v1
        .and(warp::path("chunk"))
        .and(warp::post())
        .and(warp::body::json())
        .and(with_engine(engine.clone()))
        .and_then(handlers::store_chunk);
    
    // Store entity endpoint
    let store_entity = api_v1
        .and(warp::path("entity"))
        .and(warp::post())
        .and(warp::body::json())
        .and(with_engine(engine.clone()))
        .and_then(handlers::store_entity);
    
    // Query triples endpoint
    let query_triples = api_v1
        .and(warp::path("query"))
        .and(warp::post())
        .and(warp::body::json())
        .and(with_engine(engine.clone()))
        .and_then(handlers::query_triples);
    
    // Semantic search endpoint
    let semantic_search = api_v1
        .and(warp::path("search"))
        .and(warp::post())
        .and(warp::body::json())
        .and(with_engine(engine.clone()))
        .and_then(handlers::semantic_search);
    
    // Get entity relationships endpoint
    let entity_relationships = api_v1
        .and(warp::path("relationships"))
        .and(warp::post())
        .and(warp::body::json())
        .and(with_engine(engine.clone()))
        .and_then(handlers::get_entity_relationships);
    
    // Get metrics endpoint
    let metrics = api_v1
        .and(warp::path("metrics"))
        .and(warp::get())
        .and(with_engine(engine.clone()))
        .and_then(handlers::get_metrics);
    
    // Get entity types endpoint
    let entity_types = api_v1
        .and(warp::path("entity-types"))
        .and(warp::get())
        .and(with_engine(engine.clone()))
        .and_then(handlers::get_entity_types);
    
    // Suggest predicates endpoint
    let suggest_predicates = api_v1
        .and(warp::path("suggest-predicates"))
        .and(warp::query::<SuggestPredicatesQuery>())
        .and(with_engine(engine.clone()))
        .and_then(handlers::suggest_predicates);
    
    // CORS headers
    let cors = warp::cors()
        .allow_any_origin()
        .allow_headers(vec!["content-type"])
        .allow_methods(vec!["GET", "POST", "PUT", "DELETE", "OPTIONS"]);
    
    discovery
        .or(store_triple)
        .or(store_chunk)
        .or(store_entity)
        .or(query_triples)
        .or(semantic_search)
        .or(entity_relationships)
        .or(metrics)
        .or(entity_types)
        .or(suggest_predicates)
        .with(cors)
}

fn with_engine(
    engine: SharedEngine
) -> impl Filter<Extract = (SharedEngine,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || engine.clone())
}

#[derive(Debug, serde::Deserialize)]
pub struct SuggestPredicatesQuery {
    pub context: String,
}