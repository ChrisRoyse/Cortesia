//! Common test utilities for MCP server tests

use std::sync::Arc;
use tokio::sync::RwLock;
use crate::core::knowledge_engine::KnowledgeEngine;
use crate::mcp::llm_friendly_server::types::UsageStats;
use crate::error::Result;

/// Create a test knowledge engine with optional sample data
pub async fn create_test_engine(with_data: bool) -> Result<Arc<RwLock<KnowledgeEngine>>> {
    let engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(384, 100_000)?
    ));
    
    if with_data {
        let mut engine_write = engine.write().await;
        
        // Add test triples
        engine_write.add_triple("Einstein", "is", "scientist", 1.0)?;
        engine_write.add_triple("Einstein", "invented", "relativity", 1.0)?;
        engine_write.add_triple("relativity", "is", "theory", 1.0)?;
        engine_write.add_triple("Newton", "is", "scientist", 1.0)?;
        engine_write.add_triple("Newton", "discovered", "gravity", 1.0)?;
        engine_write.add_triple("Marie Curie", "is", "scientist", 1.0)?;
        engine_write.add_triple("Marie Curie", "discovered", "radium", 1.0)?;
        
        // Add test chunks
        engine_write.add_knowledge_chunk(
            "Einstein's Theory",
            "Albert Einstein developed the theory of relativity, which revolutionized physics.",
            Some("physics"),
            Some("test source")
        )?;
        
        engine_write.add_knowledge_chunk(
            "Newton's Laws",
            "Sir Isaac Newton formulated the laws of motion and universal gravitation.",
            Some("physics"),
            Some("test source")
        )?;
        
        engine_write.add_knowledge_chunk(
            "Marie Curie's Research",
            "Marie Curie was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences.",
            Some("chemistry"),
            Some("test source")
        )?;
    }
    
    Ok(engine)
}

/// Create test usage stats
pub fn create_test_stats() -> Arc<RwLock<UsageStats>> {
    Arc::new(RwLock::new(UsageStats::default()))
}

/// Create test graph query parameters
pub fn create_test_graph_query_params() -> serde_json::Value {
    serde_json::json!({
        "natural_query": "Find all scientists and their discoveries",
        "query_language": "cypher",
        "include_explanation": true
    })
}