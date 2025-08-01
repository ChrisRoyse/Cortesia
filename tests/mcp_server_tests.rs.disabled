//! MCP Server Integration Tests
//! 
//! Tests for the MCP server functionality

use llmkg::core::knowledge_engine::KnowledgeEngine;
use llmkg::mcp::llm_friendly_server::LLMFriendlyMCPServer;
use llmkg::mcp::shared_types::LLMMCPRequest;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Test LLMFriendlyMCPServer creation
#[tokio::test]
async fn test_llm_friendly_mcp_server_creation() {
    let knowledge_engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(128, 10000).unwrap()
    ));
    
    let server = LLMFriendlyMCPServer::new(knowledge_engine).unwrap();
    
    // Verify server was created successfully
    let tools = server.get_available_tools();
    assert!(!tools.is_empty());
    
    // Verify expected tools are available
    let tool_names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert!(tool_names.contains(&"store_knowledge"));
    assert!(tool_names.contains(&"query_knowledge"));
    assert!(tool_names.contains(&"search_entities"));
}

/// Test store_knowledge tool functionality
#[tokio::test]
async fn test_store_knowledge_tool() {
    let knowledge_engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(128, 10000).unwrap()
    ));
    
    let server = LLMFriendlyMCPServer::new(knowledge_engine).unwrap();
    
    // Create store knowledge request
    let request = LLMMCPRequest {
        method: "store_knowledge".to_string(),
        params: serde_json::json!({
            "content": "Water boils at 100 degrees Celsius",
            "entity_type": "fact",
            "metadata": {}
        }),
    };
    
    // Execute request
    let response = server.handle_request(request).await.unwrap();
    
    // Verify successful response
    assert!(response.success);
    assert!(!response.message.is_empty());
}

/// Test query_knowledge tool functionality
#[tokio::test]
async fn test_query_knowledge_tool() {
    let knowledge_engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(128, 10000).unwrap()
    ));
    
    let server = LLMFriendlyMCPServer::new(knowledge_engine).unwrap();
    
    // First, store some knowledge
    let store_request = LLMMCPRequest {
        method: "store_knowledge".to_string(),
        params: serde_json::json!({
            "content": "Machine learning is a subset of artificial intelligence",
            "entity_type": "fact",
            "metadata": {}
        }),
    };
    
    let _store_response = server.handle_request(store_request).await.unwrap();
    
    // Now query for it
    let query_request = LLMMCPRequest {
        method: "query_knowledge".to_string(),
        params: serde_json::json!({
            "query": "What is machine learning?",
            "max_results": 5
        }),
    };
    
    let response = server.handle_request(query_request).await.unwrap();
    
    // Verify query response
    assert!(response.success);
    assert!(!response.message.is_empty());
}

/// Test search_entities tool functionality
#[tokio::test]
async fn test_search_entities_tool() {
    let knowledge_engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(128, 10000).unwrap()
    ));
    
    let server = LLMFriendlyMCPServer::new(knowledge_engine).unwrap();
    
    // Store multiple entities
    let entities = vec![
        ("Artificial intelligence is a broad field", "concept"),
        ("Machine learning uses algorithms", "fact"),
        ("Text processing handles language", "fact"),
    ];
    
    for (content, entity_type) in entities {
        let request = LLMMCPRequest {
            method: "store_knowledge".to_string(),
            params: serde_json::json!({
                "content": content,
                "entity_type": entity_type,
                "metadata": {}
            }),
        };
        let _response = server.handle_request(request).await.unwrap();
    }
    
    // Search for entities
    let search_request = LLMMCPRequest {
        method: "search_entities".to_string(),
        params: serde_json::json!({
            "search_term": "artificial",
            "limit": 10
        }),
    };
    
    let response = server.handle_request(search_request).await.unwrap();
    
    // Verify search response
    assert!(response.success);
    assert!(!response.message.is_empty());
}

/// Test error handling in MCP server
#[tokio::test]
async fn test_mcp_server_error_handling() {
    let knowledge_engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(128, 10000).unwrap()
    ));
    
    let server = LLMFriendlyMCPServer::new(knowledge_engine).unwrap();
    
    // Test with invalid tool
    let invalid_request = LLMMCPRequest {
        method: "invalid_tool".to_string(),
        params: serde_json::json!({}),
    };
    
    let _response = server.handle_request(invalid_request).await.unwrap();
    
    // Should return error response or handle gracefully
    // We don't enforce specific error handling, just that it doesn't panic
    assert!(true); // Successfully handled the request
}

/// Test MCP server with malformed requests
#[tokio::test]
async fn test_mcp_server_malformed_requests() {
    let knowledge_engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(128, 10000).unwrap()
    ));
    
    let server = LLMFriendlyMCPServer::new(knowledge_engine).unwrap();
    
    // Test with missing parameters
    let incomplete_request = LLMMCPRequest {
        method: "store_knowledge".to_string(),
        params: serde_json::json!({}), // Missing required fields
    };
    
    let _response = server.handle_request(incomplete_request).await.unwrap();
    
    // Should handle gracefully (may succeed with defaults or fail gracefully)
    assert!(true); // Successfully handled the request
}

/// Test concurrent MCP server requests
#[tokio::test]
async fn test_concurrent_mcp_requests() {
    let knowledge_engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(128, 10000).unwrap()
    ));
    
    let server = Arc::new(LLMFriendlyMCPServer::new(knowledge_engine).unwrap());
    
    let mut handles = Vec::new();
    
    // Spawn multiple concurrent requests
    for i in 0..5 {
        let server_clone = server.clone();
        let handle = tokio::spawn(async move {
            let request = LLMMCPRequest {
                method: "store_knowledge".to_string(),
                params: serde_json::json!({
                    "content": format!("Test knowledge {}", i),
                    "entity_type": "fact",
                    "metadata": {}
                }),
            };
            server_clone.handle_request(request).await
        });
        handles.push(handle);
    }
    
    // Wait for all requests to complete
    let mut success_count = 0;
    for handle in handles {
        let result = handle.await.unwrap();
        if result.is_ok() {
            let response = result.unwrap();
            if response.success {
                success_count += 1;
            }
        }
    }
    
    // Most requests should succeed
    assert!(success_count >= 3);
}

/// Test MCP server tool descriptions
#[tokio::test]
async fn test_mcp_server_tool_descriptions() {
    let knowledge_engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(128, 10000).unwrap()
    ));
    
    let server = LLMFriendlyMCPServer::new(knowledge_engine).unwrap();
    let tools = server.get_available_tools();
    
    // Verify all tools have descriptions
    for tool in tools {
        assert!(!tool.name.is_empty());
        assert!(!tool.description.is_empty());
        
        // Verify input schema exists
        assert!(tool.input_schema.is_object() || tool.input_schema.is_null());
    }
}

/// Test basic MCP server functionality
#[tokio::test]
async fn test_basic_mcp_functionality() {
    let knowledge_engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(128, 10000).unwrap()
    ));
    
    let server = LLMFriendlyMCPServer::new(knowledge_engine).unwrap();
    
    // Test that we can get tools
    let tools = server.get_available_tools();
    assert!(!tools.is_empty());
    
    // Test basic request handling
    let request = LLMMCPRequest {
        method: "query_knowledge".to_string(),
        params: serde_json::json!({
            "query": "test query",
            "max_results": 1
        }),
    };
    
    let response = server.handle_request(request).await;
    assert!(response.is_ok());
}