#[cfg(test)]
mod phase2_mcp_tests {
    use llmkg::mcp::brain_inspired_server::{BrainInspiredMCPServer, MCPRequest, MCPResponse};
    use llmkg::versioning::temporal_graph::TemporalKnowledgeGraph;
    use llmkg::core::graph::KnowledgeGraph;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use serde_json::json;

    async fn setup_test_server() -> BrainInspiredMCPServer {
        let graph = KnowledgeGraph::new(384).unwrap();
        let temporal_graph = Arc::new(RwLock::new(TemporalKnowledgeGraph::new(graph)));
        
        let server = BrainInspiredMCPServer::new(
            temporal_graph.clone(),
        );
        
        // Add test data
        let store_request = MCPRequest {
            tool: "store_knowledge".to_string(),
            arguments: json!({
                "text": "Dogs are mammals with four legs. Cats are also mammals with four legs. Mammals are warm-blooded animals.",
                "use_graph_construction": true
            }),
        };
        
        let _ = server.handle_tool_call(store_request).await.unwrap();
        
        server
    }

    #[tokio::test]
    async fn test_graph_store_fact() {
        let server = setup_test_server().await;
        
        // Test graph construction
        let request = MCPRequest {
            tool: "store_knowledge".to_string(),
            arguments: json!({
                "text": "Tripper is a three-legged dog who loves to play fetch.",
                "use_graph_construction": true,
                "context": "Adding information about a specific dog"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        match response {
            MCPResponse { content, is_error } => {
                assert!(!is_error);
                assert!(!content.is_empty());
                assert!(content[0].text.contains("Neural graph construction completed"));
            }
        }
    }

    #[tokio::test]
    async fn test_cognitive_reasoning_convergent() {
        let server = setup_test_server().await;
        
        // Test convergent reasoning through MCP
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "What type of animal is a dog?",
                "pattern": "convergent"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        match response {
            MCPResponse { content, is_error } => {
                assert!(!is_error);
                assert!(!content.is_empty());
                let result_text = &content[0].text;
                assert!(result_text.contains("mammal") || result_text.contains("animal"));
            }
        }
    }

    #[tokio::test]
    async fn test_cognitive_reasoning_divergent() {
        let server = setup_test_server().await;
        
        // Test divergent reasoning
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "What are examples of mammals?",
                "pattern": "divergent"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        match response {
            MCPResponse { content, is_error } => {
                assert!(!is_error);
                assert!(!content.is_empty());
                let result_text = &content[0].text;
                assert!(result_text.contains("dog") || result_text.contains("cat"));
            }
        }
    }

    #[tokio::test]
    async fn test_cognitive_reasoning_lateral() {
        let server = setup_test_server().await;
        
        // Add more data for lateral connections
        let setup_request = MCPRequest {
            tool: "store_knowledge".to_string(),
            arguments: json!({
                "text": "Dogs and cats are both popular pets. Pets provide companionship to humans.",
                "use_graph_construction": true
            }),
        };
        let _ = server.handle_tool_call(setup_request).await.unwrap();
        
        // Test lateral thinking
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "How are dogs and cats related?",
                "pattern": "lateral"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        match response {
            MCPResponse { content, is_error } => {
                assert!(!is_error);
                assert!(!content.is_empty());
                let result_text = &content[0].text;
                assert!(result_text.contains("pet") || result_text.contains("mammal") || result_text.contains("animal"));
            }
        }
    }

    #[tokio::test]
    async fn test_cognitive_reasoning_systems() {
        let server = setup_test_server().await;
        
        // Test systems thinking for hierarchical reasoning
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "What properties do dogs inherit from being mammals?",
                "pattern": "systems"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        match response {
            MCPResponse { content, is_error } => {
                assert!(!is_error);
                assert!(!content.is_empty());
                let result_text = &content[0].text;
                assert!(result_text.contains("warm-blooded") || result_text.contains("warm blooded"));
            }
        }
    }

    #[tokio::test]
    async fn test_cognitive_reasoning_critical() {
        let server = setup_test_server().await;
        
        // Add contradictory information
        let setup_request = MCPRequest {
            tool: "store_knowledge".to_string(),
            arguments: json!({
                "text": "Spike is a dog with five legs due to a rare genetic condition.",
                "use_graph_construction": true
            }),
        };
        let _ = server.handle_tool_call(setup_request).await.unwrap();
        
        // Test critical thinking
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "How many legs does Spike have?",
                "pattern": "critical"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        match response {
            MCPResponse { content, is_error } => {
                assert!(!is_error);
                assert!(!content.is_empty());
                let result_text = &content[0].text;
                // Should recognize the exception
                assert!(result_text.contains("five") || result_text.contains("5"));
            }
        }
    }

    #[tokio::test]
    async fn test_cognitive_reasoning_abstract() {
        let server = setup_test_server().await;
        
        // Add pattern data
        for i in 1..=5 {
            let request = MCPRequest {
                tool: "store_knowledge".to_string(),
                arguments: json!({
                    "text": format!("Bird{} is a bird that can fly.", i),
                    "use_graph_construction": true
                }),
            };
            let _ = server.handle_tool_call(request).await.unwrap();
        }
        
        // Test abstract pattern recognition
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "What patterns exist in the bird data?",
                "pattern": "abstract"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        match response {
            MCPResponse { content, is_error } => {
                assert!(!is_error);
                assert!(!content.is_empty());
                let result_text = &content[0].text;
                assert!(result_text.contains("pattern") || result_text.contains("fly"));
            }
        }
    }

    #[tokio::test]
    async fn test_cognitive_reasoning_adaptive() {
        let server = setup_test_server().await;
        
        // Test adaptive reasoning (automatic pattern selection)
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "Tell me about dogs",
                "pattern": "adaptive"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        match response {
            MCPResponse { content, is_error } => {
                assert!(!is_error);
                assert!(!content.is_empty());
                // Should provide a comprehensive answer using multiple patterns
                let result_text = &content[0].text;
                assert!(!result_text.is_empty());
            }
        }
    }

    #[tokio::test]
    async fn test_cognitive_reasoning_with_context() {
        let server = setup_test_server().await;
        
        // Test reasoning with context
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "How many legs?",
                "context": "We are discussing dogs",
                "pattern": "convergent"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        match response {
            MCPResponse { content, is_error } => {
                assert!(!is_error);
                assert!(!content.is_empty());
                let result_text = &content[0].text;
                assert!(result_text.contains("four") || result_text.contains("4"));
            }
        }
    }

    #[tokio::test]
    async fn test_neural_query() {
        let server = setup_test_server().await;
        
        // Test neural query functionality
        let request = MCPRequest {
            tool: "neural_query".to_string(),
            arguments: json!({
                "query": "dogs",
                "query_type": "semantic",
                "top_k": 5
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        match response {
            MCPResponse { content, is_error } => {
                assert!(!is_error);
                assert!(!content.is_empty());
                // Should return results related to dogs
                let result_text = &content[0].text;
                assert!(result_text.contains("dog") || result_text.contains("mammal"));
            }
        }
    }

    #[tokio::test]
    async fn test_invalid_cognitive_pattern() {
        let server = setup_test_server().await;
        
        // Test error handling for invalid pattern
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "What is a dog?",
                "pattern": "invalid_pattern"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        match response {
            MCPResponse { content, is_error } => {
                assert!(is_error);
                assert!(!content.is_empty());
                assert!(content[0].text.contains("Invalid cognitive pattern"));
            }
        }
    }

    #[tokio::test]
    async fn test_list_tools_includes_cognitive() {
        let server = setup_test_server().await;
        
        let tools = server.get_tools();
        assert!(!tools.is_empty());
        assert!(tools.iter().any(|t| t.name == "store_knowledge"));
        assert!(tools.iter().any(|t| t.name == "neural_query"));
        // Should include cognitive_reasoning if orchestrator is available
        assert!(tools.iter().any(|t| t.name == "cognitive_reasoning"));
    }
}