#[cfg(test)]
mod phase2_mcp_integration_tests {
    use llmkg::mcp::brain_inspired_server::{BrainInspiredMCPServer, MCPRequest, MCPResponse};
    use llmkg::versioning::temporal_graph::TemporalKnowledgeGraph;
    use llmkg::core::graph::KnowledgeGraph;
    use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use serde_json::json;
    use std::time::Instant;

    /// Setup MCP server with cognitive capabilities
    async fn setup_mcp_server_with_cognitive_capabilities() -> BrainInspiredMCPServer {
        let graph = KnowledgeGraph::new(384).unwrap();
        let temporal_graph = Arc::new(RwLock::new(TemporalKnowledgeGraph::new(graph)));
        
        // Create brain enhanced graph
        let brain_graph = Arc::new(
            BrainEnhancedKnowledgeGraph::new(
                TemporalKnowledgeGraph::new(KnowledgeGraph::new(384).unwrap())
            )
        );
        
        // Create server with cognitive capabilities
        let server = BrainInspiredMCPServer::new_with_cognitive_capabilities(
            temporal_graph,
            brain_graph,
        ).await.unwrap();
        
        // Populate with test data
        populate_test_data(&server).await;
        
        server
    }

    /// Populate server with comprehensive test data
    async fn populate_test_data(server: &BrainInspiredMCPServer) {
        // Add hierarchical knowledge
        let knowledge_requests = vec![
            json!({
                "text": "Animals are living organisms that can move and respond to their environment.",
                "use_graph_construction": true,
                "context": "Biological classification foundation"
            }),
            json!({
                "text": "Mammals are warm-blooded animals with fur or hair that feed milk to their babies.",
                "use_graph_construction": true,
                "context": "Mammal definition"
            }),
            json!({
                "text": "Dogs are domesticated mammals with four legs, known for loyalty and companionship.",
                "use_graph_construction": true,
                "context": "Dog characteristics"
            }),
            json!({
                "text": "Cats are independent mammals with four legs, known for hunting and agility.",
                "use_graph_construction": true,
                "context": "Cat characteristics"
            }),
            json!({
                "text": "Tripper is a special three-legged dog who is energetic and playful despite his condition.",
                "use_graph_construction": true,
                "context": "Individual dog with exception"
            }),
            json!({
                "text": "Artificial Intelligence involves creating systems that can perform tasks requiring human intelligence.",
                "use_graph_construction": true,
                "context": "AI definition"
            }),
            json!({
                "text": "Art is creative expression that involves imagination, skill, and aesthetic beauty.",
                "use_graph_construction": true,
                "context": "Art definition"
            }),
            json!({
                "text": "Creativity bridges technology and human expression, enabling new forms of artistic innovation.",
                "use_graph_construction": true,
                "context": "Creativity bridge"
            }),
        ];
        
        for knowledge in knowledge_requests {
            let request = MCPRequest {
                tool: "store_knowledge".to_string(),
                arguments: knowledge,
            };
            
            let _ = server.handle_tool_call(request).await;
        }
        
        // Allow some time for processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    // Test 1: MCP Tool Registration with Cognitive Capabilities
    #[tokio::test]
    async fn test_mcp_tool_registration_with_cognitive() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let tools = server.get_tools();
        
        // Should have basic tools
        assert!(tools.iter().any(|t| t.name == "store_knowledge"));
        assert!(tools.iter().any(|t| t.name == "neural_query"));
        
        // Should have cognitive reasoning tool
        assert!(tools.iter().any(|t| t.name == "cognitive_reasoning"));
        
        // Verify tool descriptions
        let cognitive_tool = tools.iter().find(|t| t.name == "cognitive_reasoning").unwrap();
        assert!(cognitive_tool.description.contains("cognitive"));
        assert!(cognitive_tool.description.contains("reasoning"));
        
        println!("Registered {} MCP tools including cognitive reasoning", tools.len());
    }

    // Test 2: Neural-Powered Graph Construction via MCP
    #[tokio::test]
    async fn test_neural_graph_construction_via_mcp() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let start = Instant::now();
        
        let request = MCPRequest {
            tool: "store_knowledge".to_string(),
            arguments: json!({
                "text": "Eagles are large birds of prey with excellent vision and powerful talons.",
                "use_graph_construction": true,
                "context": "Bird characteristics"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        let construction_time = start.elapsed();
        
        // Should complete successfully
        assert!(!response.is_error);
        assert!(!response.content.is_empty());
        assert!(response.content[0].text.contains("Neural graph construction completed"));
        
        // Should complete within reasonable time
        assert!(construction_time.as_millis() < 1000);
        
        println!("Neural graph construction time: {:?}", construction_time);
    }

    // Test 3: Cognitive Reasoning via MCP - Convergent Pattern
    #[tokio::test]
    async fn test_mcp_cognitive_reasoning_convergent() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "What type of animal is a dog?",
                "pattern": "convergent",
                "context": "Animal classification query"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        // Should provide focused answer
        assert!(!response.is_error);
        assert!(!response.content.is_empty());
        
        let result_text = &response.content[0].text.to_lowercase();
        assert!(result_text.contains("mammal") || result_text.contains("dog"));
        
        println!("Convergent reasoning result: {}", response.content[0].text);
    }

    // Test 4: Cognitive Reasoning via MCP - Divergent Pattern
    #[tokio::test]
    async fn test_mcp_cognitive_reasoning_divergent() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "What are examples of mammals?",
                "pattern": "divergent",
                "context": "Looking for multiple examples"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        // Should provide multiple examples
        assert!(!response.is_error);
        assert!(!response.content.is_empty());
        
        let result_text = &response.content[0].text.to_lowercase();
        // Should contain multiple examples
        let contains_dog = result_text.contains("dog");
        let contains_cat = result_text.contains("cat");
        
        assert!(contains_dog || contains_cat);
        
        println!("Divergent reasoning result: {}", response.content[0].text);
    }

    // Test 5: Cognitive Reasoning via MCP - Lateral Pattern
    #[tokio::test]
    async fn test_mcp_cognitive_reasoning_lateral() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "How are AI and art connected?",
                "pattern": "lateral",
                "context": "Creative connection exploration"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        // Should find creative connections
        assert!(!response.is_error);
        assert!(!response.content.is_empty());
        
        let result_text = &response.content[0].text.to_lowercase();
        assert!(result_text.contains("creativity") || result_text.contains("creative") || 
                result_text.contains("innovation") || result_text.contains("expression"));
        
        println!("Lateral reasoning result: {}", response.content[0].text);
    }

    // Test 6: Cognitive Reasoning via MCP - Systems Pattern
    #[tokio::test]
    async fn test_mcp_cognitive_reasoning_systems() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "What properties do dogs inherit from being mammals?",
                "pattern": "systems",
                "context": "Hierarchical property inheritance"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        // Should show inheritance
        assert!(!response.is_error);
        assert!(!response.content.is_empty());
        
        let result_text = &response.content[0].text.to_lowercase();
        assert!(result_text.contains("warm") || result_text.contains("fur") || 
                result_text.contains("milk") || result_text.contains("mammal"));
        
        println!("Systems reasoning result: {}", response.content[0].text);
    }

    // Test 7: Cognitive Reasoning via MCP - Critical Pattern
    #[tokio::test]
    async fn test_mcp_cognitive_reasoning_critical() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "Tripper has 3 legs but dogs normally have 4 legs. How do we resolve this?",
                "pattern": "critical",
                "context": "Contradiction resolution"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        // Should handle contradiction
        assert!(!response.is_error);
        assert!(!response.content.is_empty());
        
        let result_text = &response.content[0].text.to_lowercase();
        assert!(result_text.contains("exception") || result_text.contains("three") || 
                result_text.contains("3") || result_text.contains("special"));
        
        println!("Critical reasoning result: {}", response.content[0].text);
    }

    // Test 8: Cognitive Reasoning via MCP - Abstract Pattern
    #[tokio::test]
    async fn test_mcp_cognitive_reasoning_abstract() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "What patterns exist in animal classification?",
                "pattern": "abstract",
                "context": "Pattern identification"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        // Should identify patterns
        assert!(!response.is_error);
        assert!(!response.content.is_empty());
        
        let result_text = &response.content[0].text.to_lowercase();
        assert!(result_text.contains("pattern") || result_text.contains("hierarchy") || 
                result_text.contains("classification") || result_text.contains("structure"));
        
        println!("Abstract reasoning result: {}", response.content[0].text);
    }

    // Test 9: Cognitive Reasoning via MCP - Adaptive Pattern
    #[tokio::test]
    async fn test_mcp_cognitive_reasoning_adaptive() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "Tell me about dogs comprehensively",
                "pattern": "adaptive",
                "context": "Comprehensive analysis"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        // Should provide comprehensive answer
        assert!(!response.is_error);
        assert!(!response.content.is_empty());
        
        let result_text = &response.content[0].text;
        assert!(result_text.len() > 50); // Should be detailed
        
        println!("Adaptive reasoning result: {}", response.content[0].text);
    }

    // Test 10: Neural Query with Cognitive Pattern Integration
    #[tokio::test]
    async fn test_neural_query_with_cognitive_pattern() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let request = MCPRequest {
            tool: "neural_query".to_string(),
            arguments: json!({
                "query": "What animals are similar to dogs?",
                "query_type": "pattern",
                "top_k": 5,
                "use_cognitive_enhancement": true
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        // Should provide enhanced results
        assert!(!response.is_error);
        assert!(!response.content.is_empty());
        
        let result_text = &response.content[0].text.to_lowercase();
        assert!(result_text.contains("cat") || result_text.contains("mammal") || 
                result_text.contains("animal"));
        
        println!("Neural query with cognitive pattern result: {}", response.content[0].text);
    }

    // Test 11: MCP Performance - Latency Requirements
    #[tokio::test]
    async fn test_mcp_performance_latency() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        // Test simple cognitive reasoning latency
        let start = Instant::now();
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "What is a dog?",
                "pattern": "convergent"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        let latency = start.elapsed();
        
        // Should complete within reasonable time
        assert!(latency.as_millis() < 1000);
        assert!(!response.is_error);
        
        println!("MCP cognitive reasoning latency: {:?}", latency);
    }

    // Test 12: MCP Error Handling - Invalid Pattern
    #[tokio::test]
    async fn test_mcp_error_handling_invalid_pattern() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "What is a dog?",
                "pattern": "invalid_pattern"
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        // Should handle error gracefully
        assert!(response.is_error);
        assert!(!response.content.is_empty());
        assert!(response.content[0].text.contains("Invalid") || 
                response.content[0].text.contains("invalid"));
        
        println!("Error handling response: {}", response.content[0].text);
    }

    // Test 13: MCP Error Handling - Missing Query
    #[tokio::test]
    async fn test_mcp_error_handling_missing_query() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let request = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "pattern": "convergent"
                // Missing query parameter
            }),
        };
        
        let response = server.handle_tool_call(request).await.unwrap();
        
        // Should handle missing parameter gracefully
        assert!(response.is_error);
        assert!(!response.content.is_empty());
        assert!(response.content[0].text.contains("Missing") || 
                response.content[0].text.contains("required"));
        
        println!("Missing query error response: {}", response.content[0].text);
    }

    // Test 14: MCP Integration - Automatic Pattern Selection
    #[tokio::test]
    async fn test_mcp_automatic_pattern_selection() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let test_cases = vec![
            ("What is a dog?", "convergent"),
            ("What are examples of mammals?", "divergent"),
            ("How are AI and art connected?", "lateral"),
            ("What properties do dogs inherit?", "systems"),
        ];
        
        for (query, expected_pattern_hint) in test_cases {
            let request = MCPRequest {
                tool: "cognitive_reasoning".to_string(),
                arguments: json!({
                    "query": query,
                    "pattern": "automatic" // Let system choose
                }),
            };
            
            let response = server.handle_tool_call(request).await.unwrap();
            
            // Should complete successfully
            assert!(!response.is_error);
            assert!(!response.content.is_empty());
            
            println!("Query: '{}' -> Pattern hint: '{}' -> Result: '{}'", 
                     query, expected_pattern_hint, response.content[0].text);
        }
    }

    // Test 15: MCP Integration - Context-Aware Reasoning
    #[tokio::test]
    async fn test_mcp_context_aware_reasoning() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        // Test with context
        let request_with_context = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "How many legs?",
                "pattern": "convergent",
                "context": "We are discussing dogs in general"
            }),
        };
        
        let response_with_context = server.handle_tool_call(request_with_context).await.unwrap();
        
        // Test without context
        let request_without_context = MCPRequest {
            tool: "cognitive_reasoning".to_string(),
            arguments: json!({
                "query": "How many legs?",
                "pattern": "convergent"
            }),
        };
        
        let response_without_context = server.handle_tool_call(request_without_context).await.unwrap();
        
        // Both should work, but context should improve results
        assert!(!response_with_context.is_error);
        assert!(!response_without_context.is_error);
        
        let context_result = &response_with_context.content[0].text.to_lowercase();
        assert!(context_result.contains("four") || context_result.contains("4"));
        
        println!("With context: {}", response_with_context.content[0].text);
        println!("Without context: {}", response_without_context.content[0].text);
    }

    // Test 16: MCP Integration - Batch Processing
    #[tokio::test]
    async fn test_mcp_batch_processing() {
        let server = setup_mcp_server_with_cognitive_capabilities().await;
        
        let queries = vec![
            "What is a dog?",
            "What is a cat?",
            "What are mammals?",
            "How are dogs and cats similar?",
            "What makes mammals unique?",
        ];
        
        let mut responses = Vec::new();
        
        for query in queries {
            let request = MCPRequest {
                tool: "cognitive_reasoning".to_string(),
                arguments: json!({
                    "query": query,
                    "pattern": "automatic"
                }),
            };
            
            let response = server.handle_tool_call(request).await.unwrap();
            responses.push(response);
        }
        
        // All should complete successfully
        for (i, response) in responses.iter().enumerate() {
            assert!(!response.is_error, "Query {} failed: {}", i, response.content[0].text);
            assert!(!response.content.is_empty());
        }
        
        println!("Successfully processed {} queries in batch", responses.len());
    }
}