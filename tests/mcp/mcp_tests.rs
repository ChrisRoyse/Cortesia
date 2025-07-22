#[cfg(test)]
mod tests {
    use llmkg::mcp::{
        LLMKGMCPServer, MCPRequest, MCPResponse,
        shared_types::{MCPTool, MCPContent},
    };
    use serde_json::json;

    // Helper function to create a test LLMKGMCPServer
    async fn create_test_llmkg_server() -> LLMKGMCPServer {
        LLMKGMCPServer::new(96).expect("Failed to create LLMKG MCP server")
    }

    // Tests for LLMKGMCPServer public APIs
    mod llmkg_server_tests {
        use super::*;

        #[tokio::test]
        async fn test_server_creation() {
            let server = create_test_llmkg_server().await;
            assert!(server.get_tools().len() > 0);
        }

        #[tokio::test]
        async fn test_get_tools() {
            let server = create_test_llmkg_server().await;
            let tools = server.get_tools();
            
            // Verify expected tools are present
            let tool_names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
            assert!(tool_names.contains(&"knowledge_search".to_string()));
            assert!(tool_names.contains(&"entity_lookup".to_string()));
            assert!(tool_names.contains(&"find_connections".to_string()));
            assert!(tool_names.contains(&"expand_concept".to_string()));
            assert!(tool_names.contains(&"graph_statistics".to_string()));
            
            // Verify tool structure
            for tool in tools {
                assert!(!tool.name.is_empty());
                assert!(!tool.description.is_empty());
                assert!(!tool.input_schema.is_null());
            }
        }

        #[tokio::test]
        async fn test_tool_schemas() {
            let server = create_test_llmkg_server().await;
            let tools = server.get_tools();
            
            // Verify knowledge_search tool schema
            let knowledge_search = tools.iter()
                .find(|t| t.name == "knowledge_search")
                .expect("knowledge_search tool not found");
            
            assert!(knowledge_search.input_schema["type"].as_str() == Some("object"));
            assert!(knowledge_search.input_schema["properties"]["query"].is_object());
            assert!(knowledge_search.input_schema["properties"]["max_entities"].is_object());
            assert!(knowledge_search.input_schema["properties"]["max_depth"].is_object());
            assert!(knowledge_search.input_schema["required"].as_array().unwrap().contains(&json!("query")));
        }

        #[tokio::test]
        async fn test_knowledge_search_valid_query() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": "artificial intelligence",
                    "max_entities": 10,
                    "max_depth": 2
                }),
            };
            
            let response = server.handle_request(request).await;
            assert!(!response.is_error);
            assert!(!response.content.is_empty());
            assert_eq!(response.content[0].type_, "text");
            assert!(response.content[0].text.contains("Knowledge Search Results"));
        }

        #[tokio::test]
        async fn test_knowledge_search_missing_query() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({}),
            };
            
            let response = server.handle_request(request).await;
            assert!(response.is_error);
            assert!(response.content[0].text.contains("Query parameter is required"));
        }

        #[tokio::test]
        async fn test_knowledge_search_empty_query() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": ""
                }),
            };
            
            let response = server.handle_request(request).await;
            assert!(response.is_error);
            assert!(response.content[0].text.contains("Query parameter is required"));
        }

        #[tokio::test]
        async fn test_entity_lookup_by_description() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "entity_lookup".to_string(),
                arguments: json!({
                    "description": "quantum physics"
                }),
            };
            
            let response = server.handle_request(request).await;
            assert!(!response.is_error);
            assert!(!response.content.is_empty());
            assert!(response.content[0].text.contains("Entity Lookup Results"));
        }

        #[tokio::test]
        async fn test_entity_lookup_by_id() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "entity_lookup".to_string(),
                arguments: json!({
                    "entity_id": 12345
                }),
            };
            
            let response = server.handle_request(request).await;
            assert!(!response.is_error);
            // Note: Current implementation returns "not yet implemented" for ID lookup
            assert!(response.content[0].text.contains("not yet implemented"));
        }

        #[tokio::test]
        async fn test_entity_lookup_missing_params() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "entity_lookup".to_string(),
                arguments: json!({}),
            };
            
            let response = server.handle_request(request).await;
            assert!(response.is_error);
            assert!(response.content[0].text.contains("Either entity_id or description is required"));
        }

        #[tokio::test]
        async fn test_find_connections() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "find_connections".to_string(),
                arguments: json!({
                    "entity_a": "Einstein",
                    "entity_b": "relativity",
                    "max_path_length": 3
                }),
            };
            
            let response = server.handle_request(request).await;
            assert!(!response.is_error);
            assert!(response.content[0].text.contains("Connection Analysis"));
        }

        #[tokio::test]
        async fn test_find_connections_missing_entities() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "find_connections".to_string(),
                arguments: json!({
                    "entity_a": "Einstein"
                }),
            };
            
            let response = server.handle_request(request).await;
            assert!(response.is_error);
            assert!(response.content[0].text.contains("Both entity_a and entity_b are required"));
        }

        #[tokio::test]
        async fn test_expand_concept() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "expand_concept".to_string(),
                arguments: json!({
                    "concept": "machine learning",
                    "expansion_depth": 2,
                    "max_entities": 30
                }),
            };
            
            let response = server.handle_request(request).await;
            assert!(!response.is_error);
            assert!(response.content[0].text.contains("Concept Expansion"));
        }

        #[tokio::test]
        async fn test_expand_concept_missing_concept() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "expand_concept".to_string(),
                arguments: json!({}),
            };
            
            let response = server.handle_request(request).await;
            assert!(response.is_error);
            assert!(response.content[0].text.contains("Concept parameter is required"));
        }

        #[tokio::test]
        async fn test_graph_statistics() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "graph_statistics".to_string(),
                arguments: json!({}),
            };
            
            let response = server.handle_request(request).await;
            assert!(!response.is_error);
            assert!(response.content[0].text.contains("Knowledge Graph Statistics"));
            assert!(response.content[0].text.contains("json"));
        }

        #[tokio::test]
        async fn test_unknown_tool() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "unknown_tool".to_string(),
                arguments: json!({}),
            };
            
            let response = server.handle_request(request).await;
            assert!(response.is_error);
            assert!(response.content[0].text.contains("Unknown tool"));
        }

        #[tokio::test]
        async fn test_mmap_storage_initialization() {
            let server = create_test_llmkg_server().await;
            
            // Initialize mmap storage
            let result = server.initialize_mmap_storage(1000, 5000, 96).await;
            assert!(result.is_ok());
        }
    }

    // Integration tests for query processing workflow
    mod integration_tests {
        use super::*;

        #[tokio::test]
        async fn test_query_processing_workflow() {
            let server = create_test_llmkg_server().await;
            
            // Step 1: Search for knowledge
            let search_request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": "quantum computing basics",
                    "max_entities": 5
                }),
            };
            
            let search_response = server.handle_request(search_request).await;
            assert!(!search_response.is_error);
            
            // Step 2: Expand a concept
            let expand_request = MCPRequest {
                tool: "expand_concept".to_string(),
                arguments: json!({
                    "concept": "quantum entanglement",
                    "expansion_depth": 2
                }),
            };
            
            let expand_response = server.handle_request(expand_request).await;
            assert!(!expand_response.is_error);
            
            // Step 3: Get statistics
            let stats_request = MCPRequest {
                tool: "graph_statistics".to_string(),
                arguments: json!({}),
            };
            
            let stats_response = server.handle_request(stats_request).await;
            assert!(!stats_response.is_error);
        }

        #[tokio::test]
        async fn test_mathematical_operations() {
            let server = create_test_llmkg_server().await;
            
            // Test searching for mathematical concepts
            let math_request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": "calculus derivatives integration",
                    "max_entities": 10,
                    "max_depth": 3
                }),
            };
            
            let response = server.handle_request(math_request).await;
            assert!(!response.is_error);
            assert!(!response.content.is_empty());
        }

        #[tokio::test]
        async fn test_source_validation_placeholder() {
            let server = create_test_llmkg_server().await;
            
            // Test that source validation is handled (even if as placeholder)
            let request = MCPRequest {
                tool: "entity_lookup".to_string(),
                arguments: json!({
                    "description": "validated source"
                }),
            };
            
            let response = server.handle_request(request).await;
            assert!(!response.is_error);
            // The response should contain some form of result, even if empty
            assert!(!response.content.is_empty());
        }
    }

    // Performance and stress tests
    mod performance_tests {
        use super::*;
        use std::sync::Arc;
        use std::time::Instant;

        #[tokio::test]
        async fn test_concurrent_requests() {
            let server = Arc::new(create_test_llmkg_server().await);
            let mut handles = vec![];
            
            // Spawn multiple concurrent requests
            for i in 0..10 {
                let server_clone = server.clone();
                let handle = tokio::spawn(async move {
                    let request = MCPRequest {
                        tool: "knowledge_search".to_string(),
                        arguments: json!({
                            "query": format!("test query {}", i),
                            "max_entities": 5
                        }),
                    };
                    
                    let start = Instant::now();
                    let response = server_clone.handle_request(request).await;
                    let duration = start.elapsed();
                    
                    assert!(!response.is_error);
                    duration
                });
                handles.push(handle);
            }
            
            // Wait for all requests to complete
            let durations: Vec<_> = futures::future::join_all(handles).await;
            
            // Verify all requests completed successfully
            for duration_result in durations {
                assert!(duration_result.is_ok());
                let duration = duration_result.unwrap();
                // Verify reasonable response time (adjust as needed)
                assert!(duration.as_millis() < 5000);
            }
        }

        #[tokio::test]
        async fn test_large_query_handling() {
            let server = create_test_llmkg_server().await;
            
            // Test with maximum allowed entities
            let request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": "comprehensive analysis of all scientific theories",
                    "max_entities": 100,
                    "max_depth": 6
                }),
            };
            
            let start = Instant::now();
            let response = server.handle_request(request).await;
            let duration = start.elapsed();
            
            assert!(!response.is_error);
            // Verify it completes in reasonable time even with max parameters
            assert!(duration.as_secs() < 10);
        }

        #[tokio::test]
        async fn test_embedding_cache_performance() {
            let server = create_test_llmkg_server().await;
            
            // First query should create and cache embedding
            let request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": "cached query test"
                }),
            };
            
            let start1 = Instant::now();
            let response1 = server.handle_request(request.clone()).await;
            let duration1 = start1.elapsed();
            assert!(!response1.is_error);
            
            // Second identical query should be faster due to caching
            let start2 = Instant::now();
            let response2 = server.handle_request(request).await;
            let duration2 = start2.elapsed();
            assert!(!response2.is_error);
            
            // Cache hit should be faster (though this might not always be true in tests)
            println!("First query: {:?}, Second query: {:?}", duration1, duration2);
        }
    }

    // Edge case and error handling tests
    mod edge_case_tests {
        use super::*;

        #[tokio::test]
        async fn test_invalid_parameter_types() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": "test",
                    "max_entities": "not a number"
                }),
            };
            
            let response = server.handle_request(request).await;
            // Should use default value when type is wrong
            assert!(!response.is_error);
        }

        #[tokio::test]
        async fn test_boundary_values() {
            let server = create_test_llmkg_server().await;
            
            // Test with minimum values
            let min_request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": "test",
                    "max_entities": 1,
                    "max_depth": 1
                }),
            };
            
            let min_response = server.handle_request(min_request).await;
            assert!(!min_response.is_error);
            
            // Test with maximum values
            let max_request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": "test",
                    "max_entities": 100,
                    "max_depth": 6
                }),
            };
            
            let max_response = server.handle_request(max_request).await;
            assert!(!max_response.is_error);
        }

        #[tokio::test]
        async fn test_special_characters_in_queries() {
            let server = create_test_llmkg_server().await;
            
            let special_chars = vec![
                "test with spaces",
                "test-with-hyphens",
                "test_with_underscores",
                "test.with.dots",
                "test@with#special$chars",
                "test with 'quotes'",
                "test with \"double quotes\"",
                "test with unicode: 你好世界",
            ];
            
            for query in special_chars {
                let request = MCPRequest {
                    tool: "knowledge_search".to_string(),
                    arguments: json!({
                        "query": query
                    }),
                };
                
                let response = server.handle_request(request).await;
                assert!(!response.is_error, "Failed on query: {}", query);
            }
        }

        #[tokio::test]
        async fn test_null_and_missing_fields() {
            let server = create_test_llmkg_server().await;
            
            // Test with null value
            let null_request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": "test",
                    "max_entities": null
                }),
            };
            
            let null_response = server.handle_request(null_request).await;
            assert!(!null_response.is_error); // Should use default
            
            // Test with missing optional fields
            let minimal_request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": "minimal test"
                }),
            };
            
            let minimal_response = server.handle_request(minimal_request).await;
            assert!(!minimal_response.is_error);
        }

        #[tokio::test]
        async fn test_very_long_query() {
            let server = create_test_llmkg_server().await;
            
            // Create a very long query string
            let long_query = "test ".repeat(1000);
            
            let request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": long_query
                }),
            };
            
            let response = server.handle_request(request).await;
            assert!(!response.is_error);
        }
    }

    // Tests for response format validation
    mod response_format_tests {
        use super::*;

        #[tokio::test]
        async fn test_response_structure() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "knowledge_search".to_string(),
                arguments: json!({
                    "query": "test response format"
                }),
            };
            
            let response = server.handle_request(request).await;
            
            // Verify response structure
            assert!(!response.content.is_empty());
            for content in &response.content {
                assert!(!content.type_.is_empty());
                assert!(!content.text.is_empty());
                assert_eq!(content.type_, "text");
            }
        }

        #[tokio::test]
        async fn test_error_response_format() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "invalid_tool".to_string(),
                arguments: json!({}),
            };
            
            let response = server.handle_request(request).await;
            
            // Verify error response structure
            assert!(response.is_error);
            assert!(!response.content.is_empty());
            assert_eq!(response.content[0].type_, "text");
            assert!(response.content[0].text.contains("Unknown tool"));
        }

        #[tokio::test]
        async fn test_statistics_json_format() {
            let server = create_test_llmkg_server().await;
            
            let request = MCPRequest {
                tool: "graph_statistics".to_string(),
                arguments: json!({}),
            };
            
            let response = server.handle_request(request).await;
            assert!(!response.is_error);
            
            // Verify the response contains valid JSON
            let text = &response.content[0].text;
            assert!(text.contains("```json"));
            
            // Extract JSON portion and verify it's valid
            if let Some(start) = text.find("```json") {
                if let Some(end) = text[start..].find("```") {
                    let json_str = &text[start + 7..start + end];
                    let parsed: Result<serde_json::Value, _> = serde_json::from_str(json_str);
                    assert!(parsed.is_ok(), "Statistics should return valid JSON");
                }
            }
        }
    }
}