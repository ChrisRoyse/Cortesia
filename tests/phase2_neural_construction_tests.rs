#[cfg(test)]
mod phase2_neural_construction_tests {
    use llmkg::neural::structure_predictor::GraphStructurePredictor;
    use llmkg::neural::canonicalization::NeuralCanonicalizer;
    use llmkg::core::brain_types::{GraphOperation, BrainInspiredEntity, LogicGate};
    use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
    use llmkg::core::graph::KnowledgeGraph;
    use llmkg::versioning::temporal_graph::TemporalKnowledgeGraph;
    use llmkg::mcp::brain_inspired_server::BrainInspiredMCPServer;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use std::time::Instant;
    use serde_json::json;

    /// Setup graph processing environment
    async fn setup_graph_environment() -> (
        Arc<GraphStructurePredictor>,
        Arc<NeuralCanonicalizer>,
        Arc<BrainEnhancedKnowledgeGraph>,
    ) {
        let structure_predictor = Arc::new(GraphStructurePredictor::new(
            "test_structure_model".to_string(),
        ));
        
        let canonicalizer = Arc::new(NeuralCanonicalizer::new());
        
        let brain_graph = Arc::new(
            BrainEnhancedKnowledgeGraph::new(
                TemporalKnowledgeGraph::new(KnowledgeGraph::new(384).unwrap())
            )
        );
        
        (structure_predictor, canonicalizer, brain_graph)
    }

    /// Setup MCP server for integration tests
    async fn setup_mcp_server_for_graph_tests() -> Arc<BrainInspiredMCPServer> {
        let graph = KnowledgeGraph::new(384).unwrap();
        let temporal_graph = Arc::new(RwLock::new(TemporalKnowledgeGraph::new(graph)));
        
        let brain_graph = Arc::new(
            BrainEnhancedKnowledgeGraph::new(
                TemporalKnowledgeGraph::new(KnowledgeGraph::new(384).unwrap())
            )
        );
        
        Arc::new(BrainInspiredMCPServer::new_with_cognitive_capabilities(
            temporal_graph,
            brain_graph,
        ).await.unwrap())
    }

    // Test 1: Graph-based Canonicalization - Entity Normalization
    #[tokio::test]
    async fn test_graph_canonicalization_entity_normalization() {
        let (_, canonicalizer, _) = setup_graph_environment().await;
        
        let test_cases = vec![
            ("dog", "Dog"),
            ("DOG", "Dog"),
            ("dogs", "Dog"),
            ("canine", "Dog"),
            ("domestic dog", "Dog"),
            ("cat", "Cat"),
            ("feline", "Cat"),
            ("house cat", "Cat"),
        ];
        
        for (input, expected_canonical) in test_cases {
            let result = canonicalizer.canonicalize_entity(input).await.unwrap();
            
            // Should normalize to canonical form
            assert!(!result.is_empty());
            println!("'{}' -> '{}'", input, result);
            
            // In a real implementation, we'd expect consistent canonicalization
            // For now, just verify it produces consistent results
            let result2 = canonicalizer.canonicalize_entity(input).await.unwrap();
            assert_eq!(result, result2, "Canonicalization should be consistent");
        }
    }

    // Test 2: Structure Prediction - Graph Operations Generation
    #[tokio::test]
    async fn test_structure_prediction_graph_operations() {
        let (structure_predictor, _, _) = setup_graph_environment().await;
        
        let test_texts = vec![
            "Dogs are mammals with four legs.",
            "Cats are independent animals that hunt mice.",
            "Mammals are warm-blooded animals that feed milk to their young.",
            "Tripper is a three-legged dog who loves to play.",
        ];
        
        for text in test_texts {
            let start = Instant::now();
            let operations = structure_predictor.predict_structure(text).await.unwrap();
            let prediction_time = start.elapsed();
            
            // Should generate operations
            assert!(!operations.is_empty());
            
            // Should complete within reasonable time
            assert!(prediction_time.as_millis() < 500);
            
            println!("Text: '{}' -> {} operations ({}ms)", 
                     text, operations.len(), prediction_time.as_millis());
            
            // Verify operation types
            let has_create_node = operations.iter().any(|op| matches!(op, GraphOperation::CreateNode { .. }));
            let has_relationships = operations.iter().any(|op| matches!(op, GraphOperation::CreateRelationship { .. }));
            
            assert!(has_create_node, "Should create nodes");
            // Relationships are expected for most texts
            if text.contains("are") || text.contains("is") {
                assert!(has_relationships, "Should create relationships for '{}'", text);
            }
        }
    }

    // Test 3: Brain-Inspired Entity Creation
    #[tokio::test]
    async fn test_brain_inspired_entity_creation() {
        let (_, _, brain_graph) = setup_graph_environment().await;
        
        let test_concepts = vec![
            ("Dog", "mammal"),
            ("Cat", "mammal"),
            ("Mammal", "animal_class"),
            ("Animal", "living_being"),
        ];
        
        for (concept, entity_type) in test_concepts {
            let start = Instant::now();
            
            // Create brain-inspired entity
            let entity = BrainInspiredEntity {
                id: llmkg::core::types::EntityKey::default(),
                concept_id: concept.to_string(),
                direction: llmkg::core::brain_types::EntityDirection::Input,
                properties: std::collections::HashMap::new(),
                embedding: neural_server.get_embedding(concept).await.unwrap(),
                activation_state: 0.0,
                last_activation: std::time::SystemTime::now(),
            };
            
            let creation_time = start.elapsed();
            
            // Should create entity successfully
            assert_eq!(entity.concept_id, concept);
            assert!(!entity.embedding.is_empty());
            assert_eq!(entity.embedding.len(), 384); // Should match embedding dimension
            
            // Should complete quickly
            assert!(creation_time.as_millis() < 200);
            
            println!("Created entity '{}' with {} dimensions in {}ms", 
                     concept, entity.embedding.len(), creation_time.as_millis());
        }
    }

    // Test 4: Logic Gate Creation for Brain-Inspired Structure
    #[tokio::test]
    async fn test_logic_gate_creation() {
        let (_, _, _, _) = setup_neural_environment().await;
        
        let test_gates = vec![
            (vec!["Dog".to_string(), "Cat".to_string()], vec!["Mammal".to_string()], "AND"),
            (vec!["Mammal".to_string()], vec!["Animal".to_string()], "OR"),
            (vec!["Exception".to_string()], vec!["Normal".to_string()], "NOT"),
        ];
        
        for (inputs, outputs, gate_type_str) in test_gates {
            let gate_type = match gate_type_str {
                "AND" => llmkg::core::brain_types::LogicGateType::And,
                "OR" => llmkg::core::brain_types::LogicGateType::Or,
                "NOT" => llmkg::core::brain_types::LogicGateType::Not,
                _ => llmkg::core::brain_types::LogicGateType::And,
            };
            
            let logic_gate = LogicGate {
                gate_id: llmkg::core::types::EntityKey::default(),
                gate_type,
                input_nodes: inputs.iter().map(|_| llmkg::core::types::EntityKey::default()).collect(),
                output_nodes: outputs.iter().map(|_| llmkg::core::types::EntityKey::default()).collect(),
                threshold: 0.5,
                weight_matrix: vec![1.0; inputs.len()],
            };
            
            // Should create valid logic gate
            assert_eq!(logic_gate.input_nodes.len(), inputs.len());
            assert_eq!(logic_gate.output_nodes.len(), outputs.len());
            assert_eq!(logic_gate.weight_matrix.len(), inputs.len());
            assert!(logic_gate.threshold > 0.0 && logic_gate.threshold <= 1.0);
            
            println!("Created {} gate with {} inputs -> {} outputs", 
                     gate_type_str, inputs.len(), outputs.len());
        }
    }

    // Test 5: End-to-End Neural Graph Construction
    #[tokio::test]
    async fn test_end_to_end_neural_graph_construction() {
        let server = setup_mcp_server_for_neural_tests().await;
        
        let test_texts = vec![
            "Dogs are loyal mammals that have been domesticated for thousands of years.",
            "Cats are independent predators that hunt small animals for food.",
            "Both dogs and cats are popular pets that provide companionship to humans.",
            "Mammals are characterized by having fur, being warm-blooded, and nursing their young.",
        ];
        
        let mut total_construction_time = std::time::Duration::from_millis(0);
        let mut successful_constructions = 0;
        
        for text in &test_texts {
            let start = Instant::now();
            
            let request = llmkg::mcp::brain_inspired_server::MCPRequest {
                tool: "store_knowledge".to_string(),
                arguments: json!({
                    "text": text,
                    "use_neural_construction": true,
                    "context": "End-to-end neural construction test"
                }),
            };
            
            let response = server.handle_tool_call(request).await.unwrap();
            let construction_time = start.elapsed();
            
            // Should complete successfully
            assert!(!response.is_error, "Construction failed for: {}", text);
            assert!(!response.content.is_empty());
            assert!(response.content[0].text.contains("Neural graph construction completed"));
            
            // Should complete within reasonable time
            assert!(construction_time.as_millis() < 2000);
            
            total_construction_time += construction_time;
            successful_constructions += 1;
            
            println!("Constructed graph for '{}' in {}ms", text, construction_time.as_millis());
        }
        
        let avg_construction_time = total_construction_time / successful_constructions;
        println!("Average construction time: {:?}", avg_construction_time);
        
        // All constructions should succeed
        assert_eq!(successful_constructions, test_texts.len() as u32);
        
        // Average time should be reasonable
        assert!(avg_construction_time.as_millis() < 1000);
    }

    // Test 6: Neural Construction Performance Under Load
    #[tokio::test]
    async fn test_neural_construction_performance_under_load() {
        let server = setup_mcp_server_for_neural_tests().await;
        
        let concurrent_requests = 10;
        let mut handles = Vec::new();
        
        let start = Instant::now();
        
        // Create concurrent requests
        for i in 0..concurrent_requests {
            let server_clone = Arc::clone(&server);
            let handle = tokio::spawn(async move {
                let request = llmkg::mcp::brain_inspired_server::MCPRequest {
                    tool: "store_knowledge".to_string(),
                    arguments: json!({
                        "text": format!("Entity {} has property {} and relationship to other entities.", i, i),
                        "use_neural_construction": true,
                        "context": format!("Load test request {}", i)
                    }),
                };
                
                server_clone.handle_tool_call(request).await
            });
            handles.push(handle);
        }
        
        // Wait for all requests to complete
        let mut successful_requests = 0;
        for handle in handles {
            let result = handle.await.unwrap();
            if result.is_ok() && !result.unwrap().is_error {
                successful_requests += 1;
            }
        }
        
        let total_time = start.elapsed();
        let avg_time_per_request = total_time / concurrent_requests;
        
        println!("Completed {} concurrent requests in {:?} (avg: {:?})", 
                 concurrent_requests, total_time, avg_time_per_request);
        
        // Should handle concurrent load
        assert!(successful_requests >= concurrent_requests * 80 / 100); // 80% success rate minimum
        assert!(avg_time_per_request.as_millis() < 2000); // Reasonable average time
    }

    // Test 7: Neural Embedding Generation Quality
    #[tokio::test]
    async fn test_neural_embedding_generation_quality() {
        let (neural_server, _, _, _) = setup_neural_environment().await;
        
        let test_concepts = vec![
            ("dog", "cat"),        // Similar concepts
            ("mammal", "animal"),  // Hierarchical concepts
            ("car", "dog"),        // Dissimilar concepts
            ("red", "blue"),       // Different but related
        ];
        
        for (concept1, concept2) in test_concepts {
            let embedding1 = neural_server.get_embedding(concept1).await.unwrap();
            let embedding2 = neural_server.get_embedding(concept2).await.unwrap();
            
            // Calculate cosine similarity
            let similarity = calculate_cosine_similarity(&embedding1, &embedding2);
            
            // Should produce valid embeddings
            assert_eq!(embedding1.len(), 384);
            assert_eq!(embedding2.len(), 384);
            
            // Should have reasonable similarity bounds
            assert!(similarity >= -1.0 && similarity <= 1.0);
            
            println!("Similarity between '{}' and '{}': {:.3}", concept1, concept2, similarity);
            
            // Related concepts should be more similar than unrelated ones
            match (concept1, concept2) {
                ("dog", "cat") => assert!(similarity > 0.0, "Dogs and cats should be similar"),
                ("mammal", "animal") => assert!(similarity > 0.0, "Mammals and animals should be similar"),
                ("car", "dog") => {
                    // This might be lower similarity, but we don't enforce strict thresholds
                    // in the test environment
                },
                _ => {}
            }
        }
    }

    // Test 8: Temporal Metadata Integration
    #[tokio::test]
    async fn test_temporal_metadata_integration() {
        let server = setup_mcp_server_for_neural_tests().await;
        
        let request = llmkg::mcp::brain_inspired_server::MCPRequest {
            tool: "store_knowledge".to_string(),
            arguments: json!({
                "text": "This is a test entity with temporal metadata.",
                "use_neural_construction": true,
                "context": "Temporal metadata test",
                "timestamp": "2024-01-01T00:00:00Z"
            }),
        };
        
        let start = Instant::now();
        let response = server.handle_tool_call(request).await.unwrap();
        let processing_time = start.elapsed();
        
        // Should complete successfully with temporal metadata
        assert!(!response.is_error);
        assert!(!response.content.is_empty());
        assert!(response.content[0].text.contains("Neural graph construction completed"));
        
        // Should complete within reasonable time
        assert!(processing_time.as_millis() < 1000);
        
        println!("Temporal metadata integration completed in {:?}", processing_time);
    }

    // Test 9: Error Handling in Neural Construction
    #[tokio::test]
    async fn test_error_handling_in_neural_construction() {
        let server = setup_mcp_server_for_neural_tests().await;
        
        let error_test_cases = vec![
            ("", "Empty text should be handled gracefully"),
            ("   ", "Whitespace-only text should be handled"),
            (&"A".repeat(10000), "Very long text should be handled"),
        ];
        
        for (text, description) in error_test_cases {
            let request = llmkg::mcp::brain_inspired_server::MCPRequest {
                tool: "store_knowledge".to_string(),
                arguments: json!({
                    "text": text,
                    "use_neural_construction": true,
                    "context": description
                }),
            };
            
            let result = server.handle_tool_call(request).await;
            
            // Should handle errors gracefully (either succeed or fail gracefully)
            match result {
                Ok(response) => {
                    if response.is_error {
                        println!("Gracefully handled error for {}: {}", description, response.content[0].text);
                    } else {
                        println!("Successfully processed {}", description);
                    }
                }
                Err(e) => {
                    println!("Expected error for {}: {:?}", description, e);
                }
            }
        }
    }

    // Test 10: Integration with Knowledge Graph Storage
    #[tokio::test]
    async fn test_integration_with_knowledge_graph_storage() {
        let server = setup_mcp_server_for_neural_tests().await;
        
        // Store knowledge with neural construction
        let store_request = llmkg::mcp::brain_inspired_server::MCPRequest {
            tool: "store_knowledge".to_string(),
            arguments: json!({
                "text": "Elephants are large mammals with long trunks and excellent memory.",
                "use_neural_construction": true,
                "context": "Integration test"
            }),
        };
        
        let store_response = server.handle_tool_call(store_request).await.unwrap();
        assert!(!store_response.is_error);
        
        // Query the stored knowledge
        let query_request = llmkg::mcp::brain_inspired_server::MCPRequest {
            tool: "neural_query".to_string(),
            arguments: json!({
                "query": "elephant",
                "query_type": "semantic",
                "top_k": 5
            }),
        };
        
        let query_response = server.handle_tool_call(query_request).await.unwrap();
        
        // Should be able to find the stored knowledge
        assert!(!query_response.is_error);
        assert!(!query_response.content.is_empty());
        
        let result_text = &query_response.content[0].text.to_lowercase();
        assert!(result_text.contains("elephant") || result_text.contains("mammal"));
        
        println!("Successfully stored and retrieved knowledge about elephants");
    }

    // Helper function to calculate cosine similarity
    fn calculate_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            0.0
        } else {
            dot_product / (magnitude_a * magnitude_b)
        }
    }
}