// WebAssembly Integration Tests
// Tests LLMKG functionality in WebAssembly environment

#[cfg(target_arch = "wasm32")]
mod wasm_integration {
    use wasm_bindgen_test::*;
    use wasm_bindgen::JsValue;
    use serde_json::{json, Value};
    use web_sys::console;
    
    use llmkg_wasm::{
        WasmKnowledgeGraph, 
        WasmEmbeddingStore, 
        WasmPerformanceMonitor,
        WasmError
    };

    wasm_bindgen_test_configure!(run_in_browser);

    /// Helper to log to browser console
    fn log(message: &str) {
        console::log_1(&JsValue::from_str(message));
    }

    #[wasm_bindgen_test]
    fn test_wasm_knowledge_graph_basic_operations() {
        log("Starting WASM basic operations test");
        
        let mut kg = WasmKnowledgeGraph::new();
        
        // Test entity addition
        let entity_data = json!({
            "key": "test_entity_1",
            "name": "Test Entity 1",
            "attributes": {
                "type": "test",
                "value": "42"
            }
        });
        
        let add_result = kg.add_entity_from_json(&entity_data.to_string());
        assert!(add_result.is_ok(), "Failed to add entity: {:?}", add_result.err());
        
        // Test entity retrieval
        let retrieved = kg.get_entity_json("test_entity_1");
        assert!(retrieved.is_ok(), "Failed to retrieve entity: {:?}", retrieved.err());
        
        let retrieved_data: Value = serde_json::from_str(&retrieved.unwrap()).unwrap();
        assert_eq!(retrieved_data["name"], "Test Entity 1");
        assert_eq!(retrieved_data["attributes"]["type"], "test");
        
        // Test relationship addition
        kg.add_entity_from_json(&json!({
            "key": "test_entity_2", 
            "name": "Test Entity 2",
            "attributes": {}
        }).to_string()).unwrap();
        
        let rel_result = kg.add_relationship_json(
            "test_entity_1",
            "test_entity_2", 
            &json!({
                "name": "connects",
                "weight": 1.0,
                "type": "directed"
            }).to_string()
        );
        assert!(rel_result.is_ok(), "Failed to add relationship: {:?}", rel_result.err());
        
        // Test graph statistics
        let stats = kg.get_statistics_json();
        let stats_data: Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(stats_data["entity_count"], 2);
        assert_eq!(stats_data["relationship_count"], 1);
        
        // Test neighbor query
        let neighbors_result = kg.get_neighbors_json("test_entity_1");
        assert!(neighbors_result.is_ok());
        
        let neighbors: Value = serde_json::from_str(&neighbors_result.unwrap()).unwrap();
        assert!(neighbors.is_array());
        assert_eq!(neighbors.as_array().unwrap().len(), 1);
        assert_eq!(neighbors[0]["target"], "test_entity_2");
        
        log("WASM basic operations test completed successfully");
    }
    
    #[wasm_bindgen_test]
    fn test_wasm_embedding_similarity_search() {
        log("Starting WASM embedding similarity search test");
        
        let mut embedding_store = WasmEmbeddingStore::new(64);
        
        // Add test embeddings
        let test_entities = vec![
            ("entity_1", vec![1.0, 0.0, 0.0]),
            ("entity_2", vec![0.0, 1.0, 0.0]),
            ("entity_3", vec![0.0, 0.0, 1.0]),
            ("entity_4", vec![0.7, 0.7, 0.0]), // Similar to entity_1 and entity_2
        ];
        
        for (entity_id, embedding_base) in test_entities {
            let mut embedding = embedding_base;
            embedding.resize(64, 0.0); // Pad to 64 dimensions
            
            let result = embedding_store.add_embedding_from_array(entity_id, &embedding);
            assert!(result.is_ok(), "Failed to add embedding for {}: {:?}", 
                   entity_id, result.err());
        }
        
        // Test similarity search
        let mut query_embedding = vec![1.0, 0.1, 0.0]; // Close to entity_1
        query_embedding.resize(64, 0.0);
        
        let results = embedding_store.similarity_search_from_array(&query_embedding, 3);
        assert!(results.is_ok(), "Similarity search failed: {:?}", results.err());
        
        let results_json: Value = serde_json::from_str(&results.unwrap()).unwrap();
        let results_array = results_json.as_array().unwrap();
        
        assert_eq!(results_array.len(), 3);
        
        // First result should be entity_1 (most similar)
        assert_eq!(results_array[0]["entity"], "entity_1");
        
        // Verify distances are sorted
        let dist1 = results_array[0]["distance"].as_f64().unwrap();
        let dist2 = results_array[1]["distance"].as_f64().unwrap();
        assert!(dist1 <= dist2, "Results not sorted by distance");
        
        // Second result should likely be entity_4 (mixed similarity)
        assert_eq!(results_array[1]["entity"], "entity_4");
        
        log("WASM embedding similarity search test completed successfully");
    }
    
    #[wasm_bindgen_test]
    fn test_wasm_performance_measurement() {
        log("Starting WASM performance measurement test");
        
        let mut kg = WasmKnowledgeGraph::new();
        let mut perf_monitor = WasmPerformanceMonitor::new();
        
        // Add entities with performance monitoring
        perf_monitor.start_measurement("entity_addition");
        
        for i in 0..1000 {
            let entity_data = json!({
                "key": format!("entity_{}", i),
                "name": format!("Entity {}", i),
                "attributes": {"index": i.to_string()}
            });
            
            kg.add_entity_from_json(&entity_data.to_string()).unwrap();
        }
        
        let addition_time = perf_monitor.end_measurement("entity_addition");
        assert!(addition_time.is_ok());
        
        let time_ms = addition_time.unwrap();
        assert!(time_ms > 0.0, "Addition time should be positive");
        assert!(time_ms < 1000.0, "Addition of 1000 entities took too long: {}ms", time_ms);
        
        log(&format!("Added 1000 entities in {}ms", time_ms));
        
        // Test query performance
        perf_monitor.start_measurement("entity_queries");
        
        for i in 0..100 {
            let entity_key = format!("entity_{}", i * 10);
            kg.get_entity_json(&entity_key).unwrap();
        }
        
        let query_time = perf_monitor.end_measurement("entity_queries").unwrap();
        let avg_query_time = query_time / 100.0;
        
        assert!(avg_query_time < 1.0, "Average query time too high: {}ms", avg_query_time);
        log(&format!("Average query time: {}ms", avg_query_time));
        
        // Test memory usage
        let memory_usage = kg.get_memory_usage_mb();
        assert!(memory_usage > 0.0, "Memory usage should be positive");
        assert!(memory_usage < 50.0, "Memory usage too high for 1000 entities: {}MB", memory_usage);
        
        log(&format!("Memory usage: {}MB", memory_usage));
        
        // Get performance summary
        let perf_summary = perf_monitor.get_summary_json();
        let summary: Value = serde_json::from_str(&perf_summary).unwrap();
        
        assert!(summary["measurements"].is_object());
        assert!(summary["measurements"]["entity_addition"].is_object());
        assert!(summary["measurements"]["entity_queries"].is_object());
        
        log("WASM performance measurement test completed successfully");
    }
    
    #[wasm_bindgen_test]
    fn test_wasm_error_handling() {
        log("Starting WASM error handling test");
        
        let mut kg = WasmKnowledgeGraph::new();
        
        // Test invalid JSON
        let invalid_json_result = kg.add_entity_from_json("invalid json{");
        assert!(invalid_json_result.is_err());
        assert!(invalid_json_result.err().unwrap().contains("JSON"));
        
        // Test duplicate entity
        let entity_data = json!({
            "key": "duplicate_test",
            "name": "Duplicate Test",
            "attributes": {}
        });
        
        kg.add_entity_from_json(&entity_data.to_string()).unwrap();
        let duplicate_result = kg.add_entity_from_json(&entity_data.to_string());
        assert!(duplicate_result.is_err());
        assert!(duplicate_result.err().unwrap().contains("already exists"));
        
        // Test nonexistent entity retrieval
        let nonexistent_result = kg.get_entity_json("nonexistent_entity");
        assert!(nonexistent_result.is_err());
        assert!(nonexistent_result.err().unwrap().contains("not found"));
        
        // Test invalid relationship
        let invalid_rel_result = kg.add_relationship_json(
            "nonexistent_source",
            "nonexistent_target",
            &json!({"name": "test", "weight": 1.0}).to_string()
        );
        assert!(invalid_rel_result.is_err());
        
        log("WASM error handling test completed successfully");
    }
    
    #[wasm_bindgen_test]
    fn test_wasm_graph_algorithms() {
        log("Starting WASM graph algorithms test");
        
        let mut kg = WasmKnowledgeGraph::new();
        
        // Create a simple graph for testing algorithms
        let entities = vec![
            ("A", "Node A"),
            ("B", "Node B"),
            ("C", "Node C"),
            ("D", "Node D"),
            ("E", "Node E"),
        ];
        
        for (key, name) in entities {
            kg.add_entity_from_json(&json!({
                "key": key,
                "name": name,
                "attributes": {}
            }).to_string()).unwrap();
        }
        
        // Create relationships: A->B->C->D, A->E->D
        let relationships = vec![
            ("A", "B", 1.0),
            ("B", "C", 1.0),
            ("C", "D", 1.0),
            ("A", "E", 2.0),
            ("E", "D", 1.0),
        ];
        
        for (source, target, weight) in relationships {
            kg.add_relationship_json(
                source,
                target,
                &json!({
                    "name": format!("{}_{}", source, target),
                    "weight": weight,
                    "type": "directed"
                }).to_string()
            ).unwrap();
        }
        
        // Test shortest path
        let path_result = kg.find_shortest_path_json("A", "D");
        assert!(path_result.is_ok());
        
        let path: Value = serde_json::from_str(&path_result.unwrap()).unwrap();
        assert!(path["found"].as_bool().unwrap());
        
        let path_nodes = path["path"].as_array().unwrap();
        assert!(path_nodes.len() >= 3, "Path should have at least 3 nodes");
        assert_eq!(path_nodes[0], "A");
        assert_eq!(path_nodes[path_nodes.len() - 1], "D");
        
        // Test connected components
        let components_result = kg.find_connected_components_json();
        assert!(components_result.is_ok());
        
        let components: Value = serde_json::from_str(&components_result.unwrap()).unwrap();
        assert_eq!(components["count"], 1, "Should have 1 connected component");
        
        // Test node centrality
        let centrality_result = kg.calculate_centrality_json("degree");
        assert!(centrality_result.is_ok());
        
        let centrality: Value = serde_json::from_str(&centrality_result.unwrap()).unwrap();
        assert!(centrality["A"].as_f64().unwrap() > 0.0);
        
        log("WASM graph algorithms test completed successfully");
    }
    
    #[wasm_bindgen_test]
    fn test_wasm_batch_operations() {
        log("Starting WASM batch operations test");
        
        let mut kg = WasmKnowledgeGraph::new();
        
        // Test batch entity addition
        let mut entities = Vec::new();
        for i in 0..100 {
            entities.push(json!({
                "key": format!("batch_entity_{}", i),
                "name": format!("Batch Entity {}", i),
                "attributes": {
                    "batch_id": "test_batch",
                    "index": i
                }
            }));
        }
        
        let batch_result = kg.add_entities_batch_json(&json!(entities).to_string());
        assert!(batch_result.is_ok());
        
        let batch_response: Value = serde_json::from_str(&batch_result.unwrap()).unwrap();
        assert_eq!(batch_response["added"], 100);
        assert_eq!(batch_response["failed"], 0);
        
        // Test batch relationship addition
        let mut relationships = Vec::new();
        for i in 0..50 {
            relationships.push(json!({
                "source": format!("batch_entity_{}", i),
                "target": format!("batch_entity_{}", i + 50),
                "relationship": {
                    "name": format!("batch_rel_{}", i),
                    "weight": 1.0,
                    "type": "directed"
                }
            }));
        }
        
        let rel_batch_result = kg.add_relationships_batch_json(&json!(relationships).to_string());
        assert!(rel_batch_result.is_ok());
        
        let rel_batch_response: Value = serde_json::from_str(&rel_batch_result.unwrap()).unwrap();
        assert_eq!(rel_batch_response["added"], 50);
        
        // Verify final state
        let stats = kg.get_statistics_json();
        let stats_data: Value = serde_json::from_str(&stats).unwrap();
        assert_eq!(stats_data["entity_count"], 100);
        assert_eq!(stats_data["relationship_count"], 50);
        
        log("WASM batch operations test completed successfully");
    }
    
    #[wasm_bindgen_test]
    fn test_wasm_export_import() {
        log("Starting WASM export/import test");
        
        let mut kg1 = WasmKnowledgeGraph::new();
        
        // Create some test data
        for i in 0..10 {
            kg1.add_entity_from_json(&json!({
                "key": format!("export_entity_{}", i),
                "name": format!("Export Entity {}", i),
                "attributes": {"value": i}
            }).to_string()).unwrap();
        }
        
        for i in 0..5 {
            kg1.add_relationship_json(
                &format!("export_entity_{}", i),
                &format!("export_entity_{}", i + 5),
                &json!({
                    "name": format!("export_rel_{}", i),
                    "weight": 1.0,
                    "type": "directed"
                }).to_string()
            ).unwrap();
        }
        
        // Export to JSON
        let export_result = kg1.export_to_json();
        assert!(export_result.is_ok());
        
        let exported_data = export_result.unwrap();
        let export_json: Value = serde_json::from_str(&exported_data).unwrap();
        
        assert!(export_json["entities"].is_array());
        assert!(export_json["relationships"].is_array());
        assert_eq!(export_json["entities"].as_array().unwrap().len(), 10);
        assert_eq!(export_json["relationships"].as_array().unwrap().len(), 5);
        
        // Import into a new graph
        let mut kg2 = WasmKnowledgeGraph::new();
        let import_result = kg2.import_from_json(&exported_data);
        assert!(import_result.is_ok());
        
        // Verify imported data
        let stats2 = kg2.get_statistics_json();
        let stats2_data: Value = serde_json::from_str(&stats2).unwrap();
        assert_eq!(stats2_data["entity_count"], 10);
        assert_eq!(stats2_data["relationship_count"], 5);
        
        // Verify specific entity
        let entity_check = kg2.get_entity_json("export_entity_0");
        assert!(entity_check.is_ok());
        
        let entity_data: Value = serde_json::from_str(&entity_check.unwrap()).unwrap();
        assert_eq!(entity_data["name"], "Export Entity 0");
        assert_eq!(entity_data["attributes"]["value"], 0);
        
        log("WASM export/import test completed successfully");
    }
    
    #[wasm_bindgen_test]
    async fn test_wasm_async_operations() {
        log("Starting WASM async operations test");
        
        let kg = WasmKnowledgeGraph::new();
        
        // Test async entity loading
        let entity_data = json!({
            "key": "async_entity",
            "name": "Async Test Entity",
            "attributes": {"async": true}
        });
        
        // In real scenario, this would be an async operation
        let future_result = kg.add_entity_async(entity_data.to_string());
        let result = future_result.await;
        
        assert!(result.is_ok());
        
        // Test async query
        let query_future = kg.get_entity_async("async_entity".to_string());
        let query_result = query_future.await;
        
        assert!(query_result.is_ok());
        
        let entity: Value = serde_json::from_str(&query_result.unwrap()).unwrap();
        assert_eq!(entity["name"], "Async Test Entity");
        
        log("WASM async operations test completed successfully");
    }
    
    #[wasm_bindgen_test]
    fn test_wasm_memory_limits() {
        log("Starting WASM memory limits test");
        
        let mut kg = WasmKnowledgeGraph::new();
        let initial_memory = kg.get_memory_usage_mb();
        
        // Add entities until we approach WASM memory limits
        let mut added_count = 0;
        let target_count = 10000; // Conservative limit for testing
        
        for i in 0..target_count {
            let result = kg.add_entity_from_json(&json!({
                "key": format!("mem_test_{}", i),
                "name": format!("Memory Test Entity {}", i),
                "attributes": {
                    "data": "x".repeat(100) // Some data to consume memory
                }
            }).to_string());
            
            if result.is_err() {
                log(&format!("Memory limit reached at {} entities", i));
                break;
            }
            
            added_count += 1;
            
            // Check memory periodically
            if i % 1000 == 999 {
                let current_memory = kg.get_memory_usage_mb();
                log(&format!("After {} entities: {}MB", i + 1, current_memory));
                
                // Safety check - don't exceed reasonable limits
                if current_memory > 100.0 {
                    log("Stopping test - memory usage exceeds 100MB");
                    break;
                }
            }
        }
        
        assert!(added_count >= 5000, "Should be able to add at least 5000 entities");
        
        let final_memory = kg.get_memory_usage_mb();
        let memory_per_entity = (final_memory - initial_memory) / (added_count as f32);
        
        log(&format!("Added {} entities, {:.3}KB per entity", 
                    added_count, memory_per_entity * 1024.0));
        
        assert!(memory_per_entity < 0.01, "Memory per entity too high: {}KB", 
               memory_per_entity * 1024.0);
        
        log("WASM memory limits test completed successfully");
    }
}