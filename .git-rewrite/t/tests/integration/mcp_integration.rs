// MCP Integration Tests
// Tests Model Context Protocol integration

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::Arc;

use crate::test_infrastructure::*;
use crate::entity::{Entity, EntityKey};
use crate::relationship::{Relationship, RelationshipType};
use crate::knowledge_graph::KnowledgeGraph;
use crate::embedding::EmbeddingStore;
use crate::mcp::{
    LlmFriendlyServer, 
    McpToolRequest, 
    McpToolResponse,
    FederatedServer,
    DatabaseShard
};

use tokio;
use futures;
use serde_json::{json, Value};

#[cfg(test)]
mod mcp_integration {
    use super::*;

    #[tokio::test]
    async fn test_mcp_server_tool_integration() {
        let mut test_env = IntegrationTestEnvironment::new("mcp_integration");
        
        // Set up test knowledge graph
        let scenario = test_env.data_generator.generate_academic_scenario(200, 100, 20, 128);
        
        let mut kg = KnowledgeGraph::new();
        for entity in scenario.entities.values() {
            kg.add_entity(entity.clone()).unwrap();
        }
        for (source, target, rel) in scenario.relationships {
            kg.add_relationship(source, target, rel).unwrap();
        }
        
        let mut embedding_store = EmbeddingStore::new(128);
        for (entity_key, embedding) in scenario.embeddings {
            embedding_store.add_embedding(entity_key, embedding).unwrap();
        }
        
        // Create MCP server
        let mcp_server = LlmFriendlyServer::new(kg, embedding_store);
        
        // Test 1: knowledge_search tool
        let search_start = Instant::now();
        let search_request = McpToolRequest {
            tool_name: "knowledge_search".to_string(),
            arguments: json!({
                "query": "machine learning algorithms",
                "max_results": 10,
                "include_context": true
            }),
        };
        
        let search_response = mcp_server.handle_tool_request(search_request).await;
        let search_time = search_start.elapsed();
        
        assert!(search_response.is_ok());
        
        let search_result = search_response.unwrap();
        assert_eq!(search_result.tool_name, "knowledge_search");
        assert!(search_result.success);
        
        let result_data: Value = serde_json::from_str(&search_result.content).unwrap();
        let results = result_data["results"].as_array().unwrap();
        
        assert!(results.len() <= 10);
        assert!(!results.is_empty());
        
        // Verify result structure
        for result in results {
            assert!(result["entity"].is_string());
            assert!(result["relevance_score"].is_number());
            assert!(result["context"].is_array());
            
            let relevance = result["relevance_score"].as_f64().unwrap();
            assert!(relevance >= 0.0 && relevance <= 1.0);
        }
        
        println!("Knowledge search completed in {:?} with {} results", 
                search_time, results.len());
        
        test_env.record_performance("mcp_search_time", search_time);
        test_env.record_metric("mcp_search_results", results.len() as f64);
        
        // Test 2: entity_lookup tool
        let lookup_start = Instant::now();
        let lookup_request = McpToolRequest {
            tool_name: "entity_lookup".to_string(),
            arguments: json!({
                "entity_key": scenario.central_entities[0].to_string(),
                "include_relationships": true,
                "max_relationships": 5
            }),
        };
        
        let lookup_response = mcp_server.handle_tool_request(lookup_request).await;
        let lookup_time = lookup_start.elapsed();
        
        assert!(lookup_response.is_ok());
        
        let lookup_result = lookup_response.unwrap();
        assert!(lookup_result.success);
        
        let lookup_data: Value = serde_json::from_str(&lookup_result.content).unwrap();
        assert!(lookup_data["entity"].is_object());
        assert!(lookup_data["relationships"].is_array());
        
        let relationships = lookup_data["relationships"].as_array().unwrap();
        assert!(relationships.len() <= 5);
        
        println!("Entity lookup completed in {:?} with {} relationships",
                lookup_time, relationships.len());
        
        test_env.record_performance("mcp_lookup_time", lookup_time);
        
        // Test 3: find_connections tool
        let connection_start = Instant::now();
        let connection_request = McpToolRequest {
            tool_name: "find_connections".to_string(),
            arguments: json!({
                "source_entity": scenario.central_entities[0].to_string(),
                "target_entity": scenario.central_entities[1].to_string(),
                "max_path_length": 3,
                "max_paths": 5
            }),
        };
        
        let connection_response = mcp_server.handle_tool_request(connection_request).await;
        let connection_time = connection_start.elapsed();
        
        assert!(connection_response.is_ok());
        
        let connection_result = connection_response.unwrap();
        if connection_result.success {
            let connection_data: Value = serde_json::from_str(&connection_result.content).unwrap();
            let paths = connection_data["paths"].as_array().unwrap();
            
            for path in paths {
                let entities = path["entities"].as_array().unwrap();
                assert!(entities.len() >= 2);
                assert!(entities.len() <= 4); // max_path_length + 1
                
                // Verify path starts and ends correctly
                assert_eq!(entities[0], scenario.central_entities[0].to_string());
                assert_eq!(entities[entities.len() - 1], scenario.central_entities[1].to_string());
            }
            
            println!("Found {} connection paths in {:?}", paths.len(), connection_time);
            test_env.record_metric("mcp_paths_found", paths.len() as f64);
        } else {
            println!("No connection paths found between entities");
        }
        
        test_env.record_performance("mcp_connection_time", connection_time);
        
        // Test 4: aggregate_query tool
        let aggregate_request = McpToolRequest {
            tool_name: "aggregate_query".to_string(),
            arguments: json!({
                "entity_filter": {"type": "paper"},
                "group_by": "year",
                "aggregations": ["count", "avg_citations"]
            }),
        };
        
        let aggregate_response = mcp_server.handle_tool_request(aggregate_request).await;
        assert!(aggregate_response.is_ok());
        
        let aggregate_result = aggregate_response.unwrap();
        if aggregate_result.success {
            let aggregate_data: Value = serde_json::from_str(&aggregate_result.content).unwrap();
            assert!(aggregate_data["groups"].is_object());
            
            for (year, stats) in aggregate_data["groups"].as_object().unwrap() {
                assert!(stats["count"].is_number());
                if stats.get("avg_citations").is_some() {
                    assert!(stats["avg_citations"].is_number());
                }
            }
        }
        
        // Test 5: Error handling
        let invalid_request = McpToolRequest {
            tool_name: "nonexistent_tool".to_string(),
            arguments: json!({}),
        };
        
        let error_response = mcp_server.handle_tool_request(invalid_request).await;
        assert!(error_response.is_ok());
        
        let error_result = error_response.unwrap();
        assert!(!error_result.success);
        assert!(error_result.error_message.is_some());
        
        // Test 6: explain_entity tool
        let explain_request = McpToolRequest {
            tool_name: "explain_entity".to_string(),
            arguments: json!({
                "entity_key": scenario.central_entities[0].to_string(),
                "context_depth": 2,
                "include_embeddings": false
            }),
        };
        
        let explain_response = mcp_server.handle_tool_request(explain_request).await;
        assert!(explain_response.is_ok());
        
        let explain_result = explain_response.unwrap();
        if explain_result.success {
            let explain_data: Value = serde_json::from_str(&explain_result.content).unwrap();
            assert!(explain_data["explanation"].is_string());
            assert!(explain_data["context_entities"].is_array());
            assert!(explain_data["key_relationships"].is_array());
        }
    }
    
    #[tokio::test]
    async fn test_mcp_federated_server() {
        let mut test_env = IntegrationTestEnvironment::new("mcp_federated");
        
        // Create federated setup
        let federation_data = test_env.data_generator.generate_federation_scenario(3, 1000);
        
        let mut databases = Vec::new();
        for (i, shard) in federation_data.shards.iter().enumerate() {
            let mut kg = KnowledgeGraph::new();
            
            // Build shard
            for &entity_key in &shard.entities {
                let entity = Entity::new(entity_key, format!("Entity {:?}", entity_key));
                kg.add_entity(entity).unwrap();
            }
            
            for shard_rel in &shard.relationships {
                kg.add_relationship(
                    shard_rel.source,
                    shard_rel.target,
                    shard_rel.relationship.clone()
                ).unwrap();
            }
            
            // Add cross-shard relationships for this shard
            for cross_rel in &federation_data.cross_shard_relationships {
                if cross_rel.source_shard == i {
                    // Add placeholder for target entity if not in this shard
                    if !kg.contains_entity(cross_rel.target_entity) {
                        let placeholder = Entity::new(
                            cross_rel.target_entity,
                            format!("External Entity {:?}", cross_rel.target_entity)
                        ).with_attribute("external_shard", cross_rel.target_shard.to_string());
                        kg.add_entity(placeholder).unwrap();
                    }
                    
                    kg.add_relationship(
                        cross_rel.source_entity,
                        cross_rel.target_entity,
                        cross_rel.relationship.clone()
                    ).unwrap();
                }
            }
            
            databases.push((format!("shard_{}", i), kg));
        }
        
        let federated_server = FederatedServer::new(databases);
        
        // Test 1: Cross-database query
        let cross_db_start = Instant::now();
        let cross_db_request = McpToolRequest {
            tool_name: "federated_search".to_string(),
            arguments: json!({
                "query": "distributed entity search",
                "max_results_per_db": 5,
                "merge_strategy": "union"
            }),
        };
        
        let federated_response = federated_server.handle_tool_request(cross_db_request).await;
        let cross_db_time = cross_db_start.elapsed();
        
        assert!(federated_response.is_ok());
        
        let federated_result = federated_response.unwrap();
        assert!(federated_result.success);
        
        let result_data: Value = serde_json::from_str(&federated_result.content).unwrap();
        let databases_results = result_data["database_results"].as_object().unwrap();
        
        assert_eq!(databases_results.len(), 3); // Should query all 3 databases
        
        for (db_name, db_results) in databases_results {
            assert!(db_results.is_array());
            let results_array = db_results.as_array().unwrap();
            assert!(results_array.len() <= 5);
        }
        
        println!("Federated search completed in {:?}", cross_db_time);
        test_env.record_performance("federated_search_time", cross_db_time);
        
        // Test 2: Database-specific query
        let specific_db_request = McpToolRequest {
            tool_name: "database_query".to_string(),
            arguments: json!({
                "database_id": "shard_0",
                "query_type": "entity_count"
            }),
        };
        
        let specific_response = federated_server.handle_tool_request(specific_db_request).await;
        assert!(specific_response.is_ok());
        
        let specific_result = specific_response.unwrap();
        assert!(specific_result.success);
        
        let specific_data: Value = serde_json::from_str(&specific_result.content).unwrap();
        assert!(specific_data["count"].is_number());
        assert!(specific_data["count"].as_u64().unwrap() > 0);
        
        // Test 3: Federation statistics
        let stats_request = McpToolRequest {
            tool_name: "federation_stats".to_string(),
            arguments: json!({}),
        };
        
        let stats_response = federated_server.handle_tool_request(stats_request).await;
        assert!(stats_response.is_ok());
        
        let stats_result = stats_response.unwrap();
        assert!(stats_result.success);
        
        let stats_data: Value = serde_json::from_str(&stats_result.content).unwrap();
        assert_eq!(stats_data["total_databases"], 3);
        assert!(stats_data["total_entities"].as_u64().unwrap() >= 3000);
        assert!(stats_data["total_relationships"].is_number());
        assert!(stats_data["cross_shard_relationships"].is_number());
        
        // Test 4: Cross-shard path finding
        if !federation_data.cross_shard_relationships.is_empty() {
            let cross_shard_rel = &federation_data.cross_shard_relationships[0];
            
            let path_request = McpToolRequest {
                tool_name: "federated_path_search".to_string(),
                arguments: json!({
                    "source_entity": cross_shard_rel.source_entity.to_string(),
                    "target_entity": cross_shard_rel.target_entity.to_string(),
                    "max_length": 5,
                    "cross_shard_allowed": true
                }),
            };
            
            let path_response = federated_server.handle_tool_request(path_request).await;
            assert!(path_response.is_ok());
            
            let path_result = path_response.unwrap();
            if path_result.success {
                let path_data: Value = serde_json::from_str(&path_result.content).unwrap();
                assert!(path_data["found"].is_boolean());
                
                if path_data["found"].as_bool().unwrap() {
                    assert!(path_data["path"].is_array());
                    assert!(path_data["shards_traversed"].is_array());
                    
                    let shards_traversed = path_data["shards_traversed"].as_array().unwrap();
                    assert!(shards_traversed.len() >= 2, "Path should traverse multiple shards");
                }
            }
        }
    }
    
    #[tokio::test]
    async fn test_mcp_performance_monitoring() {
        let mut test_env = IntegrationTestEnvironment::new("mcp_performance");
        
        // Set up medium-scale test
        let scenario = test_env.data_generator.generate_performance_test_scenario(1000, 2000);
        
        let mut kg = KnowledgeGraph::new();
        for entity in scenario.entities {
            kg.add_entity(entity).unwrap();
        }
        for (source, target, rel) in scenario.relationships {
            kg.add_relationship(source, target, rel).unwrap();
        }
        
        let mcp_server = LlmFriendlyServer::new(kg, EmbeddingStore::new(128));
        
        // Test concurrent requests
        let concurrent_count = 50;
        let mut request_futures = Vec::new();
        
        for i in 0..concurrent_count {
            let server_clone = mcp_server.clone();
            let request = McpToolRequest {
                tool_name: "knowledge_search".to_string(),
                arguments: json!({
                    "query": format!("test query {}", i),
                    "max_results": 5
                }),
            };
            
            let future = async move {
                let start_time = Instant::now();
                let response = server_clone.handle_tool_request(request).await;
                let elapsed = start_time.elapsed();
                (response, elapsed)
            };
            
            request_futures.push(future);
        }
        
        // Execute all requests concurrently
        let concurrent_start = Instant::now();
        let results = futures::future::join_all(request_futures).await;
        let total_concurrent_time = concurrent_start.elapsed();
        
        // Verify all succeeded and measure performance
        let mut total_time = Duration::from_nanos(0);
        let mut success_count = 0;
        let mut response_times = Vec::new();
        
        for (response, elapsed) in results {
            assert!(response.is_ok());
            let result = response.unwrap();
            if result.success {
                success_count += 1;
                total_time += elapsed;
                response_times.push(elapsed);
                
                // Individual requests should be fast
                assert!(elapsed < Duration::from_millis(100),
                       "Individual request too slow: {:?}", elapsed);
            }
        }
        
        assert_eq!(success_count, concurrent_count);
        
        // Calculate statistics
        response_times.sort();
        let avg_time = total_time / concurrent_count as u32;
        let median_time = response_times[concurrent_count / 2];
        let p95_time = response_times[(concurrent_count as f32 * 0.95) as usize];
        let p99_time = response_times[(concurrent_count as f32 * 0.99) as usize];
        
        println!("Concurrent request statistics:");
        println!("  Total time: {:?}", total_concurrent_time);
        println!("  Average: {:?}", avg_time);
        println!("  Median: {:?}", median_time);
        println!("  P95: {:?}", p95_time);
        println!("  P99: {:?}", p99_time);
        
        assert!(avg_time < Duration::from_millis(50),
               "Average request time too slow: {:?}", avg_time);
        
        assert!(p99_time < Duration::from_millis(100),
               "P99 request time too slow: {:?}", p99_time);
        
        test_env.record_performance("avg_mcp_request_time", avg_time);
        test_env.record_performance("p95_mcp_request_time", p95_time);
        test_env.record_performance("p99_mcp_request_time", p99_time);
        test_env.record_metric("mcp_success_rate", success_count as f64 / concurrent_count as f64);
        
        // Test request queuing under load
        let queue_test_count = 100;
        let mut queue_futures = Vec::new();
        
        for i in 0..queue_test_count {
            let server_clone = mcp_server.clone();
            let request = McpToolRequest {
                tool_name: if i % 3 == 0 { "entity_lookup" } else { "knowledge_search" }.to_string(),
                arguments: json!({
                    "query": format!("queue test {}", i),
                    "max_results": 10
                }),
            };
            
            queue_futures.push(server_clone.handle_tool_request(request));
        }
        
        let queue_start = Instant::now();
        let queue_results = futures::future::join_all(queue_futures).await;
        let queue_time = queue_start.elapsed();
        
        let queue_success = queue_results.iter().filter(|r| r.as_ref().unwrap().success).count();
        assert_eq!(queue_success, queue_test_count);
        
        let throughput = queue_test_count as f64 / queue_time.as_secs_f64();
        println!("Queue test throughput: {:.2} requests/second", throughput);
        
        assert!(throughput > 100.0, "Throughput too low: {:.2} req/s", throughput);
        
        test_env.record_metric("mcp_throughput", throughput);
    }
    
    #[tokio::test]
    async fn test_mcp_streaming_responses() {
        let mut test_env = IntegrationTestEnvironment::new("mcp_streaming");
        
        // Create test data
        let scenario = test_env.data_generator.generate_academic_scenario(500, 200, 50, 128);
        
        let mut kg = KnowledgeGraph::new();
        for entity in scenario.entities.values() {
            kg.add_entity(entity.clone()).unwrap();
        }
        for (source, target, rel) in scenario.relationships {
            kg.add_relationship(source, target, rel).unwrap();
        }
        
        let mcp_server = LlmFriendlyServer::new(kg, EmbeddingStore::new(128));
        
        // Test streaming large result set
        let stream_request = McpToolRequest {
            tool_name: "stream_entities".to_string(),
            arguments: json!({
                "filter": {"type": "paper"},
                "batch_size": 10,
                "include_embeddings": false
            }),
        };
        
        let stream_start = Instant::now();
        let stream_response = mcp_server.handle_streaming_request(stream_request).await;
        
        assert!(stream_response.is_ok());
        
        let mut total_entities = 0;
        let mut batch_count = 0;
        let mut batch_times = Vec::new();
        
        let mut stream = stream_response.unwrap();
        while let Some(batch_result) = stream.next().await {
            let batch_start = Instant::now();
            
            assert!(batch_result.is_ok());
            let batch = batch_result.unwrap();
            
            let batch_data: Value = serde_json::from_str(&batch.content).unwrap();
            let entities = batch_data["entities"].as_array().unwrap();
            
            assert!(entities.len() <= 10, "Batch size exceeded");
            total_entities += entities.len();
            batch_count += 1;
            
            let batch_time = batch_start.elapsed();
            batch_times.push(batch_time);
            
            // Process batch (simulate work)
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        
        let total_stream_time = stream_start.elapsed();
        
        println!("Streamed {} entities in {} batches over {:?}",
                total_entities, batch_count, total_stream_time);
        
        let avg_batch_time = batch_times.iter().sum::<Duration>() / batch_count as u32;
        println!("Average batch processing time: {:?}", avg_batch_time);
        
        test_env.record_metric("stream_total_entities", total_entities as f64);
        test_env.record_metric("stream_batch_count", batch_count as f64);
        test_env.record_performance("stream_total_time", total_stream_time);
        test_env.record_performance("stream_avg_batch_time", avg_batch_time);
    }
    
    #[tokio::test]
    async fn test_mcp_tool_composition() {
        let mut test_env = IntegrationTestEnvironment::new("mcp_composition");
        
        // Create test scenario
        let scenario = test_env.data_generator.generate_academic_scenario(300, 100, 20, 128);
        
        let mut kg = KnowledgeGraph::new();
        for entity in scenario.entities.values() {
            kg.add_entity(entity.clone()).unwrap();
        }
        for (source, target, rel) in scenario.relationships {
            kg.add_relationship(source, target, rel).unwrap();
        }
        
        let mut embedding_store = EmbeddingStore::new(128);
        for (entity_key, embedding) in scenario.embeddings {
            embedding_store.add_embedding(entity_key, embedding).unwrap();
        }
        
        let mcp_server = LlmFriendlyServer::new(kg, embedding_store);
        
        // Test composed query: Find similar entities, then find their connections
        
        // Step 1: Find similar entities
        let target_entity = scenario.central_entities[0];
        let similar_request = McpToolRequest {
            tool_name: "find_similar_entities".to_string(),
            arguments: json!({
                "entity_key": target_entity.to_string(),
                "max_results": 5,
                "similarity_threshold": 0.7
            }),
        };
        
        let similar_response = mcp_server.handle_tool_request(similar_request).await;
        assert!(similar_response.is_ok());
        
        let similar_result = similar_response.unwrap();
        assert!(similar_result.success);
        
        let similar_data: Value = serde_json::from_str(&similar_result.content).unwrap();
        let similar_entities = similar_data["similar_entities"].as_array().unwrap();
        
        assert!(!similar_entities.is_empty());
        
        // Step 2: Find connections between similar entities
        let mut connection_requests = Vec::new();
        
        for similar_entity in similar_entities.iter().take(3) {
            let entity_key = similar_entity["entity"].as_str().unwrap();
            
            let connection_request = McpToolRequest {
                tool_name: "find_connections".to_string(),
                arguments: json!({
                    "source_entity": target_entity.to_string(),
                    "target_entity": entity_key,
                    "max_path_length": 3,
                    "max_paths": 3
                }),
            };
            
            connection_requests.push(mcp_server.handle_tool_request(connection_request));
        }
        
        // Execute connection queries in parallel
        let connection_results = futures::future::join_all(connection_requests).await;
        
        let mut total_paths = 0;
        for result in connection_results {
            assert!(result.is_ok());
            let connection_result = result.unwrap();
            
            if connection_result.success {
                let connection_data: Value = serde_json::from_str(&connection_result.content).unwrap();
                if let Some(paths) = connection_data["paths"].as_array() {
                    total_paths += paths.len();
                }
            }
        }
        
        println!("Found {} total connection paths between similar entities", total_paths);
        
        test_env.record_metric("composed_query_paths", total_paths as f64);
        
        // Test 3: Aggregate analysis of connected components
        let component_request = McpToolRequest {
            tool_name: "analyze_component".to_string(),
            arguments: json!({
                "root_entity": target_entity.to_string(),
                "max_depth": 2,
                "include_statistics": true
            }),
        };
        
        let component_response = mcp_server.handle_tool_request(component_request).await;
        assert!(component_response.is_ok());
        
        let component_result = component_response.unwrap();
        if component_result.success {
            let component_data: Value = serde_json::from_str(&component_result.content).unwrap();
            
            assert!(component_data["entity_count"].is_number());
            assert!(component_data["relationship_count"].is_number());
            assert!(component_data["density"].is_number());
            
            if let Some(type_distribution) = component_data["type_distribution"].as_object() {
                println!("Component type distribution:");
                for (entity_type, count) in type_distribution {
                    println!("  {}: {}", entity_type, count);
                }
            }
        }
    }
}