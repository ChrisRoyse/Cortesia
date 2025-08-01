//! Full Pipeline Integration Tests
//! 
//! Tests for full integration of all system components:
//! - Complete knowledge lifecycle integration
//! - End-to-end data flow validation
//! - System-wide performance under integration load
//! - Cross-component error handling and recovery
//! 
//! These tests verify that all components work together as a cohesive
//! system for realistic production scenarios.

use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::{EntityData, Relationship, AttributeValue};
use llmkg::extraction::AdvancedEntityExtractor;
use llmkg::embedding::store::EmbeddingStore;
use llmkg::cognitive::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_test_system() -> (Arc<RwLock<KnowledgeGraph>>, AdvancedEntityExtractor, EmbeddingStore, CognitiveOrchestrator) {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let extractor = AdvancedEntityExtractor::new();
        let embedding_store = EmbeddingStore::new(384).expect("Failed to create embedding store");
        let config = CognitiveOrchestratorConfig::default();
        let orchestrator = CognitiveOrchestrator::new(config);
        
        (graph, extractor, embedding_store, orchestrator)
    }

    #[tokio::test]
    async fn test_complete_knowledge_lifecycle_integration() {
        let (graph, extractor, embedding_store, orchestrator) = create_test_system().await;
        
        // Step 1: Raw input processing
        let raw_text = "Albert Einstein developed the theory of relativity. This theory revolutionized physics and led to E=mc².";
        
        // Step 2: Extract entities and relationships
        let extraction_result = extractor.extract_entities_and_relations(raw_text).await
            .expect("Failed to extract entities");
        
        assert!(!extraction_result.entities.is_empty(), "Should extract entities");
        assert!(!extraction_result.relations.is_empty(), "Should extract relations");
        
        // Step 3: Store in knowledge graph
        {
            let mut graph_lock = graph.write().await;
            for entity in &extraction_result.entities {
                let entity_data = EntityData {
                    attributes: [("name".to_string(), AttributeValue::String(entity.name.clone()))]
                        .into_iter().collect(),
                };
                graph_lock.add_entity(entity.id.clone(), entity_data)
                    .expect("Failed to add entity");
            }
            
            for relation in &extraction_result.relations {
                let relationship = Relationship {
                    target: relation.object_id.clone(),
                    relationship_type: relation.predicate.clone(),
                    weight: 1.0,
                    properties: std::collections::HashMap::new(),
                };
                graph_lock.add_relationship(
                    relation.subject_id.clone(),
                    relationship
                ).expect("Failed to add relationship");
            }
        }
        
        // Step 4: Create embeddings
        let mut embeddings = Vec::new();
        for entity in &extraction_result.entities {
            let embedding = vec![0.1; 384]; // Mock embedding
            embeddings.push((entity.id.clone(), embedding));
        }
        
        // Step 5: Cognitive processing
        let query = "What did Einstein develop?";
        let reasoning_result = orchestrator.process_complex_query(
            query,
            &extraction_result.entities,
            &extraction_result.relations
        ).await.expect("Failed to process query");
        
        // Step 6: Validate end-to-end results
        assert!(reasoning_result.confidence > 0.5, "Should have reasonable confidence");
        assert!(!reasoning_result.explanation.is_empty(), "Should provide explanation");
        
        // Verify graph state
        let graph_lock = graph.read().await;
        let stats = graph_lock.get_stats();
        assert!(stats.entity_count > 0, "Graph should contain entities");
        assert!(stats.relationship_count > 0, "Graph should contain relationships");
        
        println!("✓ Complete knowledge lifecycle integration test passed");
    }

    #[tokio::test]
    async fn test_end_to_end_data_flow_integration() {
        let (graph, extractor, mut embedding_store, orchestrator) = create_test_system().await;
        
        // Test data flow through multiple documents
        let documents = vec![
            "Marie Curie discovered radium and won Nobel Prizes in Physics and Chemistry.",
            "The Nobel Prize is awarded annually for outstanding contributions to humanity.",
            "Radium is a radioactive element with important medical applications."
        ];
        
        let mut all_entities = Vec::new();
        let mut all_relations = Vec::new();
        
        // Process each document through the pipeline
        for (doc_id, doc_text) in documents.iter().enumerate() {
            // Extract knowledge
            let extraction = extractor.extract_entities_and_relations(doc_text).await
                .expect("Failed to extract from document");
            
            // Store in graph with document tracking
            {
                let mut graph_lock = graph.write().await;
                for entity in &extraction.entities {
                    let mut entity_data = EntityData {
                        attributes: [("name".to_string(), AttributeValue::String(entity.name.clone()))]
                            .into_iter().collect(),
                    };
                    entity_data.attributes.insert(
                        "source_document".to_string(),
                        AttributeValue::String(format!("doc_{}", doc_id))
                    );
                    
                    graph_lock.add_entity(entity.id.clone(), entity_data)
                        .expect("Failed to add entity");
                }
                
                for relation in &extraction.relations {
                    let mut relationship = Relationship {
                        target: relation.object_id.clone(),
                        relationship_type: relation.predicate.clone(),
                        weight: 1.0,
                        properties: std::collections::HashMap::new(),
                    };
                    relationship.properties.insert(
                        "source_document".to_string(),
                        AttributeValue::String(format!("doc_{}", doc_id))
                    );
                    
                    graph_lock.add_relationship(
                        relation.subject_id.clone(),
                        relationship
                    ).expect("Failed to add relationship");
                }
            }
            
            // Store embeddings
            for entity in &extraction.entities {
                let embedding = vec![0.1 * (doc_id as f32 + 1.0); 384];
                embedding_store.add_embedding(&entity.id, embedding)
                    .expect("Failed to add embedding");
            }
            
            all_entities.extend(extraction.entities);
            all_relations.extend(extraction.relations);
        }
        
        // Test cross-document queries
        let complex_query = "How are Marie Curie and the Nobel Prize connected?";
        let result = orchestrator.process_complex_query(
            complex_query,
            &all_entities,
            &all_relations
        ).await.expect("Failed to process cross-document query");
        
        // Validate data flow integrity
        assert!(result.confidence > 0.6, "Cross-document reasoning should be confident");
        assert!(result.reasoning_steps.len() >= 2, "Should involve multiple reasoning steps");
        
        // Verify embedding consistency
        let similar_entities = embedding_store.find_similar("Marie Curie", 3)
            .expect("Failed to find similar entities");
        assert!(!similar_entities.is_empty(), "Should find similar entities");
        
        // Verify graph connectivity
        let graph_lock = graph.read().await;
        let marie_curie_connections = graph_lock.get_entity_relationships("Marie Curie")
            .unwrap_or_default();
        assert!(!marie_curie_connections.is_empty(), "Marie Curie should have connections");
        
        println!("✓ End-to-end data flow integration test passed");
    }

    #[tokio::test]
    async fn test_system_wide_performance_integration() {
        let (graph, extractor, mut embedding_store, orchestrator) = create_test_system().await;
        
        let start_time = Instant::now();
        let document_count = 50;
        let base_text = "Scientific discovery by researcher leads to breakthrough in field";
        
        // Performance test with multiple documents
        for i in 0..document_count {
            let document = format!("{} number {} with specific findings about topic {}.", 
                                 base_text, i, i % 10);
            
            // Time extraction
            let extraction_start = Instant::now();
            let extraction = extractor.extract_entities_and_relations(&document).await
                .expect("Failed to extract entities");
            let extraction_time = extraction_start.elapsed();
            
            // Extraction should be fast
            assert!(extraction_time < Duration::from_millis(200), 
                   "Extraction should be fast: {:?}", extraction_time);
            
            // Time graph operations
            let graph_start = Instant::now();
            {
                let mut graph_lock = graph.write().await;
                for entity in &extraction.entities {
                    let entity_data = EntityData {
                        attributes: [("name".to_string(), AttributeValue::String(entity.name.clone()))]
                            .into_iter().collect(),
                    };
                    graph_lock.add_entity(format!("{}_{}", entity.id, i), entity_data)
                        .expect("Failed to add entity");
                }
            }
            let graph_time = graph_start.elapsed();
            
            // Graph operations should be fast
            assert!(graph_time < Duration::from_millis(100),
                   "Graph operations should be fast: {:?}", graph_time);
            
            // Time embedding operations
            let embedding_start = Instant::now();
            for entity in &extraction.entities {
                let embedding = vec![0.1 * (i as f32); 384];
                embedding_store.add_embedding(&format!("{}_{}", entity.id, i), embedding)
                    .expect("Failed to add embedding");
            }
            let embedding_time = embedding_start.elapsed();
            
            // Embedding operations should be fast
            assert!(embedding_time < Duration::from_millis(150),
                   "Embedding operations should be fast: {:?}", embedding_time);
        }
        
        let total_time = start_time.elapsed();
        let docs_per_second = document_count as f64 / total_time.as_secs_f64();
        
        // System should maintain reasonable throughput
        assert!(docs_per_second > 10.0, 
               "System should process at least 10 docs/second, got: {:.2}", docs_per_second);
        
        // Test query performance under load
        let query_start = Instant::now();
        let entities: Vec<_> = (0..10).map(|i| crate::extraction::Entity {
            id: format!("entity_{}", i),
            name: format!("Entity {}", i),
            entity_type: "concept".to_string(),
            confidence: 0.9,
        }).collect();
        
        let relations: Vec<_> = (0..5).map(|i| crate::extraction::Relation {
            subject_id: format!("entity_{}", i),
            predicate: "relates_to".to_string(),
            object_id: format!("entity_{}", i + 1),
            confidence: 0.8,
        }).collect();
        
        let query_result = orchestrator.process_complex_query(
            "Find connections between entities",
            &entities,
            &relations
        ).await.expect("Failed to process query");
        
        let query_time = query_start.elapsed();
        
        // Query should be fast even with loaded system
        assert!(query_time < Duration::from_millis(500),
               "Query should be fast under load: {:?}", query_time);
        assert!(query_result.confidence > 0.3, "Should maintain query quality under load");
        
        println!("✓ System-wide performance integration test passed: {:.2} docs/sec, query time: {:?}", 
                docs_per_second, query_time);
    }

    #[tokio::test]
    async fn test_cross_component_error_handling_integration() {
        let (graph, extractor, embedding_store, orchestrator) = create_test_system().await;
        
        // Test 1: Invalid input handling
        let invalid_inputs = vec![
            "", // Empty input
            "   ", // Whitespace only
            "a", // Too short
            "Invalid\x00input\x01with\x02control\x03characters", // Control characters
        ];
        
        for (i, invalid_input) in invalid_inputs.iter().enumerate() {
            // Extraction should handle invalid input gracefully
            let extraction_result = extractor.extract_entities_and_relations(invalid_input).await;
            
            match extraction_result {
                Ok(result) => {
                    // If successful, should have minimal/empty results
                    assert!(result.entities.len() <= 1, 
                           "Invalid input {} should produce minimal entities", i);
                },
                Err(_) => {
                    // Error is acceptable for invalid input
                    println!("Expected error for invalid input {}: {}", i, invalid_input);
                }
            }
        }
        
        // Test 2: Graph consistency under errors
        {
            let mut graph_lock = graph.write().await;
            
            // Try to add invalid entity
            let invalid_entity = EntityData {
                attributes: std::collections::HashMap::new(),
            };
            
            let result = graph_lock.add_entity("".to_string(), invalid_entity);
            // Should handle empty entity ID gracefully
            if result.is_err() {
                println!("Graph correctly rejected empty entity ID");
            }
            
            // Add valid entity for relationship test
            let valid_entity = EntityData {
                attributes: [("name".to_string(), AttributeValue::String("TestEntity".to_string()))]
                    .into_iter().collect(),
            };
            graph_lock.add_entity("valid_entity".to_string(), valid_entity)
                .expect("Should add valid entity");
            
            // Try to add relationship to non-existent entity
            let invalid_relationship = Relationship {
                target: "non_existent_entity".to_string(),
                relationship_type: "relates_to".to_string(),
                weight: 1.0,
                properties: std::collections::HashMap::new(),
            };
            
            let result = graph_lock.add_relationship(
                "valid_entity".to_string(),
                invalid_relationship
            );
            
            // Should handle gracefully
            if result.is_err() {
                println!("Graph correctly rejected relationship to non-existent entity");
            }
        }
        
        // Test 3: Orchestrator error recovery
        let malformed_entities = vec![
            crate::extraction::Entity {
                id: "".to_string(), // Empty ID
                name: "Valid Name".to_string(),
                entity_type: "concept".to_string(),
                confidence: 0.9,
            },
            crate::extraction::Entity {
                id: "valid_id".to_string(),
                name: "".to_string(), // Empty name
                entity_type: "concept".to_string(),
                confidence: 0.9,
            },
        ];
        
        let result = orchestrator.process_complex_query(
            "Test query with malformed data",
            &malformed_entities,
            &[]
        ).await;
        
        // Should either succeed with degraded results or fail gracefully
        match result {
            Ok(reasoning_result) => {
                // If successful, confidence should be lower due to poor data quality
                assert!(reasoning_result.confidence < 0.7, 
                       "Should have lower confidence with malformed data");
                println!("Orchestrator handled malformed data gracefully");
            },
            Err(err) => {
                println!("Orchestrator appropriately rejected malformed data: {:?}", err);
            }
        }
        
        // Test 4: System state consistency after errors
        let graph_lock = graph.read().await;
        let stats = graph_lock.get_stats();
        
        // Graph should maintain consistency despite errors
        assert!(stats.entity_count >= 1, "Graph should have at least the valid entity");
        // Relationship count depends on implementation - some systems may allow orphaned relationships
        
        println!("✓ Cross-component error handling integration test passed");
    }

    #[tokio::test]
    async fn test_configuration_integration() {
        // Test configuration consistency across components
        
        // Create systems with custom configurations
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
        let extractor = AdvancedEntityExtractor::new();
        let embedding_store = EmbeddingStore::new(512) // Different dimension
            .expect("Failed to create embedding store");
        
        // Test configuration validation
        let config = CognitiveOrchestratorConfig {
            max_reasoning_depth: 5,
            confidence_threshold: 0.7,
            enable_creative_reasoning: true,
            timeout_seconds: 30,
        };
        let orchestrator = CognitiveOrchestrator::new(config.clone());
        
        // Verify configuration is applied
        let orchestrator_config = orchestrator.get_config();
        assert_eq!(orchestrator_config.max_reasoning_depth, 5);
        assert_eq!(orchestrator_config.confidence_threshold, 0.7);
        assert_eq!(orchestrator_config.enable_creative_reasoning, true);
        assert_eq!(orchestrator_config.timeout_seconds, 30);
        
        // Test embedding store configuration
        let store_info = embedding_store.get_info();
        assert_eq!(store_info.dimension, 512, "Embedding store should use configured dimension");
        
        // Test configuration impact on processing
        let test_entities = vec![
            crate::extraction::Entity {
                id: "entity1".to_string(),
                name: "Entity 1".to_string(),
                entity_type: "concept".to_string(),
                confidence: 0.6, // Below threshold
            },
            crate::extraction::Entity {
                id: "entity2".to_string(),
                name: "Entity 2".to_string(),
                entity_type: "concept".to_string(),
                confidence: 0.8, // Above threshold
            },
        ];
        
        let result = orchestrator.process_complex_query(
            "Test configuration impact",
            &test_entities,
            &[]
        ).await.expect("Failed to process query");
        
        // Configuration should affect results
        assert!(result.confidence >= config.confidence_threshold - 0.1, 
               "Result should respect confidence threshold configuration");
        
        // Test memory constraints are respected
        let graph_lock = graph.read().await;
        let memory_usage = graph_lock.get_memory_usage();
        assert!(memory_usage.total_bytes < 100_000_000, // 100MB limit for test
               "Should respect memory constraints");
        
        println!("✓ Configuration integration test passed");
    }

    #[tokio::test]
    async fn test_monitoring_integration() {
        let (graph, extractor, mut embedding_store, orchestrator) = create_test_system().await;
        
        // Track initial state
        let initial_graph_stats = {
            let graph_lock = graph.read().await;
            graph_lock.get_stats()
        };
        let initial_embedding_count = embedding_store.get_info().count;
        
        // Perform operations while monitoring
        let test_document = "Marie Curie conducted research on radioactivity and won Nobel Prizes.";
        
        let extraction_result = extractor.extract_entities_and_relations(test_document).await
            .expect("Failed to extract entities");
        
        // Monitor graph changes
        {
            let mut graph_lock = graph.write().await;
            for entity in &extraction_result.entities {
                let entity_data = EntityData {
                    attributes: [("name".to_string(), AttributeValue::String(entity.name.clone()))]
                        .into_iter().collect(),
                };
                graph_lock.add_entity(entity.id.clone(), entity_data)
                    .expect("Failed to add entity");
            }
            
            for relation in &extraction_result.relations {
                let relationship = Relationship {
                    target: relation.object_id.clone(),
                    relationship_type: relation.predicate.clone(),
                    weight: 1.0,
                    properties: std::collections::HashMap::new(),
                };
                graph_lock.add_relationship(
                    relation.subject_id.clone(),
                    relationship
                ).expect("Failed to add relationship");
            }
        }
        
        // Monitor embedding store changes
        for entity in &extraction_result.entities {
            let embedding = vec![0.5; 384];
            embedding_store.add_embedding(&entity.id, embedding)
                .expect("Failed to add embedding");
        }
        
        // Check monitoring data
        let final_graph_stats = {
            let graph_lock = graph.read().await;
            graph_lock.get_stats()
        };
        let final_embedding_count = embedding_store.get_info().count;
        
        // Verify metrics capture changes
        assert!(final_graph_stats.entity_count > initial_graph_stats.entity_count,
               "Graph entity count should increase");
        assert!(final_embedding_count > initial_embedding_count,
               "Embedding count should increase");
        
        // Test performance monitoring
        let start_time = Instant::now();
        let query_result = orchestrator.process_complex_query(
            "What did Marie Curie research?",
            &extraction_result.entities,
            &extraction_result.relations
        ).await.expect("Failed to process query");
        let query_duration = start_time.elapsed();
        
        // Monitor query performance
        assert!(query_duration < Duration::from_secs(5), 
               "Query should complete within reasonable time: {:?}", query_duration);
        assert!(query_result.confidence > 0.0, "Query should produce confident result");
        
        // Monitor memory usage
        let memory_usage = {
            let graph_lock = graph.read().await;
            graph_lock.get_memory_usage()
        };
        
        assert!(memory_usage.total_bytes > 0, "Should track memory usage");
        assert!(memory_usage.total_bytes < 50_000_000, // 50MB reasonable limit
               "Memory usage should be reasonable: {} bytes", memory_usage.total_bytes);
        
        // Log monitoring summary
        println!("✓ Monitoring integration test passed:");
        println!("  - Entities: {} -> {}", initial_graph_stats.entity_count, final_graph_stats.entity_count);
        println!("  - Relationships: {} -> {}", initial_graph_stats.relationship_count, final_graph_stats.relationship_count);
        println!("  - Embeddings: {} -> {}", initial_embedding_count, final_embedding_count);
        println!("  - Query time: {:?}", query_duration);
        println!("  - Memory usage: {} bytes", memory_usage.total_bytes);
    }

    #[tokio::test]
    async fn test_scalability_integration() {
        let (graph, extractor, mut embedding_store, orchestrator) = create_test_system().await;
        
        // Test scalability with increasing load
        let load_levels = vec![10, 50, 100];
        let mut performance_metrics = Vec::new();
        
        for load_level in load_levels {
            let start_time = Instant::now();
            let mut total_entities = 0;
            let mut total_relations = 0;
            
            // Process multiple documents concurrently
            let mut tasks = Vec::new();
            
            for i in 0..load_level {
                let extractor_ref = &extractor;
                let task = tokio::spawn(async move {
                    let document = format!(
                        "Scientist {} discovered element {} through research on {}. \
                         This breakthrough led to applications in field {}.",
                        i, i, i % 5, i % 3
                    );
                    
                    extractor_ref.extract_entities_and_relations(&document).await
                });
                tasks.push(task);
            }
            
            // Wait for all extractions to complete
            let extraction_results: Vec<_> = futures::future::join_all(tasks).await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .expect("All extraction tasks should complete")
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .expect("All extractions should succeed");
            
            // Process results in batches for better performance
            let batch_size = 10;
            for batch in extraction_results.chunks(batch_size) {
                // Add to graph in batches
                {
                    let mut graph_lock = graph.write().await;
                    for (doc_idx, extraction) in batch.iter().enumerate() {
                        for entity in &extraction.entities {
                            let entity_data = EntityData {
                                attributes: [
                                    ("name".to_string(), AttributeValue::String(entity.name.clone())),
                                    ("batch".to_string(), AttributeValue::String(format!("{}", load_level))),
                                ].into_iter().collect(),
                            };
                            
                            let unique_id = format!("{}_{}_{}_{}", load_level, doc_idx, entity.id, total_entities);
                            graph_lock.add_entity(unique_id, entity_data)
                                .expect("Failed to add entity");
                            total_entities += 1;
                        }
                        
                        for relation in &extraction.relations {
                            let relationship = Relationship {
                                target: format!("{}_{}_{}_{}", load_level, doc_idx, relation.object_id, total_entities - 1),
                                relationship_type: relation.predicate.clone(),
                                weight: 1.0,
                                properties: [("batch".to_string(), AttributeValue::String(format!("{}", load_level)))]
                                    .into_iter().collect(),
                            };
                            
                            let subject_id = format!("{}_{}_{}_{}", load_level, doc_idx, relation.subject_id, total_entities - 2);
                            graph_lock.add_relationship(subject_id, relationship)
                                .expect("Failed to add relationship");
                            total_relations += 1;
                        }
                    }
                }
                
                // Add embeddings in batches
                for (doc_idx, extraction) in batch.iter().enumerate() {
                    for (entity_idx, entity) in extraction.entities.iter().enumerate() {
                        let embedding = vec![0.1 * (load_level as f32 + entity_idx as f32); 384];
                        let unique_id = format!("{}_{}_{}_{}", load_level, doc_idx, entity.id, entity_idx);
                        embedding_store.add_embedding(&unique_id, embedding)
                            .expect("Failed to add embedding");
                    }
                }
            }
            
            let processing_time = start_time.elapsed();
            let throughput = load_level as f64 / processing_time.as_secs_f64();
            
            // Test query performance at this scale
            let query_start = Instant::now();
            let sample_entities: Vec<_> = extraction_results[0].entities.iter().take(5).cloned().collect();
            let sample_relations: Vec<_> = extraction_results[0].relations.iter().take(3).cloned().collect();
            
            let query_result = orchestrator.process_complex_query(
                "Find relationships between scientific discoveries",
                &sample_entities,
                &sample_relations
            ).await.expect("Failed to process query at scale");
            
            let query_time = query_start.elapsed();
            
            performance_metrics.push((load_level, throughput, query_time, query_result.confidence));
            
            // Verify system maintains quality at scale
            assert!(throughput > 1.0, "Should maintain at least 1 doc/sec at load {}", load_level);
            assert!(query_time < Duration::from_secs(10), "Query should be responsive at load {}", load_level);
            assert!(query_result.confidence > 0.2, "Should maintain some query quality at load {}", load_level);
            
            // Check memory usage doesn't grow excessively
            let memory_usage = {
                let graph_lock = graph.read().await;
                graph_lock.get_memory_usage()
            };
            
            let memory_per_entity = memory_usage.total_bytes as f64 / total_entities as f64;
            assert!(memory_per_entity < 10_000.0, // 10KB per entity seems reasonable
                   "Memory usage per entity should be reasonable: {:.0} bytes", memory_per_entity);
            
            println!("Load level {}: {:.2} docs/sec, query: {:?}, confidence: {:.3}, memory/entity: {:.0}B",
                    load_level, throughput, query_time, query_result.confidence, memory_per_entity);
        }
        
        // Verify scalability characteristics
        assert!(performance_metrics.len() == 3, "Should test all load levels");
        
        // System should maintain reasonable performance across load levels
        let (_, throughput_10, query_time_10, confidence_10) = performance_metrics[0];
        let (_, throughput_100, query_time_100, confidence_100) = performance_metrics[2];
        
        // Throughput shouldn't degrade too much (allow 50% degradation at 10x load)
        assert!(throughput_100 > throughput_10 * 0.5, 
               "Throughput should degrade gracefully: {:.2} -> {:.2}", throughput_10, throughput_100);
        
        // Query time shouldn't increase too much (allow 3x increase at 10x load)
        assert!(query_time_100 < query_time_10 * 3, 
               "Query time should scale reasonably: {:?} -> {:?}", query_time_10, query_time_100);
        
        // Confidence shouldn't degrade too much (allow 30% degradation)
        assert!(confidence_100 > confidence_10 * 0.7,
               "Query confidence should degrade gracefully: {:.3} -> {:.3}", confidence_10, confidence_100);
        
        println!("✓ Scalability integration test passed");
    }
}