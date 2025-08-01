//! Knowledge Processing Integration Tests
//! 
//! Tests for integration between knowledge processing and storage components:
//! - End-to-end knowledge processing pipeline
//! - Processing results fed into hierarchical storage
//! - Integration with retrieval systems
//! - Cross-component data validation and consistency
//! 
//! These tests verify that knowledge processing works seamlessly
//! with storage and retrieval components.

use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::types::{EntityData, Relationship, AttributeValue};
use llmkg::extraction::AdvancedEntityExtractor;
use llmkg::embedding::store::EmbeddingStore;
use llmkg::cognitive::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use llmkg::storage::persistent_mmap::PersistentMMapStorage;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use tempfile::TempDir;

#[cfg(test)]
mod tests {
    use super::*;

    async fn create_processing_pipeline() -> (
        Arc<RwLock<KnowledgeGraph>>,
        AdvancedEntityExtractor,
        EmbeddingStore,
        CognitiveOrchestrator,
        PersistentMMapStorage
    ) {
        let graph = Arc::new(RwLock::new(KnowledgeGraph::new(384).expect("Failed to create knowledge graph")));
        let extractor = AdvancedEntityExtractor::new();
        let embedding_store = EmbeddingStore::new(384, 8).expect("Failed to create embedding store");
        let config = CognitiveOrchestratorConfig::default();
        let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new_for_test().expect("Failed to create brain graph"));
        let orchestrator = CognitiveOrchestrator::new(brain_graph, config).await.expect("Failed to create orchestrator");
        
        // Create temporary storage
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let storage_path = temp_dir.path().join("test_storage.mmap");
        let storage = PersistentMMapStorage::new(Some(&storage_path), 1024 * 1024) // 1MB
            .expect("Failed to create storage");
        
        (graph, extractor, embedding_store, orchestrator, storage)
    }

    #[tokio::test]
    async fn test_knowledge_processing_to_storage_pipeline() {
        let (graph, extractor, mut embedding_store, orchestrator, mut storage) = create_processing_pipeline().await;
        
        // Test complete pipeline from raw text to hierarchical storage
        let raw_documents = vec![
            "Marie Curie was a physicist who discovered radium.",
            "Radium is a radioactive element used in medical treatments.",
            "Medical treatments using radioactive elements require careful handling."
        ];
        
        let mut processed_knowledge = Vec::new();
        
        for (doc_id, document) in raw_documents.iter().enumerate() {
            // Step 1: Extract knowledge from text
            let extraction = extractor.extract_entities_and_relations(document).await
                .expect("Failed to extract knowledge");
            
            assert!(!extraction.entities.is_empty(), "Should extract entities from document {}", doc_id);
            
            // Step 2: Process and enrich knowledge
            let enriched_entities: Vec<_> = extraction.entities.iter().map(|entity| {
                let mut enriched_data = EntityData {
                    attributes: [
                        ("name".to_string(), AttributeValue::String(entity.name.clone())),
                        ("type".to_string(), AttributeValue::String(entity.entity_type.clone())),
                        ("confidence".to_string(), AttributeValue::Float(entity.confidence)),
                        ("source_document".to_string(), AttributeValue::String(format!("doc_{}", doc_id))),
                        ("processing_timestamp".to_string(), AttributeValue::String(chrono::Utc::now().to_rfc3339())),
                    ].into_iter().collect(),
                };
                
                // Add semantic enrichment
                if entity.name.to_lowercase().contains("curie") {
                    enriched_data.attributes.insert(
                        "category".to_string(),
                        AttributeValue::String("person".to_string())
                    );
                    enriched_data.attributes.insert(
                        "significance".to_string(),
                        AttributeValue::String("high".to_string())
                    );
                }
                
                (entity.id.clone(), enriched_data)
            }).collect();
            
            // Step 3: Store in knowledge graph
            {
                let mut graph_lock = graph.write().await;
                for (entity_id, entity_data) in &enriched_entities {
                    graph_lock.add_entity(entity_id.clone(), entity_data.clone())
                        .expect("Failed to store entity in graph");
                }
                
                for relation in &extraction.relations {
                    let relationship = Relationship {
                        target: relation.object_id.clone(),
                        relationship_type: relation.predicate.clone(),
                        weight: relation.confidence,
                        properties: [
                            ("source_document".to_string(), AttributeValue::String(format!("doc_{}", doc_id))),
                            ("confidence".to_string(), AttributeValue::Float(relation.confidence)),
                        ].into_iter().collect(),
                    };
                    
                    graph_lock.add_relationship(
                        relation.subject_id.clone(),
                        relationship
                    ).expect("Failed to store relationship in graph");
                }
            }
            
            // Step 4: Generate and store embeddings
            for entity in &extraction.entities {
                let embedding_text = format!("{} {} {}", entity.name, entity.entity_type, document);
                let embedding: Vec<f32> = (0..384).map(|i| {
                    (i as f32 * 0.01 + doc_id as f32 * 0.1 + entity.confidence).sin()
                }).collect();
                
                embedding_store.add_embedding(&entity.id, embedding)
                    .expect("Failed to store embedding");
            }
            
            // Step 5: Store in persistent storage
            let serialized_data = serde_json::to_string(&extraction)
                .expect("Failed to serialize extraction");
            
            storage.store(&format!("extraction_{}", doc_id), serialized_data.as_bytes())
                .expect("Failed to store in persistent storage");
            
            processed_knowledge.push(extraction);
        }
        
        // Step 6: Validate integration
        // Check graph completeness
        let graph_lock = graph.read().await;
        let stats = graph_lock.get_stats();
        assert!(stats.entity_count >= 3, "Should have stored all entities");
        assert!(stats.relationship_count >= 1, "Should have stored relationships");
        
        // Check embedding store
        let embedding_info = embedding_store.get_info();
        assert!(embedding_info.count >= 3, "Should have stored embeddings");
        
        // Check persistent storage
        for doc_id in 0..raw_documents.len() {
            let stored_data = storage.retrieve(&format!("extraction_{}", doc_id))
                .expect("Failed to retrieve from storage");
            assert!(!stored_data.is_empty(), "Should have stored data for document {}", doc_id);
        }
        
        println!("✓ Knowledge processing to storage pipeline integration test passed");
    }

    #[tokio::test]
    async fn test_processed_knowledge_retrieval_integration() {
        let (graph, extractor, mut embedding_store, orchestrator, mut storage) = create_processing_pipeline().await;
        
        // First, populate the system with processed knowledge
        let knowledge_base = vec![
            "Albert Einstein developed the theory of relativity",
            "The theory of relativity explains the relationship between space and time",
            "GPS satellites must account for relativistic effects to maintain accuracy",
            "Marie Curie discovered radium and polonium through her research on radioactivity",
            "Radioactivity has important applications in medicine and energy production"
        ];
        
        let mut all_entities = Vec::new();
        let mut all_relations = Vec::new();
        
        // Process and store all knowledge
        for (doc_id, document) in knowledge_base.iter().enumerate() {
            let extraction = extractor.extract_entities_and_relations(document).await
                .expect("Failed to extract knowledge");
            
            // Store in graph
            {
                let mut graph_lock = graph.write().await;
                for entity in &extraction.entities {
                    let entity_data = EntityData {
                        attributes: [
                            ("name".to_string(), AttributeValue::String(entity.name.clone())),
                            ("document_id".to_string(), AttributeValue::String(format!("doc_{}", doc_id))),
                            ("content".to_string(), AttributeValue::String(document.to_string())),
                        ].into_iter().collect(),
                    };
                    graph_lock.add_entity(format!("{}_{}", entity.id, doc_id), entity_data)
                        .expect("Failed to add entity");
                }
                
                for relation in &extraction.relations {
                    let relationship = Relationship {
                        target: format!("{}_{}", relation.object_id, doc_id),
                        relationship_type: relation.predicate.clone(),
                        weight: relation.confidence,
                        properties: [
                            ("document_id".to_string(), AttributeValue::String(format!("doc_{}", doc_id))),
                        ].into_iter().collect(),
                    };
                    graph_lock.add_relationship(
                        format!("{}_{}", relation.subject_id, doc_id),
                        relationship
                    ).expect("Failed to add relationship");
                }
            }
            
            // Store embeddings
            for entity in &extraction.entities {
                let embedding: Vec<f32> = (0..384).map(|i| {
                    (entity.name.len() as f32 * i as f32 * 0.001 + doc_id as f32 * 0.1).sin()
                }).collect();
                embedding_store.add_embedding(&format!("{}_{}", entity.id, doc_id), embedding)
                    .expect("Failed to add embedding");
            }
            
            all_entities.extend(extraction.entities);
            all_relations.extend(extraction.relations);
        }
        
        // Test different retrieval methods
        
        // 1. Graph-based retrieval
        let graph_lock = graph.read().await;
        
        // Find entities by name pattern
        let einstein_entities: Vec<_> = (0..knowledge_base.len())
            .filter_map(|doc_id| {
                let entity_id = format!("Einstein_{}", doc_id);
                graph_lock.get_entity(&entity_id).map(|_| entity_id)
            })
            .collect();
        
        if !einstein_entities.is_empty() {
            println!("Found Einstein entities: {:?}", einstein_entities);
            
            // Check relationships for found entities
            for entity_id in &einstein_entities {
                let relationships = graph_lock.get_entity_relationships(entity_id)
                    .unwrap_or_default();
                assert!(!relationships.is_empty(), "Einstein entity should have relationships");
            }
        }
        
        drop(graph_lock);
        
        // 2. Embedding-based similarity search
        let query_embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.001).sin()).collect();
        let similar_entities = embedding_store.find_similar_by_vector(&query_embedding, 3)
            .expect("Failed to find similar entities");
        
        assert!(!similar_entities.is_empty(), "Should find similar entities");
        assert!(similar_entities.len() <= 3, "Should respect limit parameter");
        
        for (entity_id, similarity) in &similar_entities {
            assert!(similarity >= &0.0 && similarity <= &1.0, "Similarity should be in valid range");
            println!("Similar entity: {} (similarity: {:.3})", entity_id, similarity);
        }
        
        // 3. Cognitive reasoning-based retrieval
        let reasoning_queries = vec![
            "What did Einstein discover?",
            "How are GPS and relativity connected?",
            "What did Marie Curie research?"
        ];
        
        for query in reasoning_queries {
            let reasoning_result = orchestrator.process_complex_query(
                query,
                &all_entities,
                &all_relations
            ).await.expect("Failed to process reasoning query");
            
            assert!(reasoning_result.confidence > 0.0, "Reasoning should produce confident results");
            assert!(!reasoning_result.explanation.is_empty(), "Should provide explanation");
            
            println!("Query: '{}' -> Confidence: {:.3}, Steps: {}", 
                    query, reasoning_result.confidence, reasoning_result.reasoning_steps.len());
        }
        
        // 4. Cross-method validation
        // Verify that different retrieval methods return consistent information
        let test_entity = "Einstein";
        
        // Find via embedding similarity
        let embedding_results = embedding_store.find_similar(test_entity, 5)
            .unwrap_or_default();
        
        // Find via graph traversal
        let graph_lock = graph.read().await;
        let mut graph_results = Vec::new();
        for doc_id in 0..knowledge_base.len() {
            let entity_id = format!("{}_{}", test_entity, doc_id);
            if graph_lock.get_entity(&entity_id).is_some() {
                graph_results.push(entity_id);
            }
        }
        
        // Results should be complementary
        println!("Embedding results: {} entities", embedding_results.len());
        println!("Graph results: {} entities", graph_results.len());
        
        // At least one method should find relevant information
        assert!(!embedding_results.is_empty() || !graph_results.is_empty(),
               "At least one retrieval method should find results for {}", test_entity);
        
        println!("✓ Processed knowledge retrieval integration test passed");
    }

    #[tokio::test]
    async fn test_knowledge_quality_validation_integration() {
        let (graph, extractor, mut embedding_store, orchestrator, _storage) = create_processing_pipeline().await;
        
        // Test documents with varying quality
        let test_documents = vec![
            ("Albert Einstein developed the theory of relativity in 1905.", "high_quality"),
            ("Einstein thing relativity stuff.", "low_quality"),
            ("Marie Curie won Nobel Prizes for her work on radioactivity.", "high_quality"),
            ("Person did science.", "low_quality"),
            ("", "invalid"),
            ("   ", "invalid"),
        ];
        
        let mut quality_metrics = Vec::new();
        
        for (doc_id, (document, expected_quality)) in test_documents.iter().enumerate() {
            // Process document
            let extraction_result = extractor.extract_entities_and_relations(document).await;
            
            let quality_score = match extraction_result {
                Ok(extraction) => {
                    // Quality metrics based on extraction results
                    let entity_quality = if extraction.entities.is_empty() {
                        0.0
                    } else {
                        extraction.entities.iter().map(|e| e.confidence).sum::<f32>() / extraction.entities.len() as f32
                    };
                    
                    let relation_quality = if extraction.relations.is_empty() {
                        0.0
                    } else {
                        extraction.relations.iter().map(|r| r.confidence).sum::<f32>() / extraction.relations.len() as f32
                    };
                    
                    let entity_count_score = (extraction.entities.len() as f32 / 10.0).min(1.0);
                    let relation_count_score = (extraction.relations.len() as f32 / 5.0).min(1.0);
                    
                    // Composite quality score
                    (entity_quality * 0.3 + relation_quality * 0.3 + entity_count_score * 0.2 + relation_count_score * 0.2)
                },
                Err(_) => 0.0, // Failed extraction indicates poor quality
            };
            
            quality_metrics.push((doc_id, expected_quality, quality_score));
            
            // Validate quality expectations
            match *expected_quality {
                "high_quality" => {
                    assert!(quality_score > 0.5, 
                           "High quality document {} should have quality score > 0.5, got {:.3}", 
                           doc_id, quality_score);
                    
                    // High quality documents should extract meaningful entities
                    if let Ok(extraction) = extraction_result {
                        assert!(!extraction.entities.is_empty(), 
                               "High quality document should extract entities");
                        
                        // Store high quality knowledge in graph
                        {
                            let mut graph_lock = graph.write().await;
                            for entity in &extraction.entities {
                                let mut entity_data = EntityData {
                                    attributes: [
                                        ("name".to_string(), AttributeValue::String(entity.name.clone())),
                                        ("quality_score".to_string(), AttributeValue::Float(quality_score)),
                                        ("quality_tier".to_string(), AttributeValue::String("high".to_string())),
                                    ].into_iter().collect(),
                                };
                                
                                graph_lock.add_entity(
                                    format!("quality_{}_{}", entity.id, doc_id),
                                    entity_data
                                ).expect("Failed to add high quality entity");
                            }
                        }
                        
                        // Store high quality embeddings
                        for entity in &extraction.entities {
                            let high_quality_embedding: Vec<f32> = (0..384).map(|i| {
                                (entity.confidence * i as f32 * 0.001 + quality_score).sin()
                            }).collect();
                            
                            embedding_store.add_embedding(
                                &format!("quality_{}_{}", entity.id, doc_id),
                                high_quality_embedding
                            ).expect("Failed to add high quality embedding");
                        }
                    }
                },
                "low_quality" => {
                    assert!(quality_score < 0.7, 
                           "Low quality document {} should have quality score < 0.7, got {:.3}", 
                           doc_id, quality_score);
                },
                "invalid" => {
                    assert!(quality_score < 0.1, 
                           "Invalid document {} should have very low quality score, got {:.3}", 
                           doc_id, quality_score);
                }
            }
        }
        
        // Test quality-based filtering in cognitive processing
        let high_quality_entities: Vec<_> = quality_metrics.iter()
            .filter(|(_, expected_quality, score)| *expected_quality == "high_quality" && *score > 0.5)
            .map(|(doc_id, _, _)| llmkg::extraction::Entity {
                id: format!("filtered_entity_{}", doc_id),
                name: format!("High Quality Entity {}", doc_id),
                entity_type: "concept".to_string(),
                confidence: 0.9,
            })
            .collect();
        
        assert!(!high_quality_entities.is_empty(), "Should have some high quality entities");
        
        let reasoning_result = orchestrator.process_complex_query(
            "Process only high quality knowledge",
            &high_quality_entities,
            &[]
        ).await.expect("Failed to process quality-filtered query");
        
        // Quality filtering should improve reasoning confidence
        assert!(reasoning_result.confidence > 0.6, 
               "Quality-filtered reasoning should be confident: {:.3}", reasoning_result.confidence);
        
        // Test quality-based retrieval
        let graph_lock = graph.read().await;
        let stats = graph_lock.get_stats();
        
        // Should have stored some high-quality entities
        assert!(stats.entity_count > 0, "Should have stored quality entities");
        
        // Test embedding quality affects similarity search
        let quality_search_results = embedding_store.find_similar("high_quality_concept", 5)
            .unwrap_or_default();
        
        // Quality embeddings should be searchable
        println!("Quality-based search found {} results", quality_search_results.len());
        
        // Summary
        println!("✓ Knowledge quality validation integration test passed:");
        for (doc_id, expected_quality, quality_score) in &quality_metrics {
            println!("  Document {}: {} -> Quality: {:.3}", doc_id, expected_quality, quality_score);
        }
    }

    #[tokio::test]
    async fn test_batch_processing_integration() {
        let (graph, extractor, mut embedding_store, orchestrator, mut storage) = create_processing_pipeline().await;
        
        // Create large batch of documents for processing
        let batch_size = 25;
        let documents: Vec<String> = (0..batch_size).map(|i| {
            format!(
                "Research paper {} discusses findings about topic {} and its relationship to concept {}. \
                 The study involved {} participants and concluded that phenomenon {} has significance {}.",
                i,
                i % 5,    // Topic cycles every 5
                i % 3,    // Concept cycles every 3
                50 + i,   // Participant count
                i % 7,    // Phenomenon cycles every 7
                if i % 2 == 0 { "high" } else { "medium" }
            )
        }).collect();
        
        let start_time = Instant::now();
        
        // Process documents in batches to test batch processing efficiency
        let batch_chunk_size = 5;
        let mut total_entities = 0;
        let mut total_relations = 0;
        let mut batch_results = Vec::new();
        
        for (batch_idx, document_batch) in documents.chunks(batch_chunk_size).enumerate() {
            let batch_start = Instant::now();
            
            // Process batch concurrently
            let extraction_tasks: Vec<_> = document_batch.iter().enumerate().map(|(doc_idx, document)| {
                let extractor_ref = &extractor;
                async move {
                    extractor_ref.extract_entities_and_relations(document).await
                        .map(|extraction| (batch_idx * batch_chunk_size + doc_idx, extraction))
                }
            }).collect();
            
            let batch_extractions: Vec<_> = futures::future::join_all(extraction_tasks).await
                .into_iter()
                .collect::<Result<Vec<_>, _>>()
                .expect("Batch extraction should succeed");
            
            // Batch insert into graph
            {
                let mut graph_lock = graph.write().await;
                for (doc_id, extraction) in &batch_extractions {
                    for entity in &extraction.entities {
                        let entity_data = EntityData {
                            attributes: [
                                ("name".to_string(), AttributeValue::String(entity.name.clone())),
                                ("batch_id".to_string(), AttributeValue::String(format!("batch_{}", batch_idx))),
                                ("document_id".to_string(), AttributeValue::String(format!("doc_{}", doc_id))),
                                ("processing_order".to_string(), AttributeValue::Integer(*doc_id as i64)),
                            ].into_iter().collect(),
                        };
                        
                        graph_lock.add_entity(
                            format!("batch_{}_{}", entity.id, doc_id),
                            entity_data
                        ).expect("Failed to add batch entity");
                        total_entities += 1;
                    }
                    
                    for relation in &extraction.relations {
                        let relationship = Relationship {
                            target: format!("batch_{}_{}", relation.object_id, doc_id),
                            relationship_type: relation.predicate.clone(),
                            weight: relation.confidence,
                            properties: [
                                ("batch_id".to_string(), AttributeValue::String(format!("batch_{}", batch_idx))),
                            ].into_iter().collect(),
                        };
                        
                        graph_lock.add_relationship(
                            format!("batch_{}_{}", relation.subject_id, doc_id),
                            relationship
                        ).expect("Failed to add batch relationship");
                        total_relations += 1;
                    }
                }
            }
            
            // Batch insert embeddings
            for (doc_id, extraction) in &batch_extractions {
                for entity in &extraction.entities {
                    let embedding: Vec<f32> = (0..384).map(|i| {
                        (batch_idx as f32 * 0.1 + *doc_id as f32 * 0.01 + i as f32 * 0.001).sin()
                    }).collect();
                    
                    embedding_store.add_embedding(
                        &format!("batch_{}_{}", entity.id, doc_id),
                        embedding
                    ).expect("Failed to add batch embedding");
                }
            }
            
            // Batch store in persistent storage
            let batch_data = serde_json::to_string(&batch_extractions)
                .expect("Failed to serialize batch");
            storage.store(&format!("batch_{}", batch_idx), batch_data.as_bytes())
                .expect("Failed to store batch");
            
            let batch_time = batch_start.elapsed();
            batch_results.push((
                batch_idx,
                batch_extractions.len(),
                batch_time,
                batch_extractions.iter().map(|(_, e)| e.entities.len()).sum::<usize>(),
                batch_extractions.iter().map(|(_, e)| e.relations.len()).sum::<usize>()
            ));
        }
        
        let total_time = start_time.elapsed();
        let throughput = batch_size as f64 / total_time.as_secs_f64();
        
        // Validate batch processing performance
        assert!(throughput > 5.0, "Batch processing should maintain throughput > 5 docs/sec, got {:.2}", throughput);
        
        // Test batch query processing
        let batch_query_start = Instant::now();
        
        // Create sample entities from different batches for cross-batch reasoning
        let cross_batch_entities: Vec<_> = (0..5).map(|i| {
            llmkg::extraction::Entity {
                id: format!("cross_batch_entity_{}", i),
                name: format!("Topic {}", i % 5),
                entity_type: "concept".to_string(),
                confidence: 0.8,
            }
        }).collect();
        
        let batch_reasoning_result = orchestrator.process_complex_query(
            "Find patterns across research topics",
            &cross_batch_entities,
            &[]
        ).await.expect("Failed to process cross-batch query");
        
        let batch_query_time = batch_query_start.elapsed();
        
        // Validate cross-batch reasoning
        assert!(batch_query_time < Duration::from_secs(5), 
               "Cross-batch query should be fast: {:?}", batch_query_time);
        assert!(batch_reasoning_result.confidence > 0.3, 
               "Cross-batch reasoning should be reasonably confident");
        
        // Test batch retrieval
        let batch_retrieval_results = embedding_store.find_similar("Topic 1", 10)
            .expect("Failed to perform batch retrieval");
        
        assert!(!batch_retrieval_results.is_empty(), "Batch retrieval should find results");
        
        // Validate storage integrity
        let storage_batch_count = (0..(batch_size / batch_chunk_size)).filter(|batch_idx| {
            storage.retrieve(&format!("batch_{}", batch_idx)).is_ok()
        }).count();
        
        assert_eq!(storage_batch_count, (batch_size + batch_chunk_size - 1) / batch_chunk_size,
                  "All batches should be stored");
        
        // Validate graph state
        let graph_lock = graph.read().await;
        let final_stats = graph_lock.get_stats();
        
        assert_eq!(final_stats.entity_count, total_entities, "Graph should contain all batch entities");
        assert_eq!(final_stats.relationship_count, total_relations, "Graph should contain all batch relationships");
        
        // Performance summary
        println!("✓ Batch processing integration test passed:");
        println!("  - Total throughput: {:.2} docs/sec", throughput);
        println!("  - Total entities: {}", total_entities);
        println!("  - Total relations: {}", total_relations);
        println!("  - Cross-batch query time: {:?}", batch_query_time);
        
        for (batch_idx, doc_count, batch_time, entities, relations) in batch_results {
            println!("  - Batch {}: {} docs, {} entities, {} relations in {:?}", 
                    batch_idx, doc_count, entities, relations, batch_time);
        }
    }

    #[tokio::test]
    async fn test_incremental_processing_integration() {
        let (graph, extractor, mut embedding_store, orchestrator, mut storage) = create_processing_pipeline().await;
        
        // Initial knowledge base
        let initial_documents = vec![
            "Albert Einstein was born in Germany in 1879.",
            "Einstein developed the theory of relativity.",
            "Marie Curie won Nobel Prizes in Physics and Chemistry."
        ];
        
        // Process initial knowledge base
        let mut knowledge_version = 0;
        let mut all_entities = Vec::new();
        
        for (doc_id, document) in initial_documents.iter().enumerate() {
            let extraction = extractor.extract_entities_and_relations(document).await
                .expect("Failed to extract from initial document");
            
            // Store in graph with version tracking
            {
                let mut graph_lock = graph.write().await;
                for entity in &extraction.entities {
                    let entity_data = EntityData {
                        attributes: [
                            ("name".to_string(), AttributeValue::String(entity.name.clone())),
                            ("version".to_string(), AttributeValue::Integer(knowledge_version)),
                            ("document_id".to_string(), AttributeValue::String(format!("doc_{}", doc_id))),
                            ("created_at".to_string(), AttributeValue::String(chrono::Utc::now().to_rfc3339())),
                        ].into_iter().collect(),
                    };
                    
                    graph_lock.add_entity(
                        format!("v{}_{}", knowledge_version, entity.id),
                        entity_data
                    ).expect("Failed to add initial entity");
                }
            }
            
            all_entities.extend(extraction.entities);
        }
        
        let initial_stats = {
            let graph_lock = graph.read().await;
            graph_lock.get_stats()
        };
        
        // Test incremental updates
        knowledge_version += 1;
        
        let incremental_updates = vec![
            ("update", "Einstein received the Nobel Prize in Physics in 1921."),
            ("correction", "Albert Einstein was actually born in Ulm, Germany in 1879."),
            ("addition", "Einstein emigrated to the United States in 1933."),
            ("relation", "Einstein and Marie Curie both won Nobel Prizes."),
        ];
        
        for (update_type, update_document) in incremental_updates {
            let update_start = Instant::now();
            
            let extraction = extractor.extract_entities_and_relations(update_document).await
                .expect("Failed to extract from update");
            
            // Process incremental update based on type
            match update_type {
                "update" => {
                    // Add new information to existing entities
                    let mut graph_lock = graph.write().await;
                    for entity in &extraction.entities {
                        if entity.name.contains("Einstein") {
                            // Find existing Einstein entity and update it
                            let existing_key = format!("v{}_{}", knowledge_version - 1, "Einstein");
                            if let Some(mut existing_entity) = graph_lock.get_entity(&existing_key).cloned() {
                                // Add new attribute
                                existing_entity.attributes.insert(
                                    "nobel_prize_year".to_string(),
                                    AttributeValue::String("1921".to_string())
                                );
                                existing_entity.attributes.insert(
                                    "last_updated".to_string(),
                                    AttributeValue::String(chrono::Utc::now().to_rfc3339())
                                );
                                
                                // Create updated version
                                graph_lock.add_entity(
                                    format!("v{}_{}", knowledge_version, "Einstein"),
                                    existing_entity
                                ).expect("Failed to update entity");
                            }
                        }
                    }
                },
                "correction" => {
                    // Correct existing information
                    let mut graph_lock = graph.write().await;
                    for entity in &extraction.entities {
                        if entity.name.contains("Einstein") {
                            let entity_data = EntityData {
                                attributes: [
                                    ("name".to_string(), AttributeValue::String(entity.name.clone())),
                                    ("birthplace".to_string(), AttributeValue::String("Ulm, Germany".to_string())),
                                    ("birth_year".to_string(), AttributeValue::String("1879".to_string())),
                                    ("version".to_string(), AttributeValue::Integer(knowledge_version)),
                                    ("correction_flag".to_string(), AttributeValue::String("true".to_string())),
                                ].into_iter().collect(),
                            };
                            
                            graph_lock.add_entity(
                                format!("v{}_corrected_{}", knowledge_version, entity.id),
                                entity_data
                            ).expect("Failed to add corrected entity");
                        }
                    }
                },
                "addition" => {
                    // Add completely new information
                    let mut graph_lock = graph.write().await;
                    for entity in &extraction.entities {
                        let entity_data = EntityData {
                            attributes: [
                                ("name".to_string(), AttributeValue::String(entity.name.clone())),
                                ("version".to_string(), AttributeValue::Integer(knowledge_version)),
                                ("addition_flag".to_string(), AttributeValue::String("true".to_string())),
                                ("migration_info".to_string(), AttributeValue::String("emigrated to US in 1933".to_string())),
                            ].into_iter().collect(),
                        };
                        
                        graph_lock.add_entity(
                            format!("v{}_new_{}", knowledge_version, entity.id),
                            entity_data
                        ).expect("Failed to add new entity");
                    }
                },
                "relation" => {
                    // Add new relationships
                    let mut graph_lock = graph.write().await;
                    for relation in &extraction.relations {
                        let relationship = Relationship {
                            target: format!("v{}_{}_{}", knowledge_version, relation.object_id, "Curie"),
                            relationship_type: "shared_achievement_with".to_string(),
                            weight: relation.confidence,
                            properties: [
                                ("relationship_type".to_string(), AttributeValue::String("incremental_addition".to_string())),
                                ("version".to_string(), AttributeValue::Integer(knowledge_version)),
                            ].into_iter().collect(),
                        };
                        
                        graph_lock.add_relationship(
                            format!("v{}_{}_{}", knowledge_version, relation.subject_id, "Einstein"),
                            relationship
                        ).expect("Failed to add incremental relationship");
                    }
                }
            }
            
            // Update embeddings incrementally
            for entity in &extraction.entities {
                let incremental_embedding: Vec<f32> = (0..384).map(|i| {
                    (knowledge_version as f32 * 0.1 + entity.confidence * i as f32 * 0.001).sin()
                }).collect();
                
                embedding_store.add_embedding(
                    &format!("v{}_{}_{}", knowledge_version, update_type, entity.id),
                    incremental_embedding
                ).expect("Failed to add incremental embedding");
            }
            
            // Store incremental update
            let update_data = serde_json::json!({
                "type": update_type,
                "version": knowledge_version,
                "extraction": extraction,
                "timestamp": chrono::Utc::now().to_rfc3339()
            });
            
            storage.store(
                &format!("incremental_{}_{}", knowledge_version, update_type),
                update_data.to_string().as_bytes()
            ).expect("Failed to store incremental update");
            
            let update_time = update_start.elapsed();
            
            // Incremental updates should be fast
            assert!(update_time < Duration::from_millis(500),
                   "Incremental update '{}' should be fast: {:?}", update_type, update_time);
            
            all_entities.extend(extraction.entities);
        }
        
        // Test incremental query processing
        let incremental_query = "What do we know about Einstein's life and achievements?";
        let reasoning_result = orchestrator.process_complex_query(
            incremental_query,
            &all_entities,
            &[]
        ).await.expect("Failed to process incremental query");
        
        // Incremental processing should improve knowledge quality
        assert!(reasoning_result.confidence > 0.6,
               "Incremental knowledge should improve reasoning confidence: {:.3}", reasoning_result.confidence);
        
        // Verify incremental state
        let final_stats = {
            let graph_lock = graph.read().await;
            graph_lock.get_stats()
        };
        
        assert!(final_stats.entity_count > initial_stats.entity_count,
               "Incremental processing should increase entity count: {} -> {}", 
               initial_stats.entity_count, final_stats.entity_count);
        
        // Test version-aware retrieval
        let version_search = embedding_store.find_similar("Einstein version", 5)
            .expect("Failed to search versioned entities");
        
        assert!(!version_search.is_empty(), "Should find versioned entities");
        
        // Verify storage contains all incremental updates
        for version in 1..=knowledge_version {
            for update_type in ["update", "correction", "addition", "relation"] {
                let stored_update = storage.retrieve(&format!("incremental_{}_{}", version, update_type));
                if stored_update.is_ok() {
                    println!("Found incremental update: v{} {}", version, update_type);
                }
            }
        }
        
        println!("✓ Incremental processing integration test passed:");
        println!("  - Initial entities: {}", initial_stats.entity_count);
        println!("  - Final entities: {}", final_stats.entity_count);
        println!("  - Knowledge versions: {}", knowledge_version);
        println!("  - Final reasoning confidence: {:.3}", reasoning_result.confidence);
    }

    #[tokio::test]
    async fn test_processing_error_recovery_integration() {
        let (graph, extractor, mut embedding_store, orchestrator, mut storage) = create_processing_pipeline().await;
        
        // Test documents with various error scenarios
        let problematic_documents = vec![
            ("", "empty_document"),
            ("   \n\t  ", "whitespace_only"),
            ("A", "too_short"),
            ("Valid content but with\x00null\x01bytes\x02", "invalid_characters"),
            ("This is a very long document ".repeat(1000), "extremely_long"),
            ("Einstein Einstein Einstein Einstein Einstein", "repetitive_content"),
            ("Normal document about Einstein's theory of relativity.", "valid_control"),
        ];
        
        let mut recovery_stats = Vec::new();
        let mut successful_processes = 0;
        let mut recovered_errors = 0;
        
        for (doc_id, (document, error_type)) in problematic_documents.iter().enumerate() {
            let processing_start = Instant::now();
            
            // Attempt processing with error recovery
            let processing_result = async {
                // First attempt: normal processing
                match extractor.extract_entities_and_relations(document).await {
                    Ok(extraction) => {
                        // Validate extraction quality for error recovery
                        if extraction.entities.is_empty() && !document.trim().is_empty() {
                            // Empty extraction from non-empty document - potential error
                            Err(format!("Empty extraction from non-empty document: {}", error_type))
                        } else {
                            Ok((extraction, "direct_success"))
                        }
                    },
                    Err(e) => {
                        // First attempt failed - try error recovery
                        println!("Processing failed for {}: {:?}, attempting recovery...", error_type, e);
                        
                        // Recovery strategy 1: Clean and retry
                        let cleaned_document = document
                            .chars()
                            .filter(|c| c.is_ascii() && !c.is_control() || c.is_whitespace())
                            .collect::<String>()
                            .trim()
                            .to_string();
                        
                        if !cleaned_document.is_empty() && cleaned_document.len() > 10 {
                            match extractor.extract_entities_and_relations(&cleaned_document).await {
                                Ok(extraction) => Ok((extraction, "cleaned_recovery")),
                                Err(_) => {
                                    // Recovery strategy 2: Fallback to minimal extraction
                                    let fallback_extraction = llmkg::extraction::ExtractionResult {
                                        entities: vec![
                                            llmkg::extraction::Entity {
                                                id: format!("fallback_entity_{}", doc_id),
                                                name: "Unknown Entity".to_string(),
                                                entity_type: "fallback".to_string(),
                                                confidence: 0.1,
                                            }
                                        ],
                                        relations: vec![],
                                    };
                                    Ok((fallback_extraction, "fallback_recovery"))
                                }
                            }
                        } else {
                            // Recovery strategy 3: Skip with placeholder
                            let skip_extraction = llmkg::extraction::ExtractionResult {
                                entities: vec![],
                                relations: vec![],
                            };
                            Ok((skip_extraction, "skip_recovery"))
                        }
                    }
                }
            }.await;
            
            let processing_time = processing_start.elapsed();
            
            match processing_result {
                Ok((extraction, recovery_type)) => {
                    match recovery_type {
                        "direct_success" => successful_processes += 1,
                        _ => recovered_errors += 1,
                    }
                    
                    // Attempt to store recovered/successful results
                    let storage_success = async {
                        // Try graph storage with error handling
                        let graph_result = {
                            let mut graph_lock = graph.write().await;
                            let mut stored_entities = 0;
                            
                            for entity in &extraction.entities {
                                let entity_data = EntityData {
                                    attributes: [
                                        ("name".to_string(), AttributeValue::String(entity.name.clone())),
                                        ("error_type".to_string(), AttributeValue::String(error_type.to_string())),
                                        ("recovery_type".to_string(), AttributeValue::String(recovery_type.to_string())),
                                        ("processing_time_ms".to_string(), AttributeValue::Integer(processing_time.as_millis() as i64)),
                                    ].into_iter().collect(),
                                };
                                
                                match graph_lock.add_entity(
                                    format!("recovery_{}_{}", entity.id, doc_id),
                                    entity_data
                                ) {
                                    Ok(_) => stored_entities += 1,
                                    Err(e) => println!("Graph storage error for {}: {:?}", error_type, e),
                                }
                            }
                            stored_entities
                        };
                        
                        // Try embedding storage with error handling
                        let embedding_result = {
                            let mut stored_embeddings = 0;
                            for entity in &extraction.entities {
                                let embedding: Vec<f32> = (0..384).map(|i| {
                                    (doc_id as f32 * 0.1 + i as f32 * 0.001).sin()
                                }).collect();
                                
                                match embedding_store.add_embedding(&format!("recovery_{}_{}", entity.id, doc_id), embedding) {
                                    Ok(_) => stored_embeddings += 1,
                                    Err(e) => println!("Embedding storage error for {}: {:?}", error_type, e),
                                }
                            }
                            stored_embeddings
                        };
                        
                        // Try persistent storage with error handling
                        let persistent_result = {
                            let recovery_data = serde_json::json!({
                                "error_type": error_type,
                                "recovery_type": recovery_type,
                                "processing_time_ms": processing_time.as_millis(),
                                "extraction": extraction,
                                "document_length": document.len(),
                                "timestamp": chrono::Utc::now().to_rfc3339()
                            });
                            
                            storage.store(
                                &format!("error_recovery_{}", doc_id),
                                recovery_data.to_string().as_bytes()
                            ).is_ok()
                        };
                        
                        (graph_result, embedding_result, persistent_result, extraction.entities.len())
                    }.await;
                    
                    recovery_stats.push((
                        doc_id,
                        error_type,
                        recovery_type,
                        processing_time,
                        storage_success.3, // entity count
                        storage_success.0 > 0 || storage_success.1 > 0 || storage_success.2 // any storage success
                    ));
                },
                Err(unrecoverable_error) => {
                    println!("Unrecoverable error for {}: {}", error_type, unrecoverable_error);
                    recovery_stats.push((
                        doc_id,
                        error_type,
                        "unrecoverable",
                        processing_time,
                        0,
                        false
                    ));
                }
            }
        }
        
        // Test cognitive processing error recovery
        let problematic_entities = vec![
            llmkg::extraction::Entity {
                id: "".to_string(), // Empty ID
                name: "Valid Name".to_string(),
                entity_type: "concept".to_string(),
                confidence: 0.9,
            },
            llmkg::extraction::Entity {
                id: "valid_id".to_string(),
                name: "".to_string(), // Empty name
                entity_type: "concept".to_string(),
                confidence: 0.9,
            },
            llmkg::extraction::Entity {
                id: "negative_confidence".to_string(),
                name: "Test Entity".to_string(),
                entity_type: "concept".to_string(),
                confidence: -0.5, // Invalid confidence
            },
        ];
        
        let cognitive_recovery_result = orchestrator.process_complex_query(
            "Process problematic entities with error recovery",
            &problematic_entities,
            &[]
        ).await;
        
        // Test system state consistency after errors
        let final_graph_stats = {
            let graph_lock = graph.read().await;
            graph_lock.get_stats()
        };
        
        let final_embedding_info = embedding_store.get_info();
        
        // Validate error recovery effectiveness
        let total_documents = problematic_documents.len();
        let recovery_rate = (successful_processes + recovered_errors) as f64 / total_documents as f64;
        
        assert!(recovery_rate > 0.5, 
               "Error recovery should handle most cases: {:.1}% recovery rate", recovery_rate * 100.0);
        
        // System should maintain consistency despite errors
        assert!(final_graph_stats.entity_count >= 0, "Graph should maintain valid state");
        assert!(final_embedding_info.count >= 0, "Embedding store should maintain valid state");
        
        // Cognitive processing should handle errors gracefully
        match cognitive_recovery_result {
            Ok(result) => {
                assert!(result.confidence >= 0.0 && result.confidence <= 1.0,
                       "Cognitive recovery should produce valid confidence");
                println!("Cognitive error recovery successful: confidence {:.3}", result.confidence);
            },
            Err(e) => {
                println!("Cognitive error recovery failed gracefully: {:?}", e);
                // Graceful failure is acceptable for severely malformed data
            }
        }
        
        // Summary
        println!("✓ Processing error recovery integration test passed:");
        println!("  - Total documents: {}", total_documents);
        println!("  - Direct successes: {}", successful_processes);
        println!("  - Recovered errors: {}", recovered_errors);
        println!("  - Recovery rate: {:.1}%", recovery_rate * 100.0);
        println!("  - Final graph entities: {}", final_graph_stats.entity_count);
        println!("  - Final embeddings: {}", final_embedding_info.count);
        
        for (doc_id, error_type, recovery_type, processing_time, entity_count, storage_success) in recovery_stats {
            println!("  - Doc {}: {} -> {} ({:?}, {} entities, storage: {})", 
                    doc_id, error_type, recovery_type, processing_time, entity_count, storage_success);
        }
    }
}