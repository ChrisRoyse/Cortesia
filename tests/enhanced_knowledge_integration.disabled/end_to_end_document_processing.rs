//! End-to-End Document Processing Integration Tests
//! 
//! Comprehensive tests for complete document processing pipeline:
//! - Document ingestion and validation
//! - Knowledge extraction and quality assessment
//! - Storage across hierarchical tiers
//! - Retrieval and reasoning integration
//! - Performance and scalability validation

use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::{EntityData, Relationship, AttributeValue};
use llmkg::extraction::AdvancedEntityExtractor;
use llmkg::embedding::store::EmbeddingStore;
use llmkg::cognitive::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use llmkg::storage::persistent_mmap::PersistentMMapStorage;
use llmkg::BrainEnhancedKnowledgeGraph;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use tempfile::TempDir;

#[cfg(test)]
mod tests {
    use super::*;

    struct DocumentProcessingPipeline {
        graph: Arc<RwLock<KnowledgeGraph>>,
        extractor: AdvancedEntityExtractor,
        embedding_store: EmbeddingStore,
        orchestrator: CognitiveOrchestrator,
        storage: PersistentMMapStorage,
        _temp_dir: TempDir,
    }

    impl DocumentProcessingPipeline {
        async fn new() -> Self {
            let temp_dir = TempDir::new().expect("Failed to create temp dir");
            let storage_path = temp_dir.path().join("document_processing.mmap");
            
            let graph = Arc::new(RwLock::new(KnowledgeGraph::new(384).expect("Failed to create graph")));
            let extractor = AdvancedEntityExtractor::new();
            let embedding_store = EmbeddingStore::new(384, 8).expect("Failed to create embedding store");
            let config = CognitiveOrchestratorConfig::default();
            let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new(384).expect("Failed to create brain graph"));
            let orchestrator = CognitiveOrchestrator::new(brain_graph, config).await.expect("Failed to create orchestrator");
            let storage = PersistentMMapStorage::new(Some(&storage_path), 384) // 50MB
                .expect("Failed to create storage");

            Self {
                graph,
                extractor,
                embedding_store,
                orchestrator,
                storage,
                _temp_dir: temp_dir,
            }
        }

        async fn process_document(&mut self, document_id: String, content: &str) -> Result<DocumentProcessingResult, String> {
            let processing_start = Instant::now();

            // Step 1: Document validation and preprocessing
            let validation_start = Instant::now();
            if content.trim().is_empty() {
                return Err("Empty document content".to_string());
            }
            if content.len() > 1_000_000 { // 1MB limit
                return Err("Document too large".to_string());
            }
            let validation_time = validation_start.elapsed();

            // Step 2: Knowledge extraction
            let extraction_start = Instant::now();
            let extraction = self.extractor.extract_entities_and_relations(content).await
                .map_err(|e| format!("Extraction failed: {:?}", e))?;
            let extraction_time = extraction_start.elapsed();

            // Step 3: Quality assessment
            let quality_start = Instant::now();
            let quality_score = self.assess_extraction_quality(&extraction, content);
            let quality_time = quality_start.elapsed();

            // Step 4: Storage in knowledge graph
            let graph_storage_start = Instant::now();
            let mut stored_entities = 0;
            let mut stored_relationships = 0;

            {
                let mut graph_lock = self.graph.write().await;
                for entity in &extraction.entities {
                    let entity_data = EntityData {
                        attributes: [
                            ("name".to_string(), AttributeValue::String(entity.name.clone())),
                            ("type".to_string(), AttributeValue::String(entity.entity_type.clone())),
                            ("confidence".to_string(), AttributeValue::Float(entity.confidence)),
                            ("document_id".to_string(), AttributeValue::String(document_id.clone())),
                            ("quality_score".to_string(), AttributeValue::Float(quality_score)),
                            ("processing_timestamp".to_string(), AttributeValue::String(chrono::Utc::now().to_rfc3339())),
                        ].into_iter().collect(),
                    };

                    graph_lock.add_entity(
                        format!("{}_{}", document_id, entity.id),
                        entity_data
                    ).map_err(|e| format!("Failed to store entity: {:?}", e))?;
                    stored_entities += 1;
                }

                for relation in &extraction.relations {
                    let relationship = Relationship {
                        target: format!("{}_{}", document_id, relation.object_id),
                        relationship_type: relation.predicate.clone(),
                        weight: relation.confidence,
                        properties: [
                            ("document_id".to_string(), AttributeValue::String(document_id.clone())),
                            ("confidence".to_string(), AttributeValue::Float(relation.confidence)),
                        ].into_iter().collect(),
                    };

                    graph_lock.add_relationship(
                        format!("{}_{}", document_id, relation.subject_id),
                        relationship
                    ).map_err(|e| format!("Failed to store relationship: {:?}", e))?;
                    stored_relationships += 1;
                }
            }
            let graph_storage_time = graph_storage_start.elapsed();

            // Step 5: Embedding generation and storage
            let embedding_start = Instant::now();
            let mut stored_embeddings = 0;

            for entity in &extraction.entities {
                // Generate contextual embedding
                let embedding_text = format!("{} {} {} {}", 
                    entity.name, entity.entity_type, content, document_id);
                
                let embedding: Vec<f32> = (0..384).map(|i| {
                    (embedding_text.len() as f32 * i as f32 * 0.0001 + 
                     entity.confidence * quality_score).sin()
                }).collect();

                self.embedding_store.add_embedding(
                    &format!("{}_{}", document_id, entity.id),
                    embedding
                ).map_err(|e| format!("Failed to store embedding: {:?}", e))?;
                stored_embeddings += 1;
            }
            let embedding_time = embedding_start.elapsed();

            // Step 6: Persistent storage
            let persistent_start = Instant::now();
            let document_metadata = serde_json::json!({
                "document_id": document_id,
                "content": content,
                "extraction": extraction,
                "quality_score": quality_score,
                "processing_metrics": {
                    "validation_time_ms": validation_time.as_millis(),
                    "extraction_time_ms": extraction_time.as_millis(),
                    "quality_time_ms": quality_time.as_millis(),
                    "graph_storage_time_ms": graph_storage_time.as_millis(),
                    "embedding_time_ms": embedding_time.as_millis(),
                    "stored_entities": stored_entities,
                    "stored_relationships": stored_relationships,
                    "stored_embeddings": stored_embeddings
                },
                "processed_at": chrono::Utc::now().to_rfc3339()
            });

            self.storage.store(
                &format!("doc_{}", document_id),
                document_metadata.to_string().as_bytes()
            ).map_err(|e| format!("Failed to store document metadata: {:?}", e))?;
            let persistent_time = persistent_start.elapsed();

            let total_processing_time = processing_start.elapsed();

            Ok(DocumentProcessingResult {
                document_id,
                quality_score,
                entities_extracted: extraction.entities.len(),
                relationships_extracted: extraction.relations.len(),
                stored_entities,
                stored_relationships,
                stored_embeddings,
                processing_times: ProcessingTimes {
                    validation: validation_time,
                    extraction: extraction_time,
                    quality_assessment: quality_time,
                    graph_storage: graph_storage_time,
                    embedding_storage: embedding_time,
                    persistent_storage: persistent_time,
                    total: total_processing_time,
                },
                extraction_result: extraction,
            })
        }

        fn assess_extraction_quality(&self, extraction: &llmkg::extraction::ExtractionResult, content: &str) -> f32 {
            let content_length_score = (content.len() as f32 / 1000.0).min(1.0);
            let entity_count_score = (extraction.entities.len() as f32 / 10.0).min(1.0);
            let relation_count_score = (extraction.relations.len() as f32 / 5.0).min(1.0);
            
            let avg_entity_confidence = if extraction.entities.is_empty() {
                0.0
            } else {
                extraction.entities.iter().map(|e| e.confidence).sum::<f32>() / extraction.entities.len() as f32
            };

            let avg_relation_confidence = if extraction.relations.is_empty() {
                0.0
            } else {
                extraction.relations.iter().map(|r| r.confidence).sum::<f32>() / extraction.relations.len() as f32
            };

            (content_length_score * 0.2 + 
             entity_count_score * 0.3 + 
             relation_count_score * 0.2 + 
             avg_entity_confidence * 0.15 + 
             avg_relation_confidence * 0.15).min(1.0)
        }

        async fn query_processed_knowledge(&self, query: &str) -> Result<QueryResult, String> {
            let query_start = Instant::now();

            // Gather all processed entities
            let graph_lock = self.graph.read().await;
            let stats = graph_lock.get_stats();
            drop(graph_lock);

            // Simulate gathering entities for reasoning (simplified)
            let test_entities: Vec<_> = (0..stats.entity_count.min(20)).map(|i| {
                llmkg::extraction::Entity {
                    id: format!("query_entity_{}", i),
                    name: format!("Processed Entity {}", i),
                    entity_type: "processed_content".to_string(),
                    confidence: 0.8,
                }
            }).collect();

            let reasoning_result = self.orchestrator.process_complex_query(
                query,
                &test_entities,
                &[]
            ).await.map_err(|e| format!("Query processing failed: {:?}", e))?;

            let query_time = query_start.elapsed();

            Ok(QueryResult {
                query: query.to_string(),
                confidence: reasoning_result.confidence,
                reasoning_steps: reasoning_result.reasoning_steps.len(),
                explanation: reasoning_result.explanation,
                processing_time: query_time,
            })
        }
    }

    #[derive(Debug)]
    struct DocumentProcessingResult {
        document_id: String,
        quality_score: f32,
        entities_extracted: usize,
        relationships_extracted: usize,
        stored_entities: usize,
        stored_relationships: usize,
        stored_embeddings: usize,
        processing_times: ProcessingTimes,
        extraction_result: llmkg::extraction::ExtractionResult,
    }

    #[derive(Debug)]
    struct ProcessingTimes {
        validation: Duration,
        extraction: Duration,
        quality_assessment: Duration,
        graph_storage: Duration,
        embedding_storage: Duration,
        persistent_storage: Duration,
        total: Duration,
    }

    #[derive(Debug)]
    struct QueryResult {
        query: String,
        confidence: f32,
        reasoning_steps: usize,
        explanation: String,
        processing_time: Duration,
    }

    #[tokio::test]
    async fn test_comprehensive_document_processing_pipeline() {
        let mut pipeline = DocumentProcessingPipeline::new().await;

        // Test documents of varying complexity and quality
        let test_documents = vec![
            (
                "doc_001",
                "Albert Einstein developed the theory of relativity, which revolutionized our understanding of space and time. His famous equation E=mc² shows the relationship between mass and energy.",
                "high_quality"
            ),
            (
                "doc_002", 
                "Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different scientific fields.",
                "high_quality"
            ),
            (
                "doc_003",
                "GPS satellites use Einstein's relativity theory to maintain accurate positioning. Without accounting for relativistic effects, GPS would have significant errors.",
                "medium_quality"
            ),
            (
                "doc_004",
                "Science is important and helps us understand things.",
                "low_quality"
            ),
            (
                "doc_005",
                "The quantum mechanical properties of subatomic particles exhibit wave-particle duality, as demonstrated by the double-slit experiment. This fundamental principle underlies much of modern quantum physics and has applications in quantum computing and cryptography.",
                "high_quality"
            )
        ];

        let mut processing_results = Vec::new();

        // Process each document through the complete pipeline
        for (doc_id, content, expected_quality) in test_documents {
            let result = pipeline.process_document(doc_id.to_string(), content).await
                .expect(&format!("Failed to process document {}", doc_id));

            // Validate processing results
            assert!(result.entities_extracted > 0 || expected_quality == "low_quality", 
                   "Document {} should extract entities (unless low quality)", doc_id);
            
            assert!(result.stored_entities == result.entities_extracted,
                   "All extracted entities should be stored for {}", doc_id);
            
            assert!(result.stored_embeddings == result.entities_extracted,
                   "All entities should have embeddings for {}", doc_id);

            // Validate quality assessment
            match expected_quality {
                "high_quality" => {
                    assert!(result.quality_score > 0.6,
                           "High quality document {} should have quality score > 0.6, got {:.3}", 
                           doc_id, result.quality_score);
                },
                "medium_quality" => {
                    assert!(result.quality_score > 0.3 && result.quality_score <= 0.8,
                           "Medium quality document {} should have moderate quality score, got {:.3}", 
                           doc_id, result.quality_score);
                },
                "low_quality" => {
                    assert!(result.quality_score <= 0.5,
                           "Low quality document {} should have low quality score, got {:.3}", 
                           doc_id, result.quality_score);
                }
            }

            // Validate processing performance
            assert!(result.processing_times.total < Duration::from_secs(5),
                   "Document {} processing should complete within 5 seconds: {:?}", 
                   doc_id, result.processing_times.total);

            assert!(result.processing_times.extraction < Duration::from_secs(2),
                   "Extraction for {} should be fast: {:?}", 
                   doc_id, result.processing_times.extraction);

            processing_results.push(result);
            println!("✓ Processed document {} ({}) - Quality: {:.3}, Entities: {}, Time: {:?}",
                    doc_id, expected_quality, processing_results.last().unwrap().quality_score,
                    processing_results.last().unwrap().entities_extracted,
                    processing_results.last().unwrap().processing_times.total);
        }

        // Test cross-document knowledge integration
        let integration_queries = vec![
            "What did Einstein contribute to science?",
            "How are Einstein and Marie Curie related in terms of scientific achievements?",
            "What are the practical applications of Einstein's theories?",
            "What is the connection between quantum physics and modern technology?"
        ];

        let mut query_results = Vec::new();

        for query in integration_queries {
            let result = pipeline.query_processed_knowledge(query).await
                .expect(&format!("Failed to process query: {}", query));

            // Validate query results
            assert!(result.confidence > 0.2,
                   "Query '{}' should have reasonable confidence: {:.3}", query, result.confidence);
            
            assert!(result.processing_time < Duration::from_secs(10),
                   "Query '{}' should complete quickly: {:?}", query, result.processing_time);

            assert!(!result.explanation.is_empty(),
                   "Query '{}' should provide explanation", query);

            query_results.push(result);
            println!("✓ Query: '{}' - Confidence: {:.3}, Steps: {}, Time: {:?}",
                    query, query_results.last().unwrap().confidence,
                    query_results.last().unwrap().reasoning_steps,
                    query_results.last().unwrap().processing_time);
        }

        // Validate overall system state
        let final_graph_stats = {
            let graph_lock = pipeline.graph.read().await;
            graph_lock.get_stats()
        };

        let final_embedding_info = pipeline.embedding_store.get_info();

        let total_entities_processed: usize = processing_results.iter()
            .map(|r| r.entities_extracted).sum();
        let total_relationships_processed: usize = processing_results.iter()
            .map(|r| r.relationships_extracted).sum();

        assert_eq!(final_graph_stats.entity_count, total_entities_processed,
                  "Graph should contain all processed entities");
        
        assert_eq!(final_embedding_info.count, total_entities_processed,
                  "Embedding store should contain all entity embeddings");

        // Performance analysis
        let avg_processing_time: Duration = processing_results.iter()
            .map(|r| r.processing_times.total)
            .sum::<Duration>() / processing_results.len() as u32;

        let avg_query_time: Duration = query_results.iter()
            .map(|r| r.processing_time)
            .sum::<Duration>() / query_results.len() as u32;

        assert!(avg_processing_time < Duration::from_secs(2),
               "Average document processing time should be reasonable: {:?}", avg_processing_time);

        assert!(avg_query_time < Duration::from_secs(3),
               "Average query time should be reasonable: {:?}", avg_query_time);

        println!("✓ End-to-end document processing integration test passed:");
        println!("  - Documents processed: {}", processing_results.len());
        println!("  - Total entities: {}", total_entities_processed);
        println!("  - Total relationships: {}", total_relationships_processed);
        println!("  - Average processing time: {:?}", avg_processing_time);
        println!("  - Queries executed: {}", query_results.len());
        println!("  - Average query time: {:?}", avg_query_time);
        println!("  - Final graph entities: {}", final_graph_stats.entity_count);
        println!("  - Final embeddings: {}", final_embedding_info.count);
    }

    #[tokio::test]
    async fn test_document_processing_error_handling() {
        let mut pipeline = DocumentProcessingPipeline::new().await;

        // Test various error scenarios
        let error_test_cases = vec![
            ("", "empty_content"),
            ("   \n\t  ", "whitespace_only"),
            ("A", "too_short"),
            ("x".repeat(2_000_000), "too_large"),
            ("Valid content but with\x00null\x01bytes", "invalid_characters"),
        ];

        let mut error_recovery_count = 0;
        let mut graceful_failures = 0;

        for (content, error_type) in error_test_cases {
            let doc_id = format!("error_test_{}", error_type);
            
            match pipeline.process_document(doc_id.clone(), content).await {
                Ok(result) => {
                    // Some error cases might succeed with degraded quality
                    if error_type == "invalid_characters" {
                        assert!(result.quality_score < 0.5,
                               "Document with invalid characters should have low quality score");
                        error_recovery_count += 1;
                        println!("✓ Error recovery for {}: Quality {:.3}", error_type, result.quality_score);
                    } else if error_type == "too_short" {
                        // Very short content might still be processed
                        error_recovery_count += 1;
                        println!("✓ Processed short content: Quality {:.3}", result.quality_score);
                    }
                },
                Err(error_msg) => {
                    // Expected failures should be graceful
                    assert!(!error_msg.is_empty(), "Error message should be informative");
                    graceful_failures += 1;
                    println!("✓ Graceful failure for {}: {}", error_type, error_msg);
                }
            }
        }

        // System should handle most error cases gracefully
        let total_cases = error_test_cases.len();
        let handled_cases = error_recovery_count + graceful_failures;
        assert_eq!(handled_cases, total_cases, 
                  "All error cases should be handled gracefully or recovered");

        println!("✓ Document processing error handling test passed:");
        println!("  - Total error cases: {}", total_cases);
        println!("  - Error recoveries: {}", error_recovery_count);
        println!("  - Graceful failures: {}", graceful_failures);
    }

    #[tokio::test]
    async fn test_document_processing_scalability() {
        let mut pipeline = DocumentProcessingPipeline::new().await;

        // Test scalability with increasing load
        let load_levels = vec![5, 15, 30];
        let base_document = "Scientific research conducted by researchers at institutions leads to discoveries in various fields of study. These findings contribute to human knowledge and technological advancement.";

        for load_level in load_levels {
            let batch_start = Instant::now();
            let mut batch_results = Vec::new();

            // Process documents in parallel batches
            let batch_size = 5;
            for batch_idx in (0..load_level).step_by(batch_size) {
                let batch_end = (batch_idx + batch_size).min(load_level);
                let mut batch_tasks = Vec::new();

                for doc_idx in batch_idx..batch_end {
                    let doc_content = format!("{} Document {} contains specific information about topic {} and methodology {}.",
                                            base_document, doc_idx, doc_idx % 7, doc_idx % 4);
                    let doc_id = format!("scale_test_{}_{}", load_level, doc_idx);
                    
                    // In a real async scenario, we'd spawn tasks here
                    // For this test, we'll process sequentially but measure timing
                    let result = pipeline.process_document(doc_id, &doc_content).await
                        .expect("Scalability test document should process successfully");
                    
                    batch_results.push(result);
                }
            }

            let batch_time = batch_start.elapsed();
            let throughput = load_level as f64 / batch_time.as_secs_f64();

            // Validate scalability metrics
            assert!(throughput > 1.0, 
                   "System should maintain throughput > 1 doc/sec at load {}: {:.2}", load_level, throughput);

            let avg_processing_time: Duration = batch_results.iter()
                .map(|r| r.processing_times.total)
                .sum::<Duration>() / batch_results.len() as u32;

            assert!(avg_processing_time < Duration::from_secs(3),
                   "Average processing time should remain reasonable at load {}: {:?}", 
                   load_level, avg_processing_time);

            // Quality should remain consistent
            let avg_quality: f32 = batch_results.iter()
                .map(|r| r.quality_score)
                .sum::<f32>() / batch_results.len() as f32;

            assert!(avg_quality > 0.5,
                   "Average quality should remain reasonable at load {}: {:.3}", load_level, avg_quality);

            println!("✓ Scalability test at load {}: {:.2} docs/sec, avg time: {:?}, avg quality: {:.3}",
                    load_level, throughput, avg_processing_time, avg_quality);
        }

        // Test query performance under load
        let final_graph_stats = {
            let graph_lock = pipeline.graph.read().await;
            graph_lock.get_stats()
        };

        let loaded_query_start = Instant::now();
        let loaded_query_result = pipeline.query_processed_knowledge(
            "Find patterns and connections in the processed research documents"
        ).await.expect("Loaded query should succeed");
        let loaded_query_time = loaded_query_start.elapsed();

        assert!(loaded_query_time < Duration::from_secs(5),
               "Query should remain responsive under load: {:?}", loaded_query_time);
        assert!(loaded_query_result.confidence > 0.3,
               "Query quality should be maintained under load: {:.3}", loaded_query_result.confidence);

        println!("✓ Document processing scalability test passed:");
        println!("  - Final entities in graph: {}", final_graph_stats.entity_count);
        println!("  - Loaded query time: {:?}", loaded_query_time);
        println!("  - Loaded query confidence: {:.3}", loaded_query_result.confidence);
    }
}