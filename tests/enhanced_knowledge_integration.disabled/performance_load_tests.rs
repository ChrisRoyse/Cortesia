//! Performance and Load Integration Tests
//! 
//! Tests for system performance under various load conditions:
//! - High-throughput document processing
//! - Concurrent user scenarios
//! - Memory usage under load
//! - Query performance with large datasets
//! - System stability and recovery

use llmkg::core::graph::KnowledgeGraph;
use llmkg::core::types::{EntityData, Relationship, AttributeValue};
use llmkg::extraction::AdvancedEntityExtractor;
use llmkg::embedding::store::EmbeddingStore;
use llmkg::cognitive::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};
use futures::future::join_all;

#[cfg(test)]
mod tests {
    use super::*;

    struct PerformanceTestSystem {
        graph: Arc<RwLock<KnowledgeGraph>>,
        extractor: AdvancedEntityExtractor,
        embedding_store: EmbeddingStore,
        orchestrator: CognitiveOrchestrator,
    }

    impl PerformanceTestSystem {
        async fn new() -> Self {
            let graph = Arc::new(RwLock::new(KnowledgeGraph::new()));
            let extractor = AdvancedEntityExtractor::new();
            let embedding_store = EmbeddingStore::new(384).expect("Failed to create embedding store");
            let config = CognitiveOrchestratorConfig {
                max_reasoning_depth: 5,
                confidence_threshold: 0.4,
                enable_creative_reasoning: true,
                timeout_seconds: 10, // Shorter timeout for load testing
            };
            let orchestrator = CognitiveOrchestrator::new(config);

            Self {
                graph,
                extractor,
                embedding_store,
                orchestrator,
            }
        }

        async fn process_document_batch(&mut self, batch_id: usize, documents: Vec<String>) -> Result<BatchResult, String> {
            let batch_start = Instant::now();
            let mut successful_processes = 0;
            let mut failed_processes = 0;
            let mut total_entities = 0;
            let mut total_relations = 0;

            for (doc_index, document) in documents.iter().enumerate() {
                let doc_id = format!("batch_{}_{}", batch_id, doc_index);
                
                match self.process_single_document(&doc_id, document).await {
                    Ok(metrics) => {
                        successful_processes += 1;
                        total_entities += metrics.entities;
                        total_relations += metrics.relations;
                    },
                    Err(_) => {
                        failed_processes += 1;
                    }
                }
            }

            let batch_time = batch_start.elapsed();

            Ok(BatchResult {
                batch_id,
                total_documents: documents.len(),
                successful_processes,
                failed_processes,
                total_entities,
                total_relations,
                processing_time: batch_time,
            })
        }

        async fn process_single_document(&mut self, doc_id: &str, content: &str) -> Result<ProcessingMetrics, String> {
            // Extract knowledge
            let extraction = self.extractor.extract_entities_and_relations(content).await
                .map_err(|e| format!("Extraction failed: {:?}", e))?;

            // Store in graph
            {
                let mut graph_lock = self.graph.write().await;
                for entity in &extraction.entities {
                    let entity_data = EntityData {
                        attributes: [
                            ("name".to_string(), AttributeValue::String(entity.name.clone())),
                            ("doc_id".to_string(), AttributeValue::String(doc_id.to_string())),
                            ("load_test".to_string(), AttributeValue::String("true".to_string())),
                        ].into_iter().collect(),
                    };

                    graph_lock.add_entity(
                        format!("{}_{}", doc_id, entity.id),
                        entity_data
                    ).map_err(|e| format!("Graph storage failed: {:?}", e))?;
                }

                for relation in &extraction.relations {
                    let relationship = Relationship {
                        target: format!("{}_{}", doc_id, relation.object_id),
                        relationship_type: relation.predicate.clone(),
                        weight: relation.confidence,
                        properties: [
                            ("doc_id".to_string(), AttributeValue::String(doc_id.to_string())),
                        ].into_iter().collect(),
                    };

                    graph_lock.add_relationship(
                        format!("{}_{}", doc_id, relation.subject_id),
                        relationship
                    ).map_err(|e| format!("Relationship storage failed: {:?}", e))?;
                }
            }

            // Store embeddings
            for entity in &extraction.entities {
                let embedding: Vec<f32> = (0..384).map(|i| {
                    (doc_id.len() as f32 * i as f32 * 0.0001 + entity.confidence).sin()
                }).collect();

                self.embedding_store.add_embedding(
                    &format!("{}_{}", doc_id, entity.id),
                    embedding
                ).map_err(|e| format!("Embedding storage failed: {:?}", e))?;
            }

            Ok(ProcessingMetrics {
                entities: extraction.entities.len(),
                relations: extraction.relations.len(),
            })
        }

        async fn perform_concurrent_queries(&self, queries: Vec<String>) -> Result<Vec<QueryMetrics>, String> {
            let query_tasks: Vec<_> = queries.into_iter().enumerate().map(|(idx, query)| {
                let orchestrator = &self.orchestrator;
                async move {
                    let query_start = Instant::now();
                    
                    // Create test entities for the query
                    let test_entities: Vec<_> = (0..10).map(|i| {
                        llmkg::extraction::Entity {
                            id: format!("query_entity_{}_{}", idx, i),
                            name: format!("Test Entity {} {}", idx, i),
                            entity_type: "test_concept".to_string(),
                            confidence: 0.8,
                        }
                    }).collect();

                    let result = orchestrator.process_complex_query(
                        &query,
                        &test_entities,
                        &[]
                    ).await;

                    let query_time = query_start.elapsed();

                    match result {
                        Ok(reasoning_result) => Ok(QueryMetrics {
                            query_index: idx,
                            query: query.clone(),
                            success: true,
                            confidence: reasoning_result.confidence,
                            reasoning_steps: reasoning_result.reasoning_steps.len(),
                            processing_time: query_time,
                        }),
                        Err(e) => Ok(QueryMetrics {
                            query_index: idx,
                            query: query.clone(),
                            success: false,
                            confidence: 0.0,
                            reasoning_steps: 0,
                            processing_time: query_time,
                        })
                    }
                }
            }).collect();

            let query_results = join_all(query_tasks).await;
            query_results.into_iter().collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Query execution failed: {:?}", e))
        }

        async fn get_system_metrics(&self) -> SystemMetrics {
            let graph_stats = {
                let graph_lock = self.graph.read().await;
                graph_lock.get_stats()
            };

            let memory_usage = {
                let graph_lock = self.graph.read().await;
                graph_lock.get_memory_usage()
            };

            let embedding_info = self.embedding_store.get_info();

            SystemMetrics {
                total_entities: graph_stats.entity_count,
                total_relationships: graph_stats.relationship_count,
                memory_usage_bytes: memory_usage.total_bytes,
                embedding_count: embedding_info.count,
                embedding_dimension: embedding_info.dimension,
            }
        }
    }

    #[derive(Debug)]
    struct BatchResult {
        batch_id: usize,
        total_documents: usize,
        successful_processes: usize,
        failed_processes: usize,
        total_entities: usize,
        total_relations: usize,
        processing_time: Duration,
    }

    #[derive(Debug)]
    struct ProcessingMetrics {
        entities: usize,
        relations: usize,
    }

    #[derive(Debug)]
    struct QueryMetrics {
        query_index: usize,
        query: String,
        success: bool,
        confidence: f32,
        reasoning_steps: usize,
        processing_time: Duration,
    }

    #[derive(Debug)]
    struct SystemMetrics {
        total_entities: usize,
        total_relationships: usize,
        memory_usage_bytes: usize,
        embedding_count: usize,
        embedding_dimension: usize,
    }

    #[tokio::test]
    async fn test_high_throughput_document_processing() {
        let mut system = PerformanceTestSystem::new().await;

        // Generate large batch of test documents
        let batch_sizes = vec![10, 25, 50];
        let base_content = "Research conducted by scientists at various institutions has led to important discoveries in multiple fields including physics, chemistry, biology, and technology.";

        let mut overall_performance_metrics = Vec::new();

        for batch_size in batch_sizes {
            let throughput_start = Instant::now();

            // Generate documents for this batch size
            let documents: Vec<String> = (0..batch_size).map(|i| {
                format!("{} Document {} focuses on topic {} and explores methodology {} with findings related to concept {}.",
                       base_content, i, i % 7, i % 4, i % 5)
            }).collect();

            // Process the batch
            let batch_result = system.process_document_batch(batch_size, documents).await
                .expect(&format!("Failed to process batch of size {}", batch_size));

            let throughput_time = throughput_start.elapsed();
            let docs_per_second = batch_result.total_documents as f64 / throughput_time.as_secs_f64();

            // Validate throughput performance
            assert!(docs_per_second > 2.0,
                   "Should maintain throughput > 2 docs/sec for batch size {}: {:.2}", 
                   batch_size, docs_per_second);

            assert!(batch_result.successful_processes >= (batch_result.total_documents * 8 / 10),
                   "Should successfully process at least 80% of documents in batch size {}: {}/{}", 
                   batch_size, batch_result.successful_processes, batch_result.total_documents);

            assert!(batch_result.processing_time < Duration::from_secs(batch_size as u64 * 2),
                   "Batch processing should scale reasonably for size {}: {:?}", 
                   batch_size, batch_result.processing_time);

            overall_performance_metrics.push((batch_size, docs_per_second, batch_result.successful_processes));

            println!("✓ Throughput test batch size {}: {:.2} docs/sec, {}/{} successful, time: {:?}",
                    batch_size, docs_per_second, batch_result.successful_processes, 
                    batch_result.total_documents, batch_result.processing_time);
        }

        // Analyze performance degradation
        let performance_degradation = if overall_performance_metrics.len() >= 2 {
            let first_throughput = overall_performance_metrics[0].1;
            let last_throughput = overall_performance_metrics.last().unwrap().1;
            (first_throughput - last_throughput) / first_throughput
        } else {
            0.0
        };

        // Performance should not degrade too much with increased load
        assert!(performance_degradation < 0.7,
               "Performance degradation should be reasonable: {:.1}%", performance_degradation * 100.0);

        println!("✓ High throughput document processing test passed:");
        println!("  - Performance degradation: {:.1}%", performance_degradation * 100.0);
        
        for (batch_size, throughput, successful) in overall_performance_metrics {
            println!("  - Batch {}: {:.2} docs/sec, {} successful", batch_size, throughput, successful);
        }
    }

    #[tokio::test]
    async fn test_concurrent_user_simulation() {
        let mut system = PerformanceTestSystem::new().await;

        // Pre-populate system with some data
        let initial_documents = vec![
            "Einstein developed theories that revolutionized physics and our understanding of the universe.",
            "Quantum mechanics describes the behavior of matter and energy at the atomic scale.",
            "Modern technology relies on scientific principles discovered over centuries of research.",
            "Artificial intelligence and machine learning are transforming various industries.",
            "Space exploration has expanded human knowledge and technological capabilities."
        ];

        for (i, doc) in initial_documents.iter().enumerate() {
            system.process_single_document(&format!("initial_{}", i), doc).await
                .expect("Failed to process initial document");
        }

        // Simulate concurrent users with different query patterns
        let concurrent_scenarios = vec![
            (5, "light_load"),
            (15, "medium_load"), 
            (25, "heavy_load"),
        ];

        for (user_count, load_type) in concurrent_scenarios {
            let concurrency_start = Instant::now();

            // Generate diverse queries for concurrent users
            let concurrent_queries: Vec<String> = (0..user_count).map(|user_id| {
                match user_id % 4 {
                    0 => format!("What can you tell me about scientific discoveries? User {}", user_id),
                    1 => format!("How do modern technologies work? Query from user {}", user_id),
                    2 => format!("What are the connections between physics and technology? User {} asking", user_id),
                    _ => format!("Explain the relationship between theory and practice in science for user {}", user_id),
                }
            }).collect();

            // Execute concurrent queries
            let query_results = system.perform_concurrent_queries(concurrent_queries).await
                .expect(&format!("Failed to execute concurrent queries for {}", load_type));

            let concurrency_time = concurrency_start.elapsed();

            // Validate concurrent performance
            let successful_queries = query_results.iter().filter(|q| q.success).count();
            let success_rate = successful_queries as f64 / query_results.len() as f64;

            assert!(success_rate >= 0.7,
                   "Should maintain good success rate under {} load: {:.1}%", 
                   load_type, success_rate * 100.0);

            assert!(concurrency_time < Duration::from_secs(30),
                   "Concurrent queries should complete within reasonable time for {}: {:?}", 
                   load_type, concurrency_time);

            // Check individual query performance
            let avg_query_time: Duration = query_results.iter()
                .map(|q| q.processing_time)
                .sum::<Duration>() / query_results.len() as u32;

            assert!(avg_query_time < Duration::from_secs(5),
                   "Average query time should be reasonable under {}: {:?}", 
                   load_type, avg_query_time);

            let avg_confidence: f32 = query_results.iter()
                .filter(|q| q.success)
                .map(|q| q.confidence)
                .sum::<f32>() / successful_queries as f32;

            // Quality should remain reasonable under load
            assert!(avg_confidence > 0.3,
                   "Average confidence should remain reasonable under {}: {:.3}", 
                   load_type, avg_confidence);

            println!("✓ Concurrent user simulation {}: {} users, {:.1}% success rate, avg time: {:?}, avg confidence: {:.3}",
                    load_type, user_count, success_rate * 100.0, avg_query_time, avg_confidence);
        }
    }

    #[tokio::test]
    async fn test_memory_usage_under_load() {
        let mut system = PerformanceTestSystem::new().await;

        // Monitor memory usage as load increases
        let load_increments = vec![10, 30, 60, 100];
        let mut memory_progression = Vec::new();

        for increment in load_increments {
            let load_start = Instant::now();

            // Add documents to increase memory usage
            let documents: Vec<String> = (0..increment).map(|i| {
                format!("Memory test document {} contains detailed information about scientific research topic {} with extensive analysis of methodology {} and comprehensive conclusions about findings {} in field {}.",
                       i, i % 8, i % 5, i % 6, i % 4)
            }).collect();

            // Process documents
            for (doc_idx, document) in documents.iter().enumerate() {
                let doc_id = format!("memory_test_{}_{}", increment, doc_idx);
                system.process_single_document(&doc_id, document).await
                    .expect("Failed to process memory test document");
            }

            // Get system metrics
            let metrics = system.get_system_metrics().await;
            let load_time = load_start.elapsed();

            // Calculate memory efficiency
            let memory_per_entity = if metrics.total_entities > 0 {
                metrics.memory_usage_bytes as f64 / metrics.total_entities as f64
            } else {
                0.0
            };

            memory_progression.push((increment, metrics.memory_usage_bytes, memory_per_entity, load_time));

            // Validate memory usage is reasonable
            assert!(memory_per_entity < 50_000.0, // 50KB per entity seems reasonable
                   "Memory per entity should be reasonable at load {}: {:.0} bytes", 
                   increment, memory_per_entity);

            assert!(metrics.memory_usage_bytes < 500_000_000, // 500MB total limit
                   "Total memory usage should be reasonable at load {}: {} bytes", 
                   increment, metrics.memory_usage_bytes);

            println!("✓ Memory test load {}: {} entities, {} bytes total, {:.0} bytes/entity, time: {:?}",
                    increment, metrics.total_entities, metrics.memory_usage_bytes, memory_per_entity, load_time);
        }

        // Analyze memory growth patterns
        if memory_progression.len() >= 2 {
            let initial_memory = memory_progression[0].1 as f64;
            let final_memory = memory_progression.last().unwrap().1 as f64;
            let memory_growth_factor = final_memory / initial_memory.max(1.0);

            let initial_load = memory_progression[0].0 as f64;
            let final_load = memory_progression.last().unwrap().0 as f64;
            let load_growth_factor = final_load / initial_load;

            // Memory growth should be roughly linear with load
            let growth_efficiency = memory_growth_factor / load_growth_factor;

            assert!(growth_efficiency < 2.0,
                   "Memory growth should be roughly linear with load: {:.2}", growth_efficiency);

            println!("✓ Memory usage analysis:");
            println!("  - Memory growth factor: {:.2}x", memory_growth_factor);
            println!("  - Load growth factor: {:.2}x", load_growth_factor);
            println!("  - Growth efficiency: {:.2}", growth_efficiency);
        }
    }

    #[tokio::test]
    async fn test_query_performance_with_large_dataset() {
        let mut system = PerformanceTestSystem::new().await;

        // Build large dataset
        let dataset_sizes = vec![50, 150, 300];
        let base_content = "Scientific research paper discusses methodology and findings related to";

        for dataset_size in dataset_sizes {
            let dataset_start = Instant::now();

            // Populate large dataset
            for i in 0..dataset_size {
                let document = format!("{} topic {} using approach {} with results indicating {}.",
                                     base_content, i % 12, i % 8, i % 15);
                let doc_id = format!("large_dataset_{}_{}", dataset_size, i);
                
                system.process_single_document(&doc_id, &document).await
                    .expect("Failed to process large dataset document");
            }

            let dataset_build_time = dataset_start.elapsed();

            // Test query performance on large dataset
            let query_performance_start = Instant::now();

            let test_queries = vec![
                "What research methodologies are commonly used?",
                "What topics are covered in the research papers?",
                "How do different approaches compare in terms of results?",
                "What patterns emerge from the scientific findings?",
            ];

            let query_results = system.perform_concurrent_queries(test_queries).await
                .expect("Failed to perform queries on large dataset");

            let query_performance_time = query_performance_start.elapsed();

            // Validate query performance scales acceptably
            let avg_query_time: Duration = query_results.iter()
                .map(|q| q.processing_time)
                .sum::<Duration>() / query_results.len() as u32;

            let query_time_threshold = match dataset_size {
                50 => Duration::from_secs(3),
                150 => Duration::from_secs(8),
                300 => Duration::from_secs(15),
                _ => Duration::from_secs(20),
            };

            assert!(avg_query_time < query_time_threshold,
                   "Query performance should scale acceptably with dataset size {}: {:?} < {:?}", 
                   dataset_size, avg_query_time, query_time_threshold);

            let successful_queries = query_results.iter().filter(|q| q.success).count();
            let success_rate = successful_queries as f64 / query_results.len() as f64;

            assert!(success_rate >= 0.75,
                   "Query success rate should remain high with large dataset {}: {:.1}%", 
                   dataset_size, success_rate * 100.0);

            // Get final system state
            let final_metrics = system.get_system_metrics().await;

            println!("✓ Large dataset test (size {}): build time: {:?}, avg query time: {:?}, success rate: {:.1}%",
                    dataset_size, dataset_build_time, avg_query_time, success_rate * 100.0);
            println!("  Final system: {} entities, {} relationships, {} MB memory",
                    final_metrics.total_entities, final_metrics.total_relationships, 
                    final_metrics.memory_usage_bytes / 1024 / 1024);
        }
    }

    #[tokio::test]
    async fn test_system_stability_and_recovery() {
        let mut system = PerformanceTestSystem::new().await;

        // Test system stability under sustained load
        let stability_test_duration = Duration::from_secs(30); // Shorter for testing
        let stability_start = Instant::now();

        let mut stability_metrics = Vec::new();
        let mut cycle_count = 0;

        while stability_start.elapsed() < stability_test_duration {
            let cycle_start = Instant::now();

            // Simulate mixed load: documents + queries
            let documents = vec![
                format!("Stability test document {} at timestamp {}", cycle_count, chrono::Utc::now().timestamp()),
                format!("Additional test content {} for cycle {}", cycle_count * 2, cycle_count),
            ];

            // Process documents
            for (doc_idx, document) in documents.iter().enumerate() {
                let doc_id = format!("stability_{}_{}", cycle_count, doc_idx);
                match system.process_single_document(&doc_id, document).await {
                    Ok(_) => {},
                    Err(e) => {
                        println!("Document processing error in cycle {}: {}", cycle_count, e);
                    }
                }
            }

            // Execute queries
            let queries = vec![
                format!("Query {} about stability test content", cycle_count),
                format!("Find information related to cycle {}", cycle_count),
            ];

            match system.perform_concurrent_queries(queries).await {
                Ok(results) => {
                    let successful_queries = results.iter().filter(|q| q.success).count();
                    stability_metrics.push((cycle_count, successful_queries, results.len()));
                },
                Err(e) => {
                    println!("Query execution error in cycle {}: {}", cycle_count, e);
                    stability_metrics.push((cycle_count, 0, 2));
                }
            }

            let cycle_time = cycle_start.elapsed();
            cycle_count += 1;

            // Brief pause to prevent overwhelming the system
            if cycle_time < Duration::from_millis(500) {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }

        let total_stability_time = stability_start.elapsed();

        // Analyze stability metrics
        let total_successful_queries: usize = stability_metrics.iter().map(|(_, success, _)| success).sum();
        let total_attempted_queries: usize = stability_metrics.iter().map(|(_, _, total)| total).sum();
        let overall_success_rate = total_successful_queries as f64 / total_attempted_queries as f64;

        // System should maintain reasonable stability
        assert!(overall_success_rate >= 0.6,
               "System should maintain stability under sustained load: {:.1}% success rate", 
               overall_success_rate * 100.0);

        assert!(cycle_count >= 10,
               "Should complete reasonable number of cycles: {}", cycle_count);

        // Check final system state
        let final_metrics = system.get_system_metrics().await;

        assert!(final_metrics.total_entities > 0,
               "System should maintain data after stability test");

        // Test recovery by running a final complex query
        let recovery_start = Instant::now();
        let recovery_queries = vec![
            "Summarize all the stability test information".to_string(),
            "What patterns emerged during the stability testing?".to_string(),
        ];

        let recovery_results = system.perform_concurrent_queries(recovery_queries).await
            .expect("Recovery queries should succeed");

        let recovery_time = recovery_start.elapsed();

        let recovery_success_rate = recovery_results.iter().filter(|q| q.success).count() as f64 / recovery_results.len() as f64;

        assert!(recovery_success_rate >= 0.5,
               "System should recover and respond to queries after stability test: {:.1}%", 
               recovery_success_rate * 100.0);

        assert!(recovery_time < Duration::from_secs(10),
               "Recovery queries should complete promptly: {:?}", recovery_time);

        println!("✓ System stability and recovery test passed:");
        println!("  - Test duration: {:?}", total_stability_time);
        println!("  - Cycles completed: {}", cycle_count);
        println!("  - Overall success rate: {:.1}%", overall_success_rate * 100.0);
        println!("  - Recovery success rate: {:.1}%", recovery_success_rate * 100.0);
        println!("  - Recovery time: {:?}", recovery_time);
        println!("  - Final system: {} entities, {} relationships", 
                final_metrics.total_entities, final_metrics.total_relationships);
    }
}