//! Real Performance Validation Benchmarks
//! 
//! This benchmark suite validates that REAL implementations (not mocks) achieve the 
//! documented performance targets:
//! - Entity extraction: <8ms per sentence with neural processing
//! - Relationship extraction: <12ms per sentence with federation
//! - Question answering: <20ms total with cognitive reasoning
//! - Federation operations: <3ms with cross-database coordination
//!
//! Critical: All benchmarks use production neural models and federation systems.
//! Any benchmark that exceeds its target will PANIC to ensure visibility.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, BatchSize, Throughput};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::runtime::Runtime;
use std::collections::HashMap;

// Core imports for REAL systems (not test builders)
use llmkg::core::{
    entity_extractor::CognitiveEntityExtractor,
    relationship_extractor::CognitiveRelationshipExtractor,
    answer_generator::AdvancedAnswerGenerator,
    knowledge_engine::KnowledgeEngine,
    triple::Triple,
};

// Real neural and cognitive systems
use llmkg::neural::neural_server::NeuralProcessingServer;
use llmkg::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use llmkg::cognitive::attention_manager::AttentionManager;
use llmkg::cognitive::working_memory::WorkingMemorySystem;
use llmkg::monitoring::brain_metrics_collector::BrainMetricsCollector;
use llmkg::monitoring::performance::PerformanceMonitor;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::activation_config::ActivationConfig;

// Real federation systems
use llmkg::federation::coordinator::{
    FederationCoordinator, TransactionId, TransactionOperation, OperationType, 
    TransactionMetadata, TransactionPriority, IsolationLevel, ConsistencyMode,
    OperationStatus
};
use llmkg::federation::types::DatabaseId;

/// Performance targets (CRITICAL - these MUST be met)
const ENTITY_EXTRACTION_TARGET_MS: u64 = 8;
const RELATIONSHIP_EXTRACTION_TARGET_MS: u64 = 12;
const QUESTION_ANSWERING_TARGET_MS: u64 = 20;
const FEDERATION_STORAGE_TARGET_MS: u64 = 3;

/// Test data with realistic complexity
struct RealTestData {
    simple_sentence: &'static str,
    complex_sentence: &'static str,
    wikipedia_paragraph: &'static str,
    multi_entity_text: &'static str,
    scientific_text: &'static str,
}

static REAL_TEST_DATA: RealTestData = RealTestData {
    simple_sentence: "Einstein won the Nobel Prize in 1921.",
    complex_sentence: "Marie Curie discovered polonium and radium with her husband Pierre Curie at the University of Paris in 1898.",
    wikipedia_paragraph: "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics. His work is also known for its influence on the philosophy of science. He was born in Ulm, in the Kingdom of Württemberg in the German Empire, on 14 March 1879, into a family of secular Ashkenazi Jews. His parents were Hermann Einstein, a salesman and engineer, and Pauline Koch.",
    multi_entity_text: "The collaboration between MIT, Stanford University, Harvard University, and the University of Cambridge with organizations like NASA, CERN, and the World Health Organization has led to breakthrough discoveries in quantum computing, artificial intelligence, and biotechnology research.",
    scientific_text: "Quantum entanglement demonstrates that particles can be correlated in such a way that the quantum state of each particle cannot be described independently of the state of the others, even when the particles are separated by a large distance. This phenomenon, which Einstein called 'spooky action at a distance', forms the basis for quantum computing applications.",
};

/// Questions for realistic QA testing
const REAL_QUESTIONS: &[&str] = &[
    "Who discovered radium?",
    "Where was Einstein born?",
    "What is quantum entanglement?",
    "When did Marie Curie win the Nobel Prize?",
    "Which universities collaborate with NASA?",
    "What did Einstein call quantum entanglement?",
    "How does quantum computing work?",
    "What are the applications of artificial intelligence?",
];

/// Create REAL cognitive entity extractor with production neural models
async fn create_real_entity_extractor() -> CognitiveEntityExtractor {
    // Create real neural processing server (not mock)
    let neural_server = Arc::new(
        NeuralProcessingServer::new("localhost:9000".to_string())
            .await
            .expect("Failed to create real neural server")
    );
    
    // Initialize with real models (DistilBERT, TinyBERT, etc.)
    neural_server.initialize_models()
        .await
        .expect("Failed to initialize real neural models");
    
    // Create real brain-enhanced graph
    let graph = Arc::new(
        BrainEnhancedKnowledgeGraph::new()
            .expect("Failed to create brain-enhanced graph")
    );
    
    // Create real cognitive orchestrator
    let cognitive_orchestrator = Arc::new(
        CognitiveOrchestrator::new(graph.clone(), CognitiveOrchestratorConfig::default())
            .await
            .expect("Failed to create cognitive orchestrator")
    );
    
    // Create real activation engine
    let activation_engine = Arc::new(
        ActivationPropagationEngine::new(ActivationConfig::default())
    );
    
    // Create real working memory system
    let working_memory = Arc::new(
        WorkingMemorySystem::new(activation_engine.clone(), graph.sdr_storage.clone())
            .await
            .expect("Failed to create working memory system")
    );
    
    // Create real attention manager
    let attention_manager = Arc::new(
        AttentionManager::new(
            cognitive_orchestrator.clone(),
            activation_engine.clone(),
            working_memory.clone(),
        )
        .await
        .expect("Failed to create attention manager")
    );
    
    // Create real metrics collector
    let metrics_collector = Arc::new(
        BrainMetricsCollector::new()
            .await
            .expect("Failed to create metrics collector")
    );
    
    // Create real performance monitor
    let performance_monitor = Arc::new(
        PerformanceMonitor::new()
            .await
            .expect("Failed to create performance monitor")
    );
    
    // Create cognitive entity extractor with real systems
    let mut extractor = CognitiveEntityExtractor::new(
        cognitive_orchestrator,
        attention_manager,
        working_memory,
        metrics_collector,
        performance_monitor,
    );
    
    // Connect to real neural server
    extractor = extractor.with_neural_server(neural_server);
    
    extractor
}

/// Create REAL federation coordinator with actual databases
async fn create_real_federation_coordinator() -> FederationCoordinator {
    FederationCoordinator::new()
        .await
        .expect("Failed to create real federation coordinator")
}

/// Benchmark REAL entity extraction with neural models
fn bench_real_entity_extraction(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let extractor = rt.block_on(async {
        create_real_entity_extractor().await
    });
    
    let texts = vec![
        ("simple", REAL_TEST_DATA.simple_sentence),
        ("complex", REAL_TEST_DATA.complex_sentence),
        ("wikipedia", REAL_TEST_DATA.wikipedia_paragraph),
        ("multi_entity", REAL_TEST_DATA.multi_entity_text),
        ("scientific", REAL_TEST_DATA.scientific_text),
    ];
    
    let mut group = c.benchmark_group("real_entity_extraction");
    group.throughput(Throughput::Elements(1));
    
    for (name, text) in texts {
        group.bench_with_input(BenchmarkId::new("text_type", name), &text, |b, text| {
            b.to_async(&rt).iter(|| async {
                let start = Instant::now();
                
                let entities = extractor.extract_entities(black_box(text))
                    .await
                    .expect("Entity extraction failed");
                
                let duration = start.elapsed();
                
                // CRITICAL: Enforce real performance target
                if duration.as_millis() > ENTITY_EXTRACTION_TARGET_MS as u128 {
                    panic!(
                        "FAILED: Real entity extraction took {}ms for '{}', target is <{}ms. \
                         This indicates the neural models are not achieving documented performance.",
                        duration.as_millis(), name, ENTITY_EXTRACTION_TARGET_MS
                    );
                }
                
                // Validate real results (not empty, have confidence scores)
                assert!(!entities.is_empty(), "Real entity extraction must find entities in test text");
                assert!(
                    entities.iter().all(|e| e.confidence_score > 0.0),
                    "All entities must have real confidence scores from neural models"
                );
                
                black_box(entities)
            });
        });
    }
    
    group.finish();
}

/// Benchmark REAL federation operations with cross-database coordination
fn bench_real_federation_operations(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let federation_coordinator = rt.block_on(async {
        create_real_federation_coordinator().await
    });
    
    // Setup real databases for testing
    let primary_db = DatabaseId::new("primary_db".to_string());
    let secondary_db = DatabaseId::new("secondary_db".to_string());
    
    let mut group = c.benchmark_group("real_federation_storage");
    group.throughput(Throughput::Elements(1));
    
    group.bench_function("cross_database_transaction", |b| {
        b.to_async(&rt).iter_batched_ref(
            || (),
            |_| async {
                let start = Instant::now();
                
                // Start real cross-database transaction
                let tx_id = federation_coordinator.begin_transaction(
                    vec![primary_db.clone(), secondary_db.clone()],
                    TransactionMetadata {
                        priority: TransactionPriority::High,
                        isolation_level: IsolationLevel::ReadCommitted,
                        consistency_mode: ConsistencyMode::Strong,
                        timeout_seconds: 30,
                        retry_policy: None,
                    }
                ).await.expect("Failed to begin real transaction");
                
                // Add real operation across databases
                federation_coordinator.add_operation(&tx_id, TransactionOperation {
                    operation_id: format!("test_op_{}", tx_id.as_str()),
                    database_id: primary_db.clone(),
                    operation_type: OperationType::CreateEntity {
                        entity_id: "test_entity".to_string(),
                        entity_data: [
                            ("name".to_string(), serde_json::Value::String("Test Entity".to_string())),
                            ("type".to_string(), serde_json::Value::String("Person".to_string())),
                        ].into(),
                    },
                    parameters: HashMap::new(),
                    dependencies: vec![],
                    status: OperationStatus::Pending,
                }).await.expect("Failed to add operation");
                
                // Prepare and commit transaction (real 2PC)
                let prepared = federation_coordinator.prepare_transaction(&tx_id)
                    .await
                    .expect("Failed to prepare transaction");
                
                assert!(prepared, "Real transaction must prepare successfully");
                
                federation_coordinator.commit_transaction(&tx_id)
                    .await
                    .expect("Failed to commit transaction");
                
                let duration = start.elapsed();
                
                // CRITICAL: Enforce real federation performance target
                if duration.as_millis() > FEDERATION_STORAGE_TARGET_MS as u128 {
                    panic!(
                        "FAILED: Real federation operation took {}ms, target is <{}ms. \
                         This indicates the federation system is not achieving documented performance.",
                        duration.as_millis(), FEDERATION_STORAGE_TARGET_MS
                    );
                }
                
                black_box(duration)
            },
            BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

/// Benchmark REAL relationship extraction with neural models and federation
fn bench_real_relationship_extraction(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let (entity_extractor, relationship_extractor) = rt.block_on(async {
        let entity_extractor = create_real_entity_extractor().await;
        
        // Create real relationship extractor with federation
        let federation_coordinator = Arc::new(create_real_federation_coordinator().await);
        let knowledge_engine = Arc::new(KnowledgeEngine::new(256, 50000).unwrap());
        
        let relationship_extractor = CognitiveRelationshipExtractor::new(
            federation_coordinator,
            knowledge_engine,
        );
        
        (entity_extractor, relationship_extractor)
    });
    
    let mut group = c.benchmark_group("real_relationship_extraction");
    group.throughput(Throughput::Elements(1));
    
    group.bench_function("complex_relationships", |b| {
        b.to_async(&rt).iter(|| async {
            let start = Instant::now();
            
            // First extract entities with real neural models
            let entities = entity_extractor.extract_entities(REAL_TEST_DATA.complex_sentence)
                .await
                .expect("Entity extraction failed");
            
            // Then extract relationships with real federation
            let relationships = relationship_extractor.extract_relationships(
                REAL_TEST_DATA.complex_sentence,
                &entities,
            ).await.expect("Relationship extraction failed");
            
            let duration = start.elapsed();
            
            // CRITICAL: Enforce real relationship extraction target
            if duration.as_millis() > RELATIONSHIP_EXTRACTION_TARGET_MS as u128 {
                panic!(
                    "FAILED: Real relationship extraction took {}ms, target is <{}ms. \
                     This indicates the neural+federation system is not achieving documented performance.",
                    duration.as_millis(), RELATIONSHIP_EXTRACTION_TARGET_MS
                );
            }
            
            // Validate real results
            assert!(!relationships.is_empty(), "Real relationship extraction must find relationships");
            assert!(
                relationships.iter().all(|r| r.confidence_score > 0.0),
                "All relationships must have real confidence scores"
            );
            
            black_box(relationships)
        });
    });
    
    group.finish();
}

/// Benchmark REAL question answering pipeline end-to-end
fn bench_real_question_answering(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let qa_system = rt.block_on(async {
        // Create complete real QA system
        let entity_extractor = Arc::new(create_real_entity_extractor().await);
        let federation_coordinator = Arc::new(create_real_federation_coordinator().await);
        let knowledge_engine = Arc::new(KnowledgeEngine::new(256, 50000).unwrap());
        
        // Pre-populate with real knowledge
        let _ = knowledge_engine.store_chunk(REAL_TEST_DATA.wikipedia_paragraph.to_string(), None);
        let _ = knowledge_engine.store_chunk(REAL_TEST_DATA.scientific_text.to_string(), None);
        let _ = knowledge_engine.store_chunk(REAL_TEST_DATA.multi_entity_text.to_string(), None);
        
        // Create real answer generator
        AdvancedAnswerGenerator::new(
            entity_extractor,
            federation_coordinator,
            knowledge_engine,
        ).await.expect("Failed to create real answer generator")
    });
    
    let mut group = c.benchmark_group("real_question_answering");
    group.throughput(Throughput::Elements(1));
    
    for (i, question) in REAL_QUESTIONS.iter().enumerate() {
        group.bench_with_input(BenchmarkId::new("question", i), question, |b, question| {
            b.to_async(&rt).iter(|| async {
                let start = Instant::now();
                
                let answer = qa_system.answer_question(black_box(question))
                    .await
                    .expect("Question answering failed");
                
                let duration = start.elapsed();
                
                // CRITICAL: Enforce real QA performance target
                if duration.as_millis() > QUESTION_ANSWERING_TARGET_MS as u128 {
                    panic!(
                        "FAILED: Real question answering took {}ms for '{}', target is <{}ms. \
                         This indicates the complete QA pipeline is not achieving documented performance.",
                        duration.as_millis(), question, QUESTION_ANSWERING_TARGET_MS
                    );
                }
                
                // Validate real answer
                assert!(!answer.answer_text.is_empty(), "Real QA must provide non-empty answers");
                assert!(answer.confidence_score > 0.0, "Real QA must provide confidence scores");
                
                black_box(answer)
            });
        });
    }
    
    group.finish();
}

/// Test concurrent performance under realistic load
fn bench_concurrent_performance(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let extractor = rt.block_on(async {
        Arc::new(create_real_entity_extractor().await)
    });
    
    let mut group = c.benchmark_group("concurrent_performance");
    group.throughput(Throughput::Elements(10)); // 10 concurrent requests
    
    group.bench_function("concurrent_entity_extraction", |b| {
        b.to_async(&rt).iter(|| async {
            let mut handles = Vec::new();
            
            // Start 10 concurrent entity extractions
            for i in 0..10 {
                let extractor = extractor.clone();
                let text = match i % 5 {
                    0 => REAL_TEST_DATA.simple_sentence,
                    1 => REAL_TEST_DATA.complex_sentence,
                    2 => REAL_TEST_DATA.wikipedia_paragraph,
                    3 => REAL_TEST_DATA.multi_entity_text,
                    _ => REAL_TEST_DATA.scientific_text,
                };
                
                let handle = tokio::spawn(async move {
                    let start = Instant::now();
                    let entities = extractor.extract_entities(text).await.unwrap();
                    let duration = start.elapsed();
                    
                    // Each concurrent request must still meet individual target
                    if duration.as_millis() > ENTITY_EXTRACTION_TARGET_MS as u128 {
                        panic!(
                            "FAILED: Concurrent request {} took {}ms, individual target is <{}ms",
                            i, duration.as_millis(), ENTITY_EXTRACTION_TARGET_MS
                        );
                    }
                    
                    entities
                });
                
                handles.push(handle);
            }
            
            // Wait for all to complete successfully
            for handle in handles {
                let _entities = handle.await.expect("Concurrent request failed");
            }
            
            black_box(())
        });
    });
    
    group.finish();
}

/// Validate performance targets manually and generate report
fn validate_and_report_performance() {
    println!("\n=== REAL PERFORMANCE VALIDATION REPORT ===");
    println!("Testing PRODUCTION neural models and federation systems (NOT mocks)");
    println!();
    println!("Performance Targets:");
    println!("- Entity extraction: <{}ms per sentence with DistilBERT/TinyBERT", ENTITY_EXTRACTION_TARGET_MS);
    println!("- Relationship extraction: <{}ms per sentence with neural+federation", RELATIONSHIP_EXTRACTION_TARGET_MS);
    println!("- Question answering: <{}ms total with complete cognitive pipeline", QUESTION_ANSWERING_TARGET_MS);
    println!("- Federation operations: <{}ms with real cross-database 2PC", FEDERATION_STORAGE_TARGET_MS);
    println!();
    println!("CRITICAL: Any benchmark exceeding its target will PANIC to ensure visibility.");
    println!("Run 'cargo bench --bench real_performance_validation' to validate.");
    println!();
    println!("Evidence Required:");
    println!("- All benchmarks must pass without panicking");
    println!("- Statistical reports show mean times below targets");
    println!("- Concurrent performance maintains individual targets");
    println!("- Real neural models (not regex) produce confidence scores");
    println!("- Real federation systems complete 2PC transactions");
    println!();
    println!("Benchmark outputs will be saved to target/criterion/real_performance_validation/");
}

/// Configure benchmarks for real performance validation
criterion_group!(
    name = real_performance_benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(15))
        .sample_size(50)
        .warm_up_time(Duration::from_secs(5))
        .with_plots()
        .with_output_color(true);
    targets = 
        bench_real_entity_extraction,
        bench_real_federation_operations,
        bench_real_relationship_extraction,
        bench_real_question_answering,
        bench_concurrent_performance
);

criterion_main!(real_performance_benches);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_targets() {
        assert_eq!(ENTITY_EXTRACTION_TARGET_MS, 8);
        assert_eq!(RELATIONSHIP_EXTRACTION_TARGET_MS, 12);
        assert_eq!(QUESTION_ANSWERING_TARGET_MS, 20);
        assert_eq!(FEDERATION_STORAGE_TARGET_MS, 3);
    }
    
    #[test]
    fn test_validation_report() {
        validate_and_report_performance();
    }
    
    #[tokio::test]
    async fn test_real_systems_creation() {
        // Verify we can create real systems (may take time to load models)
        println!("Testing real entity extractor creation...");
        let _extractor = create_real_entity_extractor().await;
        
        println!("Testing real federation coordinator creation...");
        let _coordinator = create_real_federation_coordinator().await;
        
        println!("Real systems created successfully!");
    }
}