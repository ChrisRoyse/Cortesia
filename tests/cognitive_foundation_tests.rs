//! Comprehensive Cognitive Integration Test Suite
//! Tests all cognitive systems integration, federation operations, neural processing,
//! and validates performance targets as specified in documentation lines 841-940

use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio;

use llmkg::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use llmkg::cognitive::attention_manager::{AttentionManager, AttentionType};
use llmkg::cognitive::working_memory::{WorkingMemorySystem, MemoryContent, BufferType};
use llmkg::cognitive::{
    CognitivePatternType, ReasoningStrategy, QueryContext, PatternParameters, ValidationLevel
};
use llmkg::neural::neural_server::{NeuralProcessingServer, NeuralRequest, NeuralOperation, NeuralParameters};
use llmkg::federation::coordinator::{FederationCoordinator, TransactionId, CrossDatabaseTransaction};
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::entity_extractor::EntityExtractor;
use llmkg::core::relationship_extractor::RelationshipExtractor;
use llmkg::core::question_parser::QuestionParser;
use llmkg::core::answer_generator::AnswerGenerator;
use llmkg::test_support::builders::{
    AttentionManagerBuilder, CognitiveOrchestratorBuilder, WorkingMemoryBuilder,
    QueryContextBuilder, PatternParametersBuilder
};
use llmkg::test_support::fixtures::create_test_graph;
use llmkg::test_support::data::create_test_entities_and_relationships;
use llmkg::error::Result;

/// Performance timer for validating cognitive integration timing requirements
struct PerformanceTimer {
    start: Instant,
    operation: String,
}

impl PerformanceTimer {
    fn new(operation: &str) -> Self {
        Self {
            start: Instant::now(),
            operation: operation.to_string(),
        }
    }
    
    fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
    
    fn assert_within_ms(&self, max_ms: f64) {
        let elapsed = self.elapsed_ms();
        assert!(
            elapsed <= max_ms,
            "{} took {:.2}ms, expected less than {:.2}ms",
            self.operation,
            elapsed,
            max_ms
        );
        println!("✓ {} completed in {:.2}ms (target: <{:.2}ms)", 
                 self.operation, elapsed, max_ms);
    }
}

/// Test data structure for comprehensive integration testing
struct CognitiveTestSuite {
    orchestrator: Arc<CognitiveOrchestrator>,
    neural_server: Arc<NeuralProcessingServer>,
    federation_coordinator: Arc<FederationCoordinator>,
    attention_manager: Arc<AttentionManager>,
    working_memory: Arc<WorkingMemorySystem>,
    entity_extractor: Arc<EntityExtractor>,
    relationship_extractor: Arc<RelationshipExtractor>,
    question_parser: Arc<QuestionParser>,
    answer_generator: Arc<AnswerGenerator>,
}

impl CognitiveTestSuite {
    /// Creates a fully integrated cognitive test suite with all systems initialized
    async fn new() -> Result<Self> {
        let graph = create_test_graph();
        
        // Initialize cognitive orchestrator with full configuration
        let orchestrator_config = CognitiveOrchestratorConfig {
            enable_adaptive_selection: true,
            enable_ensemble_methods: true,
            default_timeout_ms: 5000,
            max_parallel_patterns: 8,
            performance_tracking: true,
        };
        let orchestrator = Arc::new(
            CognitiveOrchestrator::new(graph.clone(), orchestrator_config).await?
        );
        
        // Initialize neural processing server
        let neural_server = Arc::new(NeuralProcessingServer::new().await?);
        
        // Initialize federation coordinator
        // Federation coordinator would need registry, using mock for now
        // let federation_coordinator = Arc::new(FederationCoordinator::new(registry).await?);
        
        // Initialize attention manager with cognitive enhancement
        let attention_manager = Arc::new(
            AttentionManagerBuilder::new()
                .with_graph(graph.clone())
                .build()
                .await?
        );
        
        // Initialize working memory system
        let working_memory = Arc::new(
            WorkingMemoryBuilder::new()
                .with_graph(graph.clone())
                .build()
                .await?
        );
        
        // Initialize core extraction and processing components
        let entity_extractor = Arc::new(EntityExtractor::new(
            graph.clone(),
            Some(neural_server.clone()),
            Some(orchestrator.clone())
        ));
        
        let relationship_extractor = Arc::new(RelationshipExtractor::new(
            graph.clone(),
            Some(neural_server.clone()),
            Some(federation_coordinator.clone())
        ));
        
        let question_parser = Arc::new(QuestionParser::new(
            orchestrator.clone(),
            attention_manager.clone()
        ));
        
        let answer_generator = Arc::new(AnswerGenerator::new(
            orchestrator.clone(),
            working_memory.clone(),
            neural_server.clone()
        ));
        
        Ok(Self {
            orchestrator,
            neural_server,
            federation_coordinator,
            attention_manager,
            working_memory,
            entity_extractor,
            relationship_extractor,
            question_parser,
            answer_generator,
        })
    }
}

#[cfg(test)]
mod cognitive_integration_tests {
    use super::*;

    /// Test comprehensive cognitive entity extraction with neural processing and attention management
    /// Performance Target: <8ms per sentence with neural processing
    #[tokio::test]
    async fn test_cognitive_entity_extraction() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        let test_sentences = vec![
            "Albert Einstein developed the Theory of Relativity in 1905.",
            "Marie Curie discovered radium and polonium through radioactivity research.",
            "The quantum computer at IBM processes qubits using superconducting circuits.",
            "Neural networks learn patterns through backpropagation and gradient descent.",
            "CRISPR gene editing technology allows precise DNA modifications in living cells.",
        ];
        
        for sentence in test_sentences {
            let timer = PerformanceTimer::new(&format!("cognitive entity extraction for: '{}'", sentence));
            
            // Focus attention on the sentence for cognitive enhancement
            let _ = suite.attention_manager.focus_attention_on_text(sentence, AttentionType::Focused).await;
            
            // Extract entities with cognitive enhancement
            let entities = suite.entity_extractor.extract_entities_with_cognitive_enhancement(
                sentence,
                &QueryContextBuilder::new()
                    .with_reasoning_trace(true)
                    .build()
            ).await.unwrap();
            
            // Validate performance target: <8ms per sentence
            timer.assert_within_ms(8.0);
            
            // Verify cognitive enhancement metadata
            assert!(!entities.is_empty(), "Should extract entities from: {}", sentence);
            for entity in &entities {
                assert!(entity.reasoning_pattern != CognitivePatternType::Unknown,
                       "Entity '{}' should have cognitive reasoning pattern", entity.name);
                assert!(entity.attention_weight > 0.0,
                       "Entity '{}' should have attention weight", entity.name);
                assert!(entity.confidence > 0.5,
                       "Entity '{}' should have high confidence", entity.name);
            }
            
            println!("✓ Extracted {} entities with cognitive enhancement from: '{}'", 
                     entities.len(), sentence);
        }
    }
    
    /// Test federation-aware storage with cross-database transactions
    /// Performance Target: <3ms with cross-database coordination
    #[tokio::test]
    async fn test_federation_aware_storage() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        let timer = PerformanceTimer::new("federation-aware storage operations");
        
        // Create cross-database transaction
        let transaction_id = TransactionId::new();
        let cross_db_transaction = suite.federation_coordinator
            .begin_cross_database_transaction(transaction_id.clone(), vec!["primary", "secondary"])
            .await
            .unwrap();
        
        // Test entity creation across multiple databases
        let entity_data = create_test_entities_and_relationships();
        for (entity_name, entity_info) in entity_data.entities.iter().take(5) {
            suite.federation_coordinator
                .add_entity_to_transaction(
                    &transaction_id,
                    entity_name,
                    entity_info.clone()
                )
                .await
                .unwrap();
        }
        
        // Test relationship creation with federation coordination
        for relationship in entity_data.relationships.iter().take(5) {
            suite.federation_coordinator
                .add_relationship_to_transaction(
                    &transaction_id,
                    &relationship.from_entity,
                    &relationship.to_entity,
                    &relationship.relationship_type,
                    relationship.properties.clone()
                )
                .await
                .unwrap();
        }
        
        // Commit transaction with 2-phase commit protocol
        let commit_result = suite.federation_coordinator
            .commit_transaction(transaction_id)
            .await
            .unwrap();
        
        // Validate performance target: <3ms with federation coordination
        timer.assert_within_ms(3.0);
        
        // Verify transaction consistency
        assert!(commit_result.is_committed(), "Transaction should be committed successfully");
        assert!(commit_result.all_databases_consistent(), "All databases should be consistent");
        
        println!("✓ Federation-aware storage completed with cross-database consistency");
    }
    
    /// Test neural server integration with model training and prediction
    /// Performance Target: Model inference <5ms, training convergence
    #[tokio::test]
    async fn test_neural_server_integration() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        // Test neural model training
        let training_timer = PerformanceTimer::new("neural model training");
        
        let training_request = NeuralRequest {
            operation: NeuralOperation::Train {
                dataset: "cognitive_patterns_dataset".to_string(),
                epochs: 10,
            },
            model_id: "cognitive_pattern_classifier".to_string(),
            input_data: serde_json::json!({
                "input_dim": 96,
                "output_dim": 7, // 7 cognitive patterns
                "learning_rate": 0.001
            }),
            parameters: NeuralParameters::default(),
        };
        
        let training_result = suite.neural_server
            .process_request(training_request)
            .await
            .unwrap();
        
        assert!(training_result.confidence > 0.8, "Training should achieve high confidence");
        println!("✓ Neural model training completed with {:.3} confidence", training_result.confidence);
        
        // Test neural model prediction with performance validation
        let prediction_timer = PerformanceTimer::new("neural model prediction");
        
        let test_embeddings = vec![
            vec![0.1; 96], // Test embedding 1
            vec![0.2; 96], // Test embedding 2
            vec![0.3; 96], // Test embedding 3
        ];
        
        for (i, embedding) in test_embeddings.iter().enumerate() {
            let prediction_request = NeuralRequest {
                operation: NeuralOperation::Predict {
                    input: embedding.clone(),
                },
                model_id: "cognitive_pattern_classifier".to_string(),
                input_data: serde_json::json!({ "input": embedding }),
                parameters: NeuralParameters::default(),
            };
            
            let prediction_result = suite.neural_server
                .process_request(prediction_request)
                .await
                .unwrap();
            
            // Validate prediction quality
            assert!(prediction_result.confidence > 0.5, 
                   "Prediction {} should have reasonable confidence", i);
            
            println!("✓ Neural prediction {} completed with {:.3} confidence", 
                     i, prediction_result.confidence);
        }
        
        // Validate performance target: <5ms for inference
        prediction_timer.assert_within_ms(5.0);
    }
    
    /// Test comprehensive monitoring system with cognitive metrics
    /// Validates brain metrics collection and performance tracking
    #[tokio::test]
    async fn test_comprehensive_monitoring() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        let timer = PerformanceTimer::new("comprehensive monitoring system");
        
        // Execute various cognitive operations to generate metrics
        let test_queries = vec![
            "What is quantum computing and how does it work?",
            "Explore creative applications of artificial intelligence",
            "Analyze the relationship between consciousness and computation",
            "How do neural networks learn complex patterns?",
        ];
        
        for query in test_queries {
            // Execute cognitive reasoning
            let result = suite.orchestrator
                .reason(query, None, ReasoningStrategy::Automatic)
                .await
                .unwrap();
            
            // Collect attention metrics
            let attention_metrics = suite.attention_manager
                .get_attention_metrics()
                .await
                .unwrap();
            
            // Collect working memory metrics
            let memory_metrics = suite.working_memory
                .get_memory_metrics()
                .await
                .unwrap();
            
            // Collect neural processing metrics
            let neural_metrics = suite.neural_server
                .get_processing_metrics()
                .await
                .unwrap();
            
            // Validate cognitive metrics collection
            assert!(result.quality_metrics.overall_confidence >= 0.0);
            assert!(attention_metrics.focus_strength > 0.0);
            assert!(memory_metrics.working_memory_utilization >= 0.0);
            assert!(neural_metrics.average_inference_time_ms > 0.0);
            
            println!("✓ Collected comprehensive metrics for query: '{}'", query);
        }
        
        // Test performance monitoring aggregation
        let orchestrator_performance = suite.orchestrator
            .get_performance_metrics()
            .await
            .unwrap();
        
        assert!(orchestrator_performance.total_queries_processed > 0);
        assert!(orchestrator_performance.average_response_time_ms > 0.0);
        assert!(orchestrator_performance.success_rate >= 0.0);
        
        timer.assert_within_ms(100.0); // Monitoring should be fast
        
        println!("✓ Comprehensive monitoring system validated with aggregated metrics");
    }
    
    /// Test end-to-end cognitive pipeline from text to knowledge
    /// Performance Target: <20ms total with cognitive reasoning
    #[tokio::test]
    async fn test_end_to_end_cognitive_pipeline() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        let timer = PerformanceTimer::new("end-to-end cognitive pipeline");
        
        let input_text = "Albert Einstein developed the Theory of Relativity in 1905, revolutionizing our understanding of space, time, and gravity through mathematical equations that describe the fabric of spacetime.";
        
        // Step 1: Extract entities with cognitive enhancement
        let entities = suite.entity_extractor
            .extract_entities_with_cognitive_enhancement(
                input_text,
                &QueryContextBuilder::new()
                    .with_confidence_threshold(0.7)
                    .with_reasoning_trace(true)
                    .build()
            )
            .await
            .unwrap();
        
        // Step 2: Extract relationships with federation coordination
        let relationships = suite.relationship_extractor
            .extract_relationships_with_federation(
                input_text,
                &entities,
                &PatternParametersBuilder::new()
                    .with_validation_level(ValidationLevel::Comprehensive)
                    .build()
            )
            .await
            .unwrap();
        
        // Step 3: Parse question with cognitive intent recognition
        let question = "Who developed the Theory of Relativity and when?";
        let question_intent = suite.question_parser
            .parse_with_cognitive_enhancement(
                question,
                Some(&QueryContextBuilder::new()
                    .with_domain("physics".to_string())
                    .build())
            )
            .await
            .unwrap();
        
        // Step 4: Generate answer using cognitive reasoning
        let answer = suite.answer_generator
            .generate_answer_with_cognitive_reasoning(
                &entities,
                &relationships,
                &question_intent,
                ReasoningStrategy::Ensemble(vec![
                    CognitivePatternType::Convergent,
                    CognitivePatternType::Critical,
                ])
            )
            .await
            .unwrap();
        
        // Validate performance target: <20ms total pipeline
        timer.assert_within_ms(20.0);
        
        // Verify cognitive enhancement throughout pipeline
        assert!(!entities.is_empty(), "Should extract entities");
        assert!(entities.iter().all(|e| e.reasoning_pattern != CognitivePatternType::Unknown),
               "All entities should have cognitive reasoning patterns");
        
        assert!(!relationships.is_empty(), "Should extract relationships");
        assert!(relationships.iter().all(|r| r.confidence > 0.7),
               "All relationships should have high confidence");
        
        assert!(question_intent.cognitive_context.is_some(),
               "Question should have cognitive context");
        assert!(question_intent.expected_reasoning_patterns.len() > 0,
               "Question should identify required reasoning patterns");
        
        assert!(!answer.content.is_empty(), "Should generate answer");
        assert!(answer.confidence > 0.8, "Answer should have high confidence");
        assert!(answer.reasoning_trace.is_some(), "Answer should include reasoning trace");
        
        println!("✓ End-to-end cognitive pipeline completed successfully");
        println!("  - Extracted {} entities with cognitive patterns", entities.len());
        println!("  - Extracted {} relationships with federation", relationships.len());
        println!("  - Generated answer with {:.3} confidence", answer.confidence);
    }
    
    /// Test working memory integration with attention-based retrieval
    /// Performance Target: <2ms with attention-based retrieval
    #[tokio::test]
    async fn test_working_memory_attention_integration() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        let timer = PerformanceTimer::new("working memory attention integration");
        
        // Load working memory with test concepts
        let test_concepts = vec![
            MemoryContent::Concept("quantum computing".to_string()),
            MemoryContent::Concept("artificial intelligence".to_string()),
            MemoryContent::Concept("machine learning".to_string()),
            MemoryContent::Concept("neural networks".to_string()),
            MemoryContent::Concept("deep learning".to_string()),
        ];
        
        for concept in test_concepts {
            suite.working_memory
                .store_in_buffer(concept, BufferType::Episodic, 0.8)
                .await
                .unwrap();
        }
        
        // Test attention-guided retrieval
        let attention_query = "Find concepts related to artificial intelligence";
        suite.attention_manager
            .focus_attention_on_text(attention_query, AttentionType::Selective)
            .await
            .unwrap();
        
        let retrieved_memories = suite.working_memory
            .retrieve_with_attention_guidance(
                attention_query,
                suite.attention_manager.get_current_attention_state().await.unwrap(),
                5
            )
            .await
            .unwrap();
        
        // Validate performance target: <2ms for attention-based retrieval
        timer.assert_within_ms(2.0);
        
        // Verify attention-guided retrieval quality
        assert!(!retrieved_memories.is_empty(), "Should retrieve relevant memories");
        assert!(retrieved_memories.len() <= 5, "Should respect retrieval limit");
        
        for memory in &retrieved_memories {
            assert!(memory.activation_level > 0.5, 
                   "Retrieved memories should have high activation");
            assert!(memory.attention_weight > 0.0,
                   "Retrieved memories should have attention weights");
        }
        
        println!("✓ Working memory attention integration completed");
        println!("  - Retrieved {} relevant memories with attention guidance", retrieved_memories.len());
    }
    
    /// Test cognitive pattern coordination under high load
    /// Validates pattern switching and ensemble coordination
    #[tokio::test]
    async fn test_cognitive_pattern_coordination_under_load() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        let timer = PerformanceTimer::new("cognitive pattern coordination under load");
        
        // Create multiple concurrent reasoning tasks
        let reasoning_tasks = vec![
            ("What are the creative applications of quantum computing?", 
             ReasoningStrategy::Specific(CognitivePatternType::Divergent)),
            ("Analyze the logical structure of machine learning algorithms", 
             ReasoningStrategy::Specific(CognitivePatternType::Convergent)),
            ("How might art and technology intersect in unexpected ways?", 
             ReasoningStrategy::Specific(CognitivePatternType::Lateral)),
            ("Evaluate the claims about AI consciousness from multiple perspectives", 
             ReasoningStrategy::Specific(CognitivePatternType::Critical)),
            ("What patterns emerge when examining technology adoption cycles?", 
             ReasoningStrategy::Specific(CognitivePatternType::Abstract)),
        ];
        
        // Execute tasks concurrently to test coordination
        let mut task_handles = Vec::new();
        
        for (query, strategy) in reasoning_tasks {
            let orchestrator = suite.orchestrator.clone();
            let handle = tokio::spawn(async move {
                orchestrator.reason(query, None, strategy).await
            });
            task_handles.push(handle);
        }
        
        // Wait for all tasks to complete
        let results = futures::future::join_all(task_handles).await;
        
        // Validate all tasks completed successfully
        for (i, result) in results.into_iter().enumerate() {
            let reasoning_result = result.unwrap().unwrap();
            assert!(!reasoning_result.final_answer.is_empty(), 
                   "Task {} should produce answer", i);
            assert!(reasoning_result.quality_metrics.overall_confidence > 0.0,
                   "Task {} should have confidence score", i);
        }
        
        // Validate performance under load
        timer.assert_within_ms(5000.0); // Reasonable time for concurrent tasks
        
        // Check orchestrator coordination metrics
        let coordination_metrics = suite.orchestrator
            .get_coordination_metrics()
            .await
            .unwrap();
        
        assert!(coordination_metrics.concurrent_patterns_handled >= 5,
               "Should handle multiple concurrent patterns");
        assert!(coordination_metrics.pattern_switching_efficiency > 0.7,
               "Should maintain high pattern switching efficiency");
        
        println!("✓ Cognitive pattern coordination under load completed successfully");
        println!("  - Handled {} concurrent reasoning tasks", 5);
        println!("  - Pattern switching efficiency: {:.3}", coordination_metrics.pattern_switching_efficiency);
    }
    
    /// Test federation transaction rollback and recovery
    /// Validates error recovery and data consistency
    #[tokio::test]
    async fn test_federation_transaction_rollback_recovery() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        let timer = PerformanceTimer::new("federation transaction rollback recovery");
        
        // Create transaction that will intentionally fail partway through
        let transaction_id = TransactionId::new();
        let cross_db_transaction = suite.federation_coordinator
            .begin_cross_database_transaction(transaction_id.clone(), vec!["primary", "secondary", "tertiary"])
            .await
            .unwrap();
        
        // Add operations that should succeed
        suite.federation_coordinator
            .add_entity_to_transaction(
                &transaction_id,
                "test_entity_1",
                serde_json::json!({"name": "Test Entity 1", "type": "valid"})
            )
            .await
            .unwrap();
        
        // Add operation that will cause failure (e.g., constraint violation)
        let failure_result = suite.federation_coordinator
            .add_entity_to_transaction(
                &transaction_id,
                "invalid_entity",
                serde_json::json!({"name": "", "type": "invalid"}) // Invalid data
            )
            .await;
        
        // Transaction should detect the failure and initiate rollback
        assert!(failure_result.is_err(), "Invalid operation should fail");
        
        // Verify automatic rollback
        let transaction_status = suite.federation_coordinator
            .get_transaction_status(&transaction_id)
            .await
            .unwrap();
        
        assert!(matches!(transaction_status, crate::federation::coordinator::TransactionStatus::Aborted),
               "Transaction should be automatically aborted on failure");
        
        // Verify data consistency after rollback
        let consistency_check = suite.federation_coordinator
            .verify_cross_database_consistency(vec!["primary", "secondary", "tertiary"])
            .await
            .unwrap();
        
        assert!(consistency_check.is_consistent, "Databases should be consistent after rollback");
        
        // Validate performance of rollback process
        timer.assert_within_ms(100.0); // Rollback should be fast
        
        println!("✓ Federation transaction rollback and recovery completed successfully");
        println!("  - Transaction properly rolled back on failure");
        println!("  - Data consistency maintained across all databases");
    }
    
    /// Test neural model adaptation and online learning
    /// Validates continuous learning capabilities
    #[tokio::test] 
    async fn test_neural_model_adaptation_online_learning() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        let timer = PerformanceTimer::new("neural model adaptation and online learning");
        
        // Get baseline model performance
        let baseline_request = NeuralRequest {
            operation: NeuralOperation::Predict {
                input: vec![0.5; 96],
            },
            model_id: "adaptive_cognitive_model".to_string(),
            input_data: serde_json::json!({"input": vec![0.5; 96]}),
            parameters: NeuralParameters::default(),
        };
        
        let baseline_result = suite.neural_server
            .process_request(baseline_request)
            .await
            .unwrap();
        
        let baseline_confidence = baseline_result.confidence;
        
        // Simulate online learning with new data
        let adaptation_request = NeuralRequest {
            operation: NeuralOperation::Train {
                dataset: "online_adaptation_data".to_string(),
                epochs: 5, // Quick adaptation
            },
            model_id: "adaptive_cognitive_model".to_string(),
            input_data: serde_json::json!({
                "adaptation_mode": true,
                "learning_rate": 0.01,
                "batch_size": 16
            }),
            parameters: NeuralParameters::default(),
        };
        
        let adaptation_result = suite.neural_server
            .process_request(adaptation_request)
            .await
            .unwrap();
        
        // Test model performance after adaptation
        let post_adaptation_request = NeuralRequest {
            operation: NeuralOperation::Predict {
                input: vec![0.5; 96],
            },
            model_id: "adaptive_cognitive_model".to_string(),
            input_data: serde_json::json!({"input": vec![0.5; 96]}),
            parameters: NeuralParameters::default(),
        };
        
        let post_adaptation_result = suite.neural_server
            .process_request(post_adaptation_request)
            .await
            .unwrap();
        
        // Validate adaptation effectiveness
        assert!(adaptation_result.confidence > 0.7, 
               "Adaptation training should achieve good confidence");
        
        // Model should maintain or improve performance
        let performance_change = post_adaptation_result.confidence - baseline_confidence;
        assert!(performance_change >= -0.1, // Allow small degradation
               "Model performance should not significantly degrade after adaptation");
        
        timer.assert_within_ms(2000.0); // Online learning should be reasonably fast
        
        println!("✓ Neural model adaptation and online learning completed");
        println!("  - Baseline confidence: {:.3}", baseline_confidence);
        println!("  - Post-adaptation confidence: {:.3}", post_adaptation_result.confidence);
        println!("  - Performance change: {:+.3}", performance_change);
    }
}

/// Performance validation tests for specific timing targets
#[cfg(test)]
mod performance_validation_tests {
    use super::*;
    
    /// Validates entity extraction performance target: <8ms per sentence
    #[tokio::test]
    async fn validate_entity_extraction_performance_target() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        let test_sentences = vec![
            "Simple entity test",
            "Albert Einstein developed the Theory of Relativity",
            "The quantum computer processes information using qubits",
            "Machine learning algorithms learn patterns from data through training",
            "CRISPR gene editing allows precise modifications of DNA sequences in living organisms",
        ];
        
        for sentence in test_sentences {
            let timer = PerformanceTimer::new("entity extraction performance validation");
            
            let _entities = suite.entity_extractor
                .extract_entities(sentence)
                .await
                .unwrap();
            
            // Strict performance validation: must be <8ms
            timer.assert_within_ms(8.0);
        }
        
        println!("✓ Entity extraction performance target validated: <8ms per sentence");
    }
    
    /// Validates relationship extraction performance target: <12ms per sentence
    #[tokio::test]
    async fn validate_relationship_extraction_performance_target() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        let test_sentences = vec![
            "Einstein developed relativity",
            "The scientist discovered the element through experimentation",
            "Neural networks learn patterns by adjusting weights during training",
            "Quantum computers utilize superposition and entanglement for computational advantage",
            "CRISPR technology enables researchers to edit genes with unprecedented precision and accuracy",
        ];
        
        for sentence in test_sentences {
            let timer = PerformanceTimer::new("relationship extraction performance validation");
            
            let entities = suite.entity_extractor
                .extract_entities(sentence)
                .await
                .unwrap();
            
            let _relationships = suite.relationship_extractor
                .extract_relationships(sentence, &entities)
                .await
                .unwrap();
            
            // Strict performance validation: must be <12ms
            timer.assert_within_ms(12.0);
        }
        
        println!("✓ Relationship extraction performance target validated: <12ms per sentence");
    }
    
    /// Validates question answering performance target: <20ms total
    #[tokio::test]
    async fn validate_question_answering_performance_target() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        // Pre-populate with some test knowledge
        let knowledge_text = "Albert Einstein developed the Theory of Relativity in 1905. Marie Curie discovered radium through radioactivity research.";
        
        let entities = suite.entity_extractor
            .extract_entities(knowledge_text)
            .await
            .unwrap();
        
        let relationships = suite.relationship_extractor
            .extract_relationships(knowledge_text, &entities)
            .await
            .unwrap();
        
        let test_questions = vec![
            "Who developed the Theory of Relativity?",
            "When was the Theory of Relativity developed?",
            "What did Marie Curie discover?",
            "How did Marie Curie make her discovery?",
        ];
        
        for question in test_questions {
            let timer = PerformanceTimer::new("question answering performance validation");
            
            let question_intent = suite.question_parser
                .parse(question)
                .await
                .unwrap();
            
            let _answer = suite.answer_generator
                .generate_answer(&entities, &relationships, &question_intent)
                .await
                .unwrap();
            
            // Strict performance validation: must be <20ms total
            timer.assert_within_ms(20.0);
        }
        
        println!("✓ Question answering performance target validated: <20ms total");
    }
    
    /// Validates federation storage performance target: <3ms
    #[tokio::test]
    async fn validate_federation_storage_performance_target() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        let timer = PerformanceTimer::new("federation storage performance validation");
        
        let transaction_id = TransactionId::new();
        let _transaction = suite.federation_coordinator
            .begin_cross_database_transaction(transaction_id.clone(), vec!["db1", "db2"])
            .await
            .unwrap();
        
        suite.federation_coordinator
            .add_entity_to_transaction(
                &transaction_id,
                "test_entity",
                serde_json::json!({"name": "Test", "type": "validation"})
            )
            .await
            .unwrap();
        
        suite.federation_coordinator
            .commit_transaction(transaction_id)
            .await
            .unwrap();
        
        // Strict performance validation: must be <3ms
        timer.assert_within_ms(3.0);
        
        println!("✓ Federation storage performance target validated: <3ms");
    }
    
    /// Validates working memory performance target: <2ms for attention-based retrieval
    #[tokio::test]
    async fn validate_working_memory_performance_target() {
        let suite = CognitiveTestSuite::new().await.unwrap();
        
        // Pre-populate working memory
        for i in 0..10 {
            suite.working_memory
                .store_in_buffer(
                    MemoryContent::Concept(format!("test_concept_{}", i)),
                    BufferType::Episodic,
                    0.8
                )
                .await
                .unwrap();
        }
        
        let timer = PerformanceTimer::new("working memory retrieval performance validation");
        
        let attention_state = suite.attention_manager
            .get_current_attention_state()
            .await
            .unwrap();
        
        let _memories = suite.working_memory
            .retrieve_with_attention_guidance(
                "test concepts",
                attention_state,
                5
            )
            .await
            .unwrap();
        
        // Strict performance validation: must be <2ms
        timer.assert_within_ms(2.0);
        
        println!("✓ Working memory performance target validated: <2ms for attention-based retrieval");
    }
}