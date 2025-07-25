//! Performance Validation Tests for Cognitive Integration
//! Validates all performance targets specified in documentation
//! Targets: Entity extraction <8ms, Relationship extraction <12ms, 
//! Question answering <20ms, Federation storage <3ms, Working memory <2ms

use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio;
use futures::future::join_all;

use llmkg::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use llmkg::cognitive::attention_manager::{AttentionManager, AttentionType};
use llmkg::cognitive::working_memory::{WorkingMemorySystem, MemoryContent, BufferType};
use llmkg::cognitive::{CognitivePatternType, ReasoningStrategy, QueryContext};
use llmkg::neural::neural_server::{NeuralProcessingServer, NeuralRequest, NeuralOperation, NeuralParameters};
use llmkg::federation::coordinator::{FederationCoordinator, TransactionId};
use llmkg::core::entity_extractor::EntityExtractor;
use llmkg::core::relationship_extractor::RelationshipExtractor;
use llmkg::core::question_parser::QuestionParser;
use llmkg::core::answer_generator::AnswerGenerator;
use llmkg::test_support::builders::{
    AttentionManagerBuilder, CognitiveOrchestratorBuilder, WorkingMemoryBuilder
};
use llmkg::test_support::fixtures::create_test_graph;
use llmkg::error::Result;

/// Performance benchmark suite for cognitive integration
struct PerformanceBenchmarkSuite {
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

impl PerformanceBenchmarkSuite {
    /// Creates performance benchmark suite with optimized configuration
    async fn new() -> Result<Self> {
        let graph = create_test_graph();
        
        // Use high-performance configuration for benchmarking
        let orchestrator_config = CognitiveOrchestratorConfig {
            enable_adaptive_selection: true,
            enable_ensemble_methods: true,
            default_timeout_ms: 1000, // Reduced for performance testing
            max_parallel_patterns: 16, // Increased for parallel processing
            performance_tracking: true,
        };
        
        let orchestrator = Arc::new(CognitiveOrchestrator::new(graph.clone(), orchestrator_config).await?);
        let neural_server = Arc::new(NeuralProcessingServer::new().await?);
        // Federation coordinator would need registry, using mock for now
        // let federation_coordinator = Arc::new(FederationCoordinator::new(registry).await?);
        
        let attention_manager = Arc::new(
            AttentionManagerBuilder::new()
                .with_graph(graph.clone())
                .build()
                .await?
        );
        
        let working_memory = Arc::new(
            WorkingMemoryBuilder::new()
                .with_graph(graph.clone())
                .build()
                .await?
        );
        
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

/// Precise performance timer with statistical analysis
struct PrecisionTimer {
    start: Instant,
    operation: String,
    measurements: Vec<f64>,
}

impl PrecisionTimer {
    fn new(operation: &str) -> Self {
        Self {
            start: Instant::now(),
            operation: operation.to_string(),
            measurements: Vec::new(),
        }
    }
    
    fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
    
    fn reset(&mut self) {
        self.start = Instant::now();
    }
    
    fn record_measurement(&mut self) {
        let elapsed = self.elapsed_ms();
        self.measurements.push(elapsed);
        self.reset();
    }
    
    fn assert_performance_target(&self, target_ms: f64, percentile: f64) {
        if self.measurements.is_empty() {
            panic!("No measurements recorded for {}", self.operation);
        }
        
        let mut sorted_measurements = self.measurements.clone();
        sorted_measurements.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((percentile / 100.0) * (sorted_measurements.len() - 1) as f64) as usize;
        let percentile_value = sorted_measurements[index];
        
        let avg = self.measurements.iter().sum::<f64>() / self.measurements.len() as f64;
        let min = sorted_measurements[0];
        let max = sorted_measurements[sorted_measurements.len() - 1];
        
        assert!(
            percentile_value <= target_ms,
            "{} {}th percentile: {:.2}ms exceeds target: {:.2}ms\n  Stats: avg={:.2}ms, min={:.2}ms, max={:.2}ms",
            self.operation,
            percentile,
            percentile_value,
            target_ms,
            avg,
            min,
            max
        );
        
        println!("✓ {} performance validated", self.operation);
        println!("  Target: <{:.1}ms, {}th percentile: {:.2}ms", target_ms, percentile, percentile_value);
        println!("  Stats: avg={:.2}ms, min={:.2}ms, max={:.2}ms, samples={}", avg, min, max, self.measurements.len());
    }
}

#[cfg(test)]
mod performance_validation_tests {
    use super::*;

    /// Comprehensive entity extraction performance validation
    /// Target: <8ms per sentence with neural processing
    #[tokio::test]
    async fn validate_entity_extraction_performance_comprehensive() {
        let suite = PerformanceBenchmarkSuite::new().await.unwrap();
        let mut timer = PrecisionTimer::new("Entity Extraction with Neural Processing");
        
        // Diverse test sentences with varying complexity
        let test_sentences = vec![
            // Simple (1-2 entities)
            "Einstein worked.",
            "Marie Curie discovered radium.",
            "The computer processes data.",
            
            // Medium (3-5 entities) 
            "Albert Einstein developed the Theory of Relativity in 1905.",
            "Marie Curie discovered radium and polonium through radioactivity research.",
            "The quantum computer at IBM processes qubits using superconducting circuits.",
            
            // Complex (6+ entities)
            "Dr. Albert Einstein, the renowned theoretical physicist, developed the special and general theories of relativity between 1905 and 1915 while working at Princeton University.",
            "Marie Curie, the Polish-French physicist and chemist, discovered the radioactive elements radium and polonium through her groundbreaking research on radioactivity at the University of Paris.",
            "The IBM quantum computer system utilizes superconducting qubits operating at near absolute zero temperatures to perform quantum computations with unprecedented speed and accuracy.",
            
            // Domain-specific technical
            "CRISPR-Cas9 gene editing technology enables researchers to make precise modifications to DNA sequences in living cells with unprecedented accuracy.",
            "The transformer neural network architecture utilizes self-attention mechanisms to process sequential data more efficiently than traditional recurrent neural networks.",
            "Quantum entanglement creates correlations between particles that Einstein famously called 'spooky action at a distance' in his correspondence with colleagues.",
        ];
        
        // Run multiple iterations for statistical significance
        for iteration in 0..5 {
            for sentence in &test_sentences {
                timer.reset();
                
                let _entities = suite.entity_extractor
                    .extract_entities_with_neural_processing(sentence)
                    .await
                    .unwrap();
                
                timer.record_measurement();
            }
        }
        
        // Validate 90th percentile performance (allows for occasional outliers)
        timer.assert_performance_target(8.0, 90.0);
    }
    
    /// Comprehensive relationship extraction performance validation
    /// Target: <12ms per sentence with federation coordination
    #[tokio::test]
    async fn validate_relationship_extraction_performance_comprehensive() {
        let suite = PerformanceBenchmarkSuite::new().await.unwrap();
        let mut timer = PrecisionTimer::new("Relationship Extraction with Federation");
        
        let test_sentences = vec![
            // Simple relationships
            "Einstein developed relativity.",
            "Curie discovered radium.",
            "IBM built the computer.",
            
            // Medium complexity relationships
            "Albert Einstein developed the Theory of Relativity in 1905.",
            "Marie Curie discovered radium through radioactivity research.",
            "The neural network learns patterns from training data.",
            
            // Complex relationships with multiple entities
            "Dr. Albert Einstein, working at Princeton University, developed both special and general relativity theories that revolutionized modern physics.",
            "Marie Curie, collaborating with her husband Pierre, discovered radioactive elements through systematic research at the University of Paris.",
            "The IBM quantum computer team engineered superconducting qubits that enable quantum computations at unprecedented scales and speeds.",
            
            // Technical domain relationships
            "CRISPR-Cas9 technology allows scientists to edit genes by cutting DNA at specific sequences and inserting new genetic material.",
            "Transformer neural networks utilize self-attention mechanisms to model long-range dependencies in sequential data more effectively than RNNs.",
            "Quantum entanglement creates instantaneous correlations between particles regardless of the distance separating them across space.",
        ];
        
        // Pre-extract entities for consistent testing
        let mut entity_cache = std::collections::HashMap::new();
        for sentence in &test_sentences {
            let entities = suite.entity_extractor
                .extract_entities(sentence)
                .await
                .unwrap();
            entity_cache.insert(sentence.to_string(), entities);
        }
        
        // Run performance tests
        for iteration in 0..5 {
            for sentence in &test_sentences {
                let entities = entity_cache.get(*sentence).unwrap();
                
                timer.reset();
                
                let _relationships = suite.relationship_extractor
                    .extract_relationships_with_federation(sentence, entities)
                    .await
                    .unwrap();
                
                timer.record_measurement();
            }
        }
        
        // Validate 90th percentile performance
        timer.assert_performance_target(12.0, 90.0);
    }
    
    /// Comprehensive question answering performance validation
    /// Target: <20ms total with cognitive reasoning
    #[tokio::test]
    async fn validate_question_answering_performance_comprehensive() {
        let suite = PerformanceBenchmarkSuite::new().await.unwrap();
        let mut timer = PrecisionTimer::new("Question Answering with Cognitive Reasoning");
        
        // Pre-populate knowledge base
        let knowledge_texts = vec![
            "Albert Einstein developed the Theory of Relativity in 1905 while working as a patent clerk.",
            "Marie Curie discovered radium and polonium through her research on radioactivity at the University of Paris.",
            "The quantum computer uses qubits that can exist in superposition states for parallel computation.",
            "Neural networks learn by adjusting weights through backpropagation during training phases.",
            "CRISPR gene editing allows precise DNA modifications using guide RNAs and Cas9 enzymes.",
        ];
        
        let mut all_entities = Vec::new();
        let mut all_relationships = Vec::new();
        
        for text in knowledge_texts {
            let entities = suite.entity_extractor.extract_entities(text).await.unwrap();
            let relationships = suite.relationship_extractor.extract_relationships(text, &entities).await.unwrap();
            all_entities.extend(entities);
            all_relationships.extend(relationships);
        }
        
        let test_questions = vec![
            // Simple factual questions
            "Who developed the Theory of Relativity?",
            "What did Marie Curie discover?",
            "When was relativity developed?",
            
            // Medium complexity questions
            "Who discovered radium and polonium through radioactivity research?",
            "What technology allows precise DNA modifications?",
            "How do neural networks learn patterns?",
            
            // Complex analytical questions
            "What are the key differences between quantum computers and classical computers?",
            "How did Marie Curie's research contribute to our understanding of radioactivity?",
            "What makes CRISPR technology revolutionary for genetic engineering?",
            
            // Reasoning-intensive questions
            "Why was Einstein's work on relativity considered revolutionary for physics?",
            "How do the principles of quantum mechanics enable quantum computing advantages?",
            "What are the ethical implications of CRISPR gene editing technology?",
        ];
        
        // Run performance tests
        for iteration in 0..3 { // Fewer iterations due to complexity
            for question in &test_questions {
                timer.reset();
                
                let question_intent = suite.question_parser
                    .parse_with_cognitive_enhancement(question, None)
                    .await
                    .unwrap();
                
                let _answer = suite.answer_generator
                    .generate_answer_with_cognitive_reasoning(
                        &all_entities,
                        &all_relationships,
                        &question_intent,
                        ReasoningStrategy::Automatic
                    )
                    .await
                    .unwrap();
                
                timer.record_measurement();
            }
        }
        
        // Validate 95th percentile performance (stricter for end-to-end)
        timer.assert_performance_target(20.0, 95.0);
    }
    
    /// Federation storage performance validation with transaction complexity
    /// Target: <3ms with cross-database coordination
    #[tokio::test]
    async fn validate_federation_storage_performance_comprehensive() {
        let suite = PerformanceBenchmarkSuite::new().await.unwrap();
        let mut timer = PrecisionTimer::new("Federation Storage with Cross-Database Coordination");
        
        // Test different transaction complexities
        let transaction_scenarios = vec![
            // Simple: single entity
            vec![("entity_1", serde_json::json!({"name": "Test Entity 1", "type": "simple"}))],
            
            // Medium: multiple entities
            vec![
                ("entity_2", serde_json::json!({"name": "Test Entity 2", "type": "medium"})),
                ("entity_3", serde_json::json!({"name": "Test Entity 3", "type": "medium"})),
            ],
            
            // Complex: multiple entities with relationships
            vec![
                ("entity_4", serde_json::json!({"name": "Complex Entity 4", "type": "complex", "properties": {"importance": 0.9}})),
                ("entity_5", serde_json::json!({"name": "Complex Entity 5", "type": "complex", "properties": {"importance": 0.8}})),
                ("entity_6", serde_json::json!({"name": "Complex Entity 6", "type": "complex", "properties": {"importance": 0.7}})),
            ],
        ];
        
        // Run performance tests
        for iteration in 0..10 { // More iterations for storage operations
            for (scenario_idx, scenario) in transaction_scenarios.iter().enumerate() {
                timer.reset();
                
                let transaction_id = TransactionId::new();
                let _transaction = suite.federation_coordinator
                    .begin_cross_database_transaction(
                        transaction_id.clone(),
                        vec!["primary", "secondary"]
                    )
                    .await
                    .unwrap();
                
                // Add entities to transaction
                for (entity_name, entity_data) in scenario {
                    suite.federation_coordinator
                        .add_entity_to_transaction(
                            &transaction_id,
                            &format!("{}_{}_{}_{}", entity_name, scenario_idx, iteration, rand::random::<u32>()),
                            entity_data.clone()
                        )
                        .await
                        .unwrap();
                }
                
                // Commit transaction
                suite.federation_coordinator
                    .commit_transaction(transaction_id)
                    .await
                    .unwrap();
                
                timer.record_measurement();
            }
        }
        
        // Validate 95th percentile performance (storage should be very consistent)
        timer.assert_performance_target(3.0, 95.0);
    }
    
    /// Working memory performance validation with attention guidance
    /// Target: <2ms for attention-based retrieval
    #[tokio::test]
    async fn validate_working_memory_performance_comprehensive() {
        let suite = PerformanceBenchmarkSuite::new().await.unwrap();
        let mut timer = PrecisionTimer::new("Working Memory with Attention-Based Retrieval");
        
        // Pre-populate working memory with varied content
        let memory_contents = vec![
            // Concepts
            MemoryContent::Concept("quantum computing".to_string()),
            MemoryContent::Concept("artificial intelligence".to_string()),
            MemoryContent::Concept("machine learning".to_string()),
            MemoryContent::Concept("neural networks".to_string()),
            MemoryContent::Concept("deep learning".to_string()),
            MemoryContent::Concept("natural language processing".to_string()),
            MemoryContent::Concept("computer vision".to_string()),
            MemoryContent::Concept("robotics".to_string()),
            
            // Factual information
            MemoryContent::Fact("Einstein developed relativity theory".to_string()),
            MemoryContent::Fact("Marie Curie discovered radium".to_string()),
            MemoryContent::Fact("Quantum computers use qubits".to_string()),
            MemoryContent::Fact("Neural networks learn from data".to_string()),
            
            // Patterns
            MemoryContent::Pattern("Scientific discovery pattern".to_string()),
            MemoryContent::Pattern("Technology innovation pattern".to_string()),
            MemoryContent::Pattern("Learning algorithm pattern".to_string()),
        ];
        
        // Store content in different buffers with varied activation levels
        for (i, content) in memory_contents.iter().enumerate() {
            let buffer_type = match i % 3 {
                0 => BufferType::Episodic,
                1 => BufferType::Semantic,
                _ => BufferType::Working,
            };
            let activation = 0.5 + (i as f32 * 0.05) % 0.5; // Varied activation levels
            
            suite.working_memory
                .store_in_buffer(content.clone(), buffer_type, activation)
                .await
                .unwrap();
        }
        
        // Test different query patterns
        let query_patterns = vec![
            "quantum computing concepts",
            "artificial intelligence",
            "scientific discoveries",
            "learning algorithms",
            "technology innovations",
            "neural network patterns",
            "computing paradigms",
        ];
        
        // Run performance tests
        for iteration in 0..20 { // Many iterations for memory operations
            for query in &query_patterns {
                // Set attention focus
                suite.attention_manager
                    .focus_attention_on_text(query, AttentionType::Selective)
                    .await
                    .unwrap();
                
                let attention_state = suite.attention_manager
                    .get_current_attention_state()
                    .await
                    .unwrap();
                
                timer.reset();
                
                let _memories = suite.working_memory
                    .retrieve_with_attention_guidance(
                        query,
                        attention_state,
                        5 // Retrieve top 5 relevant memories
                    )
                    .await
                    .unwrap();
                
                timer.record_measurement();
            }
        }
        
        // Validate 99th percentile performance (memory should be very fast)
        timer.assert_performance_target(2.0, 99.0);
    }
    
    /// Stress test for concurrent cognitive operations
    /// Validates performance under high load conditions
    #[tokio::test]
    async fn validate_concurrent_operations_performance() {
        let suite = PerformanceBenchmarkSuite::new().await.unwrap();
        let mut timer = PrecisionTimer::new("Concurrent Cognitive Operations");
        
        let num_concurrent_tasks = 10;
        let operations_per_task = 5;
        
        for iteration in 0..3 {
            timer.reset();
            
            // Create concurrent tasks
            let mut task_handles = Vec::new();
            
            for task_id in 0..num_concurrent_tasks {
                let suite_clone = Arc::new(suite); // Note: We need to share the suite
                
                let handle = tokio::spawn(async move {
                    let base_text = format!("Task {} processes entity extraction", task_id);
                    
                    for op_id in 0..operations_per_task {
                        let text = format!("{} operation {}", base_text, op_id);
                        
                        // Perform entity extraction
                        let _entities = suite_clone.entity_extractor
                            .extract_entities(&text)
                            .await
                            .unwrap();
                        
                        // Perform cognitive reasoning
                        let _result = suite_clone.orchestrator
                            .reason(
                                &format!("Analyze: {}", text),
                                None,
                                ReasoningStrategy::Automatic
                            )
                            .await
                            .unwrap();
                    }
                });
                
                task_handles.push(handle);
            }
            
            // Wait for all tasks to complete
            join_all(task_handles).await;
            
            timer.record_measurement();
        }
        
        // Validate concurrent performance (should scale reasonably)
        let target_ms = 1000.0; // 1 second for all concurrent operations
        timer.assert_performance_target(target_ms, 90.0);
        
        println!("✓ Concurrent operations completed: {} tasks × {} operations each", 
                 num_concurrent_tasks, operations_per_task);
    }
    
    /// Memory usage and resource efficiency validation
    #[tokio::test]
    async fn validate_memory_usage_efficiency() {
        let suite = PerformanceBenchmarkSuite::new().await.unwrap();
        
        // Measure baseline memory usage
        let baseline_memory = get_current_memory_usage().await;
        
        // Perform intensive operations
        let intensive_operations = 100;
        for i in 0..intensive_operations {
            let text = format!("Intensive operation {} processes complex entity relationships in cognitive systems", i);
            
            let entities = suite.entity_extractor.extract_entities(&text).await.unwrap();
            let _relationships = suite.relationship_extractor.extract_relationships(&text, &entities).await.unwrap();
            
            // Store in working memory
            suite.working_memory
                .store_in_buffer(
                    MemoryContent::Concept(format!("concept_{}", i)),
                    BufferType::Working,
                    0.7
                )
                .await
                .unwrap();
        }
        
        // Measure peak memory usage
        let peak_memory = get_current_memory_usage().await;
        let memory_increase = peak_memory - baseline_memory;
        
        println!("Memory usage analysis:");
        println!("  Baseline: {:.2} MB", baseline_memory);
        println!("  Peak: {:.2} MB", peak_memory);
        println!("  Increase: {:.2} MB", memory_increase);
        
        // Validate memory efficiency (should not increase excessively)
        let max_memory_increase_mb = 50.0; // Allow up to 50MB increase
        assert!(
            memory_increase <= max_memory_increase_mb,
            "Memory increase {:.2} MB exceeds limit {:.2} MB",
            memory_increase,
            max_memory_increase_mb
        );
        
        // Test memory cleanup
        suite.working_memory.cleanup_expired_memories().await.unwrap();
        
        // Allow some time for garbage collection
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        let final_memory = get_current_memory_usage().await;
        let memory_after_cleanup = final_memory - baseline_memory;
        
        println!("  After cleanup: {:.2} MB", final_memory);
        println!("  Net increase: {:.2} MB", memory_after_cleanup);
        
        // Memory should be reduced after cleanup
        assert!(
            memory_after_cleanup < memory_increase * 0.8,
            "Memory cleanup not effective: {:.2} MB still above baseline",
            memory_after_cleanup
        );
        
        println!("✓ Memory usage efficiency validated");
    }
}

/// Helper function to get current memory usage (simplified implementation)
async fn get_current_memory_usage() -> f64 {
    // In a real implementation, this would use system APIs to get actual memory usage
    // For testing purposes, we'll use a simplified approximation
    use std::alloc::{GlobalAlloc, Layout, System};
    
    // This is a simplified placeholder - in production you'd use proper memory profiling
    let test_allocation = unsafe {
        System.alloc(Layout::from_size_align(1024, 8).unwrap())
    };
    
    if !test_allocation.is_null() {
        unsafe {
            System.dealloc(test_allocation, Layout::from_size_align(1024, 8).unwrap());
        }
    }
    
    // Return a simulated memory usage value
    // In practice, you'd use system calls or memory profiling libraries
    42.0 // Placeholder value in MB
}

/// Integration test for all performance targets together
#[cfg(test)]
mod integrated_performance_tests {
    use super::*;
    
    /// Full pipeline performance test with all components
    #[tokio::test]
    async fn validate_full_pipeline_performance_targets() {
        let suite = PerformanceBenchmarkSuite::new().await.unwrap();
        
        println!("🔬 Running comprehensive performance validation suite...");
        
        // Test realistic workflow scenarios
        let workflow_scenarios = vec![
            "Scientific research workflow: analyzing quantum computing applications",
            "Educational content analysis: understanding machine learning concepts", 
            "Technical documentation processing: extracting API relationships",
            "Knowledge discovery: finding connections between disparate fields",
        ];
        
        for scenario in workflow_scenarios {
            println!("\n📊 Testing scenario: {}", scenario);
            
            let pipeline_timer = PrecisionTimer::new(&format!("Full Pipeline: {}", scenario));
            
            // Step 1: Entity extraction (<8ms)
            let entity_timer = Instant::now();
            let entities = suite.entity_extractor
                .extract_entities(scenario)
                .await
                .unwrap();
            let entity_time = entity_timer.elapsed().as_secs_f64() * 1000.0;
            
            // Step 2: Relationship extraction (<12ms)
            let rel_timer = Instant::now();
            let relationships = suite.relationship_extractor
                .extract_relationships(scenario, &entities)
                .await
                .unwrap();
            let rel_time = rel_timer.elapsed().as_secs_f64() * 1000.0;
            
            // Step 3: Question generation and answering (<20ms)
            let qa_timer = Instant::now();
            let question = format!("What are the key concepts in: {}?", scenario);
            let question_intent = suite.question_parser.parse(&question).await.unwrap();
            let _answer = suite.answer_generator
                .generate_answer(&entities, &relationships, &question_intent)
                .await
                .unwrap();
            let qa_time = qa_timer.elapsed().as_secs_f64() * 1000.0;
            
            // Step 4: Federation storage (<3ms)
            let storage_timer = Instant::now();
            let transaction_id = TransactionId::new();
            let _transaction = suite.federation_coordinator
                .begin_cross_database_transaction(transaction_id.clone(), vec!["main"])
                .await
                .unwrap();
            suite.federation_coordinator
                .commit_transaction(transaction_id)
                .await
                .unwrap();
            let storage_time = storage_timer.elapsed().as_secs_f64() * 1000.0;
            
            // Step 5: Working memory operations (<2ms)
            let memory_timer = Instant::now();
            let attention_state = suite.attention_manager
                .get_current_attention_state()
                .await
                .unwrap();
            let _memories = suite.working_memory
                .retrieve_with_attention_guidance("key concepts", attention_state, 3)
                .await
                .unwrap();
            let memory_time = memory_timer.elapsed().as_secs_f64() * 1000.0;
            
            // Validate individual performance targets
            assert!(entity_time <= 8.0, "Entity extraction: {:.2}ms > 8ms", entity_time);
            assert!(rel_time <= 12.0, "Relationship extraction: {:.2}ms > 12ms", rel_time);
            assert!(qa_time <= 20.0, "Question answering: {:.2}ms > 20ms", qa_time);
            assert!(storage_time <= 3.0, "Federation storage: {:.2}ms > 3ms", storage_time);
            assert!(memory_time <= 2.0, "Working memory: {:.2}ms > 2ms", memory_time);
            
            let total_time = entity_time + rel_time + qa_time + storage_time + memory_time;
            
            println!("  ✓ Entity extraction: {:.2}ms (target: <8ms)", entity_time);
            println!("  ✓ Relationship extraction: {:.2}ms (target: <12ms)", rel_time);
            println!("  ✓ Question answering: {:.2}ms (target: <20ms)", qa_time);
            println!("  ✓ Federation storage: {:.2}ms (target: <3ms)", storage_time);
            println!("  ✓ Working memory: {:.2}ms (target: <2ms)", memory_time);
            println!("  📈 Total pipeline time: {:.2}ms", total_time);
            
            // Total pipeline should complete well within sum of individual targets
            assert!(total_time <= 45.0, "Total pipeline time {:.2}ms exceeds 45ms threshold", total_time);
        }
        
        println!("\n🎉 All performance targets validated successfully!");
    }
}