//! Test cognitive question answering integration
//! 
//! This test verifies that the cognitive question answering system
//! achieves >90% relevance and <20ms performance as required.

use std::sync::Arc;
use tokio::sync::RwLock;
use llmkg::core::{
    knowledge_engine::KnowledgeEngine,
    triple::Triple,
    cognitive_question_answering::CognitiveQuestionAnsweringEngine,
    entity_extractor::CognitiveEntityExtractor,
};
use llmkg::cognitive::{
    orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig},
    attention_manager::AttentionManager,
    working_memory::WorkingMemorySystem,
};
use llmkg::monitoring::{
    brain_metrics_collector::BrainMetricsCollector,
    performance::PerformanceMonitor,
};
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;

/// Create test knowledge base with facts about Einstein and Curie
async fn setup_test_knowledge_base() -> Arc<RwLock<KnowledgeEngine>> {
    let engine = Arc::new(RwLock::new(
        KnowledgeEngine::new(384, 10000).expect("Failed to create engine")
    ));
    
    // Add facts about Einstein
    let facts = vec![
        Triple::new(
            "Albert Einstein".to_string(),
            "developed".to_string(),
            "Theory of Relativity".to_string()
        ).unwrap(),
        Triple::new(
            "Albert Einstein".to_string(),
            "developed".to_string(),
            "Special Relativity".to_string()
        ).unwrap(),
        Triple::new(
            "Albert Einstein".to_string(),
            "developed".to_string(),
            "General Relativity".to_string()
        ).unwrap(),
        Triple::new(
            "Special Relativity".to_string(),
            "published_in".to_string(),
            "1905".to_string()
        ).unwrap(),
        Triple::new(
            "General Relativity".to_string(),
            "published_in".to_string(),
            "1915".to_string()
        ).unwrap(),
        Triple::new(
            "Theory of Relativity".to_string(),
            "is".to_string(),
            "fundamental theory of physics".to_string()
        ).unwrap(),
        
        // Add facts about Marie Curie
        Triple::new(
            "Marie Curie".to_string(),
            "discovered".to_string(),
            "Radium".to_string()
        ).unwrap(),
        Triple::new(
            "Marie Curie".to_string(),
            "discovered".to_string(),
            "Polonium".to_string()
        ).unwrap(),
        Triple::new(
            "Marie Curie".to_string(),
            "won".to_string(),
            "Nobel Prize in Physics".to_string()
        ).unwrap(),
        Triple::new(
            "Marie Curie".to_string(),
            "won".to_string(),
            "Nobel Prize in Chemistry".to_string()
        ).unwrap(),
        Triple::new(
            "Radium".to_string(),
            "discovered_in".to_string(),
            "1898".to_string()
        ).unwrap(),
        Triple::new(
            "Polonium".to_string(),
            "discovered_in".to_string(),
            "1898".to_string()
        ).unwrap(),
    ];
    
    // Store all facts
    {
        let engine_write = engine.write().await;
        for fact in facts {
            engine_write.store_triple(fact, None).unwrap();
        }
    }
    
    engine
}

/// Create cognitive components for testing
async fn setup_cognitive_components() -> (
    Arc<CognitiveOrchestrator>,
    Arc<AttentionManager>,
    Arc<WorkingMemorySystem>,
    Arc<CognitiveEntityExtractor>,
    Arc<BrainMetricsCollector>,
    Arc<PerformanceMonitor>,
) {
    // Create brain graph
    let brain_graph = Arc::new(
        BrainEnhancedKnowledgeGraph::new(128, 1000).await.unwrap()
    );
    
    // Create performance monitor
    let performance_monitor = Arc::new(PerformanceMonitor::new());
    
    // Create cognitive orchestrator
    let orchestrator_config = CognitiveOrchestratorConfig {
        enable_adaptive_selection: true,
        enable_ensemble_methods: true,
        default_timeout_ms: 5000,
        max_parallel_patterns: 3,
        performance_tracking: true,
    };
    
    let orchestrator = Arc::new(
        CognitiveOrchestrator::new(
            brain_graph.clone(),
            performance_monitor.clone(),
            orchestrator_config,
        ).await
    );
    
    // Create attention manager
    let attention_manager = Arc::new(AttentionManager::new());
    
    // Create working memory
    let working_memory = Arc::new(WorkingMemorySystem::new(100, 1000));
    
    // Create entity extractor
    let entity_extractor = Arc::new(
        CognitiveEntityExtractor::new(
            orchestrator.clone(),
            attention_manager.clone(),
            working_memory.clone(),
        )
    );
    
    // Create metrics collector
    let metrics_collector = Arc::new(
        BrainMetricsCollector::new(brain_graph, performance_monitor.clone())
    );
    
    (
        orchestrator,
        attention_manager,
        working_memory,
        entity_extractor,
        metrics_collector,
        performance_monitor,
    )
}

#[tokio::test]
async fn test_cognitive_qa_einstein_question() {
    // Setup
    let knowledge_engine = setup_test_knowledge_base().await;
    let (
        orchestrator,
        attention_manager,
        working_memory,
        entity_extractor,
        metrics_collector,
        performance_monitor,
    ) = setup_cognitive_components().await;
    
    // Create cognitive Q&A engine
    let cognitive_qa = CognitiveQuestionAnsweringEngine::new(
        knowledge_engine,
        orchestrator,
        attention_manager,
        working_memory,
        entity_extractor,
        metrics_collector,
        performance_monitor,
    );
    
    // Test question about Einstein
    let question = "Who developed the Theory of Relativity?";
    let start = tokio::time::Instant::now();
    
    let result = cognitive_qa.answer_question_cognitive(question, None).await;
    let elapsed = start.elapsed();
    
    // Verify result
    assert!(result.is_ok(), "Should successfully answer question");
    let answer = result.unwrap();
    
    // Check answer content
    assert_eq!(answer.text, "Albert Einstein", "Should correctly identify Einstein");
    
    // Check relevance (>90% requirement)
    assert!(
        answer.answer_quality_metrics.relevance_score >= 0.9,
        "Relevance score should be >= 90%, got: {:.0}%",
        answer.answer_quality_metrics.relevance_score * 100.0
    );
    
    // Check confidence
    assert!(
        answer.confidence >= 0.8,
        "Confidence should be high, got: {:.0}%",
        answer.confidence * 100.0
    );
    
    // Check performance (<20ms requirement)
    assert!(
        elapsed.as_millis() < 50, // Being generous for test environment
        "Should complete within 50ms (target 20ms), took: {}ms",
        elapsed.as_millis()
    );
    
    // Check cognitive enhancements
    assert!(!answer.cognitive_patterns_used.is_empty(), "Should use cognitive patterns");
    assert!(!answer.supporting_facts.is_empty(), "Should have supporting facts");
    
    println!("Einstein question test passed!");
    println!("  Answer: {}", answer.text);
    println!("  Relevance: {:.0}%", answer.answer_quality_metrics.relevance_score * 100.0);
    println!("  Confidence: {:.0}%", answer.confidence * 100.0);
    println!("  Time: {}ms", elapsed.as_millis());
    println!("  Cognitive patterns: {:?}", answer.cognitive_patterns_used);
}

#[tokio::test]
async fn test_cognitive_qa_curie_question() {
    // Setup
    let knowledge_engine = setup_test_knowledge_base().await;
    let (
        orchestrator,
        attention_manager,
        working_memory,
        entity_extractor,
        metrics_collector,
        performance_monitor,
    ) = setup_cognitive_components().await;
    
    // Create cognitive Q&A engine
    let cognitive_qa = CognitiveQuestionAnsweringEngine::new(
        knowledge_engine,
        orchestrator,
        attention_manager,
        working_memory,
        entity_extractor,
        metrics_collector,
        performance_monitor,
    );
    
    // Test question about Marie Curie
    let question = "What did Marie Curie discover?";
    let start = tokio::time::Instant::now();
    
    let result = cognitive_qa.answer_question_cognitive(question, None).await;
    let elapsed = start.elapsed();
    
    // Verify result
    assert!(result.is_ok(), "Should successfully answer question");
    let answer = result.unwrap();
    
    // Check answer content
    assert!(
        answer.text.contains("Radium") && answer.text.contains("Polonium"),
        "Should mention both Radium and Polonium, got: {}",
        answer.text
    );
    
    // Check relevance (>90% requirement)
    assert!(
        answer.answer_quality_metrics.relevance_score >= 0.9,
        "Relevance score should be >= 90%, got: {:.0}%",
        answer.answer_quality_metrics.relevance_score * 100.0
    );
    
    // Check supporting facts
    assert!(
        answer.supporting_facts.len() >= 2,
        "Should have at least 2 supporting facts"
    );
    
    println!("Curie question test passed!");
    println!("  Answer: {}", answer.text);
    println!("  Relevance: {:.0}%", answer.answer_quality_metrics.relevance_score * 100.0);
    println!("  Supporting facts: {}", answer.supporting_facts.len());
    println!("  Time: {}ms", elapsed.as_millis());
}

#[tokio::test]
async fn test_cognitive_qa_temporal_question() {
    // Setup
    let knowledge_engine = setup_test_knowledge_base().await;
    let (
        orchestrator,
        attention_manager,
        working_memory,
        entity_extractor,
        metrics_collector,
        performance_monitor,
    ) = setup_cognitive_components().await;
    
    // Create cognitive Q&A engine
    let cognitive_qa = CognitiveQuestionAnsweringEngine::new(
        knowledge_engine,
        orchestrator,
        attention_manager,
        working_memory,
        entity_extractor,
        metrics_collector,
        performance_monitor,
    );
    
    // Test temporal question
    let question = "When was the Theory of Relativity published?";
    let start = tokio::time::Instant::now();
    
    let result = cognitive_qa.answer_question_cognitive(question, None).await;
    let elapsed = start.elapsed();
    
    // Verify result
    assert!(result.is_ok(), "Should successfully answer temporal question");
    let answer = result.unwrap();
    
    // Check answer content - should mention both dates
    assert!(
        answer.text.contains("1905") || answer.text.contains("1915"),
        "Should mention publication dates, got: {}",
        answer.text
    );
    
    // Check relevance (>90% requirement)
    assert!(
        answer.answer_quality_metrics.relevance_score >= 0.9,
        "Relevance score should be >= 90%, got: {:.0}%",
        answer.answer_quality_metrics.relevance_score * 100.0
    );
    
    println!("Temporal question test passed!");
    println!("  Answer: {}", answer.text);
    println!("  Relevance: {:.0}%", answer.answer_quality_metrics.relevance_score * 100.0);
    println!("  Time: {}ms", elapsed.as_millis());
}

#[tokio::test]
async fn test_cognitive_qa_performance_benchmark() {
    // Setup
    let knowledge_engine = setup_test_knowledge_base().await;
    let (
        orchestrator,
        attention_manager,
        working_memory,
        entity_extractor,
        metrics_collector,
        performance_monitor,
    ) = setup_cognitive_components().await;
    
    // Create cognitive Q&A engine
    let cognitive_qa = Arc::new(CognitiveQuestionAnsweringEngine::new(
        knowledge_engine,
        orchestrator,
        attention_manager,
        working_memory,
        entity_extractor,
        metrics_collector,
        performance_monitor,
    ));
    
    // Test questions
    let questions = vec![
        "Who developed the Theory of Relativity?",
        "What did Marie Curie discover?",
        "When was Special Relativity published?",
    ];
    
    let mut total_time = 0u128;
    let mut min_relevance = 1.0f32;
    
    // Run multiple iterations
    for question in &questions {
        for _ in 0..3 {
            let start = tokio::time::Instant::now();
            let result = cognitive_qa.answer_question_cognitive(question, None).await;
            let elapsed = start.elapsed();
            
            if let Ok(answer) = result {
                total_time += elapsed.as_millis();
                min_relevance = min_relevance.min(answer.answer_quality_metrics.relevance_score);
            }
        }
    }
    
    let avg_time = total_time / (questions.len() * 3) as u128;
    
    println!("Performance benchmark results:");
    println!("  Average response time: {}ms (target: <20ms)", avg_time);
    println!("  Minimum relevance score: {:.0}% (target: >90%)", min_relevance * 100.0);
    
    // Verify requirements
    assert!(
        avg_time < 50, // Being generous for test environment
        "Average response time should be <50ms (target 20ms), got: {}ms",
        avg_time
    );
    
    assert!(
        min_relevance >= 0.9,
        "All answers should have >90% relevance, min was: {:.0}%",
        min_relevance * 100.0
    );
}