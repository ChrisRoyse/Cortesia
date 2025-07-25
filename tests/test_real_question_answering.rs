//! Test real question answering with neural generation

use llmkg::core::question_parser::{CognitiveQuestionParser, CognitiveQuestionType, FactualSubtype, ExplanatorySubtype};
use llmkg::core::answer_generator::{CognitiveAnswerGenerator, CognitiveFact};
use llmkg::core::entity_extractor::CognitiveEntityExtractor;
use llmkg::cognitive::orchestrator::CognitiveOrchestrator;
use llmkg::cognitive::attention_manager::AttentionManager;
use llmkg::cognitive::working_memory::WorkingMemorySystem;
use llmkg::monitoring::brain_metrics_collector::BrainMetricsCollector;
use llmkg::monitoring::performance::PerformanceMonitor;
use std::sync::Arc;
use tokio;

#[tokio::test]
async fn test_question_classification() {
    // Build test components
    let orchestrator = Arc::new(CognitiveOrchestrator::default());
    let attention_manager = Arc::new(AttentionManager::new());
    let working_memory = Arc::new(WorkingMemorySystem::new());
    let entity_extractor = Arc::new(CognitiveEntityExtractor::new());
    let metrics_collector = Arc::new(BrainMetricsCollector::new());
    let performance_monitor = Arc::new(PerformanceMonitor::new());
    
    let parser = CognitiveQuestionParser::new(
        orchestrator,
        attention_manager,
        working_memory,
        entity_extractor,
        metrics_collector,
        performance_monitor,
    );
    
    // Test factual questions
    let question = "Who discovered radium?";
    let intent = parser.parse(question).await.unwrap();
    assert!(matches!(intent.question_type, CognitiveQuestionType::Factual(FactualSubtype::Identity)));
    assert!(intent.processing_time_ms < 20); // Performance target
    
    // Test definition questions
    let question = "What is radioactivity?";
    let intent = parser.parse(question).await.unwrap();
    assert!(matches!(intent.question_type, CognitiveQuestionType::Factual(FactualSubtype::Definition)));
    
    // Test explanatory questions
    let question = "How does radioactivity work?";
    let intent = parser.parse(question).await.unwrap();
    assert!(matches!(intent.question_type, CognitiveQuestionType::Explanatory(ExplanatorySubtype::Process)));
    
    // Test comparative questions
    let question = "What's the difference between alpha and beta radiation?";
    let intent = parser.parse(question).await.unwrap();
    assert!(matches!(intent.question_type, CognitiveQuestionType::Comparative(_)));
    
    // Test temporal questions
    let question = "When did Marie Curie discover radium?";
    let intent = parser.parse(question).await.unwrap();
    assert!(matches!(intent.question_type, CognitiveQuestionType::Temporal(_)));
    
    // Test causal questions
    let question = "What caused the discovery of radioactivity?";
    let intent = parser.parse(question).await.unwrap();
    assert!(matches!(intent.question_type, CognitiveQuestionType::Causal(_)));
}

#[tokio::test]
async fn test_fact_ranking() {
    // Build test components
    let orchestrator = Arc::new(CognitiveOrchestrator::default());
    let working_memory = Arc::new(WorkingMemorySystem::new());
    let metrics_collector = Arc::new(BrainMetricsCollector::new());
    let performance_monitor = Arc::new(PerformanceMonitor::new());
    
    let generator = CognitiveAnswerGenerator::new(
        orchestrator.clone(),
        working_memory,
        metrics_collector,
        performance_monitor,
    );
    
    // Create test facts
    let facts = vec![
        CognitiveFact {
            subject: "Marie Curie".to_string(),
            predicate: "discovered".to_string(),
            object: "radium".to_string(),
            confidence: 0.9,
            source_databases: vec![],
            temporal_context: Some("1898".to_string()),
            cognitive_relevance: 0.0, // Will be calculated
            relationship_type: None,
            extracted_patterns: vec![],
            neural_salience: 0.8,
        },
        CognitiveFact {
            subject: "Pierre Curie".to_string(),
            predicate: "worked_with".to_string(),
            object: "Marie Curie".to_string(),
            confidence: 0.85,
            source_databases: vec![],
            temporal_context: None,
            cognitive_relevance: 0.0,
            relationship_type: None,
            extracted_patterns: vec![],
            neural_salience: 0.7,
        },
        CognitiveFact {
            subject: "Nobel Prize".to_string(),
            predicate: "awarded_to".to_string(),
            object: "Marie Curie".to_string(),
            confidence: 0.95,
            source_databases: vec![],
            temporal_context: Some("1903".to_string()),
            cognitive_relevance: 0.0,
            relationship_type: None,
            extracted_patterns: vec![],
            neural_salience: 0.75,
        },
    ];
    
    // Create test intent
    let attention_manager = Arc::new(AttentionManager::new());
    let entity_extractor = Arc::new(CognitiveEntityExtractor::new());
    let parser = CognitiveQuestionParser::new(
        orchestrator,
        attention_manager,
        working_memory,
        entity_extractor,
        metrics_collector,
        performance_monitor,
    );
    
    let intent = parser.parse("Who discovered radium?").await.unwrap();
    
    // Test answer generation
    let answer = generator.generate_answer(facts, intent).await.unwrap();
    
    // Verify answer quality
    assert_eq!(answer.text, "Marie Curie");
    assert!(answer.confidence > 0.8);
    assert!(!answer.supporting_facts.is_empty());
    assert!(answer.processing_time_ms < 20); // Performance target
    
    // Verify fact ranking worked
    let first_fact = &answer.supporting_facts[0];
    assert!(first_fact.cognitive_relevance > 0.5); // Should be highly relevant
}

#[tokio::test]
async fn test_answer_generation_patterns() {
    // Build test components
    let orchestrator = Arc::new(CognitiveOrchestrator::default());
    let working_memory = Arc::new(WorkingMemorySystem::new());
    let metrics_collector = Arc::new(BrainMetricsCollector::new());
    let performance_monitor = Arc::new(PerformanceMonitor::new());
    
    let generator = CognitiveAnswerGenerator::new(
        orchestrator.clone(),
        working_memory,
        metrics_collector,
        performance_monitor,
    );
    
    // Test factual identity answer
    let facts = vec![
        CognitiveFact {
            subject: "Albert Einstein".to_string(),
            predicate: "developed".to_string(),
            object: "Theory of Relativity".to_string(),
            confidence: 0.95,
            source_databases: vec![],
            temporal_context: Some("1905".to_string()),
            cognitive_relevance: 0.9,
            relationship_type: None,
            extracted_patterns: vec![],
            neural_salience: 0.85,
        },
    ];
    
    let parser = build_test_parser().await;
    let intent = parser.parse("Who developed the Theory of Relativity?").await.unwrap();
    let answer = generator.generate_answer(facts, intent).await.unwrap();
    assert_eq!(answer.text, "Albert Einstein");
    
    // Test explanatory answer
    let facts = vec![
        CognitiveFact {
            subject: "Radioactivity".to_string(),
            predicate: "works_by".to_string(),
            object: "atomic decay".to_string(),
            confidence: 0.9,
            source_databases: vec![],
            temporal_context: None,
            cognitive_relevance: 0.8,
            relationship_type: None,
            extracted_patterns: vec![],
            neural_salience: 0.8,
        },
    ];
    
    let intent = parser.parse("How does radioactivity work?").await.unwrap();
    let answer = generator.generate_answer(facts, intent).await.unwrap();
    assert!(answer.text.contains("atomic decay"));
    
    // Test no information case
    let empty_facts = vec![];
    let intent = parser.parse("What is quantum entanglement?").await.unwrap();
    let answer = generator.generate_answer(empty_facts, intent).await.unwrap();
    assert!(answer.text.contains("don't have enough information"));
    assert_eq!(answer.confidence, 0.0);
}

#[tokio::test]
async fn test_temporal_context_extraction() {
    let parser = build_test_parser().await;
    
    // Test year extraction
    let intent = parser.parse("What happened in 1905?").await.unwrap();
    assert!(intent.temporal_context.is_some());
    let temporal = intent.temporal_context.unwrap();
    assert_eq!(temporal.start_time, Some("1905".to_string()));
    assert_eq!(temporal.end_time, Some("1905".to_string()));
    
    // Test range extraction
    let intent = parser.parse("What happened between 1900 and 1910?").await.unwrap();
    assert!(intent.temporal_context.is_some());
    let temporal = intent.temporal_context.unwrap();
    assert_eq!(temporal.start_time, Some("1900".to_string()));
    assert_eq!(temporal.end_time, Some("1910".to_string()));
    
    // Test relative time
    let intent = parser.parse("What happened before Einstein's theory?").await.unwrap();
    assert!(intent.temporal_context.is_some());
    let temporal = intent.temporal_context.unwrap();
    assert_eq!(temporal.relative_timeframe, Some("relative".to_string()));
}

async fn build_test_parser() -> CognitiveQuestionParser {
    let orchestrator = Arc::new(CognitiveOrchestrator::default());
    let attention_manager = Arc::new(AttentionManager::new());
    let working_memory = Arc::new(WorkingMemorySystem::new());
    let entity_extractor = Arc::new(CognitiveEntityExtractor::new());
    let metrics_collector = Arc::new(BrainMetricsCollector::new());
    let performance_monitor = Arc::new(PerformanceMonitor::new());
    
    CognitiveQuestionParser::new(
        orchestrator,
        attention_manager,
        working_memory,
        entity_extractor,
        metrics_collector,
        performance_monitor,
    )
}