//! Integration test demonstrating real question answering

use llmkg::core::question_parser::{CognitiveQuestionParser, CognitiveQuestionType, FactualSubtype};
use llmkg::core::answer_generator::{CognitiveAnswerGenerator, CognitiveFact};
use llmkg::core::entity_extractor::CognitiveEntityExtractor;
use llmkg::cognitive::orchestrator::CognitiveOrchestrator;
use llmkg::cognitive::attention_manager::AttentionManager;
use llmkg::cognitive::working_memory::WorkingMemorySystem;
use llmkg::monitoring::brain_metrics_collector::BrainMetricsCollector;
use llmkg::monitoring::performance::PerformanceMonitor;
use std::sync::Arc;
use std::time::Instant;

#[tokio::test]
async fn test_complete_question_answering_pipeline() {
    // Setup components
    let orchestrator = Arc::new(CognitiveOrchestrator::default());
    let attention_manager = Arc::new(AttentionManager::new());
    let working_memory = Arc::new(WorkingMemorySystem::new());
    let entity_extractor = Arc::new(CognitiveEntityExtractor::new());
    let metrics_collector = Arc::new(BrainMetricsCollector::new());
    let performance_monitor = Arc::new(PerformanceMonitor::new());
    
    // Create parser and generator
    let parser = CognitiveQuestionParser::new(
        orchestrator.clone(),
        attention_manager,
        working_memory.clone(),
        entity_extractor,
        metrics_collector.clone(),
        performance_monitor.clone(),
    );
    
    let generator = CognitiveAnswerGenerator::new(
        orchestrator,
        working_memory,
        metrics_collector,
        performance_monitor,
    );
    
    // Test questions and facts
    let test_cases = vec![
        (
            "Who discovered polonium?",
            vec![CognitiveFact {
                subject: "Marie Curie".to_string(),
                predicate: "discovered".to_string(),
                object: "polonium".to_string(),
                confidence: 0.95,
                source_databases: vec![],
                temporal_context: Some("1898".to_string()),
                cognitive_relevance: 0.0,
                relationship_type: None,
                extracted_patterns: vec![],
                neural_salience: 0.9,
            }],
            "Marie Curie",
        ),
        (
            "What is the theory of relativity?",
            vec![CognitiveFact {
                subject: "Theory of relativity".to_string(),
                predicate: "is".to_string(),
                object: "physics theory describing spacetime".to_string(),
                confidence: 0.9,
                source_databases: vec![],
                temporal_context: None,
                cognitive_relevance: 0.0,
                relationship_type: None,
                extracted_patterns: vec![],
                neural_salience: 0.85,
            }],
            "Theory of relativity is physics theory describing spacetime",
        ),
        (
            "When did Einstein publish his theory?",
            vec![CognitiveFact {
                subject: "Einstein".to_string(),
                predicate: "published theory in".to_string(),
                object: "1905".to_string(),
                confidence: 0.92,
                source_databases: vec![],
                temporal_context: Some("1905".to_string()),
                cognitive_relevance: 0.0,
                relationship_type: None,
                extracted_patterns: vec![],
                neural_salience: 0.88,
            }],
            "1905",
        ),
    ];
    
    for (question, facts, expected_answer) in test_cases {
        let start = Instant::now();
        
        // Parse question
        let intent = parser.parse(question).await.unwrap();
        let parse_time = start.elapsed();
        
        // Generate answer
        let answer_start = Instant::now();
        let answer = generator.generate_answer(facts, intent).await.unwrap();
        let answer_time = answer_start.elapsed();
        
        let total_time = start.elapsed();
        
        // Verify results
        println!("\nQuestion: {}", question);
        println!("Question Type: {:?}", answer.cognitive_patterns_used);
        println!("Answer: {}", answer.text);
        println!("Confidence: {:.2}", answer.confidence);
        println!("Parse time: {:?}", parse_time);
        println!("Answer time: {:?}", answer_time);
        println!("Total time: {:?}", total_time);
        
        // Assert performance
        assert!(total_time.as_millis() < 20, "Total time exceeded 20ms target");
        assert!(answer.text.contains(expected_answer), 
            "Expected '{}' to contain '{}'", answer.text, expected_answer);
        assert!(answer.confidence > 0.7, "Confidence too low: {}", answer.confidence);
    }
}

#[tokio::test]
async fn test_fact_ranking_performance() {
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
    
    // Create many facts to test ranking performance
    let mut facts = vec![];
    for i in 0..100 {
        facts.push(CognitiveFact {
            subject: format!("Entity{}", i),
            predicate: if i % 10 == 0 { "discovered".to_string() } else { "related_to".to_string() },
            object: format!("Object{}", i),
            confidence: 0.5 + (i as f32) * 0.005,
            source_databases: vec![],
            temporal_context: None,
            cognitive_relevance: 0.0,
            relationship_type: None,
            extracted_patterns: vec![],
            neural_salience: 0.5,
        });
    }
    
    // Add highly relevant fact
    facts.push(CognitiveFact {
        subject: "Marie Curie".to_string(),
        predicate: "discovered".to_string(),
        object: "radium".to_string(),
        confidence: 0.95,
        source_databases: vec![],
        temporal_context: Some("1898".to_string()),
        cognitive_relevance: 0.0,
        relationship_type: None,
        extracted_patterns: vec![],
        neural_salience: 0.9,
    });
    
    // Parse question
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
    
    // Time the answer generation with many facts
    let start = Instant::now();
    let answer = generator.generate_answer(facts, intent).await.unwrap();
    let elapsed = start.elapsed();
    
    println!("\nFact ranking performance:");
    println!("Facts processed: 101");
    println!("Time taken: {:?}", elapsed);
    println!("Answer: {}", answer.text);
    println!("Top fact relevance: {:.2}", answer.supporting_facts[0].cognitive_relevance);
    
    // Verify performance and correctness
    assert!(elapsed.as_millis() < 20, "Fact ranking exceeded 20ms target");
    assert_eq!(answer.text, "Marie Curie");
    assert!(answer.supporting_facts[0].cognitive_relevance > 0.8);
}