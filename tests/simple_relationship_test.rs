//! Simple test to verify relationship extraction works

use llmkg::core::relationship_extractor::{
    CognitiveRelationshipExtractor, CognitiveRelationshipType
};
use llmkg::cognitive::orchestrator::CognitiveOrchestrator;
use llmkg::cognitive::attention_manager::AttentionManager;
use llmkg::cognitive::working_memory::WorkingMemorySystem;
use llmkg::monitoring::brain_metrics_collector::BrainMetricsCollector;
use llmkg::monitoring::performance::PerformanceMonitor;
use std::sync::Arc;

#[tokio::test]
async fn test_basic_relationship_extraction() {
    // Setup cognitive components with defaults
    let cognitive_orchestrator = Arc::new(CognitiveOrchestrator::default());
    let attention_manager = Arc::new(AttentionManager::new());
    let working_memory = Arc::new(WorkingMemorySystem::new(1000));
    let metrics_collector = Arc::new(BrainMetricsCollector::new());
    let performance_monitor = Arc::new(PerformanceMonitor::new());
    
    let extractor = CognitiveRelationshipExtractor::new(
        cognitive_orchestrator,
        attention_manager,
        working_memory,
        metrics_collector,
        performance_monitor,
    );
    
    // Test simple extraction
    let test_cases = vec![
        ("Albert Einstein developed the Theory of Relativity", "developed", CognitiveRelationshipType::Developed),
        ("Marie Curie discovered polonium", "discovered", CognitiveRelationshipType::Discovered),
        ("Tesla invented the AC motor", "invented", CognitiveRelationshipType::Invented),
        ("Steve Jobs founded Apple", "founded", CognitiveRelationshipType::Founded),
        ("Einstein was born in Germany", "born in", CognitiveRelationshipType::BornIn),
        ("Marie Curie won the Nobel Prize", "won", CognitiveRelationshipType::Won),
    ];
    
    let mut success_count = 0;
    
    for (text, expected_predicate, expected_type) in test_cases {
        println!("Testing: {}", text);
        
        match extractor.extract_relationships(text).await {
            Ok(relationships) => {
                if relationships.is_empty() {
                    println!("  ✗ No relationships extracted");
                } else {
                    let found = relationships.iter().any(|r| {
                        r.predicate.to_lowercase().contains(expected_predicate) &&
                        r.relationship_type == expected_type
                    });
                    
                    if found {
                        success_count += 1;
                        println!("  ✓ Successfully extracted: {:?}", 
                            relationships.iter()
                                .find(|r| r.relationship_type == expected_type)
                                .map(|r| format!("{} - {} - {}", r.subject, r.predicate, r.object))
                                .unwrap_or_default()
                        );
                    } else {
                        println!("  ✗ Did not find expected relationship type {:?}", expected_type);
                        println!("    Found: {:?}", 
                            relationships.iter()
                                .map(|r| format!("{:?}: {} - {} - {}", r.relationship_type, r.subject, r.predicate, r.object))
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
            Err(e) => {
                println!("  ✗ Error: {:?}", e);
            }
        }
    }
    
    println!("\nSuccess rate: {}/{}", success_count, test_cases.len());
    assert!(success_count >= 4, "Should extract at least 4 relationships correctly");
}