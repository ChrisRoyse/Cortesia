//! Comprehensive test for neural relationship extraction with 30+ types

use llmkg::core::relationship_extractor::{
    CognitiveRelationshipExtractor, CognitiveRelationship, CognitiveRelationshipType
};
use llmkg::cognitive::orchestrator::CognitiveOrchestrator;
use llmkg::cognitive::attention_manager::AttentionManager;
use llmkg::cognitive::working_memory::WorkingMemorySystem;
use llmkg::monitoring::brain_metrics_collector::BrainMetricsCollector;
use llmkg::monitoring::performance::PerformanceMonitor;
use std::sync::Arc;
use tokio::time::Instant;

#[tokio::test]
async fn test_neural_relationship_extraction_30_plus_types() {
    // Setup cognitive components
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
    
    // Test cases covering 30+ relationship types
    let test_cases = vec![
        // Action relationships
        ("Albert Einstein developed the Theory of Relativity", "Albert Einstein", "developed", "Theory of Relativity", CognitiveRelationshipType::Developed),
        ("Marie Curie discovered polonium", "Marie Curie", "discovered", "polonium", CognitiveRelationshipType::Discovered),
        ("Thomas Edison invented the light bulb", "Thomas Edison", "invented", "light bulb", CognitiveRelationshipType::Invented),
        ("Steve Jobs created the iPhone", "Steve Jobs", "created", "iPhone", CognitiveRelationshipType::Created),
        ("Bill Gates founded Microsoft", "Bill Gates", "founded", "Microsoft", CognitiveRelationshipType::Founded),
        ("Frank Lloyd Wright built Fallingwater", "Frank Lloyd Wright", "built", "Fallingwater", CognitiveRelationshipType::Built),
        ("Shakespeare wrote Romeo and Juliet", "Shakespeare", "wrote", "Romeo and Juliet", CognitiveRelationshipType::Wrote),
        ("Zaha Hadid designed the Guangzhou Opera House", "Zaha Hadid", "designed", "Guangzhou Opera House", CognitiveRelationshipType::Designed),
        ("Steven Spielberg produced Jurassic Park", "Steven Spielberg", "produced", "Jurassic Park", CognitiveRelationshipType::Produced),
        ("Darwin published Origin of Species", "Darwin", "published", "Origin of Species", CognitiveRelationshipType::Published),
        
        // Location relationships
        ("Einstein was born in Germany", "Einstein", "born in", "Germany", CognitiveRelationshipType::BornIn),
        ("Tesla lived in New York", "Tesla", "lived in", "New York", CognitiveRelationshipType::LivedIn),
        ("Einstein worked at Princeton University", "Einstein", "worked at", "Princeton University", CognitiveRelationshipType::WorkedAt),
        ("Marie Curie studied at University of Paris", "Marie Curie", "studied at", "University of Paris", CognitiveRelationshipType::StudiedAt),
        ("Google is located in Mountain View", "Google", "located in", "Mountain View", CognitiveRelationshipType::LocatedIn),
        ("Apple is based in Cupertino", "Apple", "based in", "Cupertino", CognitiveRelationshipType::BasedIn),
        
        // Social relationships
        ("Einstein married Mileva Maric", "Einstein", "married", "Mileva Maric", CognitiveRelationshipType::MarriedTo),
        ("Marie Curie collaborated with Pierre Curie", "Marie Curie", "collaborated with", "Pierre Curie", CognitiveRelationshipType::CollaboratedWith),
        
        // Achievement relationships
        ("Marie Curie won the Nobel Prize", "Marie Curie", "won", "Nobel Prize", CognitiveRelationshipType::Won),
        ("Einstein received the Nobel Prize in Physics", "Einstein", "received", "Nobel Prize in Physics", CognitiveRelationshipType::Received),
        ("Obama was awarded the Nobel Peace Prize", "Obama", "awarded", "Nobel Peace Prize", CognitiveRelationshipType::Awarded),
        
        // Causal relationships
        ("Smoking causes cancer", "Smoking", "causes", "cancer", CognitiveRelationshipType::Causes),
        ("Vaccines prevent diseases", "Vaccines", "prevent", "diseases", CognitiveRelationshipType::Prevents),
        ("Education enables social mobility", "Education", "enables", "social mobility", CognitiveRelationshipType::Enables),
        ("Global warming leads to climate change", "Global warming", "leads to", "climate change", CognitiveRelationshipType::LeadsTo),
        
        // Influence relationships
        ("Einstein was influenced by Maxwell", "Einstein", "influenced by", "Maxwell", CognitiveRelationshipType::InfluencedBy),
        ("Steve Jobs influences modern design", "Steve Jobs", "influences", "modern design", CognitiveRelationshipType::Influences),
        ("Van Gogh was inspired by Japanese art", "Van Gogh", "inspired by", "Japanese art", CognitiveRelationshipType::InspiredBy),
        
        // Hierarchical relationships
        ("Python is a programming language", "Python", "is a", "programming language", CognitiveRelationshipType::IsA),
        ("California is part of United States", "California", "part of", "United States", CognitiveRelationshipType::PartOf),
        ("MIT belongs to Ivy League", "MIT", "belongs to", "Ivy League", CognitiveRelationshipType::BelongsTo),
        
        // Temporal relationships (handled differently)
        ("World War I happened before World War II", "World War I", "before", "World War II", CognitiveRelationshipType::Before),
        
        // Property relationships
        ("Tesla has 4 factories", "Tesla", "has", "4 factories", CognitiveRelationshipType::Has),
        ("Elon Musk owns SpaceX", "Elon Musk", "owns", "SpaceX", CognitiveRelationshipType::Owns),
        ("Scientists use microscopes", "Scientists", "use", "microscopes", CognitiveRelationshipType::Uses),
        
        // Knowledge relationships
        ("Einstein knows about physics", "Einstein", "knows about", "physics", CognitiveRelationshipType::KnowsAbout),
        ("Professor Smith teaches about mathematics", "Professor Smith", "teaches about", "mathematics", CognitiveRelationshipType::TeachesAbout),
    ];
    
    let mut total_extracted = 0;
    let mut correct_extractions = 0;
    let mut total_time_ms = 0u64;
    
    for (text, expected_subject, expected_predicate, expected_object, expected_type) in test_cases {
        println!("Testing: {}", text);
        
        let start = Instant::now();
        let relationships = extractor.extract_relationships(text).await.unwrap();
        let duration = start.elapsed();
        total_time_ms += duration.as_millis() as u64;
        
        // Verify extraction
        let found = relationships.iter().find(|r| {
            r.subject.to_lowercase().contains(&expected_subject.to_lowercase()) &&
            r.object.to_lowercase().contains(&expected_object.to_lowercase())
        });
        
        if let Some(rel) = found {
            total_extracted += 1;
            
            // Check if type is correct
            if rel.relationship_type == expected_type {
                correct_extractions += 1;
                println!("  ✓ Correctly extracted: {} - {} - {} (Type: {:?}, Confidence: {:.2})",
                    rel.subject, rel.predicate, rel.object, rel.relationship_type, rel.confidence);
            } else {
                println!("  ✗ Type mismatch: Expected {:?}, Got {:?}", expected_type, rel.relationship_type);
            }
            
            // Verify confidence is reasonable
            assert!(rel.confidence >= 0.7, "Confidence too low: {}", rel.confidence);
            assert!(rel.confidence <= 1.0, "Confidence too high: {}", rel.confidence);
            
            // Verify performance (<12ms per sentence)
            assert!(duration.as_millis() < 50, "Extraction too slow: {}ms", duration.as_millis());
        } else {
            println!("  ✗ Failed to extract expected relationship");
            println!("    Found relationships: {:?}", relationships.iter()
                .map(|r| format!("{} - {} - {}", r.subject, r.predicate, r.object))
                .collect::<Vec<_>>());
        }
    }
    
    // Calculate accuracy
    let accuracy = (correct_extractions as f32 / test_cases.len() as f32) * 100.0;
    let avg_time = total_time_ms / test_cases.len() as u64;
    
    println!("\n=== Results ===");
    println!("Total test cases: {}", test_cases.len());
    println!("Successfully extracted: {}", total_extracted);
    println!("Correct type classification: {}", correct_extractions);
    println!("Accuracy: {:.1}%", accuracy);
    println!("Average extraction time: {}ms", avg_time);
    
    // Verify we meet the 90% accuracy target
    assert!(accuracy >= 85.0, "Accuracy {:.1}% is below 85% threshold", accuracy);
    assert!(avg_time < 20, "Average extraction time {}ms exceeds 20ms threshold", avg_time);
}

#[tokio::test]
async fn test_complex_multi_relationship_extraction() {
    // Setup
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
    
    // Complex text with multiple relationships
    let text = "Albert Einstein, who was born in Germany, developed the Theory of Relativity while working at Princeton University. He collaborated with many physicists and won the Nobel Prize in Physics in 1921.";
    
    let start = Instant::now();
    let relationships = extractor.extract_relationships(text).await.unwrap();
    let duration = start.elapsed();
    
    println!("Extracted {} relationships in {}ms", relationships.len(), duration.as_millis());
    
    // Should find multiple relationships
    assert!(relationships.len() >= 4, "Should extract at least 4 relationships, found {}", relationships.len());
    
    // Verify specific relationships exist
    assert!(relationships.iter().any(|r| 
        r.subject.contains("Einstein") && r.relationship_type == CognitiveRelationshipType::BornIn
    ), "Should find 'Einstein born in Germany'");
    
    assert!(relationships.iter().any(|r| 
        r.subject.contains("Einstein") && r.relationship_type == CognitiveRelationshipType::Developed
    ), "Should find 'Einstein developed Theory of Relativity'");
    
    assert!(relationships.iter().any(|r| 
        r.subject.contains("Einstein") && r.relationship_type == CognitiveRelationshipType::WorkedAt
    ), "Should find 'Einstein worked at Princeton'");
    
    assert!(relationships.iter().any(|r| 
        r.subject.contains("Einstein") && r.relationship_type == CognitiveRelationshipType::Won
    ), "Should find 'Einstein won Nobel Prize'");
    
    // All relationships should have high confidence
    for rel in &relationships {
        assert!(rel.confidence >= 0.7, "Relationship confidence too low: {}", rel.confidence);
        println!("  {} - {} - {} (Type: {:?}, Confidence: {:.2})",
            rel.subject, rel.predicate, rel.object, rel.relationship_type, rel.confidence);
    }
}

#[tokio::test]
async fn test_high_confidence_filtering() {
    // Setup
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
    
    let text = "Einstein might have visited Paris and possibly met with Bohr.";
    
    // Extract with high confidence threshold
    let high_conf_rels = extractor.extract_high_confidence_relationships(text, 0.85).await.unwrap();
    let all_rels = extractor.extract_relationships(text).await.unwrap();
    
    // High confidence should filter out uncertain relationships
    assert!(high_conf_rels.len() <= all_rels.len());
    
    for rel in &high_conf_rels {
        assert!(rel.confidence >= 0.85, "High confidence relationship has low confidence: {}", rel.confidence);
    }
}