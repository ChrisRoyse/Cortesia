use llmkg::cognitive::*;
use llmkg::cognitive::orchestrator::{CognitiveOrchestrator, CognitiveOrchestratorConfig};
use llmkg::cognitive::types::*;
use llmkg::versioning::temporal_graph::TemporalKnowledgeGraph;
use llmkg::neural::neural_server::NeuralProcessingServer;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::types::EntityData;
use std::sync::Arc;
use tokio::sync::RwLock;
use ahash::AHashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LLMKG Cognitive Patterns Demo ===\n");
    
    // Initialize the knowledge graph and neural server
    let temporal_graph = TemporalKnowledgeGraph::new_default();
    let brain_graph = Arc::new(BrainEnhancedKnowledgeGraph::new(temporal_graph));
    let neural_server = Arc::new(NeuralProcessingServer::new_test().await?);
    
    // Populate the graph with sample data
    populate_sample_data(&brain_graph).await?;
    
    // Create the cognitive orchestrator
    let orchestrator = CognitiveOrchestrator::new(
        brain_graph.clone(),
        neural_server.clone(),
        CognitiveOrchestratorConfig::default(),
    ).await?;
    
    // Demonstrate each cognitive pattern
    demonstrate_convergent_thinking(&orchestrator).await?;
    demonstrate_divergent_thinking(&orchestrator).await?;
    demonstrate_lateral_thinking(&orchestrator).await?;
    demonstrate_systems_thinking(&orchestrator).await?;
    demonstrate_critical_thinking(&orchestrator).await?;
    demonstrate_abstract_thinking(&orchestrator).await?;
    demonstrate_adaptive_thinking(&orchestrator).await?;
    
    println!("\n=== Demo Complete ===");
    Ok(())
}

async fn populate_sample_data(graph: &Arc<BrainEnhancedKnowledgeGraph>) -> Result<(), Box<dyn std::error::Error>> {
    println!("Populating knowledge graph with sample data...\n");
    
    // Create sample brain-inspired entities
    use llmkg::core::brain_types::{BrainInspiredEntity, EntityDirection};
    
    // Add animal hierarchy
    let animal_entity = BrainInspiredEntity::new("animal".to_string(), EntityDirection::Input);
    graph.insert_brain_entity(animal_entity).await?;
    
    let mammal_entity = BrainInspiredEntity::new("mammal".to_string(), EntityDirection::Input);
    graph.insert_brain_entity(mammal_entity).await?;
    
    let dog_entity = BrainInspiredEntity::new("dog".to_string(), EntityDirection::Input);
    graph.insert_brain_entity(dog_entity).await?;
    
    
    let cat_entity = BrainInspiredEntity::new("cat".to_string(), EntityDirection::Input);
    graph.insert_brain_entity(cat_entity).await?;
    
    let bird_entity = BrainInspiredEntity::new("bird".to_string(), EntityDirection::Input);
    graph.insert_brain_entity(bird_entity).await?;
    
    let fido_entity = BrainInspiredEntity::new("fido".to_string(), EntityDirection::Input);
    graph.insert_brain_entity(fido_entity).await?;
    
    let tripper_entity = BrainInspiredEntity::new("tripper".to_string(), EntityDirection::Input);
    graph.insert_brain_entity(tripper_entity).await?;
    
    let ai_entity = BrainInspiredEntity::new("AI".to_string(), EntityDirection::Input);
    graph.insert_brain_entity(ai_entity).await?;
    
    let art_entity = BrainInspiredEntity::new("art".to_string(), EntityDirection::Input);
    graph.insert_brain_entity(art_entity).await?;
    
    
    println!("Sample data loaded successfully.\n");
    Ok(())
}

async fn demonstrate_convergent_thinking(orchestrator: &CognitiveOrchestrator) -> Result<(), Box<dyn std::error::Error>> {
    println!("1. CONVERGENT THINKING (Direct, Focused Retrieval)");
    println!("   Query: 'What sound does a dog make?'");
    
    let result = orchestrator.reason(
        "What sound does a dog make?",
        None,
        ReasoningStrategy::Specific(CognitivePatternType::Convergent),
    ).await?;
    
    println!("   Final answer: {}", result.final_answer);
    println!("   Confidence: {:.2}", result.quality_metrics.overall_confidence);
    println!("   Reasoning steps: {}", result.reasoning_trace.len());
    println!();
    
    Ok(())
}

async fn demonstrate_divergent_thinking(orchestrator: &CognitiveOrchestrator) -> Result<(), Box<dyn std::error::Error>> {
    println!("2. DIVERGENT THINKING (Creative Exploration)");
    println!("   Query: 'What are types of animals?'");
    
    let result = orchestrator.reason(
        "What are types of animals?",
        None,
        ReasoningStrategy::Specific(CognitivePatternType::Divergent),
    ).await?;
    
    println!("   Final answer: {}", result.final_answer);
    println!("   Confidence: {:.2}", result.quality_metrics.overall_confidence);
    println!("   Patterns executed: {:?}", result.execution_metadata.patterns_executed);
    println!();
    
    Ok(())
}

async fn demonstrate_lateral_thinking(orchestrator: &CognitiveOrchestrator) -> Result<(), Box<dyn std::error::Error>> {
    println!("3. LATERAL THINKING (Creative Connections)");
    println!("   Query: 'How is AI related to art?'");
    
    let result = orchestrator.reason(
        "How is AI related to art?",
        None,
        ReasoningStrategy::Specific(CognitivePatternType::Lateral),
    ).await?;
    
    println!("   Final answer: {}", result.final_answer);
    println!("   Confidence: {:.2}", result.quality_metrics.overall_confidence);
    println!("   Novelty score: {:.2}", result.quality_metrics.novelty_score);
    println!();
    
    Ok(())
}

async fn demonstrate_systems_thinking(orchestrator: &CognitiveOrchestrator) -> Result<(), Box<dyn std::error::Error>> {
    println!("4. SYSTEMS THINKING (Hierarchical Reasoning)");
    println!("   Query: 'What properties does a dog inherit from being a mammal?'");
    
    let result = orchestrator.reason(
        "What properties does a dog inherit from being a mammal?",
        None,
        ReasoningStrategy::Specific(CognitivePatternType::Systems),
    ).await?;
    
    println!("   Final answer: {}", result.final_answer);
    println!("   Confidence: {:.2}", result.quality_metrics.overall_confidence);
    println!("   Completeness score: {:.2}", result.quality_metrics.completeness_score);
    println!();
    
    Ok(())
}

async fn demonstrate_critical_thinking(orchestrator: &CognitiveOrchestrator) -> Result<(), Box<dyn std::error::Error>> {
    println!("5. CRITICAL THINKING (Contradiction Resolution)");
    println!("   Query: 'How many legs does Tripper have?'");
    println!("   (Note: Tripper is a dog with only 3 legs)");
    
    let result = orchestrator.reason(
        "How many legs does Tripper have?",
        None,
        ReasoningStrategy::Specific(CognitivePatternType::Critical),
    ).await?;
    
    println!("   Final answer: {}", result.final_answer);
    println!("   Confidence: {:.2}", result.quality_metrics.overall_confidence);
    println!("   Consistency score: {:.2}", result.quality_metrics.consistency_score);
    println!();
    
    Ok(())
}

async fn demonstrate_abstract_thinking(orchestrator: &CognitiveOrchestrator) -> Result<(), Box<dyn std::error::Error>> {
    println!("6. ABSTRACT THINKING (Pattern Recognition)");
    println!("   Query: 'What patterns exist in the animal hierarchy?'");
    
    let result = orchestrator.reason(
        "What patterns exist in the animal hierarchy?",
        None,
        ReasoningStrategy::Specific(CognitivePatternType::Abstract),
    ).await?;
    
    println!("   Final answer: {}", result.final_answer);
    println!("   Confidence: {:.2}", result.quality_metrics.overall_confidence);
    println!("   Efficiency score: {:.2}", result.quality_metrics.efficiency_score);
    println!();
    
    Ok(())
}

async fn demonstrate_adaptive_thinking(orchestrator: &CognitiveOrchestrator) -> Result<(), Box<dyn std::error::Error>> {
    println!("7. ADAPTIVE THINKING (Automatic Strategy Selection)");
    println!("   Query: 'Tell me about pets and their relationship to AI'");
    println!("   (This complex query requires multiple cognitive patterns)");
    
    let result = orchestrator.reason(
        "Tell me about pets and their relationship to AI",
        None,
        ReasoningStrategy::Automatic,
    ).await?;
    
    println!("   Selected strategy: {:?}", result.strategy_used);
    println!("   Final answer: {}", result.final_answer);
    println!("   Patterns executed: {:?}", result.execution_metadata.patterns_executed);
    println!("   Confidence: {:.2}", result.quality_metrics.overall_confidence);
    println!("   Total execution time: {}ms", result.execution_metadata.total_time_ms);
    println!();
    
    Ok(())
}