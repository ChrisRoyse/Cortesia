use std::sync::Arc;
use tokio::main;

use llmkg::cognitive::*;
use llmkg::core::brain_enhanced_graph::BrainEnhancedKnowledgeGraph;
use llmkg::core::brain_types::{BrainInspiredEntity, EntityDirection};
use llmkg::neural::neural_server::NeuralProcessingServer;

#[main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 LLMKG Phase 2 Cognitive Demo");
    println!("=================================");
    
    // 1. Setup brain-enhanced graph and neural server
    let graph = Arc::new(BrainEnhancedKnowledgeGraph::new_for_test());
    let neural_server = Arc::new(NeuralProcessingServer::new_mock());
    
    // 2. Add test knowledge
    println!("📝 Adding test knowledge...");
    
    let dog_entity = BrainInspiredEntity::new("dog".to_string(), EntityDirection::Input);
    let dog_key = graph.insert_brain_entity(dog_entity).await?;
    
    let mammal_entity = BrainInspiredEntity::new("mammal".to_string(), EntityDirection::Input);
    let mammal_key = graph.insert_brain_entity(mammal_entity).await?;
    
    let animal_entity = BrainInspiredEntity::new("animal".to_string(), EntityDirection::Input);
    let animal_key = graph.insert_brain_entity(animal_entity).await?;
    
    println!("   ✓ Added {} entities", 3);
    
    // 3. Test Convergent Thinking
    println!("\n🎯 Testing Convergent Thinking...");
    let convergent = ConvergentThinking::new(graph.clone(), neural_server.clone());
    
    let result = convergent.execute_convergent_query("What type is dog?", None).await?;
    println!("   Query: 'What type is dog?'");
    println!("   Answer: {}", result.answer);
    println!("   Confidence: {:.2}", result.confidence);
    println!("   Execution time: {} ms", result.execution_time_ms);
    
    // 4. Test Divergent Thinking  
    println!("\n🌟 Testing Divergent Thinking...");
    let divergent = DivergentThinking::new_with_params(
        graph.clone(), 
        neural_server.clone(),
        5,  // exploration_breadth
        0.3 // creativity_threshold
    );
    
    let div_result = divergent.execute_divergent_exploration(
        "animal", 
        ExplorationType::Instances
    ).await?;
    println!("   Query: 'animal' (exploring instances)");
    println!("   Found {} explorations", div_result.explorations.len());
    for (i, exploration) in div_result.explorations.iter().take(3).enumerate() {
        println!("     {}. {}", i + 1, exploration.concept);
    }
    
    // 5. Test Lateral Thinking
    println!("\n🔗 Testing Lateral Thinking...");
    let lateral = LateralThinking::new(graph.clone(), neural_server.clone());
    
    let lat_result = lateral.find_creative_connections(
        "dog", 
        "mammal", 
        Some(3)
    ).await?;
    println!("   Finding connections: dog ↔ mammal");
    println!("   Found {} bridge paths", lat_result.bridges.len());
    if !lat_result.bridges.is_empty() {
        let best_bridge = &lat_result.bridges[0];
        println!("   Best path: {} concepts", best_bridge.path.len());
        println!("   Novelty: {:.2}", best_bridge.novelty_score);
        println!("   Plausibility: {:.2}", best_bridge.plausibility_score);
    }
    
    // 6. Test Systems Thinking
    println!("\n🏗️  Testing Systems Thinking...");
    let systems = SystemsThinking::new(graph.clone(), neural_server.clone());
    
    let sys_result = systems.execute_hierarchical_reasoning(
        "What properties do dogs inherit?",
        SystemsReasoningType::AttributeInheritance
    ).await?;
    println!("   Query: 'What properties do dogs inherit?'");
    println!("   Hierarchy depth: {}", sys_result.hierarchy_path.len());
    println!("   Inherited attributes: {}", sys_result.inherited_attributes.len());
    
    // 7. Test Critical Thinking
    println!("\n🔍 Testing Critical Thinking...");
    let critical = CriticalThinking::new(graph.clone(), neural_server.clone());
    
    let crit_result = critical.execute_critical_analysis(
        "Are all dogs mammals?",
        ValidationLevel::Basic
    ).await?;
    println!("   Query: 'Are all dogs mammals?'");
    println!("   Resolved facts: {}", crit_result.resolved_facts.len());
    println!("   Contradictions found: {}", crit_result.contradictions_found.len());
    println!("   Overall uncertainty: {:.2}", crit_result.uncertainty_analysis.overall_uncertainty);
    
    // 8. Test Abstract Thinking
    println!("\n🧩 Testing Abstract Thinking...");
    let abstract_thinking = AbstractThinking::new(graph.clone(), neural_server.clone());
    
    let abs_result = abstract_thinking.execute_pattern_analysis(
        AnalysisScope::Global,
        PatternType::Structural
    ).await?;
    println!("   Analysis scope: Global structural patterns");
    println!("   Patterns found: {}", abs_result.patterns_found.len());
    println!("   Abstraction candidates: {}", abs_result.abstractions.len());
    
    // 9. Test Adaptive Thinking
    println!("\n🤖 Testing Adaptive Thinking...");
    let adaptive = AdaptiveThinking::new(graph.clone(), neural_server.clone());
    
    let adapt_result = adaptive.execute_adaptive_reasoning(
        "What is a dog?",
        None,
        vec![
            CognitivePatternType::Convergent,
            CognitivePatternType::Systems,
            CognitivePatternType::Lateral
        ]
    ).await?;
    println!("   Query: 'What is a dog?'");
    println!("   Selected strategy: {:?}", adapt_result.strategy_used.selected_patterns);
    println!("   Final answer: {}", adapt_result.final_answer);
    println!("   Ensemble confidence: {:.2}", adapt_result.confidence_distribution.ensemble_confidence);
    
    // 10. Test Cognitive Orchestrator
    println!("\n🎼 Testing Cognitive Orchestrator...");
    let orchestrator = CognitiveOrchestrator::new(
        graph.clone(),
        neural_server.clone(),
        CognitiveOrchestratorConfig::default()
    ).await?;
    
    let orch_result = orchestrator.reason(
        "How are dogs and animals related?",
        None,
        ReasoningStrategy::Automatic
    ).await?;
    println!("   Query: 'How are dogs and animals related?'");
    println!("   Strategy: Automatic selection");
    println!("   Answer: {}", orch_result.final_answer);
    println!("   Patterns used: {:?}", orch_result.execution_metadata.patterns_executed);
    println!("   Overall confidence: {:.2}", orch_result.quality_metrics.overall_confidence);
    
    println!("\n✅ Phase 2 Cognitive Demo completed successfully!");
    println!("🚀 All 7 cognitive patterns are operational!");
    
    Ok(())
}