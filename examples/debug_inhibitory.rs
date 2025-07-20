use llmkg::core::activation_config::ActivationConfig;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    BrainInspiredRelationship, RelationType
};

#[tokio::main]
async fn main() {
    let mut config = ActivationConfig::default();
    config.inhibition_strength = 1.5;
    
    let engine = ActivationPropagationEngine::new(config);

    // Complex network with mixed connections
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Hidden);
    let entity_d = BrainInspiredEntity::new("D".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;
    let key_d = entity_d.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();
    engine.add_entity(entity_d).await.unwrap();

    // A excites B and C
    let rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    let rel_ac = BrainInspiredRelationship::new(key_a, key_c, RelationType::RelatedTo);

    // B excites D
    let rel_bd = BrainInspiredRelationship::new(key_b, key_d, RelationType::RelatedTo);

    // C inhibits D
    let mut rel_cd = BrainInspiredRelationship::new(key_c, key_d, RelationType::RelatedTo);
    rel_cd.is_inhibitory = true;
    rel_cd.weight = 0.7;

    engine.add_relationship(rel_ab).await.unwrap();
    engine.add_relationship(rel_ac).await.unwrap();
    engine.add_relationship(rel_bd).await.unwrap();
    engine.add_relationship(rel_cd).await.unwrap();

    // Activate A
    let mut pattern = ActivationPattern::new("mixed_network".to_string());
    pattern.activations.insert(key_a, 1.0);

    let result = engine.propagate_activation(&pattern).await.unwrap();

    println!("=== Debug Trace ===");
    println!("Iterations completed: {}", result.iterations_completed);
    println!("\nActivation trace (first 20 steps):");
    for (i, step) in result.activation_trace.iter().take(20).enumerate() {
        println!("{}: {} - {:?} = {:.3}", 
            i, 
            step.concept_id, 
            step.operation_type,
            step.activation_level
        );
    }

    println!("\n=== Final Activations ===");
    println!("A: {:.3}", result.final_activations.get(&key_a).copied().unwrap_or(0.0));
    println!("B: {:.3}", result.final_activations.get(&key_b).copied().unwrap_or(0.0));
    println!("C: {:.3}", result.final_activations.get(&key_c).copied().unwrap_or(0.0));
    println!("D: {:.3}", result.final_activations.get(&key_d).copied().unwrap_or(0.0));
    
    let activation_d = result.final_activations.get(&key_d).copied().unwrap_or(0.0);
    
    println!("\n=== Test Assertions ===");
    println!("D activation: {:.6}", activation_d);
    println!("D > 0.0? {} (expected: true)", activation_d > 0.0);
    println!("D < 0.5? {} (expected: true)", activation_d < 0.5);
    
    if activation_d > 0.0 && activation_d < 0.5 {
        println!("\n✓ Test would PASS");
    } else {
        println!("\n✗ Test would FAIL");
        if activation_d <= 0.0 {
            println!("  - D has no activation (too much inhibition)");
        } else if activation_d >= 0.5 {
            println!("  - D has too much activation (not enough inhibition)");
        }
    }
}