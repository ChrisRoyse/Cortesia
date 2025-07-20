use std::collections::HashMap;
use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    BrainInspiredRelationship, RelationType
};
use llmkg::core::types::EntityKey;

#[tokio::main]
async fn main() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create inhibitory chain: A -i-> B -i-> C
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();

    // A inhibits B
    let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel_ab.is_inhibitory = true;

    // B inhibits C
    let mut rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
    rel_bc.is_inhibitory = true;

    engine.add_relationship(rel_ab).await.unwrap();
    engine.add_relationship(rel_bc).await.unwrap();

    // Test with initial activations
    let mut pattern = ActivationPattern::new("inhibitory_chain".to_string());
    pattern.activations.insert(key_a, 0.8);
    pattern.activations.insert(key_b, 0.6); // B starts with some activation
    pattern.activations.insert(key_c, 0.7); // C starts with some activation

    let result = engine.propagate_activation(&pattern).await.unwrap();

    let activation_a = result.final_activations.get(&key_a).copied().unwrap_or(0.0);
    let activation_b = result.final_activations.get(&key_b).copied().unwrap_or(0.0);
    let activation_c = result.final_activations.get(&key_c).copied().unwrap_or(0.0);

    println!("Initial activations: A=0.8, B=0.6, C=0.7");
    println!("Final activations: A={:.3}, B={:.3}, C={:.3}", activation_a, activation_b, activation_c);
    println!("B < 0.3? {}", activation_b < 0.3);
    println!("C > B? {}", activation_c > activation_b);
    
    // Print trace for debugging
    println!("\nActivation trace:");
    for step in &result.activation_trace {
        println!("  Step {}: {} = {:.3} ({:?})", 
            step.step_id, 
            step.concept_id, 
            step.activation_level,
            step.operation_type
        );
    }
}