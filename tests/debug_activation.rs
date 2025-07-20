use llmkg::core::activation_config::ActivationConfig;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    BrainInspiredRelationship, RelationType
};

#[tokio::main]
async fn main() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create two separate islands: (A -> B) and (C -> D)
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Output);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Input);
    let entity_d = BrainInspiredEntity::new("D".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;
    let key_d = entity_d.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();
    engine.add_entity(entity_d).await.unwrap();

    // Connect A to B
    let rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    engine.add_relationship(rel_ab).await.unwrap();

    // Connect C to D
    let rel_cd = BrainInspiredRelationship::new(key_c, key_d, RelationType::RelatedTo);
    engine.add_relationship(rel_cd).await.unwrap();

    // Activate only A
    let mut pattern = ActivationPattern::new("island_test".to_string());
    pattern.activations.insert(key_a, 0.8);

    println!("Initial activations:");
    println!("A: 0.8");
    println!("B: 0.0");
    println!("C: 0.0");
    println!("D: 0.0");
    println!();

    // Propagate activation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Print results
    let activation_a = result.final_activations.get(&key_a).copied().unwrap_or(0.0);
    let activation_b = result.final_activations.get(&key_b).copied().unwrap_or(0.0);
    let activation_c = result.final_activations.get(&key_c).copied().unwrap_or(0.0);
    let activation_d = result.final_activations.get(&key_d).copied().unwrap_or(0.0);

    println!("Final activations after propagation:");
    println!("A: {}", activation_a);
    println!("B: {}", activation_b);
    println!("C: {}", activation_c);
    println!("D: {}", activation_d);
    println!();
    
    println!("Converged: {}", result.converged);
    println!("Iterations: {}", result.iterations_completed);
    
    // Check assertions
    println!("\nAssertion checks:");
    println!("A > 0: {} (should be true)", activation_a > 0.0);
    println!("B > 0: {} (should be true)", activation_b > 0.0);
    println!("C == 0: {} (should be true)", activation_c == 0.0);
    println!("D == 0: {} (should be true)", activation_d == 0.0);
}