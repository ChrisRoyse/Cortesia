use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    BrainInspiredRelationship, RelationType
};
use llmkg::core::types::EntityKey;

#[tokio::main]
async fn main() {
    let mut config = ActivationConfig::default();
    config.inhibition_strength = 2.0;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create network: A (excitatory) -> C, B (inhibitory) -> C
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Input);
    let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Output);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_c = entity_c.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_c).await.unwrap();

    // Excitatory connection from A to C
    let rel_ac = BrainInspiredRelationship::new(key_a, key_c, RelationType::RelatedTo);

    // Inhibitory connection from B to C
    let mut rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
    rel_bc.is_inhibitory = true;

    engine.add_relationship(rel_ac).await.unwrap();
    engine.add_relationship(rel_bc).await.unwrap();

    // Test 1: Only excitatory input
    println!("Test 1: Only excitatory input (A=0.8)");
    let mut pattern1 = ActivationPattern::new("excitatory_only".to_string());
    pattern1.activations.insert(key_a, 0.8);

    let result1 = engine.propagate_activation(&pattern1).await.unwrap();
    let c_activation_excitatory = result1.final_activations.get(&key_c).copied().unwrap_or(0.0);
    println!("C activation (excitatory only): {}", c_activation_excitatory);

    // Test 2: Both excitatory and inhibitory inputs
    println!("\nTest 2: Both excitatory and inhibitory inputs (A=0.8, B=0.6)");
    let mut pattern2 = ActivationPattern::new("with_inhibition".to_string());
    pattern2.activations.insert(key_a, 0.8);
    pattern2.activations.insert(key_b, 0.6);

    let result2 = engine.propagate_activation(&pattern2).await.unwrap();
    let c_activation_inhibited = result2.final_activations.get(&key_c).copied().unwrap_or(0.0);
    println!("C activation (with inhibition): {}", c_activation_inhibited);

    // Verify inhibition reduces activation
    println!("\nComparison:");
    println!("C with excitation only: {}", c_activation_excitatory);
    println!("C with inhibition: {}", c_activation_inhibited);
    println!("Reduction: {}", c_activation_excitatory - c_activation_inhibited);
    println!("Test passes: {}", c_activation_inhibited < c_activation_excitatory);
}