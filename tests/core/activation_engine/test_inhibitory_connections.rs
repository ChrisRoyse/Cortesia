use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    BrainInspiredRelationship, RelationType
};
use llmkg::core::types::EntityKey;

#[tokio::test]
async fn test_basic_inhibitory_connection() {
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
    let mut pattern1 = ActivationPattern::new("excitatory_only".to_string());
    pattern1.activations.insert(key_a, 0.8);

    let result1 = engine.propagate_activation(&pattern1).await.unwrap();
    let c_activation_excitatory = result1.final_activations.get(&key_c).copied().unwrap_or(0.0);

    // Test 2: Both excitatory and inhibitory inputs
    let mut pattern2 = ActivationPattern::new("with_inhibition".to_string());
    pattern2.activations.insert(key_a, 0.8);
    pattern2.activations.insert(key_b, 0.6);

    let result2 = engine.propagate_activation(&pattern2).await.unwrap();
    let c_activation_inhibited = result2.final_activations.get(&key_c).copied().unwrap_or(0.0);

    // Verify inhibition reduces activation
    assert!(
        c_activation_inhibited < c_activation_excitatory,
        "Inhibitory input should reduce target activation"
    );
    assert!(
        c_activation_excitatory > 0.0,
        "Excitatory-only should produce positive activation"
    );
}

#[tokio::test]
async fn test_complete_inhibition() {
    let mut config = ActivationConfig::default();
    config.inhibition_strength = 3.0; // Strong inhibition
    
    let engine = ActivationPropagationEngine::new(config);

    // Create simple inhibitory pair
    let entity_inhibitor = BrainInspiredEntity::new("Inhibitor".to_string(), EntityDirection::Input);
    let entity_target = BrainInspiredEntity::new("Target".to_string(), EntityDirection::Output);

    let key_inhibitor = entity_inhibitor.id;
    let key_target = entity_target.id;

    engine.add_entity(entity_inhibitor).await.unwrap();
    engine.add_entity(entity_target).await.unwrap();

    // Strong inhibitory connection
    let mut rel = BrainInspiredRelationship::new(key_inhibitor, key_target, RelationType::RelatedTo);
    rel.is_inhibitory = true;
    rel.weight = 1.0;

    engine.add_relationship(rel).await.unwrap();

    // Pre-activate target and apply inhibition
    let mut pattern = ActivationPattern::new("complete_inhibition".to_string());
    pattern.activations.insert(key_target, 0.9); // Target starts activated
    pattern.activations.insert(key_inhibitor, 1.0); // Strong inhibitor

    let result = engine.propagate_activation(&pattern).await.unwrap();

    let final_target_activation = result.final_activations.get(&key_target).copied().unwrap_or(0.0);
    
    assert!(
        final_target_activation < 0.1,
        "Strong inhibition should nearly eliminate target activation"
    );
}

#[tokio::test]
async fn test_lateral_inhibition_competition() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create competing neurons with lateral inhibition
    let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Hidden);
    let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Hidden);
    let entity_input = BrainInspiredEntity::new("Input".to_string(), EntityDirection::Input);

    let key_a = entity_a.id;
    let key_b = entity_b.id;
    let key_input = entity_input.id;

    engine.add_entity(entity_a).await.unwrap();
    engine.add_entity(entity_b).await.unwrap();
    engine.add_entity(entity_input).await.unwrap();

    // Input connects to both A and B
    let mut rel_input_a = BrainInspiredRelationship::new(key_input, key_a, RelationType::RelatedTo);
    rel_input_a.weight = 0.9;
    
    let mut rel_input_b = BrainInspiredRelationship::new(key_input, key_b, RelationType::RelatedTo);
    rel_input_b.weight = 0.7; // Slightly weaker connection

    // Lateral inhibition between A and B
    let mut rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
    rel_ab.is_inhibitory = true;
    rel_ab.weight = 0.8;

    let mut rel_ba = BrainInspiredRelationship::new(key_b, key_a, RelationType::RelatedTo);
    rel_ba.is_inhibitory = true;
    rel_ba.weight = 0.8;

    engine.add_relationship(rel_input_a).await.unwrap();
    engine.add_relationship(rel_input_b).await.unwrap();
    engine.add_relationship(rel_ab).await.unwrap();
    engine.add_relationship(rel_ba).await.unwrap();

    // Activate input
    let mut pattern = ActivationPattern::new("lateral_inhibition".to_string());
    pattern.activations.insert(key_input, 1.0);

    let result = engine.propagate_activation(&pattern).await.unwrap();

    let activation_a = result.final_activations.get(&key_a).copied().unwrap_or(0.0);
    let activation_b = result.final_activations.get(&key_b).copied().unwrap_or(0.0);

    // A should win the competition due to stronger input connection
    assert!(
        activation_a > activation_b,
        "Stronger input should win lateral inhibition competition"
    );
    assert!(
        activation_b < 0.5,
        "Weaker competitor should be significantly inhibited"
    );
}

#[tokio::test]
async fn test_inhibitory_chain() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create inhibitory chain: A -i-> B -i-> C (double negative)
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

    // Test with B pre-activated
    let mut pattern = ActivationPattern::new("inhibitory_chain".to_string());
    pattern.activations.insert(key_a, 0.8);
    pattern.activations.insert(key_b, 0.6); // B starts with some activation
    pattern.activations.insert(key_c, 0.7); // C starts with some activation

    let result = engine.propagate_activation(&pattern).await.unwrap();

    let activation_b = result.final_activations.get(&key_b).copied().unwrap_or(0.0);
    let activation_c = result.final_activations.get(&key_c).copied().unwrap_or(0.0);

    // B should be inhibited by A
    assert!(
        activation_b < 0.3,
        "B should be inhibited by A"
    );
    
    // C might maintain activation since B is weakened (disinhibition)
    assert!(
        activation_c > activation_b,
        "C should be less inhibited due to weakened B"
    );
}

#[tokio::test]
async fn test_mixed_excitatory_inhibitory_network() {
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

    let activation_d = result.final_activations.get(&key_d).copied().unwrap_or(0.0);

    // D receives both excitation (via B) and inhibition (via C)
    assert!(
        activation_d > 0.0,
        "D should have some activation despite inhibition"
    );
    assert!(
        activation_d < 0.5,
        "D activation should be reduced by inhibition"
    );
}