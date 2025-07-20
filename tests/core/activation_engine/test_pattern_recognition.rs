use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    BrainInspiredRelationship, RelationType, LogicGate, LogicGateType
};
use llmkg::core::types::EntityKey;
use std::collections::HashMap;

#[tokio::test]
async fn test_animal_classification_pattern() {
    // Integration test: Build a network that classifies animals based on features
    let mut config = ActivationConfig::default();
    config.max_iterations = 50;
    config.convergence_threshold = 0.01;
    
    let engine = ActivationPropagationEngine::new(config);

    // Feature nodes (inputs)
    let has_fur = BrainInspiredEntity::new("has_fur".to_string(), EntityDirection::Input);
    let has_feathers = BrainInspiredEntity::new("has_feathers".to_string(), EntityDirection::Input);
    let can_fly = BrainInspiredEntity::new("can_fly".to_string(), EntityDirection::Input);
    let lays_eggs = BrainInspiredEntity::new("lays_eggs".to_string(), EntityDirection::Input);
    let has_four_legs = BrainInspiredEntity::new("has_four_legs".to_string(), EntityDirection::Input);

    // Classification nodes (outputs)
    let is_mammal = BrainInspiredEntity::new("is_mammal".to_string(), EntityDirection::Output);
    let is_bird = BrainInspiredEntity::new("is_bird".to_string(), EntityDirection::Output);
    let is_dog = BrainInspiredEntity::new("is_dog".to_string(), EntityDirection::Output);

    // Store keys
    let key_fur = has_fur.id;
    let key_feathers = has_feathers.id;
    let key_fly = can_fly.id;
    let key_eggs = lays_eggs.id;
    let key_four_legs = has_four_legs.id;
    let key_mammal = is_mammal.id;
    let key_bird = is_bird.id;
    let key_dog = is_dog.id;

    // Add all entities
    engine.add_entity(has_fur).await.unwrap();
    engine.add_entity(has_feathers).await.unwrap();
    engine.add_entity(can_fly).await.unwrap();
    engine.add_entity(lays_eggs).await.unwrap();
    engine.add_entity(has_four_legs).await.unwrap();
    engine.add_entity(is_mammal).await.unwrap();
    engine.add_entity(is_bird).await.unwrap();
    engine.add_entity(is_dog).await.unwrap();

    // Create mammal detection gate (has_fur AND NOT lays_eggs)
    let mammal_gate = BrainInspiredEntity::new("mammal_gate".to_string(), EntityDirection::Gate);
    let key_mammal_gate = mammal_gate.id;
    engine.add_entity(mammal_gate).await.unwrap();

    // Relationships for mammal detection
    let mut rel_fur_mammal = BrainInspiredRelationship::new(key_fur, key_mammal_gate, RelationType::RelatedTo);
    rel_fur_mammal.weight = 0.9;
    
    let mut rel_eggs_mammal = BrainInspiredRelationship::new(key_eggs, key_mammal_gate, RelationType::RelatedTo);
    rel_eggs_mammal.is_inhibitory = true;
    rel_eggs_mammal.weight = 0.8;
    
    let rel_gate_mammal = BrainInspiredRelationship::new(key_mammal_gate, key_mammal, RelationType::RelatedTo);

    engine.add_relationship(rel_fur_mammal).await.unwrap();
    engine.add_relationship(rel_eggs_mammal).await.unwrap();
    engine.add_relationship(rel_gate_mammal).await.unwrap();

    // Bird detection (has_feathers AND can_fly)
    let mut bird_gate = LogicGate::new(LogicGateType::And, 0.5);
    bird_gate.input_nodes.push(key_feathers);
    bird_gate.input_nodes.push(key_fly);
    bird_gate.output_nodes.push(key_bird);
    engine.add_logic_gate(bird_gate).await.unwrap();

    // Dog detection (is_mammal AND has_four_legs)
    let mut dog_gate = LogicGate::new(LogicGateType::And, 0.5);
    dog_gate.input_nodes.push(key_mammal);
    dog_gate.input_nodes.push(key_four_legs);
    dog_gate.output_nodes.push(key_dog);
    engine.add_logic_gate(dog_gate).await.unwrap();

    // Test 1: Classify a dog (mammal with fur, four legs, no eggs)
    let mut dog_pattern = ActivationPattern::new("dog_test".to_string());
    dog_pattern.activations.insert(key_fur, 0.9);
    dog_pattern.activations.insert(key_four_legs, 0.9);
    dog_pattern.activations.insert(key_eggs, 0.0);
    dog_pattern.activations.insert(key_feathers, 0.0);
    dog_pattern.activations.insert(key_fly, 0.0);

    let dog_result = engine.propagate_activation(&dog_pattern).await.unwrap();
    
    assert!(dog_result.final_activations.get(&key_mammal).copied().unwrap_or(0.0) > 0.7, 
        "Dog should be classified as mammal");
    assert!(dog_result.final_activations.get(&key_dog).copied().unwrap_or(0.0) > 0.5, 
        "Dog should be classified as dog");
    assert!(dog_result.final_activations.get(&key_bird).copied().unwrap_or(0.0) < 0.1, 
        "Dog should not be classified as bird");

    // Test 2: Classify a bird (feathers, can fly, lays eggs)
    let mut bird_pattern = ActivationPattern::new("bird_test".to_string());
    bird_pattern.activations.insert(key_feathers, 0.9);
    bird_pattern.activations.insert(key_fly, 0.8);
    bird_pattern.activations.insert(key_eggs, 0.9);
    bird_pattern.activations.insert(key_fur, 0.0);
    bird_pattern.activations.insert(key_four_legs, 0.0);

    let bird_result = engine.propagate_activation(&bird_pattern).await.unwrap();
    
    assert!(bird_result.final_activations.get(&key_bird).copied().unwrap_or(0.0) > 0.7, 
        "Bird should be classified as bird");
    assert!(bird_result.final_activations.get(&key_mammal).copied().unwrap_or(0.0) < 0.2, 
        "Bird should not be classified as mammal (inhibited by egg-laying)");
}

#[tokio::test]
async fn test_semantic_concept_network() {
    // Test semantic relationships and concept activation
    let mut config = ActivationConfig::default();
    config.decay_rate = 0.1;
    
    let engine = ActivationPropagationEngine::new(config);

    // Create semantic network around "computer" concept
    let computer = BrainInspiredEntity::new("computer".to_string(), EntityDirection::Hidden);
    let technology = BrainInspiredEntity::new("technology".to_string(), EntityDirection::Hidden);
    let electronics = BrainInspiredEntity::new("electronics".to_string(), EntityDirection::Hidden);
    let programming = BrainInspiredEntity::new("programming".to_string(), EntityDirection::Hidden);
    let hardware = BrainInspiredEntity::new("hardware".to_string(), EntityDirection::Hidden);
    let software = BrainInspiredEntity::new("software".to_string(), EntityDirection::Hidden);

    let key_computer = computer.id;
    let key_tech = technology.id;
    let key_electronics = electronics.id;
    let key_programming = programming.id;
    let key_hardware = hardware.id;
    let key_software = software.id;

    // Add entities
    engine.add_entity(computer).await.unwrap();
    engine.add_entity(technology).await.unwrap();
    engine.add_entity(electronics).await.unwrap();
    engine.add_entity(programming).await.unwrap();
    engine.add_entity(hardware).await.unwrap();
    engine.add_entity(software).await.unwrap();

    // Create semantic relationships
    let relationships = vec![
        (key_computer, key_tech, 0.9, RelationType::IsA),
        (key_computer, key_electronics, 0.8, RelationType::RelatedTo),
        (key_computer, key_programming, 0.7, RelationType::RelatedTo),
        (key_computer, key_hardware, 0.9, RelationType::HasProperty),
        (key_computer, key_software, 0.9, RelationType::HasProperty),
        (key_hardware, key_electronics, 0.8, RelationType::IsA),
        (key_software, key_programming, 0.9, RelationType::RelatedTo),
    ];

    for (source, target, weight, rel_type) in relationships {
        let mut rel = BrainInspiredRelationship::new(source, target, rel_type);
        rel.weight = weight;
        engine.add_relationship(rel).await.unwrap();
    }

    // Activate "computer" and see semantic spread
    let mut pattern = ActivationPattern::new("semantic_test".to_string());
    pattern.activations.insert(key_computer, 1.0);

    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Check semantic activation spread
    assert!(result.final_activations.get(&key_tech).copied().unwrap_or(0.0) > 0.7,
        "Technology should be strongly activated");
    assert!(result.final_activations.get(&key_hardware).copied().unwrap_or(0.0) > 0.7,
        "Hardware should be strongly activated");
    assert!(result.final_activations.get(&key_software).copied().unwrap_or(0.0) > 0.7,
        "Software should be strongly activated");
    assert!(result.final_activations.get(&key_electronics).copied().unwrap_or(0.0) > 0.5,
        "Electronics should be moderately activated");
    assert!(result.final_activations.get(&key_programming).copied().unwrap_or(0.0) > 0.4,
        "Programming should be activated through indirect connections");
}

#[tokio::test]
async fn test_competitive_pattern_recognition() {
    // Test winner-take-all pattern with lateral inhibition
    let mut config = ActivationConfig::default();
    config.inhibition_strength = 2.5;
    
    let engine = ActivationPropagationEngine::new(config);

    // Input features
    let feature1 = BrainInspiredEntity::new("feature1".to_string(), EntityDirection::Input);
    let feature2 = BrainInspiredEntity::new("feature2".to_string(), EntityDirection::Input);
    let feature3 = BrainInspiredEntity::new("feature3".to_string(), EntityDirection::Input);

    // Competing pattern detectors
    let pattern_a = BrainInspiredEntity::new("pattern_a".to_string(), EntityDirection::Output);
    let pattern_b = BrainInspiredEntity::new("pattern_b".to_string(), EntityDirection::Output);
    let pattern_c = BrainInspiredEntity::new("pattern_c".to_string(), EntityDirection::Output);

    let key_f1 = feature1.id;
    let key_f2 = feature2.id;
    let key_f3 = feature3.id;
    let key_pa = pattern_a.id;
    let key_pb = pattern_b.id;
    let key_pc = pattern_c.id;

    // Add all entities
    engine.add_entity(feature1).await.unwrap();
    engine.add_entity(feature2).await.unwrap();
    engine.add_entity(feature3).await.unwrap();
    engine.add_entity(pattern_a).await.unwrap();
    engine.add_entity(pattern_b).await.unwrap();
    engine.add_entity(pattern_c).await.unwrap();

    // Pattern A strongly responds to features 1 and 2
    let mut rel_f1_pa = BrainInspiredRelationship::new(key_f1, key_pa, RelationType::RelatedTo);
    rel_f1_pa.weight = 0.9;
    let mut rel_f2_pa = BrainInspiredRelationship::new(key_f2, key_pa, RelationType::RelatedTo);
    rel_f2_pa.weight = 0.9;

    // Pattern B responds to features 2 and 3
    let mut rel_f2_pb = BrainInspiredRelationship::new(key_f2, key_pb, RelationType::RelatedTo);
    rel_f2_pb.weight = 0.7;
    let mut rel_f3_pb = BrainInspiredRelationship::new(key_f3, key_pb, RelationType::RelatedTo);
    rel_f3_pb.weight = 0.7;

    // Pattern C weakly responds to all features
    let mut rel_f1_pc = BrainInspiredRelationship::new(key_f1, key_pc, RelationType::RelatedTo);
    rel_f1_pc.weight = 0.4;
    let mut rel_f2_pc = BrainInspiredRelationship::new(key_f2, key_pc, RelationType::RelatedTo);
    rel_f2_pc.weight = 0.4;
    let mut rel_f3_pc = BrainInspiredRelationship::new(key_f3, key_pc, RelationType::RelatedTo);
    rel_f3_pc.weight = 0.4;

    // Add feature connections
    engine.add_relationship(rel_f1_pa).await.unwrap();
    engine.add_relationship(rel_f2_pa).await.unwrap();
    engine.add_relationship(rel_f2_pb).await.unwrap();
    engine.add_relationship(rel_f3_pb).await.unwrap();
    engine.add_relationship(rel_f1_pc).await.unwrap();
    engine.add_relationship(rel_f2_pc).await.unwrap();
    engine.add_relationship(rel_f3_pc).await.unwrap();

    // Add lateral inhibition between patterns
    for (source, target) in [(key_pa, key_pb), (key_pa, key_pc), (key_pb, key_pa), 
                             (key_pb, key_pc), (key_pc, key_pa), (key_pc, key_pb)] {
        let mut inhib = BrainInspiredRelationship::new(source, target, RelationType::RelatedTo);
        inhib.is_inhibitory = true;
        inhib.weight = 0.8;
        engine.add_relationship(inhib).await.unwrap();
    }

    // Test with features 1 and 2 active (should activate Pattern A)
    let mut test_pattern = ActivationPattern::new("competitive_test".to_string());
    test_pattern.activations.insert(key_f1, 1.0);
    test_pattern.activations.insert(key_f2, 1.0);
    test_pattern.activations.insert(key_f3, 0.0);

    let result = engine.propagate_activation(&test_pattern).await.unwrap();

    let act_a = result.final_activations.get(&key_pa).copied().unwrap_or(0.0);
    let act_b = result.final_activations.get(&key_pb).copied().unwrap_or(0.0);
    let act_c = result.final_activations.get(&key_pc).copied().unwrap_or(0.0);

    // Pattern A should win the competition
    assert!(act_a > 0.6, "Pattern A should win with features 1&2");
    assert!(act_b < 0.3, "Pattern B should be inhibited");
    assert!(act_c < 0.2, "Pattern C should be strongly inhibited");
    assert!(act_a > act_b && act_a > act_c, "Pattern A should have highest activation");
}

#[tokio::test]
async fn test_hierarchical_concept_activation() {
    // Test hierarchical concept network with multiple levels
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Level 0: Specific items
    let sparrow = BrainInspiredEntity::new("sparrow".to_string(), EntityDirection::Input);
    let robin = BrainInspiredEntity::new("robin".to_string(), EntityDirection::Input);
    let oak = BrainInspiredEntity::new("oak".to_string(), EntityDirection::Input);
    let pine = BrainInspiredEntity::new("pine".to_string(), EntityDirection::Input);

    // Level 1: Categories
    let bird = BrainInspiredEntity::new("bird".to_string(), EntityDirection::Hidden);
    let tree = BrainInspiredEntity::new("tree".to_string(), EntityDirection::Hidden);

    // Level 2: Superordinate
    let animal = BrainInspiredEntity::new("animal".to_string(), EntityDirection::Hidden);
    let plant = BrainInspiredEntity::new("plant".to_string(), EntityDirection::Hidden);

    // Level 3: Most general
    let living_thing = BrainInspiredEntity::new("living_thing".to_string(), EntityDirection::Output);

    // Store keys
    let keys: HashMap<String, EntityKey> = vec![
        ("sparrow", sparrow.id),
        ("robin", robin.id),
        ("oak", oak.id),
        ("pine", pine.id),
        ("bird", bird.id),
        ("tree", tree.id),
        ("animal", animal.id),
        ("plant", plant.id),
        ("living_thing", living_thing.id),
    ].into_iter().map(|(k, v)| (k.to_string(), v)).collect();

    // Add all entities
    engine.add_entity(sparrow).await.unwrap();
    engine.add_entity(robin).await.unwrap();
    engine.add_entity(oak).await.unwrap();
    engine.add_entity(pine).await.unwrap();
    engine.add_entity(bird).await.unwrap();
    engine.add_entity(tree).await.unwrap();
    engine.add_entity(animal).await.unwrap();
    engine.add_entity(plant).await.unwrap();
    engine.add_entity(living_thing).await.unwrap();

    // Create hierarchical relationships
    let hierarchy = vec![
        ("sparrow", "bird", RelationType::IsA),
        ("robin", "bird", RelationType::IsA),
        ("oak", "tree", RelationType::IsA),
        ("pine", "tree", RelationType::IsA),
        ("bird", "animal", RelationType::IsA),
        ("tree", "plant", RelationType::IsA),
        ("animal", "living_thing", RelationType::IsA),
        ("plant", "living_thing", RelationType::IsA),
    ];

    for (source_name, target_name, rel_type) in hierarchy {
        let source = keys[source_name];
        let target = keys[target_name];
        let rel = BrainInspiredRelationship::new(source, target, rel_type);
        engine.add_relationship(rel).await.unwrap();
    }

    // Activate a specific bird
    let mut pattern = ActivationPattern::new("hierarchy_test".to_string());
    pattern.activations.insert(keys["sparrow"], 1.0);

    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Check hierarchical activation
    assert!(result.final_activations.get(&keys["bird"]).copied().unwrap_or(0.0) > 0.7,
        "Bird category should be activated");
    assert!(result.final_activations.get(&keys["animal"]).copied().unwrap_or(0.0) > 0.5,
        "Animal superordinate should be activated");
    assert!(result.final_activations.get(&keys["living_thing"]).copied().unwrap_or(0.0) > 0.3,
        "Living thing should be activated at top of hierarchy");
    
    // Non-bird categories should not be activated
    assert!(result.final_activations.get(&keys["tree"]).copied().unwrap_or(0.0) < 0.1,
        "Tree category should not be activated");
    assert!(result.final_activations.get(&keys["plant"]).copied().unwrap_or(0.0) < 0.1,
        "Plant superordinate should not be activated");
}