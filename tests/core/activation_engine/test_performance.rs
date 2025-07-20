use llmkg::core::activation_config::ActivationConfig;
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    BrainInspiredRelationship, RelationType, LogicGate, LogicGateType
};
use llmkg::core::types::EntityKey;
use std::time::Instant;

#[tokio::test]
async fn test_large_network_performance() {
    let mut config = ActivationConfig::default();
    config.max_iterations = 50; // Limit iterations for performance test
    
    let engine = ActivationPropagationEngine::new(config);

    // Create a large network: 100 inputs -> 200 hidden -> 100 outputs
    let mut input_keys = Vec::new();
    let mut hidden_keys = Vec::new();
    let mut output_keys = Vec::new();

    // Add input layer
    for i in 0..100 {
        let entity = BrainInspiredEntity::new(format!("Input{}", i), EntityDirection::Input);
        input_keys.push(entity.id);
        engine.add_entity(entity).await.unwrap();
    }

    // Add hidden layer
    for i in 0..200 {
        let entity = BrainInspiredEntity::new(format!("Hidden{}", i), EntityDirection::Hidden);
        hidden_keys.push(entity.id);
        engine.add_entity(entity).await.unwrap();
    }

    // Add output layer
    for i in 0..100 {
        let entity = BrainInspiredEntity::new(format!("Output{}", i), EntityDirection::Output);
        output_keys.push(entity.id);
        engine.add_entity(entity).await.unwrap();
    }

    // Create connections: sparse connectivity (10% connection probability)
    let mut connection_count = 0;
    
    // Input to hidden connections
    for &input in &input_keys {
        for &hidden in &hidden_keys {
            if rand::random::<f32>() < 0.1 {
                let rel = BrainInspiredRelationship::new(input, hidden, RelationType::RelatedTo);
                engine.add_relationship(rel).await.unwrap();
                connection_count += 1;
            }
        }
    }

    // Hidden to output connections
    for &hidden in &hidden_keys {
        for &output in &output_keys {
            if rand::random::<f32>() < 0.1 {
                let rel = BrainInspiredRelationship::new(hidden, output, RelationType::RelatedTo);
                engine.add_relationship(rel).await.unwrap();
                connection_count += 1;
            }
        }
    }

    // Add some inhibitory connections
    for i in 0..50 {
        let source = hidden_keys[i];
        let target = hidden_keys[i + 50];
        let mut rel = BrainInspiredRelationship::new(source, target, RelationType::RelatedTo);
        rel.is_inhibitory = true;
        engine.add_relationship(rel).await.unwrap();
    }

    println!("Created network with {} entities and {} connections", 
             input_keys.len() + hidden_keys.len() + output_keys.len(),
             connection_count);

    // Create activation pattern with 20% of inputs active
    let mut pattern = ActivationPattern::new("large_network_test".to_string());
    for (i, &key) in input_keys.iter().enumerate() {
        if i % 5 == 0 {
            pattern.activations.insert(key, 0.8);
        }
    }

    // Measure propagation time
    let start = Instant::now();
    let result = engine.propagate_activation(&pattern).await.unwrap();
    let duration = start.elapsed();

    println!("Propagation completed in {:?}", duration);
    println!("Iterations: {}", result.iterations_completed);
    println!("Converged: {}", result.converged);
    println!("Active outputs: {}", 
             result.final_activations.iter()
                 .filter(|(k, v)| output_keys.contains(k) && **v > 0.1)
                 .count());

    // Performance assertions
    assert!(duration.as_secs() < 5, "Large network should complete within 5 seconds");
    assert!(result.iterations_completed <= 50, "Should respect max iterations");
}

#[tokio::test]
async fn test_deep_network_propagation() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create a deep chain: Input -> Layer1 -> Layer2 -> ... -> Layer10 -> Output
    let mut layer_keys = Vec::new();
    let depth = 10;
    let width = 5; // 5 nodes per layer

    // Create input layer
    let mut input_layer = Vec::new();
    for i in 0..width {
        let entity = BrainInspiredEntity::new(format!("Input{}", i), EntityDirection::Input);
        input_layer.push(entity.id);
        engine.add_entity(entity).await.unwrap();
    }
    layer_keys.push(input_layer);

    // Create hidden layers
    for layer in 1..depth {
        let mut hidden_layer = Vec::new();
        for i in 0..width {
            let entity = BrainInspiredEntity::new(
                format!("Layer{}Node{}", layer, i), 
                EntityDirection::Hidden
            );
            hidden_layer.push(entity.id);
            engine.add_entity(entity).await.unwrap();
        }
        layer_keys.push(hidden_layer);
    }

    // Create output layer
    let mut output_layer = Vec::new();
    for i in 0..width {
        let entity = BrainInspiredEntity::new(format!("Output{}", i), EntityDirection::Output);
        output_layer.push(entity.id);
        engine.add_entity(entity).await.unwrap();
    }
    layer_keys.push(output_layer.clone());

    // Connect layers (full connectivity between adjacent layers)
    for layer_idx in 0..layer_keys.len() - 1 {
        let current_layer = &layer_keys[layer_idx];
        let next_layer = &layer_keys[layer_idx + 1];
        
        for &source in current_layer {
            for &target in next_layer {
                let mut rel = BrainInspiredRelationship::new(source, target, RelationType::RelatedTo);
                rel.weight = 0.7; // Slight attenuation per layer
                engine.add_relationship(rel).await.unwrap();
            }
        }
    }

    // Activate first input
    let mut pattern = ActivationPattern::new("deep_network".to_string());
    pattern.activations.insert(layer_keys[0][0], 1.0);

    let start = Instant::now();
    let result = engine.propagate_activation(&pattern).await.unwrap();
    let duration = start.elapsed();

    // Check that signal reaches the output
    let output_activations: Vec<f32> = output_layer.iter()
        .map(|&key| result.final_activations.get(&key).copied().unwrap_or(0.0))
        .collect();

    println!("Deep network propagation in {:?}", duration);
    println!("Output activations: {:?}", output_activations);

    // Signal should reach output but be attenuated
    let max_output = output_activations.iter().cloned().fold(0.0, f32::max);
    assert!(max_output > 0.0, "Signal should reach output layer");
    assert!(max_output < 0.5, "Signal should be attenuated through {} layers", depth);
}

#[tokio::test]
async fn test_complex_gate_network_performance() {
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create a network with many logic gates
    let num_inputs = 50;
    let num_gates = 100;
    let num_outputs = 25;

    let mut input_keys = Vec::new();
    let mut gate_keys = Vec::new();
    let mut output_keys = Vec::new();

    // Create inputs
    for i in 0..num_inputs {
        let entity = BrainInspiredEntity::new(format!("Input{}", i), EntityDirection::Input);
        input_keys.push(entity.id);
        engine.add_entity(entity).await.unwrap();
    }

    // Create outputs
    for i in 0..num_outputs {
        let entity = BrainInspiredEntity::new(format!("Output{}", i), EntityDirection::Output);
        output_keys.push(entity.id);
        engine.add_entity(entity).await.unwrap();
    }

    // Create various types of gates
    let gate_types = vec![
        LogicGateType::And,
        LogicGateType::Or,
        LogicGateType::Threshold,
        LogicGateType::Inhibitory,
    ];

    for i in 0..num_gates {
        let gate_type = gate_types[i % gate_types.len()];
        let mut gate = LogicGate::new(gate_type, 0.5);
        
        // Random input connections (2-4 inputs per gate)
        let num_inputs_for_gate = 2 + (i % 3);
        for _ in 0..num_inputs_for_gate {
            let input_idx = rand::random::<usize>() % input_keys.len();
            gate.input_nodes.push(input_keys[input_idx]);
        }
        
        // Random output connection
        let output_idx = rand::random::<usize>() % output_keys.len();
        gate.output_nodes.push(output_keys[output_idx]);
        
        gate_keys.push(gate.gate_id);
        engine.add_logic_gate(gate).await.unwrap();
    }

    // Activate random subset of inputs
    let mut pattern = ActivationPattern::new("gate_network".to_string());
    for (i, &key) in input_keys.iter().enumerate() {
        if i % 3 == 0 {
            pattern.activations.insert(key, rand::random::<f32>());
        }
    }

    let start = Instant::now();
    let result = engine.propagate_activation(&pattern).await.unwrap();
    let duration = start.elapsed();

    println!("Gate network propagation in {:?}", duration);
    println!("Total energy: {}", result.total_energy);

    assert!(duration.as_millis() < 2000, "Gate network should complete within 2 seconds");
}

#[tokio::test] 
async fn test_memory_efficiency() {
    let mut config = ActivationConfig::default();
    config.max_iterations = 10; // Limit for memory test
    
    // Test memory usage doesn't grow excessively with trace
    let engine = ActivationPropagationEngine::new(config);

    // Create moderate network
    for i in 0..100 {
        let entity = BrainInspiredEntity::new(format!("Entity{}", i), EntityDirection::Hidden);
        engine.add_entity(entity).await.unwrap();
    }

    // Create pattern activating all entities
    let mut pattern = ActivationPattern::new("memory_test".to_string());
    let entities = engine.get_current_state().await.unwrap();
    for key in entities.keys() {
        pattern.activations.insert(*key, 0.5);
    }

    let result = engine.propagate_activation(&pattern).await.unwrap();
    
    // Check trace size is reasonable
    let trace_size = result.activation_trace.len();
    let expected_max_trace = 100 * 4 * 10; // entities * steps_per_iteration * max_iterations
    
    assert!(
        trace_size < expected_max_trace,
        "Trace size {} should be reasonable (< {})",
        trace_size, expected_max_trace
    );
}