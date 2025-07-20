use llmkg::core::activation_config::{ActivationConfig, PropagationResult};
use llmkg::core::activation_engine::ActivationPropagationEngine;
use llmkg::core::brain_types::{
    BrainInspiredEntity, EntityDirection, ActivationPattern, 
    BrainInspiredRelationship, RelationType, LogicGate, LogicGateType
};
use llmkg::core::types::EntityKey;
use std::collections::HashMap;

use super::test_constants::*;
use super::test_helpers::*;

// ==================== UNIT TESTS FOR propagate_activation ====================

#[tokio::test]
async fn test_propagate_activation_happy_path() {
    // Test normal propagation with a simple connected graph
    let (engine, entity_map) = TestNetworkBuilder::new()
        .add_input_neuron("sensory_input")
        .add_hidden_neuron("interneuron")
        .add_output_neuron("motor_output")
        .connect("sensory_input", "interneuron", MEDIUM_WEIGHT)
        .connect("interneuron", "motor_output", MEDIUM_WEIGHT)
        .build()
        .await;

    // Create activation pattern
    let pattern = create_activation_pattern(
        "test_happy_path",
        vec![("sensory_input", STRONG_ACTIVATION)],
        &entity_map
    );

    // Test propagation with timing assertion
    let result = assert_timed_execution(
        || engine.propagate_activation(&pattern),
        SIMPLE_PROPAGATION_TIMEOUT,
        "Simple chain propagation"
    ).await.unwrap();

    // Verify convergence
    assert!(
        result.converged || result.iterations_completed > 0,
        "Propagation should either converge or complete at least one iteration"
    );
    
    // Verify specific activations with detailed assertions
    let sensory_key = entity_map["sensory_input"];
    let interneuron_key = entity_map["interneuron"];
    let motor_key = entity_map["motor_output"];
    
    assert_activation_equals(
        result.final_activations[&sensory_key],
        STRONG_ACTIVATION,
        "sensory_input",
        "Input neurons should maintain their initial activation"
    );
    
    assert_neuron_activated(
        &result.final_activations,
        &interneuron_key,
        "interneuron",
        "Interneuron should receive activation from sensory input"
    );
    
    assert_neuron_activated(
        &result.final_activations,
        &motor_key,
        "motor_output",
        "Motor output should receive activation through the chain"
    );
    
    // Verify activation propagation strength
    let interneuron_activation = result.final_activations[&interneuron_key];
    let expected_interneuron = STRONG_ACTIVATION * MEDIUM_WEIGHT;
    assert_activation_in_range(
        interneuron_activation,
        expected_interneuron * 0.9,  // Allow 10% variance
        expected_interneuron * 1.1,
        "interneuron",
        "Interneuron activation should be proportional to input * weight"
    );
    
    // Verify energy calculation
    assert!(
        result.total_energy > ZERO_ACTIVATION,
        "Total energy should be positive for active network, got {}",
        result.total_energy
    );
    
    // Verify trace recording
    assert!(
        !result.activation_trace.is_empty(),
        "Activation trace should contain at least {} entries for {} iterations",
        result.iterations_completed,
        result.iterations_completed
    );
    
    // Verify trace contains expected neurons
    let trace_neurons: Vec<EntityKey> = result.activation_trace.iter()
        .map(|step| step.entity_key)
        .collect();
    assert!(
        trace_neurons.contains(&sensory_key),
        "Trace should include sensory input activations"
    );
    assert!(
        trace_neurons.contains(&interneuron_key),
        "Trace should include interneuron activations"
    );
}

#[tokio::test]
async fn test_propagate_activation_empty_graph() {
    // Test propagation on an empty graph (no entities)
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create empty activation pattern
    let pattern = ActivationPattern::new("test_empty".to_string());

    // Test propagation on empty graph
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify results with specific assertions
    assert!(
        result.final_activations.is_empty(),
        "Empty graph should have no activations, found {} activations",
        result.final_activations.len()
    );
    
    assert_activation_equals(
        result.total_energy,
        ZERO_ACTIVATION,
        "total_energy",
        "Empty graph should have exactly zero energy"
    );
    
    assert!(
        result.converged,
        "Empty graph should converge immediately (no nodes to process)"
    );
    
    assert!(
        result.activation_trace.is_empty(),
        "Empty graph should have no trace entries, found {}",
        result.activation_trace.len()
    );
    
    assert_eq!(
        result.iterations_completed, 0,
        "Empty graph should complete zero iterations"
    );
}

#[tokio::test]
async fn test_propagate_activation_single_node() {
    // Test propagation with a single isolated node
    let (engine, entity_map) = TestNetworkBuilder::new()
        .add_input_neuron("isolated_neuron")
        .build()
        .await;

    let pattern = create_activation_pattern(
        "test_single",
        vec![("isolated_neuron", MEDIUM_ACTIVATION)],
        &entity_map
    );

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify results
    let neuron_key = entity_map["isolated_neuron"];
    
    assert_eq!(
        result.final_activations.len(), 1,
        "Single node network should have exactly one activation"
    );
    
    assert_activation_equals(
        result.final_activations[&neuron_key],
        MEDIUM_ACTIVATION,
        "isolated_neuron",
        "Input node should maintain exact initial activation"
    );
    
    assert!(
        result.converged,
        "Single node with no connections should converge immediately"
    );
    
    let expected_energy = MEDIUM_ACTIVATION * MEDIUM_ACTIVATION;
    assert_activation_equals(
        result.total_energy,
        expected_energy,
        "energy",
        "Energy should equal activation squared for single node"
    );
}

#[tokio::test]
async fn test_propagate_activation_disconnected_components() {
    // Test propagation with multiple disconnected components
    let (engine, entity_map) = TestNetworkBuilder::new()
        // Component 1: Visual pathway
        .add_input_neuron("retina")
        .add_output_neuron("visual_cortex")
        .connect("retina", "visual_cortex", STRONG_WEIGHT)
        // Component 2: Auditory pathway (disconnected)
        .add_input_neuron("cochlea")
        .add_output_neuron("auditory_cortex")
        .connect("cochlea", "auditory_cortex", STRONG_WEIGHT)
        .build()
        .await;

    // Activate only visual pathway
    let pattern = create_activation_pattern(
        "test_disconnected",
        vec![("retina", STRONG_ACTIVATION)],
        &entity_map
    );

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify visual pathway is activated
    assert_neuron_activated(
        &result.final_activations,
        &entity_map["retina"],
        "retina",
        "Visual input should maintain activation"
    );
    
    assert_neuron_activated(
        &result.final_activations,
        &entity_map["visual_cortex"],
        "visual_cortex",
        "Visual cortex should receive activation from retina"
    );
    
    // Verify auditory pathway remains silent
    assert_neuron_silent(
        &result.final_activations,
        &entity_map["cochlea"],
        "cochlea",
        "Disconnected auditory input should remain inactive"
    );
    
    assert_neuron_silent(
        &result.final_activations,
        &entity_map["auditory_cortex"],
        "auditory_cortex",
        "Disconnected auditory cortex should remain inactive"
    );
    
    // Verify only active component contributes to energy
    let visual_contribution = result.final_activations[&entity_map["retina"]].powi(2) +
                            result.final_activations[&entity_map["visual_cortex"]].powi(2);
    assert!(
        (result.total_energy - visual_contribution).abs() < EPSILON,
        "Total energy {} should equal visual pathway contribution {}",
        result.total_energy, visual_contribution
    );
}

#[tokio::test]
async fn test_propagate_activation_max_iterations() {
    // Test that propagation respects max_iterations limit
    let mut config = ActivationConfig::default();
    config.max_iterations = MIN_ITERATIONS;
    config.convergence_threshold = TIGHT_CONVERGENCE;
    
    // Create a recurrent network that won't converge quickly
    let (engine, entity_map) = TestNetworkBuilder::new()
        .with_config(config)
        .add_hidden_neuron("neuron_a")
        .add_hidden_neuron("neuron_b")
        .add_hidden_neuron("neuron_c")
        .connect("neuron_a", "neuron_b", STRONG_WEIGHT)
        .connect("neuron_b", "neuron_c", STRONG_WEIGHT)
        .connect("neuron_c", "neuron_a", STRONG_WEIGHT)  // Creates cycle
        .build()
        .await;

    // Activate the cycle
    let pattern = create_activation_pattern(
        "test_max_iter",
        vec![("neuron_a", MAX_ACTIVATION)],
        &entity_map
    );

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify iteration limit was respected
    assert!(
        !result.converged,
        "Cyclic network with tight convergence threshold should not converge in {} iterations",
        MIN_ITERATIONS
    );
    
    assert_eq!(
        result.iterations_completed, MIN_ITERATIONS,
        "Should stop exactly at max_iterations limit"
    );
    
    // Verify all neurons in cycle have some activation
    for neuron_name in ["neuron_a", "neuron_b", "neuron_c"] {
        assert_neuron_activated(
            &result.final_activations,
            &entity_map[neuron_name],
            neuron_name,
            "All neurons in cycle should have received activation"
        );
    }
}

#[tokio::test]
async fn test_propagate_activation_with_all_node_types() {
    // Test propagation with all node types: Input, Hidden, Output, Gate
    let (engine, entity_map) = TestNetworkBuilder::new()
        .add_input_neuron("sensory_1")
        .add_input_neuron("sensory_2")
        .add_hidden_neuron("interneuron")
        .add_gate_neuron("and_gate")
        .add_output_neuron("motor_output")
        .connect("sensory_1", "interneuron", MEDIUM_WEIGHT)
        .add_and_gate(
            vec!["sensory_1", "sensory_2"],
            vec!["motor_output"],
            GATE_THRESHOLD
        )
        .build()
        .await;

    // Activate both inputs to trigger AND gate
    let pattern = create_activation_pattern(
        "test_all_types",
        vec![
            ("sensory_1", STRONG_ACTIVATION),
            ("sensory_2", STRONG_ACTIVATION)
        ],
        &entity_map
    );

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify all node types were processed
    assert_neuron_activated(
        &result.final_activations,
        &entity_map["sensory_1"],
        "sensory_1",
        "Input neuron 1 should maintain activation"
    );
    
    assert_neuron_activated(
        &result.final_activations,
        &entity_map["sensory_2"],
        "sensory_2",
        "Input neuron 2 should maintain activation"
    );
    
    assert_neuron_activated(
        &result.final_activations,
        &entity_map["interneuron"],
        "interneuron",
        "Hidden neuron should be activated by sensory_1"
    );
    
    assert_neuron_activated(
        &result.final_activations,
        &entity_map["motor_output"],
        "motor_output",
        "Output should be activated by AND gate (both inputs high)"
    );
    
    // Verify AND gate behavior
    let output_activation = result.final_activations[&entity_map["motor_output"]];
    assert!(
        output_activation > GATE_THRESHOLD,
        "AND gate with two high inputs should produce output above threshold, got {}",
        output_activation
    );
}

#[tokio::test]
async fn test_propagate_activation_error_handling_missing_nodes() {
    // Test behavior when relationships reference non-existent nodes
    let config = ActivationConfig::default();
    let engine = ActivationPropagationEngine::new(config);

    // Create only one entity
    let existing_neuron = BrainInspiredEntity::new("existing_neuron".to_string(), EntityDirection::Hidden);
    let key_exists = existing_neuron.id;
    engine.add_entity(existing_neuron).await.unwrap();

    // Create relationship referencing non-existent node
    let missing_entity = BrainInspiredEntity::new("missing_neuron".to_string(), EntityDirection::Hidden);
    let key_missing = missing_entity.id;
    // Note: We intentionally don't add the missing entity to the engine
    let rel = BrainInspiredRelationship::new(key_exists, key_missing, RelationType::RelatedTo);
    engine.add_relationship(rel).await.unwrap();

    // Try to propagate
    let mut pattern = ActivationPattern::new("test_missing".to_string());
    pattern.activations.insert(key_exists, MEDIUM_ACTIVATION);

    // Should not panic, should handle gracefully
    let result = engine.propagate_activation(&pattern).await.unwrap();
    
    // Verify it handled the missing node gracefully
    assert!(
        result.final_activations.contains_key(&key_exists),
        "Existing node should be processed despite dangling connection"
    );
    
    assert!(
        !result.final_activations.contains_key(&key_missing),
        "Missing node should not appear in final activations"
    );
    
    assert_activation_equals(
        result.final_activations[&key_exists],
        MEDIUM_ACTIVATION,
        "existing_neuron",
        "Existing node should maintain activation despite error"
    );
}

#[tokio::test]
async fn test_propagate_activation_extreme_values() {
    // Test propagation with extreme activation values
    let (engine, entity_map) = TestNetworkBuilder::new()
        .add_input_neuron("extreme_input")
        .add_output_neuron("clamped_output")
        .connect("extreme_input", "clamped_output", EXTREME_WEIGHT)
        .build()
        .await;

    // Test with extreme activation
    let pattern = create_activation_pattern(
        "test_extreme",
        vec![("extreme_input", EXCESSIVE_ACTIVATION)],
        &entity_map
    );

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Verify activations are clamped to reasonable range
    let output_activation = result.final_activations[&entity_map["clamped_output"]];
    assert_activation_in_range(
        output_activation,
        ZERO_ACTIVATION,
        MAX_ACTIVATION,
        "clamped_output",
        "Activation should be clamped to valid range [0, 1]"
    );
    
    // Verify input is also clamped
    let input_activation = result.final_activations[&entity_map["extreme_input"]];
    assert!(
        input_activation <= MAX_ACTIVATION,
        "Input activation {} should be clamped to maximum {}",
        input_activation, MAX_ACTIVATION
    );
}

#[tokio::test]
async fn test_propagate_activation_convergence_behavior() {
    // Test that convergence detection works properly
    let mut config = ActivationConfig::default();
    config.convergence_threshold = LOOSE_CONVERGENCE;
    config.max_iterations = MAX_ITERATIONS;
    
    // Create stable feedforward network that should converge quickly
    let (engine, entity_map) = TestNetworkBuilder::new()
        .with_config(config)
        .add_input_neuron("stable_input")
        .add_output_neuron("stable_output")
        .connect("stable_input", "stable_output", MEDIUM_WEIGHT)
        .build()
        .await;

    // Activate
    let pattern = create_activation_pattern(
        "test_convergence",
        vec![("stable_input", STRONG_ACTIVATION)],
        &entity_map
    );

    // Test propagation
    let result = engine.propagate_activation(&pattern).await.unwrap();

    // Should converge quickly for this simple network
    assert!(
        result.converged,
        "Simple feedforward network should converge with loose threshold"
    );
    
    assert!(
        result.iterations_completed < STANDARD_ITERATIONS / 5,
        "Simple network should converge in fewer than {} iterations, took {}",
        STANDARD_ITERATIONS / 5,
        result.iterations_completed
    );
    
    // Verify stable final state
    let expected_output = STRONG_ACTIVATION * MEDIUM_WEIGHT;
    assert_activation_in_range(
        result.final_activations[&entity_map["stable_output"]],
        expected_output * 0.95,
        expected_output * 1.05,
        "stable_output",
        "Converged output should be close to expected value"
    );
}