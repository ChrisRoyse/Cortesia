use llmkg::core::activation_engine::{ActivationPropagationEngine, ActivationConfig, PropagationResult};
use llmkg::core::brain_types::{
    BrainInspiredEntity, LogicGate, BrainInspiredRelationship, ActivationPattern,
    EntityDirection, LogicGateType, RelationType
};
use llmkg::core::types::EntityKey;
use std::collections::HashMap;
use slotmap::Key;

/// Helper function to create test configuration
fn create_test_config() -> ActivationConfig {
    ActivationConfig {
        max_iterations: 50,
        convergence_threshold: 0.001,
        decay_rate: 0.1,
        inhibition_strength: 2.0,
        default_threshold: 0.5,
    }
}

/// Helper function to create and add multiple entities
async fn create_entities(
    engine: &ActivationPropagationEngine,
    names: Vec<(&str, EntityDirection)>,
) -> HashMap<String, EntityKey> {
    let mut keys = HashMap::new();
    for (name, direction) in names {
        let entity = BrainInspiredEntity::new(name.to_string(), direction);
        let key = engine.add_entity(entity).await.unwrap();
        keys.insert(name.to_string(), key);
    }
    keys
}

#[tokio::test]
async fn test_pattern_recognition_cognitive_task() {
    // Create engine with configuration optimized for pattern recognition
    let mut config = create_test_config();
    config.inhibition_strength = 1.5; // Lower inhibition for pattern matching
    let engine = ActivationPropagationEngine::new(config);

    // Create pattern recognition network:
    // Input layer: feature detectors
    // Hidden layer: pattern components
    // Output layer: pattern classifiers
    let entities = create_entities(&engine, vec![
        // Input features
        ("feature_edge", EntityDirection::Input),
        ("feature_corner", EntityDirection::Input),
        ("feature_curve", EntityDirection::Input),
        ("feature_color_red", EntityDirection::Input),
        ("feature_color_blue", EntityDirection::Input),
        
        // Hidden pattern components
        ("pattern_square", EntityDirection::Hidden),
        ("pattern_circle", EntityDirection::Hidden),
        ("pattern_triangle", EntityDirection::Hidden),
        
        // Output classifiers
        ("shape_square", EntityDirection::Output),
        ("shape_circle", EntityDirection::Output),
        ("shape_triangle", EntityDirection::Output),
    ]).await;

    // Create AND gates for pattern detection
    let square_gate = LogicGate {
        gate_id: EntityKey::default(),
        gate_type: LogicGateType::And,
        input_nodes: vec![entities["feature_edge"], entities["feature_corner"]],
        output_nodes: vec![entities["pattern_square"]],
        threshold: 0.6,
        weight_matrix: vec![1.0, 1.0],
    };
    let square_gate_key = engine.add_logic_gate(square_gate).await.unwrap();

    let circle_gate = LogicGate {
        gate_id: EntityKey::default(),
        gate_type: LogicGateType::And,
        input_nodes: vec![entities["feature_curve"]],
        output_nodes: vec![entities["pattern_circle"]],
        threshold: 0.5,
        weight_matrix: vec![1.0],
    };
    let circle_gate_key = engine.add_logic_gate(circle_gate).await.unwrap();

    // Create relationships for feature → pattern → classifier flow
    let relationships = vec![
        // Square detection path
        (entities["feature_edge"], square_gate_key, 0.8, false),
        (entities["feature_corner"], square_gate_key, 0.8, false),
        (entities["pattern_square"], entities["shape_square"], 0.9, false),
        
        // Circle detection path
        (entities["feature_curve"], circle_gate_key, 0.9, false),
        (entities["pattern_circle"], entities["shape_circle"], 0.9, false),
        
        // Inhibitory connections between competing shapes
        (entities["shape_square"], entities["shape_circle"], 0.5, true),
        (entities["shape_circle"], entities["shape_square"], 0.5, true),
        (entities["shape_square"], entities["shape_triangle"], 0.3, true),
        (entities["shape_triangle"], entities["shape_square"], 0.3, true),
    ];

    for (source, target, weight, inhibitory) in relationships {
        let mut rel = BrainInspiredRelationship::new(source, target, RelationType::RelatedTo);
        rel.weight = weight;
        rel.is_inhibitory = inhibitory;
        engine.add_relationship(rel).await.unwrap();
    }

    // Test 1: Recognize square pattern
    let mut square_pattern = ActivationPattern::new("square_recognition".to_string());
    square_pattern.activations.insert(entities["feature_edge"], 0.9);
    square_pattern.activations.insert(entities["feature_corner"], 0.8);
    square_pattern.activations.insert(entities["feature_color_red"], 0.7);

    let result = engine.propagate_activation(&square_pattern).await.unwrap();
    
    // Verify square is recognized with highest activation
    let square_activation = result.final_activations.get(&entities["shape_square"]).copied().unwrap_or(0.0);
    let circle_activation = result.final_activations.get(&entities["shape_circle"]).copied().unwrap_or(0.0);
    let triangle_activation = result.final_activations.get(&entities["shape_triangle"]).copied().unwrap_or(0.0);
    
    assert!(square_activation > circle_activation, "Square should have higher activation than circle");
    assert!(square_activation > triangle_activation, "Square should have higher activation than triangle");
    assert!(square_activation > 0.5, "Square activation should be significant");
    assert!(result.converged || result.iterations_completed > 5, "Should converge or run sufficient iterations");

    // Test 2: Recognize circle pattern
    let mut circle_pattern = ActivationPattern::new("circle_recognition".to_string());
    circle_pattern.activations.insert(entities["feature_curve"], 0.95);
    circle_pattern.activations.insert(entities["feature_color_blue"], 0.6);

    let result = engine.propagate_activation(&circle_pattern).await.unwrap();
    
    let square_activation = result.final_activations.get(&entities["shape_square"]).copied().unwrap_or(0.0);
    let circle_activation = result.final_activations.get(&entities["shape_circle"]).copied().unwrap_or(0.0);
    
    assert!(circle_activation > square_activation, "Circle should have higher activation than square");
    assert!(circle_activation > 0.5, "Circle activation should be significant");

    // Test 3: Ambiguous pattern (both features present)
    let mut ambiguous_pattern = ActivationPattern::new("ambiguous_recognition".to_string());
    ambiguous_pattern.activations.insert(entities["feature_edge"], 0.7);
    ambiguous_pattern.activations.insert(entities["feature_corner"], 0.7);
    ambiguous_pattern.activations.insert(entities["feature_curve"], 0.6);

    let result = engine.propagate_activation(&ambiguous_pattern).await.unwrap();
    
    // Inhibitory connections should create competition
    let square_activation = result.final_activations.get(&entities["shape_square"]).copied().unwrap_or(0.0);
    let circle_activation = result.final_activations.get(&entities["shape_circle"]).copied().unwrap_or(0.0);
    
    // One should dominate due to inhibition
    assert!((square_activation - circle_activation).abs() > 0.1, "Inhibition should create clear winner");
}

#[tokio::test]
async fn test_hierarchical_concept_activation() {
    let config = create_test_config();
    let engine = ActivationPropagationEngine::new(config);

    // Create hierarchical concept network
    let entities = create_entities(&engine, vec![
        // Base concepts
        ("animal", EntityDirection::Input),
        ("mammal", EntityDirection::Hidden),
        ("bird", EntityDirection::Hidden),
        ("dog", EntityDirection::Hidden),
        ("cat", EntityDirection::Hidden),
        ("eagle", EntityDirection::Hidden),
        
        // Properties
        ("has_fur", EntityDirection::Hidden),
        ("can_fly", EntityDirection::Hidden),
        ("barks", EntityDirection::Hidden),
        ("meows", EntityDirection::Hidden),
        
        // Specific instances
        ("golden_retriever", EntityDirection::Output),
        ("persian_cat", EntityDirection::Output),
        ("bald_eagle", EntityDirection::Output),
    ]).await;

    // Create hierarchical relationships
    let relationships = vec![
        // ISA hierarchy
        (entities["animal"], entities["mammal"], 0.9, false),
        (entities["animal"], entities["bird"], 0.9, false),
        (entities["mammal"], entities["dog"], 0.8, false),
        (entities["mammal"], entities["cat"], 0.8, false),
        (entities["bird"], entities["eagle"], 0.8, false),
        
        // Property relationships
        (entities["mammal"], entities["has_fur"], 0.9, false),
        (entities["bird"], entities["can_fly"], 0.9, false),
        (entities["dog"], entities["barks"], 0.95, false),
        (entities["cat"], entities["meows"], 0.95, false),
        
        // Instance relationships
        (entities["dog"], entities["golden_retriever"], 0.9, false),
        (entities["cat"], entities["persian_cat"], 0.9, false),
        (entities["eagle"], entities["bald_eagle"], 0.9, false),
        
        // Cross-category inhibition
        (entities["mammal"], entities["bird"], 0.7, true),
        (entities["bird"], entities["mammal"], 0.7, true),
    ];

    for (source, target, weight, inhibitory) in relationships {
        let mut rel = BrainInspiredRelationship::new(source, target, RelationType::IsA);
        rel.weight = weight;
        rel.is_inhibitory = inhibitory;
        engine.add_relationship(rel).await.unwrap();
    }

    // Test activation spreading through hierarchy
    let mut animal_pattern = ActivationPattern::new("animal_activation".to_string());
    animal_pattern.activations.insert(entities["animal"], 1.0);

    let result = engine.propagate_activation(&animal_pattern).await.unwrap();

    // Verify hierarchical activation spread
    let mammal_activation = result.final_activations.get(&entities["mammal"]).copied().unwrap_or(0.0);
    let bird_activation = result.final_activations.get(&entities["bird"]).copied().unwrap_or(0.0);
    let dog_activation = result.final_activations.get(&entities["dog"]).copied().unwrap_or(0.0);
    let has_fur_activation = result.final_activations.get(&entities["has_fur"]).copied().unwrap_or(0.0);

    assert!(mammal_activation > 0.0, "Mammal should be activated");
    assert!(bird_activation > 0.0, "Bird should be activated");
    assert!(dog_activation > 0.0, "Dog should be activated through hierarchy");
    assert!(has_fur_activation > 0.0, "Properties should be activated");
    assert!(result.total_energy > 0.0, "Total energy should be positive");
}

#[tokio::test]
async fn test_working_memory_simulation() {
    let mut config = create_test_config();
    config.decay_rate = 0.2; // Higher decay for working memory
    config.max_iterations = 100;
    let engine = ActivationPropagationEngine::new(config);

    // Create working memory network with rehearsal loop
    let entities = create_entities(&engine, vec![
        // Sensory input
        ("visual_input", EntityDirection::Input),
        ("auditory_input", EntityDirection::Input),
        
        // Working memory buffers
        ("phonological_loop", EntityDirection::Hidden),
        ("visuospatial_sketchpad", EntityDirection::Hidden),
        ("central_executive", EntityDirection::Hidden),
        
        // Rehearsal mechanisms
        ("rehearsal_gate", EntityDirection::Gate),
        
        // Output/retrieval
        ("memory_output", EntityDirection::Output),
    ]).await;

    // Create rehearsal gate (maintains activation through feedback)
    let rehearsal_gate = LogicGate {
        gate_id: EntityKey::default(),
        gate_type: LogicGateType::Threshold,
        input_nodes: vec![entities["phonological_loop"], entities["central_executive"]],
        output_nodes: vec![entities["phonological_loop"]], // Feedback loop
        threshold: 0.3,
        weight_matrix: vec![0.7, 0.3],
    };
    let rehearsal_gate_key = engine.add_logic_gate(rehearsal_gate).await.unwrap();

    // Create memory network relationships
    let relationships = vec![
        // Input to buffers
        (entities["visual_input"], entities["visuospatial_sketchpad"], 0.9, false),
        (entities["auditory_input"], entities["phonological_loop"], 0.9, false),
        
        // Buffers to executive
        (entities["phonological_loop"], entities["central_executive"], 0.7, false),
        (entities["visuospatial_sketchpad"], entities["central_executive"], 0.7, false),
        
        // Executive control
        (entities["central_executive"], entities["memory_output"], 0.8, false),
        
        // Rehearsal loop connections
        (entities["phonological_loop"], rehearsal_gate_key, 0.6, false),
        (entities["central_executive"], rehearsal_gate_key, 0.4, false),
        
        // Competition between buffers
        (entities["phonological_loop"], entities["visuospatial_sketchpad"], 0.3, true),
        (entities["visuospatial_sketchpad"], entities["phonological_loop"], 0.3, true),
    ];

    for (source, target, weight, inhibitory) in relationships {
        let mut rel = BrainInspiredRelationship::new(source, target, RelationType::RelatedTo);
        rel.weight = weight;
        rel.is_inhibitory = inhibitory;
        engine.add_relationship(rel).await.unwrap();
    }

    // Test 1: Initial memory encoding
    let mut encoding_pattern = ActivationPattern::new("memory_encoding".to_string());
    encoding_pattern.activations.insert(entities["auditory_input"], 0.8);

    let result = engine.propagate_activation(&encoding_pattern).await.unwrap();
    
    let loop_activation = result.final_activations.get(&entities["phonological_loop"]).copied().unwrap_or(0.0);
    let output_activation = result.final_activations.get(&entities["memory_output"]).copied().unwrap_or(0.0);
    
    assert!(loop_activation > 0.5, "Phonological loop should be strongly activated");
    assert!(output_activation > 0.3, "Memory output should show activation");

    // Test 2: Maintenance without input (testing rehearsal)
    let empty_pattern = ActivationPattern::new("memory_maintenance".to_string());
    let result = engine.propagate_activation(&empty_pattern).await.unwrap();
    
    // Due to rehearsal gate, some activation should be maintained
    let maintained_activation = result.final_activations.get(&entities["phonological_loop"]).copied().unwrap_or(0.0);
    assert!(maintained_activation > 0.0, "Rehearsal should maintain some activation");
}

#[tokio::test]
async fn test_attention_network_simulation() {
    let config = create_test_config();
    let engine = ActivationPropagationEngine::new(config);

    // Create attention network based on Posner's model
    let entities = create_entities(&engine, vec![
        // Stimuli
        ("stimulus_left", EntityDirection::Input),
        ("stimulus_right", EntityDirection::Input),
        ("stimulus_center", EntityDirection::Input),
        
        // Attention networks
        ("alerting_network", EntityDirection::Hidden),
        ("orienting_network", EntityDirection::Hidden),
        ("executive_network", EntityDirection::Hidden),
        
        // Spatial attention
        ("attention_left", EntityDirection::Hidden),
        ("attention_right", EntityDirection::Hidden),
        ("attention_center", EntityDirection::Hidden),
        
        // Response selection
        ("response_left", EntityDirection::Output),
        ("response_right", EntityDirection::Output),
        ("response_inhibit", EntityDirection::Output),
    ]).await;

    // Create attention gates
    let conflict_detection_gate = LogicGate {
        gate_id: EntityKey::default(),
        gate_type: LogicGateType::Xor,
        input_nodes: vec![entities["stimulus_left"], entities["stimulus_right"]],
        output_nodes: vec![entities["executive_network"]],
        threshold: 0.5,
        weight_matrix: vec![1.0, 1.0],
    };
    engine.add_logic_gate(conflict_detection_gate).await.unwrap();

    // Create attention network relationships
    let relationships = vec![
        // Stimuli to attention networks
        (entities["stimulus_left"], entities["orienting_network"], 0.8, false),
        (entities["stimulus_right"], entities["orienting_network"], 0.8, false),
        (entities["stimulus_center"], entities["alerting_network"], 0.9, false),
        
        // Orienting to spatial attention
        (entities["orienting_network"], entities["attention_left"], 0.7, false),
        (entities["orienting_network"], entities["attention_right"], 0.7, false),
        
        // Spatial attention to responses
        (entities["attention_left"], entities["response_left"], 0.9, false),
        (entities["attention_right"], entities["response_right"], 0.9, false),
        
        // Executive control inhibition
        (entities["executive_network"], entities["response_inhibit"], 0.8, false),
        (entities["executive_network"], entities["response_left"], 0.4, true),
        (entities["executive_network"], entities["response_right"], 0.4, true),
        
        // Mutual inhibition between responses
        (entities["response_left"], entities["response_right"], 0.6, true),
        (entities["response_right"], entities["response_left"], 0.6, true),
    ];

    for (source, target, weight, inhibitory) in relationships {
        let mut rel = BrainInspiredRelationship::new(source, target, RelationType::RelatedTo);
        rel.weight = weight;
        rel.is_inhibitory = inhibitory;
        engine.add_relationship(rel).await.unwrap();
    }

    // Test spatial attention with conflict
    let mut conflict_pattern = ActivationPattern::new("attention_conflict".to_string());
    conflict_pattern.activations.insert(entities["stimulus_left"], 0.7);
    conflict_pattern.activations.insert(entities["stimulus_right"], 0.7);

    let result = engine.propagate_activation(&conflict_pattern).await.unwrap();
    
    let exec_activation = result.final_activations.get(&entities["executive_network"]).copied().unwrap_or(0.0);
    let inhibit_activation = result.final_activations.get(&entities["response_inhibit"]).copied().unwrap_or(0.0);
    
    assert!(exec_activation > 0.5, "Executive network should detect conflict");
    assert!(inhibit_activation > 0.3, "Conflict should trigger inhibition");
}

#[tokio::test]
async fn test_semantic_network_reasoning() {
    let config = create_test_config();
    let engine = ActivationPropagationEngine::new(config);

    // Create semantic network for reasoning
    let entities = create_entities(&engine, vec![
        // Concepts
        ("bird", EntityDirection::Input),
        ("penguin", EntityDirection::Hidden),
        ("robin", EntityDirection::Hidden),
        ("can_fly", EntityDirection::Hidden),
        ("has_wings", EntityDirection::Hidden),
        ("lives_antarctica", EntityDirection::Hidden),
        
        // Reasoning nodes
        ("typical_bird", EntityDirection::Hidden),
        ("atypical_bird", EntityDirection::Hidden),
        
        // Conclusions
        ("flies", EntityDirection::Output),
        ("does_not_fly", EntityDirection::Output),
    ]).await;

    // Create reasoning gates
    let typical_bird_gate = LogicGate {
        gate_id: EntityKey::default(),
        gate_type: LogicGateType::And,
        input_nodes: vec![entities["bird"], entities["has_wings"]],
        output_nodes: vec![entities["typical_bird"]],
        threshold: 0.6,
        weight_matrix: vec![1.0, 0.8],
    };
    engine.add_logic_gate(typical_bird_gate).await.unwrap();

    // Create semantic relationships
    let relationships = vec![
        // ISA relationships
        (entities["penguin"], entities["bird"], 0.9, false),
        (entities["robin"], entities["bird"], 0.9, false),
        
        // Property relationships
        (entities["bird"], entities["has_wings"], 0.95, false),
        (entities["robin"], entities["can_fly"], 0.95, false),
        (entities["penguin"], entities["lives_antarctica"], 0.9, false),
        
        // Reasoning paths
        (entities["typical_bird"], entities["flies"], 0.8, false),
        (entities["penguin"], entities["atypical_bird"], 0.8, false),
        (entities["atypical_bird"], entities["does_not_fly"], 0.9, false),
        
        // Exception handling
        (entities["penguin"], entities["can_fly"], 0.9, true), // Penguin inhibits flying
        
        // Mutual exclusion
        (entities["flies"], entities["does_not_fly"], 0.9, true),
        (entities["does_not_fly"], entities["flies"], 0.9, true),
    ];

    for (source, target, weight, inhibitory) in relationships {
        let mut rel = BrainInspiredRelationship::new(
            source, 
            target, 
            if inhibitory { RelationType::Opposite } else { RelationType::RelatedTo }
        );
        rel.weight = weight;
        rel.is_inhibitory = inhibitory;
        engine.add_relationship(rel).await.unwrap();
    }

    // Test 1: Robin reasoning (typical bird)
    let mut robin_pattern = ActivationPattern::new("robin_reasoning".to_string());
    robin_pattern.activations.insert(entities["robin"], 0.9);

    let result = engine.propagate_activation(&robin_pattern).await.unwrap();
    
    let flies = result.final_activations.get(&entities["flies"]).copied().unwrap_or(0.0);
    let not_flies = result.final_activations.get(&entities["does_not_fly"]).copied().unwrap_or(0.0);
    
    assert!(flies > not_flies, "Robin should be concluded to fly");
    assert!(flies > 0.5, "Flying conclusion should be strong");

    // Test 2: Penguin reasoning (exception handling)
    let mut penguin_pattern = ActivationPattern::new("penguin_reasoning".to_string());
    penguin_pattern.activations.insert(entities["penguin"], 0.9);

    let result = engine.propagate_activation(&penguin_pattern).await.unwrap();
    
    let flies = result.final_activations.get(&entities["flies"]).copied().unwrap_or(0.0);
    let not_flies = result.final_activations.get(&entities["does_not_fly"]).copied().unwrap_or(0.0);
    
    assert!(not_flies > flies, "Penguin should be concluded to not fly");
    assert!(not_flies > 0.5, "Not flying conclusion should be strong");
}

#[tokio::test]
async fn test_multi_modal_integration() {
    let config = create_test_config();
    let engine = ActivationPropagationEngine::new(config);

    // Create multi-modal integration network
    let entities = create_entities(&engine, vec![
        // Sensory inputs
        ("visual_red", EntityDirection::Input),
        ("visual_round", EntityDirection::Input),
        ("tactile_smooth", EntityDirection::Input),
        ("olfactory_sweet", EntityDirection::Input),
        
        // Feature integration
        ("color_processor", EntityDirection::Hidden),
        ("shape_processor", EntityDirection::Hidden),
        ("texture_processor", EntityDirection::Hidden),
        ("smell_processor", EntityDirection::Hidden),
        
        // Object representation
        ("object_apple", EntityDirection::Hidden),
        ("object_ball", EntityDirection::Hidden),
        
        // Recognition output
        ("recognized_apple", EntityDirection::Output),
        ("recognized_ball", EntityDirection::Output),
    ]).await;

    // Create integration gates
    let apple_integration = LogicGate {
        gate_id: EntityKey::default(),
        gate_type: LogicGateType::Weighted,
        input_nodes: vec![
            entities["color_processor"], 
            entities["shape_processor"],
            entities["smell_processor"]
        ],
        output_nodes: vec![entities["object_apple"]],
        threshold: 0.6,
        weight_matrix: vec![0.3, 0.3, 0.4], // Smell is distinctive for apple
    };
    engine.add_logic_gate(apple_integration).await.unwrap();

    let ball_integration = LogicGate {
        gate_id: EntityKey::default(),
        gate_type: LogicGateType::Weighted,
        input_nodes: vec![
            entities["color_processor"], 
            entities["shape_processor"],
            entities["texture_processor"]
        ],
        output_nodes: vec![entities["object_ball"]],
        threshold: 0.6,
        weight_matrix: vec![0.3, 0.4, 0.3],
    };
    engine.add_logic_gate(ball_integration).await.unwrap();

    // Create processing pathways
    let relationships = vec![
        // Sensory to processors
        (entities["visual_red"], entities["color_processor"], 0.9, false),
        (entities["visual_round"], entities["shape_processor"], 0.9, false),
        (entities["tactile_smooth"], entities["texture_processor"], 0.9, false),
        (entities["olfactory_sweet"], entities["smell_processor"], 0.9, false),
        
        // Object to recognition
        (entities["object_apple"], entities["recognized_apple"], 0.9, false),
        (entities["object_ball"], entities["recognized_ball"], 0.9, false),
        
        // Competition between objects
        (entities["recognized_apple"], entities["recognized_ball"], 0.5, true),
        (entities["recognized_ball"], entities["recognized_apple"], 0.5, true),
    ];

    for (source, target, weight, inhibitory) in relationships {
        let mut rel = BrainInspiredRelationship::new(source, target, RelationType::RelatedTo);
        rel.weight = weight;
        rel.is_inhibitory = inhibitory;
        engine.add_relationship(rel).await.unwrap();
    }

    // Test multi-modal apple recognition
    let mut apple_pattern = ActivationPattern::new("multimodal_apple".to_string());
    apple_pattern.activations.insert(entities["visual_red"], 0.9);
    apple_pattern.activations.insert(entities["visual_round"], 0.8);
    apple_pattern.activations.insert(entities["olfactory_sweet"], 0.9);

    let result = engine.propagate_activation(&apple_pattern).await.unwrap();
    
    let apple_recognition = result.final_activations.get(&entities["recognized_apple"]).copied().unwrap_or(0.0);
    let ball_recognition = result.final_activations.get(&entities["recognized_ball"]).copied().unwrap_or(0.0);
    
    assert!(apple_recognition > ball_recognition, "Apple should be recognized over ball");
    assert!(apple_recognition > 0.6, "Apple recognition should be strong with multi-modal input");
}

#[tokio::test]
async fn test_learning_and_adaptation() {
    let mut config = create_test_config();
    config.max_iterations = 30;
    let engine = ActivationPropagationEngine::new(config);

    // Create adaptive learning network
    let entities = create_entities(&engine, vec![
        // Stimuli
        ("stimulus_a", EntityDirection::Input),
        ("stimulus_b", EntityDirection::Input),
        
        // Hidden associations
        ("association_1", EntityDirection::Hidden),
        ("association_2", EntityDirection::Hidden),
        
        // Responses
        ("response_x", EntityDirection::Output),
        ("response_y", EntityDirection::Output),
    ]).await;

    // Create adaptive relationships that can strengthen over time
    let relationships = vec![
        // Initial weak associations
        (entities["stimulus_a"], entities["association_1"], 0.3, false),
        (entities["stimulus_b"], entities["association_2"], 0.3, false),
        
        // Cross associations (for flexibility)
        (entities["stimulus_a"], entities["association_2"], 0.1, false),
        (entities["stimulus_b"], entities["association_1"], 0.1, false),
        
        // Association to response
        (entities["association_1"], entities["response_x"], 0.5, false),
        (entities["association_2"], entities["response_y"], 0.5, false),
        
        // Lateral inhibition
        (entities["association_1"], entities["association_2"], 0.4, true),
        (entities["association_2"], entities["association_1"], 0.4, true),
    ];

    for (source, target, weight, inhibitory) in relationships {
        let mut rel = BrainInspiredRelationship::new(
            source, 
            target, 
            RelationType::Learned
        );
        rel.weight = weight;
        rel.is_inhibitory = inhibitory;
        engine.add_relationship(rel).await.unwrap();
    }

    // Run multiple learning trials
    let mut final_result = PropagationResult {
        final_activations: HashMap::new(),
        iterations_completed: 0,
        converged: false,
        activation_trace: Vec::new(),
        total_energy: 0.0,
    };

    for trial in 0..3 {
        let mut pattern = ActivationPattern::new(format!("learning_trial_{}", trial));
        pattern.activations.insert(entities["stimulus_a"], 0.8);
        pattern.activations.insert(entities["stimulus_b"], 0.2);

        final_result = engine.propagate_activation(&pattern).await.unwrap();
        
        // Each trial should show progressive strengthening
        let response_x = final_result.final_activations.get(&entities["response_x"]).copied().unwrap_or(0.0);
        assert!(response_x > 0.0, "Response X should show activation in trial {}", trial);
    }

    // Verify learning effect
    let response_x_final = final_result.final_activations.get(&entities["response_x"]).copied().unwrap_or(0.0);
    let response_y_final = final_result.final_activations.get(&entities["response_y"]).copied().unwrap_or(0.0);
    
    assert!(response_x_final > response_y_final, "Repeated pairing should strengthen A->X association");
}

#[tokio::test]
async fn test_complex_decision_making() {
    let config = create_test_config();
    let engine = ActivationPropagationEngine::new(config);

    // Create decision-making network
    let entities = create_entities(&engine, vec![
        // Context inputs
        ("context_urgent", EntityDirection::Input),
        ("context_safe", EntityDirection::Input),
        
        // Options
        ("option_risky_high_reward", EntityDirection::Hidden),
        ("option_safe_low_reward", EntityDirection::Hidden),
        
        // Evaluation criteria
        ("criterion_speed", EntityDirection::Hidden),
        ("criterion_safety", EntityDirection::Hidden),
        ("criterion_reward", EntityDirection::Hidden),
        
        // Decision outputs
        ("decision_go_risky", EntityDirection::Output),
        ("decision_go_safe", EntityDirection::Output),
    ]).await;

    // Create evaluation gates
    let urgency_gate = LogicGate {
        gate_id: EntityKey::default(),
        gate_type: LogicGateType::And,
        input_nodes: vec![entities["context_urgent"], entities["criterion_speed"]],
        output_nodes: vec![entities["option_risky_high_reward"]],
        threshold: 0.6,
        weight_matrix: vec![0.7, 0.3],
    };
    engine.add_logic_gate(urgency_gate).await.unwrap();

    let safety_gate = LogicGate {
        gate_id: EntityKey::default(),
        gate_type: LogicGateType::And,
        input_nodes: vec![entities["context_safe"], entities["criterion_safety"]],
        output_nodes: vec![entities["option_safe_low_reward"]],
        threshold: 0.5,
        weight_matrix: vec![0.6, 0.4],
    };
    engine.add_logic_gate(safety_gate).await.unwrap();

    // Create decision network
    let relationships = vec![
        // Context influences criteria
        (entities["context_urgent"], entities["criterion_speed"], 0.9, false),
        (entities["context_safe"], entities["criterion_safety"], 0.9, false),
        
        // Options to decisions
        (entities["option_risky_high_reward"], entities["decision_go_risky"], 0.8, false),
        (entities["option_safe_low_reward"], entities["decision_go_safe"], 0.8, false),
        
        // Mutual inhibition between decisions
        (entities["decision_go_risky"], entities["decision_go_safe"], 0.7, true),
        (entities["decision_go_safe"], entities["decision_go_risky"], 0.7, true),
        
        // Cross-inhibition between contexts
        (entities["context_urgent"], entities["criterion_safety"], 0.4, true),
        (entities["context_safe"], entities["criterion_speed"], 0.4, true),
    ];

    for (source, target, weight, inhibitory) in relationships {
        let mut rel = BrainInspiredRelationship::new(source, target, RelationType::RelatedTo);
        rel.weight = weight;
        rel.is_inhibitory = inhibitory;
        engine.add_relationship(rel).await.unwrap();
    }

    // Test urgent context decision
    let mut urgent_pattern = ActivationPattern::new("urgent_decision".to_string());
    urgent_pattern.activations.insert(entities["context_urgent"], 0.9);
    urgent_pattern.activations.insert(entities["criterion_speed"], 0.8);
    urgent_pattern.activations.insert(entities["criterion_reward"], 0.7);

    let result = engine.propagate_activation(&urgent_pattern).await.unwrap();
    
    let risky_decision = result.final_activations.get(&entities["decision_go_risky"]).copied().unwrap_or(0.0);
    let safe_decision = result.final_activations.get(&entities["decision_go_safe"]).copied().unwrap_or(0.0);
    
    assert!(risky_decision > safe_decision, "Urgent context should favor risky decision");
    assert!(risky_decision > 0.5, "Risky decision should have significant activation");
}

#[tokio::test]
async fn test_activation_statistics_accuracy() {
    let config = create_test_config();
    let engine = ActivationPropagationEngine::new(config);

    // Create a known network for statistics validation
    let num_entities = 10;
    let num_gates = 3;
    let num_inhibitory = 2;
    
    let mut entity_keys = Vec::new();
    for i in 0..num_entities {
        let entity = BrainInspiredEntity::new(
            format!("entity_{}", i),
            if i < 2 { EntityDirection::Input } else { EntityDirection::Output }
        );
        let key = engine.add_entity(entity).await.unwrap();
        entity_keys.push(key);
    }

    // Add gates
    for i in 0..num_gates {
        let gate = LogicGate {
            gate_id: EntityKey::default(),
            gate_type: LogicGateType::And,
            input_nodes: vec![entity_keys[i], entity_keys[i + 1]],
            output_nodes: vec![entity_keys[i + 2]],
            threshold: 0.5,
            weight_matrix: vec![1.0, 1.0],
        };
        engine.add_logic_gate(gate).await.unwrap();
    }

    // Add relationships with known inhibitory count
    for i in 0..5 {
        let mut rel = BrainInspiredRelationship::new(
            entity_keys[i],
            entity_keys[i + 1],
            RelationType::RelatedTo
        );
        if i < num_inhibitory {
            rel.is_inhibitory = true;
        }
        engine.add_relationship(rel).await.unwrap();
    }

    // Activate some entities
    let mut pattern = ActivationPattern::new("statistics_test".to_string());
    pattern.activations.insert(entity_keys[0], 0.8);
    pattern.activations.insert(entity_keys[1], 0.6);
    
    engine.propagate_activation(&pattern).await.unwrap();

    // Get and verify statistics
    let stats = engine.get_activation_statistics().await.unwrap();
    
    assert_eq!(stats.total_entities, num_entities, "Entity count mismatch");
    assert_eq!(stats.total_gates, num_gates, "Gate count mismatch");
    assert_eq!(stats.total_relationships, 5, "Relationship count mismatch");
    assert_eq!(stats.inhibitory_connections, num_inhibitory, "Inhibitory count mismatch");
    assert!(stats.average_activation >= 0.0 && stats.average_activation <= 1.0, "Average activation out of range");
    assert!(stats.active_entities <= stats.total_entities, "Active entities exceeds total");
}

#[tokio::test]
async fn test_state_persistence_and_reset() {
    let config = create_test_config();
    let engine = ActivationPropagationEngine::new(config);

    // Create simple network
    let entities = create_entities(&engine, vec![
        ("persistent_1", EntityDirection::Input),
        ("persistent_2", EntityDirection::Output),
    ]).await;

    let rel = BrainInspiredRelationship::new(
        entities["persistent_1"],
        entities["persistent_2"],
        RelationType::RelatedTo
    );
    engine.add_relationship(rel).await.unwrap();

    // Activate network
    let mut pattern = ActivationPattern::new("persistence_test".to_string());
    pattern.activations.insert(entities["persistent_1"], 0.9);
    
    let result = engine.propagate_activation(&pattern).await.unwrap();
    assert!(!result.final_activations.is_empty(), "Should have activations");

    // Get current state
    let state_before = engine.get_current_state().await.unwrap();
    assert!(state_before.values().any(|&v| v > 0.0), "Should have active entities");

    // Reset activations
    engine.reset_activations().await.unwrap();

    // Verify reset
    let state_after = engine.get_current_state().await.unwrap();
    assert!(state_after.values().all(|&v| v == 0.0), "All activations should be zero after reset");
    assert_eq!(state_before.len(), state_after.len(), "State size should remain the same");
}

#[tokio::test]
async fn test_large_scale_network_performance() {
    let mut config = create_test_config();
    config.max_iterations = 20; // Limit iterations for performance
    let engine = ActivationPropagationEngine::new(config);

    // Create large network
    let num_layers = 5;
    let nodes_per_layer = 20;
    let mut all_keys = Vec::new();

    // Create layers
    for layer in 0..num_layers {
        for node in 0..nodes_per_layer {
            let direction = match layer {
                0 => EntityDirection::Input,
                4 => EntityDirection::Output,
                _ => EntityDirection::Hidden,
            };
            
            let entity = BrainInspiredEntity::new(
                format!("layer_{}_node_{}", layer, node),
                direction
            );
            let key = engine.add_entity(entity).await.unwrap();
            all_keys.push((layer, key));
        }
    }

    // Create feed-forward connections with some lateral inhibition
    for (layer, source_key) in &all_keys {
        if *layer < num_layers - 1 {
            // Connect to next layer
            for (target_layer, target_key) in &all_keys {
                if *target_layer == layer + 1 {
                    let mut rel = BrainInspiredRelationship::new(
                        *source_key,
                        *target_key,
                        RelationType::RelatedTo
                    );
                    rel.weight = 0.5;
                    engine.add_relationship(rel).await.unwrap();
                }
            }
        }
        
        // Add some lateral inhibition within layer
        if *layer > 0 && *layer < num_layers - 1 {
            for (target_layer, target_key) in &all_keys {
                if *target_layer == *layer && source_key != target_key {
                    // Create some pseudo-random lateral inhibition
                    let source_idx = all_keys.iter().position(|(_, k)| k == source_key).unwrap_or(0);
                    let target_idx = all_keys.iter().position(|(_, k)| k == target_key).unwrap_or(0);
                    if (source_idx + target_idx) % 7 == 0 {
                        let mut rel = BrainInspiredRelationship::new(
                            *source_key,
                            *target_key,
                            RelationType::RelatedTo
                        );
                        rel.weight = 0.3;
                        rel.is_inhibitory = true;
                        engine.add_relationship(rel).await.unwrap();
                    }
                }
            }
        }
    }

    // Activate input layer
    let mut pattern = ActivationPattern::new("large_scale_test".to_string());
    for (layer, key) in &all_keys {
        if *layer == 0 {
            pattern.activations.insert(*key, 0.7);
        }
    }

    // Test propagation completes in reasonable time
    let start = std::time::Instant::now();
    let result = engine.propagate_activation(&pattern).await.unwrap();
    let duration = start.elapsed();

    assert!(duration.as_secs() < 5, "Large network should process in under 5 seconds");
    assert!(result.iterations_completed > 0, "Should complete some iterations");
    assert!(result.total_energy > 0.0, "Should have energy in the system");
    
    // Verify output layer activation
    let output_activations: Vec<_> = all_keys.iter()
        .filter(|(layer, _)| *layer == 4)
        .filter_map(|(_, key)| result.final_activations.get(key))
        .collect();
    
    assert!(!output_activations.is_empty(), "Output layer should have activations");
    assert!(output_activations.iter().any(|&&a| a > 0.0), "At least some outputs should be active");
}

#[tokio::test]
async fn test_edge_cases_and_error_handling() {
    let config = create_test_config();
    let engine = ActivationPropagationEngine::new(config);

    // Test 1: Empty network propagation
    let empty_pattern = ActivationPattern::new("empty_test".to_string());
    let result = engine.propagate_activation(&empty_pattern).await.unwrap();
    assert!(result.final_activations.is_empty(), "Empty network should have no activations");
    assert_eq!(result.total_energy, 0.0, "Empty network should have zero energy");

    // Test 2: Single isolated entity
    let entity = BrainInspiredEntity::new("isolated".to_string(), EntityDirection::Hidden);
    let key = engine.add_entity(entity).await.unwrap();
    
    let mut isolated_pattern = ActivationPattern::new("isolated_test".to_string());
    isolated_pattern.activations.insert(key, 0.5);
    
    let result = engine.propagate_activation(&isolated_pattern).await.unwrap();
    assert!(result.final_activations.contains_key(&key), "Isolated entity should maintain activation");

    // Test 3: Maximum activation values
    let mut max_pattern = ActivationPattern::new("max_test".to_string());
    max_pattern.activations.insert(key, 1.0);
    
    let result = engine.propagate_activation(&max_pattern).await.unwrap();
    assert!(result.final_activations.values().all(|&v| v <= 1.0), "Activations should not exceed 1.0");

    // Test 4: Very small activation values
    let mut min_pattern = ActivationPattern::new("min_test".to_string());
    min_pattern.activations.insert(key, 0.00001);
    
    let result = engine.propagate_activation(&min_pattern).await.unwrap();
    assert!(result.final_activations.values().all(|&v| v >= 0.0), "Activations should not go negative");
}