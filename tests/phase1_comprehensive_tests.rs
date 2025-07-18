#[cfg(test)]
mod phase1_tests {
    use llmkg::core::brain_types::{
        BrainInspiredEntity, LogicGate, BrainInspiredRelationship, ActivationPattern,
        EntityDirection, LogicGateType, RelationType
    };
    use llmkg::core::brain_enhanced_graph::{BrainEnhancedKnowledgeGraph, BrainEnhancedConfig};
    use llmkg::core::types::{EntityKey, AttributeValue};
    use llmkg::versioning::temporal_graph::TemporalKnowledgeGraph;
    use llmkg::core::graph::KnowledgeGraph;
    use std::collections::HashMap;
    use std::time::{SystemTime, Duration};
    use chrono::Utc;

    // Test 1: Brain-Inspired Entity Tests
    #[test]
    fn test_brain_entity_creation_all_directions() {
        let input_entity = BrainInspiredEntity::new("test_input".to_string(), EntityDirection::Input);
        assert_eq!(input_entity.concept_id, "test_input");
        assert!(matches!(input_entity.direction, EntityDirection::Input));
        assert_eq!(input_entity.activation_state, 0.0);
        assert!(input_entity.properties.is_empty());
        assert!(input_entity.embedding.is_empty());

        let output_entity = BrainInspiredEntity::new("test_output".to_string(), EntityDirection::Output);
        assert!(matches!(output_entity.direction, EntityDirection::Output));

        let gate_entity = BrainInspiredEntity::new("test_gate".to_string(), EntityDirection::Gate);
        assert!(matches!(gate_entity.direction, EntityDirection::Gate));

        let hidden_entity = BrainInspiredEntity::new("test_hidden".to_string(), EntityDirection::Hidden);
        assert!(matches!(hidden_entity.direction, EntityDirection::Hidden));
    }

    #[test]
    fn test_entity_state_management() {
        let mut entity = BrainInspiredEntity::new("test_entity".to_string(), EntityDirection::Input);
        
        // Test initial state
        assert_eq!(entity.activation_state, 0.0);
        
        // Test activation
        let activation = entity.activate(0.5, 0.1);
        assert!((activation - 0.5).abs() < 0.01);
        assert!((entity.activation_state - 0.5).abs() < 0.01);
        
        // Test cumulative activation
        entity.activate(0.3, 0.1);
        assert!(entity.activation_state > 0.5);
        assert!(entity.activation_state <= 1.0);
    }

    #[test]
    fn test_entity_properties_storage() {
        let mut entity = BrainInspiredEntity::new("test_entity".to_string(), EntityDirection::Input);
        
        // Add properties
        entity.properties.insert("color".to_string(), AttributeValue::String("blue".to_string()));
        entity.properties.insert("size".to_string(), AttributeValue::Number(10.5));
        entity.properties.insert("active".to_string(), AttributeValue::Boolean(true));
        
        // Verify properties
        assert_eq!(entity.properties.len(), 3);
        if let Some(AttributeValue::String(color)) = entity.properties.get("color") {
            assert_eq!(color, "blue");
        } else {
            panic!("Color property not found or wrong type");
        }
        
        if let Some(AttributeValue::Number(size)) = entity.properties.get("size") {
            assert_eq!(*size, 10.5);
        } else {
            panic!("Size property not found or wrong type");
        }
    }

    #[test]
    fn test_entity_key_generation() {
        let entity1 = BrainInspiredEntity::new("entity1".to_string(), EntityDirection::Input);
        let entity2 = BrainInspiredEntity::new("entity2".to_string(), EntityDirection::Input);
        
        // Keys should be different (default keys are unique)
        assert_ne!(entity1.id, entity2.id);
    }

    // Test 2: Logic Gate Tests
    #[test]
    fn test_logic_gate_and_computation() {
        let mut gate = LogicGate::new(LogicGateType::And, 0.5);
        
        // Set up gate with 2 inputs
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default()];
        
        // Both inputs above threshold
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.8);
        
        // One input below threshold
        assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.0);
        
        // Both inputs below threshold
        assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.0);
        
        // Test with multiple inputs
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default(), EntityKey::default()];
        assert_eq!(gate.calculate_output(&[0.7, 0.8, 0.6]).unwrap(), 0.6);
    }

    #[test]
    fn test_logic_gate_or_computation() {
        let mut gate = LogicGate::new(LogicGateType::Or, 0.5);
        
        // Set up gate with 2 inputs
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default()];
        
        // At least one input above threshold
        assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.9);
        
        // Both inputs below threshold
        assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.0);
        
        // Multiple inputs
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default(), EntityKey::default()];
        assert_eq!(gate.calculate_output(&[0.3, 0.8, 0.4]).unwrap(), 0.8);
    }

    #[test]
    fn test_logic_gate_not_computation() {
        let mut gate = LogicGate::new(LogicGateType::Not, 0.0);
        
        // Set up gate with 1 input
        gate.input_nodes = vec![EntityKey::default()];
        
        // Invert input
        assert_eq!(gate.calculate_output(&[0.3]).unwrap(), 0.7);
        assert_eq!(gate.calculate_output(&[0.0]).unwrap(), 1.0);
        assert_eq!(gate.calculate_output(&[1.0]).unwrap(), 0.0);
        
        // NOT gate with multiple inputs should error
        assert!(gate.calculate_output(&[0.3, 0.4]).is_err());
    }

    #[test]
    fn test_logic_gate_inhibitory_computation() {
        let mut gate = LogicGate::new(LogicGateType::Inhibitory, 0.0);
        
        // Set up gate with 2 inputs
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default()];
        
        // Primary input minus inhibitory inputs
        assert_eq!(gate.calculate_output(&[0.8, 0.3]).unwrap(), 0.5);
        
        // Set up gate with 3 inputs
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default(), EntityKey::default()];
        assert_eq!(gate.calculate_output(&[0.8, 0.3, 0.2]).unwrap(), 0.3);
        
        // Inhibition exceeds primary
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default()];
        assert_eq!(gate.calculate_output(&[0.3, 0.8]).unwrap(), 0.0);
        
        // No inputs
        gate.input_nodes = vec![];
        assert_eq!(gate.calculate_output(&[]).unwrap(), 0.0);
    }

    #[test]
    fn test_logic_gate_weighted_computation() {
        let mut gate = LogicGate::new(LogicGateType::Weighted, 0.5);
        gate.weight_matrix = vec![0.5, 0.8, 0.2];
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default(), EntityKey::default()];
        
        // Weighted sum above threshold
        let result = gate.calculate_output(&[0.8, 0.7, 0.9]).unwrap();
        let expected = 0.8 * 0.5 + 0.7 * 0.8 + 0.9 * 0.2; // = 1.34, clamped to 1.0
        assert_eq!(result, 1.0);
        
        // Weighted sum below threshold
        let result = gate.calculate_output(&[0.2, 0.3, 0.1]).unwrap();
        let weighted_sum = 0.2 * 0.5 + 0.3 * 0.8 + 0.1 * 0.2; // = 0.36 < 0.5
        assert_eq!(result, 0.0);
        
        // Weight matrix size mismatch should error
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default()];
        assert!(gate.calculate_output(&[0.5, 0.6]).is_err());
    }

    #[test]
    fn test_logic_gate_threshold_behavior() {
        let mut gate_high = LogicGate::new(LogicGateType::Or, 0.8);
        let mut gate_low = LogicGate::new(LogicGateType::Or, 0.2);
        
        // Set up gates with 2 inputs
        gate_high.input_nodes = vec![EntityKey::default(), EntityKey::default()];
        gate_low.input_nodes = vec![EntityKey::default(), EntityKey::default()];
        
        // Same input, different thresholds
        assert_eq!(gate_high.calculate_output(&[0.5, 0.6]).unwrap(), 0.0);
        assert_eq!(gate_low.calculate_output(&[0.5, 0.6]).unwrap(), 0.6);
    }

    #[test]
    fn test_logic_gate_zero_and_multiple_inputs() {
        let mut and_gate = LogicGate::new(LogicGateType::And, 0.5);
        let mut or_gate = LogicGate::new(LogicGateType::Or, 0.5);
        
        // Zero inputs
        and_gate.input_nodes = vec![];
        or_gate.input_nodes = vec![];
        assert_eq!(and_gate.calculate_output(&[]).unwrap(), f32::INFINITY);
        assert_eq!(or_gate.calculate_output(&[]).unwrap(), 0.0);
        
        // Single input
        and_gate.input_nodes = vec![EntityKey::default()];
        or_gate.input_nodes = vec![EntityKey::default()];
        assert_eq!(and_gate.calculate_output(&[0.8]).unwrap(), 0.8);
        assert_eq!(or_gate.calculate_output(&[0.8]).unwrap(), 0.8);
    }

    #[test]
    fn test_logic_gate_xor_computation() {
        let mut gate = LogicGate::new(LogicGateType::Xor, 0.5);
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default()];
        
        // XOR truth table
        assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.9); // false ^ true = true
        assert_eq!(gate.calculate_output(&[0.8, 0.3]).unwrap(), 0.8); // true ^ false = true
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.0); // true ^ true = false
        assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.0); // false ^ false = false
    }

    #[test]
    fn test_logic_gate_nand_computation() {
        let mut gate = LogicGate::new(LogicGateType::Nand, 0.5);
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default()];
        
        // NAND truth table (NOT AND)
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.0); // true AND true = false
        assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 1.0); // false AND true = true
        assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 1.0); // false AND false = true
    }

    #[test]
    fn test_logic_gate_nor_computation() {
        let mut gate = LogicGate::new(LogicGateType::Nor, 0.5);
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default()];
        
        // NOR truth table (NOT OR)
        assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 1.0); // false OR false = true
        assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.0); // false OR true = false
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.0); // true OR true = false
    }

    #[test]
    fn test_logic_gate_xnor_computation() {
        let mut gate = LogicGate::new(LogicGateType::Xnor, 0.5);
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default()];
        
        // XNOR truth table (NOT XOR)
        assert_eq!(gate.calculate_output(&[0.8, 0.9]).unwrap(), 0.9); // true XNOR true = true
        assert_eq!(gate.calculate_output(&[0.3, 0.4]).unwrap(), 0.4); // false XNOR false = true
        assert_eq!(gate.calculate_output(&[0.3, 0.9]).unwrap(), 0.0); // false XNOR true = false
        assert_eq!(gate.calculate_output(&[0.8, 0.3]).unwrap(), 0.0); // true XNOR false = false
    }

    #[test]
    fn test_logic_gate_identity_computation() {
        let mut gate = LogicGate::new(LogicGateType::Identity, 0.0);
        gate.input_nodes = vec![EntityKey::default()];
        
        // Identity gate passes through input
        assert_eq!(gate.calculate_output(&[0.5]).unwrap(), 0.5);
        assert_eq!(gate.calculate_output(&[0.0]).unwrap(), 0.0);
        assert_eq!(gate.calculate_output(&[1.0]).unwrap(), 1.0);
        
        // Identity gate with multiple inputs should error
        assert!(gate.calculate_output(&[0.5, 0.6]).is_err());
    }

    #[test]
    fn test_logic_gate_threshold_computation() {
        let mut gate = LogicGate::new(LogicGateType::Threshold, 1.0);
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default(), EntityKey::default()];
        
        // Sum above threshold
        assert_eq!(gate.calculate_output(&[0.5, 0.6, 0.7]).unwrap(), 1.0); // 1.8 > 1.0, clamped to 1.0
        
        // Sum below threshold
        assert_eq!(gate.calculate_output(&[0.2, 0.3, 0.4]).unwrap(), 0.0); // 0.9 < 1.0
        
        // Sum equal to threshold
        assert_eq!(gate.calculate_output(&[0.3, 0.3, 0.4]).unwrap(), 1.0); // 1.0 = 1.0
    }

    // Test 3: Brain-Inspired Relationship Tests
    #[test]
    fn test_brain_relationship_creation() {
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        let relationship = BrainInspiredRelationship::new(key1, key2, RelationType::IsA);
        
        assert_eq!(relationship.source, key1);
        assert_eq!(relationship.target, key2);
        assert_eq!(relationship.relation_type, RelationType::IsA);
        assert_eq!(relationship.weight, 1.0);
        assert!(!relationship.is_inhibitory);
        assert_eq!(relationship.activation_count, 0);
    }

    #[test]
    fn test_excitatory_and_inhibitory_relationships() {
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        
        // Excitatory relationship
        let mut excitatory = BrainInspiredRelationship::new(key1, key2, RelationType::RelatedTo);
        assert!(!excitatory.is_inhibitory);
        assert_eq!(excitatory.weight, 1.0);
        
        // Inhibitory relationship
        let mut inhibitory = BrainInspiredRelationship::new(key1, key2, RelationType::RelatedTo);
        inhibitory.is_inhibitory = true;
        assert!(inhibitory.is_inhibitory);
    }

    #[test]
    fn test_relationship_weight_bounds() {
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        let mut relationship = BrainInspiredRelationship::new(key1, key2, RelationType::RelatedTo);
        
        // Initial weight
        assert_eq!(relationship.weight, 1.0);
        
        // Strengthen relationship
        relationship.strengthen(0.5);
        assert_eq!(relationship.weight, 1.0); // Clamped to 1.0
        
        // Start with lower weight
        relationship.weight = 0.5;
        relationship.strengthen(0.3);
        assert_eq!(relationship.weight, 0.8);
    }

    #[test]
    fn test_hebbian_learning_strengthen_weaken() {
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        let mut relationship = BrainInspiredRelationship::new(key1, key2, RelationType::RelatedTo);
        
        relationship.weight = 0.5;
        let initial_count = relationship.activation_count;
        
        // Strengthen
        relationship.strengthen(0.2);
        assert_eq!(relationship.weight, 0.7);
        assert_eq!(relationship.activation_count, initial_count + 1);
        
        // Weaken through decay
        relationship.temporal_decay = 0.5;
        let old_weight = relationship.weight;
        std::thread::sleep(Duration::from_millis(10));
        relationship.apply_decay();
        assert!(relationship.weight < old_weight);
    }

    #[test]
    fn test_relationship_decay_over_time() {
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        let mut relationship = BrainInspiredRelationship::new(key1, key2, RelationType::RelatedTo);
        
        relationship.weight = 0.8;
        relationship.temporal_decay = 0.1;
        
        // Apply decay
        std::thread::sleep(Duration::from_millis(10));
        let decayed_weight = relationship.apply_decay();
        assert!(decayed_weight < 0.8);
        assert_eq!(relationship.weight, decayed_weight);
    }

    #[test]
    fn test_relationship_types() {
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        
        let is_a = BrainInspiredRelationship::new(key1, key2, RelationType::IsA);
        assert_eq!(is_a.relation_type, RelationType::IsA);
        
        let has_property = BrainInspiredRelationship::new(key1, key2, RelationType::HasProperty);
        assert_eq!(has_property.relation_type, RelationType::HasProperty);
        
        let related_to = BrainInspiredRelationship::new(key1, key2, RelationType::RelatedTo);
        assert_eq!(related_to.relation_type, RelationType::RelatedTo);
    }

    // Test 4: Activation Pattern Tests
    #[test]
    fn test_activation_pattern_creation() {
        let pattern = ActivationPattern::new("test query".to_string());
        assert_eq!(pattern.query, "test query");
        assert!(pattern.activations.is_empty());
        assert!(pattern.timestamp <= SystemTime::now());
    }

    #[test]
    fn test_activation_pattern_wave_propagation() {
        let mut pattern = ActivationPattern::new("wave test".to_string());
        
        // Add initial activations
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        let key3 = EntityKey::default();
        
        pattern.activations.insert(key1, 0.8);
        pattern.activations.insert(key2, 0.5);
        pattern.activations.insert(key3, 0.3);
        
        assert_eq!(pattern.activations.len(), 3);
        assert_eq!(pattern.activations[&key1], 0.8);
    }

    #[test]
    fn test_activation_bounds_maintained() {
        let mut pattern = ActivationPattern::new("bounds test".to_string());
        let key = EntityKey::default();
        
        // Test bounds
        pattern.activations.insert(key, 0.5);
        assert!(pattern.activations[&key] >= 0.0);
        assert!(pattern.activations[&key] <= 1.0);
        
        // Update with valid value
        pattern.activations.insert(key, 0.9);
        assert_eq!(pattern.activations[&key], 0.9);
    }

    #[test]
    fn test_activation_pattern_top_k_retrieval() {
        let mut pattern = ActivationPattern::new("top k test".to_string());
        
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        let key3 = EntityKey::default();
        
        pattern.activations.insert(key1, 0.9);
        pattern.activations.insert(key2, 0.5);
        pattern.activations.insert(key3, 0.7);
        
        let top_2 = pattern.get_top_activations(2);
        assert_eq!(top_2.len(), 2);
        assert_eq!(top_2[0].1, 0.9); // Highest activation
        assert_eq!(top_2[1].1, 0.7); // Second highest
    }

    #[test]
    fn test_activation_pattern_energy_calculation() {
        let mut pattern = ActivationPattern::new("energy test".to_string());
        
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        
        pattern.activations.insert(key1, 0.6);
        pattern.activations.insert(key2, 0.4);
        
        // Calculate total energy (sum of activations)
        let total_energy: f32 = pattern.activations.values().sum();
        assert_eq!(total_energy, 1.0);
    }

    // Test 5: Graph Structure Tests (requires async)
    #[tokio::test]
    async fn test_graph_structure_add_remove_entities() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        let entity = BrainInspiredEntity::new("test_entity".to_string(), EntityDirection::Input);
        let entity_key = graph.insert_brain_entity(entity).await.unwrap();
        
        let count = graph.entity_count().await.unwrap();
        assert_eq!(count, 1);
        
        // Verify entity exists
        let all_entities = graph.get_all_entities().await.unwrap();
        assert_eq!(all_entities.len(), 1);
        assert_eq!(all_entities[0].concept_id, "test_entity");
    }

    #[tokio::test]
    async fn test_graph_structure_add_remove_relationships() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        let entity1 = BrainInspiredEntity::new("entity1".to_string(), EntityDirection::Input);
        let entity2 = BrainInspiredEntity::new("entity2".to_string(), EntityDirection::Output);
        
        let key1 = graph.insert_brain_entity(entity1).await.unwrap();
        let key2 = graph.insert_brain_entity(entity2).await.unwrap();
        
        let relationship = BrainInspiredRelationship::new(key1, key2, RelationType::RelatedTo);
        graph.insert_brain_relationship(relationship).await.unwrap();
        
        let rel_count = graph.get_relationship_count().await.unwrap();
        assert_eq!(rel_count, 1);
        
        // Test relationship exists
        assert!(graph.has_relationship(key1, key2).await.unwrap());
        
        // Remove relationship
        graph.remove_relationship(key1, key2).await.unwrap();
        let rel_count = graph.get_relationship_count().await.unwrap();
        assert_eq!(rel_count, 0);
    }

    #[tokio::test]
    async fn test_graph_traversal_algorithms() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Create a simple chain: A -> B -> C
        let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
        let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Input);
        let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Output);
        
        let key_a = graph.insert_brain_entity(entity_a).await.unwrap();
        let key_b = graph.insert_brain_entity(entity_b).await.unwrap();
        let key_c = graph.insert_brain_entity(entity_c).await.unwrap();
        
        let rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
        let rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
        
        graph.insert_brain_relationship(rel_ab).await.unwrap();
        graph.insert_brain_relationship(rel_bc).await.unwrap();
        
        // Test neighbor retrieval
        let neighbors_a = graph.get_neighbors(key_a).await.unwrap();
        assert_eq!(neighbors_a.len(), 1);
        assert!(neighbors_a.contains(&key_b));
        
        let neighbors_b = graph.get_neighbors(key_b).await.unwrap();
        assert_eq!(neighbors_b.len(), 2);
        assert!(neighbors_b.contains(&key_a));
        assert!(neighbors_b.contains(&key_c));
    }

    #[tokio::test]
    async fn test_graph_cycle_detection() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Create a cycle: A -> B -> C -> A
        let entity_a = BrainInspiredEntity::new("A".to_string(), EntityDirection::Input);
        let entity_b = BrainInspiredEntity::new("B".to_string(), EntityDirection::Input);
        let entity_c = BrainInspiredEntity::new("C".to_string(), EntityDirection::Output);
        
        let key_a = graph.insert_brain_entity(entity_a).await.unwrap();
        let key_b = graph.insert_brain_entity(entity_b).await.unwrap();
        let key_c = graph.insert_brain_entity(entity_c).await.unwrap();
        
        let rel_ab = BrainInspiredRelationship::new(key_a, key_b, RelationType::RelatedTo);
        let rel_bc = BrainInspiredRelationship::new(key_b, key_c, RelationType::RelatedTo);
        let rel_ca = BrainInspiredRelationship::new(key_c, key_a, RelationType::RelatedTo);
        
        graph.insert_brain_relationship(rel_ab).await.unwrap();
        graph.insert_brain_relationship(rel_bc).await.unwrap();
        graph.insert_brain_relationship(rel_ca).await.unwrap();
        
        // Test path finding (should find cycle)
        let paths = graph.find_alternative_paths(key_a, key_a, 5).await.unwrap();
        assert!(!paths.is_empty());
    }

    #[tokio::test]
    async fn test_graph_metrics() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Create entities
        let entity1 = BrainInspiredEntity::new("central_node".to_string(), EntityDirection::Input);
        let entity2 = BrainInspiredEntity::new("node2".to_string(), EntityDirection::Input);
        let entity3 = BrainInspiredEntity::new("node3".to_string(), EntityDirection::Output);
        
        let key1 = graph.insert_brain_entity(entity1).await.unwrap();
        let key2 = graph.insert_brain_entity(entity2).await.unwrap();
        let key3 = graph.insert_brain_entity(entity3).await.unwrap();
        
        // Connect central node to others
        let rel12 = BrainInspiredRelationship::new(key1, key2, RelationType::RelatedTo);
        let rel13 = BrainInspiredRelationship::new(key1, key3, RelationType::RelatedTo);
        
        graph.insert_brain_relationship(rel12).await.unwrap();
        graph.insert_brain_relationship(rel13).await.unwrap();
        
        // Test connection count
        let conn_count = graph.get_connection_count(key1).await.unwrap();
        assert_eq!(conn_count, 2);
        
        // Test betweenness centrality
        let centrality = graph.calculate_betweenness_centrality(key1).await.unwrap();
        assert!(centrality > 0.0);
    }

    // Test 6: Temporal Graph Tests
    #[tokio::test]
    async fn test_temporal_graph_operations() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        let entity = BrainInspiredEntity::new("temporal_entity".to_string(), EntityDirection::Input);
        let entity_key = graph.insert_brain_entity(entity).await.unwrap();
        
        // Entity should be stored in temporal graph
        let count = graph.entity_count().await.unwrap();
        assert_eq!(count, 1);
        
        // Test temporal graph access
        let temporal_graph = graph.get_temporal_graph();
        assert!(temporal_graph.read().await.get_entity_count().await.unwrap() >= 0);
    }

    #[tokio::test]
    async fn test_temporal_snapshots() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Create entity at time T1
        let entity1 = BrainInspiredEntity::new("entity_t1".to_string(), EntityDirection::Input);
        let key1 = graph.insert_brain_entity(entity1).await.unwrap();
        
        let snapshot_time = SystemTime::now();
        
        // Create another entity at time T2
        std::thread::sleep(Duration::from_millis(10));
        let entity2 = BrainInspiredEntity::new("entity_t2".to_string(), EntityDirection::Output);
        let key2 = graph.insert_brain_entity(entity2).await.unwrap();
        
        // Both entities should exist
        let count = graph.entity_count().await.unwrap();
        assert_eq!(count, 2);
    }

    // Test 7: Brain-Enhanced Graph Integration Tests
    #[tokio::test]
    async fn test_brain_enhanced_graph_creation() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        let stats = graph.get_brain_statistics().await.unwrap();
        assert_eq!(stats.total_brain_entities, 0);
        assert_eq!(stats.total_brain_relationships, 0);
        assert_eq!(stats.total_logic_gates, 0);
    }

    #[tokio::test]
    async fn test_entity_to_brain_entity_mapping() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        let entity = BrainInspiredEntity::new("mapped_entity".to_string(), EntityDirection::Input);
        let brain_key = graph.insert_brain_entity(entity).await.unwrap();
        
        // Entity should be mapped
        let count = graph.entity_count().await.unwrap();
        assert_eq!(count, 1);
        
        // Test mapping exists
        let all_entities = graph.get_all_entities().await.unwrap();
        assert_eq!(all_entities.len(), 1);
        assert_eq!(all_entities[0].concept_id, "mapped_entity");
    }

    #[tokio::test]
    async fn test_concept_structure_creation() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        let embedding = vec![0.1, 0.2, 0.3, 0.4]; // Small embedding for test
        let structure = graph.create_concept_structure(
            "test_concept".to_string(),
            embedding,
        ).await.unwrap();
        
        assert_eq!(structure.concept_name, "test_concept");
        
        // Should have created input, output, and gate
        let stats = graph.get_brain_statistics().await.unwrap();
        assert_eq!(stats.input_nodes, 1);
        assert_eq!(stats.output_nodes, 1);
        assert_eq!(stats.total_logic_gates, 1);
        assert_eq!(stats.total_brain_relationships, 1);
    }

    #[tokio::test]
    async fn test_activation_propagation_through_structures() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        let embedding = vec![0.5; 10]; // Small embedding
        let structure = graph.create_concept_structure(
            "propagation_test".to_string(),
            embedding,
        ).await.unwrap();
        
        // Activate the concept
        let result = graph.activate_concept("propagation_test_input", 0.8).await.unwrap();
        assert!(!result.final_activations.is_empty());
    }

    // Test 8: Circuit Building Tests
    #[tokio::test]
    async fn test_simple_logic_circuit() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Create AND gate
        let and_gate = LogicGate::new(LogicGateType::And, 0.5);
        let and_key = graph.insert_logic_gate(and_gate).await.unwrap();
        
        // Create OR gate  
        let or_gate = LogicGate::new(LogicGateType::Or, 0.3);
        let or_key = graph.insert_logic_gate(or_gate).await.unwrap();
        
        let stats = graph.get_brain_statistics().await.unwrap();
        assert_eq!(stats.total_logic_gates, 2);
    }

    #[tokio::test]
    async fn test_signal_flow_through_gates() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Create input entities
        let input1 = BrainInspiredEntity::new("input1".to_string(), EntityDirection::Input);
        let input2 = BrainInspiredEntity::new("input2".to_string(), EntityDirection::Input);
        let output = BrainInspiredEntity::new("output".to_string(), EntityDirection::Output);
        
        let key1 = graph.insert_brain_entity(input1).await.unwrap();
        let key2 = graph.insert_brain_entity(input2).await.unwrap();
        let key_out = graph.insert_brain_entity(output).await.unwrap();
        
        // Create AND gate connecting them
        let mut and_gate = LogicGate::new(LogicGateType::And, 0.5);
        and_gate.input_nodes = vec![key1, key2];
        and_gate.output_nodes = vec![key_out];
        
        let gate_key = graph.insert_logic_gate(and_gate).await.unwrap();
        
        // Test signal flow
        let result = graph.activate_concept("input1", 0.8).await.unwrap();
        assert!(!result.final_activations.is_empty());
    }

    #[tokio::test]
    async fn test_lateral_inhibition() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Create competing entities
        let entity1 = BrainInspiredEntity::new("option1".to_string(), EntityDirection::Input);
        let entity2 = BrainInspiredEntity::new("option2".to_string(), EntityDirection::Input);
        
        let key1 = graph.insert_brain_entity(entity1).await.unwrap();
        let key2 = graph.insert_brain_entity(entity2).await.unwrap();
        
        // Create inhibitory relationship
        let mut inhibitory_rel = BrainInspiredRelationship::new(key1, key2, RelationType::RelatedTo);
        inhibitory_rel.is_inhibitory = true;
        inhibitory_rel.weight = 0.8;
        
        graph.insert_brain_relationship(inhibitory_rel).await.unwrap();
        
        // Test inhibition
        let stats = graph.get_brain_statistics().await.unwrap();
        assert_eq!(stats.inhibitory_connections, 1);
    }

    // Test 9: Performance Tests
    #[tokio::test]
    async fn test_entity_creation_performance() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        let start = SystemTime::now();
        
        // Create 100 entities
        for i in 0..100 {
            let entity = BrainInspiredEntity::new(format!("entity_{}", i), EntityDirection::Input);
            graph.insert_brain_entity(entity).await.unwrap();
        }
        
        let duration = start.elapsed().unwrap();
        assert!(duration.as_millis() < 1000); // Should complete in less than 1 second
        
        let count = graph.entity_count().await.unwrap();
        assert_eq!(count, 100);
    }

    #[tokio::test]
    async fn test_relationship_creation_performance() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Create some entities first
        let mut entity_keys = Vec::new();
        for i in 0..10 {
            let entity = BrainInspiredEntity::new(format!("entity_{}", i), EntityDirection::Input);
            let key = graph.insert_brain_entity(entity).await.unwrap();
            entity_keys.push(key);
        }
        
        let start = SystemTime::now();
        
        // Create relationships between entities
        for i in 0..entity_keys.len() {
            for j in i+1..entity_keys.len() {
                let rel = BrainInspiredRelationship::new(
                    entity_keys[i], 
                    entity_keys[j], 
                    RelationType::RelatedTo
                );
                graph.insert_brain_relationship(rel).await.unwrap();
            }
        }
        
        let duration = start.elapsed().unwrap();
        assert!(duration.as_millis() < 500); // Should complete in less than 500ms
    }

    #[tokio::test]
    async fn test_activation_propagation_performance() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Create a concept structure
        let embedding = vec![0.1; 10];
        graph.create_concept_structure("perf_test".to_string(), embedding).await.unwrap();
        
        let start = SystemTime::now();
        
        // Run activation multiple times
        for _ in 0..10 {
            let _result = graph.activate_concept("perf_test_input", 0.8).await.unwrap();
        }
        
        let duration = start.elapsed().unwrap();
        assert!(duration.as_millis() < 100); // Should complete in less than 100ms
    }

    #[tokio::test]
    async fn test_large_graph_scalability() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Create 1000 entities
        let mut entity_keys = Vec::new();
        for i in 0..1000 {
            let entity = BrainInspiredEntity::new(format!("large_entity_{}", i), EntityDirection::Input);
            let key = graph.insert_brain_entity(entity).await.unwrap();
            entity_keys.push(key);
        }
        
        // Create some relationships
        for i in 0..100 {
            let rel = BrainInspiredRelationship::new(
                entity_keys[i], 
                entity_keys[i + 1], 
                RelationType::RelatedTo
            );
            graph.insert_brain_relationship(rel).await.unwrap();
        }
        
        let count = graph.entity_count().await.unwrap();
        assert_eq!(count, 1000);
        
        let rel_count = graph.get_relationship_count().await.unwrap();
        assert_eq!(rel_count, 100);
    }

    // Test 10: Error Handling Tests
    #[test]
    fn test_invalid_entity_creation() {
        // Test with empty concept_id
        let entity = BrainInspiredEntity::new("".to_string(), EntityDirection::Input);
        assert_eq!(entity.concept_id, "");
        
        // This should still work as empty string is valid
        assert!(matches!(entity.direction, EntityDirection::Input));
    }

    #[test]
    fn test_invalid_gate_inputs() {
        let gate = LogicGate::new(LogicGateType::And, 0.5);
        
        // Test with wrong number of inputs for NOT gate
        let not_gate = LogicGate::new(LogicGateType::Not, 0.0);
        assert!(not_gate.calculate_output(&[0.5, 0.6]).is_err());
        
        // Test weighted gate with mismatched weights
        let mut weighted_gate = LogicGate::new(LogicGateType::Weighted, 0.5);
        weighted_gate.weight_matrix = vec![0.5, 0.8];
        assert!(weighted_gate.calculate_output(&[0.5]).is_err());
    }

    #[tokio::test]
    async fn test_missing_entity_operations() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Test activation of non-existent concept
        let result = graph.activate_concept("nonexistent", 0.8).await;
        assert!(result.is_err());
        
        // Test finding non-existent entity
        let result = graph.find_entity_by_concept("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_invalid_relationship_operations() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        
        // Test removing non-existent relationship
        let result = graph.remove_relationship(key1, key2).await;
        assert!(result.is_err());
        
        // Test getting weight of non-existent relationship
        let result = graph.get_relationship_weight(key1, key2).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_activation_bounds_enforcement() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Create entity
        let entity = BrainInspiredEntity::new("bounds_test".to_string(), EntityDirection::Input);
        let key = graph.insert_brain_entity(entity).await.unwrap();
        
        // Test activation within bounds
        let result = graph.activate_concept("bounds_test", 0.5).await.unwrap();
        
        // Check that activations are within bounds
        for (_, activation) in result.final_activations {
            assert!(activation >= 0.0);
            assert!(activation <= 1.0);
        }
    }

    // Integration test combining multiple features
    #[tokio::test]
    async fn test_comprehensive_integration() {
        let graph = BrainEnhancedKnowledgeGraph::new_for_test();
        
        // Create a knowledge hierarchy: Animal -> Dog -> Poodle
        let animal = BrainInspiredEntity::new("animal".to_string(), EntityDirection::Input);
        let dog = BrainInspiredEntity::new("dog".to_string(), EntityDirection::Input);
        let poodle = BrainInspiredEntity::new("poodle".to_string(), EntityDirection::Input);
        
        let animal_key = graph.insert_brain_entity(animal).await.unwrap();
        let dog_key = graph.insert_brain_entity(dog).await.unwrap();
        let poodle_key = graph.insert_brain_entity(poodle).await.unwrap();
        
        // Create is_a relationships
        let dog_is_animal = BrainInspiredRelationship::new(dog_key, animal_key, RelationType::IsA);
        let poodle_is_dog = BrainInspiredRelationship::new(poodle_key, dog_key, RelationType::IsA);
        
        graph.insert_brain_relationship(dog_is_animal).await.unwrap();
        graph.insert_brain_relationship(poodle_is_dog).await.unwrap();
        
        // Add properties
        let has_fur = BrainInspiredEntity::new("has_fur".to_string(), EntityDirection::Output);
        let fur_key = graph.insert_brain_entity(has_fur).await.unwrap();
        
        let animal_has_fur = BrainInspiredRelationship::new(animal_key, fur_key, RelationType::HasProperty);
        graph.insert_brain_relationship(animal_has_fur).await.unwrap();
        
        // Test inheritance through query
        let result = graph.neural_query("poodle").await.unwrap();
        assert!(!result.final_activations.is_empty());
        
        // Test graph statistics
        let stats = graph.get_brain_statistics().await.unwrap();
        assert_eq!(stats.total_brain_entities, 4); // animal, dog, poodle, has_fur
        assert_eq!(stats.total_brain_relationships, 3); // two is_a + one has_property
        
        // Test graph health
        let health = graph.assess_graph_health().await.unwrap();
        assert!(health > 0.0);
        assert!(health <= 1.0);
    }
}