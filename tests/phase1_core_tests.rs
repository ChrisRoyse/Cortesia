#[cfg(test)]
mod phase1_core_tests {
    use llmkg::core::brain_types::{
        BrainInspiredEntity, EntityDirection, LogicGate, LogicGateType,
        BrainInspiredRelationship, ActivationPattern
    };
    use llmkg::core::types::{EntityKey, EntityData};
    use llmkg::core::graph::KnowledgeGraph;
    use llmkg::versioning::temporal_graph::{TemporalKnowledgeGraph, TimeRange};
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use chrono::Utc;

    #[test]
    fn test_brain_inspired_entity_creation() {
        let entity = BrainInspiredEntity::new(
            "test_concept".to_string(),
            EntityDirection::Input
        );
        
        assert_eq!(entity.concept_id, "test_concept");
        assert!(matches!(entity.direction, EntityDirection::Input));
        assert_eq!(entity.activation_state, 0.0);
    }

    #[test]
    fn test_entity_directions() {
        let input_entity = BrainInspiredEntity::new(
            "input_concept".to_string(),
            EntityDirection::Input
        );
        let output_entity = BrainInspiredEntity::new(
            "output_concept".to_string(),
            EntityDirection::Output
        );
        let hidden_entity = BrainInspiredEntity::new(
            "hidden_concept".to_string(),
            EntityDirection::Hidden
        );
        let mixed_entity = BrainInspiredEntity::new(
            "mixed_concept".to_string(),
            EntityDirection::Mixed
        );

        assert!(matches!(input_entity.direction, EntityDirection::Input));
        assert!(matches!(output_entity.direction, EntityDirection::Output));
        assert!(matches!(hidden_entity.direction, EntityDirection::Hidden));
        assert!(matches!(mixed_entity.direction, EntityDirection::Mixed));
    }

    #[test]
    fn test_logic_gate_and_computation() {
        let mut gate = LogicGate::new(LogicGateType::And, 0.5);
        gate.input_nodes = vec![Default::default(), Default::default()];
        
        // Test AND gate with both inputs above threshold
        let result = gate.calculate_output(&[0.8, 0.9]).unwrap();
        assert_eq!(result, 0.8); // Should return minimum of inputs
        
        // Test AND gate with one input below threshold
        let result = gate.calculate_output(&[0.3, 0.9]).unwrap();
        assert_eq!(result, 0.0); // Should return 0 when one input is below threshold
    }

    #[test]
    fn test_logic_gate_or_computation() {
        let mut gate = LogicGate::new(LogicGateType::Or, 0.5);
        gate.input_nodes = vec![Default::default(), Default::default()];
        
        // Test OR gate with both inputs above threshold
        let result = gate.calculate_output(&[0.8, 0.9]).unwrap();
        assert_eq!(result, 0.9); // Should return maximum of inputs
        
        // Test OR gate with one input below threshold
        let result = gate.calculate_output(&[0.3, 0.9]).unwrap();
        assert_eq!(result, 0.9); // Should return the higher input
        
        // Test OR gate with both inputs below threshold
        let result = gate.calculate_output(&[0.3, 0.4]).unwrap();
        assert_eq!(result, 0.0); // Should return 0 when both inputs are below threshold
    }

    #[test]
    fn test_logic_gate_not_computation() {
        let mut gate = LogicGate::new(LogicGateType::Not, 0.0);
        gate.input_nodes = vec![Default::default()];
        
        // Test NOT gate
        let result = gate.calculate_output(&[0.8]).unwrap();
        assert!((result - 0.2).abs() < 0.001, "Expected ~0.2, got {}", result);
        
        let result = gate.calculate_output(&[0.3]).unwrap();
        assert!((result - 0.7).abs() < 0.001, "Expected ~0.7, got {}", result);
    }

    #[test]
    fn test_logic_gate_inhibitory_computation() {
        let mut gate = LogicGate::new(LogicGateType::Inhibitory, 0.0);
        gate.input_nodes = vec![Default::default(), Default::default()];
        
        // Test inhibitory gate: first input minus second input
        let result = gate.calculate_output(&[0.8, 0.3]).unwrap();
        assert_eq!(result, 0.5); // 0.8 - 0.3 = 0.5
        
        let result = gate.calculate_output(&[0.5, 0.8]).unwrap();
        assert_eq!(result, 0.0); // max(0.5 - 0.8, 0.0) = 0.0
    }

    #[test]
    fn test_activation_pattern() {
        let mut pattern = ActivationPattern::new("test query".to_string());
        
        // Create a dummy graph to generate proper entity keys
        let graph = KnowledgeGraph::new(384).unwrap();
        let entity1 = graph.insert_entity(1, EntityData {
            type_id: 1,
            properties: "name:entity1".to_string(),
            embedding: vec![0.1; 384],
        }).unwrap();
        let entity2 = graph.insert_entity(2, EntityData {
            type_id: 1,
            properties: "name:entity2".to_string(),
            embedding: vec![0.2; 384],
        }).unwrap();
        let entity3 = graph.insert_entity(3, EntityData {
            type_id: 1,
            properties: "name:entity3".to_string(),
            embedding: vec![0.3; 384],
        }).unwrap();
        
        pattern.activations.insert(entity1, 0.9);
        pattern.activations.insert(entity2, 0.7);
        pattern.activations.insert(entity3, 0.5);
        
        let top_activations = pattern.get_top_activations(2);
        assert_eq!(top_activations.len(), 2);
        assert_eq!(top_activations[0].1, 0.9);
        assert_eq!(top_activations[1].1, 0.7);
    }

    #[test]
    fn test_brain_inspired_relationship() {
        let graph = KnowledgeGraph::new(384).unwrap();
        let entity1 = graph.insert_entity(1, EntityData {
            type_id: 1,
            properties: "name:entity1".to_string(),
            embedding: vec![0.1; 384],
        }).unwrap();
        let entity2 = graph.insert_entity(2, EntityData {
            type_id: 1,
            properties: "name:entity2".to_string(),
            embedding: vec![0.2; 384],
        }).unwrap();
        
        let mut relationship = BrainInspiredRelationship::new(
            entity1,
            entity2,
            "test_relation".to_string()
        );
        
        assert_eq!(relationship.weight, 1.0);
        assert!(!relationship.is_inhibitory);
        
        // Test Hebbian learning
        relationship.strengthen(0.1);
        assert!(relationship.weight == 1.0); // Should be capped at 1.0
        assert_eq!(relationship.activation_count, 1);
    }

    #[test]
    fn test_relationship_weakening() {
        let graph = KnowledgeGraph::new(384).unwrap();
        let entity1 = graph.insert_entity(1, EntityData {
            type_id: 1,
            properties: "name:entity1".to_string(),
            embedding: vec![0.1; 384],
        }).unwrap();
        let entity2 = graph.insert_entity(2, EntityData {
            type_id: 1,
            properties: "name:entity2".to_string(),
            embedding: vec![0.2; 384],
        }).unwrap();
        
        let mut relationship = BrainInspiredRelationship::new(
            entity1,
            entity2,
            "test_relation".to_string()
        );
        
        // Set initial weight lower than 1.0
        relationship.weight = 0.8;
        
        // Test weakening
        relationship.weaken(0.2);
        assert_eq!(relationship.weight, 0.6);
        
        // Test that it doesn't go below 0.0
        relationship.weaken(1.0);
        assert_eq!(relationship.weight, 0.0);
    }

    #[tokio::test]
    async fn test_temporal_knowledge_graph() {
        let graph = KnowledgeGraph::new(384).unwrap();
        let temporal_graph = TemporalKnowledgeGraph::new(graph);
        
        let entity = BrainInspiredEntity::new(
            "temporal_entity".to_string(),
            EntityDirection::Output
        );
        
        let valid_time = TimeRange::new(Utc::now());
        let result = temporal_graph.insert_temporal_entity(entity, valid_time).await;
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_knowledge_graph_basic_operations() {
        let graph = KnowledgeGraph::new(384).unwrap();
        
        let entity1 = graph.insert_entity(1, EntityData {
            type_id: 1,
            properties: "name:first_entity".to_string(),
            embedding: vec![0.1; 384],
        }).unwrap();
        
        let entity2 = graph.insert_entity(2, EntityData {
            type_id: 1,
            properties: "name:second_entity".to_string(),
            embedding: vec![0.2; 384],
        }).unwrap();
        
        // Test entity retrieval
        let retrieved = graph.get_entity(entity1).unwrap();
        assert_eq!(retrieved.properties, "name:first_entity");
        
        // Test entity existence
        assert!(graph.get_entity(entity1).is_ok());
        assert!(graph.get_entity(entity2).is_ok());
    }

    #[test]
    fn test_entity_key_functionality() {
        let graph = KnowledgeGraph::new(384).unwrap();
        
        let entity1 = graph.insert_entity(1, EntityData {
            type_id: 1,
            properties: "name:entity1".to_string(),
            embedding: vec![0.1; 384],
        }).unwrap();
        
        let entity2 = graph.insert_entity(2, EntityData {
            type_id: 1,
            properties: "name:entity2".to_string(),
            embedding: vec![0.2; 384],
        }).unwrap();
        
        // Test that entity keys are different
        assert_ne!(entity1, entity2);
        
        // Test that entity keys are valid
        assert!(entity1 != EntityKey::default());
        assert!(entity2 != EntityKey::default());
    }

    #[test]  
    fn test_activation_pattern_operations() {
        let mut pattern = ActivationPattern::new("test operations".to_string());
        assert_eq!(pattern.query, "test operations");
        assert!(pattern.activations.is_empty());
        
        // Test adding activations
        let graph = KnowledgeGraph::new(384).unwrap();
        let entity = graph.insert_entity(1, EntityData {
            type_id: 1,
            properties: "name:test_entity".to_string(),
            embedding: vec![0.1; 384],
        }).unwrap();
        
        pattern.activations.insert(entity, 0.8);
        assert_eq!(pattern.activations.len(), 1);
        assert_eq!(pattern.activations[&entity], 0.8);
        
        // Test top activations with empty pattern
        let empty_pattern = ActivationPattern::new("empty".to_string());
        let top_empty = empty_pattern.get_top_activations(5);
        assert!(top_empty.is_empty());
    }
}