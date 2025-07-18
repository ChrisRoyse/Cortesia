#[cfg(test)]
mod phase1_minimal_test {
    use llmkg::core::brain_types::{
        BrainInspiredEntity, EntityDirection, LogicGate, LogicGateType
    };

    #[test]
    fn test_basic_entity_creation() {
        let entity = BrainInspiredEntity::new(
            "test_concept".to_string(),
            EntityDirection::Input
        );
        
        assert_eq!(entity.concept_id, "test_concept");
        assert!(matches!(entity.direction, EntityDirection::Input));
        assert_eq!(entity.activation_state, 0.0);
    }

    #[test]
    fn test_basic_logic_gate() {
        let mut gate = LogicGate::new(LogicGateType::And, 0.5);
        gate.input_nodes = vec![Default::default(), Default::default()];
        
        let result = gate.calculate_output(&[0.8, 0.9]).unwrap();
        assert_eq!(result, 0.8);
    }
}