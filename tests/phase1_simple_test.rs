#[cfg(test)]
mod simple_phase1_tests {
    use llmkg::core::brain_types::{
        BrainInspiredEntity, LogicGate, BrainInspiredRelationship, ActivationPattern,
        EntityDirection, LogicGateType, RelationType
    };
    use llmkg::core::types::EntityKey;
    use std::collections::HashMap;
    
    #[test]
    fn test_basic_entity_creation() {
        let entity = BrainInspiredEntity::new("test".to_string(), EntityDirection::Input);
        assert_eq!(entity.concept_id, "test");
        assert!(matches!(entity.direction, EntityDirection::Input));
    }
    
    #[test]
    fn test_basic_gate_creation() {
        let mut gate = LogicGate::new(LogicGateType::And, 0.5);
        gate.input_nodes = vec![EntityKey::default(), EntityKey::default()];
        
        let result = gate.calculate_output(&[0.8, 0.9]).unwrap();
        assert_eq!(result, 0.8);
    }
    
    #[test]
    fn test_basic_relationship_creation() {
        let key1 = EntityKey::default();
        let key2 = EntityKey::default();
        let relationship = BrainInspiredRelationship::new(key1, key2, RelationType::IsA);
        
        assert_eq!(relationship.source, key1);
        assert_eq!(relationship.target, key2);
    }
    
    #[test]
    fn test_basic_activation_pattern() {
        let pattern = ActivationPattern::new("test".to_string());
        assert_eq!(pattern.query, "test");
    }
}