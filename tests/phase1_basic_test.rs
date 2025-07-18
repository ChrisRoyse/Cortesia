#[cfg(test)]
mod phase1_basic_test {
    use llmkg::core::brain_types::{BrainInspiredEntity, EntityDirection};

    #[test]
    fn test_entity_creation() {
        let entity = BrainInspiredEntity::new(
            "test_concept".to_string(),
            EntityDirection::Input
        );
        
        assert_eq!(entity.concept_id, "test_concept");
        assert!(matches!(entity.direction, EntityDirection::Input));
        assert_eq!(entity.activation_state, 0.0);
        println!("✅ Phase 1 Basic Entity Test Passed!");
    }
}