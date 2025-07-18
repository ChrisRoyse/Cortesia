#[cfg(test)]
mod minimal_tests {
    use llmkg::core::brain_types::EntityDirection;
    
    #[test]
    fn test_entity_direction() {
        let dir = EntityDirection::Input;
        assert!(matches!(dir, EntityDirection::Input));
    }
}