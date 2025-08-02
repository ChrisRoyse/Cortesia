# Task 04f7: Test Polymorphic Relationship Operations

**Estimated Time**: 8 minutes  
**Dependencies**: 04f6_implement_neural_trait.md  
**Next Task**: 05a_create_basic_node_operations.md  

## Objective
Create comprehensive tests for polymorphic relationship operations.

## Single Action
Add test module for relationship trait testing.

## Code to Add
Add to `src/storage/relationship_types.rs`:
```rust
#[cfg(test)]
mod relationship_trait_tests {
    use super::*;
    use crate::storage::node_types::PropertySource;
    
    #[test]
    fn test_polymorphic_relationship_operations() {
        let relationships: Vec<Box<dyn GraphRelationship>> = vec![
            Box::new(InheritsFromRelationship::new(
                "child".to_string(),
                "parent".to_string(),
                InheritanceType::Direct,
                1,
            )),
            Box::new(HasPropertyRelationship::new(
                "node".to_string(),
                "property".to_string(),
                PropertySource::Direct,
                "system".to_string(),
            )),
        ];
        
        for (i, rel) in relationships.iter().enumerate() {
            println!("Testing relationship {}: {}", i, rel.relationship_type());
            
            assert!(!rel.id().is_empty());
            assert!(!rel.relationship_type().is_empty());
            assert!(!rel.source_node_id().is_empty());
            assert!(!rel.target_node_id().is_empty());
            assert!(rel.validate());
            assert!(rel.strength() >= 0.0);
            
            let json_result = rel.to_json();
            assert!(json_result.is_ok(), "JSON serialization failed for {}", rel.relationship_type());
        }
    }
    
    #[test]
    fn test_relationship_type_identification() {
        let inheritance_rel = InheritsFromRelationship::new(
            "a".to_string(),
            "b".to_string(),
            InheritanceType::Direct,
            1,
        );
        
        assert_eq!(inheritance_rel.relationship_type(), "INHERITS_FROM");
    }
}
```

## Success Check
```bash
cargo test relationship_trait_tests
```

## Acceptance Criteria
- [ ] All relationship types tested polymorphically
- [ ] Type identification works correctly
- [ ] JSON serialization tested
- [ ] All assertions pass

## Duration
6-8 minutes for comprehensive testing.