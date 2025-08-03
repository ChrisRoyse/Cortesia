# Task 03g6: Test Polymorphic Node Operations

**Estimated Time**: 8 minutes  
**Dependencies**: 03g5_implement_remaining_node_traits.md  
**Next Task**: 03h_test_all_node_types.md  

## Objective
Create comprehensive tests for polymorphic node operations.

## Single Action
Add test module for node trait testing.

## Code to Add
Add to `src/storage/node_types.rs`:
```rust
#[cfg(test)]
mod trait_tests {
    use super::*;
    
    #[test]
    fn test_polymorphic_node_operations() {
        let nodes: Vec<Box<dyn GraphNode>> = vec![
            Box::new(ConceptNode::new("test".to_string(), "entity".to_string())),
            Box::new(MemoryNode::new("memory".to_string(), MemoryType::Episodic)),
            Box::new(PropertyNode::new(
                "prop".to_string(), 
                PropertyValue::String("value".to_string()), 
                DataType::Text
            )),
        ];
        
        for node in &nodes {
            assert!(!node.id().is_empty());
            assert!(!node.node_type().is_empty());
            assert!(node.validate());
            assert!(node.to_json().is_ok());
        }
    }
    
    #[test]
    fn test_node_type_identification() {
        let concept = ConceptNode::new("test".to_string(), "entity".to_string());
        let memory = MemoryNode::new("test".to_string(), MemoryType::Semantic);
        
        assert_eq!(concept.node_type(), "Concept");
        assert_eq!(memory.node_type(), "Memory");
    }
    
    #[test]
    fn test_node_validation() {
        let valid_concept = ConceptNode::new("test".to_string(), "entity".to_string());
        assert!(valid_concept.validate());
        
        let mut invalid_concept = valid_concept.clone();
        invalid_concept.confidence_score = 1.5; // Invalid: > 1.0
        assert!(!invalid_concept.validate());
    }
}
```

## Success Check
```bash
cargo test trait_tests
```

## Acceptance Criteria
- [ ] Polymorphic operations tested
- [ ] Node type identification works
- [ ] Validation edge cases tested
- [ ] All assertions pass

## Duration
6-8 minutes for comprehensive testing.