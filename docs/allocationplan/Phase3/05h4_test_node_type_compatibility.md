# Task 05h4: Test Node Type CRUD Compatibility

**Estimated Time**: 10 minutes  
**Dependencies**: 05h3_test_error_handling.md  
**Next Task**: 05h5_test_filter_combinations.md  

## Objective
Test that all node types work correctly with CRUD operations.

## Single Action
Add one test to verify all node types are CRUD compatible.

## Code to Add
Add to `tests/crud_operations_integration_test.rs`:
```rust
    #[test]
    fn test_all_node_types_crud_compatibility() {
        // Test that all node types are compatible with CRUD operations
        let nodes: Vec<Box<dyn GraphNode>> = vec![
            Box::new(ConceptNode::new("Test".to_string(), "Entity".to_string())),
            Box::new(MemoryNode::new("Test memory".to_string(), MemoryType::Episodic)),
            Box::new(PropertyNode::new(
                "test_prop".to_string(),
                PropertyValue::String("value".to_string()),
                DataType::Text,
            )),
            Box::new(ExceptionNode::new(
                ExceptionType::Override,
                "prop".to_string(),
                PropertyValue::Boolean(true),
                "system".to_string(),
                "Test exception".to_string(),
            )),
            Box::new(VersionNode::new(
                "main".to_string(),
                1,
                "Test version".to_string(),
                ChangeType::Create,
                "system".to_string(),
            )),
            Box::new(NeuralPathwayNode::new(
                PathwayType::Excitatory,
                "source".to_string(),
                "target".to_string(),
            )),
        ];
        
        for node in &nodes {
            // Test that all nodes implement required traits
            assert!(!node.id().is_empty());
            assert!(!node.node_type().is_empty());
            assert!(node.validate());
            
            // Test JSON serialization
            let json_result = node.to_json();
            assert!(json_result.is_ok(), "JSON serialization failed for {}", node.node_type());
            
            let json = json_result.unwrap();
            assert!(json.contains("\"id\""));
            assert!(json.len() > 10); // Reasonable JSON length
            
            println!("✅ {} - CRUD compatible", node.node_type());
        }
    }
```

## Success Check
```bash
cargo test test_all_node_types_crud_compatibility
```

## Acceptance Criteria
- [ ] All 6 node types tested
- [ ] Each node validates correctly
- [ ] JSON serialization works for all
- [ ] Test passes for all node types

## Duration
8-10 minutes for compatibility testing.