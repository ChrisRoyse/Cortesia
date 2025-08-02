# Task 03h: Test All Node Types Integration

**Estimated Time**: 10 minutes  
**Dependencies**: 03g_create_node_trait_interface.md  
**Next Task**: 04a_create_inheritance_relationship.md  

## Objective
Create comprehensive integration tests for all node types and their interactions.

## Single Action
Create integration test file to verify all node types work together correctly.

## File to Create
File: `tests/node_types_integration_test.rs`
```rust
use llmkg::storage::node_types::*;
use chrono::Utc;

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_all_node_types_compilation() {
        // Test that all node types can be created without compilation errors
        let concept = ConceptNode::new("TestConcept".to_string(), "Entity".to_string());
        let memory = MemoryNode::new("Test memory".to_string(), MemoryType::Semantic);
        let property = PropertyNode::new(
            "test_prop".to_string(),
            PropertyValue::String("value".to_string()),
            DataType::Text,
        );
        let exception = ExceptionNode::new(
            ExceptionType::Override,
            "prop".to_string(),
            PropertyValue::Boolean(true),
            "system".to_string(),
            "Test exception".to_string(),
        );
        let version = VersionNode::new(
            "main".to_string(),
            1,
            "Initial version".to_string(),
            ChangeType::Create,
            "system".to_string(),
        );
        let pathway = NeuralPathwayNode::new(
            PathwayType::Excitatory,
            concept.id.clone(),
            memory.id.clone(),
        );
        
        // Verify all nodes have valid IDs
        assert!(!concept.id.is_empty());
        assert!(!memory.id.is_empty());
        assert!(!property.id.is_empty());
        assert!(!exception.id.is_empty());
        assert!(!version.id.is_empty());
        assert!(!pathway.id.is_empty());
    }
    
    #[test]
    fn test_node_trait_polymorphism() {
        let nodes: Vec<Box<dyn GraphNode>> = vec![
            Box::new(ConceptNode::new("Concept1".to_string(), "Entity".to_string())),
            Box::new(MemoryNode::new("Memory1".to_string(), MemoryType::Episodic)),
            Box::new(PropertyNode::new(
                "Property1".to_string(),
                PropertyValue::Number(42.0),
                DataType::Float,
            )),
            Box::new(ExceptionNode::new(
                ExceptionType::Block,
                "blocked_prop".to_string(),
                PropertyValue::Null,
                "user".to_string(),
                "Blocking inheritance".to_string(),
            )),
            Box::new(VersionNode::new(
                "feature".to_string(),
                2,
                "Feature addition".to_string(),
                ChangeType::Update,
                "developer".to_string(),
            )),
            Box::new(NeuralPathwayNode::new(
                PathwayType::Inhibitory,
                "node1".to_string(),
                "node2".to_string(),
            )),
        ];
        
        // Test polymorphic operations
        for (i, node) in nodes.iter().enumerate() {
            println!("Testing node {}: {}", i, node.node_type());
            
            assert!(!node.id().is_empty());
            assert!(!node.node_type().is_empty());
            assert!(node.validate());
            
            let json_result = node.to_json();
            assert!(json_result.is_ok(), "JSON serialization failed for {}", node.node_type());
            
            let json_str = json_result.unwrap();
            assert!(json_str.contains("\"id\""), "JSON should contain id field");
        }
    }
    
    #[test]
    fn test_node_builder_patterns() {
        // Test fluent builder patterns
        let concept = ConceptNode::new("BuilderTest".to_string(), "Test".to_string())
            .with_ttfs_encoding(0.75)
            .with_semantic_embedding(vec![0.1, 0.2, 0.3, 0.4]);
        
        assert_eq!(concept.ttfs_encoding, Some(0.75));
        assert_eq!(concept.semantic_embedding, vec![0.1, 0.2, 0.3, 0.4]);
        
        let memory = MemoryNode::new("Builder Memory".to_string(), MemoryType::Working)
            .with_context("Test context".to_string())
            .with_strength(0.8);
        
        assert_eq!(memory.context, Some("Test context".to_string()));
        assert_eq!(memory.strength, 0.8);
        
        let property = PropertyNode::new(
            "builder_prop".to_string(),
            PropertyValue::Boolean(false),
            DataType::Boolean,
        ).with_inheritance_priority(5)
         .set_inheritable(false);
        
        assert_eq!(property.inheritance_priority, 5);
        assert!(!property.is_inheritable);
    }
    
    #[test]
    fn test_node_validation_rules() {
        // Test valid nodes
        let valid_concept = ConceptNode::new("Valid".to_string(), "Type".to_string());
        assert!(valid_concept.validate());
        
        let valid_memory = MemoryNode::new("Valid memory".to_string(), MemoryType::Semantic);
        assert!(valid_memory.validate());
        
        // Test edge cases
        let mut invalid_concept = ConceptNode::new("".to_string(), "Type".to_string());
        assert!(!invalid_concept.validate()); // Empty name should be invalid
        
        invalid_concept.confidence_score = 2.0; // Invalid confidence
        assert!(!invalid_concept.validate());
        
        let mut invalid_memory = MemoryNode::new("Test".to_string(), MemoryType::Procedural);
        invalid_memory.strength = -0.5; // Invalid strength
        assert!(!invalid_memory.validate());
    }
    
    #[test]
    fn test_serialization_roundtrip() {
        // Test JSON serialization and deserialization
        let original_concept = ConceptNode::new("Serialization Test".to_string(), "Entity".to_string())
            .with_ttfs_encoding(0.9);
        
        let json_str = original_concept.to_json().unwrap();
        let deserialized: ConceptNode = serde_json::from_str(&json_str).unwrap();
        
        assert_eq!(original_concept.id, deserialized.id);
        assert_eq!(original_concept.name, deserialized.name);
        assert_eq!(original_concept.ttfs_encoding, deserialized.ttfs_encoding);
        
        // Test with all node types
        let nodes_json = vec![
            MemoryNode::new("Test".to_string(), MemoryType::Episodic).to_json().unwrap(),
            PropertyNode::new("prop".to_string(), PropertyValue::Number(123.0), DataType::Float).to_json().unwrap(),
        ];
        
        for json in nodes_json {
            assert!(json.len() > 0);
            assert!(json.contains("\"id\""));
        }
    }
    
    #[test]
    fn test_node_relationships() {
        // Test that nodes can reference each other
        let source_concept = ConceptNode::new("Source".to_string(), "Entity".to_string());
        let target_concept = ConceptNode::new("Target".to_string(), "Entity".to_string());
        
        let pathway = NeuralPathwayNode::new(
            PathwayType::Excitatory,
            source_concept.id.clone(),
            target_concept.id.clone(),
        );
        
        assert_eq!(pathway.source_node, source_concept.id);
        assert_eq!(pathway.target_node, target_concept.id);
        
        let exception = ExceptionNode::new(
            ExceptionType::Override,
            "some_property".to_string(),
            PropertyValue::String("overridden".to_string()),
            "system".to_string(),
            "Test override for source concept".to_string(),
        );
        
        // Simulate applying exception to concept
        assert!(exception.is_applicable());
        assert_eq!(exception.target_property, "some_property");
    }
}
```

## Success Check
```bash
# Run comprehensive integration tests
cargo test integration_tests --test node_types_integration_test

# Verify all tests pass
cargo test node_types --lib
```

## Performance Test (Optional)
```bash
# Quick performance check
cargo test integration_tests --release
```

## Expected Results
- All 6 integration tests should pass
- No compilation warnings
- JSON serialization working for all types
- Polymorphic operations functioning

## Acceptance Criteria
- [ ] All integration tests pass
- [ ] Node polymorphism works correctly
- [ ] Builder patterns function as expected
- [ ] Validation rules work properly
- [ ] JSON serialization roundtrip succeeds
- [ ] Node relationship references work

## Duration
8-10 minutes for comprehensive testing and verification.