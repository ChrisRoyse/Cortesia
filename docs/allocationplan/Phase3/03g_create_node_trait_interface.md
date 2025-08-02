# Task 03g: Create Node Trait Interface

**Estimated Time**: 8 minutes  
**Dependencies**: 03f_create_neural_pathway_struct.md  
**Next Task**: 03h_test_all_node_types.md  

## Objective
Create a common trait interface for all node types to enable polymorphic operations.

## Single Action
Add GraphNode trait and implement it for all node types.

## Code to Add
Add to `src/storage/node_types.rs`:
```rust
use std::any::Any;

/// Common trait for all graph node types
pub trait GraphNode: Send + Sync {
    fn id(&self) -> &str;
    fn node_type(&self) -> &str;
    fn created_at(&self) -> DateTime<Utc>;
    fn last_modified(&self) -> DateTime<Utc>;
    fn as_any(&self) -> &dyn Any;
    fn validate(&self) -> bool;
    fn to_json(&self) -> Result<String, serde_json::Error>;
}

// Implement GraphNode for ConceptNode
impl GraphNode for ConceptNode {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn node_type(&self) -> &str {
        "Concept"
    }
    
    fn created_at(&self) -> DateTime<Utc> {
        self.creation_timestamp
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.last_accessed
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        !self.name.is_empty() && 
        self.confidence_score >= 0.0 && 
        self.confidence_score <= 1.0 &&
        self.inheritance_depth >= 0
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// Implement GraphNode for MemoryNode
impl GraphNode for MemoryNode {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn node_type(&self) -> &str {
        "Memory"
    }
    
    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.last_strengthened
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        !self.content.is_empty() && 
        self.strength >= 0.0 && 
        self.strength <= 1.0 &&
        self.decay_rate >= 0.0
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// Implement GraphNode for PropertyNode
impl GraphNode for PropertyNode {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn node_type(&self) -> &str {
        "Property"
    }
    
    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.modified_at
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        self.is_valid()
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// Implement GraphNode for ExceptionNode
impl GraphNode for ExceptionNode {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn node_type(&self) -> &str {
        "Exception"
    }
    
    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.applied_at.unwrap_or(self.created_at)
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        !self.target_property.is_empty() && 
        !self.justification.is_empty() &&
        self.confidence >= 0.0 && 
        self.confidence <= 1.0
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// Implement GraphNode for VersionNode
impl GraphNode for VersionNode {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn node_type(&self) -> &str {
        "Version"
    }
    
    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.created_at
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        !self.branch_name.is_empty() && 
        self.version_number > 0 &&
        !self.change_summary.is_empty()
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

// Implement GraphNode for NeuralPathwayNode
impl GraphNode for NeuralPathwayNode {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn node_type(&self) -> &str {
        "NeuralPathway"
    }
    
    fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.last_activated
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        !self.source_node.is_empty() && 
        !self.target_node.is_empty() &&
        self.connection_strength >= 0.0 && 
        self.connection_strength <= 1.0
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

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
}
```

## Success Check
```bash
# Verify compilation
cargo check
# Should compile successfully

# Run trait tests
cargo test trait_tests
```

## Acceptance Criteria
- [ ] GraphNode trait compiles without errors
- [ ] All node types implement the trait
- [ ] Polymorphic operations work correctly
- [ ] JSON serialization functions
- [ ] Tests pass

## Duration
6-8 minutes for trait implementation and testing.