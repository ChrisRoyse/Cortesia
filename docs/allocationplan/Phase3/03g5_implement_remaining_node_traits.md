# Task 03g5: Implement Traits for Exception, Version, and Neural Nodes

**Estimated Time**: 12 minutes  
**Dependencies**: 03g4_implement_property_node_trait.md  
**Next Task**: 03g6_test_polymorphic_node_operations.md  

## Objective
Implement GraphNode trait for ExceptionNode, VersionNode, and NeuralPathwayNode.

## Single Action
Add trait implementations for the remaining three node types.

## Code to Add
Add to `src/storage/node_types.rs`:
```rust
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
```

## Success Check
```bash
cargo check
cargo test exception_node version_node neural_pathway
```

## Acceptance Criteria
- [ ] All three node types implement GraphNode
- [ ] Each has appropriate validation logic
- [ ] Each handles last_modified correctly
- [ ] All compile without errors

## Duration
10-12 minutes for three trait implementations.