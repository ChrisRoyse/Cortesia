# Task 03g2: Implement Trait for ConceptNode

**Estimated Time**: 6 minutes  
**Dependencies**: 03g1_define_base_node_trait.md  
**Next Task**: 03g3_implement_memory_node_trait.md  

## Objective
Implement the GraphNode trait for ConceptNode only.

## Single Action
Add trait implementation for concept nodes.

## Code to Add
Add to `src/storage/node_types.rs`:
```rust
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
```

## Success Check
```bash
cargo check
cargo test concept_node
```

## Acceptance Criteria
- [ ] ConceptNode implements GraphNode
- [ ] Validation checks all constraints
- [ ] JSON serialization works
- [ ] Compiles without errors

## Duration
4-6 minutes for single trait implementation.