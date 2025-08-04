# Task 03g3: Implement Trait for MemoryNode

**Estimated Time**: 6 minutes  
**Dependencies**: 03g2_implement_concept_node_trait.md  
**Next Task**: 03g4_implement_property_node_trait.md  

## Objective
Implement the GraphNode trait for MemoryNode only.

## Single Action
Add trait implementation for memory nodes.

## Code to Add
Add to `src/storage/node_types.rs`:
```rust
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
```

## Success Check
```bash
cargo check
cargo test memory_node
```

## Acceptance Criteria
- [ ] MemoryNode implements GraphNode
- [ ] Uses last_strengthened as last_modified
- [ ] Validates strength bounds and content
- [ ] Compiles without errors

## Duration
4-6 minutes for single trait implementation.