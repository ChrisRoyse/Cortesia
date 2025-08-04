# Task 03g4: Implement Trait for PropertyNode

**Estimated Time**: 5 minutes  
**Dependencies**: 03g3_implement_memory_node_trait.md  
**Next Task**: 03g5_implement_remaining_node_traits.md  

## Objective
Implement the GraphNode trait for PropertyNode only.

## Single Action
Add trait implementation for property nodes.

## Code to Add
Add to `src/storage/node_types.rs`:
```rust
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
```

## Success Check
```bash
cargo check
cargo test property_node
```

## Acceptance Criteria
- [ ] PropertyNode implements GraphNode
- [ ] Delegates validation to existing is_valid method
- [ ] Uses modified_at for last_modified
- [ ] Compiles without errors

## Duration
3-5 minutes for single trait implementation.