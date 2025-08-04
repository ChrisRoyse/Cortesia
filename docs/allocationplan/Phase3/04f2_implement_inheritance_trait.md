# Task 04f2: Implement Trait for InheritsFromRelationship

**Estimated Time**: 7 minutes  
**Dependencies**: 04f1_define_base_relationship_trait.md  
**Next Task**: 04f3_implement_property_trait.md  

## Objective
Implement the GraphRelationship trait for InheritsFromRelationship only.

## Single Action
Add trait implementation for inheritance relationships.

## Code to Add
Add to `src/storage/relationship_types.rs`:
```rust
// Implement GraphRelationship for InheritsFromRelationship
impl GraphRelationship for InheritsFromRelationship {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn relationship_type(&self) -> &str {
        "INHERITS_FROM"
    }
    
    fn source_node_id(&self) -> &str {
        &self.source_node_id
    }
    
    fn target_node_id(&self) -> &str {
        &self.target_node_id
    }
    
    fn established_at(&self) -> DateTime<Utc> {
        self.established_at
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.last_validated
    }
    
    fn strength(&self) -> f32 {
        self.strength
    }
    
    fn is_active(&self) -> bool {
        self.is_active
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        let mut mutable_self = self.clone();
        mutable_self.validate()
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}
```

## Success Check
```bash
cargo check
cargo test --lib inheritance
```

## Acceptance Criteria
- [ ] InheritsFromRelationship implements GraphRelationship
- [ ] All methods return correct values
- [ ] Compiles without errors

## Duration
5-7 minutes for single trait implementation.