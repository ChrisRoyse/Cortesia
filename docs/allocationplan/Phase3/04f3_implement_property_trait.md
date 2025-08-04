# Task 04f3: Implement Trait for HasPropertyRelationship

**Estimated Time**: 7 minutes  
**Dependencies**: 04f2_implement_inheritance_trait.md  
**Next Task**: 04f4_implement_semantic_trait.md  

## Objective
Implement the GraphRelationship trait for HasPropertyRelationship only.

## Single Action
Add trait implementation for property relationships.

## Code to Add
Add to `src/storage/relationship_types.rs`:
```rust
// Implement GraphRelationship for HasPropertyRelationship
impl GraphRelationship for HasPropertyRelationship {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn relationship_type(&self) -> &str {
        "HAS_PROPERTY"
    }
    
    fn source_node_id(&self) -> &str {
        &self.source_node_id
    }
    
    fn target_node_id(&self) -> &str {
        &self.target_property_id
    }
    
    fn established_at(&self) -> DateTime<Utc> {
        self.established_at
    }
    
    fn last_modified(&self) -> DateTime<Utc> {
        self.last_modified
    }
    
    fn strength(&self) -> f32 {
        self.confidence
    }
    
    fn is_active(&self) -> bool {
        self.is_active
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
    
    fn validate(&self) -> bool {
        Self::validate(self)
    }
    
    fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}
```

## Success Check
```bash
cargo check
cargo test --lib property
```

## Acceptance Criteria
- [ ] HasPropertyRelationship implements GraphRelationship
- [ ] Maps target_property_id to target_node_id correctly
- [ ] Compiles without errors

## Duration
5-7 minutes for single trait implementation.