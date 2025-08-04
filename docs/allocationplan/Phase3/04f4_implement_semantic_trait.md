# Task 04f4: Implement Trait for SemanticallyRelatedRelationship

**Estimated Time**: 7 minutes  
**Dependencies**: 04f3_implement_property_trait.md  
**Next Task**: 04f5_implement_temporal_trait.md  

## Objective
Implement the GraphRelationship trait for SemanticallyRelatedRelationship only.

## Single Action
Add trait implementation for semantic relationships.

## Code to Add
Add to `src/storage/relationship_types.rs`:
```rust
// Implement GraphRelationship for SemanticallyRelatedRelationship
impl GraphRelationship for SemanticallyRelatedRelationship {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn relationship_type(&self) -> &str {
        "SEMANTICALLY_RELATED"
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
        self.last_computed
    }
    
    fn strength(&self) -> f32 {
        self.similarity_score
    }
    
    fn is_active(&self) -> bool {
        !self.is_weak_relationship()
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
cargo test --lib semantic
```

## Acceptance Criteria
- [ ] SemanticallyRelatedRelationship implements GraphRelationship
- [ ] Uses similarity_score as strength
- [ ] Uses is_weak_relationship() for is_active logic

## Duration
5-7 minutes for single trait implementation.