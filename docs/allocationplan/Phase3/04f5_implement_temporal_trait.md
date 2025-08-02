# Task 04f5: Implement Trait for TemporalSequenceRelationship

**Estimated Time**: 7 minutes  
**Dependencies**: 04f4_implement_semantic_trait.md  
**Next Task**: 04f6_implement_neural_trait.md  

## Objective
Implement the GraphRelationship trait for TemporalSequenceRelationship only.

## Single Action
Add trait implementation for temporal relationships.

## Code to Add
Add to `src/storage/relationship_types.rs`:
```rust
// Implement GraphRelationship for TemporalSequenceRelationship
impl GraphRelationship for TemporalSequenceRelationship {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn relationship_type(&self) -> &str {
        "TEMPORAL_SEQUENCE"
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
        self.established_at
    }
    
    fn strength(&self) -> f32 {
        self.confidence * self.causal_strength
    }
    
    fn is_active(&self) -> bool {
        self.confidence > 0.3
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
cargo test --lib temporal
```

## Acceptance Criteria
- [ ] TemporalSequenceRelationship implements GraphRelationship
- [ ] Strength calculated as confidence * causal_strength
- [ ] Active when confidence > 0.3

## Duration
5-7 minutes for single trait implementation.