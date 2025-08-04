# Task 04f6: Implement Trait for NeuralConnectionRelationship

**Estimated Time**: 7 minutes  
**Dependencies**: 04f5_implement_temporal_trait.md  
**Next Task**: 04f7_test_polymorphic_operations.md  

## Objective
Implement the GraphRelationship trait for NeuralConnectionRelationship only.

## Single Action
Add trait implementation for neural relationships.

## Code to Add
Add to `src/storage/relationship_types.rs`:
```rust
// Implement GraphRelationship for NeuralConnectionRelationship
impl GraphRelationship for NeuralConnectionRelationship {
    fn id(&self) -> &str {
        &self.id
    }
    
    fn relationship_type(&self) -> &str {
        "NEURAL_CONNECTION"
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
        self.last_activated
    }
    
    fn strength(&self) -> f32 {
        self.get_effective_strength()
    }
    
    fn is_active(&self) -> bool {
        Self::is_active(self)
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
cargo test --lib neural
```

## Acceptance Criteria
- [ ] NeuralConnectionRelationship implements GraphRelationship
- [ ] Uses get_effective_strength() method
- [ ] Delegates to existing is_active and validate methods

## Duration
5-7 minutes for single trait implementation.