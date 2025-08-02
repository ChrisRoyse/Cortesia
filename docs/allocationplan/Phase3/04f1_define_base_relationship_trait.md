# Task 04f1: Define Base Relationship Trait

**Estimated Time**: 5 minutes  
**Dependencies**: 04e_create_neural_relationship.md  
**Next Task**: 04f2_implement_inheritance_trait.md  

## Objective
Define the core GraphRelationship trait interface with essential methods.

## Single Action
Create the basic trait definition with core methods only.

## Code to Add
Add to `src/storage/relationship_types.rs`:
```rust
use std::any::Any;
use chrono::{DateTime, Utc};

/// Common trait for all graph relationship types
pub trait GraphRelationship: Send + Sync {
    fn id(&self) -> &str;
    fn relationship_type(&self) -> &str;
    fn source_node_id(&self) -> &str;
    fn target_node_id(&self) -> &str;
    fn established_at(&self) -> DateTime<Utc>;
    fn last_modified(&self) -> DateTime<Utc>;
    fn strength(&self) -> f32;
    fn is_active(&self) -> bool;
    fn as_any(&self) -> &dyn Any;
    fn validate(&self) -> bool;
    fn to_json(&self) -> Result<String, serde_json::Error>;
}
```

## Success Check
```bash
cargo check
# Should compile without errors
```

## Acceptance Criteria
- [ ] GraphRelationship trait compiles without errors
- [ ] All required methods defined
- [ ] Trait bounds include Send + Sync

## Duration
3-5 minutes for trait definition only.