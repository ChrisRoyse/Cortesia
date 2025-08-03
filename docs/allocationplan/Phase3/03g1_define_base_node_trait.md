# Task 03g1: Define Base Node Trait

**Estimated Time**: 5 minutes  
**Dependencies**: 03f_create_neural_pathway_struct.md  
**Next Task**: 03g2_implement_concept_node_trait.md  

## Objective
Define the core GraphNode trait interface with essential methods.

## Single Action
Create the basic trait definition with core methods only.

## Code to Add
Add to `src/storage/node_types.rs`:
```rust
use std::any::Any;
use chrono::{DateTime, Utc};

/// Common trait for all graph node types
pub trait GraphNode: Send + Sync {
    fn id(&self) -> &str;
    fn node_type(&self) -> &str;
    fn created_at(&self) -> DateTime<Utc>;
    fn last_modified(&self) -> DateTime<Utc>;
    fn as_any(&self) -> &dyn Any;
    fn validate(&self) -> bool;
    fn to_json(&self) -> Result<String, serde_json::Error>;
}
```

## Success Check
```bash
cargo check
```

## Acceptance Criteria
- [ ] GraphNode trait compiles without errors
- [ ] All required methods defined
- [ ] Trait bounds include Send + Sync

## Duration
3-5 minutes for trait definition only.